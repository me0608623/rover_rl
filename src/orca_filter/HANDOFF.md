# ORCA 車端部署 — Claude Code 交接文檔

> 寫給車端 Claude Code:這份記錄 ORCA 在車端的部署狀態、兩種模式(旁觀/控制車)、做了什麼、怎麼測/debug/改。
> 建立日期 2026-07-08。來源:PC-A IsaacLab `play_eval/rvo2_safety_filter.py` 移植 + 接 policy 控制車。

---

## 1. 這是什麼

把 IsaacLab 模擬器的 ORCA(RVO2)安全過濾器移植到車端 ROS2 humble,有**兩種模式**:

| 模式 | 節點 | 控制車? | 用途 |
|------|------|---------|------|
| **旁觀** | `orca_bystander` | ❌ 只印+RViz | 驗證 ORCA 對真實 LV-DOT 障礙的閃躲建議合不合理 |
| **控制車** | `orca_safety` | ✅ 接 cmd 鏈 | 取代 vo_safety_node,A/B 比較後上線 |

**non-cooperative 設計**:行人不會避讓機器人 → `ego_responsibility=1.0`(機器人扛 100% 避讓)+ `obs_inflation=1.8`(障礙半徑 ×1.8 提早反應)。

兩模式共用 `orca_core.py`(純 ORCA solver,1:1 移植自 sim)。

---

## 2. 已完成的工作(2026-07-08,全在車端 ~/rover_rl/)

### 2a. pyrvo2 安裝(車端 ARM)
- aarch64 / Python 3.10.12,PyPI 無 ARM wheel → 從 https://github.com/mit-acl/Python-RVO2 源碼編譯,裝 `~/.local`
- 重裝見 §7 坑 2

### 2b. orca_filter ROS2 package(`~/rover_rl/src/orca_filter/`)
- ament_python,**不動 vo_interface / vo_safety_node / policy_node**(只新增)
- 含兩個節點:`orca_bystander`(旁觀)+ `orca_safety`(控制車)
- colcon build OK

### 2c. orca_safety_node 控制車版(取代 VO)
- 獨立節點,topic 對稱 `vo_safety_node`
- launch 用 `enable_orca` 切換;policy 改道靠 `topic_cmd_vel` 覆寫(policy 無感知)
- 安全網:看門狗 + front_brake + slew(無 escape/commit,ORCA `lateral_evasion` 替代)

### 2d. deploy 選單整合
- `deploy_select.sh` 安全層選擇:`1=VO / 2=ORCA / 3=都不`(皆控制車)
- 選 2 設 `ORCA_ARG="enable_orca:=true"`,launch 啟 `orca_safety_node`

---

## 3. 架構

### 旁觀模式(bystander)
```
/vo_interface/tracked_obstacles + /odom → [orca_bystander] → 印 + RViz(不發 cmd_vel)
```

### 控制車模式(safety,取代 VO)
```
policy_node ──/rover_rl/cmd_vel_desired──► [orca_safety_node] ──/input/nav_cmd_vel──► mux ──► 底盤
                         ▲
   /odom + /vo_interface/tracked_obstacles + /rover_rl_policy/status(front_m)──┘
```

policy_node 對 enable_orca **無感知**:切換靠 launch `topic_cmd_vel` 覆寫(enable_orca=true → policy 改發 `/rover_rl/cmd_vel_desired`)。ORCA 與 VO topic 完全對稱,可隨時 A/B 切換。

---

## 4. 關鍵設計決策

| 決策 | 值 | 為什麼 |
|------|-----|--------|
| non-cooperative | `ego_responsibility=1.0` | 行人不會減速讓機器人 |
| 障礙膨脹 | `obs_inflation=1.8` | 行人不讓 → 提早反應 |
| 只處理動態 | `static_vel_threshold=0.15` | 靜態交 policy/LiDAR + front_brake |
| 分離時域 | dynamic 2.5s / static 0.3s | 動態提早反應;靜態<0.5s 過走廊 |
| 側向逃生 | `LATERAL_EVASION_*` | ORCA 凍結(輸出≈0)時注入垂直速度 |
| NH-ORCA 投影 | `STEER_KP=3.0` | 差分車不能側移,world v_safe→body (v,ω) |
| 介入門檻 | `INTERVENTION_EPSILON=0.05 m/s` | minimal intervention |
| v_pref 預測 | `yaw_next = yaw + ω·dt` | 對齊 sim filter_step |

所有常數 1:1 移植自 PC-A `rvo2_safety_filter.py`。

---

## 5. 怎麼跑

### 旁觀(不控制車)
```bash
lv-dot                                    # T2:LV-DOT+vo_interface+RViz
source ~/rover_rl/setup_env.sh && ros2 launch orca_filter bystander.launch.py   # T3
```

### 控制車(取代 VO,實車)
```bash
lv-dot                                    # 障礙來源(前提)
deploy_rl_shell                           # 選 checkpoint → 安全層選 2(ORCA)
```

⚠️ **Zenoh**:車端用 `rmw_zenoh_cpp`(`setup_env.sh`)。任何 ORCA 節點啟動前必須 `source ~/rover_rl/setup_env.sh`,否則跑得起來但收不到 topic。`deploy_rl_shell` / `lv-dot` 已自動 source。

---

## 6. orca_safety_node 控制車版詳解

### 6.1 接線(對稱 vo_safety_node)
- 訂:`/rover_rl/cmd_vel_desired`(Twist,policy 期望)、`/odom`、`/vo_interface/tracked_obstacles`、`/rover_rl_policy/status`(String JSON,取 `front_m`)
- 發:`/input/nav_cmd_vel`(Twist,送 mux)、`~/status`(String JSON,給 TUI)

### 6.2 主迴圈 `_tick_ctrl`(20Hz)
1. **看門狗**:odom 逾 0.3s/desired 逾 0.5s → hard-zero;障礙逾 1.0s → 視為無障礙(pass-through)
2. **ORCA**:`_build_orca_input`(v_pref = policy desired 經 `yaw_next` 轉 world)→ `ORCAFilter.solve` → `(v_safe_linear, v_safe_omega)`
3. **front_brake**(最後蓋過 ORCA):`front_m ≤ 0.5m` → v 鉗≤0(stop);`≤0.55m` emergency 兜底;`0.5-0.6m` 線性縮速(slow)
4. **slew/brake 輸出**:stop 走 `_publish_brake`(前進瞬間歸零);其他走 `_publish_slew`(accel×dt 限速)

### 6.3 安全網(與 VO 對比)
| 網 | ORCA 版 | VO 版(差異) |
|----|---------|--------------|
| 看門狗 | ✅(相同) | ✅ |
| front_brake | ✅ 簡化(純距離) | ✅ + defer_to_vo + block_ratio + reverse |
| slew | ✅(相同) | ✅ |
| escape(卡死倒車) | ❌(ORCA `lateral_evasion` 替代) | ✅ commit_side K 轉 |
| commit(繞行側鎖定) | ❌(ORCA 每次獨立求解) | ✅ |

### 6.4 首次測保守參數(已設 `config/orca_params.yaml`)
- `accel_v=0.6` / `accel_w=1.5`(原 VO 1.2/3.0 降半,確認 slew 平滑再放開)
- `front_brake_slow_m=1.0`(原 0.6,早介入壓制 ORCA 側向不確定性)
- `min_vel_confidence=0.0`(全收速度,信任 KF;防剛冒出快速闖入者漏避)

---

## 7. 踩過的坑(必讀)

1. **package.xml 註解不能有 `--`**:XML 禁止註解內 `--`。`<!-- ... --user ... -->` → ParseError → colcon 讀不到 package.xml → 識別成 `(python)` 非 `(ros.ament_python)` → 不產生 ament hook → ros2 找不到 package(但 build 成功 + entry point 能跑)。診斷:`python3 -c "import xml.etree.ElementTree as ET; ET.parse('package.xml')"` + `colcon list`。
2. **pyrvo2 PyPI 無 ARM wheel**:`pip install pyrvo2` 失敗。修:`git clone https://github.com/mit-acl/Python-RVO2` → `pip3 install --user --break-system-packages cython` → `pip3 install --user --break-system-packages --no-build-isolation .`(`--no-build-isolation` 必須)。需 cmake+gcc+python3-dev。
3. **Zenoh 環境**:不 source `setup_env.sh` → 節點跑但收不到 topic(預設 DDS 跟車端 Zenoh 不通)。
4. **VO 與 ORCA 不是替代關係(語義)**:VO 是 DWA 取樣+rollout;ORCA 是 RVO2 half-plane+LP。但 deploy 選單讓它們互斥切換(topic 對稱),可 A/B 比較。

---

## 8. 檔案地圖

```
~/rover_rl/src/orca_filter/
├── orca_filter/
│   ├── orca_core.py         # 純 ORCA solver(兩模式共用)
│   ├── bystander_node.py    # 旁觀節點(不控制車)
│   ├── orca_safety_node.py  # ★控制車節點(取代 vo_safety_node)
│   └── __init__.py
├── config/orca_params.yaml  # 兩節點參數(orca_bystander + orca_safety_node 區)
├── launch/bystander.launch.py
├── package.xml / setup.py / setup.cfg / resource/orca_filter
├── README.md
└── HANDOFF.md               # 本檔

~/rover_rl/src/rover_rl_bringup/launch/deploy_full.launch.py  # 加 enable_orca + orca_safety_node + policy 改道
~/rover_rl/deploy_select.sh      # 安全層 1=VO/2=ORCA/3=none(ORCA_ARG)
~/rover_rl/deploy_rl_shell.sh    # 加 "$ORCA_ARG",移除背景 bystander
~/rover_rl/deploy_rl.sh          # 加 "$ORCA_ARG"

備份:*.bak_orca_safety_20260708_173538
```

**不碰**:`vo_safety_node.py`、`vo_layer.py`、`policy_node.py`、`orca_core.py`(A/B 並存)。

---

## 9. ★車端 Claude Code 接手清單(實做下一步)

> PC-A 已做完程式碼(build 通過、節點啟動驗證 OK)。**未實車測試**。以下是車端 Claude Code 接手要做的。

### 9.1 實車測試(動車,建議用戶在場監督)

```bash
# 前提:lv-dot 在跑(障礙來源)+ 底盤 driver(/odom)
deploy_rl_shell        # 選 checkpoint → 安全層選 2(ORCA)
```

驗證清單(依序):
1. `ros2 node list | grep -E "orca_safety|vo_safety"` → 有 `orca_safety_node`、**無** `vo_safety_node`(互斥)
2. `ros2 topic info /input/nav_cmd_vel` → 只 1 個 publisher(orca_safety_node)
3. `ros2 topic echo /rover_rl_policy/status` 確認 `front_m` 有值(policy 在跑)
4. **空曠處**(無障礙):車正常走(ORCA pass-through,`orca_active=false`)
5. 人走動靠近車頭(2-3m)→ `ros2 topic echo /orca_safety_node/status` 看 `orca_active=true` + 車減速/轉向
6. 人站車頭 0.5m 內 → status `front_brake=stop` + 車停
7. A/B:選 1(VO)→ vo_safety 啟;選 2(ORCA)→ orca_safety 啟
8. 按 q 離開 TUI 自動收棧

**安全**:第一次 ORCA 控制車,先空曠處測 pass-through;準備 joy 急停;人站車頭測 front_brake 時保持距離。

### 9.2 debug 流程

看節點狀態:
```bash
ros2 topic echo /orca_safety_node/status   # orca_active/front_brake/fail/v_out/w_out
```
看 log:`~/rover_rl/logs/deploy_*.log`(grep INTERVENE / front_brake / fail)

常見問題:
| 症狀 | 可能原因 | 檢查 |
|------|---------|------|
| 車不動 | 看門狗 fail(desired/odom 逾時) | status `fail` 欄;確認 policy/odom 在跑 |
| ORCA 沒介入 | 障礙物靜態(被排除)/ 無障礙 / min_vel_confidence 太高 | status `orca_active`;echo tracked_obstacles 看速度 |
| 車亂動/晃 | ORCA lateral_evasion + slew 不穩 | 降 `accel_v`/`accel_w`;增 `front_brake_slow_m` |
| 撞動態物 | 反應太慢 / min_vel_confidence 漏掉 | 增 `time_horizon_dynamic`;`min_vel_confidence=0.0` |
| 撞靜態牆 | front_brake 沒觸發 | 確認 policy status `front_m` 有值;降 `front_brake_stop_m` |

### 9.3 調參指引(`config/orca_params.yaml`,改完重啟節點)
- 反應太慢 → `time_horizon_dynamic` 2.5→3.0
- 繞路太多 → `time_horizon_dynamic` 2.5→2.0
- 撞 → `safety_margin` 0.10→0.15 或 `obs_inflation` 1.8→2.0
- 走廊卡 → `time_horizon_static` 0.3→0.2
- 介入太頻繁 → `INTERVENTION_EPSILON`(orca_core.py)0.05→0.10
- 改 .py → `colcon build --packages-select orca_filter && source install/setup.bash`;改 .yaml 不用 rebuild

### 9.4 驗證通過後(放開保守參數)
首次測確認 ORCA 控制車平穩後,把 yaml 參數放回正常值:
- `accel_v` 0.6→1.2、`accel_w` 1.5→3.0(對齊 VO)
- `front_brake_slow_m` 1.0→0.6(對齊 VO)

---

## 10. 未來工作

- **ORCA 上線取代 VO**:A/B 測試 ORCA ≥ VO 後,把 deploy 預設改 ORCA(或移除 VO)
- **加 VO cone 可視化**:`orca_core` 已輸出 `vo_cone_data`,bystander_node 可加 RViz LINE_STRIP
- **加 escape**(若 ORCA 全堵死表現差):借 vo_safety_node escape 狀態機(需 ORCA result 加 blocked/feasible 欄位)
- **效能**:pyrvo2 20Hz 量測單拍 solve 耗時(>5ms 需優化 sim 重用)

---

## 11. 重要提醒

- **不碰** vo_safety_node / vo_layer / policy_node / orca_core(除非 §10 未來工作)
- 改 `orca_filter/*.py` → `colcon build --packages-select orca_filter` + source;改 `.yaml` 不用 rebuild
- pyrvo2 在 `~/.local`,重裝系統會掉(§7 坑 2)
- 車端 Claude:動 orca_filter 後務必 rebuild + source install/setup.bash
- PC-A 對應源碼(參考):`/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/play_eval/rvo2_safety_filter.py`
