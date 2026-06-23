# rover_rl — RL Policy 部署到 CampusRover 實車

> 把 IsaacLab 訓練好的 RL navigation policy（LiDAR sweep → RNN → `cmd_vel`）打包成
> **ROS 2 Humble** workspace，部署到 CampusRover 實車。
> **只做推論、不訓練、不改網路** — 訓練在 PC 端 [`/home/aa/IsaacLab`](https://github.com/me0608623/rover_rl)。

| | |
|---|---|
| **平台** | Jetson AGX Orin · Ubuntu 22.04 · ROS 2 Humble |
| **LiDAR** | Velodyne VLP-16（`/velodyne_points`, 10 Hz） |
| **定位** | NDT localizer（`map→odom` TF）+ 底盤 odom（`odom→base` TF） |
| **通訊** | `rmw_zenoh_cpp`，`ROS_DOMAIN_ID=55`，zenoh router hub |
| **最新模型** | `sa4_v3f_240000`（79D, 2026-06-22） |

---

## 1. 這是什麼

`rover_rl` 是一個**獨立的 ROS 2 workspace**，與既有 `rover2_ws` / `ndt_ws` 平行運作、不取代它。
職責只有一件事：**把訓練好的 policy 推論結果，安全地送到底盤**。

```
VLP-16 ─► lidar_preprocessor ─► 72-bin sweep ─► policy_node ─► (vo_safety) ─► /input/nav_cmd_vel
                                                                    │
   NDT/odom/goal/path ─────────────────────────► (組 obs, 5Hz RNN)   └─► lcr_cmd_vel_mux ─► /output/cmd_vel ─► 底盤
```

policy 的 obs 障礙欄位目前**補 0**（看不到動態物），動態障礙的預測避障交由夾在 policy 與底盤之間的
**VO 安全層**（`vo_safety_node`）處理。

---

## 2. 支援的模型（切勿混用設定！）

不同模型 obs 維度 / 校準不同，**換模型必須同步換對應 yaml**，否則 `raw_obs` 維度不符或行為失準。

| 模型 `.ts` | obs | hidden | r_min | 動作 ω_max | 訓練場景 | 對應 yaml |
|---|---|---|---|---|---|---|
| `sa6_tc_dense_420k.ts` ⭐ | 79 / 139D | 30 | 0.9 | 2.0 | T 走廊 dense 障礙 | `policy_params.yaml` |
| `sa7_tc_dense_300000.ts` | 79 / 139D | 30 | 0.9 | 2.0 | T 走廊高壓（occlusion + 8 dyn） | `policy_params.yaml` |
| `sa5_tc_g1_p30_270000.ts` | 79 / 139D | 30 | 0.9 | 2.0 | T 走廊 g1_p30（早期） | `policy_params.yaml` |
| `sa3_v3c_240000.ts` | **83D** | 64 | **0.25** | 0.785 | 牆壁+crossing，action stacking | `policy_params_v3c.yaml` |
| `sa2_v3e_240000.ts` / `sa1_v3e_60000.ts` | **83D** | 64 | **0.25** | **1.2** | v3e 延遲感知重訓 | `policy_params_v3e.yaml` |
| `sa4_v3f_240000.ts` 🆕 | **79D** | 64 | **0.25** | **1.2** | SA4 空間規劃最終（act_hist 移除） | `policy_params_v3f.yaml` |

- **三個 ω 常數別混**：obs[2] 正規化 = **1.5**（全部模型）、動作上限 = 上表值、act_stack 正規化 = π/15。
- obs 佈局（79D）：`ego(4) + goal(2) + lidar(72) + time(1)`；83D 多 `action_stack(4)`；139D 多 60D 訓練用障礙（部署補 0）。
- `.ts` 模型檔由 PC 端 scp、**不入庫**（`.gitignore` 的 `models/*.ts`）；`models/*.obs_spec.{json,md}` 規格檔入庫。
- v3 系列部署細節見 [`V3C_DEPLOY.md`](V3C_DEPLOY.md) / [`V3E_DEPLOY.md`](V3E_DEPLOY.md)。

### 換模型（不需重啟節點）

```bash
ros2 param set /rover_rl_policy model_path /home/aa/rover_rl/models/<new>.ts
ros2 service call /rover_rl_policy/load_model std_srvs/srv/Trigger
```

---

## 3. 系統架構

### 3.1 完整資料流

```
        velodyne_driver ── /velodyne_points (10Hz)
              │
   ┌──────────┴──────────┐
   ▼                     ▼
 lidar_preprocessor    bev_play_node        (純 debug 可視化)
   │                     └─► /rover_rl/bev_image
   └─► /rover_rl/lidar_sweep_72 (Float32MultiArray[72], 10Hz)
              │
              ▼
        policy_node  ◄── /odom, /goal_pose, /global_path, /ndt_pose(活性), TF map→base
        (5Hz RNN 推論 → low-pass+slew → 20Hz target)
              │
              │  VO 開：/rover_rl/cmd_vel_desired
              │  VO 關：直接 /input/nav_cmd_vel
              ▼
   (vo_safety_node)  ◄── /vo_interface/tracked_obstacles (動態障礙追蹤)
        (DWA rollout 預測避障 + 卡死逃脫)
              │
              └─► /input/nav_cmd_vel ─► lcr_cmd_vel_mux ─► /output/cmd_vel ─► 底盤
```

### 3.2 TF tree（運行時）

```
world ─(static)─ map ─(NDT 動態)─ odom ─(底盤 driver)─ base_link ─(URDF)─ base_footprint / velodyne_link
```

> ⚠ 本機 NDT 的 `/ndt_pose` 內容 == `map→odom` TF（**不是車姿**）。車姿走 **TF `map→base_footprint`**
> （`tf2_buffer.lookup_transform` 取最新可用，跨機時鐘安全）。詳見 [`CLAUDE.md`](CLAUDE.md) NDT 段。

### 3.3 cmd_vel 鏈與 VO

| VO 狀態 | policy 發到 | 經 vo_safety | 最終 |
|---|---|---|---|
| 關 (`enable_vo:=false`) | `/input/nav_cmd_vel` | — | → mux → 底盤 |
| 開 (`enable_vo:=true`) | `/rover_rl/cmd_vel_desired` | → 預測避障濾波 → | `/input/nav_cmd_vel` → mux → 底盤 |

VO 沒有動態障礙來源時會退化為「放行 + ω 限幅」（正常安全退化）。

---

## 4. 套件與節點

### `rover_rl_inference`（核心，13 個 executable）

| Executable | 職責 |
|---|---|
| `policy_node` | 主推論：72-bin sweep + odom/goal → 79D obs → RNN → `cmd_vel`（5 Hz 推論 / 20 Hz cmd） |
| `lidar_preprocessor` | PointCloud2 → 72-bin 正規化 sweep（獨立節點，與 policy 解耦） |
| `vo_safety` | VO 安全層：動態障礙做 DWA 預測避障 + 卡死後退逃脫 |
| `bev_play` | BEV 極座標可視化 → `/rover_rl/bev_image`（純 debug） |
| `diag_logger` | 被動診斷：訂閱各 topic 逐列寫 CSV（每個 goal/path 一段） |
| `analyze_diag` | 離線分析 diag CSV（速度三層對比、延遲、晃動報告） |
| `status_tui` | 繁中 curses 即時儀表板（三層速度 + 延遲 + RNN + 動態障礙 + LiDAR 雷達） |
| `routing_to_path` | campusrover_routing service → `/global_path` 橋接（2Hz republish + 站序） |
| `routing_click_bridge` | RViz「Publish Point」兩點 → 呼叫 routing service |
| `ros_smoke_test` | 上電前 ROS 通訊冒煙測試（不需 torch） |
| `export_policy` | 訓練 checkpoint → 部署用 TorchScript（含 normalizer，auto-detect 架構） |

### 其他套件

| 套件 | 用途 |
|---|---|
| `rover_rl_bringup` | launch + config（`deploy*.launch.py`、`config/*.yaml`） |
| `vo_interface` | 動態障礙追蹤介面（CV-Kalman 重追蹤，輸出持久 ID / 平滑速度） |
| `jie_deamon_deploy` | Web 地圖導航（`jie_deamon`，顯示機器人位置） |

---

## 5. 啟動方式

### 5.1 三種 launch

| Launch | 用途 | 含哪些節點 |
|---|---|---|
| `deploy.launch.py` | 最簡：只跑 policy | policy_node |
| `deploy_with_bev.launch.py` | 標準：policy + preprocessor + BEV | preprocessor + policy + bev_play |
| `deploy_full.launch.py` | **完整棧**：+ NDT + routing + costmap + MOT + RViz | 上面全部 + NDT + routing 橋 + costmap + MOT |

### 5.2 Alias（`.bashrc` 已設定，最快）

| Alias | 行為 | 適用 |
|---|---|---|
| `deploy_rl` | 前景純 `ros2 launch deploy_full`（滾動 log），**預設不開 NDT，`enable_vo:=true`** | **Claude / 非互動 shell** |
| `deploy_rl_shell` | 背景 launch + 前景繁中 TUI 儀表板，launch 前互動選 model + 啟用 VO | **人（真實終端機）** |
| `deploy_rl_stop` | 停止整個棧 | — |
| `deploy_all` | `deploy_full` 完整棧 + 動態障礙偵測，全部背景啟動 | 一次全開 |

```bash
# 人用（真實終端機）：互動選 model + VO，背景 launch + 前景 TUI
deploy_rl_shell initial_mode:=idle

# Claude / 非互動：純前景 launch（跳過選單走預設）
deploy_rl initial_mode:=nav

# 一次全開（NDT + RL + 動態偵測都背景）
deploy_all initial_mode:=idle
```

> ⚠ `deploy_rl` 預設不開 NDT（用 `ndt` alias 分開啟）；要一鍵全開用 `deploy_all`。

### 5.3 首次部署（完整步驟見 [`DEPLOY_CAMPUSROVER.md`](DEPLOY_CAMPUSROVER.md)）

```bash
# 1. 環境（zenoh router 必須先在線）
systemctl status zenoh-router.service
source ~/rover_rl/setup_env.sh    # RMW=zenoh, DOMAIN=55

# 2. Build（campusrover_msgs 依賴要先 source rover2_ws）
source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash
cd ~/rover_rl && colcon build --symlink-install \
  --packages-select rover_rl_inference rover_rl_bringup
source install/setup.bash

# 3. 放模型到 ~/rover_rl/models/（PC 端 scp；.ts 不入庫）
# 4. 啟動（建議首次 initial_mode:=idle，架空車輪 + estop 待命）
deploy_rl_shell initial_mode:=idle
```

---

## 6. 多 Mode 與控制

### 5 種 mode

| Mode | 推論 | cmd_vel | 用途 |
|---|---|---|---|
| `nav`（預設） | ✓ | policy 輸出 | 正常運作 |
| `idle` | ✗ | 強制 0 | 待命 |
| `estop` | ✗ | 強制 0 | 緊急停車 |
| `manual` | ✗ | 讓出 topic | 搖桿接管 |
| `paused` | ✗ | 強制 0 | 暫停但保留 RNN hidden |

```bash
# 切 mode（topic 訂閱式最方便）
ros2 topic pub --once /rover_rl_policy/mode std_msgs/String "data: 'estop'"
# service 式
ros2 service call /rover_rl_policy/set_mode std_srvs/srv/SetBool "{data: false}"   # idle
ros2 service call /rover_rl_policy/set_mode std_srvs/srv/SetBool "{data: true}"    # nav
# 重置 RNN hidden
ros2 service call /rover_rl_policy/reset_hidden std_srvs/srv/Trigger
```

### Timer 結構

| Timer | Rate | 工作 |
|---|---|---|
| inference | 5 Hz (`control_dt=0.2`, **訓練值勿改**) | RL 推論 → 更新 target |
| cmd | 20 Hz | low-pass(α) + slew-rate + deadband → 發 cmd_vel |
| marker | 10 Hz | RViz markers |
| heartbeat | 0.5 Hz | 系統狀態 log |

---

## 7. VO 安全層

夾在 policy 與底盤之間的預測式避障濾波（`vo_safety_node`）。policy 的 obs 障礙欄補 0、
看不到動態物，故動態障礙（如走動的人）的閃避由 VO 處理。

- **障礙來源**：訂閱 `/vo_interface/tracked_obstacles`（持久 ID / 平滑速度 / 協方差）。
- **演算法**：DWA 式 rollout，障礙線性外推；可行弧線中選最貼近 RL 意圖 + 朝 goal 推進者。
- **實車調校真值**（`config/vo_params.yaml`）：
  - `margin 0.30`（至少距人 30cm 才繞）、`obstacle_radius_max 0.6`（封偶發假巨框）、
    `w_goal 0.6`（DWA 進展獎勵主動繞）、`stuck_escape`（堵死 + 後方淨空 → 慢退開空間）。
- **熱調**：`ros2 param set /vo_safety_node w_goal 0.4`；status 見 `/vo_safety_node/status`。

---

## 8. 診斷與分析

| 工具 | 用途 |
|---|---|
| `status_tui` | 即時儀表板：模式 / 三層速度(想/送/實) / 延遲 / RNN hidden / 動態障礙 / VO 介入 / 右側 LiDAR 雷達 |
| `diag_logger` | 被動寫 CSV，**每個 goal/path 一段獨立紀錄**（到終點自動停 + 自動 re-arm）。資料在 `~/rover_rl/logs/diag/diag_<時間>/` |
| `analyze_diag` | 離線報告：速度三層對比、延遲、goal 型態×距離×速度、晃動歸因 |

```bash
# 分析最新一次測試
ros2 run rover_rl_inference analyze_diag "$(ls -td ~/rover_rl/logs/diag/*/ | head -1)"/*.csv
# 即時看狀態（JSON）
ros2 topic echo /rover_rl_policy/status
```

> `diag_logger` 的 `wandb_mode` 一律用 **`offline`**（online 會阻塞單執行緒 executor 卡死記錄），回頭再 `wandb sync`。

---

## 9. 訓練 ↔ 部署對齊保證

LiDAR sweep 公式兩端**完全相同**，避免 sim-to-real obs mismatch：

| 處理 | 訓練端 (`obs_functions.py`) | 部署端 (`lidar_preprocess.py`) |
|---|---|---|
| z_filter | `\|z\|>0.5` 過濾 | 同 |
| 角度 binning | `atan2(y,x)` → 72 bins | 同 |
| Min-pool | 同 bin 取最小距離 | 同 |
| 正規化 | `(d - r_robot)/(r_max - r_robot)` | 同 |
| obs normalizer | running mean/var | **baked into `.ts`** |

任何 mismatch → policy 表現必然不如預期。

---

## 10. Sim-to-Real 已知 gap

1. **LiDAR 高度**：訓練 1.6m vs 實車 1.43m（beam 落點不同）
2. **ω 上限**：SA5/6/7 訓練 ω_max=2.0 > 底盤 1.2（全力轉實際比 policy 預期慢 40%）；v3e/f 已校正到 1.2
3. **wheel 不對稱 1.9%** → odom drift ~1%/m
4. **訓練場景 vs 實車任意場景**：泛化性未完全驗證（SA5/6/7 是 T 走廊，v3 是牆壁+crossing）
5. **cmd 死時間**：實車 goal-following 曾出 0.4Hz「舞龍舞獅」振盪（底盤 ω 死時間 ~0.2s）。`policy_node` 有 `cmd_delay_comp_s` 補償；v3e/f 已做延遲感知重訓
6. **底盤 deadband**：policy 微小輸出可能低於馬達死區 → 觀察到「policy 有輸出但輪子靜止」需量測實際死區

詳見 [`docs/`](docs/) 的測試筆記與 [`CLAUDE.md`](CLAUDE.md)。

---

## 11. 安全條款（絕對遵守）

1. **第一次跑：架空車輪 + 遙控器 deadman 隨時待命**
2. **首次用 `initial_mode:=idle`**，確認 cmd_vel 為 0 後再切 `nav`
3. **動作上限不可放寬**：即使底盤能更快，policy 沒見過 OOD 速度會 extrapolate 失敗
4. **policy 異常 = 立刻 E-stop**，不要「再跑一下看看」
5. **任何 cmd_vel 超出 `[-1, 1] m/s` 或 `[-2, 2] rad/s`** = 通訊/匯出錯誤，停車檢查
6. **`/rover_rl/lidar_sweep_72` 全 = 1.0** = LiDAR 沒讀到任何點 → 別上電
7. **改 yaml 前先備份**：`cp policy_params.yaml policy_params.yaml.bak`

---

## 12. 故障排除速查

| 症狀 | 檢查 / 修復 |
|---|---|
| `ros2 topic list` 無回應 | zenoh router 沒連上：`systemctl status zenoh-router`，先解決再啟節點 |
| policy_node 啟動 crash | 99% 是 `model_path` 錯或 torch 沒裝；看 launch log |
| 永遠輸出 0 cmd | LiDAR/odom 訊號中斷：`ros2 topic hz /velodyne_points` `/odom` |
| `sweep_src=inline_fallback` | preprocessor 節點死了：`ros2 topic hz /rover_rl/lidar_sweep_72`，重啟 preprocessor |
| goal 方位/距離全歪 | 確認 `pose_src=tf`、`pose_x/y` 與 `tf2_echo map base_footprint` 一致 |
| 車原地震 | normalizer 維度與 model 不符（79/83/139 互錯，如 v3c 用了 SA6 yaml） |
| cmd 振幅異常大 | normalizer 沒 bake 進 model，重新 `export_policy.py` |
| RViz 沒顯示 | `deploy_full` 引用本機 `/home/aa/rviz/demo.rviz`（未入庫），換機須手動複製 |

更多見 [`CLAUDE.md`](CLAUDE.md) 故障排除表。

---

## 13. 相關文件

| 文件 | 內容 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | **最權威最新指南**（車上 Claude 用，含所有節點/參數/調校/gap 全記錄） |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 完整架構（節點 / topic / service / TF 一覽，SA6 時代但結構仍參考） |
| [`V3C_DEPLOY.md`](V3C_DEPLOY.md) / [`V3E_DEPLOY.md`](V3E_DEPLOY.md) | v3c(83D) / v3e(83D) 模型部署細節 |
| [`DEPLOY_CAMPUSROVER.md`](DEPLOY_CAMPUSROVER.md) | CampusRover 實車首次部署完整步驟 |
| [`RNN_HANDLING.md`](RNN_HANDLING.md) | RNN hidden state 生命週期處理 |
| [`docs/`](docs/) | 測試筆記、開發日記、啟動指令 |

PC 端訓練 repo：`/home/aa/IsaacLab/rover_rl`（需重訓 model / 匯出新 `.ts` / 改 obs 規格時找 PC 端訓練組）。
