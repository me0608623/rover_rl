# orca_filter — ORCA (RVO2) 安全過濾器(車端部署版)

從 IsaacLab `play_eval/rvo2_safety_filter.py` 抽取的純 ORCA 核心,**無 Isaac Lab 依賴**,
可獨立在車端 ROS2 humble 上跑。**只新增此 package,不動 vo_interface、不動 rover_rl 其他代碼。**

## 設計:non-cooperative(行人不會避讓機器人)

ORCA 原生假設「雙方各退一半」(reciprocal),但實車場景行人不會為機器人減速。故:

| 參數 | 值 | 作用 |
|------|-----|------|
| `ego_responsibility` | 1.0 | 機器人扛 100% 避讓,障礙物 `max_speed≈0`(不會減速讓你) |
| `obs_inflation` | 1.8 | 算碰撞時障礙半徑 ×1.8,提早反應補償行人不讓 |

只處理**動態**障礙物;靜態(|vel|<0.15)完全排除,留給 policy/LiDAR。

## 依賴

- ROS2 humble
- `vo_interface`(提供 `TrackedObstacleArray` msg)
- `pyrvo2`(mit-acl/Python-RVO2,已 `pip install --user` 裝在 `~/.local`)
- numpy

pyrvo2 安裝(若重裝):
```bash
cd /tmp && git clone https://github.com/mit-acl/Python-RVO2
cd Python-RVO2
pip3 install --user --break-system-packages cython
pip3 install --user --break-system-packages --no-build-isolation .
python3 -c "import rvo2; print('OK')"  # 驗證
```

## 檔案

| 檔案 | 說明 |
|------|------|
| `orca_filter/orca_core.py` | 純 ORCA solver(`ORCAFilter.solve`),1:1 移植自 rvo2_safety_filter.py |
| `orca_filter/bystander_node.py` | 旁觀測試節點:訂 LV-DOT + odom → ORCA → 印 + RViz |
| `config/orca_params.yaml` | 所有參數(預設 = PC-A 1:1) |
| `launch/bystander.launch.py` | 啟動旁觀節點 |

## Build

```bash
cd ~/rover_rl
colcon build --packages-select orca_filter
source install/setup.bash
```

## 跑(旁觀測試,不控制車)

```bash
# 1. 先啟動 vo_interface(發 /vo_interface/tracked_obstacles)+ 確認 /odom 有發
# 2. 啟動 ORCA 旁觀節點
ros2 launch orca_filter bystander.launch.py
```

RViz 看:
- **綠箭頭** = v_pref(機器人想要的速度)
- **紅箭頭** = v_safe(ORCA 建議,介入時才畫)
- **黃球** = 動態障礙物位置;灰球 = 靜態

終端會印介入訊息:
```
[INTERVENE #1] v_pref=(+0.80,+0.00) v_safe=(+0.62,+0.31) → body v=+0.62 ω=+0.45 nearby=2(dyn2)
```

## v_pref 策略(bystander 無 policy 輸出)

bystander 不接 policy,故用參數模擬「意圖速度」:
- `forward`(預設):固定前進 `v_pref_speed`,測「直走遇到障礙時 ORCA 怎麼閃」
- `current`:用當前 odom 速度,測「ORCA 對當前狀態的修正」

## 未來:接進 policy_node(會碰 rover_rl)

bystander 驗證 ORCA 算得合理後,要真的控制車需把 `ORCAFilter.solve` 插進
`rover_rl_inference/policy_node.py` 的 `_tick_inference`(policy 推論輸出後、存 `_target_v/w` 前),
v_pref 改用 policy 真實輸出。那一步會修改 rover_rl,須另行評估。

## 參數調整

見 `config/orca_params.yaml` 註解。重要:
- 反應太慢 → 增 `time_horizon_dynamic`(2.5→3.0)
- 繞路太多 → 減 `time_horizon_dynamic`(2.5→2.0)
- 撞到 → 增 `safety_margin`(0.10→0.15)或 `obs_inflation`(1.8→2.0)
- 走廊卡住 → 減 `time_horizon_static`(0.3→0.2)
