# rover_rl — 給車上 Claude Code 的指南

> 此 repo 由 PC 端訓練組（/home/aa/IsaacLab）匯出，部署到 CampusRover 實車。
> 您（車上的 Claude）的任務：**驗證 ROS 通訊運作正常，然後協助首次上電部署**。

> **最後更新：2026-06-02** — 已實作 routing_to_path / routing_click_bridge / bev_play_node；
> deploy_full.launch.py 可一次啟動完整棧（NDT + routing + RL policy）。
> 技術細節參見 `ARCHITECTURE.md`。

## 語言偏好

- 所有回覆、log、註解一律 **繁體中文**
- 程式碼/變數/topic 名稱保持英文

## 此 repo 的角色

把 IsaacLab 訓練好的 RL policy（PointCloud2 → 79D obs → RNN → cmd_vel）
打包成 ROS 2 workspace。**不訓練、不修改網路、只做推論**。

### 模型架構（已固定，不要動）

```
obs_raw [B, 139] ── normalize + slice → obs79 [B, 79]
                                          ├─→ LidarStateExtractor (Conv1d) → feat [B, 96]
                                          ├─→ PreprocessRNN(hidden=30) → preprocess [B, 12]
                                          │     └─ 維護 RNN hidden state（episode 內延續）
                                          └─→ cat(obs79, preprocess) → [B, 91]
                                                └─→ PolicyHead → logits [B, 38]
                                                      └─→ argmax → MultiDiscrete([19, 19])
                                                            └─→ cmd_vel (linear, angular)
```

- **raw_obs_dim=139**（SA5/6/7 T-Corridor）— 60D 障礙物欄位部署時補 0
- **5 Hz 控制週期**（dt=0.2s）— 訓練值，**勿改**
- **動作上限**：v=1.0 m/s, a=0.5 m/s², ω=2.0 rad/s — **不要 extrapolate** 即使底盤更強

### ⭐ v3c (SA3_v3c) 新模型 — 2026-06-12 加入（83D，與 SA6 不同！）

> **完整說明見 `V3C_DEPLOY.md`。** 這裡是給車端 Claude 的速覽，避免誤用 SA6 設定跑 v3c。

PC 端新增 **SA3_v3c** 模型，與 SA6 (79/139D) **不相容**，靠「換模型 + 換 yaml」切換，
**SA6 路徑完全保留、行為 byte 不變**（不要為了 v3c 去改 SA6 的預設 yaml 或 79/139 程式分支）。

| 項目 | SA6（舊，預設） | **v3c（新）** |
|------|------|------|
| raw_obs_dim | 79 / 139 | **83**（79D + 4D action stacking） |
| RNN hidden | 30 | **64** |
| LiDAR r_min | 0.9 | **0.25**（VLP-16 表面→人物中心校準） |
| 角速度上限 | 2.0 | **0.25π≈0.785** |
| 角速度 α slew | 無 | **3.0 rad/s²**（防舞龍舞獅） |
| 模型 .ts | `sa6_tc_dense_420k.ts` | `sa3_v3c_240000.ts` |
| yaml | `policy_params.yaml` | `policy_params_v3c.yaml` |
| preprocessor yaml | `lidar_preprocessor_params.yaml` | `lidar_preprocessor_params_v3c.yaml` |

83D obs 佈局：`[0:4]ego [4:6]goal [6:78]lidar(72) [78]time [79:83]action_stack`。
其中 **state_mlp 分支=11D**（ego4+goal2+time1+act_hist4）、**lidar 分支=72D**。

**action_stack（[79:83]）不是硬體量測**，是 policy 自己上 2 步 `decode_logits_to_cmd`
回傳的 `(actual_accel, cmd_w)`（cmd_w 已過 slew），`policy_node` 內 `deque(maxlen=2)`
推論後 push、停車/新 goal 清空。程式已接好，車端不需動。

**啟動 v3c**（覆寫兩個 params_file）：
```bash
ros2 launch rover_rl_bringup deploy_with_bev.launch.py \
  params_file:=$(ros2 pkg prefix rover_rl_bringup)/share/rover_rl_bringup/config/policy_params_v3c.yaml \
  preprocessor_params_file:=$(ros2 pkg prefix rover_rl_bringup)/share/rover_rl_bringup/config/lidar_preprocessor_params_v3c.yaml \
  initial_mode:=idle
```
模型放在 `~/rover_rl/models/sa3_v3c_240000.ts`（已由 PC 端 scp，gitignore 不入庫）。

**v3c 注意事項**：
- v3c yaml `speed_rate=1.0`（action_stack 與 slew 僅 rate=1 精確對齊）；首次上電靠
  「架空車輪 + estop + 保守 goal」確保安全，**不要靠降 speed_rate**。
- `episode_horizon_s=60.0` 為暫沿用值，只影響 obs[78] time-ramp，非安全關鍵，待驗證。
- 訓練端 `modular_rnn_models.py` / `export_policy.py` 已支援 83D 自動偵測；重匯出新
  checkpoint 不用改程式。

## LiDAR 前處理：獨立節點架構（採 spot_rl/spot_obs_process.cpp pattern）

**重要**：rover_rl 把 LiDAR 前處理切成**獨立節點**，與 RL 推論節點解耦。
這跟 spot_rl 的設計一致（spot_obs_process.cpp 是獨立節點，AI 訂閱處理後的 obs）。

```
/velodyne_points (raw PointCloud2, 10Hz)
       │
       ▼
┌────────────────────────────────┐
│ rover_rl_lidar_preprocessor    │
│  • PointCloud2 → 72-bin sweep  │   (對齊訓練 wd_like_sweep_72)
│  • r_min=0.9, r_max=20         │
│  • z_filter=0.5                │
│  • motion compensation         │
└────────────────────────────────┘
       │
       ├─→ /rover_rl/lidar_sweep_72  (Float32MultiArray[72], 正規化 [0,1])
       │      ↓
       │      ▼
       │   policy_node 訂閱此 topic → RL 推論
       │
       └─→ /rover_rl/lidar_scan      (LaserScan, 可選, 給 RViz/costmap)
```

### 為何要切獨立節點？

1. **可驗證**：`ros2 topic echo /rover_rl/lidar_sweep_72` 直接看處理結果
2. **獨立除錯**：preprocessor 對不對是一回事，policy 對不對是另一回事
3. **複用**：其他 policy / 視覺化都可訂閱同一個 sweep
4. **匹配 spot_rl 架構**：spot 是 `spot_obs_process.cpp` 獨立節點發 `RL_body_obs`，
   `ai_model_action.py` 訂閱 — 同樣 pattern

### 啟動方式

**3 種 launch 對應 3 種場景**：

| Launch 檔 | 用途 | 含哪些節點 |
|---|---|---|
| `deploy.launch.py` | 最簡：只跑 policy | policy_node 單一節點 |
| `deploy_with_bev.launch.py` | 標準：policy + preprocessor + BEV | preprocessor + policy + bev_play |
| `deploy_full.launch.py` | 完整棧：rover_rl + campusrover | 上面三個 + NDT + routing + costmap + MOT + RViz |

```bash
# 標準部署（NDT 由用戶另行啟動）
ros2 launch rover_rl_bringup deploy_with_bev.launch.py
# 內含：
#   - lidar_preprocessor   (/velodyne_points → /rover_rl/lidar_sweep_72)
#   - policy_node          (訂閱 sweep → /input/nav_cmd_vel)
#   - bev_play_node        (→ /rover_rl/bev_image, matplotlib headless)

# 完整棧（含 NDT + routing，一鍵全啟）
source ~/rover_rl/setup_env.sh   # ROS_DOMAIN_ID=30, RMW=fastrtps 等
ros2 launch rover_rl_bringup deploy_full.launch.py initial_mode:=idle
```

### 切換成「policy_node inline 處理」（不推薦，但保留）

```yaml
# policy_params.yaml
use_inline_preprocess: true   # policy_node 自己訂閱 /velodyne_points 並處理
```

```bash
ros2 launch ... enable_preprocessor:=false   # 關閉獨立 preprocessor
```

### 驗證 preprocessor 正常

```bash
# 1. 看處理後的 sweep 數值（應介於 0~1，0=非常近，1=>=20m 或無回波）
ros2 topic echo /rover_rl/lidar_sweep_72 --once | head -20

# 2. 用 RViz 看 LaserScan
ros2 run rviz2 rviz2
# Add Display → LaserScan → topic = /rover_rl/lidar_scan

# 3. 確認頻率
ros2 topic hz /rover_rl/lidar_sweep_72   # 預期 ~10 Hz
```

### 訓練端 vs 部署端對齊

兩邊用**完全相同**的 sweep 公式（避免 sim-to-real obs mismatch）：

| 處理步驟 | 訓練端（obs_functions.py） | 部署端（lidar_preprocess.py） |
|---|---|---|
| 輸入 | Isaac Sim ray_hits_w | VLP-16 PointCloud2 |
| z_filter | `\|z_hit\| > 0.5` 過濾 | 同 |
| 角度 binning | `atan2(y, x)` → 72 bins | 同 |
| Min-pool | 同 bin 取最小距離 | 同 |
| 正規化 | `(d - r_robot) / (r_max - r_robot)` | 同 |
| r_min | 0.9 (VLP-16 盲區) | 0.9 |
| r_max | 20.0 | 20.0 |
| r_robot | 0.35 (body_radius) | 0.35 |

## 多 Mode 設計 + Path 整合（採 spot_rl pattern）

### 支援的 5 種 mode

| Mode | 行為 | cmd_vel 發布 | 推論 |
|---|---|---|---|
| `nav` | 正常運作（預設） | yes（policy 輸出） | yes |
| `idle` | 待命 | yes（強制 0） | no |
| `estop` | 緊急停車 | yes（強制 0） | no |
| `manual` | 外部接管（搖桿） | **no**（讓出 topic） | no |
| `paused` | 暫停推論但保留 hidden | yes（強制 0） | no |

### 切換方式（3 種）

```bash
# A. ROS topic 訂閱式（最方便）
ros2 topic pub --once /rover_rl_policy/mode std_msgs/String "data: 'estop'"

# B. ROS service（適合 supervisor 整合）
ros2 service call /rover_rl_policy/set_mode std_srvs/srv/SetBool "{data: false}"  # → idle
ros2 service call /rover_rl_policy/set_mode std_srvs/srv/SetBool "{data: true}"   # → nav

# C. launch param
ros2 launch rover_rl_bringup deploy_with_bev.launch.py initial_mode:=idle
```

### Hot-swap model (不需重啟 node)

```bash
ros2 param set /rover_rl_policy model_path /path/to/new_model.ts
ros2 service call /rover_rl_policy/load_model std_srvs/srv/Trigger
```

### Path / Subgoal 整合

policy_node 同時訂閱兩個來源：

| Topic | Type | 用途 | 來源 |
|---|---|---|---|
| `/goal_pose` | `geometry_msgs/PoseStamped` | 單一目標（RViz Nav2 Goal） | 手動 / Nav2 |
| `/global_path` | `nav_msgs/Path` | 多 waypoint 路徑 | **campusrover_routing** / AIT* |

**Subgoal 選取邏輯**（rover_rl_inference/subgoal_selector.py）：
1. 收到 Path 時 `prefer_path=True`（覆蓋 single goal）
2. 從 path 找離機器人最近的 waypoint
3. 往前找第一個距離 ≥ `path_lookahead_m`（預設 2m）的 waypoint = carrot
4. 走到尾還沒達 lookahead → 取最後一點作為 final goal

**campusrover_routing 整合**：routing 是 service-based（`RoutingPath.srv`），**已有橋接節點**：

```
RViz Publish Point (兩點)
      │
      ▼
routing_click_bridge      ← 點第1次選起點，點第2次選終點 → 自動呼叫 routing
      │ srv call
      ▼
routing_to_path_node  ── generation_path svc ──→ pick routing[0] ──publish──→ /global_path ──→ policy_node
```

或手動呼叫 service：
```bash
ros2 service call /rover_rl/routing_call campusrover_msgs/srv/RoutingPath \
  "{origin: 'c1', destination: ['e0']}"
# 結果自動 publish 到 /global_path
```

## TF tree 與啟動順序（重要）

實車跑起來時完整 TF chain：

```
world ─(static, 由 ndt_localizer/tf_static)─ map
                                                │
map ─(動態, 由 ndt_localizer_node 發布)── odom
                                                │
odom ─(由 campusrover_base driver 發布)── base_link
                                                │
base_link ─(URDF static)─ base_footprint / velodyne_link / imu_link
```

各段負責方：

| TF 段 | 發布者 | 啟動方式 |
|---|---|---|
| world → map | static_transform_publisher | NDT launch 自帶 |
| **map → odom** | ndt_localizer_node | `ros2 launch ndt_localizer ndt_localizer_launch.py` |
| odom → base_link | rover driver | campusrover_base launch |
| base_link → child | URDF | robot_state_publisher |

**⚠ 重要修正（2026-06-04）：本機 NDT 的 `/ndt_pose` 是 `map→odom` 變換、不是車姿。**
這顆 `ndt_ws` 的 `ndt_localizer` 發的 `/ndt_pose` 內容 == `map→odom` TF（實測），
真實車姿 = `map→odom ∘ odom→base`，差一個 `odom→base`。早期把 `/ndt_pose` 當車姿 →
車離 odom 原點越遠、goal 方位/距離越歪（曾出現點正前方 3.5m 卻算成 7.3m/-113°）。

**現行 pattern（policy_node `_robot_pose_in_map`）**：
- 車姿走 **TF `map→base_footprint`**（`tf2_buffer.lookup_transform(goal_frame, base_frame, Time())`，
  用 Time() 取最新可用、避免跨機時鐘不同步）—— 與 RViz 同一條鏈，由 tf2 正確合成 map→odom∘odom→base。
- `/ndt_pose` 僅用於 **NDT 活性判定**（is_ndt_stable / ndt_age，基於訊息到達 monotonic 時間，跨機可靠）。
- body-frame goal：`goal_body = R(-yaw) · (goal_map - robot_map)`（robot_map 來自上面 TF）。
- Fallback：TF 查不到 → odom_only 來源（`require_ndt: false` 時允許）。
- status/HB 的 `off_x/off_y/off_yaw` = 真實 `map→odom` TF；`pose_src` 正常為 `"tf"`。

驗證：`pose_src` 應為 `tf`、`pose_x/y` 與 `ros2 run tf2_ros tf2_echo map base_footprint` 一致；
點正前方 goal → `goal_ang_deg ≈ 0`。

（舊註：`localization.MapOdomOffsetTracker` 的 cached-offset 車姿計算現已停用、改走 TF；
其 `offset` 套用未旋轉 odom 位移的潛在 bug 也因此繞過。若日後換成「/ndt_pose 真的是車姿」的
NDT，再評估是否回退。）

### 啟動順序

**方案 A：deploy_with_bev（分步啟動，推薦首次部署）**
```bash
# Terminal 1: NDT 定位
cd ~/Documents/ndt_ws && source install/setup.bash
ros2 launch ndt_localizer ndt_localizer_launch.py

# Terminal 2: 底盤 driver（/odom + odom→base_link TF）
cd ~/rover2_ws && source install/setup.bash
ros2 launch campusrover_base <existing_launch>.py

# Terminal 3: VLP-16 driver
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py

# Terminal 4: rover_rl（preprocessor + policy + bev）
cd ~/rover_rl && source install/setup.bash
source ~/rover2_ws/install/setup.bash   # campusrover_msgs 依賴
source setup_env.sh                     # ROS_DOMAIN_ID + RMW
ros2 launch rover_rl_bringup deploy_with_bev.launch.py initial_mode:=idle
```

**方案 B：deploy_full（一鍵完整棧）**
```bash
# Terminal 1: 底盤 driver
cd ~/rover2_ws && source install/setup.bash
ros2 launch campusrover_base <existing_launch>.py

# Terminal 2: VLP-16 driver
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py

# Terminal 3: 完整棧（包含 NDT + routing + RL + BEV）
cd ~/rover_rl && source install/setup.bash
source ~/rover2_ws/install/setup.bash
source setup_env.sh
ros2 launch rover_rl_bringup deploy_full.launch.py initial_mode:=idle
```

### Timer 結構（採 spot_rl 多 timer pattern）

| Timer | Rate | 工作 |
|---|---|---|
| inference_timer | 5 Hz (control_dt=0.2) | RL 推論 → 更新 target cmd |
| cmd_timer | **20 Hz** (cmd_rate_hz) | low-pass + slew-rate → 發 cmd_vel |
| marker_timer | 10 Hz (marker_rate_hz) | 發 RViz markers |
| heartbeat_timer | 0.5 Hz | log 系統狀態 |

**為何分開？**
- Inference 5 Hz 固定（訓練 dt=0.2s），但 cmd_vel 20 Hz republish 避免底盤/mux watchdog
- Cmd timer 用 last_target + filter，inference 暫時延遲也不會斷流
- Marker 10 Hz 對 RViz 視覺夠用，省 CPU

### cmd_vel 過濾管線（採 spot_rl/pid.cpp）

```
policy raw cmd ──→ first-order low-pass (α=0.3) ──→ slew-rate limit (±1 m/s²) ──→ deadband ──→ cmd_vel
```

- α=0.3: 70% 舊 + 30% 新，平滑離散動作跳階
- Slew: |Δcmd| ≤ max_accel × dt，防 policy 從 +0.5 直跳 -0.5
- Deadband: |cmd| < 0.02 強制歸 0，避免 jitter

### deploy_with_bev.launch.py 參數

| arg | 預設 | 說明 |
|---|---|---|
| `params_file` | 內建 yaml | 覆寫 policy_params.yaml |
| `model_path` | "" | 覆寫 yaml 的 model_path |
| `enable_bev` | true | false 則不啟 bev_play |
| `enable_preprocessor` | true | false 則不啟 lidar_preprocessor |
| `log_level` | info | debug / info / warn / error |

### deploy_full.launch.py 參數（完整棧）

| arg | 預設 | 說明 |
|---|---|---|
| `initial_mode` | idle | 建議首次用 idle，確認後再改 nav |
| `map_file` | `/home/aa/maps/4v3F.yaml` | NDT map |
| `enable_mot` | true | 動態障礙物追蹤 |
| `enable_costmap` | true | costmap |
| `rviz` | true | RViz |
| `enable_bev` | true | BEV play node |

### 驗證 TF 是否完整

```bash
# 檢查 map → base_footprint 可不可以查
ros2 run tf2_ros tf2_echo map base_footprint

# 看不到 → 缺 NDT，或 base_frame 名字錯
# 看得到但跳動 → NDT 還沒收斂，等 5~10 秒
```

## 參考：rover2_ws 既有 stack

實車上已有 ROS 2 stack 在 `/home/aa/rover2_ws/`，部分元件可重用：

| 路徑 | 用途 | rover_rl 整合方式 |
|---|---|---|
| `src/campusrover_base/` | 底盤 driver + URDF + TF | 不動。提供 `/odom` 與 `base_footprint`/`velodyne_link` TF |
| `src/spot_rl/` | 舊版 spot RL policy（spot_model + warp_device_ros） | **參考，不依賴**。rover_rl 是獨立新版 |
| `src/campusrover_navigation/` | 地圖 + costmap + planner | 可提供 `/goal_pose` 來源（RViz Nav2 goal） |
| `launch_rl.sh` | 舊 RL 完整啟動腳本 | 已過時；改用 `deploy_rl` alias（見下） |
| `/home/aa/maps/4v3F.yaml` | 預設地圖 | 給 NDT localizer 用 |

### 注意：與 spot_rl 的差異

- spot_rl 用 spot_model（不同 obs 維度、不同 action space）
- rover_rl 用 SA6_TC checkpoint（VLP16 + 72-bin sweep + 19×19 discrete action）
- **兩者不可互換 checkpoint**
- **spot_rl 核心（ai_model_action.py / spot_obs_process.cpp）是 ROS 1，從未完成 ROS 2 移植**；rover_rl 才是完整的 ROS 2 RL deployment

### deploy_rl alias（最快啟動方式）

`.bashrc` 已有現成 alias，包含完整 source 順序與環境設定：

```bash
deploy_rl            # 純 ros2 launch（前景滾動 log）。任何 shell 皆可，含 Claude 非互動環境
deploy_rl initial_mode:=nav     # 參數原樣轉給 launch
deploy_rl_shell      # 互動式：完整棧背景 + 前景繁中 TUI 儀表板（需真實 TTY，給人用）
deploy_rl_shell initial_mode:=nav
deploy_rl_stop       # 停止整個棧
```

**兩者分工（重要）**：
- `deploy_rl` = 純 `ros2 launch deploy_full`，前景滾動 log。**Claude / 非互動 shell 用這個**（curses 在 pipe 會卡死，故 deploy_rl 不含 TUI）。Claude 要看狀態改用 `ros2 topic echo /rover_rl_policy/status`（JSON）。
- `deploy_rl_shell` → `bash ~/rover_rl/deploy_rl_shell.sh`（**給人在真實終端機用**）：
  1. **TTY 守門**：偵測非互動（`! -t 0/1`）→ 友善退出不硬跑 curses
  2. **launch 前兩段互動詢問**（命令列已帶同名參數則各自跳過、尊重覆寫）：
     - **選 checkpoint**：列 `~/rover_rl/models/*.ts`，Enter=沿用 yaml 預設；選到 `*v3c*` 自動帶 v3c config
     - **是否啟用 VO 安全層**：`[Y/n]（Enter=啟用）`。選 Y→`enable_vo:=true`、n→`enable_vo:=false`。
       預設啟用（沿用原行為）；啟用後 policy 改道 `/rover_rl/cmd_vel_desired`→vo_safety→mux，
       且 TUI 多顯示「VO / VO參數」兩列。⚠ 此腳本預設不開 lv-dot → 即使選啟用，沒另開 `lv-dot`
       時 VO 無障礙來源、退化為「放行 + ω 限幅」（TUI 顯示「放行（障礙源逾時/未偵測）」），這是正常安全退化。
  3. `ros2 launch deploy_full` 丟**背景**（帶上選定的 model_path / `$VO_ARG`），log 導到 `~/rover_rl/logs/deploy_<時間>.log`
  4. 等 `rover_rl_policy` 起來 → 前景跑 `status_tui`（curses 取得真實 TTY）
  5. 按 `q` 或 Ctrl+C → trap（`trap - EXIT INT TERM` 先解除自身避免重入，**不用 `''` 遮蔽訊號**）呼叫 `rover_rl_stop.sh` 收棧

**為何不把 TUI 放進 launch**：launch 子行程無 TTY，curses 會崩；故採「launch 背景 + TUI 前景」分離，且只在 `deploy_rl_shell` 提供。

⚠️ 直接用 alias，不要手動拼環境指令。

### Zenoh Router 必須先運行

所有 ROS 2 通訊走 `rmw_zenoh_cpp`，依賴一台 zenoh router hub（`192.168.3.13:7447`）。
**節點啟動前先確認 zenoh router 在線**：

```bash
# 確認本機 zenoh router service 狀態
systemctl status zenoh-router.service   # 應為 active (running)

# 確認能連到遠端 router
ping 192.168.3.13

# 驗證 ROS 2 通訊
ros2 topic list    # 有輸出 = zenoh 正常
```

若 `ros2 topic list` 無回應或超時 → zenoh router 沒連上，先解決這個再啟動任何節點。

## ROS 通訊驗證流程（首要任務）

### Step 1：環境檢查

```bash
# ⚠️ 正確環境設定（RMW=zenoh, DOMAIN=55）— 直接用 alias 最安全
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
# 或直接：source ~/rover_rl/setup_env.sh

# 檢查必要 topic 是否已被既有節點發布
ros2 topic list | grep -E "velodyne_points|odom|goal_pose|cmd_vel"

# 預期看到：
# /velodyne_points         ← VLP-16 driver
# /odom                    ← campusrover_base driver
# /input/nav_cmd_vel       ← 我們要發的（mux 接收端）
# /output/cmd_vel          ← mux 給 driver 的
# /goal_pose               ← Nav2 / RViz 發
```

### Step 2：Build rover_rl

```bash
cd ~/rover_rl               # 實際路徑（/home/aa/rover_rl）
source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash   # campusrover_msgs 依賴
colcon build --symlink-install --packages-select rover_rl_inference rover_rl_bringup
source install/setup.bash
```

**常見錯誤**：
- `ModuleNotFoundError: torch` → Jetson 用 NVIDIA wheel，見 DEPLOY_CAMPUSROVER.md
- `Cannot find sensor_msgs` → 沒 source ROS 環境
- `ModuleNotFoundError: campusrover_msgs` → `routing_to_path` / `routing_click_bridge` 需要此 package。
  `rover_rl_inference/package.xml` 未宣告此依賴（已知缺失）。
  修復：確認 `source ~/rover2_ws/install/setup.bash` 在 build 和 launch 前都執行。

### Step 3：把 model 放到位

```bash
mkdir -p ~/rover_rl_ws/models
# 從 PC 端拷貝（或從 GitHub release）
scp pc-a:/home/aa/IsaacLab/rover_rl/models/sa6_tc_dense_420k.ts ~/rover_rl_ws/models/

# 修改 config 中的 model_path 絕對路徑
nano ~/rover_rl_ws/src/rover_rl_bringup/config/policy_params.yaml
# 改：model_path: "/home/<user>/rover_rl_ws/models/sa6_tc_dense_420k.ts"
```

### Step 4：純通訊驗證（不上電）

```bash
# Terminal A: 啟動 LiDAR + odom（用既有 stack）
cd /home/aa/rover2_ws && source install/setup.bash
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py &
ros2 launch campusrover_base rover_driver.launch.py &     # 看實際 launch 名

# Terminal B: 啟動 rover_rl（preprocessor + policy + bev）
cd ~/rover_rl && source install/setup.bash
source setup_env.sh    # ROS_DOMAIN_ID=30, RMW=fastrtps
ros2 launch rover_rl_bringup deploy_with_bev.launch.py log_level:=debug initial_mode:=idle

# Terminal C: 確認資料流
ros2 topic hz /velodyne_points              # 應 ≈ 10 Hz
ros2 topic hz /odom                          # 應 ≈ 20 Hz
ros2 topic echo /input/nav_cmd_vel --once   # 在發 goal 後應有訊息

# Terminal D: 發測試 goal
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 2.0, y: 0.0}, orientation: {w: 1.0}}}'

# 然後在 Terminal C 看 /input/nav_cmd_vel 是否：
# - 以 5 Hz 持續發布
# - linear.x 在 [-1.0, 1.0] 範圍
# - angular.z 在 [-2.0, 2.0] 範圍
# - 不全是零（policy 有反應）
# - 不爆走（不會跳到絕對值最大）
```

### Step 5：離地測試（車架起）

policy_node 已具備 watchdog（LiDAR/odom timeout 自動發 0），但首次測試請：
1. 把車架起讓輪子離地
2. 開機跑完整流程
3. 觀察輪子方向是否朝期望（goal 在前 → 兩輪正轉）
4. 用遙控器 deadman 隨時切回 joy 模式

## 新增節點（2026-06 加入）

### routing_to_path
- **Executable**: `routing_to_path`
- **職責**: campusrover_routing `generation_path` service → `/global_path` topic 橋接
- **呼叫方式**: `ros2 service call /rover_rl/routing_call campusrover_msgs/srv/RoutingPath "{origin: 'c1', destination: ['e0']}"`
- **效果**: 結果自動 2Hz republish 到 `/global_path` → policy_node 的 SubgoalSelector 接收
- **具名站序（2026-06-09 加入，地鐵式儀表板用）**: 額外抓 `/get_route_info`（ModuleInfo）拿
  拓撲節點表（name→xy），把 `/global_path` 的 waypoint snap 到最近具名節點（門檻 `station_snap_m`
  預設 1.5m、去重相鄰同名）→ 推出**有序站名 + 各站起始 waypoint index**，2Hz 發到
  `/rover_rl/route_stations`（std_msgs/String JSON `{stations, wp_idx, n_wp}`）。
  - RoutingPath response 本身**不含站名**（只有 `nav_msgs/Path[]` 純 pose），站名/座標來自 get_route_info。
  - status_tui 用 policy 發的 `path_i`（同一條 path 的 index）對 `wp_idx` 求目前在第幾站，
    **不做 frame 轉換**（path_i 與 wp_idx 同屬 path 索引空間）。節點表未載入 → stations 空 → TUI
    自動退回幾何進度條。

### routing_click_bridge
- **Executable**: `routing_click_bridge`
- **職責**: RViz "Publish Point" 兩次點擊 → 自動呼叫 routing service → path 顯示 marker
- **參數**: `building: itc`, `floor: 3`
- **使用**: RViz 開 "Publish Point" 工具 → 第1點選起點 → 第2點選終點 → 自動觸發 routing

兩者已整合進 `deploy_full.launch.py`，也可單獨啟動：
```bash
ros2 run rover_rl_inference routing_to_path
ros2 run rover_rl_inference routing_click_bridge
```

### diag_logger（診斷記錄）
- **Executable**: `diag_logger`
- **職責**: 被動訂閱 odom/ndt/goal/cmd_vel/obs，逐列寫 CSV（不影響推論），供事後分析晃動/不朝 goal
- **⭐ 資料存放位置（用戶委託分析時，第一個來這裡找）**: `~/rover_rl/logs/diag/`
  - **每次實驗一個獨立資料夾**：`~/rover_rl/logs/diag/diag_<YYYYMMDD>_<HHMMSS>[_<實驗名>]/`
  - 該次所有檔案都在裡面：
    - `diag_<時間>.csv` — 20Hz 時間序列（goal/位置/cmd_vel/三層速度/延遲…）
    - `diag_<時間>_params.json` — 當下 policy_node 全部參數（speed_rate / cmd_alpha 等，重現性用）
  - 找最新一次測試 = `ls -td ~/rover_rl/logs/diag/*/ | head -1`
- **何時建資料夾**: deploy 啟動後**待命**，**收到第一個 goal/path 才建資料夾開錄**（沒走 goal 不留空資料夾）。
  `<實驗名>` 來自 record start 的 label（`_safe_label()` 清特殊字元；不給就只有時間戳）
- **⭐ 一個 goal/path = 一段獨立紀錄（2026-06-09 加入 auto-stop + auto-rearm）**:
  - **到終點自動停**: robot 到 goal `auto_stop_goal_tol`（0.6m）內連續 `auto_stop_goal_ticks`（10 tick=0.5s）
    → 印摘要、關 CSV、`wandb.finish()`、發 `/rover_rl/diag_event` 的 `goal_reached`。
  - **方式 B（routing path）**: 取 `poses[-1]`（終點）當停止判據；中途 waypoint 不停，只在**最後一點**停。
  - **自動 re-arm（`auto_rearm`，預設 True）**: 停完自動回待命，**下一個 goal/path 進來開全新一段**
    （新資料夾 + 新 CSV + 新 wandb run）。多 goal 連續操作完全自動，不必手動 record start。
  - **每段一個 wandb run**（`enable_wandb` 預設 True、`wandb_mode` 預設 **online**），run 名=資料夾名。
  - **stale path 防護**: `routing_to_path` 是 2Hz 無限 republish，故 ①進行中同終點只刷新座標不重置
    auto-stop tick（否則永遠停不了）②re-arm 後用 `_last_done_goal` + `goal_change_eps_m`（0.8m）擋掉
    「剛完成終點」的殘留發布，避免原地秒重開迴圈。新 routing（終點 >0.8m 變化）才算新一段。
  - 手動 `record stop` 是明確停止，**不** re-arm。
- **⚙ 設定真值**: `src/rover_rl_bringup/config/diag_logger_params.yaml`（auto_rearm / goal_change_eps_m /
  auto_stop_* / wandb 等都在此）。deploy_full 載入它；改 yaml 後重 build bringup 即生效。
  `require_start` / `enable_wandb` / `wandb_mode` / `auto_rearm` / `goal_change_eps_m` 五個也可用
  CLI arg 現場熱調（留空走 yaml，有給才覆寫）：`deploy_rl auto_rearm:=false goal_change_eps_m:=1.5`。
- **舊紀錄（2026-06-04 之前）**: 平鋪在 `~/rover_rl/logs/diag/` 根目錄（未分 run 資料夾）
- **Ctrl+C 後**: 自動印「診斷摘要 + CSV 完整路徑 + analyze_diag 指令」
- **分析**: `ros2 run rover_rl_inference analyze_diag ~/rover_rl/logs/diag/diag_<時間>/diag_<時間>.csv`

### status_tui（即時狀態儀表板，繁中 TUI）
- **Executable**: `status_tui`
- **職責**: 訂閱 policy_node 發布的精簡狀態（`/rover_rl_policy/status`, std_msgs/String 內含 JSON），
  用 curses 畫成乾淨方框儀表板即時刷新。**純訂閱純渲染，不影響推論**。
- **解決痛點**: `deploy_full` 一次拉起十幾個節點、log 全擠同一 terminal 難觀察 → 此節點獨立一個
  terminal 跑，模式 / cmd_vel / LiDAR 最近距離 / 里程計 / NDT / goal 方向一目了然。
- **顏色**: 綠=正常、黃=注意（idle/paused/NDT 未穩）、紅=危險（estop/逾時/障礙過近）。
- **資料來源**: policy_node 以 5 Hz 發 `~/status`；TUI 收不到 >1.5s 顯示「等待 policy_node…」。
- **互動式啟動**：`deploy_rl_shell`（**人用**，背景 launch + 此 TUI 前景，見「deploy_rl alias」）。
  `deploy_rl`（Claude/非互動用）**不含** TUI。
- **單獨啟動**（接已在跑的棧，需真實終端機）:
  ```bash
  source ~/rover_rl/install/setup.bash && source ~/rover_rl/setup_env.sh
  ros2 run rover_rl_inference status_tui      # 按 q 離開
  ```
- **不放進 launch 檔 / 不放進 deploy_rl**：launch 子行程與非互動 shell 無 TTY，curses 會卡死/亂碼；
  故 curses TUI 僅由 `deploy_rl_shell.sh`（含 TTY 守門）前景啟動。
- **導航型態 + 地鐵式路線（2026-06-09 加入）**：「導航」列標示目前是
  `路徑導航 (routing)`（方式 B，RViz Publish Point 兩點觸發 routing）或 `單一 goal 導航`
  （方式 A，RViz 2D Goal Pose）或 `待命（無目標）`。資料來自 status JSON 的 `nav_type`
  （由 policy 的 subgoal source 推導：`path_*`→path、`goal_pose`→single）。
  - **路徑導航時多一列「路線」地鐵式進度條**：
    - 有具名站序（訂閱到 `/rover_rl/route_stations`）→ 畫站名線 `c1 - c3 - e0`，目前站綠色粗體，
      站太多時以 `…` 視窗化；目前站由 `path_i` 對 `wp_idx` 求得。「導航」列同步顯示 `c3 (第 2/3 站)`。
    - 無站序（routing 節點表沒載入/手發 path）→ 退回幾何進度條 `起[==O-->--]終 12/47`，
      `O`=目前最近 waypoint、`>`=lookahead carrot。
- **三層速度對比**：儀表板「速度v / 速度ω」列同框顯示
  `想{RL意圖} 送{濾波後送底盤} 實{odom實測}`；角速度超出底盤 `chassis_omega_max`（預設 1.2）時
  標 `⚠超1.2`（對應 gap #2）；底盤實測明顯跟不上送出值時轉黃（飽和/deadband/延遲）。
- **延遲量測**：`latency.py` 的 `LagEstimator` 以滑窗正規化互相關估「送出 cmd → odom 實測」的
  時間落差（線/角通道各一，取相關係數高者）。儀表板「延遲」列顯示 `XXXms (相關 r, 通道)`；
  >200ms 轉黃、>400ms 或相關低轉紅（對應 gap #5 振盪風險）。站著不動訊號太平 → 顯示「需移動中」。
- **RNN 狀態**：`PolicyRunner` 記 `reset_count/step_count` + `hidden_norm()`（hidden L2 範數）。
  儀表板「RNN」列顯示 `‖h‖=X 本段步數 N 重置 M 次`；hidden 歸零（idle/剛 reset）→ 灰字「待命/已重置」。
  每收新 goal/path、切模式、換模型都會 reset hidden（episode 記憶重來）。
- **LV-DOT 動態障礙**：儀表板「動態」列**直接訂閱** `/onboard_detector/dynamic_bboxes`（MarkerArray，
  與 policy 解耦，policy 推論不吃此資料、obs 障礙欄仍補 0）。顯示動態障礙框數；未啟動→灰、
  >2s 無更新→黃「偵測器可能死」、0 個→綠。topic 可用 `topic_dynamic_bboxes` 參數改。
- **VO 安全層（2026-06-16 加入）**：`vo_safety_node` 以 ~5Hz 發 `~/status`（`/vo_safety_node/status`,
  std_msgs/String JSON，純觀察不影響控制）；TUI 訂閱後在「動態」列下多畫**兩列**，與 LV-DOT 同款解耦
  （`enable_vo` 沒開 / 收不到 status → **自動隱藏這兩列**，topic 可用 `topic_vo_status` 參數改）：
  - **「VO」列（介入狀態，依嚴重度上色）**：紅`⚠ 看門狗發 0（odom/RL 逾時）`、紅`⛔ 全堵死→停車（近障N）`、
    黃粗`介入中 RL(v,w)→(v,w) 近障N 可行M`、黃`放行（障礙源逾時/未偵測，僅 ω 限幅）`、
    青`監看中（近障N，未改寫放行）`、綠`✓ 放行（無近障）`。判據同 vo_safety 的 log
    （`intervening = engaged 且 VO 解明顯偏離 RL 意圖`）。
  - **「VO參數」列**：`ω≤{w_max} 預測{horizon}s 觸發{engage_range}m 餘裕{margin}m 追蹤{n_tracked}`，
    對應 `vo_params.yaml`，現場確認 VO 用哪組參數。
  - ⚠ `deploy_rl_shell` 詢問啟用 VO 但沒另開 `lv-dot` 時，VO 列固定顯示「放行（障礙源逾時/未偵測）」
    （無 `vo_interface/tracked_obstacles` 障礙來源）——正常安全退化，要看到真正介入需同時 `lv-dot`。
- **右側 LiDAR 雷達（2026-06-09 加入）**：主框右邊空白區多開一個子視窗，把 BEV 的極座標散點
  **用純文字重畫**（不把 `/rover_rl/bev_image` 點陣圖塞進 curses——curses 不支援影像、且吃終端機）。
  **直接訂閱 `/rover_rl/lidar_sweep_72`**（與 status JSON 解耦，狀態逾時也能看 LiDAR），反算公尺後
  畫散點：綠=安全(>5m)、黃=注意(2~5m)、紅=危險(<2m)、貼近量程上限視為無回波不畫；機器人置中 `↑`、
  goal 黃 `◆`（來自 status 的 `goal_dist`/`goal_ang_deg`，與 sweep 同 `atan2(left,fwd)` 慣例、含
  BEV 同款左右鏡像）。**前方朝上**。
  - **兩種風格可熱切換**：`radar_style:=braille`（預設，每字元 2×4 點、≈80×80 點解析、輪廓最接近 BEV）
    或 `dots`（彩色 `●` 散點、最簡）。`ros2 param set /rover_rl_status_tui radar_style dots` 現場切。
  - 參數：`enable_radar`(預設 True)、`radar_range_m`(顯示半徑，預設 10m，超出貼邊)、`topic_sweep`、
    `r_max`/`r_robot`(正規化反算，與 preprocessor 對齊)。
  - **窄螢幕自動退化**：右側可用寬度 <26 欄就不畫雷達（比照站線放不下的退化邏輯），純文字終端無關。
  - 想要照片級畫質（trail/距離環/點大小分級）仍用 `rqt_image_view /rover_rl/bev_image`；雷達只做一眼態勢感知。

### 速度/延遲離線分析（diag_logger + analyze_diag）
- `diag_logger` 已加訂閱 `/rover_rl_policy/status`，CSV 多記 `rl_v/rl_w`（RL 意圖）、
  `sent_v/sent_w`（送出）、`act_v/act_w`（實測）、`v_over/w_over`（飽和旗標）、`lag_ms/lag_corr`。
- `analyze_diag` 多出兩段報告：
  - **【速度三層對比】**：各通道 想要→送出→實測 平均值 + 底盤跟隨率（<60% 警示飽和/deadband）
  - **【延遲】**：整段離線互相關估計（較穩）+ 即時 lag_ms 摘要，附 >200ms/>300ms 判讀
  ```bash
  # 資料在 ~/rover_rl/logs/diag/diag_<時間>/ 下；分析最新一次：
  ros2 run rover_rl_inference analyze_diag "$(ls -td ~/rover_rl/logs/diag/*/ | head -1)"/*.csv
  ```

### pingpong_test（兩固定點往返避障測試，2026-06-24 加入）
- **Executable**: `pingpong_test`
- **職責**: 把車手動開到 A/B 兩拓撲節點（預設 c24/c27）任一點停穩 → **每段都按空白鍵才出發**：
  按下後規劃到對向點，到達**自動停車回就緒**、再按空白鍵走回來，A↔B 一段一段確認來回，
  供兩段間重擺障礙物反覆測避障。與 RL 推論/VO 避障完全解耦（只訂閱 policy status、呼叫
  routing、必要時切 mode=nav/idle）。
- **三態狀態機**：
  - `waiting`：等車開到 A/B 停穩（距 < `arrive_tol_m`=0.8m、低速 < `start_max_speed`=0.15m/s、
    連續 `start_dwell_ticks`=4 拍≈2s）。mode **只擋 estop**（estop=明確要停，不就緒）；
    **nav/manual/idle 皆可就緒**——預設 nav 把車停到點上即就緒，不必先切 idle。安全靠「停穩在點上
    + 空白鍵閘門」兩道把關（RL 若在開速度/距離一定不過關）。
  - `ready`：就緒，發狀態給 TUI 跳大字提示。**只有按空白鍵**（TUI 攔截 → `/rover_rl/pingpong/start`
    Empty）才真正開始；車一離開點/加速/切 estop 即退回 waiting。
  - `running`：切 mode=nav（`auto_set_nav`，預設開）+ 呼叫 routing 到對向點；**到達對向點
    自動停車（切 idle）回 `ready`、等空白鍵走下一段**（非自動折返）。mode 一旦離開 nav
    （抓搖桿→manual / estop）→ **立即中斷回 waiting**，須手動開回任一點停穩、再按空白鍵重啟。
    watchdog：3s 無有效路徑自動重規劃（防 routing 偶發回空）。
- **介接**：座標來自 `/get_route_info` 拓撲節點表（同 routing_click_bridge 作法）；pose 用 policy
  status 的 map-frame 車姿；routing 走 `/routing_to_path/routing_call`；mode 發 `/rover_rl_policy/mode`。
  自身狀態 5Hz 發 `/rover_rl/pingpong/status`（JSON `{state, ready_point, from, target, a, b, nodes_loaded}`）
  供 status_tui 顯示提示與底部狀態列。
- **⚠ 需 NDT + routing 在跑**：靠拓撲節點 map-frame 定位判到點，沒 NDT（pose_src≠tf）會警告且判定失準。
  `deploy_rl_shell` 預設不啟 NDT → 先另開 `ndt` alias，或改用 `deploy_all`。
- **啟動**：`deploy_rl_shell` 互動詢問「是否啟用往返測試」並可填 A/B 點名；或 launch 直接帶參數：
  ```bash
  ros2 launch rover_rl_bringup deploy_full.launch.py \
    enable_pingpong:=true pingpong_a:=c24 pingpong_b:=c27
  ```
  launch 參數：`enable_pingpong`(預設 false) / `pingpong_a`(c24) / `pingpong_b`(c27) /
  `pingpong_auto_nav`(true)。
- **操作流程**：① 手動把車開到 c24 或 c27 停穩 → ② TUI 跳「就緒，按空白鍵」→ ③ 按空白 → 車往對向點 →
  ④ 到達自動停車、TUI 再跳「就緒，按空白鍵」→ ⑤ 按空白走回來（如此每段確認來回）→
  要停就抓搖桿/estop（中斷）→ 手動開回任一點停穩、再按空白鍵重啟。

## 故障排除清單

| 症狀 | 檢查 | 修復 |
|---|---|---|
| policy_node 啟動 crash | `journalctl -xe` 或 launch output | 99% 是 `model_path` 錯或 torch 沒裝 |
| /input/nav_cmd_vel 沒輸出 | `ros2 topic info /input/nav_cmd_vel -v` | 確認 mux config 接此 topic；或直接改 `topic_cmd_vel: /cmd_vel` 測試 |
| 永遠輸出 0 cmd | log 看 `lidar timeout` 或 `odom timeout` | LiDAR/odom 訊號中斷；檢查 hz |
| sweep_src=inline_fallback | preprocessor 節點死了 | `ros2 topic hz /rover_rl/lidar_sweep_72`；重啟 preprocessor |
| Goal 永遠收不到 | `goal_frame` 設錯 | 改 `goal_frame: odom`（沒 map）或 `map`（有 NDT/AMCL） |
| goal 方位/距離全歪（點正前方卻算成遠處側後方） | `/ndt_pose` 是 map→odom 非車姿，被當車姿用 | 已修：車姿改走 TF `map→base_footprint`（見上方 NDT 段）。確認 `pose_src=tf`、`pose_x/y` 與 `tf2_echo map base_footprint` 一致 |
| routing_click_bridge 找不到節點 | 地圖節點未啟 | 先確認 `/get_route_info` service 存在 |
| 往返測試一直「待命」不就緒 | mode=estop、車離點太遠(>0.8m)、沒停穩(速度>0.15)、或 pose_src≠tf | 把車開到 A/B 停穩（nav/manual/idle 皆可，只要別 estop）；`ros2 topic echo /rover_rl/pingpong/status` 看 state/nodes_loaded、policy status 看 pose_x/y 與點距；無 NDT 先開 `ndt` |
| 往返測試按空白鍵沒反應 | 非 ready 狀態才會忽略空白鍵；或不在 TUI 焦點 | 先確認 TUI 跳「就緒」綠字才按；空白鍵只由 status_tui 攔截（deploy_rl_shell 才有 TUI） |
| 往返測試 nodes_loaded=false | routing/get_route_info 沒起來 | 確認 routing_to_path + mapinfo_db_handler 在跑、`/get_route_info` service 存在 |
| 跑起來但車原地震 | normalizer 期望維度與 model 不符（79/83/139 互錯，如 v3c 用了 SA6 yaml） | 看 launch log 的 `raw_obs=X used_obs=Y`，與 model 對照；v3c 應為 raw_obs=83 |
| cmd 振幅異常大 | normalizer mean/var 沒 bake 進 model | 重新 export_policy.py（必須帶有 obs_normalizer 的 checkpoint） |
| RViz Nav2 goal 不被收 | topic remap | 確認 `/goal_pose` 是 Nav2 standard，不是 `/move_base_simple/goal` |
| /rover_rl/bev_image 沒畫面 | matplotlib Agg 依賴 | 確認 `pip install matplotlib` 已裝 |
| RViz 啟動但沒有 display 設定 | rviz config 不在 repo | `deploy_full` 引用 `/home/aa/rviz/demo.rviz`（本機路徑，未納入 git）。換機器部署須手動複製或改 launch 路徑 |
| 啟動後第一幀 LiDAR timeout | velodyne 比 policy 慢啟 | 正常現象（非 crash）。spot_rl 有 `delay.py` 延遲保護；rover_rl 沒有，第一幀 timeout 只是 warn log，約 1s 後自動恢復 |

## 任務分流（您與用戶協作時）

| 任務 | 您（車上 Claude）做 | 用戶決定 |
|---|---|---|
| Build workspace | ✅ 自動 colcon build | — |
| 改 yaml topic 名 | ✅ 依用戶實測 topic 名 | 告訴您正確 topic |
| 改 yaml 動作上限 | ❌ **不要動**（policy 訓練 distribution） | — |
| 改 yaml safety / tolerance | ✅ 提供 3 段預設值（保守/中/激進） | 選哪段 |
| 改 model_path | ✅ | 確認用哪個 .ts |
| 重訓 model | ❌ 您沒訓練環境 | PC 端訓練組 |
| 上電遙控 | ❌ | 用戶手動 |
| 寫 ros2 service / E-stop | ✅ 用戶若要求 | 提需求 |
| 序列多停靠點任務 | ❌ 目前不支援 | spot_rl 有 `service_seq.py`；rover_rl 僅支援單 goal 或單段 path，跨房間 A→B→C 需要用戶手動多次呼叫 routing_call |

## 模型選擇建議

| Model | obs | 訓練場景 | 建議使用情境 |
|---|---|---|---|
| `sa3_v3c_240000.ts` 🆕 | **83D** | 牆壁+crossing，goal 動態，action stacking 抗抽動 | **最新**；用 `*_v3c.yaml`，見上方 v3c 段 + `V3C_DEPLOY.md` |
| `sa6_tc_dense_420k.ts` ⭐ | 79/139D | T 型走廊 dense 障礙物，420k steps | SA6 首選，最久訓練 + 中等難度 |
| `sa7_tc_dense_300000.ts` | 79/139D | T 走廊高壓（occlusion 20%, dyn=8） | 進階場景；首次部署別用 |
| `sa5_tc_g1_p30_270000.ts` | 79/139D | T 走廊 g1_p30（goal=1, penalty=-30） | 訓練早期；當前不建議 |

⚠️ **SA5/6/7 是 T-Corridor 課程，v3c 是牆壁+crossing 課程**。實車場景差異大時 policy 可能
表現不佳；實測有問題就跟 PC 端訓練組要求對應場景重訓。
⚠️ **v3c 與 SA6 不可混用設定**：v3c 必須配 `*_v3c.yaml`（83D / r_min 0.25 / ω 0.785 / slew），
用 SA6 yaml 跑 v3c 會「raw_obs 維度不符」或行為失準。

## BEV 處理（純可視化，不是 policy 輸入）

**rover_rl 推論完全不需要 BEV**。模型輸入是 72-bin 1D LiDAR sweep，不是 2D 影像。

兩條獨立分支：
```
/velodyne_points (PointCloud2)
   │
   ├─→ rover_rl_lidar_preprocessor ── /rover_rl/lidar_sweep_72 ─→ policy_node ─→ cmd_vel
   │
   └─→ rover_rl bev_play_node ── matplotlib Agg 極座標圖 ─→ /rover_rl/bev_image  (debug)
```

### bev_play_node（現已整合進 rover_rl）

`rover_rl_inference/bev_play_node.py` 移植自訓練端 `play_rnn_car.py::LiveBEVVisualizer`。

**訂閱**：
- `/rover_rl/lidar_sweep_72` (Float32MultiArray[72])
- `/rover_rl_policy/obs_debug` (Float32MultiArray[79], 可選，取 goal body)
- `/input/nav_cmd_vel` (Twist, 可選)
- `/odom` (Odometry, 可選，trail + yaw)

**發布**：
- `/rover_rl/bev_image` (sensor_msgs/Image, rgb8, ~5 Hz)

**deploy_with_bev.launch.py** 預設 `enable_bev:=true` 自動帶起此節點。

### 是否啟動 BEV

**建議啟動**，作為 debug 工具：
- 上電前肉眼確認 LiDAR 看得到牆/障礙物
- 排查 sensor 異常時直接看極座標圖比 raw PointCloud2 直覺

```bash
# deploy_with_bev 已自動啟動 bev_play；用 rqt_image_view 查看：
ros2 run rqt_image_view rqt_image_view /rover_rl/bev_image
```

⚠️ 如果 BEV 沒看到任何障礙物 → policy 也看不到 → 別上電。

## ONNX / TensorRT 轉換（目前不做）

**結論：用 TorchScript 就好，不要花時間轉 ONNX。**

數字證明：
- Model 大小：1.4 MB
- Jetson Orin TorchScript 推論延遲：~3-5 ms
- Policy 控制週期：200 ms (5 Hz)
- 延遲占用：~2%（剩 195 ms 完全空閒）

ONNX/TensorRT 在這個 model 上**完全沒收益**：
- 即使 TensorRT FP16 把延遲降到 1 ms，5 Hz policy 也用不到
- ONNX export 對 vanilla RNN + index_select 需要手動拆，多 debug 時間
- 跨平台價值零（Jetson 已有 PyTorch）

**什麼時候才考慮 ONNX**：
1. 模型放大 10x（不會發生，這是 30-hidden RNN）
2. 整合 ONNX-only inference server（rover2_ws 全部 PyTorch，不需）
3. 用 CPU-only 邊緣裝置（Orin 有 GPU，跳過）

**重要禁令**：車上 Claude **不要主動** 寫 `export_policy_onnx.py`，
除非用戶明確要求並說明理由。

## 速度/加速度參數對照：DWA（舊）vs RL（新）

`deploy_full.launch.py` 用 RL policy 取代了舊的 DWA 局部規劃器。兩者速度設定差很多，
解釋「為什麼換 RL 後車變快、轉彎手感不同」時參考此表。

| 項目 | DWA（舊，已移除） | rover_rl RL（新） | 底盤實際上限 |
|---|---|---|---|
| 線速度 | 0.5 m/s | **1.0 m/s** | 1.5 m/s |
| 角速度 | 0.5 rad/s | **2.0 rad/s** | **1.2 rad/s** |
| 線加速度 | 5.0 m/s²（取樣用） | 0.5 m/s²（`act_max_linear_accel`） | 1.2 m/s² |
| 角加速度 | 5.0 rad/s²（取樣用） | 3.0 rad/s²（`cmd_max_accel_angular` 防甩） | — |

來源：
- DWA：`campusrover_demo_launch.py:225-230`（`dwa_planner` node 參數）
- RL：`policy_params.yaml`（`act_max_*` / `cmd_max_accel_*`）
- 底盤：`driver_chgh.yaml`（`max_speed: 1.5` / `profile_omega_max: 1.2` / `acc_max: 1.2`）

重點：
1. **DWA 跑超保守**（0.5 / 0.5），遠低於底盤能力；學長調慢求穩。
2. **RL 比 DWA 激進**：線速度 2 倍、角速度 4 倍（訓練分布就較快）。
3. **角速度隱憂**：DWA 0.5 < 底盤 1.2 安全；但 **RL 2.0 > 底盤 1.2**，RL 會叫出底盤做不到的轉速
   → 見下方 sim-to-real gap #2。DWA 沒此問題因為只到 0.5。
4. DWA 的加速度 5.0 是「速度 pair 取樣用」，不是真的輸出 5.0 猛加速（速度本身卡在 0.5）。

## sim-to-real 已知 gap（按嚴重度排序）

1. **LiDAR 高度 1.6m (訓練) vs 1.43m (實車)** — beam 角度落點不同
2. **訓練 ω_max=2.0 vs 底盤 1.2** — 全力轉時實際比 policy 預期慢 40%（見 `driver_chgh.yaml: profile_omega_max: 1.2`）
3. **wheel 不對稱 1.9%** — odom drift 1%/m
4. **訓練 T 走廊 vs 實車任意場景** — 泛化性未驗證
5. **cmd_delay 補償（2026-06-08 已實作）** — 實車實測：goal-following 時車頭 0.42Hz「舞龍舞獅」振盪。
   根因 = 底盤 cmd→實測死時間（diag 互相關，**ω 通道 ~0.2s = 1 控制步**；註：v 通道因等速直行訊號太平、互相關全平坦，所報 600ms 是假象，勿信）+ 5Hz 控制步 → policy 對 1 步前的舊車姿過度修正 → 延遲驅動極限環。
   實驗證明 `speed_rate` 降速**只治標**（0.3→0.2 擺幅僅 −20%、頻率不變）。
   **解法（已加）**：`policy_node` 新增 `cmd_delay_comp_s` 參數（預設 0.0=關）。>0 時推論前用 odom 測得速度把車姿往前積分這麼多秒、重算 goal_body，讓 obs 對齊「動作生效時」的車姿（仿 spot_rl `fast_info_calculate`，但只動 goal_body 視角、不動 velocity obs/網路/動作上限；用測得速度非命令速度，因 ω 跟隨率僅 ~12%、用命令會過補）。
   可熱調：`ros2 param set /rover_rl_policy cmd_delay_comp_s 0.2`；status JSON 有 `cmd_delay_comp_s` 可驗證。建議從 0.2 起、過大會領先過頭。
6. **底盤 deadband 未知** — spot_rl 強制最小移動速度 `minimum_will_move_speed=0.14 m/s`（避免 policy 輸出微小速度但馬達不動）。rover_rl deadband 僅 0.02 m/s。若實車觀察到 policy 有輸出但輪子靜止，需量測實際底盤死區並調高 `cmd_alpha_linear` 或在 action_decoder 加 floor。

## 與 PC 端的溝通介面

PC 端 repo：`/home/aa/IsaacLab/rover_rl`（私 SSH origin: `me0608623/rover_rl.git`）

需要 PC 端配合的事：
- 重訓 model（改 LiDAR 高度 / open scene / 新場景）
- 匯出新的 .ts（用 `scripts/export_policy.py`，已 auto-detect 架構）
- 提供新版 obs / action 規格

## ROS MCP Server 使用指南

**ros-mcp-server** 已全域安裝，可在此 session 直接對 rover_rl 的 ROS 2 環境下指令。

### 啟動 rosbridge（每次使用前）

```bash
source /opt/ros/humble/setup.bash
source ~/rover_rl/setup_env.sh   # ROS_DOMAIN_ID=55, RMW=rmw_zenoh_cpp
ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
# 看到 "Rosbridge WebSocket server started on port 9090" 即成功
```

### 連線

在 Claude 對話中直接呼叫：
```
connect_to_robot(ip="127.0.0.1", port=9090)
```

### 常用操作範例

```
# 看所有 topic
get_topics()

# 確認 policy 有沒有在發 cmd_vel
subscribe_once(topic="/input/nav_cmd_vel", timeout=3)

# 切換 rover_rl mode
call_service(service="/rover_rl_policy/set_mode", ...)

# 查 lidar sweep 數值
subscribe_once(topic="/rover_rl/lidar_sweep_72", timeout=2)

# 確認 odom 頻率
get_topic_details(topic="/odom")

# 看 RealSense 彩色影像（先 subscribe 存圖，再 analyze）
subscribe_once(topic="/camera/camera/color/image_raw",
               msg_type="sensor_msgs/msg/Image", timeout=5, expects_image="true")
analyze_previously_received_image()
```

### RealSense 相機（隨 lv-dot 共同啟動，2026-06-10 更新）

相機（D435i）接在主機 `192.168.3.13`（帳號 humble，Jetson 本機無相機裝置）。
**舊的開機自啟 systemd 服務已不存在**（實查無 realsense.service / cron / autostart）。
現行機制：`lv-dot` launch（`run_detector.launch.py`）內建 `use_camera:=true`（預設開），
透過免密 SSH（aa@jetson → humble@.13，金鑰已建）跑遠端 `~/start_realsense.sh`
（含 `enable_depth:=true` + depth 640,480,30 + color 640,480,15 + zenoh RMW）。
**冪等 + 脫鉤**（2026-06-10 實測）：相機已在跑 → 沿用不重啟（避免雙開搶 USB）；
沒在跑 → setsid+nohup 背景拉起（log: `~/realsense_lvdot.log` on .13）。
**關 lv-dot 不會關相機**——要手動關：`ssh humble@192.168.3.13 "pkill -f 'realsense2_camera_[n]ode'"`
（pattern 必須用 [n] bracket trick，否則 pkill 比中 ssh 自己的指令字串）。

rover_rl 推論**不使用**此相機（policy 只吃 72-bin LiDAR sweep）；
LV-DOT 的視覺分支（depth UV detector + YOLO 2D）與人工確認場景需要它。

| Topic | Type | 說明 |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | 640×480 rgb8 @ 15fps |
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/Image` | 640×480 @ 30fps |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 內參 |

frame_id = `camera_color_optical_frame`。透過 ROS MCP 看畫面：
```
subscribe_once(topic="/camera/camera/color/image_raw",
               msg_type="sensor_msgs/msg/Image", timeout=5, expects_image="true")
analyze_previously_received_image()
```

⚠️ 注意：`ros2 topic list` 看到 `/camera/*` 不代表相機在發（detector 訂閱端也會讓
topic 名字出現）——要用 `ros2 topic info -v` 看 **Publisher count**。
（2026-06-10 曾發生 D435i 整支從 USB 消失 → 遠端一直 `No RealSense devices were found!`，
重插即恢復。看到此訊息先查 .13 的 `lsusb | grep -i intel`。）

### 與 rover_rl 部署整合

rosbridge 要在 rover2_ws / rover_rl 之前啟動，並使用相同 RMW 環境：
```bash
source ~/rover_rl/setup_env.sh   # ← 必須，確保 zenoh + DOMAIN_ID=55 一致
ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
```

若 rosbridge 沒有 source rover_rl 環境就啟動，會連到不同 zenoh domain，看不到 rover_rl 的 topics。

## LV-DOT 動態障礙偵測 + VO 介面（2026-06-15 整合）

> LV-DOT（`src/LV-DOT/onboard_detector`）：LiDAR + 視覺融合的動態障礙偵測追蹤，**與 RL policy 完全解耦**
> （policy 不吃此資料、obs 障礙欄仍補 0）。輸出供 RViz 觀察 / status_tui / 下游 VO 避障規劃。
> 移植與調校全紀錄見記憶 `lvdot-ros2-port.md` / `vo-interface-node.md`。

### 一鍵啟動（lv-dot alias）

```bash
lv-dot          # = ros2 launch onboard_detector run_detector.launch.py use_yolo:=true
lv-dot_stop     # 停掉 detector + yolo + vo + launch（已含全部）
```

`lv-dot` 一鍵起**完整一套**：遠端相機(SSH .13) + detector_node + YOLO(GPU venv) + **vo_interface**。
- **防雙開**：launch 啟動前先 `pkill` 殘留 detector/yolo/vo（`stale_cleanup` + `OnProcessExit` 排序），
  **不先 lv-dot_stop 直接重啟也不會雙開**。（舊問題：Ctrl+C 偶爾沒收乾淨 YOLO venv 子行程 → 殘留雙開。）
- **enable_vo**（預設 true）：vo_interface 隨 lv-dot 一起起；`lv-dot ... enable_vo:=false` 可關。
- **相機**：`use_camera:=true`（預設）SSH 到 humble@192.168.3.13 跑 `~/start_realsense.sh`；
  關 lv-dot 連帶殺相機（見上方相機段）。
- detector 終端 ~1Hz 印 `[動態障礙] N 個: #i pos(x,y) v(vx,vy)|spd 尺寸(...) [YOLO人]`，空場景印 `0 個`。

### 關鍵輸出 topic

| Topic | Type | 說明 |
|---|---|---|
| `/onboard_detector/dynamic_bboxes` | MarkerArray | 動態障礙框（藍，已分類為 dynamic），frame=**odom**（vis_frame）|
| `/onboard_detector/dynamic_point_cloud` | PointCloud2 | 動態點雲 |
| `/vo_interface/tracked_obstacles` | `vo_interface/TrackedObstacleArray` | **給 VO 規劃器**：持久 ID/平滑速度/協方差/age |
| `/vo_interface/markers` | MarkerArray | RViz 視覺化（粉紅速度箭頭 + ID/age 文字）|

`TrackedObstacle.msg` 欄位：`id`(持久) `age`(秒) `position` `velocity`(平滑絕對速度,odom frame)
`size` `radius`(=0.5·hypot(x,y),Minkowski 用) `vel_confidence`(0~1) `position/velocity_covariance[4]`(PVO 用)。
vo_interface 純訂閱 dynamic_bboxes、自做 CV-Kalman 重追蹤（LV-DOT 原生 box.id 是每幀索引非持久，
且 LiDAR-only 速度偏跳）；KF coasting 還會補平 dynamic_bboxes 的斷續發布 → VO 拿到連續障礙物流。

### RViz

```bash
ros2 run rviz2 rviz2 -d /home/aa/rviz/lvdot.rviz   # 或 lv-dot use_rviz:=true（X11 有 DISPLAY 時）
```
`lvdot.rviz` 已含 DynamicBBoxes(藍) / VO_TrackedObstacles(粉紅箭頭) / DynamicPoints / LidarClusters。

### 偵測覆蓋與調校現況（2026-06-15）

兩條分類路徑：**YOLO 快速通道**（`is_human`，相機 FOV ±43° 內，YOLO 認出人→直接判動態）vs
**LiDAR-only**（側/後方，靠速度投票+位移+一致性，門檻較嚴）。實測各方位覆蓋率差異大：

| 方位 | 覆蓋 | 狀態 |
|---|---|---|
| **前**（相機 FOV） | YOLO 看到人→**99%** 變動態框 | ✅ 基本解決 |
| 側 | ~16% | LiDAR-only + 部分超出相機 FOV |
| 後 | ~2% | 硬體盲區（無相機 + VLP-16 稀疏，疑遮蔽）|

**前方兩個關鍵修正**（從 31%/4% 一路救到 99%）：
1. **YOLO 模型 `yolo11n→yolo11s` + 推論解析度 `352→640`**（`yolov11_detector.py`；weights/yolo11s.pt 已入庫）
   → YOLO person recall 55%→**100%**。推論 23→40ms（仍 < 67ms 相機週期）。
2. **`is_human` 跳過尺寸約束**（`dynamicDetector.cpp` classificationCB）→ P(動態|YOLO人) 4%→**99%**。
   原因：VLP-16 稀疏下人物融合框 z_width 常 <0.5（只掃到上半身），被人形尺寸 `target_constrain_size`
   容差誤殺。YOLO 已確認是人不該再用尺寸濾；LiDAR-only 候選**仍受約束**濾家具（淨空誤報實測 0%）。

**LiDAR-only（側/後方）救活的調校**（記憶 lvdot-ros2-port 有全紀錄）：外參改 TF 實測值、停用隨機降採樣/
force_dynamic 自鎖閂、位移閘門隨可用歷史縮放、`max_match_range` 0.5→0.8、`image_cols` 848→640。
側/後方仍受 VLP-16 稀疏硬體限制（YOLO 照不到），這是天花板。對 VO 避障，最該顧的前方扇區已穩。

**⚠ 診斷覆蓋率的方法論教訓**：用「移動 LiDAR 簇」當 ground truth **嚴重低估**（給 39%）；用「YOLO 看到人」
當 GT 才準（同場景 4%）。量覆蓋率務必選可靠 GT，否則會追錯瓶頸。

### 故障排除（LV-DOT 專屬）

| 症狀 | 真因 | 處置 |
|---|---|---|
| 藍框/traj 在 RViz 延遲 ~2 秒 | **RViz QoS `Depth:100` 訊息積壓**（非 pipeline！資料其實即時，粉紅箭頭即時可證）| lvdot.rviz 已改 `Depth:1`；重載 RViz。X11 轉送畫線框本身偏重，治本=RViz 跑 PC 本機走 zenoh |
| 動態框抓固定點雜訊 / 真人難抓 | 外參範例值（已修為 TF 實測）/ 隨機降採樣假速度 / force_dynamic 自鎖閂 | 已全修，見記憶 lvdot-ros2-port |
| 前方的人 YOLO 有抓到卻不變動態框 | `target_constrain_size` 把 YOLO 確認的人（z_width<0.5）誤殺 | 已修：is_human 跳過尺寸約束（見上「偵測覆蓋」段）|
| 前方 YOLO recall 低（人在前卻常漏） | YOLO 用 nano 模型 + 352 低解析度 | 已改 yolo11s + 640（weights/yolo11s.pt）|
| `[python-N] libEGL warning: DRI3: failed to query the version` | **無害警告**（libEGL 在無 GL 顯示脈絡下查 DRI3 失敗，matplotlib/GL python 節點都會印）| **忽略**，不影響偵測 |
| YOLO 每幀洗版 `0: 352x352 N chairs` | ultralytics 預設 verbose | 已設 `verbose=False`（yolov11_detector.py:92）|
| 重啟後 YOLO/vo 雙開 | 上次沒收乾淨殘留 | 已修（stale_cleanup）；直接 `lv-dot` 重啟即可 |
| dynamic_bboxes frame 對不上 | vis_frame | odom mode 下發 odom frame（=LV-DOT world frame）；RViz Fixed Frame 用 odom |

## 安全條款（絕對遵守）

1. **第一次跑：架空 + 遙控器隨時待命**
2. **超過訓練 distribution 的場景**（走廊比訓練窄、障礙物比訓練密）— 先模擬、別貿然上電
3. **policy 異常 = 立刻 E-stop**，不要嘗試「再跑一下看看」
4. **任何 cmd_vel 超出 [-1, 1] m/s 或 [-2, 2] rad/s** = 通訊或匯出錯誤，停車檢查
5. **改 yaml 之前一定 copy 一份**：`cp policy_params.yaml policy_params.yaml.bak`
