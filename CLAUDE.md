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

**rover_rl policy_node 採 spot_rl + rover2_ws 驗證過的 pattern**：
- 訂閱 `/ndt_pose`（NDT 只在 is_converged=True 時發）
- 用「近 1 秒收到且累積 ≥ 5 次」判定 NDT 穩定
- 機器人靜止 + NDT 穩定時 cache `map→odom` offset（每 5 秒重算，delta > 0.3m 拒絕）
- 即時計算：`robot_in_map = odom_xy + cached_offset`
- 手動算 body-frame goal：`goal_body = R(-yaw) · (goal_map - robot_map)`
- Fallback：NDT 未穩定 → 用 odom_only 來源（`require_ndt: false` 時允許）

**為何不直接用 `tf_buffer.transform(PoseStamped, target)`**？
- NDT 更新頻率低（1-10 Hz），TF lookup 可能 stale 或失敗
- cached offset 即使 NDT 暫時消失仍能跑（safer fallback）
- 跟 rover2_ws 既有 rl_policy_node 同 pattern，行為一致

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
deploy_rl
# 等同：
# source /opt/ros/humble/setup.bash
# source ~/rover2_ws/install/setup.bash
# source ~/rover_rl/install/setup.bash
# export ROS_DOMAIN_ID=55
# export RMW_IMPLEMENTATION=rmw_zenoh_cpp
# ros2 launch rover_rl_bringup deploy_full.launch.py
```

⚠️ 直接用這個 alias，不要手動拼環境指令。

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
- **CSV 存檔位置**: `~/rover_rl/logs/`（`log_dir` 參數預設，`deploy_full.launch.py` 亦設同值）
- **檔名規則**: `diag_<YYYYMMDD>_<HHMMSS>[_<實驗名>].csv`
  （`<實驗名>` 來自 start 時給的 label，經 `_safe_label()` 清特殊字元；不給就只有時間戳）
- **Ctrl+C 後**: 自動印「診斷摘要 + CSV 完整路徑 + analyze_diag 指令」
- **分析**: `ros2 run rover_rl_inference analyze_diag ~/rover_rl/logs/diag_<...>.csv`

## 故障排除清單

| 症狀 | 檢查 | 修復 |
|---|---|---|
| policy_node 啟動 crash | `journalctl -xe` 或 launch output | 99% 是 `model_path` 錯或 torch 沒裝 |
| /input/nav_cmd_vel 沒輸出 | `ros2 topic info /input/nav_cmd_vel -v` | 確認 mux config 接此 topic；或直接改 `topic_cmd_vel: /cmd_vel` 測試 |
| 永遠輸出 0 cmd | log 看 `lidar timeout` 或 `odom timeout` | LiDAR/odom 訊號中斷；檢查 hz |
| sweep_src=inline_fallback | preprocessor 節點死了 | `ros2 topic hz /rover_rl/lidar_sweep_72`；重啟 preprocessor |
| Goal 永遠收不到 | `goal_frame` 設錯 | 改 `goal_frame: odom`（沒 map）或 `map`（有 NDT/AMCL） |
| routing_click_bridge 找不到節點 | 地圖節點未啟 | 先確認 `/get_route_info` service 存在 |
| 跑起來但車原地震 | normalizer 期望 139D 你給 79D（或反） | 看 launch log 的 `raw_obs=X used_obs=Y`，與 model 對照 |
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

| Model | 訓練場景 | 建議使用情境 |
|---|---|---|
| `sa6_tc_dense_420k.ts` ⭐ | T 型走廊 dense 障礙物，420k steps | **首選**，最久訓練 + 中等難度 |
| `sa7_tc_dense_300000.ts` | T 走廊高壓（occlusion 20%, dyn=8） | 進階場景；首次部署別用 |
| `sa5_tc_g1_p30_270000.ts` | T 走廊 g1_p30（goal=1, penalty=-30） | 訓練早期；當前不建議 |

⚠️ **這三個都是 T-Corridor 課程**。實車場景若不是 T 走廊（例：開放廣場、長廊），policy 可能表現不佳。
如果實測有問題，跟 PC 端訓練組要求一個 open-scene baseline 重訓。

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
5. **無 cmd_delay 補償** — spot_rl 有 `cmd_delay_time=1.0s` 的死時間補償（`fast_info_calculate()`：基於預測的延遲位姿更新 obs 中的速度和目標座標）。rover_rl 沒有此機制；若底盤響應 latency 明顯（>0.2s），policy 可能振盪。初步觀察若有振盪，考慮加大 `cmd_alpha_linear` 濾波。
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

### RealSense 相機（開機自動啟動）

主機 `192.168.3.13` 開機會**自動啟動** RealSense 相機，影像透過 zenoh 傳到全網。
rover_rl 推論**不使用**此相機（policy 只吃 72-bin LiDAR sweep），相機純供
人工/AI 視覺確認場景用。

啟動指令（已設為開機自啟，列出供參考）：
```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_depth:=false rgb_camera.color_profile:=640,480,15
```

| Topic | Type | 說明 |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | 640×480 rgb8 @ 15fps |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 內參 |

frame_id = `camera_color_optical_frame`。透過 ROS MCP 看畫面：
```
subscribe_once(topic="/camera/camera/color/image_raw",
               msg_type="sensor_msgs/msg/Image", timeout=5, expects_image="true")
analyze_previously_received_image()
```

⚠️ depth 預設關閉（`enable_depth:=false`）。若要點雲/深度需另開 launch 參數。

### 與 rover_rl 部署整合

rosbridge 要在 rover2_ws / rover_rl 之前啟動，並使用相同 RMW 環境：
```bash
source ~/rover_rl/setup_env.sh   # ← 必須，確保 zenoh + DOMAIN_ID=55 一致
ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
```

若 rosbridge 沒有 source rover_rl 環境就啟動，會連到不同 zenoh domain，看不到 rover_rl 的 topics。

## 安全條款（絕對遵守）

1. **第一次跑：架空 + 遙控器隨時待命**
2. **超過訓練 distribution 的場景**（走廊比訓練窄、障礙物比訓練密）— 先模擬、別貿然上電
3. **policy 異常 = 立刻 E-stop**，不要嘗試「再跑一下看看」
4. **任何 cmd_vel 超出 [-1, 1] m/s 或 [-2, 2] rad/s** = 通訊或匯出錯誤，停車檢查
5. **改 yaml 之前一定 copy 一份**：`cp policy_params.yaml policy_params.yaml.bak`
