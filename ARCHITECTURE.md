# rover_rl 架構文件

> SA6_TC RL policy 部署在 CampusRover 實車（Jetson AGX Orin + ROS 2 Humble）的完整架構說明。
> 所有節點、topic、service 一覽 + 與既有 rover2_ws / NDT 整合方式。

---

## 1. 系統總覽

### 1.1 角色定位

`rover_rl` 是一個**獨立的 ROS 2 workspace**，負責把 IsaacLab 訓練好的 RL policy 部署到實車。
它**不取代** rover2_ws 的既有 stack，而是平行運作：

```
┌────────────────────────────────────────────────────────────────┐
│ 既有 stack（rover2_ws、ndt_ws）                                  │
│  • campusrover_base   — 底盤 driver、URDF、TF                   │
│  • ndt_localizer      — NDT 定位，發 map→odom                   │
│  • velodyne driver    — VLP-16 PointCloud2                      │
│  • campusrover_routing— 拓樸路徑規劃（service）                 │
│  • lcr_cmd_vel_mux    — cmd_vel 多路復用                        │
│  • bev_node           — BEV 視覺化（rover2_ws 內，可選）        │
├────────────────────────────────────────────────────────────────┤
│ rover_rl（本 repo，3 個節點）                                    │
│  • lidar_preprocessor — PointCloud2 → 72-bin sweep              │
│  • policy_node        — RL 推論 → cmd_vel                       │
│  • ros_smoke_test     — 純通訊驗證（部署前用）                  │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 資料流總圖

```
        velodyne_driver
              │
              │ /velodyne_points (PointCloud2, 10Hz)
              │
              ├─────────────────────────┐
              │                         │
              ▼                         ▼
   ┌────────────────────┐    ┌────────────────────┐
   │ rover_rl_lidar_    │    │ rover2_ws bev_node │ (純可視化)
   │ preprocessor       │    │                    │
   └────────────────────┘    └────────────────────┘
              │                         │
   /rover_rl/lidar_sweep_72   /bev_polar_image
   (Float32MultiArray[72])    (Image, for RViz)
              │
              │
   ┌─────────────────────────────────────────────────┐
   │              rover_rl_policy                    │
   │                                                 │
   │  訂閱：sweep, odom, ndt_pose, goal, path, mode  │
   │  推論：5 Hz RL → target cmd                    │
   │  Filter: low-pass + slew-rate → 20 Hz cmd_vel  │
   │  發布：cmd_vel, markers, obs_debug              │
   │  Services: set_mode, load_model, reset_hidden  │
   └─────────────────────────────────────────────────┘
              │
              │ /input/nav_cmd_vel (Twist, 20Hz, filtered)
              ▼
        lcr_cmd_vel_mux ──→ /output/cmd_vel ──→ 底盤 driver
```

---

## 2. 節點清單與職責

### 2.1 `rover_rl_lidar_preprocessor`

**Package**: `rover_rl_inference`
**Executable**: `lidar_preprocessor`
**檔案**: `rover_rl_inference/lidar_preprocessor_node.py`
**控制週期**: 10 Hz（與 VLP-16 同步）

#### 職責
把原始 VLP-16 PointCloud2 轉成 RL 模型直接可吃的 **72-bin normalized sweep**，
邏輯與 IsaacLab 訓練端 `wd_like_sweep_72()` 完全對齊（避免 sim-to-real obs mismatch）。

#### 處理流程
1. 訂閱 `PointCloud2` → numpy `[R, 3]`
2. 過濾 `|z| > 0.5m` 的地板/天花板點
3. 過濾 `r < r_min=0.9m`（VLP-16 硬體盲區）與 `r > r_max=20m`
4. 計算每點 `atan2(y, x)` → 分到 72 個 5° bins
5. 每 bin 取**最小**距離（min-pool）
6. 正規化 `(d - r_robot) / (r_max - r_robot)` → `[0, 1]`
7. 可選：依 odom 速度做 motion compensation（補償掃描期間位移）

#### 訂閱 topics

| Topic | Type | QoS | 用途 |
|---|---|---|---|
| `/velodyne_points` | `sensor_msgs/PointCloud2` | BEST_EFFORT | 原始 LiDAR |
| `/odom` | `nav_msgs/Odometry` | RELIABLE | motion compensation 用 |

#### 發布 topics

| Topic | Type | 頻率 | 用途 |
|---|---|---|---|
| `/rover_rl/lidar_sweep_72` | `std_msgs/Float32MultiArray` | 10 Hz | **policy_node 訂閱** |
| `/rover_rl/lidar_scan` | `sensor_msgs/LaserScan` | 10 Hz | RViz / costmap 視覺化（可選） |

#### Services

無

#### 關鍵參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `r_min` | 0.9 | VLP-16 盲區（**訓練值，勿改**） |
| `r_max` | 20.0 | 最遠有效距離（**訓練值**） |
| `r_robot` | 0.35 | 機器人半徑（**訓練值**） |
| `z_filter` | 0.5 | sensor frame z 過濾（**訓練值**） |
| `num_bins` | 72 | 角度 bin 數（**訓練值**） |
| `motion_compensation` | true | 補償 LiDAR 掃描期間位移 |
| `publish_laserscan` | true | 是否同時發 LaserScan |
| `laserscan_frame` | velodyne_link | LaserScan 的 frame_id |

---

### 2.2 `rover_rl_policy`

**Package**: `rover_rl_inference`
**Executable**: `policy_node`
**檔案**: `rover_rl_inference/policy_node.py`
**控制週期**:
- inference 5 Hz（control_dt=0.2s，**訓練 dt 一致，勿改**）
- cmd_vel republish 20 Hz
- markers 10 Hz
- heartbeat 0.5 Hz

#### 職責
主推論節點。把感測器資料組成 79D / 139D obs → 跑 RNN policy → 解碼成 `Twist` cmd_vel。
含定位（NDT cached offset）、subgoal 選擇（Path lookahead 或單目標）、cmd 過濾、多模式管理。

#### 處理流程（inference 一個 tick）
1. 取最新 sweep / odom / goal / NDT
2. 用 `MapOdomOffsetTracker` 算出機器人在 map frame 的 pose
3. `SubgoalSelector` 選 current subgoal（path lookahead 或 single goal）
4. `world_to_body` 把 subgoal 投到 body frame → `(gx, gy)`
5. 組 obs vector（ego 4D + goal 2D + LiDAR 72D + time 1D = 79D；或 139D w/ 60D zeros）
6. RNN forward：`extractor → preprocess → policy_head → 38 logits`
7. `decode_logits_to_cmd` → `(target_v, target_w, accel)`
8. 存到 `_target_v`/`_target_w`，等 cmd_timer 取用

cmd_timer (20 Hz) 獨立執行：
1. 讀 `_target_v`/`_target_w`
2. `CmdFilter`: low-pass(α=0.3) + slew-rate(±1 m/s²) + deadband
3. 發 `Twist` 到 `/input/nav_cmd_vel`

#### 訂閱 topics

| Topic | Type | 用途 | 必要性 |
|---|---|---|---|
| `/rover_rl/lidar_sweep_72` | `Float32MultiArray` | 已預處理 LiDAR | **必要** |
| `/velodyne_points` | `PointCloud2` | fallback (use_inline_preprocess=true 時) | 選用 |
| `/odom` | `Odometry` | 機器人速度 + odom-frame pose | **必要** |
| `/goal_pose` | `PoseStamped` | 單一目標（RViz Nav2 Goal） | 至少一個 |
| `/global_path` | `nav_msgs/Path` | 多 waypoint 路徑 | 至少一個 |
| `/ndt_pose` | `PoseStamped` | NDT 定位（map frame） | 可選（無則 fallback odom） |
| `~/mode` | `std_msgs/String` | 動態切 mode | 可選 |

#### 發布 topics

| Topic | Type | 頻率 | 用途 |
|---|---|---|---|
| `/input/nav_cmd_vel` | `Twist` | 20 Hz | 進 lcr_cmd_vel_mux → 底盤 |
| `~/markers` | `MarkerArray` | 10 Hz | RViz 視覺化 |
| `~/obs_debug` | `Float32MultiArray` | 5 Hz | obs 內容（可選） |

#### Services

| Service | Type | 用途 |
|---|---|---|
| `~/set_mode` | `std_srvs/SetBool` | true=nav, false=idle（快速切） |
| `~/load_model` | `std_srvs/Trigger` | hot-swap `model_path` 指向的 .ts |
| `~/reset_hidden` | `std_srvs/Trigger` | 重置 RNN hidden + cmd filter |

#### Mode 切換對照

| Mode | 推論 | cmd_vel 發布 | 用途 |
|---|---|---|---|
| `nav` | ✓ | ✓（policy 輸出） | 正常運作 |
| `idle` | ✗ | ✓（強制 0） | 待命 |
| `estop` | ✗ | ✓（強制 0） | 緊急停車 |
| `manual` | ✗ | ✗（讓出 topic） | 搖桿接管 |
| `paused` | ✗ | ✓（強制 0） | 暫停但保留 hidden state |

#### 關鍵參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `model_path` | "" | .ts 檔絕對路徑（**必填**） |
| `device` | cuda:0 | cpu / cuda:0 |
| `control_dt` | 0.2 | inference 週期（5 Hz，**勿改**） |
| `cmd_rate_hz` | 20.0 | cmd_vel republish 頻率 |
| `marker_rate_hz` | 10.0 | markers 發布頻率 |
| `goal_frame` | "map" | map / odom |
| `base_frame` | "base_footprint" | 機器人 footprint frame |
| `path_lookahead_m` | 2.0 | path 模式 carrot 前瞻距離 |
| `goal_tolerance_m` | 0.6 | 到達 goal 容忍距離 |
| `require_ndt` | false | true=NDT 未穩定就拒絕動 |
| `safety_lidar_emergency_stop_m` | 0.40 | LiDAR < 此距離強制 ESTOP |
| `cmd_alpha_linear` | 0.3 | low-pass α |
| `cmd_max_accel_linear` | 1.0 | slew-rate m/s² |
| `act_max_linear_velocity` | 1.0 | 動作上限（**訓練值，勿放寬**） |
| `act_max_angular_velocity` | 2.0 | 動作上限（**訓練值**） |
| `initial_mode` | "nav" | 啟動 mode |

---

### 2.3 `rover_rl_smoke` (測試用)

**Package**: `rover_rl_inference`
**Executable**: `ros_smoke_test`
**檔案**: `rover_rl_inference/ros_smoke_test.py`

#### 職責
**部署前驗證 ROS 通訊正常**的測試節點，**不需 torch**。
跑這個確認所有 topic 都正常後再啟動 `policy_node`。

#### 訂閱 / 發布

| Topic | 方向 | Type |
|---|---|---|
| `/velodyne_points` | sub | PointCloud2 |
| `/odom` | sub | Odometry |
| `/goal_pose` | sub | PoseStamped |
| `/input/nav_cmd_vel` | pub（可選） | Twist (random tiny twist, 5 Hz) |

每 2 秒 log 一次收訊計數與 Hz。

---

## 3. 整合既有 stack

### 3.1 完整 TF tree（運行時）

```
world
  └── map                  (static, NDT 啟動時 tf_static_publisher 發)
        └── odom            (動態, ndt_localizer_node 發 map→odom)
              └── base_link       (campusrover_base driver 發 odom→base_link)
                    ├── base_footprint   (URDF static)
                    ├── velodyne_link    (URDF static)
                    └── imu_link         (URDF static)
```

| TF 段 | 誰負責 | 啟動 launch |
|---|---|---|
| world → map | NDT tf_static | `ros2 launch ndt_localizer ndt_localizer_launch.py` |
| **map → odom** | ndt_localizer_node | 同上 |
| odom → base_link | campusrover_base driver | rover2_ws bringup |
| base_link → 子 frames | URDF + robot_state_publisher | 同上 |

### 3.2 啟動順序（用戶手動，rover_rl 不接管）

```bash
# Terminal 1: NDT 定位
cd ~/Documents/ndt_ws && source install/setup.bash
ros2 launch ndt_localizer ndt_localizer_launch.py

# Terminal 2: 底盤 + URDF + TF
cd ~/rover2_ws && source install/setup.bash
ros2 launch campusrover_base <bringup>.launch.py     # 視實際 launch 名

# Terminal 3: VLP-16
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py

# Terminal 4: rover_rl（包 preprocessor + policy + bev）
cd ~/rover_rl_ws && source install/setup.bash
source ~/rover2_ws/install/setup.bash    # ★ 才找得到 bev_node
ros2 launch rover_rl_bringup deploy_with_bev.launch.py

# Terminal 5: 路徑/目標來源（擇一）
# 5a. Manual goal: RViz Nav2 Goal 點選
ros2 run rviz2 rviz2

# 5b. campusrover_routing（service-based topology planner）
#     需要中間節點呼叫 RoutingPath service 並 publish 到 /global_path
#     —— 目前 rover_rl 沒做這個 bridge，用戶若要可請求加上
```

### 3.3 與 lcr_cmd_vel_mux 整合

```
        rover_rl_policy           joy                emergency_stop
              │                    │                       │
              │                    │                       │
              ▼                    ▼                       ▼
   /input/nav_cmd_vel    /input/joy_cmd_vel       /input/stop_cmd_vel
              │                    │                       │
              └────────────────────┴───────────────────────┘
                                   │
                            lcr_cmd_vel_mux
                                   │
                                   ▼
                            /output/cmd_vel
                                   │
                                   ▼
                            底盤 driver
```

rover_rl 發到 `/input/nav_cmd_vel`，**不要直接發 `/cmd_vel`**（會繞過 mux，搖桿無法接管）。

### 3.4 與 BEV 視覺化整合

`rover2_ws/src/campusrover_navigation/campusrover_rl_policy/launch/bev.launch.py`
獨立節點，與 rover_rl 並行；訂閱同一個 `/velodyne_points`。

```
/velodyne_points
    ├─→ rover_rl_lidar_preprocessor → /rover_rl/lidar_sweep_72 → policy_node
    └─→ rover2_ws bev_node          → /bev_polar_image (給 RViz 看)
```

兩條獨立分支，互不干擾。

---

## 4. 模型架構（內部，給研發參考）

```
obs_raw [B, 139]                     (139D for SA6_TC; 79D for SA1_v2)
    │
    │  normalize (running mean/var from training)
    │  + index_select [0..77, 138]  →  obs_79d [B, 79]
    ▼
LidarStateExtractor                  (2-branch: Conv1d LiDAR + MLP State)
    │  out: [B, 96]
    ▼
PreprocessRNN (hidden=30 or 64)      (vanilla RNN, 維護 episode 內 hidden state)
    │  out: preprocess_feat [B, 12]
    ▼
cat(obs_79d, preprocess_feat) → [B, 91]
    │
    ▼
PolicyHead (91→256→256→256→512→38)
    │  out: logits [B, 38]
    ▼
split into 19 + 19 (linear_accel idx + omega idx)
    │
    │  argmax → idx_a, idx_w ∈ [0, 18]
    │  decode with dynamic acceleration bounds
    ▼
(linear_vel_cmd, angular_vel_cmd)  ∈ [-1.0, 1.0] m/s × [-2.0, 2.0] rad/s
```

---

## 5. 可用 model checkpoints

| Model | 訓練場景 | Steps | Arch | 預設用途 |
|---|---|---|---|---|
| `sa6_tc_dense_420k.ts` ⭐ | T 走廊 dense 障礙物 | 420k | hidden=30, 139D | **首選** |
| `sa7_tc_dense_300000.ts` | T 走廊 + occlusion + 8 dyn | 300k | hidden=30, 139D | 進階場景 |
| `sa5_tc_g1_p30_270000.ts` | T 走廊 g1_p30 | 270k | hidden=30, 139D | 早期備案 |

切換方式（不需重啟節點）：
```bash
ros2 param set /rover_rl_policy model_path /path/to/other_model.ts
ros2 service call /rover_rl_policy/load_model std_srvs/srv/Trigger
```

---

## 6. 偵錯指令快速參考

### 通訊驗證
```bash
# 確認感測器都在
ros2 topic hz /velodyne_points          # ~10 Hz
ros2 topic hz /odom                      # ~20 Hz
ros2 topic hz /ndt_pose                  # ~5-10 Hz（NDT 收斂後）

# 確認 preprocessor 工作
ros2 topic hz /rover_rl/lidar_sweep_72   # ~10 Hz
ros2 topic echo /rover_rl/lidar_sweep_72 --once   # 看 72 個 [0,1] 值

# 確認 policy 工作
ros2 topic hz /input/nav_cmd_vel         # ~20 Hz（filtered cmd）

# 確認 TF
ros2 run tf2_ros tf2_echo map base_footprint    # 穩定 = NDT chain 完整
```

### Mode 切換
```bash
# Emergency stop
ros2 topic pub --once /rover_rl_policy/mode std_msgs/String "data: 'estop'"

# 切 manual（讓搖桿接管）
ros2 topic pub --once /rover_rl_policy/mode std_msgs/String "data: 'manual'"

# 回 nav
ros2 service call /rover_rl_policy/set_mode std_srvs/srv/SetBool "{data: true}"

# 重置 RNN
ros2 service call /rover_rl_policy/reset_hidden std_srvs/srv/Trigger
```

### Heartbeat log 解讀
```
[HB] mode=nav sweep_src=topic | sweep_age=0.10s pc_age=0.05s odom_age=0.02s \
     ndt=yes(age=0.5s,offset=(+1.23,-0.45)) | target v=+0.50 w=+0.20
```
- `sweep_src=topic` → 從 preprocessor 收到（首選），`inline_fallback` 表示降級
- `ndt=yes(offset=...)` → NDT 穩定 + offset 已 cache
- `target v/w` → policy 推論結果（filter 前）

---

## 7. 安全條款（運行時）

1. **第一次跑：架空 + 遙控器隨時待命**
2. **Mode 預設 nav 才會動**；初次部署可 `initial_mode:=idle`，確認 cmd_vel 為 0 後再切 nav
3. **動作上限不可放寬**：即使底盤能跑更快，policy 沒見過 OOD 速度，會 extrapolate 失敗
4. **NDT 未穩定時 require_ndt=true** 會強制停車；想 fallback odom 設 false 自己承擔
5. **任何 cmd_vel 超出 [-1, 1] m/s 或 [-2, 2] rad/s** = 通訊或匯出錯誤，立即停車檢查
6. **`/rover_rl/lidar_sweep_72` 全部 = 1.0** 表示 LiDAR 沒讀到任何點 → 別上電

---

## 8. 與訓練端對齊保證

| 處理項 | 訓練 (obs_functions.py / charge_env_cfg_vlp16.py) | 部署 (lidar_preprocess.py / obs_builder.py) | 對齊 |
|---|---|---|---|
| LiDAR r_min | 0.9 | 0.9 | ✓ |
| LiDAR r_max | 20.0 | 20.0 | ✓ |
| LiDAR r_robot | 0.35 | 0.35 | ✓ |
| LiDAR z_filter | 0.5 | 0.5 | ✓ |
| Num bins | 72 | 72 | ✓ |
| obs[0] accel normalize | / 1.0 | / 1.0 | ✓ |
| obs[1] vel normalize | / 1.0 | / 1.0 | ✓ |
| obs[2] omega normalize | / 1.5 | / 1.5 | ✓ |
| obs[3] radius | 0.35 | 0.35 | ✓ |
| obs normalizer mean/var | running 統計 | baked into .ts | ✓ |
| Action v_max | 1.0 | 1.0 | ✓ |
| Action a_max | 0.5 | 0.5 | ✓ |
| Action ω_max | 2.0 | 2.0 | ✓ |
| Control dt | 0.2s | 0.2s | ✓ |

任何 mismatch → policy 表現必然不如預期。
