# rover_rl — 給車上 Claude Code 的指南

> 此 repo 由 PC 端訓練組（/home/aa/IsaacLab）匯出，部署到 CampusRover 實車。
> 您（車上的 Claude）的任務：**驗證 ROS 通訊運作正常，然後協助首次上電部署**。

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

## 參考：rover2_ws 既有 stack

實車上已有 ROS 2 stack 在 `/home/aa/rover2_ws/`，部分元件可重用：

| 路徑 | 用途 | rover_rl 整合方式 |
|---|---|---|
| `src/campusrover_base/` | 底盤 driver + URDF + TF | 不動。提供 `/odom` 與 `base_footprint`/`velodyne_link` TF |
| `src/spot_rl/` | 舊版 spot RL policy（spot_model + warp_device_ros） | **參考，不依賴**。rover_rl 是獨立新版 |
| `src/campusrover_navigation/` | 地圖 + costmap + planner | 可提供 `/goal_pose` 來源（RViz Nav2 goal） |
| `launch_rl.sh` | 舊 RL 完整啟動腳本 | **照抄環境設定**：ROS_DOMAIN_ID=30, RMW=fastrtps |
| `/home/aa/maps/4v3F.yaml` | 預設地圖 | 給 NDT localizer 用 |

### 注意：與 spot_rl 的差異

- spot_rl 用 spot_model（不同 obs 維度、不同 action space）
- rover_rl 用 SA6_TC checkpoint（VLP16 + 72-bin sweep + 19×19 discrete action）
- **兩者不可互換 checkpoint**

## ROS 通訊驗證流程（首要任務）

### Step 1：環境檢查

```bash
source /opt/ros/humble/setup.bash      # 或 jazzy，依車上版本
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

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
cd ~/rover_rl_ws            # 假設 user 把 src/ 放這裡
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rover_rl_inference rover_rl_bringup
source install/setup.bash
```

**常見錯誤**：
- `ModuleNotFoundError: torch` → Jetson 用 NVIDIA wheel，見 DEPLOY_CAMPUSROVER.md
- `Cannot find sensor_msgs` → 沒 source ROS 環境

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

# Terminal B: 啟動 rover_rl policy
cd ~/rover_rl_ws && source install/setup.bash
ros2 launch rover_rl_bringup deploy.launch.py log_level:=debug

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

## 故障排除清單

| 症狀 | 檢查 | 修復 |
|---|---|---|
| policy_node 啟動 crash | `journalctl -xe` 或 launch output | 99% 是 `model_path` 錯或 torch 沒裝 |
| /input/nav_cmd_vel 沒輸出 | `ros2 topic info /input/nav_cmd_vel -v` | 確認 mux config 接此 topic；或直接改 `topic_cmd_vel: /cmd_vel` 測試 |
| 永遠輸出 0 cmd | log 看 `lidar timeout` 或 `odom timeout` | LiDAR/odom 訊號中斷；檢查 hz |
| Goal 永遠收不到 | `goal_frame` 設錯 | 改 `goal_frame: odom`（沒 map）或 `map`（有 NDT/AMCL） |
| 跑起來但車原地震 | normalizer 期望 139D 你給 79D（或反） | 看 launch log 的 `raw_obs=X used_obs=Y`，與 model 對照 |
| cmd 振幅異常大 | normalizer mean/var 沒 bake 進 model | 重新 export_policy.py（必須帶有 obs_normalizer 的 checkpoint） |
| RViz Nav2 goal 不被收 | topic remap | 確認 `/goal_pose` 是 Nav2 standard，不是 `/move_base_simple/goal` |

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
   ├─→ rover_rl policy_node ── 72-bin sweep ─→ RL ─→ cmd_vel
   │
   └─→ rover2_ws bev_node ── 極座標影像 ─→ /bev_polar_image  (可選，純 debug)
```

### 是否啟動 BEV

**建議啟動**，作為 debug 工具：
- 上電前肉眼確認 LiDAR 看得到牆/障礙物
- 排查 sensor 異常時直接看極座標圖比 raw PointCloud2 直覺
- **不要把 BEV 程式抄進 rover_rl**，重用 rover2_ws 的版本就好

### 啟動方式（車上 Claude 不需要寫，照抄）

```bash
# 啟動 rover2_ws 的 bev_node（與 rover_rl 並行）
cd /home/aa/rover2_ws && source install/setup.bash
ros2 launch campusrover_rl_policy bev.launch.py use_rviz:=true
# 或 use_image_view:=true 用 rqt_image_view 看
```

預期 RViz 看到：
- 中央機器人（圈）
- 72-bin 極座標 LiDAR sweep（圓周上的距離值）
- 障礙物在對應角度顯示為近距離 bin

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

## sim-to-real 已知 gap（按嚴重度排序）

1. **LiDAR 高度 1.6m (訓練) vs 1.43m (實車)** — beam 角度落點不同
2. **訓練 ω_max=2.0 vs 底盤 1.2** — 全力轉時實際比 policy 預期慢 40%
3. **wheel 不對稱 1.9%** — odom drift 1%/m
4. **訓練 T 走廊 vs 實車任意場景** — 泛化性未驗證

## 與 PC 端的溝通介面

PC 端 repo：`/home/aa/IsaacLab/rover_rl`（私 SSH origin: `me0608623/rover_rl.git`）

需要 PC 端配合的事：
- 重訓 model（改 LiDAR 高度 / open scene / 新場景）
- 匯出新的 .ts（用 `scripts/export_policy.py`，已 auto-detect 架構）
- 提供新版 obs / action 規格

## 安全條款（絕對遵守）

1. **第一次跑：架空 + 遙控器隨時待命**
2. **超過訓練 distribution 的場景**（走廊比訓練窄、障礙物比訓練密）— 先模擬、別貿然上電
3. **policy 異常 = 立刻 E-stop**，不要嘗試「再跑一下看看」
4. **任何 cmd_vel 超出 [-1, 1] m/s 或 [-2, 2] rad/s** = 通訊或匯出錯誤，停車檢查
5. **改 yaml 之前一定 copy 一份**：`cp policy_params.yaml policy_params.yaml.bak`
