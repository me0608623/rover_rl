# CampusRover 部署清單（Jetson AGX Orin + ROS 2 Humble）

## 硬體規格鎖定（用戶提供 2026-06-01）

| 項目 | 值 | 對訓練的影響 |
|---|---|---|
| 底盤 | Differential drive | ✓ 訓練 model 假設 |
| 輪距 | 0.5592 m | odom 校準用 |
| 輪徑 | L 0.2442 / R 0.2396 m | odom 有偏差，goal 投影 long term 會 drift |
| 機器人半徑 | 實際 ~0.26 m | 訓練 0.35 m，模型偏保守，**不改** |
| LiDAR 安裝高度 | 1.43 m 距地 | 訓練 1.6 m，差 17 cm（**待重訓 SA1_v3 或接受**） |
| LiDAR yaw | 0° vs base_link | ✓ 無偏移 |
| 底盤上限 | v 1.5 / a 1.2 / ω 1.2 | 訓練 v 1.0 / a 0.5 / ω 2.0；**動作鎖在訓練內** |
| Compute | Jetson AGX Orin (iGPU + CUDA 12.6) | 用 `device: cuda:0` |
| ROS | Humble (Ubuntu 22.04.5) | ✓ |

## sim-to-real 已知 gap（影響部署）

1. **LiDAR 高度 1.43m vs 訓練 1.6m**
   - 影響：低矮障礙物的 hit 角度不同（離 sensor 同距離但 vertical 偏差會讓 ring index 落點不同）
   - 但您訓練 obs 用 `wd_like_sweep_72`（z_filter=0.5），只取水平 ±0.5m 區段，所以差 17cm 不會把地板/天花板誤入
   - 真正會被影響的是：1.93m 以上的低矮天花板（無）和 0.93m 以下的桌面/椅面（有些會落入過濾窗外）→ 邊緣案例可能有 phantom hit
   - 建議：先用現有 SA1_v2 model 部署，看實測 collision rate；若有問題重訓 v3 把高度設 1.43

2. **wheel 不對稱 (L 0.244 vs R 0.240, 1.9% 差)**
   - odom long-term drift 約 1%/m
   - 影響：goal 在 body frame 的 (Δx, Δy) 會慢慢偏，導致到達 tolerance 失準
   - 對策：goal_tolerance_m 已調大到 0.6；若仍誤差大，建議 driver 端校準 wheel diameter

3. **cmd_vel mux**
   - 不可直接發 `/cmd_vel`，會被 mux 鎖；必須發到 `/input/nav_cmd_vel`
   - 已在 `policy_params.yaml` 改正

## 安裝步驟（Jetson AGX Orin）

### A. PyTorch for Jetson (CUDA 12.6)

```bash
# Jetson 不能用 PyPI 的 wheel，必須用 NVIDIA 提供的版本
# 參考 https://developer.nvidia.com/embedded/downloads
# 或 conda-forge for aarch64
pip install --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
            torch torchvision

# 驗證
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 預期: 2.x.0 True
```

### B. ROS 2 workspace

```bash
mkdir -p ~/rover_rl_ws/src
cd ~/rover_rl_ws

# 從 PC-A 拷貝（assumes /home/aa/IsaacLab/rover_rl/ 已 rsync 到 Jetson）
cp -r ~/IsaacLab/rover_rl/src/* src/
mkdir -p models

# build
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### C. 匯出模型（在 PC-A 訓練端先做）

```bash
cd /home/aa/IsaacLab
conda activate env_isaaclab

# 找到 SA1_v2 checkpoint（目前最後一個）
ls logs/skrl/wd_sa1_v2*/checkpoint_*.pt | tail -1

python3 rover_rl/src/rover_rl_inference/rover_rl_inference/export_policy.py \
  --checkpoint logs/skrl/wd_sa1_v2_xxx/checkpoint_270000.pt \
  --output /tmp/sa1_v2_policy.ts \
  --hidden-dim 64 --preprocess-dim 12 --fc-dim 64 --middle-dim 48 \
  --rnn-type RNN --used-obs-dim 79

# 傳到 Jetson
scp /tmp/sa1_v2_policy.ts jetson@<IP>:~/rover_rl_ws/models/
```

### D. 上車測試

```bash
# Terminal 1: LiDAR driver
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py

# Terminal 2: rover driver（您 existing stack）
ros2 launch rover_driver rover_driver.launch.py

# Terminal 3: TF / odom（您 existing SLAM 或 robot_state_publisher）

# Terminal 4: policy node
cd ~/rover_rl_ws && source install/setup.bash
ros2 launch rover_rl_bringup deploy.launch.py

# Terminal 5: 監控
ros2 topic hz /input/nav_cmd_vel    # 應 ≈ 5 Hz
ros2 topic echo /input/nav_cmd_vel  # 看 linear/angular 數值是否合理
ros2 topic hz /velodyne_points      # 應 ≈ 10 Hz
ros2 topic hz /odom                 # 應 ≈ 20 Hz

# 送 goal（用 RViz 點 2D Goal Pose，或手動）
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 3.0, y: 0.0}, orientation: {w: 1.0}}}'
```

## 首次上電 safety protocol（極重要）

1. **離地測試**：把車架起來，發 goal，觀察輪子是否朝期望方向轉
2. **低速地板測試**：把 `act_max_linear_velocity` 暫時降到 0.3 m/s 試水溫
3. **遙控器 deadman 隨時待命**：lcr_cmd_vel_mux 可切回 joy 模式立即接管
4. **每次 episode 限時**：節點預設 `episode_horizon_s: 60.0`，超過會 `time_remaining=0`；policy 可能行為突變，建議第一次部署人為 60s 內把車叫停
5. **失敗回退**：如果 policy 異常，按 E-stop 後檢查 `ros2 topic echo /input/nav_cmd_vel`，確認是否 = 0

## 預期問題（事先準備）

| 症狀 | 原因 | 對策 |
|---|---|---|
| 車子原地震盪 | obs[2] ω 用 1.5 normalize，但底盤 ω_max 只有 1.2 → clip | 把 `act_max_angular_velocity` 降到 1.2 |
| Goal 永遠到不了 | odom drift 累積 | 縮短 goal 距離；或加 AMCL re-localize |
| 撞牆 | LiDAR 高度差導致 phantom hit | 重訓 SA1_v3 with `lidar_z=1.43` |
| Stop 太頻繁 | `safety_lidar_emergency_stop_m` 太大 | 降到 0.30；確認 r_min=0.9 把盲區擋住 |
| cmd_vel 沒輸出 | mux 不認 `/input/nav_cmd_vel` | `ros2 topic list \| grep nav_cmd_vel` 看名字；或檢查 mux config |

## 動作端 ω_max 衝突警告

訓練 `act_max_angular_velocity = 2.0 rad/s` 但底盤實測 `omega_max = 1.2 rad/s`。

當 policy 輸出 idx=18 (全右轉) 會發 `cmd_vel.angular.z = +2.0`，但底盤只能跑 1.2。
- **影響**：實際轉彎比預期慢 40% → policy 以為 "我已在轉" 但實際 yaw 變化更小 → 可能持續輸出全右
- **建議**：把 `act_max_angular_velocity` 改成 1.2，或重訓時把 max_angular_vel 改回符合實車的值

```yaml
# 保守做法：clip 到底盤能力
act_max_angular_velocity: 1.2
```

這會讓 policy 的 omega ratio 範圍從 [-2, 2] 縮成 [-1.2, 1.2]，但因為 obs[2] 還是用 1.5 normalize，
所以 policy 的「自我感知」不變，只是命令端被截短。建議實測後決定。
