# rover_rl — SA1_v2 RL Policy Deployment

把 IsaacLab 訓練的 **SA1_v2** (Asymmetric Critic + RNN64 + Vel Aux 13D) 模型部署到實車。
目標平台：**Ubuntu 22.04 + ROS 2 Humble**。

## 訓練側對齊規格（不要動）

| 項目 | 值 |
|---|---|
| Obs 維度 | 79D = ego(4) + goal(2) + LiDAR(72) + time(1) |
| Action | MultiDiscrete([19, 19]) → 38 logits → (linear_vel, angular_vel) |
| 物理上限 | v_max=1.0 m/s, a_max=0.5 m/s², ω_max(action)=2.0 rad/s |
| Obs normalizer | a/1.0, v/1.0, ω/1.5（注意 ω obs 用 1.5，動作端用 2.0） |
| 機器人半徑 | 0.35 m |
| 控制週期 | dt = 0.2 s (5 Hz) |
| LiDAR | VLP-16 → 72-bin sweep, r_min=0.9, r_max=20.0, z_filter=0.5 |
| Network | LidarStateExtractor(96D) → PreprocessRNN(hidden=64, preprocess=12) → PolicyHead(91→...→38) |

## Workspace 結構

```
rover_rl/
├── lidar_preprocess.py              ← 原 reference 實作（保留）
├── models/                          ← 放匯出的 .ts 檔
├── src/
│   ├── rover_rl_inference/         ← Python ROS 2 node
│   │   └── rover_rl_inference/
│   │       ├── policy_node.py      ← ROS 2 主節點
│   │       ├── lidar_preprocess.py ← 點雲 → 72-bin sweep
│   │       ├── obs_builder.py      ← 組 79D obs
│   │       ├── action_decoder.py   ← 38 logits → cmd_vel
│   │       ├── model_runtime.py    ← TorchScript runner + hidden state
│   │       └── export_policy.py    ← checkpoint → .ts
│   └── rover_rl_bringup/           ← launch + config
│       ├── launch/deploy.launch.py
│       └── config/policy_params.yaml
```

## 部署步驟

### 0. 前置（IsaacLab 訓練端 PC，匯出模型）

```bash
cd /home/aa/IsaacLab
conda activate env_isaaclab

python -m rover_rl.src.rover_rl_inference.rover_rl_inference.export_policy \
  --checkpoint /path/to/checkpoint_xxx.pt \
  --output /home/aa/IsaacLab/rover_rl/models/sa1_v2_policy.ts \
  --hidden-dim 64 --preprocess-dim 12 --fc-dim 64 --middle-dim 48 \
  --rnn-type RNN --used-obs-dim 79
```

### 1. 實車 ROS 2 端

```bash
# 建立 colcon workspace
cd ~/                                # 或您偏好的位置
mkdir -p rover_rl_ws
cp -r /path/to/IsaacLab/rover_rl/src rover_rl_ws/
cp -r /path/to/IsaacLab/rover_rl/models rover_rl_ws/

# 依賴
sudo apt install ros-humble-velodyne ros-humble-tf2-ros python3-numpy
pip install torch                    # 或 pip install torch --index-url https://download.pytorch.org/whl/cpu

cd rover_rl_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. 修改 `policy_params.yaml`

最重要的 4 件事：
1. `model_path`: 改成實車上 `.ts` 的絕對路徑
2. `topic_lidar` / `topic_odom` / `topic_cmd_vel` / `topic_goal`: 改成實車 topic 名稱
3. `lidar_yaw_offset_deg`: 量 sensor 朝向，校正成 base_link x_front
4. `goal_frame`: 有 `/map` 用 map；沒有則用 `odom`

### 3. 啟動

```bash
# Terminal A: VLP-16 driver
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py

# Terminal B: 您自家 odometry / SLAM stack（產 /odom 或 /map → odom TF）

# Terminal C: policy node
ros2 launch rover_rl_bringup deploy.launch.py

# Terminal D: 用 RViz 點選 2D Goal Pose（會自動發到 /goal_pose）
ros2 run rviz2 rviz2
```

## Sim-to-Real 校驗 Checklist

1. **LiDAR yaw offset**：車身前進方向 = LiDAR x_front 嗎？若否，量出偏角填到 `lidar_yaw_offset_deg`
2. **z_filter**：sensor 安裝高度若 < 0.5m，會把地板誤判為障礙 → 提高 z_filter 或抬高安裝
3. **r_min=0.9m**：實機 VLP-16 盲區實測值。若您的盲區不同請覆寫
4. **cmd_vel 反應時間**：訓練 dt=0.2s，底盤應在 0.2s 內收到並執行；若底盤延遲 > 100ms 建議再縮減 v_max
5. **odom 校驗**：靜止時 `linear.x` 與 `angular.z` 應 ≈ 0；轉一圈量 yaw 累積誤差
6. **goal 容忍**：訓練 0.5m，實車可放大到 0.8~1.0m 避免在 goal 附近震盪
7. **emergency stop**：第一次部署設高一點（如 0.4m），確認 policy 不會撞才放寬

## 仍需用戶提供（部署前）

請填以下空格後我可以針對您的硬體調整 `policy_params.yaml` 與 launch 檔：

| 欄位 | 用途 | 您填 |
|---|---|---|
| 車型／底盤 | 確認 v_max、ω_max 物理可達 | _____ |
| LiDAR topic 名 | sensor_msgs/PointCloud2 | _____ |
| LiDAR QoS | best_effort / reliable | _____ |
| LiDAR frame_id | TF 對齊 base_link | _____ |
| LiDAR 安裝高度 z | z_filter 校正 | _____ |
| Odom topic 名 | nav_msgs/Odometry | _____ |
| cmd_vel topic 名 | Twist | _____ |
| cmd_vel msg type | Twist / TwistStamped | _____ |
| 是否有 /map | 影響 goal_frame 設定 | _____ |
| GPU on 實車？ | CPU vs CUDA inference | _____ |

## 已知限制

- **只匯出 actor**：critic / aux predict head / obs_policy 沒匯出（部署不需要）
- **單環境推論**：batch=1, hidden state 自管，episode reset 在收到新 `/goal_pose` 時觸發
- **無 dynamic obstacle TopK**：訓練 SA1_v2 已不用 60D ground-truth 障礙物（asymmetric critic 只給訓練端），這對部署有利
- **TorchScript 可移植性**：CPU 與 GPU 共用同一 `.ts`，由 `device` 參數切換
