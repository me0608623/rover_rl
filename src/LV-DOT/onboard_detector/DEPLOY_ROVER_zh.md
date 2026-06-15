# LV-DOT 在 CampusRover 上的部署筆記（ROS2 Humble）

> 此檔記錄把 LV-DOT（onboard_detector）從 ROS1 移植到 ROS2 Humble、並部署到 rover_rl 實車的方式。
> 上游 README 是 ROS1 用法，**實車請看本檔**。最後驗證：2026-06-04。

## 0. 一句話現況

LV-DOT 已移植成 ROS2 Humble，`colcon build` 通過。實車已驗證 **LiDAR + NDT 定位 + 相機 depth 融合 + YOLO 人形分類**全棧在 640×480 下可跑、無 crash。

## 1. 與上游的差異（移植重點）

- C++：`dynamicDetector` 改繼承 `rclcpp::Node`；publisher/subscription/wall_timer/service 全改 rclcpp；message_filters ApproximateTime sync 改 ROS2 API。
- srv：`GetDynamicObstacles.srv` 用 `rosidl_generate_interfaces` 產生。
- YOLO 節點：`rospy` → `rclpy`。
- build：catkin → `ament_cmake`；launch：`.launch` → `run_detector.launch.py`；cfg：ROS2 `/**:` 參數格式。
- vision_msgs API：`bbox.center.x` → `bbox.center.position.x`（ROS2 用 `vision_msgs/Pose2D`）。
- **未移植**：`fake_detector`（Gazebo 模擬用、依賴 gazebo_msgs），已從 build 排除。
- 坑：`dbscan.h` 的 `#define SUCCESS 0` 會撞 `rclcpp::FutureReturnCode::SUCCESS`，已用 `lidarDetector.h` 的 include 順序修掉（utils.h 先於 dbscan.h），**勿改回**。

## 2. 編譯

```bash
cd ~/rover_rl
source /opt/ros/humble/setup.bash
colcon build --packages-select onboard_detector --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## 3. topic 對齊（實車）

| 用途 | topic | 備註 |
|---|---|---|
| LiDAR | `/velodyne_points` | rover 本機 velodyne driver |
| 定位 pose | `/ndt_pose` (PoseStamped, map frame) | NDT，`localization_mode: 0` |
| odom（替代） | `/odom` | 若無 NDT，改 `localization_mode: 1` |
| 彩色 | `/camera/camera/color/image_raw` | 640×480×15，.13 的 D435I |
| 深度 | `/camera/camera/depth/image_rect_raw` | **必須 640×480**（見 §5） |

## 4. 相機（在 192.168.3.13）

相機 D435I 接在 `192.168.3.13`，由 systemd `realsense.service`（`/home/humble/start_realsense.sh`）開機自啟。

**重點：開機腳本必須 export zenoh RMW**，否則 node 跑在預設 FastDDS，rover（zenoh）看不到：
```bash
# /home/humble/start_realsense.sh
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_zenoh_cpp     # ← 缺這行相機就「看得到 topic 但 0 publisher」
exec ros2 launch realsense2_camera rs_launch.py \
    enable_depth:=true \
    depth_module.depth_profile:=640,480,30 \  # ← 對齊 LV-DOT cfg，避免 buffer overflow
    rgb_camera.color_profile:=640,480,15
```

排錯：
- rover 看不到 `/camera/camera` 或 publisher count=0 → 多半是 RMW 沒設成 zenoh。
- 開機時 dmesg 會閃一波 `usb 2-2: GET_CUR ... -32`(EPIPE)，~20s 後停、不影響串流，別誤判硬體壞。
- 別同時手動 launch 又讓 systemd 跑（雙開搶 USB → 裝置 re-enumerate → 卡死）。

## 5. ⚠️ depth 解析度必須對齊（重要）

LV-DOT `cfg/detector_param.yaml` 的 `image_cols/rows = 640/480`。
若相機 depth 串流不是 640×480（D435 預設 848×480），`projectDepthImage()` 的 `projPoints_` buffer 會 **overflow → crash**。
→ 已在相機開機腳本鎖定 `depth_module.depth_profile:=640,480,30`。換相機/改設定時務必保持一致。

## 6. YOLO 人形分類（GPU，venv 隔離）

系統 python3 的 torch 是 `2.10.0+cpu`（policy 在用、**勿動**）。為了讓 YOLO 吃 Orin GPU 又不影響 policy，
另開一個 venv 裝 **CUDA torch**，YOLO 節點用該 venv 的 python 跑：

```bash
# 一次性建置（已完成）
python3 -m venv --system-site-packages ~/yolo_venv
~/yolo_venv/bin/pip install -U pip
~/yolo_venv/bin/pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126   # JetPack6/CUDA12.6 預編譯 wheel
# ultralytics 沿用系統 ~/.local 那份(8.4.60)即可，import 時會用到 venv 的 CUDA torch
~/yolo_venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # → True (device: Orin)
```

- launch 已用 `prefix=[yolo_python]` 讓 YOLO 節點自動用 `~/yolo_venv/bin/python` → **GPU**。
  改用系統 python(CPU)：`ros2 launch ... yolo_python:=''`（或裝置沒 venv 時）。
- 實測：CPU 2.6s/幀 → **GPU ~23ms/幀（yolo_time≈0.029s）**，約 90×。
- venv 與 policy 的 torch 完全隔離（venv site-packages 在 sys.path 前面、自然蓋過 ~/.local 的 cpu torch；不需 PYTHONNOUSERSITE）。
- 不要 YOLO 時 `use_yolo:=false`，LiDAR+depth 融合照常。
- 移除：`rm -rf ~/yolo_venv`（policy 不受影響）。
- ⚠️ policy 本身仍跑 CPU torch（30-hidden RNN，本就只要幾 ms，不需 GPU）。若要 policy 也上 GPU 才需動系統 torch（風險高，見 git 筆記策略 B）。

## 7. 啟動 + RViz

```bash
source /opt/ros/humble/setup.bash && source ~/rover_rl/install/setup.bash && source ~/rover_rl/setup_env.sh
ros2 launch onboard_detector run_detector.launch.py use_yolo:=true use_rviz:=true
```

launch 參數：`params_file`（cfg）、`use_yolo`（預設 true）、`use_rviz`（預設 false）。

RViz 看（Fixed Frame = **map**）：
- MarkerArray → `/onboard_detector/dynamic_bboxes`、`/onboard_detector/filtered_bboxes`
- PointCloud2 → `/onboard_detector/lidar_clusters`、`/onboard_detector/filtered_depth_cloud`

## 8. 待辦（之後校正）

- cfg 相機內參仍是上游範例值；要精準改實機值：color 640×480 `fx614.28 cx325.55 cy239.97`、depth 640×480 `fx≈382.6 cx322.1 cy236.9`（`rs-enumerate-devices -c`）。
- `body_to_camera_*` / `body_to_lidar` 外參仍是範例值，要量實機相機/光達安裝位置。
