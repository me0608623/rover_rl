#!/bin/bash
# deploy_full_stop：停止 deploy_full 棧 + NDT + LV-DOT，並清掉殘留殭屍。
# 從檔案執行 → 本進程 cmdline 不含目標關鍵字，pkill 不會誤殺呼叫端。
# 不動：感測器(velodyne/camera/imu)、底盤 driver、rosbridge、zenoh router 等基礎服務。
echo "[deploy_full_stop] 1) 優雅停止 rover_rl 棧..."
bash /home/aa/rover_rl_stop.sh 2>/dev/null | tail -2

echo "[deploy_full_stop] 2) 強制清掉殘留 (NDT / policy / LV-DOT / routing / costmap / MOT / bev / 殭屍)..."
for pat in \
  "deploy_full.launch" "rover_rl_bringup" "run_detector.launch" \
  "ndt_localizer_launch" "points_downsample_launch" "map_loader_launch" "tf_static_launch" \
  "rover_rl_inference/lib/rover_rl_inference/policy_node" \
  "rover_rl_inference/lib/rover_rl_inference/lidar_preprocessor" \
  "rover_rl_inference/lib/rover_rl_inference/bev_play" \
  "rover_rl_inference/lib/rover_rl_inference/diag_logger" \
  "rover_rl_inference/lib/rover_rl_inference/routing_to_path" \
  "rover_rl_inference/lib/rover_rl_inference/routing_click_bridge" \
  "rover_rl_inference/lib/rover_rl_inference/status_tui" \
  "onboard_detector/lib/onboard_detector/detector_node" \
  "yolov11_detector_node" "yolo_venv/bin/python" \
  "ndt_localizer_node" "ndt_localizer/lib" "voxel_grid_filter" "map_loader" \
  "campusrover_routing/lib" "campusrover_costmap_ros2/lib" "campusrover_mot/lib" \
  "campusrover_demo/lib/campusrover_demo/simple_map_publisher" \
  "static_transform_publisher.*world.*map" \
  "rviz2.*demo.rviz" ; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 3

echo "[deploy_full_stop] 3) 檢查殘留..."
n=$(ps -eo cmd --no-headers | grep -E "deploy_full.launch|detector_node|policy_node|ndt_localizer|voxel_grid_filter|world_to_map|yolov11|status_tui|lidar_preprocessor|routing_engine|routing_to_path|routing_click|local_costmap|global_costmap|mot_node|mot_marker|bev_play|map_loader|simple_map_publisher|yolo_venv" | grep -v grep | wc -l)
if [ "$n" -eq 0 ]; then
  echo "[deploy_full_stop] ✅ 全部已停、清乾淨 (殘留 0)"
else
  echo "[deploy_full_stop] ⚠ 仍有 $n 個殘留:"
  ps -eo pid,cmd --no-headers | grep -E "deploy_full.launch|detector_node|policy_node|ndt_localizer|voxel_grid_filter|world_to_map|yolov11|status_tui|lidar_preprocessor|routing|costmap|mot_node|bev_play|map_loader|simple_map_publisher|yolo_venv" | grep -v grep | head
fi
