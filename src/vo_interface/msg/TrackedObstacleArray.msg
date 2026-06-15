# LV-DOT 動態障礙物經 vo_interface 重追蹤後的輸出清單，供下游 Velocity Obstacle 規劃器訂閱。
std_msgs/Header header             # stamp + frame_id（部署為 odom）
TrackedObstacle[] obstacles
