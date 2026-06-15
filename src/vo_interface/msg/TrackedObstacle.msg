# 單一被追蹤的動態障礙物，介面供 Velocity Obstacle (VO) / 避障規劃使用。
# 座標系見 TrackedObstacleArray.header.frame_id（部署為 odom = LV-DOT 的 world frame，速度為絕對速度）。
uint32 id                       # 持久追蹤 ID（跨幀穩定；LV-DOT 原生 box.id 是每幀索引，這裡自行配對重編）
float64 age                     # 已穩定追蹤時間（秒）。VO 可據此忽略過年輕、速度尚未收斂的 track
geometry_msgs/Point position    # 障礙物中心位置 p_B
geometry_msgs/Vector3 velocity  # 絕對速度向量 v_B（CV-Kalman 平滑後，odom frame）
geometry_msgs/Vector3 size      # 邊界框尺寸 (x, y, z)
float64 radius                  # 等效邊界圓半徑 r_B = 0.5 * hypot(size.x, size.y)，VO 取 Minkowski 和用
float64 vel_confidence          # 速度可信度 0~1（track 越久、協方差越小→越高）
float64[4] position_covariance  # 位置 2x2 協方差 (row-major: xx, xy, yx, yy)，供機率式 VO (PVO)
float64[4] velocity_covariance  # 速度 2x2 協方差 (row-major)，供機率式 VO (PVO)
