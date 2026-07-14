# 正式的 LV-DOT 動態障礙介面。座標系與量測時間見 DynamicObstacleArray.header。
uint32 id
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 acceleration
geometry_msgs/Vector3 size
string classification          # person / unknown
string source                  # yolo_lidar / lidar
float64 confidence             # 0..1，目前為偵測/動態分類可信度
bool is_moving                 # 幾何速度是否超過 dynamic_velocity_threshold
float64[4] position_covariance # 2x2 row-major: xx,xy,yx,yy
float64[4] velocity_covariance # 2x2 row-major: xx,xy,yx,yy
