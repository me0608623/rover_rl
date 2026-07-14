# header.stamp 是產生這批障礙的 LiDAR 量測時間，非發布時間。
std_msgs/Header header
bool source_valid                 # false = LiDAR/odom 已逾時；下游必須立即清空/coast stop
DynamicObstacle[] obstacles
