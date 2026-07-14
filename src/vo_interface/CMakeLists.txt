cmake_minimum_required(VERSION 3.8)
project(vo_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(onboard_detector REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 產生 VO 介面訊息
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedObstacle.msg"
  "msg/TrackedObstacleArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# 安裝 python 節點（rclpy）
install(PROGRAMS
  scripts/vo_interface_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安裝 launch / config
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime onboard_detector)
ament_package()
