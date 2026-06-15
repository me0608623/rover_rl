cmake_minimum_required(VERSION 3.8)
project(onboard_detector)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetDynamicObstacles.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

set(dependencies
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  vision_msgs
  cv_bridge
  image_transport
  message_filters
  pcl_conversions
  pcl_ros
  tf2
  tf2_geometry_msgs
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# core library
add_library(${PROJECT_NAME}_lib
  include/${PROJECT_NAME}/dynamicDetector.cpp
  include/${PROJECT_NAME}/uvDetector.cpp
  include/${PROJECT_NAME}/dbscan.cpp
  include/${PROJECT_NAME}/kalmanFilter.cpp
  include/${PROJECT_NAME}/lidarDetector.cpp
)
ament_target_dependencies(${PROJECT_NAME}_lib ${dependencies})
target_link_libraries(${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBS})

# link generated interfaces to the library
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_lib "${cpp_typesupport_target}")

# detector node executable
add_executable(detector_node src/detector_node.cpp)
ament_target_dependencies(detector_node ${dependencies})
target_link_libraries(detector_node ${PROJECT_NAME}_lib)

# install C++ targets
install(TARGETS
  ${PROJECT_NAME}_lib
  detector_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# install headers
install(DIRECTORY include/
  DESTINATION include
)

# install python yolo detector scripts (flattened so node finds its siblings + weights)
install(DIRECTORY scripts/yolo_detector/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# install launch and config
install(DIRECTORY launch cfg rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime ${dependencies})
ament_package()
