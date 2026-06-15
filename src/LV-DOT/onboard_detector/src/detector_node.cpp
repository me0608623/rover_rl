/*
	FILE: detector_node.cpp
	--------------------------
	Run detector node (ROS2 Humble)
*/
#include <rclcpp/rclcpp.hpp>
#include <onboard_detector/dynamicDetector.h>

int main(int argc, char** argv){
	rclcpp::init(argc, argv);
	auto node = std::make_shared<onboardDetector::dynamicDetector>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
