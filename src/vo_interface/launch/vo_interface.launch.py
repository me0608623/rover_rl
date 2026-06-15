#!/usr/bin/env python3
"""vo_interface 獨立啟動。LV-DOT（lv-dot alias）需先在跑，本節點純訂閱其 dynamic_bboxes。"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("vo_interface")
    default_params = os.path.join(pkg_share, "config", "vo_interface_params.yaml")

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="vo_interface 參數 yaml"),
        Node(
            package="vo_interface",
            executable="vo_interface_node.py",
            name="vo_interface",
            output="screen",
            parameters=[params_file],
        ),
    ])
