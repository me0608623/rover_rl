"""Launch ORCA bystander 測試節點。

不控制車,只訂 LV-DOT + odom → 跑 ORCA → 印 + RViz。
前提:vo_interface 已啟動(發 /vo_interface/tracked_obstacles)、odom 有發布。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('orca_filter'), 'config', 'orca_params.yaml')
    return LaunchDescription([
        Node(
            package='orca_filter',
            executable='orca_bystander',
            name='orca_bystander',
            parameters=[params],
            output='screen',
        ),
    ])
