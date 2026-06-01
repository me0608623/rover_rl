"""rover_rl 部署 launch — policy 推論節點."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("rover_rl_bringup")
    default_params = os.path.join(pkg_share, "config", "policy_params.yaml")

    params_file = LaunchConfiguration("params_file")
    model_path = LaunchConfiguration("model_path")
    log_level = LaunchConfiguration("log_level")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Override params file model_path; "
                            "leave empty to use yaml value",
            ),
            DeclareLaunchArgument("log_level", default_value="info"),
            Node(
                package="rover_rl_inference",
                executable="policy_node",
                name="rover_rl_policy",
                output="screen",
                parameters=[
                    params_file,
                    {"model_path": model_path},
                ],
                arguments=["--ros-args", "--log-level", log_level],
            ),
        ]
    )
