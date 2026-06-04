"""rover_rl 部署 launch — policy 推論節點.

這是最精簡的 launch：只啟動 policy_node 一個節點，
不含 preprocessor / BEV / NDT / routing。
適用情境：preprocessor 已由外部啟動，或用 inline 前處理時只想單獨跑推論。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 取得本 package 安裝後的 share 路徑，組出預設參數檔位置
    pkg_share = get_package_share_directory("rover_rl_bringup")
    default_params = os.path.join(pkg_share, "config", "policy_params.yaml")

    # LaunchConfiguration 是「延遲取值」的代理物件，實際值在啟動時才解析
    params_file = LaunchConfiguration("params_file")   # policy 參數 yaml 路徑
    model_path = LaunchConfiguration("model_path")     # 覆寫 yaml 內的 model_path
    log_level = LaunchConfiguration("log_level")       # ROS log 等級

    return LaunchDescription(
        [
            # 宣告可由命令列覆寫的啟動參數（含預設值）
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Override params file model_path; "
                            "leave empty to use yaml value",
            ),
            DeclareLaunchArgument("log_level", default_value="info"),
            # RL 推論節點：訂閱 sweep/odom/goal → 發 cmd_vel
            Node(
                package="rover_rl_inference",
                executable="policy_node",
                name="rover_rl_policy",
                output="screen",
                # 參數套用順序：先載 yaml，再用 dict 覆寫 model_path
                # （注意：此檔即使 model_path 為空字串也會覆寫成空，
                #  與 deploy_with_bev/deploy_full 用 OpaqueFunction 的寫法不同）
                parameters=[
                    params_file,
                    {"model_path": model_path},
                ],
                arguments=["--ros-args", "--log-level", log_level],
            ),
        ]
    )
