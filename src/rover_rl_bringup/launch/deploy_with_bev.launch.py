"""rover_rl + BEV 合併 launch（NDT 由用戶獨立啟動）.

TF 鏈依賴：
    world ── (static, 由 NDT 的 tf_static_launch) ── map
                                                       │
    map ── (動態, 由 ndt_localizer_node 發布) ── odom
                                                  │
    odom ── (由 campusrover_base driver 發布) ── base_link
                                                  │
    base_link ── (URDF static) ── base_footprint / velodyne_link / imu_link

啟動順序建議（不同 terminal）：
    1. 用戶手動：cd ~/Documents/ndt_ws && source install/setup.bash
                ros2 launch ndt_localizer ndt_localizer_launch.py
    2. 用戶手動：底盤 driver（提供 /odom + odom→base_link TF）
    3. 用戶手動：velodyne driver（提供 /velodyne_points）
    4. 本 launch：ros2 launch rover_rl_bringup deploy_with_bev.launch.py

Args（可選）：
    params_file    覆寫 policy_params.yaml 路徑
    model_path     覆寫 model_path（空字串 = 用 yaml 預設）
    enable_bev     true/false（預設 true）
    bev_show       rviz / image_view / window / none（預設 image_view）
    log_level      debug / info / warn / error（預設 info）
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("rover_rl_bringup")
    default_params = os.path.join(pkg_share, "config", "policy_params.yaml")

    params_file = LaunchConfiguration("params_file")
    model_path = LaunchConfiguration("model_path")
    enable_bev = LaunchConfiguration("enable_bev")
    bev_show = LaunchConfiguration("bev_show")
    log_level = LaunchConfiguration("log_level")

    # ── 1. Policy node ────────────────────────────────────────────────────
    policy_node = Node(
        package="rover_rl_inference",
        executable="policy_node",
        name="rover_rl_policy",
        output="screen",
        emulate_tty=True,
        parameters=[
            params_file,
            {"model_path": model_path},
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    # ── 2. BEV node（讀 rover2_ws 的 bev.launch.py） ────────────────────────
    # 若 rover2_ws 不在 workspace path，可改用直接 Node() 方式
    try:
        bev_pkg_share = get_package_share_directory("campusrover_rl_policy")
        bev_launch_path = os.path.join(bev_pkg_share, "launch", "bev.launch.py")
        bev_action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bev_launch_path),
            launch_arguments={
                "pointcloud_topic": "/velodyne_points",
                "use_rviz": PythonExpression(["'", bev_show, "' == 'rviz'"]),
                "use_image_view": PythonExpression(["'", bev_show, "' == 'image_view'"]),
                "show_window": PythonExpression(["'", bev_show, "' == 'window'"]),
            }.items(),
        )
        bev_group = GroupAction(
            [bev_action],
            condition=IfCondition(enable_bev),
        )
        bev_unavailable_msg = []
    except Exception:
        bev_group = LogInfo(
            msg="[rover_rl] campusrover_rl_policy 未安裝，BEV 無法啟動（請 source rover2_ws）"
        )
        bev_unavailable_msg = []

    # ── 3. Banner ─────────────────────────────────────────────────────────
    banner = LogInfo(msg=(
        "================================\n"
        "rover_rl deploy_with_bev 啟動\n"
        "  - policy_node: 訂閱 /velodyne_points + /odom + /goal_pose\n"
        "  - bev_node:    可視化 /bev_polar_image (由 rover2_ws 提供)\n"
        "  - 假設：NDT、底盤 driver、velodyne driver 已由其他 launch 啟動\n"
        "================================"
    ))

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument(
            "model_path", default_value="",
            description="覆寫 yaml model_path；空字串=用 yaml 預設",
        ),
        DeclareLaunchArgument("enable_bev", default_value="true"),
        DeclareLaunchArgument(
            "bev_show", default_value="image_view",
            description="rviz / image_view / window / none",
        ),
        DeclareLaunchArgument("log_level", default_value="info"),
        banner,
        policy_node,
        bev_group,
    ])
