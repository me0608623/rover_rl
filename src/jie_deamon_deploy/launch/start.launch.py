"""
start.launch.py — jie_deamon 主啟動檔（campusrover 差速機器人版本）

啟動方式:
  # 僅啟動 robot_nexus（底盤和感測器由外部腳本啟動）
  ros2 launch jie_deamon start.launch.py

  # 同時啟動底盤驅動（適合開發測試，實機建議用 cr_start_drive 腳本）
  ros2 launch jie_deamon start.launch.py with_driver:=true

前置條件:
  需要有 /scan_front topic（YDLidar 前光達）
  實機環境由 cr_sensor 腳本啟動感測器

Topic remap:
  /scan     → /scan_front          你的前光達 topic
  /cmd_vel  → /input/nav_cmd_vel   經由 lcr_cmd_vel_mux 仲裁,搖桿可搶斷

完整的實機啟動流程:
  1. 執行 cr_start_drive   → 底盤 + 搖桿 + mux
  2. 執行 cr_sensor        → Velodyne + IMU + YDLidar + TF
  3. ros2 launch jie_deamon start.launch.py  → 追蹤 + Web 介面
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory('jie_deamon')

    # ======================== 啟動參數 ========================

    active_arg = DeclareLaunchArgument(
        'active', default_value='true',
        description='是否啟用雷達跟隨功能'
    )
    enable_opencv_arg = DeclareLaunchArgument(
        'enable_opencv', default_value='false',
        description='是否啟用 OpenCV 視覺化（除錯用,吃 CPU）'
    )
    enable_web_arg = DeclareLaunchArgument(
        'enable_web', default_value='true',
        description='是否啟用 Web 可視化介面（http://<IP>:8080）'
    )
    with_driver_arg = DeclareLaunchArgument(
        'with_driver', default_value='false',
        description='是否同時啟動底盤驅動（實機建議用 cr_start_drive 腳本）'
    )

    # ======================== 可選：底盤驅動 ========================
    # campusrover_driver: rover_driver + joy_to_twist + lcr_cmd_vel_mux
    # rover_driver 訂閱 /output/cmd_vel（mux 輸出）
    # 實機通常由 cr_start_drive 腳本啟動,這裡提供 with_driver:=true 做開發測試用
    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('campusrover_driver'),
                'launch', 'campusrover_driver_launch.py'
            )
        ),
        condition=IfCondition(LaunchConfiguration('with_driver'))
    )

    # ======================== jie_deamon 中樞節點 ========================
    # 需要外部提供: /scan_front (LaserScan)
    # 輸出: /input/nav_cmd_vel (Twist) → 送進 mux
    robot_nexus_node = Node(
        package='jie_deamon',
        executable='robot_nexus',
        name='robot_nexus',
        output='screen',
        parameters=[{
            'active': LaunchConfiguration('active'),
            'enable_opencv': LaunchConfiguration('enable_opencv'),
            'enable_web': LaunchConfiguration('enable_web'),
            'web_root': os.path.join(pkg_share, 'web'),
        }],
        remappings=[
            ('/scan', '/scan_front'),           # 前光達 topic
            ('/cmd_vel', '/input/nav_cmd_vel'), # 送進 mux,搖桿可搶斷
        ]
    )

    return LaunchDescription([
        active_arg,
        enable_opencv_arg,
        enable_web_arg,
        with_driver_arg,
        driver_launch,       # 可選：底盤驅動
        robot_nexus_node,    # 追蹤節點（需要 /scan_front 已存在）
    ])
