"""
test.launch.py — jie_deamon 測試啟動檔（不含底盤驅動）

檔案說明:
  啟動 robot_nexus 節點與雷達驅動，用於不連接底盤的桌面測試。
  與 start.launch.py 的差異:
    - 無 with_lidar 條件判斷，雷達驅動一律啟動
    - 適合在沒有實體底盤的情況下測試感知與 Web 可視化

Humble 相容性:
  - launch / launch_ros API 與 Jazzy 基本相容
  - ament_index_python 路徑在 Humble / Jazzy 中一致
  - 注意: 若 lidar_pkg 不存在，get_package_share_directory 會在啟動時報錯
"""

import os  # 檔案路徑操作
from launch import LaunchDescription  # ROS 2 啟動描述容器
from launch.actions import DeclareLaunchArgument  # 宣告可由命令列傳入的啟動參數
from launch.substitutions import LaunchConfiguration  # 在執行期取得啟動參數值
from launch_ros.actions import Node  # 啟動 ROS 2 節點
from launch.actions import IncludeLaunchDescription  # 包含其他 launch 檔案
from ament_index_python.packages import get_package_share_directory  # 取得套件 share 目錄路徑
from launch.launch_description_sources import PythonLaunchDescriptionSource  # 指定 Python launch 來源


def generate_launch_description():
    """產生啟動描述 — ROS 2 launch 系統的進入點"""

    # 取得本套件（jie_deamon）的 share 目錄，用於定位 web 靜態檔案
    pkg_share = get_package_share_directory('jie_deamon')

    # === 宣告啟動參數 ===

    # active: 是否啟用跟隨功能（預設開啟）
    active_arg = DeclareLaunchArgument(
        'active',
        default_value='true',
        description='是否激活跟隨功能'
    )

    # enable_opencv: 是否開啟 OpenCV 視窗顯示（僅除錯用）
    enable_opencv_arg = DeclareLaunchArgument(
        'enable_opencv',
        default_value='false',
        description='是否啟用OpenCV可視化(調試用)'
    )

    # enable_web: 是否啟用 Web 即時可視化介面（預設開啟）
    enable_web_arg = DeclareLaunchArgument(
        'enable_web',
        default_value='true',
        description='是否啟用Web可視化'
    )

    # === 建立 robot_nexus 節點 ===
    # robot_nexus 是主要後端服務節點，負責跟隨邏輯、Web 通訊等
    robot_nexus_node = Node(
        package='jie_deamon',          # 所屬套件名稱
        executable='robot_nexus',       # CMakeLists.txt 中定義的可執行檔名
        name='robot_nexus',             # 節點名稱
        output='screen',                # 將 log 輸出到終端機
        parameters=[{
            'active': LaunchConfiguration('active'),                # 跟隨功能開關
            'enable_opencv': LaunchConfiguration('enable_opencv'),  # OpenCV 顯示開關
            'enable_web': LaunchConfiguration('enable_web'),        # Web 可視化開關
            'web_root': os.path.join(pkg_share, 'web'),             # Web 靜態檔案根目錄
        }]
    )

    # === 雷達驅動（測試模式下一律啟動） ===
    # 包含 lidar_pkg 的 launch 檔案來啟動雷達驅動節點
    # 注意: 須確保 lidar_pkg 已安裝，否則此處會報錯
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("lidar_pkg"), 'launch', 'lidar.launch.py')
        )
    )

    # === 組合所有啟動項目並回傳 ===
    return LaunchDescription([
        active_arg,          # 宣告 active 參數
        enable_opencv_arg,   # 宣告 enable_opencv 參數
        enable_web_arg,      # 宣告 enable_web 參數
        robot_nexus_node,    # 啟動 robot_nexus 節點
        lidar_launch,        # 啟動雷達驅動
    ])
