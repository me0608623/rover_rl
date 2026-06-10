"""
d1_bringup.launch.py — D1 四足機械狗底盤啟動檔（已棄用）

檔案說明:
  原本用於啟動 D1 四足機械狗的底盤核心節點 (d1_core)。
  專案已改為差速驅動機器人，此檔案保留僅作參考。

  *** 已棄用 — 請勿在正式環境使用 ***

Humble 相容性:
  - launch / launch_ros API 與 Jazzy 基本相容
  - ament_index_python 路徑在 Humble / Jazzy 中一致
  - 重要警告: 此檔案引用的 d1_bringup 套件在差速機器人環境中不存在，
    啟動時 get_package_share_directory('d1_bringup') 會拋出
    PackageNotFoundError，導致整個 launch 失敗
"""

import os  # 檔案路徑操作
from ament_index_python.packages import get_package_share_directory  # 取得套件 share 目錄路徑
from launch import LaunchDescription  # ROS 2 啟動描述容器
from launch_ros.actions import Node  # 啟動 ROS 2 節點


def generate_launch_description():
    """產生啟動描述 — 啟動 D1 底盤核心節點（已棄用）"""

    pkg_name = 'd1_bringup'  # D1 機械狗底盤套件名稱（需另外安裝）

    # 取得 D1 參數設定檔路徑（d1_params.yaml）
    # 注意: 若 d1_bringup 套件不存在，此行會拋出 PackageNotFoundError
    config_file = os.path.join(
        get_package_share_directory(pkg_name),  # d1_bringup 的 share 目錄
        'config',                                # config 子目錄
        'd1_params.yaml'                         # D1 底盤參數設定檔
    )

    # 回傳啟動描述，包含 D1 底盤核心節點
    return LaunchDescription([
        Node(
            package=pkg_name,           # 所屬套件: d1_bringup
            executable='d1_core',       # 可執行檔: D1 底盤核心程式
            name='d1_core_node',        # 節點名稱
            # output='screen',          # 已註解 — 取消註解可在終端機看到 log
            parameters=[config_file]    # 載入 d1_params.yaml 參數設定檔
        )
    ])
