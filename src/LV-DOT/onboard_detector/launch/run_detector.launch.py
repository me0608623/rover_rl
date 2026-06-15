#!/usr/bin/env python3
"""LV-DOT dynamic obstacle detector launch (ROS2 Humble)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown, OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 啟動前先清掉殘留的 detector / yolo / vo 行程，避免「上次沒收乾淨 → 這次雙開」（YOLO 雙開吃雙倍 GPU）。
# pkill pattern 用 [n] bracket trick：避免比中本指令字串自身。detector + yolo + vo 都清。
STALE_CLEANUP_CMD = (
    "pkill -9 -f 'yolov11_detector_[n]ode' 2>/dev/null; "
    "pkill -9 -f 'onboard_detector/lib/onboard_detector/[d]etector_node' 2>/dev/null; "
    "pkill -9 -f 'vo_interface/vo_interface_[n]ode' 2>/dev/null; "
    "sleep 1; echo '[lvdot] 已清除殘留 detector/yolo/vo 行程（防雙開）'; true"
)

# RealSense 接在主機 192.168.3.13、帳號 humble（本機 Jetson 無相機裝置），use_camera 走 SSH 遠端啟動。
# 免密金鑰已建（aa@jetson → humble@.13）。遠端 ~/start_realsense.sh 已含
# enable_depth:=true + depth 640,480,30 + color 640,480,15 + zenoh RMW，直接複用。
# 冪等啟動：已在跑→沿用不動（避免雙開搶 USB re-enumerate 卡死）；
# 沒在跑→setsid+nohup 背景拉起後 ssh 立刻收掉。
# pattern 用 [n] bracket trick：pgrep -f 否則會比中 ssh 遠端 shell 自己的指令字串（誤判已在跑）
# ⚠ 2026-06-10 行為變更：改為「關 lv-dot 連帶殺相機」（見下方 CAMERA_KILL_CMD + OnShutdown）。
CAMERA_REMOTE_CMD = (
    "if pgrep -f 'realsense2_camera_[n]ode' >/dev/null; then "
    "echo '[camera] RealSense 已在跑，沿用既有行程'; "
    "else "
    "setsid nohup ~/start_realsense.sh > ~/realsense_lvdot.log 2>&1 < /dev/null & "
    "sleep 2 && echo '[camera] RealSense 已背景啟動（log: ~/realsense_lvdot.log）'; "
    "fi"
)

# lv-dot 關閉時連帶殺掉遠端 RealSense（pkill 同 [n] bracket trick，避免比中 ssh 自身指令字串）。
CAMERA_KILL_CMD = (
    "pkill -f 'realsense2_camera_[n]ode' && "
    "echo '[camera] 已隨 lv-dot 關閉 RealSense' || "
    "echo '[camera] 無 RealSense 行程可關'"
)


def generate_launch_description():
    pkg_share = get_package_share_directory('onboard_detector')
    default_params = os.path.join(pkg_share, 'cfg', 'detector_param.yaml')
    # 注意：包內 detector_lv.rviz 是 ROS1 格式，rviz2 載入後 display 全失效；改用本機 rviz2 設定
    default_rviz = '/home/aa/rviz/lvdot.rviz'

    params_file = LaunchConfiguration('params_file')
    use_yolo = LaunchConfiguration('use_yolo')
    use_rviz = LaunchConfiguration('use_rviz')
    use_camera = LaunchConfiguration('use_camera')
    enable_vo = LaunchConfiguration('enable_vo')
    # 用哪個 python 跑 YOLO 節點。預設用裝了 CUDA torch 的 venv → YOLO 吃 GPU。
    # 設成 '' 則用系統 python3（CPU torch，會很慢）。
    yolo_python = LaunchConfiguration('yolo_python')

    # detector + yolo 在 stale_cleanup 退出後才啟動（保證乾淨起跑、防雙開）
    detector_node = Node(
        package='onboard_detector',
        executable='detector_node',
        name='dynamic_detector',
        output='screen',
        parameters=[params_file],
    )
    yolo_node = Node(
        package='onboard_detector',
        executable='yolov11_detector_node.py',
        name='yolov11_detector_node',
        output='screen',
        prefix=[yolo_python],   # 用 CUDA-torch venv 的 python 跑 → GPU
        condition=IfCondition(use_yolo),
    )

    stale_cleanup = ExecuteProcess(
        cmd=['bash', '-c', STALE_CLEANUP_CMD],
        name='lvdot_stale_cleanup',
        output='screen',
    )

    # vo_interface（LV-DOT → Velocity Obstacle 介面）隨 lv-dot 一起啟動（enable_vo 預設 true）。
    # 程式仍解耦（純訂閱 dynamic_bboxes），只是 launch 層一起拉起方便用。vo_interface 未 build 時跳過。
    vo_actions = []
    try:
        vo_share = get_package_share_directory('vo_interface')
        vo_params = os.path.join(vo_share, 'config', 'vo_interface_params.yaml')
        vo_actions.append(Node(
            package='vo_interface',
            executable='vo_interface_node.py',
            name='vo_interface',
            output='screen',
            parameters=[vo_params],
            condition=IfCondition(enable_vo),
        ))
    except Exception:
        pass  # vo_interface 尚未 build → lv-dot 照常啟動，只是不帶 vo

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Detector parameter yaml'),
        DeclareLaunchArgument('use_yolo', default_value='true',
                              description='Launch the YOLOv11 color detector'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='Launch RViz'),
        DeclareLaunchArgument('yolo_python',
                              default_value=os.path.expanduser('~/yolo_venv/bin/python'),
                              description='Python interpreter for the YOLO node (CUDA-torch venv)'),
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='SSH 啟動相機主機上的 RealSense（含 depth）'),
        DeclareLaunchArgument('camera_ssh', default_value='humble@192.168.3.13',
                              description='RealSense 所在主機的 ssh 目標'),
        DeclareLaunchArgument('enable_vo', default_value='true',
                              description='隨 lv-dot 一起啟動 vo_interface（VO 介面節點）'),

        ExecuteProcess(
            cmd=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                 LaunchConfiguration('camera_ssh'), CAMERA_REMOTE_CMD],
            name='realsense_remote',
            output='screen',
            condition=IfCondition(use_camera),
        ),

        # 先跑清理，清完（process exit）再起 detector + yolo + vo → 防雙開
        # （vo 必須也排在清理後，否則 top-level 與清理同時跑會被 pkill vo 殺掉剛起的新 vo）
        stale_cleanup,
        RegisterEventHandler(
            OnProcessExit(
                target_action=stale_cleanup,
                on_exit=[detector_node, yolo_node, *vo_actions],
            ),
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', default_rviz],
            condition=IfCondition(use_rviz),
        ),

        # lv-dot 收棧（Ctrl+C / 子節點全退）時，SSH 過去把遠端 RealSense 一起殺掉。
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    ExecuteProcess(
                        cmd=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                             LaunchConfiguration('camera_ssh'), CAMERA_KILL_CMD],
                        name='realsense_remote_kill',
                        output='screen',
                        condition=IfCondition(use_camera),
                    ),
                ],
            ),
        ),
    ])
