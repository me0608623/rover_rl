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
    GroupAction,           # 把多個 action 包成一組（此處用來包 BEV 節點）
    IncludeLaunchDescription,
    LogInfo,               # 啟動時印 banner 訊息
    OpaqueFunction,        # 延遲執行函式，可在啟動時讀取參數實際值再決定節點設定
)
from launch.conditions import IfCondition   # 依 bool 參數決定節點是否啟動
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 預設參數檔路徑（policy 與 lidar preprocessor 各一份 yaml）
    pkg_share = get_package_share_directory("rover_rl_bringup")
    default_params = os.path.join(pkg_share, "config", "policy_params.yaml")
    default_pre_params = os.path.join(pkg_share, "config",
                                       "lidar_preprocessor_params.yaml")

    # 啟動參數代理（啟動時才解析）
    params_file = LaunchConfiguration("params_file")                      # policy yaml
    preprocessor_params_file = LaunchConfiguration("preprocessor_params_file")  # 前處理 yaml
    model_path = LaunchConfiguration("model_path")                        # 覆寫模型路徑
    enable_bev = LaunchConfiguration("enable_bev")                        # 是否開 BEV
    enable_preprocessor = LaunchConfiguration("enable_preprocessor")      # 是否開前處理
    log_level = LaunchConfiguration("log_level")

    # ── 0. LiDAR preprocessor node（先處理再給 RL） ────────────────────────
    preprocessor_node = Node(
        package="rover_rl_inference",
        executable="lidar_preprocessor",
        name="rover_rl_lidar_preprocessor",
        output="screen",
        emulate_tty=True,
        parameters=[preprocessor_params_file],
        arguments=["--ros-args", "--log-level", log_level],
        condition=IfCondition(enable_preprocessor),
    )

    # ── 1. Policy node（用 OpaqueFunction 避免空字串覆蓋 yaml model_path）────
    # OpaqueFunction 會在啟動階段被呼叫，此時才能用 .perform() 取得參數真值。
    # 關鍵差異：只有當 model_path 非空時才加入覆寫 dict；空字串就完全不覆寫，
    # 讓 yaml 內的 model_path 生效（避免被空字串蓋掉）。
    def make_policy_node(context, *args, **kwargs):
        mp = LaunchConfiguration("model_path").perform(context)
        extra = [{"model_path": mp}] if mp else []   # 空字串 → 不覆寫
        return [Node(
            package="rover_rl_inference",
            executable="policy_node",
            name="rover_rl_policy",
            output="screen",
            emulate_tty=True,
            parameters=[params_file] + extra,
            arguments=["--ros-args", "--log-level",
                       LaunchConfiguration("log_level").perform(context)],
        )]
    policy_node = OpaqueFunction(function=make_policy_node)

    # ── 2. BEV node（rover_rl 自包，移植自訓練端 play_eval/bev_renderer.py） ──
    bev_play_node = Node(
        package="rover_rl_inference",
        executable="bev_play",
        name="rover_rl_bev_play",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"frame_mode": "body",
             "rate_hz": 5.0,
             "r_max": 20.0,
             "r_robot": 0.35,
             "topic_obs_debug": "/rover_rl_policy/obs_debug"},
        ],
        arguments=["--ros-args", "--log-level", log_level],
        condition=IfCondition(enable_bev),
    )
    # 用 GroupAction 包起來（目前僅一個節點，保留分組以利日後擴充/命名空間）
    bev_group = GroupAction([bev_play_node])

    # ── 3. Banner ─────────────────────────────────────────────────────────
    banner = LogInfo(msg=(
        "================================\n"
        "rover_rl deploy_with_bev 啟動\n"
        "  - policy_node: 訂閱 /velodyne_points + /odom + /goal_pose\n"
        "  - bev_node:    可視化 /bev_polar_image (由 rover2_ws 提供)\n"
        "  - 假設：NDT、底盤 driver、velodyne driver 已由其他 launch 啟動\n"
        "================================"
    ))

    # 組裝啟動描述：先宣告所有可覆寫參數，再依序加入節點
    # （preprocessor → policy → bev，符合資料流先後順序）
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument(
            "preprocessor_params_file", default_value=default_pre_params,
        ),
        DeclareLaunchArgument(
            "model_path", default_value="",
            description="覆寫 yaml model_path；空字串=用 yaml 預設",
        ),
        DeclareLaunchArgument(
            "enable_bev", default_value="true",
            description="啟動 rover_rl 自包的 play-style BEV node "
                        "（matplotlib 渲染 → /rover_rl/bev_image）",
        ),
        DeclareLaunchArgument(
            "enable_preprocessor", default_value="true",
            description="是否啟動 rover_rl 自己的 lidar_preprocessor；"
                        "若已有外部 preprocessor 可設 false",
        ),
        DeclareLaunchArgument("log_level", default_value="info"),
        banner,
        preprocessor_node,
        policy_node,
        bev_group,
    ])
