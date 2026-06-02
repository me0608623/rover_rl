"""rover_rl 完整部署 — campusrover 全棧 (排除 DWA + AIT*) + RL policy + BEV.

取代 campusrover_demo_launch.py：
  - 移除 DWA（RL policy 取代）
  - 移除 AIT*（改用 campusrover_routing service-based 全局路徑）
  - 加入 routing_to_path 橋接（routing service → /global_path topic）

啟動節點：
  campusrover 棧:
    [0] Map Server         — /map
    [1] NDT 定位           — /velodyne_points → /ndt_pose + map→odom TF
    [2] Routing Engine     — generation_path service（拓撲路徑）
    [3] routing_to_path    — 橋接：routing service → /global_path topic
    [4] Costmap            — local_costmap (RViz debug 用)
    [5] MOT                — 動態障礙物追蹤
    [6] RViz               — 可視化

  rover_rl 棧:
    [7] lidar_preprocessor — /velodyne_points → /rover_rl/lidar_sweep_72
    [8] policy_node        — sweep + odom + goal → /input/nav_cmd_vel
    [9] bev_play           — 即時 BEV 圖 → /rover_rl/bev_image

使用方式：
  Terminal 1: 底盤 driver（提供 /odom + odom→base_link TF）
  Terminal 2: VLP-16 driver（提供 /velodyne_points）
  Terminal 3:
    source ~/rover_rl/install/setup.bash
    source ~/rover2_ws/install/setup.bash
    source ~/rover_rl/setup_env.sh
    ros2 launch rover_rl_bringup deploy_full.launch.py

  規劃路徑：
    ros2 service call /rover_rl/routing_call campusrover_msgs/srv/RoutingPath \
      "{origin: 'c1', destination: ['e0']}"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── package paths ──
    rl_pkg = get_package_share_directory("rover_rl_bringup")
    ndt_pkg = get_package_share_directory("ndt_localizer")
    costmap_pkg = get_package_share_directory("campusrover_costmap_ros2")
    routing_pkg = get_package_share_directory("campusrover_routing")

    default_params = os.path.join(rl_pkg, "config", "policy_params.yaml")
    default_pre_params = os.path.join(rl_pkg, "config",
                                       "lidar_preprocessor_params.yaml")

    # ── args ──
    model_path = LaunchConfiguration("model_path")
    initial_mode = LaunchConfiguration("initial_mode")
    params_file = LaunchConfiguration("params_file")
    pre_params_file = LaunchConfiguration("preprocessor_params_file")
    enable_bev = LaunchConfiguration("enable_bev")
    enable_preprocessor = LaunchConfiguration("enable_preprocessor")
    enable_mot = LaunchConfiguration("enable_mot")
    enable_costmap = LaunchConfiguration("enable_costmap")
    rviz = LaunchConfiguration("rviz")
    log_level = LaunchConfiguration("log_level")
    map_file = LaunchConfiguration("map_file")

    # ── Part 0: Map Server ──
    map_server_node = Node(
        package="campusrover_demo",
        executable="simple_map_publisher",
        name="map_server",
        output="log",
        parameters=[{"map_file": map_file}],
    )

    # ── Part 1: RViz ──
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_demo",
        arguments=["-d", "/home/aa/rviz/demo.rviz"],
        output="log",
        condition=IfCondition(rviz),
    )

    # ── Part 2: NDT Localization ──
    ndt_localizer_node = Node(
        package="ndt_localizer",
        executable="ndt_localizer_node",
        name="ndt_localizer_node",
        output="log",
        arguments=["--ros-args", "--log-level", "ndt_localizer_node:=warn"],
        parameters=[{
            "resolution": LaunchConfiguration("ndt_resolution"),
            "step_size": LaunchConfiguration("step_size"),
            "trans_epsilon": LaunchConfiguration("trans_epsilon"),
            "max_iterations": LaunchConfiguration("max_iterations"),
            "converged_param_transform_probability":
                LaunchConfiguration("converged_param"),
            "debug": False,
            "base_frame": "base_link",
            "odom_frame": "odom",
            "map_frame": "map",
        }],
        remappings=[
            ("ndt_pose", "/ndt_pose"),
            ("diagnostics", "/diagnostics"),
        ],
    )

    tf_static_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ndt_pkg, "launch", "tf_static_launch.py")),
    )
    points_downsample_launch = TimerAction(
        period=2.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ndt_pkg, "launch", "points_downsample_launch.py")),
        )],
    )
    map_loader_launch = TimerAction(
        period=3.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ndt_pkg, "launch", "map_loader_launch.py")),
            launch_arguments={"x": "0.0", "y": "0.0", "z": "0.0",
                              "roll": "0.0", "pitch": "0.0", "yaw": "0.0"}.items(),
        )],
    )

    # ── Part 3: Routing (取代 AIT*) ──
    routing_engine_node = Node(
        package="campusrover_routing",
        executable="routing_engine_node",
        name="routing_engine_node",
        output="screen",
        parameters=[{
            "enable_one_way": False,
            "use_csv": False,
            "path_orienation": False,
            "file_path1": os.path.join(routing_pkg, "share/node_module/3F_modul.csv"),
            "file_path2": os.path.join(routing_pkg, "share/node_module/3F_modul.csv"),
            "file_path3": os.path.join(routing_pkg, "share/node_module/3F_modul.csv"),
            "file_node_info": os.path.join(routing_pkg, "share/node_module/3F_info.csv"),
            "connect_method": "BezierCurve",
            "path_resolution": 0.05,
            "bezier_length": 1.5,
            "bezier_resolution": 0.01,
            "BSpline_k": 3,
            "BSpline_resolution": 0.001,
            "path_frame": "map",
        }],
    )
    mapinfo_db_handler_node = Node(
        package="campusrover_routing",
        executable="mapinfo_db_handler.py",
        name="mapinfo_db_handler",
        output="log",
        parameters=[
            {"use_database": False},
            {"json_folder": os.path.join(routing_pkg, "share/json/")},
        ],
    )
    routes_visualization_node = Node(
        package="campusrover_routing",
        executable="routes_visualization",
        name="routes_visualization_node",
        output="log",
    )

    # routing_to_path 橋接：routing service → /global_path topic
    routing_to_path_node = Node(
        package="rover_rl_inference",
        executable="routing_to_path",
        name="routing_to_path",
        output="screen",
        parameters=[{
            "building": "itc",
            "floor": "3",
            "topic_global_path": "/global_path",
        }],
    )

    # RViz Publish Point → routing 橋接（點兩下選起終點）
    routing_click_bridge_node = Node(
        package="rover_rl_inference",
        executable="routing_click_bridge",
        name="routing_click_bridge",
        output="screen",
        parameters=[{"building": "itc", "floor": "3"}],
    )

    tf_world_to_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_map_broadcaster",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                    "--frame-id", "world", "--child-frame-id", "map"],
    )

    # ── Part 4: Costmap (可選) ──
    local_costmap_params = os.path.join(costmap_pkg, "config",
                                         "local_costmap.yaml")
    local_costmap_node = Node(
        package="campusrover_costmap_ros2",
        executable="local_costmap_node",
        name="campusrover_costmap",
        output="log",
        parameters=[local_costmap_params],
        remappings=[
            ("points2", LaunchConfiguration("pointcloud_topic",
                                             default="velodyne_points")),
        ],
        condition=IfCondition(enable_costmap),
    )
    global_costmap_node = Node(
        package="campusrover_costmap_ros2",
        executable="global_costmap_node",
        name="global_costmap_node",
        output="log",
        parameters=[{
            "costmap_resolution": 0.0,
            "inflation_radius": 0.5,
            "cost_scaling_factor": 10.0,
        }],
        remappings=[
            ("map", "/map"),
            ("global_costmap", "/global_costmap"),
        ],
        condition=IfCondition(enable_costmap),
    )

    # ── Part 5: MOT (可選) ──
    mot_node = Node(
        package="campusrover_mot",
        executable="campusrover_mot_node",
        name="campusrover_mot_node",
        output="screen",
        remappings=[("points", "/velodyne_points")],
        parameters=[{
            "detection_area_min_x": -10.0, "detection_area_max_x": 10.0,
            "detection_area_min_y": -10.0, "detection_area_max_y": 10.0,
            "detection_area_min_z": -0.05, "detection_area_max_z": 0.5,
            "track_dead_time": 1.0, "track_older_age": 0.5,
            "cluster_dist": 0.35, "false_alarm_min": 10,
            "false_alarm_max": 3000, "weight_min_tolerate": 0.01,
            "cov_scale": 20.0, "inherit_ratio": 0.6,
            "history_length": 20, "anchor_dist_threshold": 0.3,
            "speed_threshold": 0.3,
            "trackers_update_period": 0.05, "label_update_period": 0.1,
            "map_frame": "map", "laser_frame": "scan",
            "camera_frame": "camera_link",
            "h_scale": 2.0, "v_scale": 3.0,
            "sync_tolerate": 0.08, "tf_tolerate": 1.0,
            "debug_mode": True, "is_use_laser": False,
            "is_map_filter": True, "is_img_label": False,
            "only_dynamic_obstacle": False,
        }],
        condition=IfCondition(enable_mot),
    )
    mot_marker_node = Node(
        package="campusrover_mot",
        executable="mot_marker_node.py",
        name="mot_marker_node",
        output="screen",
        remappings=[
            ("tracked_obstacles", "/tracked_label_obstacle"),
            ("obstacles_marker", "/obstacles_marker_3d_marker"),
        ],
        condition=IfCondition(enable_mot),
    )

    # ── Part 6: rover_rl — LiDAR Preprocessor ──
    preprocessor_node = Node(
        package="rover_rl_inference",
        executable="lidar_preprocessor",
        name="rover_rl_lidar_preprocessor",
        output="screen",
        emulate_tty=True,
        parameters=[pre_params_file],
        arguments=["--ros-args", "--log-level", log_level],
        condition=IfCondition(enable_preprocessor),
    )

    # ── Part 7: rover_rl — Policy Node ──
    def make_policy_node(context, *args, **kwargs):
        mp = LaunchConfiguration("model_path").perform(context)
        mode = LaunchConfiguration("initial_mode").perform(context)
        lv = LaunchConfiguration("log_level").perform(context)
        extra = {}
        if mp:
            extra["model_path"] = mp
        extra["initial_mode"] = mode
        return [Node(
            package="rover_rl_inference",
            executable="policy_node",
            name="rover_rl_policy",
            output="screen",
            emulate_tty=True,
            parameters=[params_file, extra],
            arguments=["--ros-args", "--log-level", lv],
        )]
    policy_node = OpaqueFunction(function=make_policy_node)

    # ── Part 8: rover_rl — BEV Play ──
    bev_play_node = Node(
        package="rover_rl_inference",
        executable="bev_play",
        name="rover_rl_bev_play",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "frame_mode": "body",
            "rate_hz": 5.0,
            "r_max": 20.0,
            "r_robot": 0.35,
            "topic_obs_debug": "/rover_rl_policy/obs_debug",
        }],
        arguments=["--ros-args", "--log-level", log_level],
        condition=IfCondition(enable_bev),
    )

    # ── Part 9: rover_rl — 診斷記錄（被動，不影響推論）──
    diag_logger_node = Node(
        package="rover_rl_inference",
        executable="diag_logger",
        name="rover_rl_diag_logger",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "rate_hz": 20.0,
            "log_dir": os.path.expanduser("~/rover_rl/logs"),
            "topic_cmd_vel": "/input/nav_cmd_vel",
            "topic_obs_debug": "/rover_rl_policy/obs_debug",
            "topic_record_ctrl": "/rover_rl/record",
            "require_start": LaunchConfiguration("require_start"),
            "log_only_with_goal": True,
            "enable_wandb": LaunchConfiguration("enable_wandb"),
            "wandb_mode": LaunchConfiguration("wandb_mode"),
        }],
        condition=IfCondition(LaunchConfiguration("enable_diag")),
    )

    # ── Banner ──
    banner = LogInfo(msg=(
        "================================\n"
        "rover_rl deploy_full 啟動\n"
        "  campusrover 棧:\n"
        "    [0] Map Server (/map)\n"
        "    [1] NDT Localization (/ndt_pose)\n"
        "    [2] Routing Engine (generation_path svc)\n"
        "    [3] routing_to_path → /global_path\n"
        "    [4] Costmap (可選)\n"
        "    [5] MOT (可選)\n"
        "    [6] RViz\n"
        "  rover_rl 棧:\n"
        "    [7] lidar_preprocessor\n"
        "    [8] policy_node\n"
        "    [9] bev_play\n"
        "  排除: DWA + AIT* (由 RL policy + routing 取代)\n"
        "================================"
    ))

    return LaunchDescription([
        # ── args ──
        DeclareLaunchArgument("model_path", default_value="",
                              description="覆寫 yaml model_path"),
        DeclareLaunchArgument("initial_mode", default_value="nav"),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("preprocessor_params_file",
                              default_value=default_pre_params),
        DeclareLaunchArgument("enable_bev", default_value="true"),
        DeclareLaunchArgument("enable_preprocessor", default_value="true"),
        DeclareLaunchArgument("enable_mot", default_value="true"),
        DeclareLaunchArgument("enable_costmap", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("enable_diag", default_value="true",
                              description="診斷記錄節點（goal 後記 CSV 到 ~/rover_rl/logs）"),
        DeclareLaunchArgument("require_start", default_value="true",
                              description="true=待命，需另一終端送 start 才開錄（每次=乾淨實驗）"),
        DeclareLaunchArgument("enable_wandb", default_value="false",
                              description="診斷記錄同步上 wandb（需先 pip install wandb）"),
        DeclareLaunchArgument("wandb_mode", default_value="offline",
                              description="wandb 模式：offline(實車建議)/online/disabled"),
        DeclareLaunchArgument("map_file",
                              default_value="/home/aa/maps/4v3F.yaml"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument("step_size", default_value="0.1"),
        DeclareLaunchArgument("trans_epsilon", default_value="0.00001"),
        DeclareLaunchArgument("max_iterations", default_value="10"),
        DeclareLaunchArgument("converged_param", default_value="1.5"),

        banner,

        # campusrover 棧
        map_server_node,
        rviz_node,
        ndt_localizer_node,
        tf_static_launch,
        points_downsample_launch,
        map_loader_launch,
        routing_engine_node,
        mapinfo_db_handler_node,
        routes_visualization_node,
        routing_to_path_node,
        routing_click_bridge_node,
        tf_world_to_map,
        local_costmap_node,
        global_costmap_node,
        mot_node,
        mot_marker_node,

        # rover_rl 棧
        preprocessor_node,
        policy_node,
        bev_play_node,
        diag_logger_node,
    ])
