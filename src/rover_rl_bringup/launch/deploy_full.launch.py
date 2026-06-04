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
    IncludeLaunchDescription,   # 引入其他 launch 檔（NDT 子模組用）
    LogInfo,                    # 啟動 banner
    OpaqueFunction,             # 啟動時讀參數真值再建節點（policy 用）
    TimerAction,                # 延遲啟動（NDT 子模組需等地圖/降採樣就緒）
)
from launch.conditions import IfCondition   # 依 bool 參數決定節點是否啟動
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── package paths ──
    # 取得各 package 安裝後的 share 路徑，用來組參數檔/子 launch 的絕對路徑
    rl_pkg = get_package_share_directory("rover_rl_bringup")          # 本 package
    ndt_pkg = get_package_share_directory("ndt_localizer")            # NDT 定位
    costmap_pkg = get_package_share_directory("campusrover_costmap_ros2")  # costmap
    routing_pkg = get_package_share_directory("campusrover_routing")  # 拓撲路徑規劃

    default_params = os.path.join(rl_pkg, "config", "policy_params.yaml")
    default_pre_params = os.path.join(rl_pkg, "config",
                                       "lidar_preprocessor_params.yaml")

    # ── args ──
    # 全部宣告為 LaunchConfiguration（延遲取值），實際預設值在 return 區的
    # DeclareLaunchArgument 設定，可由命令列覆寫
    model_path = LaunchConfiguration("model_path")
    initial_mode = LaunchConfiguration("initial_mode")
    params_file = LaunchConfiguration("params_file")
    pre_params_file = LaunchConfiguration("preprocessor_params_file")
    enable_bev = LaunchConfiguration("enable_bev")
    enable_preprocessor = LaunchConfiguration("enable_preprocessor")
    enable_mot = LaunchConfiguration("enable_mot")
    enable_costmap = LaunchConfiguration("enable_costmap")
    rviz = LaunchConfiguration("rviz")
    enable_ndt = LaunchConfiguration("enable_ndt")
    log_level = LaunchConfiguration("log_level")
    map_file = LaunchConfiguration("map_file")

    # ── Part 0: Map Server ──
    # 讀 yaml 地圖檔並持續發布到 /map（供 RViz / global_costmap / routing 用）
    map_server_node = Node(
        package="campusrover_demo",
        executable="simple_map_publisher",
        name="map_server",
        output="log",
        parameters=[{"map_file": map_file}],
    )

    # ── Part 1: RViz ──
    # 可視化介面，載入固定的 .rviz 設定檔（本機路徑，未納入 git）
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_demo",
        arguments=["-d", "/home/aa/rviz/demo.rviz"],
        output="log",
        condition=IfCondition(rviz),
    )

    # ── Part 2: NDT Localization ──
    # 用 NDT 點雲配準算出 map→odom TF + /ndt_pose（提供全域定位）
    # 收斂參數（resolution/step_size 等）皆由 LaunchConfiguration 帶入，可調
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
        condition=IfCondition(enable_ndt),
    )

    # NDT 三個子 launch，用 TimerAction 錯開啟動時間避免相依未就緒：
    # tf_static(0s) → points_downsample(2s) → map_loader(3s)
    tf_static_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ndt_pkg, "launch", "tf_static_launch.py")),
        condition=IfCondition(enable_ndt),
    )
    # 點雲降採樣：減少 NDT 配準計算量；延後 2 秒等 TF 就緒
    points_downsample_launch = TimerAction(
        period=2.0,
        condition=IfCondition(enable_ndt),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ndt_pkg, "launch", "points_downsample_launch.py")),
        )],
    )
    # 載入 PCD 點雲地圖；延後 3 秒，並給定初始位姿（全 0）
    map_loader_launch = TimerAction(
        period=3.0,
        condition=IfCondition(enable_ndt),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ndt_pkg, "launch", "map_loader_launch.py")),
            launch_arguments={"x": "0.0", "y": "0.0", "z": "0.0",
                              "roll": "0.0", "pitch": "0.0", "yaw": "0.0"}.items(),
        )],
    )

    # ── Part 3: Routing (取代 AIT*) ──
    # 拓撲路徑規劃引擎：讀 node CSV 圖，提供 generation_path service
    # 用 Bezier 曲線連接節點，輸出平滑全域路徑（path_frame=map）
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
    # 地圖節點資訊處理（從 json 讀節點資料；不接資料庫）
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
    # 把規劃出的路徑畫成 RViz marker
    routes_visualization_node = Node(
        package="campusrover_routing",
        executable="routes_visualization",
        name="routes_visualization_node",
        output="log",
    )

    # routing_to_path 橋接：呼叫 routing service 取得路徑 → 2Hz republish 到
    # /global_path topic，讓 policy_node 的 SubgoalSelector 能訂閱
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

    # RViz Publish Point → routing 橋接：在 RViz 點兩下（第1點起點、第2點終點）
    # 即自動呼叫 routing service 規劃路徑
    routing_click_bridge_node = Node(
        package="rover_rl_inference",
        executable="routing_click_bridge",
        name="routing_click_bridge",
        output="screen",
        parameters=[{"building": "itc", "floor": "3"}],
    )

    # world→map 靜態 TF（單位轉換，補 TF 鏈最上層；僅 NDT 模式需要）
    tf_world_to_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_map_broadcaster",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                    "--frame-id", "world", "--child-frame-id", "map"],
        condition=IfCondition(enable_ndt),
    )

    # ── Part 4: Costmap (可選) ──
    # 注意：RL policy 不吃 costmap，這裡純供 RViz debug / 對照用
    local_costmap_params = os.path.join(costmap_pkg, "config",
                                         "local_costmap.yaml")
    # 局部 costmap：由即時點雲建障礙物層
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
    # 全域 costmap：由 /map 加膨脹層（inflation 0.5m）
    global_costmap_node = Node(
        package="campusrover_costmap_ros2",
        executable="global_costmap_node",
        name="global_costmap_node",
        output="log",
        parameters=[{
            "costmap_resolution": 0.0,    # 0=沿用地圖原解析度
            "inflation_radius": 0.5,      # 障礙物膨脹半徑 (m)
            "cost_scaling_factor": 10.0,  # 代價衰減速率
        }],
        remappings=[
            ("map", "/map"),
            ("global_costmap", "/global_costmap"),
        ],
        condition=IfCondition(enable_costmap),
    )

    # ── Part 5: MOT (可選) ──
    # 多目標追蹤：從點雲分群偵測並追蹤動態障礙物（行人等）
    # 偵測範圍 ±10m、z 軸 -0.05~0.5m（濾地板/高處），輸出追蹤框
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
    # 把追蹤到的障礙物轉成 RViz 3D marker
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
    # 把 /velodyne_points 處理成 72-bin sweep（對齊訓練端公式）→ 發給 policy
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
    # 用 OpaqueFunction 在啟動時讀取參數真值：
    #   - model_path 為空時不覆寫（保留 yaml 預設，避免空字串蓋掉）
    #   - initial_mode 一律覆寫（首次部署建議 idle，確認後再切 nav）
    #   - enable_vo=true 時把 policy 輸出改道到 /rover_rl/cmd_vel_desired，
    #     讓 VO 安全層接手後才送進 mux（policy 自己不直接發 /input/nav_cmd_vel）
    def make_policy_node(context, *args, **kwargs):
        mp = LaunchConfiguration("model_path").perform(context)
        mode = LaunchConfiguration("initial_mode").perform(context)
        lv = LaunchConfiguration("log_level").perform(context)
        vo_on = LaunchConfiguration("enable_vo").perform(context).lower() == "true"
        extra = {}
        if mp:
            extra["model_path"] = mp     # 非空才覆寫
        extra["initial_mode"] = mode
        if vo_on:
            extra["topic_cmd_vel"] = "/rover_rl/cmd_vel_desired"   # 改道給 VO
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
    # 純可視化：把 sweep + goal + cmd_vel 畫成極座標 BEV 圖（matplotlib Agg）
    # → /rover_rl/bev_image，供上電前肉眼確認 LiDAR 看得到障礙物。policy 不吃此圖
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
    # 訂閱 odom/ndt/goal/cmd_vel/obs，20Hz 寫 CSV 到 ~/rover_rl/logs/diag/
    # 收到第一個 goal/path 才開始建資料夾記錄（log_only_with_goal），可同步 wandb
    diag_logger_node = Node(
        package="rover_rl_inference",
        executable="diag_logger",
        name="rover_rl_diag_logger",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "rate_hz": 20.0,
            "log_dir": os.path.expanduser("~/rover_rl/logs/diag"),
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

    # ── Part 10: LV-DOT 動態障礙物偵測（LiDAR+depth 融合，發 map frame markers）──
    # 與 policy 解耦：偵測結果發到 /onboard_detector/*，供 status_tui / RViz 觀察，
    # policy 推論不吃此資料（obs 障礙欄仍補 0）
    lvdot_pkg = get_package_share_directory("onboard_detector")
    lvdot_params = os.path.join(lvdot_pkg, "cfg", "detector_param.yaml")
    # 主偵測器：LiDAR + depth 融合輸出動態障礙框
    lvdot_detector_node = Node(
        package="onboard_detector",
        executable="detector_node",
        name="dynamic_detector",
        output="screen",
        parameters=[lvdot_params],
        condition=IfCondition(LaunchConfiguration("enable_lvdot")),
    )
    # YOLOv11 視覺輔助（需 ultralytics，預設關閉）
    lvdot_yolo_node = Node(
        package="onboard_detector",
        executable="yolov11_detector_node.py",
        name="yolov11_detector_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_lvdot_yolo")),
    )

    # ── Part 11: VO 安全層（夾在 RL policy 與底盤 mux 之間）──
    # policy → /rover_rl/cmd_vel_desired → [vo_safety] → /input/nav_cmd_vel → mux
    # 用 LV-DOT 的 get_dynamic_obstacles service 做動態障礙預測式避障濾波。
    # ⚠️ 預設 enable_vo=false：這是新的安全關鍵層，請先架空 + 單獨驗證行為後再開。
    vo_params = os.path.join(rl_pkg, "config", "vo_params.yaml")
    vo_safety_node = Node(
        package="rover_rl_inference",
        executable="vo_safety",
        name="vo_safety_node",
        output="screen",
        emulate_tty=True,
        parameters=[vo_params],
        arguments=["--ros-args", "--log-level", log_level],
        condition=IfCondition(LaunchConfiguration("enable_vo")),
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
        "    [10] LV-DOT 動態偵測 (/onboard_detector/*)\n"
        "  排除: DWA + AIT* (由 RL policy + routing 取代)\n"
        "================================"
    ))

    return LaunchDescription([
        # ── args ──
        # 在此設定所有啟動參數的「預設值」與說明，可在命令列覆寫
        # 例：ros2 launch ... deploy_full.launch.py initial_mode:=idle enable_mot:=false
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
        DeclareLaunchArgument("enable_ndt", default_value="true",
                              description="false 則不啟內建 NDT（改用單獨的 ndt alias）"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("enable_diag", default_value="true",
                              description="診斷記錄節點（goal 後記 CSV 到 ~/rover_rl/logs）"),
        DeclareLaunchArgument("require_start", default_value="false",
                              description="false=deploy 後自動待錄，發 goal 即開始記錄；"
                                          "true=需另送 /rover_rl/record start 才開錄"),
        DeclareLaunchArgument("enable_wandb", default_value="true",
                              description="診斷記錄同步上 wandb（run 名=diag_日期_時間）"),
        DeclareLaunchArgument("wandb_mode", default_value="offline",
                              description="wandb 模式：offline(實車建議，回頭 wandb sync)/online/disabled"),
        DeclareLaunchArgument("enable_lvdot", default_value="true",
                              description="LV-DOT 動態障礙物偵測（→ /onboard_detector/*）"),
        DeclareLaunchArgument("enable_lvdot_yolo", default_value="false",
                              description="LV-DOT YOLOv11 視覺輔助（需 ultralytics，預設關）"),
        DeclareLaunchArgument("enable_vo", default_value="false",
                              description="VO 安全層（RL→VO→mux，用 LV-DOT 動態障礙避障）。"
                                          "新的安全關鍵層，驗證前預設關；開啟同時會把 "
                                          "policy 輸出改道到 /rover_rl/cmd_vel_desired"),
        DeclareLaunchArgument("map_file",
                              default_value="/home/aa/maps/4v3F.yaml"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument("step_size", default_value="0.1"),
        DeclareLaunchArgument("trans_epsilon", default_value="0.00001"),
        DeclareLaunchArgument("max_iterations", default_value="10"),
        DeclareLaunchArgument("converged_param", default_value="1.5"),

        banner,   # 先印啟動橫幅

        # campusrover 棧（定位 / 路徑 / costmap / MOT / RViz）
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

        # LV-DOT 動態障礙物偵測
        lvdot_detector_node,
        lvdot_yolo_node,

        # VO 安全層（預設關，enable_vo:=true 開啟）
        vo_safety_node,
    ])
