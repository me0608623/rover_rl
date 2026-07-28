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
from launch.substitutions import LaunchConfiguration, PythonExpression
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
    # diag_logger 設定真值（auto_rearm / goal_change_eps_m / wandb 等都在這）
    diag_params_file = os.path.join(rl_pkg, "config", "diag_logger_params.yaml")
    recovery_params_file = os.path.join(rl_pkg, "config", "recovery_supervisor_params.yaml")

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
            "connect_method": "common",
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
            {"json_file": "itc_3f_3.json"},  # 直接讀此檔，忽略 building/floor 拼名
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
    #   - enable_vo/enable_orca/enable_recovery=true 時把 policy 輸出改道到
    #     /rover_rl/cmd_vel_desired，讓外層 wrapper 接手後才送進 mux
    #     （policy 自己不直接發 /input/nav_cmd_vel）
    def make_policy_node(context, *args, **kwargs):
        mp = LaunchConfiguration("model_path").perform(context)
        mode = LaunchConfiguration("initial_mode").perform(context)
        lv = LaunchConfiguration("log_level").perform(context)
        vo_on = LaunchConfiguration("enable_vo").perform(context).lower() == "true"
        orca_on = LaunchConfiguration("enable_orca").perform(context).lower() == "true"
        recovery_on = LaunchConfiguration("enable_recovery").perform(context).lower() == "true"
        mppi_on = LaunchConfiguration("enable_mppi").perform(context).lower() == "true"
        extra = {}
        if mp:
            extra["model_path"] = mp     # 非空才覆寫
        extra["initial_mode"] = mode
        if vo_on or orca_on or recovery_on or mppi_on:
            extra["topic_cmd_vel"] = "/rover_rl/cmd_vel_desired"   # 改道給外層 wrapper
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
    # 設定真值在 config/diag_logger_params.yaml；用 OpaqueFunction 讓 CLI arg
    # 留空(預設"")時走 yaml、有給才覆寫（一個 goal/path = 一段，到終點自動停+re-arm）
    def make_diag_node(context, *args, **kwargs):
        ov = {}
        _b = lambda v: v.lower() == "true"     # "true"/"false" → bool
        rs = LaunchConfiguration("require_start").perform(context)
        ew = LaunchConfiguration("enable_wandb").perform(context)
        wm = LaunchConfiguration("wandb_mode").perform(context)
        ar = LaunchConfiguration("auto_rearm").perform(context)
        ge = LaunchConfiguration("goal_change_eps_m").perform(context)
        if rs != "":
            ov["require_start"] = _b(rs)
        if ew != "":
            ov["enable_wandb"] = _b(ew)
        if wm != "":
            ov["wandb_mode"] = wm
        if ar != "":
            ov["auto_rearm"] = _b(ar)
        if ge != "":
            ov["goal_change_eps_m"] = float(ge)
        return [Node(
            package="rover_rl_inference",
            executable="diag_logger",
            name="rover_rl_diag_logger",
            output="screen",
            emulate_tty=True,
            parameters=[diag_params_file, ov],   # yaml 為底，ov 只覆寫有給的 CLI arg
            condition=IfCondition(LaunchConfiguration("enable_diag")),
        )]
    diag_logger_node = OpaqueFunction(function=make_diag_node)

    # ── Part 10: LV-DOT 動態障礙物偵測（LiDAR+depth 融合，發 odom frame markers，見 vis_frame 參數）──
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
    # YOLOv11 視覺輔助：預設跟隨 LV-DOT，一併啟停；仍可顯式 false 關閉。
    lvdot_yolo_node = Node(
        package="onboard_detector",
        executable="yolov11_detector_node.py",
        name="yolov11_detector_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_lvdot_yolo")),
    )

    # ── Part 10b: MPPI 靜態避障層（RL 導航 + MPPI 靜態 + VO 動態，三層協作）──
    # policy → /rover_rl/cmd_vel_desired → [MPPI static_guard] → /rover_rl/cmd_vel_mppi → VO → …
    # MPPI 吃 RL 意圖當 reference（導航），只用 /campusrover_local_costmap 做靜態避障，
    # dynamic critic 關（yaml），啟動即 autostart。輸出下一站：VO 開→cmd_vel_mppi；
    # 無 VO 有 recovery→cmd_vel_recovery_in；都無→直接 /input/nav_cmd_vel。
    def make_mppi_static_guard_node(context, *args, **kwargs):
        mppi_on = LaunchConfiguration("enable_mppi").perform(context).lower() == "true"
        if not mppi_on:
            return []
        vo_on = LaunchConfiguration("enable_vo").perform(context).lower() == "true"
        recovery_on = LaunchConfiguration("enable_recovery").perform(context).lower() == "true"
        lv = LaunchConfiguration("log_level").perform(context)
        if vo_on:
            out_topic = "/rover_rl/cmd_vel_mppi"
        elif recovery_on:
            out_topic = "/rover_rl/cmd_vel_recovery_in"
        else:
            out_topic = "/input/nav_cmd_vel"
        cm_pkg = get_package_share_directory("campusrover_move")
        mppi_cfg = LaunchConfiguration("mppi_params_file").perform(context)
        if not mppi_cfg:
            mppi_cfg = os.path.join(cm_pkg, "config", "mppi_static_guard.yaml")
        costmap_topic = LaunchConfiguration("mppi_costmap_topic").perform(context)
        return [Node(
            package="campusrover_move",
            executable="mppi_planner",
            name="mppi_planner_node",
            output="screen",
            emulate_tty=True,
            parameters=[mppi_cfg],
            arguments=["--ros-args", "--log-level", lv],
            remappings=[
                ("reference_cmd", "/rover_rl/cmd_vel_desired"),
                ("costmap", costmap_topic),
                ("cmd_vel", out_topic),
                ("odom", "/odom"),
                ("global_path", "/rover_rl/_mppi_unused_path"),
                ("elevator_path", "/rover_rl/_mppi_unused_elevator"),
                ("dynamic_obstacles", "/rover_rl/_mppi_unused_dynobs"),
            ],
        )]
    mppi_static_guard_node = OpaqueFunction(function=make_mppi_static_guard_node)

    # ── Part 11: VO 安全層（夾在 RL policy 與底盤 mux 之間）──
    # 一般模式:
    #   policy → /rover_rl/cmd_vel_desired → [vo_safety] → /input/nav_cmd_vel → mux
    # Recovery 模式:
    #   policy → /rover_rl/cmd_vel_desired → [vo_safety] → /rover_rl/cmd_vel_recovery_in
    #          → [recovery_supervisor] → /input/nav_cmd_vel → mux
    # 吃 vo_interface/tracked_obstacles（KF 平滑速度）做動態障礙預測式避障/煞停濾波。
    # ⚠️ enable_vo 預設跟隨 enable_lvdot；首次仍請先架空 + 單獨驗證行為。
    # VO 參數檔可用 vo_params_file:= 覆寫（預設 vo_params.yaml；滿血版帶 vo_params_full.yaml）。
    # 兩份完全獨立、同一個 vo_safety_node，換 yaml 就換行為，不互相干擾。
    def make_vo_safety_node(context, *args, **kwargs):
        vo_on = LaunchConfiguration("enable_vo").perform(context).lower() == "true"
        if not vo_on:
            return []
        recovery_on = LaunchConfiguration("enable_recovery").perform(context).lower() == "true"
        static_avoid_on = LaunchConfiguration("enable_static_avoid").perform(context).lower() == "true"
        mppi_on = LaunchConfiguration("enable_mppi").perform(context).lower() == "true"
        lv = LaunchConfiguration("log_level").perform(context)
        extra = {}
        if static_avoid_on:
            extra["static_avoid_enable"] = True
        if mppi_on:
            # 三層協作：RL→MPPI(靜態)→VO(動態)。VO 改吃 MPPI 靜態濾波後的輸出。
            extra["topic_cmd_in"] = "/rover_rl/cmd_vel_mppi"
            # MPPI gap guard 會在 costmap 上提供偏好 gap，最後由 MPPI rollout 選命令；
            # VO 的舊式 front_freeze 會在 0.7m 內把 (vx,wz) 一起歸零，導致車頭明明有
            # 側向 gap 卻只會停住。
            # 保留 front_brake 的 0.55m 絕對底線，取消會抹掉轉向的 freeze。
            extra["front_freeze_enable"] = False
        if recovery_on:
            # Keep VO's ordinary dynamic-obstacle filtering, but replace VO's own
            # backing-up escape with recovery_supervisor downstream.
            # ⚠ 此組合下 VO 自己絕不後退，若下游 recovery 也沒武裝（例：人偏一側、±30° 填滿率
            #   0.38 落在 recovery 的 ratio 死區），VO 的 0.6m 硬停會把 v/ω 一起歸零而無人接手
            #   → 靠 vo_params.yaml 的 deadlock_release_s（VO 給 0 且車不動 3.5s → 交還 RL）兜底。
            extra.update({
                "topic_cmd_out": "/rover_rl/cmd_vel_recovery_in",
                "stuck_escape_enable": False,
                "front_brake_reverse": False,
                "front_hardstop_dwell_s": 999.0,
            })
        params = [LaunchConfiguration("vo_params_file")]
        if extra:
            params.append(extra)
        return [Node(
            package="rover_rl_inference",
            executable="vo_safety",
            name="vo_safety_node",
            output="screen",
            emulate_tty=True,
            parameters=params,
            arguments=["--ros-args", "--log-level", lv],
        )]
    vo_safety_node = OpaqueFunction(function=make_vo_safety_node)

    # ── Part 11b: ORCA 安全層（獨立於 VO，enable_orca 控制）──
    # policy → /rover_rl/cmd_vel_desired → [orca_safety] → /input/nav_cmd_vel → mux
    # 吃 vo_interface/tracked_obstacles 跑 RVO2 non-cooperative 避讓 + lateral_evasion。
    # 與 vo_safety_node 互斥（deploy_select 保證 enable_vo 與 enable_orca 不同時 true）。
    # Recovery 啟用時也停用 ORCA，避免多個 wrapper 同時發布 /input/nav_cmd_vel。
    # 安全網：看門狗 + front_brake（吃 policy status front_m）+ slew 限速。無 escape/commit。
    orca_safety_node = Node(
        package="orca_filter",
        executable="orca_safety",
        name="orca_safety_node",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("orca_params_file")],
        arguments=["--ros-args", "--log-level", log_level],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("enable_orca"), "' == 'true' and '",
            LaunchConfiguration("enable_recovery"), "' != 'true'"
        ])),
    )

    # ── Part 11c: Recovery Supervisor（獨立 cmd_vel wrapper）──
    # 無 VO:
    #   policy → /rover_rl/cmd_vel_desired → [recovery_supervisor] → /input/nav_cmd_vel → mux
    # 有 VO:
    #   policy → VO → /rover_rl/cmd_vel_recovery_in → [recovery_supervisor] → /input/nav_cmd_vel
    def make_recovery_supervisor_node(context, *args, **kwargs):
        recovery_on = LaunchConfiguration("enable_recovery").perform(context).lower() == "true"
        if not recovery_on:
            return []
        vo_on = LaunchConfiguration("enable_vo").perform(context).lower() == "true"
        lv = LaunchConfiguration("log_level").perform(context)
        extra = {
            "topic_cmd_in": (
                "/rover_rl/cmd_vel_recovery_in"
                if vo_on else "/rover_rl/cmd_vel_desired"
            )
        }
        return [Node(
            package="rover_rl_inference",
            executable="recovery_supervisor",
            name="recovery_supervisor_node",
            output="screen",
            emulate_tty=True,
            parameters=[LaunchConfiguration("recovery_params_file"), extra],
            arguments=["--ros-args", "--log-level", lv],
        )]
    recovery_supervisor_node = OpaqueFunction(function=make_recovery_supervisor_node)

    # ── Part 12: 兩固定點往返避障測試（預設關，enable_pingpong:=true 開啟）──
    # 把車手動開到 A/B 任一點停穩 → 自動規劃往對向點，A↔B 無限來回，供反覆測避障。
    # 與 RL 推論/避障完全解耦：只訂閱 policy status、呼叫 routing、必要時切 mode=nav。
    pingpong_test_node = Node(
        package="rover_rl_inference",
        executable="pingpong_test",
        name="pingpong_test",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "point_a": LaunchConfiguration("pingpong_a"),
            "point_b": LaunchConfiguration("pingpong_b"),
            "building": "itc",
            "floor": "3",
            "auto_set_nav": LaunchConfiguration("pingpong_auto_nav"),
            "auto_continue": LaunchConfiguration("pingpong_auto_continue"),
        }],
        arguments=["--ros-args", "--log-level", log_level],
        condition=IfCondition(LaunchConfiguration("enable_pingpong")),
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
        # 以下 5 個 diag 參數的真值在 config/diag_logger_params.yaml；
        # 預設留空("")=走 yaml，CLI 有給才覆寫（現場熱調，免改 yaml）
        DeclareLaunchArgument("require_start", default_value="",
                              description="覆寫 yaml：false=發 goal 即開錄；true=需 record start。空=走 yaml"),
        DeclareLaunchArgument("enable_wandb", default_value="",
                              description="覆寫 yaml：診斷同步上 wandb（true/false）。空=走 yaml"),
        DeclareLaunchArgument("wandb_mode", default_value="",
                              description="覆寫 yaml：online/offline/disabled。空=走 yaml"),
        DeclareLaunchArgument("auto_rearm", default_value="",
                              description="覆寫 yaml：到 goal 停後自動 re-arm 開新一段（true/false）。空=走 yaml"),
        DeclareLaunchArgument("goal_change_eps_m", default_value="",
                              description="覆寫 yaml：新目標 vs 同終點重複發布的位移門檻(m)。空=走 yaml"),
        DeclareLaunchArgument("enable_lvdot", default_value="true",
                              description="LV-DOT 動態障礙物偵測（→ /onboard_detector/*）"),
        DeclareLaunchArgument("enable_lvdot_yolo",
                              default_value=LaunchConfiguration("enable_lvdot"),
                              description="LV-DOT YOLOv11 視覺輔助（需 ultralytics；預設跟隨 enable_lvdot）"),
        DeclareLaunchArgument("enable_vo", default_value=LaunchConfiguration("enable_lvdot"),
                              description="VO 安全層（RL→VO→mux，吃 vo_interface 動態障礙避障）。"
                                          "預設跟隨 enable_lvdot（有 LV-DOT 才開 VO，因 VO 障礙來自 "
                                          "vo_interface/tracked_obstacles，需 lv-dot 在跑）；開啟同時會把 "
                                          "policy 輸出改道到 /rover_rl/cmd_vel_desired。"
                                          "首次驗證可顯式 enable_vo:=false 關閉"),
        DeclareLaunchArgument("vo_params_file",
                              default_value=os.path.join(rl_pkg, "config", "vo_params.yaml"),
                              description="VO 安全層參數檔。預設 vo_params.yaml；"
                                          "滿血版帶 vo_params_full.yaml（engage_range=4m + 積極繞行，"
                                          "與預設完全獨立）。deploy_rl_shell 選 VO=Y 後可互動選滿血版"),
        DeclareLaunchArgument("enable_static_avoid", default_value="false",
                              description="VO 內建靜態早期避障：吃 policy LiDAR front/left/right，"
                                          "3m 內提早減速並往空側轉。需 enable_vo=true。"),
        DeclareLaunchArgument("enable_mppi", default_value="false",
                              description="MPPI 靜態避障層（RL 導航 + MPPI 靜態 + VO 動態）。"
                                          "啟用時 policy 輸出改道 /rover_rl/cmd_vel_desired → "
                                          "MPPI(static_guard) → /rover_rl/cmd_vel_mppi → VO → mux。"
                                          "MPPI 吃 RL 意圖當 reference、只用 local costmap 避靜態、"
                                          "dynamic critic 關（讓 VO 管動態）。需 campusrover_move 已 build。"),
        DeclareLaunchArgument("mppi_params_file", default_value="",
                              description="MPPI static_guard 參數檔。空=用 campusrover_move 內建 "
                                          "config/mppi_static_guard.yaml"),
        DeclareLaunchArgument("mppi_costmap_topic", default_value="/campusrover_local_costmap",
                              description="MPPI 靜態避障吃的 local costmap topic（需 enable_costmap）"),
        DeclareLaunchArgument("enable_orca", default_value="false",
                              description="ORCA 安全層（RL→ORCA→mux，RVO2 non-cooperative 避讓）。"
                                          "啟用時 policy 輸出改道到 /rover_rl/cmd_vel_desired → "
                                          "orca_safety_node → /input/nav_cmd_vel。與 enable_vo 互斥。"),
        DeclareLaunchArgument("orca_params_file",
                              default_value=os.path.join(
                                  get_package_share_directory("orca_filter"), "config", "orca_params.yaml"),
                              description="ORCA 安全層參數檔（orca_filter/config/orca_params.yaml）。"),
        DeclareLaunchArgument("enable_recovery", default_value="false",
                              description="Recovery Supervisor 脫困 wrapper。啟用時 policy 輸出改道到 "
                                          "/rover_rl/cmd_vel_desired；若 enable_vo=true，流程為 "
                                          "RL→VO→recovery_supervisor→/input/nav_cmd_vel，"
                                          "並關閉 VO 內建倒退逃脫，讓 Recovery 專管倒退/脫困。"),
        DeclareLaunchArgument("recovery_params_file",
                              default_value=recovery_params_file,
                              description="Recovery Supervisor 參數檔。"),
        DeclareLaunchArgument("enable_pingpong", default_value="false",
                              description="兩固定點往返避障測試：車停在 A/B 任一點即自動往對向點來回"),
        DeclareLaunchArgument("pingpong_a", default_value="c24",
                              description="往返測試 A 點（routing 拓撲節點名）"),
        DeclareLaunchArgument("pingpong_b", default_value="c27",
                              description="往返測試 B 點（routing 拓撲節點名）"),
        DeclareLaunchArgument("pingpong_auto_nav", default_value="true",
                              description="往返測試啟動時自動切 mode=nav 讓 RL 接手"),
        DeclareLaunchArgument("pingpong_auto_continue", default_value="false",
                              description="往返測試全自動模式：就緒後不必按空白鍵、停穩自動出發下一段"
                                          "（執行中可用 TUI 'a' 鍵熱切換）"),
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

        # MPPI 靜態避障層（預設關，enable_mppi:=true 開啟；RL→MPPI→VO 三層協作）
        mppi_static_guard_node,

        # VO 安全層（預設關，enable_vo:=true 開啟）
        vo_safety_node,
        orca_safety_node,
        recovery_supervisor_node,

        # 兩固定點往返避障測試（預設關，enable_pingpong:=true 開啟）
        pingpong_test_node,
    ])
