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
    [4] Costmap            — local_costmap (RViz debug 用；controller=dwa/pid_vo 時是避障輸入)
    [5] MOT                — 動態障礙物追蹤
    [6] RViz               — 可視化

  rover_rl 棧:
    [7] lidar_preprocessor — /velodyne_points → /rover_rl/lidar_sweep_72
                             （所有 controller 都啟：diag 的碰撞指標 + VO 的前方安全煞備援都吃它）
    [8] policy_node        — sweep + odom + goal → /input/nav_cmd_vel（僅 controller=rl）
    [9] bev_play           — 即時 BEV 圖 → /rover_rl/bev_image（僅 controller=rl）

  消融實驗 baseline（controller:=dwa|pid|pid_vo|mppi，論文用傳統演算法對照組）:
    controller=dwa      — campusrover_move/dwa_planner（軌跡取樣 DWA，吃 /campusrover_local_costmap
                           做靜態避障）→ /input/nav_cmd_vel。速度/角速度上限統一為底盤真實上限
                           v=1.0 m/s, ω=1.2 rad/s（與 RL 實際可達上限一致，見 dwa_baseline 參數）。
    controller=pid       — campusrover_move/path_following，關掉內建 DWA/costmap 避障
                           （純路徑跟蹤，無避障）→ /input/nav_cmd_vel。同一組速度上限。
    controller=pid_vo    — path_following（純跟蹤）→ /rover_rl/cmd_vel_baseline_desired
                           → vo_safety_node（動態避障）→ /input/nav_cmd_vel。
                           ⚠ PID 本身零避障、VO 只管動態障礙（靜態原本靠 RL 的 LiDAR 擋），
                           故 vo_safety_node 在無 policy status 時會改直接吃 72-bin sweep
                           還原 front/left/right_m，補回前方 LiDAR 安全煞（0.5m 硬停），
                           否則靜態障礙（牆/家具）在此組合下完全沒人擋。
    controller=mppi      — campusrover_move/mppi_planner 的原生 path-following 模式
                           （取樣式 MPC：batch 230 × horizon 70 步 × dt 0.05 = 3.5s 前瞻，
                           path_follow/path_align/goal/obstacle cost）→ /input/nav_cmd_vel。
                           參數 rover_rl_bringup/config/mppi_baseline.yaml，vx_max 拉到 1.0
                           與其他組對齊、dynamic critic 關（型別接不上 LV-DOT），
                           故與 dwa 同條件：都只吃 /campusrover_local_costmap 的靜態障礙。
                           ⚠ 與 enable_mppi:=true 的 static_guard 三層協作模式是兩回事，
                           後者吃 RL 的 reference_cmd、不吃 global_path，只在 controller=rl 啟。

    ⚠ 四組 baseline 都需要 baseline_arm 節點呼叫 planner_function* service 才會動
      （三個 planner 都是 action_flag_=false 開機，不 arm 就完全不發 cmd_vel 且不報錯），
      該 service 同時覆寫速度上限與避障開關 → 統一上限的真值在 baseline_arm，見 Part 11e。

    四者共用同一套 NDT/costmap/routing/LV-DOT/diag_logger/pingpong_test 基礎設施，只換
    「誰在發 /input/nav_cmd_vel」，確保與 controller=rl 比較時公平（同 TF、同 costmap、
    同測試協定）。diag_logger 的 experiment_tag 預設自動帶 controller 值。
    pingpong_test 在非 rl controller 下會用 require_policy_status:=false（TF+/odom 取代
    policy status），因為沒有 policy_node 可訂閱。

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

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,   # 引入其他 launch 檔（NDT 子模組用）
    LogInfo,                    # 啟動 banner
    OpaqueFunction,             # 啟動時讀參數真值再建節點（policy 用）
    SetLaunchConfiguration,     # 算好的 baseline 速度上限廣播給多個節點共用
    TimerAction,                # 延遲啟動（NDT 子模組需等地圖/降採樣就緒）
)
from launch.conditions import IfCondition   # 依 bool 參數決定節點是否啟動
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    controller = LaunchConfiguration("controller")

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
    # ⚠ 所有 controller 都要啟（含 dwa/pid/pid_vo baseline），不是只有 RL 需要：
    #   1. diag_logger 用 sweep 算 lidar_collision（<lidar_collision_m 判碰撞）與
    #      leg_min_sweep_m —— 那是論文 SR/TO/CR 的碰撞率來源，關掉等於 baseline 沒有碰撞指標。
    #   2. vo_safety_node 在沒有 policy status 時（controller=pid_vo）改直接吃 sweep 還原
    #      front/left/right_m，補回前方 LiDAR 安全煞（見 vo_safety_node._cb_lidar_sweep）。
    #   policy 本身不在時它只是多發一個 topic，成本極低。
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
        if LaunchConfiguration("controller").perform(context) != "rl":
            return []   # 消融實驗 baseline（dwa/pid/pid_vo）不跑 RL policy
        mp = LaunchConfiguration("model_path").perform(context)
        mode = LaunchConfiguration("initial_mode").perform(context)
        lv = LaunchConfiguration("log_level").perform(context)
        vo_on = LaunchConfiguration("enable_vo").perform(context).lower() == "true"
        orca_on = LaunchConfiguration("enable_orca").perform(context).lower() == "true"
        recovery_on = LaunchConfiguration("enable_recovery").perform(context).lower() == "true"
        mppi_on = LaunchConfiguration("enable_mppi").perform(context).lower() == "true"
        sr = LaunchConfiguration("speed_rate").perform(context)
        extra = {}
        if mp:
            extra["model_path"] = mp     # 非空才覆寫
        extra["initial_mode"] = mode
        if sr != "":
            extra["speed_rate"] = float(sr)   # 空=走 yaml；有給才覆寫（deploy_rl_shell 選單會帶）
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
        # 吃 policy 專屬的 obs_debug/lidar_sweep，baseline（dwa/pid/pid_vo）沒有這些資料
        condition=IfCondition(PythonExpression([
            "'", enable_bev, "' == 'true' and '", controller, "' == 'rl'"
        ])),
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
        et = LaunchConfiguration("experiment_tag").perform(context)
        ctrl = LaunchConfiguration("controller").perform(context)
        # 消融實驗：留空自動帶 controller（rl/dwa/pid/pid_vo/mppi），有給才用自訂值
        # （可自訂加場景後綴，如 dwa_fixed_obstacle）
        ov["experiment_tag"] = et if et != "" else ctrl
        # 消融實驗重現性：diag 的 <csv>_params.json 是「事後證明兩組跑在同一組上限」的
        # 唯一載體，但它只抓 policy_node_name 指到的那一個節點的參數，baseline 沒有
        # policy_node → 15 秒後逾時、params 整個空白，速度上限無從查證。
        # 改指向該 controller 的 planner 節點，記下它實際拿到的速度上限與演算法參數。
        # （planner 的 C++ service callback 改的是成員變數、不寫回 parameter server，
        #   但那份值與 baseline_arm 送進去的來自同一個 resolve_baseline_limits，故一致。）
        if ctrl != "rl":
            ov["policy_node_name"] = {
                "dwa": "dwa_planner",
                "pid": "path_following",
                "pid_vo": "path_following",
                "mppi": "mppi_planner_node",
            }.get(ctrl, "baseline_arm")
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
        ctrl = LaunchConfiguration("controller").perform(context)
        if not mppi_on or ctrl != "rl":
            return []   # MPPI 吃 policy 的 /rover_rl/cmd_vel_desired，非 rl controller 沒有這個來源
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
        ctrl = LaunchConfiguration("controller").perform(context)
        lv0 = LaunchConfiguration("log_level").perform(context)
        if ctrl == "pid_vo":
            # 消融實驗 PID+VO 組：VO 吃 path_following 的純路徑跟蹤輸出（動態避障），
            # 直接輸出到 mux；不套 RL 專屬的 mppi/recovery 相關 extra 覆寫。
            # ⚠ 這裡「無視 enable_vo」刻意為之：選 controller=pid_vo 本身就是在要 VO。
            #   若在此尊重 enable_vo:=false，path_following 已被 remap 去發
            #   /rover_rl/cmd_vel_baseline_desired、卻沒有 VO 接手轉發到 /input/nav_cmd_vel
            #   → 整條鏈斷掉、沒有任何節點發 cmd_vel，車完全不動且不報錯（靜默死路）。
            #   要純 PID 無避障請用 controller:=pid，不要用 pid_vo + enable_vo:=false。
            return [Node(
                package="rover_rl_inference",
                executable="vo_safety",
                name="vo_safety_node",
                output="screen",
                emulate_tty=True,
                parameters=[LaunchConfiguration("vo_params_file"), {
                    "topic_cmd_in": "/rover_rl/cmd_vel_baseline_desired",
                }],
                arguments=["--ros-args", "--log-level", lv0],
            )]
        if ctrl != "rl" or not vo_on:
            return []   # dwa / pid（純）自己不接 VO；rl 則照 enable_vo 決定
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
            LaunchConfiguration("enable_recovery"), "' != 'true' and '",
            controller, "' == 'rl'"
        ])),
    )

    # ── Part 11c: Recovery Supervisor（獨立 cmd_vel wrapper）──
    # 無 VO:
    #   policy → /rover_rl/cmd_vel_desired → [recovery_supervisor] → /input/nav_cmd_vel → mux
    # 有 VO:
    #   policy → VO → /rover_rl/cmd_vel_recovery_in → [recovery_supervisor] → /input/nav_cmd_vel
    def make_recovery_supervisor_node(context, *args, **kwargs):
        recovery_on = LaunchConfiguration("enable_recovery").perform(context).lower() == "true"
        ctrl = LaunchConfiguration("controller").perform(context)
        if not recovery_on or ctrl != "rl":
            return []   # Recovery 吃 policy 的 /rover_rl/cmd_vel_desired 鏈，非 rl controller 不適用
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

    # ── Part 11c-2: 算出 baseline 的速度上限（消融實驗公平性的核心）──
    # ⚠ 這段是「跟 RL 對齊」的唯一真值來源，dwa / pid / mppi / baseline_arm 四處共用。
    #
    # RL 的實體速度上限不是 yaml 的 act_max_*_velocity，而是：
    #     實體 v = act_max_linear_velocity  × speed_rate
    #     實體 ω = act_max_angular_velocity × speed_rate        (policy_node.py:117-118)
    # speed_rate 是「時間膨脹」，**同時**縮線速度與角速度，而且：
    #   · 每個 checkpoint 的 yaml 值都不同（0.35 ~ 1.0，實體 v 從 0.35 到 1.00 m/s）
    #   · deploy_rl_shell 啟動時還會再問一次 speed_rate，現場覆寫 yaml
    # → 把 baseline 硬編成「底盤上限 1.0/1.2」會讓它比 RL 快 43%~186%，
    #   「相同點位的路線與速度變化」這個比較直接失效。
    #
    # 用法（二選一）：
    #   align_rl_config:=sa4r2              ← 推薦：自動讀該 RL yaml 換算，不會手算錯
    #   align_rl_config:=sa4r2 align_speed_rate:=0.7   ← RL 啟動時現場改過 speed_rate 就補這個
    #   baseline_max_v:=0.7 baseline_max_w:=0.84       ← 直接給數字
    def resolve_baseline_limits(context, *args, **kwargs):
        if LaunchConfiguration("controller").perform(context) == "rl":
            return []
        v = float(LaunchConfiguration("baseline_max_v").perform(context))
        w = float(LaunchConfiguration("baseline_max_w").perform(context))
        align = LaunchConfiguration("align_rl_config").perform(context).strip()
        src = "baseline_max_v/w 直接指定"

        if align:
            # 接受 variant 名（sa4r2）、檔名（policy_params_sa4r2.yaml）或絕對路徑
            if os.path.isabs(align):
                path = align
            elif align.endswith(".yaml"):
                path = os.path.join(rl_pkg, "config", align)
            else:
                path = os.path.join(rl_pkg, "config", f"policy_params_{align}.yaml")
            if not os.path.isfile(path):
                return [LogInfo(msg=(
                    f"\n🔴 align_rl_config='{align}' 找不到對應檔案：{path}\n"
                    f"   速度上限無法對齊 RL，這次的比較數據不可用。請確認 variant 名稱。\n"
                ))]
            with open(path, encoding="utf-8") as fh:
                rp = list(yaml.safe_load(fh).values())[0]["ros__parameters"]
            act_v = float(rp.get("act_max_linear_velocity", 1.0))
            act_w = float(rp.get("act_max_angular_velocity", 1.2))
            rate = float(rp.get("speed_rate", 1.0))
            # speed_rate 也縮加速度（policy_node.py:1328/1332），所以 RL 的實體加速度
            # 同樣要 ×rate。這兩個數字**目前不會自動套到 baseline**（見下方 LogInfo 的
            # 提醒）：加速度限制在 dwa 是取樣範圍（改了不限制輸出）、在 mppi 才是真限制，
            # 硬壓會改變演算法本身的行為。先顯性印出落差，要不要對齊是論文方法論決策。
            acc_v = float(rp.get("act_max_linear_accel", 0.5)) * rate
            acc_w = float(rp.get("act_max_angular_accel", 3.0)) * rate
            override = LaunchConfiguration("align_speed_rate").perform(context).strip()
            if override:
                rate = float(override)
                src = f"對齊 {os.path.basename(path)}（speed_rate 由命令列覆寫為 {rate}）"
            else:
                src = f"對齊 {os.path.basename(path)}（yaml speed_rate={rate}）"
            v = act_v * rate
            w = act_w * rate

        return [
            SetLaunchConfiguration("baseline_v_final", f"{v:.4f}"),
            SetLaunchConfiguration("baseline_w_final", f"{w:.4f}"),
            SetLaunchConfiguration("baseline_v_neg", f"{-v:.4f}"),
            SetLaunchConfiguration("baseline_w_neg", f"{-w:.4f}"),
            LogInfo(msg=(
                f"\n[消融實驗] baseline 速度上限 v={v:.3f} m/s  ω={w:.3f} rad/s"
                f"\n            來源：{src}"
                f"\n            ⚠ RL 組必須跑在同一組上限（RL 實體上限 = act_max × speed_rate），"
                f"否則路線/速度比較無效"
                + (f"\n            ── 加速度（目前未對齊，僅供判讀）──"
                   f"\n            RL 實體加速度：線 {acc_v:.2f} m/s²  角 {acc_w:.2f} rad/s²"
                   f"\n            baseline：dwa 無輸出加速度限制（5.0 僅取樣範圍）／"
                   f"mppi 線 1.5 角 10.0／pid 內部步進"
                   f"\n            → baseline 起步會比 RL 猛，畫「速度變化」曲線時要一併說明"
                   if align else "")
                + "\n"
            )),
        ]
    baseline_limits = OpaqueFunction(function=resolve_baseline_limits)

    # ── Part 11d: 消融實驗 baseline 演算法（controller:=dwa|pid|pid_vo）──
    # 兩個節點都是既有、已在此車上跑過的 campusrover_move 演算法（非新寫），只是這裡
    # 加上「速度/角速度上限統一為底盤真實上限」的參數組，跟 RL 做公平比較。
    #
    # controller=dwa：真正的軌跡取樣 DWA，吃 /campusrover_local_costmap 做靜態避障，
    #   直接輸出 /input/nav_cmd_vel（需 enable_costmap:=true，預設已開）。
    dwa_baseline_node = Node(
        package="campusrover_move",
        executable="dwa_planner",
        name="dwa_planner",
        output="screen",
        remappings=[
            ("elevator_path", "/rover_rl/_baseline_unused_elevator"),
            ("global_path", "/global_path"),
            ("costmap", "/campusrover_local_costmap"),
            ("cmd_vel", "/input/nav_cmd_vel"),
            ("odom", "/odom"),
        ],
        parameters=[{
            "robot_frame": "base_link",
            "arriving_range_dis": 0.1,
            "arriving_range_angle": 0.05,
            "max_linear_acceleration": 5.0,   # 只影響 DWA 內部速度取樣範圍，非致動器限制
            "max_angular_acceleration": 5.0,
            # ⚠ 速度上限一律由 Part 11c-2 的 resolve_baseline_limits 算出（對齊 RL 的
            #   act_max × speed_rate），不要在這裡填死數字——RL 的實體上限隨 checkpoint
            #   與啟動時選的 speed_rate 變動（0.35~1.00 m/s），填死等於送 baseline 速度優勢。
            #   另注意：這四個值會在 arm 時被 planner_function_dwa 的 speed_parameter 覆寫，
            #   baseline_arm 吃的是同一組 LaunchConfiguration，兩邊保證一致。
            "max_linear_velocity": ParameterValue(LaunchConfiguration("baseline_v_final"), value_type=float),
            "min_linear_velocity": ParameterValue(LaunchConfiguration("baseline_v_neg"), value_type=float),
            "max_angular_velocity": ParameterValue(LaunchConfiguration("baseline_w_final"), value_type=float),
            "min_angular_velocity": ParameterValue(LaunchConfiguration("baseline_w_neg"), value_type=float),
            "target_point_dis": 3.0,
            "threshold_occupied": 2.0,
            # footprint_* 在 dwa_planner.cpp 裡宣告/讀取後未再使用（死參數，真正的靜態避障
            # 由下面 obstacle_max_dis/min_dis 配合 costmap cost weight 做軌跡評分達成）。
            # 仍填真實車身外框數字（見 pid_baseline_node 同款註解），純粹避免將來對照時混淆。
            "footprint_max_x": 0.8,
            "footprint_min_x": -0.40,
            "footprint_max_y": 0.40,
            "footprint_min_y": -0.40,
            "obstacle_max_dis": 3.0,
            "obstacle_min_dis": 0.3,
            "obstable_cost_weight": 1.5,
            "target_dis_weight": 1.0,
            "velocity_weight": 1.0,
            "trajectory_num": 10,
            "trajectory_point_num": 10,
            "simulation_time": 6.0,
            "target_bias": 0.1,
            "min_angle_of_linear_profile": 0.1,
            "max_angle_of_linear_profile": 0.8,
            "enable_linear_depend_angular": True,
            "enable_costmap_obstacle": True,
            "direction_inverse": False,
        }],
        condition=IfCondition(PythonExpression([
            "'", controller, "' == 'dwa'"
        ])),
    )

    # controller=pid / pid_vo：path_following，關掉內建 DWA/costmap 避障（純路徑跟蹤），
    #   pid   → 直接輸出 /input/nav_cmd_vel（無避障，論文原始對照組）
    #   pid_vo→ 輸出到 /rover_rl/cmd_vel_baseline_desired，交給上面的 vo_safety_node
    #           做動態避障後再送 /input/nav_cmd_vel
    pid_baseline_node = Node(
        package="campusrover_move",
        executable="path_following",
        name="path_following",
        output="screen",
        remappings=[
            ("elevator_path", "/rover_rl/_baseline_unused_elevator"),
            ("global_path", "/global_path"),
            ("costmap", "/campusrover_local_costmap"),
            ("odom", "/odom"),
            ("cmd_vel", PythonExpression([
                "'/rover_rl/cmd_vel_baseline_desired' if '", controller,
                "' == 'pid_vo' else '/input/nav_cmd_vel'"
            ])),
        ],
        parameters=[{
            "robot_frame": "base_link",
            "arriving_range_dis": 0.1,
            "arriving_range_angle": 0.05,
            # ⚠ 同 dwa：對齊 RL 的實體上限，由 resolve_baseline_limits 算出
            "max_linear_velocity": ParameterValue(LaunchConfiguration("baseline_v_final"), value_type=float),
            "max_angular_velocity": ParameterValue(LaunchConfiguration("baseline_w_final"), value_type=float),
            "target_point_dis": 0.6,
            "threshold_occupied": 2.0,
            # footprint_* 是 path_following.cpp CostmapCallback() 的「停車框」（障礙物落入即
            # obstacle_stop_cmd_=true），但目前 enable_costmap_obstacle=False 時該函式開頭就
            # return，此框連同 obstacle_detect_max_dis/min_dis 一併不執行（純路徑跟蹤、零避障，
            # 已讀原始碼確認）。數字仍對齊 RL 側真值（policy_params.yaml 訓練半徑 0.35、實際
            # 半寬 0.26、diag_logger/vo_params 的車體轉彎外接圓 0.40）：左右取 ±0.40（與 VO
            # r_robot 一致），前方 0.40 車身 + 0.42 煞停距離（v=1.0 / chassis acc_max=1.2 算得
            # v²/2a）≈0.8。只在未來把 enable_costmap_obstacle 打開做「PID+costmap 停車」變體
            # 時才會生效，先備著避免踩到舊的離譜數字。
            "footprint_max_x": 0.8,
            "footprint_min_x": -0.40,
            "footprint_max_y": 0.40,
            "footprint_min_y": -0.40,
            "speed_pid_k": 0.8,
            "min_angle_of_linear_profile": 0.1,
            "max_angle_of_linear_profile": 0.8,
            "obstacle_range": 0.3,
            "enable_linear_depend_angular": True,
            "enable_costmap_obstacle": False,        # 純路徑跟蹤，避障交給 VO 或不做
            "enable_dwa_obstacle_avoidance": False,   # 不掛內建 DWA 避障，保持「傳統 PID」對照組乾淨
            "enable_pullover_mode": False,
            "direction_inverse": False,
        }],
        condition=IfCondition(PythonExpression([
            "'", controller, "' in ('pid', 'pid_vo')"
        ])),
    )

    # controller=mppi：取樣式 MPC baseline（campusrover_move/mppi_planner 原生 path 模式）。
    #   ⚠ 與 Part 10b 的 static_guard 是同一支 executable、完全不同用法，別混淆：
    #       static_guard = 吃 RL 的 reference_cmd、不吃 global_path（RL 導航 + MPPI 只做靜態避障）
    #       baseline     = 吃 /global_path 自己導航，是獨立的傳統演算法對照組
    #   參數檔 mppi_baseline.yaml 沒設 static_guard_mode → C++ 預設 false（mppi_planner.cpp:138），
    #   即原生 path-following 模式：path_follow/path_align/goal/obstacle cost 全部生效。
    def make_mppi_baseline_node(context, *args, **kwargs):
        if LaunchConfiguration("controller").perform(context) != "mppi":
            return []
        lv = LaunchConfiguration("log_level").perform(context)
        cfg = LaunchConfiguration("mppi_baseline_params_file").perform(context)
        if not cfg:
            cfg = os.path.join(rl_pkg, "config", "mppi_baseline.yaml")
        return [Node(
            package="campusrover_move",
            executable="mppi_planner",
            name="mppi_planner_node",
            output="screen",
            emulate_tty=True,
            # yaml 之後疊一層速度覆寫：mppi 的 serviceCallback 不吃 speed_parameter
            # （dwa/pid 才吃），所以它的上限只能從 parameters 進來。
            # vx_min 保持 yaml 的 0.0（正常行進禁止倒車），只覆寫上限與角速度對稱範圍。
            parameters=[cfg, {
                "vx_max": ParameterValue(LaunchConfiguration("baseline_v_final"),
                                         value_type=float),
                "wz_max": ParameterValue(LaunchConfiguration("baseline_w_final"),
                                         value_type=float),
                "wz_min": ParameterValue(LaunchConfiguration("baseline_w_neg"),
                                         value_type=float),
            }],
            arguments=["--ros-args", "--log-level", lv],
            remappings=[
                ("global_path", "/global_path"),
                ("costmap", LaunchConfiguration("mppi_costmap_topic")),
                ("cmd_vel", "/input/nav_cmd_vel"),
                ("odom", "/odom"),
                ("elevator_path", "/rover_rl/_baseline_unused_elevator"),
                # LV-DOT 發的是 onboard_detector/DynamicObstacleArray，這裡要的是
                # campusrover_msgs/DynamicObstacleArray —— 型別不相容接不上，
                # 故 yaml 已關 dynamic critic，此處 remap 到空 topic（與 dwa 同條件：只吃 costmap）
                ("dynamic_obstacles", "/rover_rl/_baseline_unused_dynobs"),
                ("reference_cmd", "/rover_rl/_baseline_unused_refcmd"),
            ],
        )]
    mppi_baseline_node = OpaqueFunction(function=make_mppi_baseline_node)

    # ── Part 11e: baseline planner 的 arm 橋接（controller!=rl 才啟）──
    # 為什麼一定要有：dwa_planner / path_following / mppi_planner 三者都是
    # action_flag_=false 開機，控制迴圈開頭就 return（dwa:327 / pid:358 / mppi:414）
    # → 不呼叫 planner_function* service，車完全不動而且不印任何錯誤。
    # 且該 service 會覆寫 launch parameters 裡的速度上限與避障開關：
    #     max_linear_velocity_ = req->speed_parameter.linear.x;   (dwa:1179 / pid:1035)
    #     enable_costmap_obstacle_ = req->obstacle_avoidance.data;
    # → **消融實驗「統一速度上限 v=1.0 / ω=1.2」的真正生效點在這個節點，不是上面的
    #   parameters 區塊**（那份只在 service 呼叫前短暫生效）。兩邊數字必須一致。
    def make_baseline_arm_node(context, *args, **kwargs):
        ctrl = LaunchConfiguration("controller").perform(context)
        if ctrl == "rl":
            return []
        lv = LaunchConfiguration("log_level").perform(context)
        return [Node(
            package="rover_rl_inference",
            executable="baseline_arm",
            name="baseline_arm",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "controller": ctrl,
                # 與 dwa/pid/mppi 同一組數字（resolve_baseline_limits 算出）。
                # ⚠ 這裡才是 dwa/pid 真正生效的上限：service 的 speed_parameter 會覆寫
                #   planner 的 parameters，兩邊必須來自同一個來源，否則悄悄不一致。
                "max_linear_velocity": float(
                    LaunchConfiguration("baseline_v_final").perform(context)),
                "max_angular_velocity": float(
                    LaunchConfiguration("baseline_w_final").perform(context)),
                # dwa 要吃 costmap 做靜態避障；pid/pid_vo 刻意零避障（交給 VO 或不做）。
                # mppi 的 serviceCallback 不讀這欄，靜態避障由 yaml 的 obstacle critic 決定。
                "obstacle_avoidance": ctrl == "dwa",
            }],
            arguments=["--ros-args", "--log-level", lv],
        )]
    baseline_arm_node = OpaqueFunction(function=make_baseline_arm_node)

    # ── Part 12: 兩固定點往返避障測試（預設關，enable_pingpong:=true 開啟）──
    # 把車手動開到 A/B 任一點停穩 → 自動規劃往對向點，A↔B 無限來回，供反覆測避障。
    # 與 RL 推論/避障完全解耦：只訂閱 policy status、呼叫 routing、必要時切 mode=nav。
    # 消融實驗（controller!=rl）沒有 policy_node，姿態/mode 改走 TF+/odom
    # （require_policy_status:=false，見 pingpong_test_node.py 的 baseline 模式）
    def make_pingpong_test_node(context, *args, **kwargs):
        ctrl = LaunchConfiguration("controller").perform(context)
        lv = LaunchConfiguration("log_level").perform(context)
        return [Node(
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
                "require_policy_status": ctrl == "rl",
            }],
            arguments=["--ros-args", "--log-level", lv],
            condition=IfCondition(LaunchConfiguration("enable_pingpong")),
        )]
    pingpong_test_node = OpaqueFunction(function=make_pingpong_test_node)

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
        "  rover_rl 棧 (controller=rl 才啟):\n"
        "    [7] lidar_preprocessor\n"
        "    [8] policy_node\n"
        "    [9] bev_play\n"
        "    [10] LV-DOT 動態偵測 (/onboard_detector/*)\n"
        "  消融實驗 baseline (controller=dwa|pid|pid_vo|mppi 才啟):\n"
        "    [11] dwa_planner / path_following / mppi_planner → /input/nav_cmd_vel\n"
        "    [12] baseline_arm（arm 上面的 planner，缺它車不會動）\n"
        "  排除: DWA + AIT* 預設關（controller=dwa 時由此開回 DWA；"
        "AIT* 一律用 routing 取代）\n"
        "================================"
    ))

    return LaunchDescription([
        # ── args ──
        # 在此設定所有啟動參數的「預設值」與說明，可在命令列覆寫
        # 例：ros2 launch ... deploy_full.launch.py initial_mode:=idle enable_mot:=false
        DeclareLaunchArgument("controller", default_value="rl",
                              description="rl|dwa|pid|pid_vo|mppi — 消融實驗用，決定誰發 "
                                          "/input/nav_cmd_vel。rl=policy_node(可疊 mppi/vo/orca/"
                                          "recovery，現況預設)；dwa=campusrover_move/dwa_planner"
                                          "（軌跡取樣 DWA + costmap 靜態避障）；pid="
                                          "campusrover_move/path_following（純路徑跟蹤，無避障）；"
                                          "pid_vo=path_following + vo_safety_node（動態避障）；"
                                          "mppi=campusrover_move/mppi_planner 原生 path 模式"
                                          "（取樣式 MPC，3.5s 前瞻 + costmap 靜態避障，"
                                          "參數 mppi_baseline.yaml）。"
                                          "非 rl 時 lidar_preprocessor/policy_node/bev/mppi/orca/"
                                          "recovery 一律不啟，NDT/costmap/routing/LV-DOT/diag_logger/"
                                          "pingpong_test 照常共用。"),
        DeclareLaunchArgument("experiment_tag", default_value="",
                              description="診斷記錄 experiment_tag。留空自動帶 controller 值"
                                          "（rl/dwa/pid/pid_vo/mppi），可自訂加場景後綴如 "
                                          "dwa_fixed_obstacle"),
        DeclareLaunchArgument("model_path", default_value="",
                              description="覆寫 yaml model_path"),
        DeclareLaunchArgument("initial_mode", default_value="nav"),
        DeclareLaunchArgument("speed_rate", default_value="",
                              description="覆寫 policy yaml 的 speed_rate（時間膨脹式降速，"
                                          "rate<1 時感知量放大 1/rate、動作上限縮 ×rate）。"
                                          "有效 0.05~1.0，policy_node 會 clamp。空=走 yaml。"
                                          "deploy_rl_shell 選單會互動詢問（預設 0.6）"),
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
        DeclareLaunchArgument("align_rl_config", default_value="",
                              description="⭐消融實驗公平性：把 baseline 的速度上限自動對齊某組 RL "
                                          "設定。給 variant 名（sa4r2）、檔名"
                                          "（policy_params_sa4r2.yaml）或絕對路徑；"
                                          "換算 = act_max_*_velocity × speed_rate"
                                          "（policy_node.py:117-118，speed_rate 同時縮 v 與 ω）。"
                                          "留空則用 baseline_max_v/w 的值。"),
        DeclareLaunchArgument("align_speed_rate", default_value="",
                              description="覆寫 align_rl_config 讀到的 speed_rate。"
                                          "⚠ deploy_rl_shell 啟動 RL 時會現場問一次 speed_rate "
                                          "（Enter=0.6）並覆寫 yaml → RL 那次實際跑的值若與 yaml "
                                          "不同，這裡要填一樣的數字，否則兩組上限不一致。"),
        DeclareLaunchArgument("baseline_max_v", default_value="1.0",
                              description="baseline 線速度上限 (m/s)。⚠ 預設 1.0 是底盤真實上限，"
                                          "只有 RL 也跑在 speed_rate=1.0（如 sa1r1）時才等價；"
                                          "其他 checkpoint 請用 align_rl_config 自動換算。"),
        DeclareLaunchArgument("baseline_max_w", default_value="1.2",
                              description="baseline 角速度上限 (rad/s)。同上，預設是底盤真實上限。"),
        DeclareLaunchArgument("mppi_baseline_params_file", default_value="",
                              description="controller:=mppi 的參數檔。留空用 rover_rl_bringup/"
                                          "config/mppi_baseline.yaml（原生 path 模式，vx_max 對齊 "
                                          "1.0、dynamic critic 關）。⚠ 不要拿 mppi_static_guard.yaml "
                                          "餵這裡，那份是 RL 三層協作用的、不吃 global_path"),
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
        DeclareLaunchArgument("pingpong_auto_continue",
                              default_value=PythonExpression([
                                  "'false' if '", controller, "' == 'rl' else 'true'"
                              ]),
                              description="往返測試全自動模式：就緒後不必按空白鍵、停穩自動出發下一段"
                                          "（執行中可用 TUI 'a' 鍵熱切換）。"
                                          "⚠ controller!=rl 時預設 true：空白鍵是 status_tui 攔截後發的，"
                                          "而 baseline（dwa/pid/pid_vo）沒有 policy status 可餵 TUI、"
                                          "不會跑 status_tui → 沒人能按空白鍵，false 會永遠卡在「就緒」。"),
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

        # 消融實驗 baseline（controller:=dwa|pid|pid_vo|mppi 才啟，預設 rl 一個都不啟）
        # ⚠ baseline_limits 必須排在四個節點之前：它用 SetLaunchConfiguration 廣播
        #   算好的速度上限，下面四個都靠那組值（順序顛倒會拿到未定義的 configuration）。
        baseline_limits,
        dwa_baseline_node,
        pid_baseline_node,
        mppi_baseline_node,
        # ⚠ 上面三個 planner 都需要 arm 才會發 cmd_vel，缺這個節點車不動且不報錯
        baseline_arm_node,

        # 兩固定點往返避障測試（預設關，enable_pingpong:=true 開啟）
        pingpong_test_node,
    ])
