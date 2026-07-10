"""rover_rl 診斷記錄節點 — 獨立、不影響推論.

目的：每次「實驗」把 goal / agent 位置 / 速度 / NDT / policy 看到的 goal 方向 /
cmd_vel 連續記錄成 CSV（+可選 wandb），供事後分析「角速度晃動」「沒往 goal
前進」等問題。

設計（對齊 repo「獨立節點」哲學）：
  - 完全被動訂閱，不發任何 cmd，不碰 policy_node
  - **待命機制**：deploy_rl 啟動後不立刻記錄（避免錄到 NDT/RViz 收斂期雜訊）。
    在另一終端送 start → 開一個全新實驗（新 CSV + 新 wandb run，帶 label）。
        ros2 topic pub --once /rover_rl/record std_msgs/String "{data: 'start exp_name'}"
        ros2 topic pub --once /rover_rl/record std_msgs/String "{data: 'stop'}"
    （require_start=false 時退回舊行為：啟動即待錄，收到 goal 就寫）
  - 20 Hz timer 取最新快取 → 寫一列 CSV（時間序列，可畫圖）
  - 衍生指標：
      dist_to_goal        : robot(map) → goal(map) 距離
      heading_err_deg     : 真實「車頭朝向 vs goal 方向」誤差（NDT ground truth）
      policy_goal_ang_deg : policy obs 內的 goal body 角度（policy 實際在追的方向）
      → 兩者應一致；不一致 = 定位/TF/座標問題
  - wandb 可選；實車網路不穩建議 wandb_mode=offline，回頭 `wandb sync`
  - 開錄時抓 policy_node 全部參數（speed_rate / cmd_alpha_* 等）→ 寫
    `<csv>_params.json` sidecar + wandb run config（論文重現性 + 關聯晃動診斷）

訂閱：
  /odom /ndt_pose /goal_pose /global_path /input/nav_cmd_vel
  /rover_rl_policy/obs_debug  (79D obs，goal body 在 [4],[5])
  /rover_rl/lidar_sweep_72
  /rover_rl/record            (std_msgs/String) start/stop 控制
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from datetime import datetime

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Empty, Float32MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener

# 動態障礙追蹤（LV-DOT→vo_interface）；供碰撞判定用。vo_interface 未 build 時
# 缺此 msg，碰撞指標自動停用（不擋 diag 其餘功能）。
try:
    from vo_interface.msg import TrackedObstacleArray
    _HAS_TRACKED = True
except ImportError:
    TrackedObstacleArray = None
    _HAS_TRACKED = False


def _pv_to_py(pv):
    """rcl_interfaces ParameterValue → python（避免版本差異，手動轉）."""
    # pv.type 是整數列舉，按 rcl_interfaces ParameterType 對應取出正確欄位；
    # 不同 ROS 版本的 helper 介面有差異，故這裡直接查表轉，最穩
    t = pv.type
    return {
        1: lambda: pv.bool_value,
        2: lambda: pv.integer_value,
        3: lambda: pv.double_value,
        4: lambda: pv.string_value,
        5: lambda: list(pv.byte_array_value),
        6: lambda: list(pv.bool_array_value),
        7: lambda: list(pv.integer_array_value),
        8: lambda: list(pv.double_array_value),
        9: lambda: list(pv.string_array_value),
    }.get(t, lambda: None)()

# CSV 欄位順序（DictWriter 依此寫表頭）；analyze_diag 依賴這些欄名，勿隨意改名。
# *_age 欄記「該筆快取距現在幾秒」，事後可判斷某感測是否在掉訊。
CSV_FIELDS = [
    "t_wall", "t_rel",                                       # 牆鐘時間 / 相對開錄秒數
    "goal_seq", "has_goal",                                  # 第幾個 goal / 當下是否有 goal
    "odom_x", "odom_y", "odom_yaw_deg", "odom_v", "odom_w",  # 里程計位姿與實測速度
    "ndt_x", "ndt_y", "ndt_yaw_deg", "ndt_age",             # NDT /ndt_pose（此部署=map→odom 變換，非車姿！僅供活性判定）
    "map_x", "map_y", "map_yaw_deg", "pose_src",            # 車在 goal_frame 的真實位姿（TF goal_frame→base，與 policy 同源）；pose_src=tf/odom_only
    "goal_x", "goal_y", "goal_frame",
    "dist_to_goal", "heading_err_deg",                       # 衍生：到 goal 距離 / 車頭朝向誤差（用 map_* 算，非 /ndt_pose）
    "cmd_v", "cmd_w", "cmd_age",                             # 實際發出的 cmd_vel
    "policy_goal_bx", "policy_goal_by", "policy_goal_ang_deg",  # policy obs 內的 goal body 方向
    # policy 實際吃到的 goal 距離 = hypot(obs[4],obs[5])，已 clamp 18m + ×1/speed_rate。
    # 與 dist_to_goal（真實剩餘距離）對照：path 走 lookahead carrot ≈2m、single goal 給真實距離
    # → 兩者脫節時 = policy 感知距離 ≠ 真實距離（遠 single goal OOD 假設的核心證據）。
    "policy_goal_dist",
    "policy_v_norm", "policy_w_norm", "obs_age",
    "sweep_min_m", "sweep_age",                              # 72-bin sweep 還原成公尺後的最近障礙
    # ★07-06: 方向距離(來自 policy status; front=bins28:45 等) + 72束原始obs(分號串接,校準/前錐分析用)
    "front_m", "back_m", "left_m", "right_m", "obs_lidar72",
    # lidar_collision：sweep 最近障礙 < lidar_collision_m(param,預設0.5) → 1，與 TUI 碰撞閃紅同步。
    # ⚠ 這是「raw LiDAR 任一方向過近」旗標（含牆），與下方論文用 collision(VO 動態障礙中心距)不同，
    #   刻意分開避免走廊牆面污染 CR 指標。要看實際接觸仍以 collision / 人工 collision_mark 為準。
    "lidar_collision",
    # 三層速度 + 延遲（來自 /rover_rl_policy/status）：rl=RL意圖, sent=濾波後送出, act=odom實測
    "rl_v", "rl_w", "sent_v", "sent_w", "act_v", "act_w",
    "v_over", "w_over", "lag_ms", "lag_corr", "lag_ch",      # 飽和旗標 / 延遲估計值
    # VO 安全層（來自 /vo_safety_node/status）：本次有沒有走 VO + 本拍 VO 有沒有在動手。
    # vo_active=0 整段 → 這次 policy 沒接 VO（enable_vo:=false），cmd 即 RL 直出；
    # vo_active=1 但 vo_intervening=0 → VO 在線但只放行+slew（前方無障礙時的起步 sin 波屬此類）。
    # driver = 本拍「誰在動手」的單欄判定（給 VO 介入距離校準用）：
    #   rl=RL 主導（沒走 VO / VO 只放行）、vo=VO 改寫 cmd 繞行、vo_stop=VO 全堵死煞停。
    # vo_engage_range = 當下 VO 設定的 engage_range（整段常數，供 analyze 對照實測介入距離）。
    "vo_active", "vo_intervening", "vo_blocked", "driver", "vo_engage_range",
    # VO 量化欄（逼近過程診斷用）：vo_n_obs=engage_range 內障礙數（0→VO 沒看到/沒介入）、
    # vo_des_*=RL 給 VO 的期望、vo_out_*=VO slew 後實際送 mux 的、vo_engaged=有障進範圍、
    # vo_n_feasible=可行候選數、vo_min_ttc=選定候選最近碰撞時間。
    # 看「逼近時 des_v 與 out_v 怎麼變」可判 VO 是逐步減速還是最後一秒才反應。
    "vo_engaged", "vo_n_obs", "vo_n_feasible",
    "vo_des_v", "vo_des_w", "vo_out_v", "vo_out_w", "vo_min_ttc",
    # 此次 goal 的決定方式：single_goal=RViz 2D Goal Pose 單點(/goal_pose)；
    # routing_path=RViz Publish Point 兩點選起點/終點 → routing → /global_path。
    "nav_type",
    # 碰撞判定（SR/TO/CR 論文指標）：以 odom frame 車姿 vs vo_interface 追蹤動態障礙。
    # min_sep_m = min(中心距 − 障礙等效半徑 − robot_radius_collision)，即「車體表面↔人表面」最近間隙；
    #   ≤ collision_margin 即判本拍接觸。n_dyn_obs=追蹤到的動態障礙數；collision=本拍是否接觸(0/1)。
    # dyn_obs_min_m = 最近動態障礙的「車中心↔障礙中心」距離（不扣半徑、不受 engage_range 過濾，
    #   任何距離都記）＝與 VO engage_range 同一量測基準 → 對照 driver 欄即可校準介入距離。
    # ⚠ 三者同在 odom frame → 皆為相對量測，對 NDT/odom 全域漂移免疫（不受定位誤差影響）。
    "min_sep_m", "dyn_obs_min_m", "n_dyn_obs", "collision",
]


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    # 只取 yaw（繞 z）：機器人在平面上移動，roll/pitch 不需要
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_pi(a: float) -> float:
    # 角度差包回 (-π, π]，避免 heading_err 出現 ±360° 的假大值
    return math.atan2(math.sin(a), math.cos(a))


def _safe_label(s: str) -> str:
    # 實驗名會直接拿來當資料夾/檔名，清掉空白與特殊字元防路徑注入/非法檔名，截 48 字
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_\-]", "", s)[:48]


class DiagLoggerNode(Node):
    def __init__(self):
        super().__init__("rover_rl_diag_logger")
        self._declare_params()
        self._read_params()
        self._init_state()
        self._init_pubsub()
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._tick)

        if not self.require_start:
            self.start_experiment("")        # 啟動即待命（收到第一個 goal 才建資料夾）
            ready = "待命中 — 收到第一個 goal 自動建資料夾開錄"
        else:
            ready = ("待命中 — 在另一終端送 start 才開錄：\n"
                     "    ros2 topic pub --once /rover_rl/record std_msgs/String "
                     "\"{data: 'start 實驗名'}\"")
        self.get_logger().info(
            f"診斷記錄節點啟動  rate={self.rate_hz}Hz  "
            f"wandb={'on(' + self.wandb_mode + ')' if self.enable_wandb else 'off'}\n"
            f"  log_dir={self.log_dir}\n  {ready}"
        )

    # ── params ──
    def _declare_params(self) -> None:
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("log_dir", os.path.expanduser("~/rover_rl/logs/diag"))
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_ndt_pose", "/ndt_pose")
        self.declare_parameter("topic_goal_pose", "/goal_pose")
        self.declare_parameter("topic_global_path", "/global_path")
        self.declare_parameter("topic_cmd_vel", "/input/nav_cmd_vel")
        self.declare_parameter("topic_obs_debug", "/rover_rl_policy/obs_debug")
        self.declare_parameter("topic_lidar_sweep", "/rover_rl/lidar_sweep_72")
        self.declare_parameter("topic_record_ctrl", "/rover_rl/record")
        self.declare_parameter("topic_status", "/rover_rl_policy/status")
        self.declare_parameter("topic_vo_status", "/vo_safety_node/status")
        self.declare_parameter("topic_diag_event", "/rover_rl/diag_event")
        # 開錄時抓此節點的全部參數（speed_rate 等）寫進 sidecar + wandb config
        self.declare_parameter("policy_node_name", "rover_rl_policy")
        self.declare_parameter("lidar_r_max_m", 20.0)
        self.declare_parameter("robot_radius_m", 0.35)
        # === 碰撞判定（SR/TO/CR 論文指標）===
        # 追蹤動態障礙來源（odom frame，與 /odom 同 frame → min_sep 為相對量測、免疫全域漂移）。
        self.declare_parameter("topic_tracked_obstacles", "/vo_interface/tracked_obstacles")
        # 手動 ground-truth 標註：安全員看到接觸/急閃時發 Empty → 強制本段記為 collision（論文權威標籤）。
        self.declare_parameter("topic_collision_mark", "/rover_rl/collision_mark")
        # ⚠ 碰撞用車體半徑（車體最大外緣，實測 0.40）— 與上面 robot_radius_m(0.35，sweep 還原用)無關，勿混用。
        self.declare_parameter("robot_radius_collision_m", 0.40)
        # 接觸門檻：min_sep ≤ 此值即本拍接觸。0=表面相接觸；設負值可要求更深侵入才算（更嚴）。
        self.declare_parameter("collision_margin_m", 0.0)
        # raw LiDAR 過近旗標門檻(m)：sweep 最近障礙 < 此值 → lidar_collision=1（與 TUI 同步，含牆）。
        self.declare_parameter("lidar_collision_m", 0.5)
        # 連續 N tick 接觸才算一次確認碰撞事件（濾單幀雜訊/假框）。20Hz × 2 = 0.1s。
        self.declare_parameter("collision_persist_ticks", 2)
        # 動態障礙 track 最短年齡(秒)：太年輕（速度未收斂）的 track 不納入碰撞判定，濾假框。0=全納入。
        self.declare_parameter("collision_min_track_age_s", 0.0)
        # 每段（leg）超時上限(秒)：t_rel 超過仍未到 goal → 判 timeout(TO)。<=0 關閉超時判定。
        self.declare_parameter("episode_timeout_s", 60.0)
        # 實驗條件標籤：區分不同部署模型/情境（如 "noise_dr" vs "baseline"），寫進 session 指標
        # 每列 + 每段 wandb config，供 pingpong_report 分組對比。可 launch 帶或 ros2 param set 現場改。
        self.declare_parameter("experiment_tag", "")
        # 開錄前輸入新鮮度門檻：任一輸入 *_age 超過此值 → 紅字警告「記錄可能被阻塞」
        self.declare_parameter("stale_warn_s", 0.5)
        # 車姿來源 frame：與 policy_node 同名參數一致（dist/heading 走 TF goal_frame→base）
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        # false(預設)=啟動即待錄，發 goal 即開始；true=需送 record start 才開錄
        self.declare_parameter("require_start", False)
        # true(預設)=收到 goal 後還要等「真正有速度輸出」才開檔+wandb（避免發了 goal 但車
        #            沒動就佔一個空 run）；false=收到 goal 立即開檔（舊行為）
        self.declare_parameter("start_on_motion", True)
        # 判定「有速度輸出」的門檻（m/s 或 rad/s）：|cmd_v| 或 |cmd_w| 超過即算啟動
        self.declare_parameter("start_motion_eps", 0.03)
        # 到達 goal 自動停止：距離低於此值連續 N tick → stop_experiment
        self.declare_parameter("auto_stop_goal_tol", 0.6)
        self.declare_parameter("auto_stop_goal_ticks", 10)   # 20Hz × 10 = 0.5s
        # 到 goal 自動停後是否自動 re-arm（下一個 goal/path 進來開新一段紀錄）
        self.declare_parameter("auto_rearm", True)
        # 判定「新目標 vs 同目標重複發布」的位移門檻（公尺）。
        # 用途①：path 由 routing_to_path 2Hz republish 同一條 → 終點不變則不重置 close_ticks、不 ++seq
        #         （否則 auto-stop 的連續 tick 計數會被 2Hz 回呼一直歸零，終點停觸發不了）
        # 用途②：re-arm 後舊 path 仍 2Hz 殘留 → 終點與「剛完成的 goal」相同則忽略，避免原地秒重開迴圈
        self.declare_parameter("goal_change_eps_m", 0.8)
        # 只在有 goal 時寫列（省空間）；false=start 後一直寫
        self.declare_parameter("log_only_with_goal", True)
        self.declare_parameter("enable_wandb", False)
        self.declare_parameter("wandb_project", "rover_rl_deploy")
        # online / offline / disabled；實車建議 offline
        self.declare_parameter("wandb_mode", "offline")

    def _read_params(self) -> None:
        gp = self.get_parameter
        sv = lambda k: gp(k).get_parameter_value().string_value
        self.rate_hz = float(gp("rate_hz").value)
        self.log_dir = sv("log_dir")
        self.topic_odom = sv("topic_odom")
        self.topic_ndt = sv("topic_ndt_pose")
        self.topic_goal = sv("topic_goal_pose")
        self.topic_path = sv("topic_global_path")
        self.topic_cmd = sv("topic_cmd_vel")
        self.topic_obs = sv("topic_obs_debug")
        self.topic_sweep = sv("topic_lidar_sweep")
        self.topic_ctrl = sv("topic_record_ctrl")
        self.topic_status = sv("topic_status")
        self.topic_vo_status = sv("topic_vo_status")
        self.topic_diag_event = sv("topic_diag_event")
        self.policy_node_name = sv("policy_node_name")
        self.r_max = float(gp("lidar_r_max_m").value)
        self.r_robot = float(gp("robot_radius_m").value)
        self.topic_tracked = sv("topic_tracked_obstacles")
        self.topic_collision_mark = sv("topic_collision_mark")
        self.r_robot_collision = float(gp("robot_radius_collision_m").value)
        self.collision_margin = float(gp("collision_margin_m").value)
        self.lidar_collision_m = float(gp("lidar_collision_m").value)
        self.collision_persist_ticks = int(gp("collision_persist_ticks").value)
        self.collision_min_track_age = float(gp("collision_min_track_age_s").value)
        self.episode_timeout_s = float(gp("episode_timeout_s").value)
        self.experiment_tag = sv("experiment_tag")
        self.stale_warn_s = float(gp("stale_warn_s").value)
        self.goal_frame = sv("goal_frame") or "map"
        self.base_frame = sv("base_frame") or "base_footprint"
        self.require_start = bool(gp("require_start").value)
        self.start_on_motion = bool(gp("start_on_motion").value)
        self.start_motion_eps = float(gp("start_motion_eps").value)
        self.auto_stop_goal_tol = float(gp("auto_stop_goal_tol").value)
        self.auto_stop_goal_ticks = int(gp("auto_stop_goal_ticks").value)
        self.auto_rearm = bool(gp("auto_rearm").value)
        self.goal_change_eps_m = float(gp("goal_change_eps_m").value)
        self.log_only_with_goal = bool(gp("log_only_with_goal").value)
        self.enable_wandb = bool(gp("enable_wandb").value)
        self.wandb_project = sv("wandb_project")
        self.wandb_mode = sv("wandb_mode") or "offline"

    # ── state ──
    def _init_state(self) -> None:
        self._t0 = time.monotonic()
        self._odom = None          # (x, y, yaw, v, w, t)
        self._ndt = None           # (x, y, yaw, t)
        self._goal = None          # (x, y, frame)
        self._goal_method = ""     # 此次 goal 決定方式：single_goal / routing_path（見 CSV nav_type）
        self._goal_seq = 0
        self._last_done_goal = None  # (x, y) 剛 auto-stop 完成的終點，用來擋 stale path 殘留發布
        self._cmd = None
        self._obs = None
        self._sweep_min = None
        self._status = None        # dict（三層速度 + 延遲）
        self._status_t = 0.0       # 最近一次收到 policy status 的 monotonic 時間（開錄新鮮度檢查用）
        self._vo = None            # dict（VO 安全層 status；None=從未收到）
        self._vo_t = 0.0           # 最近一次收到 VO status 的 monotonic 時間
        self._vo_seen_run = False  # 本段紀錄期間是否曾收到 VO status（→ params.json vo_enabled）
        self._model_name = None    # 本次使用的 model .ts 檔名（從 status 的 "model" 欄即時抓）
        self._goal_close_ticks = 0  # dist_to_goal < tol 連續 tick 計數
        self._all_csv_paths = []   # 本次 session 建立的所有 CSV（finalize 時全部列出）
        # 碰撞判定：追蹤障礙快取 + 本 session SR/TO/CR 統計
        self._obstacles = []       # list[(x, y, radius, age)]（odom frame）
        self._obstacles_t = 0.0
        self._session_metrics_path = None   # pingpong_metrics_<session>.csv（每段一列 + 累計 SR/TO/CR）
        self._sess = {"success": 0, "timeout": 0, "collision": 0}
        # 實驗 / 檔案
        self._started = False
        self._armed = False        # 已待命但尚未建資料夾（等第一個 goal）
        self._pending_motion = False  # 已收 goal、等待真正速度輸出才開檔（start_on_motion）
        self._exp_label = ""
        self._fh = None
        self._writer = None
        self.run_dir = None
        self.csv_path = None
        self.wandb_run = None
        # policy 參數擷取（用底層 rcl_interfaces service client）
        self._cli_list = self.create_client(
            ListParameters, f"/{self.policy_node_name}/list_parameters")
        self._cli_get = self.create_client(
            GetParameters, f"/{self.policy_node_name}/get_parameters")
        self._params_timer = None
        self._params_attempts = 0
        self._policy_params = {}
        self._reset_run_stats()

    def _reset_run_stats(self) -> None:
        self._n_rows = 0
        self._vo_seen_run = False   # 每段重置：是否走 VO 是 per-run 屬性
        self._cmd_w_prev = None
        self._dw_sq = 0.0
        self._dw_n = 0
        self._cmd_w_abs_sum = 0.0
        self._heading_abs_sum = 0.0
        self._heading_n = 0
        # 碰撞（per-leg）：本段最近間隙、接觸連續計數、確認事件數、是否曾碰撞、是否人工標註
        self._min_sep_run = float("inf")
        self._min_sweep_run = float("inf")   # 本段最近障礙(含牆，來自 72-bin sweep)
        self._contact_ticks = 0
        self._collision_events = 0
        self._collided = False
        self._collided_manual = False
        self._path_len = 0.0        # 里程積分（leg 路徑長度，公尺）
        self._prev_odom_xy = None

    def _init_pubsub(self) -> None:
        # sweep 是高頻 sensor 流，用 BEST_EFFORT 匹配 preprocessor 端 QoS（RELIABLE 會對不上收不到）
        be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        )
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)
        self.create_subscription(PoseStamped, self.topic_ndt, self._cb_ndt, 10)
        self.create_subscription(PoseStamped, self.topic_goal, self._cb_goal, 10)
        self.create_subscription(NavPath, self.topic_path, self._cb_path, 10)
        self.create_subscription(Twist, self.topic_cmd, self._cb_cmd, 10)
        self.create_subscription(Float32MultiArray, self.topic_obs, self._cb_obs, 10)
        self.create_subscription(Float32MultiArray, self.topic_sweep, self._cb_sweep, be)
        self.create_subscription(String, self.topic_ctrl, self._cb_ctrl, 10)
        self.create_subscription(String, self.topic_status, self._cb_status, 10)
        # VO 安全層 status（純觀察）：node 沒起 → 整段收不到 → vo_active 恆 0 = 本次沒走 VO
        self.create_subscription(String, self.topic_vo_status, self._cb_vo_status, 10)
        # 動態障礙追蹤（碰撞判定）：與 /odom 同 odom frame。vo_interface 未 build → 停用、不訂閱。
        if _HAS_TRACKED:
            self.create_subscription(
                TrackedObstacleArray, self.topic_tracked, self._cb_obstacles, 10)
        else:
            self.get_logger().warn(
                "vo_interface 未 build（缺 TrackedObstacleArray）→ 碰撞指標停用；"
                "min_sep_m/collision 欄將留空。build vo_interface 後即啟用。")
        # 手動 ground-truth 碰撞標註（安全員按鍵）：強制本段記為 collision
        self.create_subscription(Empty, self.topic_collision_mark, self._cb_collision_mark, 10)
        self._pub_event = self.create_publisher(String, self.topic_diag_event, 10)
        # TF：以 goal_frame→base_frame 取車在 map 的真實位姿（與 policy_node 同一條鏈，
        # tf2 正確合成 map→odom(NDT)∘odom→base）。不可用 /ndt_pose 當車姿（它是 map→odom）。
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _robot_pose_in_map(self, frame: str):
        """車在 `frame` 的真實位姿，取自 TF frame→base_frame（與 policy_node 同源）。
        回傳 (x, y, yaw, "tf")；查不到 TF 回 (None, None, None, "")。
        用 Time()（最新可用）避免跨機時鐘不同步造成 extrapolation 失敗。"""
        try:
            tf = self._tf_buffer.lookup_transform(frame, self.base_frame, Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            return t.x, t.y, _yaw_from_quat(q.x, q.y, q.z, q.w), "tf"
        except TransformException:
            return None, None, None, ""

    # ── 實驗開關 ──
    def start_experiment(self, label: str) -> None:
        # 一個「實驗」= 一個獨立資料夾 + 一份 CSV(+wandb run)；重新 start 先收掉上一個
        if self._started or self._armed:
            self.stop_experiment()      # 先收掉上一個
        self._exp_label = label
        self._reset_run_stats()
        self._goal_seq = 0
        self._goal = None
        self._goal_method = ""
        self._pending_motion = False
        self._last_done_goal = None
        if self.log_only_with_goal:
            # 待命：收到第一個 goal 才建資料夾 + 開檔（避免沒走 goal 留空資料夾）
            self._armed = True
            self.get_logger().info("▶ 待命中 — 收到第一個 goal 才建立紀錄資料夾")
        else:
            self._open_run_files(label)   # 不挑 goal → 立即開檔

    def _check_input_freshness(self) -> None:
        """開錄前檢查各輸入快取新鮮度：任一 *_age > stale_warn_s → 紅字（error）警告。
        抓兩種會讓整段 CSV 報廢的情境：①開錄當下 topic 已掉線（odom/sweep/preprocessor 死）；
        ②上一拍有東西卡住單執行緒 executor（如 wandb online 阻塞）讓所有訂閱一起 stale。
        純診斷提示，不擋開錄（仍照常記錄，只是讓你當場知道這段恐不可用）。"""
        now = time.monotonic()
        thr = self.stale_warn_s
        checks = (
            ("odom", self._odom[5] if self._odom else None),
            ("cmd", self._cmd[2] if self._cmd else None),
            ("status", self._status_t if self._status is not None else None),
            ("sweep", self._sweep_min[1] if self._sweep_min else None),
            ("obs", self._obs[1] if self._obs else None),
            ("ndt", self._ndt[3] if self._ndt else None),
        )
        bad = []
        for name, t in checks:
            if t is None:
                bad.append(f"{name}=未收到")
            elif (age := now - t) > thr:
                bad.append(f"{name}={age:.2f}s")
        if bad:
            # logger.error → rcl console 紅字
            self.get_logger().error(
                f"⚠ 記錄可能被阻塞：開錄時輸入不新鮮（>{thr:.1f}s）→ {', '.join(bad)}；"
                "這段 CSV 恐只寫得出零星 stale 列。查 odom hz / preprocessor / "
                "wandb 是否 online 阻塞 executor"
            )
        else:
            self.get_logger().info("✓ 開錄輸入新鮮度 OK（odom/cmd/status/sweep/obs/ndt 皆 <門檻）")

    def _open_run_files(self, label: str) -> None:
        """真正建立 run 資料夾並開 CSV（由 start 或第一個 goal 觸發）."""
        self._check_input_freshness()   # 開錄前先確認輸入沒 stale（阻塞/掉線即紅字示警）
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lab = _safe_label(label)
        name = f"diag_{stamp}" + (f"_{lab}" if lab else "")
        # 每次實驗開一個獨立資料夾（以日期時間[_實驗名]命名），
        # 該次所有檔案（CSV / params.json / 未來其他）都放裡面
        self.run_dir = os.path.join(self.log_dir, name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, name + ".csv")
        self._all_csv_paths.append(self.csv_path)
        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()
        self._t0 = time.monotonic()
        self._init_wandb(name)
        self._armed = False
        self._started = True
        self.get_logger().info(
            f"▶ 開始實驗 '{name}' → {self.csv_path}\n"
            f"  使用 model：{self._model_name or '（尚未收到 policy status，稍後補記）'}"
        )
        self._schedule_param_capture()

    # ── 擷取 policy_node 參數（speed_rate 等）→ sidecar json + wandb config ──
    def _schedule_param_capture(self) -> None:
        if self._params_timer is not None:
            self._params_timer.cancel()
        self._params_attempts = 0
        self._policy_params = {}
        self._params_csv_path = self.csv_path     # 綁定本次實驗
        self._params_timer = self.create_timer(1.0, self._try_capture_params)

    def _try_capture_params(self) -> None:
        # policy_node 可能比 diag_logger 晚起，故用 1Hz timer 輪詢 service，最多重試 15 次
        self._params_attempts += 1
        if self._params_attempts > 15:
            self.get_logger().warn(
                f"擷取 {self.policy_node_name} 參數逾時（節點未起？）— CSV 仍正常記錄"
            )
            self._params_timer.cancel()
            return
        if not (self._cli_list.service_is_ready() and self._cli_get.service_is_ready()):
            return                                # 下個 tick 再試
        self._params_timer.cancel()
        self._cli_list.call_async(
            ListParameters.Request()).add_done_callback(self._on_list_params)

    def _on_list_params(self, future) -> None:
        try:
            names = list(future.result().result.names)
        except Exception as e:
            self.get_logger().warn(f"list_parameters 失敗：{e}")
            return
        if not names:
            return
        req = GetParameters.Request()
        req.names = names
        self._cli_get.call_async(req).add_done_callback(
            lambda f: self._on_get_params(f, names)
        )

    def _on_get_params(self, future, names) -> None:
        try:
            values = future.result().values
        except Exception as e:
            self.get_logger().warn(f"get_parameters 失敗：{e}")
            return
        params = {n: _pv_to_py(v) for n, v in zip(names, values)}
        self._policy_params = params
        # sidecar json（與 CSV 同名 _params.json）：論文重現性 + 把當下 speed_rate/cmd_alpha
        # 等參數和這次晃動資料綁在一起，事後才能對照「這組參數造成的行為」
        meta = {
            "csv": os.path.basename(self._params_csv_path or ""),
            "policy_node": self.policy_node_name,
            "captured_wall": time.time(),
            # 本次使用的 model：優先 status 的即時檔名，退回 params 的 model_path
            "model": self._model_name or params.get("model_path"),
            # 本次 policy 是否走 VO 安全層（截至擷取參數當下是否已收到 VO status）
            "vo_enabled": bool(self._vo_seen_run),
            # 此次 goal 的決定方式：single_goal=2D Goal Pose 單點；routing_path=Publish Point 兩點 routing
            "goal_method": self._goal_method or None,
            "params": params,
        }
        try:
            side = (self._params_csv_path or "").rsplit(".", 1)[0] + "_params.json"
            with open(side, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(f"寫 params sidecar 失敗：{e}")
        # wandb config：除 policy 參數外，補推 meta 級欄位（vo_enabled / goal_method / model），
        # 與 sidecar 對齊，確保 wandb 端也完整可查本 run 的 goal 決定方式與是否走 VO。
        if self.wandb_run is not None:
            try:
                self.wandb_run.config.update(params, allow_val_change=True)
                self.wandb_run.config.update(
                    {k: meta[k] for k in ("vo_enabled", "goal_method", "model")},
                    allow_val_change=True)
            except Exception as e:
                self.get_logger().warn(f"wandb config 更新失敗：{e}")
        key = {k: params.get(k) for k in (
            "speed_rate", "cmd_alpha_linear", "cmd_alpha_angular",
            "cmd_max_accel_angular", "act_max_angular_velocity", "control_dt",
        ) if k in params}
        self.get_logger().info(
            f"已擷取 {self.policy_node_name} 參數 {len(params)} 項 → sidecar/wandb；"
            f"關鍵：{key}"
        )

    def _init_wandb(self, run_name: str) -> None:
        self.wandb_run = None
        if not self.enable_wandb or self.wandb_mode == "disabled":
            return
        try:
            os.environ.setdefault("WANDB_MODE", self.wandb_mode)
            import wandb
            self.wandb = wandb
            self.wandb_run = wandb.init(
                project=self.wandb_project, name=run_name,
                mode=self.wandb_mode,
                # goal_method 此時已由 _handle_target 設好（goal/path 觸發開檔），放進 config
                # 確保 wandb 端也記得本 run 是 2D Goal Pose 還是 routing（時間序列無法存字串）。
                # model 補記；vo_enabled 在 param 擷取後由 config.update 補（見 _on_get_params）。
                config={"rate_hz": self.rate_hz,
                        "goal_method": self._goal_method or None,
                        "model": self._model_name,
                        "experiment_tag": self.experiment_tag or None},
            )
        except Exception as e:
            self.get_logger().warn(f"wandb 初始化失敗，改用純 CSV：{e}")
            self.wandb_run = None

    def stop_experiment(self) -> None:
        if self._armed and not self._started:
            self._armed = False
            self._pending_motion = False
            self.get_logger().info("⏹ 取消待命（未開始紀錄，無資料夾產生）")
            return
        if not self._started:
            return
        self._print_summary()
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass
            self.wandb_run = None
        self._started = False

    def _rearm(self) -> None:
        """auto-stop 後重新待命：清掉上一段狀態，等下一個 goal/path 自動開新一段。"""
        self._reset_run_stats()
        self._goal_seq = 0
        self._goal = None
        self._goal_method = ""
        self._pending_motion = False
        self._goal_close_ticks = 0
        self._armed = True
        self.get_logger().info(
            "▶ 已重新待命 — 下一個 goal/path 進來自動開新一段紀錄（auto_rearm）"
        )

    def _cb_ctrl(self, msg: String) -> None:
        parts = msg.data.strip().split(None, 1)
        if not parts:
            return
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "start":
            self.start_experiment(arg)
        elif cmd == "stop":
            if self._started:
                self.get_logger().info("⏹ 收到 stop")
                self.stop_experiment()
            else:
                self.get_logger().info("收到 stop 但目前沒有進行中的實驗")
        else:
            self.get_logger().warn(f"未知 record 指令: {msg.data!r}（用 start / stop）")

    def _cb_status(self, msg: String) -> None:
        try:
            self._status = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._status_t = time.monotonic()
        # 即時抓本次使用的 model 檔名（status 5Hz 發、比 param service 更早可用）
        m = self._status.get("model")
        if m:
            self._model_name = m

    def _cb_vo_status(self, msg: String) -> None:
        # VO 安全層 5Hz status；收到即代表 VO 在線。只在記錄中標記本段「走過 VO」。
        try:
            self._vo = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._vo_t = time.monotonic()
        if self._started:
            self._vo_seen_run = True

    def _cb_obstacles(self, msg) -> None:
        # 動態障礙（odom frame）：存中心 x,y、等效半徑、track 年齡，供 _tick 算 min_sep。
        self._obstacles = [
            (float(ob.position.x), float(ob.position.y),
             float(ob.radius), float(ob.age))
            for ob in msg.obstacles
        ]
        self._obstacles_t = time.monotonic()

    def _cb_collision_mark(self, _msg: Empty) -> None:
        # 安全員人工標註：本段記為碰撞（論文權威 ground truth，凌駕自動判定）
        if self._started:
            self._collided = True
            self._collided_manual = True
            self.get_logger().warn("🚨 收到人工碰撞標註 → 本段將記為 collision")
        else:
            self.get_logger().info("收到碰撞標註但目前無進行中紀錄，忽略")

    # ── callbacks ──
    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._odom = (
            p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w),
            msg.twist.twist.linear.x, msg.twist.twist.angular.z, time.monotonic(),
        )

    def _cb_ndt(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self._ndt = (
            msg.pose.position.x, msg.pose.position.y,
            _yaw_from_quat(q.x, q.y, q.z, q.w), time.monotonic(),
        )

    def _cb_goal(self, msg: PoseStamped) -> None:
        frame = msg.header.frame_id or "map"
        self._handle_target(msg.pose.position.x, msg.pose.position.y, frame, "goal", "")

    def _cb_path(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        last = msg.poses[-1].pose.position
        frame = msg.header.frame_id or "map"
        # path 的「目標」= 終點（poses[-1]）；中途 waypoint 由 policy 的 SubgoalSelector 處理，
        # diag 只關心「到終點才停」
        self._handle_target(last.x, last.y, frame, "path", f" {len(msg.poses)} pts")

    @staticmethod
    def _xy_dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _handle_target(self, x: float, y: float, frame: str, kind: str, extra: str) -> None:
        """goal_pose / global_path 的共用入口：判斷是新目標還是同目標重複發布，
        並在 armed / started / 已停 三種狀態下做對的事。"""
        # 記下此次 goal 的決定方式（kind: goal=2D Goal Pose 單點, path=Publish Point 兩點 routing）
        self._goal_method = "single_goal" if kind == "goal" else "routing_path"
        new_xy = (x, y)
        if self._started:
            # 進行中：path 2Hz republish 同終點 → 只刷新座標，不重置 close_ticks、不 ++seq
            # （否則 auto-stop 連續 tick 永遠被歸零，終點停觸發不了）
            if self._goal is not None and self._xy_dist(new_xy, self._goal[:2]) <= self.goal_change_eps_m:
                self._goal = (x, y, frame)
                return
            # 終點明顯改變（中途重新路由）→ 視為同一段內換目標
            self._goal = (x, y, frame)
            self._goal_seq += 1
            self._goal_close_ticks = 0
            self.get_logger().info(
                f"[{kind} #{self._goal_seq}] 換新目標 ({x:.2f},{y:.2f}) frame={frame}{extra}"
            )
            return
        if self._armed:
            # 待命中：忽略「剛 auto-stop 完成的同一終點」殘留發布（stale path 2Hz），避免秒重開迴圈
            if (self._last_done_goal is not None
                    and self._xy_dist(new_xy, self._last_done_goal) <= self.goal_change_eps_m):
                return
            # 已在等速度輸出（pending）：同終點 2Hz republish 只刷新座標，不重印 log
            if (self._pending_motion and self._goal is not None
                    and self._xy_dist(new_xy, self._goal[:2]) <= self.goal_change_eps_m):
                self._goal = (x, y, frame)
                return
            self._reset_run_stats()
            self._goal = (x, y, frame)
            self._goal_seq = 1
            self._goal_close_ticks = 0
            if self.start_on_motion:
                # 收到 goal 但先不開檔，等 _cb_cmd 偵測到真正速度輸出才建資料夾+wandb
                self._pending_motion = True
                self.get_logger().info(
                    f"[{kind} #1] ({x:.2f},{y:.2f}) frame={frame}{extra}"
                    f" → 收到目標，等待速度輸出後開始紀錄"
                )
            else:
                self._open_run_files(self._exp_label)   # 舊行為：第一個 goal 立即開檔
                self.get_logger().info(
                    f"[{kind} #1] ({x:.2f},{y:.2f}) frame={frame}{extra} → 開始新一段紀錄"
                )
            return
        # 既非 started 也非 armed（auto_rearm=false 已停）→ 不再記錄，忽略

    def _cb_cmd(self, msg: Twist) -> None:
        self._cmd = (msg.linear.x, msg.angular.z, time.monotonic())
        # pending（已收 goal 等速度）：偵測到真正速度輸出 → 建資料夾開檔 + wandb
        if self._pending_motion and (
                abs(msg.linear.x) > self.start_motion_eps
                or abs(msg.angular.z) > self.start_motion_eps):
            self._pending_motion = False
            self._open_run_files(self._exp_label)
            self.get_logger().info(
                f"▶ 偵測到速度輸出（v={msg.linear.x:+.2f} w={msg.angular.z:+.2f}）→ 開始紀錄 + wandb"
            )

    def _cb_obs(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 6:
            self._obs = (list(msg.data), time.monotonic())

    def _cb_sweep(self, msg: Float32MultiArray) -> None:
        if not msg.data:
            return
        # sweep 是正規化 [0,1]，這裡用與訓練/preprocessor 相同公式還原成公尺，取最近一格 = 最近障礙
        m = min(msg.data)
        dist_m = m * (self.r_max - self.r_robot) + self.r_robot
        self._sweep_min = (dist_m, time.monotonic())

    # ── 20Hz 寫列 ──
    def _tick(self) -> None:
        if not self._started:
            return
        now = time.monotonic()
        has_goal = self._goal is not None
        if self.log_only_with_goal and not has_goal:
            return

        row = {k: "" for k in CSV_FIELDS}
        row["t_wall"] = f"{time.time():.3f}"
        row["t_rel"] = f"{now - self._t0:.3f}"
        row["goal_seq"] = self._goal_seq
        row["has_goal"] = int(has_goal)

        if self._odom is not None:
            ox, oy, oyaw, ov, ow, ot = self._odom
            row["odom_x"] = f"{ox:.3f}"
            row["odom_y"] = f"{oy:.3f}"
            row["odom_yaw_deg"] = f"{math.degrees(oyaw):.2f}"
            row["odom_v"] = f"{ov:.3f}"
            row["odom_w"] = f"{ow:.3f}"
            # 里程積分 → 本段路徑長度（供指標 path_len；跳大步視為 odom 斷流不累加）
            if self._prev_odom_xy is not None:
                step = math.hypot(ox - self._prev_odom_xy[0], oy - self._prev_odom_xy[1])
                if step < 1.0:
                    self._path_len += step
            self._prev_odom_xy = (ox, oy)

        # /ndt_pose 僅記錄供活性判定（ndt_age）；⚠ 此部署它是 map→odom 變換、非車姿，不拿來算 dist/heading
        if self._ndt is not None:
            nx, ny, nyaw, nt = self._ndt
            row["ndt_x"] = f"{nx:.3f}"
            row["ndt_y"] = f"{ny:.3f}"
            row["ndt_yaw_deg"] = f"{math.degrees(nyaw):.2f}"
            row["ndt_age"] = f"{now - nt:.2f}"

        # 車在 goal_frame 的真實位姿：走 TF goal_frame→base（與 policy_node._robot_pose_in_map 同源）。
        # 查不到 TF 才退回 odom（odom frame 非 map，dist/heading 僅近似）。
        robot_x, robot_y, robot_yaw, pose_src = self._robot_pose_in_map(self.goal_frame)
        if robot_x is None and self._odom is not None:
            robot_x, robot_y, robot_yaw, pose_src = (
                self._odom[0], self._odom[1], self._odom[2], "odom_only")
        if robot_x is not None:
            row["map_x"] = f"{robot_x:.3f}"
            row["map_y"] = f"{robot_y:.3f}"
            row["map_yaw_deg"] = f"{math.degrees(robot_yaw):.2f}"
            row["pose_src"] = pose_src

        if has_goal:
            gx, gy, gframe = self._goal
            row["goal_x"] = f"{gx:.3f}"
            row["goal_y"] = f"{gy:.3f}"
            row["goal_frame"] = gframe
            if robot_x is not None:
                # heading_err = 「指向 goal 的方向」減「車頭實際朝向」，車姿走 TF（與 policy 同源）；
                # 應與 policy_goal_ang_deg 一致，不一致代表定位/TF/座標出問題
                dx, dy = gx - robot_x, gy - robot_y
                dist = math.hypot(dx, dy)
                heading_err = _wrap_pi(math.atan2(dy, dx) - robot_yaw)
                row["dist_to_goal"] = f"{dist:.3f}"
                row["heading_err_deg"] = f"{math.degrees(heading_err):.2f}"
                self._heading_abs_sum += abs(math.degrees(heading_err))
                self._heading_n += 1

                # 每段超時（TO）：t_rel 超過上限仍未到 goal → 判 timeout（碰撞優先，見 _finish_leg）
                if (self.episode_timeout_s > 0
                        and (now - self._t0) > self.episode_timeout_s):
                    self._finish_leg("timeout", dist, gx, gy)
                    return
                # 到達 goal 自動停止（SR）：連續 N tick 距離 < tolerance → 判 success
                if self.auto_stop_goal_tol > 0 and dist < self.auto_stop_goal_tol:
                    self._goal_close_ticks += 1
                    if self._goal_close_ticks >= self.auto_stop_goal_ticks:
                        self._finish_leg("success", dist, gx, gy)
                        return
                else:
                    self._goal_close_ticks = 0

        if self._cmd is not None:
            cv, cw, ct = self._cmd
            row["cmd_v"] = f"{cv:.3f}"
            row["cmd_w"] = f"{cw:.3f}"
            row["cmd_age"] = f"{now - ct:.2f}"
            if self._cmd_w_prev is not None:
                dw = cw - self._cmd_w_prev
                self._dw_sq += dw * dw
                self._dw_n += 1
            self._cmd_w_prev = cw
            self._cmd_w_abs_sum += abs(cw)

        if self._obs is not None:
            obs, obt = self._obs
            bx, by = obs[4], obs[5]      # 79D obs 固定佈局：[4],[5] = goal 在 body frame 的 x,y
            row["policy_goal_bx"] = f"{bx:.3f}"
            row["policy_goal_by"] = f"{by:.3f}"
            row["policy_goal_ang_deg"] = f"{math.degrees(math.atan2(by, bx)):.2f}"
            row["policy_goal_dist"] = f"{math.hypot(bx, by):.3f}"   # policy 感知的 goal 距離
            row["policy_v_norm"] = f"{obs[1]:.3f}"
            row["policy_w_norm"] = f"{obs[2]:.3f}"
            row["obs_age"] = f"{now - obt:.2f}"
            # ★07-06: 72 束原始 obs（正規化值, [6:78]）分號串接 — bin 校準/前錐分析
            if len(obs) >= 78:
                row["obs_lidar72"] = ";".join(f"{v:.3f}" for v in obs[6:78])

        if self._sweep_min is not None:
            sm, st = self._sweep_min
            row["sweep_min_m"] = f"{sm:.3f}"
            row["sweep_age"] = f"{now - st:.2f}"
            if (now - st) < 1.0:
                # raw LiDAR 過近旗標（新鮮 sweep 才判）：最近障礙 < 門檻 → 1（與 TUI 碰撞閃紅同步）
                row["lidar_collision"] = int(sm < self.lidar_collision_m)
                if sm < self._min_sweep_run:
                    self._min_sweep_run = sm     # 本段最近障礙(含牆)，新鮮 sweep 才計

        st = self._status
        if st is not None:
            for k in ("rl_v", "rl_w", "sent_v", "sent_w", "act_v", "act_w",
                      "lag_ms", "lag_corr", "lag_ch",
                      "front_m", "back_m", "left_m", "right_m"):   # ★07-06 方向距離
                v = st.get(k)
                if v is not None:
                    row[k] = v
            row["v_over"] = int(bool(st.get("v_over")))
            row["w_over"] = int(bool(st.get("w_over")))

        # VO 安全層：node 在線(<1s 內有 status)=active；用其 status 旗標標記本拍有沒有動手
        vo_active = self._vo is not None and (now - self._vo_t) < 1.0
        row["vo_active"] = int(vo_active)
        if vo_active:
            row["vo_intervening"] = int(bool(self._vo.get("intervening")))
            row["vo_blocked"] = int(bool(self._vo.get("blocked")))
            row["vo_engaged"] = int(bool(self._vo.get("engaged")))
            er = self._vo.get("engage_range")
            if er is not None:
                row["vo_engage_range"] = er     # 整段常數，供 analyze 對照實測介入距離
            for src, dst in (("n_obs", "vo_n_obs"), ("n_feasible", "vo_n_feasible"),
                             ("des_v", "vo_des_v"), ("des_w", "vo_des_w"),
                             ("out_v", "vo_out_v"), ("out_w", "vo_out_w"),
                             ("min_ttc", "vo_min_ttc")):
                v = self._vo.get(src)
                if v is not None:          # min_ttc 可能為 null → 留空
                    row[dst] = v
        else:
            row["vo_intervening"] = 0
            row["vo_blocked"] = 0
            row["vo_engaged"] = 0
        # 本拍「誰在動手」單欄判定：VO 煞停 > VO 改寫 > 否則 RL 主導（含沒走 VO / VO 只放行）
        if row["vo_blocked"] == 1:
            row["driver"] = "vo_stop"
        elif row["vo_intervening"] == 1:
            row["driver"] = "vo"
        else:
            row["driver"] = "rl"

        row["nav_type"] = self._goal_method

        # 碰撞判定（填 min_sep_m / n_dyn_obs / collision + 累計事件）
        self._update_collision(row, now)

        self._writer.writerow(row)
        self._n_rows += 1
        if self._n_rows % 20 == 0:
            self._fh.flush()          # 每秒(20Hz)落盤一次，意外斷電也保住大部分資料

        if self.wandb_run is not None:
            # 同步到 wandb：只送可轉 float 的數值欄。⚠ 用 try/except 跳過所有不可轉字串欄
            # （pose_src='tf'/goal_frame/lag_ch/nav_type…）—— 硬 float() 黑名單一漏掉某欄
            # 就會 ValueError 讓整個 node 當場死掉（曾因 pose_src 漏列，每段只寫 1 行就崩）。
            # t_wall 牆鐘時間數值太大會壓壞圖表 x 軸，明確排除。
            metrics = {}
            for k, v in row.items():
                if v == "" or k == "t_wall":
                    continue
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    continue
            self.wandb.log(metrics)

    # ── 碰撞判定 ──
    def _update_collision(self, row: dict, now: float) -> None:
        """以 odom frame 車姿 vs 追蹤動態障礙算「車體表面↔人表面」最近間隙 min_sep，
        填 CSV 欄並偵測確認碰撞事件。obstacles 與 /odom 同 odom frame → 相對量測、免疫全域漂移。"""
        if not _HAS_TRACKED:
            return
        # 追蹤快取新鮮度：>1s 未更新視為偵測器沒起/死 → 無有效障礙（不誤判為安全，只是無資料）
        fresh = bool(self._obstacles_t) and (now - self._obstacles_t) < 1.0
        obs = self._obstacles if fresh else []
        if self.collision_min_track_age > 0:
            obs = [o for o in obs if o[3] >= self.collision_min_track_age]
        row["n_dyn_obs"] = len(obs)
        if self._odom is None or not obs:
            self._contact_ticks = 0
            row["collision"] = 0
            return
        rx, ry = self._odom[0], self._odom[1]
        # 中心↔中心距離（不扣半徑）＝與 VO engage_range 同基準，供校準介入距離；不受 engage 過濾
        cdist = [math.hypot(ox - rx, oy - ry) for ox, oy, _r, _age in obs]
        row["dyn_obs_min_m"] = f"{min(cdist):.3f}"
        # 表面↔表面間隙（碰撞判定用，扣掉障礙半徑 + 車體外緣半徑）
        min_sep = min(d - r - self.r_robot_collision
                      for d, (_ox, _oy, r, _age) in zip(cdist, obs))
        row["min_sep_m"] = f"{min_sep:.3f}"
        if min_sep < self._min_sep_run:
            self._min_sep_run = min_sep
        contact = min_sep <= self.collision_margin
        row["collision"] = int(contact)
        if not contact:
            self._contact_ticks = 0
            return
        self._contact_ticks += 1
        # 上升緣（剛滿 persist 拍）→ 記一次確認碰撞事件；持續接觸不重複計數
        if self._contact_ticks == self.collision_persist_ticks:
            self._collision_events += 1
            self._collided = True
            self.get_logger().warn(
                f"💥 碰撞事件 #{self._collision_events}：min_sep={min_sep:.2f}m "
                f"(≤{self.collision_margin:.2f}m 連續 {self.collision_persist_ticks} 拍)")
            ev = String()
            ev.data = json.dumps({"event": "collision", "min_sep": round(min_sep, 2),
                                  "n_events": self._collision_events})
            self._pub_event.publish(ev)

    # ── 一段（leg）結束：分類 SR/TO/CR、累計、寫 session 指標、停止並 re-arm ──
    def _finish_leg(self, outcome: str, dist: float, gx: float, gy: float) -> None:
        """outcome: success / timeout。碰撞優先：本段曾碰撞 → 一律改判 collision（CR>TO>SR）。"""
        if self._collided:
            outcome = "collision"
        dur = time.monotonic() - self._t0
        csv = self.csv_path or ""
        self._sess[outcome] = self._sess.get(outcome, 0) + 1
        icon = {"success": "🏁", "timeout": "⏱", "collision": "💥"}.get(outcome, "•")
        reason = {
            "success": f"到達 goal（dist={dist:.2f}m）",
            "timeout": f"超時 {self.episode_timeout_s:.0f}s 未達 goal（dist={dist:.2f}m）",
            "collision": (f"發生碰撞（{self._collision_events} 次事件"
                          f"{'＋人工標註' if self._collided_manual else ''}，"
                          f"min_sep={self._min_sep_run:.2f}m）"),
        }.get(outcome, outcome)
        self.get_logger().info(
            f"{icon} 本段結果 = {outcome.upper()}：{reason}\n  完整 log：{csv}")
        # 發 event（TUI / 外部監聽）
        ev = String()
        ev.data = json.dumps({"event": "leg_done", "outcome": outcome,
                              "dist": round(dist, 2), "dur": round(dur, 1),
                              "min_sep": round(self._min_sep_run, 2)
                              if math.isfinite(self._min_sep_run) else None,
                              "n_collision": self._collision_events, "csv": csv})
        self._pub_event.publish(ev)
        # wandb run summary（此段一個 run）：outcome one-hot + 距離指標。wandb UI 依 config.model /
        # experiment_tag group-by 後對 outcome_* 取平均 = 各條件的 SR/TO/CR，直接對比有無雜訊/DR 模型。
        if self.wandb_run is not None:
            try:
                self.wandb_run.summary.update({
                    "outcome": outcome,
                    "outcome_success": int(outcome == "success"),
                    "outcome_timeout": int(outcome == "timeout"),
                    "outcome_collision": int(outcome == "collision"),
                    "leg_duration_s": round(dur, 2),
                    "leg_path_len_m": round(self._path_len, 2),
                    "leg_min_sep_m": (round(self._min_sep_run, 3)
                                      if math.isfinite(self._min_sep_run) else None),
                    "leg_min_sweep_m": (round(self._min_sweep_run, 3)
                                        if math.isfinite(self._min_sweep_run) else None),
                    "leg_n_collision_events": self._collision_events,
                    "experiment_tag": self.experiment_tag,
                })
            except Exception as e:
                self.get_logger().warn(f"wandb summary 更新失敗：{e}")
        # 寫 session 級指標 + 印累計 SR/TO/CR
        self._write_session_metric(outcome, dur, gx, gy)
        self.stop_experiment()
        if self.auto_rearm:
            self._last_done_goal = (gx, gy)   # 擋 stale path 殘留；再待命等下一段
            self._rearm()

    def _write_session_metric(self, outcome: str, dur: float, gx: float, gy: float) -> None:
        """把每段結果附到 session 級 CSV（pingpong_metrics_<session>.csv），並印累計 SR/TO/CR。
        這份檔就是論文直接引用的成功率/超時率/碰撞率原始資料。"""
        n_ok = self._sess["success"]
        n_to = self._sess["timeout"]
        n_cr = self._sess["collision"]
        total = n_ok + n_to + n_cr
        min_sep = (f"{self._min_sep_run:.3f}"
                   if math.isfinite(self._min_sep_run) else "")
        try:
            if self._session_metrics_path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._session_metrics_path = os.path.join(
                    self.log_dir, f"pingpong_metrics_{stamp}.csv")
                os.makedirs(self.log_dir, exist_ok=True)
                with open(self._session_metrics_path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "leg", "t_end_wall", "experiment_tag", "model", "outcome",
                        "duration_s", "path_len_m", "min_sep_m", "min_sweep_m",
                        "n_collision_events", "collided_manual",
                        "goal_x", "goal_y", "nav_type", "csv",
                    ])
            min_sweep = (f"{self._min_sweep_run:.3f}"
                         if math.isfinite(self._min_sweep_run) else "")
            with open(self._session_metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    total, f"{time.time():.3f}", self.experiment_tag,
                    self._model_name or "", outcome, f"{dur:.1f}",
                    f"{self._path_len:.2f}", min_sep, min_sweep,
                    self._collision_events, int(self._collided_manual),
                    f"{gx:.3f}", f"{gy:.3f}", self._goal_method,
                    os.path.basename(self.csv_path or ""),
                ])
        except Exception as e:
            self.get_logger().warn(f"寫 session 指標失敗：{e}")
        pct = lambda n: 100.0 * n / total if total else 0.0
        self.get_logger().info(
            "──────── 本 session 累計 ────────\n"
            f"  段數 total={total}  成功 SR={n_ok}({pct(n_ok):.0f}%)  "
            f"超時 TO={n_to}({pct(n_to):.0f}%)  碰撞 CR={n_cr}({pct(n_cr):.0f}%)\n"
            f"  指標檔：{self._session_metrics_path}\n"
            "────────────────────────────────")

    # ── 摘要 ──
    def _print_summary(self) -> None:
        # Δω RMS = 相鄰兩次 cmd 角速度差的均方根，是「角速度晃動」的核心量化指標（越大越抖）
        dw_std = math.sqrt(self._dw_sq / self._dw_n) if self._dw_n > 0 else 0.0
        cmd_w_abs = self._cmd_w_abs_sum / max(self._dw_n + 1, 1)
        head_abs = self._heading_abs_sum / max(self._heading_n, 1)
        nav_desc = {
            "single_goal": "單一 goal（RViz 2D Goal Pose 單點）",
            "routing_path": "路徑導航（RViz Publish Point 兩點選起點/終點 → routing）",
        }.get(self._goal_method, "未知（未收到 goal/path）")
        sep = (f"{self._min_sep_run:.2f}m" if math.isfinite(self._min_sep_run)
               else "無動態障礙資料")
        col = (f"碰撞 {self._collision_events} 次事件"
               + ("＋人工標註" if self._collided_manual else "")
               if self._collided else "無碰撞")
        self.get_logger().info(
            "================ 診斷摘要 ================\n"
            f"  使用 model  : {self._model_name or '未知（未收到 policy status）'}\n"
            f"  goal 決定方式: {nav_desc}\n"
            f"  CSV: {self.csv_path}  ({self._n_rows} 列)\n"
            f"  角速度晃動  : Δω RMS={dw_std:.3f} rad/s/step, |ω|平均={cmd_w_abs:.3f}\n"
            f"  朝向誤差    : |heading_err|平均={head_abs:.1f}°  (理想<20°)\n"
            f"  碰撞/間隙   : {col}；本段最近間隙 min_sep={sep}；路徑長={self._path_len:.1f}m\n"
            f"  分析: ros2 run rover_rl_inference analyze_diag {self.csv_path or ''}\n"
            "=========================================="
        )

    def finalize(self) -> None:
        self.stop_experiment()
        if self._all_csv_paths:
            paths = "\n".join(f"  • {p}" for p in self._all_csv_paths)
            self.get_logger().info(
                f"本次 session 全部診斷 log：\n{paths}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = DiagLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()   # 印出診斷摘要 + CSV 存檔位置（若有在記錄）
        if node.csv_path is None:
            node.get_logger().info(
                "⏹ diag_logger 已停止（本次未啟動記錄，無 CSV 產生）"
            )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
