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
from std_msgs.msg import Float32MultiArray, String


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
    "ndt_x", "ndt_y", "ndt_yaw_deg", "ndt_age",             # NDT map-frame 定位（ground truth）
    "goal_x", "goal_y", "goal_frame",
    "dist_to_goal", "heading_err_deg",                       # 衍生：到 goal 距離 / 車頭朝向誤差
    "cmd_v", "cmd_w", "cmd_age",                             # 實際發出的 cmd_vel
    "policy_goal_bx", "policy_goal_by", "policy_goal_ang_deg",  # policy obs 內的 goal body 方向
    "policy_v_norm", "policy_w_norm", "obs_age",
    "sweep_min_m", "sweep_age",                              # 72-bin sweep 還原成公尺後的最近障礙
    # 三層速度 + 延遲（來自 /rover_rl_policy/status）：rl=RL意圖, sent=濾波後送出, act=odom實測
    "rl_v", "rl_w", "sent_v", "sent_w", "act_v", "act_w",
    "v_over", "w_over", "lag_ms", "lag_corr", "lag_ch",      # 飽和旗標 / 延遲估計值
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
        self.declare_parameter("topic_diag_event", "/rover_rl/diag_event")
        # 開錄時抓此節點的全部參數（speed_rate 等）寫進 sidecar + wandb config
        self.declare_parameter("policy_node_name", "rover_rl_policy")
        self.declare_parameter("lidar_r_max_m", 20.0)
        self.declare_parameter("robot_radius_m", 0.35)
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
        self.topic_diag_event = sv("topic_diag_event")
        self.policy_node_name = sv("policy_node_name")
        self.r_max = float(gp("lidar_r_max_m").value)
        self.r_robot = float(gp("robot_radius_m").value)
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
        self._goal_seq = 0
        self._last_done_goal = None  # (x, y) 剛 auto-stop 完成的終點，用來擋 stale path 殘留發布
        self._cmd = None
        self._obs = None
        self._sweep_min = None
        self._status = None        # dict（三層速度 + 延遲）
        self._goal_close_ticks = 0  # dist_to_goal < tol 連續 tick 計數
        self._all_csv_paths = []   # 本次 session 建立的所有 CSV（finalize 時全部列出）
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
        self._cmd_w_prev = None
        self._dw_sq = 0.0
        self._dw_n = 0
        self._cmd_w_abs_sum = 0.0
        self._heading_abs_sum = 0.0
        self._heading_n = 0

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
        self._pub_event = self.create_publisher(String, self.topic_diag_event, 10)

    # ── 實驗開關 ──
    def start_experiment(self, label: str) -> None:
        # 一個「實驗」= 一個獨立資料夾 + 一份 CSV(+wandb run)；重新 start 先收掉上一個
        if self._started or self._armed:
            self.stop_experiment()      # 先收掉上一個
        self._exp_label = label
        self._reset_run_stats()
        self._goal_seq = 0
        self._goal = None
        self._pending_motion = False
        self._last_done_goal = None
        if self.log_only_with_goal:
            # 待命：收到第一個 goal 才建資料夾 + 開檔（避免沒走 goal 留空資料夾）
            self._armed = True
            self.get_logger().info("▶ 待命中 — 收到第一個 goal 才建立紀錄資料夾")
        else:
            self._open_run_files(label)   # 不挑 goal → 立即開檔

    def _open_run_files(self, label: str) -> None:
        """真正建立 run 資料夾並開 CSV（由 start 或第一個 goal 觸發）."""
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
        self.get_logger().info(f"▶ 開始實驗 '{name}' → {self.csv_path}")
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
            "params": params,
        }
        try:
            side = (self._params_csv_path or "").rsplit(".", 1)[0] + "_params.json"
            with open(side, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(f"寫 params sidecar 失敗：{e}")
        # wandb config
        if self.wandb_run is not None:
            try:
                self.wandb_run.config.update(params, allow_val_change=True)
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
                config={"rate_hz": self.rate_hz},
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
            pass

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

        robot_x = robot_y = robot_yaw = None
        if self._odom is not None:
            ox, oy, oyaw, ov, ow, ot = self._odom
            row["odom_x"] = f"{ox:.3f}"
            row["odom_y"] = f"{oy:.3f}"
            row["odom_yaw_deg"] = f"{math.degrees(oyaw):.2f}"
            row["odom_v"] = f"{ov:.3f}"
            row["odom_w"] = f"{ow:.3f}"

        if self._ndt is not None:
            nx, ny, nyaw, nt = self._ndt
            row["ndt_x"] = f"{nx:.3f}"
            row["ndt_y"] = f"{ny:.3f}"
            row["ndt_yaw_deg"] = f"{math.degrees(nyaw):.2f}"
            row["ndt_age"] = f"{now - nt:.2f}"
            robot_x, robot_y, robot_yaw = nx, ny, nyaw   # 優先用 NDT(map frame) 算 dist/heading

        # NDT 還沒來時退回 odom，至少 dist/heading 有個近似值（odom 會漂但短時間可用）
        if robot_x is None and self._odom is not None:
            robot_x, robot_y, robot_yaw = self._odom[0], self._odom[1], self._odom[2]

        if has_goal:
            gx, gy, gframe = self._goal
            row["goal_x"] = f"{gx:.3f}"
            row["goal_y"] = f"{gy:.3f}"
            row["goal_frame"] = gframe
            if robot_x is not None:
                # heading_err = 「指向 goal 的方向」減「車頭實際朝向」（NDT ground truth）；
                # 應與 policy_goal_ang_deg 一致，不一致代表定位/TF/座標出問題
                dx, dy = gx - robot_x, gy - robot_y
                dist = math.hypot(dx, dy)
                heading_err = _wrap_pi(math.atan2(dy, dx) - robot_yaw)
                row["dist_to_goal"] = f"{dist:.3f}"
                row["heading_err_deg"] = f"{math.degrees(heading_err):.2f}"
                self._heading_abs_sum += abs(math.degrees(heading_err))
                self._heading_n += 1

                # 到達 goal 自動停止：連續 N tick 距離 < tolerance → stop_experiment
                if self.auto_stop_goal_tol > 0 and dist < self.auto_stop_goal_tol:
                    self._goal_close_ticks += 1
                    if self._goal_close_ticks >= self.auto_stop_goal_ticks:
                        csv = self.csv_path or ""
                        self.get_logger().info(
                            f"🏁 到達 goal（dist={dist:.2f}m < {self.auto_stop_goal_tol}m"
                            f" 連續 {self._goal_close_ticks} tick）→ 自動停止紀錄\n"
                            f"  完整 log 保存於：{csv}"
                        )
                        # 發 event 給 TUI / 外部監聽
                        ev = String()
                        ev.data = json.dumps({"event": "goal_reached",
                                              "dist": round(dist, 2), "csv": csv})
                        self._pub_event.publish(ev)
                        self.stop_experiment()
                        if self.auto_rearm:
                            # 記住剛完成的終點 → 擋掉 stale path 殘留發布；再待命等下一個 goal/path
                            self._last_done_goal = (gx, gy)
                            self._rearm()
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
            row["policy_v_norm"] = f"{obs[1]:.3f}"
            row["policy_w_norm"] = f"{obs[2]:.3f}"
            row["obs_age"] = f"{now - obt:.2f}"

        if self._sweep_min is not None:
            sm, st = self._sweep_min
            row["sweep_min_m"] = f"{sm:.3f}"
            row["sweep_age"] = f"{now - st:.2f}"

        st = self._status
        if st is not None:
            for k in ("rl_v", "rl_w", "sent_v", "sent_w", "act_v", "act_w",
                      "lag_ms", "lag_corr", "lag_ch"):
                v = st.get(k)
                if v is not None:
                    row[k] = v
            row["v_over"] = int(bool(st.get("v_over")))
            row["w_over"] = int(bool(st.get("w_over")))

        self._writer.writerow(row)
        self._n_rows += 1
        if self._n_rows % 20 == 0:
            self._fh.flush()          # 每秒(20Hz)落盤一次，意外斷電也保住大部分資料

        if self.wandb_run is not None:
            # 同步到 wandb：只送可轉 float 的數值欄，排除字串欄(goal_frame/lag_ch)與
            # 牆鐘時間(t_wall，數值太大會壓壞圖表 x 軸)
            metrics = {k: float(v) for k, v in row.items()
                       if v != "" and k not in ("goal_frame", "t_wall", "lag_ch")}
            self.wandb.log(metrics)

    # ── 摘要 ──
    def _print_summary(self) -> None:
        # Δω RMS = 相鄰兩次 cmd 角速度差的均方根，是「角速度晃動」的核心量化指標（越大越抖）
        dw_std = math.sqrt(self._dw_sq / self._dw_n) if self._dw_n > 0 else 0.0
        cmd_w_abs = self._cmd_w_abs_sum / max(self._dw_n + 1, 1)
        head_abs = self._heading_abs_sum / max(self._heading_n, 1)
        self.get_logger().info(
            "================ 診斷摘要 ================\n"
            f"  CSV: {self.csv_path}  ({self._n_rows} 列)\n"
            f"  角速度晃動  : Δω RMS={dw_std:.3f} rad/s/step, |ω|平均={cmd_w_abs:.3f}\n"
            f"  朝向誤差    : |heading_err|平均={head_abs:.1f}°  (理想<20°)\n"
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
