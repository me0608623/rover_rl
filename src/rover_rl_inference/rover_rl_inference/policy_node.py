"""rover_rl_inference policy node — full deployment 版.

整合元件：
  - localization.MapOdomOffsetTracker  : NDT cached offset + fallback
  - subgoal_selector.SubgoalSelector   : Path/PoseStamped → carrot lookahead
  - cmd_filter.CmdFilter               : low-pass + slew-rate
  - mode_manager.ModeManager           : nav / idle / estop / manual / paused
  - markers                            : RViz MarkerArray debug
  - lidar_preprocess.lidar_sweep_72_real (+ motion compensation)

Timer 結構（採 spot_rl 多 timer pattern，刻意把推論與發布解耦）：
  - inference_timer (5 Hz):  RL 推論 → 更新 target cmd
      推論週期鎖 5 Hz（dt=0.2s）是訓練值，動作分布綁在此頻率上，勿改。
  - cmd_timer (20 Hz):       low-pass/slew filter → 發 cmd_vel
      cmd 要 20 Hz republish 是為了餵飽底盤 / cmd_vel mux 的 watchdog；
      即使某次推論延遲，cmd timer 仍以 last_target + filter 持續出流不斷檔。
  - marker_timer (10 Hz):    發 RViz markers（純除錯視覺，10 Hz 對人眼夠用又省 CPU）
  - heartbeat_timer (1 Hz):  log 狀態

Mode 切換：
  - 訂閱 `~/mode` (std_msgs/String): "nav" / "idle" / "estop" / "manual" / "paused"
  - 服務 `~/set_mode` (std_srvs/SetBool): true=nav, false=idle
  - 服務 `~/load_model` (std_srvs/Trigger): 重載 model_path 指向的 .ts

sim-to-real 核心對策（細節見 rover_rl/CLAUDE.md）：
  - speed_rate 時間膨脹：rate<1 時把感知量放大 1/rate、動作上限縮 ×rate，
    讓 policy 自以為仍在訓練的原速世界 → 不離開訓練分布。
  - NDT 定位走 cached map→odom offset（非直接 TF lookup），NDT 短暫消失仍可跑。
  - LiDAR / odom 皆有 timeout watchdog，逾時即強制 cmd=0（fail-safe）。
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import MarkerArray

from .action_decoder import ActionParams, decode_logits_to_cmd
from .cmd_filter import CmdFilter, CmdFilterParams
from .latency import LagEstimator
from .lidar_preprocess import lidar_sweep_72_real, pointcloud2_to_xyz
from .localization import MapOdomOffsetTracker, RobotPose, world_to_body
from .markers import build_marker_array
from .mode_manager import Mode, ModeManager
from .model_runtime import PolicyRunner, load_bundle
from .obs_builder import ObsParams, build_obs_raw
from .subgoal_selector import SubgoalSelector


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    # 只取 yaw（平面機器人不需 roll/pitch），避免引入 tf_transformations 額外依賴
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PolicyNode(Node):
    """RL 推論主節點：sweep → obs → RNN → cmd_vel，外加多 mode / subgoal / NDT 定位。

    初始化順序刻意固定：先讀參數 → 載模型（失敗即早 raise，不啟空節點）→
    建狀態與元件 → 再建 pub/sub 與 timer（確保 callback 觸發時依賴已就緒）。
    """

    def __init__(self):
        super().__init__("rover_rl_policy")
        self._declare_params()
        self._read_params()
        self._load_model()
        self._init_state()
        self._init_pubsub()
        self._init_timers()
        self._init_services()

        self.get_logger().info(
            f"rover_rl_policy 啟動完成\n"
            f"  模型: {self._model_path} (raw_obs={self.bundle.raw_obs_dim}, "
            f"hidden={self.bundle.hidden_dim})\n"
            f"  模式: {self.mode_mgr.mode.value} (require_ndt={self.require_ndt})\n"
            f"  速度: speed_rate={self.speed_rate:.2f} "
            f"(實體上限 v={self.act_params.max_linear_velocity*self.speed_rate:.2f}m/s "
            f"ω={self.act_params.max_angular_velocity_action*self.speed_rate:.2f}rad/s)\n"
            f"  時鐘: inference {1/self.control_dt:.1f}Hz, cmd {self.cmd_rate_hz}Hz, "
            f"marker {self.marker_rate_hz}Hz"
        )

    # ──────────────────────────── 參數宣告 ────────────────────────────

    def _declare_params(self) -> None:
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cpu")
        # rates
        self.declare_parameter("control_dt", 0.2)        # inference
        self.declare_parameter("cmd_rate_hz", 20.0)
        self.declare_parameter("marker_rate_hz", 10.0)
        self.declare_parameter("publish_markers", True)
        # timeouts
        self.declare_parameter("timeout_lidar_s", 0.5)
        self.declare_parameter("timeout_odom_s", 0.3)
        # policy
        self.declare_parameter("deterministic", True)
        # topics
        # 預設：訂閱「已預處理」的 72-bin sweep（由 lidar_preprocessor_node 發布）
        # fallback：若沒收到 preprocessed sweep，自動 fallback 訂閱 raw PointCloud2 並自己處理
        self.declare_parameter("topic_lidar_sweep", "/rover_rl/lidar_sweep_72")
        self.declare_parameter("topic_lidar_raw", "/velodyne_points")
        self.declare_parameter("use_inline_preprocess", False)
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_goal_pose", "/goal_pose")
        self.declare_parameter("topic_global_path", "/global_path")
        self.declare_parameter("topic_cmd_vel", "/input/nav_cmd_vel")
        self.declare_parameter("topic_ndt_pose", "/ndt_pose")
        self.declare_parameter("topic_markers", "~/markers")
        self.declare_parameter("topic_obs_debug", "~/obs_debug")
        self.declare_parameter("publish_obs_debug", False)
        # qos
        self.declare_parameter("lidar_qos_best_effort", True)
        # LiDAR
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)
        self.declare_parameter("lidar_z_filter_m", 0.5)
        self.declare_parameter("lidar_r_min_m", 0.9)
        self.declare_parameter("lidar_r_max_m", 20.0)
        self.declare_parameter("lidar_motion_compensation", True)
        # obs normalizer (與訓練端對齊)
        # 這幾個 max_* 是 obs 正規化的分母，必須跟訓練端 obs_functions.py 完全一致；
        # 不一致 → policy 看到的數值尺度錯位，車會原地震/亂走（sim-to-real mismatch）。
        self.declare_parameter("obs_max_acceleration", 1.0)
        self.declare_parameter("obs_max_linear_velocity", 1.0)
        self.declare_parameter("obs_max_angular_velocity", 1.5)
        self.declare_parameter("robot_radius_m", 0.35)
        self.declare_parameter("episode_horizon_s", 60.0)
        # action limits (鎖在訓練分布)
        # 即使底盤實體更強也不要把這些往上開：policy 沒在更高速分布訓練過，
        # extrapolate 會行為不可預測（見 CLAUDE.md「動作上限勿改」）。
        self.declare_parameter("act_max_linear_velocity", 1.0)
        self.declare_parameter("act_max_linear_accel", 0.5)
        self.declare_parameter("act_max_angular_velocity", 2.0)
        # 全域速度縮放（spot_rl 時間膨脹）：(0,1]，1.0=原速，0.5=半速
        # 動作上限 ×rate，感知速度/目標 ÷rate → policy 留在訓練分布內
        self.declare_parameter("speed_rate", 1.0)
        # goal / localization
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("goal_tolerance_m", 0.6)
        self.declare_parameter("path_lookahead_m", 2.0)
        self.declare_parameter("require_ndt", False)
        # safety
        self.declare_parameter("safety_lidar_emergency_stop_m", 0.40)
        # cmd filter
        self.declare_parameter("cmd_alpha_linear", 0.3)
        self.declare_parameter("cmd_alpha_angular", 0.5)
        self.declare_parameter("cmd_max_accel_linear", 1.0)
        self.declare_parameter("cmd_max_accel_angular", 3.0)
        # 底盤實體上限（僅供診斷：判定 cmd 是否超出底盤能力，見 CLAUDE.md gap #2）
        self.declare_parameter("chassis_v_max", 1.5)
        self.declare_parameter("chassis_omega_max", 1.2)
        # initial mode
        self.declare_parameter("initial_mode", "nav")

    def _read_params(self) -> None:
        gp = self.get_parameter
        self._model_path = gp("model_path").get_parameter_value().string_value
        self._device = gp("device").get_parameter_value().string_value
        self.control_dt = float(gp("control_dt").value)
        self.cmd_rate_hz = float(gp("cmd_rate_hz").value)
        self.marker_rate_hz = float(gp("marker_rate_hz").value)
        self.publish_markers = bool(gp("publish_markers").value)
        self.timeout_lidar_s = float(gp("timeout_lidar_s").value)
        self.timeout_odom_s = float(gp("timeout_odom_s").value)
        self.deterministic = bool(gp("deterministic").value)
        self.topic_lidar_sweep = gp("topic_lidar_sweep").get_parameter_value().string_value
        self.topic_lidar_raw = gp("topic_lidar_raw").get_parameter_value().string_value
        self.use_inline_preprocess = bool(gp("use_inline_preprocess").value)
        self.topic_odom = gp("topic_odom").get_parameter_value().string_value
        self.topic_goal = gp("topic_goal_pose").get_parameter_value().string_value
        self.topic_path = gp("topic_global_path").get_parameter_value().string_value
        self.topic_cmd = gp("topic_cmd_vel").get_parameter_value().string_value
        self.topic_ndt = gp("topic_ndt_pose").get_parameter_value().string_value
        self.topic_markers = gp("topic_markers").get_parameter_value().string_value
        self.topic_obs_debug = gp("topic_obs_debug").get_parameter_value().string_value
        self.publish_obs_debug = bool(gp("publish_obs_debug").value)
        self.lidar_qos_be = bool(gp("lidar_qos_best_effort").value)
        self.lidar_yaw_offset = math.radians(float(gp("lidar_yaw_offset_deg").value))
        self.lidar_z_filter = float(gp("lidar_z_filter_m").value)
        self.lidar_r_min = float(gp("lidar_r_min_m").value)
        self.lidar_r_max = float(gp("lidar_r_max_m").value)
        self.lidar_motion_comp = bool(gp("lidar_motion_compensation").value)
        self.obs_params = ObsParams(
            max_acceleration=float(gp("obs_max_acceleration").value),
            max_linear_velocity=float(gp("obs_max_linear_velocity").value),
            max_angular_velocity_obs=float(gp("obs_max_angular_velocity").value),
            robot_radius=float(gp("robot_radius_m").value),
            episode_horizon_s=float(gp("episode_horizon_s").value),
        )
        self.act_params = ActionParams(
            num_bins=19,
            max_linear_velocity=float(gp("act_max_linear_velocity").value),
            max_linear_accel=float(gp("act_max_linear_accel").value),
            max_angular_velocity_action=float(gp("act_max_angular_velocity").value),
            dt=self.control_dt,
        )
        self.speed_rate = self._clamp_rate(float(gp("speed_rate").value))
        self.goal_frame = gp("goal_frame").get_parameter_value().string_value
        self.base_frame = gp("base_frame").get_parameter_value().string_value
        self.goal_tolerance = float(gp("goal_tolerance_m").value)
        self.path_lookahead = float(gp("path_lookahead_m").value)
        self.require_ndt = bool(gp("require_ndt").value)
        self.safety_estop_m = float(gp("safety_lidar_emergency_stop_m").value)
        self.cmd_filter_params = CmdFilterParams(
            alpha_linear=float(gp("cmd_alpha_linear").value),
            alpha_angular=float(gp("cmd_alpha_angular").value),
            max_accel_linear=float(gp("cmd_max_accel_linear").value),
            max_accel_angular=float(gp("cmd_max_accel_angular").value),
        )
        self._initial_mode = gp("initial_mode").get_parameter_value().string_value or "nav"

    # ──────────────────────────── 模型載入 ────────────────────────────

    def _load_model(self) -> None:
        # 模型路徑無效就直接 raise：寧可啟動失敗，也不要起一個發不出 cmd 的空節點
        if not self._model_path or not os.path.isfile(self._model_path):
            raise RuntimeError(
                f"model_path 無效: {self._model_path!r}。"
                "請 export_policy.py 後設定為 .ts 絕對路徑"
            )
        self.bundle = load_bundle(self._model_path, device=self._device)
        self.runner = PolicyRunner(self.bundle)

    # ──────────────────────────── 狀態與元件 ────────────────────────────

    def _init_state(self) -> None:
        # 所有 sensor cache / target cmd 都跨 timer 與 callback 執行緒共享，
        # rclpy callback 可能在不同 thread 觸發 → 一律用 self._lock 保護讀寫
        self._lock = threading.Lock()
        # sensor cache
        self._latest_sweep: np.ndarray | None = None
        self._latest_sweep_t = 0.0
        self._latest_pc: np.ndarray | None = None
        self._latest_pc_t = 0.0
        self._sweep_source: str = "none"   # 'preprocessor_topic' / 'inline'
        self._odom_xy = (0.0, 0.0)
        self._odom_yaw = 0.0
        self._odom_v = 0.0
        self._odom_w = 0.0
        self._odom_t = 0.0
        self._last_accel = 0.0
        self._start_t = time.monotonic()
        # 上次 policy 推論的「目標 cmd」（給 cmd timer 取用）
        # 推論(5Hz)與發布(20Hz)解耦的關鍵橋樑：inference 只寫這組目標值，
        # cmd timer 讀它做 filter 後送出，兩端頻率互不阻塞
        self._target_v = 0.0
        self._target_w = 0.0
        self._target_set_t = 0.0
        # 上次 sweep（給 marker 用）
        self._last_sweep: np.ndarray | None = None
        # 上次 subgoal（給 marker 用）
        self._last_subgoal_body: tuple[float, float] | None = None
        self._last_subgoal_source: str | None = None
        # map→odom 的 child frame（NDT 發布此段）；_cb_odom 會以實際 frame_id 更新
        self.odom_frame = "odom"
        # 元件
        # 註：localizer 現僅用於「NDT 是否活著且收斂」的判定（is_ndt_stable / ndt_age，
        # 基於 /ndt_pose 訊息到達時間）。機器人 map 位姿改由 _robot_pose_in_map 走 TF 取得，
        # 因為此部署的 ndt_localizer 發的 /ndt_pose 是 map→odom 變換、不是車姿（見下方註解）。
        self.localizer = MapOdomOffsetTracker(logger=self.get_logger())
        self.subgoals = SubgoalSelector(lookahead_m=self.path_lookahead)
        self.cmd_filter = CmdFilter(self.cmd_filter_params)
        # 延遲估計（送出 cmd ↔ odom 實測），線速度/角速度各一
        _cmd_dt = 1.0 / max(self.cmd_rate_hz, 1.0)
        self.lag_v = LagEstimator(_cmd_dt)
        self.lag_w = LagEstimator(_cmd_dt)
        self.chassis_v_max = float(self.get_parameter("chassis_v_max").value)
        self.chassis_omega_max = float(self.get_parameter("chassis_omega_max").value)
        try:
            initial = Mode.parse(self._initial_mode)
        except ValueError:
            self.get_logger().warn(f"未知 initial_mode={self._initial_mode!r}，使用 nav")
            initial = Mode.NAV
        self.mode_mgr = ModeManager(initial=initial, on_change=self._on_mode_change)

    # ──────────────────────────── ROS 介面 ────────────────────────────

    def _init_pubsub(self) -> None:
        # LiDAR 走 BEST_EFFORT + VOLATILE：感測資料寧可丟舊幀也不要塞滿佇列阻塞，
        # 且大多數 LiDAR driver（velodyne）發布端就是 BEST_EFFORT，QoS 不相容會收不到
        sensor_qos = QoSProfile(
            reliability=(QoSReliabilityPolicy.BEST_EFFORT
                         if self.lidar_qos_be else QoSReliabilityPolicy.RELIABLE),
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # 優先：訂閱已預處理的 sweep（來自 lidar_preprocessor_node）
        self.create_subscription(
            Float32MultiArray, self.topic_lidar_sweep, self._cb_sweep, 10,
        )
        # Fallback：raw PointCloud2（use_inline_preprocess=true 或外部 preprocessor 沒啟動）
        if self.use_inline_preprocess:
            self.create_subscription(
                PointCloud2, self.topic_lidar_raw, self._cb_lidar_raw, sensor_qos,
            )
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)
        self.create_subscription(PoseStamped, self.topic_goal, self._cb_goal, 10)
        self.create_subscription(NavPath, self.topic_path, self._cb_path, 10)
        self.create_subscription(PoseStamped, self.topic_ndt, self._cb_ndt_pose, 10)
        self.create_subscription(String, "~/mode", self._cb_mode_topic, 10)

        self.pub_cmd = self.create_publisher(Twist, self.topic_cmd, 10)
        self.pub_markers = (
            self.create_publisher(MarkerArray, self.topic_markers, 10)
            if self.publish_markers else None
        )
        self.pub_obs = (
            self.create_publisher(Float32MultiArray, self.topic_obs_debug, 10)
            if self.publish_obs_debug else None
        )
        # 供 TUI 儀表板訂閱的精簡狀態（JSON String），與 log 解耦
        self.pub_status = self.create_publisher(String, "~/status", 10)

        # TF2：以 map→base_frame 取機器人真實 map 位姿（與 RViz 顯示同一條鏈）。
        # 此部署的 ndt_localizer 發的 /ndt_pose 內容是 map→odom 變換、不是車姿，直接拿來
        # 當車姿會差一個 odom→base（車離 odom 原點越遠差越多）。改由 tf2 正確合成
        # map→odom(NDT) ∘ odom→base(底盤)。
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _init_timers(self) -> None:
        # 多 timer 分頻：推論(control_dt)/發布(cmd_rate)/marker/heartbeat/status 各跑各的
        self.timer_infer = self.create_timer(self.control_dt, self._tick_inference)
        self.timer_cmd = self.create_timer(1.0 / max(self.cmd_rate_hz, 1.0), self._tick_cmd)
        if self.publish_markers:
            self.timer_marker = self.create_timer(
                1.0 / max(self.marker_rate_hz, 1.0), self._tick_marker,
            )
        self.timer_heartbeat = self.create_timer(2.0, self._tick_heartbeat)
        self.timer_status = self.create_timer(0.2, self._publish_status)
        self._cmd_last_t = time.monotonic()

    def _init_services(self) -> None:
        self.create_service(SetBool, "~/set_mode", self._srv_set_mode)
        self.create_service(Trigger, "~/load_model", self._srv_load_model)
        self.create_service(Trigger, "~/reset_hidden", self._srv_reset_hidden)
        # 允許 ros2 param set /rover_rl_policy speed_rate 0.5 即時調整
        self.add_on_set_parameters_callback(self._on_param_update)

    def _clamp_rate(self, r: float) -> float:
        """speed_rate 限制在 (0, 1]；超出範圍 clamp 並警告."""
        clamped = float(np.clip(r, 0.05, 1.0))
        if abs(clamped - r) > 1e-6:
            self.get_logger().warn(
                f"speed_rate={r} 超出 [0.05, 1.0]，clamp 成 {clamped}"
            )
        return clamped

    def _on_param_update(self, params) -> SetParametersResult:
        """即時更新 speed_rate（其他參數不在此處理）.

        只攔 speed_rate 是因為它是少數能在跑動中安全熱調的旋鈕（時間膨脹，不離分布）；
        動作上限 / obs normalizer 改了會破壞 sim-to-real 對齊，故不在此開放熱改。
        """
        for p in params:
            if p.name == "speed_rate":
                new_rate = self._clamp_rate(float(p.value))
                old = self.speed_rate
                self.speed_rate = new_rate
                self.get_logger().info(f"speed_rate: {old:.2f} → {new_rate:.2f}")
        return SetParametersResult(successful=True)

    # ──────────────────────────── Callbacks ────────────────────────────

    def _cb_sweep(self, msg: Float32MultiArray) -> None:
        """收到外部 preprocessor 處理好的 72-bin sweep（首選來源）."""
        if len(msg.data) != self.obs_params.lidar_num_bins:
            self.get_logger().warn(
                f"sweep 維度錯 ({len(msg.data)} != {self.obs_params.lidar_num_bins})，忽略",
                throttle_duration_sec=5.0,
            )
            return
        now = time.monotonic()
        sweep = np.asarray(msg.data, dtype=np.float32)
        with self._lock:
            self._latest_sweep = sweep
            self._latest_sweep_t = now
            self._sweep_source = "preprocessor_topic"

    def _cb_lidar_raw(self, msg: PointCloud2) -> None:
        """Fallback：raw PointCloud2，自己處理（use_inline_preprocess=true 時啟用）."""
        pts = pointcloud2_to_xyz(msg)
        now = time.monotonic()
        with self._lock:
            self._latest_pc = pts
            self._latest_pc_t = now

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        tw = msg.twist.twist
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        if msg.header.frame_id:
            self.odom_frame = msg.header.frame_id
        with self._lock:
            self._odom_xy = (p.x, p.y)
            self._odom_yaw = yaw
            self._odom_v = tw.linear.x
            self._odom_w = tw.angular.z
            self._odom_t = time.monotonic()

    def _cb_goal(self, msg: PoseStamped) -> None:
        # 新目標 = 新 episode：reset RNN hidden（清掉上一段的記憶）與 cmd filter，
        # 並重置 elapsed 計時（episode_horizon 從 0 起算），對齊訓練時每 episode 的初始狀態
        frame = msg.header.frame_id or self.goal_frame
        self.subgoals.set_single_goal(msg.pose.position.x, msg.pose.position.y, frame)
        self.runner.reset()
        self.cmd_filter.reset()
        with self._lock:
            self._start_t = time.monotonic()
        self.get_logger().info(
            f"收到 goal_pose frame={frame} ({msg.pose.position.x:.2f},{msg.pose.position.y:.2f})"
        )

    def _cb_path(self, msg: NavPath) -> None:
        if not msg.poses:
            self.subgoals.clear_path()
            return
        frame = msg.header.frame_id or self.goal_frame
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.subgoals.set_path(pts, frame)
        # 一旦收到 path 就讓 path 蓋過單一 goal_pose（routing 規劃出的路徑優先於手點目標）
        self.subgoals.prefer_path = True
        self.runner.reset()
        self.cmd_filter.reset()
        with self._lock:
            self._start_t = time.monotonic()
        self.get_logger().info(
            f"收到 path frame={frame} ({len(pts)} waypoints, lookahead={self.path_lookahead}m)"
        )

    def _cb_ndt_pose(self, msg: PoseStamped) -> None:
        # ⚠ 此部署的 ndt_localizer 發的 /ndt_pose 內容是 map→odom 變換、不是車在 map 的位姿
        #   （實測 /ndt_pose == map→odom TF；真實車姿 = map→odom ∘ odom→base，差一個 odom→base）。
        #   故這裡**不**拿它算車姿——車姿改由 _robot_pose_in_map 走 TF map→base 取得。
        #   本 callback 只用「有持續收到 /ndt_pose」判定 NDT 還活著且收斂中（is_ndt_stable /
        #   ndt_age，基於訊息到達的 monotonic 時間，跨機器時鐘不同步也可靠）。
        ndt_yaw = _yaw_from_quat(
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        )
        with self._lock:
            ox, oy = self._odom_xy
            oyaw = self._odom_yaw
        self.localizer.on_ndt_pose(
            msg.pose.position.x, msg.pose.position.y, ndt_yaw,
            ox, oy, oyaw,
        )

    def _cb_mode_topic(self, msg: String) -> None:
        try:
            self.mode_mgr.set(Mode.parse(msg.data), reason="topic ~/mode")
        except ValueError as e:
            self.get_logger().warn(f"mode topic 收到無效值: {e}")

    # ──────────────────────────── Services ────────────────────────────

    def _srv_set_mode(self, req: SetBool.Request, res: SetBool.Response):
        target = Mode.NAV if req.data else Mode.IDLE
        changed = self.mode_mgr.set(target, reason="service ~/set_mode")
        res.success = True
        res.message = f"mode={self.mode_mgr.mode.value} (changed={changed})"
        return res

    def _srv_load_model(self, req: Trigger.Request, res: Trigger.Response):
        new_path = self.get_parameter("model_path").get_parameter_value().string_value
        try:
            if not new_path or not os.path.isfile(new_path):
                raise RuntimeError(f"model_path 無效: {new_path!r}")
            new_bundle = load_bundle(new_path, device=self._device)
            self.bundle = new_bundle
            self.runner = PolicyRunner(new_bundle)
            self.cmd_filter.reset()
            self._model_path = new_path
            res.success = True
            res.message = (f"已重載: {new_path} (raw_obs={new_bundle.raw_obs_dim}, "
                           f"hidden={new_bundle.hidden_dim})")
            self.get_logger().info(res.message)
        except Exception as e:
            res.success = False
            res.message = f"重載失敗: {e}"
            self.get_logger().error(res.message)
        return res

    def _srv_reset_hidden(self, req: Trigger.Request, res: Trigger.Response):
        self.runner.reset()
        self.cmd_filter.reset()
        with self._lock:
            self._target_v = 0.0
            self._target_w = 0.0
        res.success = True
        res.message = "RNN hidden state 與 cmd filter 已重置"
        return res

    def _on_mode_change(self, old: Mode, new: Mode, reason: str) -> None:
        self.get_logger().info(f"mode: {old.value} → {new.value} (reason: {reason})")
        # 切到任何非行駛模式時立刻把 target 歸 0 並重置 filter，
        # 避免切回 nav 時殘留舊 target 或 filter 慣性造成突跳
        if new in (Mode.IDLE, Mode.ESTOP, Mode.PAUSED):
            with self._lock:
                self._target_v = 0.0
                self._target_w = 0.0
            self.cmd_filter.reset()

    # ──────────────────────────── 定位（TF map→base）────────────────────────────

    def _robot_pose_in_map(self, odom_x: float, odom_y: float, odom_yaw: float) -> RobotPose:
        """機器人在 map frame 的真實位姿，取自 TF map→base_frame（與 RViz 同一條鏈）.

        為何不用 /ndt_pose：此 ndt_localizer 發的 /ndt_pose 是 map→odom 變換、非車姿，
        當成車姿會差一個 odom→base（車離 odom 原點越遠差越多，曾導致 goal 方位/距離全歪）。
        TF 鏈 map→odom(NDT) ∘ odom→base(底盤) 由 tf2 正確合成。查不到 TF 時退回純 odom。
        用 Time()（最新可用）而非當下時刻：避免跨機器時鐘不同步造成 extrapolation 失敗。
        """
        try:
            tf = self.tf_buffer.lookup_transform(self.goal_frame, self.base_frame, Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            return RobotPose(
                x=t.x, y=t.y,
                yaw=_yaw_from_quat(q.x, q.y, q.z, q.w), source="tf",
            )
        except TransformException:
            return RobotPose(x=odom_x, y=odom_y, yaw=odom_yaw, source="odom_only")

    def _map_odom_from_tf(self) -> tuple[float, float, float] | None:
        """回傳 map→odom 變換 (x, y, yaw)，供狀態顯示 NDT 修正量；查不到回 None."""
        try:
            tf = self.tf_buffer.lookup_transform(self.goal_frame, self.odom_frame, Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            return t.x, t.y, _yaw_from_quat(q.x, q.y, q.z, q.w)
        except TransformException:
            return None

    # ──────────────────────────── Timer: 推論 (5 Hz) ────────────────────────────

    def _tick_inference(self) -> None:
        # 非 active mode（idle/estop/manual/paused）完全不推論，省 CPU 也避免誤動
        if not self.mode_mgr.is_active():
            return

        now = time.monotonic()
        # 先在鎖內快照所有共享狀態，後續重運算（推論）都在鎖外做，縮短臨界區
        with self._lock:
            sweep_pre = self._latest_sweep
            sweep_age = now - self._latest_sweep_t if sweep_pre is not None else float("inf")
            pc = self._latest_pc
            pc_age = now - self._latest_pc_t if pc is not None else float("inf")
            odom_age = now - self._odom_t
            odom_x, odom_y = self._odom_xy
            odom_yaw = self._odom_yaw
            v = self._odom_v
            w = self._odom_w
            last_accel = self._last_accel
            elapsed = now - self._start_t

        # 選 sweep 來源：preprocessor topic 首選；沒收到/過期 → fallback inline
        # 獨立 preprocessor 節點（spot_rl pattern）發布的 sweep 較可驗證；
        # 它掛了才退回自己從 raw PointCloud2 算（需 use_inline_preprocess=true）
        sweep_from_topic = sweep_pre is not None and sweep_age <= self.timeout_lidar_s
        if sweep_from_topic:
            sweep_active = sweep_pre
            sweep_source_tag = "topic"
        elif self.use_inline_preprocess and pc is not None and pc_age <= self.timeout_lidar_s:
            sweep_active = None       # 等下方算
            sweep_source_tag = "inline_fallback"
        else:
            reason = "LiDAR timeout"
            if not self.use_inline_preprocess:
                reason += "（且 use_inline_preprocess=false，需啟動 lidar_preprocessor_node）"
            return self._set_target_stop(reason)
        # watchdog：odom 逾時 = 失去自身速度/位姿基準，立刻停車
        if odom_age > self.timeout_odom_s:
            return self._set_target_stop("Odom timeout")
        # 沒目標就停（warn=False：待命狀態屬正常，不洗 log）
        if not self.subgoals.has_target():
            return self._set_target_stop("尚未收到 goal/path", warn=False)

        # 機器人在 map frame 的位姿：走 TF map→base（map→odom 由 NDT 發、odom→base 由底盤發）
        robot_pose = self._robot_pose_in_map(odom_x, odom_y, odom_yaw)
        # require_ndt=true：沒有 map→base TF（NDT 未提供 map→odom）或 NDT 不穩定就不動
        if self.require_ndt and (robot_pose.source != "tf"
                                 or not self.localizer.is_ndt_stable()):
            return self._set_target_stop("require_ndt=true 但 NDT/TF 未就緒")

        # 取得 sub-goal（lookahead 或 single）
        choice = self.subgoals.select(robot_pose.x, robot_pose.y)
        if choice is None:
            return self._set_target_stop("無法選定 subgoal")

        # frame 一致性保護：subgoal 在 odom frame 時，robot 位姿也必須用 odom 系，
        # 否則 world_to_body 會把兩個不同座標系的點相減 → goal 方向算錯
        if choice.frame == "odom":
            robot_x, robot_y, robot_yaw_use = odom_x, odom_y, odom_yaw
        else:
            robot_x, robot_y, robot_yaw_use = robot_pose.x, robot_pose.y, robot_pose.yaw

        # 把 goal 轉到 body frame（policy obs 用的是相對機器人的座標）
        gx, gy = world_to_body(choice.x, choice.y, robot_x, robot_y, robot_yaw_use)
        dist = math.hypot(gx, gy)
        # path_lookahead 的 carrot 會一直往前滑，不該因「接近 carrot」就判定到達；
        # 只有 single goal / path 終點等真正終點才用 goal_tolerance 判停
        if choice.source != "path_lookahead" and dist < self.goal_tolerance:
            return self._set_target_stop(
                f"到達 {choice.source} (dist={dist:.2f})", warn=False,
            )

        # LiDAR sweep 取得
        if sweep_active is None:
            # inline fallback：自己從 PointCloud2 處理
            # motion compensation：掃描期間機器人在動，用 odom 速度×掃描齡補償點雲位移，
            # 避免高速時 sweep 出現拖影；dt_scan clamp 在 0.15s 內防 stale 幀過度補償
            motion_comp = None
            if self.lidar_motion_comp:
                dt_scan = max(min(pc_age, 0.15), 0.0)
                motion_comp = (v * dt_scan, 0.0, w * dt_scan)
            sweep_active = lidar_sweep_72_real(
                pc,
                r_max=self.lidar_r_max,
                r_robot=self.obs_params.robot_radius,
                r_min=self.lidar_r_min,
                z_filter=self.lidar_z_filter,
                num_bins=self.obs_params.lidar_num_bins,
                yaw_offset=self.lidar_yaw_offset,
                motion_compensation=motion_comp,
            )
        sweep = sweep_active
        with self._lock:
            self._last_sweep = sweep
            self._sweep_source = sweep_source_tag
            self._last_subgoal_body = (gx, gy)
            self._last_subgoal_source = f"{choice.source}/{robot_pose.source}"

        # 硬性緊急停車：繞過 policy，只要任一 bin 距離 < safety_estop 立刻 ESTOP。
        # 這是 policy 之外的最後一道安全網，不信任 policy 在極近距離的判斷。
        if self._too_close(sweep):
            self.mode_mgr.set(Mode.ESTOP, reason="LiDAR < safety_estop")
            return self._set_target_stop("EMERGENCY: LiDAR 進入安全區")

        # 時間膨脹（spot_rl speed_rate）：rate<1 時把感知量放大 1/rate，
        # 動作上限縮小 ×rate，policy 以為在原速世界 → 留在訓練分布內。
        rate = self.speed_rate
        inv = 1.0 / rate
        # goal 放大後 clamp 距離上限（避免超出訓練 goal 範圍，對齊 spot 的 18m cap）
        g_inflate_x, g_inflate_y = gx * inv, gy * inv
        g_dist = math.hypot(g_inflate_x, g_inflate_y)
        if g_dist > 18.0:
            s = 18.0 / g_dist
            g_inflate_x *= s
            g_inflate_y *= s

        # RL 推論（餵入「感知世界」的放大量）
        obs = build_obs_raw(
            self.bundle.raw_obs_dim,
            last_accel=last_accel * inv, linear_vel=v * inv, angular_vel=w * inv,
            goal_body_x=g_inflate_x, goal_body_y=g_inflate_y,
            lidar_sweep_72=sweep, elapsed_s=elapsed,
            params=self.obs_params,
        )
        logits = self.runner.step(obs)
        # 動作端：上限 ×rate 縮回實體速度；current_vel 用真實 v 做積分
        # （obs 端放大 1/rate、action 端縮小 ×rate 的不對稱，正是時間膨脹的本質：
        #  policy 在「放大的感知世界」決策，輸出再縮回真實世界的慢速指令）
        act_eff = (self.act_params if rate >= 0.999 else ActionParams(
            num_bins=self.act_params.num_bins,
            max_linear_velocity=self.act_params.max_linear_velocity * rate,
            max_linear_accel=self.act_params.max_linear_accel * rate,
            max_angular_velocity_action=self.act_params.max_angular_velocity_action * rate,
            dt=self.act_params.dt,
        ))
        cmd_v, cmd_w, accel = decode_logits_to_cmd(
            logits, current_linear_vel=v,
            params=act_eff, deterministic=self.deterministic,
        )

        with self._lock:
            self._last_accel = accel
            self._target_v = cmd_v
            self._target_w = cmd_w
            self._target_set_t = now

        if self.pub_obs is not None:
            m = Float32MultiArray()
            m.data = obs.tolist()
            self.pub_obs.publish(m)

    def _set_target_stop(self, reason: str, warn: bool = True) -> None:
        with self._lock:
            self._target_v = 0.0
            self._target_w = 0.0
        if warn:
            self.get_logger().warn(reason, throttle_duration_sec=2.0)

    def _too_close(self, sweep_norm: np.ndarray) -> bool:
        # sweep 是正規化值 [0,1]，這裡把 safety 閾值（公尺）換算成同樣的正規化尺度再比，
        # 避免每幀把整條 sweep 反正規化回公尺（省運算）
        denom = max(self.lidar_r_max - self.obs_params.robot_radius, 1e-6)
        thr_norm = (self.safety_estop_m - self.obs_params.robot_radius) / denom
        return bool((sweep_norm < thr_norm).any())

    # ──────────────────────────── Timer: cmd_vel 發布 (20 Hz) ────────────────────────────

    def _tick_cmd(self) -> None:
        # manual 模式 should_publish_cmd()=False → 完全不發，把 cmd_vel topic 讓給搖桿
        if not self.mode_mgr.should_publish_cmd():
            return
        now = time.monotonic()
        dt = max(now - self._cmd_last_t, 1e-3)
        self._cmd_last_t = now

        with self._lock:
            tgt_v = self._target_v
            tgt_w = self._target_w
        # idle/estop/paused 仍會發 cmd 但強制 0（持續宣告「我在控制且要停」給 mux）
        if self.mode_mgr.force_zero_cmd():
            tgt_v = 0.0
            tgt_w = 0.0

        # Inference 過期保護：cmd timer 跑得比推論快，若推論卡住（target 超過 5 個
        # cmd_dt 沒更新）就強制 0，避免一直 republish 最後一個過時的 target 而衝出去
        if (self.mode_mgr.mode == Mode.NAV
                and now - self._target_set_t > 5.0 / max(self.cmd_rate_hz, 1.0)):
            tgt_v, tgt_w = 0.0, 0.0

        # low-pass + slew-rate 平滑：把 5Hz 離散動作的跳階磨平再以 20Hz 送出
        out_v, out_w = self.cmd_filter.step(tgt_v, tgt_w, dt)
        msg = Twist()
        msg.linear.x = out_v
        msg.angular.z = out_w
        self.pub_cmd.publish(msg)

        # 餵延遲估計：拿「送出的 cmd」對「底盤實測 odom twist」做互相關，
        # 估 cmd→實際響應的死時間（rover_rl 無 cmd_delay 補償，靠這診斷振盪風險）
        with self._lock:
            act_v, act_w = self._odom_v, self._odom_w
        self.lag_v.push(out_v, act_v)
        self.lag_w.push(out_w, act_w)

    # ──────────────────────────── Timer: markers (10 Hz) ────────────────────────────

    def _tick_marker(self) -> None:
        if self.pub_markers is None:
            return
        with self._lock:
            sweep = self._last_sweep
            subgoal = self._last_subgoal_body
            source = self._last_subgoal_source
            out_v = self.cmd_filter._last_v
            out_w = self.cmd_filter._last_w

        # 找最近障礙物 bin 並反算回 (角度, 公尺) 給 RViz 畫；純視覺，不影響推論。
        # 反正規化：dist = norm × (r_max - r_robot) + r_robot；bin → 角度為 binning 的逆運算
        nearest = None
        if sweep is not None and sweep.size > 0:
            idx_min = int(np.argmin(sweep))
            denom = self.lidar_r_max - self.obs_params.robot_radius
            dist = float(sweep[idx_min]) * denom + self.obs_params.robot_radius
            angle = (2.0 * math.pi * idx_min / sweep.size) - math.pi
            nearest = (angle, dist)

        marr = build_marker_array(
            frame_id=self.base_frame,
            stamp=self.get_clock().now().to_msg(),
            robot_radius=self.obs_params.robot_radius,
            subgoal_xy=subgoal,
            subgoal_source=source,
            cmd_v=out_v, cmd_w=out_w,
            nearest_lidar_bin=nearest,
            mode=self.mode_mgr.mode.value,
        )
        self.pub_markers.publish(marr)

    # ──────────────────────────── Heartbeat (0.5 Hz) ────────────────────────────

    def _sweep_to_meters(self, sweep_norm) -> "np.ndarray | None":
        """正規化 sweep [0,1] → 實際距離 (m)；1.0=無回波/>=r_max。"""
        if sweep_norm is None or len(sweep_norm) == 0:
            return None
        r_robot = self.obs_params.robot_radius
        return np.asarray(sweep_norm, dtype=np.float32) * (self.lidar_r_max - r_robot) + r_robot

    def _publish_status(self) -> None:
        """以 ~5 Hz 發布精簡狀態 JSON，供 status_tui 儀表板渲染（不影響推論）。"""
        now = time.monotonic()
        with self._lock:
            # 用 _latest_sweep（topic 收到即有，idle 也更新），非僅推論時的 _last_sweep
            sweep = self._latest_sweep
            sweep_age = (now - self._latest_sweep_t
                         if self._latest_sweep is not None else float("inf"))
            odom_age = now - self._odom_t if self._odom_t else float("inf")
            tgt_v = self._target_v
            tgt_w = self._target_w
            sweep_src = self._sweep_source
            subgoal = self._last_subgoal_body
            subgoal_src = self._last_subgoal_source
            odom_x, odom_y = self._odom_xy
            odom_yaw = self._odom_yaw
            rl_v, rl_w = tgt_v, tgt_w
            act_v, act_w = self._odom_v, self._odom_w
            sent_v = self.cmd_filter._last_v
            sent_w = self.cmd_filter._last_w

        # sweep → 公尺；最近距離 + 四向扇區（bin: 36=前 54=左 18=右 0=後，每 bin 5°）
        nearest_m = front_m = back_m = left_m = right_m = None
        meters = self._sweep_to_meters(sweep)
        if meters is not None and len(meters) == 72:
            def _sec(idxs):
                v = float(np.min(meters[idxs]))
                return round(v, 2)
            nearest_m = round(float(np.min(meters)), 2)
            front_m = _sec(np.r_[28:45])
            left_m = _sec(np.r_[46:63])
            back_m = _sec(np.r_[64:72, 0:10])
            right_m = _sec(np.r_[10:27])

        # 機器人在 map frame 位姿（走 TF map→base；來源 tf / odom_only）
        rp = self._robot_pose_in_map(odom_x, odom_y, odom_yaw)
        # NDT 修正量（map→odom），純供顯示
        mo = self._map_odom_from_tf()
        goal_dist = goal_ang = None
        if subgoal is not None:
            goal_dist = math.hypot(subgoal[0], subgoal[1])
            goal_ang = math.degrees(math.atan2(subgoal[1], subgoal[0]))

        # 延遲：取線/角通道中相關係數較高者回報
        lag_v_s, corr_v = self.lag_v.estimate()
        lag_w_s, corr_w = self.lag_w.estimate()
        lag_ms = lag_corr = lag_ch = None
        cand = [(corr_v, lag_v_s, "v"), (corr_w, lag_w_s, "w")]
        cand = [c for c in cand if c[0] is not None]
        if cand:
            corr, lag_s, ch = max(cand, key=lambda c: c[0])
            lag_ms = round(lag_s * 1000.0)
            lag_corr = round(corr, 2)
            lag_ch = ch

        status = {
            "mode": self.mode_mgr.mode.value,
            "cmd_v": round(tgt_v, 3),
            "cmd_w": round(tgt_w, 3),
            "speed_rate": round(self.speed_rate, 2),
            # 三層速度：RL 想要 → 送出底盤 → 底盤實測
            "rl_v": round(rl_v, 3), "rl_w": round(rl_w, 3),
            "sent_v": round(sent_v, 3), "sent_w": round(sent_w, 3),
            "act_v": round(act_v, 3), "act_w": round(act_w, 3),
            # cmd 是否超出底盤實體上限（飽和 → 底盤做不到）
            "v_over": bool(abs(sent_v) > self.chassis_v_max * 0.99),
            "w_over": bool(abs(sent_w) > self.chassis_omega_max * 0.99),
            "chassis_v_max": self.chassis_v_max,
            "chassis_w_max": self.chassis_omega_max,
            # 延遲（送出 cmd ↔ 實測）
            "lag_ms": lag_ms, "lag_corr": lag_corr, "lag_ch": lag_ch,
            # RNN hidden state（episode 內記憶）
            "rnn_norm": round(self.runner.hidden_norm(), 2),
            "rnn_steps": self.runner.step_count,
            "rnn_resets": self.runner.reset_count,
            "lidar_age": round(sweep_age, 3) if sweep_age != float("inf") else None,
            "lidar_src": sweep_src,
            "nearest_m": nearest_m,
            "front_m": front_m, "back_m": back_m, "left_m": left_m, "right_m": right_m,
            "odom_age": round(odom_age, 3) if odom_age != float("inf") else None,
            "ndt_age": round(self.localizer.ndt_age_s, 2),
            "ndt_ok": bool(self.localizer.is_ndt_stable()),
            "pose_x": round(rp.x, 2), "pose_y": round(rp.y, 2),
            "pose_yaw_deg": round(math.degrees(rp.yaw), 1), "pose_src": rp.source,
            "off_x": round(mo[0], 2) if mo else None,
            "off_y": round(mo[1], 2) if mo else None,
            "off_yaw_deg": round(math.degrees(mo[2]), 1) if mo else None,
            "goal_dist": round(goal_dist, 2) if goal_dist is not None else None,
            "goal_ang_deg": round(goal_ang, 1) if goal_ang is not None else None,
            "goal_src": subgoal_src,
            "model": os.path.basename(self._model_path) if self._model_path else None,
        }
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)

    def _tick_heartbeat(self) -> None:
        now = time.monotonic()
        with self._lock:
            sweep_age = (now - self._latest_sweep_t
                         if self._latest_sweep is not None else float("inf"))
            pc_age = (now - self._latest_pc_t
                      if self._latest_pc is not None else float("inf"))
            odom_age = now - self._odom_t if self._odom_t else float("inf")
            tgt_v = self._target_v
            tgt_w = self._target_w
            sweep_src = self._sweep_source
        import math as _math
        ndt_age = self.localizer.ndt_age_s
        ndt_ok = self.localizer.is_ndt_stable()
        mo = self._map_odom_from_tf()

        def _h(age, limit):   # 健康標記：新鮮=✓，逾時=⚠
            return "✓" if age < limit else "⚠"

        # 資料健康度（lidar/odom/ndt）；inline 模式才額外顯示 raw pc
        health = [
            f"lidar {_h(sweep_age, 0.5)}{sweep_age:.2f}s",
            f"odom {_h(odom_age, 0.5)}{odom_age:.2f}s",
            f"ndt {'✓' if ndt_ok else '⚠'}{ndt_age:.1f}s",
        ]
        if sweep_src != "preprocessor_topic":
            health.insert(1, f"pc {_h(pc_age, 0.5)}{pc_age:.2f}s")
        off_str = (f"({mo[0]:+.2f},{mo[1]:+.2f},{_math.degrees(mo[2]):+.1f}°)"
                   if mo else "—")
        self.get_logger().info(
            f"[HB] {self.mode_mgr.mode.value:<6} cmd v={tgt_v:+.2f} w={tgt_w:+.2f} "
            f"(rate {self.speed_rate:.2f}) │ " + "  ".join(health) +
            f" │ map→odom {off_str}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 關閉時不主動補發 0 cmd：mode 切離 nav 與 cmd timer 停轉後底盤端 watchdog
        # 會在收不到 cmd 時自行停車；這裡只負責乾淨釋放 node 資源
        node.get_logger().info("⏹ policy_node 已停止（cmd_vel 已歸 0，安全關閉）")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
