"""rover_rl_inference policy node — full deployment 版.

整合元件：
  - localization.MapOdomOffsetTracker  : NDT cached offset + fallback
  - subgoal_selector.SubgoalSelector   : Path/PoseStamped → carrot lookahead
  - cmd_filter.CmdFilter               : low-pass + slew-rate
  - mode_manager.ModeManager           : nav / idle / estop / manual / paused
  - markers                            : RViz MarkerArray debug
  - lidar_preprocess.lidar_sweep_72_real (+ motion compensation)

Timer 結構：
  - inference_timer (5 Hz):  RL 推論 → 更新 target cmd
  - cmd_timer (20 Hz):       low-pass/slew filter → 發 cmd_vel
  - marker_timer (10 Hz):    發 RViz markers
  - heartbeat_timer (1 Hz):  log 狀態

Mode 切換：
  - 訂閱 `~/mode` (std_msgs/String): "nav" / "idle" / "estop" / "manual" / "paused"
  - 服務 `~/set_mode` (std_srvs/SetBool): true=nav, false=idle
  - 服務 `~/load_model` (std_srvs/Trigger): 重載 model_path 指向的 .ts
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import MarkerArray

from .action_decoder import ActionParams, decode_logits_to_cmd
from .cmd_filter import CmdFilter, CmdFilterParams
from .lidar_preprocess import lidar_sweep_72_real, pointcloud2_to_xyz
from .localization import MapOdomOffsetTracker, world_to_body
from .markers import build_marker_array
from .mode_manager import Mode, ModeManager
from .model_runtime import PolicyRunner, load_bundle
from .obs_builder import ObsParams, build_obs_raw
from .subgoal_selector import SubgoalSelector


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PolicyNode(Node):
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
        self.declare_parameter("obs_max_acceleration", 1.0)
        self.declare_parameter("obs_max_linear_velocity", 1.0)
        self.declare_parameter("obs_max_angular_velocity", 1.5)
        self.declare_parameter("robot_radius_m", 0.35)
        self.declare_parameter("episode_horizon_s", 60.0)
        # action limits (鎖在訓練分布)
        self.declare_parameter("act_max_linear_velocity", 1.0)
        self.declare_parameter("act_max_linear_accel", 0.5)
        self.declare_parameter("act_max_angular_velocity", 2.0)
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
        if not self._model_path or not os.path.isfile(self._model_path):
            raise RuntimeError(
                f"model_path 無效: {self._model_path!r}。"
                "請 export_policy.py 後設定為 .ts 絕對路徑"
            )
        self.bundle = load_bundle(self._model_path, device=self._device)
        self.runner = PolicyRunner(self.bundle)

    # ──────────────────────────── 狀態與元件 ────────────────────────────

    def _init_state(self) -> None:
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
        self._target_v = 0.0
        self._target_w = 0.0
        self._target_set_t = 0.0
        # 上次 sweep（給 marker 用）
        self._last_sweep: np.ndarray | None = None
        # 上次 subgoal（給 marker 用）
        self._last_subgoal_body: tuple[float, float] | None = None
        self._last_subgoal_source: str | None = None
        # 元件
        self.localizer = MapOdomOffsetTracker(logger=self.get_logger())
        self.subgoals = SubgoalSelector(lookahead_m=self.path_lookahead)
        self.cmd_filter = CmdFilter(self.cmd_filter_params)
        try:
            initial = Mode.parse(self._initial_mode)
        except ValueError:
            self.get_logger().warn(f"未知 initial_mode={self._initial_mode!r}，使用 nav")
            initial = Mode.NAV
        self.mode_mgr = ModeManager(initial=initial, on_change=self._on_mode_change)

    # ──────────────────────────── ROS 介面 ────────────────────────────

    def _init_pubsub(self) -> None:
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

    def _init_timers(self) -> None:
        self.timer_infer = self.create_timer(self.control_dt, self._tick_inference)
        self.timer_cmd = self.create_timer(1.0 / max(self.cmd_rate_hz, 1.0), self._tick_cmd)
        if self.publish_markers:
            self.timer_marker = self.create_timer(
                1.0 / max(self.marker_rate_hz, 1.0), self._tick_marker,
            )
        self.timer_heartbeat = self.create_timer(2.0, self._tick_heartbeat)
        self._cmd_last_t = time.monotonic()

    def _init_services(self) -> None:
        self.create_service(SetBool, "~/set_mode", self._srv_set_mode)
        self.create_service(Trigger, "~/load_model", self._srv_load_model)
        self.create_service(Trigger, "~/reset_hidden", self._srv_reset_hidden)

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
        with self._lock:
            self._odom_xy = (p.x, p.y)
            self._odom_yaw = yaw
            self._odom_v = tw.linear.x
            self._odom_w = tw.angular.z
            self._odom_t = time.monotonic()

    def _cb_goal(self, msg: PoseStamped) -> None:
        frame = msg.header.frame_id or self.goal_frame
        self.subgoals.set_single_goal(msg.pose.position.x, msg.pose.position.y, frame)
        self.runner.reset()
        self.cmd_filter.reset()
        with self._lock:
            self._start_t = time.monotonic()
        self.get_logger().info(
            f"收到 goal_pose frame={frame} ({msg.pose.position.x:.2f},{msg.pose.position.y:.2f})"
        )

    def _cb_path(self, msg: Path) -> None:
        if not msg.poses:
            self.subgoals.clear_path()
            return
        frame = msg.header.frame_id or self.goal_frame
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.subgoals.set_path(pts, frame)
        self.subgoals.prefer_path = True
        self.runner.reset()
        self.cmd_filter.reset()
        with self._lock:
            self._start_t = time.monotonic()
        self.get_logger().info(
            f"收到 path frame={frame} ({len(pts)} waypoints, lookahead={self.path_lookahead}m)"
        )

    def _cb_ndt_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            ox, oy = self._odom_xy
            speed = self._odom_v
        self.localizer.on_ndt_pose(msg.pose.position.x, msg.pose.position.y, ox, oy)
        self.localizer.try_update_offset(robot_speed_mps=speed)

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
        if new in (Mode.IDLE, Mode.ESTOP, Mode.PAUSED):
            with self._lock:
                self._target_v = 0.0
                self._target_w = 0.0
            self.cmd_filter.reset()

    # ──────────────────────────── Timer: 推論 (5 Hz) ────────────────────────────

    def _tick_inference(self) -> None:
        if not self.mode_mgr.is_active():
            return

        now = time.monotonic()
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
        if odom_age > self.timeout_odom_s:
            return self._set_target_stop("Odom timeout")
        if not self.subgoals.has_target():
            return self._set_target_stop("尚未收到 goal/path", warn=False)

        # 機器人在 map frame 的位姿
        robot_pose = self.localizer.get_robot_pose_in_map(odom_x, odom_y, odom_yaw)
        if self.require_ndt and robot_pose.source == "odom_only":
            return self._set_target_stop("require_ndt=true 但 NDT 未穩定")

        # 取得 sub-goal（lookahead 或 single）
        choice = self.subgoals.select(robot_pose.x, robot_pose.y)
        if choice is None:
            return self._set_target_stop("無法選定 subgoal")

        # 若 subgoal 與 robot 在不同 frame：goal_frame=odom 強制用 odom_only pose
        if choice.frame == "odom":
            robot_x, robot_y, robot_yaw_use = odom_x, odom_y, odom_yaw
        else:
            robot_x, robot_y, robot_yaw_use = robot_pose.x, robot_pose.y, robot_pose.yaw

        gx, gy = world_to_body(choice.x, choice.y, robot_x, robot_y, robot_yaw_use)
        dist = math.hypot(gx, gy)
        if choice.source != "path_lookahead" and dist < self.goal_tolerance:
            return self._set_target_stop(
                f"到達 {choice.source} (dist={dist:.2f})", warn=False,
            )

        # LiDAR sweep 取得
        if sweep_active is None:
            # inline fallback：自己從 PointCloud2 處理
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

        # 緊急停車（在 normalize 前的 raw m）
        if self._too_close(sweep):
            self.mode_mgr.set(Mode.ESTOP, reason="LiDAR < safety_estop")
            return self._set_target_stop("EMERGENCY: LiDAR 進入安全區")

        # RL 推論
        obs = build_obs_raw(
            self.bundle.raw_obs_dim,
            last_accel=last_accel, linear_vel=v, angular_vel=w,
            goal_body_x=gx, goal_body_y=gy,
            lidar_sweep_72=sweep, elapsed_s=elapsed,
            params=self.obs_params,
        )
        logits = self.runner.step(obs)
        cmd_v, cmd_w, accel = decode_logits_to_cmd(
            logits, current_linear_vel=v,
            params=self.act_params, deterministic=self.deterministic,
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
        denom = max(self.lidar_r_max - self.obs_params.robot_radius, 1e-6)
        thr_norm = (self.safety_estop_m - self.obs_params.robot_radius) / denom
        return bool((sweep_norm < thr_norm).any())

    # ──────────────────────────── Timer: cmd_vel 發布 (20 Hz) ────────────────────────────

    def _tick_cmd(self) -> None:
        if not self.mode_mgr.should_publish_cmd():
            return
        now = time.monotonic()
        dt = max(now - self._cmd_last_t, 1e-3)
        self._cmd_last_t = now

        with self._lock:
            tgt_v = self._target_v
            tgt_w = self._target_w
        if self.mode_mgr.force_zero_cmd():
            tgt_v = 0.0
            tgt_w = 0.0

        # Inference 過期保護：若 target 已 > 5 個 cmd_dt 沒更新 → 強制 0
        if (self.mode_mgr.mode == Mode.NAV
                and now - self._target_set_t > 5.0 / max(self.cmd_rate_hz, 1.0)):
            tgt_v, tgt_w = 0.0, 0.0

        out_v, out_w = self.cmd_filter.step(tgt_v, tgt_w, dt)
        msg = Twist()
        msg.linear.x = out_v
        msg.angular.z = out_w
        self.pub_cmd.publish(msg)

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
        ndt_age = self.localizer.ndt_age_s
        ndt_ok = "yes" if self.localizer.is_ndt_stable() else "no"
        off = self.localizer.offset
        off_str = f"({off[0]:+.2f},{off[1]:+.2f})" if off else "—"
        self.get_logger().info(
            f"[HB] mode={self.mode_mgr.mode.value} sweep_src={sweep_src} | "
            f"sweep_age={sweep_age:.2f}s pc_age={pc_age:.2f}s odom_age={odom_age:.2f}s "
            f"ndt={ndt_ok}(age={ndt_age:.1f}s,offset={off_str}) | "
            f"target v={tgt_v:+.2f} w={tgt_w:+.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
