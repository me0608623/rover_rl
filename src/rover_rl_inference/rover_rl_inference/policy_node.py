"""rover_rl_inference policy node (ROS 2 Humble).

訂閱：
    LiDAR (sensor_msgs/PointCloud2)
    Odometry (nav_msgs/Odometry)
    Goal pose (geometry_msgs/PoseStamped)

發布：
    cmd_vel (geometry_msgs/Twist)
    obs_debug (std_msgs/Float32MultiArray, optional)

控制週期：固定 dt=0.2s (5 Hz)；訊號中斷 > timeout 自動停車。

所有 topic 名 / frame / 參數均可由 ros2 param 覆寫，預設取自 params.yaml。
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray

from .action_decoder import ActionParams, decode_logits_to_cmd
from .lidar_preprocess import lidar_sweep_72_real, pointcloud2_to_xyz
from .model_runtime import PolicyRunner, load_bundle
from .obs_builder import ObsParams, build_obs_raw, world_goal_to_body


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("rover_rl_policy")

        # --- declare params ---
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cpu")  # "cpu" or "cuda:0"
        self.declare_parameter("control_dt", 0.2)
        self.declare_parameter("timeout_lidar_s", 0.5)
        self.declare_parameter("timeout_odom_s", 0.5)
        self.declare_parameter("deterministic", True)

        self.declare_parameter("topic_lidar", "/velodyne_points")
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_goal", "/goal_pose")
        self.declare_parameter("topic_cmd_vel", "/cmd_vel")
        self.declare_parameter("publish_obs_debug", False)

        self.declare_parameter("lidar_qos_best_effort", True)
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)
        self.declare_parameter("lidar_z_filter_m", 0.5)
        self.declare_parameter("lidar_r_min_m", 0.9)
        self.declare_parameter("lidar_r_max_m", 20.0)

        # obs normalizers (match training)
        self.declare_parameter("obs_max_acceleration", 1.0)
        self.declare_parameter("obs_max_linear_velocity", 1.0)
        self.declare_parameter("obs_max_angular_velocity", 1.5)
        self.declare_parameter("robot_radius_m", 0.35)
        self.declare_parameter("episode_horizon_s", 60.0)

        # action params (match training DiscreteDifferentialDriveActionCfg)
        self.declare_parameter("act_max_linear_velocity", 1.0)
        self.declare_parameter("act_max_linear_accel", 0.5)
        self.declare_parameter("act_max_angular_velocity", 2.0)

        # goal handling
        self.declare_parameter("goal_frame", "map")  # map / odom / base_link
        self.declare_parameter("goal_tolerance_m", 0.5)

        # safety
        self.declare_parameter("safety_lidar_emergency_stop_m", 0.30)

        # --- read params ---
        gp = self.get_parameter
        model_path = gp("model_path").get_parameter_value().string_value
        device = gp("device").get_parameter_value().string_value
        self.control_dt = float(gp("control_dt").value)
        self.timeout_lidar_s = float(gp("timeout_lidar_s").value)
        self.timeout_odom_s = float(gp("timeout_odom_s").value)
        self.deterministic = bool(gp("deterministic").value)

        topic_lidar = gp("topic_lidar").get_parameter_value().string_value
        topic_odom = gp("topic_odom").get_parameter_value().string_value
        topic_goal = gp("topic_goal").get_parameter_value().string_value
        topic_cmd_vel = gp("topic_cmd_vel").get_parameter_value().string_value
        self.publish_obs_debug = bool(gp("publish_obs_debug").value)

        self.lidar_yaw_offset = math.radians(float(gp("lidar_yaw_offset_deg").value))
        self.lidar_z_filter = float(gp("lidar_z_filter_m").value)
        self.lidar_r_min = float(gp("lidar_r_min_m").value)
        self.lidar_r_max = float(gp("lidar_r_max_m").value)

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
        self.goal_tolerance = float(gp("goal_tolerance_m").value)
        self.safety_estop_m = float(gp("safety_lidar_emergency_stop_m").value)

        # --- model ---
        if not model_path or not os.path.isfile(model_path):
            raise RuntimeError(
                f"model_path missing or invalid: {model_path!r}. "
                "Set ros2 param model_path to a torchscript bundle from export_policy.py"
            )
        self.bundle = load_bundle(model_path, device=device)
        self.runner = PolicyRunner(self.bundle)
        self.get_logger().info(
            f"loaded policy bundle: {model_path} on {device} "
            f"(raw_obs={self.bundle.raw_obs_dim}, used_obs={self.bundle.used_obs_dim}, "
            f"hidden={self.bundle.hidden_dim}, preprocess={self.bundle.preprocess_dim})"
        )

        # --- state ---
        self._lock = threading.Lock()
        self._latest_pc: np.ndarray | None = None
        self._latest_pc_t = 0.0
        self._odom_xy = (0.0, 0.0)
        self._odom_yaw = 0.0
        self._odom_v = 0.0
        self._odom_w = 0.0
        self._odom_t = 0.0
        self._goal_world: tuple[float, float] | None = None
        self._goal_frame_used: str | None = None
        self._last_accel = 0.0
        self._start_t = time.monotonic()

        # --- pubs/subs ---
        sensor_qos = QoSProfile(
            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if bool(gp("lidar_qos_best_effort").value)
                else QoSReliabilityPolicy.RELIABLE
            ),
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(PointCloud2, topic_lidar, self._cb_lidar, sensor_qos)
        self.create_subscription(Odometry, topic_odom, self._cb_odom, 10)
        self.create_subscription(PoseStamped, topic_goal, self._cb_goal, 10)
        self.pub_cmd = self.create_publisher(Twist, topic_cmd_vel, 10)
        self.pub_obs = (
            self.create_publisher(Float32MultiArray, "~/obs_debug", 10)
            if self.publish_obs_debug
            else None
        )

        self.timer = self.create_timer(self.control_dt, self._tick)
        self.get_logger().info(
            f"subs: lidar={topic_lidar}, odom={topic_odom}, goal={topic_goal} | "
            f"pub: cmd_vel={topic_cmd_vel}"
        )

    # ----- callbacks -----

    def _cb_lidar(self, msg: PointCloud2) -> None:
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
        with self._lock:
            self._goal_world = (msg.pose.position.x, msg.pose.position.y)
            self._goal_frame_used = msg.header.frame_id or self.goal_frame
            self.runner.reset()
            self._start_t = time.monotonic()
        self.get_logger().info(
            f"new goal in frame={self._goal_frame_used}: "
            f"({self._goal_world[0]:.2f}, {self._goal_world[1]:.2f}) — hidden state reset"
        )

    # ----- main loop -----

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            pc = self._latest_pc
            pc_age = now - self._latest_pc_t
            odom_age = now - self._odom_t
            odom_xy = self._odom_xy
            odom_yaw = self._odom_yaw
            v = self._odom_v
            w = self._odom_w
            goal_world = self._goal_world
            last_accel = self._last_accel
            elapsed = now - self._start_t

        if pc is None or pc_age > self.timeout_lidar_s:
            return self._publish_stop("lidar timeout")
        if odom_age > self.timeout_odom_s:
            return self._publish_stop("odom timeout")
        if goal_world is None:
            return self._publish_stop("no goal", warn=False)

        # goal → body
        gx, gy = world_goal_to_body(
            goal_world[0], goal_world[1], odom_xy[0], odom_xy[1], odom_yaw,
        )
        dist = math.hypot(gx, gy)
        if dist < self.goal_tolerance:
            self._publish_stop("goal reached", warn=False)
            return

        # lidar sweep
        sweep = lidar_sweep_72_real(
            pc,
            r_max=self.lidar_r_max,
            r_robot=self.obs_params.robot_radius,
            r_min=self.lidar_r_min,
            z_filter=self.lidar_z_filter,
            num_bins=self.obs_params.lidar_num_bins,
            yaw_offset=self.lidar_yaw_offset,
        )

        # emergency stop on hard obstacle
        if self._too_close(sweep):
            return self._publish_stop("emergency stop: obstacle within safety zone")

        obs = build_obs_raw(
            self.bundle.raw_obs_dim,
            last_accel=last_accel,
            linear_vel=v,
            angular_vel=w,
            goal_body_x=gx,
            goal_body_y=gy,
            lidar_sweep_72=sweep,
            elapsed_s=elapsed,
            params=self.obs_params,
        )

        logits = self.runner.step(obs)
        cmd_v, cmd_w, accel = decode_logits_to_cmd(
            logits,
            current_linear_vel=v,
            params=self.act_params,
            deterministic=self.deterministic,
        )

        with self._lock:
            self._last_accel = accel

        msg = Twist()
        msg.linear.x = cmd_v
        msg.angular.z = cmd_w
        self.pub_cmd.publish(msg)

        if self.pub_obs is not None:
            out = Float32MultiArray()
            out.data = obs.tolist()
            self.pub_obs.publish(out)

    def _too_close(self, sweep_norm: np.ndarray) -> bool:
        denom = max(self.lidar_r_max - self.obs_params.robot_radius, 1e-6)
        # normalized value at safety threshold (m from sensor center)
        thr_norm = (self.safety_estop_m - self.obs_params.robot_radius) / denom
        return bool((sweep_norm < thr_norm).any())

    def _publish_stop(self, reason: str, warn: bool = True) -> None:
        if warn:
            self.get_logger().warn(reason, throttle_duration_sec=1.0)
        msg = Twist()
        self.pub_cmd.publish(msg)


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
