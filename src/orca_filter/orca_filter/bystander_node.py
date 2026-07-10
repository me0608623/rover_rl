#!/usr/bin/env python3
"""orca_bystander_node — ORCA 旁觀測試節點(不控制車)。

訂 LV-DOT /vo_interface/tracked_obstacles + /odom,餵 ORCA solver,
印出「ORCA 建議速度 vs policy 偏好速度」+ RViz 畫箭頭,驗證 ORCA 在車端
用真實 LV-DOT 資料算出的閃躲方向合不合理。

不發 cmd_vel、不碰 policy_node、不碰 vo_interface。純觀察 + 可視化。

v_pref 策略(bystander 無 policy 輸出,故模擬「意圖速度」):
  - forward : 固定前進 v_pref_speed(測「直走時 ORCA 怎麼閃」)
  - current : 當前 odom 速度(測「ORCA 對當前狀態的修正」)
"""
import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from vo_interface.msg import TrackedObstacleArray

from .orca_core import ORCAFilter, ORCAInput


def _yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class ORCABystanderNode(Node):
    def __init__(self):
        super().__init__("orca_bystander")

        # ── topics / 頻率 ──
        self.declare_parameter("topic_obstacles", "/vo_interface/tracked_obstacles")
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("decision_hz", 5.0)
        # v_pref 策略
        self.declare_parameter("v_pref_mode", "forward")
        self.declare_parameter("v_pref_speed", 0.8)         # forward mode 速度(m/s)
        self.declare_parameter("min_vel_confidence", 0.0)   # 低於此 confidence 的 track 忽略
        # ORCA 參數(預設 = PC-A 1:1,non-cooperative)
        self.declare_parameter("robot_radius", 0.35)
        self.declare_parameter("safety_margin", 0.10)
        self.declare_parameter("max_speed", 1.0)
        self.declare_parameter("culling_radius", 5.0)
        self.declare_parameter("neighbor_dist", 10.0)
        self.declare_parameter("max_neighbors", 20)
        self.declare_parameter("time_horizon_dynamic", 2.5)
        self.declare_parameter("time_horizon_static", 0.3)
        self.declare_parameter("static_vel_threshold", 0.15)
        self.declare_parameter("angle_threshold_deg", 60.0)
        self.declare_parameter("omega_max", 0.25 * math.pi)
        self.declare_parameter("obs_inflation", 1.8)
        self.declare_parameter("ego_responsibility", 1.0)

        g = self.get_parameter
        self.frame_id = str(g("frame_id").value)
        self.v_pref_mode = str(g("v_pref_mode").value)
        self.v_pref_speed = float(g("v_pref_speed").value)
        self.min_vel_conf = float(g("min_vel_confidence").value)
        self.culling_radius = float(g("culling_radius").value)
        self.static_vel_threshold = float(g("static_vel_threshold").value)
        self.dt = 1.0 / float(g("decision_hz").value)

        self.filter = ORCAFilter(
            robot_radius=float(g("robot_radius").value),
            safety_margin=float(g("safety_margin").value),
            max_speed=float(g("max_speed").value),
            neighbor_dist=float(g("neighbor_dist").value),
            max_neighbors=int(g("max_neighbors").value),
            time_horizon_dynamic=float(g("time_horizon_dynamic").value),
            time_horizon_static=float(g("time_horizon_static").value),
            static_vel_threshold=self.static_vel_threshold,
            angle_threshold_deg=float(g("angle_threshold_deg").value),
            omega_max=float(g("omega_max").value),
            obs_inflation=float(g("obs_inflation").value),
            ego_responsibility=float(g("ego_responsibility").value),
        )

        self._lock = threading.Lock()
        self._obstacles_msg = None
        self._odom = None  # (x, y, yaw, v_fwd)

        self.create_subscription(TrackedObstacleArray, str(g("topic_obstacles").value),
                                 self._cb_obstacles, 10)
        self.create_subscription(Odometry, str(g("topic_odom").value),
                                 self._cb_odom, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/markers", 10)

        self.create_timer(self.dt, self._tick)
        self._intervene_count = 0
        self.get_logger().info(
            f"[orca_bystander] 訂 {g('topic_obstacles').value} + {g('topic_odom').value} "
            f"@ {1.0/self.dt:.1f}Hz | v_pref={self.v_pref_mode}({self.v_pref_speed}) | "
            f"culling={self.culling_radius}m | inflation={self.filter.obs_inflation} | "
            f"ego_resp={self.filter.ego_responsibility}")

    # ────────────────────────────────────────────────────────────────────────
    def _cb_obstacles(self, msg: TrackedObstacleArray):
        with self._lock:
            self._obstacles_msg = msg

    def _cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._lock:
            self._odom = (p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w),
                          msg.twist.twist.linear.x)

    # ────────────────────────────────────────────────────────────────────────
    def _tick(self):
        with self._lock:
            odom = self._odom
            obs_msg = self._obstacles_msg
        if odom is None:
            return
        rx, ry, yaw, v_fwd = odom
        # robot world velocity(差動底盤 body twist → world)
        robot_vx = math.cos(yaw) * v_fwd
        robot_vy = math.sin(yaw) * v_fwd

        # v_pref(bystander 無 policy,用模式模擬意圖速度)
        if self.v_pref_mode == "current":
            v_pref = (robot_vx, robot_vy)
            v_next_body = abs(v_fwd)
        else:  # forward
            v_pref = (math.cos(yaw) * self.v_pref_speed,
                      math.sin(yaw) * self.v_pref_speed)
            v_next_body = self.v_pref_speed

        # 組障礙物(world frame,已是 odom 絕對速度)+ culling
        obs_pos, obs_vel, obs_radii, is_static = self._gather_obstacles(obs_msg, rx, ry)

        inp = ORCAInput(
            robot_pos=(rx, ry), robot_vel=(robot_vx, robot_vy), yaw=yaw,
            v_pref=v_pref, v_next_body=v_next_body, omega_rl=0.0,
            obs_pos=obs_pos, obs_vel=obs_vel, obs_radii=obs_radii, obs_is_static=is_static,
        )
        out = self.filter.solve(inp, self.dt)

        if out.orca_active:
            self._intervene_count += 1
            if self._intervene_count % 5 == 1:
                n_dyn = int((~is_static).sum()) if len(is_static) else 0
                self.get_logger().info(
                    f"[INTERVENE #{self._intervene_count}] "
                    f"v_pref=({v_pref[0]:+.2f},{v_pref[1]:+.2f}) "
                    f"v_safe=({out.v_safe_world[0]:+.2f},{out.v_safe_world[1]:+.2f}) "
                    f"→ body v={out.v_safe_linear:+.2f} ω={out.v_safe_omega:+.2f} "
                    f"nearby={len(obs_pos)}(dyn{n_dyn})")

        self._publish_markers(rx, ry, v_pref, out, obs_pos, obs_radii, is_static)

    # ────────────────────────────────────────────────────────────────────────
    def _gather_obstacles(self, obs_msg, rx, ry):
        """TrackedObstacleArray → culled numpy arrays(world frame)。"""
        if obs_msg is None or not obs_msg.obstacles:
            z = np.zeros((0, 2))
            return z, z, np.zeros(0), np.zeros(0, dtype=bool)

        rows = []
        for ob in obs_msg.obstacles:
            if ob.vel_confidence < self.min_vel_conf:
                continue
            rows.append((ob.position.x, ob.position.y,
                         ob.velocity.x, ob.velocity.y, ob.radius))
        if not rows:
            z = np.zeros((0, 2))
            return z, z, np.zeros(0), np.zeros(0, dtype=bool)

        arr = np.array(rows, dtype=float)
        obs_pos = arr[:, :2]
        obs_vel = arr[:, 2:4]
        obs_radii = arr[:, 4]

        # culling:只保留 culling_radius 內、且非 self(dist>0.01)
        rel = obs_pos - np.array([rx, ry])
        dist = np.linalg.norm(rel, axis=1)
        mask = (dist < self.culling_radius) & (dist > 0.01)
        obs_pos, obs_vel, obs_radii = obs_pos[mask], obs_vel[mask], obs_radii[mask]

        is_static = (np.linalg.norm(obs_vel, axis=1) < self.static_vel_threshold
                     if len(obs_pos) else np.zeros(0, dtype=bool))
        return obs_pos, obs_vel, obs_radii, is_static

    # ────────────────────────────────────────────────────────────────────────
    def _publish_markers(self, rx, ry, v_pref, out, obs_pos, obs_radii, is_static):
        marr = MarkerArray()
        h = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)

        # v_pref 箭頭(綠)= 機器人想要的速度
        marr.markers.append(self._arrow(h, "v_pref", 0, rx, ry,
                                        v_pref[0], v_pref[1],
                                        ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.9)))
        # v_safe 箭頭(紅)= ORCA 建議(介入時才畫)
        if out.orca_active:
            marr.markers.append(self._arrow(h, "v_safe", 1, rx, ry,
                                            float(out.v_safe_world[0]),
                                            float(out.v_safe_world[1]),
                                            ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)))
        # 障礙物球(黃=動態,灰=靜態)
        for i, (pos, r) in enumerate(zip(obs_pos, obs_radii)):
            st = bool(is_static[i]) if i < len(is_static) else False
            m = Marker(header=h, ns="obstacle", id=i + 100,
                       type=Marker.SPHERE, action=Marker.ADD)
            m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=0.0)
            m.scale = Vector3(x=float(r * 2), y=float(r * 2), z=0.1)
            m.color = (ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.4) if st
                       else ColorRGBA(r=1.0, g=0.9, b=0.2, a=0.6))
            m.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
            marr.markers.append(m)

        self.marker_pub.publish(marr)

    def _arrow(self, h, ns, idd, x, y, vx, vy, color):
        m = Marker(header=h, ns=ns, id=idd, type=Marker.ARROW, action=Marker.ADD)
        m.scale = Vector3(x=0.06, y=0.12, z=0.0)  # shaft / head 直徑
        m.color = color
        m.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        m.points = [Point(x=float(x), y=float(y), z=0.2),
                    Point(x=float(x + vx), y=float(y + vy), z=0.2)]
        return m


def main(args=None):
    rclpy.init(args=args)
    node = ORCABystanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
