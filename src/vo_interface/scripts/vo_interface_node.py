#!/usr/bin/env python3
"""vo_interface_node — LV-DOT 動態障礙物 → Velocity Obstacle 介面。

純訂閱 onboard_detector 的 /dynamic_bboxes（MarkerArray，只含幾何），
自行做 constant-velocity Kalman 追蹤，輸出帶「持久 ID / 平滑絕對速度 / 協方差 /
track age」的 TrackedObstacleArray，供下游 VO 規劃器使用。完全不修改 LV-DOT。

為何不直接用 LV-DOT 的 box.Vx/Vy？
  ① LV-DOT 原生 box.id 是每幀索引、非持久（VO 要 ID 做時間平滑）；
  ② LiDAR-only 速度估計偏跳（本專案實測），VO 對速度跳動極敏感；
  ③ 自做 CV-KF 可 coasting 補平 dynamic_bboxes 的斷續發布 → 給 VO 更連續的障礙物流。
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray

from vo_interface.msg import TrackedObstacle, TrackedObstacleArray


class CVTrack:
    """單一障礙物的 constant-velocity Kalman 追蹤器。

    state x = [px, py, vx, vy]，量測 = 位置 (px, py)。
    速度與協方差直接從 KF 取得（協方差供 PVO，速度供 VO 碰撞錐）。
    """

    _next_id = 1

    def __init__(self, px, py, size, t, sigma_a, sigma_z):
        self.id = CVTrack._next_id
        CVTrack._next_id += 1
        self.x = np.array([px, py, 0.0, 0.0], dtype=float)
        # 初始：位置依量測噪音、速度大不確定（尚未觀測）
        self.P = np.diag([sigma_z ** 2, sigma_z ** 2, 4.0, 4.0])
        self.size = np.array(size, dtype=float)  # x_width, y_width, z_width
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        self.t_last = t
        self.t_created = t
        self.hits = 1
        self.misses = 0

    def predict(self, t):
        dt = t - self.t_last
        self.t_last = t
        dt = min(max(dt, 1e-3), 0.5)  # 夾住異常 dt（啟動/長斷流）
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        # constant-acceleration 過程噪音（白噪加速度模型）
        q = self.sigma_a ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = np.array([[dt4 / 4, 0, dt3 / 2, 0],
                      [0, dt4 / 4, 0, dt3 / 2],
                      [dt3 / 2, 0, dt2, 0],
                      [0, dt3 / 2, 0, dt2]], dtype=float) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, px, py, size, alpha_size=0.4):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = np.diag([self.sigma_z ** 2, self.sigma_z ** 2])
        z = np.array([px, py], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        # 尺寸用 EMA 平滑（避免框抖動讓半徑跳）
        self.size = alpha_size * np.array(size) + (1 - alpha_size) * self.size
        self.hits += 1
        self.misses = 0

    @property
    def pos(self):
        return self.x[0], self.x[1]

    @property
    def vel(self):
        return self.x[2], self.x[3]

    def age(self, t):
        return t - self.t_created


class VOInterfaceNode(Node):
    def __init__(self):
        super().__init__("vo_interface")

        self.declare_parameter("input_topic", "/onboard_detector/dynamic_bboxes")
        self.declare_parameter("output_topic", "/vo_interface/tracked_obstacles")
        self.declare_parameter("marker_topic", "/vo_interface/markers")
        self.declare_parameter("frame_id", "odom")          # fallback；優先用 marker 自帶 frame
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("assoc_max_dist", 0.7)       # 配對最大距離 (m)
        self.declare_parameter("min_hits", 3)               # 連續命中幾次才算確認 track（才輸出）
        self.declare_parameter("max_misses", 10)            # 連續漏配幾次才刪除（coasting 上限）
        self.declare_parameter("process_noise_accel", 2.0)  # sigma_a (m/s^2)：人加減速的尺度
        self.declare_parameter("meas_noise_pos", 0.10)      # sigma_z (m)：偵測位置噪音
        self.declare_parameter("vel_confidence_time", 0.5)  # 達到滿可信度所需的 track age (s)
        self.declare_parameter("min_radius", 0.2)           # 半徑下限 (m)，避免太小框讓 VO 漏避
        self.declare_parameter("publish_markers", True)

        g = self.get_parameter
        self.input_topic = g("input_topic").value
        self.output_topic = g("output_topic").value
        self.marker_topic = g("marker_topic").value
        self.frame_id_fallback = g("frame_id").value
        self.rate_hz = float(g("publish_rate_hz").value)
        self.assoc_max_dist = float(g("assoc_max_dist").value)
        self.min_hits = int(g("min_hits").value)
        self.max_misses = int(g("max_misses").value)
        self.sigma_a = float(g("process_noise_accel").value)
        self.sigma_z = float(g("meas_noise_pos").value)
        self.vel_conf_time = float(g("vel_confidence_time").value)
        self.min_radius = float(g("min_radius").value)
        self.publish_markers = bool(g("publish_markers").value)

        self.tracks = []          # list[CVTrack]
        self.latest_dets = None   # (frame_id, [(cx,cy,size3), ...]) 最近一筆量測
        self.det_frame_id = self.frame_id_fallback

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(MarkerArray, self.input_topic,
                                            self._markers_cb, sensor_qos)
        self.pub = self.create_publisher(TrackedObstacleArray, self.output_topic, 10)
        self.marker_pub = (self.create_publisher(MarkerArray, self.marker_topic, 10)
                           if self.publish_markers else None)

        # 固定頻率 predict+publish（與輸入解耦，輸入斷流時靠 coasting）
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"[vo_interface] 訂閱 {self.input_topic} → 發布 {self.output_topic} "
            f"@ {self.rate_hz:.0f}Hz；assoc<{self.assoc_max_dist}m, min_hits={self.min_hits}")

    # ---- 量測：把 LINE_LIST 邊界框 marker 還原成 (中心, 尺寸) ----
    def _markers_cb(self, msg: MarkerArray):
        dets = []
        frame = self.frame_id_fallback
        for m in msg.markers:
            if m.header.frame_id:
                frame = m.header.frame_id
            if not m.points:
                continue
            cx = m.pose.position.x
            cy = m.pose.position.y
            xs = [p.x for p in m.points]
            ys = [p.y for p in m.points]
            zs = [p.z for p in m.points]
            size = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            dets.append((cx, cy, size))
        self.det_frame_id = frame
        self.latest_dets = dets

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- 主迴圈：predict → 配對 latest_dets → update/coast → publish ----
    def _tick(self):
        t = self._now()
        for trk in self.tracks:
            trk.predict(t)

        dets = self.latest_dets
        self.latest_dets = None  # 消費掉，下一拍沒新量測就純 coasting

        if dets is not None:
            self._associate_and_update(dets, t)

        # 刪除久未命中的 track
        self.tracks = [trk for trk in self.tracks if trk.misses <= self.max_misses]

        self._publish(t)

    def _associate_and_update(self, dets, t):
        # greedy 最近鄰配對（障礙物數量少，夠用且穩定）
        unmatched_dets = set(range(len(dets)))
        pairs = []  # (dist, trk_idx, det_idx)
        for ti, trk in enumerate(self.tracks):
            tx, ty = trk.pos
            for di, (cx, cy, _s) in enumerate(dets):
                d = math.hypot(cx - tx, cy - ty)
                if d <= self.assoc_max_dist:
                    pairs.append((d, ti, di))
        pairs.sort(key=lambda p: p[0])

        used_trk, used_det = set(), set()
        for d, ti, di in pairs:
            if ti in used_trk or di in used_det:
                continue
            used_trk.add(ti)
            used_det.add(di)
            unmatched_dets.discard(di)
            cx, cy, size = dets[di]
            self.tracks[ti].update(cx, cy, size)

        # 未命中的既有 track → coasting（predict 已做，這裡只記 miss）
        for ti, trk in enumerate(self.tracks):
            if ti not in used_trk:
                trk.misses += 1

        # 未配對到的偵測 → 新生 track
        for di in unmatched_dets:
            cx, cy, size = dets[di]
            self.tracks.append(CVTrack(cx, cy, size, t, self.sigma_a, self.sigma_z))

    def _vel_confidence(self, trk, t):
        # 由 track age 與速度協方差跡共同決定（年輕或速度不確定 → 低）
        age_term = min(1.0, trk.age(t) / max(self.vel_conf_time, 1e-3))
        vel_var = trk.P[2, 2] + trk.P[3, 3]
        cov_term = math.exp(-vel_var / 2.0)
        return max(0.0, min(1.0, age_term * cov_term))

    def _publish(self, t):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.det_frame_id

        # RViz marker 專用 header：不設 stamp(=0)→ RViz 用「最新可用 TF」轉換，避免 Fixed Frame=map 時
        # marker 的 now() 比 NDT map→odom(~10Hz 有延遲) 新→暫時 transform 不出來閃錯。
        # （TrackedObstacleArray 資料訊息仍用 now() header，給 VO 規劃器做精確時間對齊用。）
        marker_header = Header()
        marker_header.frame_id = self.det_frame_id

        out = TrackedObstacleArray()
        out.header = header
        markers = MarkerArray()

        for trk in self.tracks:
            if trk.hits < self.min_hits:
                continue  # 未確認 track 不輸出（速度尚未收斂）
            px, py = trk.pos
            vx, vy = trk.vel
            radius = max(self.min_radius, 0.5 * math.hypot(trk.size[0], trk.size[1]))

            ob = TrackedObstacle()
            ob.id = int(trk.id)
            ob.age = float(trk.age(t))
            ob.position = Point(x=float(px), y=float(py), z=0.0)
            ob.velocity = Vector3(x=float(vx), y=float(vy), z=0.0)
            ob.size = Vector3(x=float(trk.size[0]), y=float(trk.size[1]), z=float(trk.size[2]))
            ob.radius = float(radius)
            ob.vel_confidence = float(self._vel_confidence(trk, t))
            ob.position_covariance = [float(trk.P[0, 0]), float(trk.P[0, 1]),
                                      float(trk.P[1, 0]), float(trk.P[1, 1])]
            ob.velocity_covariance = [float(trk.P[2, 2]), float(trk.P[2, 3]),
                                      float(trk.P[3, 2]), float(trk.P[3, 3])]
            out.obstacles.append(ob)

            if self.marker_pub is not None:
                markers.markers.extend(self._track_markers(trk, marker_header, px, py, vx, vy, radius))

        self.pub.publish(out)
        if self.marker_pub is not None:
            self._add_deletion_if_needed(markers, header)
            self.marker_pub.publish(markers)

    # ---- RViz 視覺化：速度箭頭 + ID/age 文字 ----
    def _track_markers(self, trk, header, px, py, vx, vy, radius):
        out = []
        arrow = Marker()
        arrow.header = header
        arrow.ns = "vo_velocity"
        arrow.id = int(trk.id)
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.06   # shaft 直徑
        arrow.scale.y = 0.12   # head 直徑
        arrow.scale.z = 0.0
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.3, 0.9, 1.0
        arrow.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        arrow.points = [Point(x=float(px), y=float(py), z=0.2),
                        Point(x=float(px + vx), y=float(py + vy), z=0.2)]
        out.append(arrow)

        txt = Marker()
        txt.header = header
        txt.ns = "vo_label"
        txt.id = int(trk.id)
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position = Point(x=float(px), y=float(py), z=float(radius + 0.5))
        txt.scale.z = 0.25
        txt.color.r, txt.color.g, txt.color.b, txt.color.a = 1.0, 1.0, 1.0, 1.0
        txt.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        spd = math.hypot(vx, vy)
        txt.text = f"#{trk.id} {spd:.1f}m/s {trk.age(self._now()):.1f}s"
        out.append(txt)
        return out

    def _add_deletion_if_needed(self, markers, header):
        # 簡化：靠 lifetime 0.3s 自動消，不主動 DELETEALL（避免閃爍）
        return


def main(args=None):
    rclpy.init(args=args)
    node = VOInterfaceNode()
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
