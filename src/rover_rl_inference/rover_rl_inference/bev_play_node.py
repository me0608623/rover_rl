"""bev_play_node — 對齊 play_rnn_car.py LiveBEVVisualizer 風格.

訓練端 play 看到的 BEV (scripts/.../play_eval/play_rnn_car.py::LiveBEVVisualizer)：

  ┌─────────────────────────────────────────────┐ ← text_ax (上方資料面板)
  │ Charge RL BEV — 72-bin LiDAR (Body)         │
  │ AGENT/ACTION   LIDAR/GOAL    MODE / STATE   │
  │   Step       │   Nearest    │  Status       │
  │   Raw idx    │   Mean       │  Pose src     │
  │   v          │   <2m/2~5    │  Hidden       │
  │   omega      │   Goal body  │  Sweep src    │
  │   accel      │   Goal dist  │  Frame yaw    │
  ├─────────────────────────────────────────────┤
  │                                             │ ← ax (BEV 視覺層)
  │     · · · · · · · ·                         │
  │   · · · 距離環 ·  · · ·                      │
  │  ·    + 黃箭頭→  · ·                         │
  │  ·    白機器人      ·                        │
  │   · · · trail · · ·                          │
  │     · · · · · · · ·                         │
  └─────────────────────────────────────────────┘

部署版差異（vs 訓練端）：
  - 移除 ORCA/RVO2 column → 改 MODE/STATE（rover_rl 模式狀態）
  - 移除 termination goal（無 ground truth）→ 只顯示 obs 中 subgoal
  - 移除多 goal 場景（無 simulator）
  - Trail 改從 /odom 取（不是 simulator ground truth）
  - Agg backend headless render → sensor_msgs/Image

訂閱:
  /rover_rl/lidar_sweep_72   Float32MultiArray[72]
  /rover_rl_policy/obs_debug Float32MultiArray[79]   (optional, 取 goal body)
  /input/nav_cmd_vel         Twist                     (optional, 動作)
  /odom                      Odometry                   (optional, yaw + trail)
  /rover_rl_policy/mode      可選 (status badge)

發布:
  /rover_rl/bev_image  sensor_msgs/Image (rgb8, ~5 Hz)
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, String
import sensor_msgs_py.point_cloud2 as pc2

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


class BevPlayNode(Node):
    def __init__(self):
        super().__init__("rover_rl_bev_play")

        self.declare_parameter("topic_sweep", "/rover_rl/lidar_sweep_72")
        self.declare_parameter("topic_obs_debug", "/rover_rl_policy/obs_debug")
        self.declare_parameter("topic_cmd_vel", "/input/nav_cmd_vel")
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_mode", "/rover_rl_policy/mode")
        self.declare_parameter("topic_image", "/rover_rl/bev_image")
        self.declare_parameter("frame_mode", "body")         # 'body' / 'world'
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("r_max", 20.0)
        self.declare_parameter("r_robot", 0.35)
        self.declare_parameter("trail_length", 500)
        self.declare_parameter("figure_dpi", 90)
        self.declare_parameter("figure_w", 7.5)
        self.declare_parameter("figure_h", 7.2)

        gp = self.get_parameter
        self.frame_mode = gp("frame_mode").get_parameter_value().string_value
        self.rate_hz = float(gp("rate_hz").value)
        self.r_max = float(gp("r_max").value)
        self.r_robot = float(gp("r_robot").value)
        self.trail_length = int(gp("trail_length").value)
        self.dpi = int(gp("figure_dpi").value)
        self.fig_w = float(gp("figure_w").value)
        self.fig_h = float(gp("figure_h").value)

        # ── 狀態 ──
        self._lock = threading.Lock()
        self._sweep: np.ndarray | None = None
        self._obs_79: np.ndarray | None = None
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0
        self._odom_xy = (0.0, 0.0)
        self._odom_yaw = 0.0
        self._mode_str = "unknown"
        self._step = 0
        self._trail: deque[tuple[float, float]] = deque(maxlen=max(self.trail_length, 1))
        # sweep / odom 來源狀態
        self._sweep_t = 0.0
        self._odom_t = 0.0
        # goal_pose（直接訂閱，不依賴 obs_debug）
        self._goal_map: tuple[float, float] | None = None   # map frame (x, y)
        # 原始點雲最近距離（不受 r_min 過濾）
        self._raw_nearest: float | None = None
        self._raw_nearest_ang: float | None = None

        # ── matplotlib figure（重用） ──
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans", "Noto Sans CJK JP", "Noto Sans CJK TC",
            "WenQuanYi Micro Hei",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        self.fig = plt.figure(figsize=(self.fig_w, self.fig_h),
                              constrained_layout=True, dpi=self.dpi)
        gs = self.fig.add_gridspec(nrows=2, ncols=1,
                                    height_ratios=[1.8, 5.2], hspace=0.03)
        self.text_ax = self.fig.add_subplot(gs[0])
        self.ax = self.fig.add_subplot(gs[1])
        self.fig.patch.set_facecolor("#141414")

        # ── 訂閱 / 發布 ──
        self.create_subscription(Float32MultiArray,
            gp("topic_sweep").get_parameter_value().string_value, self._cb_sweep, 10)
        self.create_subscription(Float32MultiArray,
            gp("topic_obs_debug").get_parameter_value().string_value, self._cb_obs, 10)
        self.create_subscription(Twist,
            gp("topic_cmd_vel").get_parameter_value().string_value, self._cb_cmd, 10)
        self.create_subscription(Odometry,
            gp("topic_odom").get_parameter_value().string_value, self._cb_odom, 10)
        self.create_subscription(String,
            gp("topic_mode").get_parameter_value().string_value, self._cb_mode, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._cb_goal, 10)
        self.create_subscription(PointCloud2, "/velodyne_points", self._cb_pc, 5)

        self.pub_image = self.create_publisher(Image,
            gp("topic_image").get_parameter_value().string_value, 5)

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._tick)
        self.get_logger().info(
            f"rover_rl_bev_play (LiveBEVVisualizer style)\n"
            f"  訂閱 sweep={gp('topic_sweep').value}, obs={gp('topic_obs_debug').value}, "
            f"cmd={gp('topic_cmd_vel').value}, odom={gp('topic_odom').value}, "
            f"mode={gp('topic_mode').value}\n"
            f"  發布 image={gp('topic_image').value} "
            f"({int(self.fig_w*self.dpi)}×{int(self.fig_h*self.dpi)} px @ {self.rate_hz}Hz)"
        )

    # ── callbacks ──
    def _cb_sweep(self, msg: Float32MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        with self._lock:
            self._sweep = arr
            self._sweep_t = time.monotonic()

    def _cb_obs(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 6:
            with self._lock:
                self._obs_79 = np.asarray(msg.data, dtype=np.float32)

    def _cb_cmd(self, msg: Twist) -> None:
        with self._lock:
            self._last_cmd_v = float(msg.linear.x)
            self._last_cmd_w = float(msg.angular.z)

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._lock:
            self._odom_xy = (p.x, p.y)
            self._odom_yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
            self._odom_t = time.monotonic()
            # trail
            if self._trail:
                last = self._trail[-1]
                d2 = (p.x - last[0]) ** 2 + (p.y - last[1]) ** 2
                if d2 > 4.0:   # 跳躍 > 2m 視為 reset → 清空
                    self._trail.clear()
            self._trail.append((p.x, p.y))

    def _cb_mode(self, msg: String) -> None:
        with self._lock:
            self._mode_str = msg.data or "unknown"

    def _cb_goal(self, msg: PoseStamped) -> None:
        with self._lock:
            self._goal_map = (msg.pose.position.x, msg.pose.position.y)

    def _cb_pc(self, msg: PointCloud2) -> None:
        try:
            pts = np.array(list(pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)))
            if pts.shape[0] == 0:
                return
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            # z_filter: 過濾地面與高處（與 preprocessor 對齊）
            mask = np.abs(z) < 0.5
            if mask.sum() == 0:
                return
            x, y = x[mask], y[mask]
            d = np.sqrt(x**2 + y**2)
            idx = int(d.argmin())
            nearest = float(d[idx])
            ang = float(math.degrees(math.atan2(y[idx], x[idx])))
            with self._lock:
                self._raw_nearest = nearest
                self._raw_nearest_ang = ang
        except Exception:
            pass

    # ── main tick ──
    def _tick(self) -> None:
        with self._lock:
            sweep = self._sweep
            obs = self._obs_79
            cmd_v = self._last_cmd_v
            cmd_w = self._last_cmd_w
            yaw = self._odom_yaw
            xy = self._odom_xy
            trail = list(self._trail)
            mode = self._mode_str
            goal_map = self._goal_map
            raw_nearest = self._raw_nearest
            raw_nearest_ang = self._raw_nearest_ang
            self._step += 1
            step = self._step
            sweep_age = time.monotonic() - self._sweep_t if self._sweep_t else float("inf")
            odom_age = time.monotonic() - self._odom_t if self._odom_t else float("inf")
        if sweep is None:
            return

        # goal_map → body frame（用 odom 近似，NDT 未收斂時仍有參考價值）
        goal_body: tuple[float, float] | None = None
        if goal_map is not None:
            dx = goal_map[0] - xy[0]
            dy = goal_map[1] - xy[1]
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            gx_body = cos_y * dx + sin_y * dy   # forward
            gy_body = -sin_y * dx + cos_y * dy  # left
            goal_body = (gx_body, gy_body)

        self._draw_bev(sweep, obs, yaw, xy, trail, goal_body)
        self._draw_top_panel(step, cmd_v, cmd_w, sweep, obs, mode,
                              yaw, sweep_age, odom_age, goal_body,
                              raw_nearest, raw_nearest_ang)
        msg = self._fig_to_image_msg()
        self.pub_image.publish(msg)

    # ── BEV plot ──
    def _draw_bev(self, sweep_norm, obs_79, yaw, robot_xy, trail, goal_body=None):
        ax = self.ax
        ax.cla()
        ax.set_facecolor("#141414")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.r_max, self.r_max)
        ax.set_ylim(-self.r_max, self.r_max)
        if self.frame_mode == "world":
            xlabel, ylabel = "World X rel. robot (m)", "World Y rel. robot (m)"
        else:
            xlabel, ylabel = "Left-Right y (m)", "Forward x (m)"
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, color="#383838", linewidth=0.7)

        # 距離環
        for r_m in [2, 5, 10, 15, 20]:
            ax.add_patch(plt.Circle((0, 0), r_m, fill=False,
                                     color="#4a4a4a", linewidth=0.8))

        # sweep → metric
        denom = self.r_max - self.r_robot
        real_dist = sweep_norm * denom + self.r_robot
        n_bins = sweep_norm.shape[0]
        # 訓練端用 angles_deg = -180 + i*5 (沒加 0.5 offset)，這裡完全跟著
        angles = np.radians(-180.0 + np.arange(n_bins) * (360.0 / n_bins))
        x_forward = real_dist * np.cos(angles)
        y_left = real_dist * np.sin(angles)
        if self.frame_mode == "world":
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            plot_x = cos_y * x_forward - sin_y * y_left
            plot_y = sin_y * x_forward + cos_y * y_left
        else:
            plot_x = y_left
            plot_y = x_forward

        # 折線
        ax.plot(plot_x, plot_y, color="#7fbf7f", linewidth=1.2, alpha=0.8)
        # 點：綠/橙/紅/灰
        colors = np.full(real_dist.shape, "#50dc50", dtype=object)
        colors[real_dist < 5.0] = "#ffa500"
        colors[real_dist < 2.0] = "#ff3030"
        colors[real_dist >= self.r_max - 0.5] = "#808080"
        sizes = np.full(real_dist.shape, 22.0)
        sizes[real_dist < 5.0] = 36.0
        sizes[real_dist < 2.0] = 54.0
        ax.scatter(plot_x, plot_y, c=colors.tolist(), s=sizes, zorder=3)

        # 機器人：白圓 + 箭頭 + 中心點
        ax.add_patch(plt.Circle((0, 0), 0.3, fill=False, color="white",
                                 linewidth=2.0, zorder=4))
        if self.frame_mode == "world":
            head_x, head_y = 1.2 * math.cos(yaw), 1.2 * math.sin(yaw)
        else:
            head_x, head_y = 0.0, 1.2
        ax.arrow(0, 0, head_x, head_y, color="white", width=0.04,
                 head_width=0.35, length_includes_head=True, zorder=5)
        ax.scatter([0], [0], c="white", s=28, marker="o",
                   edgecolors="black", linewidths=0.6, zorder=7)

        # 軌跡（漸變色，橙→青）
        if len(trail) >= 2:
            arr = np.array(trail)
            rel = arr - np.array(robot_xy)[None, :]
            if self.frame_mode == "world":
                tx, ty = rel[:, 0], rel[:, 1]
            else:
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                xf = cos_y * rel[:, 0] + sin_y * rel[:, 1]
                yl = -sin_y * rel[:, 0] + cos_y * rel[:, 1]
                tx, ty = yl, xf
            n = len(tx)
            segs = np.column_stack([tx[:-1], ty[:-1],
                                     tx[1:], ty[1:]]).reshape(-1, 2, 2)
            alphas = np.linspace(0.15, 0.9, n - 1)
            t = np.linspace(0, 1, n - 1)
            cc = np.zeros((n - 1, 4))
            cc[:, 0] = 0.6 * (1 - t) + 0.3 * t
            cc[:, 1] = 0.3 * (1 - t) + 0.9 * t
            cc[:, 2] = 0.1 * (1 - t) + 1.0 * t
            cc[:, 3] = alphas
            ax.add_collection(LineCollection(segs, colors=cc,
                                              linewidths=1.5, zorder=2))

        # subgoal 優先來自 obs_debug（policy 推論中），fallback 用 goal_pose 直接算
        raw_gx = raw_gy = None
        if obs_79 is not None and obs_79.shape[0] >= 6:
            raw_gx, raw_gy = float(obs_79[4]), float(obs_79[5])
        elif goal_body is not None:
            raw_gx, raw_gy = goal_body[0], goal_body[1]  # forward, left

        if raw_gx is not None:
            gx_c = float(np.clip(raw_gx, -self.r_max, self.r_max))
            gy_c = float(np.clip(raw_gy, -self.r_max, self.r_max))
            if self.frame_mode == "world":
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                goal_x = cos_y * gx_c - sin_y * gy_c
                goal_y = sin_y * gx_c + cos_y * gy_c
            else:
                goal_x, goal_y = gy_c, gx_c   # BEV: x=left, y=forward
            dist_m = math.hypot(raw_gx, raw_gy)
            ax.arrow(0, 0, goal_x, goal_y, color="#ffd040", width=0.035,
                     head_width=0.45, length_includes_head=True, zorder=4)
            ax.scatter([goal_x], [goal_y], c=["#ffd040"], s=80, zorder=5)
            # 距離標注
            label_x = goal_x * 0.55
            label_y = goal_y * 0.55
            ax.text(label_x, label_y, f"{dist_m:.1f}m",
                    color="#ffd040", fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=6,
                    bbox={"facecolor": "#1a1a1a", "edgecolor": "none",
                          "alpha": 0.7, "boxstyle": "round,pad=0.2"})

    # ── 上方資料面板（三欄 grid，對齊 LiveBEVVisualizer._draw_top_panel） ──
    def _draw_top_panel(self, step, cmd_v, cmd_w, sweep, obs_79, mode,
                         yaw, sweep_age, odom_age, goal_body=None,
                         raw_nearest=None, raw_nearest_ang=None):
        ax = self.text_ax
        ax.cla()
        ax.set_facecolor("#101214")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.axhline(0.02, color="#3c3f44", linewidth=1.0)
        for x in (0.33, 0.66):
            ax.axvline(x, ymin=0.08, ymax=0.82, color="#272b30", linewidth=0.8)

        # Status badge（右上）
        if mode == "nav":
            badge_color, badge_text = "#1f8f3a", " [ RL Control: nav ] "
        elif mode == "estop":
            badge_color, badge_text = "#b3261e", " [ EMERGENCY STOP ] "
        elif mode == "manual":
            badge_color, badge_text = "#ce8a17", " [ Manual override ] "
        elif mode == "paused":
            badge_color, badge_text = "#5b6175", " [ Paused ] "
        elif mode == "idle":
            badge_color, badge_text = "#3b4250", " [ Idle ] "
        else:
            badge_color, badge_text = "#3b4250", f" [ {mode} ] "
        ax.text(0.985, 0.91, badge_text, transform=ax.transAxes,
                ha="right", va="center", color="white", fontsize=10.5,
                fontweight="bold",
                bbox={"facecolor": badge_color, "edgecolor": "none",
                      "alpha": 0.88, "boxstyle": "round,pad=0.45"})

        # 標題
        title_suffix = "World" if self.frame_mode == "world" else "Body (fwd=up)"
        ax.text(0.02, 0.91,
                f"rover_rl BEV — 72-bin LiDAR ({title_suffix})",
                transform=ax.transAxes, color="white", fontsize=10.3,
                fontweight="bold", ha="left", va="center")

        x_left, x_mid, x_right = 0.02, 0.36, 0.69
        y_header = 0.78
        y_rows = [0.64, 0.53, 0.42, 0.31, 0.20, 0.09]
        mono = "DejaVu Sans Mono"

        # ── Column 1: AGENT / ACTION ──
        ax.text(x_left, y_header, "AGENT / ACTION", transform=ax.transAxes,
                color="#fff4a8", fontsize=9.2, fontweight="bold",
                ha="left", va="center")
        left_lines = [
            f"Step   : {step}",
            f"v cmd  : {cmd_v:+.3f} m/s",
            f"w cmd  : {cmd_w:+.3f} rad/s",
            f"w deg  : {math.degrees(cmd_w):+.1f}°/s",
            f"sweep  : {'OK' if sweep_age < 0.5 else f'STALE {sweep_age:.1f}s'}",
            f"odom   : {'OK' if odom_age < 0.3 else f'STALE {odom_age:.1f}s'}",
        ]
        for y, line in zip(y_rows, left_lines):
            ax.text(x_left, y, line, transform=ax.transAxes,
                    color="#ffe9a8", fontsize=7.6, fontfamily=mono,
                    ha="left", va="center")

        # ── Column 2: LIDAR / GOAL ──
        ax.text(x_mid, y_header, "LIDAR / GOAL", transform=ax.transAxes,
                color="#c9f7c9", fontsize=9.2, fontweight="bold",
                ha="left", va="center")
        denom = self.r_max - self.r_robot
        real = sweep * denom + self.r_robot
        n = real.shape[0]
        near_idx = int(real.argmin())
        near_d = float(real[near_idx])
        near_ang = -180.0 + near_idx * (360.0 / n)
        lt2 = int((real < 2.0).sum())
        mid = int(((real >= 2.0) & (real < 5.0)).sum())
        gx = gy = float("nan")
        d_goal = float("nan")
        goal_src = ""
        if obs_79 is not None and obs_79.shape[0] >= 6:
            gx, gy = float(obs_79[4]), float(obs_79[5])
            d_goal = math.hypot(gx, gy)
            goal_src = "[obs]"
        elif goal_body is not None:
            gx, gy = goal_body[0], goal_body[1]
            d_goal = math.hypot(gx, gy)
            goal_src = "[pose]"
        raw_str = (f"{raw_nearest:.2f}m@{raw_nearest_ang:+.0f}°"
                   if raw_nearest is not None else "N/A")
        mid_lines = [
            f"Nearest: {near_d:.2f}m @ bin{near_idx}",
            f"Raw min: {raw_str}",
            f"Mean   : {float(real.mean()):.2f}m",
            f"<2m/2~5: {lt2}/{n}  {mid}/{n}",
            f"Goal xy: ({gx:+.2f},{gy:+.2f}){goal_src}" if not math.isnan(gx) else "Goal xy: N/A",
            f"Goal d : {d_goal:.2f}m" if not math.isnan(d_goal) else "Goal d : N/A",
        ]
        for y, line in zip(y_rows, mid_lines):
            ax.text(x_mid, y, line, transform=ax.transAxes,
                    color="#d8f7d8", fontsize=7.4, fontfamily=mono,
                    ha="left", va="center")

        # ── Column 3: MODE / STATE ──（部署版替換 ORCA/RVO2）
        ax.text(x_right, y_header, "MODE / STATE", transform=ax.transAxes,
                color="#88ccff", fontsize=9.2, fontweight="bold",
                ha="left", va="center")
        right_lines = [
            f"Mode   : {mode}",
            f"Frame  : {self.frame_mode}",
            f"Yaw    : {math.degrees(yaw):+.1f}°",
            f"Trail  : {len(self._trail)} pts",
            f"sweep age: {sweep_age:.2f}s",
            f"odom age : {odom_age:.2f}s",
        ]
        for y, line in zip(y_rows, right_lines):
            ax.text(x_right, y, line, transform=ax.transAxes,
                    color="#d8e6ff", fontsize=7.6, fontfamily=mono,
                    ha="left", va="center")

    def _fig_to_image_msg(self) -> Image:
        self.fig.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        try:
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb = buf.reshape(height, width, 3)
        except AttributeError:
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            rgb = buf.reshape(height, width, 4)[:, :, :3].copy()
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "rover_rl_bev"
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = rgb.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = BevPlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(node.fig)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
