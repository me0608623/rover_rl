"""bev_play_node — 移植訓練端 play_eval/bev_renderer.py 到 ROS 2.

訓練側 BEV 風格（matplotlib 黑底科研風）：
  • 72-bin LiDAR 極座標圖（紅<2m / 橙<5m / 綠>5m / 灰=max_range）
  • 距離環 2m / 5m / 10m / 15m / 20m
  • 機器人圓圈 + 航向箭頭（白）
  • 目標箭頭（黃）
  • 中文資訊面板（步驟/動作/航向/最近障礙/平均距離）

與訓練端差異：
  • 移除 RNN aux 7D 預測 panel（部署沒 ground truth）
  • 移除多 goal 場景（沒 simulator）
  • 用 Agg backend headless render → 發布為 sensor_msgs/Image
    可直接在 RViz / rqt_image_view 看，不需 X display

訂閱：
  /rover_rl/lidar_sweep_72   Float32MultiArray[72]   ← 已預處理 sweep
  /rover_rl_policy/obs_debug Float32MultiArray[79]   ← obs (含 goal body frame, optional)
  /input/nav_cmd_vel         Twist                     ← 動作顯示用 (optional)
  /odom                      Odometry                   ← yaw 顯示 (optional)

發布：
  /rover_rl/bev_image        sensor_msgs/Image (rgb8)

注意：matplotlib render 比較重，預設 5 Hz 更新。
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# Headless rendering — 不依賴 X display
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


class BevPlayNode(Node):
    def __init__(self):
        super().__init__("rover_rl_bev_play")

        # ── 參數 ──
        self.declare_parameter("topic_sweep", "/rover_rl/lidar_sweep_72")
        self.declare_parameter("topic_obs_debug", "/rover_rl_policy/obs_debug")
        self.declare_parameter("topic_cmd_vel", "/input/nav_cmd_vel")
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_image", "/rover_rl/bev_image")
        self.declare_parameter("frame_mode", "body")        # 'body' 或 'world'
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("r_max", 20.0)
        self.declare_parameter("r_robot", 0.35)
        self.declare_parameter("figure_dpi", 80)             # 80 = 約 480x480
        self.declare_parameter("figure_size", 6.0)

        gp = self.get_parameter
        topic_sweep = gp("topic_sweep").get_parameter_value().string_value
        topic_obs = gp("topic_obs_debug").get_parameter_value().string_value
        topic_cmd = gp("topic_cmd_vel").get_parameter_value().string_value
        topic_odom = gp("topic_odom").get_parameter_value().string_value
        topic_image = gp("topic_image").get_parameter_value().string_value
        self.frame_mode = gp("frame_mode").get_parameter_value().string_value
        self.rate_hz = float(gp("rate_hz").value)
        self.r_max = float(gp("r_max").value)
        self.r_robot = float(gp("r_robot").value)
        self.dpi = int(gp("figure_dpi").value)
        self.fig_size = float(gp("figure_size").value)

        # ── 狀態 ──
        self._lock = threading.Lock()
        self._sweep_norm: np.ndarray | None = None
        self._sweep_t = 0.0
        self._obs_79: np.ndarray | None = None
        self._last_cmd = (0.0, 0.0)
        self._odom_yaw = 0.0
        self._step = 0

        # ── matplotlib figure（單張，重複使用以省 GC） ──
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK TC", "WenQuanYi Micro Hei", "DejaVu Sans"
        ]
        plt.rcParams["axes.unicode_minus"] = False
        self.fig, self.ax = plt.subplots(
            figsize=(self.fig_size, self.fig_size), dpi=self.dpi,
        )
        self.fig.patch.set_facecolor("#141414")

        # ── 訂閱 / 發布 ──
        self.create_subscription(Float32MultiArray, topic_sweep, self._cb_sweep, 10)
        self.create_subscription(Float32MultiArray, topic_obs, self._cb_obs, 10)
        self.create_subscription(Twist, topic_cmd, self._cb_cmd, 10)
        self.create_subscription(Odometry, topic_odom, self._cb_odom, 10)
        self.pub_image = self.create_publisher(Image, topic_image, 5)

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._tick)

        self.get_logger().info(
            f"rover_rl_bev_play 啟動完成\n"
            f"  訂閱: {topic_sweep}, {topic_obs}, {topic_cmd}, {topic_odom}\n"
            f"  發布: {topic_image} ({self.fig_size*self.dpi:.0f}×{self.fig_size*self.dpi:.0f} px @ {self.rate_hz}Hz)\n"
            f"  座標系: {self.frame_mode}"
        )

    def _cb_sweep(self, msg: Float32MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        with self._lock:
            self._sweep_norm = arr
            self._sweep_t = time.monotonic()

    def _cb_obs(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 6:
            with self._lock:
                self._obs_79 = np.asarray(msg.data, dtype=np.float32)

    def _cb_cmd(self, msg: Twist) -> None:
        with self._lock:
            self._last_cmd = (float(msg.linear.x), float(msg.angular.z))

    def _cb_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        with self._lock:
            self._odom_yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)

    def _tick(self) -> None:
        with self._lock:
            sweep = self._sweep_norm
            obs = self._obs_79
            cmd = self._last_cmd
            yaw = self._odom_yaw
            self._step += 1
            step = self._step
        if sweep is None:
            return
        self._render(sweep, obs, cmd, yaw, step)
        msg = self._fig_to_image_msg()
        self.pub_image.publish(msg)

    def _render(self, sweep_norm, obs_79, cmd, yaw, step):
        ax = self.ax
        ax.cla()
        ax.set_facecolor("#141414")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.r_max, self.r_max)
        ax.set_ylim(-self.r_max, self.r_max)
        frame_label = "車體座標（前方為上）" if self.frame_mode == "body" else "世界座標"
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, color="#383838", linewidth=0.7)
        ax.set_title(f"Charge BEV — 72-bin LiDAR ({frame_label})",
                     color="white", fontsize=11)

        # 距離環
        for r_m in [2, 5, 10, 15, 20]:
            ax.add_patch(plt.Circle((0, 0), r_m, fill=False,
                                     color="#4a4a4a", linewidth=0.8))
            ax.text(0.2, r_m, f"{r_m}m", color="#888888", fontsize=8)

        # sweep 反正規化回 metric distance
        # sweep_norm = (d - r_robot) / (r_max - r_robot)
        # → d = sweep_norm * (r_max - r_robot) + r_robot
        denom = self.r_max - self.r_robot
        real_dist = sweep_norm * denom + self.r_robot

        # 角度：訓練端 atan2(y, x) + π → bin_idx；bin 中心 = -π + (i + 0.5) × (2π/72)
        n_bins = sweep_norm.shape[0]
        angles = -math.pi + (np.arange(n_bins) + 0.5) * (2 * math.pi / n_bins)

        # body frame: x_fwd = d·cos(θ), y_left = d·sin(θ)；圖上 x=y_left, y=x_fwd
        x_fwd = real_dist * np.cos(angles)
        y_left = real_dist * np.sin(angles)
        if self.frame_mode == "world":
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            plot_x = cos_y * x_fwd - sin_y * y_left
            plot_y = sin_y * x_fwd + cos_y * y_left
        else:
            plot_x = y_left
            plot_y = x_fwd

        # 連線
        ax.plot(plot_x, plot_y, color="#7fbf7f", linewidth=1.2, alpha=0.8)

        # 點顏色
        colors = np.full(real_dist.shape, "#50dc50", dtype=object)
        colors[real_dist < 5.0] = "#ffa500"
        colors[real_dist < 2.0] = "#ff3030"
        colors[real_dist >= self.r_max - 0.5] = "#808080"
        sizes = np.full(real_dist.shape, 22.0)
        sizes[real_dist < 5.0] = 36.0
        sizes[real_dist < 2.0] = 54.0
        ax.scatter(plot_x, plot_y, c=colors.tolist(), s=sizes, zorder=3)

        # 機器人（白圈 + 航向箭頭）
        ax.add_patch(plt.Circle((0, 0), 0.3, fill=False, color="white",
                                 linewidth=2.0, zorder=4))
        if self.frame_mode == "world":
            hx = 1.2 * math.cos(yaw)
            hy = 1.2 * math.sin(yaw)
        else:
            hx, hy = 0.0, 1.2
        ax.arrow(0, 0, hx, hy, color="white", width=0.04, head_width=0.35,
                 length_includes_head=True, zorder=5)

        # 目標箭頭（從 obs[4:6] 取 body-frame goal）
        if obs_79 is not None and obs_79.shape[0] >= 6:
            gx_body = float(obs_79[4])
            gy_body = float(obs_79[5])
            # clip 進視窗
            gx_body_c = float(np.clip(gx_body, -self.r_max, self.r_max))
            gy_body_c = float(np.clip(gy_body, -self.r_max, self.r_max))
            if self.frame_mode == "world":
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                goal_x = cos_y * gx_body_c - sin_y * gy_body_c
                goal_y = sin_y * gx_body_c + cos_y * gy_body_c
            else:
                goal_x = gy_body_c
                goal_y = gx_body_c
            ax.arrow(0, 0, goal_x, goal_y, color="#ffd040", width=0.035,
                     head_width=0.45, length_includes_head=True, zorder=4)
            ax.scatter([goal_x], [goal_y], c=["#ffd040"], s=80, zorder=5)
        else:
            gx_body = float("nan")
            gy_body = float("nan")

        # 資訊面板
        near_idx = int(real_dist.argmin())
        near_d = float(real_dist[near_idx])
        near_angle = -180.0 + (near_idx + 0.5) * (360.0 / n_bins)
        text_lines = [
            f"步驟={step}  動作 v={cmd[0]:+.2f}m/s ω={cmd[1]:+.2f}rad/s",
            f"座標系={frame_label}  航向={math.degrees(yaw):+.1f}°",
            f"最近障礙: {near_d:.2f}m @ bin{near_idx} ({near_angle:+.0f}°)",
            f"平均距離={float(real_dist.mean()):.2f}m  <2m={int((real_dist < 2.0).sum())}/{n_bins}",
        ]
        if not math.isnan(gx_body):
            d_goal = math.hypot(gx_body, gy_body)
            text_lines.append(f"目標(body)=({gx_body:+.2f},{gy_body:+.2f}) dist={d_goal:.2f}m")
        text_lines.append("白=機器人  黃=目標  紅=近距危險  灰=無回波")
        ax.text(
            0.02, 0.98, "\n".join(text_lines),
            transform=ax.transAxes, va="top", ha="left", color="white", fontsize=9,
            bbox={"facecolor": "#202020", "edgecolor": "#606060", "alpha": 0.85},
        )

    def _fig_to_image_msg(self) -> Image:
        """matplotlib canvas → sensor_msgs/Image (rgb8)，不需 cv_bridge."""
        self.fig.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        try:
            # matplotlib 3.x
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        except AttributeError:
            # 新版 matplotlib 改 API
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(height, width, 4)[:, :, :3].copy().reshape(-1)
        rgb = buf.reshape(height, width, 3)

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
