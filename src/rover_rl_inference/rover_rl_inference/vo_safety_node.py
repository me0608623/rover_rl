"""VO 安全層 ROS 節點 — 夾在 RL policy 與底盤 mux 之間.

資料流（由 deploy_full 串接）：
    policy_node ──/rover_rl/cmd_vel_desired──► [vo_safety_node] ──/input/nav_cmd_vel──► mux ──► 底盤
                                                    ▲
                       LV-DOT  onboard_detector/get_dynamic_obstacles (service, ~15Hz 輪詢快取)

職責：訂閱 RL 期望 cmd + odom，輪詢 LV-DOT 動態障礙，跑 vo_layer.compute_safe_cmd，
     以 20Hz 發出安全 cmd。演算法在 vo_layer.py（純函式可測），本檔只做 ROS 接線 + 看門狗。

看門狗（fail-safe，安全優先）：
  - RL desired 逾時 → 發 0（RL 沉默不亂動）
  - odom 逾時 → 發 0（不知道自身狀態不敢動）
  - 障礙快取逾時 → 視為無障礙（VO 退化成純放行 desired，靜態仍由 RL 的 LiDAR sweep 擋）

⚠️ 注意：此層只處理「動態障礙」（service 回傳的 dynamicBBoxes_ 已篩動態）；
   靜態避障仍由 RL policy 的 72-bin sweep 負責，避免雙重保守導致原地凍結。
⚠️ 多一層會略增延遲（對應 CLAUDE.md sim-to-real gap #5），故保持輕量、單純。
"""
from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from onboard_detector.srv import GetDynamicObstacles

from .vo_layer import Obstacle, RobotState, VOParams, _clamp, compute_safe_cmd


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """四元數 → yaw（平面導航只需繞 z 的旋轉角）。"""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class VOSafetyNode(Node):
    def __init__(self) -> None:
        super().__init__("vo_safety_node")

        gp = self.declare_parameter
        # --- Topics / service ---
        gp("topic_cmd_in", "/rover_rl/cmd_vel_desired")   # RL 期望（policy 改發到這）
        gp("topic_cmd_out", "/input/nav_cmd_vel")         # 安全輸出（送進 mux）
        gp("topic_odom", "/odom")
        gp("service_obstacles", "onboard_detector/get_dynamic_obstacles")
        # --- 看門狗逾時 ---
        gp("timeout_desired_s", 0.5)   # RL cmd 超過此值沒更新 → 發 0
        gp("timeout_odom_s", 0.3)
        gp("timeout_obstacles_s", 1.0) # 障礙快取超過此值 → 視為無障礙
        # --- 頻率 ---
        gp("ctrl_rate_hz", 20.0)
        gp("obstacle_poll_hz", 15.0)
        # --- VO 演算法參數（對應 vo_layer.VOParams）---
        gp("v_max", 1.0)
        gp("v_min", -1.0)
        gp("w_max", 1.2)
        gp("accel_v", 1.2)
        gp("accel_w", 3.0)
        gp("window_time", 1.0)   # 候選視窗可達時間（與 ctrl_dt 解耦，讓 VO 能繞行）
        gp("n_v", 7)
        gp("n_w", 15)
        gp("horizon", 2.0)
        gp("sim_dt", 0.1)
        gp("r_robot", 0.35)
        gp("margin", 0.10)
        gp("w_v", 1.0)
        gp("w_w", 0.3)
        gp("engage_range", 6.0)
        gp("obstacle_radius_max", 1.5)   # bbox 換算半徑的上限（防超大框把路全擋死）

        g = self.get_parameter
        self.topic_in = g("topic_cmd_in").get_parameter_value().string_value
        self.topic_out = g("topic_cmd_out").get_parameter_value().string_value
        self.topic_odom = g("topic_odom").get_parameter_value().string_value
        self.srv_name = g("service_obstacles").get_parameter_value().string_value
        self.timeout_desired = float(g("timeout_desired_s").value)
        self.timeout_odom = float(g("timeout_odom_s").value)
        self.timeout_obs = float(g("timeout_obstacles_s").value)
        ctrl_hz = float(g("ctrl_rate_hz").value)
        poll_hz = float(g("obstacle_poll_hz").value)
        self.r_obs_max = float(g("obstacle_radius_max").value)

        # 控制週期 ctrl_dt 即 dynamic window 的可達時間（與 ctrl_rate 對齊）
        self.vo = VOParams(
            v_max=float(g("v_max").value), v_min=float(g("v_min").value),
            w_max=float(g("w_max").value),
            accel_v=float(g("accel_v").value), accel_w=float(g("accel_w").value),
            ctrl_dt=1.0 / ctrl_hz,
            window_time=float(g("window_time").value),
            n_v=int(g("n_v").value), n_w=int(g("n_w").value),
            horizon=float(g("horizon").value), sim_dt=float(g("sim_dt").value),
            r_robot=float(g("r_robot").value), margin=float(g("margin").value),
            w_v=float(g("w_v").value), w_w=float(g("w_w").value),
            engage_range=float(g("engage_range").value),
        )

        # --- 狀態 ---
        self._des_v = 0.0
        self._des_w = 0.0
        self._des_t = 0.0
        self._robot: RobotState | None = None
        self._odom_t = 0.0
        self._obstacles: list[Obstacle] = []
        self._obs_t = 0.0
        self._obs_inflight = False   # 防同時多個未完成 service 呼叫
        self._out_v = 0.0            # 上次輸出（輸出端 slew 限速用）
        self._out_w = 0.0

        # --- ROS 介面 ---
        self.pub_cmd = self.create_publisher(Twist, self.topic_out, 10)
        self.create_subscription(Twist, self.topic_in, self._cb_desired, 10)
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)
        self.cli_obs = self.create_client(GetDynamicObstacles, self.srv_name)

        self.create_timer(1.0 / poll_hz, self._poll_obstacles)
        self.create_timer(1.0 / ctrl_hz, self._tick_ctrl)

        self.get_logger().info(
            f"VO 安全層啟動：{self.topic_in} → (VO) → {self.topic_out}；"
            f"障礙 service={self.srv_name}；w_max={self.vo.w_max} horizon={self.vo.horizon}s"
        )

    # ── callbacks ──
    def _cb_desired(self, msg: Twist) -> None:
        self._des_v = msg.linear.x
        self._des_w = msg.angular.z
        self._des_t = time.monotonic()

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        tw = msg.twist.twist
        self._robot = RobotState(
            x=p.x, y=p.y, yaw=_yaw_from_quat(q.x, q.y, q.z, q.w),
            v=tw.linear.x, w=tw.angular.z,
        )
        self._odom_t = time.monotonic()

    # ── 障礙輪詢（非阻塞 async，避免卡住控制迴圈）──
    def _poll_obstacles(self) -> None:
        if self._robot is None or self._obs_inflight:
            return
        if not self.cli_obs.service_is_ready():
            return  # LV-DOT 還沒起來，當作無障礙（看門狗會處理）
        req = GetDynamicObstacles.Request()
        req.current_position = Point(x=self._robot.x, y=self._robot.y, z=0.0)
        req.range = self.vo.engage_range + 2.0   # 多抓一點緩衝
        self._obs_inflight = True
        fut = self.cli_obs.call_async(req)
        fut.add_done_callback(self._on_obstacles)

    def _on_obstacles(self, fut) -> None:
        self._obs_inflight = False
        try:
            res = fut.result()
        except Exception as e:  # service 失敗不影響控制，下拍重試
            self.get_logger().warn(f"取障礙 service 失敗：{e}", throttle_duration_sec=5.0)
            return
        obstacles: list[Obstacle] = []
        for pos, vel, size in zip(res.position, res.velocity, res.size):
            # bbox 寬度 → 等效半徑（取 xy 較大邊半徑，封頂防超大框擋死全路）
            r = min(0.5 * max(size.x, size.y), self.r_obs_max)
            obstacles.append(Obstacle(x=pos.x, y=pos.y, vx=vel.x, vy=vel.y, r=r))
        self._obstacles = obstacles
        self._obs_t = time.monotonic()

    # ── 20Hz 控制迴圈 ──
    def _tick_ctrl(self) -> None:
        now = time.monotonic()

        # 看門狗：RL 沉默 或 odom 掉線 → 立即發 0（fail-safe，不 slew，讓底盤盡快煞）
        if self._robot is None or (now - self._odom_t) > self.timeout_odom:
            self._publish_hard_zero()
            return
        if (now - self._des_t) > self.timeout_desired:
            self._publish_hard_zero()
            return

        # 障礙快取過期 → 視為無障礙（靜態仍由 RL sweep 擋）
        obstacles = self._obstacles if (now - self._obs_t) <= self.timeout_obs else []

        res = compute_safe_cmd(self._des_v, self._des_w, self._robot, obstacles, self.vo)

        # 介入時節流 log，方便現場觀察 VO 何時動作
        if res.blocked:
            self.get_logger().warn(
                f"VO 全堵死 → 停車（近障 {res.n_obstacles}）", throttle_duration_sec=1.0)
        elif res.engaged and (abs(res.v - self._des_v) > 0.05 or abs(res.w - self._des_w) > 0.1):
            self.get_logger().info(
                f"VO 介入：RL({self._des_v:+.2f},{self._des_w:+.2f}) → "
                f"目標({res.v:+.2f},{res.w:+.2f}) 近障{res.n_obstacles} 可行{res.n_feasible}",
                throttle_duration_sec=0.5)

        # 輸出端 slew 限速：候選窗較寬，靠這裡把輸出每拍限在 accel×ctrl_dt 內，
        # 保證實際送底盤的 cmd 平滑可達（不會因 VO 切解而跳階）。
        self._publish_slew(res.v, res.w)

    def _publish_slew(self, target_v: float, target_w: float) -> None:
        max_dv = self.vo.accel_v * self.vo.ctrl_dt
        max_dw = self.vo.accel_w * self.vo.ctrl_dt
        self._out_v += _clamp(target_v - self._out_v, -max_dv, max_dv)
        self._out_w += _clamp(target_w - self._out_w, -max_dw, max_dw)
        self._emit(self._out_v, self._out_w)

    def _publish_hard_zero(self) -> None:
        self._out_v = 0.0
        self._out_w = 0.0
        self._emit(0.0, 0.0)

    def _emit(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub_cmd.publish(msg)


def main() -> None:
    rclpy.init()
    node = VOSafetyNode()
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
