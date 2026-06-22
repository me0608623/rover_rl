"""VO 安全層 ROS 節點 — 夾在 RL policy 與底盤 mux 之間.

資料流（由 deploy_full 串接）：
    policy_node ──/rover_rl/cmd_vel_desired──► [vo_safety_node] ──/input/nav_cmd_vel──► mux ──► 底盤
                                                    ▲
                       vo_interface  /vo_interface/tracked_obstacles (topic, 20Hz)

職責：訂閱 RL 期望 cmd + odom + vo_interface 追蹤障礙，跑 vo_layer.compute_safe_cmd，
     以 20Hz 發出安全 cmd。演算法在 vo_layer.py（純函式可測），本檔只做 ROS 接線 + 看門狗。

為何吃 vo_interface 而非直接輪詢 LV-DOT service？
  vo_interface 已對 LV-DOT 動態框做 CV-Kalman 重追蹤，給出「持久 ID + 平滑絕對速度」。
  rollout 碰撞預測靠 p_i + v_i·t 外推，最吃速度準度；LV-DOT raw 速度偏跳會讓快速障礙物
  的預測碰撞點亂飄（該煞沒煞 / 亂煞）。改吃平滑速度後，前/側快速切入的軌跡外推更穩。

看門狗（fail-safe，安全優先）：
  - RL desired 逾時 → 發 0（RL 沉默不亂動）
  - odom 逾時 → 發 0（不知道自身狀態不敢動）
  - 障礙逾時 → 視為無障礙（VO 退化成純放行 desired，靜態仍由 RL 的 LiDAR sweep 擋）

⚠️ 注意：此層只處理「動態障礙」（service 回傳的 dynamicBBoxes_ 已篩動態）；
   靜態避障仍由 RL policy 的 72-bin sweep 負責，避免雙重保守導致原地凍結。
⚠️ 多一層會略增延遲（對應 CLAUDE.md sim-to-real gap #5），故保持輕量、單純。
"""
from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from vo_interface.msg import TrackedObstacleArray

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
        gp("topic_obstacles", "/vo_interface/tracked_obstacles")  # vo_interface 平滑追蹤
        gp("topic_goal_status", "/rover_rl_policy/status")  # policy status JSON（取 goal_dist/goal_ang_deg）
        # --- 看門狗逾時 ---
        gp("timeout_desired_s", 0.5)   # RL cmd 超過此值沒更新 → 發 0
        gp("timeout_odom_s", 0.3)
        gp("timeout_obstacles_s", 1.0) # 障礙超過此值沒更新 → 視為無障礙
        gp("timeout_goal_s", 1.5)      # goal 超過此值沒更新 → 視為無 goal（VO 退回純貼近 RL）
        # --- 年輕 track 過濾：速度尚未收斂的 track 不參與 rollout（避免亂煞）---
        gp("min_vel_confidence", 0.0)  # vo_interface vel_confidence 低於此值的 track 忽略其速度
        # --- 最小障礙速度：站著不動的人(speed≈0)整顆 drop，交給 RL sweep，避雙重處理凍結 ---
        gp("min_obstacle_speed", 0.2)  # speed 低於此值的 track 不加進 VO 障礙清單（0=關）
        # --- 頻率 ---
        gp("ctrl_rate_hz", 20.0)
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
        gp("w_goal", 0.0)   # goal 導向項權重（0=關，純貼近 RL；>0 被擋時偏好繞向 goal）
        gp("engage_range", 6.0)
        gp("obstacle_radius_max", 1.5)   # bbox 換算半徑的上限（防超大框把路全擋死）

        g = self.get_parameter
        self.topic_in = g("topic_cmd_in").get_parameter_value().string_value
        self.topic_out = g("topic_cmd_out").get_parameter_value().string_value
        self.topic_odom = g("topic_odom").get_parameter_value().string_value
        self.topic_obs = g("topic_obstacles").get_parameter_value().string_value
        self.topic_goal_status = g("topic_goal_status").get_parameter_value().string_value
        self.timeout_desired = float(g("timeout_desired_s").value)
        self.timeout_odom = float(g("timeout_odom_s").value)
        self.timeout_obs = float(g("timeout_obstacles_s").value)
        self.timeout_goal = float(g("timeout_goal_s").value)
        self.min_vel_conf = float(g("min_vel_confidence").value)
        self.min_obs_speed = float(g("min_obstacle_speed").value)
        ctrl_hz = float(g("ctrl_rate_hz").value)
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
            w_goal=float(g("w_goal").value),
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
        self._goal_active = False    # 本拍是否確實把 goal 導向項納入（供 status 顯示）
        # goal（取自 policy status，body-frame 相對量；每拍配 odom_yaw 反推 odom 座標）
        self._goal_dist: float | None = None      # m，body-frame goal 距離
        self._goal_bearing: float | None = None   # rad，body-frame goal 方位（相對車頭）
        self._goal_t = 0.0
        self._out_v = 0.0            # 上次輸出（輸出端 slew 限速用）
        self._out_w = 0.0
        # status 節流：control loop 20Hz，狀態只需 ~5Hz（對齊 policy_node ~/status）
        self._status_every = max(1, int(round(ctrl_hz / 5.0)))
        self._status_tick = 0

        # --- ROS 介面 ---
        self.pub_cmd = self.create_publisher(Twist, self.topic_out, 10)
        # 精簡狀態 JSON（供 status_tui 顯示 VO 介入/參數，純觀察不影響控制）→ /vo_safety_node/status
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.create_subscription(Twist, self.topic_in, self._cb_desired, 10)
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)
        self.create_subscription(TrackedObstacleArray, self.topic_obs,
                                 self._cb_obstacles, 10)
        # goal 來源：policy status JSON（純訂閱觀察，不耦合控制安全；收不到→退回純貼近 RL）
        self.create_subscription(String, self.topic_goal_status,
                                 self._cb_goal_status, 10)

        self.create_timer(1.0 / ctrl_hz, self._tick_ctrl)

        self.get_logger().info(
            f"VO 安全層啟動：{self.topic_in} → (VO) → {self.topic_out}；"
            f"障礙 topic={self.topic_obs}；w_max={self.vo.w_max} horizon={self.vo.horizon}s"
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

    # ── 障礙訂閱（vo_interface 已做 KF 平滑追蹤，直接吃 topic）──
    def _cb_obstacles(self, msg: TrackedObstacleArray) -> None:
        obstacles: list[Obstacle] = []
        for ob in msg.obstacles:
            # 站著不動的人(含 YOLO 標 dynamic 但 speed≈0)整顆 drop：交給 RL 的 72-bin sweep
            # 自己繞，VO 只管移動中的人。避免「固定阻擋 + 正面對衝邏輯」誤觸原地凍結。
            speed = math.hypot(float(ob.velocity.x), float(ob.velocity.y))
            if speed < self.min_obs_speed:
                continue
            # vo_interface 已給等效半徑；仍封頂防超大框把整條路擋死
            r = min(float(ob.radius), self.r_obs_max)
            # 速度尚未收斂的年輕 track：忽略其速度（vx=vy=0），只靠位置 + 機器人自身運動
            # 做碰撞預測，避免拿跳動速度亂外推（保守）
            if ob.vel_confidence < self.min_vel_conf:
                vx, vy = 0.0, 0.0
            else:
                vx, vy = float(ob.velocity.x), float(ob.velocity.y)
            obstacles.append(Obstacle(x=float(ob.position.x), y=float(ob.position.y),
                                      vx=vx, vy=vy, r=r))
        self._obstacles = obstacles
        self._obs_t = time.monotonic()

    # ── goal 訂閱（policy status JSON；取 body-frame goal_dist/goal_ang_deg）──
    def _cb_goal_status(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        gd = d.get("goal_dist")
        ga = d.get("goal_ang_deg")
        if gd is None or ga is None:
            # 無目標（待命）→ 清掉 goal，VO 退回純貼近 RL
            self._goal_dist = None
            self._goal_bearing = None
            return
        self._goal_dist = float(gd)
        self._goal_bearing = math.radians(float(ga))
        self._goal_t = time.monotonic()

    def _goal_in_odom(self, now: float) -> tuple[float, float] | None:
        """body-frame goal 配「當下 odom 車姿」反推 odom 座標.

        goal_bearing 為 body-relative（frame-independent），故 odom 系方位 =
        odom_yaw + goal_bearing，再沿此方位推 goal_dist → 與 VO rollout 同在 odom 系。
        逾時 / 無 goal / 無車姿 → None（VO 退回純貼近 RL）。
        """
        if (self._goal_dist is None or self._goal_bearing is None
                or self._robot is None
                or (now - self._goal_t) > self.timeout_goal):
            return None
        bearing_odom = self._robot.yaw + self._goal_bearing
        gx = self._robot.x + self._goal_dist * math.cos(bearing_odom)
        gy = self._robot.y + self._goal_dist * math.sin(bearing_odom)
        return (gx, gy)

    # ── 20Hz 控制迴圈 ──
    def _tick_ctrl(self) -> None:
        now = time.monotonic()

        # 看門狗：RL 沉默 或 odom 掉線 → 立即發 0（fail-safe，不 slew，讓底盤盡快煞）
        if self._robot is None or (now - self._odom_t) > self.timeout_odom:
            self._publish_hard_zero()
            self._publish_status(fail="odom")
            return
        if (now - self._des_t) > self.timeout_desired:
            self._publish_hard_zero()
            self._publish_status(fail="desired")
            return

        # 障礙快取過期 → 視為無障礙（靜態仍由 RL sweep 擋）
        obs_stale = (now - self._obs_t) > self.timeout_obs
        obstacles = [] if obs_stale else self._obstacles

        goal = self._goal_in_odom(now)   # odom-frame goal（None=無/逾時，退回純貼近 RL）
        self._goal_active = goal is not None and self.vo.w_goal > 0.0
        res = compute_safe_cmd(self._des_v, self._des_w, self._robot, obstacles,
                               self.vo, goal=goal)

        # 介入 = 有近障且 VO 解明顯偏離 RL 意圖（與下方 status/log 同一判據）
        intervening = res.engaged and (
            abs(res.v - self._des_v) > 0.05 or abs(res.w - self._des_w) > 0.1)

        # 介入時節流 log，方便現場觀察 VO 何時動作
        if res.blocked:
            self.get_logger().warn(
                f"VO 全堵死 → 停車（近障 {res.n_obstacles}）", throttle_duration_sec=1.0)
        elif intervening:
            self.get_logger().info(
                f"VO 介入：RL({self._des_v:+.2f},{self._des_w:+.2f}) → "
                f"目標({res.v:+.2f},{res.w:+.2f}) 近障{res.n_obstacles} 可行{res.n_feasible}",
                throttle_duration_sec=0.5)

        # 輸出端 slew 限速：候選窗較寬，靠這裡把輸出每拍限在 accel×ctrl_dt 內，
        # 保證實際送底盤的 cmd 平滑可達（不會因 VO 切解而跳階）。
        self._publish_slew(res.v, res.w)
        self._publish_status(res=res, intervening=intervening, obs_stale=obs_stale)

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

    # ── 精簡狀態 JSON（供 status_tui；節流到 ~5Hz）──
    def _publish_status(self, res=None, intervening: bool = False,
                        obs_stale: bool = False, fail: str = "") -> None:
        self._status_tick += 1
        if self._status_tick % self._status_every:
            return
        d = {
            "fail": fail,                       # ""=正常；"odom"/"desired"=看門狗發 0
            "obs_stale": bool(obs_stale),       # 障礙源逾時 → VO 退化為放行
            "des_v": round(self._des_v, 3),     # RL 期望（VO 輸入）
            "des_w": round(self._des_w, 3),
            "out_v": round(self._out_v, 3),     # 實際送 mux 的 slew 後輸出
            "out_w": round(self._out_w, 3),
            "n_tracked": len(self._obstacles),  # 收到的追蹤障礙總數（engage 過濾前）
            # 參數狀況（供 TUI「VO參數」列）
            "w_max": self.vo.w_max, "horizon": self.vo.horizon,
            "engage_range": self.vo.engage_range, "margin": self.vo.margin,
            "window_time": self.vo.window_time,
            # goal 導向項：w_goal>0 且本拍有有效 goal → 繞行方向由 goal 主導
            "w_goal": self.vo.w_goal, "goal_active": bool(self._goal_active),
        }
        if res is not None:
            d.update({
                "engaged": bool(res.engaged),       # 近障進入 engage_range
                "blocked": bool(res.blocked),       # 全堵死 → 停車
                "intervening": bool(intervening),   # VO 解明顯偏離 RL 意圖
                "n_obs": int(res.n_obstacles),      # engage_range 內障礙數
                "n_feasible": int(res.n_feasible),
                "vo_v": round(res.v, 3), "vo_w": round(res.w, 3),  # VO 選定目標
                "min_ttc": (None if math.isinf(res.min_ttc) else round(res.min_ttc, 2)),
            })
        else:   # 看門狗 hard-zero：未跑 VO
            d.update({"engaged": False, "blocked": False, "intervening": False,
                      "n_obs": 0, "n_feasible": 0,
                      "vo_v": 0.0, "vo_w": 0.0, "min_ttc": None})
        self.pub_status.publish(String(data=json.dumps(d)))


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
