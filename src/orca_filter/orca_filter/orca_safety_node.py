#!/usr/bin/env python3
"""orca_safety_node — ORCA 安全層(控制車),取代 vo_safety_node。

資料流(對稱 vo_safety_node):
    policy_node ──/rover_rl/cmd_vel_desired──► [orca_safety_node] ──/input/nav_cmd_vel──► mux ──► 底盤
                       ▲
   /odom + /vo_interface/tracked_obstacles + /rover_rl_policy/status(front_m)──┘

職責:訂 RL 期望 cmd + odom + vo_interface 障礙,跑 ORCAFilter.solve(non-cooperative RVO2),
     20Hz 發安全 cmd。看門狗 + front_brake + slew 安全網。

與 vo_safety_node 差別:
  - 演算法:ORCA(RVO2 half-plane + LP)取代 VO(DWA 取樣+rollout)
  - 無 escape/commit(ORCA _lateral_evasion_if_frozen 替代)
  - 無 defer_to_vo/front_block_ratio(front_brake 純距離硬底線)
  - non-cooperative(ego_responsibility=1.0 + obs_inflation=1.8,行人不會避讓)

不碰 vo_safety_node / vo_layer / policy_node / orca_core。A/B 並存(launch enable_orca 切換)。
"""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from vo_interface.msg import TrackedObstacleArray

from .orca_core import ORCAFilter, ORCAInput


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class ORCASafetyNode(Node):
    def __init__(self):
        super().__init__("orca_safety_node")

        gp = self.declare_parameter
        # ── topics ──
        gp("topic_cmd_in", "/rover_rl/cmd_vel_desired")
        gp("topic_cmd_out", "/input/nav_cmd_vel")
        gp("topic_odom", "/odom")
        gp("topic_obstacles", "/vo_interface/tracked_obstacles")
        gp("topic_policy_status", "/rover_rl_policy/status")
        # ── 頻率 ──
        gp("ctrl_rate_hz", 20.0)
        # ── 看門狗逾時 ──
        gp("timeout_desired_s", 0.5)
        gp("timeout_odom_s", 0.3)
        gp("timeout_obstacles_s", 1.0)
        gp("timeout_status_s", 1.5)          # policy status(front_m)逾時
        # ── ORCA 核心(預設 = orca_params.yaml 1:1,non-cooperative)──
        gp("robot_radius", 0.35)
        gp("safety_margin", 0.10)
        gp("max_speed", 1.0)
        gp("culling_radius", 5.0)
        gp("neighbor_dist", 10.0)
        gp("max_neighbors", 20)
        gp("time_horizon_dynamic", 2.5)
        gp("time_horizon_static", 0.3)
        gp("static_vel_threshold", 0.15)
        gp("angle_threshold_deg", 60.0)
        gp("omega_max", 0.25 * math.pi)
        gp("obs_inflation", 1.8)
        gp("ego_responsibility", 1.0)
        gp("min_vel_confidence", 0.0)        # 首次測 0.0(全收速度,信任 KF)
        gp("obstacle_radius_max", 2.0)       # 障礙半徑封頂
        # ── slew 限速 ──
        gp("accel_v", 1.2)
        gp("accel_w", 3.0)
        gp("v_max", 1.0)
        gp("v_min", -1.0)
        gp("w_max", 1.2)
        # ── front_brake 簡化版 ──
        gp("front_brake_enable", True)
        gp("front_brake_slow_m", 0.6)
        gp("front_brake_stop_m", 0.5)
        gp("front_brake_emergency_m", 0.55)

        g = self.get_parameter
        self.ctrl_dt = 1.0 / float(g("ctrl_rate_hz").value)
        self.timeout_desired = float(g("timeout_desired_s").value)
        self.timeout_odom = float(g("timeout_odom_s").value)
        self.timeout_obs = float(g("timeout_obstacles_s").value)
        self.timeout_status = float(g("timeout_status_s").value)
        self.min_vel_conf = float(g("min_vel_confidence").value)
        self.obs_radius_max = float(g("obstacle_radius_max").value)
        self.culling_radius = float(g("culling_radius").value)
        self.static_vel_threshold = float(g("static_vel_threshold").value)
        self.accel_v = float(g("accel_v").value)
        self.accel_w = float(g("accel_w").value)
        self.v_max = float(g("v_max").value)
        self.v_min = float(g("v_min").value)
        self.w_max = float(g("w_max").value)
        self.front_brake_enable = bool(g("front_brake_enable").value)
        self.front_slow_m = float(g("front_brake_slow_m").value)
        self.front_stop_m = float(g("front_brake_stop_m").value)
        self.front_emergency_m = float(g("front_brake_emergency_m").value)

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

        # ── 狀態 ──
        self._lock = threading.Lock()
        self._des_v = 0.0; self._des_w = 0.0; self._des_t = 0.0
        self._robot = None        # (x, y, yaw, v_fwd)
        self._odom_t = 0.0
        self._obs_msg = None; self._obs_t = 0.0
        self._front_m = None; self._status_t = 0.0
        self._out_v = 0.0; self._out_w = 0.0
        self._front_brake = ""
        self._intervene_count = 0
        self._status_tick = 0
        self._hb_tick = 0
        self._driver = "rl"
        # 心跳 log 週期(每秒印一次「當下誰主導 + 三層速度」到 deploy log)
        gp("hb_rate_hz", 1.0)
        self.hb_every = max(1, round(float(g("ctrl_rate_hz").value) / max(0.1, float(g("hb_rate_hz").value))))

        # ── pub/sub ──
        self.create_subscription(Twist, str(g("topic_cmd_in").value), self._cb_desired, 10)
        self.create_subscription(Odometry, str(g("topic_odom").value), self._cb_odom, 10)
        self.create_subscription(TrackedObstacleArray, str(g("topic_obstacles").value),
                                 self._cb_obstacles, 10)
        self.create_subscription(String, str(g("topic_policy_status").value),
                                 self._cb_policy_status, 10)
        self.pub_cmd = self.create_publisher(Twist, str(g("topic_cmd_out").value), 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)

        self.create_timer(self.ctrl_dt, self._tick_ctrl)
        self.get_logger().info(
            f"[orca_safety] 訂 {g('topic_cmd_in').value} → 發 {g('topic_cmd_out').value} "
            f"@ {1.0/self.ctrl_dt:.0f}Hz | inflation={self.filter.obs_inflation} "
            f"ego_resp={self.filter.ego_responsibility} | front_brake stop={self.front_stop_m}m")

    # ────────────────────────────────────────────────────────────────────────
    # callbacks
    # ────────────────────────────────────────────────────────────────────────
    def _cb_desired(self, msg: Twist):
        with self._lock:
            self._des_v = msg.linear.x
            self._des_w = msg.angular.z
            self._des_t = time.monotonic()

    def _cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._lock:
            self._robot = (p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w),
                           msg.twist.twist.linear.x)
            self._odom_t = time.monotonic()

    def _cb_obstacles(self, msg: TrackedObstacleArray):
        with self._lock:
            self._obs_msg = msg
            self._obs_t = time.monotonic()

    def _cb_policy_status(self, msg: String):
        try:
            d = json.loads(msg.data)
            fm = d.get("front_m")
            fm = float(fm) if isinstance(fm, (int, float)) else None
        except (ValueError, TypeError):
            fm = None
        with self._lock:
            self._front_m = fm if (fm is not None and math.isfinite(fm)) else None
            self._status_t = time.monotonic()

    # ────────────────────────────────────────────────────────────────────────
    # 主迴圈(對稱 vo_safety_node._tick_ctrl,但 ORCA 取代 VO + 去 escape/commit)
    # ────────────────────────────────────────────────────────────────────────
    def _tick_ctrl(self):
        now = time.monotonic()
        with self._lock:
            robot = self._robot
            odom_t = self._odom_t
            des_v, des_w, des_t = self._des_v, self._des_w, self._des_t
            obs_msg = self._obs_msg
            obs_t = self._obs_t
            front_m, status_t = self._front_m, self._status_t

        # 看門狗:odom/desired 逾時 → hard-zero(fail-safe,不 slew)
        if robot is None or (now - odom_t) > self.timeout_odom:
            self._driver = "fail"
            self._publish_hard_zero(); self._publish_status(fail="odom", driver="fail"); return
        if (now - des_t) > self.timeout_desired:
            self._driver = "fail"
            self._publish_hard_zero(); self._publish_status(fail="desired", driver="fail"); return

        obs_stale = (now - obs_t) > self.timeout_obs
        status_stale = (now - status_t) > self.timeout_status

        # ORCA 演算法
        inp = self._build_orca_input(robot, des_v, des_w, obs_msg, obs_stale)
        out = self.filter.solve(inp, self.ctrl_dt)
        tv, tw = out.v_safe_linear, out.v_safe_omega

        if out.orca_active:
            self._intervene_count += 1
            if self._intervene_count % 10 == 1:
                self.get_logger().info(
                    f"[INTERVENE #{self._intervene_count}] "
                    f"v_pref=({out.v_pref_world[0]:+.2f},{out.v_pref_world[1]:+.2f}) "
                    f"v_safe=({out.v_safe_world[0]:+.2f},{out.v_safe_world[1]:+.2f}) "
                    f"→ v={tv:+.2f} ω={tw:+.2f}")

        # front_brake(最後蓋過 ORCA)
        tv, tw, fb = self._apply_front_brake(tv, tw, front_m, status_stale)
        if fb != self._front_brake and (fb or self._front_brake):
            self.get_logger().info(
                f"[front_brake] {self._front_brake or '·'}→{fb or '·'} front_m={front_m}")
        self._front_brake = fb

        # driver 歸因:當下輸出到底由誰主導(回答「rl or orca」)
        #   stop  = front_brake 硬煞(距離兜底,蓋過一切)
        #   orca  = ORCA 介入改寫(orca_active,可能還疊 slow 減速)
        #   rl    = 純放行 RL 意圖(ORCA 沒動手;大 ω 若在此列即 RL 自己要轉)
        n_dyn = len(out.vo_cone_data)
        if fb == "stop":
            driver = "stop"
        elif out.orca_active:
            driver = "orca"
        else:
            driver = "rl"
        self._driver = driver

        # 輸出:stop 走 brake(前進瞬間歸零),其他走 slew
        if fb == "stop":
            self._publish_brake(tv, tw)
        else:
            self._publish_slew(tv, tw)
        self._publish_status(out=out, obs_stale=obs_stale, front_brake=fb,
                             driver=driver, n_dyn=n_dyn)

        # ── 心跳(1Hz):把「當下 driver + RL意圖 vs 送出 + 障礙數」記進 deploy log ──
        self._hb_tick += 1
        if self._hb_tick % self.hb_every == 0:
            self.get_logger().info(
                f"[HB] driver={driver.upper()} "
                f"rl(v,w)=({des_v:+.2f},{des_w:+.2f}) "
                f"out(v,w)=({self._out_v:+.2f},{self._out_w:+.2f}) "
                f"orca_active={'T' if out.orca_active else 'F'} fb={fb or '·'} "
                f"dyn_obs={n_dyn} front_m={front_m}")

    # ────────────────────────────────────────────────────────────────────────
    # ORCA 適配:policy desired + odom + obstacles → ORCAInput(world frame)
    # ────────────────────────────────────────────────────────────────────────
    def _build_orca_input(self, robot, des_v, des_w, obs_msg, obs_stale):
        rx, ry, yaw, v_fwd = robot
        # robot world vel(差動底盤 body twist → world)
        robot_vx = math.cos(yaw) * v_fwd
        robot_vy = math.sin(yaw) * v_fwd
        # v_pref:policy desired (v,w) → world,用 yaw_next 預測(對齊 sim filter_step)
        yaw_next = yaw + des_w * self.ctrl_dt
        v_pref = (des_v * math.cos(yaw_next), des_v * math.sin(yaw_next))
        obs_pos, obs_vel, obs_radii, is_static = self._gather_obstacles(
            obs_msg, rx, ry, obs_stale)
        return ORCAInput(
            robot_pos=(rx, ry), robot_vel=(robot_vx, robot_vy), yaw=yaw,
            v_pref=v_pref, v_next_body=des_v, omega_rl=des_w,
            obs_pos=obs_pos, obs_vel=obs_vel, obs_radii=obs_radii, obs_is_static=is_static,
        )

    def _gather_obstacles(self, obs_msg, rx, ry, obs_stale):
        """TrackedObstacleArray → culled numpy arrays(world frame)。"""
        z2 = np.zeros((0, 2))
        if obs_stale or obs_msg is None or not obs_msg.obstacles:
            return z2, z2, np.zeros(0), np.zeros(0, dtype=bool)
        rows = []
        for ob in obs_msg.obstacles:
            ox, oy = ob.position.x, ob.position.y
            if ob.vel_confidence < self.min_vel_conf:
                ovx, ovy = 0.0, 0.0       # 低 confidence 速度歸零(當靜態位置避)
            else:
                ovx, ovy = ob.velocity.x, ob.velocity.y
            r = min(float(ob.radius), self.obs_radius_max)
            rows.append((ox, oy, ovx, ovy, r))
        if not rows:
            return z2, z2, np.zeros(0), np.zeros(0, dtype=bool)
        arr = np.array(rows, dtype=float)
        obs_pos = arr[:, :2]
        obs_vel = arr[:, 2:4]
        obs_radii = arr[:, 4]
        # culling:只保留 culling_radius 內、非 self
        rel = obs_pos - np.array([rx, ry])
        dist = np.linalg.norm(rel, axis=1)
        mask = (dist < self.culling_radius) & (dist > 0.01)
        obs_pos, obs_vel, obs_radii = obs_pos[mask], obs_vel[mask], obs_radii[mask]
        is_static = (np.linalg.norm(obs_vel, axis=1) < self.static_vel_threshold
                     if len(obs_pos) else np.zeros(0, dtype=bool))
        return obs_pos, obs_vel, obs_radii, is_static

    # ────────────────────────────────────────────────────────────────────────
    # front_brake 簡化版(借 vo_safety_node _apply_front_brake,去 defer/reverse/block_ratio)
    # ────────────────────────────────────────────────────────────────────────
    def _apply_front_brake(self, v, w, front_m, status_stale):
        """吃 policy status 的 front_m;≤stop/emergency 禁止前進;≤slow 線性縮速。"""
        if not self.front_brake_enable or front_m is None or status_stale:
            return v, w, ""
        f = front_m
        if f <= self.front_stop_m or f <= self.front_emergency_m:
            return min(v, 0.0), w, "stop"
        if f <= self.front_slow_m and v > 0.0:
            frac = (f - self.front_stop_m) / max(1e-3, self.front_slow_m - self.front_stop_m)
            return v * frac, w, "slow"
        return v, w, ""

    # ────────────────────────────────────────────────────────────────────────
    # 輸出(借 vo_safety_node _publish_slew/_publish_brake/_publish_hard_zero)
    # ────────────────────────────────────────────────────────────────────────
    def _publish_slew(self, target_v, target_w):
        max_dv = self.accel_v * self.ctrl_dt
        max_dw = self.accel_w * self.ctrl_dt
        self._out_v = _clamp(
            self._out_v + _clamp(target_v - self._out_v, -max_dv, max_dv), self.v_min, self.v_max)
        self._out_w = _clamp(
            self._out_w + _clamp(target_w - self._out_w, -max_dw, max_dw), -self.w_max, self.w_max)
        self._emit(self._out_v, self._out_w)

    def _publish_brake(self, target_v, target_w):
        """stop 專用:前進瞬間歸零(不走 slew 慢收,免多前衝撞)。"""
        max_dv = self.accel_v * self.ctrl_dt
        max_dw = self.accel_w * self.ctrl_dt
        if self._out_v > 0.0:
            self._out_v = 0.0
        self._out_v = _clamp(
            self._out_v + _clamp(target_v - self._out_v, -max_dv, max_dv), self.v_min, self.v_max)
        self._out_w = _clamp(
            self._out_w + _clamp(target_w - self._out_w, -max_dw, max_dw), -self.w_max, self.w_max)
        self._emit(self._out_v, self._out_w)

    def _publish_hard_zero(self):
        self._out_v = 0.0
        self._out_w = 0.0
        self._emit(0.0, 0.0)

    def _emit(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def _publish_status(self, out=None, fail="", obs_stale=False, front_brake="",
                        driver="rl", n_dyn=0):
        self._status_tick += 1
        if self._status_tick % 4 != 0:    # ~5Hz(20Hz/4)
            return
        d = {
            "orca_active": bool(out.orca_active) if out else False,
            "driver": driver,             # rl / orca / stop / fail(當下誰主導輸出)
            "v_out": round(self._out_v, 3),
            "w_out": round(self._out_w, 3),
            "v_des": round(self._des_v, 3),
            "w_des": round(self._des_w, 3),
            "front_brake": front_brake,
            "n_dyn_obs": n_dyn,
            "fail": fail,
            "obs_stale": obs_stale,
        }
        msg = String()
        msg.data = json.dumps(d)
        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ORCASafetyNode()
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
