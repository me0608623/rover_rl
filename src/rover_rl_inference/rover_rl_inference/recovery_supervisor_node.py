"""Recovery supervisor for rover_rl navigation.

This node is a conservative cmd_vel wrapper:

    policy_node -> /rover_rl/cmd_vel_desired -> recovery_supervisor
        -> /input/nav_cmd_vel -> mux -> rover driver

It does not change the RL observation, model, action decoder, or training flow.
In NORMAL_RL it passes the RL command through. When the robot is front-blocked
and either stops making goal progress or the RL command hesitates, it briefly
stops, backs up if the rear sector is clear, chooses the more open side, drives
toward a short local goal, and then returns control to RL.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import time
from enum import Enum

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class RecoveryState(Enum):
    NORMAL_RL = "normal_rl"
    RECOVERY_STOP = "recovery_stop"
    RECOVERY_BACKUP = "recovery_backup"
    RECOVERY_SELECT_GAP = "recovery_select_gap"
    RECOVERY_GO_LOCAL = "recovery_go_local"
    RETURN_TO_RL = "return_to_rl"
    STUCK = "stuck"


class RecoverySupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("recovery_supervisor_node")

        gp = self.declare_parameter
        gp("topic_cmd_in", "/rover_rl/cmd_vel_desired")
        gp("topic_cmd_out", "/input/nav_cmd_vel")
        gp("topic_odom", "/odom")
        gp("topic_policy_status", "/rover_rl_policy/status")
        gp("ctrl_rate_hz", 20.0)
        gp("status_rate_hz", 5.0)
        gp("csv_log_enable", True)
        gp("csv_log_dir", "/home/aa/rover_rl/logs/recovery")
        gp("csv_flush_every_rows", 20)

        gp("timeout_cmd_s", 0.5)
        gp("timeout_odom_s", 0.3)
        gp("timeout_status_s", 1.0)

        gp("front_blocked_m", 0.8)
        gp("front_block_ratio_thresh", 0.9)
        gp("front_hard_blocked_m", 0.55)
        gp("front_hard_block_cmd_v_abs_mps", 0.08)
        gp("front_hard_block_cmd_w_abs_radps", 0.15)
        gp("front_blocked_dwell_s", 2.0)
        gp("front_clear_m", 1.2)
        gp("rear_clear_start_m", 0.8)
        gp("rear_clear_stop_m", 0.55)
        gp("side_clear_min_m", 0.55)
        gp("side_gap_trigger_enable", True)
        gp("side_gap_front_m", 0.60)
        gp("side_gap_ratio_max", 0.35)
        gp("side_gap_clear_m", 1.20)
        gp("side_gap_delta_m", 0.80)
        gp("side_gap_dwell_s", 1.2)
        gp("side_gap_cmd_v_abs_mps", 0.08)
        gp("side_gap_cmd_w_abs_radps", 0.75)
        gp("side_gap_v_max", 0.18)

        gp("no_progress_window_s", 2.5)
        gp("no_progress_min_delta_m", 0.15)
        gp("rl_hesitate_v_abs_mps", 0.05)
        gp("rl_hesitate_window_s", 1.0)
        gp("trigger_cooldown_s", 2.0)

        gp("stop_duration_s", 0.3)
        gp("backup_v_mps", -0.2)
        gp("backup_time_s", 1.2)
        gp("backup_dist_m", 0.4)
        gp("rotate_scan_w_radps", 0.35)
        gp("rotate_scan_time_s", 0.8)

        gp("local_goal_x_m", 0.7)
        gp("local_goal_y_m", 0.7)
        gp("local_goal_tolerance_m", 0.25)
        gp("local_go_min_time_s", 1.2)
        gp("local_ctrl_v_max", 0.35)
        gp("local_ctrl_w_max", 0.6)
        gp("local_ctrl_k_v", 0.5)
        gp("local_ctrl_k_w", 1.0)

        gp("recovery_max_time_s", 5.0)
        gp("max_fail_count", 3)
        gp("return_stop_s", 0.2)

        g = self.get_parameter
        self.topic_cmd_in = g("topic_cmd_in").get_parameter_value().string_value
        self.topic_cmd_out = g("topic_cmd_out").get_parameter_value().string_value
        self.topic_odom = g("topic_odom").get_parameter_value().string_value
        self.topic_policy_status = g("topic_policy_status").get_parameter_value().string_value
        self.ctrl_rate_hz = float(g("ctrl_rate_hz").value)
        self.status_rate_hz = float(g("status_rate_hz").value)
        self.csv_log_enable = bool(g("csv_log_enable").value)
        self.csv_log_dir = g("csv_log_dir").get_parameter_value().string_value
        self.csv_flush_every_rows = max(1, int(g("csv_flush_every_rows").value))

        self.timeout_cmd_s = float(g("timeout_cmd_s").value)
        self.timeout_odom_s = float(g("timeout_odom_s").value)
        self.timeout_status_s = float(g("timeout_status_s").value)

        self.front_blocked_m = float(g("front_blocked_m").value)
        self.front_block_ratio_thresh = float(g("front_block_ratio_thresh").value)
        self.front_hard_blocked_m = float(g("front_hard_blocked_m").value)
        self.front_hard_block_cmd_v_abs_mps = float(g("front_hard_block_cmd_v_abs_mps").value)
        self.front_hard_block_cmd_w_abs_radps = float(g("front_hard_block_cmd_w_abs_radps").value)
        self.front_blocked_dwell_s = float(g("front_blocked_dwell_s").value)
        self.front_clear_m = float(g("front_clear_m").value)
        self.rear_clear_start_m = float(g("rear_clear_start_m").value)
        self.rear_clear_stop_m = float(g("rear_clear_stop_m").value)
        self.side_clear_min_m = float(g("side_clear_min_m").value)
        self.side_gap_trigger_enable = bool(g("side_gap_trigger_enable").value)
        self.side_gap_front_m = float(g("side_gap_front_m").value)
        self.side_gap_ratio_max = float(g("side_gap_ratio_max").value)
        self.side_gap_clear_m = float(g("side_gap_clear_m").value)
        self.side_gap_delta_m = float(g("side_gap_delta_m").value)
        self.side_gap_dwell_s = float(g("side_gap_dwell_s").value)
        self.side_gap_cmd_v_abs_mps = float(g("side_gap_cmd_v_abs_mps").value)
        self.side_gap_cmd_w_abs_radps = float(g("side_gap_cmd_w_abs_radps").value)
        self.side_gap_v_max = float(g("side_gap_v_max").value)

        self.no_progress_window_s = float(g("no_progress_window_s").value)
        self.no_progress_min_delta_m = float(g("no_progress_min_delta_m").value)
        self.rl_hesitate_v_abs_mps = float(g("rl_hesitate_v_abs_mps").value)
        self.rl_hesitate_window_s = float(g("rl_hesitate_window_s").value)
        self.trigger_cooldown_s = float(g("trigger_cooldown_s").value)

        self.stop_duration_s = float(g("stop_duration_s").value)
        self.backup_v_mps = -abs(float(g("backup_v_mps").value))
        self.backup_time_s = float(g("backup_time_s").value)
        self.backup_dist_m = float(g("backup_dist_m").value)
        self.rotate_scan_w_radps = float(g("rotate_scan_w_radps").value)
        self.rotate_scan_time_s = float(g("rotate_scan_time_s").value)

        self.local_goal_x_m = float(g("local_goal_x_m").value)
        self.local_goal_y_m = abs(float(g("local_goal_y_m").value))
        self.local_goal_tolerance_m = float(g("local_goal_tolerance_m").value)
        self.local_go_min_time_s = float(g("local_go_min_time_s").value)
        self.local_ctrl_v_max = float(g("local_ctrl_v_max").value)
        self.local_ctrl_w_max = float(g("local_ctrl_w_max").value)
        self.local_ctrl_k_v = float(g("local_ctrl_k_v").value)
        self.local_ctrl_k_w = float(g("local_ctrl_k_w").value)

        self.recovery_max_time_s = float(g("recovery_max_time_s").value)
        self.max_fail_count = int(g("max_fail_count").value)
        self.return_stop_s = float(g("return_stop_s").value)

        self._state = RecoveryState.NORMAL_RL
        self._state_t = time.monotonic()
        self._recovery_t = 0.0
        self._last_recovery_end_t = -1e9
        self._fail_count = 0
        self._return_success = False
        self._last_trigger_reason = ""

        self._des_v = 0.0
        self._des_w = 0.0
        self._des_t = 0.0

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._odom_t = 0.0

        self._front_m: float | None = None
        self._front_block_ratio: float | None = None
        self._front_blocked_since: float | None = None
        self._side_gap_blocked_since: float | None = None
        self._back_m: float | None = None
        self._left_m: float | None = None
        self._right_m: float | None = None
        self._goal_dist: float | None = None
        self._goal_ang_deg: float | None = None
        self._status_t = 0.0

        self._goal_hist: collections.deque[tuple[float, float]] = collections.deque()
        self._cmd_hist: collections.deque[tuple[float, float]] = collections.deque()

        self._backup_start_xy = (0.0, 0.0)
        self._selected_side = 0  # +1 left, -1 right
        self._local_goal_odom: tuple[float, float] | None = None
        self._direct_side_gap = False
        self._rot_scan_dir = 1.0
        self._last_out_v = 0.0
        self._last_out_w = 0.0

        self._csv_tick_fh = None
        self._csv_event_fh = None
        self._csv_tick = None
        self._csv_event = None
        self._csv_rows_since_flush = 0
        self._csv_tick_path = ""
        self._csv_event_path = ""
        self._init_csv_logger()

        self.pub_cmd = self.create_publisher(Twist, self.topic_cmd_out, 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.create_subscription(Twist, self.topic_cmd_in, self._cb_cmd, 10)
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)
        self.create_subscription(String, self.topic_policy_status, self._cb_policy_status, 10)

        self.create_timer(1.0 / max(self.ctrl_rate_hz, 1.0), self._tick_ctrl)
        self.create_timer(1.0 / max(self.status_rate_hz, 1.0), self._publish_status)

        self.get_logger().info(
            f"recovery_supervisor 啟動：{self.topic_cmd_in} -> {self.topic_cmd_out}, "
            f"front_blocked<{self.front_blocked_m:.2f}m, rear_start>{self.rear_clear_start_m:.2f}m"
        )
        if self.csv_log_enable:
            self.get_logger().info(
                f"recovery CSV: ticks={self._csv_tick_path} events={self._csv_event_path}"
            )

    def _init_csv_logger(self) -> None:
        if not self.csv_log_enable:
            return
        os.makedirs(os.path.expanduser(self.csv_log_dir), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._csv_tick_path = os.path.join(
            os.path.expanduser(self.csv_log_dir), f"recovery_ticks_{stamp}.csv"
        )
        self._csv_event_path = os.path.join(
            os.path.expanduser(self.csv_log_dir), f"recovery_events_{stamp}.csv"
        )
        self._csv_tick_fh = open(self._csv_tick_path, "w", newline="", encoding="utf-8")
        self._csv_event_fh = open(self._csv_event_path, "w", newline="", encoding="utf-8")
        tick_fields = [
            "t", "state", "state_age_s", "recovery_elapsed_s", "fail_count",
            "front_m", "front_block_ratio", "front_blocked_age_s",
            "back_m", "left_m", "right_m",
            "goal_dist", "goal_ang_deg",
            "des_v", "des_w", "out_v", "out_w",
            "odom_x", "odom_y", "odom_yaw",
            "backup_dist_m", "selected_side",
            "local_goal_x", "local_goal_y", "local_err_x", "local_err_y",
            "direct_side_gap", "side_gap_blocked_age_s", "trigger_reason",
        ]
        event_fields = [
            "t", "event", "from_state", "to_state", "reason",
            "front_m", "front_block_ratio", "front_blocked_age_s",
            "back_m", "left_m", "right_m",
            "goal_dist", "goal_ang_deg",
            "des_v", "des_w", "out_v", "out_w",
            "odom_x", "odom_y", "odom_yaw",
            "backup_dist_m", "selected_side", "fail_count",
            "direct_side_gap", "side_gap_blocked_age_s", "trigger_reason",
        ]
        self._csv_tick = csv.DictWriter(self._csv_tick_fh, fieldnames=tick_fields)
        self._csv_event = csv.DictWriter(self._csv_event_fh, fieldnames=event_fields)
        self._csv_tick.writeheader()
        self._csv_event.writeheader()
        self._csv_tick_fh.flush()
        self._csv_event_fh.flush()

    def _cb_cmd(self, msg: Twist) -> None:
        self._des_v = float(msg.linear.x)
        self._des_w = float(msg.angular.z)
        self._des_t = time.monotonic()

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._odom_x = float(p.x)
        self._odom_y = float(p.y)
        self._odom_yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        self._odom_t = time.monotonic()

    def _cb_policy_status(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        now = time.monotonic()
        self._front_m = _safe_float(d.get("front_m"))
        self._front_block_ratio = _safe_float(d.get("front_block_ratio"))
        self._back_m = _safe_float(d.get("back_m"))
        self._left_m = _safe_float(d.get("left_m"))
        self._right_m = _safe_float(d.get("right_m"))
        self._goal_dist = _safe_float(d.get("goal_dist"))
        self._goal_ang_deg = _safe_float(d.get("goal_ang_deg"))
        self._status_t = now
        if self._goal_dist is not None:
            self._goal_hist.append((now, self._goal_dist))
            self._trim_hist(self._goal_hist, now, self.no_progress_window_s + 0.5)

    @staticmethod
    def _trim_hist(hist: collections.deque, now: float, keep_s: float) -> None:
        while hist and now - hist[0][0] > keep_s:
            hist.popleft()

    def _set_state(self, state: RecoveryState, reason: str = "") -> None:
        if state == self._state:
            return
        old = self._state
        self._state = state
        self._state_t = time.monotonic()
        self._log_event("state_transition", old, state, reason)
        self.get_logger().warn(
            f"recovery state: {old.value} -> {state.value}"
            + (f" ({reason})" if reason else "")
        )

    def _tick_ctrl(self) -> None:
        now = time.monotonic()
        if now - self._odom_t > self.timeout_odom_s:
            self._emit(0.0, 0.0)
            return
        if now - self._des_t > self.timeout_cmd_s:
            self._emit(0.0, 0.0)
            return

        self._cmd_hist.append((now, abs(self._des_v)))
        self._trim_hist(self._cmd_hist, now, self.rl_hesitate_window_s + 0.2)

        if self._state == RecoveryState.NORMAL_RL:
            if self._should_trigger_recovery(now):
                self._recovery_t = now
                self._backup_start_xy = (self._odom_x, self._odom_y)
                self._local_goal_odom = None
                if self._direct_side_gap:
                    self._set_state(RecoveryState.RECOVERY_SELECT_GAP, self._last_trigger_reason)
                else:
                    self._set_state(RecoveryState.RECOVERY_STOP, self._last_trigger_reason)
                self._emit(0.0, 0.0)
            else:
                self._emit(self._des_v, self._des_w)
            return

        if self._state == RecoveryState.STUCK:
            self._emit(0.0, 0.0)
            return

        if self._is_active_recovery_state() and now - self._recovery_t > self.recovery_max_time_s:
            self._fail_or_return("recovery timeout")
            self._emit(0.0, 0.0)
            return

        if self._state == RecoveryState.RECOVERY_STOP:
            self._emit(0.0, 0.0)
            if now - self._state_t >= self.stop_duration_s:
                self._backup_start_xy = (self._odom_x, self._odom_y)
                if self._rear_clear_for_start():
                    self._set_state(RecoveryState.RECOVERY_BACKUP, "rear clear")
                else:
                    self._rot_scan_dir = 1.0 if self._left_score() >= self._right_score() else -1.0
                    self._set_state(RecoveryState.RECOVERY_BACKUP, "rear blocked, rotate scan")
            return

        if self._state == RecoveryState.RECOVERY_BACKUP:
            if self._rear_clear_for_continue():
                dist = math.hypot(self._odom_x - self._backup_start_xy[0],
                                  self._odom_y - self._backup_start_xy[1])
                if (dist >= self.backup_dist_m
                        or now - self._state_t >= self.backup_time_s
                        or not self._rear_clear_for_continue()):
                    self._set_state(RecoveryState.RECOVERY_SELECT_GAP, "backup done")
                    self._emit(0.0, 0.0)
                else:
                    self._emit(self.backup_v_mps, 0.0)
            else:
                if now - self._state_t >= self.rotate_scan_time_s:
                    self._set_state(RecoveryState.RECOVERY_SELECT_GAP, "rotate scan done")
                    self._emit(0.0, 0.0)
                else:
                    self._emit(0.0, self._rot_scan_dir * abs(self.rotate_scan_w_radps))
            return

        if self._state == RecoveryState.RECOVERY_SELECT_GAP:
            self._select_local_goal()
            self._set_state(RecoveryState.RECOVERY_GO_LOCAL, "local goal selected")
            self._emit(0.0, 0.0)
            return

        if self._state == RecoveryState.RECOVERY_GO_LOCAL:
            min_go_done = now - self._state_t >= self.local_go_min_time_s
            if self._local_goal_reached() or (min_go_done and self._front_clear_for_rl()):
                self._return_success = True
                self._set_state(RecoveryState.RETURN_TO_RL, "clear/reached")
                self._emit(0.0, 0.0)
            else:
                v, w = self._local_controller()
                if self._front_m is not None and self._front_m < self.rear_clear_stop_m:
                    if self._direct_side_gap and self._selected_side_has_clearance():
                        v = min(v, self.side_gap_v_max)
                    else:
                        v = min(v, 0.0)
                self._emit(v, w)
            return

        if self._state == RecoveryState.RETURN_TO_RL:
            self._emit(0.0, 0.0)
            if now - self._state_t >= self.return_stop_s:
                self._last_recovery_end_t = now
                if self._return_success or self._front_clear_for_rl():
                    self._fail_count = 0
                self._return_success = False
                self._direct_side_gap = False
                self._set_state(RecoveryState.NORMAL_RL, "return control")

    def _is_active_recovery_state(self) -> bool:
        return self._state in (
            RecoveryState.RECOVERY_STOP,
            RecoveryState.RECOVERY_BACKUP,
            RecoveryState.RECOVERY_SELECT_GAP,
            RecoveryState.RECOVERY_GO_LOCAL,
        )

    def _should_trigger_recovery(self, now: float) -> bool:
        if now - self._status_t > self.timeout_status_s:
            return False
        if now - self._last_recovery_end_t < self.trigger_cooldown_s:
            return False
        self._direct_side_gap = False
        front_blocked = self._front_blocked_by_lidar_bins(now)
        side_gap_blocked = self._side_gap_blocked(now)
        if not front_blocked and not side_gap_blocked:
            return False
        no_progress = self._no_progress(now)
        hesitating = self._rl_hesitating(now)
        age = (
            0.0 if self._front_blocked_since is None
            else now - self._front_blocked_since
        )
        side_age = (
            0.0 if self._side_gap_blocked_since is None
            else now - self._side_gap_blocked_since
        )
        self._direct_side_gap = side_gap_blocked and side_age >= self.side_gap_dwell_s
        mode = "side_gap_direct" if self._direct_side_gap else "front_block_backup"
        self._last_trigger_reason = (
            f"mode={mode}, "
            f"front_m={self._front_m:.2f}, "
            f"front_block_ratio={self._front_block_ratio:.2f}, "
            f"hard_close={self._front_hard_close_and_stopped()}, "
            f"blocked_age={age:.1f}s, "
            f"side_gap_age={side_age:.1f}s, "
            f"left={self._left_score():.2f}, right={self._right_score():.2f}, "
            f"no_progress={no_progress}, rl_hesitating={hesitating}"
        )
        return self._direct_side_gap or (front_blocked and age >= self.front_blocked_dwell_s)

    def _front_blocked_by_lidar_bins(self, now: float) -> bool:
        fill_blocked = (
            self._front_m is not None
            and self._front_m <= self.front_blocked_m
            and self._front_block_ratio is not None
            and self._front_block_ratio >= self.front_block_ratio_thresh
        )
        blocked = fill_blocked or self._front_hard_close_and_stopped()
        if blocked:
            if self._front_blocked_since is None:
                self._front_blocked_since = now
        else:
            self._front_blocked_since = None
        return blocked

    def _side_gap_blocked(self, now: float) -> bool:
        if not self.side_gap_trigger_enable:
            self._side_gap_blocked_since = None
            return False
        left = self._left_score()
        right = self._right_score()
        best = max(left, right)
        worst = min(left, right)
        blocked = (
            self._front_m is not None
            and self._front_m <= self.side_gap_front_m
            and self._front_block_ratio is not None
            and self._front_block_ratio <= self.side_gap_ratio_max
            and best >= self.side_gap_clear_m
            and (best - worst) >= self.side_gap_delta_m
            and abs(self._des_v) <= self.side_gap_cmd_v_abs_mps
            and abs(self._des_w) <= self.side_gap_cmd_w_abs_radps
        )
        if blocked:
            if self._side_gap_blocked_since is None:
                self._side_gap_blocked_since = now
        else:
            self._side_gap_blocked_since = None
        return blocked

    def _front_hard_close_and_stopped(self) -> bool:
        return (
            self._front_m is not None
            and self._front_m <= self.front_hard_blocked_m
            and abs(self._des_v) <= self.front_hard_block_cmd_v_abs_mps
            and abs(self._des_w) <= self.front_hard_block_cmd_w_abs_radps
        )

    def _no_progress(self, now: float) -> bool:
        self._trim_hist(self._goal_hist, now, self.no_progress_window_s + 0.5)
        if len(self._goal_hist) < 2:
            return False
        oldest_t, oldest_d = self._goal_hist[0]
        newest_t, newest_d = self._goal_hist[-1]
        if newest_t - oldest_t < self.no_progress_window_s:
            return False
        return (oldest_d - newest_d) < self.no_progress_min_delta_m

    def _rl_hesitating(self, now: float) -> bool:
        self._trim_hist(self._cmd_hist, now, self.rl_hesitate_window_s + 0.2)
        if not self._cmd_hist:
            return False
        if self._cmd_hist[-1][0] - self._cmd_hist[0][0] < self.rl_hesitate_window_s:
            return False
        return max(v for _, v in self._cmd_hist) < self.rl_hesitate_v_abs_mps

    def _rear_clear_for_start(self) -> bool:
        return self._back_m is not None and self._back_m > self.rear_clear_start_m

    def _rear_clear_for_continue(self) -> bool:
        return self._back_m is not None and self._back_m > self.rear_clear_stop_m

    def _front_clear_for_rl(self) -> bool:
        if self._front_m is None:
            return False
        if self._goal_ang_deg is not None and abs(self._goal_ang_deg) > 60.0:
            return False
        return self._front_m >= self.front_clear_m

    def _left_score(self) -> float:
        return self._left_m if self._left_m is not None else 0.0

    def _right_score(self) -> float:
        return self._right_m if self._right_m is not None else 0.0

    def _selected_side_has_clearance(self) -> bool:
        if self._selected_side > 0:
            return self._left_score() >= self.side_gap_clear_m
        if self._selected_side < 0:
            return self._right_score() >= self.side_gap_clear_m
        return False

    def _select_local_goal(self) -> None:
        left = self._left_score()
        right = self._right_score()
        if left < self.side_clear_min_m and right < self.side_clear_min_m:
            self._selected_side = 1 if left >= right else -1
        else:
            self._selected_side = 1 if left >= right else -1

        gx_b = self.local_goal_x_m
        gy_b = self._selected_side * self.local_goal_y_m
        c = math.cos(self._odom_yaw)
        s = math.sin(self._odom_yaw)
        gx = self._odom_x + c * gx_b - s * gy_b
        gy = self._odom_y + s * gx_b + c * gy_b
        self._local_goal_odom = (gx, gy)

    def _local_goal_body_error(self) -> tuple[float, float] | None:
        if self._local_goal_odom is None:
            return None
        dx = self._local_goal_odom[0] - self._odom_x
        dy = self._local_goal_odom[1] - self._odom_y
        c = math.cos(-self._odom_yaw)
        s = math.sin(-self._odom_yaw)
        return c * dx - s * dy, s * dx + c * dy

    def _local_goal_reached(self) -> bool:
        err = self._local_goal_body_error()
        if err is None:
            return True
        return math.hypot(err[0], err[1]) <= self.local_goal_tolerance_m

    def _local_controller(self) -> tuple[float, float]:
        err = self._local_goal_body_error()
        if err is None:
            return 0.0, 0.0
        ex, ey = err
        dist = math.hypot(ex, ey)
        heading = math.atan2(ey, max(ex, 1e-3))
        v = _clamp(self.local_ctrl_k_v * dist, 0.0, self.local_ctrl_v_max)
        if abs(heading) > math.radians(65.0):
            v *= 0.3
        w = _clamp(self.local_ctrl_k_w * heading,
                   -self.local_ctrl_w_max, self.local_ctrl_w_max)
        return v, w

    def _fail_or_return(self, reason: str) -> None:
        self._return_success = False
        self._fail_count += 1
        if self._fail_count >= self.max_fail_count:
            self._set_state(RecoveryState.STUCK, reason)
        else:
            self._set_state(RecoveryState.RETURN_TO_RL, reason)

    def _emit(self, v: float, w: float) -> None:
        self._last_out_v = float(v)
        self._last_out_w = float(w)
        msg = Twist()
        msg.linear.x = self._last_out_v
        msg.angular.z = self._last_out_w
        self.pub_cmd.publish(msg)
        self._log_tick(time.monotonic())

    def _backup_dist(self) -> float:
        return math.hypot(self._odom_x - self._backup_start_xy[0],
                          self._odom_y - self._backup_start_xy[1])

    def _recovery_elapsed(self, now: float) -> float:
        if self._state in (RecoveryState.NORMAL_RL, RecoveryState.STUCK):
            return 0.0
        return max(0.0, now - self._recovery_t)

    def _float_or_empty(self, v):
        return "" if v is None else v

    def _front_blocked_age(self, now: float) -> float | None:
        if self._front_blocked_since is None:
            return None
        return max(0.0, now - self._front_blocked_since)

    def _side_gap_blocked_age(self, now: float) -> float | None:
        if self._side_gap_blocked_since is None:
            return None
        return max(0.0, now - self._side_gap_blocked_since)

    def _snapshot_row(self, now: float) -> dict:
        local_x = local_y = ""
        err_x = err_y = ""
        if self._local_goal_odom is not None:
            local_x, local_y = self._local_goal_odom
            err = self._local_goal_body_error()
            if err is not None:
                err_x, err_y = err
        return {
            "t": f"{now:.3f}",
            "state": self._state.value,
            "state_age_s": f"{now - self._state_t:.3f}",
            "recovery_elapsed_s": f"{self._recovery_elapsed(now):.3f}",
            "fail_count": self._fail_count,
            "front_m": self._float_or_empty(self._front_m),
            "front_block_ratio": self._float_or_empty(self._front_block_ratio),
            "front_blocked_age_s": self._float_or_empty(self._front_blocked_age(now)),
            "back_m": self._float_or_empty(self._back_m),
            "left_m": self._float_or_empty(self._left_m),
            "right_m": self._float_or_empty(self._right_m),
            "goal_dist": self._float_or_empty(self._goal_dist),
            "goal_ang_deg": self._float_or_empty(self._goal_ang_deg),
            "des_v": f"{self._des_v:.4f}",
            "des_w": f"{self._des_w:.4f}",
            "out_v": f"{self._last_out_v:.4f}",
            "out_w": f"{self._last_out_w:.4f}",
            "odom_x": f"{self._odom_x:.4f}",
            "odom_y": f"{self._odom_y:.4f}",
            "odom_yaw": f"{self._odom_yaw:.4f}",
            "backup_dist_m": f"{self._backup_dist():.4f}",
            "selected_side": self._selected_side,
            "local_goal_x": self._float_or_empty(local_x),
            "local_goal_y": self._float_or_empty(local_y),
            "local_err_x": self._float_or_empty(err_x),
            "local_err_y": self._float_or_empty(err_y),
            "direct_side_gap": int(self._direct_side_gap),
            "side_gap_blocked_age_s": self._float_or_empty(self._side_gap_blocked_age(now)),
            "trigger_reason": self._last_trigger_reason,
        }

    def _log_tick(self, now: float) -> None:
        if self._csv_tick is None:
            return
        self._csv_tick.writerow(self._snapshot_row(now))
        self._csv_rows_since_flush += 1
        if self._csv_rows_since_flush >= self.csv_flush_every_rows:
            self._csv_rows_since_flush = 0
            self._csv_tick_fh.flush()

    def _log_event(
        self,
        event: str,
        old_state: RecoveryState | None,
        new_state: RecoveryState | None,
        reason: str,
    ) -> None:
        if self._csv_event is None:
            return
        now = time.monotonic()
        row = self._snapshot_row(now)
        row.update({
            "event": event,
            "from_state": "" if old_state is None else old_state.value,
            "to_state": "" if new_state is None else new_state.value,
            "reason": reason,
        })
        keep = self._csv_event.fieldnames or []
        self._csv_event.writerow({k: row.get(k, "") for k in keep})
        self._csv_event_fh.flush()

    def _publish_status(self) -> None:
        d = {
            "state": self._state.value,
            "fail_count": self._fail_count,
            "trigger_reason": self._last_trigger_reason,
            "front_m": None if self._front_m is None else round(self._front_m, 2),
            "front_block_ratio": (
                None if self._front_block_ratio is None else round(self._front_block_ratio, 2)
            ),
            "front_blocked_age_s": (
                None if self._front_blocked_age(time.monotonic()) is None
                else round(self._front_blocked_age(time.monotonic()), 2)
            ),
            "direct_side_gap": self._direct_side_gap,
            "side_gap_blocked_age_s": (
                None if self._side_gap_blocked_age(time.monotonic()) is None
                else round(self._side_gap_blocked_age(time.monotonic()), 2)
            ),
            "back_m": None if self._back_m is None else round(self._back_m, 2),
            "left_m": None if self._left_m is None else round(self._left_m, 2),
            "right_m": None if self._right_m is None else round(self._right_m, 2),
            "goal_dist": None if self._goal_dist is None else round(self._goal_dist, 2),
            "selected_side": self._selected_side,
            "local_goal_odom": (
                None if self._local_goal_odom is None
                else [round(self._local_goal_odom[0], 2), round(self._local_goal_odom[1], 2)]
            ),
        }
        self.pub_status.publish(String(data=json.dumps(d)))


def main() -> None:
    rclpy.init()
    node = RecoverySupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, "_csv_tick_fh", None) is not None:
            node._csv_tick_fh.flush()
            node._csv_tick_fh.close()
        if getattr(node, "_csv_event_fh", None) is not None:
            node._csv_event_fh.flush()
            node._csv_event_fh.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
