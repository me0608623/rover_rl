#!/usr/bin/env python3
"""baseline_status_line.py — 消融實驗 baseline（dwa/pid/pid_vo）的前景簡易狀態行。

為什麼不用 status_tui：status_tui 幾乎所有欄位都從 /rover_rl_policy/status 解出來，
baseline 沒有 policy_node → 那個 topic 永遠不發，整個面板會卡在「等待 policy_node…」。
這裡改成純訂閱「不依賴 policy_node 的來源」：TF 車姿、送進 mux 的 cmd_vel、72-bin sweep、
往返測試狀態。純訂閱純列印，不影響任何控制。

由 deploy_baseline_shell.sh 前景啟動；也可單獨接已在跑的棧：
    source ~/rover_rl/install/setup.bash && source ~/rover_rl/setup_env.sh
    python3 ~/rover_rl/baseline_status_line.py --controller dwa
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformListener

# 與 policy_node / vo_safety_node 同一套扇區切法（bin36=前，每 bin 5°）
FRONT_BINS = slice(28, 45)
STALE_S = 3.0        # 超過此秒數沒更新 → 顯示為失聯


class BaselineStatusLine(Node):
    def __init__(self, controller: str, rate_hz: float) -> None:
        super().__init__("baseline_status_line")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("r_max_m", 20.0)
        self.declare_parameter("r_robot_m", 0.35)
        self._map_frame = self.get_parameter("map_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._r_max = float(self.get_parameter("r_max_m").value)
        self._r_robot = float(self.get_parameter("r_robot_m").value)

        self._controller = controller
        self._cmd = (0.0, 0.0)
        self._cmd_t = 0.0
        self._odom_v = 0.0
        self._front_m: float | None = None
        self._sweep_t = 0.0
        self._pp: dict | None = None
        self._vo: dict | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(Twist, "/input/nav_cmd_vel", self._on_cmd, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Float32MultiArray, "/rover_rl/lidar_sweep_72",
                                 self._on_sweep, 5)
        self.create_subscription(String, "/rover_rl/pingpong/status", self._on_pp, 5)
        self.create_subscription(String, "/vo_safety_node/status", self._on_vo, 5)

        self._tty = sys.stdout.isatty()
        self.create_timer(1.0 / max(rate_hz, 0.1), self._tick)

    # ── 訂閱 ──
    def _on_cmd(self, msg: Twist) -> None:
        self._cmd = (msg.linear.x, msg.angular.z)
        self._cmd_t = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        self._odom_v = msg.twist.twist.linear.x

    def _on_sweep(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != 72:
            return
        span = self._r_max - self._r_robot
        self._front_m = min(v * span + self._r_robot for v in msg.data[FRONT_BINS])
        self._sweep_t = time.monotonic()

    def _on_pp(self, msg: String) -> None:
        try:
            self._pp = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _on_vo(self, msg: String) -> None:
        try:
            self._vo = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    # ── 渲染 ──
    def _pose_str(self) -> str:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, Time())
        except Exception:
            return "pose=(等 TF)      "
        return (f"pose=({tf.transform.translation.x:6.2f},"
                f"{tf.transform.translation.y:6.2f})")

    def _tick(self) -> None:
        now = time.monotonic()
        parts = [f"{self._controller:<6}"]

        if self._pp is not None:
            st = self._pp.get("state", "-")
            tgt = self._pp.get("target") or self._pp.get("ready_point") or "-"
            parts.append(f"往返={st:<7} →{tgt:<4}")

        parts.append(self._pose_str())

        v, w = self._cmd
        cmd_stale = (now - self._cmd_t) > STALE_S if self._cmd_t else True
        parts.append(f"cmd=({v:+.2f},{w:+.2f}){'!斷' if cmd_stale else '  '}")
        parts.append(f"實測v={self._odom_v:+.2f}")

        if self._front_m is not None and (now - self._sweep_t) <= STALE_S:
            parts.append(f"前障={self._front_m:5.2f}m")
        else:
            parts.append("前障= --  ")

        if self._vo is not None:
            fail = self._vo.get("fail") or ""
            if fail:
                parts.append(f"VO={fail}")
            elif self._vo.get("obs_stale"):
                parts.append("VO=放行")
            else:
                parts.append("VO=監看")

        line = "  ".join(parts)
        if self._tty:
            print(f"\r{line}\033[K", end="", flush=True)
        else:
            print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="baseline")
    ap.add_argument("--rate", type=float, default=1.0)
    args, ros_args = ap.parse_known_args()

    print(f"[狀態行] controller={args.controller}｜Ctrl+C 離開並收棧")
    print("[狀態行] 欄位：往返測試狀態 / map 車姿 / 送進 mux 的 cmd / odom 實測 v / "
          "前方最近障礙 / VO 狀態")
    rclpy.init(args=ros_args)
    node = BaselineStatusLine(args.controller, args.rate)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C（SIGINT）或被 stop 腳本 SIGTERM 掉都算正常離開，不要噴 traceback
        pass
    finally:
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
