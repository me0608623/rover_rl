#!/usr/bin/env python3
"""端到端延遲預算 — 實機線上量測（S1 感測 / S2 前處理 / S3 推論取樣）.

純訂閱觀察者：不發任何 topic、不改任何參數，對推論零影響。補 latency_budget.py
（離線 bag 只能量 S4/S5）量不到的前段：

  S1 感測    /velodyne_points header.stamp → 本機收到（含掃描封包累積 + driver + 傳輸）
  S2 前處理  點雲收到 → /rover_rl/lidar_sweep_72 收到（含 10Hz timer 取樣 + sweep 計算）
  S3 推論    policy status 的 sweep_age_ms（5Hz 取樣造成的 obs 老化）+ infer_ms（純計算）
  S4 指令    /rover_rl/cmd_vel_desired → /output/cmd_vel（同值配對，跨 wrapper 全鏈）

前提：velodyne driver + lidar_preprocessor 在跑（S3 還需 policy_node 在 nav 模式推論中）。
S1 假設 driver 與本機同一時鐘（同機或已 NTP 同步）；跨機未同步時 S1 不可信，會標記警告。

用法：
  source /opt/ros/humble/setup.bash && source ~/rover_rl/install/setup.bash
  source ~/rover_rl/setup_env.sh
  python3 scripts/latency_probe.py --secs 60
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return {
        "n": n, "p50": s[n // 2], "p90": s[min(int(n * 0.9), n - 1)],
        "max": s[-1], "mean": statistics.fmean(s),
    }


def _fmt(label: str, st: dict | None, note: str = "") -> str:
    if st is None:
        return f"  {label:26s} 無樣本 {note}"
    return (f"  {label:26s} p50 {st['p50']:7.1f} ms  p90 {st['p90']:7.1f} ms  "
            f"max {st['max']:7.1f} ms  n={st['n']}{note}")


class LatencyProbe(Node):
    def __init__(self, args):
        super().__init__("rover_rl_latency_probe")
        self.args = args
        self.s1: list[float] = []          # header.stamp → 收到
        self.s2: list[float] = []          # 點雲收到 → sweep 收到
        self.s4: list[float] = []          # cmd_vel_desired → output/cmd_vel（同值配對）
        self.infer_ms: list[float] = []
        self.sweep_age_ms: list[float] = []
        self.lag_ms: list[float] = []
        self.cloud_iv: list[float] = []
        self.sweep_iv: list[float] = []
        self._t_cloud: float | None = None
        self._t_cloud_prev: float | None = None
        self._t_sweep_prev: float | None = None
        self._sweep_consumed = True        # 一筆點雲只配一次 sweep，避免重複計數
        self._desired: list[tuple[float, float, float]] = []   # (t, v, w)
        self._clock_warn = False

        self.create_subscription(PointCloud2, args.topic_cloud, self._cb_cloud, SENSOR_QOS)
        self.create_subscription(Float32MultiArray, args.topic_sweep, self._cb_sweep, 10)
        self.create_subscription(String, args.topic_status, self._cb_status, 10)
        self.create_subscription(Twist, args.topic_desired, self._cb_desired, 20)
        self.create_subscription(Twist, args.topic_out, self._cb_out, 20)

        self._t0 = time.monotonic()
        self.create_timer(5.0, self._tick_report)
        self.get_logger().info(
            f"延遲探針啟動：{args.topic_cloud} / {args.topic_sweep} / {args.topic_status}"
            f"（{args.secs}s 後自動結束）"
        )

    # ── S1：感測 ──
    def _cb_cloud(self, msg: PointCloud2) -> None:
        now_mono = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        age_ms = (now_ros - stamp) * 1e3
        if -50.0 < age_ms < 2000.0:
            self.s1.append(age_ms)
        elif not self._clock_warn:
            self._clock_warn = True
            self.get_logger().warn(
                f"header.stamp 與本機時鐘差 {age_ms:.0f} ms（超出合理範圍）→ "
                f"S1 感測延遲不可信，driver 與本機時鐘未同步"
            )
        if self._t_cloud is not None:
            self.cloud_iv.append((now_mono - self._t_cloud) * 1e3)
        self._t_cloud = now_mono
        self._sweep_consumed = False

    # ── S2：前處理 ──
    def _cb_sweep(self, _msg: Float32MultiArray) -> None:
        now = time.monotonic()
        if self._t_sweep_prev is not None:
            self.sweep_iv.append((now - self._t_sweep_prev) * 1e3)
        self._t_sweep_prev = now
        if self._t_cloud is not None and not self._sweep_consumed:
            self.s2.append((now - self._t_cloud) * 1e3)
            self._sweep_consumed = True

    # ── S3：推論（讀 policy 自報的欄位）──
    def _cb_status(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        for key, sink in (("infer_ms", self.infer_ms),
                          ("sweep_age_ms", self.sweep_age_ms),
                          ("lag_ms", self.lag_ms)):
            val = d.get(key)
            if isinstance(val, (int, float)):
                sink.append(float(val))

    # ── S4：指令鏈（同值配對）──
    def _cb_desired(self, msg: Twist) -> None:
        self._desired.append((time.monotonic(), msg.linear.x, msg.angular.z))
        if len(self._desired) > 200:
            del self._desired[:100]

    def _cb_out(self, msg: Twist) -> None:
        if abs(msg.linear.x) < 1e-3 and abs(msg.angular.z) < 1e-3:
            return                                   # 靜止時整條鏈都是 0，配對無鑑別力
        now = time.monotonic()
        for t, v, w in reversed(self._desired):
            if now - t > 1.0:
                break
            if abs(v - msg.linear.x) < 1e-6 and abs(w - msg.angular.z) < 1e-6:
                self.s4.append((now - t) * 1e3)
                break

    # ── 報表 ──
    def _tick_report(self) -> None:
        el = time.monotonic() - self._t0
        print(f"\n──── 延遲探針 {el:.0f}s ────")
        self.print_report()
        if el >= self.args.secs:
            raise KeyboardInterrupt

    def print_report(self) -> None:
        warn = "  ⚠時鐘未同步" if self._clock_warn else ""
        print(_fmt("S1 感測(stamp→收到)", _stats(self.s1), warn))
        print(_fmt("S2 前處理(雲→sweep)", _stats(self.s2)))
        print(_fmt("S3 sweep 老化(取樣)", _stats(self.sweep_age_ms)))
        print(_fmt("S3 推論純計算", _stats(self.infer_ms)))
        print(_fmt("S4 desired→output", _stats(self.s4)))
        print(_fmt("S5 致動死時間(policy)", _stats(self.lag_ms)))
        print(_fmt("  參考: 點雲到達間隔", _stats(self.cloud_iv)))
        print(_fmt("  參考: sweep 間隔", _stats(self.sweep_iv)))
        parts = [_stats(x) for x in (self.s1, self.s2, self.sweep_age_ms,
                                     self.infer_ms, self.s4, self.lag_ms)]
        if all(p is not None for p in parts):
            tot = sum(p["p50"] for p in parts)
            print(f"  {'端到端 p50 累計':26s} {tot:7.1f} ms  "
                  f"（= 感測+前處理+取樣+推論+指令鏈+致動）")
        else:
            miss = [n for n, p in zip(["S1", "S2", "S3老化", "S3計算", "S4", "S5"], parts)
                    if p is None]
            print(f"  端到端累計：尚缺 {', '.join(miss)}（相關節點未在跑或車未移動）")


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端延遲預算 — 實機線上量測")
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--topic-cloud", default="/velodyne_points")
    ap.add_argument("--topic-sweep", default="/rover_rl/lidar_sweep_72")
    ap.add_argument("--topic-status", default="/rover_rl_policy/status")
    ap.add_argument("--topic-desired", default="/rover_rl/cmd_vel_desired")
    ap.add_argument("--topic-out", default="/output/cmd_vel")
    args = ap.parse_args()

    rclpy.init()
    node = LatencyProbe(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n════ 最終統計 ════")
        node.print_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
