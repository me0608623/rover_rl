#!/usr/bin/env python3
"""ω 通道「原地轉圈」系統辨識 — 不用架空車子.

只發角速度命令、線速度恆為 0 → 車原地打轉、不會跑掉，輪子在地上量到的是真實載重動態。
一格一格灌階梯轉速（左右交替，車不會越轉越偏），同時錄 odom 實測角速度，
跑完自動算：死區(deadband) / 靜態增益(聽幾成話) / 死時間(慢半拍) / 時間常數(多久跟上)。

⚠ 安全：
  - 找一小塊空地（車原地轉的半徑即可），手拿遙控器/急停。
  - 跑之前先把 policy 停掉（deploy_rl_stop），只留底盤 driver + odom 在跑，避免兩邊搶命令。
  - Ctrl+C 立即送 0 停車。--dry 只印排程不發命令（先看會做什麼）。

用法：
  source /opt/ros/humble/setup.bash && source ~/rover_rl/setup_env.sh
  source ~/rover2_ws/install/setup.bash
  python3 scripts/sysid_inject.py                 # 預設排程
  python3 scripts/sysid_inject.py --dry           # 只看排程不動車
  python3 scripts/sysid_inject.py --cmd-topic /cmd_vel   # 改命令 topic
"""
from __future__ import annotations
import argparse, csv, math, os, time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def _yaw_rate_from_odom(msg):
    return msg.twist.twist.angular.z


class SysIdInject(Node):
    def __init__(self, args):
        super().__init__("rover_rl_sysid_inject")
        self.cmd_topic = args.cmd_topic
        self.rate_hz = 20.0
        self.levels = args.levels
        self.hold_s = args.hold
        self.pause_s = args.pause
        self.dry = args.dry
        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self._odom_w = 0.0
        self._have_odom = False
        self.create_subscription(Odometry, args.odom_topic, self._cb_odom, 10)
        # 紀錄：(t_rel, cmd_w, odom_w) + 區段表 (t0, t1, cmd_level)
        self.log = []
        self.segments = []
        self._t0 = time.monotonic()
        self._cur_cmd = 0.0

    def _cb_odom(self, msg):
        self._odom_w = _yaw_rate_from_odom(msg)
        self._have_odom = True

    def _publish(self, w):
        self._cur_cmd = w
        if not self.dry:
            t = Twist()
            t.linear.x = 0.0           # 永遠不前進
            t.angular.z = float(w)
            self.pub.publish(t)

    def _spin_for(self, dur, w, label):
        """按住命令 w 持續 dur 秒，期間 20Hz 發命令 + 錄資料."""
        t_seg0 = time.monotonic() - self._t0
        n = max(1, int(dur * self.rate_hz))
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.0)
            self._publish(w)
            self.log.append((time.monotonic() - self._t0, w, self._odom_w))
            time.sleep(1.0 / self.rate_hz)
        t_seg1 = time.monotonic() - self._t0
        if abs(w) > 1e-9:
            self.segments.append((t_seg0, t_seg1, w, label))

    def run(self):
        # 等 odom
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._have_odom:
                break
        if not self._have_odom and not self.dry:
            self.get_logger().error(f"收不到 odom（{self.get_namespace()}）；確認底盤 driver 在跑。中止。")
            return
        sched = [(lvl, sgn) for lvl in self.levels for sgn in (+1, -1)]
        total = len(sched) * (self.hold_s + self.pause_s)
        print(f"\n排程：{len(self.levels)} 種轉速 × 左右 = {len(sched)} 段，"
              f"每段按住 {self.hold_s}s + 停 {self.pause_s}s ≈ 共 {total:.0f}s")
        print(f"命令 topic = {self.cmd_topic}   {'(乾跑, 不動車)' if self.dry else ''}")
        print(f"轉速階梯 (rad/s): {self.levels}")
        if self.dry:
            for lvl, sgn in sched:
                print(f"  按住 {sgn*lvl:+.2f} {self.hold_s}s → 停 {self.pause_s}s")
            return
        print("\n⚠ 3 秒後開始原地轉 — 手拿急停！")
        for i in (3, 2, 1):
            print(f"  {i}…"); time.sleep(1.0)
        try:
            for lvl, sgn in sched:
                self._spin_for(self.hold_s, sgn * lvl, f"{sgn*lvl:+.2f}")
                self._spin_for(self.pause_s, 0.0, "pause")
        finally:
            for _ in range(10):           # 確實送 0 收車
                self._publish(0.0); time.sleep(0.02)
        print("完成，車已停。")

    # ── 存檔 + 擬合 ──
    def save_and_fit(self):
        if not self.log:
            return
        d = os.path.expanduser("~/rover_rl/logs/sysid")
        os.makedirs(d, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(d, f"sysid_omega_{stamp}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_rel", "cmd_w", "odom_w"])
            w.writerows([(f"{t:.3f}", f"{c:.4f}", f"{o:.4f}") for t, c, o in self.log])
        print(f"\n原始資料存於：{path}")
        if self.dry:
            return
        self._fit(path)

    def _fit(self, path):
        import numpy as np
        arr = np.array(self.log)  # [t, cmd, odom]
        t, cmd, odw = arr[:, 0], arr[:, 1], arr[:, 2]
        print("\n========= ω 通道辨識結果 =========")
        rows = []
        for t0, t1, lvl, lab in self.segments:
            m = (t >= t0) & (t < t1)
            if m.sum() < 5:
                continue
            tt, oo = t[m], odw[m]
            steady = oo[int(len(oo) * 0.6):]          # 後 40% 視為穩態
            y_ss = float(np.mean(steady)) * math.copysign(1, lvl)  # 取與命令同號的量值
            # 死時間：odom 從段首升到 20% 穩態的時間
            target = 0.2 * abs(np.mean(steady))
            dead = float("nan")
            for ti, oi in zip(tt, oo):
                if abs(oi) >= target and abs(np.mean(steady)) > 0.02:
                    dead = ti - t0; break
            # 時間常數：升到 63% 的時間 - 死時間
            tau = float("nan")
            tg = 0.632 * abs(np.mean(steady))
            for ti, oi in zip(tt, oo):
                if abs(oi) >= tg and abs(np.mean(steady)) > 0.02:
                    tau = (ti - t0) - (dead if dead == dead else 0); break
            rows.append((abs(lvl), y_ss, dead, tau))
        # 依命令大小聚合（左右兩段平均）
        from collections import defaultdict
        byl = defaultdict(list)
        for lvl, yss, dead, tau in rows:
            byl[round(lvl, 3)].append((yss, dead, tau))
        print("命令ω   實測ω(穩態)  聽幾成  死時間   時間常數")
        levels_sorted = sorted(byl)
        xs, ys = [], []
        for lvl in levels_sorted:
            vals = byl[lvl]
            yss = float(np.mean([abs(v[0]) for v in vals]))
            dead = float(np.nanmean([v[1] for v in vals]))
            tau = float(np.nanmean([v[2] for v in vals]))
            ratio = yss / lvl * 100 if lvl > 1e-6 else 0
            xs.append(lvl); ys.append(yss)
            print(f"  {lvl:.2f}      {yss:.3f}      {ratio:3.0f}%   "
                  f"{dead*1000 if dead==dead else float('nan'):5.0f}ms  "
                  f"{tau*1000 if tau==tau else float('nan'):5.0f}ms")
        # 死區 + 大訊號增益：對「有明顯回應」的點線性擬合 y = g*(x - deadband)
        xs, ys = np.array(xs), np.array(ys)
        good = ys > 0.02
        if good.sum() >= 2:
            g, c = np.polyfit(xs[good], ys[good], 1)
            db = -c / g if abs(g) > 1e-6 else float("nan")
            print(f"\n死區 deadband ≈ {db:.3f} rad/s   （轉速命令小於此值輪子幾乎不動）")
            print(f"大訊號增益 g ≈ {g:.2f}        （死區以上，命令每 +1 實際 +{g:.2f}）")
            dd = [r[2] for r in rows if r[2] == r[2]]
            tt2 = [r[3] for r in rows if r[3] == r[3]]
            if dd:
                print(f"死時間 ≈ {np.median(dd)*1000:.0f} ms   時間常數 τ ≈ "
                      f"{(np.median(tt2)*1000 if tt2 else float('nan')):.0f} ms")
            print("\n→ 這組 (死區, 增益, 死時間, τ) 才是建 Smith-predictor 補償該用的真模型。")
        else:
            print("\n回應太弱/雜，建議加大 hold 時間或確認車真的在轉。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd-topic", default="/input/nav_cmd_vel",
                    help="發角速度命令的 topic（底盤實際吃的那個）")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--hold", type=float, default=3.0, help="每段按住秒數")
    ap.add_argument("--pause", type=float, default=2.0, help="段間停止秒數")
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    help="測試的角速度階梯 (rad/s)")
    ap.add_argument("--dry", action="store_true", help="只印排程不發命令")
    args = ap.parse_args()

    rclpy.init()
    node = SysIdInject(args)
    try:
        node.run()
    except KeyboardInterrupt:
        for _ in range(10):
            node._publish(0.0); time.sleep(0.02)
        print("\n已中斷，車已送 0。")
    finally:
        node.save_and_fit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
