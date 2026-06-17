#!/usr/bin/env python3
"""ω 通道「邊開邊錄」被動辨識 — 你用遙控器開車，我錄並擬合.

原地轉受 in-place 摩擦/黏滑污染量不準（見 sysid_inject 結果）；車「邊前進邊轉」摩擦小、
才是 policy 真正的工作條件。此程式純被動錄製（不發任何命令）：
  輸入 u = 底盤實際吃到的角速度命令（自動從常見 topic 偵測，不用你指定）
  輸出 y = /odom 量到的實際角速度
錄完自動算 死區 / 增益 / 死時間 / 時間常數（與 sysid_omega.py 同方法）。

用法（先把車的遙控/joy 開好，能手動開車）：
  source /opt/ros/humble/setup.bash && source ~/rover_rl/setup_env.sh
  source ~/rover2_ws/install/setup.bash
  python3 scripts/sysid_record.py            # 錄到 Ctrl+C
  python3 scripts/sysid_record.py --secs 90  # 錄 90 秒自動停
開始後就拿遙控器多轉幾種彎（左右、大小彎、邊走邊轉）約 1 分鐘。
"""
from __future__ import annotations
import argparse, csv, math, os, time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry

# 候選「底盤命令」topic（依優先序；自動用正在發訊的那個）
CMD_CANDIDATES = ["/output/cmd_vel", "/cmd_vel", "/input/nav_cmd_vel",
                  "/rover_rl/cmd_vel_desired", "/joy_cmd_vel"]


class Recorder(Node):
    def __init__(self, args):
        super().__init__("rover_rl_sysid_record")
        self.rate_hz = 20.0
        self._cmd = {}          # topic -> (w, t_mono)
        self._odom_w = 0.0
        self._have_odom = False
        self.create_subscription(Odometry, args.odom_topic, self._cb_odom, 10)
        for tp in CMD_CANDIDATES:
            self.create_subscription(Twist, tp, lambda m, t=tp: self._cb_cmd(t, m.angular.z), 10)
        self.log = []
        self.used = {}
        self._t0 = time.monotonic()

    def _cb_odom(self, msg):
        self._odom_w = msg.twist.twist.angular.z
        self._have_odom = True

    def _cb_cmd(self, topic, wz):
        self._cmd[topic] = (float(wz), time.monotonic())

    def _latest_cmd(self):
        """取最近 0.5s 內、優先序最高且有發訊的命令角速度."""
        now = time.monotonic()
        for tp in CMD_CANDIDATES:
            if tp in self._cmd and now - self._cmd[tp][1] < 0.5:
                self.used[tp] = self.used.get(tp, 0) + 1
                return self._cmd[tp][0]
        return 0.0

    def run(self, secs):
        print(f"開始錄製 — 拿遙控器把車開一開、多轉幾種彎"
              + (f"（{secs:.0f} 秒後自動停）" if secs else "（按 Ctrl+C 結束）"))
        end = (time.monotonic() + secs) if secs else None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            w = self._latest_cmd()
            self.log.append((time.monotonic() - self._t0, w, self._odom_w))
            time.sleep(1.0 / self.rate_hz)
            if end and time.monotonic() >= end:
                break

    def save_and_fit(self):
        if not self.log:
            print("沒錄到資料。"); return
        d = os.path.expanduser("~/rover_rl/logs/sysid"); os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"sysid_record_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_rel", "cmd_w", "odom_w", "has_goal"])
            w.writerows([(f"{t:.3f}", f"{c:.4f}", f"{o:.4f}", 1) for t, c, o in self.log])
        print(f"\n原始資料存於：{path}")
        print(f"命令來源 topic 使用次數：{self.used or '（沒偵測到任何命令 topic！確認遙控有在發）'}")
        self._fit(path)

    def _fit(self, path):
        import numpy as np
        arr = np.array(self.log)
        t, c, o = arr[:, 0], arr[:, 1], arr[:, 2]
        dt = 0.05
        g = np.arange(t[0], t[-1], dt)
        u = np.interp(g, t, c); y = np.interp(g, t, o)
        au = np.abs(u)
        print(f"\n========= ω 通道辨識（邊開邊錄）=========")
        print(f"樣本={len(g)} 時長={t[-1]-t[0]:.0f}s  |命令|平均={au.mean():.3f}  "
              f"命令>0.1 佔比={100*np.mean(au>0.1):.0f}%  飽和(>0.55)佔比={100*np.mean(au>0.55):.0f}%")
        if np.mean(au > 0.1) < 0.2:
            print("⚠ 轉彎命令太少 → 多轉幾種彎再錄一次（這樣才有激勵）。")
        # 死時間：互相關
        u0, y0 = u - u.mean(), y - y.mean()
        best = (0, -2)
        for dd in range(0, int(0.6 / dt) + 1):
            if dd >= len(u0): break
            a, b = u0[:len(u0)-dd], y0[dd:]
            den = math.sqrt(float(a@a)*float(b@b))
            r = float(a@b)/den if den > 1e-9 else 0
            if r > best[1]: best = (dd, r)
        d, r = best
        print(f"死時間 ≈ {d*dt*1000:.0f} ms  (峰值相關 r={r:.2f})")
        # 一階擬合
        k0 = max(1, d)
        Y = y[k0:]; X = np.column_stack([y[k0-1:len(y)-1], u[k0-d:len(u)-d]])
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        K = b/(1-a) if abs(1-a) > 1e-6 else float("nan")
        tau = -dt/math.log(a) if 0 < a < 1 else float("nan")
        print(f"靜態增益 K ≈ {K:.2f}（聽 {K*100:.0f}% 話）  時間常數 τ ≈ {tau*1000:.0f} ms")
        # 死區 + 大訊號增益（分箱）
        ud = au[:len(au)-d] if d else au
        yd = np.abs(y[d:]) if d else np.abs(y)
        edges = np.linspace(0, ud.max(), 11)
        pts = []
        print("命令|ω|   實測|ω|   聽幾成")
        for i in range(10):
            m = (ud >= edges[i]) & (ud < edges[i+1])
            if m.sum() >= 5:
                uc = 0.5*(edges[i]+edges[i+1]); ym = float(yd[m].mean())
                pts.append((uc, ym))
                print(f"  {uc:.2f}      {ym:.3f}     {ym/uc*100 if uc>1e-6 else 0:3.0f}%")
        good = [(x, yy) for x, yy in pts if yy > 0.02]
        if len(good) >= 2:
            xs = np.array([p[0] for p in good]); ys = np.array([p[1] for p in good])
            gg, cc = np.polyfit(xs, ys, 1)
            db = -cc/gg if abs(gg) > 1e-6 else float("nan")
            print(f"\n死區 deadband ≈ {db:.3f} rad/s   大訊號增益 g ≈ {gg:.2f}")
            print(f"→ Smith-predictor 該用：死區 {db:.2f}、增益 {gg:.2f}、死時間 {d*dt*1000:.0f}ms、τ {tau*1000:.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--secs", type=float, default=0.0, help="錄幾秒(0=到Ctrl+C)")
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args)
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node._have_odom: break
    if not node._have_odom:
        print("⚠ 收不到 /odom，確認底盤 driver 在跑。")
    try:
        node.run(args.secs)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，結束錄製。")
    finally:
        node.save_and_fit()
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
