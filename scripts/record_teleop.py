#!/usr/bin/env python3
"""記錄「人工示範」的操控速度 — 你用遙控器親自開一次，我把你怎麼控速全錄下來.

用途：VO 的後退/避障行為很怪，改用模仿學習的想法——你手動示範「該怎麼後退」，
本腳本被動錄下你的搖桿指令 + 底盤實測 + 當下障礙距離，離線就能看出你的速度曲線
（尤其倒退：多快起步、退多久、退多遠、有沒有邊退邊轉），供之後調 VO 逃脫參數
或訓練一個小模仿策略的 ground truth。

純被動（不發任何 cmd_vel、不切 mode）：
  搖桿指令  u = /input/joy_cmd_vel（mux 手動輸入；joy 模式下 = 你真正下的速度意圖）
  底盤送出  = /output/cmd_vel（mux 最終送底盤，joy 模式下 ≈ 你的搖桿）
  實測     y = /odom（車實際跑出來的 v/ω）
  障礙脈絡  = /rover_rl_policy/status JSON（front_m/back_m/left_m/right_m，示範倒退時的距離）
以固定 50Hz 取樣「最新值」寫成等距時間序列 CSV（不受各 topic 頻率影響）。

用法（先確保能手動開車：搖桿在 joy 模式）：
  source /opt/ros/humble/setup.bash && source ~/rover_rl/setup_env.sh
  source ~/rover2_ws/install/setup.bash
  python3 scripts/record_teleop.py backup_demo_1    # 標籤；錄到 Ctrl+C
  python3 scripts/record_teleop.py backup_demo_1 --secs 60   # 錄 60 秒自動停

輸出：~/rover_rl/logs/teleop/teleop_<時間>_<標籤>/teleop_<時間>.csv
Ctrl+C 後自動印「倒退段落分析」摘要 + CSV 路徑。
"""
from __future__ import annotations
import argparse, csv, json, math, os, time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def _yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class TeleopRecorder(Node):
    def __init__(self, args):
        super().__init__("rover_rl_teleop_record")
        self.rate_hz = float(args.rate)
        # 最新值快取（取樣時抓）
        self._joy = (0.0, 0.0, 0.0)     # (v, w, t_mono)
        self._out = (0.0, 0.0, 0.0)
        self._odom = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (v, w, x, y, yaw, t_mono)
        self._front = self._back = self._left = self._right = None
        self._status_t = 0.0

        self.create_subscription(Twist, args.joy_topic,
                                 lambda m: self._cb_twist(m, "joy"), 10)
        self.create_subscription(Twist, args.out_topic,
                                 lambda m: self._cb_twist(m, "out"), 10)
        self.create_subscription(Odometry, args.odom_topic, self._cb_odom, 20)
        self.create_subscription(String, args.status_topic, self._cb_status, 10)

        self.rows = []
        self._t0 = time.monotonic()
        self._t0_wall = time.time()
        self._seen = {"joy": 0, "out": 0, "odom": 0, "status": 0}
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self._secs = args.secs

        self.get_logger().info(
            f"開始錄製人工示範：joy={args.joy_topic} out={args.out_topic} "
            f"odom={args.odom_topic} @ {self.rate_hz:.0f}Hz。開車示範，Ctrl+C 停。")

    def _cb_twist(self, msg: Twist, which: str) -> None:
        now = time.monotonic()
        val = (msg.linear.x, msg.angular.z, now)
        if which == "joy":
            self._joy = val
        else:
            self._out = val
        self._seen[which] += 1

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        tw = msg.twist.twist
        self._odom = (tw.linear.x, tw.angular.z, p.x, p.y,
                      _yaw(msg.pose.pose.orientation), time.monotonic())
        self._seen["odom"] += 1

    def _cb_status(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._front = _safe_float(d.get("front_m"))
        self._back = _safe_float(d.get("back_m"))
        self._left = _safe_float(d.get("left_m"))
        self._right = _safe_float(d.get("right_m"))
        self._status_t = time.monotonic()
        self._seen["status"] += 1

    def _fresh(self, cache, timeout=0.5):
        """cache=(...,t_mono)：>timeout 沒更新 → 視為無效（回 0，避免拿舊值當示範）。"""
        return (time.monotonic() - cache[-1]) <= timeout

    def _tick(self) -> None:
        now = time.monotonic()
        t = now - self._t0
        joy_ok = self._fresh(self._joy)
        out_ok = self._fresh(self._out)
        odom_ok = self._fresh(self._odom)
        stat_ok = (now - self._status_t) <= 1.0
        self.rows.append({
            "t": round(t, 3),
            "t_wall": round(self._t0_wall + t, 3),
            # 你的搖桿意圖（模仿學習的輸入/label 核心）
            "joy_v": round(self._joy[0], 4) if joy_ok else 0.0,
            "joy_w": round(self._joy[1], 4) if joy_ok else 0.0,
            # mux 送底盤
            "out_v": round(self._out[0], 4) if out_ok else 0.0,
            "out_w": round(self._out[1], 4) if out_ok else 0.0,
            # odom 實測
            "meas_v": round(self._odom[0], 4) if odom_ok else 0.0,
            "meas_w": round(self._odom[1], 4) if odom_ok else 0.0,
            "x": round(self._odom[2], 4),
            "y": round(self._odom[3], 4),
            "yaw": round(self._odom[4], 4),
            # 障礙距離脈絡（示範倒退時前後左右多空）
            "front_m": self._front if stat_ok else None,
            "back_m": self._back if stat_ok else None,
            "left_m": self._left if stat_ok else None,
            "right_m": self._right if stat_ok else None,
            "joy_ok": int(joy_ok),
        })
        if self._secs and t >= self._secs:
            raise KeyboardInterrupt

    # ── 存檔 + 倒退段落分析 ──
    def save_and_report(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._t0_wall))
        csv_path = os.path.join(out_dir, f"teleop_{stamp}.csv")
        if not self.rows:
            self.get_logger().warn("沒有錄到任何資料（topic 都沒發訊？）")
            return csv_path
        cols = list(self.rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.rows)
        self._report(csv_path)
        return csv_path

    def _report(self, csv_path: str) -> None:
        rows = self.rows
        dur = rows[-1]["t"] - rows[0]["t"] if len(rows) > 1 else 0.0
        # 用 joy 判意圖（你下的）；joy 沒發訊時退用 out（mux 送底盤值）
        def cmd_v(r):
            return r["joy_v"] if r["joy_ok"] else r["out_v"]
        def cmd_w(r):
            return r["joy_w"] if r["joy_ok"] else r["out_w"]
        # 動作分段：前進(F, cmd_v>+0.03) / 倒退(R, cmd_v<-0.03) / 停微動(0)，連續同類合併
        DEAD = 0.03
        def cls(r):
            v = cmd_v(r)
            return "F" if v > DEAD else ("R" if v < -DEAD else "0")
        segs, cur = [], None   # cur=[class, a, b]
        for i, r in enumerate(rows):
            s = cls(r)
            if cur is not None and cur[0] == s:
                cur[2] = i
            else:
                if cur is not None:
                    segs.append(tuple(cur))
                cur = [s, i, i]
        if cur is not None:
            segs.append(tuple(cur))

        print("\n" + "=" * 62)
        print("  人工示範錄製摘要（完整動作序列：倒退 + 前進繞出）")
        print("=" * 62)
        print(f"  總長 {dur:.1f}s，{len(rows)} 筆（{self.rate_hz:.0f}Hz）")
        print(f"  收訊：joy={self._seen['joy']} out={self._seen['out']} "
              f"odom={self._seen['odom']} status={self._seen['status']}")
        if self._seen["joy"] == 0:
            print("  ⚠ /input/joy_cmd_vel 完全沒收到——搖桿沒在 joy 模式，或 topic 名不同。")
            print("    分析改用 out_v（mux 送底盤值）。")
        name = {"F": "前進", "R": "倒退", "0": "停/微動"}
        k = 0
        for s, a, b in segs:
            sub = rows[a:b + 1]
            seg_dur = sub[-1]["t"] - sub[0]["t"]
            if seg_dur < 0.15:           # 太短（切換抖動）→ 跳過不列
                continue
            k += 1
            vs = [cmd_v(r) for r in sub]
            ws = [cmd_w(r) for r in sub]
            peak = max(vs, key=abs)      # 該段峰值線速（保號）
            peak_w = max(ws, key=abs) if ws else 0.0
            dist = math.hypot(sub[-1]["x"] - sub[0]["x"], sub[-1]["y"] - sub[0]["y"])
            t0, t1 = sub[0]["t"], sub[-1]["t"]
            head = (f"  ── #{k} [{name[s]}] {seg_dur:4.1f}s  "
                    f"峰v={peak:+.2f} 峰w={peak_w:+.2f} rad/s  位移{dist:.2f}m  "
                    f"(t={t0:.1f}→{t1:.1f})")
            print(head)
            if s in ("R", "F"):
                # 起步斜率：進段到觸及峰值用多久（看你踩得猛還是柔）
                i_pk = vs.index(peak)
                t2pk = sub[i_pk]["t"] - t0
                ramp = peak / t2pk if t2pk > 1e-2 else float("nan")
                turn = "邊走邊轉" if abs(peak_w) > 0.1 else "幾乎直行"
                side = "" if abs(peak_w) <= 0.1 else ("（往左）" if peak_w > 0 else "（往右）")
                print(f"       起步斜率 {ramp:+.2f} m/s²（{t2pk:.2f}s 到峰值）；{turn}{side}")
                fm, bm = sub[0]["front_m"], sub[0]["back_m"]
                if fm is not None or bm is not None:
                    fm_s = f"{fm:.2f}" if fm is not None else "?"
                    bm_s = f"{bm:.2f}" if bm is not None else "?"
                    print(f"       進段時 front_m={fm_s} back_m={bm_s}")
        print("-" * 62)
        print(f"  CSV：{csv_path}")
        print("=" * 62 + "\n")


def _safe_label(s: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "._-") else "_" for c in s.strip())
    return keep or "demo"


def main() -> None:
    ap = argparse.ArgumentParser(description="記錄人工示範操控速度（模仿學習用）")
    ap.add_argument("label", nargs="?", default="demo", help="本段示範標籤")
    ap.add_argument("--secs", type=float, default=0.0, help="錄幾秒自動停（0=到 Ctrl+C）")
    ap.add_argument("--rate", type=float, default=50.0, help="取樣頻率 Hz（預設 50）")
    ap.add_argument("--joy-topic", default="/input/joy_cmd_vel")
    ap.add_argument("--out-topic", default="/output/cmd_vel")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--status-topic", default="/rover_rl_policy/status")
    ap.add_argument("--outdir", default=os.path.expanduser("~/rover_rl/logs/teleop"))
    args = ap.parse_args()

    rclpy.init()
    node = TeleopRecorder(args)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(node._t0_wall))
    out_dir = os.path.join(args.outdir, f"teleop_{stamp}_{_safe_label(args.label)}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_report(out_dir)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
