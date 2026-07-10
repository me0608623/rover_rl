#!/usr/bin/env python3
"""LV-DOT 感測器特性量測 harness（給 sim DR 校準用）。

用途：把 LV-DOT 的「真實髒法」量成數字，回填 PC 端訓練的 domain randomization。
純訂閱純量測，不影響推論、不發任何控制。與 adapter（格式對齊）無關——本工具
只管數值分佈對齊（延遲 / 抖動 / 漏偵 / 追蹤連續性）。

量什麼
------
1. 反應延遲：YOLO 看到人（上升緣） → dynamic_bbox 冒出 / tracked_obstacle 冒出 的 Δt。
   這是 L3 投票+追蹤延遲（避障最在意、也是唯一 sim 要模擬的那一段）。
2. 前方 recall proxy：YOLO 看到人的 N 次事件裡，有幾次在 timeout 內真的變動態框。
3. 追蹤連續性：每個持久 id 的存活時間、掉幀率（該在的幀有沒有到）。
4. 位置/速度抖動：每個 id 滑窗 std（人穩定走動時的高頻抖動 ≈ 感測雜訊）。

前置
----
  source ~/rover_rl/install/setup.bash    # 需要 vo_interface 訊息
  source ~/rover_rl/setup_env.sh          # zenoh + DOMAIN 55
  # 車端要先跑：deploy_rl + lv-dot（含 YOLO + vo_interface）

跑法
----
  python3 scripts/lvdot_characterize.py            # 開始量測，Ctrl+C 出報告 + JSON
  # 操作：讓一個人「走進 FOV → 停一下 → 完全離開 FOV ≥2s → 再走進」反覆 10+ 次，
  #       每次乾淨的進出 = 一筆延遲樣本。側/後方另做（見報告註記）。

輸出
----
  logs/lvdot_char/lvdot_char_<時間>.json   # 誤差模型，交給 PC 端當 DR 依據
  同時建議另開一個 terminal 存 raw bag（開頭會印指令）。
"""
from __future__ import annotations

import json
import math
import os
import signal
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray

try:
    from vo_interface.msg import TrackedObstacleArray
    HAVE_VO = True
except Exception:  # noqa: BLE001
    HAVE_VO = False


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _stat(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 4),
        "std": round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0,
        "p50": round(_pct(xs, 50), 4),
        "p95": round(_pct(xs, 95), 4),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
    }


class Characterizer(Node):
    def __init__(self):
        super().__init__("lvdot_characterizer")

        # topics（可用參數覆寫）
        self.t_yolo = self.declare_parameter("topic_yolo", "/yolo_detector/detected_bounding_boxes").value
        self.t_dyn = self.declare_parameter("topic_dynamic", "/onboard_detector/dynamic_bboxes").value
        self.t_trk = self.declare_parameter("topic_tracked", "/vo_interface/tracked_obstacles").value
        # 延遲事件參數
        self.edge_timeout = float(self.declare_parameter("edge_timeout_s", 3.0).value)   # 逾時視為漏偵
        self.min_absent = float(self.declare_parameter("min_absent_s", 2.0).value)       # 重新武裝前人要離開多久
        self.jitter_win = int(self.declare_parameter("jitter_window", 12).value)         # 抖動滑窗樣本數

        self.create_subscription(Detection2DArray, self.t_yolo, self._yolo_cb, 10)
        self.create_subscription(MarkerArray, self.t_dyn, self._dyn_cb, 10)
        if HAVE_VO:
            self.create_subscription(TrackedObstacleArray, self.t_trk, self._trk_cb, 10)

        # ── 延遲事件狀態機 ──
        self._person = False           # YOLO 目前是否看到人
        self._last_person_t = 0.0      # 最近一次看到人的時間
        self._last_absent_t = time.monotonic()  # 最近一次「確定沒人」的時間
        self._armed = False            # 是否已武裝等待 dynamic/tracked 冒出
        self._edge_t0 = 0.0
        self._got_dyn = False
        self._got_trk = False
        self.lat_dyn = []              # YOLO→dynamic Δt 樣本
        self.lat_trk = []              # YOLO→tracked Δt 樣本
        self.edges = 0                 # 總事件數
        self.misses = 0                # 逾時沒變動態框的事件數（前方 recall 失敗）

        # ── 訊號存在旗標 + 頻率 ──
        self._dyn_present = False
        self._trk_present = False
        self._dyn_stamps = []
        self._trk_stamps = []

        # ── 追蹤連續性 / 抖動（by id）──
        self.tracks = {}   # id -> {"first":t,"last":t,"n":int,"pos":[(x,y)...],"vel":[(vx,vy)...]}

        self._t_start = time.monotonic()
        self.create_timer(2.0, self._report_live)

        print("=" * 68)
        print("LV-DOT 特性量測中… 讓人反覆走進/離開 FOV（每次離開≥%.0fs）" % self.min_absent)
        print("Ctrl+C 出報告。建議另開 terminal 存 raw bag：")
        print("  ros2 bag record %s %s %s /velodyne_points "
              "/camera/camera/color/image_raw /odom" % (self.t_yolo, self.t_dyn, self.t_trk))
        if not HAVE_VO:
            print("⚠ 找不到 vo_interface.msg → tracked_obstacles 那半量不到；"
                  "先 source ~/rover_rl/install/setup.bash")
        print("=" * 68)

    # ---------- callbacks ----------
    def _yolo_cb(self, msg: Detection2DArray):
        now = time.monotonic()
        present = len(msg.detections) > 0
        if present:
            self._last_person_t = now
            # 上升緣：之前確定沒人夠久 → 武裝一次延遲事件
            if not self._person and (now - self._last_absent_t) >= self.min_absent and not self._armed:
                self._armed = True
                self._edge_t0 = now
                self._got_dyn = False
                self._got_trk = False
                self.edges += 1
            self._person = True
        else:
            # 連續沒人才更新 absent 時戳
            if self._person and (now - self._last_person_t) > 0.3:
                self._person = False
            if not self._person:
                self._last_absent_t = now
        self._check_edge_timeout(now)

    def _dyn_cb(self, msg: MarkerArray):
        now = time.monotonic()
        present = any(
            m.action == 0 and (m.scale.x > 0.01 or len(m.points) > 0)
            for m in msg.markers
        )
        self._dyn_present = present
        if present:
            self._dyn_stamps.append(now)
            if self._armed and not self._got_dyn:
                self.lat_dyn.append(now - self._edge_t0)
                self._got_dyn = True
                self._maybe_disarm()

    def _trk_cb(self, msg):
        now = time.monotonic()
        present = len(msg.obstacles) > 0
        self._trk_present = present
        if present:
            self._trk_stamps.append(now)
            if self._armed and not self._got_trk:
                self.lat_trk.append(now - self._edge_t0)
                self._got_trk = True
                self._maybe_disarm()
        for ob in msg.obstacles:
            tr = self.tracks.setdefault(
                int(ob.id), {"first": now, "last": now, "n": 0, "pos": [], "vel": []})
            tr["last"] = now
            tr["n"] += 1
            tr["pos"].append((ob.position.x, ob.position.y))
            tr["vel"].append((ob.velocity.x, ob.velocity.y))
            if len(tr["pos"]) > self.jitter_win:
                tr["pos"] = tr["pos"][-self.jitter_win:]
                tr["vel"] = tr["vel"][-self.jitter_win:]

    # ---------- edge bookkeeping ----------
    def _maybe_disarm(self):
        need_trk = HAVE_VO
        if self._got_dyn and (self._got_trk or not need_trk):
            self._armed = False

    def _check_edge_timeout(self, now):
        if self._armed and (now - self._edge_t0) > self.edge_timeout:
            if not self._got_dyn:
                self.misses += 1   # YOLO 看到人卻沒變動態框 = 前方 recall 漏
            self._armed = False

    # ---------- reports ----------
    def _rate(self, stamps, win=5.0):
        now = time.monotonic()
        recent = [t for t in stamps if now - t <= win]
        return round(len(recent) / win, 1) if recent else 0.0

    def _report_live(self):
        el = time.monotonic() - self._t_start
        d = _stat(self.lat_dyn)
        line = (f"[{el:5.0f}s] 事件{self.edges} 漏{self.misses} | "
                f"YOLO→動態框 中位={d.get('p50','—')}s (n={d['n']}) | "
                f"dyn_hz={self._rate(self._dyn_stamps)} trk_hz={self._rate(self._trk_stamps)} | "
                f"活躍id={sum(1 for t in self.tracks.values() if time.monotonic()-t['last']<1.0)}")
        print(line, flush=True)

    def build_model(self):
        # 追蹤連續性
        life = [t["last"] - t["first"] for t in self.tracks.values() if t["last"] > t["first"]]
        drops = []
        trk_hz = self._rate(self._trk_stamps) or 20.0
        for t in self.tracks.values():
            span = t["last"] - t["first"]
            if span > 0.5:
                expected = span * trk_hz
                if expected > 1:
                    drops.append(max(0.0, 1.0 - t["n"] / expected))
        # 抖動：所有 id 的滑窗 std 聚合
        pos_std, vel_std = [], []
        for t in self.tracks.values():
            if len(t["pos"]) >= 4:
                xs = [p[0] for p in t["pos"]]
                ys = [p[1] for p in t["pos"]]
                pos_std.append(math.hypot(statistics.pstdev(xs), statistics.pstdev(ys)))
                vx = [v[0] for v in t["vel"]]
                vy = [v[1] for v in t["vel"]]
                vel_std.append(math.hypot(statistics.pstdev(vx), statistics.pstdev(vy)))
        recall = None
        if self.edges > 0:
            recall = round(1.0 - self.misses / self.edges, 3)
        return {
            "note": "LV-DOT 感測器誤差模型（實測）→ 交 PC 端注入 sim 障礙物 obs 的 DR",
            "duration_s": round(time.monotonic() - self._t_start, 1),
            "latency_yolo_to_dynamic_s": _stat(self.lat_dyn),
            "latency_yolo_to_tracked_s": _stat(self.lat_trk),
            "front_recall_proxy": recall,
            "front_recall_note": "YOLO看到人的事件中真的變動態框的比例；側/後方YOLO照不到，此工具量不到，需人工標註",
            "n_edges": self.edges,
            "n_misses": self.misses,
            "track_dropout_frac": _stat(drops),
            "track_id_lifetime_s": _stat(life),
            "track_id_churn_hint": "id 生命期若遠短於單次走動時間 → ID 常跳（vo_interface 重追蹤沒接上）",
            "n_unique_ids": len(self.tracks),
            "pos_jitter_std_m": _stat(pos_std),
            "vel_jitter_std_mps": _stat(vel_std),
            "tracked_rate_hz": trk_hz,
            "have_vo_interface": HAVE_VO,
        }


def main():
    rclpy.init()
    node = Characterizer()

    def dump_and_exit(*_):
        model = node.build_model()
        os.makedirs(os.path.expanduser("~/rover_rl/logs/lvdot_char"), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.expanduser(f"~/rover_rl/logs/lvdot_char/lvdot_char_{stamp}.json")
        with open(path, "w") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 68)
        print("LV-DOT 誤差模型：")
        print(json.dumps(model, ensure_ascii=False, indent=2))
        print("=" * 68)
        print(f"已存：{path}")
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, dump_and_exit)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
