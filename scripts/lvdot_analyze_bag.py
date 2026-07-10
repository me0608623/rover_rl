#!/usr/bin/env python3
"""離線分析 LV-DOT 特性 bag → 感測器誤差模型 JSON（給 sim DR 校準）。

比 live 量測準：用 bag 記錄時間戳（非 wall-clock），且可任意重跑、換參數重算。

量什麼（同 lvdot_characterize.py，但修正了上升緣配對）
------
1. 反應延遲：YOLO「人出現」rising edge → dynamic_bbox rising edge / tracked rising edge 的 Δt。
   ⭐ 改用「各訊號各自的 rising edge 再配對」：只配「人剛進來 且 動態框剛冒出」，
      避免場上已有持續動態物時把延遲誤算成 ~0。
2. 前方 recall proxy：YOLO rising edge 有幾次在 timeout 內配到 dynamic rising edge。
3. 追蹤連續性：每個持久 id 的存活時間（p50 遠短於單次走動 → ID 常跳）。
4. 位置/速度抖動：每 id 滑窗 std（→ DR 高斯噪聲 σ）。

用法
----
  source ~/rover_rl/install/setup.bash && source ~/rover_rl/setup_env.sh
  python3 scripts/lvdot_analyze_bag.py <bag_dir>
  # 例：python3 scripts/lvdot_analyze_bag.py ~/rover_rl/logs/lvdot_char/bag_20260707_191043
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

T_YOLO = "/yolo_detector/detected_bounding_boxes"
T_DYN = "/onboard_detector/dynamic_bboxes"
T_TRK = "/vo_interface/tracked_obstacles"

MIN_ABSENT = 2.0      # YOLO：人要離開多久才算一次乾淨「進來」事件
GAP_SIGNAL = 1.0      # dynamic/tracked：空多久才算一次乾淨「冒出」rising edge
DEBOUNCE = 0.3        # 短暫掉幀不算離開
TIMEOUT = 3.0         # YOLO 進來後多久內沒配到 → 算漏偵
JITTER_WIN = 12


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


def rising_edges(series, gap):
    """series: [(t, present_bool)] 依時間排序。回傳「離開≥gap 後再出現」的上升緣時刻。"""
    edges = []
    present = False
    last_true = -1e9
    absent_since = -1e9
    for t, p in series:
        if p:
            if not present:
                if (t - absent_since) >= gap:
                    edges.append(t)
                present = True
            last_true = t
        else:
            if present and (t - last_true) > DEBOUNCE:
                present = False
                absent_since = last_true
    return edges


def dyn_present(msg):
    return any(m.action == 0 and (m.scale.x > 0.01 or len(m.points) > 0)
               for m in msg.markers)


def main():
    if len(sys.argv) < 2:
        print("用法：python3 lvdot_analyze_bag.py <bag_dir>")
        sys.exit(1)
    bag = os.path.expanduser(sys.argv[1])

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    want = {T_YOLO, T_DYN, T_TRK}
    msgcls = {t: get_message(type_map[t]) for t in want if t in type_map}

    yolo_series, dyn_series, trk_series = [], [], []
    tracks = {}          # id -> {first,last,n,pos[],vel[]}
    t0 = t1 = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in want:
            continue
        t = t_ns * 1e-9
        t0 = t if t0 is None else t0
        t1 = t
        msg = deserialize_message(data, msgcls[topic])
        if topic == T_YOLO:
            yolo_series.append((t, len(msg.detections) > 0))
        elif topic == T_DYN:
            dyn_series.append((t, dyn_present(msg)))
        elif topic == T_TRK:
            trk_series.append((t, len(msg.obstacles) > 0))
            for ob in msg.obstacles:
                tr = tracks.setdefault(
                    int(ob.id), {"first": t, "last": t, "n": 0, "pos": [], "vel": []})
                tr["last"] = t
                tr["n"] += 1
                tr["pos"].append((ob.position.x, ob.position.y))
                tr["vel"].append((ob.velocity.x, ob.velocity.y))
                if len(tr["pos"]) > JITTER_WIN:
                    tr["pos"] = tr["pos"][-JITTER_WIN:]
                    tr["vel"] = tr["vel"][-JITTER_WIN:]

    dur = (t1 - t0) if t0 else 0.0

    # ── 延遲：YOLO rising edge 配 dynamic/tracked rising edge ──
    y_edges = rising_edges(yolo_series, MIN_ABSENT)
    d_edges = rising_edges(dyn_series, GAP_SIGNAL)
    k_edges = rising_edges(trk_series, GAP_SIGNAL)

    def pair(y_list, s_list):
        lat, misses = [], 0
        for y in y_list:
            cand = [s - y for s in s_list if 0.0 <= (s - y) <= TIMEOUT]
            if cand:
                lat.append(min(cand))
            else:
                misses += 1
        return lat, misses

    lat_dyn, miss_dyn = pair(y_edges, d_edges)
    lat_trk, miss_trk = pair(y_edges, k_edges)
    recall = round(1.0 - miss_dyn / len(y_edges), 3) if y_edges else None

    # ── 追蹤連續性 ──
    trk_hz = (len(trk_series) / dur) if dur else 20.0
    life = [t["last"] - t["first"] for t in tracks.values() if t["last"] > t["first"]]
    drops = []
    for t in tracks.values():
        span = t["last"] - t["first"]
        if span > 0.5:
            expected = span * trk_hz
            if expected > 1:
                drops.append(max(0.0, 1.0 - t["n"] / expected))

    # ── 抖動 ──
    pos_std, vel_std = [], []
    for t in tracks.values():
        if len(t["pos"]) >= 4:
            xs = [p[0] for p in t["pos"]]
            ys = [p[1] for p in t["pos"]]
            pos_std.append(math.hypot(statistics.pstdev(xs), statistics.pstdev(ys)))
            vx = [v[0] for v in t["vel"]]
            vy = [v[1] for v in t["vel"]]
            vel_std.append(math.hypot(statistics.pstdev(vx), statistics.pstdev(vy)))

    model = {
        "note": "LV-DOT 感測器誤差模型（bag 離線實測）→ 交 PC 端注入 sim 障礙物 obs 的 DR",
        "bag": bag,
        "duration_s": round(dur, 1),
        "n_yolo_person_events": len(y_edges),
        "latency_yolo_to_dynamic_s": _stat(lat_dyn),
        "latency_yolo_to_tracked_s": _stat(lat_trk),
        "front_recall_proxy": recall,
        "n_misses_dynamic": miss_dyn,
        "front_recall_note": "YOLO看到人的事件中真的變動態框的比例；側/後方YOLO照不到，需人工標註",
        "track_dropout_frac": _stat(drops),
        "track_id_lifetime_s": _stat(life),
        "track_id_churn_hint": "id 生命期 p50 若遠短於單次走動秒數 → ID 常跳（vo_interface 重追蹤沒接上）",
        "n_unique_ids": len(tracks),
        "pos_jitter_std_m": _stat(pos_std),
        "vel_jitter_std_mps": _stat(vel_std),
        "tracked_rate_hz": round(trk_hz, 1),
        "counts": {"yolo": len(yolo_series), "dynamic": len(dyn_series), "tracked": len(trk_series)},
    }

    out_dir = os.path.expanduser("~/rover_rl/logs/lvdot_char")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(bag.rstrip("/"))
    path = os.path.join(out_dir, f"model_{base}.json")
    with open(path, "w") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(json.dumps(model, ensure_ascii=False, indent=2))
    print(f"\n已存：{path}")


if __name__ == "__main__":
    main()
