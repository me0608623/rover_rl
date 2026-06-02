"""分析 diag_logger 產生的 CSV — 量化「角速度晃動」與「是否朝 goal 前進」.

用法：
  ros2 run rover_rl_inference analyze_diag <csv路徑>
  # 或不給路徑 → 自動取 ~/rover_rl/logs/ 最新一個
  ros2 run rover_rl_inference analyze_diag

輸出：
  - 終端統計摘要（晃動程度、朝向誤差、是否接近 goal）
  - 若裝了 matplotlib：在 CSV 同目錄存一張 <csv>.png（cmd_w / heading_err / dist 時間圖）
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys


def _load(path: str):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v == "":
            out.append(None)
        else:
            try:
                out.append(float(v))
            except ValueError:
                out.append(None)
    return out


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"n": n, "mean": mean, "std": math.sqrt(var),
            "min": min(xs), "max": max(xs)}


def _print_params(csv_path: str) -> None:
    """印出 sidecar _params.json 內的 policy 設定（若有）."""
    side = csv_path.rsplit(".", 1)[0] + "_params.json"
    if not os.path.isfile(side):
        return
    try:
        with open(side) as f:
            meta = json.load(f)
        p = meta.get("params", {})
    except Exception:
        return
    keys = ["speed_rate", "cmd_alpha_linear", "cmd_alpha_angular",
            "cmd_max_accel_linear", "cmd_max_accel_angular",
            "act_max_linear_velocity", "act_max_angular_velocity",
            "control_dt", "goal_tolerance_m", "path_lookahead_m",
            "require_ndt", "deterministic", "model_path"]
    shown = [(k, p[k]) for k in keys if k in p]
    print("-" * 60)
    print(f"【當時 policy 參數】(來自 {os.path.basename(side)})")
    for k, v in shown:
        print(f"  {k:24s}= {v}")
    extra = [k for k in p if k not in keys and k != "use_sim_time"]
    if extra:
        print(f"  (另有 {len(extra)} 項：{', '.join(extra[:8])}{' …' if len(extra) > 8 else ''})")


def analyze(path: str) -> None:
    rows = _load(path)
    if not rows:
        print(f"[!] {path} 沒有資料列")
        return

    t = _col(rows, "t_rel")
    cmd_w = _col(rows, "cmd_w")
    cmd_v = _col(rows, "cmd_v")
    odom_w = _col(rows, "odom_w")
    heading = _col(rows, "heading_err_deg")
    dist = _col(rows, "dist_to_goal")
    pol_ang = _col(rows, "policy_goal_ang_deg")

    # Δω（相鄰列）
    dws = [b - a for a, b in zip(cmd_w, cmd_w[1:])
           if a is not None and b is not None]
    dw_st = _stats(dws)
    cw_st = _stats([abs(x) for x in cmd_w if x is not None])
    hd_st = _stats([abs(x) for x in heading if x is not None])
    di_st = _stats(dist)
    pol_st = _stats([abs(x) for x in pol_ang if x is not None])

    # heading 與 policy_goal_ang 一致性（檢查定位/TF）
    diff = [abs(h - p) for h, p in zip(heading, pol_ang)
            if h is not None and p is not None]
    diff_st = _stats(diff)

    # 朝 goal 前進？distance 趨勢（頭 20% vs 尾 20%）
    dvals = [d for d in dist if d is not None]
    trend = None
    if len(dvals) >= 10:
        k = max(1, len(dvals) // 5)
        trend = sum(dvals[-k:]) / k - sum(dvals[:k]) / k

    dur = (t[-1] - t[0]) if t and t[0] is not None and t[-1] is not None else 0.0

    print("=" * 60)
    print(f"檔案: {path}")
    print(f"列數: {len(rows)}   時長: {dur:.1f}s")
    _print_params(path)
    print("-" * 60)
    print("【角速度晃動】(變化越大越晃)")
    if dw_st:
        print(f"  Δω(相鄰) std = {dw_st['std']:.3f} rad/s   "
              f"range=[{dw_st['min']:+.2f},{dw_st['max']:+.2f}]")
        print(f"    判讀: <0.05 平順 / 0.05~0.15 中等 / >0.15 明顯晃動")
    if cw_st:
        print(f"  |ω| 平均 = {cw_st['mean']:.3f} rad/s  max={cw_st['max']:.3f}")
    print("-" * 60)
    print("【是否朝 goal 前進】")
    if hd_st:
        print(f"  |朝向誤差| 平均 = {hd_st['mean']:.1f}°  max={hd_st['max']:.1f}°")
        print(f"    判讀: <20° 大致朝向 / 20~60° 偏 / >60° 幾乎沒朝 goal")
    if di_st:
        print(f"  與 goal 距離: 起 {dvals[0]:.2f}m → 終 {dvals[-1]:.2f}m  "
              f"min={di_st['min']:.2f}m")
    if trend is not None:
        verdict = "✓ 有靠近" if trend < -0.3 else (
            "✗ 反而變遠" if trend > 0.3 else "→ 幾乎沒進展")
        print(f"  距離趨勢(尾-頭) = {trend:+.2f}m  {verdict}")
    print("-" * 60)
    print("【定位/TF 一致性】(policy 看到的方向 vs NDT 真實方向)")
    if pol_st:
        print(f"  policy_goal 角度 |平均| = {pol_st['mean']:.1f}°")
    if diff_st:
        consistent = "✓ 一致" if diff_st['mean'] < 15 else "✗ 不一致(疑 TF/座標問題)"
        print(f"  |heading - policy_goal_ang| 平均 = {diff_st['mean']:.1f}°  "
              f"{consistent}")
    print("=" * 60)

    _maybe_plot(path, t, cmd_w, cmd_v, odom_w, heading, dist, pol_ang)


def _maybe_plot(path, t, cmd_w, cmd_v, odom_w, heading, dist, pol_ang):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[i] 未安裝 matplotlib，略過繪圖（pip install matplotlib 可開啟）")
        return

    def clean(xs):
        return [(ti, xi) for ti, xi in zip(t, xs)
                if ti is not None and xi is not None]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    cw = clean(cmd_w); cv = clean(cmd_v); ow = clean(odom_w)
    if cw:
        axes[0].plot(*zip(*cw), label="cmd_w (rad/s)", color="tab:red")
    if ow:
        axes[0].plot(*zip(*ow), label="odom_w (measured)", color="tab:orange", alpha=0.5)
    if cv:
        axes[0].plot(*zip(*cv), label="cmd_v (m/s)", color="tab:blue", alpha=0.7)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("cmd"); axes[0].legend(loc="upper right")
    axes[0].set_title("Angular wobble (cmd_w bouncing = wobble)")

    hd = clean(heading); pa = clean(pol_ang)
    if hd:
        axes[1].plot(*zip(*hd), label="heading_err (NDT truth)", color="tab:green")
    if pa:
        axes[1].plot(*zip(*pa), label="policy_goal_ang (policy view)",
                     color="tab:purple", alpha=0.6)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("angle (deg)"); axes[1].legend(loc="upper right")
    axes[1].set_title("Heading error (near 0 = toward goal; gap = TF issue)")

    di = clean(dist)
    if di:
        axes[2].plot(*zip(*di), label="dist_to_goal (m)", color="tab:brown")
    axes[2].set_ylabel("dist (m)"); axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Distance to goal (should keep decreasing)")

    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=110)
    print(f"[✓] 圖已存: {out}")


def main(args=None):
    argv = sys.argv[1:]
    if argv:
        path = argv[0]
    else:
        cands = sorted(glob.glob(os.path.expanduser("~/rover_rl/logs/diag_*.csv")))
        if not cands:
            print("用法: analyze_diag <csv>  (或先跑一次部署產生 ~/rover_rl/logs/diag_*.csv)")
            return
        path = cands[-1]
        print(f"[i] 未指定路徑，分析最新: {path}")
    analyze(path)


if __name__ == "__main__":
    main()
