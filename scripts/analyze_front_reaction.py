#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_front_reaction.py — 前方障礙「反應強度 vs 距離」A/B 對比（sim2real 有無雜訊）

論文主張：訓練不加 LiDAR 雜訊(clean) → 實車部署對真實 sweep 抖動「高增益」放大，
表現為近距離時角速度 ω 抖動/過衝/飽和明顯大於 noise+DR 模型。此腳本把 diag_logger
每趟自動錄的 CSV，依「模型 / experiment_tag」分組，算出：

  1. 反應曲線（依前方距離 front_m 分桶）：
       mean|ω| / ω 標準差(=抖動,高增益頭號指標) / Δω RMS(急拉) / mean v / ω 飽和率
  2. 近距離帶(預設 front_m ≤ 2m)的高增益純量摘要（每組一列，論文表格用）
  3. 推 wandb 對比線圖（wandb.plot.line_series，clean vs noise 疊在同一面板）
     + 本地 PNG / curves.csv / summary.{csv,md} 供論文直接引用

分組鍵來自各 run 的 _params.json（"model" + "experiment_tag"），與 aggregate_diag.py
同源；front_m 來自 sweep(sensor 相對量)，不受 NDT/odom 漂移影響。

用法：
  python3 scripts/analyze_front_reaction.py [ROOT] [--group-by tag|model]
       [--since 20260720] [--near 2.0] [--wandb] [--wandb-mode online|offline]
       [--project rover_rl_sim2real] [--no-plot]

  ROOT       diag 根目錄（預設 ~/rover_rl/logs/diag）
  --group-by tag=用 experiment_tag 分組(預設,沒 tag 的 run 退回用 model)；model=一律用模型名
  --since    只看資料夾名 >= 此日期字串(例 20260720)
  --near     近距離帶上界(m)，純量高增益摘要只統計 front_m ≤ 此值，預設 2.0
  --wandb    推 wandb（預設不推；只想先看本地圖表可不加）
  --wandb-mode  online(即時,post-hoc 獨立程式不會卡 ROS)/offline(存本機回頭 sync)
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

# front_m 分桶邊界(m)：近密遠疏，涵蓋盲區上緣~開闊
BUCKET_EDGES = [0.4, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
BUCKET_CENTERS = [(BUCKET_EDGES[i] + BUCKET_EDGES[i + 1]) / 2 for i in range(len(BUCKET_EDGES) - 1)]

DT = 0.05  # 20Hz


def _read_csv(path):
    """robust CSV → dict of np arrays（float 欄）+ dict of str 欄。清 NUL byte。"""
    with open(path, "r", errors="replace", newline="") as f:
        text = f.read().replace("\x00", "")
    rdr = csv.reader(text.splitlines())
    try:
        header = next(rdr)
    except StopIteration:
        return {}, {}, 0
    rows = [r for r in rdr if r]
    n = len(rows)
    fcols, scols = {}, {}
    for j, name in enumerate(header):
        vals_f = np.full(n, np.nan, dtype=float)
        vals_s = [""] * n
        for i, r in enumerate(rows):
            if j < len(r):
                s = r[j]
                vals_s[i] = s
                try:
                    vals_f[i] = float(s)
                except (ValueError, TypeError):
                    pass
        fcols[name] = vals_f
        scols[name] = vals_s
    return fcols, scols, n


def _load_params(run_dir):
    """讀 _params.json → (model, experiment_tag)。"""
    for p in glob.glob(os.path.join(run_dir, "*_params.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        model = d.get("model") or d.get("model_path") or ""
        model = os.path.basename(str(model)).replace(".ts", "")
        tag = d.get("experiment_tag") or ""
        return model, tag
    return "", ""


def _bucket_idx(front_m):
    """回傳 front_m 落在哪個桶(0..len-1)，超界回 -1。"""
    if not math.isfinite(front_m):
        return -1
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= front_m < BUCKET_EDGES[i + 1]:
            return i
    return -1


def collect(root, group_by, since):
    """走訪所有 run，回傳 groups[key] = list of per-row records (dict of arrays)。"""
    groups = defaultdict(list)
    run_dirs = sorted(glob.glob(os.path.join(root, "diag_*")))
    n_runs = 0
    for rd in run_dirs:
        base = os.path.basename(rd)
        if since and base[5:5 + len(since)] < since:
            continue
        csvs = glob.glob(os.path.join(rd, "diag_*.csv"))
        if not csvs:
            continue
        fcols, scols, n = _read_csv(csvs[0])
        if n == 0 or "front_m" not in fcols or "rl_w" not in fcols:
            continue  # 07-06 前舊檔無 front_m，跳過
        model, tag = _load_params(rd)
        if group_by == "tag":
            key = tag if tag else (model or "untagged")
        else:
            key = model or "unknown"
        # 只取有 goal 的列(policy 真正在跑)
        has_goal = fcols.get("has_goal", np.zeros(n))
        mask = has_goal > 0.5
        if mask.sum() < 5:
            continue
        rec = {
            "run": base,
            "front_m": fcols["front_m"][mask],
            "rl_w": fcols["rl_w"][mask],
            "rl_v": fcols.get("rl_v", np.full(n, np.nan))[mask],
            "sent_w": fcols.get("sent_w", np.full(n, np.nan))[mask],
            "act_w": fcols.get("act_w", np.full(n, np.nan))[mask],
            "w_over": fcols.get("w_over", np.zeros(n))[mask],
            "v_over": fcols.get("v_over", np.zeros(n))[mask],
            "sweep_min_m": fcols.get("sweep_min_m", np.full(n, np.nan))[mask],
        }
        groups[key].append(rec)
        n_runs += 1
    return groups, n_runs


def curve_for_group(recs):
    """把一組(多 run)的列，依 front_m 分桶算反應曲線。回傳 dict[metric] = array(len buckets)。"""
    nb = len(BUCKET_CENTERS)
    acc = {k: [[] for _ in range(nb)] for k in
           ("abs_w", "rl_w", "rl_v", "w_over", "dw")}
    for rec in recs:
        fm = rec["front_m"]
        w = rec["rl_w"]
        # 相鄰列 Δω(同一 run 內)：急拉/反轉指標
        dw = np.abs(np.diff(w, prepend=w[:1]))
        for i in range(len(fm)):
            b = _bucket_idx(fm[i])
            if b < 0:
                continue
            if math.isfinite(w[i]):
                acc["abs_w"][b].append(abs(w[i]))
                acc["rl_w"][b].append(w[i])
                acc["dw"][b].append(dw[i])
            if math.isfinite(rec["rl_v"][i]):
                acc["rl_v"][b].append(rec["rl_v"][i])
            acc["w_over"][b].append(1.0 if rec["w_over"][i] > 0.5 else 0.0)
    out = {
        "n": np.array([len(acc["abs_w"][b]) for b in range(nb)], float),
        "mean_abs_w": np.array([np.mean(acc["abs_w"][b]) if acc["abs_w"][b] else np.nan for b in range(nb)]),
        "std_w": np.array([np.std(acc["rl_w"][b]) if len(acc["rl_w"][b]) > 1 else np.nan for b in range(nb)]),
        "dw_rms": np.array([np.sqrt(np.mean(np.square(acc["dw"][b]))) if acc["dw"][b] else np.nan for b in range(nb)]),
        "mean_v": np.array([np.mean(acc["rl_v"][b]) if acc["rl_v"][b] else np.nan for b in range(nb)]),
        "w_over_rate": np.array([np.mean(acc["w_over"][b]) if acc["w_over"][b] else np.nan for b in range(nb)]),
        "max_abs_w": np.array([np.max(acc["abs_w"][b]) if acc["abs_w"][b] else np.nan for b in range(nb)]),
    }
    return out


def scalar_summary(recs, near):
    """近距離帶(front_m ≤ near)的高增益純量摘要。"""
    fm = np.concatenate([r["front_m"] for r in recs])
    w = np.concatenate([r["rl_w"] for r in recs])
    v = np.concatenate([r["rl_v"] for r in recs])
    wov = np.concatenate([r["w_over"] for r in recs])
    # 每 run 內算 Δω 再併(避免跨 run 邊界假急拉)
    dw_all = np.concatenate([np.abs(np.diff(r["rl_w"], prepend=r["rl_w"][:1])) for r in recs])
    m = np.isfinite(fm) & np.isfinite(w) & (fm <= near) & (fm >= BUCKET_EDGES[0])
    if m.sum() < 3:
        return None
    wn = w[m]
    return {
        "n_samples": int(m.sum()),
        "n_runs": len(recs),
        "omega_chatter_std": float(np.std(wn)),          # ★高增益頭號指標
        "domega_rms": float(np.sqrt(np.mean(np.square(dw_all[m])))),
        "max_abs_omega": float(np.max(np.abs(wn))),
        "mean_abs_omega": float(np.mean(np.abs(wn))),
        "w_saturation_rate": float(np.mean(wov[m] > 0.5)),
        "mean_v_near": float(np.nanmean(v[m])),
        "min_front_m": float(np.nanmin(fm[m])),
    }


def mannwhitney_chatter(groups):
    """兩組間 |ω|(近帶已在 caller 過濾前) 的 Mann-Whitney U（抖動分布差異顯著性）。"""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy.stats import mannwhitneyu
    except Exception:
        return None
    keys = list(groups.keys())
    if len(keys) != 2:
        return None
    return keys, mannwhitneyu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.expanduser("~/rover_rl/logs/diag"))
    ap.add_argument("--group-by", choices=["tag", "model"], default="tag")
    ap.add_argument("--since", default="")
    ap.add_argument("--near", type=float, default=2.0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--project", default="rover_rl_sim2real")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    groups, n_runs = collect(args.root, args.group_by, args.since)
    if not groups:
        print("✗ 找不到含 front_m 的 diag run（07-06 後才有 front_m 欄）。", file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.expanduser(f"~/rover_rl/logs/analysis/front_reaction_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    print(f"=== 前方障礙反應 A/B 分析 ===  ({n_runs} runs, {len(groups)} 組)")
    curves = {k: curve_for_group(v) for k, v in groups.items()}
    summaries = {k: scalar_summary(v, args.near) for k, v in groups.items()}

    # ---- curves.csv ----
    with open(os.path.join(outdir, "curves.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["group", "front_m_center", "n", "mean_abs_w", "std_w",
                      "dw_rms", "mean_v", "w_over_rate", "max_abs_w"])
        for k, c in curves.items():
            for i, ctr in enumerate(BUCKET_CENTERS):
                wtr.writerow([k, ctr, int(c["n"][i]),
                              f"{c['mean_abs_w'][i]:.4f}", f"{c['std_w'][i]:.4f}",
                              f"{c['dw_rms'][i]:.4f}", f"{c['mean_v'][i]:.4f}",
                              f"{c['w_over_rate'][i]:.4f}", f"{c['max_abs_w'][i]:.4f}"])

    # ---- summary.csv + .md ----
    cols = ["n_runs", "n_samples", "omega_chatter_std", "domega_rms",
            "max_abs_omega", "mean_abs_omega", "w_saturation_rate", "mean_v_near", "min_front_m"]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["group"] + cols)
        for k, s in summaries.items():
            if s:
                wtr.writerow([k] + [s[c] for c in cols])
    md = [f"# 前方障礙高增益摘要（front_m ≤ {args.near} m）\n",
          "| 組 | runs | 樣本 | ω抖動std | Δω RMS | max|ω| | mean|ω| | ω飽和率 | 近帶mean v | 最近front |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in summaries.items():
        if s:
            md.append(f"| {k} | {s['n_runs']} | {s['n_samples']} | {s['omega_chatter_std']:.3f} | "
                      f"{s['domega_rms']:.3f} | {s['max_abs_omega']:.3f} | {s['mean_abs_omega']:.3f} | "
                      f"{s['w_saturation_rate']:.2%} | {s['mean_v_near']:.3f} | {s['min_front_m']:.2f} |")
    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- Mann-Whitney U（兩組近帶 |ω| 抖動分布）----
    mw = mannwhitney_chatter(groups)
    if mw:
        keys, mwu = mw
        def near_absw(recs):
            fm = np.concatenate([r["front_m"] for r in recs])
            w = np.concatenate([r["rl_w"] for r in recs])
            m = np.isfinite(fm) & np.isfinite(w) & (fm <= args.near) & (fm >= BUCKET_EDGES[0])
            return np.abs(w[m])
        a, b = near_absw(groups[keys[0]]), near_absw(groups[keys[1]])
        if len(a) > 3 and len(b) > 3:
            try:
                u, p = mwu(a, b, alternative="two-sided")
                line = f"\nMann-Whitney U（近帶 |ω|，{keys[0]} vs {keys[1]}）：U={u:.0f}, p={p:.2e}"
                print(line)
                open(os.path.join(outdir, "summary.md"), "a").write(line + "\n")
            except Exception as e:
                print("MWU 失敗:", e)

    # ---- 本地 PNG ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            panels = [("std_w", "omega chatter std(w) [rad/s]  * high-gain"),
                      ("mean_abs_w", "mean |w| [rad/s]"),
                      ("dw_rms", "d-omega RMS (jerk/reversal) [rad/s/step]"),
                      ("mean_v", "mean forward v [m/s]")]
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for ax, (metric, title) in zip(axes.flat, panels):
                for k, c in curves.items():
                    ax.plot(BUCKET_CENTERS, c[metric], marker="o", label=k)
                ax.set_title(title); ax.set_xlabel("front obstacle distance front_m [m]")
                ax.grid(alpha=0.3); ax.legend(); ax.invert_xaxis()  # closer -> right
            fig.suptitle("Front-obstacle reaction vs distance (noise A/B)")
            fig.tight_layout()
            png = os.path.join(outdir, "reaction_curves.png")
            fig.savefig(png, dpi=130); plt.close(fig)
            print(f"\n圖：{png}")
        except Exception as e:
            print("繪圖失敗（不影響數據）:", e)

    # ---- wandb 對比線圖 ----
    if args.wandb and args.wandb_mode != "disabled":
        try:
            os.environ["WANDB_MODE"] = args.wandb_mode
            import wandb
            run = wandb.init(project=args.project, name=f"front_reaction_{stamp}",
                             job_type="analysis", config={
                                 "group_by": args.group_by, "near_m": args.near,
                                 "n_runs": n_runs, "groups": list(groups.keys())})
            keys = list(curves.keys())
            xs = BUCKET_CENTERS
            for metric, title in [("std_w", "omega_chatter_std_vs_dist"),
                                  ("mean_abs_w", "mean_abs_omega_vs_dist"),
                                  ("dw_rms", "domega_rms_vs_dist"),
                                  ("mean_v", "mean_v_vs_dist"),
                                  ("w_over_rate", "omega_saturation_vs_dist")]:
                ys = [list(np.nan_to_num(curves[k][metric], nan=0.0)) for k in keys]
                run.log({title: wandb.plot.line_series(
                    xs=xs, ys=ys, keys=keys, title=title, xname="front_m [m]")})
            # 純量摘要表
            tbl = wandb.Table(columns=["group"] + cols)
            for k, s in summaries.items():
                if s:
                    tbl.add_data(k, *[s[c] for c in cols])
            run.log({"highgain_summary": tbl})
            for k, s in summaries.items():
                if s:
                    for c in cols:
                        run.summary[f"{k}/{c}"] = s[c]
            run.finish()
            print(f"\nwandb: project={args.project} run=front_reaction_{stamp} (mode={args.wandb_mode})")
            if args.wandb_mode == "offline":
                print("  回頭上傳：wandb sync ~/rover_rl/wandb/offline-run-*")
        except Exception as e:
            print("wandb 推送失敗（本地數據已存）:", e)

    print(f"\n輸出資料夾：{outdir}")


if __name__ == "__main__":
    main()
