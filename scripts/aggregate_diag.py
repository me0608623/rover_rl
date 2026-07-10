#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate_diag.py — 跨趟 diag CSV 彙總分析（A/B 模型比較 E3 主結果）

獨立離線工具，與 deploy 棧解耦：讀 deploy_rl_shell 每趟自動產生的 diag CSV
（~/rover_rl/logs/diag/<run>/diag_*.csv + _params.json），逐趟算導航效能指標，
依「模型」分組彙總、做 Mann-Whitney U 檢定、輸出彙總表（+ 可選圖）。

關鍵正確性：
  - 距離/成功一律「自己用 map_x/map_y 與 goal_x/goal_y 幾何重算」，
    繞開 diag 已知有 bug 的 dist_to_goal / heading_err 欄
    （見記憶 ndt-pose-is-map-odom：那兩欄整欄可能 ≈180° 錯）。
  - 模型名來自 _params.json 的 "model"；場景來自 run 資料夾名的 label 後綴。

用法：
  python3 scripts/aggregate_diag.py [ROOT] [--tol 0.6] [--since 20260624] [--plot]
    ROOT     diag 根目錄（預設 ~/rover_rl/logs/diag）
    --tol    到達 goal 容差(m)，預設 0.6（對齊 auto_stop_goal_tol）
    --since  只看資料夾名 >= 此日期字串者（例 20260624）
    --plot   產生長條圖 png（需 matplotlib）
    --out    彙總 CSV 輸出路徑（預設 ./aggregate_summary.csv）
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:  # scipy 在此機可 import（僅 numpy 版本 warning，不影響 mannwhitneyu）
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

DT = 0.05  # diag 預設 20Hz；僅用於把「步差 RMS」標註單位，實際用相鄰列差分


# ---------- CSV 讀取小工具（self-contained，不依賴 ROS / 套件）----------
def _read_csv(path):
    """回傳 (cols_float dict, cols_str dict, n_rows)。非數值欄存成字串。"""
    # 容錯：早期 logger 崩潰會在 CSV 留 NUL byte，先清掉再解析
    with open(path, "r", errors="replace", newline="") as f:
        text = f.read().replace("\x00", "")
    rdr = csv.reader(text.splitlines())
    header = next(rdr)
    rows = [r for r in rdr if r]
    n = len(rows)
    fcols, scols = {}, {}
    for j, name in enumerate(header):
        vals_f = np.full(n, np.nan, dtype=float)
        vals_s = [""] * n
        numeric = True
        for i, r in enumerate(rows):
            v = r[j] if j < len(r) else ""
            vals_s[i] = v
            try:
                vals_f[i] = float(v)
            except (ValueError, TypeError):
                numeric = False
        if numeric:
            fcols[name] = vals_f
        else:
            scols[name] = vals_s
    return fcols, scols, n


def _finite_pair(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], m


# ---------- 單趟指標 ----------
def trial_metrics(csv_path, tol):
    f, s, n = _read_csv(csv_path)
    if n < 5:
        return None

    def col(name):
        return f.get(name, np.full(n, np.nan))

    map_x, map_y = col("map_x"), col("map_y")
    goal_x, goal_y = col("goal_x"), col("goal_y")
    sweep_min = col("sweep_min_m")
    cmd_w = col("cmd_w")
    sent_v, sent_w = col("sent_v"), col("sent_w")
    t_rel = col("t_rel")
    lag_ms = col("lag_ms")

    # 幾何重算 車→goal 距離（繞開 buggy dist_to_goal）
    valid = np.isfinite(map_x) & np.isfinite(map_y) & np.isfinite(goal_x) & np.isfinite(goal_y)
    if valid.sum() < 5:
        return None
    d = np.hypot(goal_x - map_x, goal_y - map_y)
    d_valid = d[valid]

    # 起點（第一個有效車姿）、goal（取有效中位數，穩健）
    first_idx = np.argmax(valid)
    start_x, start_y = map_x[first_idx], map_y[first_idx]
    gx, gy = np.nanmedian(goal_x[valid]), np.nanmedian(goal_y[valid])
    straight = math.hypot(gx - start_x, gy - start_y)

    # 路徑長（有效車姿相鄰差分）
    vx, vy = map_x[valid], map_y[valid]
    path_len = float(np.sum(np.hypot(np.diff(vx), np.diff(vy))))

    min_dist = float(np.nanmin(d_valid))
    final_dist = float(d_valid[-1])
    reached = bool(min_dist < tol)

    # 到達時間：第一次 d<tol 的 t_rel
    ttg = float("nan")
    if reached:
        idx_reach = np.where((d < tol) & valid)[0]
        if idx_reach.size and np.isfinite(t_rel[idx_reach[0]]):
            ttg = float(t_rel[idx_reach[0]])
    duration = float(np.nanmax(t_rel)) if np.isfinite(t_rel).any() else float("nan")

    # SPL 成分（成功才計；比值夾 ≤1）
    spl = 0.0
    if reached and path_len > 1e-3:
        spl = float(min(1.0, straight / max(path_len, straight, 1e-6)))

    # 最小淨空
    min_clear = float(np.nanmin(sweep_min)) if np.isfinite(sweep_min).any() else float("nan")

    # 角速度抖動（舞龍舞獅）：cmd_w 相鄰步差的 RMS + 平均 |cmd_w|
    cw = cmd_w[np.isfinite(cmd_w)]
    dw_rms = float(np.sqrt(np.mean(np.diff(cw) ** 2))) if cw.size > 2 else float("nan")
    mean_abs_w = float(np.mean(np.abs(cw))) if cw.size else float("nan")

    # 卡住比例：送出速度近 0 但還沒到 goal 的時間佔比
    stuck_mask = (
        np.isfinite(sent_v) & np.isfinite(sent_w) & valid
        & (np.abs(sent_v) < 0.02) & (np.abs(sent_w) < 0.05) & (d > tol)
    )
    denom = (valid & (d > tol)).sum()
    stuck_frac = float(stuck_mask.sum() / denom) if denom > 0 else 0.0

    mean_lag = float(np.nanmean(lag_ms)) if np.isfinite(lag_ms).any() else float("nan")

    return dict(
        reached=int(reached),
        min_dist_m=min_dist,
        final_dist_m=final_dist,
        time_to_goal_s=ttg,
        duration_s=duration,
        path_len_m=path_len,
        straight_m=straight,
        spl=spl,
        min_clearance_m=min_clear,
        dw_rms=dw_rms,
        mean_abs_w=mean_abs_w,
        stuck_frac=stuck_frac,
        mean_lag_ms=mean_lag,
        n_rows=n,
    )


def _params_meta(run_dir):
    """從 _params.json 抓 model；從資料夾名抓 label(場景)。"""
    model, scenario = "unknown", ""
    pj = glob.glob(os.path.join(run_dir, "*_params.json"))
    if pj:
        try:
            with open(pj[0]) as fp:
                model = json.load(fp).get("model", "unknown")
        except Exception:
            pass
    # 資料夾名 diag_<date>_<time>[_<label>] → label
    base = os.path.basename(run_dir.rstrip("/"))
    parts = base.split("_")
    if len(parts) > 3:  # diag, date, time, label...
        scenario = "_".join(parts[3:])
    return model, scenario


# ---------- 彙總 ----------
METRICS = [
    ("reached", "成功率", "max"),
    ("spl", "SPL", "max"),
    ("min_clearance_m", "最小淨空(m)", "max"),
    ("dw_rms", "Δω RMS(抖動)", "min"),
    ("mean_abs_w", "平均|ω|", "min"),
    ("stuck_frac", "卡住比例", "min"),
    ("time_to_goal_s", "到達時間(s)", "min"),
    ("path_len_m", "路徑長(m)", "min"),
    ("mean_lag_ms", "平均延遲(ms)", "min"),
]


def _fmt(mean, std, n):
    if not np.isfinite(mean):
        return "  n/a"
    return f"{mean:.3f}±{std:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.expanduser("~/rover_rl/logs/diag"))
    ap.add_argument("--tol", type=float, default=0.6)
    ap.add_argument("--since", default="")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="aggregate_summary.csv")
    args = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*/")) if os.path.isdir(d))
    if args.since:
        run_dirs = [d for d in run_dirs if os.path.basename(d.rstrip("/")) >= f"diag_{args.since}"]
    if not run_dirs:
        print(f"找不到 diag run：{args.root}", file=sys.stderr)
        sys.exit(1)

    trials = []  # (model, scenario, dir, metrics)
    for rd in run_dirs:
        csvs = glob.glob(os.path.join(rd, "diag_*.csv"))
        if not csvs:
            continue
        model, scenario = _params_meta(rd)
        for cp in csvs:
            try:
                m = trial_metrics(cp, args.tol)
            except Exception as e:
                print(f"  跳過 {os.path.basename(cp)}：{e}", file=sys.stderr)
                continue
            if m:
                trials.append((model, scenario, os.path.basename(rd.rstrip("/")), m))

    if not trials:
        print("沒有可用的趟次（CSV 太短或缺 map/goal 欄）。", file=sys.stderr)
        sys.exit(1)

    # 寫每趟明細 CSV
    keys = list(trials[0][3].keys())
    with open(args.out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["model", "scenario", "run", *keys])
        for model, scen, run, m in trials:
            w.writerow([model, scen, run, *[m[k] for k in keys]])

    # 依模型分組
    by_model = defaultdict(list)
    for model, scen, run, m in trials:
        by_model[model].append(m)

    print("=" * 70)
    print(f"跨趟彙總：{len(trials)} 趟，{len(by_model)} 個模型，tol={args.tol}m")
    print(f"明細已寫：{os.path.abspath(args.out)}")
    print("=" * 70)

    models = list(by_model.keys())
    # 表頭
    head = f"{'指標':<16}" + "".join(f"{mm[:22]:>24}" for mm in models)
    print(head)
    print("-" * len(head))
    arrs = {mm: {k: np.array([t[k] for t in by_model[mm]], float) for k in keys} for mm in models}
    for key, label, _better in METRICS:
        line = f"{label:<16}"
        for mm in models:
            a = arrs[mm][key]
            a = a[np.isfinite(a)]
            if a.size:
                line += f"{_fmt(a.mean(), a.std(), a.size):>24}"
            else:
                line += f"{'n/a':>24}"
        print(line)
    print(f"{'趟數 n':<16}" + "".join(f"{len(by_model[mm]):>24}" for mm in models))

    # Mann-Whitney U（恰兩個模型時）
    if len(models) == 2 and HAVE_SCIPY:
        a_name, b_name = models
        print("\n" + "=" * 70)
        print(f"Mann-Whitney U 檢定：{a_name}  vs  {b_name}")
        print("-" * 70)
        for key, label, better in METRICS:
            xa = arrs[a_name][key]; xa = xa[np.isfinite(xa)]
            xb = arrs[b_name][key]; xb = xb[np.isfinite(xb)]
            if xa.size < 3 or xb.size < 3:
                print(f"{label:<16} 樣本不足")
                continue
            try:
                u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            except Exception as e:
                print(f"{label:<16} 檢定失敗 {e}")
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ma, mb = xa.mean(), xb.mean()
            win = a_name if ((ma > mb) == (better == "max")) else b_name
            print(f"{label:<16} p={p:.4f} {sig:<3} 較佳={win[:24]}")
        print("（* p<.05  ** p<.01  *** p<.001  ns=不顯著）")
    elif len(models) == 2 and not HAVE_SCIPY:
        print("\n（scipy 不可用，略過 Mann-Whitney；明細 CSV 仍可自行檢定）")

    # 可選圖
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            keys_plot = [k for k, _l, _b in METRICS]
            fig, axes = plt.subplots(3, 3, figsize=(15, 11))
            for ax, (key, label, _b) in zip(axes.ravel(), METRICS):
                means = [np.nanmean(arrs[mm][key]) if np.isfinite(arrs[mm][key]).any() else 0 for mm in models]
                stds = [np.nanstd(arrs[mm][key]) if np.isfinite(arrs[mm][key]).any() else 0 for mm in models]
                ax.bar(range(len(models)), means, yerr=stds, capsize=4)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([mm[:14] for mm in models], rotation=15, fontsize=7)
                ax.set_title(label, fontsize=9)
            fig.tight_layout()
            png = os.path.splitext(args.out)[0] + ".png"
            fig.savefig(png, dpi=110)
            print(f"\n圖已存：{os.path.abspath(png)}")
        except Exception as e:
            print(f"\n繪圖失敗（略過）：{e}", file=sys.stderr)


if __name__ == "__main__":
    main()
