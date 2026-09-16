#!/usr/bin/env python3
"""消融實驗論文數據：速度變化 + 障礙淨距 的曲線、分佈與統計表.

與 plot_ablation_routes.py 分工：
  plot_ablation_routes.py  → 空間圖（軌跡疊點雲地圖、軌跡依速度上色）
  plot_ablation_metrics.py → 本檔：**沿路徑進度**的速度/淨距曲線 + 分佈 + 論文統計表

為什麼 x 軸預設是「路徑進度」而不是時間：每段 leg 的長度與耗時都不同，
按時間疊圖會讓「走比較久的那組」被拉長、看起來像是行為不同，實際上只是慢。
正規化成 0~1 的路徑進度後，四組才落在同一個橫軸上可比。
（仍可用 --x time / --x dist 看原始時間或里程。）

每組畫法：個別 leg 細線（半透明）+ 該組在共同格點上的**中位數粗線 + IQR 帶**，
避免十幾條線糊成一團而看不出組間差異。

資料來源與 plot_ablation_routes.py 相同（共用 load_from_index），
所以同一個 pingpong_metrics_<session>.csv 可以同時餵給兩支腳本。

用法：
    # 一次產出全部（速度圖 + 淨距圖 + 統計表）
    python3 scripts/plot_ablation_metrics.py --index ~/rover_rl/logs/diag/pingpong_metrics_<時間>.csv

    # 只要統計表（貼論文用，同時輸出 csv/md/tex）
    python3 scripts/plot_ablation_metrics.py --index ... --only table

    # 換速度來源（預設 act_v=底盤實測；cmd_v=送出指令；rl_v=RL 意圖，僅 RL 組有）
    python3 scripts/plot_ablation_metrics.py --index ... --speed-col cmd_v

    # 沒有索引檔時明確指定（同 tag 可給多次）
    python3 scripts/plot_ablation_metrics.py --run dwa=<run_dir> --run rl=<run_dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 共用資料索引/配色/中文字型，避免兩支腳本各寫一份而漂移
from plot_ablation_routes import (  # noqa: E402
    FALLBACK_COLORS, L, _color_of, _csv_of, _use_cjk_font, load_from_index,
)

N_GRID = 200          # 共同格點數（中位數/IQR 帶用）
MOVING_V = 0.05       # |v| 低於此值視為靜止，不納入速度統計（避免等待段拉低平均）


# ── 讀單段 ──
def load_series(csv_path: str, speed_col: str) -> dict | None:
    """讀一段 diag CSV → 進度軸 + 速度 + 障礙淨距。

    只取 pose_src=tf 的列：NDT 沒在跑時退化的 odom 座標算出來的里程不可信。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠ 讀不到 {csv_path}: {e}", file=sys.stderr)
        return None
    if "pose_src" in df.columns:
        df = df[df["pose_src"] == "tf"]
    if df.empty or "map_x" not in df.columns:
        print(f"  ⚠ {os.path.basename(csv_path)} 無有效 map frame 位姿，略過", file=sys.stderr)
        return None

    num = lambda c: (pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                     if c in df.columns else None)

    x, y = num("map_x"), num("map_y")
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5:
        return None
    # 累積里程（實際走過的距離，不是直線距離）
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x[ok]), np.diff(y[ok])))])
    total = dist[-1] if dist[-1] > 0 else 1.0

    v = num(speed_col)
    if v is None or np.all(np.isnan(v)):        # baseline 沒有 rl_* 欄，退回實測/指令
        for alt in ("act_v", "odom_v", "cmd_v"):
            v = num(alt)
            if v is not None and not np.all(np.isnan(v)):
                speed_col = alt
                break
    w = num(speed_col.replace("_v", "_w"))
    if w is None or np.all(np.isnan(w)):
        w = num("act_w") if num("act_w") is not None else num("odom_w")

    t = num("t_rel")
    return {
        "name": os.path.basename(csv_path),
        "t": (t[ok] - t[ok][0]) if t is not None else np.arange(ok.sum(), dtype=float) * 0.05,
        "dist": dist,
        "prog": dist / total,
        "v": None if v is None else v[ok],
        "w": None if w is None else w[ok],
        "min_sep": (lambda a: None if a is None else a[ok])(num("min_sep_m")),
        "dyn_min": (lambda a: None if a is None else a[ok])(num("dyn_obs_min_m")),
        "front": (lambda a: None if a is None else a[ok])(num("front_m")),
        "collision": (lambda a: None if a is None else a[ok])(num("collision")),
        "speed_col": speed_col,
    }


def load_groups(args) -> "OrderedDict[str, list]":
    """索引檔或 --run → {tag: [series, ...]}。"""
    skip = {s.strip() for s in args.skip_outcome.split(",") if s.strip()}
    if args.index:
        paths = load_from_index(args.index, skip)
    else:
        paths = OrderedDict()
        for spec in args.run:
            tag, _, p = spec.partition("=")
            c = _csv_of(p)
            if c:
                paths.setdefault(tag.strip(), []).append(c)
            else:
                print(f"  ⚠ 找不到 CSV：{p}", file=sys.stderr)
    out: "OrderedDict[str, list]" = OrderedDict()
    for tag, files in paths.items():
        for f in files:
            s = load_series(f, args.speed_col)
            if s:
                out.setdefault(tag, []).append(s)
    return out


# ── 共同格點上的中位數帶 ──
def _band(series: list, xkey: str, ykey: str):
    """把長度不一的多段 resample 到共同格點，回傳 (grid, median, q25, q75)。"""
    valid = [s for s in series if s.get(ykey) is not None]
    if not valid:
        return None
    hi = 1.0 if xkey == "prog" else max(float(s[xkey][-1]) for s in valid)
    grid = np.linspace(0.0, hi, N_GRID)
    stack = []
    for s in valid:
        xs, ys = np.asarray(s[xkey], dtype=float), np.asarray(s[ykey], dtype=float)
        m = ~np.isnan(ys)
        if m.sum() < 5:
            continue
        stack.append(np.interp(grid, xs[m], ys[m], left=np.nan, right=np.nan))
    if not stack:
        return None
    arr = np.vstack(stack)
    with np.errstate(all="ignore"):
        return (grid, np.nanmedian(arr, axis=0),
                np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0))


XLABEL = {
    "prog": ("路徑進度 (0=起點, 1=終點)", "Path progress (0=start, 1=goal)"),
    "dist": ("已行駛距離 (m)", "Distance travelled (m)"),
    "time": ("時間 (s)", "Time (s)"),
}


def _draw_panel(ax, groups, xkey, ykey, ylabel, spare):
    """一個子圖：每組畫 個別 leg 細線 + 中位數粗線 + IQR 帶。"""
    for tag, series in groups.items():
        c = _color_of(tag, spare)
        for s in series:
            if s.get(ykey) is None:
                continue
            ax.plot(s[xkey], s[ykey], color=c, lw=0.6, alpha=0.18, zorder=2)
        b = _band(series, xkey, ykey)
        if b is None:
            continue
        grid, med, q25, q75 = b
        ax.fill_between(grid, q25, q75, color=c, alpha=0.15, lw=0, zorder=3)
        ax.plot(grid, med, color=c, lw=2.0, label=f"{tag} (n={len(series)})", zorder=4)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, lw=0.5)


def plot_speed(groups, args, stamp) -> str:
    """速度變化圖：線速度 + 角速度，沿路徑進度。"""
    fig, axes = plt.subplots(2, 1, figsize=(args.width, args.height), sharex=True)
    spare = list(FALLBACK_COLORS)
    _draw_panel(axes[0], groups, args.x, "v", L("線速度 v (m/s)", "Linear v (m/s)"), spare)
    _draw_panel(axes[1], groups, args.x, "w", L("角速度 ω (rad/s)", "Angular ω (rad/s)"),
                list(FALLBACK_COLORS))
    axes[1].set_xlabel(L(*XLABEL[args.x]))
    axes[0].legend(fontsize=8, loc="best")
    src = next(iter(next(iter(groups.values()))))["speed_col"] if groups else "?"
    axes[0].set_title(args.title or L(
        f"速度變化（{src}，粗線=各組中位數，帶=IQR）",
        f"Speed profile ({src}; bold=median, band=IQR)"))
    fig.tight_layout()
    out = args.out or os.path.join(args.outdir, f"ablation_speed_{stamp}.png")
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def plot_clearance(groups, args, stamp) -> str:
    """障礙淨距圖：沿路徑的 min_sep 曲線 + 各組分佈箱型圖。"""
    fig = plt.figure(figsize=(args.width, args.height))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.1], height_ratios=[1, 1])
    ax_sep, ax_front = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_box = fig.add_subplot(gs[:, 1])

    _draw_panel(ax_sep, groups, args.x, "min_sep",
                L("動態障礙淨距 (m)", "Dynamic clearance (m)"), list(FALLBACK_COLORS))
    ax_sep.axhline(0.0, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax_sep.set_title(args.title or L(
        "與障礙的最近距離（min_sep = 車體表面↔行人表面；0 = 接觸）",
        "Clearance to obstacles (min_sep = body surface to pedestrian surface; 0 = contact)"))
    ax_sep.legend(fontsize=8, loc="best")

    _draw_panel(ax_front, groups, args.x, "front",
                L("前方 LiDAR 距離 (m)", "Front LiDAR range (m)"), list(FALLBACK_COLORS))
    ax_front.set_xlabel(L(*XLABEL[args.x]))

    data, labels, colors, spare = [], [], [], list(FALLBACK_COLORS)
    for tag, series in groups.items():
        vals = np.concatenate([s["min_sep"][~np.isnan(s["min_sep"])]
                               for s in series if s.get("min_sep") is not None]
                              or [np.array([])])
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(tag)
        colors.append(_color_of(tag, spare))
    if data:
        bp = ax_box.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax_box.axhline(0.0, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax_box.set_title(L("淨距分佈", "Clearance dist."), fontsize=10)
    ax_box.grid(alpha=0.3, lw=0.5, axis="y")
    plt.setp(ax_box.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    out = os.path.join(args.outdir, f"ablation_clearance_{stamp}.png")
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


# ── 統計表 ──
def _rms_rate(y, t):
    """|dy/dt| 的 RMS —— 「速度變化量」的量化：越大代表加減速/轉向越劇烈。"""
    if y is None or t is None or len(y) < 3:
        return None
    dy, dt = np.diff(np.asarray(y, float)), np.diff(np.asarray(t, float))
    m = (~np.isnan(dy)) & (dt > 1e-6)
    return float(np.sqrt(np.mean((dy[m] / dt[m]) ** 2))) if m.sum() >= 2 else None


def _agg(vals, fn):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(fn(vals)) if vals else None


def summarize(groups) -> list[dict]:
    """每組一列論文統計。"""
    rows = []
    for tag, series in groups.items():
        cat = lambda k: (np.concatenate([s[k][~np.isnan(s[k])] for s in series
                                         if s.get(k) is not None] or [np.array([])]))
        v_all, w_all = cat("v"), cat("w")
        moving = v_all[np.abs(v_all) > MOVING_V] if v_all.size else v_all
        sep, front, coll = cat("min_sep"), cat("front"), cat("collision")
        rows.append({
            "group": tag,
            "n_legs": len(series),
            # 速度
            "v_mean": float(np.mean(moving)) if moving.size else None,
            "v_max": float(np.max(np.abs(v_all))) if v_all.size else None,
            "w_rms": float(np.sqrt(np.mean(w_all ** 2))) if w_all.size else None,
            # 速度變化量（加加速度感）：每段各算再取平均，避免跨段接縫造成假尖峰
            "dv_dt_rms": _agg([_rms_rate(s.get("v"), s["t"]) for s in series], np.mean),
            "dw_dt_rms": _agg([_rms_rate(s.get("w"), s["t"]) for s in series], np.mean),
            # 與障礙距離
            "sep_min": float(np.min(sep)) if sep.size else None,
            "sep_mean": float(np.mean(sep)) if sep.size else None,
            "sep_p5": float(np.percentile(sep, 5)) if sep.size else None,
            "front_min": float(np.min(front)) if front.size else None,
            "front_mean": float(np.mean(front)) if front.size else None,
            "collision_ratio": float(np.mean(coll > 0.5)) if coll.size else None,
        })
    return rows


COLS = [
    ("group", "組", "Method", ""), ("n_legs", "段數", "Legs", "d"),
    ("v_mean", "平均速度", "v mean", "3"), ("v_max", "最大速度", "v max", "3"),
    ("dv_dt_rms", "線速度變化量", "|dv/dt| rms", "3"),
    ("w_rms", "角速度 RMS", "w rms", "3"),
    ("dw_dt_rms", "角速度變化量", "|dw/dt| rms", "3"),
    ("sep_min", "淨距最近", "sep min", "3"), ("sep_mean", "淨距平均", "sep mean", "3"),
    ("sep_p5", "淨距 P5", "sep p5", "3"),
    ("front_min", "前方最近", "front min", "3"), ("front_mean", "前方平均", "front mean", "3"),
    ("collision_ratio", "接觸比例", "contact", "3"),
]


def _cell(v, fmt):
    """fmt: ""=原樣字串（組名）／"d"=整數／數字=小數位數。None 一律顯示 —。"""
    if v is None:
        return "—"
    if not fmt:
        return str(v)
    return f"{int(v)}" if fmt == "d" else f"{v:.{int(fmt)}f}"


def write_tables(rows, outdir, stamp) -> list[str]:
    """終端 + CSV + Markdown + LaTeX（與 pingpong_report 同一套輸出慣例）。"""
    import csv as _csv
    heads = [L(zh, en) for _, zh, en, _ in COLS]
    print("\n" + L("【消融實驗：速度與障礙淨距】", "[Ablation: speed & clearance]"))
    widths = [max(len(h), 12) for h in heads]
    print("  " + " ".join(h.ljust(w) for h, w in zip(heads, widths)))
    for r in rows:
        print("  " + " ".join(_cell(r[k], f).ljust(w)
                              for (k, _, _, f), w in zip(COLS, widths)))

    paths = []
    p = os.path.join(outdir, f"ablation_metrics_{stamp}.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wr = _csv.DictWriter(fh, fieldnames=[k for k, _, _, _ in COLS])
        wr.writeheader()
        wr.writerows([{k: r[k] for k, _, _, _ in COLS} for r in rows])
    paths.append(p)

    p = os.path.join(outdir, f"ablation_metrics_{stamp}.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(heads) + " |\n")
        fh.write("|" + "---|" * len(heads) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(_cell(r[k], f) for k, _, _, f in COLS) + " |\n")
    paths.append(p)

    p = os.path.join(outdir, f"ablation_metrics_{stamp}.tex")
    with open(p, "w", encoding="utf-8") as fh:
        en = [e for _, _, e, _ in COLS]
        fh.write("\\begin{tabular}{l" + "r" * (len(COLS) - 1) + "}\n\\toprule\n")
        fh.write(" & ".join(en) + " \\\\\n\\midrule\n")
        for r in rows:
            fh.write(" & ".join(_cell(r[k], f) for k, _, _, f in COLS) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description="消融實驗：速度變化 + 障礙淨距 的曲線、分佈與論文統計表",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_argument_group("資料來源（擇一）")
    src.add_argument("--index", help="pingpong_metrics_<session>.csv，依 experiment_tag 分組")
    src.add_argument("--run", action="append", default=[], metavar="TAG=PATH",
                     help="明確指定（可重複，同 tag 疊同組）")
    ap.add_argument("--skip-outcome", default="",
                    help="略過的 outcome（逗號分隔，如 timeout,collision）")
    ap.add_argument("--only", choices=["speed", "clearance", "table"], default="",
                    help="只產出其中一項（預設全出）")
    ap.add_argument("--x", choices=["prog", "dist", "time"], default="prog",
                    help="橫軸：prog=路徑進度(預設，跨段可比) / dist=里程 / time=時間")
    ap.add_argument("--speed-col", default="act_v",
                    help="速度來源欄：act_v(底盤實測，預設) / cmd_v(送出) / rl_v(RL 意圖)")
    ap.add_argument("--outdir", default=os.path.expanduser("~/rover_rl/logs/diag"))
    ap.add_argument("--out", default="", help="速度圖輸出路徑（其餘仍走 --outdir）")
    ap.add_argument("--width", type=float, default=10.0)
    ap.add_argument("--height", type=float, default=7.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    if not args.index and not args.run:
        ap.error("需要 --index 或 --run")
    _use_cjk_font()
    groups = load_groups(args)
    if not groups:
        print("⚠ 沒有讀到任何有效資料（NDT 沒在跑時 pose_src 不是 tf，整段會被濾掉）",
              file=sys.stderr)
        return 1
    print(f"共 {len(groups)} 組：" + "、".join(f"{k}({len(v)} 段)" for k, v in groups.items()))

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    made = []
    if args.only in ("", "speed"):
        made.append(plot_speed(groups, args, stamp))
    if args.only in ("", "clearance"):
        made.append(plot_clearance(groups, args, stamp))
    if args.only in ("", "table"):
        made += write_tables(summarize(groups), args.outdir, stamp)
    print("\n產出：")
    for p in made:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
