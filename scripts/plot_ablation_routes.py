#!/usr/bin/env python3
"""plot_ablation_routes.py — 消融實驗路線疊圖：各演算法在相同 A↔B 點位的軌跡，畫在點雲地圖上。

論文用途：把 RL / DWA / PID / PID+VO 在同一組往返點位跑出來的軌跡疊在同一張圖上比較，
背景是 NDT 用的點雲地圖（或 2D 佔據地圖）。軌跡來自 diag CSV 的 map_x/map_y
（TF map→base_footprint，與 RViz 同一條鏈），與地圖同屬 map frame，可直接疊。

為什麼不用 rosbag 回放截圖：RViz 一次只能放一組 bag，四種演算法疊不到同一張圖上。
rosbag 回放適合定性檢查/錄影，這支適合出論文圖。

── 用法 ─────────────────────────────────────────────────────────────
# 1) 最省事：吃 pingpong_metrics 索引，依 experiment_tag 自動分組
python3 scripts/plot_ablation_routes.py --index ~/rover_rl/logs/diag/pingpong_metrics_<時間>.csv

# 2) 沒有索引檔時明確指定（值可以是 run 資料夾或 csv；同一 tag 可給多次 = 多段疊同色）
python3 scripts/plot_ablation_routes.py \
    --run dwa=~/rover_rl/logs/diag/diag_A \
    --run pid=~/rover_rl/logs/diag/diag_B \
    --run pid_vo=~/rover_rl/logs/diag/diag_C \
    --run mppi=~/rover_rl/logs/diag/diag_D \
    --run rl=~/rover_rl/logs/diag/diag_E

# 3) 速度變化：每個演算法一張子圖，軌跡依速度上色
python3 scripts/plot_ablation_routes.py --index ... --speed --speed-col cmd_v

# 4) 背景換成 2D 佔據地圖（比較乾淨，適合黑白印刷）
python3 scripts/plot_ablation_routes.py --index ... --map grid

# 5) 分段圖：每段一格，看單段行為與段間一致性（多組時每組一列，同段上下對齊）
python3 scripts/plot_ablation_routes.py --index ... --per-leg
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _use_cjk_font() -> bool:
    """matplotlib 預設字型沒有中文字 → 圖上的中文會變成空心方框（論文圖直接報廢）。
    找系統已裝的 CJK 字型來用；找不到就回 False，由呼叫端改用英文標籤。"""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    # ⚠ Noto CJK 是 .ttc 集合檔，matplotlib 只註冊集合裡第一個 face → 系統明明裝了
    # TC/SC，字型清單裡卻只看得到 "…CJK JP"。但那是同一份完整 CJK 字集，繁體字照樣有，
    # 只有少數共用碼位的字形風格偏日式。故 JP 名稱也要列入候選，否則會誤判成「沒中文字型」。
    for name in ("Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
                 "Noto Serif CJK TC", "Noto Serif CJK SC", "Noto Serif CJK JP",
                 "WenQuanYi Zen Hei", "Droid Sans Fallback"):
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False   # CJK 字型的負號會破圖，關掉
            return True
    return False


HAS_CJK = _use_cjk_font()


def L(zh: str, en: str) -> str:
    """沒有中文字型時自動退回英文標籤，避免出圖變方框。"""
    return zh if HAS_CJK else en

DEFAULT_PCD = "/home/aa/ndt_ws/src/ndt_localizer/map/3F_314.pcd"
DEFAULT_GRID_YAML = "/home/aa/maps/4v3F.yaml"
# 各 controller 的固定配色：同一演算法在不同圖表中顏色一致，論文才好對照
TAG_COLORS = {
    "rl": "#d62728", "dwa": "#1f77b4", "pid": "#2ca02c", "pid_vo": "#ff7f0e",
    "mppi": "#9467bd",
}
# ⚠ 這裡不能再放 TAG_COLORS 已用掉的顏色，否則某組 baseline 會跟固定配色撞色
FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# ── 讀資料 ──
def _csv_of(path: str) -> str | None:
    """給 run 資料夾或 csv 路徑，回傳實際 csv 路徑。"""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        cands = sorted(f for f in os.listdir(path)
                       if f.endswith(".csv") and not f.endswith("_params.csv"))
        return os.path.join(path, cands[0]) if cands else None
    return path if os.path.isfile(path) else None


def load_from_index(index_path: str, skip_outcomes: set[str]) -> "OrderedDict[str, list]":
    """吃 pingpong_metrics_<session>.csv：每列一個 leg，含 experiment_tag 與該段 csv 檔名。"""
    index_path = os.path.expanduser(index_path)
    base = os.path.dirname(index_path)
    groups: "OrderedDict[str, list]" = OrderedDict()
    with open(index_path, newline="") as f:
        for row in csv.DictReader(f):
            outcome = (row.get("outcome") or "").strip()
            if outcome in skip_outcomes:
                continue
            tag = (row.get("experiment_tag") or "").strip() or "(untagged)"
            name = (row.get("csv") or "").strip()
            if not name:
                continue
            # metrics 裡存的是檔名；實際檔案在 diag_<時間>/ 子資料夾內
            stem = name[:-4] if name.endswith(".csv") else name
            for cand in (os.path.join(base, stem, name), os.path.join(base, name)):
                if os.path.isfile(cand):
                    groups.setdefault(tag, []).append(cand)
                    break
    return groups


def load_traj(csv_path: str, speed_col: str) -> dict | None:
    """讀一段 diag CSV → map frame 軌跡 + 速度。只取 pose_src=tf 的列（odom 退化的座標不可信）。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠ 讀不到 {csv_path}: {e}", file=sys.stderr)
        return None
    if "map_x" not in df.columns or "map_y" not in df.columns:
        print(f"  ⚠ {os.path.basename(csv_path)} 無 map_x/map_y 欄，略過", file=sys.stderr)
        return None
    if "pose_src" in df.columns:
        df = df[df["pose_src"] == "tf"]
    df = df.dropna(subset=["map_x", "map_y"])
    if df.empty:
        print(f"  ⚠ {os.path.basename(csv_path)} 無有效 map frame 位姿（NDT 沒在跑？）",
              file=sys.stderr)
        return None
    spd = None
    if speed_col in df.columns:
        spd = pd.to_numeric(df[speed_col], errors="coerce").to_numpy()
    return {
        "x": df["map_x"].to_numpy(dtype=float),
        "y": df["map_y"].to_numpy(dtype=float),
        "v": spd,
        "name": os.path.basename(csv_path),
    }


# ── 背景地圖 ──
def draw_pcd(ax, pcd_path: str, bbox, z_min: float, z_max: float, max_pts: int) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("⚠ 沒有 open3d，改用 --map grid 或 --map none", file=sys.stderr)
        return
    pts = np.asarray(o3d.io.read_point_cloud(os.path.expanduser(pcd_path)).points)
    if pts.size == 0:
        print(f"⚠ 點雲讀不到內容：{pcd_path}", file=sys.stderr)
        return
    x0, x1, y0, y1 = bbox
    m = ((pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
         & (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
         & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
    pts = pts[m]
    if len(pts) > max_pts:      # 下採樣純為出圖速度與檔案大小，不影響幾何
        pts = pts[np.random.default_rng(0).choice(len(pts), max_pts, replace=False)]
    print(f"  背景點雲：{len(pts)} 點（z {z_min}~{z_max}m）")
    ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c="#b0b0b0", linewidths=0, zorder=1)


def draw_grid(ax, yaml_path: str) -> None:
    import yaml
    from PIL import Image
    yaml_path = os.path.expanduser(yaml_path)
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = os.path.join(os.path.dirname(yaml_path), meta["image"])
    img = np.array(Image.open(img_path))
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    h, w = img.shape[:2]
    # PGM 原點在左下、影像列由上而下 → origin="lower" 搭配翻轉列
    ax.imshow(np.flipud(img), cmap="gray", origin="lower",
              extent=[ox, ox + w * res, oy, oy + h * res], zorder=1, alpha=0.85)


# ── 畫圖 ──
def _color_of(tag: str, spare: list) -> str:
    """依 tag 取固定配色。

    ⚠ 必須「由長到短」比對前綴：tag 會帶場景後綴（dwa_fixed_obstacle、
    pid_vo_random…），若用 split("_")[0] 或短前綴先比，pid_vo 會被當成 pid
    拿到同一個顏色 → 兩條線在圖上完全分不出來。
    """
    if tag in TAG_COLORS:
        return TAG_COLORS[tag]
    for key in sorted(TAG_COLORS, key=len, reverse=True):
        if tag.startswith(key):
            return TAG_COLORS[key]
    return spare.pop(0) if spare else "#333333"


def _bbox_of(groups: dict, margin: float):
    xs = np.concatenate([t["x"] for ts in groups.values() for t in ts])
    ys = np.concatenate([t["y"] for ts in groups.values() for t in ts])
    return (xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin)


def _bg(ax, args, bbox):
    if args.map == "pcd":
        draw_pcd(ax, args.pcd, bbox, args.z_min, args.z_max, args.max_points)
    elif args.map == "grid":
        draw_grid(ax, args.grid_yaml)


def _finish(ax, bbox, title):
    x0, x1, y0, y1 = bbox
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")


def plot_compare(groups, args, bbox):
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    _bg(ax, args, bbox)
    spare = list(FALLBACK_COLORS)
    for tag, trajs in groups.items():
        color = _color_of(tag, spare)
        for i, t in enumerate(trajs):
            ax.plot(t["x"], t["y"], color=color, lw=1.8, alpha=0.85, zorder=3,
                    label=f"{tag} (n={len(trajs)})" if i == 0 else None)
            ax.plot(t["x"][0], t["y"][0], "o", color=color, ms=6, mec="k",
                    mew=0.6, zorder=4)
            ax.plot(t["x"][-1], t["y"][-1], "s", color=color, ms=6, mec="k",
                    mew=0.6, zorder=4)
    _finish(ax, bbox, args.title or L("各演算法在相同往返點位的行駛路線（○起點 □終點）",
                                      "Routes of each algorithm over the same A-B run "
                                      "(o=start, s=end)"))
    ax.legend(loc="best", framealpha=0.9)
    return fig


def plot_per_leg(groups, args, bbox):
    """每段一張子圖，共用座標範圍與背景 —— 看單段行為與段間一致性。

    疊圖能看整體分布，但看不出「哪一段偏了、偏在哪裡」。分段圖每格只畫一段，
    座標範圍與背景固定，段與段之間可直接對位比較。

    排版：單組時排成接近正方形的網格；多組時每組佔一列（同一行=同一段序），
    四組對比時同一段的四種走法會上下對齊。
    """
    multi = len(groups) > 1
    if multi:
        nrow = len(groups)
        ncol = max(len(v) for v in groups.values())
    else:
        n = len(next(iter(groups.values())))
        ncol = int(np.ceil(np.sqrt(n)))
        nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(args.width * ncol / 2.6,
                                                  args.height * nrow / 2.2),
                             squeeze=False)
    # 背景點雲每格都要畫，總點數 = max_points × 格數 會很慢；分攤後每格用少一點
    bg_args = argparse.Namespace(**vars(args))
    bg_args.max_points = max(8000, args.max_points // max(1, nrow * ncol))

    spare = list(FALLBACK_COLORS)
    # (row, col, tag, color, traj) 的攤平清單：多組→每組一列；單組→依序填滿網格
    cells = []
    for r, (tag, trajs) in enumerate(groups.items()):
        color = _color_of(tag, spare)
        for i, t in enumerate(trajs):
            row, col = (r, i) if multi else (i // ncol, i % ncol)
            cells.append((row, col, tag, color, t, i + 1))
    used = {(c[0], c[1]) for c in cells}
    for row in range(nrow):
        for col in range(ncol):
            if (row, col) not in used:
                axes[row][col].axis("off")

    for row, col, tag, color, t, leg_no in cells:
            ax = axes[row][col]
            _bg(ax, bg_args, bbox)
            ax.plot(t["x"], t["y"], color=color, lw=2.0, alpha=0.95, zorder=3)
            ax.plot(t["x"][0], t["y"][0], "o", color=color, ms=7, mec="k", mew=0.7, zorder=4)
            ax.plot(t["x"][-1], t["y"][-1], "s", color=color, ms=7, mec="k", mew=0.7, zorder=4)
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{tag} #{leg_no}", fontsize=9)
    fig.suptitle(args.title or L("分段行駛路線（每格一段，○起點 □終點）",
                                 "Per-run routes (one panel per run; o=start, s=end)"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_speed(groups, args, bbox):
    n = len(groups)
    ncol = min(n, 2)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(args.width * ncol / 1.6,
                                                 args.height * nrow / 1.6),
                             squeeze=False)
    vals = np.concatenate([t["v"][~np.isnan(t["v"])]
                           for ts in groups.values() for t in ts
                           if t["v"] is not None and np.isfinite(t["v"]).any()]
                          or [np.array([0.0, 1.0])])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    sc = None
    for idx, (tag, trajs) in enumerate(groups.items()):
        ax = axes[idx // ncol][idx % ncol]
        _bg(ax, args, bbox)
        for t in trajs:
            v = t["v"]
            if v is None or not np.isfinite(v).any():
                ax.plot(t["x"], t["y"], color="#333", lw=1.5, zorder=3)
                continue
            sc = ax.scatter(t["x"], t["y"], c=v, cmap="viridis", s=6,
                            vmin=vmin, vmax=vmax, zorder=3, linewidths=0)
        _finish(ax, bbox, f"{tag}")
    for k in range(len(groups), nrow * ncol):     # 多餘的空格子關掉
        axes[k // ncol][k % ncol].axis("off")
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85,
                     label=f"{args.speed_col} (m/s)")
    fig.suptitle(args.title or L(f"各演算法行駛路線與速度變化（{args.speed_col}）",
                                 f"Route and speed profile per algorithm ({args.speed_col})"))
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(
        description="消融實驗路線疊圖（點雲地圖背景）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_argument_group("資料來源（擇一）")
    src.add_argument("--index", help="pingpong_metrics_<session>.csv，依 experiment_tag 自動分組")
    src.add_argument("--run", action="append", default=[], metavar="TAG=PATH",
                     help="明確指定；PATH 可為 run 資料夾或 csv。可重複，同 tag 會疊同色")
    ap.add_argument("--skip-outcome", default="",
                    help="用 --index 時要略過的 outcome，逗號分隔（如 timeout,collision）")

    bg = ap.add_argument_group("背景地圖")
    bg.add_argument("--map", choices=["pcd", "grid", "none"], default="pcd")
    bg.add_argument("--pcd", default=DEFAULT_PCD)
    bg.add_argument("--grid-yaml", default=DEFAULT_GRID_YAML)
    bg.add_argument("--z-min", type=float, default=-1.0, help="點雲 z 下界（濾地板）")
    bg.add_argument("--z-max", type=float, default=1.5, help="點雲 z 上界（濾天花板）")
    bg.add_argument("--max-points", type=int, default=120000, help="背景點雲下採樣上限")

    ap.add_argument("--speed", action="store_true", help="改畫「每演算法一張子圖、依速度上色」")
    ap.add_argument("--per-leg", action="store_true",
                    help="改畫「每段一張子圖」（看單段行為與段間一致性；多組時每組一列）")
    ap.add_argument("--speed-col", default="cmd_v",
                    help="速度欄：cmd_v(送進mux) / act_v(odom實測) / odom_v。預設 cmd_v")
    ap.add_argument("--margin", type=float, default=3.0, help="軌跡外擴邊界(m)")
    ap.add_argument("--width", type=float, default=9.0)
    ap.add_argument("--height", type=float, default=7.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.index and not args.run:
        ap.error("要給 --index 或至少一個 --run")

    skip = {s.strip() for s in args.skip_outcome.split(",") if s.strip()}
    groups: "OrderedDict[str, list]" = OrderedDict()

    if args.index:
        raw = load_from_index(args.index, skip)
        print(f"索引 {os.path.basename(args.index)}：{len(raw)} 個 tag")
        for tag, paths in raw.items():
            for p in paths:
                t = load_traj(p, args.speed_col)
                if t:
                    groups.setdefault(tag, []).append(t)
    for item in args.run:
        if "=" not in item:
            ap.error(f"--run 要用 TAG=PATH 格式：{item}")
        tag, path = item.split("=", 1)
        c = _csv_of(path)
        if not c:
            print(f"⚠ 找不到 csv：{path}", file=sys.stderr)
            continue
        t = load_traj(c, args.speed_col)
        if t:
            groups.setdefault(tag.strip(), []).append(t)

    if not groups:
        print("✗ 沒有任何可用軌跡。檢查：① NDT 有沒有在跑（pose_src 要是 tf）"
              "② CSV 路徑是否正確", file=sys.stderr)
        return 1

    total = sum(len(v) for v in groups.values())
    print(f"載入 {total} 段軌跡：")
    for tag, ts in groups.items():
        pts = sum(len(t['x']) for t in ts)
        print(f"  {tag:<16} {len(ts)} 段 / {pts} 點")

    bbox = _bbox_of(groups, args.margin)
    print(f"繪圖範圍 x[{bbox[0]:.1f},{bbox[1]:.1f}] y[{bbox[2]:.1f},{bbox[3]:.1f}]")

    if args.per_leg:
        fig = plot_per_leg(groups, args, bbox)
    elif args.speed:
        fig = plot_speed(groups, args, bbox)
    else:
        fig = plot_compare(groups, args, bbox)

    out = args.out or os.path.expanduser(
        f"~/rover_rl/logs/ablation_routes_{'speed' if args.speed else 'compare'}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"✅ 已輸出：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
