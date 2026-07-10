"""pingpong_report — 把往返測試的 session 指標彙整成論文用對比表格.

輸入：diag_logger 產生的 `pingpong_metrics_<session>.csv`（每段 leg 一列，含
outcome / duration / path_len / min_sep / experiment_tag / model）。

用途：把「多個部署模型 / 多個實驗條件」的往返結果**分組彙整**成成功率 SR、超時率
TO、碰撞率 CR 及距離/時間統計，一次輸出：
  1. 終端機對比表（人看）
  2. `pingpong_report_<stamp>.csv`  — 每組一列（Excel / 再處理用）
  3. `pingpong_report_<stamp>.md`   — Markdown 表格（貼報告 / GitHub）
  4. `pingpong_report_<stamp>.tex`  — LaTeX booktabs 表格（直接 \\input 進論文）
  5. （可選 --wandb）推一個彙整 run 到 wandb 對比

分組鍵 = (experiment_tag, model)。跑實驗時用
  deploy_rl experiment_tag:=noise_dr     # 有雜訊+域隨機化模型
  deploy_rl experiment_tag:=baseline     # 無雜訊 baseline
分開跑，各自產生一份 session CSV；本工具彙整後即得對比表，一目了然看虛實遷移差異。

用法：
  # 彙整 logs/diag 下所有 session（自動分組）
  ros2 run rover_rl_inference pingpong_report
  # 指定檔案
  ros2 run rover_rl_inference pingpong_report a.csv b.csv --out ~/paper/tables
  # 一併推 wandb
  ros2 run rover_rl_inference pingpong_report --wandb --wandb-project rover_rl_deploy
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from datetime import datetime


DEFAULT_DIR = os.path.expanduser("~/rover_rl/logs/diag")


def _f(x):
    """字串 → float；空字串/非數 → None。"""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        try:
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    r["_src"] = os.path.basename(p)
                    rows.append(r)
        except Exception as e:
            print(f"⚠ 讀取 {p} 失敗：{e}")
    return rows


def _group_key(r: dict) -> tuple[str, str]:
    tag = (r.get("experiment_tag") or "").strip() or "(untagged)"
    model = (r.get("model") or "").strip() or "(unknown)"
    return tag, model


def _mean_std(vals: list[float]):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _aggregate(rows: list[dict]) -> list[dict]:
    """依 (tag, model) 分組，算 SR/TO/CR 與距離/時間統計。回傳每組一個 dict。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault(_group_key(r), []).append(r)

    out = []
    for (tag, model), grp in sorted(groups.items()):
        n = len(grp)
        n_ok = sum(1 for r in grp if r.get("outcome") == "success")
        n_to = sum(1 for r in grp if r.get("outcome") == "timeout")
        n_cr = sum(1 for r in grp if r.get("outcome") == "collision")
        n_col_ev = sum(int(_f(r.get("n_collision_events")) or 0) for r in grp)
        # 時間 / 路徑長：只取成功段（失敗段的時間/長度沒有可比性）
        ok = [r for r in grp if r.get("outcome") == "success"]
        t_m, t_s = _mean_std([_f(r.get("duration_s")) for r in ok])
        pl_m, pl_s = _mean_std([_f(r.get("path_len_m")) for r in ok])
        # 最近間隙 min_sep：全段都算（避障品質，越大越安全）
        seps = [_f(r.get("min_sep_m")) for r in grp]
        seps = [s for s in seps if s is not None]
        sep_min = min(seps) if seps else None
        sep_mean, _ = _mean_std(seps)
        sweeps = [_f(r.get("min_sweep_m")) for r in grp]
        sweeps = [s for s in sweeps if s is not None]
        sweep_min = min(sweeps) if sweeps else None
        pct = lambda k: 100.0 * k / n if n else 0.0
        out.append({
            "experiment_tag": tag,
            "model": model,
            "n_legs": n,
            "n_roundtrips": round(n / 2.0, 1),
            "SR_pct": round(pct(n_ok), 1),
            "TO_pct": round(pct(n_to), 1),
            "CR_pct": round(pct(n_cr), 1),
            "n_success": n_ok, "n_timeout": n_to, "n_collision": n_cr,
            "n_collision_events": n_col_ev,
            "time_to_goal_s_mean": None if t_m is None else round(t_m, 1),
            "time_to_goal_s_std": None if t_s is None else round(t_s, 1),
            "path_len_m_mean": None if pl_m is None else round(pl_m, 2),
            "min_sep_m_min": None if sep_min is None else round(sep_min, 3),
            "min_sep_m_mean": None if sep_mean is None else round(sep_mean, 3),
            "min_sweep_m_min": None if sweep_min is None else round(sweep_min, 3),
        })
    return out


def _cell(v, dash="—"):
    return dash if v is None else str(v)


def _print_console(agg: list[dict]) -> None:
    hdr = ["條件(tag)", "model", "段數", "來回", "SR%", "TO%", "CR%",
           "撞次", "到點時間s", "路徑m", "最近人m", "最近障m"]
    def rowvals(a):
        t = _cell(a["time_to_goal_s_mean"])
        if a["time_to_goal_s_mean"] is not None:
            t = f"{a['time_to_goal_s_mean']}±{a['time_to_goal_s_std']}"
        return [a["experiment_tag"], a["model"], a["n_legs"], a["n_roundtrips"],
                a["SR_pct"], a["TO_pct"], a["CR_pct"], a["n_collision_events"],
                t, _cell(a["path_len_m_mean"]), _cell(a["min_sep_m_mean"]),
                _cell(a["min_sweep_m_min"])]
    table = [hdr] + [[str(x) for x in rowvals(a)] for a in agg]
    widths = [max(len(row[i]) for row in table) for i in range(len(hdr))]
    print("\n================= 往返測試對比（SR / TO / CR） =================")
    for ri, row in enumerate(table):
        line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))
        print("  " + line)
        if ri == 0:
            print("  " + "  ".join("-" * widths[i] for i in range(len(hdr))))
    print("================================================================\n")


def _write_csv(agg: list[dict], path: str) -> None:
    if not agg:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)


def _fmt(v):
    return "--" if v is None else v


def _write_md(agg: list[dict], path: str) -> None:
    cols = [("Condition", "experiment_tag"), ("Model", "model"),
            ("N (legs)", "n_legs"), ("SR (%)", "SR_pct"), ("TO (%)", "TO_pct"),
            ("CR (%)", "CR_pct"), ("Collisions", "n_collision_events"),
            ("Time (s)", None), ("Path (m)", "path_len_m_mean"),
            ("Min-Sep (m)", "min_sep_m_mean")]
    lines = ["| " + " | ".join(h for h, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for a in agg:
        t = _fmt(a["time_to_goal_s_mean"])
        if a["time_to_goal_s_mean"] is not None:
            t = f"{a['time_to_goal_s_mean']}±{a['time_to_goal_s_std']}"
        cells = []
        for h, k in cols:
            cells.append(str(t) if k is None else str(_fmt(a[k])))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_tex(agg: list[dict], path: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Real-robot round-trip navigation performance "
        r"(success SR, timeout TO, collision CR).}",
        r"  \label{tab:pingpong_sr_cr}",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    Condition & Model & $N$ & SR(\%) & TO(\%) & CR(\%) "
        r"& Time(s) & Min-Sep(m) \\",
        r"    \midrule",
    ]
    for a in agg:
        t = "--"
        if a["time_to_goal_s_mean"] is not None:
            t = f"{a['time_to_goal_s_mean']}$\\pm${a['time_to_goal_s_std']}"
        model = str(a["model"]).replace("_", r"\_")
        tag = str(a["experiment_tag"]).replace("_", r"\_")
        lines.append(
            f"    {tag} & {model} & {a['n_legs']} & {a['SR_pct']} & "
            f"{a['TO_pct']} & {a['CR_pct']} & {t} & "
            f"{_fmt(a['min_sep_m_mean'])} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _push_wandb(agg: list[dict], project: str, run_name: str) -> None:
    try:
        os.environ.setdefault("WANDB_MODE", "offline")
        import wandb
    except Exception as e:
        print(f"⚠ wandb 不可用，略過：{e}")
        return
    run = wandb.init(project=project, name=run_name, job_type="report")
    tbl = wandb.Table(columns=list(agg[0].keys()))
    for a in agg:
        tbl.add_data(*[a[k] for k in agg[0].keys()])
    run.log({"pingpong_comparison": tbl})
    # 每組 summary 指標（方便 wandb 面板 group-by 條件比 SR/CR）
    for a in agg:
        run.log({f"{a['experiment_tag']}/{a['model']}/SR_pct": a["SR_pct"],
                 f"{a['experiment_tag']}/{a['model']}/CR_pct": a["CR_pct"],
                 f"{a['experiment_tag']}/{a['model']}/TO_pct": a["TO_pct"]})
    run.finish()
    print(f"✓ 已推 wandb（project={project}, run={run_name}, mode={os.environ.get('WANDB_MODE')}）")


def main(argv=None):
    ap = argparse.ArgumentParser(description="彙整往返測試 SR/TO/CR 對比表格")
    ap.add_argument("paths", nargs="*",
                    help="pingpong_metrics_*.csv（留空=彙整 logs/diag 下全部）")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="session CSV 搜尋目錄")
    ap.add_argument("--out", default=None, help="輸出目錄（預設同 --dir）")
    ap.add_argument("--wandb", action="store_true", help="一併推 wandb 彙整 run")
    ap.add_argument("--wandb-project", default="rover_rl_deploy")
    args = ap.parse_args(argv)

    paths = args.paths or sorted(glob.glob(
        os.path.join(args.dir, "pingpong_metrics_*.csv")))
    if not paths:
        print(f"找不到 session 指標 CSV（{args.dir}/pingpong_metrics_*.csv）。\n"
              "先跑往返測試累積幾段，diag_logger 會自動產生。")
        return
    print(f"讀取 {len(paths)} 份 session CSV：")
    for p in paths:
        print(f"  • {p}")

    rows = _load_rows(paths)
    if not rows:
        print("沒有可用的 leg 資料。")
        return
    agg = _aggregate(rows)
    _print_console(agg)

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"pingpong_report_{stamp}")
    _write_csv(agg, base + ".csv")
    _write_md(agg, base + ".md")
    _write_tex(agg, base + ".tex")
    print(f"已輸出論文表格：\n  {base}.csv\n  {base}.md\n  {base}.tex")
    if args.wandb:
        _push_wandb(agg, args.wandb_project, f"pingpong_report_{stamp}")


if __name__ == "__main__":
    main()
