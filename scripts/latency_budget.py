#!/usr/bin/env python3
"""端到端延遲預算分解 — 離線（rosbag）+ 推論台架量測.

把 RL 導航閉迴路拆成 5 段，各段量出 ms 級延遲並累加成端到端：

  S1 感測    VLP-16 掃描完成(header.stamp) → PointCloud2 抵達本機     [需 live，見 latency_probe.py]
  S2 前處理  點雲抵達 → /rover_rl/lidar_sweep_72 發布（含 10Hz 取樣）  [需 live]
  S3 推論    sweep 抵達 → RL 決策完成（含 5Hz 取樣 + obs + forward）   [bench 量純算 + live 量取樣]
  S4 指令    RL target → /output/cmd_vel（20Hz 取樣 + 濾波相位 + VO/recovery/mux 各 hop）
  S5 致動    /output/cmd_vel → 底盤實測速度（odom）死時間

本程式負責 **S4 hop / S5 死時間 / S3 純推論算力**，資料來源為既有 rosbag 與模型檔，
不需要車在跑。S1/S2 與 S3 取樣抖動需 LiDAR 在線，用 scripts/latency_probe.py 量。

用法：
  python3 scripts/latency_budget.py --bag ~/rover_rl/logs/bags/deploy_20260722_164122
  python3 scripts/latency_budget.py --bench ~/rover_rl/models/sa5_v3f_tcadapt_60000.ts
  python3 scripts/latency_budget.py --bag <dir> --bench <model.ts>   # 兩段一起
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys

import numpy as np

# cmd 鏈路各 hop：(上游 topic, 下游 topic, 這一 hop 代表的處理節點)
CHAIN = [
    ("/rover_rl/cmd_vel_desired", "/rover_rl/cmd_vel_mppi", "MPPI static_guard"),
    ("/rover_rl/cmd_vel_mppi", "/rover_rl/cmd_vel_recovery_in", "VO safety(接 MPPI)"),
    ("/rover_rl/cmd_vel_desired", "/rover_rl/cmd_vel_recovery_in", "VO safety"),
    ("/rover_rl/cmd_vel_recovery_in", "/input/nav_cmd_vel", "recovery supervisor"),
    ("/rover_rl/cmd_vel_desired", "/input/nav_cmd_vel", "RL→mux 入口(無 wrapper)"),
    ("/input/nav_cmd_vel", "/output/cmd_vel", "cmd_vel mux"),
]


# ─────────────────────────── bag 讀取 ───────────────────────────

def _bag_db(bag_dir: str) -> str:
    cand = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
    if not cand:
        raise SystemExit(f"找不到 .db3：{bag_dir}")
    return os.path.join(bag_dir, cand[0])


def read_bag(bag_dir: str) -> dict[str, list[tuple[float, object]]]:
    """讀 bag → {topic: [(t_recv_秒, msg), ...]}。只解析用得到的型別。"""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    con = sqlite3.connect(_bag_db(bag_dir))
    topics = {tid: (name, typ) for tid, name, typ
              in con.execute("SELECT id, name, type FROM topics")}
    wanted = {t for hop in CHAIN for t in hop[:2]} | {"/odom", "/rover_rl/lidar_sweep_72"}

    out: dict[str, list[tuple[float, object]]] = {}
    for tid, (name, typ) in topics.items():
        if name not in wanted:
            continue
        try:
            cls = get_message(typ)
        except Exception:
            continue
        rows = con.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,),
        ).fetchall()
        if not rows:
            continue
        out[name] = [(ts / 1e9, deserialize_message(bytes(d), cls)) for ts, d in rows]
    con.close()
    return out


def _twist_series(msgs: list[tuple[float, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.array([m[0] for m in msgs])
    v = np.array([m[1].linear.x for m in msgs])
    w = np.array([m[1].angular.z for m in msgs])
    return t, v, w


# ─────────────────────────── S4：鏈路 hop ───────────────────────────

def hop_latency(up: list[tuple[float, object]], dn: list[tuple[float, object]],
                tol: float = 1e-6, win_s: float = 0.5) -> dict | None:
    """值配對法量 hop 延遲：下游每則訊息回頭找「數值相同且最近」的上游訊息。

    wrapper（VO/recovery）改寫數值時配不到 → 該樣本略過，並以 pass_rate 回報改寫比例。
    只用非零樣本：靜止時整條鏈都是 0，配對無鑑別力。
    """
    tu, vu, wu = _twist_series(up)
    td, vd, wd = _twist_series(dn)
    nz = (np.abs(vd) > 1e-3) | (np.abs(wd) > 1e-3)
    lat: list[float] = []
    matched = 0
    total = int(nz.sum())
    j = 0
    for i in np.flatnonzero(nz):
        t_d = td[i]
        # 上游游標推進到 t_d
        while j + 1 < len(tu) and tu[j + 1] <= t_d:
            j += 1
        # 從游標往回找數值相同者
        k = j
        while k >= 0 and t_d - tu[k] <= win_s:
            if abs(vu[k] - vd[i]) <= tol and abs(wu[k] - wd[i]) <= tol:
                lat.append(t_d - tu[k])
                matched += 1
                break
            k -= 1
    if not lat:
        return None
    a = np.array(lat)
    return {
        "n": matched,
        "pass_rate": matched / max(total, 1),
        "p50": float(np.median(a)) * 1e3,
        "p90": float(np.percentile(a, 90)) * 1e3,
        "mean": float(a.mean()) * 1e3,
        "max": float(a.max()) * 1e3,
    }


# ─────────────────────────── S5：致動死時間 ───────────────────────────

def _resample(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, t, y)


def dead_time(cmd: list[tuple[float, object]], odom: list[tuple[float, object]],
              dt: float = 0.02, max_lag_s: float = 1.0,
              min_std: float = 0.04) -> dict | None:
    """互相關估 /output/cmd_vel → odom 實測 的死時間（ω 通道，與 latency.py 同法）。

    以 50Hz 重取樣（比 20Hz 控制率細）→ 死時間解析度 20ms，再對相關峰做拋物線內插。
    v 通道等速直行時訊號太平、互相關不可信（見 docs/2026-06-08 ESCALATION），故只報 ω。
    """
    tc, _, wc = _twist_series(cmd)
    to = np.array([m[0] for m in odom])
    wo = np.array([m[1].twist.twist.angular.z for m in odom])
    t0 = max(tc[0], to[0])
    t1 = min(tc[-1], to[-1])
    if t1 - t0 < 5.0:
        return None
    grid = np.arange(t0, t1, dt)
    s = _resample(tc, wc, grid)
    a = _resample(to, wo, grid)
    if s.std() < min_std or a.std() < min_std:
        return None
    s = (s - s.mean()) / s.std()
    a = (a - a.mean()) / a.std()
    n = len(s)
    max_k = int(max_lag_s / dt)
    cs = [float(np.mean(s[: n - k] * a[k:])) for k in range(max_k + 1)]
    k = int(np.argmax(cs))
    # 拋物線內插取次取樣峰位（相鄰三點）
    k_ref = float(k)
    if 0 < k < len(cs) - 1:
        d = cs[k - 1] - 2 * cs[k] + cs[k + 1]
        if abs(d) > 1e-12:
            k_ref = k + 0.5 * (cs[k - 1] - cs[k + 1]) / d
    return {"lag_ms": k_ref * dt * 1e3, "corr": cs[k], "secs": t1 - t0}


# ─────────────────────────── 取樣週期 ───────────────────────────

def rate_stats(msgs: list[tuple[float, object]]) -> dict:
    t = np.array([m[0] for m in msgs])
    d = np.diff(t)
    d = d[(d > 0) & (d < 2.0)]
    return {
        "n": len(t), "hz": 1.0 / float(np.median(d)) if len(d) else float("nan"),
        "p50_ms": float(np.median(d)) * 1e3 if len(d) else float("nan"),
        "p95_ms": float(np.percentile(d, 95)) * 1e3 if len(d) else float("nan"),
    }


# ─────────────────────────── S3：推論台架 ───────────────────────────

def bench_model(ts_path: str, iters: int = 300, warmup: int = 30,
                device: str = "cpu") -> dict:
    """量純推論算力：走部署同一條路徑（obs build → runner.step → decode），回報 ms 分位。

    直接用 rover_rl_inference 的 PolicyRunner / build_obs_raw / decode_logits_to_cmd，
    避免自己拼 forward 簽章與部署不一致。
    """
    import time

    from rover_rl_inference.action_decoder import ActionParams, decode_logits_to_cmd
    from rover_rl_inference.model_runtime import PolicyRunner, load_bundle
    from rover_rl_inference.obs_builder import ObsParams, build_obs_raw

    bundle = load_bundle(ts_path, device=device)
    runner = PolicyRunner(bundle)
    op = ObsParams()
    ap_ = ActionParams()
    sweep = np.full(72, 0.5, dtype=np.float32)

    def one_step() -> tuple[float, float, float]:
        t0 = time.perf_counter()
        obs = build_obs_raw(
            bundle.raw_obs_dim, last_accel=0.0, linear_vel=0.5, angular_vel=0.1,
            goal_body_x=3.0, goal_body_y=1.0, lidar_sweep_72=sweep, elapsed_s=5.0,
            params=op,
            action_history=(np.zeros(4, dtype=np.float32)
                            if bundle.raw_obs_dim >= 83 else None),
        )
        t1 = time.perf_counter()
        logits = runner.step(obs)
        t2 = time.perf_counter()
        decode_logits_to_cmd(logits, current_linear_vel=0.5, params=ap_,
                             deterministic=True, current_angular_vel=0.1)
        t3 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3

    for _ in range(warmup):
        one_step()
    rows = np.array([one_step() for _ in range(iters)])       # [N, 3]
    tot = rows.sum(axis=1)
    return {
        "path": os.path.basename(ts_path), "raw_obs_dim": bundle.raw_obs_dim,
        "e2e": bundle.end_to_end, "device": device, "iters": iters,
        "obs_ms": float(np.median(rows[:, 0])),
        "fwd_ms": float(np.median(rows[:, 1])),
        "dec_ms": float(np.median(rows[:, 2])),
        "p50": float(np.median(tot)), "p90": float(np.percentile(tot, 90)),
        "p99": float(np.percentile(tot, 99)), "max": float(tot.max()),
    }


# ─────────────────────────── 報表 ───────────────────────────

def report_bag(bag_dir: str) -> None:
    data = read_bag(bag_dir)
    print(f"\n=== bag: {bag_dir} ===")
    print("\n【topic 取樣週期】(p50 = 中位間隔)")
    for tp in ["/rover_rl/lidar_sweep_72", "/rover_rl/cmd_vel_desired",
               "/input/nav_cmd_vel", "/output/cmd_vel", "/odom"]:
        if tp in data:
            r = rate_stats(data[tp])
            print(f"  {tp:34s} {r['hz']:5.1f} Hz  p50 {r['p50_ms']:6.1f} ms  "
                  f"p95 {r['p95_ms']:6.1f} ms  (n={r['n']})")

    print("\n【S4 cmd 鏈路各 hop 延遲】(值配對；pass=數值原樣通過比例)")
    for up, dn, who in CHAIN:
        if up not in data or dn not in data:
            continue
        r = hop_latency(data[up], data[dn])
        if r is None:
            print(f"  {who:24s} {up} → {dn}: 無可配對樣本")
            continue
        print(f"  {who:24s} p50 {r['p50']:6.1f} ms  p90 {r['p90']:6.1f} ms  "
              f"max {r['max']:7.1f} ms  n={r['n']}  pass={r['pass_rate']*100:.0f}%")

    print("\n【S5 致動死時間】(/output/cmd_vel → odom 實測，ω 通道互相關)")
    if "/output/cmd_vel" in data and "/odom" in data:
        r = dead_time(data["/output/cmd_vel"], data["/odom"])
        if r:
            print(f"  死時間 {r['lag_ms']:.0f} ms  (相關 r={r['corr']:.2f}, "
                  f"分析長度 {r['secs']:.0f}s)")
        else:
            print("  訊號太平或重疊不足 → 無法估計（車需有轉向動作）")


def report_bench(ts_path: str) -> None:
    r = bench_model(ts_path)
    print(f"\n【S3 純推論算力台架】{r['path']}  (raw_obs_dim={r['raw_obs_dim']}, "
          f"e2e={r['e2e']}, {r['iters']} 次, device={r['device']})")
    print(f"  obs build {r['obs_ms']:.2f} ms + forward {r['fwd_ms']:.2f} ms + "
          f"decode {r['dec_ms']:.2f} ms")
    print(f"  合計 p50 {r['p50']:.2f} ms   p90 {r['p90']:.2f} ms   "
          f"p99 {r['p99']:.2f} ms   max {r['max']:.2f} ms")


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端延遲預算分解（離線）")
    ap.add_argument("--bag", help="rosbag 資料夾（logs/bags/deploy_*）")
    ap.add_argument("--bench", help="模型 .ts 路徑，量純推論時間")
    args = ap.parse_args()
    if not args.bag and not args.bench:
        ap.print_help()
        return 1
    if args.bag:
        report_bag(os.path.expanduser(args.bag))
    if args.bench:
        report_bench(os.path.expanduser(args.bench))
    return 0


if __name__ == "__main__":
    sys.exit(main())
