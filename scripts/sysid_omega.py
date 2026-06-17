#!/usr/bin/env python3
"""ω 通道系統辨識 — 從 diag CSV 量底盤 cmd→實測 的 gain / 遲滯(死時間+時間常數) / 死區.

用途：判斷「車頭擺動」是延遲驅動還是訓練問題，並為 Smith-predictor 式延遲補償
提供真實的 cmd→response 模型參數（取代裸積分 measured/commanded 速度）。

輸入  u = cmd_w   （policy 實際送底盤的角速度命令, rad/s）
輸出  y = odom_w  （odom 量到的實際角速度, rad/s）
模型  一階遲滯 + 純死時間 + 死區：
        y[k] = a*y[k-1] + b*u[k-d]
        靜態增益 K = b/(1-a)，時間常數 τ = -dt/ln(a)，死時間 = d*dt
用法  python3 sysid_omega.py <diag.csv> [--in cmd_w] [--out odom_w]
"""
from __future__ import annotations
import argparse, csv, math
import numpy as np


def load(path, kin, kout):
    rows = list(csv.DictReader(open(path)))
    def fn(r, k):
        try: return float(r.get(k, ""))
        except: return None
    t, u, y = [], [], []
    for r in rows:
        if r.get("has_goal") not in ("1", "True", "true"):
            continue
        tt, uu, yy = fn(r, "t_rel"), fn(r, kin), fn(r, kout)
        if None in (tt, uu, yy):
            continue
        t.append(tt); u.append(uu); y.append(yy)
    return np.array(t), np.array(u), np.array(y)


def resample(t, u, y, dt):
    """線性內插到等間隔 dt 網格（互相關需等間隔）."""
    grid = np.arange(t[0], t[-1], dt)
    return grid, np.interp(grid, t, u), np.interp(grid, t, y)


def deadtime_xcorr(u, y, dt, max_lag_s=0.6):
    """正規化互相關估死時間：找讓 corr(u[k-d], y[k]) 最大的 d."""
    u0, y0 = u - u.mean(), y - y.mean()
    n = int(max_lag_s / dt)
    best_d, best_r = 0, -2.0
    rs = []
    for d in range(0, n + 1):
        if d >= len(u0):
            break
        a, b = u0[: len(u0) - d], y0[d:]
        denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        r = float(np.dot(a, b)) / denom if denom > 1e-9 else 0.0
        rs.append((d * dt, r))
        if r > best_r:
            best_r, best_d = r, d
    return best_d, best_r, rs


def fit_first_order(u, y, d):
    """最小二乘擬合 y[k] = a*y[k-1] + b*u[k-d]."""
    k0 = max(1, d)
    Y = y[k0:]
    X = np.column_stack([y[k0 - 1 : len(y) - 1], u[k0 - d : len(u) - d]])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = X @ coef
    ss_res = float(np.sum((Y - yhat) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return a, b, r2


def deadband_gain(u, y, d, nbins=12):
    """死區 + 大訊號增益：把 |u| 分箱，量每箱平均 |y|（已對齊死時間）.
    再對「明顯有回應」的箱做線性擬合 |y| = g*(|u|-db)，回 db, g."""
    ud = u[: len(u) - d] if d > 0 else u
    yd = y[d:] if d > 0 else y
    au, ay = np.abs(ud), np.abs(yd)
    umax = au.max()
    edges = np.linspace(0, umax, nbins + 1)
    table = []
    for i in range(nbins):
        m = (au >= edges[i]) & (au < edges[i + 1])
        if m.sum() >= 5:
            table.append((0.5 * (edges[i] + edges[i + 1]),
                          float(ay[m].mean()), int(m.sum())))
    # 線性擬合：取回應 > 0.02 的箱，y = g*u + c → db = -c/g
    pts = [(uc, ym) for uc, ym, n in table if ym > 0.02]
    db = g = float("nan")
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        g, c = np.polyfit(xs, ys, 1)
        db = -c / g if g > 1e-6 else float("nan")
    return table, db, float(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--in", dest="kin", default="cmd_w")
    ap.add_argument("--out", dest="kout", default="odom_w")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    t, u, y = load(a.csv, a.kin, a.kout)
    print(f"檔案: {a.csv}")
    print(f"輸入={a.kin}  輸出={a.kout}  has_goal樣本={len(t)}  時長={t[-1]-t[0]:.0f}s")
    g, ug, yg = resample(t, u, y, a.dt)
    print(f"重取樣到 {a.dt*1000:.0f}ms 網格: {len(g)} 點")
    print(f"|u|平均={np.abs(ug).mean():.3f}  |y|平均={np.abs(yg).mean():.3f}  "
          f"裸跟隨率={np.abs(yg).mean()/max(1e-6,np.abs(ug).mean())*100:.0f}%")

    d, r, rs = deadtime_xcorr(ug, yg, a.dt)
    print(f"\n【死時間 (cross-corr)】 = {d*a.dt*1000:.0f} ms ({d} 步)  峰值相關 r={r:.2f}")
    print("  lag(ms): r  → " + "  ".join(f"{int(L*1000)}:{rr:.2f}" for L, rr in rs[:10]))

    aa, bb, r2 = fit_first_order(ug, yg, d)
    K = bb / (1 - aa) if abs(1 - aa) > 1e-6 else float("nan")
    tau = -a.dt / math.log(aa) if 0 < aa < 1 else float("nan")
    print(f"\n【一階遲滯擬合】 y[k]={aa:.3f}·y[k-1]+{bb:.3f}·u[k-{d}]   R²={r2:.2f}")
    print(f"  靜態增益 K={K:.2f}   時間常數 τ={tau*1000:.0f} ms   死時間={d*a.dt*1000:.0f} ms")

    table, db, gain = deadband_gain(ug, yg, d)
    print(f"\n【死區 / 大訊號增益】 |u|分箱 → 平均|y|:")
    for uc, ym, n in table:
        bar = "█" * int(ym / max(1e-6, max(tt[1] for tt in table)) * 30)
        print(f"  |cmd_w|≈{uc:.2f}  →  |odom_w|≈{ym:.3f}  (n={n:4d}) {bar}")
    print(f"  線性擬合死區 deadband≈{db:.3f} rad/s   大訊號增益 g≈{gain:.2f}")

    print("\n【判讀】")
    print(f"  • 跟隨率/增益 K≈{K:.2f} → 底盤只做到命令的 ~{K*100:.0f}%（其餘被死區+飽和+遲滯吃掉）")
    print(f"  • 死時間 {d*a.dt*1000:.0f}ms ≈ {d*a.dt/0.2:.1f} 個控制步(0.2s) → 延遲確實存在")
    print(f"  • Smith-predictor 該用此模型(K,τ,死時間,deadband)預測，而非裸積分速度")


if __name__ == "__main__":
    main()
