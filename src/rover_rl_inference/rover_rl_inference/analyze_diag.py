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
    # 通用統計摘要（忽略 None）。注意 median 用 sorted[n//2] 是簡化版（偶數筆
    # 取偏上位數而非平均兩中值），對診斷判讀夠用、不需精確中位數。
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    median = sorted(xs)[n // 2]
    return {"n": n, "mean": mean, "std": math.sqrt(var),
            "min": min(xs), "max": max(xs), "median": median}


def _dominant_str(rows, key):
    """回傳某字串欄出現最多次的非空值（用於 nav_type 這種整段固定的標籤）."""
    counts: dict[str, int] = {}
    for r in rows:
        v = (r.get(key) or "").strip()
        if v:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _print_goal_speed_tracking(nav_type, true_d, perc_d, cmd_v, cmd_w,
                               far_thr=5.0, lookahead_hint=2.0) -> None:
    """重點追蹤：goal 型態 × 距離 × 速度。檢驗「遠 single goal → OOD/全速延遲極限環」假設。

    true_d  = dist_to_goal     真實剩餘距離（到終點）
    perc_d  = policy_goal_dist  policy 實際吃到的 goal 距離（clamp 18m + ×1/rate）
    依 policy 感知距離把每拍分「遠 / 近」兩段，比較 v 與 |Δω| 抖動。
    """
    print("【goal 型態 × 距離 × 速度 追蹤】(遠 single goal OOD 假設)")
    nt_desc = {"single_goal": "單一 goal (RViz 2D Goal Pose)",
               "routing_path": "路徑導航 (Publish Point → routing)"}.get(
        nav_type, nav_type or "未知")
    print(f"  goal 型態: {nt_desc}")

    tv = [d for d in true_d if d is not None]
    pv = [d for d in perc_d if d is not None]
    if tv:
        print(f"  真實距離(到終點): 起 {tv[0]:.1f}m → 終 {tv[-1]:.1f}m  max {max(tv):.1f}m")
    if pv:
        pmean = sum(pv) / len(pv)
        pmax = max(pv)
        tag = ""
        if pmax > 17.0:
            tag = " ← 觸 18m clamp（goal 超出，被截斷餵入）"
        elif nav_type == "routing_path" and pmean < lookahead_hint * 1.6:
            tag = " ← carrot 限幅在 lookahead（近、分布內）"
        print(f"  policy 感知距離: 平均 {pmean:.1f}m  max {pmax:.1f}m{tag}")
        # 感知 vs 真實 是否脫節（path carrot 的關鍵特徵 / single goal 的 OOD 風險）
        if tv and max(tv) > far_thr:
            t0 = sum(tv[:max(1, len(tv) // 10)]) / max(1, len(tv) // 10)
            p0 = sum(pv[:max(1, len(pv) // 10)]) / max(1, len(pv) // 10)
            if t0 - p0 > far_thr:
                print(f"    脫節: 開頭真實 {t0:.1f}m 但 policy 只看到 {p0:.1f}m"
                      "（carrot 把遠 goal 切近 → 這正是 path 平順的原因）")
            elif p0 > far_thr:
                print(f"    直給: policy 開頭就吃到 {p0:.1f}m 的遠 goal（single goal OOD 風險）")

    # 依 policy 感知距離分「遠/近」兩段，比較速度與角速度抖動
    far_v, near_v, far_jit, near_jit = [], [], [], []
    prev_w = None
    for i, d in enumerate(perc_d):
        v = cmd_v[i] if i < len(cmd_v) else None
        w = cmd_w[i] if i < len(cmd_w) else None
        if d is not None and v is not None:
            (far_v if d > far_thr else near_v).append(v)
        if d is not None and w is not None and prev_w is not None:
            (far_jit if d > far_thr else near_jit).append(abs(w - prev_w))
        if w is not None:
            prev_w = w
    _m = lambda xs: (sum(xs) / len(xs)) if xs else None
    fv, nv, fj, nj = _m(far_v), _m(near_v), _m(far_jit), _m(near_jit)
    print(f"  以 policy 感知 {far_thr:.0f}m 為界分段:")
    if fv is not None:
        print(f"    遠(>{far_thr:.0f}m): v={fv:.2f} m/s  |Δω|={fj:.3f} rad/s/拍  (n={len(far_v)})")
    if nv is not None:
        print(f"    近(≤{far_thr:.0f}m): v={nv:.2f} m/s  |Δω|={nj:.3f} rad/s/拍  (n={len(near_v)})")
    if fj is not None and nj is not None:
        if fj > nj * 1.5 and fj > 0.08:
            print("    → 遠距抖動明顯較大 = 與『遠 goal→全速→延遲極限環/OOD』一致 ✓")
        else:
            print("    → 遠近抖動相近，遠 goal 未顯著惡化（此段證據不足，需更遠 goal 對比）")
    elif fj is None:
        print("    → 本段 policy 全程在近距（無遠段可比，多半是 path carrot 或近 goal）")
    print("-" * 60)


def _step_jitter(xs, wrap_deg=False):
    """相鄰兩筆的變化量統計（抖動指標）。wrap_deg=True 時把角度差包回 (-180,180]，
    避免 ±360° 假跳。回傳 _stats(|Δ|) 或 None。"""
    ds = []
    prev = None
    for x in xs:
        if x is None:
            continue
        if prev is not None:
            d = x - prev
            if wrap_deg:
                d = (d + 180.0) % 360.0 - 180.0
            ds.append(abs(d))
        prev = x
    return _stats(ds)


def _print_jump_attribution(pol_ang, ndt_yaw, map_yaw) -> None:
    """goal 方向角抖動 + 跳動歸因：policy 追的 goal 角抖不抖，是 NDT 還是車姿在跳。"""
    print("【goal 方向角抖動 / NDT·車姿跳動歸因】")
    pj = _step_jitter(pol_ang, wrap_deg=True)
    if pj:
        v = ("平順" if pj["mean"] < 1.0 else
             "中等" if pj["mean"] < 3.0 else "明顯抖")
        print(f"  policy_goal_ang 每拍|Δ| = 平均 {pj['mean']:.2f}° / max {pj['max']:.1f}°  ({v})")
        print("    判讀: >3°/拍 = policy 看到的 goal 方向一直跳 → 會驅動 ω 來回修 = sin 波源")
    # 把抖動往上游拆：NDT(map→odom) yaw 跳？合成車姿 yaw 跳？
    nyawj = _step_jitter(ndt_yaw, wrap_deg=True)
    myawj = _step_jitter(map_yaw, wrap_deg=True)
    if nyawj:
        print(f"  NDT(map→odom) yaw 每拍|Δ| = 平均 {nyawj['mean']:.2f}° / max {nyawj['max']:.1f}°"
              "   (>2° = NDT 重定位在跳)")
    if myawj:
        print(f"  合成車姿 map yaw 每拍|Δ| = 平均 {myawj['mean']:.2f}° / max {myawj['max']:.1f}°"
              "   (車姿朝向跳 → goal body 角直接跟著跳)")
    if pj and (nyawj or myawj):
        upstream = max((nyawj or {}).get("mean", 0), (myawj or {}).get("mean", 0))
        if pj["mean"] > 3.0 and upstream > 2.0:
            print("    → goal 角抖動與 NDT/車姿跳動同步 = 你的假設成立：定位跳 → goal 方向跳 → 晃")
        elif pj["mean"] > 3.0:
            print("    → goal 角抖但 NDT/車姿穩 → 抖動另有源（policy 內部/obs），非定位跳")
    print("-" * 60)


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
    model = meta.get("model") or p.get("model_path")
    if model:
        print(f"【本次使用 model】{model}")
    nav_desc = {
        "single_goal": "單一 goal（RViz 2D Goal Pose 單點）",
        "routing_path": "路徑導航（RViz Publish Point 兩點 → routing）",
    }.get(meta.get("goal_method"))
    if nav_desc:
        print(f"【goal 決定方式】{nav_desc}")
    print(f"【當時 policy 參數】(來自 {os.path.basename(side)})")
    for k, v in shown:
        print(f"  {k:24s}= {v}")
    extra = [k for k in p if k not in keys and k != "use_sim_time"]
    if extra:
        print(f"  (另有 {len(extra)} 項：{', '.join(extra[:8])}{' …' if len(extra) > 8 else ''})")


def _xcorr_lag(t, sent, act, max_lag_s=1.0, min_std=0.04):
    """離線互相關估 sent→act 延遲。回傳 (lag_s, corr) 或 (None, None)。

    原理：把「送出 cmd」往後平移 k 格再和「實測」逐點相乘求平均（即正規化
    互相關），讓兩條訊號最對齊的位移量 k×dt 就是底盤的反應延遲。先各自減均值
    除標準差做 z-score 正規化，使相關值落在 [-1,1]、與訊號振幅無關。

    若訊號太平穩（std < min_std，多半是站著沒動）相關沒有意義，回 None 要求
    使用者在移動中重錄。
    """
    pairs = [(s, a) for s, a in zip(sent, act) if s is not None and a is not None]
    if len(pairs) < 30:
        return None, None
    import statistics
    s = [p[0] for p in pairs]
    a = [p[1] for p in pairs]
    ts = [x for x in t if x is not None]
    # 由時間欄推平均取樣間隔 dt，把「平移幾格」換算回秒；缺時間欄退回 0.05s(20Hz)。
    dt = (ts[-1] - ts[0]) / max(len(ts) - 1, 1) if len(ts) >= 2 else 0.05
    sd_s = statistics.pstdev(s)
    sd_a = statistics.pstdev(a)
    if sd_s < min_std or sd_a < min_std:
        return None, None
    ms, ma = statistics.fmean(s), statistics.fmean(a)
    sn = [(x - ms) / sd_s for x in s]
    an = [(x - ma) / sd_a for x in a]
    n = len(sn)
    # 只搜尋 0 ~ max_lag_s 的正向延遲（實測不可能領先送出）；逐個 k 算相關，
    # 取相關最高者為估計延遲。n-k<=5 時樣本太少不可信，提早結束。
    max_k = max(int(max_lag_s / dt), 1)
    best_k, best_c = 0, -2.0
    for k in range(0, max_k + 1):
        if n - k <= 5:
            break
        c = sum(sn[i] * an[i + k] for i in range(n - k)) / (n - k)
        if c > best_c:
            best_c, best_k = c, k
    return best_k * dt, best_c


def _print_speed_tracking(rl_w, sent_w, act_w, rl_v, sent_v, act_v) -> None:
    # 三層拆解定位問題出在哪：RL想要→送出 之間的落差是濾波器(low-pass/slew)造成，
    # 送出→實測 之間的落差是底盤端(飽和/deadband/延遲)造成。
    print("【速度三層對比】RL想要 → 送出底盤 → 底盤實測")

    def _avg_abs(xs):
        v = [abs(x) for x in xs if x is not None]
        return sum(v) / len(v) if v else None

    for name, rl, sent, act, unit in (
        ("ω", rl_w, sent_w, act_w, "rad/s"),
        ("v", rl_v, sent_v, act_v, "m/s"),
    ):
        a_rl, a_sent, a_act = _avg_abs(rl), _avg_abs(sent), _avg_abs(act)
        if a_sent is None or a_act is None:
            continue
        # 濾波損失：RL想要 vs 送出；底盤跟隨：送出 vs 實測
        track = a_act / a_sent if a_sent > 1e-3 else float("nan")
        line = (f"  |{name}| 想要={_o(a_rl)} 送出={a_sent:.3f} 實測={a_act:.3f} {unit}"
                f"   底盤跟隨率={track:.0%}")
        print(line)
        if a_sent > 0.1 and track < 0.6:
            print(f"    ⚠ 底盤實測僅達送出 {track:.0%} → 飽和/deadband/延遲（見 CLAUDE.md gap #2/#6）")


def _print_latency(t, sent_w, act_w, sent_v, act_v, lag_ms, lag_corr) -> None:
    print("【延遲（送出 cmd → 底盤實測）】")
    # 1) 離線整段互相關（較穩）
    best = None
    for ch, sent, act in (("角速ω", sent_w, act_w), ("線速v", sent_v, act_v)):
        lag_s, corr = _xcorr_lag(t, sent, act)
        if corr is None:
            continue
        if best is None or corr > best[2]:
            best = (ch, lag_s, corr)
    if best is None:
        print("  訊號太平穩（多在原地）→ 無法估延遲；請在移動中重錄一段")
    else:
        ch, lag_s, corr = best
        ms = lag_s * 1000.0
        verdict = ("✓ 可忽略" if ms <= 100 else
                   "△ 注意（>200ms 易振盪）" if ms <= 300 else
                   "⚠ 偏大，建議加大 cmd_alpha_linear 濾波或評估 cmd_delay 補償")
        print(f"  整段互相關估計 = {ms:.0f} ms  (相關 {corr:.2f}, {ch}通道)  {verdict}")
    # 2) 即時 lag_ms 欄位摘要（policy_node 滑窗估計）
    live = [x for x in lag_ms if x is not None]
    if live:
        st = _stats(live)
        print(f"  即時估計 lag_ms: 平均 {st['mean']:.0f} / 中位 {st['median']:.0f} / "
              f"max {st['max']:.0f} ms（共 {len(live)} 筆）")


def _o(x):
    return f"{x:.3f}" if x is not None else "—"


def _print_vo_approach(t, dist, n_obs, des_v, out_v, blocked, min_ttc) -> None:
    """VO 逼近減速分析：抓「障礙進入 engage 範圍」(n_obs 0→≥1) 的時刻，印當下
    des_v→out_v、後續減速斜率，並驗 out_v 有沒有超過 des_v（speedup bug 回歸測試）。"""
    print("【VO 逼近減速】障礙進範圍→VO 怎麼減速")
    na = [x for x in n_obs if x is not None]
    if not na or max(na) < 1:
        print("  本段 VO 從未偵測到 engage_range 內障礙（vo_n_obs 全 0）→ 無逼近事件")
        print("    （人沒進 engage_range / 本段沒走 VO / 無動態障礙；首測那種「直直衝過來」屬此類）")
        print("-" * 60)
        return

    # 安全層回歸測試：介入(n_obs≥1)期間 out_v 是否曾超過 des_v（修好後應 ≤0）
    over = [out_v[i] - des_v[i] for i in range(len(n_obs))
            if (n_obs[i] or 0) >= 1 and out_v[i] is not None and des_v[i] is not None]
    if over:
        mx = max(over)
        if mx > 0.05:
            print(f"  ⚠ 介入時 out_v 一度超過 des_v 達 +{mx:.2f} m/s → 安全層在「加速」！"
                  "(v 鉗位失效，檢查 vo_layer v_hi=min(v_hi,des_v))")
        else:
            print(f"  ✓ 介入時 out_v 始終 ≤ des_v（最大超出 {mx:+.2f} m/s）→ 安全層只減速不加速")

    # 找 n_obs 0→≥1 的上升緣 = 逼近事件起點
    episodes = []
    prev = 0
    for i, x in enumerate(n_obs):
        cur = int(x) if x is not None else 0
        if prev == 0 and cur >= 1:
            episodes.append(i)
        prev = cur
    print(f"  偵測到 {len(episodes)} 次「障礙進入 engage 範圍」事件：")

    for k, i0 in enumerate(episodes[:4]):     # 最多印 4 段
        t0, d0, dv0, ov0, ttc0 = t[i0], dist[i0], des_v[i0], out_v[i0], min_ttc[i0]
        # 往後掃到 n_obs 退回 0 = 本事件區間；記錄最低 out_v 與其時刻
        j = i0
        ov_min, j_min = (ov0 if ov0 is not None else float("inf")), i0
        while j < len(n_obs) and (n_obs[j] or 0) >= 1:
            if out_v[j] is not None and out_v[j] < ov_min:
                ov_min, j_min = out_v[j], j
            j += 1
        head = f"  #{k + 1} t={_o(t0)}s"
        if d0 is not None:
            head += f"  (距goal {d0:.2f}m)"
        if ttc0 is not None:
            head += f"  進範圍時 min_ttc={ttc0:.2f}s"
        print(head)
        print(f"      進範圍當下: des_v={_o(dv0)} → out_v={_o(ov0)} m/s")
        dt = (t[j_min] - t0) if (t[j_min] is not None and t0 is not None) else None
        if dt is not None and dt > 1e-3 and ov0 is not None and ov_min < ov0:
            rate = (ov0 - ov_min) / dt
            kind = ("逐步平順減速" if dt >= 0.5 else
                    "偏急" if dt >= 0.2 else "⚠ 懸崖式急煞(最後一秒才反應)")
            print(f"      減速到最低 out_v={ov_min:.2f} 歷時 {dt:.2f}s → 平均 {rate:.2f} m/s²  ({kind})")
        elif ov0 is not None and ov_min >= ov0 - 1e-3:
            print("      進範圍後 out_v 未下降（VO 判定可直接通過/未介入減速）")
        # 區間內是否堵死(煞停)
        bl = [blocked[x] for x in range(i0, j) if blocked[x] is not None]
        if bl and any(b > 0.5 for b in bl):
            print(f"      期間堵死(blocked) {sum(1 for b in bl if b > 0.5)}/{len(bl)} 拍 → 一度全候選不可行=煞停")
    print("    判讀: 斜率小+歷時長=逐步減速(好)；懸崖式=engage太晚或horizon太短，"
          "考慮調大 engage_range/horizon 或補管線延遲")
    print("-" * 60)


def _print_vo_engage_calib(dyn_min, vo_active, vo_interv, vo_blocked, engage_vals) -> None:
    """VO 介入距離校準：把每拍按「最近動態障礙中心距離」分桶，統計各距離下
    誰在動手（RL / VO 繞行 / VO 煞停）→ 直接看出 VO 實際在幾公尺開始接管，好調 engage_range。

    dyn_min      = dyn_obs_min_m   最近動態障礙的車中心↔障礙中心距離（與 engage_range 同基準）
    engage_vals  = vo_engage_range 當下 VO 設定值（整段常數）
    """
    print("【VO 介入距離校準】最近動態障礙距離(中心↔中心) × 誰在動手")
    ev = [x for x in engage_vals if x is not None]
    er = ev[0] if ev else None

    def _driver(i):
        act = vo_active[i]
        if act is None or act < 0.5:
            return "rl"                       # VO 沒在線 → RL 直出
        if vo_blocked[i] and vo_blocked[i] > 0.5:
            return "stop"
        if vo_interv[i] and vo_interv[i] > 0.5:
            return "vo"
        return "rl"                           # VO 在線但只放行 = RL 主導

    pts = [(d, _driver(i)) for i, d in enumerate(dyn_min) if d is not None]
    if not pts:
        print("  本段無動態障礙距離資料（vo_interface 未起 / 場上無人）→ 無法校準")
        print("    需 lv-dot 在跑（發 /vo_interface/tracked_obstacles）才有動態障礙距離")
        print("-" * 60)
        return
    if er is not None:
        print(f"  目前設定 engage_range = {er:.2f} m（vo_params.yaml；VO 只在此距離內跑）")

    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
            (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]
    labels = ["<0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–5", ">5"]
    print(f"    {'距離(m)':>7} {'列數':>6} {'VO繞行':>7} {'VO煞停':>7} {'RL主導':>7}")
    for (lo, hi), lab in zip(bins, labels):
        sub = [drv for d, drv in pts if lo <= d < hi]
        if not sub:
            continue
        n = len(sub)
        vo = sum(1 for x in sub if x == "vo") / n
        stp = sum(1 for x in sub if x == "stop") / n
        rl = sum(1 for x in sub if x == "rl") / n
        print(f"    {lab:>7} {n:>6} {vo:>6.0%} {stp:>6.0%} {rl:>6.0%}")

    vo_ds = [d for d, drv in pts if drv in ("vo", "stop")]
    if vo_ds:
        print(f"  → 實測 VO 動手(繞行/煞停)最遠距離 = {max(vo_ds):.2f} m")
        if er is not None:
            print(f"    （engage_range={er:.2f}m 是理論上限；實測最遠 {max(vo_ds):.2f}m"
                  " 是障礙真的進到那個距離才動）")
        print("    調法: 想讓 VO 更晚、更貼近車頭才動 → 調小 engage_range（或連 margin 一起）；")
        print("          想更早反應 → 調大 engage_range。挑上表『VO 動手率開始明顯上升』的距離當 engage_range。")
    else:
        print("  → 本段 VO 全程沒動手（各距離都 RL 主導）")
        print("    可能：人從沒進 engage_range / 這段沒走 VO / 人不在前進軌跡上（VO 靠 rollout 只擋擋路的）")
    print("-" * 60)


def analyze(path: str) -> None:
    rows = _load(path)
    if not rows:
        print(f"[!] {path} 沒有資料列")
        return

    # 以下逐欄抽出 diag_logger 寫入的時間序列（缺值為 None）：
    # t_rel=相對時間秒；cmd_*=最終 cmd_vel；odom_w=實測角速；
    # heading_err_deg=NDT 真實朝向誤差；dist_to_goal=與目標距離；
    # policy_goal_ang_deg=policy 自己算的目標方位（拿來和 heading 比對檢查 TF）。
    t = _col(rows, "t_rel")
    cmd_w = _col(rows, "cmd_w")
    cmd_v = _col(rows, "cmd_v")
    odom_w = _col(rows, "odom_w")
    heading = _col(rows, "heading_err_deg")
    dist = _col(rows, "dist_to_goal")
    pol_ang = _col(rows, "policy_goal_ang_deg")
    perc_d = _col(rows, "policy_goal_dist")   # policy 感知的 goal 距離（clamp 18m + ×1/rate）
    nav_type = _dominant_str(rows, "nav_type")
    # 三層速度 + 延遲
    rl_w = _col(rows, "rl_w")
    sent_w = _col(rows, "sent_w")
    act_w = _col(rows, "act_w")
    rl_v = _col(rows, "rl_v")
    sent_v = _col(rows, "sent_v")
    act_v = _col(rows, "act_v")
    lag_ms = _col(rows, "lag_ms")
    lag_corr = _col(rows, "lag_corr")
    # VO 安全層：本段是否走 VO（vo_active）+ 各拍有沒有介入/堵死
    vo_active = _col(rows, "vo_active")
    vo_interv = _col(rows, "vo_intervening")
    vo_blocked = _col(rows, "vo_blocked")
    # VO 逼近量化欄：障礙進範圍數 / RL 期望 vs VO 輸出線速度 / 最近碰撞時間
    vo_n_obs = _col(rows, "vo_n_obs")
    vo_des_v = _col(rows, "vo_des_v")
    vo_out_v = _col(rows, "vo_out_v")
    vo_min_ttc = _col(rows, "vo_min_ttc")
    # VO 介入距離校準：最近動態障礙中心距離（與 engage_range 同基準）+ 當下設定值
    dyn_obs_min = _col(rows, "dyn_obs_min_m")
    vo_engage_range = _col(rows, "vo_engage_range")
    # goal 方向角抖動歸因：NDT(map→odom) yaw 與合成車姿 yaw 的每拍跳動
    ndt_yaw = _col(rows, "ndt_yaw_deg")
    map_yaw = _col(rows, "map_yaw_deg")

    # Δω（相鄰列）：相鄰兩筆角速度的差分，std 越大代表 cmd_w 抖動越劇烈＝晃動。
    # 用差分而非 ω 本身，是因為穩定大轉彎的 ω 也很大但不算「晃」。
    dws = [b - a for a, b in zip(cmd_w, cmd_w[1:])
           if a is not None and b is not None]
    dw_st = _stats(dws)
    cw_st = _stats([abs(x) for x in cmd_w if x is not None])
    hd_st = _stats([abs(x) for x in heading if x is not None])
    di_st = _stats(dist)
    pol_st = _stats([abs(x) for x in pol_ang if x is not None])

    # heading 與 policy_goal_ang 一致性（檢查定位/TF）：兩者理應接近，差很大
    # 代表 policy 看到的目標方向和 NDT 真實方向不一致 → TF/座標轉換有問題。
    diff = [abs(h - p) for h, p in zip(heading, pol_ang)
            if h is not None and p is not None]
    diff_st = _stats(diff)

    # 朝 goal 前進？比較頭 20% 與尾 20% 的平均距離，看整體是靠近還是變遠，
    # 取區段平均而非單點首尾以抑制定位噪聲。
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
    # VO 在不在線 → 判斷這段晃動「是不是 VO 造成」的關鍵脈絡
    va = [x for x in vo_active if x is not None]
    if va:
        on = sum(1 for x in va if x > 0.5)
        frac = on / len(va)
        if frac < 0.05:
            print("  VO: 本段未走 VO（vo_active≈0）→ cmd 即 RL 直出，晃動屬 policy/底盤")
        else:
            iv = [x for x in vo_interv if x is not None]
            bl = [x for x in vo_blocked if x is not None]
            iv_f = (sum(1 for x in iv if x > 0.5) / len(iv)) if iv else 0.0
            bl_f = (sum(1 for x in bl if x > 0.5) / len(bl)) if bl else 0.0
            print(f"  VO: 在線 {frac:.0%} 的列；其中介入(改寫cmd) {iv_f:.0%}、堵死停車 {bl_f:.0%}")
            print("    判讀: 晃動時段 vo_intervening≈0 → 非 VO 繞行造成（VO 僅放行+slew），"
                  "病根在 policy/底盤；intervening 高才是 VO 在左右搖")
    print("-" * 60)
    _print_goal_speed_tracking(nav_type, dist, perc_d, cmd_v, cmd_w)
    _print_jump_attribution(pol_ang, ndt_yaw, map_yaw)
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
    print("-" * 60)
    _print_speed_tracking(rl_w, sent_w, act_w, rl_v, sent_v, act_v)
    print("-" * 60)
    _print_vo_approach(t, dist, vo_n_obs, vo_des_v, vo_out_v, vo_blocked, vo_min_ttc)
    _print_vo_engage_calib(dyn_obs_min, vo_active, vo_interv, vo_blocked, vo_engage_range)
    _print_latency(t, sent_w, act_w, sent_v, act_v, lag_ms, lag_corr)
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
