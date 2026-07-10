"""Velocity Obstacle 安全層 — 純演算法（無 ROS 依賴，可單元測試）.

夾在 RL policy 與底盤之間：RL 給「期望速度」，本層用 LV-DOT 的動態障礙
(position + velocity) 做短期碰撞預測，輸出「最接近 RL 意圖、且 horizon 內不撞」
的安全速度。RL 不知道被改寫，但下一拍會從 odom 速度回授看到實際執行值（閉環）。

為何不用教科書 VO？
  差速底盤不能橫移，**瞬時平移速度方向永遠沿 heading**，只有大小 v 可變。
  純瞬時 VO 因此只會「叫你減速」無法「叫你繞」（繞要靠 ω 隨時間累積轉向）。
  故改採 **dynamic-window 取樣 (v, ω) + 軌跡 rollout 對動態障礙做預測碰撞檢查**，
  等價於把 Velocity Obstacle 原理套用在非全向載具（DWA-style，每 20Hz 重規劃，
  小步累積即可平滑繞行）。

每個 tick 流程：
  1. RL 期望 (v_des, ω_des) → 先 clamp 進輸出上限（ω 順便壓到底盤真實 1.2，解 gap #2）。
  2. 以 odom 實測 (v0, ω0) 為中心，按加速度上限 × 控制週期取 dynamic window（保證可達+平滑）。
  3. 視窗內撒候選 (v, ω)，並把 clamp 後的 desired 也塞進候選（可行時零偏差直接放行）。
  4. 每個候選以等速弧線 rollout（horizon T），障礙線性外推 p_i + v_i·t；
     任一時刻距離 < r_robot + r_obs + margin → 該候選在 horizon 內碰撞 = 不可行。
  5. 可行候選中選 cost 最小者：cost = w_v·Δv² + w_ω·Δω²（貼近 RL）+ w_goal·(d_end−d_now)
     （朝 goal 推進獎勵）。進展項讓「靜止障礙在前」時主動繞過去往 goal 走，而非慢爬卡在障礙前。
  6. 全部不可行 → fallback：clearance_fallback 開→慢爬向碰撞最晚方向；關→停車 (0,0)。

座標系：robot 與 obstacle 一律在 **odom frame**（LV-DOT localization_mode:1 用 odom）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VOParams:
    # --- 輸出硬上限（最終 cmd 不超過這些；對齊底盤真實能力）---
    v_max: float = 1.0          # m/s — RL 訓練上限
    v_min: float = -1.0         # m/s — 允許倒車（fallback/閃避用）
    w_max: float = 1.2          # rad/s — ★底盤真實上限，順便壓掉 RL 可能叫出的 2.0
    # --- dynamic window：以實測速度為中心、加速度上限決定可達範圍 ---
    accel_v: float = 1.2        # m/s²  — 底盤 acc_max
    accel_w: float = 3.0        # rad/s²
    ctrl_dt: float = 0.05       # s — 控制週期(20Hz)，輸出 slew 限速用
    # window_time 與 ctrl_dt 解耦：候選視窗取「window_time 內可達」的較大範圍，
    # 讓 VO 能選到大幅轉向的繞行解（否則單週期窗太窄只會退化成減速/停車）。
    # 平滑性改由輸出端 slew（accel × ctrl_dt）保證，不靠縮窄候選窗。
    window_time: float = 1.0    # s
    # --- 取樣解析度（候選數 = n_v × n_w）---
    n_v: int = 7
    n_w: int = 15
    # --- 軌跡 rollout（碰撞預測）---
    horizon: float = 2.0        # s — 往前預測多久
    sim_dt: float = 0.1         # s — rollout 步長
    # --- 安全幾何 ---
    r_robot: float = 0.35       # m — 與訓練/底盤一致
    margin: float = 0.10        # m — 額外安全餘裕
    # --- 成本權重（越大越貼近 RL 該通道；線速度通常較重要）---
    w_v: float = 1.0
    w_w: float = 0.3
    # --- goal 進展獎勵（DWA 式）：可行弧線 rollout「終點離 goal 的距離」獎勵權重 ---
    # 0.0=關（純貼近 RL，遇障只會慢/停在障礙前）；>0=偏好「rollout 終點更接近 goal」的弧線
    #   → 靜止障礙在前時主動往側邊繞過去再朝 goal 推進，而非慢爬到障礙前卡住。
    # ⚠ 舊版用「終點朝向 vs goal 方位」的夾角誤差，會懲罰必要的閃避轉向(轉開=暫時背離 goal)→
    #   反而卡死不繞，已改為「終點離 goal 距離」進展獎勵（cost += w_goal·(d_end−d_now)）。
    # 為何能分流「靜止→繞 / 對衝→停」：移動逼近的人會走進側弧 → 側弧 rollout 撞 → 不可行 →
    #   只剩停/倒退（正確讓行）；靜止的人側弧不撞 → 進展獎勵讓繞行弧勝出。分流靠可行性，非權重。
    w_goal: float = 0.0
    # --- 繞行側 commit（解「往右退→又往左轉」左右橫跳）---
    # 一旦選定一側繞行(commit_side=±1)，對「弧線終點偏到 commit 反側」的候選加罰，壓過 w_w/w_goal
    # 往另一側拉的力 → 選定一側走到底。0=關。commit_side 由 node 在 escape 鎖定、過障礙後過期。
    w_commit: float = 0.0
    # --- 「鼻子已對準」判據：有 v>0 且 |w|≤此門檻 的可行候選 = 接近直行就能前進 = 鼻子對準開闊側 ---
    # 卡死逃脫用：一直倒車/原地轉到 straight_feasible 才退出，確保「對準才前進」（仿真人 K 轉）。
    w_aim_thresh: float = 0.3   # rad/s
    # --- 啟動條件：障礙進入此距離才跑 VO（省運算、避免遠處誤煞）---
    engage_range: float = 6.0   # m
    # --- clearance 退化（解「全候選不可行就急停、不繞」）---
    # False（預設）=舊行為：無可行候選 → hard-stop (0,0)。
    # True = 無可行候選時不急停，改選「碰撞時刻最晚」(=最往縫隙挪) 的候選，
    #        速度封到 v_creep 慢爬、保留其轉向 → 逐拍(20Hz)往空檔擠，自然湧現繞行。
    #        仍回報 blocked=True（status 照顯示「全堵死」），只是動作從停車改成「慢爬向縫隙」。
    clearance_fallback: bool = False
    v_creep: float = 0.3        # m/s — 退化時的線速度上限（慢爬，安全優先）
    # VO 主動繞行前進速度上限：RL 期望低(timid，rl_v≈0)時 VO 仍可前進到此速度執行繞行。
    # 0.0=純守 v_des（RL 多慢 VO 多慢→RL 縮手就繞不動，只原地轉）；0.3=RL 近乎停 VO 也能慢繞。
    # 安全：輸出仍 ≤ max(v_des, this)，RL 正常驅動(v_des≥this)時不超過 v_des（不重現 speedup 0.8）；
    # 只在 RL 縮手(v_des<this)時補到 this。且只在 engaged(有近障跑候選)時生效；沿不撞的弧前進。
    goaround_v_cap: float = 0.0


@dataclass(frozen=True)
class Obstacle:
    """動態障礙（odom frame）。r 由 LV-DOT bbox 寬度換算的等效半徑。"""
    x: float
    y: float
    vx: float
    vy: float
    r: float


@dataclass(frozen=True)
class RobotState:
    """機器人狀態（odom frame）。v/w 取 odom 實測，作 dynamic window 中心。"""
    x: float
    y: float
    yaw: float
    v: float
    w: float


@dataclass(frozen=True)
class VOResult:
    v: float                 # 輸出安全線速度
    w: float                 # 輸出安全角速度
    engaged: bool            # 是否有障礙進入 engage_range 而啟動 VO（False=直接放行 desired）
    blocked: bool            # 是否所有候選都不可行 → 已 fallback 停車
    n_obstacles: int         # engage_range 內障礙數
    n_feasible: int          # 可行候選數
    min_ttc: float           # 選定候選的最近碰撞時間（inf=horizon 內安全）
    # 「最該往哪側轉」提示（w 的符號才是重點）：有可行解→選定弧的 w；全堵死→碰撞時刻最晚
    # 候選的 w（=最有空間繞過去的方向）。供 node 卡死逃脫選邊用——與 go-around 同源（同一份
    # rollout），確保「逃脫倒車轉向」和「事後繞行」往同一側，不再打架（倒車左打卻往右前進）。
    best_turn_w: float = 0.0
    # 鼻子是否已對準：有 v>0 且 |w|≤w_aim_thresh 的可行候選（接近直行就能前進）。卡死逃脫據此
    # 決定「還要繼續轉對準」還是「對準了可前進」→ 仿真人「倒車打方向到對準才走」。
    straight_feasible: bool = False


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1 or hi <= lo:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def _rollout(v: float, w: float, robot: RobotState,
             obstacles: list[Obstacle], p: VOParams) -> tuple[float, float, float, float]:
    """以等速 (v, w) rollout 機器人弧線，回 (min_ttc, end_x, end_y, end_yaw).

    機器人：θ(t)、x(t)、y(t) 由 (v, w) 積分（odom frame）。
    障礙：線性外推 p_i + v_i·t（假設等速，短 horizon 內合理）。
    碰撞：中心距 < r_robot + r_obs + margin。
    - 無碰撞：min_ttc=inf，end_* 為 horizon 終點位姿（給 goal 導向成本用）。
    - 有碰撞：min_ttc=碰撞時刻 t，end_* 為碰撞當下位姿（候選會被淘汰，end_* 不使用）。
    """
    steps = max(1, int(round(p.horizon / p.sim_dt)))
    x, y, yaw = robot.x, robot.y, robot.yaw
    for k in range(1, steps + 1):
        # 等速弧線積分（先轉再走，midpoint 近似已足夠）
        yaw += w * p.sim_dt
        x += v * math.cos(yaw) * p.sim_dt
        y += v * math.sin(yaw) * p.sim_dt
        t = k * p.sim_dt
        for ob in obstacles:
            ox = ob.x + ob.vx * t
            oy = ob.y + ob.vy * t
            rr = p.r_robot + ob.r + p.margin
            dx = x - ox
            dy = y - oy
            if dx * dx + dy * dy <= rr * rr:
                return t, x, y, yaw  # 在 t 秒碰撞
    return math.inf, x, y, yaw


def compute_safe_cmd(v_des: float, w_des: float, robot: RobotState,
                     obstacles: list[Obstacle], p: VOParams,
                     goal: tuple[float, float] | None = None,
                     commit_side: int = 0) -> VOResult:
    """VO 主入口：給 RL 期望速度 + 障礙，回最接近且安全的 (v, w)。

    goal（odom frame, 選填）：提供時且 p.w_goal>0，成本多加「弧線終點朝向 vs
    終點→goal 方位」的夾角誤差項，讓直走被擋時 VO 偏好「繞過去又對準 goal」的弧線，
    而非只挑最貼近 RL 直走的（常是慢/停）。不提供→退回純貼近 RL 行為。
    """
    # 1) desired clamp 進上限（ω 壓到底盤真實 w_max）
    v_des = _clamp(v_des, p.v_min, p.v_max)
    w_des = _clamp(w_des, -p.w_max, p.w_max)

    # 2) 只保留 engage_range 內的障礙（遠處不影響短期決策）
    er2 = p.engage_range * p.engage_range
    near = [ob for ob in obstacles
            if (ob.x - robot.x) ** 2 + (ob.y - robot.y) ** 2 <= er2]

    if not near:
        # 無近障 → 直接放行 clamp 後 desired（VO 不介入）
        return VOResult(v_des, w_des, engaged=False, blocked=False,
                        n_obstacles=0, n_feasible=1, min_ttc=math.inf,
                        best_turn_w=w_des, straight_feasible=True)

    # 3) dynamic window（以實測速度為中心，window_time 內可達範圍 → 候選夠廣可繞行）
    v_lo = _clamp(robot.v - p.accel_v * p.window_time, p.v_min, p.v_max)
    v_hi = _clamp(robot.v + p.accel_v * p.window_time, p.v_min, p.v_max)
    # ★安全層不變快：VO 線速度上限鉗到 max(v_des, goaround_v_cap)。
    # 主鉗 v_des：RL 正常驅動時不超過 RL 期望（不重現 speedup：實測 des_v=0.5→VO 曾衝 0.8）。
    # goaround_v_cap 地板：RL 縮手(v_des≈0)且近障時，仍允許前進到此速度沿不撞弧繞行
    # （否則 v_des≈0→v_hi=0→VO 只能原地轉、繞不過去）。仍只在 engaged 候選搜尋中生效。
    v_hi = min(v_hi, max(v_des, p.goaround_v_cap))
    if v_hi < v_lo:          # RL 期望比可達下限還慢（如 RL 要停）→ 收斂到 v_des，仍允許倒退候選
        v_lo = min(v_lo, v_hi)
    w_lo = _clamp(robot.w - p.accel_w * p.window_time, -p.w_max, p.w_max)
    w_hi = _clamp(robot.w + p.accel_w * p.window_time, -p.w_max, p.w_max)

    v_cands = _linspace(v_lo, v_hi, p.n_v)
    w_cands = _linspace(w_lo, w_hi, p.n_w)
    # 把 clamp 到視窗內的 desired 也加入候選：可行時讓輸出 == desired（零偏差放行）
    v_cands.append(_clamp(v_des, v_lo, v_hi))
    w_cands.append(_clamp(w_des, w_lo, w_hi))
    # 確保「停在原地 (0,0)」一定在候選內：若連停都安全，blocked 才不會誤報
    if v_lo <= 0.0 <= v_hi:
        v_cands.append(0.0)
    if w_lo <= 0.0 <= w_hi:
        w_cands.append(0.0)

    # 4-5) 掃所有候選，挑「可行且成本最低」者
    #   成本 = w_v·Δv² + w_w·Δω²（貼近 RL）+ w_goal·(d_end−d_now)（朝 goal 推進，選填）
    use_goal = goal is not None and p.w_goal > 0.0
    d_now = math.hypot(goal[0] - robot.x, goal[1] - robot.y) if use_goal else 0.0
    # commit 側偏置：終點偏到已選定繞行側的反側時加罰（壓 w_w/w_goal 往另一側拉→不橫跳）
    use_commit = commit_side != 0 and p.w_commit > 0.0
    sin_yaw = math.sin(robot.yaw)
    cos_yaw = math.cos(robot.yaw)
    best: tuple[float, float] | None = None
    best_cost = math.inf
    best_ttc = math.inf
    n_feasible = 0
    # clearance 退化用：記「碰撞時刻最晚」的不可行候選（=最往縫隙挪的方向）。
    # 同 ttc 時偏好較慢（|v| 小）以策安全。
    inf_best: tuple[float, float] | None = None
    inf_best_ttc = -1.0
    straight_feasible = False   # 有「v>0 且接近直行」可行候選 = 鼻子已對準開闊側
    for v in v_cands:
        for w in w_cands:
            ttc, ex, ey, _eyaw = _rollout(v, w, robot, near, p)
            if ttc == math.inf:  # horizon 內不撞
                n_feasible += 1
                if v > 1e-3 and abs(w) <= p.w_aim_thresh:
                    straight_feasible = True
                cost = p.w_v * (v - v_des) ** 2 + p.w_w * (w - w_des) ** 2
                if use_goal:
                    # 進展獎勵：弧線終點離 goal 的距離（越接近 goal=越省）。
                    # d_end−d_now<0 表示有推進 → 降低成本，獎勵真的繞過去往 goal 走。
                    d_end = math.hypot(goal[0] - ex, goal[1] - ey)
                    cost += p.w_goal * (d_end - d_now)
                if use_commit:
                    # 終點相對車頭橫向位置 dy_body（>0=左/+y、<0=右/−y）；偏到 commit 反側→加罰。
                    # commit_side=+1(左)罰右(dy_body<0)；=-1(右)罰左(dy_body>0)。鎖定一側不橫跳。
                    dy_body = -sin_yaw * (ex - robot.x) + cos_yaw * (ey - robot.y)
                    cost += p.w_commit * max(0.0, -commit_side * dy_body)
                if cost < best_cost:
                    best_cost = cost
                    best = (v, w)
                    best_ttc = math.inf
            else:
                # 不可行候選：永遠留住碰撞最晚者（ttc 大 = 撐最久 = 最往空檔挪）。
                # 不再只在 clearance_fallback 時記——其 w 同時是「卡死逃脫該往哪側轉」的提示
                # （best_turn_w），讓逃脫轉向與 go-around 同源、不打架。
                if ttc > inf_best_ttc + 1e-3 or (
                        abs(ttc - inf_best_ttc) <= 1e-3
                        and inf_best is not None and abs(v) < abs(inf_best[0])):
                    inf_best_ttc = ttc
                    inf_best = (v, w)

    # 6) 全不可行 → fallback
    # best_turn_w（卡死逃脫選邊提示）= 往「最近障礙的反側」繞——這正是 go-around 可行弧天生會走
    # 的側（障礙那側的弧會撞→不可行；反側才有解）。用 body-frame 橫向位置判，貼身 penetrating 時
    # 也穩（ttc 在貼身會全部飽和失真，故不用 ttc）。正前(|y_body|小)→0，留給 commit/clearance。
    turn_hint = 0.0
    nb = min(near, key=lambda o: (o.x - robot.x) ** 2 + (o.y - robot.y) ** 2)
    y_body = -sin_yaw * (nb.x - robot.x) + cos_yaw * (nb.y - robot.y)
    if y_body > 0.05:          # 最近障礙偏左 → 往右繞（w<0）
        turn_hint = -p.w_max
    elif y_body < -0.05:       # 最近障礙偏右 → 往左繞（w>0）
        turn_hint = p.w_max
    elif inf_best is not None:  # 正前方對稱 → 退回「碰撞最晚」候選的轉向
        turn_hint = inf_best[1]
    if best is None:
        if p.clearance_fallback and inf_best is not None:
            # 不急停：慢爬向「碰撞最晚」的方向（保留轉向、線速度封到 v_creep）。
            # 逐拍 20Hz 重規劃 → 往縫隙擠出繞行。仍標 blocked（場面確實全堵）。
            fv = _clamp(inf_best[0], p.v_min, p.v_creep)
            return VOResult(fv, inf_best[1], engaged=True, blocked=True,
                            n_obstacles=len(near), n_feasible=0, min_ttc=inf_best_ttc,
                            best_turn_w=turn_hint, straight_feasible=straight_feasible)
        return VOResult(0.0, 0.0, engaged=True, blocked=True,
                        n_obstacles=len(near), n_feasible=0, min_ttc=0.0,
                        best_turn_w=turn_hint, straight_feasible=straight_feasible)

    return VOResult(best[0], best[1], engaged=True, blocked=False,
                    n_obstacles=len(near), n_feasible=n_feasible, min_ttc=best_ttc,
                    best_turn_w=best[1], straight_feasible=straight_feasible)
