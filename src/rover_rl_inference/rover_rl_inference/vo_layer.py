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
  5. 可行候選中選 cost = w_v·Δv² + w_ω·Δω² 最小者（最貼近 RL 意圖）。
  6. 全部不可行 → fallback：停車 (0, 0)（安全優先，對應使用者選定行為）。

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
    # --- 啟動條件：障礙進入此距離才跑 VO（省運算、避免遠處誤煞）---
    engage_range: float = 6.0   # m


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


def _rollout_min_ttc(v: float, w: float, robot: RobotState,
                     obstacles: list[Obstacle], p: VOParams) -> float:
    """以等速 (v, w) rollout 機器人弧線，回傳 horizon 內最近碰撞時間；無碰撞回 inf.

    機器人：θ(t)、x(t)、y(t) 由 (v, w) 積分（odom frame）。
    障礙：線性外推 p_i + v_i·t（假設等速，短 horizon 內合理）。
    碰撞：中心距 < r_robot + r_obs + margin。
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
                return t  # 在 t 秒碰撞
    return math.inf


def compute_safe_cmd(v_des: float, w_des: float, robot: RobotState,
                     obstacles: list[Obstacle], p: VOParams) -> VOResult:
    """VO 主入口：給 RL 期望速度 + 障礙，回最接近且安全的 (v, w)。"""
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
                        n_obstacles=0, n_feasible=1, min_ttc=math.inf)

    # 3) dynamic window（以實測速度為中心，window_time 內可達範圍 → 候選夠廣可繞行）
    v_lo = _clamp(robot.v - p.accel_v * p.window_time, p.v_min, p.v_max)
    v_hi = _clamp(robot.v + p.accel_v * p.window_time, p.v_min, p.v_max)
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

    # 4-5) 掃所有候選，挑「可行且最貼近 desired」者
    best: tuple[float, float] | None = None
    best_cost = math.inf
    best_ttc = math.inf
    n_feasible = 0
    for v in v_cands:
        for w in w_cands:
            if _rollout_min_ttc(v, w, robot, near, p) == math.inf:  # horizon 內不撞
                n_feasible += 1
                cost = p.w_v * (v - v_des) ** 2 + p.w_w * (w - w_des) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best = (v, w)
                    best_ttc = math.inf

    # 6) 全不可行 → fallback 停車（安全優先）
    if best is None:
        return VOResult(0.0, 0.0, engaged=True, blocked=True,
                        n_obstacles=len(near), n_feasible=0, min_ttc=0.0)

    return VOResult(best[0], best[1], engaged=True, blocked=False,
                    n_obstacles=len(near), n_feasible=n_feasible, min_ttc=best_ttc)
