"""map→base 位姿跳變 fail-closed guard（handoff #3）。

實車失效：`map_yaw` 在約 0.05 s 內跳了約 97.4°。物理上限是
`ω_max × dt = 1.2 × 0.05 = 0.06 rad = 3.4°` —— **超標約 28 倍**。
policy 信了這個跳變，於是拼命打方向盤要「轉回來」。

為什麼既有的閘門攔不到：`localization.py` 那道 delta gate 守的是
**map→odom 偏移量的更新**（拒絕壞的 NDT 修正），但 policy 用的位姿是
`odom + offset`。只要 odom 自己跳，位姿照樣跳，那道閘門完全在鏈外。
handoff 明寫：「An unused tracker elsewhere is not sufficient.」
所以本 guard 必須裝在 policy **實際消費**位姿的那一點。

判準不靠人工調參，直接用車輛物理極限反推：
    位移上限 = v_max × dt × margin
    轉角上限 = ω_max × dt × margin

分級處理（2026-08-21 加入）：實車量到 NDT 在此環境的 transform_probability
在 1.0~2.1 間游移、收斂門檻 1.6 正好切在分布中央，於是 map→odom 每次更新平均
跳 0.134 m（車靜止時只有 5.6 mm）。這種**常態定位噪聲**只超標一點點，卻和
97° 那種災難跳變吃同一套 fail-closed，害車每 16 秒硬停 0.6 s + 重新加速 2 s
（走走停停）。故依超標倍率 ratio 分兩級：

    ratio ≤ 1              → ok        位姿可用
    1 < ratio < hard_ratio → soft      位姿不可信但不急停：呼叫端改用 odom 遞推
                                       位姿（等於暫時凍結 map→odom offset）繼續推論
    ratio ≥ hard_ratio     → rejected  災難跳變，維持原本 fail-closed

soft 連續超過 soft_max_consecutive 拍 → 升級 rejected：NDT 真的掛掉時不能
無限用 odom 遞推漂下去。

rejected 後（fail-closed）：
    1. 輸出 0（不沿用上一個命令）
    2. 清空 RNN state 與 action history —— 錯誤觀測已進 K=8 幀歷史與 4 維
       動作歷史，只清當拍不夠，污染會延續數拍
    3. 記錄原因與數值
    4. 連續 N 拍合理才恢復（避免在跳變邊緣反覆進出）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STATE_OK = "ok"
STATE_SOFT = "soft"
STATE_REJECTED = "rejected"
STATE_RECOVERING = "recovering"


def _wrap_pi(a: float) -> float:
    """把角度差包到 [-π, π]，避免 ±π 邊界被誤判成大跳變。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class GuardResult:
    ok: bool                 # True = 位姿可用（soft 也是 False，呼叫端須改用遞推位姿）
    state: str               # ok / soft / rejected / recovering
    reason: str | None
    dt: float | None
    dpos: float | None
    dyaw: float | None
    ratio: float | None = None   # 超標倍率 = max(dpos/上限, dyaw/上限)；≤1 表示正常

    @property
    def use_extrapolation(self) -> bool:
        """True = 位姿不可信但不必停車，改用 odom 遞推值。"""
        return self.state == STATE_SOFT


class PoseJumpGuard:
    """連續兩次位姿之間的變化必須能用「經過時間 × 物理極速」解釋。

    Args:
        v_max: 線速度上限 (m/s)，用車輛規格值
        w_max: 角速度上限 (rad/s)
        margin: 容忍倍率。1.5 已足夠吸收取樣抖動與 TF 內插誤差，
                而實車那次超標 28 倍，任何合理 margin 都攔得到。
        recover_samples: 連續幾拍合理才解除拒絕
        max_gap_s: 間隔超過此值視為重新開始（例如節點剛啟動、長時間 estop），
                   不做判定也不算跳變 —— 否則恢復時必定誤判一次。
        hard_ratio: 超標幾倍才算災難跳變（fail-closed）。1.0 = 關閉分級，
                    全部走舊的硬停行為。實車 NDT 常態噪聲約 1~1.5 倍，
                    97° 那次是 28 倍，2.0 能把兩者分開。
        soft_max_consecutive: 連續幾拍軟修正就升級成硬停（定位真的掛了）。
    """

    def __init__(self, v_max: float = 1.0, w_max: float = 1.2,
                 margin: float = 1.5, recover_samples: int = 3,
                 max_gap_s: float = 1.0, hard_ratio: float = 2.0,
                 soft_max_consecutive: int = 10):
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.margin = float(margin)
        self.recover_samples = int(recover_samples)
        self.max_gap_s = float(max_gap_s)
        self.hard_ratio = max(float(hard_ratio), 1.0)
        self.soft_max_consecutive = int(soft_max_consecutive)

        self._last: tuple[float, float, float, float] | None = None  # x,y,yaw,t
        self._state = STATE_OK
        self._good_streak = 0
        self._soft_streak = 0
        self.reject_count = 0
        self.soft_count = 0

    @property
    def state(self) -> str:
        return self._state

    def reset(self, reason: str = "") -> None:
        """節點重啟／模式切換時呼叫，避免拿舊位姿當基準。"""
        self._last = None
        self._state = STATE_OK
        self._good_streak = 0
        self._soft_streak = 0

    def set_reference(self, x: float, y: float, yaw: float, t: float) -> None:
        """把基準改成呼叫端實際採用的位姿（soft 時傳遞推值）。

        soft 那拍的原始位姿是髒的，不能當下一拍的比較基準 —— 否則髒值一旦被
        記成基準，下一拍相對它就「看起來正常」，等於默默吃下跳變。
        """
        self._last = (float(x), float(y), float(yaw), float(t))

    def check(self, x: float, y: float, yaw: float, t: float) -> GuardResult:
        prev = self._last

        if prev is None:
            self._last = (x, y, yaw, t)
            self._state = STATE_OK
            return GuardResult(True, STATE_OK, None, None, None, None, 0.0)

        px, py, pyaw, pt = prev
        dt = t - pt

        # 時間倒退或間隔過長：不判定，重新以本次為基準
        if dt <= 0.0 or dt > self.max_gap_s:
            self._last = (x, y, yaw, t)
            self._state = STATE_OK
            self._good_streak = 0
            self._soft_streak = 0
            return GuardResult(True, STATE_OK, "gap_reset", dt, None, None, 0.0)

        dpos = math.hypot(x - px, y - py)
        dyaw = abs(_wrap_pi(yaw - pyaw))
        pos_lim = self.v_max * dt * self.margin
        yaw_lim = self.w_max * dt * self.margin
        ratio = max(dpos / pos_lim if pos_lim > 0.0 else 0.0,
                    dyaw / yaw_lim if yaw_lim > 0.0 else 0.0)

        bad = None
        if dpos > pos_lim:
            bad = (f"位移跳變 {dpos:.3f}m > 上限 {pos_lim:.3f}m "
                   f"(v_max {self.v_max} × dt {dt:.3f} × {self.margin})")
        elif dyaw > yaw_lim:
            bad = (f"角度跳變 {math.degrees(dyaw):.1f}° > 上限 "
                   f"{math.degrees(yaw_lim):.1f}° "
                   f"(ω_max {self.w_max} × dt {dt:.3f} × {self.margin})")

        if bad is not None:
            # 分級：只超標一點 = NDT 常態噪聲，交給呼叫端用 odom 遞推，不急停。
            # 連續軟修正太久代表定位真的失效，升級成 fail-closed。
            soft_ok = (ratio < self.hard_ratio
                       and self._soft_streak < self.soft_max_consecutive)
            if soft_ok:
                # 基準不更新為髒值：呼叫端算出遞推位姿後會 set_reference() 補上
                self._state = STATE_SOFT
                self._good_streak = 0
                self._soft_streak += 1
                self.soft_count += 1
                return GuardResult(
                    False, STATE_SOFT,
                    f"{bad}；超標 {ratio:.1f}× < {self.hard_ratio}× "
                    f"→ 改用 odom 遞推位姿（連續 {self._soft_streak}"
                    f"/{self.soft_max_consecutive}）",
                    dt, dpos, dyaw, ratio,
                )
            self._last = (x, y, yaw, t)
            self._state = STATE_REJECTED
            self._good_streak = 0
            if self._soft_streak >= self.soft_max_consecutive:
                bad += f"；軟修正已連續 {self._soft_streak} 拍，升級硬停"
            self._soft_streak = 0
            self.reject_count += 1
            return GuardResult(False, STATE_REJECTED, bad, dt, dpos, dyaw, ratio)

        self._last = (x, y, yaw, t)
        self._soft_streak = 0

        # 這一拍合理。soft 不需要恢復期（它從未停車），直接回 ok
        if self._state in (STATE_OK, STATE_SOFT):
            self._state = STATE_OK
            return GuardResult(True, STATE_OK, None, dt, dpos, dyaw, ratio)

        # 硬停後的恢復期：要連續數拍合理才放行
        self._good_streak += 1
        if self._good_streak >= self.recover_samples:
            self._state = STATE_OK
            self._good_streak = 0
            return GuardResult(True, STATE_OK, "recovered", dt, dpos, dyaw, ratio)
        self._state = STATE_RECOVERING
        return GuardResult(
            False, STATE_RECOVERING,
            f"恢復中 {self._good_streak}/{self.recover_samples}",
            dt, dpos, dyaw, ratio,
        )


class OdomDeadReckoner:
    """把最後一個可信 map 位姿用 odom 增量往前推。

    guard 判 soft 那拍，NDT 給的 map 位姿髒了但 odom 是乾淨的（實車量到 odom
    逐拍 0.03 m/2.7°、與 odom_v/odom_w 完全一致，抖的只有 map→odom）。此時用
    「上一個可信 map 位姿 ⊕ odom 這段時間走的量」代替，效果等同暫時凍結
    map→odom offset：短時間內位姿連續、不吃 NDT 噪聲。

    只適合撐幾拍 —— odom 本身有 1%/m drift，撐久了要交回 fail-closed。
    """

    def __init__(self) -> None:
        self._ref: tuple[float, float, float, float, float, float] | None = None

    def reset(self) -> None:
        self._ref = None

    @property
    def has_reference(self) -> bool:
        return self._ref is not None

    def update(self, map_x: float, map_y: float, map_yaw: float,
               odom_x: float, odom_y: float, odom_yaw: float) -> None:
        """記下一組「可信 map 位姿 ↔ 同一拍的 odom 位姿」配對。"""
        self._ref = (float(map_x), float(map_y), float(map_yaw),
                     float(odom_x), float(odom_y), float(odom_yaw))

    def extrapolate(self, odom_x: float, odom_y: float,
                    odom_yaw: float) -> tuple[float, float, float] | None:
        """回傳遞推後的 (map_x, map_y, map_yaw)；還沒有基準則 None。"""
        if self._ref is None:
            return None
        mx, my, myaw, ox, oy, oyaw = self._ref
        # map→odom 的旋轉量：兩個 frame 的 yaw 差。odom 位移在 odom frame，
        # 要先轉到 map frame 再疊加，否則車頭方向一偏就整個推歪。
        th = myaw - oyaw
        dx, dy = odom_x - ox, odom_y - oy
        cos_t, sin_t = math.cos(th), math.sin(th)
        return (mx + cos_t * dx - sin_t * dy,
                my + sin_t * dx + cos_t * dy,
                _wrap_pi(myaw + _wrap_pi(odom_yaw - oyaw)))
