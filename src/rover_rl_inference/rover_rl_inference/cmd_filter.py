"""cmd_vel 平滑與限速 — 採 spot_rl/pid.cpp pattern.

兩段處理：
  1. First-order low-pass: smooth_x = (1-α)·smooth_x + α·new_x
     用途：消除 5 Hz 離散動作跳階；α=0.3 預設。
  2. Slew-rate limit: |Δcmd| ≤ ramp_per_s · dt
     用途：避免 policy 從 +0.5 跳到 -0.5 m/s 把馬達搞爆。

順序：raw → low-pass → slew-rate → final。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CmdFilterParams:
    # Low-pass coefficient（每 tick 取入多少新值）。0=完全不變 1=不平滑。
    alpha_linear: float = 0.3
    alpha_angular: float = 0.5    # 角度通常希望反應較快，平滑少一點
    # Slew-rate limit（單位/秒）。
    max_accel_linear: float = 1.0     # m/s² — 不能超過底盤實際 a_max=1.2
    max_accel_angular: float = 3.0    # rad/s² — 經驗值
    # 真正停車的 deadband：|v| < deadband 直接設 0（避免 jitter）
    deadband_linear: float = 0.02
    deadband_angular: float = 0.02


class CmdFilter:
    """單通道 (linear_x, angular_z) low-pass + slew-rate 過濾器."""

    def __init__(self, params: CmdFilterParams | None = None):
        self.p = params or CmdFilterParams()
        self._smooth_v: float = 0.0
        self._smooth_w: float = 0.0
        self._last_v: float = 0.0
        self._last_w: float = 0.0

    def reset(self) -> None:
        self._smooth_v = 0.0
        self._smooth_w = 0.0
        self._last_v = 0.0
        self._last_w = 0.0

    def step(self, target_v: float, target_w: float, dt: float) -> tuple[float, float]:
        """Args:
            target_v: policy 輸出的 linear.x (m/s)
            target_w: policy 輸出的 angular.z (rad/s)
            dt:       距上次 step 的時間 (s)
        """
        p = self.p
        if dt <= 0:
            return self._last_v, self._last_w

        # 1) Low-pass
        self._smooth_v = (1.0 - p.alpha_linear) * self._smooth_v + p.alpha_linear * target_v
        self._smooth_w = (1.0 - p.alpha_angular) * self._smooth_w + p.alpha_angular * target_w

        # 2) Slew-rate limit
        max_dv = p.max_accel_linear * dt
        max_dw = p.max_accel_angular * dt
        dv = _clamp(self._smooth_v - self._last_v, -max_dv, max_dv)
        dw = _clamp(self._smooth_w - self._last_w, -max_dw, max_dw)
        out_v = self._last_v + dv
        out_w = self._last_w + dw

        # 3) Deadband
        if abs(out_v) < p.deadband_linear:
            out_v = 0.0
        if abs(out_w) < p.deadband_angular:
            out_w = 0.0

        self._last_v = out_v
        self._last_w = out_w
        return out_v, out_w


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
