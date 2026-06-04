"""38 logits → (linear_vel_cmd, angular_vel_cmd).

訓練側 (DiscreteDifferentialDriveAction.process_actions)：
  - 19×19 中心對稱 idx；center=9
  - ratio_lin = (idx_a - 9) / 9       ∈ [-1, +1]
  - ratio_ang = (idx_w - 9) / 9       ∈ [-1, +1]
  - 動態加速邊界:
        a_hi = min(+a_max, (+v_max - v) / dt)
        a_lo = max(-a_max, (-v_max - v) / dt)
        accel = ratio_lin * a_hi  if ratio_lin >= 0 else -ratio_lin * a_lo
  - next_v = clamp(v + accel * dt, -v_max, +v_max)
  - actual_ω = ratio_ang * ω_max_action  (≠ obs normalizer)

Deterministic policy: argmax over each 19-way head.
Stochastic: sample from softmax (含 entropy 探索)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionParams:
    num_bins: int = 19
    max_linear_velocity: float = 1.0
    max_linear_accel: float = 0.5
    max_angular_velocity_action: float = 2.0
    dt: float = 0.2


def decode_logits_to_cmd(
    logits_38: np.ndarray,
    current_linear_vel: float,
    params: ActionParams = ActionParams(),
    deterministic: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Args:
        logits_38: shape [38]，前 19 為 accel idx，後 19 為 omega idx
        current_linear_vel: 當前線速度 (m/s)，用於動態加速邊界
        deterministic: argmax；False = softmax sampling
        rng: numpy Generator（sampling 時用）

    Returns:
        (cmd_linear_vel, cmd_angular_vel, actual_accel)
            cmd_linear_vel: m/s, clamped to [-v_max, +v_max]
            cmd_angular_vel: rad/s, clamped to [-ω_max, +ω_max]
            actual_accel: m/s²，可回填到下一輪 obs[0]
    """
    if logits_38.shape != (38,):
        raise ValueError(f"logits shape {logits_38.shape} != (38,)")

    # 38 logits = 兩個獨立的 19-way head（MultiDiscrete([19,19])）：前段 accel、後段 omega
    logits_a = logits_38[: params.num_bins]
    logits_w = logits_38[params.num_bins :]

    if deterministic:
        # 部署預設 deterministic：各 head 取 argmax，去掉訓練時的 entropy 探索抖動
        idx_a = int(np.argmax(logits_a))
        idx_w = int(np.argmax(logits_w))
    else:
        rng = rng or np.random.default_rng()
        idx_a = _sample_categorical(logits_a, rng)
        idx_w = _sample_categorical(logits_w, rng)

    # 19 bins 中心對稱（center=9）：idx 映到 [-1,+1] 比例，9=不變、0=全負、18=全正
    center = (params.num_bins - 1) / 2.0
    ratio_a = (idx_a - center) / center
    ratio_w = (idx_w - center) / center

    v = float(current_linear_vel)
    a_max = params.max_linear_accel
    v_max = params.max_linear_velocity
    dt = params.dt

    # 動態加速邊界：除 ±a_max 外，再夾住「這一步最多能加/減多少才不超過 v_max」，
    # 與訓練端 process_actions 完全一致，避免實車 next_v 飽和點與訓練不同
    a_hi = min(+a_max, (+v_max - v) / dt)
    a_lo = max(-a_max, (-v_max - v) / dt)

    # ratio 正→朝上界加速、負→朝下界減速；兩側各自縮放確保 accel 落在 [a_lo, a_hi]
    if ratio_a >= 0.0:
        accel = ratio_a * a_hi
    else:
        accel = -ratio_a * a_lo

    accel = float(np.clip(accel, a_lo, a_hi))
    next_v = float(np.clip(v + accel * dt, -v_max, +v_max))
    # omega 是直接速度映射（非積分加速），用 action ω_max=2.0（≠ obs normalizer 1.5）
    cmd_w = float(np.clip(ratio_w * params.max_angular_velocity_action,
                          -params.max_angular_velocity_action,
                          +params.max_angular_velocity_action))
    return next_v, cmd_w, accel


def _sample_categorical(logits: np.ndarray, rng: np.random.Generator) -> int:
    p = logits - logits.max()
    p = np.exp(p)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))
