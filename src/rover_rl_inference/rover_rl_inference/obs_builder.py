"""79D observation builder — 對齊 SA1_v2 PolicyCfg.

Layout (與 ObservationsCfgVLP16.PolicyCfg 完全一致)：
    [0]      ā_t   = a_t / max_acceleration       # normalized linear accel
    [1]      v̄_t   = v_t / max_linear_velocity   # normalized linear vel
    [2]      ω̄_t   = ω_t / max_angular_velocity_obs   # normalized angular vel
    [3]      r_a   = robot_radius                  # 常數
    [4:6]    goal in robot body frame (x_fwd, y_left)
    [6:78]   LiDAR 72-bin sweep (normalized [0, 1])
    [78]     time_remaining ratio [0, 1]

注意：
- obs 內 ω 用 max_angular_velocity_obs=1.5 正規化（與 vlp16 ObsTerm 一致），
  動作端 max_angular_vel=2.0；兩者不同，請勿混用。
- goal 為 body frame：x 前方 (+), y 左方 (+)。需要從 world goal 與當前 odom 算 TF。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsParams:
    max_acceleration: float = 1.0
    max_linear_velocity: float = 1.0
    max_angular_velocity_obs: float = 1.5
    robot_radius: float = 0.35
    lidar_num_bins: int = 72
    episode_horizon_s: float = 60.0  # time_remaining 分母（訓練 episode 長度）


def build_obs_79d(
    last_accel: float,
    linear_vel: float,
    angular_vel: float,
    goal_body_x: float,
    goal_body_y: float,
    lidar_sweep_72: np.ndarray,
    elapsed_s: float,
    params: ObsParams = ObsParams(),
) -> np.ndarray:
    if lidar_sweep_72.shape != (params.lidar_num_bins,):
        raise ValueError(
            f"lidar_sweep_72 shape {lidar_sweep_72.shape} != ({params.lidar_num_bins},)"
        )

    ego_a = np.clip(last_accel / max(params.max_acceleration, 1e-6), -1.0, 1.0)
    ego_v = np.clip(linear_vel / max(params.max_linear_velocity, 1e-6), -1.0, 1.0)
    ego_w = np.clip(angular_vel / max(params.max_angular_velocity_obs, 1e-6), -1.0, 1.0)
    radius = params.robot_radius

    time_rem = max(
        0.0,
        1.0 - elapsed_s / max(params.episode_horizon_s, 1e-6),
    )

    obs = np.empty(79, dtype=np.float32)
    obs[0] = ego_a
    obs[1] = ego_v
    obs[2] = ego_w
    obs[3] = radius
    obs[4] = goal_body_x
    obs[5] = goal_body_y
    obs[6:78] = lidar_sweep_72.astype(np.float32)
    obs[78] = time_rem
    return obs


def build_obs_raw(
    raw_obs_dim: int,
    *,
    last_accel: float,
    linear_vel: float,
    angular_vel: float,
    goal_body_x: float,
    goal_body_y: float,
    lidar_sweep_72: np.ndarray,
    elapsed_s: float,
    params: ObsParams = ObsParams(),
) -> np.ndarray:
    """Build raw obs vector matching the bundle's expected raw_obs_dim.

    - 79D (SA1_v2): ego(4) + goal(2) + LiDAR(72) + time(1)
    - 139D (SA5/6/7): ego(4) + goal(2) + LiDAR(72) + obstacles(60 zeros) + time(1)

    部署沒有 ground-truth 障礙物，60D 補零；normalizer 會把這些零依訓練統計轉換，
    再 slice 掉（export_policy 已內建 [0..77, 138] 切片）。
    """
    o79 = build_obs_79d(
        last_accel, linear_vel, angular_vel, goal_body_x, goal_body_y,
        lidar_sweep_72, elapsed_s, params,
    )
    if raw_obs_dim == 79:
        return o79
    if raw_obs_dim == 139:
        out = np.zeros(139, dtype=np.float32)
        out[0:78] = o79[0:78]      # ego + goal + LiDAR
        # out[78:138] = 0           # obstacles ground-truth unavailable
        out[138] = o79[78]          # time
        return out
    raise ValueError(f"unsupported raw_obs_dim={raw_obs_dim}")


def world_goal_to_body(
    goal_world_x: float,
    goal_world_y: float,
    robot_world_x: float,
    robot_world_y: float,
    robot_world_yaw: float,
) -> tuple[float, float]:
    """world frame goal → body frame (x_fwd, y_left)."""
    dx = goal_world_x - robot_world_x
    dy = goal_world_y - robot_world_y
    c = math.cos(-robot_world_yaw)
    s = math.sin(-robot_world_yaw)
    return c * dx - s * dy, s * dx + c * dy
