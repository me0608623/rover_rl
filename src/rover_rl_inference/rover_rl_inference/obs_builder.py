"""Observation builder — 79D (SA1_v2 / SA5/6/7) 與 83D (v3c) 兩種 layout.

79D Layout (與 ObservationsCfgVLP16.PolicyCfg 完全一致)：
    [0]      ā_t   = a_t / max_acceleration       # normalized linear accel
    [1]      v̄_t   = v_t / max_linear_velocity   # normalized linear vel
    [2]      ω̄_t   = ω_t / max_angular_velocity_obs   # normalized angular vel
    [3]      r_a   = robot_radius                  # 常數
    [4:6]    goal in robot body frame (x_fwd, y_left)
    [6:78]   LiDAR 72-bin sweep (normalized [0, 1])
    [78]     time_remaining ratio [0, 1]

83D Layout (v3c — 79D 之後接 4D action history，抗抽動)：
    [0:79]   同上 79D（ego + goal + LiDAR + time）
    [79]     a_{t-1} = applied_accel_{t-1} / act_hist_a_max,   clip[-2, 2]
    [80]     ω_{t-1} = applied_omega_{t-1} / act_hist_omega_max, clip[-2, 2]
    [81]     a_{t-2} = applied_accel_{t-2} / act_hist_a_max,   clip[-2, 2]
    [82]     ω_{t-2} = applied_omega_{t-2} / act_hist_omega_max, clip[-2, 2]

注意：
- obs 內 ω 用 max_angular_velocity_obs=1.5 正規化（與 vlp16 ObsTerm 一致），
  動作端 max_angular_vel 另計；兩者不同，請勿混用。
- action history 用 act_hist_a_max=0.2、act_hist_omega_max=π/15（≈0.209）正規化，
  與訓練端 discrete_applied_action_history 一致；**這不是動作上限，是歷史欄正規化分母**。
- action history 來源是 policy 自己上 2 步「下達並經動態 clamp/slew 後」的指令
  (decode_logits_to_cmd 回傳的 actual_accel / actual_omega)，**不是 IMU/odom 量測值**。
- goal 為 body frame：x 前方 (+), y 左方 (+)。需要從 world goal 與當前 odom 算 TF。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# 這些常數必須與訓練端 ObservationsCfgVLP16 完全一致；任一不符都會造成
# obs 分布偏移（normalizer 是用訓練統計 bake 進去的），policy 行為立刻失準。
@dataclass(frozen=True)
class ObsParams:
    max_acceleration: float = 1.0          # obs[0] 正規化分母（≠ action 的 act_max_linear_accel=0.5）
    max_linear_velocity: float = 1.0       # obs[1] 正規化分母
    max_angular_velocity_obs: float = 1.5  # obs[2] 正規化分母；刻意 ≠ action ω_max=2.0，勿混用
    robot_radius: float = 0.35             # obs[3] 常數欄；同 sweep 正規化用的 r_robot
    lidar_num_bins: int = 72
    episode_horizon_s: float = 60.0  # time_remaining 分母（訓練 episode 長度）
    # v3c action history 正規化分母（對齊 discrete_applied_action_history 的 a_max / omega_max）
    act_hist_a_max: float = 0.2                   # [79]/[81] applied accel 正規化分母
    act_hist_omega_max: float = math.pi / 15.0    # [80]/[82] applied omega 正規化分母 ≈ 0.2094


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

    # ego 三項都正規化到 [-1, 1]：訓練時 obs 即此範圍，clip 防實車瞬時超標污染分布
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


def normalize_applied_action(
    accel: float,
    omega: float,
    params: ObsParams = ObsParams(),
) -> tuple[float, float]:
    """把一步 applied action (accel, omega) 正規化成 action-history 欄位用的 (a_norm, ω_norm).

    對齊訓練端 discrete_applied_action_history：a / act_hist_a_max、ω / act_hist_omega_max，
    再 clip 到 [-2, 2]（防實車瞬時超標污染分布 / NaN 傳播）。

    Args:
        accel: 該步實際 applied 線加速度 (m/s²) = decode_logits_to_cmd 回傳的 actual_accel
        omega: 該步實際 applied 角速度 (rad/s) = decode_logits_to_cmd 回傳的 cmd_w（已 slew）
    Returns:
        (a_norm, ω_norm)，皆 clip 在 [-2, 2]
    """
    a = float(np.clip(accel / max(params.act_hist_a_max, 1e-6), -2.0, 2.0))
    w = float(np.clip(omega / max(params.act_hist_omega_max, 1e-6), -2.0, 2.0))
    return a, w


def build_obs_83d(
    last_accel: float,
    linear_vel: float,
    angular_vel: float,
    goal_body_x: float,
    goal_body_y: float,
    lidar_sweep_72: np.ndarray,
    elapsed_s: float,
    action_hist_norm: np.ndarray | None = None,
    params: ObsParams = ObsParams(),
) -> np.ndarray:
    """83D = 79D base + 4D action history (v3c).

    Args:
        action_hist_norm: shape (4,) = [a_{t-1}, ω_{t-1}, a_{t-2}, ω_{t-2}]，**已正規化**
            (用 normalize_applied_action 算好，新→舊排列)。None → 全 0（episode/goal reset）。
    """
    o79 = build_obs_79d(
        last_accel, linear_vel, angular_vel, goal_body_x, goal_body_y,
        lidar_sweep_72, elapsed_s, params,
    )
    obs = np.empty(83, dtype=np.float32)
    obs[0:79] = o79                    # ego + goal + LiDAR + time（time 在 [78]）
    if action_hist_norm is None:
        obs[79:83] = 0.0
    else:
        ah = np.asarray(action_hist_norm, dtype=np.float32).reshape(-1)
        if ah.shape != (4,):
            raise ValueError(f"action_hist_norm shape {ah.shape} != (4,)")
        obs[79:83] = ah
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
    action_hist_norm: np.ndarray | None = None,
    params: ObsParams = ObsParams(),
) -> np.ndarray:
    """Build raw obs vector matching the bundle's expected raw_obs_dim.

    - 79D (SA1_v2): ego(4) + goal(2) + LiDAR(72) + time(1)
    - 83D (v3c): 79D + action history(4)；需傳 action_hist_norm（None → 補 0）
    - 139D (SA5/6/7): ego(4) + goal(2) + LiDAR(72) + obstacles(60 zeros) + time(1)

    部署沒有 ground-truth 障礙物，60D 補零；normalizer 會把這些零依訓練統計轉換，
    再 slice 掉（export_policy 已內建 [0..77, 138] 切片）。
    """
    if raw_obs_dim == 83:
        return build_obs_83d(
            last_accel, linear_vel, angular_vel, goal_body_x, goal_body_y,
            lidar_sweep_72, elapsed_s, action_hist_norm, params,
        )
    o79 = build_obs_79d(
        last_accel, linear_vel, angular_vel, goal_body_x, goal_body_y,
        lidar_sweep_72, elapsed_s, params,
    )
    if raw_obs_dim == 79:
        return o79
    if raw_obs_dim == 139:
        # 139D layout 把 60D 障礙物欄插在 LiDAR 與 time 之間，因此不能直接接 o79，
        # 要把 o79 的 time 欄搬到 index 138；中間 [78:138] 維持 0（實車無 ground-truth
        # 障礙物），normalizer 會依訓練統計轉換這些 0，再由 bundle 內建切片丟掉。
        out = np.zeros(139, dtype=np.float32)
        out[0:78] = o79[0:78]      # ego + goal + LiDAR
        # out[78:138] = 0           # obstacles ground-truth unavailable
        out[138] = o79[78]          # time
        return out
    raise ValueError(f"unsupported raw_obs_dim={raw_obs_dim} (expected 79, 83 or 139)")


def world_goal_to_body(
    goal_world_x: float,
    goal_world_y: float,
    robot_world_x: float,
    robot_world_y: float,
    robot_world_yaw: float,
) -> tuple[float, float]:
    """world frame goal → body frame (x_fwd, y_left).

    訓練的 goal obs 是 body frame（前 x、左 y），故部署必須把 world goal 用
    -yaw 旋轉回機器人本體座標；少這步 policy 會以為 goal 永遠在固定方位。
    """
    dx = goal_world_x - robot_world_x
    dy = goal_world_y - robot_world_y
    c = math.cos(-robot_world_yaw)
    s = math.sin(-robot_world_yaw)
    return c * dx - s * dy, s * dx + c * dy
