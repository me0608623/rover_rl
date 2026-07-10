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
    action_history: np.ndarray | None = None,
) -> np.ndarray:
    """Build raw obs vector matching the bundle's expected raw_obs_dim.

    - 79D (SA1_v2): ego(4) + goal(2) + LiDAR(72) + time(1)
    - 139D (SA5/6/7): ego(4) + goal(2) + LiDAR(72) + obstacles(60 zeros) + time(1)
    - 83D (v3c action stacking): 79D + action_history(4) = [a_t-1, ω_t-1, a_t-2, ω_t-2]

    部署沒有 ground-truth 障礙物，60D 補零；normalizer 會把這些零依訓練統計轉換，
    再 slice 掉（export_policy 已內建 [0..77, 138] 切片）。

    action_history 僅 83D（v3c）會用到；79/139D 忽略。內容為「正規化後的近 2 步動作」，
    由 policy_node 維護（appendleft 最新）。⚠ 正規化常數必須與 v3c 訓練端一致。
    """
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
    if raw_obs_dim == 83:
        # v3c action stacking：79D（time 在 index 78）後直接接 4D action history，
        # 與訓練端 concat([..., [time], act_hist]) 的順序逐字一致。
        if action_history is None or action_history.shape != (4,):
            raise ValueError(
                f"raw_obs_dim=83 需 action_history shape (4,)，got "
                f"{None if action_history is None else action_history.shape}"
            )
        out = np.empty(83, dtype=np.float32)
        out[0:79] = o79
        out[79:83] = action_history.astype(np.float32)
        return out
    raise ValueError(f"unsupported raw_obs_dim={raw_obs_dim}")


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


# ============================================================================
# LV-DOT 動態障礙 channel (30D) — 對齊訓練端 dynamic_obstacles_lvdot
# ----------------------------------------------------------------------------
# parity 已驗 (IsaacLab: scripts/.../tools/test_lvdot_parity.py):
#   本函數的計算 = 訓練 torch 函數 dynamic_obstacles_lvdot，4 情境逐位相同。
# 特徵順序 per obstacle: [px_norm, py_norm, vx_norm, vy_norm, r, valid]
#   px/py: body frame 位置 (+x 前, +y 左), ÷max_distance clamp[-1,1]
#   vx/vy: body frame 相對速度 (障礙−機器人, 轉 body), ÷v_max clamp[-2,2]
#   r:     等效半徑 (m);  valid: 1=有效, 0=填充/無
# ============================================================================
@dataclass(frozen=True)
class LvdotObsParams:
    top_k: int = 5
    max_distance: float = 8.0             # 位置正規化分母 + range cull (訓練一致)
    v_max: float = 1.5                    # 速度正規化分母 (訓練一致)
    dynamic_speed_threshold: float = 0.3  # LV-DOT dynamic_velocity_threshold (實測 config)


def build_lvdot_channel(
    obstacles,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    robot_vx_w: float,
    robot_vy_w: float,
    params: LvdotObsParams = LvdotObsParams(),
) -> np.ndarray:
    """把 LV-DOT/vo_interface tracked obstacles 轉成訓練端 30D 障礙 channel。

    obstacles: iterable，每個有 .x .y .vx .vy .r（odom frame，速度為 LV-DOT 絕對速度）。
               = vo_interface TrackedObstacle 的 position/velocity/radius，或 vo_layer.Obstacle。

    ⚠ robot_vx_w / robot_vy_w 必須是「世界(odom) frame」速度，不是 body frame twist。
      差動底盤 body twist (v_fwd, v_lat≈0) → 世界: vx_w=cos(yaw)·v_fwd, vy_w=sin(yaw)·v_fwd。
      (訓練端用 root_lin_vel_w = 世界 frame；此處也要世界 frame 才對齊 parity。)

    座標/正規化/排序/動態過濾與訓練端 dynamic_obstacles_lvdot 完全一致 (parity 已驗)。
    """
    cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
    md, vmax, dth = params.max_distance, params.v_max, params.dynamic_speed_threshold
    cand = []  # (dist, px_n, py_n, vx_n, vy_n, r)
    for ob in obstacles:
        abs_speed = math.hypot(ob.vx, ob.vy)
        if abs_speed <= dth:            # 動態過濾 (LV-DOT 已只給動態，冗餘但保 parity)
            continue
        dx, dy = ob.x - robot_x, ob.y - robot_y
        dist = math.sqrt(dx * dx + dy * dy + 1e-8)
        if dist > md:                   # range cull
            continue
        px = cos_y * dx + sin_y * dy
        py = -sin_y * dx + cos_y * dy
        rvx, rvy = ob.vx - robot_vx_w, ob.vy - robot_vy_w
        vxb = cos_y * rvx + sin_y * rvy
        vyb = -sin_y * rvx + cos_y * rvy
        px_n = float(np.clip(px / md, -1.0, 1.0))
        py_n = float(np.clip(py / md, -1.0, 1.0))
        vx_n = float(np.clip(vxb / vmax, -2.0, 2.0))
        vy_n = float(np.clip(vyb / vmax, -2.0, 2.0))
        cand.append((dist, px_n, py_n, vx_n, vy_n, float(ob.r)))
    cand.sort(key=lambda t: t[0])       # 由近到遠
    out = np.zeros(params.top_k * 6, dtype=np.float32)
    for slot in range(min(params.top_k, len(cand))):
        _, px_n, py_n, vx_n, vy_n, r = cand[slot]
        out[slot * 6:slot * 6 + 6] = [px_n, py_n, vx_n, vy_n, r, 1.0]  # valid=1
    return out


def build_obs_lvdot_109d(
    *,
    last_accel: float,
    linear_vel: float,
    angular_vel: float,
    goal_body_x: float,
    goal_body_y: float,
    lidar_sweep_72: np.ndarray,
    elapsed_s: float,
    obstacles,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    robot_vx_w: float,
    robot_vy_w: float,
    params: ObsParams = ObsParams(),
    lvdot_params: LvdotObsParams = LvdotObsParams(),
) -> np.ndarray:
    """109D obs = build_obs_79d (79D) + LV-DOT 動態障礙 channel (30D)。

    對齊訓練端 CHARGE_USE_LVDOT_OBS=1 佈局:
        ego(4) + goal(2) + LiDAR(72) + time(1) + dynamic_obstacles(30) = 109D
    (障礙 channel 接在 time 之後，與訓練 concat 順序逐字一致。)
    """
    o79 = build_obs_79d(
        last_accel, linear_vel, angular_vel, goal_body_x, goal_body_y,
        lidar_sweep_72, elapsed_s, params,
    )
    ch = build_lvdot_channel(
        obstacles, robot_x, robot_y, robot_yaw, robot_vx_w, robot_vy_w, lvdot_params,
    )
    return np.concatenate([o79, ch]).astype(np.float32)  # 109D
