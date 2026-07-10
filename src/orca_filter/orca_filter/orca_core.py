"""orca_core.py — 純 ORCA 安全過濾器(無 Isaac Lab 依賴)。

從 IsaacLab play_eval/rvo2_safety_filter.py 抽取核心三函數,1:1 移植邏輯與常數:
  - solve()                     = _solve_orca_single_env
  - _project_to_body_frame      = world v_safe → body (v_linear, omega)
  - _lateral_evasion_if_frozen  = ORCA 凍結時注入垂直逃生

設計:non-cooperative(行人不會避讓機器人)
  - ego_responsibility=1.0:機器人扛 100% 避讓,障礙物 max_speed≈0(不會減速讓你)
  - obs_inflation=1.8:算碰撞時障礙半徑 ×1.8,提早反應補償行人不讓

只處理動態障礙物;靜態(|vel|<threshold)完全排除,留給 policy/LiDAR。

依賴:pyrvo2(mit-acl/Python-RVO2)、numpy。無 Isaac Lab / GPU 依賴。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import rvo2


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ORCAInput:
    """單一環境的 ORCA 輸入(全部 world/odom frame)。

    obs_* 已經過 culling(只含 culling_radius 內的附近障礙物)。
    """
    robot_pos: tuple[float, float]      # 機器人世界位置
    robot_vel: tuple[float, float]      # 機器人世界速度
    yaw: float                          # 機器人朝向(rad)
    v_pref: tuple[float, float]         # policy「偏好速度」(world),= R(yaw_next)·[v_next_body,0]
    v_next_body: float                  # policy 輸出 body 線速度(未介入時 pass-through 用)
    omega_rl: float                     # policy 輸出 body 角速度
    # 障礙物(culling 後,K 個)
    obs_pos: np.ndarray                 # [K,2] world
    obs_vel: np.ndarray                 # [K,2] world 絕對速度
    obs_radii: np.ndarray               # [K]
    obs_is_static: np.ndarray           # [K] bool


@dataclass
class ORCAOutput:
    """ORCA 求解結果。"""
    v_safe_linear: float                # body frame 線速度(寫回 cmd)
    v_safe_omega: float                 # body frame 角速度
    orca_active: bool                   # 是否介入(||v_safe-v_pref|| > eps)
    v_pref_world: np.ndarray            # [2] 給 RViz 畫
    v_safe_world: np.ndarray            # [2] 給 RViz 畫
    vo_cone_data: list = field(default_factory=list)  # [(rel_xy[2], combined_r)] 給 RViz 畫 cone


# ══════════════════════════════════════════════════════════════════════════════
# ORCA Filter
# ══════════════════════════════════════════════════════════════════════════════

class ORCAFilter:
    """純 ORCA 安全過濾器(non-cooperative, ego 100% 避讓)。

    只處理動態障礙物;靜態(|vel|<threshold)完全排除,留給 policy/LiDAR。
    ORCA 輸出近零速度(凍結)時,注入垂直逃生速度避免坐以待斃。
    """

    # ── 介入判定 ──
    INTERVENTION_EPSILON = 0.05          # m/s,||v_safe - v_pref|| 超過此值才算介入
    TELEPORT_DISP_THRESHOLD = 1.0        # m/decision step,單步位移超過此值視為 reset 瞬移

    # ── 非完整約束投影(NH-ORCA,Forward-Biased)──
    STEER_KP: float = 1.5                # ω = Kp·angle_err 的 P gain(原 3.0 太高→稍偏就飽和;降到 1.5 讓繞行溫和)
    REVERSE_ANGLE_THRESHOLD: float = 2.6 # rad (~150°),超過才允許倒車
    FORWARD_DETOUR_BIAS: float = 0.25    # 側向繞行時維持的前進速度 floor

    # ── 側向逃生(ORCA 凍結時)──
    LATERAL_EVASION_V_THRESH: float = 0.08    # v_safe < 此值視為凍結
    LATERAL_EVASION_OBS_SPEED: float = 0.3    # 障礙物速度低於此不視為威脅
    LATERAL_EVASION_TTC: float = 4.0          # TTC 窗(s)
    LATERAL_EVASION_MISS_DIST: float = 1.2    # 垂直錯過距離閾值(robot_r+obs_r+margin)
    LATERAL_EVASION_SPEED: float = 0.4        # 注入的逃生速度大小

    def __init__(
        self,
        robot_radius: float = 0.35,
        safety_margin: float = 0.10,
        max_speed: float = 1.0,
        neighbor_dist: float = 10.0,
        max_neighbors: int = 20,
        time_horizon_dynamic: float = 2.5,
        time_horizon_static: float = 0.3,
        static_vel_threshold: float = 0.15,
        angle_threshold_deg: float = 60.0,
        omega_max: float = 0.25 * math.pi,
        obs_inflation: float = 1.8,
        ego_responsibility: float = 1.0,
    ):
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.rvo_radius = robot_radius + safety_margin
        self.max_speed = max_speed
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon_dynamic = time_horizon_dynamic
        self.time_horizon_static = time_horizon_static
        self.static_vel_threshold = static_vel_threshold
        self.angle_threshold = math.radians(angle_threshold_deg)
        self.omega_max = omega_max
        self.obs_inflation = obs_inflation
        self.ego_responsibility = ego_responsibility

    # ────────────────────────────────────────────────────────────────────────
    def solve(self, inp: ORCAInput, dt: float) -> ORCAOutput:
        """ORCA 求解單一環境。

        只處理動態障礙物。靜態完全排除,留給 RL policy。
        若 ORCA 輸出近零速度(凍結),注入垂直逃生速度。
        """
        # Dynamic maxSpeed:必須 >= |v_pref|,否則 RVO2 截斷產生 ghost
        v_pref_mag = math.sqrt(inp.v_pref[0] ** 2 + inp.v_pref[1] ** 2)
        agent_max_speed = max(self.max_speed, v_pref_mag + 0.1)

        # 建 RVO2 simulator(動態時域)
        sim = rvo2.PyRVOSimulator(
            dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon_dynamic,
            self.time_horizon_static,
            self.rvo_radius,
            agent_max_speed,
        )

        # 機器人 agent
        robot_agent = sim.addAgent(inp.robot_pos)
        sim.setAgentVelocity(robot_agent, inp.robot_vel)
        sim.setAgentPrefVelocity(robot_agent, inp.v_pref)

        # 過濾動態障礙物(靜態排除 — ORCA 只處理動態)
        is_static_mask = np.asarray(inp.obs_is_static, dtype=bool)
        dyn_mask = ~is_static_mask
        dyn_idx_np = np.where(dyn_mask)[0]
        dynamic_indices = dyn_idx_np.tolist()
        valid_count = int(dyn_mask.sum())

        if valid_count > 0:
            dyn_pos = inp.obs_pos[dyn_idx_np]
            dyn_vel = inp.obs_vel[dyn_idx_np]
            dyn_r = inp.obs_radii[dyn_idx_np]
            dyn_speed_mag = np.linalg.norm(dyn_vel, axis=1)
            effective_r = dyn_r + self.safety_margin
            if self.obs_inflation > 1.0:
                effective_r = effective_r * self.obs_inflation
            # non-cooperative:障礙物 max_speed≈0(不會減速避讓),ego 扛 100%
            obs_max_speed = dyn_speed_mag * (1.0 - self.ego_responsibility) + 0.01
            th, th_obst = self.time_horizon_dynamic, self.time_horizon_static

            def _add_one_dynamic(args):
                idx_local, pos, vel, eff_r, max_v = args
                pos_t = (float(pos[0]), float(pos[1]))
                vel_t = (float(vel[0]), float(vel[1]))
                sim.addAgent(pos_t, self.neighbor_dist, self.max_neighbors,
                             th, th_obst, float(eff_r), float(max_v), vel_t)
                sim.setAgentPrefVelocity(idx_local + 1, vel_t)
                return None

            list(map(_add_one_dynamic, zip(
                range(valid_count), dyn_pos, dyn_vel, effective_r, obs_max_speed
            )))

        # 無動態障礙物 → 純 pass-through
        if valid_count == 0:
            return ORCAOutput(
                v_safe_linear=inp.v_next_body,
                v_safe_omega=inp.omega_rl,
                orca_active=False,
                v_pref_world=np.array(inp.v_pref),
                v_safe_world=np.array(inp.v_pref),
                vo_cone_data=[],
            )

        sim.doStep()
        v_safe = sim.getAgentVelocity(robot_agent)
        v_safe_arr = np.array(v_safe)

        # 側向逃生:ORCA 說「停」但有快速障礙物逼近 → 注入垂直逃生
        v_safe_mag = math.sqrt(v_safe[0] ** 2 + v_safe[1] ** 2)
        if v_safe_mag < self.LATERAL_EVASION_V_THRESH:
            v_safe_arr = self._lateral_evasion_if_frozen(v_safe_arr, inp, dynamic_indices)

        # 介入判定
        v_pref_arr = np.array(inp.v_pref)
        v_diff = float(np.linalg.norm(v_safe_arr - v_pref_arr))
        orca_active = v_diff > self.INTERVENTION_EPSILON

        # 投影到 body frame 或 pass-through
        if orca_active:
            v_safe_linear, v_safe_omega = self._project_to_body_frame(
                (float(v_safe_arr[0]), float(v_safe_arr[1])), inp.yaw, dt
            )
        else:
            v_safe_linear = inp.v_next_body
            v_safe_omega = inp.omega_rl

        # VO cone 可視化資料(所有動態障礙物)
        if dynamic_indices:
            dyn_idx = np.asarray(dynamic_indices, dtype=np.int64)
            robot_xy = np.asarray(inp.robot_pos)
            rel_xy_all = inp.obs_pos[dyn_idx] - robot_xy
            combined_r_all = self.rvo_radius + inp.obs_radii[dyn_idx] + self.safety_margin
            vo_cone_data = list(zip(
                [r.copy() for r in rel_xy_all],
                combined_r_all.tolist(),
            ))
        else:
            vo_cone_data = []

        return ORCAOutput(
            v_safe_linear=v_safe_linear,
            v_safe_omega=v_safe_omega,
            orca_active=orca_active,
            v_pref_world=v_pref_arr.copy(),
            v_safe_world=v_safe_arr.copy(),
            vo_cone_data=vo_cone_data,
        )

    # ────────────────────────────────────────────────────────────────────────
    def _lateral_evasion_if_frozen(
        self, v_safe_arr: np.ndarray, inp: ORCAInput, dynamic_indices: list[int]
    ) -> np.ndarray:
        """ORCA 輸出 ≈0 時,若 ego 在快速逼近障礙物的碰撞軌跡上,注入垂直逃生速度。

        避免「原地煞車讓側向快速障礙物撞上來」的坐以待斃失效模式。
        """
        if not dynamic_indices:
            return v_safe_arr

        dyn_idx = np.asarray(dynamic_indices, dtype=np.int64)
        robot_xy = np.asarray(inp.robot_pos, dtype=np.float64)
        obs_xy = inp.obs_pos[dyn_idx]
        obs_v = inp.obs_vel[dyn_idx]
        obs_r = inp.obs_radii[dyn_idx]
        obs_speed = np.linalg.norm(obs_v, axis=1)

        valid_speed = obs_speed >= self.LATERAL_EVASION_OBS_SPEED
        safe_speed = np.where(obs_speed > 1e-6, obs_speed, 1.0)
        obs_dir = obs_v / safe_speed[:, None]

        rel_pos = robot_xy[None, :] - obs_xy
        along = (rel_pos * obs_dir).sum(axis=1)
        ttc = along / safe_speed
        valid_dir = along >= 0                       # 障礙朝 ego 來
        valid_ttc = ttc <= self.LATERAL_EVASION_TTC

        perp_dist = np.abs(rel_pos[:, 0] * (-obs_dir[:, 1]) + rel_pos[:, 1] * obs_dir[:, 0])
        combined_r = obs_r + self.rvo_radius
        valid_perp = perp_dist <= (self.LATERAL_EVASION_MISS_DIST + combined_r)

        threats = valid_speed & valid_dir & valid_ttc & valid_perp
        if not threats.any():
            return v_safe_arr

        # 挑 TTC 最小的威脅,注入垂直逃生(選與 v_pref 同側的方向)
        ttc_masked = np.where(threats, ttc, np.inf)
        best_k = int(np.argmin(ttc_masked))
        best_dir = obs_dir[best_k]
        perp_left = np.array([-best_dir[1], best_dir[0]])
        perp_right = np.array([best_dir[1], -best_dir[0]])
        v_pref_arr = np.asarray(inp.v_pref)
        best_perp = perp_left if np.dot(perp_left, v_pref_arr) >= np.dot(perp_right, v_pref_arr) else perp_right

        escape_v = best_perp * self.LATERAL_EVASION_SPEED
        return v_safe_arr + escape_v

    # ────────────────────────────────────────────────────────────────────────
    def _project_to_body_frame(
        self, v_safe: tuple[float, float], yaw: float, dt: float
    ) -> tuple[float, float]:
        """World v_safe → body (v_linear, omega_z)。Forward-Biased NH-ORCA 投影。

        角度偏差對應行為:
            0°-60°    全速前進 + 微轉向
           60°-120°   維持前進(≥25% |v_safe|)+ 大轉向
          120°-150°   降速 + 最大轉向(原地強轉)
          150°-180°   允許倒車(緊急,障礙在正後方)
        """
        vx_safe, vy_safe = v_safe
        v_safe_mag = math.sqrt(vx_safe ** 2 + vy_safe ** 2)

        if v_safe_mag < 0.01:
            return 0.0, 0.0

        angle_v_safe = math.atan2(vy_safe, vx_safe)
        angle_err = math.atan2(
            math.sin(angle_v_safe - yaw),
            math.cos(angle_v_safe - yaw),
        )
        abs_angle_err = abs(angle_err)

        if abs_angle_err < math.radians(60):
            # Zone A:小偏差 — 全速前進,微轉向
            v_linear = v_safe_mag * math.cos(angle_err)
        elif abs_angle_err < self.REVERSE_ANGLE_THRESHOLD:
            # Zone B:大側向偏差 — 維持前進 floor + 大轉向
            cos_component = v_safe_mag * math.cos(angle_err)
            forward_floor = v_safe_mag * self.FORWARD_DETOUR_BIAS
            v_linear = max(cos_component, forward_floor)
        else:
            # Zone C:近反向(>150°)— 允許倒車(緊急)
            v_linear = v_safe_mag * math.cos(angle_err)

        # ω = Kp·angle_err(繞行區 1.5x gain)
        if abs_angle_err > math.radians(60):
            omega_z = self.STEER_KP * 1.5 * angle_err
        else:
            omega_z = self.STEER_KP * angle_err

        # clamp 到物理極限
        v_linear = max(-self.max_speed, min(self.max_speed, v_linear))
        omega_z = max(-self.omega_max, min(self.omega_max, omega_z))

        return v_linear, omega_z
