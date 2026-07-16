"""節點級整合 smoke test — 走真實部署鏈，不需 rclpy / 實體感測器 / 車輛動作。

鏈路 = build_obs_raw(79D) → PolicyRunner(e2e bundle).step → decode_logits_to_cmd
用 policy_params_e2e.yaml 的校準值 + 合成感測器。含行為健全檢查：
  A 前方淨空 → 期望前進 (cmd_v>0)
  B 正前方障礙 → 期望減速/轉向 (cmd_v 降 或 |cmd_w| 升)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PKG = Path("/home/aa/rover_rl/src/rover_rl_inference/rover_rl_inference")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


obs_builder = _load("obs_builder")
model_runtime = _load("model_runtime")
action_decoder = _load("action_decoder")

BUNDLE = "/home/aa/rover_rl/models/sa4_e2e_fs4_cleanppo_89600.ts"
R_MAX, R_ROBOT = 20.0, 0.35

# e2e config 校準值（policy_params_e2e.yaml）
obs_params = obs_builder.ObsParams(
    max_acceleration=1.0, max_linear_velocity=1.0, max_angular_velocity_obs=1.5,
    robot_radius=R_ROBOT, lidar_num_bins=72, episode_horizon_s=60.0,
)
act_params = action_decoder.ActionParams(
    num_bins=19, max_linear_velocity=1.0, max_linear_accel=0.5,
    max_angular_velocity_action=1.2, dt=0.2, max_angular_accel=3.0,
)


def dist_to_norm(d_m: float) -> float:
    """公尺 → sweep 正規化值 [0,1]；<r_min(0.5) 判盲=遠(1.0)。"""
    if d_m < 0.5:
        return 1.0
    return float(np.clip((d_m - R_ROBOT) / (R_MAX - R_ROBOT), 0.0, 1.0))


def make_sweep(front_obstacle_m=None):
    sweep = np.ones(72, dtype=np.float32)   # 全淨空
    if front_obstacle_m is not None:
        # 正前方 0° ≈ bin 36（-180°..+175°，5°/bin）；放一叢障礙
        for b in range(33, 40):
            sweep[b] = dist_to_norm(front_obstacle_m)
    return sweep


def run_scenario(tag, front_obstacle_m, ticks=6):
    bundle = model_runtime.load_bundle(BUNDLE)
    runner = model_runtime.PolicyRunner(bundle)
    runner.reset()
    v = 0.0
    last_cmd = (0.0, 0.0)
    for t in range(ticks):
        sweep = make_sweep(front_obstacle_m)
        obs = obs_builder.build_obs_raw(
            bundle.raw_obs_dim,
            last_accel=0.0, linear_vel=v, angular_vel=0.0,
            goal_body_x=3.0, goal_body_y=0.0,   # 目標正前方 3m
            lidar_sweep_72=sweep, elapsed_s=t * 0.2,
            params=obs_params, action_history=None,
        )
        assert obs.shape == (79,), f"obs shape {obs.shape}"
        logits = runner.step(obs)
        assert logits.shape == (38,), f"logits {logits.shape}"
        cmd_v, cmd_w, accel = action_decoder.decode_logits_to_cmd(
            logits, current_linear_vel=v, params=act_params,
            deterministic=True, current_angular_vel=last_cmd[1],
        )
        v = cmd_v
        last_cmd = (cmd_v, cmd_w)
    print(f"  [{tag}] 末拍 cmd_v={cmd_v:+.3f} m/s  cmd_w={cmd_w:+.3f} rad/s  "
          f"buffer_norm={runner.hidden_norm():.1f}")
    return cmd_v, cmd_w


def main():
    print("=== e2e 節點級 smoke（build_obs_raw → runner.step → decode） ===")
    print(f"bundle raw_obs_dim / e2e / frame_stack 由 load_bundle 偵測：")
    b = model_runtime.load_bundle(BUNDLE)
    print(f"  raw_obs_dim={b.raw_obs_dim} end_to_end={b.end_to_end} "
          f"frame_stack={b.frame_stack} lidar_hist={b.lidar_hist_dim}")
    assert b.raw_obs_dim == 79 and b.end_to_end and b.frame_stack == 4

    print("--- 情境 A：前方淨空、目標正前 3m ---")
    va, wa = run_scenario("clear", None)
    print("--- 情境 B：正前方 1.3m 障礙、目標正前 3m ---")
    vb, wb = run_scenario("obstacle@1.3m", 1.3)

    print("\n=== 健全判定 ===")
    ok_pipeline = True   # 走到這裡代表 obs/step/decode 全程無錯、維度對
    print(f"[1] pipeline 無錯 + 維度對 (obs 79D / logits 38 / cmd 有值): PASS")
    forward = va > 0.05
    print(f"[2] 淨空前進 cmd_v={va:+.3f}>0.05: {'PASS' if forward else 'WARN'}")
    reacts = (vb < va - 0.02) or (abs(wb) > abs(wa) + 0.02)
    print(f"[3] 障礙時反應（減速 {vb:+.3f}<{va:+.3f} 或 轉向 |{wb:+.3f}|>|{wa:+.3f}|）: "
          f"{'PASS' if reacts else 'WARN（行為未如預期，查 bin 慣例/需實測確認）'}")
    print("\n注意：此為離線行為健全 smoke，非避障驗收；合成 sweep 與真實分布有差，"
          "WARN 不代表模型壞，實體 bring-up 仍以架空實測為準。")


if __name__ == "__main__":
    main()
