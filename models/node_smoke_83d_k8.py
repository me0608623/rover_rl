"""83D/K8 家族節點級 smoke — 走真實部署鏈，不需 rclpy / 感測器 / 車輛動作。

適用 SA1-R1 c1700 / SA4-R3 it50 / SA4-R2 c6400（契約逐字相同，只換模型與 yaml）。

鏈路（與 policy_node._tick_inference 同順序）：
    build_obs_raw(83D, act_hist) → PolicyRunner(e2e K=8).step
      → decode_logits_to_cmd → allow_reverse clamp → push act_hist(÷0.5, ÷1.2, clip±2)

校準值**全部從對應的 policy_params_<variant>.yaml 讀**，不在此檔重打一份：這支 smoke 同時是
「yaml 的值餵進真實鏈路會怎樣」的檢查，重打就失去意義。

⚠ 這是離線健全 smoke，不是避障驗收、也不是延遲極限環的判據。
   合成 sweep 與真實 VLP-16 分布有差、此處無底盤動力學與 200ms 致動延遲。
   實體 bring-up 仍以「架空實測 rl_w 波形」為準（handoff §5）。

用法：python3 models/node_smoke_83d_k8.py [sa1r1|sa4r3|sa4r2]   （預設 sa1r1）
"""
from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/home/aa/rover_rl")
PKG = ROOT / "src/rover_rl_inference/rover_rl_inference"

# variant → .ts 檔名（與 test_deploy_contract_83d_k8.py 的 MODELS 表一致）
VARIANTS = {
    "sa1r1": "sa1r1_c1700_83d_k8.ts",
    "sa4r3": "sa4_r3_it50_83d_k8.ts",
    "sa4r2": "sa4_r2_c6400_83d_k8.ts",
}
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "sa1r1"
if VARIANT not in VARIANTS:
    raise SystemExit(f"未知 variant {VARIANT!r}；可用：{', '.join(VARIANTS)}")
BUNDLE = ROOT / "models" / VARIANTS[VARIANT]
POLICY_YAML = ROOT / f"src/rover_rl_bringup/config/policy_params_{VARIANT}.yaml"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


obs_builder = _load("obs_builder")
model_runtime = _load("model_runtime")
action_decoder = _load("action_decoder")

P = yaml.safe_load(POLICY_YAML.read_text(encoding="utf-8"))["rover_rl_policy"]["ros__parameters"]

R_MAX = float(P["lidar_r_max_m"])
R_ROBOT = float(P["robot_radius_m"])
R_MIN = float(P["lidar_r_min_m"])
RATE = float(P["speed_rate"])
INV = 1.0 / RATE
A_NORM = float(P["act_stack_a_max"])
W_NORM = float(P["act_stack_omega_max"])
ALLOW_REVERSE = bool(P["allow_reverse"])
DT = float(P["control_dt"])

obs_params = obs_builder.ObsParams(
    max_acceleration=float(P["obs_max_acceleration"]),
    max_linear_velocity=float(P["obs_max_linear_velocity"]),
    max_angular_velocity_obs=float(P["obs_max_angular_velocity"]),
    robot_radius=R_ROBOT,
    lidar_num_bins=72,
    episode_horizon_s=float(P["episode_horizon_s"]),
)
# 動作上限 ×rate（時間膨脹的動作端），與 policy_node 的 act_eff 一致
act_params = action_decoder.ActionParams(
    num_bins=19,
    max_linear_velocity=float(P["act_max_linear_velocity"]) * RATE,
    max_linear_accel=float(P["act_max_linear_accel"]) * RATE,
    max_angular_velocity_action=float(P["act_max_angular_velocity"]) * RATE,
    dt=DT,
    max_angular_accel=float(P["act_max_angular_accel"]) * RATE,
    reverse_velocity_scale=float(P["reverse_velocity_scale"]),
)


def dist_to_norm(d_m: float) -> float:
    """公尺 → sweep 正規化值 [0,1]；< r_min 判盲區＝最遠(1.0)，與 preprocessor 同慣例。"""
    if d_m < R_MIN:
        return 1.0
    return float(np.clip((d_m - R_ROBOT) / (R_MAX - R_ROBOT), 0.0, 1.0))


def make_sweep(front_obstacle_m=None):
    sweep = np.ones(72, dtype=np.float32)          # 全淨空
    if front_obstacle_m is not None:
        for b in range(33, 40):                    # 正前方 0° ≈ bin 36（5°/bin）
            sweep[b] = dist_to_norm(front_obstacle_m)
    return sweep


def run_scenario(tag, front_obstacle_m, ticks=15):
    bundle = model_runtime.load_bundle(str(BUNDLE))
    runner = model_runtime.PolicyRunner(bundle)
    runner.reset()
    hist = collections.deque([np.zeros(2, np.float32)] * 2, maxlen=2)  # newest-first
    v = 0.0
    prev_w = 0.0
    ws, vs, hist_absmax = [], [], 0.0
    for t in range(ticks):
        obs = obs_builder.build_obs_raw(
            bundle.raw_obs_dim,
            last_accel=0.0, linear_vel=v * INV, angular_vel=prev_w * INV,
            goal_body_x=3.0 * INV, goal_body_y=0.0,       # 目標正前方 3m
            lidar_sweep_72=make_sweep(front_obstacle_m), elapsed_s=t * DT,
            params=obs_params,
            action_history=np.concatenate(list(hist)).astype(np.float32),
        )
        assert obs.shape == (83,), f"obs shape {obs.shape}"
        hist_absmax = max(hist_absmax, float(np.abs(obs[79:83]).max()))
        logits = runner.step(obs)
        assert logits.shape == (38,), f"logits {logits.shape}"
        cmd_v, cmd_w, accel = action_decoder.decode_logits_to_cmd(
            logits, current_linear_vel=v, params=act_params,
            deterministic=True, current_angular_vel=prev_w,
        )
        if not ALLOW_REVERSE and cmd_v < 0.0:        # 與 policy_node 同一層 clamp
            cmd_v = 0.0
            accel = (cmd_v - v) / max(DT, 1e-6)
        # issued 層 = 上面這組；act_hist 記的就是它（÷ 分母、clip ±2、newest-first）
        hist.appendleft(np.array([
            np.clip(accel * INV / A_NORM, -2.0, 2.0),
            np.clip(cmd_w * INV / W_NORM, -2.0, 2.0),
        ], dtype=np.float32))
        v, prev_w = cmd_v, cmd_w
        vs.append(cmd_v)
        ws.append(cmd_w)
    flips = sum(1 for i in range(1, len(ws))
                if ws[i] * ws[i - 1] < 0 and min(abs(ws[i]), abs(ws[i - 1])) > 0.05)
    print(f"  [{tag}] v: " + " ".join(f"{x:+.2f}" for x in vs))
    print(f"  [{tag}] ω: " + " ".join(f"{x:+.2f}" for x in ws))
    print(f"  [{tag}] 末拍 cmd_v={vs[-1]:+.3f} cmd_w={ws[-1]:+.3f}  "
          f"|ω|max={max(abs(x) for x in ws):.3f}  ω 變號 {flips} 次  "
          f"act_hist |max|={hist_absmax:.3f}  buffer_norm={runner.hidden_norm():.1f}")
    return vs[-1], ws[-1], hist_absmax


def main():
    print("=== 83D/K8 節點級 smoke（83D + act_hist → K=8 CNN → decode）===")
    print(f"variant={VARIANT}  model={BUNDLE.name}\nyaml={POLICY_YAML.name}  speed_rate={RATE}  allow_reverse={ALLOW_REVERSE}")
    print(f"act_hist 分母 a={A_NORM:g} ω={W_NORM:g}   r_min={R_MIN} r_max={R_MAX}")
    b = model_runtime.load_bundle(str(BUNDLE))
    print(f"  raw_obs_dim={b.raw_obs_dim} end_to_end={b.end_to_end} "
          f"frame_stack={b.frame_stack} lidar_hist={b.lidar_hist_dim}")
    assert b.raw_obs_dim == 83 and b.end_to_end and b.frame_stack == 8
    assert b.lidar_hist_dim == 504

    print("--- 情境 A：前方淨空、目標正前 3m ---")
    va, wa, ha = run_scenario("clear", None)
    print("--- 情境 B：正前方 1.3m 障礙、目標正前 3m ---")
    vb, wb, hb = run_scenario("obstacle@1.3m", 1.3)

    print("\n=== 健全判定 ===")
    print("[1] pipeline 無錯 + 維度對 (obs 83D / logits 38 / hist 4D): PASS")
    forward = va > 0.05
    print(f"[2] 淨空前進 cmd_v={va:+.3f}>0.05: {'PASS' if forward else 'WARN'}")
    reacts = (vb < va - 0.02) or (abs(wb) > abs(wa) + 0.02)
    print(f"[3] 障礙時反應（減速 {vb:+.3f}<{va:+.3f} 或 轉向 |{wb:+.3f}|>|{wa:+.3f}|）: "
          f"{'PASS' if reacts else 'WARN（合成 sweep，非避障驗收）'}")
    scale_ok = max(ha, hb) <= 1.0 + 1e-6
    print(f"[4] act_hist 落在 ±1（分母 = 動作上限，不該撞 ±2 clip）"
          f" |max|={max(ha, hb):.3f}: {'PASS' if scale_ok else 'FAIL — 分母尺度可疑'}")
    print("\n注意：離線 smoke，無底盤動力學與 200ms 致動延遲；ω 變號次數只能當粗略提示，"
          "極限環判定一律以架空實測為準（handoff §5）。")


if __name__ == "__main__":
    main()
