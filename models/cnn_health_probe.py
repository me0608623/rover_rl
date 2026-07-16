"""CNN 健康探針 — 證明 e2e bundle 的 4 幀 LiDAR CNN 真的在編碼「運動」。

方法：給 extractor 兩個「當前幀相同、只有歷史幀不同」的輸入：
    STATIC   四幀都 1.5m（無運動）
    APPROACH 3.0→2.5→2.0→1.5m（障礙接近）
    RECEDE   0.6→0.8→1.0→1.5m（障礙遠離）
三者「當前幀」都是 1.5m。若 extractor 的 96D 輸出彼此不同 → CNN 有讀時間維度
（單幀網路會輸出相同）。同時做空間對照（不同當前幀必須不同）+ NaN 檢查。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path("/home/aa/rover_rl/src/rover_rl_inference/rover_rl_inference")
BUNDLE = "/home/aa/rover_rl/models/sa4_e2e_fs4_cleanppo_89600.ts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


obs_builder = _load("obs_builder")
model_runtime = _load("model_runtime")

R_MAX, R_ROBOT = 20.0, 0.35
obs_params = obs_builder.ObsParams(robot_radius=R_ROBOT, lidar_num_bins=72)


def dist_to_norm(d_m):
    if d_m < 0.5:
        return 1.0
    return float(np.clip((d_m - R_ROBOT) / (R_MAX - R_ROBOT), 0.0, 1.0))


def make_raw_obs(front_dist_m):
    sweep = np.ones(72, dtype=np.float32)
    for b in range(33, 40):                       # 正前方障礙叢
        sweep[b] = dist_to_norm(front_dist_m)
    return obs_builder.build_obs_raw(
        79, last_accel=0.0, linear_vel=0.0, angular_vel=0.0,
        goal_body_x=3.0, goal_body_y=0.0, lidar_sweep_72=sweep,
        elapsed_s=0.0, params=obs_params, action_history=None,
    )


def main():
    b = model_runtime.load_bundle(BUNDLE)
    blob = b.blob
    assert b.end_to_end and b.frame_stack == 4

    def normed_lidar(d):
        raw = torch.from_numpy(make_raw_obs(d)).float().unsqueeze(0)
        return blob.preprocess(raw)[0, 6:78]      # 正規化後的當前幀 lidar [72]

    def feat_of(frames):
        """frames = [d_t-3, d_t-2, d_t-1, d_t]（公尺）→ extractor 96D 輸出。"""
        d_t3, d_t2, d_t1, d_t = frames
        raw = torch.from_numpy(make_raw_obs(d_t)).float().unsqueeze(0)
        obs79 = blob.preprocess(raw)              # [1,79] normalized
        hist = torch.cat([normed_lidar(d_t1), normed_lidar(d_t2), normed_lidar(d_t3)]).unsqueeze(0)  # [1,216]
        ext_in = torch.cat([obs79, hist], dim=-1)  # [1,295]
        with torch.no_grad():
            return blob.extractor(ext_in)[0]       # [96]

    f_static = feat_of([1.5, 1.5, 1.5, 1.5])
    f_appr = feat_of([3.0, 2.5, 2.0, 1.5])
    f_rec = feat_of([0.6, 0.8, 1.0, 1.5])
    f_space = feat_of([1.0, 1.0, 1.0, 1.0])        # 空間對照：當前幀 1.0m（≠1.5m）

    def d(a, c):
        return float((a - c).norm())

    nrm = float(f_static.norm())
    lidar_static, lidar_appr, lidar_rec = f_static[:64], f_appr[:64], f_rec[:64]

    print("=== CNN 健康探針（e2e bundle extractor 96D = LiDAR64 + state32）===")
    print(f"NaN 檢查: static={bool(torch.isnan(f_static).any())} "
          f"approach={bool(torch.isnan(f_appr).any())} recede={bool(torch.isnan(f_rec).any())}")
    print(f"||feat_static|| = {nrm:.3f}  (非 0 = extractor 有輸出)")
    print()
    print("── 時序敏感度（當前幀都 1.5m，只差歷史→純運動訊號）──")
    print(f"  ||approach − static||          = {d(f_appr,f_static):.4f}")
    print(f"  ||recede   − static||          = {d(f_rec,f_static):.4f}")
    print(f"  ||approach − recede|| (方向)   = {d(f_appr,f_rec):.4f}")
    print(f"  其中 LiDAR 分支(前64D) approach−recede = {d(lidar_appr,lidar_rec):.4f}")
    print()
    print("── 空間對照（不同當前幀，必須不同）──")
    print(f"  ||current1.0 − current1.5||    = {d(f_space,f_static):.4f}")
    print()

    rel = d(f_appr, f_rec) / max(nrm, 1e-6)
    motion_alive = d(f_appr, f_rec) > 1e-3 and d(f_appr, f_static) > 1e-3
    space_alive = d(f_space, f_static) > 1e-3
    no_nan = not bool(torch.isnan(f_static).any())
    print("=== 判定 ===")
    print(f"[1] 無 NaN:                       {'PASS' if no_nan else 'FAIL'}")
    print(f"[2] 空間響應(不同場景→不同特徵):  {'PASS' if space_alive else 'FAIL'}")
    print(f"[3] 時序/運動通道活著(approach≠recede，同當前幀): "
          f"{'PASS' if motion_alive else 'FAIL — CNN 未讀時間維度!'}"
          f"  (方向差佔特徵 {rel*100:.1f}%)")


if __name__ == "__main__":
    main()
