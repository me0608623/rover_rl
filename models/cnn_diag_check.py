import importlib.util
import sys
from pathlib import Path

import numpy as np

PKG = Path("/home/aa/rover_rl/src/rover_rl_inference/rover_rl_inference")


def L(n):
    sp = importlib.util.spec_from_file_location(n, PKG / f"{n}.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m
    sp.loader.exec_module(m)
    return m


ob = L("obs_builder")
mr = L("model_runtime")
op = ob.ObsParams(robot_radius=0.35, lidar_num_bins=72)


def raw(front):
    sw = np.ones(72, np.float32)
    if front is not None:
        val = max((front - 0.35) / 19.65, 0.0) if front >= 0.5 else 1.0
        for b in range(33, 40):
            sw[b] = val
    return ob.build_obs_raw(79, last_accel=0., linear_vel=0., angular_vel=0.,
                            goal_body_x=3., goal_body_y=0., lidar_sweep_72=sw,
                            elapsed_s=0., params=op, action_history=None)


b = mr.load_bundle("/home/aa/rover_rl/models/sa4_e2e_fs4_cleanppo_89600.ts")
r = mr.PolicyRunner(b)
r.reset()
print("reset 後:", r.cnn_diag(), " buf_norm(rnn_norm)=", round(r.hidden_norm(), 2))
print("--- 障礙接近序列 3.0→2.5→2.0→1.5，再靜止兩步 ---")
for i, d in enumerate([3.0, 2.5, 2.0, 1.5, 1.5, 1.5]):
    r.step(raw(d))
    cd = r.cnn_diag()
    bn = round(r.hidden_norm(), 2)
    print(f" step{i} front={d}m  buf_fill={cd['buf_fill']}  "
          f"frame_motion={cd['frame_motion']}  buf_norm={bn}")
print("預期: buf_fill 0→.33→.67→1.0；移動時 frame_motion>0，靜止後 frame_motion→0")
