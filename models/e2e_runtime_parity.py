"""車端：驗證 PolicyRunner(e2e) step 逐 bit 重現 golden oracle（env 0 序列）。

驗證完整 runtime 路徑：load_bundle meta 解析 + PolicyRunner e2e 分支 + reset + step。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np

MR_PATH = "/home/aa/rover_rl/src/rover_rl_inference/rover_rl_inference/model_runtime.py"


def _load_mr():
    spec = importlib.util.spec_from_file_location("model_runtime", MR_PATH)
    mod = importlib.util.module_from_spec(spec)
    # dataclass + PEP563 需模組已在 sys.modules 才能解析註解（正式部署以套件匯入不受影響）。
    sys.modules["model_runtime"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    mr = _load_mr()
    g = np.load(args.golden)
    obs_seq = g["obs_seq"]        # [T, N, 79] RAW
    gold_logits = g["logits"]     # [T, N, 38]
    gold_actions = g["actions"]   # [T, N, 2]
    T, N, _ = obs_seq.shape

    bundle = mr.load_bundle(args.bundle)
    print(f"[rt-parity] end_to_end={bundle.end_to_end} raw_obs={bundle.raw_obs_dim} "
          f"hidden_dim(meta2)={bundle.hidden_dim} lidar_hist={bundle.lidar_hist_dim} "
          f"frame_stack={bundle.frame_stack}")
    assert bundle.end_to_end, "load_bundle 未偵測到 e2e"

    runner = mr.PolicyRunner(bundle)
    runner.reset()   # env 0，state 歸零
    print(f"[rt-parity] reset 後 hidden_norm={runner.hidden_norm():.3f} (應 0)")

    max_dl = 0.0
    mism = 0
    for t in range(T):
        lg = runner.step(obs_seq[t, 0])   # [38]
        dl = float(np.abs(lg - gold_logits[t, 0]).max())
        max_dl = max(max_dl, dl)
        aa = int(lg[:19].argmax())
        aw = int(lg[19:].argmax())
        if aa != int(gold_actions[t, 0, 0]) or aw != int(gold_actions[t, 0, 1]):
            mism += 1

    print(f"[rt-parity] steps={T} (env 0)")
    print(f"[rt-parity] max |delta logits| = {max_dl:.3e}  (tol {args.tol})")
    print(f"[rt-parity] action mismatches = {mism}/{T}")
    print(f"[rt-parity] 末步 hidden_norm={runner.hidden_norm():.3f} (應 >0，buffer 已填)")
    if max_dl <= args.tol and mism == 0:
        print("[rt-parity] PASS — PolicyRunner e2e 路徑逐 bit 重現 golden")
    else:
        print("[rt-parity] FAIL — runtime 與訓練端不一致，禁止上車")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
