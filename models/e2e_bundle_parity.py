"""車端：驗證 E2EBundle TorchScript 逐 bit 重現訓練機 golden oracle。

用法:
    python e2e_bundle_parity.py --bundle <e2e.ts> --golden <golden.npz>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    g = np.load(args.golden)
    obs_seq = torch.from_numpy(g["obs_seq"]).float()   # [T, N, 79] RAW
    gold_logits = torch.from_numpy(g["logits"]).float()  # [T, N, 38]
    gold_actions = torch.from_numpy(g["actions"]).long()  # [T, N, 2]
    K = int(g["frame_stack"])
    T, N, _ = obs_seq.shape
    hist_dim = (K - 1) * 72

    bundle = torch.jit.load(args.bundle)
    meta = bundle._meta_dims
    print(f"[parity] bundle meta={meta.tolist()}  K={K} hist_dim={hist_dim}")
    assert int(meta[5]) == 1, "bundle 非 e2e (meta[5]!=1)"
    assert int(meta[2]) == hist_dim, f"meta lidar_hist_dim {int(meta[2])} != {hist_dim}"

    state = torch.zeros(N, hist_dim)
    max_dl = 0.0
    act_mismatch = 0
    with torch.no_grad():
        for t in range(T):
            logits, state = bundle(obs_seq[t], state)      # [N,38],[N,hist]
            dl = (logits - gold_logits[t]).abs().max().item()
            max_dl = max(max_dl, dl)
            act_a = logits[:, :19].argmax(-1)
            act_w = logits[:, 19:].argmax(-1)
            act = torch.stack([act_a, act_w], dim=-1)
            act_mismatch += int((act != gold_actions[t]).any(-1).sum())

    print(f"[parity] steps={T} envs={N}")
    print(f"[parity] max |delta logits| = {max_dl:.3e}  (tol {args.tol})")
    print(f"[parity] action mismatches = {act_mismatch}/{T*N}")
    if max_dl <= args.tol and act_mismatch == 0:
        print("[parity] PASS — bundle 逐 bit 重現 golden oracle")
    else:
        print("[parity] FAIL — bundle 與訓練端不一致，禁止上車")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
