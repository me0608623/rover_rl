"""Policy runtime — 載入 TorchScript bundle 並維護 RNN hidden state.

Bundle 內容（由 export_policy.py 匯出）：
    preprocess: (obs_raw[B, raw_obs_dim]) → obs79d[B, 79]   # normalize + slice
    extractor:  (obs79d) → feat[B, 96]
    rnn:        (feat, hidden[1, B, H]) → (preprocess_feat[B, P], new_hidden)
    policy:     (rl_input[B, 79+P]) → logits[B, 38]
    meta:       raw_obs_dim, used_obs_dim, hidden_dim, preprocess_dim, total_logits

執行流程：
    obs79 = bundle.preprocess(obs_raw)              # 含 normalize + slice
    feat  = bundle.extractor(obs79)
    pp, h = bundle.rnn(feat, hidden)
    rl_in = cat(obs79, pp) → [B, 79+P]
    logits = bundle.policy(rl_in)

raw_obs_dim 可能是 79（SA1_v2）或 139（SA5/6/7）。runtime 不需關心 — 只要餵入正確維度的 raw obs。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PolicyBundle:
    preprocess: torch.jit.ScriptModule
    extractor: torch.jit.ScriptModule
    rnn: torch.jit.ScriptModule
    policy: torch.jit.ScriptModule
    raw_obs_dim: int
    used_obs_dim: int
    hidden_dim: int
    preprocess_dim: int
    total_logits: int
    device: torch.device
    # E2E frame-stack 專用（RNN bundle 用預設）：
    end_to_end: bool = False
    lidar_hist_dim: int = 0
    frame_stack: int = 1
    blob: object = None


def load_bundle(path: str | Path, device: str = "cpu") -> PolicyBundle:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"policy bundle not found: {p}")

    dev = torch.device(device)
    blob = torch.jit.load(str(p), map_location=dev)
    blob.train(False)

    meta = blob._meta_dims.cpu().tolist()
    # meta 佈局：RNN bundle = 5 entries；E2E bundle = 7 entries，末 2 = [e2e_flag, frame_stack]。
    end_to_end = len(meta) >= 7 and int(meta[5]) == 1
    # E2E bundle 無 rnn 子模組（RNN 被繞過），且 meta[2]=lidar_hist_dim(216) 而非 hidden_dim。
    rnn_mod = None if end_to_end else blob.rnn.train(False)
    return PolicyBundle(
        preprocess=blob.preprocess.train(False),
        extractor=blob.extractor.train(False),
        rnn=rnn_mod,
        policy=blob.policy.train(False),
        raw_obs_dim=int(meta[0]),
        used_obs_dim=int(meta[1]),
        hidden_dim=int(meta[2]),            # e2e：= lidar_hist_dim (216)
        preprocess_dim=int(meta[3]),
        total_logits=int(meta[4]),
        device=dev,
        end_to_end=end_to_end,
        lidar_hist_dim=(int(meta[2]) if end_to_end else 0),
        frame_stack=(int(meta[6]) if end_to_end else 1),
        blob=blob,
    )


class PolicyRunner:
    """單 env 推論 wrapper，維護 recurrent state（RNN=hidden；E2E=LiDAR 疊幀 buffer）."""

    def __init__(self, bundle: PolicyBundle):
        self.bundle = bundle
        if bundle.end_to_end:
            # E2E：state = lidar_hist [1, (K-1)*72]，2D；取代 RNN hidden。
            self.state = torch.zeros(
                1, bundle.lidar_hist_dim, device=bundle.device, dtype=torch.float32
            )
        else:
            # RNN：hidden [1, 1, H]，3D。
            self.state = torch.zeros(
                1, 1, bundle.hidden_dim, device=bundle.device, dtype=torch.float32
            )
        # 診斷用 telemetry（不影響推論）
        self.reset_count = 0
        self.step_count = 0

    def reset(self) -> None:
        # 收到新 goal/path、切 mode、換模型時呼叫：清掉 recurrent state（RNN 記憶 或
        # LiDAR 疊幀歷史），否則殘留 state 會讓 policy 帶著上一段情境做新任務（行為錯亂）。
        self.state.zero_()
        self.reset_count += 1
        self.step_count = 0

    def hidden_norm(self) -> float:
        """recurrent state L2 norm；0=剛重置/待命，>0=episode 內累積中。

        RNN：hidden 記憶量；E2E：LiDAR 疊幀 buffer 填充量（reset 後幾步內從 0 爬升）。
        """
        return float(self.state.norm().item())

    @torch.no_grad()
    def step(self, obs_raw_np: np.ndarray) -> np.ndarray:
        """obs_raw[raw_obs_dim] → logits[38]."""
        b = self.bundle
        if obs_raw_np.shape != (b.raw_obs_dim,):
            raise ValueError(
                f"obs shape {obs_raw_np.shape} != ({b.raw_obs_dim},) "
                f"— bundle expects raw_obs_dim={b.raw_obs_dim}"
            )
        raw = torch.from_numpy(obs_raw_np.astype(np.float32)).unsqueeze(0).to(b.device)
        if b.end_to_end:
            # E2E bundle.forward(obs_raw, lidar_hist) → (logits, new_hist)。
            # 內含 normalize+slice → 4 幀 LiDAR CNN → cat(obs79, feat96) → policy（RNN 繞過）。
            logits, self.state = b.blob(raw, self.state)          # [1, 38], [1, 216]
        else:
            obs79 = b.preprocess(raw)                             # [1, 79]（normalize + 139→79 slice）
            feat = b.extractor(obs79)                             # [1, 96]
            # RNN 吃上一步的 hidden 並回傳新的；逐步覆寫以在整段 episode 內延續時序記憶。
            preprocess, new_hidden = b.rnn(feat, self.state)      # [1, P], [1, 1, H]
            self.state = new_hidden
            # policy head 同時看「當下 obs79」與「RNN 摘要 preprocess」，故 cat 後再餵。
            rl_input = torch.cat([obs79, preprocess], dim=-1)     # [1, 79+P]
            logits = b.policy(rl_input)                           # [1, 38]
        self.step_count += 1
        return logits.squeeze(0).cpu().numpy()
