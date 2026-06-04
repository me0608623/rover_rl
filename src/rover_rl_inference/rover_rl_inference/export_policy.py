"""Export training checkpoint → TorchScript bundle for ROS 2 deployment.

支援兩種訓練架構（自動偵測）：

A. **79D 新架構 (SA1_v2 等)**：
   env obs 已是 79D，obs_normalizer 也是 79D。policy_obs_indices = [0..78]。

B. **139D 舊架構 (SA5/6/7 等 T-Corridor / dense 系列)**：
   env obs 是 139D（含 60D obstacle ground truth），obs_normalizer 是 139D。
   policy 實際只看 79D：indices = [0..77, 138]。部署時 obstacle 區段補 0。

Bundle 結構：
    blob.preprocess: (obs_raw[B, D_raw]) → obs79d[B, 79]    # 內含 normalizer + slice
    blob.extractor:  (obs79d) → feat[B, 96]
    blob.rnn:        (feat, hidden[1,B,H]) → (preprocess[B, P], new_hidden)
    blob.policy:     (rl_input[B, 79+P]) → logits[B, 38]
    blob.raw_obs_dim / used_obs_dim / hidden_dim / preprocess_dim / total_logits

用法：
    python -m rover_rl_inference.export_policy \
        --checkpoint /path/to/checkpoint_xxx.pt \
        --output /path/to/policy.ts
    # 架構參數自動從 checkpoint['args'] 與 obs_normalizer 推斷；可用 --override 強制覆寫
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_PATH = REPO_ROOT / "scripts/reinforcement_learning/skrl/models/modular_rnn_models.py"


def _load_training_models():
    # 直接從訓練 repo 載入網路定義（不複製一份到部署端），確保部署用的 class
    # 與訓練時 100% 相同；checkpoint 的 state_dict 才能正確 load 進來
    spec = importlib.util.spec_from_file_location("modular_rnn_models", MODELS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training models from {MODELS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modular_rnn_models"] = mod
    spec.loader.exec_module(mod)
    return mod


class PreprocessNormalizer(nn.Module):
    """Raw obs (139D 或 79D) → normalize → slice 79D for policy.

    若訓練 obs = 79D，slice indices = [0..78]（恆等）。
    若訓練 obs = 139D，slice indices = [0..77, 138]（跳過 60D obstacle）。
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor,
                 policy_indices: list[int], clip: float = 5.0):
        super().__init__()
        # 把訓練時的 obs_normalizer (mean/std) 與 slice 索引「bake」成 buffer 存進模型。
        # 這是匯出的關鍵：部署端 ROS 節點只丟 raw obs 進來，正規化完全內含在 .ts，
        # 不需要在部署端另外維護一份 mean/var（避免漏帶或數值不一致導致 cmd 爆走）
        self.register_buffer("obs_mean", mean.clone())
        self.register_buffer("obs_std", std.clone())
        self.register_buffer(
            "policy_idx",
            torch.tensor(policy_indices, dtype=torch.long),
        )
        self.clip = float(clip)

    def forward(self, obs_raw: torch.Tensor) -> torch.Tensor:
        # 標準化後 clip 到 ±5（與訓練端相同），再 index_select 抽出 policy 真正看的 79D
        normed = torch.clamp(
            (obs_raw - self.obs_mean) / (self.obs_std + 1e-8),
            -self.clip, self.clip,
        )
        return normed.index_select(-1, self.policy_idx)


class RNNStep(nn.Module):
    """把訓練端 PreprocessRNN 拆成「單步」版本供部署用.

    訓練時 RNN 吃整段序列；部署是每個 control tick 跑一步並把 hidden state
    傳回給呼叫端自己保管（episode 內延續）。這裡只抽出需要的子模組重組 forward，
    讓 hidden 變成顯式輸入/輸出，避免 TorchScript 內部 stateful 的麻煩。
    """

    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.fc_front = rnn.fc_front
        self.rnn = rnn.rnn
        self.fc_middle = rnn.fc_middle
        self.concat_rnn = bool(rnn.concat_rnn)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        fc_out = self.fc_front(features)
        rnn_in = fc_out.unsqueeze(0)  # 補上 seq_len=1 維度餵 nn.RNN
        rnn_out, new_hidden = self.rnn(rnn_in, hidden)
        rnn_out = rnn_out.squeeze(0)
        if self.concat_rnn:
            combined = torch.cat([rnn_out, fc_out], dim=-1)
        else:
            combined = rnn_out
        return self.fc_middle(combined), new_hidden


class PolicyOnly(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.net = head.net

    def forward(self, rl_input: torch.Tensor) -> torch.Tensor:
        return self.net(rl_input)


class Bundle(nn.Module):
    """Self-contained policy bundle. Meta dims stored as registered buffers
    so they survive torch.jit.script without type annotation pitfalls.

    把 preprocess / extractor / rnn / policy 四段封成單一可儲存模型。
    維度 meta（raw_obs_dim 等）刻意用 register_buffer 存成 tensor 而非 Python int 屬性：
    torch.jit.script 對純 Python int 屬性的型別推斷常出錯，存成 buffer 可安全保留。
    """

    def __init__(self, preprocess, extractor, rnn, policy, raw_obs_dim,
                 used_obs_dim, hidden_dim, preprocess_dim, total_logits):
        super().__init__()
        self.preprocess = preprocess
        self.extractor = extractor
        self.rnn = rnn
        self.policy = policy
        self.register_buffer("_meta_dims",
                             torch.tensor([raw_obs_dim, used_obs_dim, hidden_dim,
                                           preprocess_dim, total_logits],
                                          dtype=torch.int64))

    def forward(self, obs_raw: torch.Tensor, hidden: torch.Tensor):
        obs79 = self.preprocess(obs_raw)
        feat = self.extractor(obs79)
        pp, new_hidden = self.rnn(feat, hidden)
        rl_in = torch.cat([obs79, pp], dim=-1)
        logits = self.policy(rl_in)
        return logits, new_hidden


def _autodetect(ckpt: dict, overrides: dict) -> dict:
    """從 checkpoint 的 args 與 obs_normalizer 形狀自動推斷網路架構參數.

    讓同一支匯出腳本能吃 79D 新架構與 139D 舊架構（見模組 docstring），
    不必每個 model 手動指定維度；任何推斷可被 --override 旗標覆寫。
    """
    args = ckpt.get("args", {}) or {}
    norm = ckpt.get("obs_normalizer", {}) or {}
    mean = norm.get("mean")
    # 用 normalizer mean 的長度判定 raw obs 維度（這是區分 79D / 139D 架構的依據）
    raw_obs_dim = int(mean.numel()) if mean is not None and hasattr(mean, "numel") else 79

    cfg = {
        "raw_obs_dim": raw_obs_dim,
        "used_obs_dim": 79,
        "hidden_dim": int(args.get("hidden_dim") or 30),
        "preprocess_dim": int(args.get("preprocess_dim") or 12),
        "fc_dim": int(args.get("fc_dim") or 48),
        "middle_dim": int(args.get("wd_middle_dim") or 0),
        # predict_dim 訓練側預設 7（WD value）；SA1_v2 用 13
        "predict_dim": int(args.get("predict_dim") or 7),
        "rnn_type": str(args.get("rnn_type") or "RNN"),
    }
    cfg.update(overrides)

    # 決定從 raw obs 抽哪些維度給 policy：
    # 79D → 全取（恆等）；139D → 取前 78D + 第 138D，跳過中間 60D obstacle ground truth
    # （該欄位 sim 才有、實車拿不到，部署時補 0，故匯出時也排除）
    if raw_obs_dim == 79:
        cfg["policy_indices"] = list(range(0, 79))
    elif raw_obs_dim == 139:
        cfg["policy_indices"] = list(range(0, 78)) + [138]
    else:
        raise ValueError(
            f"unknown obs layout: raw_obs_dim={raw_obs_dim}; "
            f"expected 79 or 139. Override via flags if needed."
        )
    return cfg


def export(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for k in ("extractor", "preprocess_rnn", "policy_head"):
        if k not in ckpt:
            raise KeyError(
                f"checkpoint missing {k}. Available: {sorted(ckpt.keys())}"
            )

    overrides: dict = {}
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.preprocess_dim is not None:
        overrides["preprocess_dim"] = args.preprocess_dim
    if args.fc_dim is not None:
        overrides["fc_dim"] = args.fc_dim
    if args.middle_dim is not None:
        overrides["middle_dim"] = args.middle_dim
    if args.predict_dim is not None:
        overrides["predict_dim"] = args.predict_dim
    if args.rnn_type is not None:
        overrides["rnn_type"] = args.rnn_type
    cfg = _autodetect(ckpt, overrides)

    print("[CFG] autodetected:")
    for k, v in cfg.items():
        if k == "policy_indices":
            print(f"      policy_indices: len={len(v)} ({'139→79 slice' if cfg['raw_obs_dim']==139 else 'identity'})")
        else:
            print(f"      {k}: {v}")

    mm = _load_training_models()
    extractor = mm.LidarStateExtractor()
    rnn = mm.PreprocessRNN(
        input_dim=extractor.output_dim,
        fc_dim=cfg["fc_dim"],
        hidden_dim=cfg["hidden_dim"],
        preprocess_dim=cfg["preprocess_dim"],
        predict_dim=cfg["predict_dim"],
        middle_dim=cfg["middle_dim"] if cfg["middle_dim"] > 0 else None,
        concat_rnn=True,
        rnn_type=cfg["rnn_type"],
    )
    head_input_dim = cfg["used_obs_dim"] + cfg["preprocess_dim"]
    policy_head = mm.PolicyHead(input_dim=head_input_dim)

    extractor.load_state_dict(ckpt["extractor"])
    rnn.load_state_dict(ckpt["preprocess_rnn"])
    policy_head.load_state_dict(ckpt["policy_head"])

    # normalizer 存的是 var，需開根號還原 std。沒帶 normalizer 的 checkpoint 退化成恆等
    # （mean=0,std=1）——這通常代表 obs 沒被正規化過，部署後極易 cmd 爆走，故印 WARN
    norm = ckpt.get("obs_normalizer", {})
    if norm and "mean" in norm:
        mean = norm["mean"].float().cpu()
        std = norm["var"].float().cpu().sqrt()
    else:
        print("[WARN] no obs_normalizer; identity (mean=0, std=1)")
        mean = torch.zeros(cfg["raw_obs_dim"])
        std = torch.ones(cfg["raw_obs_dim"])
    if mean.numel() != cfg["raw_obs_dim"]:
        raise ValueError(
            f"normalizer dim {mean.numel()} != raw_obs_dim {cfg['raw_obs_dim']}"
        )

    pre = PreprocessNormalizer(mean, std, cfg["policy_indices"]).eval()
    ext = extractor.eval()
    rnn_step = RNNStep(rnn.eval()).eval()
    pol = PolicyOnly(policy_head.eval()).eval()

    # 各子模組先用 jit.trace（餵 dummy 輸入記錄計算圖），整體 Bundle 再用 jit.script。
    # 採 trace+script 混合：子模組無控制流，trace 足夠且穩；外層 Bundle 用 script
    # 保留 meta buffer 與明確的多輸出 forward 介面
    sample_raw = torch.zeros(1, cfg["raw_obs_dim"])
    sample_hidden = torch.zeros(1, 1, cfg["hidden_dim"])
    with torch.no_grad():
        traced_pre = torch.jit.trace(pre, sample_raw)
        traced_ext = torch.jit.trace(ext, torch.zeros(1, cfg["used_obs_dim"]))
        traced_rnn = torch.jit.trace(rnn_step,
                                      (torch.zeros(1, extractor.output_dim), sample_hidden))
        traced_pol = torch.jit.trace(pol, torch.zeros(1, head_input_dim))

    bundle = Bundle(
        preprocess=traced_pre,
        extractor=traced_ext,
        rnn=traced_rnn,
        policy=traced_pol,
        raw_obs_dim=cfg["raw_obs_dim"],
        used_obs_dim=cfg["used_obs_dim"],
        hidden_dim=cfg["hidden_dim"],
        preprocess_dim=cfg["preprocess_dim"],
        total_logits=2 * 19,
    )
    scripted = torch.jit.script(bundle)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    # Sanity：重新 load 存好的 .ts 跑一遍完整 pipeline，確認 logits 形狀=38（19×2 MultiDiscrete）。
    # 在匯出端就抓出維度錯誤，避免上車才發現 .ts 壞掉
    reloaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        o79 = reloaded.preprocess(sample_raw)
        f = reloaded.extractor(o79)
        p, h = reloaded.rnn(f, sample_hidden)
        rl_in = torch.cat([o79, p], dim=-1)
        logits = reloaded.policy(rl_in)
    assert logits.shape == (1, 38), f"logits {logits.shape}"
    print(f"[OK] exported: {out_path}")
    print(f"     raw_obs={cfg['raw_obs_dim']} → used={cfg['used_obs_dim']}, "
          f"hidden={cfg['hidden_dim']}, preprocess={cfg['preprocess_dim']}, logits=38")


def build_parser():
    p = argparse.ArgumentParser(description="Export checkpoint → TorchScript bundle (auto-detect)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden-dim", type=int, default=None, help="override")
    p.add_argument("--preprocess-dim", type=int, default=None)
    p.add_argument("--fc-dim", type=int, default=None)
    p.add_argument("--middle-dim", type=int, default=None,
                   help="0 = no middle FC (SA5/6/7); 48 = SA1_v2")
    p.add_argument("--predict-dim", type=int, default=None,
                   help="7=SA5/6/7 (WD default), 13=SA1_v2")
    p.add_argument("--rnn-type", choices=["RNN", "GRU"], default=None)
    return p


def main():
    export(build_parser().parse_args())


if __name__ == "__main__":
    main()
