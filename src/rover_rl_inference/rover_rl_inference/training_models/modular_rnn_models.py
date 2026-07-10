"""
Modular RNN Models — 移植自 Warp Drive CustomModuleConnected

核心設計 (來源: new_warp_drive/custom_envs/module_connected.py):
  1. PreprocessRNN: FC → RNN → concat → middle FC → predict head
  2. Gradient detach: preprocess output 傳給 RL head 前切斷梯度
     WD 有 3 個 .detach():
       - line 504: rl_ip = ip.detach()  (obs 副本給 RL concat)
       - line 537: memory_reg.detach()  (hidden state 存儲)
       - line 573: concat_input = rl_in_.detach()  (RL 最終輸入)
     效果: RL loss.backward() 不會流到 preprocess/RNN
  3. Auxiliary loss: RNN 只被 module prediction loss 訓練
     WD target: 7D preprocess_real_data (2 nearest neighbors × (x,y,dist) + timestep)
     WD loss:   log(clamp(L1, 0.01)) × per-dim weight
  4. RL head: 只看 detached features，PPO 只訓練 head

架構:
  Obs (139D env output, 只用 79D)
    → LidarStateExtractor (LiDAR Conv1d 64D + StateMLP 32D = 96D)
      [IsaacLab adaptation] WD 用 Linear(113→64)+ReLU 單層; IL 用 Conv1d+MLP 2-branch
    → PreprocessRNN (FC→RNN→concat→FC = 12D) + .detach()
    → cat(79D_obs, 12D_preprocess) = 91D
    → PolicyHead (91→256→256→256→512→38)
    → ValueHead (91→256→256→256→512→512→1)

WD 對齊參數:
  - RNN type: vanilla RNN (not GRU)
  - module_connect_dim (preprocess_dim): 12
  - memory_dim (hidden_dim): 30
  - PolicyHead: [256, 256, 256, 512]
  - ValueHead: [256, 256, 256, 512, 512]

WD 與 IL 的有意差異 (保留 WD modular principle，搭配 IL 環境適配):
  - Extractor: WD=Linear(113→64); IL=Conv1d(72D LiDAR)+MLP(7D state)=96D
    原因: IL obs layout 不同（79D vs 113D），Conv1d 對 LiDAR 更適合

不依賴 obs60 (MOT features) — 部署可行。
注意: WD train_rnn_car.py 中 spot_state_obs_agent_rate=0，obs 113D 中 60D 障礙物區塊
多為 default/placeholder 值（type-2 state obstacles 未生成）。有效障礙資訊來自 36D LiDAR
+ 7D preprocess_real_data (2 nearest neighbors)。
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# Observation layout (from 139D env output — 只用 79D)
# ============================================================================

EGO_START, EGO_END = 0, 4        # accel + vel + omega + radius
GOAL_START, GOAL_END = 4, 6      # waypoint (x, y)
LIDAR_START, LIDAR_END = 6, 78   # 72 bins
LIDAR_LEN = LIDAR_END - LIDAR_START   # 72 (多幀 stacking 用)
TIME_START, TIME_END = 78, 79    # remaining ratio (was 138:139 when obs=139D)
ACT_HIST_START, ACT_HIST_END = 79, 83   # v3c: past 2 actions (a, ω) × 2 = 4D

ACT_HIST_DIM = ACT_HIST_END - ACT_HIST_START  # 4
STATE_DIM = 7 + ACT_HIST_DIM   # ego(4) + goal(2) + time(1) + act_hist(4) = 11
LIDAR_DIM = 72
LIDAR_CONV_CH = 64           # Conv1d 最終 channel 數
LIDAR_CONV_LEN = LIDAR_DIM // 4  # 72 → 18，經兩次 stride=2 卷積後的角度軸長度
USED_OBS_DIM = 83  # 4 + 2 + 72 + 1 + 4 (v3c: +4 action history)
RAW_OBS_DIM = 83   # env output dim (v3c: 79 + 4 action history)
NUM_BINS = 19
TOTAL_LOGITS = NUM_BINS * 2  # 38


# ============================================================================
# 2-Branch Feature Extractor (no obs60)
# ============================================================================

class LidarStateExtractor(nn.Module):
    """2-branch extractor: LiDAR Conv1d (64D) + StateMLP (32D) = 96D.

    從 79D env obs 中取 LiDAR(72D) 和 state(7D) 部分。保留 WD modular principle，無 TopK obs。

    角度語意 (2026-06-02 變更):
      原本 Conv1d → AdaptiveMaxPool1d(1) 在角度軸做 global max pooling，
      把 18 個 spatial tokens 壓成 channel-wise scalar → translation-invariant，
      丟失「障礙在哪個方位角」的定位資訊（policy 無法定位 static walls）。
      改為 Conv1d → flatten → Linear：flatten 保留 18 個角度位置為獨立 input index，
      Linear 對每個位置有獨立權重 → translation-equivariant，角度身份保留
      （等同 WD per-index Linear 精神，前面多了卷積特徵抽取）。
      ⚠️ lidar_proj 形狀由 (64→64) 變為 (1152→64)，與舊 checkpoint 不相容，
         該層需重新初始化；其餘層（conv / state / 下游 RNN+head）形狀不變可遷移。
      Conv1d padding_mode='circular'：LiDAR 角度環狀，零填充會割斷 355°↔0° 連續性
      （影響正前方那段角度的卷積特徵），改環狀填充修正；不改 tensor 形狀、不影響相容性。
    """

    def __init__(self, legacy: bool = False, include_act_hist: bool = True, act_hist_dropout: float = 0.0,
                 frame_stack: int = 1):
        super().__init__()
        # frame_stack>1 (多幀 LiDAR): obs 尾端附 (K-1)×72 幀歷史 LiDAR,讓 Conv1d 在「原始 LiDAR 層」
        #   直接算跨幀運動(類光流),RNN 再整合 → 給 RNN 做 MOT 必需的連續幀。frame_stack=1=現狀。
        self.frame_stack = int(frame_stack)
        # legacy=True：還原 2026-06-02 之前的舊架構（Conv1d 零填充 + AdaptiveMaxPool1d(1)
        #   + Linear(64,64)），用來載入舊 checkpoint（lidar_proj 形狀 64→64）。
        #   僅供 play/eval 相容，不影響新訓練（預設 legacy=False = 新架構）。
        # include_act_hist=False：還原 v3b/v3 的 7D state（無 4D 動作歷史），
        #   讓 play/eval 能載入 action-stacking 之前的 checkpoint（state_mlp 輸入 7D）。
        #   forward 會自動略過 obs 尾端的 act_hist，不影響新訓練（預設 True = v3c 11D）。
        self.legacy = legacy
        self.include_act_hist = include_act_hist
        # v3d (2026-06-12): 訓練時對 act_hist 做 dropout，當「保險絲」逼 policy 不可
        #   100% 依賴歷史動作（即使被 mask 也要靠 RNN hidden + ego 推斷）→ 弱化
        #   "copy 上一步 action" 的 shortcut。eval/play 時 self.training=False 自動關閉。
        #   0.0 = 不啟用（v3c 及之前的行為完全不變）。
        self.act_hist_dropout = float(act_hist_dropout)
        # Branch 1: LiDAR Conv1d
        # padding_mode='circular'：LiDAR 角度為環狀（bin71 355° ↔ bin0 0° 是鄰居），
        # 零填充會在邊界假裝外面是空的，割斷正前方那段角度的卷積連續性 → 用環狀填充修正。
        # legacy 模式用零填充（'zeros'），對齊舊 checkpoint 訓練時的行為。
        pad_mode = "zeros" if legacy else "circular"
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(self.frame_stack, 32, kernel_size=5, padding=2, padding_mode=pad_mode),  # 多幀:K input channels
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, padding_mode=pad_mode),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode=pad_mode),
            nn.ReLU(),
        )
        if legacy:
            # 舊架構：global max pooling 壓成 channel-wise scalar → Linear(64,64)
            self.lidar_pool = nn.AdaptiveMaxPool1d(1)
            self.lidar_proj = nn.Linear(LIDAR_CONV_CH, 64)  # 64 → 64
        else:
            # 新架構：flatten 角度軸 → Linear，保留每個角度位置的身份（取代 AdaptiveMaxPool1d）
            self.lidar_proj = nn.Linear(LIDAR_CONV_CH * LIDAR_CONV_LEN, 64)  # 1152 → 64
        self.lidar_ln = nn.LayerNorm(64)

        # Branch 2: State MLP (ego + goal + time + act_hist = 11D in v3c, 7D legacy)
        # v3c: STATE_DIM=11 包含 4D 動作歷史 (past 2 steps × (a_norm, ω_norm))
        # include_act_hist=False: 回到 7D（ego+goal+time），相容 v3b/v3 checkpoint
        state_in_dim = STATE_DIM if include_act_hist else (STATE_DIM - ACT_HIST_DIM)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.state_ln = nn.LayerNorm(32)

    @property
    def output_dim(self):
        return 96  # 64 + 32

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, 79] env observation
        Returns:
            [B, 96] feature embedding
        """
        if self.frame_stack > 1:
            # 多幀: current frame 在 LIDAR_START:LIDAR_END,前 (K-1) 幀附在 obs 尾端 (rollout 時 augment)
            cur = obs[:, LIDAR_START:LIDAR_END]                              # [B, 72] 當前幀
            hist = obs[:, -(self.frame_stack - 1) * LIDAR_LEN:]             # [B, (K-1)*72] 歷史幀
            lidar = torch.cat([cur, hist], dim=-1).reshape(
                obs.shape[0], self.frame_stack, LIDAR_LEN)                   # [B, K, 72]
        else:
            lidar = obs[:, LIDAR_START:LIDAR_END].unsqueeze(1)   # [B, 1, 72]
        lidar = self.lidar_conv(lidar)                        # [B, 64, 18]
        if self.legacy:
            lidar = self.lidar_pool(lidar).squeeze(-1)        # [B, 64]  舊架構 global max pool
        else:
            lidar = lidar.flatten(1)                          # [B, 64*18=1152]  保留角度位置
        lidar = self.lidar_ln(self.lidar_proj(lidar))         # [B, 64]

        ego = obs[:, EGO_START:EGO_END]                       # [B, 4]
        goal = obs[:, GOAL_START:GOAL_END]                     # [B, 2]
        time_feat = obs[:, TIME_START:TIME_END]                # [B, 1]
        if self.include_act_hist:
            act_hist = obs[:, ACT_HIST_START:ACT_HIST_END]     # [B, 4] v3c: 過去 2 步 action
            if self.act_hist_dropout > 0.0:
                # v3d: 訓練時隨機 mask act_hist（eval 時 training=False → 直接 passthrough）
                act_hist = nn.functional.dropout(
                    act_hist, p=self.act_hist_dropout, training=self.training
                )
            state_in = torch.cat([ego, goal, time_feat, act_hist], dim=-1)  # [B, 11]
        else:
            # v3b/v3 相容：無 act_hist，state 只用 ego+goal+time = 7D
            state_in = torch.cat([ego, goal, time_feat], dim=-1)  # [B, 7]
        state = self.state_ln(self.state_mlp(state_in))       # [B, 32]

        return torch.cat([lidar, state], dim=-1)              # [B, 96]


# ============================================================================
# Preprocess RNN Module (WD: vanilla RNN, not GRU)
# ============================================================================

class PreprocessRNN(nn.Module):
    """模組化 RNN — 移植自 Warp Drive module_connected.py。

    WD 原始設計 (module_connected.py lines 214-573):
      preprocess_info_front: Linear(obs→64)+ReLU
      → RNN(64→30)
      → [if concat_rnn] cat(rnn_out, fc_out)
      → preprocess_info_middle: Linear(→module_connect_dim)+ReLU
      → [training] preprocess_info_back: Linear(→network_feture_dim)+ReLU (for aux loss)
      → .detach() → concat(detached_obs, preprocess_feat) → RL head

    WD detach 位置 (line 573): concat_input = rl_in_.detach()
    → RL loss 永遠不訓練此模組。RNN 只被 module loss (aux) 訓練。

    Predict head (WD-style):
      - 單層 Linear(preprocess_dim→predict_dim) — 2026-06-23 移除末端 ReLU(dead-head bug,見 predict_head 註解)
        (WD config: module_network=[64,'rnn',32,'rl'], network_feture_dim=7 → back=[12→7])
      - Target: 7D privileged geometry (2 nearest obstacles body-frame x,y,dist + timestep)
      - Loss: WD module loss — log(clamp(L1, 0.01)) × per-dim weight
      - See wd_aux_targets.py for target construction and loss computation

    WD 對齊:
      - rnn_type='RNN' (vanilla, not GRU) — WD: rnn_layer_type='RNN'
      - hidden_dim=30 — WD: memory_dim=30
      - preprocess_dim=12 — WD: module_connect_dim=12
      - concat_rnn=True — WD train_rnn_car.py 同樣用 True
      - predict_head: Linear(12→7)+ReLU — WD preprocess_info_back 原樣
    """

    def __init__(
        self,
        input_dim: int = 96,
        fc_dim: int = 48,
        hidden_dim: int = 30,
        preprocess_dim: int = 12,
        predict_dim: int = 7,
        middle_dim: int | None = None,
        concat_rnn: bool = True,
        rnn_type: str = "RNN",  # "RNN" (WD default) or "GRU"
        aux_skip_input: bool = False,  # True: predict_head 直接 concat extractor 輸入(繞過 RNN 洗位置)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.concat_rnn = concat_rnn
        self.rnn_type = rnn_type
        self.fc_dim = fc_dim
        self.middle_dim = middle_dim
        self.aux_skip_input = aux_skip_input
        self.input_dim = input_dim

        # FC front
        self.fc_front = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
        )

        # RNN (WD: vanilla RNN by default)
        if rnn_type == "GRU":
            self.rnn = nn.GRU(fc_dim, hidden_dim, num_layers=1, batch_first=False)
        else:
            self.rnn = nn.RNN(fc_dim, hidden_dim, num_layers=1, batch_first=False,
                              nonlinearity='relu')

        # FC middle (after concat)
        # WD exact car config uses spot_module_network=[64, 'rnn', 32, 'rl']
        # and spot_module_connect_dim=12, i.e. concat(30,64)=94 -> FC32 -> FC12.
        # Legacy IsaacLab modes keep the previous direct concat -> FC12 mapping.
        middle_input = hidden_dim + fc_dim if concat_rnn else hidden_dim
        if middle_dim is None or middle_dim <= 0:
            self.fc_middle = nn.Sequential(
                nn.Linear(middle_input, preprocess_dim),
                nn.ReLU(),
            )
        else:
            self.fc_middle = nn.Sequential(
                nn.Linear(middle_input, middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, preprocess_dim),
                nn.ReLU(),
            )

        # Predict head — predicts privileged geometry+velocity target for aux module loss.
        # ⚠️ 2026-06-23 修 dead-head bug：原本末端有 nn.ReLU()，但 target 含「有正負」的
        #   body-frame 位置/速度（vbx/vby 可負）→ ReLU 強制 ≥0 + dead-ReLU 死亡螺旋 →
        #   predict_head 輸出恆 0、零梯度、RNN 從未被監督學會預測障礙運動。移除末端 ReLU。
        #   保留 nn.Sequential 結構使 state_dict key 仍為 predict_head.0.* (相容舊 checkpoint)。
        # aux_skip_input: predict_head 額外吃 extractor 輸入(input_dim)，位置不過 RNN 洗
        _ph_in = preprocess_dim + input_dim if aux_skip_input else preprocess_dim
        self.predict_head = nn.Sequential(
            nn.Linear(_ph_in, predict_dim),
        )

        self.preprocess_dim = preprocess_dim

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        training: bool = False,
        detach_output: bool = False,
    ):
        """Supports both single-step [B, D] and sequence [L, B, D] inputs.

        Single-step mode (features.dim() == 2):
            features: [B, 96]
            Returns: feat [B, 12], prediction [B, 7] or None, new_hidden [1, B, H]

        Sequence mode (features.dim() == 3, for TBPTT aux training):
            features: [L, B, 96]
            Returns: feat [L, B, 12], prediction [L, B, 7] or None, new_hidden [1, B, H]

        Args:
            features: [B, 96] or [L, B, 96] from extractor
            hidden:   [1, B, hidden_dim] RNN hidden state
            training: if True, compute 7D prediction for WD module loss
            detach_output: if True, explicitly detach preprocess feat before return.
        """
        seq_mode = features.dim() == 3  # [L, B, D]

        if seq_mode:
            L, B, D = features.shape
            # FC front: reshape to [L*B, D], apply, reshape back
            fc_out = self.fc_front(features.reshape(L * B, D))  # [L*B, 48]
            fc_out = fc_out.reshape(L, B, -1)                   # [L, B, 48]

            # RNN unroll: input [L, B, 48], hidden [1, B, H]
            rnn_out, new_hidden = self.rnn(fc_out, hidden)      # [L, B, 30], [1, B, 30]

            if self.concat_rnn:
                combined = torch.cat([rnn_out, fc_out], dim=-1) # [L, B, 78]
            else:
                combined = rnn_out                              # [L, B, 30]

            preprocess_feat = self.fc_middle(
                combined.reshape(L * B, -1)).reshape(L, B, -1)  # [L, B, 12]

            prediction = None
            if training:
                _ph = preprocess_feat.reshape(L * B, -1)
                if self.aux_skip_input:
                    _ph = torch.cat([_ph, features.reshape(L * B, D)], dim=-1)  # skip extractor 輸入
                prediction = self.predict_head(_ph).reshape(L, B, -1)  # [L, B, predict_dim]

            if detach_output:
                return preprocess_feat.detach(), prediction, new_hidden
            return preprocess_feat, prediction, new_hidden

        # --- Single-step mode (original path) ---
        fc_out = self.fc_front(features)                    # [B, 48]
        rnn_in = fc_out.unsqueeze(0)                        # [1, B, 48] (seq_len=1)
        rnn_out, new_hidden = self.rnn(rnn_in, hidden)      # [1, B, 30], [1, B, 30]
        rnn_out = rnn_out.squeeze(0)                        # [B, 30]

        if self.concat_rnn:
            combined = torch.cat([rnn_out, fc_out], dim=-1) # [B, 78]
        else:
            combined = rnn_out                              # [B, 30]

        preprocess_feat = self.fc_middle(combined)          # [B, 12]

        prediction = None
        if training:
            _ph = preprocess_feat
            if self.aux_skip_input:
                _ph = torch.cat([_ph, features], dim=-1)  # skip extractor 輸入(繞過 RNN 洗位置)
            prediction = self.predict_head(_ph)  # [B, predict_dim] WD-style privileged target

        # WD 原版 (line 573): concat_input = rl_in_.detach()
        # WD 設計: RNN 永遠不吃 RL gradient。detach 在 RL concat 階段完成。
        # IL: rollout 用 torch.no_grad() 達到同樣效果; PPO 只用 cached rl_in。
        if detach_output:
            return preprocess_feat.detach(), prediction, new_hidden
        return preprocess_feat, prediction, new_hidden


# Backward compatibility alias
PreprocessGRU = PreprocessRNN


# ============================================================================
# RNN State Manager
# ============================================================================

class RNNStateManager:
    """Per-env GRU hidden state 管理器。"""

    def __init__(self, num_envs: int, hidden_dim: int, device: torch.device):
        self.hidden = torch.zeros(1, num_envs, hidden_dim, device=device)

    def get(self) -> torch.Tensor:
        return self.hidden

    def update(self, new_hidden: torch.Tensor):
        self.hidden = new_hidden.detach()

    def reset(self, env_ids: torch.Tensor):
        """env reset 時歸零對應 env 的 hidden state。"""
        if len(env_ids) > 0:
            self.hidden[:, env_ids, :] = 0.0


# ============================================================================
# RL Heads (PPO 只訓練這些)
# ============================================================================

class PolicyHead(nn.Module):
    """Policy head: input_dim → 38 logits (19 accel + 19 omega).

    WD 對齊: spot_policy = [256, 256, 256, 512] (4 hidden layers)
    input = cat(obs, preprocess_12D) = 91D (legacy) or 125D (wd_exact_rnn)

    Note vs WD: WD 用 2 個獨立 Linear(512→19) 各出 19 logits;
    此處用 1 個 Linear(512→38) 然後 split。數學等價（同尺寸），
    唯一差異是初始化時兩半的 weight 不是獨立 sample 的，
    但經過幾步訓練後差異消失。不影響學習行為。
    """

    def __init__(self, input_dim: int = 91, privileged_dim: int = 0):
        super().__init__()
        self._privileged_dim = privileged_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, TOTAL_LOGITS),
        )
        # Oracle-obstacle 殘差分支（--oracle_obstacles_to_policy 診斷用）:
        # logits = net(rl_input) + priv_branch(privileged)。末層 init 0 →
        # 起始輸出恆 0 → policy 與無 priv 的 checkpoint 完全相同（完美暖啟動），
        # 之後才學會用障礙特權狀態。這是「感知 vs 策略」oracle 上限測試的注入點。
        if privileged_dim > 0:
            self.priv_branch = nn.Sequential(
                nn.Linear(privileged_dim, 128),
                nn.ReLU(),
                nn.Linear(128, TOTAL_LOGITS),
            )
            nn.init.zeros_(self.priv_branch[-1].weight)
            nn.init.zeros_(self.priv_branch[-1].bias)

    def forward(self, rl_input: torch.Tensor, privileged: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(rl_input)  # [B, 38]
        if self._privileged_dim > 0 and privileged is not None:
            logits = logits + self.priv_branch(privileged)  # 殘差；init 0 → 起始不改變 policy
        return logits


class ValueHead(nn.Module):
    """Value head: input_dim → scalar.

    WD 對齊: spot_critic = [256, 256, 256, 512, 512] (5 hidden layers)
    input = cat(obs_79D, preprocess_12D) = 91D

    Asymmetric critic: input_dim = 91 + privileged_dim (50D) = 141D.
    When privileged_dim > 0, a separate projection merges the privileged
    features before the shared MLP, allowing checkpoint migration from
    symmetric → asymmetric (the rl_proj weights carry over).
    """

    def __init__(self, input_dim: int = 91, privileged_dim: int = 0):
        super().__init__()
        self._privileged_dim = privileged_dim

        if privileged_dim > 0:
            self.rl_proj = nn.Linear(input_dim, 128)
            self.priv_proj = nn.Linear(privileged_dim, 128)
            self.merge = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            trunk_input = 256
        else:
            trunk_input = input_dim

        self.net = nn.Sequential(
            nn.Linear(trunk_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

        if privileged_dim > 0:
            nn.init.zeros_(self.priv_proj.weight)
            nn.init.zeros_(self.priv_proj.bias)

    def forward(self, rl_input: torch.Tensor, privileged: torch.Tensor | None = None) -> torch.Tensor:
        if self._privileged_dim > 0 and privileged is not None:
            h_rl = self.rl_proj(rl_input)       # [B, 128]
            h_priv = self.priv_proj(privileged)  # [B, 128]
            h = self.merge(torch.cat([h_rl, h_priv], dim=-1))  # [B, 256]
        else:
            h = rl_input
        return self.net(h)  # [B, 1]


# ============================================================================
# Obstacle Agent — 已遷移到 obstacle_agent/ 模組
# ============================================================================
# 向後相容 import (訓練腳本仍 from modular_rnn_models import ObstaclePolicyFC)
try:
    from obstacle_agent import ObstaclePolicyFC, ObstacleValueFC, OBS_DIM as OBS_POLICY_OBS_DIM, ACT_DIM as OBS_POLICY_ACT_DIM  # noqa: F401, E501
except (ImportError, ModuleNotFoundError):
    pass  # play scripts don't need obstacle_agent
