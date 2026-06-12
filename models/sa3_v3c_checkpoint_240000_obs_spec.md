# SA3_v3c Checkpoint — 觀測維度規格 (Observation Spec)

> 部署 / 推論用。此 checkpoint 的 **actor 輸入為 83D**（`obs_normalizer.mean/var` shape = 83）。
> 實機端只需重建這 83D 向量並依相同順序餵入即可，privileged 50D 僅訓練時的 critic 使用，部署不需要。

## Checkpoint 來源

| 項目 | 值 |
|------|-----|
| 檔名 | `checkpoint_240000.pt` |
| Run | `sa3_v3c_ne1024_s42` (`wd_sa3_v3c`) |
| iteration | 799 (iter 800 / 1400) |
| total_steps | 240,000 |
| 狀態 | 訓練進行中，此為**當下最佳/最新** checkpoint (SR ≈ 73% / CR ≈ 27% @ Stage 3) |
| 起點 | `sa2_v3c_ne1024_s42/checkpoint_360000.pt` |
| Task | `Isaac-Navigation-Charge-VLP16-Curriculum-WD` |
| curriculum | `warp_drive_single_agent_v3`, fixed_stage=3 (walls + crossing) |

## 物理 / 標定參數 (v3c)

| 參數 | 值 |
|------|-----|
| LiDAR `r_min` | **0.25 m** (表面→人物中心 0.2m + 物理半徑 0.0515m) |
| LiDAR beams | 72 bins (5° 解析, VLP-16 min-pool 16 ch → 72 方位) |
| LiDAR 正規化 | distance ÷ 20.0 m → [0, 1] |
| v_max | 1.0 m/s |
| a_max | 0.5 m/s² (obs 正規化用 0.2 尺度) |
| ω_max | 0.25π rad/s |
| dt | 0.2 s |
| body_radius | 0.35 m |
| obs_delay_steps | [0, 1] (觀測延遲 DR) |

---

## 83D Actor 觀測佈局

**總計 83D = ego(4) + goal(2) + lidar(72) + time(1) + action_stack(4)**

| 範圍 | 名稱 | 維度 | 說明 | 正規化 |
|------|------|------|------|--------|
| `[0]`     | linear_accel | 1 | 線加速度 āₜ | ÷ a_max 尺度 |
| `[1]`     | linear_vel   | 1 | 線速度 v̄ₜ（含倒車，∈[-1,1]） | ÷ v_max (1.0) |
| `[2]`     | angular_vel  | 1 | 角速度 ω̄ₜ | ÷ ω_max |
| `[3]`     | robot_radius | 1 | 機器人半徑 (常數 0.35m) | 固定 |
| `[4:6]`   | goal_xy      | 2 | 目標點於 robot body-frame (xₘ, yₘ) | body-frame 相對 |
| `[6:78]`  | lidar        | 72 | 72 方位 bin 正規化距離 ∈ [0,1] | (d, r_min=0.25) ÷ 20m |
| `[78]`    | time_rem     | 1 | episode 剩餘時間比例 (0→1) | [0,1] |
| `[79]`    | a_{t-1}      | 1 | 上一步 applied 線加速度 | ÷ 0.2，clip[-2,2] |
| `[80]`    | ω_{t-1}      | 1 | 上一步 applied 角速度 | ÷ (π/15)，clip[-2,2] |
| `[81]`    | a_{t-2}      | 1 | 前兩步 applied 線加速度 | ÷ 0.2，clip[-2,2] |
| `[82]`    | ω_{t-2}      | 1 | 前兩步 applied 角速度 | ÷ (π/15)，clip[-2,2] |

> **Action stacking (4D, `[79:83]`)**：v3c 抗抽動設計，提供顯式「煞車訊號」讓 policy 做阻尼修正。
> episode reset 時 (episode_length_buf==0) 此 buffer 歸零。

---

## 網路雙分支輸入切分 (extractor)

模型 encoder 把 83D obs 切成兩支：

| 分支 | 維度 | 內容 | 對應 obs index | checkpoint tensor |
|------|------|------|----------------|-------------------|
| **lidar_conv** | 72 | LiDAR 72 bins | `[6:78]` | `extractor.lidar_conv.0.weight (32,1,5)` → `lidar_proj (64,1152)` |
| **state_mlp**  | 11 | ego(4)+goal(2)+time(1)+action_stack(4) | `[0:6] ∪ [78:83]` | `extractor.state_mlp.0.weight (32, 11)` |

- lidar 分支輸出 64D，state 分支輸出 32D → 合併 **96D** → `preprocess_rnn.fc_front (64, 96)` (RNN64)。
- RNN type = `RNN` (vanilla)，hidden_dim=64，preprocess_dim=12，wd_middle_dim=48。

## 動作空間 (Action)

| 項目 | 值 |
|------|-----|
| 型態 | MultiDiscrete([19, 19]) → policy head 輸出 **38 logits** (19+19) |
| 維度 | dim0 = 線加速度 19 檔；dim1 = 角速度 19 檔，中心對稱 (idx 9 = 0) |
| tensor | `policy_head.net.8.weight (38, 512)`；input `net.0.weight (256, 95)` |

## Auxiliary / Critic（部署不需要）

| 項目 | 維度 | 說明 |
|------|------|------|
| predict_head (velocity aux) | 13 | `wd_7d_geometry`，aux_velocity_topk=3 |
| critic rl 分支 | 95 | `value_head.rl_proj (128, 95)` |
| critic privileged 分支 | **50** | `value_head.priv_proj (128, 50)` — 10 障礙 × 5D [x,y,dir,v,size] ground-truth，**僅訓練 critic 用，部署省略** |

---

## 部署檢查清單

1. 重建 83D 向量，**順序嚴格依上表** `[0:83]`。
2. LiDAR：72 方位 bins，`r_min=0.25`，÷20m，缺失補 1.0（max range）。
3. goal 必須轉到 robot body-frame。
4. action_stack 維護過去 2 步 applied action，新→舊排列，reset 歸零。
5. 載入後套用 `obs_normalizer` (mean/var, shape 83) 做 whiten 再進網路。
6. RNN hidden state 每 episode reset 歸零。

## 來源檔案

- obs 函式：`source/isaaclab_tasks/.../charge_skrl/mdp/observations/obs_functions.py`
  - `wd_like_sweep_72()`（72D LiDAR）、`discrete_applied_action_history(stack_size=2)`（4D action stack）
- 觀測群組順序：`source/isaaclab_tasks/.../charge_skrl/cfg/charge_env_cfg_vlp16.py` PolicyCfg
- 訓練入口：`scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py`
- config：`scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa3_v3c.yaml`
