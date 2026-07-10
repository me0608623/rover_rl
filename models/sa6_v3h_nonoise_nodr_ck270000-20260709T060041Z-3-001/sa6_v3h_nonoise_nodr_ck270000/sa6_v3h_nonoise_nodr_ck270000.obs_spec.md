# sa6_v3h_nonoise_nodr_ck270000 — 觀測 / 動作 逐維規格

> 來源: `logs/rnn_car/sa6_v3h_nonoise_nodr_ne1024_s42/checkpoint_270000.pt`
> ★SA6 手動中止 (iter ~993/1400) 的 best 存檔點 (iter 900) · clean baseline（無雜訊 / 無 DR）
> obs 79D · GRU64 · act_hist OFF · r_min 0.25。每一維定義從訓練端 `functions.py` / `obs_functions.py` / config 抄，非憑印象。

## 觀測 raw_obs = 79D（單幀，無 history stacking）— 組裝順序

順序 = **ego(4) + goal(2) + lidar(72) + time(1)**（見 `charge_env_cfg_vlp16.py` PolicyCfg 第二 obs group）。
所有維度進網路前先過 **obs_normalizer（running mean/var，79D，已 bake 在 .pt）** 再正規化一次。

### ego state（維 0–3）

| idx | 名稱 | 定義 | 訓練端正規化 | 值域 |
|-----|------|------|------|------|
| **0** | applied_accel | 前進線加速度 `a=(v_t − v_{t-1})/dt`（上一步的速度差；第一步=0）| `÷ max_acceleration = 1.0` | ~[-1, 1] |
| **1** | linear_velocity | body-frame 前進速度 `v_x`（差速車主要沿 x 前進）| `÷ max_linear_velocity = 1.0` | [-1, 1] |
| **2** | angular_velocity | yaw rate `ω_z`（繞 Z 軸角速度，供角速度阻尼）| **`÷ max_angular_velocity = 1.5`** | [-1, 1]（+1=最大左轉、−1=最大右轉）|
| **3** | body_radius | 車體碰撞半徑常數（讓 policy 感知自身大小）| 原值（不正規化）| **常數 0.35** |

> ⚠️ 維 2 的正規化除數是 **1.5**（obs 用），**不是**動作的 omega_max=1.2。車端算此 obs 要 `ω_z / 1.5`。

### goal state（維 4–5）— robot-frame，**原始公尺（不正規化）**

| idx | 名稱 | 定義 | 值域 |
|-----|------|------|------|
| **4** | goal_x | 目標在 **robot frame 的前後距離 (m)**，`+` = 前方 | 實距（m）|
| **5** | goal_y | 目標在 **robot frame 的左右距離 (m)**，`+` = 左、`−` = 右 | 實距（m）|

> 範例（docstring）：`[2.5, -1.0]` = 前方 2.5 m、右邊 1 m。原始公尺，尺度由 obs_normalizer 吸收。

### lidar state（維 6–77）= 72 bins，2D 方位掃描

- 每維 = 該 5° 方位扇區內的**最近障礙表面距離**，正規化 = `clamp(d − 0.35, 0) / 20`，`clip(0,1)`。
  - **1.0 = 淨空**（≥20 m 或無命中/盲區）；**越小 = 障礙越近**；0 = 貼到車體。
  - 取 min 兩層：5° 扇區內 5 條水平射線取最近 × VLP-16 16 條垂直環取最近（最保守）。
  - `r_min=0.25`（< 0.25 m 命中視為盲區→當 r_max）、`z_filter=0.5`（濾地板/天花板鬼影）。
- **bin↔角度映射**（body-frame 方位角 `θ = atan2(y_左, x_前)`，`bin = floor((θ+π)/(2π)·72)`，5°/bin）：

| bin idx | 方位 θ | 方向 |
|---------|--------|------|
| 6 (lidar[0]) | −180°…−175° | 正後方 |
| 6+18 = 24 (lidar[18]) | −90° | **正右方** |
| 6+36 = 42 (lidar[36]) | 0° | **正前方** |
| 6+54 = 60 (lidar[54]) | +90° | **正左方** |
| 6+71 = 77 (lidar[71]) | +175°…+180° | 正後方 |

> bin index 隨方位角遞增（右 → 前 → 左 → 後）。**車端 LiDAR 前處理必須產生相同 bin 排序**（同 atan2 方位量化），否則角度錯位。

### time state（維 78）

| idx | 名稱 | 定義 | 值域 |
|-----|------|------|------|
| **78** | time_remaining_ratio | episode 剩餘時間比例 | [0, 1]（1=剛開始，0=即將超時）|

---

## 動作 MultiDiscrete([19, 19]) = 38 logits

| head | logits idx | bins | 物理量 | 解碼 |
|------|-----------|------|--------|------|
| linear_accel | 0–18 | 19 | 線加速度 | idx∈[0,18], center=9, ratio=(idx−9)/9, `accel=ratio·a_max(0.5)` |
| angular_vel | 19–37 | 19 | 角速度 | idx∈[0,18], center=9, ratio=(idx−9)/9, `omega=ratio·omega_max(1.2)` |

中心 idx=9 → ratio 0（零動作）。**部署 deterministic：各 head 取 argmax**。

## 物理邊界（★車端 act_max_* 必須完全一致）

| 量 | 值 |
|----|-----|
| v_max | 1.0 m/s |
| v_reverse_max | −0.2 m/s（reverse_scale=0.2）|
| a_max | 0.5 m/s² |
| **omega_max** | **1.2 rad/s**（★勿設 0.785；2026-06-02 從 2.0→1.2 對齊馬達）|
| omega_accel_max | 3.0 rad/s² |
| dt | 0.2 s（decimation=20, sim.dt=0.01）|

## 網路架構（.pt state_dict 內容）

```
extractor        : 3-branch (Conv1d LiDAR + ObsMLP + StateMLP) → 96D feat
feat_normalizer  : running mean/var (96D)  ← 已 bake
preprocess_rnn   : fc_front(96→64) → GRU(64→64) → fc_middle(128→48)
policy_head      : 91 → … → 38   (MultiDiscrete 19×19)
value_head       : rl_proj(91→128) + priv_proj(50→128)  ← asymmetric critic（部署不需要）
obs_normalizer   : running mean/var (79D)  ← 已 bake
```
