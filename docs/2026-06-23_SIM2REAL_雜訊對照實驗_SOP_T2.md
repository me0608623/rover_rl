# VLP-16 Sim-to-Real 雜訊模型對照實驗 SOP（T2 控制動態）

> 日期：2026-06-23 ｜ 目標：以**真車對照實驗**回答「把實測 VLP-16 雜訊模型 + 隨機化帶進訓練，真的對 sim→real transfer 有用嗎？」
> 核心貢獻對應：C1（資料驅動雜訊模型，MF 67×→1.5×）、C2（min-pool-aware TLNI，三定理）、C3（policy 層面實證）。

---

## 0. 核心問題與結論 framing（先讀這段，否則白跑）

**不要用「SR 越高越好」當結論**——那是陷阱。因為 TLNI 在 sim 內 SR 通常**低於** no-noise（它在更難的雜訊環境訓練）。

正解：用 **sim→real gap** 當主指標：

```
Gap(arm) = SR_sim(arm) − SR_real(arm)
```

**論文主張 =「TLNI 最小化 |Gap|」**，不是「TLNI 最高 SR」。原因：no-noise 可能在 sim 有 SR≈100% 但真車崩（gap 大）；TLNI 在 sim 中高、真車中高（gap 小）。這個 framing 直接對應 C1「觀測分布對齊」。

兩個可驗證假設：
- **H1（C1 價值，主）**：data-driven 雜訊模型 → 真車 transfer 比規格書假設好。
- **H2（C2 價值，輔）**：min-pool-aware 注入 → 比天真 per-ray 注入 transfer 好（**主要靠 sim 消融 + 定理，不必上車**）。

---

## 1. 實驗矩陣總覽

| 對照臂 | 訓練 LiDAR 設定 | Sim 評估 | 真車 T2 | 角色 |
|--------|----------------|:--------:|:-------:|------|
| **A. no-noise** | 理想 LiDAR（全關）| ✓ | ✓ | sim-only strawman，最大 gap |
| **B. spec-Gaussian** | 規格書 ±3cm 零均值 per-ray 高斯 | ✓ | （選配）| 傳統 datasheet 做法 |
| **C. TLNI（data-driven）** | 你實際部署的全套 | ✓ | ✓ | treatment，最小 gap |

**三臂固定變數（務必逐項鎖死）**：
- 同 `curriculum_version`（= v3 家族，**r_min=0.25**）、同 scene/physics/obstacle 設定
- 同 seed、同 `timesteps`、同 `num_envs`、同 RL 超參
- **只動 LiDAR 雜訊區塊**

> ⚠️ 課程陷阱：你的 noise 原本「隨 SA1→SA7 漸進開」。所以**不能**直接拿 SA1(clean) 比 SA7(TLNI)——那是比難度不是比雜訊。三臂必須在**同一個固定難度 stage**（建議你部署的那個 stage，例 SA4 或 SA5）從頭訓，只差 LiDAR。

---

## 2. 三對照臂定義（LiDAR config 區塊）

**Base = 你實際部署的 config**（確認它含 full TLNI：`per_ring_bias + distance_bias + distractor + block_dropout + L3 DR`）。假設為 `wd_sa4_v3f.yaml`（r_min=0.25，curriculum v3e）。若你部署的是別的，把 base 換成那個即可。

### 複製三份 config，**只改 LiDAR 區塊**：

```bash
cd scripts/reinforcement_learning/skrl/rnn_car_modular/configs
cp wd_sa4_v3f.yaml wd_sa4_ablA_nonoise.yaml   # Arm A
cp wd_sa4_v3f.yaml wd_sa4_ablB_specgauss.yaml # Arm B
# Arm C = wd_sa4_v3f.yaml 原檔（沿用已訓好的，省一 run）
```

### Arm A — `wd_sa4_ablA_nonoise.yaml`（理想 LiDAR，strawman）

```yaml
# === LiDAR: 理想感測器，全關（r_min 仍 0.25，由 curriculum 控制）===
lidar_no_noise: true
lidar_distance_bias: false
lidar_per_ring_bias: false
```

### Arm B — `wd_sa4_ablB_specgauss.yaml`（規格書做法：只有 per-ray ±3cm 高斯）

```yaml
# === LiDAR: 規格書 ±3cm 零均值 per-ray 高斯，其餘全關 ===
lidar_no_noise: false
lidar_displacement_std: 0.03          # 規格書 ±3cm，fixed-σ per-ray（定理1：會被 min-pool 吞）
lidar_displacement_std_per_meter: 0.0
lidar_displacement_std_soft: 0.0
lidar_hole_rate: 0.0
lidar_distractor_rate: 0.0
lidar_distance_bias: false
lidar_per_ring_bias: false
lidar_obs_noise_std: 0.0
lidar_block_dropout_prob: 0.0
lidar_displacement_std_per_meter_dr: null
lidar_displacement_std_soft_dr: null
lidar_hole_rate_dr: null
lidar_distance_bias_k_dr: null
lidar_distance_bias_b_dr: null
```

> 設計理由：Arm B 故意只放「會被 min-pool 吞掉」的天真高斯（定理1）——這正是「傳統 datasheet sim-to-real」的寫照。若 C 真的贏 B，就**同時證明 C1（data-driven）+ C2（min-pool-aware）**。

### Arm C — `wd_sa4_v3f.yaml`（不動，full TLNI）

確認含：`lidar_distance_bias: true`、`lidar_per_ring_bias: true`、`lidar_distractor_rate: 0.002`、`lidar_block_dropout_prob: 0.03~0.10`、L3 DR ranges 非 null。

> **（選配）Arm D（sim-only，測 H2/C2 純度）**：把 C 的 distractor + block_dropout 拿掉，只留 data-driven per-ring + distance_bias + per-ray 高斯。證明「少掉 min-pool-aware 那兩項 → transfer 變差」。只在 sim 跑，不上車。

---

## 3. 訓練 SOP（PC 端，三臂 fresh run）

**環境**：`conda activate env_isaaclab`，工作目錄 `/home/aa/IsaacLab`。

```bash
# 共用變數（三臂一致）
COMMON="--headless --num_envs 1024 --seed 42 --timesteps <你部署C的總步數>"

# Arm A — no-noise
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa4_ablA_nonoise $COMMON \
  --run_name ablA_nonoise_ne1024_s42

# Arm B — spec-Gaussian
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa4_ablB_specgauss $COMMON \
  --run_name ablB_specgauss_ne1024_s42

# Arm C — TLNI（若已訓好可直接沿用其 checkpoint；否則重訓）
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa4_v3f $COMMON \
  --run_name ablC_tlni_ne1024_s42
```

**確認 checklist**：
- [ ] 訓完 grep console 的 `[SIM2REAL] LiDAR noise config` 印記，確認三臂的 noise 參數**真的如設計**（A 全 0、B 只有 displacement 0.03、C 全套）。這是論文附錄的鐵證。
- [ ] 三臂訓到**相同 timesteps / 相同 stage**。
- [ ] （理想）每臂跑 3 seed 報 mean±std；碩士底線 1 seed + 誠實標註。

---

## 4. Sim 評估 SOP（算 SR_sim）

在 sim 內用**固定的 held-out eval 場景**（不是訓練分布）跑三臂 policy，deterministic（關探索雜訊）：

```bash
# 用 play/eval 腳本，deterministic rollout，固定 seed 場景
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py \
  --checkpoint <ckpt_A> --num_envs 256 --seed 999 --eval_episodes 200 \
  --run_name evalA_sim   # 同樣跑 B、C
```

記錄每臂：**SR_sim / CR_sim / TO_sim**（用「完成 episode」正規化）。場景要涵蓋 T2 的對應情境（對向 / 橫穿）。

---

## 5. 車端部署 SOP（切換三 checkpoint）

車端：`aa@192.168.3.14`，repo `~/rover_rl`，ROS_DOMAIN_ID=55。

### 5.1 把三個 checkpoint + policy_params 推到車上

```bash
# PC → 車（checkpoint 放 ~/rover_rl/models/）
scp logs/isaac_lab/<run_A>/model_*.pt aa@192.168.3.14:~/rover_rl/models/ablA_nonoise.pt
scp logs/isaac_lab/<run_B>/model_*.pt aa@192.168.3.14:~/rover_rl/models/ablB_specgauss.pt
# C 若沿用已部署的，已經在車上
```

### 5.2 非互動切換 checkpoint（批次實驗用 `deploy_rl.sh`，不開 TUI）

```bash
# ssh 進車
ssh aa@192.168.3.14

# 用 model_path:= + params_file:= 直接指定，跳過互動選單
cd ~/rover_rl
bash deploy_rl.sh model_path:=/home/aa/rover_rl/models/ablA_nonoise.pt \
                   params_file:=/home/aa/rover_rl/src/rover_rl_bringup/config/policy_params_v3f.yaml
# （C 臂用你原本部署的 checkpoint + policy_params_v3f.yaml）
```

**⚠️ 三臂共用同一份 `policy_params_v3f.yaml` + `lidar_preprocessor_params_v3f.yaml`**（r_min=0.25、ω_max=1.2、obs omega 正規化 1.5），**只有 model_path 變**。這保證車端前處理三臂完全一致，只有 policy 權重不同。

**鎖死確認**：
- [ ] `policy_params_v3f.yaml` → `lidar_r_min_m: 0.25`（已驗證車端為 0.25 ✓）
- [ ] `lidar_preprocessor_params_v3f.yaml` → `r_min: 0.25`（已驗證 ✓）
- [ ] `omega_max` decode = 1.2（不是 0.785；見 `finding_v3c_omega_max_decode_bug`）
- [ ] 三臂用**同台車、同顆 VLP-16、同一天**跑

> 註：`lidar_preprocess.py:39` 的 `r_min=0.9` 只是 default，會被 v3* yaml 覆寫成 0.25。但**實驗當天務必 grep 確認 launch log 的實際 r_min=0.25**，別只信檔名。

---

## 6. T2 固定路徑行人測試 protocol

### 6.1 場地座標系（先在地上貼膠帶標記）

以機器人起點為原點，朝向 +x：
```
機器人 start (0, 0)  ─────────────────►  goal (G, 0)
            │
            │  走廊寬度建議 1.5~2.0 m
```
建議 G = 6~8 m。所有行人軌跡用膠帶標起點、轉折點、終點。

### 6.2 三個 canonical 情境（對應你實測的 σ_human 18–123mm regime）

| 情境 | 行人軌跡（相對機器人座標） | 速度 | 觸發時機 | 測什麼 |
|------|---------------------------|:----:|----------|--------|
| **S1 對向** | (G,0) → (0,0)，沿中心線朝機器人走 | 1.0 m/s | 機器人啟動同時 | 正面避讓、轉向決策 |
| **S2 橫穿** | (G/2, +1.5) → (G/2, −1.5)，垂直穿越路徑 | 1.0 m/s | 機器人離交點 ~3m 時行人起步 | 側向偵測 + 減速/停等 |
| **S3 同向慢行** | (G/2, 0) 起，朝 goal 同向慢走 | 0.5 m/s | 機器人啟動同時 | 跟車/超車、不被誘導碰撞 |

> 行人 = **真人**（這樣才 match 你 σ_human 實測；穿一般衣物）。為可重現性：固定同一位行人、固定步頻（可用節拍器）、固定軌跡。

### 6.3 Trial 設計（MVP 預算）

```
arms(真車) × {S1, S2, S3} × N trials
= {A, C} × 3 情境 × 10 trials = 60 trials   （MVP：只上車 A vs C 兩極端）
```
- 真車 MVP：**A vs C**（對比最大、故事最乾淨）；B 在 sim 證即可，預算夠再加真車。
- 每 trial **隨機化順序**（別先跑完 A 再跑 C，避免操作員/環境漂移偏差）。
- 每 trial 開新 rosbag + 新 diag CSV（deploy_rl_shell 結束會列本次 CSV）。

### 6.4 單一 trial 執行步驟

1. 機器人回 start (0,0)，朝向 +x，速度上限調 **0.4 m/s**（安全）。
2. 操作員手持 e-stop，行人就位 S1/S2/S3 起點。
3. 開 rosbag：`ros2 bag record -o ~/rover_rl/logs/ablation/trial_<arm>_S<i>_<n> /cmd_vel /scan /tf /tf_static /rover_rl_policy/status <goal_topic> <collision_topic>`（topic 名以車端實際為準，先 `ros2 topic list` 確認）。
4. 下 goal (G,0) + 啟動 RL 推理；行人依情境計時起步。
5. 結束條件：到達 goal / 碰撞 / timeout（建議 30s）→ 停 bag → 標記結果（SR/CR/TO）。
6. 記錄：arm、情境、trial#、結果、min clearance、備註（任何異常）。

---

## 7. 安全規範（no-noise 臂預期會撞）

- [ ] 機器人速度上限 **≤ 0.4 m/s**，加速度限縮
- [ ] e-stop 隨時可觸發，操作員全程在 2m 內
- [ ] 行人穿醒目衣物，預先約定「機器人逼近 0.5m 就閃開」（保護人，不計入碰撞）
- [ ] 先跑 **Arm A（no-noise）**——預期 gap 最大、最可能異常；確認車況安全後再跑 C
- [ ] 走廊淨空，移除不必要障礙（避免非實驗變數）

---

## 8. 資料記錄

每 trial 一個 rosbag + 一行 metadata：

```csv
arm,scenario,trial,result(SR/CR/TO),min_clearance_m,path_len_m,t_goal_s,jitter_omega_std,notes
A,S1,1,CR,0.12,2.3,30.0,0.41,左側擦撞柱
A,S1,2,SR,0.28,6.1,18.5,0.15,
C,S1,1,SR,0.33,6.0,17.2,0.08,
...
```

rosbag 必錄 topic（以車端 `ros2 topic list` 為準）：
- `/scan`（VLP-16 原始）、`/cmd_vel`（policy 輸出）、`/tf` `/tf_static`（車體位姿）
- `/rover_rl_policy/status`（policy 內部狀態 JSON）、goal topic、collision/contact topic

---

## 9. Metrics 計算與 gap 表

聚合每臂每情境 → 算 SR_real/CR_real/TO_real（binomial，報 95% CI）→ 算 gap：

```
Gap(arm, scenario) = SR_sim(arm, scenario) − SR_real(arm, scenario)
```

### 論文核心結果表（範本）

| Arm | SR_sim | SR_real | **Gap** | CR_real | jitter↓ |
|-----|:------:|:-------:|:-------:|:-------:|:-------:|
| A no-noise | ~100% | 低 | **大** | 高 | 高 |
| B spec-Gauss | 高 | 中 | 中 | 中 | 中 |
| C TLNI | 中高 | 中高 | **小** | 低 | 低 |

### 統計：binomial 用 Wilson 95% CI 或 Fisher's exact 比 A vs C。**不要**報「N=5 的 100% vs 80%」。

### 計算腳本骨架（PC 端，讀 rosbag + metadata CSV）

```python
# compute_gap.py（骨架，topic/型別以車端實際為準）
import pandas as pd
from rosbags.highlevel import AnyReader   # pip install rosbags
# 1. 讀 metadata CSV → 每臂每情境 SR/CR/TO 比例 + Wilson CI
# 2. (選配) 從 rosbag 重算 min_clearance / jitter：parse /tf + /cmd_vel
# 3. 合併 sim 評估表 → 輸出 gap 表
# 輸出：results_gap_table.csv + 圖 (arm × {sim,real,gap})
```

---

## 10. 寫論文時的宣稱範圍（這實驗能講什麼）

**✅ 跑完這實驗可宣稱**：
> 「在固定任務難度下，以真機量測擬合的 VLP-16 雜訊模型（TLNI）訓練之策略，其 sim→real 成功率 gap 顯著小於理想 LiDAR（A）與規格書高斯（B）訓練者；min-pool 等效分析（§3.6.3）解釋了為何天真的 per-ray 注入（B）被聚合運算吞噬而 transfer 不佳。」

**❌ 不要過度宣稱**：
- 別講「TLNI 提升 sim SR」——它的 sim SR 可能反而較低。
- 別從單一 seed / 小 N 下強結論；標明 seed 數與 CI。
- 別把行人結果當「自由場景泛化」——T2 是固定路徑，要講泛化需 T3（自由行人）。

---

## 11. 時程 checklist

| 階段 | 工作 | 預估 |
|------|------|------|
| 1 | 建 3 config（A/B/C） | 0.5h |
| 2 | 訓 3 fresh run（A/B，C 沿用） | 視 timesteps |
| 3 | Sim held-out eval（三臂 SR_sim） | 2h |
| 4 | 推 checkpoint 到車 + 驗證 r_min/ω_max | 1h |
| 5 | 真車 T2：A vs C × 3 情境 × 10 trial = 60 trial | 1~2 天 |
| 6 | （選配）B 真車、Arm D sim | 視預算 |
| 7 | Metrics + gap 表 + 圖 | 0.5 天 |

---

## 附錄 A：關鍵參考（Obsidian vault）

- `VLP-16 Sim-to-Real/LiDAR Sim-to-Real（一~四）`：實測數據 + TLNI 數學 + SA1-SA7 參數
- `論文/架構/第3章.../3.6_State 設計與 VLP-16 真實數據雜訊注入.md`：三定理、MF 定義
- `論文/架構/第5章.../5.5_Sim-to-Real 校準.md`：MF 67×→1.5× 證據
- `isaaclab_v2/架構與演算法研究/TLNI-三層雜訊注入-驗證與學術創新分析.md`：C 章待補實驗清單（本 SOP 即在補）

## 附錄 B：已驗證的車端事實（2026-06-23）

- r_min：`lidar_preprocess.py` default=0.9，但 v3c/v3e/v3f yaml 覆寫為 **0.25** ✓（與訓練一致）
- checkpoint 目錄：`~/rover_rl/models/`（現有 sa2_v3c_90k、sa3_v3c_240k）
- 非互動切換：`deploy_rl.sh model_path:=... params_file:=...`
- ROS_DOMAIN_ID=55，rmw_zenoh
