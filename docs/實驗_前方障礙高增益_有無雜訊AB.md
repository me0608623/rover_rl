# 實驗：前方障礙高增益反應 — 有無雜訊 A/B（sim2real 論文用）

> 目標：證明「訓練不加 LiDAR 雜訊(clean)」的模型，實車部署對真實 sweep 抖動
> 產生**高增益反應**（近距離 ω 抖動/過衝/急拉大於 noise+DR 模型），並收成 wandb 線圖 + SR。
> 對照組：`sa6_v3h_nonoise_nodr_ck270000`（clean，已上車）vs PC 端 noise+DR 臂（待上車，同架構）。
> 場景：**定點正面接近**（車正對靜止假人/牆，固定起點直線接近同一障礙，多趟）。

---

## 0. 安全（絕對遵守）

- 首次 clean baseline 對實車雜訊 robust 性低 → **架空車輪 + estop + 遙控器待命**先跑一輪確認方向。
- 落地趟務必 estop 隨手可按；任一 cmd 超 [-1,1] m/s 或 [-2,2] rad/s 立即停。
- clean 模型近障可能急拉，落地時人員站障礙側邊、非正後方。

---

## 1. 每個模型的兩種子實驗

### A. 靜態定距保持（★高增益對雜訊的最乾淨證據）
車**架空**（或 estop 鎖住不落地），車頭正對障礙，人工把車擺在固定前距，
policy 跑 nav，錄 ~30s。同一障礙、輸入只有真實 LiDAR 雜訊在變 → clean 模型 rl_w 會抖，
noise 模型平穩。距離掃：**3.0 / 2.0 / 1.5 / 1.0 m**（皮尺量到障礙）。

### B. 動態直線接近（反應曲線 + 安全指標）
車落地，固定起點（距障礙 ~4m），goal 設在障礙正後方 ~2m，讓 policy 直線朝障礙走，
記錄 v/ω 隨 front_m 收縮的軌跡、最近淨空、是否急煞/甩頭。每模型 **≥6 趟**（統計檢定要 n）。

---

## 2. 啟動（帶 experiment_tag 分組 + obs_debug）

```bash
# clean 臂
deploy_rl experiment_tag:=clean_v3h publish_obs_debug:=true \
  model_path:=/home/aa/rover_rl/models/sa6_v3h_nonoise_nodr_ck270000.ts \
  params_file:=$(ros2 pkg prefix rover_rl_bringup)/share/rover_rl_bringup/config/policy_params_v3h.yaml \
  preprocessor_params_file:=$(ros2 pkg prefix rover_rl_bringup)/share/rover_rl_bringup/config/lidar_preprocessor_params_v3h.yaml \
  initial_mode:=idle
# 或直接 deploy_rl_shell 互動選單選 v3h 條目（自動帶 v3h config），再手動加 experiment_tag。
```

> noise 臂上車後：同上，只換 `model_path`/`params_file` 與 `experiment_tag:=noise_dr`。
> `experiment_tag` 寫進每段 `_params.json` + pingpong_metrics + wandb config，分析腳本靠它分組。

**建議把定距保持改用手動開錄**（不靠 goal 到達判定）：deploy 加 `require_start:=true`，
每個距離：
```bash
ros2 topic pub --once /rover_rl/record std_msgs/String "{data: 'start hold_3p0m'}"   # 等 30s
ros2 topic pub --once /rover_rl/record std_msgs/String "{data: 'stop'}"
```
label 用 `hold_<距離>`（clean/noise 分組靠 experiment_tag，label 只標距離）。

動態接近趟則用預設 `require_start:=false`：發一個 goal＝自動開一段、到終點自動停+re-arm。

---

## 3. （選配）額外錄三層 LiDAR bag 供 E1/E2

另一 terminal，與 diag CSV 靠牆鐘對齊：
```bash
# 靜態定距（要 raw 點雲量 sweep 層雜訊 E1）
scripts/record_experiment.sh clean_hold_3p0m
# 動態接近（趟多，省 raw）
scripts/record_experiment.sh clean_approach_t1 --no-raw
```

---

## 4. 分析 + wandb 線圖

跑完 clean（之後補 noise）後：
```bash
# 依 experiment_tag 分組（clean_v3h vs noise_dr）；--near 近帶上界；--wandb 推線圖
python3 scripts/analyze_front_reaction.py --group-by tag --since <當天日期> \
  --near 2.0 --wandb --wandb-mode online --project rover_rl_sim2real
```
產出：
- **wandb 面板**（clean vs noise 疊同軸）：`omega_chatter_std_vs_dist`（★頭號）、
  `mean_abs_omega_vs_dist`、`domega_rms_vs_dist`、`mean_v_vs_dist`、`omega_saturation_vs_dist`
  + `highgain_summary` 表 + 每組純量 summary。
- **本地**：`logs/analysis/front_reaction_<stamp>/` 內 `reaction_curves.png`、`curves.csv`、
  `summary.{csv,md}`（近帶高增益表 + 兩組 Mann-Whitney U p 值，論文直接引用）。

SR/TO/CR（任務級）另用既有：`ros2 run rover_rl_inference pingpong_report`（讀 pingpong_metrics_*.csv，
依 experiment_tag+model 分組輸出 console 表 + .tex）。跨趟導航效能用 `scripts/aggregate_diag.py`。

---

## 5. 論文預期結論（既有歷史資料已見雛形）

近帶(front_m≤2m) clean(v3h) ω 抖動 std ≈ 0.68 vs noise 家族 ≈ 0.48（高約 40%），
mean|ω|、Δω RMS 亦系統性偏高 → **無雜訊訓練→實車高增益**。受控定點接近會讓差距更顯著、曲線更乾淨。
