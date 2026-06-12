# v3c (SA3_v3c) 部署遷移說明

> 從 SA6_TC (79/139D) 升級到 SA3_v3c (**83D + action stacking + α slew + r_min 0.25**)。
> SA6 路徑完全保留（預設 yaml 與 79/139 obs 行為 byte 不變）；v3c 只透過新模型 + 新 yaml 啟用。

## 模型

| 項目 | 值 |
|------|-----|
| Checkpoint | `sa3_v3c_ne1024_s42/checkpoint_240000.pt` (iter 800, SR≈73%/CR≈27% @ Stage3) |
| 匯出 .ts | `models/sa3_v3c_240000.ts` (raw_obs=83, hidden=64, logits=38) |
| 觀測規格 | 見 `models/sa3_v3c_checkpoint_240000_obs_spec.md`（隨 .ts scp 到車上） |

匯出指令（PC 端，會自動偵測 83D / hidden64 / middle48 / predict13）：
```bash
python -m rover_rl_inference.export_policy \
  --checkpoint logs/rnn_car/sa3_v3c_ne1024_s42/checkpoint_240000.pt \
  --output rover_rl/models/sa3_v3c_240000.ts
```

## 83D 觀測佈局

```
[0:4]   ego        accel, vel, omega, radius
[4:6]   goal_xy    body-frame 目標
[6:78]  lidar      72 bins, r_min=0.25, ÷20m
[78]    time_rem   episode 剩餘比例
[79:83] action_stack  past 2 步 applied (a,ω)×2 — v3c 抗抽動
```

- **state_mlp 分支 = 11D** = ego(4)+goal(2)+time(1)+action_stack(4)；**lidar 分支 = 72D**。
- action_stack 正規化：`a/0.2`、`ω/(π/15≈0.209)`，clip[-2,2]（≠ 動作上限，是歷史欄正規化）。

## action_stack 來源（重要）

**不是從硬體量測**，而是 policy 自己上 2 步 `decode_logits_to_cmd` 回傳的
`(actual_accel, cmd_w)`（cmd_w 已過 slew）。`policy_node` 內維護 `deque(maxlen=2)`，
推論後 push、停車/新 goal 時清空（≈ episode reset）。詳見 `policy_node._tick_inference`。

## 四項 v3c 改動（vs SA6）

| 改動 | SA6 | v3c | 落點 |
|------|-----|-----|------|
| obs 維度 | 79/139 | **83** | `obs_builder.build_obs_83d` + `build_obs_raw` 新增 83 分支 |
| LiDAR r_min | 0.9 | **0.25** | `lidar_preprocessor_params_v3c.yaml` 的 `r_min`（程式預設不變） |
| 角速度上限 | 2.0 | **0.25π≈0.785** | `policy_params_v3c.yaml` 的 `act_max_angular_velocity` |
| 角速度 α slew | 無 | **3.0 rad/s²** | `action_decoder`（`max_angular_accel`，0=舊行為）+ yaml `act_max_angular_accel` |

> ⚠️ ω 上限與 slew 必須**同時**改：少了 slew，action_stack 的 ω 欄會與訓練不一致。

## 啟動

```bash
ros2 launch rover_rl_bringup deploy_with_bev.launch.py \
  params_file:=$(ros2 pkg prefix rover_rl_bringup)/share/rover_rl_bringup/config/policy_params_v3c.yaml \
  preprocessor_params_file:=$(ros2 pkg prefix rover_rl_bringup)/share/rover_rl_bringup/config/lidar_preprocessor_params_v3c.yaml \
  initial_mode:=idle
```

或在 v3c yaml 內 `model_path` 已指向 `sa3_v3c_240000.ts`，直接覆寫兩個 params_file 即可。

## 注意事項

- **speed_rate 建議 1.0**：action_stack 正規化與 α slew 僅在 rate=1 時與訓練「精確」對齊。
  首次上電請用「架空車輪 + estop + 保守 goal」確保安全，而非降 speed_rate。
- `episode_horizon_s=60.0` 為暫沿用值（v3c stage-3 episode_length_s 由課程動態設定）；
  實車無 episode timeout，只影響 obs[78] time-ramp，非安全關鍵。待驗證。
- 舊 SA6 模型仍可用原 `policy_params.yaml` / `lidar_preprocessor_params.yaml` 啟動，行為不變。
