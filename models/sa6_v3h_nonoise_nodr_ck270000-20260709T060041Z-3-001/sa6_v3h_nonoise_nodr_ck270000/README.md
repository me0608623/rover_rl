# 部署包：sa6_v3h_nonoise_nodr_ck270000

SA6 v3h **clean baseline 控制臂**（machine B / 5070，無 LiDAR 雜訊 + 無 DR）。
★SA6 於 iter ~993/1400 **手動中止**，此為最新已存 checkpoint（iter 900）。
sim2real 對照臂，量 sim→real gap，**不是雜訊 treatment**。

## 包內容

| 檔案 | 用途 |
|------|------|
| `checkpoint_270000.pt` | 訓練 state_dict（含 **obs_normalizer** 79D + **feat_normalizer** 96D）· sha256 `458164ff…` |
| `sa6_v3h_nonoise_nodr_ck270000_deploy_info.yaml` | 部署資訊卡 |
| `sa6_v3h_nonoise_nodr_ck270000.obs_spec.json` | 機器可讀 obs/action 規格 |
| `sa6_v3h_nonoise_nodr_ck270000.obs_spec.md` | 人看欄位表 + 正規化公式 |
| `full_args.json` | 完整訓練 args |
| `README.md` | 本檔 |

## 關鍵參數（車端對齊用）

| 項目 | 值 |
|------|-----|
| raw_obs / used_obs | **79 / 79** |
| encoder / rnn | extractor_rnn / **GRU** hidden 64，preprocess_dim 12，feat_norm **on** |
| act_hist / frame_stack | **無** / 1 |
| lidar | bins 72，r_min 0.25，r_max 20.0，r_robot 0.35，normalize `clamp(d−0.35,0)/20` |
| 動作 | MultiDiscrete([19,19])=38 logits，deterministic=argmax_per_head |
| 物理 | v_max 1.0，a_max 0.5，**omega_max 1.2**，dt 0.2，v_reverse −0.2 |

## ⚠️ 尚未產生 TorchScript（.ts）

本機（+所有 git branch）**沒有 `scripts/export_policy.py`**，無法 bake `.ts`。此包提供原始 .pt + 完整規格。
要 .ts 在有 export 腳本的機器（machine A / 車端 rover_rl）跑，務必把 obs_normalizer + feat_normalizer 一起 bake（本 .pt 內都有）。

## scp 到車端

```bash
scp -r models/sa6_v3h_nonoise_nodr_ck270000 <車端>:/home/aa/rover_rl/models/
```

## 注意事項

- ★這顆是 **SA6 中止 (iter ~993/1400) 的 best 存檔點 (iter 900)**，train SR ~89% 平台，非完訓。
- ★obs_normalizer/feat_normalizer 不 bake → 車原地爆走（頭號部署失敗）。
- ★car act_max_* 必須完全一致（尤其 omega_max=1.2，勿設 0.785），勿 extrapolate。
- 對應車端管線：v3f 家族（79D obs、act_hist OFF、r_min 0.25）。
- 架構與同 run 較早的 `sa6_v3h_nonoise_nodr_ck210000` 包完全相同，僅 checkpoint 步數不同（270000 > 210000，較晚較優）。
