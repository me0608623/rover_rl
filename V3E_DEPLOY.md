# V3E_DEPLOY — rover_rl 部署 v3e（sa1/sa2_v3e）

> 給車上 Claude / 操作員的 v3e 部署速覽。架構與 v3c 相同（83D），靠「換模型 + 換 yaml」切換。
> 最後更新：2026-06-17（PC 端訓練組匯出）。v3c 段見 `V3C_DEPLOY.md`，總覽見 `CLAUDE.md`。

## 1. v3e 是什麼（vs v3c）

v3e 與 v3c **網路架構完全相同**（raw_obs=83 / RNN hidden=64 / fc=64 / middle=48 / predict=13 / RNN，logits=38），
所以沿用 v3c 的 83D 部署管線。只有兩處不同：

| 項目 | v3c | **v3e** |
|------|-----|---------|
| 動作角速度上限 `act_max_angular_velocity` | 0.785（已知 bug） | **1.2**（修正＝訓練實際值，底盤 profile_omega_max 也是 1.2） |
| `obs[81:83]` 語義（act_hist 後 2 維） | raw：`a_{t-2}, ω_{t-2}` | **action_error：致動追蹤誤差 `err_v, err_w`** |

模型：
- `sa2_v3e_240000.ts` — 動態場景最終（**預設**）
- `sa1_v3e_60000.ts` — 靜態（= SA2 起點）

## 2. 啟動方式

```bash
deploy_rl_shell        # 選單選 sa2_v3e_240000.ts → 自動帶 v3e config + 印觀測語義表
```

選到 `*v3e*` 會：
1. 印出該 checkpoint 的**觀測維度語義表**（讀 `models/<model>.obs_spec.md`）。
2. 自動帶 `policy_params_v3e.yaml` + `lidar_preprocessor_params_v3e.yaml`。
3. policy_node 啟動時讀 `models/<model>.obs_spec.json` → 自動進 `action_error` 模式。

> **模型自描述**：每個 `.ts` 旁都有 `<model>.obs_spec.json`（機器讀）+ `.obs_spec.md`（人看）。
> 語義跟著模型走，換 checkpoint 不會用錯 act_hist 填法。由 PC 端 `export_policy.py` 產生。

## 3. ⭐ obs[79:83]（act_hist）與四個參數

83D 模型 obs 最後 4 維 = 「policy 看到自己上一步的動作」，給它煞車/阻尼訊號抗抽動：

```
obs[79]   obs[80]   | obs[81]   obs[82]
a_{t-1}   ω_{t-1}   |   ?         ?
(上一拍指令，兩模式相同) | (依 act_hist_mode 不同)
```

- **raw 模式（v3c）**：後 2 維 = 上上拍指令 `a_{t-2}, ω_{t-2}`。
- **action_error 模式（v3e）**：後 2 維 = **err（致動追蹤誤差）= 馬達沒跟上的量**。

四個 ROS 參數（在 `policy_params_v3e.yaml`，平常不用動）：

| 參數 | 預設 | 說明 |
|------|------|------|
| `act_hist_mode` | `auto` | `auto`＝讀模型 obs_spec.json 自動判（v3e→action_error）。可硬填 `raw`/`action_error` 覆寫 |
| `act_hist_err_source` | `measured` | **err 來源**（見下節）。`measured` / `zero` |
| `act_err_v_max` | `1.0` | err_v 正規化分母（訓練 max_linear_velocity） |
| `act_err_w_max` | `1.2` | err_w 正規化分母（訓練 max_angular_vel） |

## 4. ⭐⭐ act_hist_err_source 詳解（最重要的調整鈕）

### err 是什麼

> **err =「我叫車跑多快」−「車實際跑多快」**＝ 馬達沒跟上的量。

訓練時在模擬器內 err = 指令速度(pre-DR) − 實際寫入 sim 的速度(post-DR)，代表**馬達延遲/響應不及**。
穩定直行時馬達跟得上 → err ≈ 0。v3e 把它餵進 obs，讓 policy 知道指令有沒有被執行到（抗致動延遲，論文 §34）。

### measured vs zero

真車上要怎麼得到 err？這就是 `act_hist_err_source` 控制的：

| 值 | 怎麼算 err | 白話 | 取捨 |
|----|-----------|------|------|
| **`measured`**（預設） | `err = 上拍指令速度 − odom 實測速度`，正規化後 clamp[-1,1] | 真的去量指令 vs 實測，最貼近訓練語義 | 最忠實，但 odom 噪聲會回授 → **可能誘發振盪** |
| **`zero`** | `err = 0` | 當作馬達完美跟上、無誤差 | 訓練穩態 err 本來就≈0，安全；不灌 odom 噪聲 |

公式（measured）：
```
err_v = clip((上拍指令_v − odom實測_v) / act_err_v_max(1.0), -1, 1)
err_w = clip((上拍指令_w − odom實測_w) / act_err_w_max(1.2), -1, 1)
```
（內部用 policy-frame，即與 ego 速度同樣 ×inv＝1/speed_rate；1 步延遲＝odom 實測本身是上拍指令的實現結果。）

### ⚠️ 為什麼要有 zero（抗振盪安全開關）

`measured` 的 err 被餵回 obs 又影響下一步動作 → 可能形成**自我回授振盪**（PC 端 `finding_action_error_selfref_oscillation`
記錄過：實車曾出現車頭規律左右擺＝「舞龍舞獅」）。訓練 err 穩態≈0，故 `zero` 是合理又安全的近似。

### 實際操作

- 預設先用 `measured`（最忠實還原訓練）。
- **上路若車頭開始規律左右擺**，現場切 zero（不用重啟，立即生效）：
  ```bash
  ros2 param set /rover_rl_policy act_hist_err_source zero
  ```
  擺動消失 → 確認是 err 回授引起，維持 zero。

> v3c / 79D 舊模型完全不受影響（raw 模式，走原邏輯，這 4 個參數不作用）。

## 5. 首次上路安全（同 v3c）

1. **架空車輪 + estop + 遙控器待命**。
2. 確認 `deploy_rl_shell` 印的語義表顯示 `act_hist_mode: action_error`、ω_max=1.2。
3. 看 launch log `raw_obs=83`、`[act_hist] mode=action_error err_source=measured`。
4. 車頭舞龍舞獅 → 先 `act_hist_err_source zero`；仍不行 → estop，回報 PC 端。

## 6. 改了哪些檔（2026-06-17，原檔皆有 .bak）

- `src/rover_rl_inference/.../policy_node.py` — mode-aware act_hist + `_resolve_act_hist_mode` + 4 參數
- `src/rover_rl_inference/.../export_policy.py` — 匯出產生 obs_spec sidecar
- `src/rover_rl_bringup/config/policy_params_v3e.yaml`、`lidar_preprocessor_params_v3e.yaml`（新）
- `deploy_rl_shell.sh` — 選 checkpoint 印語義表 + 認 v3e
- `models/sa{1,2}_v3e_*.ts` + `.obs_spec.{json,md}`；另補 v3c 模型的 obs_spec

> symlink-install → 改 policy_node.py / yaml 不用 rebuild，下次 `deploy_rl_shell` 生效。
