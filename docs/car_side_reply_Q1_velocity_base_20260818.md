# 回覆 Q1：積分基準是 **issued**，不是量測

2026-08-18，PC 訓練端 → 車端。針對 `2026-08-17_速度變化量與位姿guard_給PC訓練端.md` §5 Q1。

---

## TL;DR

1. **訓練端的積分基準 `v` = 上一拍 post-delay 的 issued 速度**，是一個純積分器狀態，
   **從不回讀物理量測**。
2. 車端目前餵量測值（`policy_node.py:1289`）→ **與訓練不一致，要改**。
3. **改完，Q6 的起步死鎖會自己消失** —— 那兩題是同一個根因，不需要另外加死區補償。
4. ⚠️ **`v` 與 `ω` 的基準不同，不要統一**：`v` 用 post-delay，`ω` slew 用 pre-delay。
5. 順帶回覆 Q3，並回報一個你們沒問到、但比 Q3 嚴重的不符（§7）。

---

## 1. 證據：逐行追蹤

檔案：`source/isaaclab_tasks/.../charge_skrl/mdp/actions/discrete_differential_drive.py`

`_current_velocity` 在全檔只有這幾個出現點：

```python
 83:  self._current_velocity = torch.zeros(N, device=self.device)   # 初始化為 0
202:  v = self._current_velocity                                    # ← 積分基準（就是 Q1 問的 v）
309:  self._current_velocity[:] = next_velocity                     # ← 唯一的寫入點
323:  self._current_velocity[ids] = 0.0                             # reset 歸零
340:  local_velocity[:, 0] = self._current_velocity                 # 寫進模擬器
377:  vel_goal_xy = self._current_velocity.unsqueeze(1)             # 只是畫箭頭
```

第 309 行是**唯一**的資料來源，而 `next_velocity` 是解碼器自己算出來的：

```python
251:  next_velocity = (v + actual_linear_accel * dt).clamp(-v_max_neg, +v_max_pos)
      ...  actuator DR（延遲 / 速度尺度 / 一階 lag）在此套用  ...
283:  next_velocity = target_vel[:, 0].clamp(-v_max_neg, +v_max_pos)   # post-delay
309:  self._current_velocity[:] = next_velocity                        # 存的是 post-delay 的值
```

**沒有任何一行讀 `root_lin_vel` 或其他物理量測。**

結論：`v` = **上一拍 post-delay 的 issued 速度**。

---

## 2. 為什麼訓練期完全看不出這個問題

第 340 行把這個速度**直接寫進剛體**：

```python
340:  local_velocity[:, 0] = self._current_velocity
      global_linear_velocity = math_utils.quat_apply(robot_quat_w, local_velocity)
      → write_root_velocity_to_sim
```

模擬裡**沒有馬達**。命令多少，剛體就是多少。

> 所以在 sim 中「量測速度」與「issued 速度」是**同一個數字**，
> 訓練期無法區分這兩種定義，任何單元測試 / parity 檢查也不會失敗。

這也是為什麼 e2e parity oracle 過了、契約 MANIFEST 全 PASS，卻仍有這個落差 ——
**parity 驗的是網路圖，不是積分語意**。

真車上兩者才分岔：你們量到 v 跟隨率 96~126%、ω 只有 28~44%。

---

## 3. ⚠️ v 與 ω 的基準不同，不要統一

這是最容易改錯的地方。

```python
# ω 的 slew 基準：pre-delay（第 261 行，在 actuator DR 區塊之前）
238:  target_angular_vel = ratio_angular * self.cfg.max_angular_vel
240:  actual_angular_vel = self._commanded_omega + clamp(
          target_angular_vel - self._commanded_omega, -max_dw, +max_dw)
261:  self._commanded_omega[:] = actual_angular_vel      # ← pre-delay

# v 的積分基準：post-delay（第 309 行，在 actuator DR 區塊之後）
309:  self._current_velocity[:] = next_velocity          # ← post-delay
```

| 量 | 下一拍的基準 | 語意 |
|---|---|---|
| `v` | **post-delay** issued | 車體實際在跑的速度（sim 中即寫入剛體的值） |
| `ω` slew | **pre-delay** commanded | slew 限制的是「命令變化率」，所以沿命令走 |

這個不對稱是**刻意的**，不是 bug。ω 的 slew 若改沿實際 ω，會在延遲存在時產生額外的
正回授。請照原樣複製，不要為了「一致性」把兩者統一。

---

## 4. 車端要怎麼改

### 現況

```python
# policy_node.py:1289 附近
v_base = <odom 量測速度>
next_v = clip(v_base + accel * dt, ...)
```

### 應改為

```python
# 維護一個純積分器狀態，初始 0，只由自己更新
self._issued_v = 0.0          # 對應訓練端 _current_velocity

# 每拍：
next_v = clip(self._issued_v + accel * dt, -v_max*reverse_scale, +v_max)
#   ↑ 基準用上一拍存下來的值，不是 odom

sent_v = next_v               # 實際送出（若有 speed_rate 縮放，見下方注意事項）
self._issued_v = next_v       # 存回，供下一拍使用
```

### reset 時要歸零

對應訓練端第 323 行。以下情境都要把 `_issued_v` 歸零：

- 新的 goal / episode 開始
- `pose_jump_guard` 觸發
- odom timeout / RNN reset
- 任何會清 `act_hist` 的時機

**歸零時機必須與清 `act_hist`、`lidar_hist` 完全一致**，否則三者的時間基準會錯開。

### ⚠️ `speed_rate` 的處理

如果 `speed_rate < 1.0`，請確認縮放是在**送出前**做、而積分器存的是**未縮放**的值 ——
否則積分器會以縮放後的尺度累積，等效於偷偷改了 `a_max`，與訓練不符。

```python
next_v = clip(self._issued_v + accel * dt, ...)   # 未縮放
self._issued_v = next_v                            # 存未縮放
sent_v = next_v * speed_rate                       # 只有送出時縮放
```

（這一點與 Q4 的 act_hist 偏差是相關的，但 Q4 我還沒查完，見 §8。）

---

## 5. 這同時解掉 Q6（起步死鎖）

兩題是同一個根因。

| 拍 | 車端現況（餵量測 v） | 改成 issued 後 |
|---|---|---|
| 1 | odom 0 → 命令 `0.5×0.2×0.6` = **0.060** | 0 → **0.060** |
| — | 0.060 < 底盤死區，輪子不轉 | 同樣不轉 |
| 2 | odom 仍 **0** → 又是 **0.060** | 基準 0.060 → **0.120** |
| 3 | odom 仍 0 → **0.060** | 基準 0.120 → **0.180** ← 突破死區 |
| 結果 | **永遠 0.060，死鎖** | 正常起步 |

我從你們的 diag CSV 獨立確認了這個機制：

```
20:47:02   sent_v p50 = 0.060   odom_v p50 = 0.000   前進 0 m   TIMEOUT
20:41:29   sent_v p50 = 0.100   odom_v p50 = 0.000   前進 0 m   TIMEOUT
```

> **所以不需要為 Q6 另外加死區補償。** 加了反而會偏離訓練語意，
> 因為訓練端從來沒有死區補償這個東西。

---

## 6. 改完怎麼驗

### 驗收 1：靜止起步（架空即可）

給一個前方無障礙的 goal，記錄前 10 拍的 `sent_v`。

```
期待：0.060 → 0.120 → 0.180 → 0.240 → ...   （每拍 +a_max×dt×rate）
錯誤：0.060 → 0.060 → 0.060 → ...           （沒改到 / 改錯位置）
```

這個測試**架空就能做**，不需要地面。

### 驗收 2：積分器不被 odom 污染

人為擋住輪子（或架空），確認 `sent_v` 仍會持續爬升到 `v_max`。
若停在某個值不動，表示基準還在讀量測。

### 驗收 3：reset 一致性

觸發一次 `pose_jump_guard`，確認 `_issued_v`、`act_hist`、`lidar_hist`
三者在**同一拍**歸零。

### 驗收 4：回歸

改完重跑一次 e2e parity。**parity 應該完全不受影響**（它不碰積分器），
若 parity 變了表示動到不該動的地方。

---

## 7. 順帶回覆 Q3，以及一個你們沒問到的更嚴重問題

### Q3 前半：延遲模型 — 範圍對，中心偏低

我從你們 9 段共 **7,114 個有效樣本**（`lag_ms` 欄，`lag_corr` 0.88~0.98）算：

```
p10 =   0 ms
p50 = 300 ms      ← 1.5 個控制拍
p90 = 400 ms
```

訓練端設定：

```python
actuator_delay_range = (0, 2)    # U{0,1,2} 步 = 0 / 200 / 400 ms，均值 200 ms
```

**支撐範圍涵蓋到了，但中心低了 100 ms**，且約 10% 的樣本達到或超過訓練上限。
建議下次訓練改 `(1, 2)` 或 `(1, 3)`。這是訓練端的事，車端不用動。

### Q3 後半：一階 ω lag 模型 — 已經有了

```python
actuator_motor_lag      = 0.3          # 一階 lag α
actuator_velocity_scale = (0.9, 1.1)   # 速度尺度
```

### ★ 但第二行有嚴重問題

| | 訓練假設 | 你們實測（rate 1.0） |
|---|---|---|
| ω 尺度 | **0.9 ~ 1.1** | **0.28 ~ 0.44** |

**訓練期告訴 policy「底盤會給你 90~110% 的角速度」，現實只給 28~44%。**

這不是偏一點，是量級錯。policy 一直在對一個「它以為很聽話、實際只做三分之一」的
底盤下命令 —— 這就是它每拍加碼、打滿 slew、然後左右急甩的來源，
與你們 §7.2 量到的「ω 飽和佔比」「急甩次數」完全一致。

這是**訓練端要修的**，會需要重訓。但請車端幫忙確認一件事：

> 那個 28~44% 是**穩態跟隨率**，還是**含延遲的瞬時比值**？
> 若是後者，真正的穩態尺度可能更接近 1.0，而問題主要在延遲而非尺度。
> 判別法：取一段 `sent_w` 維持同號且幅值穩定 ≥ 1.5 s 的窗口，
> 比較該窗口**末段**的 `odom_w / sent_w`。

這個區分會直接決定訓練端要改 `actuator_velocity_scale` 還是 `actuator_delay_range`，
**兩者的修法不同，不能一起亂調**。

---

## 8. 尚未回覆的問題

| 題 | 狀態 |
|---|---|
| Q2（訓練端 Δω 分布是否也貼滿 slew） | 未查 |
| Q4（`speed_rate=0.6` 的 act_hist 偏差可否接受） | 未查 |
| Q5（訓練時倒車出現頻率） | 未查 |

這三題需要跑訓練端的統計，本機 GPU 剛空出來，會另外回覆。

---

## 9. 這次沒有改變的東西

- 模型檔、golden、parity 門檻：**不變**
- 83D 佈局、解碼器參數、act_hist 除數（0.5 / 1.2）、`r_min=0.5`：**不變**
- `episode_horizon_s`（SA1–SA4 = 60）：**不變**
- SA4 仍為 **HOLD**、`accepted_parent = null` —— Q1 修好也不代表能力達標，
  它修的是**契約一致性**，不是導航能力

---

## 附：我這邊的驗證邊界

- Q1 的結論來自逐行追蹤 `_current_velocity` 的全部出現點，**確定**。
- obs 的 `speed` 那一維我沒找到對應函式（`a_bar` / `omega_bar` 已確認是解碼器 property）。
  但因 `_current_velocity` 就是寫進剛體的值，任何物理讀取在數值上都等於它，
  對本文的建議無影響。
- 延遲統計用的是你們 CSV 的 `lag_ms` 欄，我沒有自己從 `sent_w`／`odom_w` 重做互相關。
  `lag_corr` 0.88~0.98 顯示該欄可信，但這是**採信你們的量測**，不是獨立複驗。
