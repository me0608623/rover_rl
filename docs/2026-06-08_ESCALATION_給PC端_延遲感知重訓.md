# ESCALATION → PC 端訓練組：延遲感知重訓需求

**日期**：2026-06-08
**來自**：車端（CampusRover 實車部署）
**對象**：PC 端訓練組（`/home/aa/IsaacLab/rover_rl`）
**模型**：`sa6_tc_dense_420k.ts`（SA6 T-Corridor, 420k steps）
**優先級**：高（影響實車可用性，目前 goal-following 會持續震盪）

---

## 1. 問題（一句話）

實車 goal-following 時車頭持續左右擺動（舞龍舞獅），無法平順朝目標前進。
**根因為 sim-to-real 延遲不匹配：訓練零致動延遲，實車有 ~200ms 死時間。**
車端已驗證這是 **obs→policy→cmd 閉迴路的延遲驅動極限環**，非底盤/濾波問題，
**需 PC 端重訓才能根治**（車端 obs 補償已試，反而更糟，見 §4）。

---

## 2. 證據（車端實測，2026-06-08）

資料：`~/rover_rl/logs/diag/diag_20260608_*/`（20Hz CSV，含速度三層 + 延遲）。

| 觀察 | 數據 | 含義 |
|---|---|---|
| 震盪頻率 | 0.2~0.43Hz（依路徑） | 慢繞，肉眼可見 |
| 震盪源頭 | **在 `rl_w`（policy 原始輸出）就存在** | policy 自己在震，非濾波/底盤造成 |
| 致動死時間 | **~200ms**（ω 通道互相關，兩趟一致；即時 LagEstimator 150ms, r=0.97） | ≈ 1 個 5Hz 控制步 |
| delay / control_dt | **≈ 1.0**（200ms / 200ms） | **控制理論最差相位裕度區，必出極限環** |

> 註：線速度 v 通道的互相關不可信（等速直行訊號太平、曲線全平坦，誤報 600ms）。死時間只看會起伏的 ω 通道。

### 車端已試的止血手段（皆無效或只治標）
- **cmd_vel 濾波**（`cmd_alpha_angular` / slew）：❌ 無效。震盪 0.42Hz 低於最激進濾波截止 0.71Hz，且濾波在 policy 下游。
- **`speed_rate` 降速** 0.3→0.2：只 **−20% 幅度、頻率不變**（極限環週期由死時間定，與速度無關）。治標。

---

## 3. 為什麼是「訓練」要改、而非調控制頻率

死時間是物理固定（底盤+通訊）。改善 delay/dt 比值只能降延遲（難）或改週期：
- **調快（10Hz）→ 更糟**：死時間變 2 個控制步，delay/dt=2.0，過度修正累積更多。
- **調慢（3~4Hz）→ 理論有幫助** 但反應變鈍、需重訓、是最後手段。
- **最佳解：維持 5Hz，在訓練裡把延遲建模進去**（治本、不損反應速度）。

---

## 4. 車端已嘗試的 obs 補償（失敗，供參考避免重蹈）

車端在 `policy_node` 加了推論端 cmd_delay 補償（`cmd_delay_comp_s`，預設關）：
推論前用 odom 測得速度把車姿往前預測 0.2s 再算 goal_body。

**結果反而更糟**（已用參數快照驗證對照有效）：

| 趟 | comp | 主頻 | rl_w RMS | act_w RMS |
|---|---|---|---|---|
| 基準 | 0.0 | 0.22Hz | 0.386 | 0.309 |
| 補償 | 0.2 | **0.43Hz↑** | 0.423 | **0.417 (+35%)** |

**原因**：補償在數學上是對 w 的負回饋（應 damp），但那個 damping 項本身也要過 200ms 死時間
才生效 → 變成「透過延遲的微分回饋」，過增益 → 把系統推到更高頻不穩定模態（0.22→0.43Hz）。
推論端線性預測難贏延遲；**訓練端讓 policy 自己學會處理延遲才對**。
（車端已 revert comp=0，程式碼保留預設關，不影響運作。）

**追加（2026-06-09）**：又試了「命令速度」版本（`cmd_delay_comp_src=commanded`，Smith-predictor 思路，
理論上不耦合 odom 震盪）@0.2 —— **一樣沒救、略糟、主頻同樣上移**（0.38→0.44Hz）。
→ 證實病根不是速度來源，是 **0.2s 前向預測在延遲迴路裡加太多相位領先**；且震盪本就在 policy 原始
輸出 `rl_w`。**車端 obs 預測補償（measured/commanded）已確定救不了，唯一出路是重訓。**

---

## 5. 請 PC 端做的事（依優先序）

### 🥇 首選：5Hz 不變 + 延遲感知重訓
1. **訓練時注入致動延遲**：action 進入環境前過一個 **~1 步（0.2s）的 delay buffer**，
   並做 domain randomization（建議 delay ∈ [0.1, 0.3]s 隨機）→ policy 學會不對剛下的指令立刻追加。
   （等同 spot_rl `cmd_delay` 的訓練端版本，更徹底）
2. **加 action-rate / smoothness penalty 進 reward**：懲罰相鄰兩步動作差（尤其角速度），
   直接壓 policy 天生 twitchy（震盪在 rl_w 就有 → 這條針對它）。
3. **訓練時就套部署用的 cmd 濾波**（low-pass α + slew），讓訓練/部署一致。

### 🥈 次選（首選做完仍邊際才考慮）
- 把訓練 control_dt 從 0.2s 調到 **0.3~0.33s（3~4Hz）**，使 delay/dt<1、改善相位裕度。
  代價：反應變鈍。**務必先試首選**。

### 🚫 不要做
- 調高控制頻率（10Hz）—— delay/dt 惡化，震盪必更嚴重。

---

## 6. 驗收標準（車端會這樣驗）

重訓匯出新 `.ts` 後，車端跑同場景 goal，對 `has_goal` 段的 `rl_w` 做 FFT：
- **目標：0.2~0.5Hz 頻帶的 RMS 擺幅較現行基準（~0.39 rad/s）下降 ≥ 50%**，
  且主頻不上移（不是把慢繞換成快抖）。
- 肉眼：車頭朝 goal 平順收斂，無持續左右繞。

驗證指令：
```bash
ros2 run rover_rl_inference analyze_diag "$(ls -td ~/rover_rl/logs/diag/*/ | head -1)"/*.csv
```

---

## 7. 其他規格（重訓時順帶確認，見車端 CLAUDE.md sim-to-real gap）

- **LiDAR 高度**：訓練 1.6m vs 實車 1.43m（gap #1）—— 可一併修。
- **ω_max**：訓練 2.0 vs 底盤實際 1.2（gap #2）—— policy 會叫出底盤做不到的轉速。
- **場景泛化**：目前三個 baseline 都是 T 走廊；實車若非 T 走廊需 open-scene baseline（gap #4）。

匯出：`scripts/export_policy.py`（須帶 obs_normalizer 的 checkpoint）。
車端 repo：`/home/aa/rover_rl`；PC 端 repo：`/home/aa/IsaacLab/rover_rl`。
