# RNN Hidden State 在實車的處理

> rover_rl 用的是 vanilla **RNN（hidden_dim=30 for SA6_TC; 64 for SA1_v2）**。
> 部署時必須在每次推論之間正確維護 hidden state，否則 policy 等於沒有時序記憶。

## 1. 核心問題

訓練時，環境是 **episodic**：
- Episode 開始 → hidden state = 0
- 每步：`new_hidden = rnn(feat, hidden); hidden ← new_hidden`
- Episode 結束 → 重置 hidden

實車**沒有 episode 概念**，但 RNN 仍需要：
- 連續累積 hidden state 跨 5 Hz 推論
- 在「邏輯上的新 episode」時 reset（新 goal、緊急停車後恢復、模型切換）
- Reset 時機錯了 → policy 用舊場景的 memory 跑新場景 → 行為怪異

## 2. rover_rl 的處理（簡單、對齊訓練）

### 2.1 儲存位置

```python
# rover_rl_inference/model_runtime.py
class PolicyRunner:
    def __init__(self, bundle: PolicyBundle):
        self.bundle = bundle
        self.hidden = torch.zeros(
            1, 1, bundle.hidden_dim, device=bundle.device, dtype=torch.float32
        )

    def reset(self) -> None:
        self.hidden.zero_()

    @torch.no_grad()
    def step(self, obs_raw_np: np.ndarray) -> np.ndarray:
        obs79 = self.bundle.preprocess(raw)
        feat = self.bundle.extractor(obs79)
        preprocess, new_hidden = self.bundle.rnn(feat, self.hidden)
        self.hidden = new_hidden                    # ← 持續累積
        rl_input = torch.cat([obs79, preprocess], dim=-1)
        return self.bundle.policy(rl_input).squeeze(0).cpu().numpy()
```

每次 inference tick (5 Hz)：
1. 讀當前 `self.hidden`
2. RNN forward → `new_hidden`
3. `self.hidden ← new_hidden`（覆寫，但不清零）

完全對應訓練端 `RNNStateManager`：

```python
# scripts/.../modular_rnn_models.py
class RNNStateManager:
    def update(self, new_hidden):
        self.hidden = new_hidden.detach()
    def reset(self, env_ids):
        self.hidden[:, env_ids, :] = 0.0
```

### 2.2 何時 reset hidden

| 事件 | 動作 | 原因 |
|---|---|---|
| 收到新 `/goal_pose` | `self.runner.reset()` | 新 episode 開始 |
| 收到新 `/global_path` | `self.runner.reset()` | 新路徑 = 新導航段 |
| Service `~/reset_hidden` 被呼叫 | reset | 手動干預 |
| Hot-swap model (`~/load_model`) | reset | 新 model 對應新 hidden shape，必須重來 |
| Mode 切換到 `idle`/`estop`/`paused` | **不 reset** | 切回 `nav` 時希望保留情境 |
| 緊急停車自動觸發 ESTOP | **不 reset** | 解 ESTOP 後想接續前一個情境 |

### 2.3 Inference 暫停時的 hidden 處理

**重要**：mode = `idle`/`estop`/`paused` 時，`_tick_inference` 直接 `return`，
不呼叫 `self.runner.step()`，所以 hidden 也不更新。**hidden 凍結在最後一次 nav 推論的值**。

切回 nav 後 hidden 就從凍結點繼續累積。

**為什麼這樣設計？**
- 訓練時，episode 中途若機器人「停下」也只是 v=0，但 RNN 仍持續吃 obs 累積 hidden
- 部署 mode=idle 時不發 obs 給 RNN → 與訓練分布略有偏差
- 但這個偏差在實務上不嚴重（RNN 對短暫凍結容忍度高）
- 替代方案：idle 時仍跑 RNN，只是不發 cmd → 增加 CPU 沒必要

如果發現切回 nav 後表現異常，可以改：mode 切換時 reset hidden（更保守）。

## 3. spot_rl 怎麼處理 RNN

spot_rl 用更**複雜**的時間戳記 memory dict（含 delay compensation）：

### 3.1 結構

```python
# spot_rl/src/system/ai_model_action.py
self.memory = torch.zeros(memory_dim)
self.momory_dict = {}           # {timestamp: memory_state}

def inference_tick(self):
    # 1. 從 dict 找「target_time = now - 1/rl_fps」時間最接近的 memory
    target_time = time.time() - 1.0 / self.rl_fps / self.dynamic_speed_rate
    for momory_time, momory_data in self.momory_dict.items():
        if abs(momory_time - target_time) < min_diff:
            self.memory = momory_data  # 選最近的

    # 2. 用該 memory 跑 RNN
    output, value, new_memory, _ = self.pre_train_model(
        obs=dynamic_observation, memory=self.memory,
    )

    # 3. 存新 memory 進 dict
    self.momory_dict[time.time()] = new_memory

    # 4. 清掉太舊的
    # for old keys > N seconds ago: del self.momory_dict[key]
```

### 3.2 為什麼這麼做？

spot_rl 處理的是「**指令延遲補償**」：
- spot 的指令發出到實際執行有可觀延遲（電機反應、無線傳輸）
- 當前 obs 反映的是「過去 N ms 的狀態」
- 用「N ms 前的 memory」跑 RNN → 時序對齊更好

但**這對 rover_rl 不適用**：
1. CampusRover 是有線、低延遲底盤（< 50 ms）
2. 訓練端也沒做這種補償（IsaacLab 直接同步 obs↔action）
3. rover_rl 訓練端與部署端用一致的 "current obs + current hidden" 模式 → **直接照搬最簡單**

### 3.3 spot_rl HRL 上層另一種寫法

```python
# spot_rl/.scripts/HRL_ros.py
model_output, _state = self.HRL_model_.predict(self.HRL_model_input_)
#                      ^ 直接丟掉 _state
```

這是 stable-baselines3 PPO API；`_state` 是 RNN hidden，但因為 HRL 用 MLP policy，
不關心 hidden → 直接丟掉。

## 4. rover_rl vs spot_rl 對照

| 項目 | rover_rl | spot_rl (ai_model_action) | spot_rl (HRL_ros) |
|---|---|---|---|
| Hidden 儲存 | `self.hidden` 一份 | `self.momory_dict[ts]` 時間索引 | 丟掉 |
| 累積方式 | 每 tick 覆寫 | 取 target_time 最近的 | 不累積 |
| Delay 補償 | 無 | 有（用過去 memory） | N/A |
| Reset 時機 | 新 goal/path | 應該也是新 episode | N/A |
| 與訓練端對齊 | ✓ 完全一致 | △ 訓練時不延遲 | N/A |
| 複雜度 | 低 | 中（dict + 時間查找） | 最低 |

**rover_rl 選擇最簡單對齊訓練的做法**，因為：
1. CampusRover 沒有顯著 cmd delay
2. 訓練端 RNNStateManager 也是這個 pattern
3. 程式碼少 → 出錯機率低

## 5. 部署時的 hidden 行為驗證

### 5.1 確認 hidden 有累積

```bash
# 啟動 policy 並訂閱 obs_debug
ros2 launch rover_rl_bringup deploy_with_bev.launch.py
ros2 topic echo /rover_rl_policy/obs_debug --once

# 假設手動發兩次同樣的 goal，policy 應該每次的 action 都不太一樣（因為 hidden 累積）
ros2 topic pub --once /goal_pose ...
# 看 cmd_vel 行為

ros2 service call /rover_rl_policy/reset_hidden std_srvs/srv/Trigger
# 再發同樣 goal，初始 action 應該回到「初始 hidden=0」狀態
```

### 5.2 確認 reset 時機正確

啟動時開 debug log，發 goal 應該看到：

```
[INFO] 收到 goal_pose frame=map (3.00, 0.00)
```

policy_node 在 `_cb_goal()` 內呼叫了 `self.runner.reset()`，可以加 debug log 驗證
（已內建：`runner.reset()` 是顯式呼叫，可從程式碼追溯）。

### 5.3 緊急停車後 hidden 行為

```
[WARN] EMERGENCY: LiDAR 進入安全區
[INFO] mode: nav → estop (reason: LiDAR < safety_estop)

(手動清除危險)
ros2 topic pub --once /rover_rl_policy/mode std_msgs/String "data: 'nav'"

[INFO] mode: estop → nav
```

切回 nav 後，hidden **保留 estop 前的狀態**繼續累積。如果想清除：

```bash
ros2 service call /rover_rl_policy/reset_hidden std_srvs/srv/Trigger
```

## 6. 與訓練端的細節對齊

| 訓練端（`models/modular_rnn_models.py`） | 部署端（`model_runtime.py`） | 對齊 |
|---|---|---|
| `RNNStateManager.__init__: hidden=zeros` | `PolicyRunner.__init__: hidden=zeros` | ✓ |
| `update(new_hidden): self.hidden = new_hidden.detach()` | `self.hidden = new_hidden` | ✓ |
| `reset(env_ids): self.hidden[:, env_ids, :] = 0` | `reset(): self.hidden.zero_()` | ✓ |
| rollout 內逐 step 跑：input=current obs + current hidden | tick 內逐次跑：input=current obs + current hidden | ✓ |
| Episode 結束時 reset | 新 goal/path 時 reset | ✓ 概念相同 |
| Hidden shape: `[1, num_envs, hidden_dim]` | `[1, 1, hidden_dim]` (單環境) | ✓ |
| Forward: `rnn(features, hidden) → (preprocess, new_hidden)` | 同 | ✓ |

**完全對齊**。實車推論時的 RNN 行為與訓練 rollout 一致。

## 7. 常見誤區

❌ **誤區 1：每 tick reset hidden**
→ 等同於 MLP policy，失去時序記憶優勢。RNN 部署的價值就是跨步累積資訊。

❌ **誤區 2：以為 ROS 重啟自動清 hidden**
→ rclpy spin 期間 Python 物件持續存在；只有 `node.destroy_node()` 後才會清。
中途切 mode 不會清。

❌ **誤區 3：Hot-swap model 後忘記 reset**
→ 新 model 的 hidden_dim 可能不同（30 vs 64），shape 不對會 RNN forward 失敗。
`~/load_model` service 內部已自動 reset。

❌ **誤區 4：用 `time.sleep()` 期間 hidden 仍會更新**
→ 不會。沒呼叫 `runner.step()` 就不更新。

❌ **誤區 5：CPU 與 GPU 切換不清 hidden**
→ tensor device 不同時 forward 會 error。切換 device 必須 reset 或手動 `hidden.to(device)`。

## 8. 故障排除

| 症狀 | 可能原因 | 修復 |
|---|---|---|
| 切到 nav 後車原地震 | hidden 帶舊場景情境 | `ros2 service call ~/reset_hidden` |
| 接連兩個 goal 第二個表現怪 | _cb_goal 應該已自動 reset，檢查 log 是否有 "RNN hidden state 已重置" | 升級 rover_rl |
| Hot-swap model 後 forward 失敗 | hidden_dim 不匹配 | 確認 `~/load_model` service 有 reset；如果沒 → bug |
| `ros2 topic echo obs_debug` 看 obs 正常但 cmd 怪 | hidden 累積了異常值（NaN？） | reset_hidden，看是否解決 |
