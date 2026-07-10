# 給 PC 端訓練組：LV-DOT → RL obstacle obs 規格確認清單

## 背景（車端現況）

我們想把車上 LV-DOT 偵測到的動態障礙物，填進 policy 的障礙物 observation 欄，
讓 RL **直接看到動態障礙**（目前 obs 障礙欄補 0、被 export 切掉，RL 完全看不到，
避障全靠外掛 VO 層）。

車端要寫一個 **adapter**（翻譯層）把 LV-DOT 輸出轉成訓練端 obstacle obs 的精確格式。
但現有匯出模型（79D v3f / 83D v3c / 139D SA6）的 obstacle channel 都被
`export_policy.py` 的 `policy_indices` 切掉了（139D 用 `range(0,78)+[138]`）。
**所以這需要 PC 端重訓 + 重匯出（obstacle channel 不再 slice）**，車端才有洞可以插。

要動工，需要你們把下列 5 項**定死**（規格對了 adapter 當天能寫死 + 用假 track 單測驗，
不用等上車）。

## 車端 LV-DOT 實際吐出的東西（vo_interface/TrackedObstacle）

| 欄位 | 內容 |
|---|---|
| `position` | 障礙物中心 (x,y,z)，**odom frame** |
| `velocity` | **絕對速度**向量 (vx,vy,vz)，**odom frame**（CV-Kalman 平滑後） |
| `size` | 邊界框 (x,y,z) |
| `radius` | 等效圓半徑 = 0.5·hypot(size.x, size.y) |
| `id` / `age` / `vel_confidence` / `covariance[4]` | 持久 ID、追蹤秒數、速度可信度、協方差 |

即：車端拿到的是 **odom frame 的絕對位置 + 絕對速度**。adapter 要轉成訓練格式。

---

## 需要定死的 5 項

### 1. 【最致命】速度是「相對」還是「絕對」？
訓練端 obstacle obs 的 velocity 欄，是：
- (a) 障礙物**絕對速度**轉 body frame，還是
- (b) **相對速度 (障礙物 − 車)** 轉 body frame？

> 車端拿到的是絕對速度(odom)。若訓練用 (b)，adapter 要先減車速再轉 body；
> 用 (a) 只轉 body。**這項錯了 policy 會把接近判成遠離、直接不閃**。
> （註：先前車端文件裡這點自相矛盾，務必一句話定死。）

### 2. 60D 的維度佈局？
60D = 幾個障礙物 × 每個幾維？每一維依序是什麼？
例：`5 obstacles × 12 dims`，每個 = `[px, py, vx, vy, size_x, size_y, radius, ...?]`？
- 座標系確認：body frame（+x 車頭、+y 左），對嗎？
- 有沒有「存在旗標 / age / confidence」欄？

### 3. 缺位怎麼填？
排序後不足額（偵測到的障礙 < 上限）的空位，obs 填什麼？
- 補 0？還是某 sentinel（例如 px=遠距離、vel=0）？
- 完全沒偵測到任何障礙時整欄的值？

### 4. 排序 key？
取「最近 N 個」是用什麼距離排序？
- body-frame 歐氏距離最近？還是含速度/TTC 加權？
- 超過 N 個障礙時多的怎麼丟？

### 5. 正規化分母 + clip？
每個欄位除以多少、clip 到哪？（要跟 obs normalizer bake 的統計一致）
- position ÷ ?（8m？量程？）
- velocity ÷ ?（1.5？對齊 max_linear_velocity_obs？）
- size / radius ÷ ?
- clip 範圍？

---

## 另外（車端這邊會補上，供你們設計 DR 參考）

車端正在用真人走動實錄 LV-DOT 的「感測器誤差模型」（延遲 / 位置抖動 σ /
速度抖動 σ / 掉幀率 / ID 跳動 / 各方位覆蓋率），輸出成 JSON 交給你們，
讓 sim 訓練時對 obstacle obs 注入**實測分佈**的 DR（延遲 buffer + 高斯噪聲 +
random dropout + FOV 盲區），避免用「零延遲乾淨 ground-truth」訓練、部署在
「遲鈍稀疏」LV-DOT 上重演 LiDAR 那種 sim-to-real gap。

⚠ **覆蓋率硬體天花板**：實測前方(相機FOV)~99%、側~16%、後~2%（VLP-16 稀疏 +
無後相機）。DR **必須**模擬 FOV 盲區（後方常態漏偵），否則 RL 會誤以為後方一定沒人。
