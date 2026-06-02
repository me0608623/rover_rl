---
name: start
description: rover_rl 上路前 preflight + 上路後即時盯場 — 透過 ROS MCP 確認系統健康、看 camera 確認場景、盯 cmd_vel 晃動與是否朝 goal。使用者打 "start" 時啟動。
user-invocable: true
---

# /start — rover_rl preflight + 即時盯場

**記錄是自動的**：`deploy_rl` 後 diag_logger 已待命，使用者一發 goal 就自動開始寫
`~/rover_rl/logs/diag_<時間>.csv`（不需任何 start 指令）。所以本 skill **不負責開錄**，
專做兩件事：**(A) 發 goal 前的 preflight 確認、(B) 發 goal 後的即時盯場**。

**核心分工：記錄由 diag_logger 節點負責（20Hz 可靠）；你用 ROS MCP 只做確認與盯場，
不要把 MCP 當記錄器。**

## Step 0：連線 ROS MCP

1. `connect_to_robot(ip="127.0.0.1", port=9090)`
   - 失敗 → 提醒使用者：另一終端要先 `source ~/rover_rl/setup_env.sh` 再
     `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`（同 zenoh DOMAIN=55）。停在這。

## Step 1：Preflight 檢查（開錄前確認系統健康）

用 ROS MCP 逐項確認，任何一項紅燈就**先報告、不要急著開錄**：

| 檢查 | 方法 | 通過條件 |
|---|---|---|
| 必要 topic 在線 | `get_topics()` | 有 `/odom` `/ndt_pose` `/velodyne_points` `/rover_rl/lidar_sweep_72` `/input/nav_cmd_vel` |
| odom 在動 | `subscribe_once("/odom")` | 有訊息、frame 合理 |
| NDT 已收斂 | `subscribe_once("/ndt_pose")` | 有訊息且 position 穩定（連兩次差 < 0.3m） |
| LiDAR 看得到東西 | `subscribe_once("/rover_rl/lidar_sweep_72")` | 不是全 1.0（全 1.0=沒回波，別開） |
| policy 活著 | `get_nodes()` | 有 `rover_rl_policy` |
| 目前 mode | 看 policy log 或假設 launch 的 initial_mode | 提醒首次建議 idle，確認後再切 nav |

## Step 2：看 camera 確認場景（可選但建議）

```
subscribe_once(topic="/camera/camera/color/image_raw",
               msg_type="sensor_msgs/msg/Image", timeout=5, expects_image="true")
analyze_previously_received_image()
```
描述畫面：前方是否淨空、有無人/障礙物。**場景明顯超出 T-corridor 訓練分布（開放廣場、極窄道）就提醒風險。**

## Step 3：preflight 結論 + 提示發 goal

把 Step 1/2 結果給使用者一句話結論（綠燈/紅燈）。綠燈就告訴他：
**可以發 goal 了（RViz Nav2 Goal 或 routing），diag_logger 收到 goal 會自動開始寫
`~/rover_rl/logs/diag_<時間>.csv`。** 紅燈先別發 goal，列出要修的項目。

## Step 4：發 goal 後即時盯場

每隔數秒抽樣盯（用 MCP subscribe_once，不需高頻）：
- `/input/nav_cmd_vel`：`angular.z` 是否在 [-2,2]、是否劇烈跳動（晃動徵兆）
- `/ndt_pose` vs goal：距離是否在縮短
- `/rover_rl/lidar_sweep_72`：最近距離是否進入安全區

**異常立即提醒使用者 E-stop**（cmd 超界 / 劇烈震盪 / 衝向障礙物）。不要說「再跑看看」。

## Step 5：結束與分析

實驗結束（使用者 Ctrl-C 關掉 deploy_rl）時 diag_logger 會自動印晃動/朝向摘要。
然後跑分析：

```bash
ros2 run rover_rl_inference analyze_diag      # 自動取最新 CSV，存 png
```
（若想中途切下一段而不關 deploy_rl，可選擇送 `ros2 topic pub --once /rover_rl/record
std_msgs/String "{data: 'start 下一段'}"` 開新檔；非必要。）
把 `Δω RMS`、`|heading_err|平均`、`距離趨勢`、`heading vs policy_goal 一致性` 解讀給使用者：
- heading 與 policy_goal_ang **不一致** → 定位/TF/座標 bug
- 一致但都偏大 → policy 真的沒朝 goal（場景超訓練分布）
- Δω RMS 大 → 晃動，嫌疑 `cmd_alpha_angular=0.5` / `speed_rate=0.3`

## 注意

- 記錄的可靠來源永遠是 diag_logger 的 CSV，不是你 MCP 抽樣到的零星值。
- 不要改 policy 動作上限 yaml；調濾波/ speed_rate 前先 `cp policy_params.yaml policy_params.yaml.bak`。
