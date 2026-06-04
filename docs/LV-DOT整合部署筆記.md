# LV-DOT 動態障礙物偵測 — 整合部署筆記

> 日期：2026-06-04
> 機器：CampusRover 實車（rover_rl workspace）
> 目的：`deploy_rl` 啟動完整棧時一併啟動 LV-DOT，並能在 RViz 訂閱其話題

---

## 1. 本次更改總覽

| 檔案 | 更改內容 |
|---|---|
| `src/rover_rl_bringup/launch/deploy_full.launch.py` | 新增 LV-DOT detector 節點（Part 10）+ 兩個 launch arg + banner 一行 |
| `/home/aa/rviz/demo.rviz` | 新增 6 個 LV-DOT display（MarkerArray ×5 + PointCloud2 ×1） |

> `deploy_rl` alias = `ros2 launch rover_rl_bringup deploy_full.launch.py`，
> 故改 deploy_full 後，`deploy_rl` 自動帶起 LV-DOT。
> install→build→src 全是 symlink（`--symlink-install`），改 src 直接生效，**免重 build**。

### 1.1 deploy_full.launch.py 新增節點

```python
# ── Part 10: LV-DOT 動態障礙物偵測（LiDAR+depth 融合，發 map frame markers）──
lvdot_pkg = get_package_share_directory("onboard_detector")
lvdot_params = os.path.join(lvdot_pkg, "cfg", "detector_param.yaml")
lvdot_detector_node = Node(
    package="onboard_detector",
    executable="detector_node",
    name="dynamic_detector",
    output="screen",
    parameters=[lvdot_params],
    condition=IfCondition(LaunchConfiguration("enable_lvdot")),
)
lvdot_yolo_node = Node(
    package="onboard_detector",
    executable="yolov11_detector_node.py",
    name="yolov11_detector_node",
    output="screen",
    condition=IfCondition(LaunchConfiguration("enable_lvdot_yolo")),
)
```

### 1.2 新增 launch 參數

| arg | 預設 | 說明 |
|---|---|---|
| `enable_lvdot` | `true` | LV-DOT 動態障礙物偵測（→ `/onboard_detector/*`） |
| `enable_lvdot_yolo` | **`false`** | LV-DOT YOLOv11 視覺輔助 |

> **為何 yolo 預設關**：這台 Jetson **未安裝 `ultralytics`**，開 yolo 節點會 crash。
> LiDAR + depth 融合不靠 yolo 也能正常跑。日後裝了 ultralytics 再加
> `enable_lvdot_yolo:=true`。

### 1.3 RViz demo.rviz 新增的 display

| Display 名稱 | Topic | 型別 | 預設開關 |
|---|---|---|---|
| LVDOT_dynamic_bboxes | `/onboard_detector/dynamic_bboxes` | MarkerArray | 開 |
| LVDOT_tracked_bboxes | `/onboard_detector/tracked_bboxes` | MarkerArray | 開 |
| LVDOT_velocity | `/onboard_detector/velocity_visualizaton` | MarkerArray | 開 |
| LVDOT_dynamic_cloud（紅點）| `/onboard_detector/dynamic_point_cloud` | PointCloud2 | 開 |
| LVDOT_history_traj | `/onboard_detector/history_trajectories` | MarkerArray | 關 |
| LVDOT_lidar_bboxes | `/onboard_detector/lidar_bboxes` | MarkerArray | 關 |

> 全部發在 `map` frame，與 demo.rviz Fixed Frame 一致，開 RViz 即訂閱到，不用手動 Add。

---

## 2. LV-DOT 架構

LV-DOT = **L**iDAR-**V**isual **D**ynamic **O**bject **T**racking。
LiDAR 點雲 + RGB-D 影像（+ 可選 YOLO）融合 → 偵測並追蹤動態障礙物 → 發 3D bbox + 速度。

```
                  ┌─────────────────────────────────────────────┐
 /velodyne_points │  onboard_detector / detector_node (C++)      │
 ───────────────► │   (name: dynamic_detector)                   │
                  │                                              │
 /camera/.../     │  1. LiDAR DBSCAN 聚類 → 3D cluster bbox      │
   depth_image_   │  2. Depth 影像 U-depth/UV 偵測（可選）       │
   rect_raw  ───► │  3. (可選) YOLOv11 視覺框輔助過濾            │
                  │  4. IOU 融合 LiDAR ↔ visual bbox            │
 /ndt_pose   ───► │  5. Kalman 追蹤 + 資料關聯（history=100）    │
   (定位 pose)    │  6. 速度估計 → 動態分類（投票）             │
 /odom  ───────► │                                              │
                  └──────────────┬──────────────────────────────┘
                                 │ 全部輸出 map frame
                                 ▼
              static cluster ──► lidar_bboxes（灰框）
              tracked       ──► tracked_bboxes
              moving (>0.2m/s, 投票通過) ──► dynamic_bboxes（紅框）
                                          + dynamic_point_cloud
                                          + velocity_visualizaton（速度箭頭）
                                          + history_trajectories（軌跡）
```

### 2.1 訂閱（輸入）topic — 來自 `detector_param.yaml`

| 參數 | Topic | 說明 |
|---|---|---|
| `depth_image_topic` | `/camera/camera/depth/image_rect_raw` | **需 RealSense `enable_depth:=true` 才有**（目前預設關） |
| `color_image_topic` | `/camera/camera/color/image_raw` | RealSense RGB（640×480） |
| `lidar_pointcloud_topic` | `/velodyne_points` | VLP-16 主要輸入 |
| `pose_topic` | `/ndt_pose` | NDT 定位（`localization_mode: 0` = pose） |
| `odom_topic` | `/odom` | 底盤里程（localization_mode 1 才用） |

### 2.2 發布（輸出）topic — 全部 `/onboard_detector/` 前綴

| Topic | 型別 | 用途 |
|---|---|---|
| `dynamic_bboxes` | MarkerArray | **動態障礙物 3D 框（紅）— 最重要** |
| `tracked_bboxes` | MarkerArray | 追蹤中物體框 |
| `lidar_bboxes` | MarkerArray | LiDAR 聚類框（含靜態，灰） |
| `dbscan_bboxes` | MarkerArray | DBSCAN 原始聚類框 |
| `visual_bboxes` | MarkerArray | 視覺（depth/yolo）框 |
| `uv_bboxes` | MarkerArray | U-depth map 偵測框 |
| `filtered_bboxes` | MarkerArray | 融合過濾後框 |
| `filtered_before_yolo_bboxes` | MarkerArray | yolo 前的過濾框 |
| `velocity_visualizaton` | MarkerArray | 速度向量箭頭 |
| `history_trajectories` | MarkerArray | 物體歷史軌跡 |
| `dynamic_point_cloud` | PointCloud2 | 動態點雲（屬於動態物的點） |
| `raw_dynamic_point_cloud` | PointCloud2 | 原始動態點雲 |
| `lidar_clusters` | PointCloud2 | LiDAR 聚類著色點雲 |
| `filtered_point_cloud` | PointCloud2 | 過濾後點雲 |
| `filtered_depth_cloud` | PointCloud2 | depth 轉點雲 |
| `downsampled_point_cloud` | PointCloud2 | 降採樣點雲 |
| `raw_lidar_point_cloud` | PointCloud2 | 原始 LiDAR 點雲 |
| `detected_color_image` | Image | YOLO/偵測疊框彩圖 |
| `detected_depth_map` | Image | 偵測 depth map |
| `detected_u_depth_map` | Image | U-depth map |
| `u_depth_bird_view` | Image | U-depth 鳥瞰 |

### 2.3 關鍵參數（detector_param.yaml）

| 類別 | 參數 | 值 |
|---|---|---|
| 定位 | `localization_mode` | 0（pose） |
| LiDAR DBSCAN | `lidar_DBSCAN_min_points` / `lidar_DBSCAN_epsilon` | 10 / 0.05 |
| 降採樣 | `downsample_threshold` / `gaussian_downsample_rate` | 3500 / 6 |
| 融合 | `filtering_BBox_IOU_threshold` | 0.2 |
| 追蹤 | `history_size` / `kalman_filter_averaging_frames` | 100 / 10 |
| **動態分類** | `dynamic_velocity_threshold` | **0.2 m/s** |
| 動態分類 | `dynamic_voting_threshold` | 0.8 |
| 動態分類 | `frames_force_dynamic` / `dynamic_consistency_threshold` | 10 / 15 |
| 尺寸限制 | `target_object_size` / `max_object_size` | [0.5,0.5,1.5] / [3,3,2] |
| 高度過濾 | `ground_height` / `roof_height` | 0.2 / 2.0 |

### 2.4 感測器外參（body↔sensor，row-major 4×4）

- `body_to_lidar`：z 偏移 +0.15 m
- `body_to_camera_depth` / `body_to_camera_color`：x +0.09, z +0.095，相機光軸朝前

---

## 3. ROS MCP 驗證結果（2026-06-04，deploy_rl 實跑）

| 檢查項 | 結果 |
|---|---|
| rosbridge / MCP 連線 | ✅ port 9090，Fully_accessible |
| `/onboard_detector/*` 話題 | ✅ 21 個全部存在 |
| `/ndt_pose`（定位依賴） | ✅ 已收斂正常發 pose |
| `/velodyne_points` | ✅ 有在發 |
| `/onboard_detector/lidar_bboxes` | ✅ **16 個 bbox**（map frame）→ 聚類偵測在跑 |
| `/onboard_detector/dynamic_bboxes` | ⚪ 空 `markers:[]` |

**結論**：`dynamic_bboxes` 空是**正常**，不是故障。
- `lidar_bboxes` 抓到 16 cluster → detector 有在處理點雲。
- 這 16 個目前全被分類為**靜態**（牆/柱/固定物）。
- 場景沒有東西真的在動（速度 < 0.2 m/s 門檻），所以沒有動態框。
- 要看到紅框：找人在車前走動幾秒（>0.2 m/s），投票通過後 `dynamic_bboxes` /
  `dynamic_point_cloud` 會即時跳出，RViz 同步顯示。

---

## 4. 使用方式

```bash
# 一鍵完整棧（含 LV-DOT，yolo 預設關）
deploy_rl

# 臨時關掉 LV-DOT
ros2 launch rover_rl_bringup deploy_full.launch.py enable_lvdot:=false

# 日後裝了 ultralytics 才開 yolo
ros2 launch rover_rl_bringup deploy_full.launch.py enable_lvdot_yolo:=true

# 單獨啟動 detector（除錯用）
ros2 run onboard_detector detector_node --ros-args \
  --params-file ~/rover_rl/install/onboard_detector/share/onboard_detector/cfg/detector_param.yaml

# 透過 ROS MCP 看動態框（需先起 rosbridge）
#   subscribe_once(topic="/onboard_detector/dynamic_bboxes",
#                  msg_type="visualization_msgs/msg/MarkerArray")
```

---

## 5. 注意事項 / 待辦

1. **depth 分支未啟用**：RealSense 開機是 `enable_depth:=false`，
   detector 目前只跑 **LiDAR 分支**。要 depth+LiDAR 融合須重啟相機帶
   `enable_depth:=true`。
2. **ultralytics 未安裝**：yolo 視覺輔助關閉中。LiDAR-only 偵測已可運作。
3. **隔離測試孤兒進程**：單獨 `ros2 run detector_node` 後若用 `kill` 沒清乾淨，
   會殘留一隻與 launch 正式節點重複發話題 → 先 `pgrep -af detector_node` 確認只剩一隻。
4. **rosbridge**：MCP 操作前須先起（`/tmp/rosbridge.log`），且要 source
   `~/rover_rl/setup_env.sh`（DOMAIN=55, zenoh）才看得到 rover_rl topics。
5. **frame 一致性**：detector 全部輸出 `map` frame，依賴 NDT 收斂。NDT 沒收斂時
   bbox 位置會飄。
