# jie_deamon → Humble 機器部署交接文件

> 本文件由 Jazzy 開發機 (aa@192.168.x.x) 的 Claude 產出
> 目標機器: humble@192.168.3.13

---

## 背景

jie_deamon 是從 https://github.com/6-robot/jie_deamon 下載的機器狗後端服務節點，
已在 Jazzy 開發機上完成以下改造：

1. **差速驅動適配** — 所有 `linear.y` 歸零，橫向排斥力轉為 `angular.z`
2. **移除機械狗動作** — `/d1_cmd` publisher 已移除，趴下/站起按鈕已刪
3. **機器人框架尺寸** — 根據 campusrover_chgh.xacro URDF 設定排除區域
4. **Topic remap** — `/scan` → `/scan_front`, `/cmd_vel` → `/input/nav_cmd_vel`
5. **Launch 整合** — start.launch.py 已配置好 campusrover 的底盤和 mux 對接
6. **全檔案中文註解** — 19 個檔案全部加入繁體中文行內註解 + Humble 相容性提示

## 你需要做的事

### Step 1: 複製 jie_deamon 到 Humble 機器

在 Jazzy 開發機上執行：
```bash
cd ~/rover2_ws/src
rsync -avz --exclude='.git' jie_deamon/ humble@192.168.3.13:~/test_campusrover_base/src/jie_deamon/
```

或者在 Humble 機器上 git clone 後覆蓋：
```bash
cd ~/test_campusrover_base/src
# 如果已有舊版 jie_deamon，先備份
mv jie_deamon jie_deamon.bak
# 從開發機複製
scp -r aa@<JAZZY_IP>:~/rover2_ws/src/jie_deamon .
```

### Step 2: 確認依賴

jie_deamon 需要這些 ROS 2 套件（Humble 上應該都有）：
- `rclcpp`
- `sensor_msgs`
- `geometry_msgs`
- `std_msgs`
- `OpenCV` (libopencv-dev)

另外需要 `cpp-httplib`（已包含在 include/httplib.h 中，不需額外安裝）。

### Step 3: 編譯

```bash
cd ~/test_campusrover_base
source /opt/ros/humble/setup.bash
colcon build --packages-select jie_deamon
source install/setup.bash
```

#### 可能的 Humble 編譯問題

1. **C++17 支持** — CMakeLists.txt 已設定 `CMAKE_CXX_STANDARD 17`，Humble 的 GCC 應該支持
2. **OpenCV 版本** — Humble 預設 OpenCV 4.x，API 與 Jazzy 相同
3. **httplib.h** — 如果出現 OpenSSL 相關錯誤，檢查是否有 `CPPHTTPLIB_NO_EXCEPTIONS` 定義（已在 web_comm.hpp 中定義）

### Step 4: 確認 YDLidar 前光達已啟用

你已經在 `start_sensors_launch.py` 中取消了 `yd_front` 的註解：
```python
# 確認這行不是被註解的
yd_front,   # ← 需要啟動，產生 /scan_front topic
```

### Step 5: 啟動測試

三個終端：

```bash
# 終端 1: 底盤驅動
cr_start_drive

# 終端 2: 感測器
cr_sensor

# 終端 3: jie_deamon
source ~/test_campusrover_base/install/setup.bash
ros2 launch jie_deamon start.launch.py
```

### Step 6: 驗證

1. **確認 topic 存在**：
```bash
ros2 topic list | grep -E "scan_front|nav_cmd_vel|output/cmd_vel"
# 應該看到：
# /scan_front
# /input/nav_cmd_vel
# /output/cmd_vel
```

2. **確認 robot_nexus 有訂閱到 scan_front**：
```bash
ros2 topic info /scan_front
# 應該看到 robot_nexus 在 Subscriptions 裡
```

3. **開瀏覽器**：
```
http://192.168.3.13:8080
```
應該看到 Web 控制介面，雷達跟隨 tab 有即時點雲。

4. **搖桿安全測試**：
   - 在 Web 介面切到「直接控制」tab，推搖桿看車有沒有動
   - 然後切到「雷達跟隨」tab，按「開始運動」
   - 隨時推實體搖桿，應該能搶斷追蹤指令

---

## 可能需要在 Humble 上調整的項目

### 1. 光達方向正負號

如果 Web 畫面上點雲前後或左右反了，需要改 `lidar_tracker.hpp:148-149`：

```cpp
// 目前是取負號（原始 D1 機械狗的光達安裝方向）
double point_x = -range * cos(angle);
double point_y = -range * sin(angle);

// 如果點雲前後反了，改為不取負號：
double point_x = range * cos(angle);
double point_y = range * sin(angle);

// 如果只有左右反了：
double point_x = -range * cos(angle);
double point_y = range * sin(angle);   // 去掉 y 的負號
```

### 2. 機器人框架排除區域

如果 Web 畫面上看到車身結構的點沒被過濾掉，調整 `common_types.hpp`：
```cpp
constexpr double ROBOT_FRAME_FRONT = 0.10;  // 加大 → 過濾更多前方點
constexpr double ROBOT_FRAME_BACK = 0.55;   // 加大 → 過濾更多後方點
constexpr double ROBOT_FRAME_LEFT = 0.20;   // 加大 → 過濾更多左側點
constexpr double ROBOT_FRAME_RIGHT = 0.30;  // 加大 → 過濾更多右側點
```

### 3. QoS 不匹配

Humble 的 YDLidar 驅動可能用 `BEST_EFFORT` QoS，但 robot_nexus 目前用 depth=1（預設 `RELIABLE`）。如果 `/scan_front` 有資料但 robot_nexus 收不到，需要改 `robot_nexus.cpp` 的訂閱 QoS：

```cpp
// 原始（可能跟 BEST_EFFORT 的光達不匹配）
scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", 1, ...);

// 改為明確 BEST_EFFORT
auto qos = rclcpp::QoS(1).best_effort();
scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", qos, ...);
```

### 4. 追蹤參數調整

在 `common_types.hpp` 中：
```cpp
FOLLOW_DIST = 0.4         // 跟隨距離(m)，太近可能撞人，太遠容易丟失
TARGET_RADIUS = 0.3       // 搜尋半徑(m)，太小容易丟失，太大可能鎖到牆壁
MAX_LINEAR_SPEED = 1.0    // 最大線速度，第一次測試建議降到 0.3
MAX_ANGULAR_SPEED = 1.0   // 最大角速度，第一次測試建議降到 0.5
```

---

## 架構總覽

```
                    ┌─────────────────────────────────────┐
cr_sensor           │  YDLidar 前                          │
                    │  /scan_front (LaserScan)             │
                    └────────────┬────────────────────────┘
                                 │ remap
                                 ▼
                    ┌─────────────────────────────────────┐
jie_deamon          │  robot_nexus                         │
start.launch.py     │  ├── lidar_tracker (追蹤+APF避障)    │
                    │  ├── direct_control (Web遙控)        │
                    │  ├── web_comm (HTTP:8080 + WS:8890)  │
                    │  └── android_comm (UDP:8888/8889)    │
                    └────────────┬────────────────────────┘
                                 │ remap
                                 ▼
                    ┌─────────────────────────────────────┐
cr_start_drive      │  /input/nav_cmd_vel                  │
                    │       ↓                              │
                    │  lcr_cmd_vel_mux (搖桿優先)           │
                    │       ↓                              │
                    │  /output/cmd_vel                     │
                    │       ↓                              │
                    │  rover_driver (馬達)                  │
                    └─────────────────────────────────────┘
```

## 修改過的檔案清單

| 檔案 | 修改類型 |
|------|---------|
| `include/common_types.hpp` | LINEAR_Y_SCALE_FACTOR→0, ROBOT_FRAME_* 改為 rover2 尺寸 |
| `include/lidar_tracker.hpp` | linear.y=0, repulse_y→angular.z, 差速適配 |
| `include/direct_control.hpp` | linear.y=0, setVelocity 傳 0 |
| `src/robot_nexus.cpp` | 移除 /d1_cmd, action_cmd 改為 WARN 忽略 |
| `src/keyboard_cmd.cpp` | A/D 改為旋轉, 移除 linear.y 顯示 |
| `web/index.html` | 移除趴下/站起按鈕 |
| `web/app.js` | joystick vy=0, 移除 action handler |
| `launch/start.launch.py` | remap scan_front + nav_cmd_vel, 可選 with_driver |
| `package.xml` | 描述改為差速機器人 |
| `README.md` | 更新啟動流程和架構說明 |
| 全部 19 個檔案 | 加入繁體中文行內註解 + Humble 相容性提示 |
