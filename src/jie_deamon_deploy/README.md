# jie_deamon

机器人后端服务节点（已适配二轮差速驱动）。基于 ROS 2 构建，集成激光雷达目标追踪、Web 可视化控制、Android App 通讯等功能。

> 原始版本来自 [6-robot/jie_deamon](https://github.com/6-robot/jie_deamon)（智元机器狗），  
> 已修改为仅使用 `linear.x` + `angular.z` 的差速驱动控制模型。

## 架构概览

```
robot_nexus (中枢节点)
├── lidar_tracker    — 激光雷达目标追踪（含卡尔曼滤波、势场避障）
├── direct_control   — 直接速度控制
├── web_comm         — Web 可视化与 WebSocket 通讯
└── android_comm     — Android App UDP 通讯
```

**ROS 2 话题：**

| 方向 | 话题 | 类型 | 说明 |
|------|------|------|------|
| 订阅 | `/scan` | `sensor_msgs/LaserScan` | 激光雷达扫描数据 |
| 发布 | `/cmd_vel` | `geometry_msgs/Twist` | 底盘速度指令 (linear.x + angular.z) |

**网络端口：**

| 端口 | 协议 | 用途 |
|------|------|------|
| 8080 | HTTP | Web 控制界面 |
| 8890 | WebSocket | 实时数据推送 |
| 8888 | UDP | 雷达数据发送至 App |
| 8889 | UDP | 接收 App 控制指令 |

## 部署

### 环境要求

- ROS 2 (Humble 或更高版本)
- C++17
- OpenCV

### 编译

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## 使用

### 实机启动流程（campusrover）

分三步启动（各自独立终端）：

```bash
# 1. 底盘驱动（搖桿 + mux + 馬達）
cr_start_drive

# 2. 感测器（Velodyne + IMU + YDLidar 前 + TF）
cr_sensor

# 3. jie_deamon 追踪节点
ros2 launch jie_deamon start.launch.py
```

或者开发测试时一步到位（底盘 + 追踪，但感测器仍需 cr_sensor）：

```bash
ros2 launch jie_deamon start.launch.py with_driver:=true
```

**Topic remap（已在 launch 中配置）：**

| 原始 topic | remap 到 | 原因 |
|-----------|---------|------|
| `/scan` | `/scan_front` | 使用前方 YDLidar（cr_sensor 启动）|
| `/cmd_vel` | `/input/nav_cmd_vel` | 经由 lcr_cmd_vel_mux 仲裁,摇杆可抢断 |

**cmd_vel 数据流：**

```
robot_nexus → /input/nav_cmd_vel ─┐
                                   ├→ lcr_cmd_vel_mux → /output/cmd_vel → rover_driver
joy_to_twist → /input/joy_cmd_vel ┘    (摇杆优先)
```

**启动参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `active` | `true` | 激活跟随功能 |
| `enable_web` | `true` | 启用 Web 可视化 |
| `enable_opencv` | `false` | 启用 OpenCV 调试窗口 |
| `with_driver` | `false` | 同时启动底盘驱动 |

示例 — 关闭跟随，仅使用 Web 遥控：

```bash
ros2 launch jie_deamon start.launch.py active:=false
```

### Web 控制界面

启动后访问 `http://<机器人IP>:8080`，支持：

- 跟随/直控模式切换
- 虚拟摇杆直接控制
- 差速机器人速度控制
- 雷达点云实时可视化
- 双击设置跟随目标坐标

### 键盘控制（调试）

```bash
ros2 run jie_deamon keyboard_cmd
```

## 目录结构

```
jie_deamon/
├── src/
│   ├── robot_nexus.cpp      # 中枢节点主程序
│   ├── web_comm.cpp         # Web 通讯实现
│   ├── android_comm.cpp     # Android 通讯实现
│   └── keyboard_cmd.cpp     # 键盘控制节点
├── include/
│   ├── common_types.hpp     # 公共类型与常量
│   ├── lidar_tracker.hpp    # 雷达追踪模块
│   ├── kalman_filter.hpp    # 卡尔曼滤波器
│   ├── direct_control.hpp   # 直接控制模块
│   ├── web_comm.hpp         # Web 通讯头文件
│   ├── web_server.hpp       # HTTP/WebSocket 服务
│   └── android_comm.hpp     # Android 通讯头文件
├── web/                     # 前端静态资源
├── launch/                  # ROS 2 启动文件
├── CMakeLists.txt
└── package.xml
```

## 许可证

MIT
