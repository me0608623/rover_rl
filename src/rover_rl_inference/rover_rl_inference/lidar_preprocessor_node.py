"""rover_rl_lidar_preprocessor — 獨立預處理節點（採 spot_rl/spot_obs_process.cpp pattern）.

把原始 VLP-16 PointCloud2 處理成 RL 模型直接吃的 72-bin sweep，發布到 topic 讓
policy_node（或其他 consumer）訂閱。

對齊訓練端 wd_like_sweep_72（IsaacLab：obs_functions.py:276）：
  r_max=20.0, r_robot=0.35, r_min=0.9, z_filter=0.5, num_bins=72
  輸出 normalized to [0, 1]：(d - r_robot) / (r_max - r_robot)

訂閱：
  /velodyne_points          PointCloud2     (sensor_msgs)
  /odom                     Odometry        — 用於 motion compensation（可選）

發布：
  /rover_rl/lidar_sweep_72  Float32MultiArray  [72]      — 給 policy_node 吃
  /rover_rl/lidar_scan      LaserScan          — 給 RViz 視覺化（可選）

執行：
  ros2 run rover_rl_inference lidar_preprocessor

獨立節點的好處（vs inline in policy_node）：
  - 可 `ros2 topic echo /rover_rl/lidar_sweep_72` 直接看處理結果
  - preprocessor 與 RL 各自獨立除錯
  - 多個 policy 可共用同一個 preprocessor
  - LaserScan 可選輸出，直接給 RViz / costmap 用
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from .lidar_preprocess import lidar_sweep_72_real, pointcloud2_to_xyz


class LidarPreprocessorNode(Node):
    """獨立 LiDAR 前處理節點：/velodyne_points → 72-bin sweep + (可選) LaserScan.

    與 policy 解耦的好處見模組 docstring。本節點只負責把點雲壓成 sweep 並發布，
    sweep 公式全交給 lidar_preprocess.lidar_sweep_72_real（與訓練端對齊）。
    """

    def __init__(self):
        super().__init__("rover_rl_lidar_preprocessor")

        # ── 參數宣告 ──
        self.declare_parameter("topic_in", "/velodyne_points")
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_out_sweep", "/rover_rl/lidar_sweep_72")
        self.declare_parameter("topic_out_scan", "/rover_rl/lidar_scan")
        self.declare_parameter("publish_laserscan", True)
        self.declare_parameter("laserscan_frame", "velodyne_link")
        self.declare_parameter("lidar_qos_best_effort", True)
        self.declare_parameter("publish_rate_hz", 10.0)  # 比 VLP-16 同步：10Hz
        # 與訓練端對齊
        self.declare_parameter("num_bins", 72)
        self.declare_parameter("r_max", 20.0)
        self.declare_parameter("r_robot", 0.35)
        self.declare_parameter("r_min", 0.9)
        self.declare_parameter("z_filter", 0.5)
        self.declare_parameter("yaw_offset_deg", 0.0)
        self.declare_parameter("motion_compensation", True)

        gp = self.get_parameter
        self.topic_in = gp("topic_in").get_parameter_value().string_value
        self.topic_odom = gp("topic_odom").get_parameter_value().string_value
        self.topic_out_sweep = gp("topic_out_sweep").get_parameter_value().string_value
        self.topic_out_scan = gp("topic_out_scan").get_parameter_value().string_value
        self.publish_laserscan = bool(gp("publish_laserscan").value)
        self.laserscan_frame = gp("laserscan_frame").get_parameter_value().string_value
        self.qos_be = bool(gp("lidar_qos_best_effort").value)
        self.rate_hz = float(gp("publish_rate_hz").value)
        self.num_bins = int(gp("num_bins").value)
        self.r_max = float(gp("r_max").value)
        self.r_robot = float(gp("r_robot").value)
        self.r_min = float(gp("r_min").value)
        self.z_filter = float(gp("z_filter").value)
        self.yaw_offset = math.radians(float(gp("yaw_offset_deg").value))
        self.motion_comp = bool(gp("motion_compensation").value)

        # ── 狀態 ──
        self._lock = threading.Lock()
        self._latest_pc: np.ndarray | None = None
        self._latest_pc_t = 0.0
        self._odom_v = 0.0
        self._odom_w = 0.0
        self._recv_count = 0
        self._hb_last_n = 0
        self._hb_last_t = time.monotonic()

        # ── QoS / sub / pub ──
        # LiDAR driver 多半用 BEST_EFFORT 發 sensor data；訂閱端 QoS 不相容會收不到，
        # 故預設 BEST_EFFORT 與 VLP-16 driver 對齊（可用 lidar_qos_best_effort 參數切換）
        sensor_qos = QoSProfile(
            reliability=(QoSReliabilityPolicy.BEST_EFFORT if self.qos_be
                         else QoSReliabilityPolicy.RELIABLE),
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(PointCloud2, self.topic_in, self._cb_lidar, sensor_qos)
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)

        self.pub_sweep = self.create_publisher(
            Float32MultiArray, self.topic_out_sweep, 10,
        )
        self.pub_scan = (
            self.create_publisher(LaserScan, self.topic_out_scan, 10)
            if self.publish_laserscan else None
        )

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._tick)
        self.create_timer(5.0, self._heartbeat)

        self.get_logger().info(
            f"rover_rl_lidar_preprocessor 啟動完成\n"
            f"  輸入: {self.topic_in} (QoS={'BE' if self.qos_be else 'REL'})\n"
            f"  輸出: {self.topic_out_sweep} (Float32MultiArray[{self.num_bins}])"
            + (f"\n  + {self.topic_out_scan} (LaserScan, frame={self.laserscan_frame})"
               if self.publish_laserscan else "")
            + f"\n  參數: r_min={self.r_min} r_max={self.r_max} r_robot={self.r_robot} "
              f"z_filter={self.z_filter} bins={self.num_bins} motion_comp={self.motion_comp}"
        )

    def _cb_lidar(self, msg: PointCloud2) -> None:
        # callback 只解點雲存最新一筆（不在這裡算 sweep），實際處理交給 timer 定速跑，
        # 把 sweep 輸出頻率與 LiDAR 到達頻率解耦，避免 callback 阻塞影響收訊
        pts = pointcloud2_to_xyz(msg)
        with self._lock:
            self._latest_pc = pts
            self._latest_pc_t = time.monotonic()
            self._recv_count += 1

    def _cb_odom(self, msg: Odometry) -> None:
        tw = msg.twist.twist
        with self._lock:
            self._odom_v = tw.linear.x
            self._odom_w = tw.angular.z

    def _tick(self) -> None:
        # 定速 timer：取最新點雲算 sweep 並發布。鎖內只快速複製狀態，計算放鎖外
        with self._lock:
            pts = self._latest_pc
            pc_age = time.monotonic() - self._latest_pc_t if pts is not None else float("inf")
            v = self._odom_v
            w = self._odom_w
        # 點雲太舊(>0.5s) 視為 LiDAR 斷訊，不發 stale sweep（讓 consumer 自己 timeout）
        if pts is None or pc_age > 0.5:
            return

        # motion compensation：用 odom 當下速度 × 估計掃描時長，估這段時間機器人位移量，
        # 傳給 sweep 函式扣回畸變。dt_scan clamp 在 0.15s 上限避免點雲過舊時補償過頭
        motion_comp = None
        if self.motion_comp:
            dt_scan = min(pc_age, 0.15)
            motion_comp = (v * dt_scan, 0.0, w * dt_scan)

        sweep = lidar_sweep_72_real(
            pts,
            r_max=self.r_max,
            r_robot=self.r_robot,
            r_min=self.r_min,
            z_filter=self.z_filter,
            num_bins=self.num_bins,
            yaw_offset=self.yaw_offset,
            motion_compensation=motion_comp,
        )

        # 發 Float32MultiArray（給 policy_node）
        sweep_msg = Float32MultiArray()
        sweep_msg.layout.dim.append(MultiArrayDimension(
            label="bin", size=self.num_bins, stride=self.num_bins,
        ))
        sweep_msg.data = sweep.tolist()
        self.pub_sweep.publish(sweep_msg)

        # 發 LaserScan（給 RViz / costmap 視覺化）
        if self.pub_scan is not None:
            scan = LaserScan()
            scan.header.stamp = self.get_clock().now().to_msg()
            scan.header.frame_id = self.laserscan_frame
            scan.angle_min = -math.pi
            scan.angle_max = math.pi - (2 * math.pi / self.num_bins)
            scan.angle_increment = 2 * math.pi / self.num_bins
            scan.scan_time = 1.0 / max(self.rate_hz, 1.0)
            scan.time_increment = 0.0
            scan.range_min = self.r_min
            scan.range_max = self.r_max
            # LaserScan 是給人/RViz 看的，需 metric 距離；把發給 policy 的正規化 sweep
            # 反正規化回公尺：(normalized) × (r_max - r_robot) + r_robot
            denom = self.r_max - self.r_robot
            ranges_m = (sweep * denom + self.r_robot).astype(np.float32)
            scan.ranges = ranges_m.tolist()
            self.pub_scan.publish(scan)

    def _heartbeat(self) -> None:
        now = time.monotonic()
        with self._lock:
            n = self._recv_count
        dt = now - self._hb_last_t
        hz = (n - self._hb_last_n) / dt if dt > 0 else 0.0
        self._hb_last_n, self._hb_last_t = n, now
        mark = "✓" if hz > 1.0 else "⚠"
        self.get_logger().info(
            f"[HB] PointCloud2 {mark}{hz:.1f}Hz (累計 {n})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarPreprocessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("⏹ lidar_preprocessor 已停止")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
