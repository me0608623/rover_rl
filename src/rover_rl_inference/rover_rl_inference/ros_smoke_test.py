"""ros_smoke_test — 純 ROS 通訊驗證，不需 torch.

用途：在實車上跑此節點代替 policy_node，驗證：
  - 能訂閱 /velodyne_points 與 /odom 並收到資料
  - 能收 /goal_pose
  - 能依 5 Hz 發 /input/nav_cmd_vel（內容是 random tiny twist 給 mux 確認連線）

通過此測試後再換成真正的 policy_node。

執行：
    ros2 run rover_rl_inference ros_smoke_test
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2


class SmokeTest(Node):
    def __init__(self):
        super().__init__("rover_rl_smoke")

        self.declare_parameter("topic_lidar", "/velodyne_points")
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_goal", "/goal_pose")
        self.declare_parameter("topic_cmd_vel", "/input/nav_cmd_vel")
        self.declare_parameter("control_dt", 0.2)
        self.declare_parameter("publish_twist", False)  # 預設只觀察，不發 cmd

        gp = self.get_parameter
        topic_lidar = gp("topic_lidar").get_parameter_value().string_value
        topic_odom = gp("topic_odom").get_parameter_value().string_value
        topic_goal = gp("topic_goal").get_parameter_value().string_value
        topic_cmd_vel = gp("topic_cmd_vel").get_parameter_value().string_value
        self.dt = float(gp("control_dt").value)
        self.publish_twist = bool(gp("publish_twist").value)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.lidar_count = 0
        self.odom_count = 0
        self.goal_count = 0
        self.last_lidar_t = 0.0
        self.last_odom_t = 0.0

        self.create_subscription(PointCloud2, topic_lidar, self._cb_lidar, sensor_qos)
        self.create_subscription(Odometry, topic_odom, self._cb_odom, 10)
        self.create_subscription(PoseStamped, topic_goal, self._cb_goal, 10)
        self.pub_cmd = self.create_publisher(Twist, topic_cmd_vel, 10)

        self.create_timer(self.dt, self._tick)
        self.create_timer(2.0, self._report)
        self.t0 = time.monotonic()

        self.get_logger().info(
            "smoke test 啟動 ===========\n"
            f"  訂閱: {topic_lidar} (PointCloud2)\n"
            f"        {topic_odom}   (Odometry)\n"
            f"        {topic_goal}   (PoseStamped)\n"
            f"  發布: {topic_cmd_vel}  (Twist) {'[啟用]' if self.publish_twist else '[僅觀察]'}\n"
            f"  控制週期: {self.dt}s ({1/self.dt:.1f} Hz)\n"
            "=========================="
        )

    def _cb_lidar(self, msg: PointCloud2) -> None:
        self.lidar_count += 1
        self.last_lidar_t = time.monotonic()
        if self.lidar_count == 1:
            self.get_logger().info(
                f"✓ 首次收到 LiDAR：{msg.width}×{msg.height} pts, "
                f"frame={msg.header.frame_id}, fields={[f.name for f in msg.fields]}"
            )

    def _cb_odom(self, msg: Odometry) -> None:
        self.odom_count += 1
        self.last_odom_t = time.monotonic()
        if self.odom_count == 1:
            p = msg.pose.pose.position
            tw = msg.twist.twist
            self.get_logger().info(
                f"✓ 首次收到 Odom：pos=({p.x:.2f}, {p.y:.2f}), "
                f"v={tw.linear.x:.3f} m/s, ω={tw.angular.z:.3f} rad/s, "
                f"frame={msg.header.frame_id}"
            )

    def _cb_goal(self, msg: PoseStamped) -> None:
        self.goal_count += 1
        p = msg.pose.position
        self.get_logger().info(
            f"✓ 收到 Goal #{self.goal_count}: ({p.x:.2f}, {p.y:.2f}), "
            f"frame={msg.header.frame_id}"
        )

    def _tick(self) -> None:
        if not self.publish_twist:
            return
        # 發極小 random twist 確認 mux 接收（用戶可看 /output/cmd_vel 確認）
        msg = Twist()
        msg.linear.x = float(np.clip(np.random.randn() * 0.02, -0.05, 0.05))
        msg.angular.z = float(np.clip(np.random.randn() * 0.02, -0.05, 0.05))
        self.pub_cmd.publish(msg)

    def _report(self) -> None:
        now = time.monotonic()
        elapsed = now - self.t0
        lidar_hz = self.lidar_count / max(elapsed, 1e-3)
        odom_hz = self.odom_count / max(elapsed, 1e-3)
        lidar_age = now - self.last_lidar_t if self.lidar_count else float("inf")
        odom_age = now - self.last_odom_t if self.odom_count else float("inf")
        self.get_logger().info(
            f"[{elapsed:6.1f}s] LiDAR {self.lidar_count}收 ({lidar_hz:.1f}Hz, last {lidar_age:.2f}s), "
            f"Odom {self.odom_count}收 ({odom_hz:.1f}Hz, last {odom_age:.2f}s), "
            f"Goal {self.goal_count}收"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SmokeTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
