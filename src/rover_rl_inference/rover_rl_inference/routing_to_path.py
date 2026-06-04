"""routing_to_path — campusrover_routing service → /global_path topic 橋接.

campusrover_routing 的 generation_path service 回傳 nav_msgs/Path[]，
但 policy_node 訂閱 /global_path (nav_msgs/Path) topic。
此節點：
  1. 發布 working_floor 讓 routing_engine 載入樓層拓撲
  2. 提供 call_path service 讓外部呼叫（或定時查詢）
  3. 收到 routing response 後 publish 到 /global_path

用法：
  ros2 run rover_rl_inference routing_to_path
  # 或在 launch 中自動啟動

  # 呼叫 service 規劃路徑：
  ros2 service call /rover_rl/routing_call campusrover_msgs/srv/RoutingPath \
    "{origin: 'c1', destination: ['e0']}"

  # 結果會自動發布到 /global_path
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Path
from campusrover_msgs.srv import RoutingPath
from campusrover_msgs.msg import WorkingFloor


class RoutingToPathNode(Node):
    def __init__(self):
        super().__init__("routing_to_path")

        self.declare_parameter("building", "itc")
        self.declare_parameter("floor", "3")
        self.declare_parameter("topic_global_path", "/global_path")
        self.declare_parameter("default_origin", "c1")
        self.declare_parameter("default_destination", ["e0"])

        building = self.get_parameter("building").get_parameter_value().string_value
        floor = self.get_parameter("floor").get_parameter_value().string_value
        topic_path = self.get_parameter("topic_global_path").get_parameter_value().string_value

        # 發布 /global_path。routing 是 service-based 一次性回傳，但 policy_node 的
        # SubgoalSelector 訂閱的是 topic；故把路徑緩存後 2Hz 持續 republish，
        # 確保晚啟動或重連的訂閱者（RViz / policy_node）都拿得到最新路徑。
        self.pub_path = self.create_publisher(Path, topic_path, 5)
        self._last_path: Path | None = None
        # 2Hz 定期重發最後一條路徑，避免 TF 更新後 RViz 失去顯示
        self.create_timer(0.5, self._repub_path)

        # 發布 working_floor 觸發 routing 載入
        self.pub_floor = self.create_publisher(WorkingFloor, "working_floor", 5)

        # ReentrantCallbackGroup：_handle_call 是 service callback，內部又要等
        # generation_path 的 future 完成。預設 callback group 會死鎖（同 group 不能重入），
        # 用 Reentrant + MultiThreadedExecutor 才能在 callback 內阻塞等待下游 service。
        self._cb_group = ReentrantCallbackGroup()
        self.routing_client = self.create_client(
            RoutingPath, "generation_path", callback_group=self._cb_group)

        # 對外 service（讓外部透過此 node 呼叫 routing）
        self.srv = self.create_service(
            RoutingPath, "~/routing_call", self._handle_call,
            callback_group=self._cb_group)

        self._lock = threading.Lock()
        self._floor_timer = self.create_timer(10.0, self._publish_floor)

        self.get_logger().info(
            f"routing_to_path 啟動\n"
            f"  building={building}, floor={floor}\n"
            f"  輸出: {topic_path}\n"
            f"  service: ~/routing_call (RoutingPath)"
        )
        self._building = building
        self._floor = floor

    def _repub_path(self):
        if self._last_path is not None:
            self.pub_path.publish(self._last_path)

    def _publish_floor(self):
        # routing_engine 要先收到 working_floor 才會載入對應樓層的拓撲圖；
        # 發一次就夠（cancel timer），重複發無益。
        msg = WorkingFloor()
        msg.building = self._building
        msg.floor = self._floor
        self.pub_floor.publish(msg)
        self.get_logger().info(f"已發布 working_floor: building={self._building}, floor={self._floor}")
        self._floor_timer.cancel()  # 只發一次

    def _handle_call(self, request, response):
        """外部呼叫 → 轉發給 routing_engine → 結果 publish 到 /global_path."""
        if not self.routing_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("generation_path service 不可用")
            response.routing = []
            return response

        # 用 Event 把 async future 轉成同步等待：service handler 必須回傳 response，
        # 不能 return future；故在此阻塞等下游 routing 完成（靠 Reentrant group 不死鎖）。
        event = threading.Event()
        result_holder: list = []

        def _done(future):
            try:
                result_holder.append(future.result())
            except Exception as e:
                self.get_logger().error(f"routing callback 異常: {e}")
            event.set()

        future = self.routing_client.call_async(request)
        future.add_done_callback(_done)

        if not event.wait(timeout=8.0):
            self.get_logger().warn("routing service 超時")
            response.routing = []
            return response

        if result_holder and result_holder[0].routing:
            # generation_path 回傳 Path[]（可能多段）；policy 只需單一連續路徑，
            # 取第 0 條，存進 _last_path 供 2Hz republish 並立即發一次。
            result = result_holder[0]
            path = result.routing[0]
            self._last_path = path
            self.pub_path.publish(path)
            self.get_logger().info(
                f"路徑已發布到 /global_path: {len(path.poses)} poses, "
                f"origin={request.origin} → dest={request.destination}"
            )
            response.routing = result.routing
        else:
            self.get_logger().warn("routing 回傳空路徑")
            response.routing = []

        return response


def main(args=None):
    rclpy.init(args=args)
    node = RoutingToPathNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("⏹ routing_to_path 已停止")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
