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

import json
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from campusrover_msgs.srv import RoutingPath, ModuleInfo
from campusrover_msgs.msg import WorkingFloor


class RoutingToPathNode(Node):
    def __init__(self):
        super().__init__("routing_to_path")

        self.declare_parameter("building", "itc")
        self.declare_parameter("floor", "3")
        self.declare_parameter("topic_global_path", "/global_path")
        self.declare_parameter("topic_route_stations", "/rover_rl/route_stations")
        self.declare_parameter("default_origin", "c1")
        self.declare_parameter("default_destination", ["e0"])
        # waypoint snap 到具名節點的距離門檻（公尺）；超過視為「不在任一站」
        self.declare_parameter("station_snap_m", 1.5)
        # 到達路徑終點自動清除：避免到點後紅色路徑一直 republish 留在 RViz。
        # 訂閱 policy status，當 goal_src=path_final 且 goal_dist < tol 連續 N 拍 → 清路徑。
        self.declare_parameter("topic_status", "/rover_rl_policy/status")
        self.declare_parameter("auto_clear_on_arrival", True)
        self.declare_parameter("arrival_clear_tol_m", 0.7)
        self.declare_parameter("arrival_clear_ticks", 5)   # status 5Hz × 5 = 1s

        building = self.get_parameter("building").get_parameter_value().string_value
        floor = self.get_parameter("floor").get_parameter_value().string_value
        topic_path = self.get_parameter("topic_global_path").get_parameter_value().string_value
        topic_stations = self.get_parameter(
            "topic_route_stations").get_parameter_value().string_value
        self._snap_m = float(self.get_parameter("station_snap_m").value)

        # 發布 /global_path。routing 是 service-based 一次性回傳，但 policy_node 的
        # SubgoalSelector 訂閱的是 topic；故把路徑緩存後 2Hz 持續 republish，
        # 確保晚啟動或重連的訂閱者（RViz / policy_node）都拿得到最新路徑。
        self.pub_path = self.create_publisher(Path, topic_path, 5)
        self._last_path: Path | None = None

        # 具名站序（地鐵式儀表板用）：把路徑 snap 到拓撲節點 → 有序站名 + 各站 waypoint index。
        # 用 waypoint index 而非座標，讓 status_tui 直接拿 policy 發的 path_i 對應目前在第幾站，
        # 完全避開 node/path frame 不一致的轉換問題。
        self.pub_stations = self.create_publisher(String, topic_stations, 5)
        self._last_stations: dict | None = None
        # 路由拓撲節點座標表（name → (x,y)），由 /get_route_info 載入，snap 用
        self._nodes: dict[str, tuple[float, float]] = {}
        self._nodes_loaded = False
        self.route_info_client = self.create_client(ModuleInfo, "/get_route_info")
        self.create_timer(2.0, self._fetch_nodes)

        # 2Hz 定期重發最後一條路徑 + 站序，避免晚啟動的訂閱者（RViz / TUI）漏接
        self.create_timer(0.5, self._repub_path)

        # 監聽手動 2D Goal Pose：使用者手點單一目標 = 放棄目前 routing 路徑。
        # 收到就清掉緩存並停止 2Hz 重發，否則重發的舊 path 會持續蓋過手點目標
        # （policy_node 端 path 優先於 single goal）。下次明確再呼叫 routing 才重新發。
        topic_goal = self.declare_parameter(
            "topic_goal_pose", "/goal_pose").get_parameter_value().string_value
        self.create_subscription(PoseStamped, topic_goal, self._on_manual_goal, 5)

        # 到達終點自動清除：訂閱 policy status 偵測「走到 path 終點」→ 清路徑 + 清 RViz 紅線。
        self._auto_clear = bool(self.get_parameter("auto_clear_on_arrival").value)
        self._arrival_tol = float(self.get_parameter("arrival_clear_tol_m").value)
        self._arrival_ticks = int(self.get_parameter("arrival_clear_ticks").value)
        self._arrival_count = 0
        if self._auto_clear:
            topic_status = self.get_parameter(
                "topic_status").get_parameter_value().string_value
            self.create_subscription(String, topic_status, self._on_status, 5)

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
        if self._last_stations is not None:
            msg = String()
            msg.data = json.dumps(self._last_stations)
            self.pub_stations.publish(msg)

    def _on_manual_goal(self, _msg: PoseStamped) -> None:
        # 手動 2D Goal Pose → 放棄 routing 路徑：清緩存、停止 2Hz 重發。
        if self._last_path is None and self._last_stations is None:
            return
        self._clear_path("收到手動 goal_pose → 停止 republish routing 路徑（手點目標接管）",
                         publish_empty=False)

    def _on_status(self, msg: String) -> None:
        # 到達 path 終點自動清除：policy 走到 path_final 且 goal_dist < tol 連續 N 拍 → 清路徑。
        # 用 path_final（非 lookahead carrot）+ 距離雙條件，確保是真的到終點才清，不是中途。
        if self._last_path is None:
            self._arrival_count = 0
            return
        try:
            st = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        src = st.get("goal_src") or ""
        dist = st.get("goal_dist")
        if str(src).startswith("path_final") and dist is not None and dist < self._arrival_tol:
            self._arrival_count += 1
            if self._arrival_count >= self._arrival_ticks:
                self._clear_path(
                    f"到達 path 終點（goal_dist={dist:.2f}m < {self._arrival_tol}m"
                    f" 連續 {self._arrival_count} 拍）→ 清除 routing 路徑 + RViz 紅線",
                    publish_empty=True)
        else:
            self._arrival_count = 0

    def _clear_path(self, reason: str, publish_empty: bool) -> None:
        """清掉緩存路徑/站序並停止 2Hz republish（手動 goal 接管 或 到達終點）。
        publish_empty=True 時補發一條空 Path，主動清掉 RViz latched 的紅色路徑顯示。"""
        frame = self._last_path.header.frame_id if self._last_path is not None else "map"
        with self._lock:
            self._last_path = None
            self._last_stations = None
            self._arrival_count = 0
        if publish_empty:
            empty = Path()
            empty.header.frame_id = frame
            empty.header.stamp = self.get_clock().now().to_msg()
            self.pub_path.publish(empty)
            # 站序也清空，TUI 退回無路線狀態
            self.pub_stations.publish(String(data=json.dumps(
                {"stations": [], "wp_idx": [], "n_wp": 0})))
        self.get_logger().info(reason)

    def _fetch_nodes(self):
        # routing 用具名節點（c1/e0…），先抓拓撲節點表把路徑 snap 成站序。
        # timer 重試直到 /get_route_info 上線且載入成功。
        if self._nodes_loaded:
            return
        if not self.route_info_client.wait_for_service(timeout_sec=0.1):
            return
        req = ModuleInfo.Request()
        req.building = self._building
        req.floor = self._floor
        future = self.route_info_client.call_async(req)
        future.add_done_callback(self._on_route_info)

    def _on_route_info(self, future):
        try:
            resp = future.result()
            for node in resp.node:
                self._nodes[node.name] = (node.pose.position.x, node.pose.position.y)
            self._nodes_loaded = True
            self.get_logger().info(f"載入 {len(self._nodes)} 個路由節點（站序用）")
            # 節點表晚於路徑載入時，補算一次站序
            if self._last_path is not None:
                self._last_stations = self._derive_stations(self._last_path)
        except Exception as e:
            self.get_logger().error(f"get_route_info 失敗: {e}")

    def _derive_stations(self, path: Path) -> dict:
        """把路徑 waypoint snap 到最近具名節點 → 有序站名 + 各站起始 waypoint index.

        作法：逐 waypoint 找最近節點，距離 <= snap 門檻才算「在該站」；連續同名只記
        第一個 index（去重相鄰重複）。回傳 {stations, wp_idx, n_wp}。
        n_wp 讓 TUI 知道總點數，wp_idx 對應 policy 的 path_i 求目前在第幾站。
        """
        n_wp = len(path.poses)
        if not self._nodes or n_wp == 0:
            return {"stations": [], "wp_idx": [], "n_wp": n_wp}
        seq: list[tuple[int, str]] = []
        for i, ps in enumerate(path.poses):
            name, d = self._nearest_node(ps.pose.position.x, ps.pose.position.y)
            if name is None or d > self._snap_m:
                continue
            if seq and seq[-1][1] == name:
                continue  # 去重相鄰同名
            seq.append((i, name))
        return {"stations": [n for _, n in seq],
                "wp_idx": [i for i, _ in seq], "n_wp": n_wp}

    def _nearest_node(self, x: float, y: float) -> tuple[str | None, float]:
        if not self._nodes:
            return None, float("inf")
        name = min(self._nodes, key=lambda n: math.hypot(
            self._nodes[n][0] - x, self._nodes[n][1] - y))
        nx, ny = self._nodes[name]
        return name, math.hypot(nx - x, ny - y)

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
            self._arrival_count = 0      # 新路徑 → 重置到達計數，避免沿用舊段殘值
            self.pub_path.publish(path)
            # 推導具名站序並發布（節點表未載入時 stations 為空，TUI 自動退回幾何進度條）
            self._last_stations = self._derive_stations(path)
            self.pub_stations.publish(
                String(data=json.dumps(self._last_stations)))
            self.get_logger().info(
                f"路徑已發布到 /global_path: {len(path.poses)} poses, "
                f"origin={request.origin} → dest={request.destination}, "
                f"站序={self._last_stations['stations']}"
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
