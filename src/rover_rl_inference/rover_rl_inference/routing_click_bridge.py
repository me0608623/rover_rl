"""RViz Publish Point → Routing 橋接節點.

使用方式：
  在 RViz 選 「Publish Point」工具，點擊地圖兩次：
    第一次點擊 → 設定起點（origin）
    第二次點擊 → 設定終點（destination）→ 自動呼叫 routing service

節點會發布視覺化 marker 讓你知道目前選到哪個路由節點。
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from campusrover_msgs.srv import RoutingPath, ModuleInfo


class RoutingClickBridge(Node):
    def __init__(self):
        super().__init__("routing_click_bridge")

        self.declare_parameter("building", "itc")
        self.declare_parameter("floor", "3")

        building = self.get_parameter("building").get_parameter_value().string_value
        floor = self.get_parameter("floor").get_parameter_value().string_value

        self._nodes: dict[str, tuple[float, float]] = {}  # name → (x, y)，路由拓撲節點座標表
        self._origin: str | None = None   # 已選起點（等第二點當終點），構成兩點選取狀態機
        self._click_count = 0             # 點擊計數：奇數=選起點、偶數=選終點並觸發 routing

        # 訂閱 RViz Publish Point
        self.sub = self.create_subscription(
            PointStamped, "/clicked_point", self._on_click, 10)

        # 呼叫 routing service
        self.routing_client = self.create_client(
            RoutingPath, "/routing_to_path/routing_call")

        # 發布選點 marker
        self.marker_pub = self.create_publisher(MarkerArray, "/routing_click/markers", 10)

        # 取得地圖節點座標：routing 用節點名（不是任意座標），故先抓拓撲節點表，
        # 之後把點擊座標 snap 到最近的節點。用 timer 重試直到 service 上線且載入成功。
        self.route_info_client = self.create_client(ModuleInfo, "/get_route_info")
        self.create_timer(2.0, self._fetch_nodes)
        self._nodes_loaded = False
        self._building = building
        self._floor = floor

        self.get_logger().info(
            f"routing_click_bridge 啟動\n"
            f"  第一次 Publish Point → 選起點\n"
            f"  第二次 Publish Point → 選終點 → 自動呼叫 routing"
        )

    def _fetch_nodes(self):
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
            self.get_logger().info(f"載入 {len(self._nodes)} 個路由節點")
        except Exception as e:
            self.get_logger().error(f"get_route_info 失敗: {e}")

    def _nearest_node(self, x: float, y: float) -> str | None:
        # RViz 點擊不會剛好落在拓撲節點上，snap 到歐氏距離最近的節點名供 routing 使用。
        if not self._nodes:
            return None
        return min(self._nodes, key=lambda n: math.hypot(
            self._nodes[n][0] - x, self._nodes[n][1] - y))

    def _on_click(self, msg: PointStamped):
        # 兩點選取狀態機：第 1 次點擊存起點、第 2 次點擊存終點並自動呼叫 routing，
        # 之後 reset 回到等起點狀態，方便連續規劃下一段路徑。
        if not self._nodes_loaded:
            self.get_logger().warn("節點地圖尚未載入，請稍後再試")
            return

        cx, cy = msg.point.x, msg.point.y
        nearest = self._nearest_node(cx, cy)
        nx, ny = self._nodes[nearest]
        dist = math.hypot(nx - cx, ny - cy)

        self._click_count += 1

        if self._click_count % 2 == 1:
            # 奇數次 → 設起點。開始一段新路線：先清掉上一段的起/終點 marker，
            # 否則舊 marker（lifetime 30s）會殘留，與新路線同時顯示成「兩條」。
            self._clear_markers()
            self._origin = nearest
            self.get_logger().info(f"[1/2] 起點選定: {nearest}  (距離點擊 {dist:.2f}m)")
            self._publish_markers(origin=nearest, dest=None)
        else:
            # 偶數次 → 設終點，呼叫 routing
            dest = nearest
            self.get_logger().info(f"[2/2] 終點選定: {dest}  (距離點擊 {dist:.2f}m)")
            self._publish_markers(origin=self._origin, dest=dest)
            self._call_routing(self._origin, dest)
            self._origin = None  # reset

    def _call_routing(self, origin: str, dest: str):
        if not self.routing_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("routing service 不可用")
            return

        req = RoutingPath.Request()
        req.origin = origin
        req.destination = [dest]
        future = self.routing_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f"routing 完成: {origin} → {dest}  path points={sum(len(p.poses) for p in f.result().routing)}"
            ) if f.result() else None
        )

    def _clear_markers(self):
        # 發 DELETEALL 清掉本節點所有選點 marker（起點/終點 sphere + text），
        # 兩個 namespace 各發一顆，確保 RViz 不殘留上一段路線的標記。
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        for ns in ("routing_click", "routing_click_text"):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = ns
            m.action = Marker.DELETEALL
            markers.markers.append(m)
        self.marker_pub.publish(markers)

    def _publish_markers(self, origin: str | None, dest: str | None):
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        mid = 0

        def _sphere(name, color_rgb, scale=0.6):
            nonlocal mid
            if name not in self._nodes:
                return
            x, y = self._nodes[name]
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "routing_click"
            m.id = mid; mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b = color_rgb
            m.color.a = 0.9
            m.lifetime.sec = 30
            markers.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns = "routing_click_text"
            t.id = mid; mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = x
            t.pose.position.y = y
            t.pose.position.z = 0.0
            t.pose.orientation.w = 1.0
            t.scale.z = 0.4
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 1.0
            t.text = name
            t.lifetime.sec = 30
            markers.markers.append(t)

        if origin:
            _sphere(origin, (0.0, 1.0, 0.0))   # 綠色 = 起點
        if dest:
            _sphere(dest, (1.0, 0.2, 0.0))      # 橘紅色 = 終點

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = RoutingClickBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("⏹ routing_click_bridge 已停止")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
