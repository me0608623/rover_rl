"""pingpong_test_node — 兩固定點往返避障測試（連續來回，避障效果評估用）.

用途：把車手動開到 A 點（預設 c24）或 B 點（預設 c27），車偵測到「已停在其中一點」就
就緒等候；**每一段都要按空白鍵才出發**——按下後規劃到「對向點」交給 RL policy 走過去，
到了對向點**自動停車回就緒**、再按空白鍵走回來，如此 A↔B 一段一段確認來回，方便在
兩段之間重新擺放障礙物、反覆測試避障表現。

啟動條件（對應安全需求）：
  • **只有車確實停在 A 或 B 其中一點**（距離 < arrive_tol、低速、連續 dwell 拍）才會「就緒」。
    這兩道硬檢查 + 空白鍵閘門即為安全保證：RL 若在開，速度/距離一定不過關。
  • mode 只擋 estop（estop = 你明確要它停，不就緒）；nav / manual / idle 皆可就緒——
    預設 nav 把車開到點上停穩即就緒，不必先切 idle。
  • 就緒後 **TUI 會跳出提示，按【空白鍵】才真正開始**（按空白 → 本節點切 mode=nav
    並呼叫 routing 規劃到對向點）。空白鍵觸發走 `/rover_rl/pingpong/start` topic，
    由 status_tui 偵測到就緒狀態時發布。
  • 本節點 5Hz 發布自身狀態到 `/rover_rl/pingpong/status`（JSON），供 TUI 顯示提示。

中斷與重啟（對應「中途我停了，手動開回去其中一點才重新運作」）：
  • 來回過程中 mode 一旦離開 nav（你抓搖桿→manual、或按 estop / 切 idle）即視為中斷，
    回到「待命」狀態，停止下指令。
  • 要重啟：手動把車開回 A 或 B 任一點停好（manual/idle + 低速 dwell）→ 自動開始新一輪。

⚠ 此測試靠 map frame 車姿（routing 拓撲節點座標在 map）判定到點，**需要 NDT 在跑**
   （pose_src=tf）。沒有 NDT 時 pose 退化為 odom，到點判定會失準——本節點偵測到會警告。

只在使用者於 deploy_rl_shell / deploy_rl 互動詢問選「啟用往返測試」時，由 launch 帶
enable_pingpong:=true 啟動；預設不啟。純測試輔助，與 RL 推論/避障邏輯完全解耦。
"""
from __future__ import annotations

import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
from campusrover_msgs.srv import RoutingPath, ModuleInfo


class PingpongTestNode(Node):
    def __init__(self):
        super().__init__("pingpong_test")

        # 兩個往返點（routing 拓撲節點名）
        self.declare_parameter("point_a", "c24")
        self.declare_parameter("point_b", "c27")
        self.declare_parameter("building", "itc")
        self.declare_parameter("floor", "3")
        # 到點判定：距離門檻（公尺）、低速門檻（m/s）、啟動 dwell 拍數、到達 dwell 拍數
        self.declare_parameter("arrive_tol_m", 0.8)
        self.declare_parameter("start_max_speed", 0.15)
        self.declare_parameter("start_dwell_ticks", 4)    # 2Hz × 4 = 2s 停穩才算「停好在點上」
        self.declare_parameter("arrive_dwell_ticks", 2)    # 2Hz × 2 = 1s 到對向點才折返
        # 啟動時是否自動把 mode 切成 nav（讓 RL 接手）。關掉則只規劃路徑、需你自己切 nav。
        self.declare_parameter("auto_set_nav", True)
        # 全自動模式：就緒後不必按空白鍵，停穩後自動出發下一段（可執行中用 TUI 'a' 鍵熱切換）。
        self.declare_parameter("auto_continue", False)      # 初始是否啟用全自動
        self.declare_parameter("auto_pause_ticks", 4)       # 2Hz × 4 = 2s 就緒緩衝再自動發（給中斷空間）
        # 介接 topic / service
        self.declare_parameter("topic_status", "/rover_rl_policy/status")
        self.declare_parameter("topic_mode", "/rover_rl_policy/mode")
        self.declare_parameter("routing_service", "/routing_to_path/routing_call")
        self.declare_parameter("status_stale_s", 2.0)     # status 超過此秒數視為失聯
        # 本節點對外狀態 / 開始觸發（TUI 顯示提示 + 空白鍵發 start）
        self.declare_parameter("topic_pp_status", "/rover_rl/pingpong/status")
        self.declare_parameter("topic_pp_start", "/rover_rl/pingpong/start")
        self.declare_parameter("topic_pp_auto", "/rover_rl/pingpong/auto_toggle")
        self.declare_parameter("topic_pp_stop", "/rover_rl/pingpong/stop")

        self._pa = self.get_parameter("point_a").get_parameter_value().string_value
        self._pb = self.get_parameter("point_b").get_parameter_value().string_value
        self._building = self.get_parameter("building").get_parameter_value().string_value
        self._floor = self.get_parameter("floor").get_parameter_value().string_value
        self._arrive_tol = float(self.get_parameter("arrive_tol_m").value)
        self._start_max_speed = float(self.get_parameter("start_max_speed").value)
        self._start_dwell = int(self.get_parameter("start_dwell_ticks").value)
        self._arrive_dwell = int(self.get_parameter("arrive_dwell_ticks").value)
        self._auto_set_nav = bool(self.get_parameter("auto_set_nav").value)
        self._auto_continue = bool(self.get_parameter("auto_continue").value)
        self._auto_pause_ticks = int(self.get_parameter("auto_pause_ticks").value)
        self._stale_s = float(self.get_parameter("status_stale_s").value)

        # ── 路由拓撲節點座標表（name → (x,y)），由 /get_route_info 載入 ──
        self._nodes: dict[str, tuple[float, float]] = {}
        self._nodes_loaded = False
        self.route_info_client = self.create_client(ModuleInfo, "/get_route_info")
        self.create_timer(2.0, self._fetch_nodes)

        # ── routing 呼叫 client（→ routing_to_path 橋接 → /global_path）──
        self.routing_client = self.create_client(
            RoutingPath, self.get_parameter("routing_service")
            .get_parameter_value().string_value)
        self._routing_inflight = False

        # ── mode 發布（啟動時切 nav）──
        self.pub_mode = self.create_publisher(
            String, self.get_parameter("topic_mode")
            .get_parameter_value().string_value, 5)

        # ── 訂閱 policy status（拿車姿 / mode / nav_type）──
        self._status: dict | None = None
        self._status_t = 0.0
        self.create_subscription(
            String, self.get_parameter("topic_status")
            .get_parameter_value().string_value, self._on_status, 5)

        # ── 對外狀態發布 + 開始觸發訂閱（TUI 提示 / 空白鍵開始）──
        self.pub_pp = self.create_publisher(
            String, self.get_parameter("topic_pp_status")
            .get_parameter_value().string_value, 5)
        self._start_requested = False
        self.create_subscription(
            Empty, self.get_parameter("topic_pp_start")
            .get_parameter_value().string_value, self._on_start_trigger, 5)
        # 全自動模式熱切換（TUI 'a' 鍵發 Empty → flip）
        self.create_subscription(
            Empty, self.get_parameter("topic_pp_auto")
            .get_parameter_value().string_value, self._on_auto_toggle, 5)
        # 主動中斷（TUI 's' 鍵發 Empty → running 中回待命、停車）
        self._stop_requested = False
        self.create_subscription(
            Empty, self.get_parameter("topic_pp_stop")
            .get_parameter_value().string_value, self._on_stop_trigger, 5)

        # ── 狀態機 ──
        # _state: "waiting"=等開到點上 / "ready"=已停在點上等空白鍵 / "running"=來回中
        self._state = "waiting"
        self._ready_point: str | None = None  # ready 時停在哪個點
        self._from: str | None = None  # 上一個點（origin）
        self._target: str | None = None  # 目前要去的點（destination）
        self._start_count = 0          # 就緒 dwell 計數
        self._arrive_count = 0         # 到達 dwell 計數
        self._lost_path_count = 0      # RUNNING 中無 path 計數（觸發重新規劃）
        self._auto_pause_count = 0     # 全自動模式：就緒緩衝計數
        self._warned_pose_src = False

        # 主狀態機 timer（2Hz）
        self.create_timer(0.5, self._tick)
        # 對外狀態發布（5Hz，供 TUI 顯示提示）
        self.create_timer(0.2, self._publish_pp_status)
        # 待命時定期提示（避免使用者不知道在等什麼）
        self.create_timer(5.0, self._idle_hint)

        self.get_logger().info(
            f"往返測試節點啟動：{self._pa} ↔ {self._pb}\n"
            f"  把車手動開到 {self._pa} 或 {self._pb} 停穩 → TUI 提示按【空白鍵】開始往對向點來回\n"
            f"  到點門檻 {self._arrive_tol}m、低速 {self._start_max_speed}m/s、"
            f"auto_set_nav={self._auto_set_nav}\n"
            f"  中途切 manual/estop/idle 即中斷，手動開回任一點重啟"
        )

    def _on_start_trigger(self, _msg: Empty):
        # 空白鍵（TUI 發）→ 標記請求開始；只有 ready 狀態 _tick 才會真正啟動
        self._start_requested = True

    def _on_auto_toggle(self, _msg: Empty):
        # TUI 'a' 鍵 → 熱切換全自動模式（就緒後不必按空白鍵、停穩自動發下一段）
        self._auto_continue = not self._auto_continue
        self._auto_pause_count = 0
        self.get_logger().info(
            f"{'🔁 全自動模式：開（停穩自動出發，無需空白鍵）' if self._auto_continue else '✋ 全自動模式：關（每段需按空白鍵）'}")

    def _on_stop_trigger(self, _msg: Empty):
        # TUI 's' 鍵 → 標記中斷請求；_tick 在 running 中消費（切 idle 停車、回待命）
        self._stop_requested = True

    def _publish_pp_status(self):
        st = {
            "state": self._state,        # waiting / ready / running
            "ready_point": self._ready_point,
            "from": self._from,
            "target": self._target,
            "a": self._pa, "b": self._pb,
            "nodes_loaded": self._nodes_loaded,
            "auto": self._auto_continue,
        }
        self.pub_pp.publish(String(data=json.dumps(st)))

    # ── 節點座標載入（同 routing_to_path / click_bridge 作法）──
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
            missing = [p for p in (self._pa, self._pb) if p not in self._nodes]
            if missing:
                self.get_logger().error(
                    f"路由節點表載入 {len(self._nodes)} 個，但找不到往返點 {missing}！"
                    f"請確認 point_a/point_b 是有效節點名。")
            else:
                self.get_logger().info(
                    f"載入 {len(self._nodes)} 個路由節點，"
                    f"{self._pa}={self._nodes[self._pa]} {self._pb}={self._nodes[self._pb]}")
        except Exception as e:
            self.get_logger().error(f"get_route_info 失敗: {e}")

    def _on_status(self, msg: String):
        try:
            self._status = json.loads(msg.data)
            self._status_t = time.monotonic()
        except (ValueError, TypeError):
            pass

    # ── 工具 ──
    def _pose(self):
        """回傳 (x, y, speed, mode, nav_type) 或 None（無有效 status）。"""
        if self._status is None:
            return None
        if time.monotonic() - self._status_t > self._stale_s:
            return None
        st = self._status
        x = st.get("pose_x")
        y = st.get("pose_y")
        if x is None or y is None:
            return None
        # NDT 沒在跑 → pose_src 非 tf，到點判定會失準，警告一次
        if not self._warned_pose_src and st.get("pose_src") != "tf":
            self.get_logger().warn(
                f"pose_src={st.get('pose_src')}（非 tf）→ 沒有 NDT map 定位，"
                f"往返到點判定可能失準。建議先啟動 NDT（ndt alias）。")
            self._warned_pose_src = True
        speed = abs(st.get("act_v") or 0.0)
        return x, y, speed, st.get("mode"), st.get("nav_type")

    def _dist_to(self, name: str, x: float, y: float):
        if name not in self._nodes:
            return None
        nx, ny = self._nodes[name]
        return math.hypot(nx - x, ny - y)

    def _nearest_endpoint(self, x: float, y: float):
        """回傳 (最近的往返點名, 距離)；節點未載入回 (None, inf)。"""
        cands = [(p, self._dist_to(p, x, y)) for p in (self._pa, self._pb)]
        cands = [(p, d) for p, d in cands if d is not None]
        if not cands:
            return None, float("inf")
        return min(cands, key=lambda c: c[1])

    def _call_routing(self, origin: str, dest: str):
        if self._routing_inflight:
            return
        if not self.routing_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("routing service 不可用（routing_to_path 沒在跑？）")
            return
        self._routing_inflight = True
        req = RoutingPath.Request()
        req.origin = origin
        req.destination = [dest]
        self.get_logger().info(f"規劃路徑：{origin} → {dest}")
        future = self.routing_client.call_async(req)
        future.add_done_callback(self._on_routing_done)

    def _on_routing_done(self, future):
        self._routing_inflight = False
        try:
            res = future.result()
            n = sum(len(p.poses) for p in res.routing) if res and res.routing else 0
            if n == 0:
                self.get_logger().warn("routing 回空路徑（稍後 watchdog 會重試）")
        except Exception as e:
            self.get_logger().error(f"routing 呼叫異常: {e}")

    def _set_nav(self):
        if self._auto_set_nav:
            self.pub_mode.publish(String(data="nav"))

    def _set_idle(self):
        # 到點後切 idle 讓車明確停住等下一段（auto_set_nav 關掉則不動 mode，靠路徑清空停車）
        if self._auto_set_nav:
            self.pub_mode.publish(String(data="idle"))

    # ── 待命提示 ──
    def _idle_hint(self):
        if self._state == "running" or not self._nodes_loaded:
            return
        p = self._pose()
        if p is None:
            self.get_logger().info("待命中：收不到 policy status（等 policy_node / NDT）…")
            return
        x, y, _, mode, _ = p
        name, d = self._nearest_endpoint(x, y)
        if self._state == "ready":
            other = self._pb if self._ready_point == self._pa else self._pa
            self.get_logger().info(
                f"已就緒：車停在 {self._ready_point} → "
                f"{f'全自動模式自動出發往 {other}' if self._auto_continue else f'在 TUI 按【空白鍵】開始往 {other}'}。")
        else:
            self.get_logger().info(
                f"待命中：最近往返點 {name} 距 {d:.2f}m（門檻 {self._arrive_tol}m），"
                f"mode={mode}。把車開到 {self._pa}/{self._pb} 停穩即就緒（再按空白鍵開始）。")

    # ── 主狀態機 ──
    def _tick(self):
        if not self._nodes_loaded:
            return
        p = self._pose()
        if p is None:
            return
        x, y, speed, mode, nav_type = p

        # 's' 鍵中斷：只在 running 生效（切 idle 停車、回待命）；其他狀態清掉避免殘留
        if self._stop_requested:
            self._stop_requested = False
            if self._state == "running":
                self._set_idle()
                self._state = "waiting"
                self._ready_point = None
                self._start_count = 0
                self._start_requested = False
                self._arrive_count = 0
                self._lost_path_count = 0
                self.get_logger().warn(
                    f"⏹ 收到中斷（s 鍵）→ 停車、回待命。"
                    f"把車開回 {self._pa}/{self._pb} 任一點停穩、再按空白鍵即重啟。")
                return

        if self._state == "waiting":
            self._tick_waiting(x, y, speed, mode)
        elif self._state == "ready":
            self._tick_ready(x, y, speed, mode)
        else:
            self._tick_running(x, y, mode, nav_type)

    def _tick_waiting(self, x, y, speed, mode):
        """待命：等車開到 A 或 B 停穩（停在點上 + 低速 + dwell）→ 就緒。

        mode 只擋 estop（= 使用者明確要停，不就緒）；nav/manual/idle 皆可就緒——
        真正的啟動閘門是「按空白鍵」，加上「距點 < arrive_tol + 速度 < start_max_speed」
        兩道硬檢查（RL 若在開，速度/距離一定不過關），故預設 nav 停在點上也能就緒。
        與 _tick_ready 的放行標準一致。
        """
        # estop 不就緒：estop = 使用者明確要停，不覆寫成 nav
        if mode == "estop":
            self._start_count = 0
            return
        name, d = self._nearest_endpoint(x, y)
        if name is None or d > self._arrive_tol or speed > self._start_max_speed:
            self._start_count = 0
            return
        self._start_count += 1
        if self._start_count < self._start_dwell:
            return
        # 就緒：等使用者按空白鍵（全自動模式則稍後自動出發）
        self._state = "ready"
        self._ready_point = name
        self._start_requested = False   # 清掉就緒前殘留的觸發
        self._auto_pause_count = 0
        self.get_logger().info(
            f"🟢 車已停在 {name}（距 {d:.2f}m）→ 就緒，"
            f"{'全自動模式將自動出發往 ' if self._auto_continue else '請在 TUI 按【空白鍵】開始往 '}"
            f"{self._pb if name == self._pa else self._pa}")

    def _tick_ready(self, x, y, speed, mode):
        """就緒（停在點上等空白鍵走下一段）：estop / 離開點 / 加速 → 退回待命。

        允許 mode 為 idle/manual/nav（初次手動停放為 manual/idle；到點停車為 idle，
        或 auto_set_nav 關時為 nav）；唯獨 estop 視為使用者要停，退回待命不放行。
        """
        name, d = self._nearest_endpoint(x, y)
        if (mode == "estop" or name != self._ready_point
                or d > self._arrive_tol or speed > self._start_max_speed):
            self._state = "waiting"
            self._ready_point = None
            self._start_count = 0
            self._start_requested = False
            self._auto_pause_count = 0
            return
        # 全自動模式：不必按空白鍵，就緒緩衝數拍後自動請求開始（緩衝期給中斷空間）
        if self._auto_continue and not self._start_requested:
            self._auto_pause_count += 1
            if self._auto_pause_count >= self._auto_pause_ticks:
                self._auto_pause_count = 0
                self._start_requested = True
                self.get_logger().info("🔁 全自動模式 → 自動出發下一段")
        # 收到空白鍵（或全自動觸發）→ 走下一段（往對向點）
        if self._start_requested:
            self._start_requested = False
            self._from = self._ready_point
            self._target = self._pb if self._from == self._pa else self._pa
            self._arrive_count = 0
            self._lost_path_count = 0
            self._state = "running"
            self.get_logger().info(
                f"✅ 收到開始（空白鍵）→ 出發往 {self._target}")
            self._set_nav()
            self._call_routing(self._from, self._target)

    def _tick_running(self, x, y, mode, nav_type):
        """前往對向點中：偵測中斷（mode 離開 nav）/ 到達（停車等空白鍵）/ 路徑遺失（重規劃）。"""
        # 中斷：使用者抓搖桿(manual) / estop / idle → 回待命
        if mode != "nav":
            self._state = "waiting"
            self._ready_point = None
            self._start_count = 0
            self._start_requested = False
            self.get_logger().warn(
                f"⏸ 偵測到中斷（mode={mode}）→ 停止往返。"
                f"手動把車開回 {self._pa}/{self._pb} 任一點停穩、再按空白鍵即重啟。")
            return

        # 到達對向點 → 停車回就緒，等空白鍵才走下一段（方便兩段間調整障礙物）
        d = self._dist_to(self._target, x, y)
        if d is not None and d < self._arrive_tol:
            self._arrive_count += 1
            if self._arrive_count >= self._arrive_dwell:
                arrived = self._target
                self._state = "ready"
                self._ready_point = arrived
                self._arrive_count = 0
                self._start_requested = False
                self._auto_pause_count = 0
                self._set_idle()   # 停住等下一段
                other = self._pb if arrived == self._pa else self._pa
                self.get_logger().info(
                    f"🏁 到達 {arrived}（距 {d:.2f}m）→ 停車就緒，"
                    f"{f'全自動模式將自動往 {other}' if self._auto_continue else f'按【空白鍵】往 {other}'}")
            return
        self._arrive_count = 0

        # 路徑遺失 watchdog：RUNNING 但無 path（routing 回空 / 被清）連續數拍 → 重規劃
        if nav_type != "path" and not self._routing_inflight:
            self._lost_path_count += 1
            if self._lost_path_count >= 6:   # 2Hz × 6 = 3s
                self.get_logger().warn(f"無有效路徑 3s → 重新規劃 {self._from} → {self._target}")
                self._lost_path_count = 0
                self._call_routing(self._from, self._target)
        else:
            self._lost_path_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = PingpongTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("⏹ pingpong_test 已停止")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
