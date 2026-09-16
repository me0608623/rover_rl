"""baseline_arm — 消融實驗 baseline planner 的 arm（啟動）橋接。

為什麼需要這個節點（不是多餘的膠水）：
  campusrover_move 的 dwa_planner / path_following / mppi_planner 三者都以
  `action_flag_ = false` 開機，控制迴圈開頭就 `if (!action_flag_) return;`
  → **不呼叫 planner_function* service，車完全不動而且不印任何錯誤**。
  （dwa_planner.cpp:205/327、path_following.cpp:214/358、mppi_planner.cpp:414）

  更容易踩到的是：service 的 speed_parameter 會「覆寫」launch 參數裡的速度上限：
      max_linear_velocity_  = req->speed_parameter.linear.x;    // dwa:1179 / pid:1035
      max_angular_velocity_ = req->speed_parameter.angular.z;
  → 就算有呼叫，漏帶 speed_parameter 上限會變 0，一樣不動。
  同理 obstacle_avoidance 會覆寫 enable_costmap_obstacle（決定 dwa 吃不吃 costmap、
  pid 是不是真的零避障），所以消融實驗的「統一速度上限」與「避障開關」真值在這裡，
  不在 launch 的 parameters 區塊。

行為：
  訂閱 /global_path，在「首次拿到路徑」與「路徑終點換位置」時 arm 對應的 planner。
  routing_to_path 是 2Hz 無限 republish 同一條 path，所以用終點位移當判據，
  而不是每次收到都 call（重複 call 會重置 path_following 的加減速 step，造成頓挫）。

  mppi_planner 到達終點會自我 disarm（mppi_planner.cpp:466 action_flag_=false），
  往返測試的下一段路徑終點會換到對向點 → 由同一條「終點變化」規則自動 re-arm。
  dwa/path_following 不自我 disarm，但 re-arm 會清掉 arriving_end_point_，
  這正是每段新路徑需要的。

用法（一般由 deploy_full.launch.py 在 controller!=rl 時自動啟動）：
  ros2 run rover_rl_inference baseline_arm --ros-args -p controller:=dwa
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from campusrover_msgs.srv import PlannerFunction

# controller → 該 planner 的 arm service 名稱
# （三個節點各自 create_service 的相對名，在全域 namespace 下解析成 /xxx）
SERVICE_BY_CONTROLLER = {
    "dwa": "/planner_function_dwa",       # dwa_planner.cpp:134
    "pid": "/planner_function",           # path_following.cpp:111
    "pid_vo": "/planner_function",        # 同上（pid_vo 只是下游多串 VO）
    "mppi": "/planner_function_mppi",     # mppi_planner.cpp:38
}

MODE_GLOBAL_PATH = 1  # PlannerFunction.Request.MODE_GLOBAL_PATH


class BaselineArmNode(Node):
    def __init__(self):
        super().__init__("baseline_arm")

        self.declare_parameter("controller", "dwa")
        self.declare_parameter("topic_global_path", "/global_path")
        # 消融實驗統一的底盤真實上限（與 RL 實際可達上限一致）。
        # ⚠ 這兩個值會覆寫 launch parameters 裡的 max_*_velocity，是真正生效的那份。
        self.declare_parameter("max_linear_velocity", 1.0)
        self.declare_parameter("max_angular_velocity", 1.2)
        # 覆寫 planner 的 enable_costmap_obstacle：
        #   dwa=True（吃 costmap 做靜態避障）/ pid、pid_vo=False（零避障，交給 VO 或不做）
        self.declare_parameter("obstacle_avoidance", True)
        # 終點位移超過此距離才視為「新的一段」→ re-arm
        self.declare_parameter("goal_change_eps_m", 0.5)
        # 保險：距上次 arm 超過此秒數仍在同一段，再 arm 一次（0=關）。
        # 給「service 先於 planner 起來、第一次 call 掉了」這種時序意外留後路。
        self.declare_parameter("rearm_period_s", 0.0)

        self.controller = self.get_parameter("controller").value
        self.max_v = float(self.get_parameter("max_linear_velocity").value)
        self.max_w = float(self.get_parameter("max_angular_velocity").value)
        self.obstacle_avoidance = bool(self.get_parameter("obstacle_avoidance").value)
        self.goal_eps = float(self.get_parameter("goal_change_eps_m").value)
        self.rearm_period = float(self.get_parameter("rearm_period_s").value)

        srv_name = SERVICE_BY_CONTROLLER.get(self.controller)
        if srv_name is None:
            self.get_logger().error(
                f"未知的 controller='{self.controller}'，"
                f"可用：{sorted(SERVICE_BY_CONTROLLER)}；本節點不做任何事"
            )
            self.client = None
            return

        self.srv_name = srv_name
        self.client = self.create_client(PlannerFunction, srv_name)

        self._last_goal = None       # 上次 arm 用的終點 (x, y)
        self._last_arm_t = None      # 上次 arm 的時間（monotonic 用 node clock）
        self._pending = False        # 有 call 在途中，避免重複轟炸

        self.create_subscription(
            Path, self.get_parameter("topic_global_path").value, self._path_cb, 10
        )

        self.get_logger().info(
            f"[baseline_arm] controller={self.controller} → {srv_name}；"
            f"上限 v={self.max_v} ω={self.max_w}；"
            f"obstacle_avoidance={self.obstacle_avoidance}"
        )
        self.get_logger().info("[baseline_arm] 等待 /global_path…（沒有路徑不會 arm，車不動是正常的）")

    # ------------------------------------------------------------------
    def _path_cb(self, msg: Path):
        if self.client is None or not msg.poses:
            return

        goal = (msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y)
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._last_goal is None:
            reason = "首次收到路徑"
        elif math.hypot(goal[0] - self._last_goal[0], goal[1] - self._last_goal[1]) > self.goal_eps:
            reason = "路徑終點改變（新的一段）"
        elif (
            self.rearm_period > 0.0
            and self._last_arm_t is not None
            and (now - self._last_arm_t) > self.rearm_period
        ):
            reason = "週期性保險 re-arm"
        else:
            return

        self._arm(goal, now, reason)

    # ------------------------------------------------------------------
    def _arm(self, goal, now, reason: str):
        if self._pending:
            return
        if not self.client.service_is_ready():
            # planner 可能比本節點晚起；不阻塞 spin，下一次 path（2Hz）再試
            self.get_logger().warn(
                f"[baseline_arm] {self.srv_name} 尚未就緒，等下一次路徑再試"
                "（planner 節點還沒起來？）",
                throttle_duration_sec=5.0,
            )
            return

        req = PlannerFunction.Request()
        req.action.data = True
        req.direction_inverse.data = False
        req.obstacle_avoidance.data = self.obstacle_avoidance
        req.mode = MODE_GLOBAL_PATH
        req.speed_parameter.linear.x = self.max_v
        req.speed_parameter.angular.z = self.max_w

        self._pending = True
        future = self.client.call_async(req)
        future.add_done_callback(lambda f: self._on_armed(f, goal, now, reason))

    def _on_armed(self, future, goal, now, reason: str):
        self._pending = False
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001 - service 失敗要看得到原因
            self.get_logger().error(f"[baseline_arm] arm 失敗：{exc}")
            return

        self._last_goal = goal
        self._last_arm_t = now
        self.get_logger().info(
            f"[baseline_arm] ✅ 已 arm {self.controller}（{reason}）"
            f" 終點=({goal[0]:.2f}, {goal[1]:.2f}) v≤{self.max_v} ω≤{self.max_w}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BaselineArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
