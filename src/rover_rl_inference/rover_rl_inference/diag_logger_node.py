"""rover_rl 診斷記錄節點 — 獨立、不影響推論.

目的：每次「實驗」把 goal / agent 位置 / 速度 / NDT / policy 看到的 goal 方向 /
cmd_vel 連續記錄成 CSV（+可選 wandb），供事後分析「角速度晃動」「沒往 goal
前進」等問題。

設計（對齊 repo「獨立節點」哲學）：
  - 完全被動訂閱，不發任何 cmd，不碰 policy_node
  - **待命機制**：deploy_rl 啟動後不立刻記錄（避免錄到 NDT/RViz 收斂期雜訊）。
    在另一終端送 start → 開一個全新實驗（新 CSV + 新 wandb run，帶 label）。
        ros2 topic pub --once /rover_rl/record std_msgs/String "{data: 'start exp_name'}"
        ros2 topic pub --once /rover_rl/record std_msgs/String "{data: 'stop'}"
    （require_start=false 時退回舊行為：啟動即待錄，收到 goal 就寫）
  - 20 Hz timer 取最新快取 → 寫一列 CSV（時間序列，可畫圖）
  - 衍生指標：
      dist_to_goal        : robot(map) → goal(map) 距離
      heading_err_deg     : 真實「車頭朝向 vs goal 方向」誤差（NDT ground truth）
      policy_goal_ang_deg : policy obs 內的 goal body 角度（policy 實際在追的方向）
      → 兩者應一致；不一致 = 定位/TF/座標問題
  - wandb 可選；實車網路不穩建議 wandb_mode=offline，回頭 `wandb sync`
  - 開錄時抓 policy_node 全部參數（speed_rate / cmd_alpha_* 等）→ 寫
    `<csv>_params.json` sidecar + wandb run config（論文重現性 + 關聯晃動診斷）

訂閱：
  /odom /ndt_pose /goal_pose /global_path /input/nav_cmd_vel
  /rover_rl_policy/obs_debug  (79D obs，goal body 在 [4],[5])
  /rover_rl/lidar_sweep_72
  /rover_rl/record            (std_msgs/String) start/stop 控制
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from datetime import datetime

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String


def _pv_to_py(pv):
    """rcl_interfaces ParameterValue → python（避免版本差異，手動轉）."""
    t = pv.type
    return {
        1: lambda: pv.bool_value,
        2: lambda: pv.integer_value,
        3: lambda: pv.double_value,
        4: lambda: pv.string_value,
        5: lambda: list(pv.byte_array_value),
        6: lambda: list(pv.bool_array_value),
        7: lambda: list(pv.integer_array_value),
        8: lambda: list(pv.double_array_value),
        9: lambda: list(pv.string_array_value),
    }.get(t, lambda: None)()

CSV_FIELDS = [
    "t_wall", "t_rel",
    "goal_seq", "has_goal",
    "odom_x", "odom_y", "odom_yaw_deg", "odom_v", "odom_w",
    "ndt_x", "ndt_y", "ndt_yaw_deg", "ndt_age",
    "goal_x", "goal_y", "goal_frame",
    "dist_to_goal", "heading_err_deg",
    "cmd_v", "cmd_w", "cmd_age",
    "policy_goal_bx", "policy_goal_by", "policy_goal_ang_deg",
    "policy_v_norm", "policy_w_norm", "obs_age",
    "sweep_min_m", "sweep_age",
]


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _safe_label(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_\-]", "", s)[:48]


class DiagLoggerNode(Node):
    def __init__(self):
        super().__init__("rover_rl_diag_logger")
        self._declare_params()
        self._read_params()
        self._init_state()
        self._init_pubsub()
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._tick)

        if not self.require_start:
            self.start_experiment("")        # 舊行為：啟動即待錄
            ready = "已開錄（收到 goal 後寫列）"
        else:
            ready = ("待命中 — 在另一終端送 start 才開錄：\n"
                     "    ros2 topic pub --once /rover_rl/record std_msgs/String "
                     "\"{data: 'start 實驗名'}\"")
        self.get_logger().info(
            f"診斷記錄節點啟動  rate={self.rate_hz}Hz  "
            f"wandb={'on(' + self.wandb_mode + ')' if self.enable_wandb else 'off'}\n"
            f"  log_dir={self.log_dir}\n  {ready}"
        )

    # ── params ──
    def _declare_params(self) -> None:
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("log_dir", os.path.expanduser("~/rover_rl/logs"))
        self.declare_parameter("topic_odom", "/odom")
        self.declare_parameter("topic_ndt_pose", "/ndt_pose")
        self.declare_parameter("topic_goal_pose", "/goal_pose")
        self.declare_parameter("topic_global_path", "/global_path")
        self.declare_parameter("topic_cmd_vel", "/input/nav_cmd_vel")
        self.declare_parameter("topic_obs_debug", "/rover_rl_policy/obs_debug")
        self.declare_parameter("topic_lidar_sweep", "/rover_rl/lidar_sweep_72")
        self.declare_parameter("topic_record_ctrl", "/rover_rl/record")
        # 開錄時抓此節點的全部參數（speed_rate 等）寫進 sidecar + wandb config
        self.declare_parameter("policy_node_name", "rover_rl_policy")
        self.declare_parameter("lidar_r_max_m", 20.0)
        self.declare_parameter("robot_radius_m", 0.35)
        # false(預設)=啟動即待錄，發 goal 即開始；true=需送 record start 才開錄
        self.declare_parameter("require_start", False)
        # 只在有 goal 時寫列（省空間）；false=start 後一直寫
        self.declare_parameter("log_only_with_goal", True)
        self.declare_parameter("enable_wandb", False)
        self.declare_parameter("wandb_project", "rover_rl_deploy")
        # online / offline / disabled；實車建議 offline
        self.declare_parameter("wandb_mode", "offline")

    def _read_params(self) -> None:
        gp = self.get_parameter
        sv = lambda k: gp(k).get_parameter_value().string_value
        self.rate_hz = float(gp("rate_hz").value)
        self.log_dir = sv("log_dir")
        self.topic_odom = sv("topic_odom")
        self.topic_ndt = sv("topic_ndt_pose")
        self.topic_goal = sv("topic_goal_pose")
        self.topic_path = sv("topic_global_path")
        self.topic_cmd = sv("topic_cmd_vel")
        self.topic_obs = sv("topic_obs_debug")
        self.topic_sweep = sv("topic_lidar_sweep")
        self.topic_ctrl = sv("topic_record_ctrl")
        self.policy_node_name = sv("policy_node_name")
        self.r_max = float(gp("lidar_r_max_m").value)
        self.r_robot = float(gp("robot_radius_m").value)
        self.require_start = bool(gp("require_start").value)
        self.log_only_with_goal = bool(gp("log_only_with_goal").value)
        self.enable_wandb = bool(gp("enable_wandb").value)
        self.wandb_project = sv("wandb_project")
        self.wandb_mode = sv("wandb_mode") or "offline"

    # ── state ──
    def _init_state(self) -> None:
        self._t0 = time.monotonic()
        self._odom = None          # (x, y, yaw, v, w, t)
        self._ndt = None           # (x, y, yaw, t)
        self._goal = None          # (x, y, frame)
        self._goal_seq = 0
        self._cmd = None
        self._obs = None
        self._sweep_min = None
        # 實驗 / 檔案
        self._started = False
        self._fh = None
        self._writer = None
        self.csv_path = None
        self.wandb_run = None
        # policy 參數擷取（用底層 rcl_interfaces service client）
        self._cli_list = self.create_client(
            ListParameters, f"/{self.policy_node_name}/list_parameters")
        self._cli_get = self.create_client(
            GetParameters, f"/{self.policy_node_name}/get_parameters")
        self._params_timer = None
        self._params_attempts = 0
        self._policy_params = {}
        self._reset_run_stats()

    def _reset_run_stats(self) -> None:
        self._n_rows = 0
        self._cmd_w_prev = None
        self._dw_sq = 0.0
        self._dw_n = 0
        self._cmd_w_abs_sum = 0.0
        self._heading_abs_sum = 0.0
        self._heading_n = 0

    def _init_pubsub(self) -> None:
        be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        )
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)
        self.create_subscription(PoseStamped, self.topic_ndt, self._cb_ndt, 10)
        self.create_subscription(PoseStamped, self.topic_goal, self._cb_goal, 10)
        self.create_subscription(NavPath, self.topic_path, self._cb_path, 10)
        self.create_subscription(Twist, self.topic_cmd, self._cb_cmd, 10)
        self.create_subscription(Float32MultiArray, self.topic_obs, self._cb_obs, 10)
        self.create_subscription(Float32MultiArray, self.topic_sweep, self._cb_sweep, be)
        self.create_subscription(String, self.topic_ctrl, self._cb_ctrl, 10)

    # ── 實驗開關 ──
    def start_experiment(self, label: str) -> None:
        if self._started:
            self.stop_experiment()      # 先收掉上一個
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lab = _safe_label(label)
        name = f"diag_{stamp}" + (f"_{lab}" if lab else "")
        self.csv_path = os.path.join(self.log_dir, name + ".csv")
        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()
        self._reset_run_stats()
        self._goal_seq = 0
        self._goal = None
        self._t0 = time.monotonic()
        self._init_wandb(name)
        self._started = True
        self.get_logger().info(
            f"▶ 開始實驗 '{name}' → {self.csv_path}"
            + ("（指定 goal 後開始寫資料）" if self.log_only_with_goal else "")
        )
        self._schedule_param_capture()

    # ── 擷取 policy_node 參數（speed_rate 等）→ sidecar json + wandb config ──
    def _schedule_param_capture(self) -> None:
        if self._params_timer is not None:
            self._params_timer.cancel()
        self._params_attempts = 0
        self._policy_params = {}
        self._params_csv_path = self.csv_path     # 綁定本次實驗
        self._params_timer = self.create_timer(1.0, self._try_capture_params)

    def _try_capture_params(self) -> None:
        self._params_attempts += 1
        if self._params_attempts > 15:
            self.get_logger().warn(
                f"擷取 {self.policy_node_name} 參數逾時（節點未起？）— CSV 仍正常記錄"
            )
            self._params_timer.cancel()
            return
        if not (self._cli_list.service_is_ready() and self._cli_get.service_is_ready()):
            return                                # 下個 tick 再試
        self._params_timer.cancel()
        self._cli_list.call_async(
            ListParameters.Request()).add_done_callback(self._on_list_params)

    def _on_list_params(self, future) -> None:
        try:
            names = list(future.result().result.names)
        except Exception as e:
            self.get_logger().warn(f"list_parameters 失敗：{e}")
            return
        if not names:
            return
        req = GetParameters.Request()
        req.names = names
        self._cli_get.call_async(req).add_done_callback(
            lambda f: self._on_get_params(f, names)
        )

    def _on_get_params(self, future, names) -> None:
        try:
            values = future.result().values
        except Exception as e:
            self.get_logger().warn(f"get_parameters 失敗：{e}")
            return
        params = {n: _pv_to_py(v) for n, v in zip(names, values)}
        self._policy_params = params
        # sidecar json（與 CSV 同名 _params.json）
        meta = {
            "csv": os.path.basename(self._params_csv_path or ""),
            "policy_node": self.policy_node_name,
            "captured_wall": time.time(),
            "params": params,
        }
        try:
            side = (self._params_csv_path or "").rsplit(".", 1)[0] + "_params.json"
            with open(side, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(f"寫 params sidecar 失敗：{e}")
        # wandb config
        if self.wandb_run is not None:
            try:
                self.wandb_run.config.update(params, allow_val_change=True)
            except Exception as e:
                self.get_logger().warn(f"wandb config 更新失敗：{e}")
        key = {k: params.get(k) for k in (
            "speed_rate", "cmd_alpha_linear", "cmd_alpha_angular",
            "cmd_max_accel_angular", "act_max_angular_velocity", "control_dt",
        ) if k in params}
        self.get_logger().info(
            f"已擷取 {self.policy_node_name} 參數 {len(params)} 項 → sidecar/wandb；"
            f"關鍵：{key}"
        )

    def _init_wandb(self, run_name: str) -> None:
        self.wandb_run = None
        if not self.enable_wandb or self.wandb_mode == "disabled":
            return
        try:
            os.environ.setdefault("WANDB_MODE", self.wandb_mode)
            import wandb
            self.wandb = wandb
            self.wandb_run = wandb.init(
                project=self.wandb_project, name=run_name,
                mode=self.wandb_mode,
                config={"rate_hz": self.rate_hz},
            )
        except Exception as e:
            self.get_logger().warn(f"wandb 初始化失敗，改用純 CSV：{e}")
            self.wandb_run = None

    def stop_experiment(self) -> None:
        if not self._started:
            return
        self._print_summary()
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass
            self.wandb_run = None
        self._started = False

    def _cb_ctrl(self, msg: String) -> None:
        parts = msg.data.strip().split(None, 1)
        if not parts:
            return
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "start":
            self.start_experiment(arg)
        elif cmd == "stop":
            if self._started:
                self.get_logger().info("⏹ 收到 stop")
                self.stop_experiment()
            else:
                self.get_logger().info("收到 stop 但目前沒有進行中的實驗")
        else:
            self.get_logger().warn(f"未知 record 指令: {msg.data!r}（用 start / stop）")

    # ── callbacks ──
    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._odom = (
            p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w),
            msg.twist.twist.linear.x, msg.twist.twist.angular.z, time.monotonic(),
        )

    def _cb_ndt(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self._ndt = (
            msg.pose.position.x, msg.pose.position.y,
            _yaw_from_quat(q.x, q.y, q.z, q.w), time.monotonic(),
        )

    def _cb_goal(self, msg: PoseStamped) -> None:
        frame = msg.header.frame_id or "map"
        self._goal = (msg.pose.position.x, msg.pose.position.y, frame)
        self._goal_seq += 1
        if self._started:
            self.get_logger().info(
                f"[goal #{self._goal_seq}] ({msg.pose.position.x:.2f},"
                f"{msg.pose.position.y:.2f}) frame={frame}"
            )

    def _cb_path(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        last = msg.poses[-1].pose.position
        frame = msg.header.frame_id or "map"
        self._goal = (last.x, last.y, frame)
        self._goal_seq += 1
        if self._started:
            self.get_logger().info(
                f"[path #{self._goal_seq}] {len(msg.poses)} pts, 終點 "
                f"({last.x:.2f},{last.y:.2f}) frame={frame}"
            )

    def _cb_cmd(self, msg: Twist) -> None:
        self._cmd = (msg.linear.x, msg.angular.z, time.monotonic())

    def _cb_obs(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 6:
            self._obs = (list(msg.data), time.monotonic())

    def _cb_sweep(self, msg: Float32MultiArray) -> None:
        if not msg.data:
            return
        m = min(msg.data)
        dist_m = m * (self.r_max - self.r_robot) + self.r_robot
        self._sweep_min = (dist_m, time.monotonic())

    # ── 20Hz 寫列 ──
    def _tick(self) -> None:
        if not self._started:
            return
        now = time.monotonic()
        has_goal = self._goal is not None
        if self.log_only_with_goal and not has_goal:
            return

        row = {k: "" for k in CSV_FIELDS}
        row["t_wall"] = f"{time.time():.3f}"
        row["t_rel"] = f"{now - self._t0:.3f}"
        row["goal_seq"] = self._goal_seq
        row["has_goal"] = int(has_goal)

        robot_x = robot_y = robot_yaw = None
        if self._odom is not None:
            ox, oy, oyaw, ov, ow, ot = self._odom
            row["odom_x"] = f"{ox:.3f}"
            row["odom_y"] = f"{oy:.3f}"
            row["odom_yaw_deg"] = f"{math.degrees(oyaw):.2f}"
            row["odom_v"] = f"{ov:.3f}"
            row["odom_w"] = f"{ow:.3f}"

        if self._ndt is not None:
            nx, ny, nyaw, nt = self._ndt
            row["ndt_x"] = f"{nx:.3f}"
            row["ndt_y"] = f"{ny:.3f}"
            row["ndt_yaw_deg"] = f"{math.degrees(nyaw):.2f}"
            row["ndt_age"] = f"{now - nt:.2f}"
            robot_x, robot_y, robot_yaw = nx, ny, nyaw

        if robot_x is None and self._odom is not None:
            robot_x, robot_y, robot_yaw = self._odom[0], self._odom[1], self._odom[2]

        if has_goal:
            gx, gy, gframe = self._goal
            row["goal_x"] = f"{gx:.3f}"
            row["goal_y"] = f"{gy:.3f}"
            row["goal_frame"] = gframe
            if robot_x is not None:
                dx, dy = gx - robot_x, gy - robot_y
                dist = math.hypot(dx, dy)
                heading_err = _wrap_pi(math.atan2(dy, dx) - robot_yaw)
                row["dist_to_goal"] = f"{dist:.3f}"
                row["heading_err_deg"] = f"{math.degrees(heading_err):.2f}"
                self._heading_abs_sum += abs(math.degrees(heading_err))
                self._heading_n += 1

        if self._cmd is not None:
            cv, cw, ct = self._cmd
            row["cmd_v"] = f"{cv:.3f}"
            row["cmd_w"] = f"{cw:.3f}"
            row["cmd_age"] = f"{now - ct:.2f}"
            if self._cmd_w_prev is not None:
                dw = cw - self._cmd_w_prev
                self._dw_sq += dw * dw
                self._dw_n += 1
            self._cmd_w_prev = cw
            self._cmd_w_abs_sum += abs(cw)

        if self._obs is not None:
            obs, obt = self._obs
            bx, by = obs[4], obs[5]
            row["policy_goal_bx"] = f"{bx:.3f}"
            row["policy_goal_by"] = f"{by:.3f}"
            row["policy_goal_ang_deg"] = f"{math.degrees(math.atan2(by, bx)):.2f}"
            row["policy_v_norm"] = f"{obs[1]:.3f}"
            row["policy_w_norm"] = f"{obs[2]:.3f}"
            row["obs_age"] = f"{now - obt:.2f}"

        if self._sweep_min is not None:
            sm, st = self._sweep_min
            row["sweep_min_m"] = f"{sm:.3f}"
            row["sweep_age"] = f"{now - st:.2f}"

        self._writer.writerow(row)
        self._n_rows += 1
        if self._n_rows % 20 == 0:
            self._fh.flush()

        if self.wandb_run is not None:
            metrics = {k: float(v) for k, v in row.items()
                       if v != "" and k not in ("goal_frame", "t_wall")}
            self.wandb.log(metrics)

    # ── 摘要 ──
    def _print_summary(self) -> None:
        dw_std = math.sqrt(self._dw_sq / self._dw_n) if self._dw_n > 0 else 0.0
        cmd_w_abs = self._cmd_w_abs_sum / max(self._dw_n + 1, 1)
        head_abs = self._heading_abs_sum / max(self._heading_n, 1)
        self.get_logger().info(
            "================ 診斷摘要 ================\n"
            f"  CSV: {self.csv_path}  ({self._n_rows} 列)\n"
            f"  角速度晃動  : Δω RMS={dw_std:.3f} rad/s/step, |ω|平均={cmd_w_abs:.3f}\n"
            f"  朝向誤差    : |heading_err|平均={head_abs:.1f}°  (理想<20°)\n"
            f"  分析: ros2 run rover_rl_inference analyze_diag {self.csv_path or ''}\n"
            "=========================================="
        )

    def finalize(self) -> None:
        self.stop_experiment()


def main(args=None):
    rclpy.init(args=args)
    node = DiagLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()   # 印出診斷摘要 + CSV 存檔位置（若有在記錄）
        if node.csv_path is None:
            node.get_logger().info(
                "⏹ diag_logger 已停止（本次未啟動記錄，無 CSV 產生）"
            )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
