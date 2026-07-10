"""VO 安全層 ROS 節點 — 夾在 RL policy 與底盤 mux 之間.

資料流（由 deploy_full 串接）：
    policy_node ──/rover_rl/cmd_vel_desired──► [vo_safety_node] ──/input/nav_cmd_vel──► mux ──► 底盤
                                                    ▲
                       vo_interface  /vo_interface/tracked_obstacles (topic, 20Hz)

職責：訂閱 RL 期望 cmd + odom + vo_interface 追蹤障礙，跑 vo_layer.compute_safe_cmd，
     以 20Hz 發出安全 cmd。演算法在 vo_layer.py（純函式可測），本檔只做 ROS 接線 + 看門狗。

為何吃 vo_interface 而非直接輪詢 LV-DOT service？
  vo_interface 已對 LV-DOT 動態框做 CV-Kalman 重追蹤，給出「持久 ID + 平滑絕對速度」。
  rollout 碰撞預測靠 p_i + v_i·t 外推，最吃速度準度；LV-DOT raw 速度偏跳會讓快速障礙物
  的預測碰撞點亂飄（該煞沒煞 / 亂煞）。改吃平滑速度後，前/側快速切入的軌跡外推更穩。

看門狗（fail-safe，安全優先）：
  - RL desired 逾時 → 發 0（RL 沉默不亂動）
  - odom 逾時 → 發 0（不知道自身狀態不敢動）
  - 障礙逾時 → 視為無障礙（VO 退化成純放行 desired，靜態仍由 RL 的 LiDAR sweep 擋）

⚠️ 注意：此層只處理「動態障礙」（service 回傳的 dynamicBBoxes_ 已篩動態）；
   靜態避障仍由 RL policy 的 72-bin sweep 負責，避免雙重保守導致原地凍結。
⚠️ 多一層會略增延遲（對應 CLAUDE.md sim-to-real gap #5），故保持輕量、單純。
"""
from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from vo_interface.msg import TrackedObstacleArray
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray

from .vo_layer import Obstacle, RobotState, VOParams, _clamp, compute_safe_cmd


def _safe_float(x) -> float | None:
    """JSON 欄位安全轉 float：非數字 / NaN / inf → None（防污染控制比較）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """四元數 → yaw（平面導航只需繞 z 的旋轉角）。"""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class VOSafetyNode(Node):
    def __init__(self) -> None:
        super().__init__("vo_safety_node")

        gp = self.declare_parameter
        # --- Topics / service ---
        gp("topic_cmd_in", "/rover_rl/cmd_vel_desired")   # RL 期望（policy 改發到這）
        gp("topic_cmd_out", "/input/nav_cmd_vel")         # 安全輸出（送進 mux）
        gp("topic_odom", "/odom")
        gp("topic_obstacles", "/vo_interface/tracked_obstacles")  # vo_interface 平滑追蹤
        gp("topic_goal_status", "/rover_rl_policy/status")  # policy status JSON（取 goal_dist/goal_ang_deg）
        # --- 看門狗逾時 ---
        gp("timeout_desired_s", 0.5)   # RL cmd 超過此值沒更新 → 發 0
        gp("timeout_odom_s", 0.3)
        gp("timeout_obstacles_s", 1.0) # 障礙超過此值沒更新 → 視為無障礙
        gp("timeout_goal_s", 1.5)      # goal 超過此值沒更新 → 視為無 goal（VO 退回純貼近 RL）
        # --- 年輕 track 過濾：速度尚未收斂的 track 不參與 rollout（避免亂煞）---
        gp("min_vel_confidence", 0.3)  # vel_confidence 低於此值的 track 當靜態位置避（不外推跳動速度，但不 drop）
        # --- 頻率 ---
        gp("ctrl_rate_hz", 20.0)
        # --- VO 演算法參數（對應 vo_layer.VOParams）---
        gp("v_max", 1.0)
        gp("v_min", -1.0)
        gp("w_max", 1.2)
        gp("accel_v", 1.2)
        gp("accel_w", 3.0)
        gp("window_time", 1.0)   # 候選視窗可達時間（與 ctrl_dt 解耦，讓 VO 能繞行）
        gp("n_v", 7)
        gp("n_w", 15)
        gp("horizon", 2.0)
        gp("sim_dt", 0.1)
        gp("r_robot", 0.35)
        gp("margin", 0.10)
        gp("w_v", 1.0)
        gp("w_w", 0.3)
        gp("w_goal", 0.0)   # goal 導向項權重（0=關，純貼近 RL；>0 被擋時偏好繞向 goal）
        gp("engage_range", 6.0)
        gp("clearance_fallback", False)  # 無可行候選時改慢爬向縫隙（True）或急停（False, 預設）
        gp("v_creep", 0.3)               # clearance 退化時的線速度上限
        gp("goaround_v_cap", 0.0)        # RL 縮手時 VO 繞行前進速度地板（0=純守 v_des；0.3=能慢繞）
        # --- 卡死逃脫（opt-in）：blocked+停住超過 N 秒且後方有空間 → 慢退開出空間再繞 ---
        gp("stuck_escape_enable", False) # True=啟用卡死後退逃脫；False(預設)=堵死就停在原地
        gp("stuck_timeout_s", 3.0)       # blocked+停住連續多久才觸發後退（設長避免有人只是路過就亂退）
        gp("rear_clear_min_m", 1.2)      # 後退【中止】門檻：back_m 掉到此值→立刻停（防倒退撞牆/人）
        gp("rear_clear_start_m", 1.2)    # 後退【觸發】門檻：back_m 需 > 此值才敢啟動後退（純 LiDAR，牆/人都算）
        gp("escape_block_if_rear_person", False) # 後方額外用 VO 追蹤人擋退（後方 LV-DOT 覆蓋僅~2%，雞肋；預設關）
        gp("escape_rear_person_range_m", 2.0)    # 後方多遠內有追蹤到的人就擋退（m，僅 enable 時作用）
        gp("escape_reverse_v", -0.2)     # 逃脫後退速度（m/s，負值；慢退）
        gp("escape_reverse_accel", 0.3)  # ★建立倒退速度時的「溫柔加速度」(m/s²)：對齊人工示範慢慢加速，
                                         #   不共用底盤 accel_v(1.2)那種猛倒。只管「越退越快」的建立段。
        gp("escape_max_dist_m", 1.0)     # 後退距離上限：退這麼多仍堵死→放棄停車（防前方一直逼→無限後退）
        gp("escape_turn_w", 0.0)         # 後退時朝「LiDAR 較開側」轉的角速度（rad/s）；0=直退（舊行為）
        gp("escape_turn_min_side_m", 0.8)  # 轉向側的 LiDAR 淨空需 > 此才邊退邊轉；兩側都窄→直直退（純退讓）
        gp("escape_aim_w_thresh", 0.3)   # 「鼻子已對準」判據：有 v>0 且 |w|≤此 的可行候選＝接近直行能前進
        gp("escape_aim_timeout_s", 5.0)  # 逃脫(倒車+轉向對準)總時限；逾時仍對不準開闊側→放棄停車
        gp("escape_aim_cone_deg", 50.0)  # 對準判據：前向±此角度錐內須無近障才准退出前進（防退遠→straight_feasible假True→鼻子仍朝障礙直行撞上）
        gp("obstacle_radius_max", 1.5)   # bbox 換算半徑的上限（防超大框把路全擋死）
        # --- 繞行側 commit（解「往右退→又往左轉」橫跳）---
        gp("w_commit", 2.0)              # commit 偏置權重：終點偏 commit 反側的加罰（0=關）
        gp("commit_hold_s", 3.0)         # escape 鎖定側後，過障礙仍持續 commit 的保持秒數（engaged 會續命）
        # --- 前方 LiDAR 安全煞（★只預設版 vo_params.yaml 開；吃 RL policy 的 front_m，含牆任何物）---
        # VO 的幾何式「停」旁邊有縫會繞、不保證某距離停。此煞是距離硬底線：front_m≤slow 線性減速、
        # ≤stop 禁止前進（可倒退/轉向逃離）→ 保證「0.5m 一定停下不前進」，不論障礙是人是牆。
        gp("front_brake_enable", False)  # True=啟用前方 LiDAR 安全煞（預設版開，滿血版不設→關）
        gp("front_brake_slow_m", 0.6)    # front_m ≤ 此值 → 線性減速（介入起點）
        gp("front_brake_stop_m", 0.5)    # front_m ≤ 此值 → 禁止前進（必停；仍可倒退/原地轉逃離）
        # 前方煞「直退避讓」（opt-in）：≤stop 且後方夠空 → 直直倒車退開，退到 ≥release 才收（遲滯防抖）。
        # 純直退（w=0），適用「牆等靜物 VO 看不到」場景；後方被擋則不退、退回只停。
        gp("front_brake_reverse", False)   # True=觸發 stop 時直退避讓；False=只停+保留 RL 轉向
        gp("front_brake_release_m", 0.65)  # 倒車退到 front_m ≥ 此值才收手（須 > slow_m，遲滯）
        gp("front_brake_reverse_delay_s", 2.0)  # ≤stop 先停住連續這麼久才開始退（仿舊 stuck_escape 2s）
        # 前方煞「讓 VO 繞」：前方 front_m 若是 VO 追蹤的障礙(人)造成 → 交給 VO 繞/停/退，front-brake
        # 讓開（否則硬停會砍掉繞行的前進弧）。只對「VO 看不到的物(牆)」硬管；另設絕對緊急距離兜底。
        gp("front_brake_defer_to_vo", True)     # True=VO 看得到的前方障礙交給 VO；False=一律硬管(舊行為)
        gp("front_brake_defer_cone_deg", 45.0)  # 判「前方障礙」的錐角(±deg)
        gp("front_brake_emergency_m", 0.55)     # 絕對硬停底線：任何物(含 VO 正在繞的人)到此距離一律停
        # --- 前方「定時硬凍結」（opt-in，獨立於 front_brake_enable）---
        # 近障(front_m≤trigger) → v/w 瞬間全歸 0、凍結 hold_s 秒 → 時間到「無條件」把控制權交回 RL，
        # 即使障礙還在也放行（RL 自身 LiDAR sweep 仍會避）。解除後須退到 ≥rearm_m 才允許再次觸發，
        # 避免同一障礙每 hold_s 秒重觸發抽動。吃 RL front_m（含牆任何物）。
        gp("front_freeze_enable", False)     # True=啟用兩段式前方安全（0.7m freeze + 0.5m 硬停後退）
        # 第一段 freeze：front_m ≤ trigger → 停 hold_s → 交回 RL（暫停讓 RL 重試）
        gp("front_freeze_trigger_m", 0.7)    # front_m ≤ 此值 → 觸發 freeze
        gp("front_freeze_hold_s", 1.0)       # freeze 持續秒數（到點交回 RL）
        gp("front_freeze_rearm_m", 0.9)      # freeze 解除後退到 front_m ≥ 此值才能再次觸發（遲滯，須 > trigger）
        # 第二段硬停+後退：front_m ≤ hardstop_m → 持續硬停（車過不了此距離）；持續 ≤hardstop_m 達
        #   dwell_s 秒 → 觸發後退（用 escape_reverse_v/accel，受後方安全檢查），退到 front_m ≥ trigger 才收。
        gp("front_hardstop_m", 0.5)          # front_m ≤ 此值 → 強制硬停（持續，禁前進）
        gp("front_hardstop_dwell_s", 1.0)    # 前方持續 ≤hardstop_m 這麼久才觸發後退（確認障礙沒走）
        # ★前方「全身堵滿」才停：front_m 近 + ±30°扇區填滿率 ≥ 此值才算「人正對堵死」→ 停。
        # 填滿率低（人偏一側/有縫）→ 不停，讓 RL/VO 繞。取代舊「單點近就停」。
        gp("front_block_ratio_thresh", 0.8)     # ±30° 前方近物覆蓋率門檻(0~1)；0.8=80%bin是近物才停
        # ★「嚴格前方」模式（TUI 'f' 鍵熱切換）：ON 時 VO 只處理「前方±cone° 且(卡住≥stuck_s 或
        #   直衝車端)」的障礙；側邊/路過的人忽略。用途：讓 VO 不再對側向路人減速，只擋真正威脅。
        gp("vo_strict_front", False)            # True=啟用嚴格前方過濾（運行時由 topic toggle）
        gp("topic_strict_front_toggle", "/vo_safety/strict_front_toggle")
        gp("strict_front_cone_deg", 15.0)       # 嚴格模式前方錐角(±deg)
        gp("strict_front_stuck_s", 2.0)         # 障礙在錐內停滯≥此秒才算「卡住」
        gp("strict_front_stuck_vel", 0.3)       # 障礙速度低於此(m/s)才算「停滯」
        gp("strict_front_closing_ms", 0.15)     # 接近率(closing)>此 m/s 才算「直衝車端」
        gp("strict_front_lateral_ms", 0.25)     # 同時橫向相對速<此才算「直直衝」(非橫越)
        # --- 靜止人源（★只滿血版 vo_params_full.yaml 開；預設版不設→關）---
        # 站著不動的人 LV-DOT 不判 dynamic（YOLO↔3D 關聯常斷）→ VO 看不到。此源自建：拿 YOLO
        # person 2D + visual_bboxes 3D 做方位角關聯，補「人的靜態位置障礙」給 VO（v=0，只避人不避牆）。
        gp("static_human_enable", False)
        gp("topic_yolo", "/yolo_detector/detected_bounding_boxes")
        gp("topic_visual_bboxes", "/onboard_detector/visual_bboxes")
        gp("cam_fx", 614.2774658203125)          # color camera_info k[0]
        gp("cam_cx", 325.5475158691406)          # color camera_info k[2]
        gp("static_human_assoc_tol_deg", 12.0)   # 方位角關聯容差（實測人誤差≤11°、牆≥15°）
        gp("static_human_timeout_s", 0.5)        # yolo/visual 任一逾時 → 不補人源
        gp("static_human_radius_max", 0.45)      # 人等效半徑上限（box 換算，下限 0.25）
        gp("static_human_dedup_m", 0.6)          # 與既有動態 track 這麼近 → 同一人不重複加

        g = self.get_parameter
        self.topic_in = g("topic_cmd_in").get_parameter_value().string_value
        self.topic_out = g("topic_cmd_out").get_parameter_value().string_value
        self.topic_odom = g("topic_odom").get_parameter_value().string_value
        self.topic_obs = g("topic_obstacles").get_parameter_value().string_value
        self.topic_goal_status = g("topic_goal_status").get_parameter_value().string_value
        self.timeout_desired = float(g("timeout_desired_s").value)
        self.timeout_odom = float(g("timeout_odom_s").value)
        self.timeout_obs = float(g("timeout_obstacles_s").value)
        self.timeout_goal = float(g("timeout_goal_s").value)
        self.min_vel_conf = float(g("min_vel_confidence").value)
        ctrl_hz = float(g("ctrl_rate_hz").value)
        self.r_obs_max = float(g("obstacle_radius_max").value)

        # 控制週期 ctrl_dt 即 dynamic window 的可達時間（與 ctrl_rate 對齊）
        self.vo = VOParams(
            v_max=float(g("v_max").value), v_min=float(g("v_min").value),
            w_max=float(g("w_max").value),
            accel_v=float(g("accel_v").value), accel_w=float(g("accel_w").value),
            ctrl_dt=1.0 / ctrl_hz,
            window_time=float(g("window_time").value),
            n_v=int(g("n_v").value), n_w=int(g("n_w").value),
            horizon=float(g("horizon").value), sim_dt=float(g("sim_dt").value),
            w_aim_thresh=float(g("escape_aim_w_thresh").value),
            r_robot=float(g("r_robot").value), margin=float(g("margin").value),
            w_v=float(g("w_v").value), w_w=float(g("w_w").value),
            w_goal=float(g("w_goal").value),
            w_commit=float(g("w_commit").value),
            engage_range=float(g("engage_range").value),
            clearance_fallback=bool(g("clearance_fallback").value),
            v_creep=float(g("v_creep").value),
            goaround_v_cap=float(g("goaround_v_cap").value),
        )
        # 卡死逃脫參數（node 層狀態機，不進 vo_layer 純函式）
        self.stuck_escape_enable = bool(g("stuck_escape_enable").value)
        self.stuck_timeout_s = float(g("stuck_timeout_s").value)
        self.rear_clear_min = float(g("rear_clear_min_m").value)
        self.rear_clear_start = float(g("rear_clear_start_m").value)
        self.escape_block_if_rear_person = bool(g("escape_block_if_rear_person").value)
        self.escape_rear_person_range = float(g("escape_rear_person_range_m").value)
        self.escape_reverse_v = float(g("escape_reverse_v").value)
        self.escape_reverse_accel = float(g("escape_reverse_accel").value)
        self.escape_max_dist = float(g("escape_max_dist_m").value)
        self.escape_turn_w = float(g("escape_turn_w").value)
        self.escape_turn_min_side = float(g("escape_turn_min_side_m").value)
        self.escape_aim_timeout = float(g("escape_aim_timeout_s").value)
        self.escape_aim_cone = math.radians(float(g("escape_aim_cone_deg").value))
        self.commit_hold_s = float(g("commit_hold_s").value)
        # 前方 LiDAR 安全煞
        self.front_brake_enable = bool(g("front_brake_enable").value)
        self.front_slow_m = float(g("front_brake_slow_m").value)
        self.front_stop_m = float(g("front_brake_stop_m").value)
        self.front_brake_reverse = bool(g("front_brake_reverse").value)
        self.front_release_m = float(g("front_brake_release_m").value)
        self.front_reverse_delay = float(g("front_brake_reverse_delay_s").value)
        self.front_defer_to_vo = bool(g("front_brake_defer_to_vo").value)
        self.front_defer_cone = math.radians(float(g("front_brake_defer_cone_deg").value))
        self.front_emergency_m = float(g("front_brake_emergency_m").value)
        # 前方定時硬凍結
        self.front_freeze_enable = bool(g("front_freeze_enable").value)
        self.front_freeze_trigger_m = float(g("front_freeze_trigger_m").value)
        self.front_freeze_hold_s = float(g("front_freeze_hold_s").value)
        self.front_freeze_rearm_m = float(g("front_freeze_rearm_m").value)
        self.front_hardstop_m = float(g("front_hardstop_m").value)
        self.front_hardstop_dwell_s = float(g("front_hardstop_dwell_s").value)
        self.front_block_ratio_thresh = float(g("front_block_ratio_thresh").value)
        # 嚴格前方模式
        self._strict_front = bool(g("vo_strict_front").value)
        topic_strict = g("topic_strict_front_toggle").get_parameter_value().string_value
        self.strict_cone = math.radians(float(g("strict_front_cone_deg").value))
        self.strict_stuck_s = float(g("strict_front_stuck_s").value)
        self.strict_stuck_vel = float(g("strict_front_stuck_vel").value)
        self.strict_closing = float(g("strict_front_closing_ms").value)
        self.strict_lateral = float(g("strict_front_lateral_ms").value)
        # 靜止人源
        self.static_human_enable = bool(g("static_human_enable").value)
        self.topic_yolo = g("topic_yolo").get_parameter_value().string_value
        self.topic_visual = g("topic_visual_bboxes").get_parameter_value().string_value
        self.cam_fx = float(g("cam_fx").value)
        self.cam_cx = float(g("cam_cx").value)
        self.static_human_tol = math.radians(float(g("static_human_assoc_tol_deg").value))
        self.static_human_timeout = float(g("static_human_timeout_s").value)
        self.static_human_r_max = float(g("static_human_radius_max").value)
        self.static_human_dedup = float(g("static_human_dedup_m").value)

        # --- 狀態 ---
        self._des_v = 0.0
        self._des_w = 0.0
        self._des_t = 0.0
        self._robot: RobotState | None = None
        self._odom_t = 0.0
        self._obstacles: list[Obstacle] = []
        self._obs_t = 0.0
        self._goal_active = False    # 本拍是否確實把 goal 導向項納入（供 status 顯示）
        # goal（取自 policy status，body-frame 相對量；每拍配 odom_yaw 反推 odom 座標）
        self._goal_dist: float | None = None      # m，body-frame goal 距離
        self._goal_bearing: float | None = None   # rad，body-frame goal 方位（相對車頭）
        self._goal_t = 0.0
        self._out_v = 0.0            # 上次輸出（輸出端 slew 限速用）
        self._out_w = 0.0
        # 卡死逃脫狀態
        self._back_m: float | None = None   # 後方 LiDAR 淨空（取自 policy status back_m）
        self._left_m: float | None = None   # 左側 LiDAR 淨空（escape 牆感知轉向用）
        self._right_m: float | None = None  # 右側 LiDAR 淨空
        self._front_m: float | None = None  # 前方 LiDAR 最近距（前方安全煞用；含牆任何物）
        self._front_block_ratio: float | None = None  # ±30° 前方扇區填滿率（0~1）
        self._sweep_t = 0.0                  # policy status（含 front_m）最後到達 monotonic 時間
        # 靜止人源快取
        self._yolo_betas: list[float] = []  # YOLO person 相機方位（rad，右正）
        self._yolo_t = 0.0
        self._visual_boxes: list[tuple[float, float, float]] = []  # (x,y,r) odom
        self._visual_t = 0.0
        self._human_obs: list[Obstacle] = []  # 本拍補的人源障礙（供 status/cone 檢查）
        self._front_brake = ""               # 前方安全煞狀態："stop"/"slow"/"reverse"/""（供 status）
        self._front_brake_prev = ""          # 前一拍前方煞狀態（狀態變化 log 用）
        self._front_reversing = False        # 前方煞直退避讓遲滯狀態
        self._front_freeze_since = None      # 定時凍結起算 monotonic（None=未凍結）
        self._front_freeze_armed = True      # 是否允許觸發凍結（False=待退到 rearm 外重置）
        self._front_hardstop_since = None    # 前方硬停(≤hardstop_m)持續起算 monotonic（dwell 計時）
        self._front_hardstop_reversing = False  # 硬停後退相位是否進行中
        # 嚴格前方模式：錐內障礙年齡追蹤 {pos_grid: [first_seen, last_seen]}
        self._cone_age: dict[tuple, list[float]] = {}
        self._front_stop_since = None        # 進入前方 stop 區的 monotonic 時間（2s 才退用）
        self._stuck_since: float | None = None  # 開始 blocked+停住的 monotonic 時間
        self._escaping = False       # 是否正在後退逃脫
        self._escape_start_xy = (0.0, 0.0)  # 後退起點 odom（量後退距離）
        self._escape_start_t = 0.0   # 逃脫開始 monotonic 時間（對準逾時用）
        self._escape_exhausted = False  # 已退滿上限仍堵死→放棄，待前方有解才重置
        # 繞行側 commit：+1=左、-1=右、0=未鎖；過障礙後（commit_until 過期）自動清
        self._commit_side = 0
        self._commit_until = 0.0
        # status 節流：control loop 20Hz，狀態只需 ~5Hz（對齊 policy_node ~/status）
        self._status_every = max(1, int(round(ctrl_hz / 5.0)))
        self._status_tick = 0

        # --- ROS 介面 ---
        self.pub_cmd = self.create_publisher(Twist, self.topic_out, 10)
        # 精簡狀態 JSON（供 status_tui 顯示 VO 介入/參數，純觀察不影響控制）→ /vo_safety_node/status
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.create_subscription(Twist, self.topic_in, self._cb_desired, 10)
        self.create_subscription(Odometry, self.topic_odom, self._cb_odom, 10)
        self.create_subscription(TrackedObstacleArray, self.topic_obs,
                                 self._cb_obstacles, 10)
        # goal 來源：policy status JSON（純訂閱觀察，不耦合控制安全；收不到→退回純貼近 RL）
        self.create_subscription(String, self.topic_goal_status,
                                 self._cb_goal_status, 10)
        # 靜止人源訂閱（只滿血版啟用時才接）
        if self.static_human_enable:
            self.create_subscription(Detection2DArray, self.topic_yolo, self._cb_yolo, 10)
            self.create_subscription(MarkerArray, self.topic_visual, self._cb_visual, 5)
        # 嚴格前方模式熱切換（TUI 'f' 鍵發 Empty）
        from std_msgs.msg import Empty as EmptyMsg
        self.create_subscription(EmptyMsg, topic_strict, self._cb_strict_toggle, 5)

        self.create_timer(1.0 / ctrl_hz, self._tick_ctrl)

        self.get_logger().info(
            f"VO 安全層啟動：{self.topic_in} → (VO) → {self.topic_out}；"
            f"障礙 topic={self.topic_obs}；w_max={self.vo.w_max} horizon={self.vo.horizon}s"
        )

    # ── callbacks ──
    def _cb_desired(self, msg: Twist) -> None:
        self._des_v = msg.linear.x
        self._des_w = msg.angular.z
        self._des_t = time.monotonic()

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        tw = msg.twist.twist
        self._robot = RobotState(
            x=p.x, y=p.y, yaw=_yaw_from_quat(q.x, q.y, q.z, q.w),
            v=tw.linear.x, w=tw.angular.z,
        )
        self._odom_t = time.monotonic()

    # ── 障礙訂閱（vo_interface 已做 KF 平滑追蹤，直接吃 topic）──
    def _cb_obstacles(self, msg: TrackedObstacleArray) -> None:
        obstacles: list[Obstacle] = []
        for ob in msg.obstacles:
            # ⚠ 不對任何障礙做「整顆 drop」——drop 等於製造安全洞（曾加 min_obstacle_speed
            # 速度門檻，會把近前小碎步的人、KF 速度未收斂的新闖入者整顆丟掉 → 已收回）。
            # 每顆障礙都進 VO；只在「速度估得準不準」這層做保守處理（見下）。
            # vo_interface 已給等效半徑；仍封頂防超大框把整條路擋死
            r = min(float(ob.radius), self.r_obs_max)
            # 速度尚未收斂的年輕 track（含剛冒出的快速闖入者、站著小碎步的人）：
            # 忽略其跳動速度（vx=vy=0），當「靜態位置」避——VO 仍看得到會減速/停，只是不拿
            # 沒收斂的速度亂外推。比 drop 安全（不漏看），比硬信跳動速度穩（不亂煞）。
            if ob.vel_confidence < self.min_vel_conf:
                vx, vy = 0.0, 0.0
            else:
                vx, vy = float(ob.velocity.x), float(ob.velocity.y)
            obstacles.append(Obstacle(x=float(ob.position.x), y=float(ob.position.y),
                                      vx=vx, vy=vy, r=r))
        self._obstacles = obstacles
        self._obs_t = time.monotonic()

    # ── goal 訂閱（policy status JSON；取 body-frame goal_dist/goal_ang_deg）──
    def _cb_goal_status(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        # 後方淨空（72-bin sweep 還原的後方最近障礙）：卡死逃脫倒退前的安全判據。
        # goal 為 None（待命）時也要更新，故放在 goal 判斷之前。
        self._back_m = _safe_float(d.get("back_m"))
        self._left_m = _safe_float(d.get("left_m"))
        self._right_m = _safe_float(d.get("right_m"))
        self._front_m = _safe_float(d.get("front_m"))  # 前方安全煞用
        self._front_block_ratio = _safe_float(d.get("front_block_ratio"))  # ±30° 前方填滿率
        self._sweep_t = time.monotonic()               # 前方安全煞的新鮮度基準
        gd = _safe_float(d.get("goal_dist"))
        ga = _safe_float(d.get("goal_ang_deg"))
        if gd is None or ga is None:
            # 無目標（待命）/ 數值異常 → 清掉 goal，VO 退回純貼近 RL
            self._goal_dist = None
            self._goal_bearing = None
            return
        self._goal_dist = gd
        self._goal_bearing = math.radians(ga)
        self._goal_t = time.monotonic()

    def _goal_in_odom(self, now: float) -> tuple[float, float] | None:
        """body-frame goal 配「當下 odom 車姿」反推 odom 座標.

        goal_bearing 為 body-relative（frame-independent），故 odom 系方位 =
        odom_yaw + goal_bearing，再沿此方位推 goal_dist → 與 VO rollout 同在 odom 系。
        逾時 / 無 goal / 無車姿 → None（VO 退回純貼近 RL）。
        """
        if (self._goal_dist is None or self._goal_bearing is None
                or self._robot is None
                or (now - self._goal_t) > self.timeout_goal):
            return None
        bearing_odom = self._robot.yaw + self._goal_bearing
        gx = self._robot.x + self._goal_dist * math.cos(bearing_odom)
        gy = self._robot.y + self._goal_dist * math.sin(bearing_odom)
        return (gx, gy)

    # ── 靜止人源：YOLO person 2D 相機方位 + visual_bboxes 3D（只滿血版）──
    def _cb_yolo(self, msg: Detection2DArray) -> None:
        # YOLO 節點已只發 person 框；bbox.center.position=左上角、size=寬高（640×480）
        betas = []
        for det in msg.detections:
            uc = det.bbox.center.position.x + det.bbox.size_x / 2.0   # 水平中心 pixel
            betas.append(math.atan2(uc - self.cam_cx, self.cam_fx))   # 相機方位（右正）
        self._yolo_betas = betas
        self._yolo_t = time.monotonic()

    def _cb_visual(self, msg: MarkerArray) -> None:
        boxes = []
        for mk in msg.markers:
            if not mk.points:
                continue
            xs = [p.x for p in mk.points]
            ys = [p.y for p in mk.points]
            r = min(max(0.25, 0.5 * math.hypot(max(xs) - min(xs), max(ys) - min(ys))),
                    self.static_human_r_max)
            boxes.append((mk.pose.position.x, mk.pose.position.y, r))
        self._visual_boxes = boxes
        self._visual_t = time.monotonic()

    def _static_human_obstacles(self, now: float,
                                dyn: list[Obstacle]) -> list[Obstacle]:
        """visual_bboxes 3D × YOLO person 2D 方位角關聯 → 站著的人當靜態障礙(v=0).

        每個 YOLO person 取「方位角相符且最近」的 visual 框（人遮住其後的牆 → 取最近，
        避免遠處牆同方位被誤標）。與既有動態 track（dyn）太近則跳過（同一人不重複加）。
        yolo/visual 任一逾時或無車姿 → 空（安全退化，不補人源）。座標同 odom frame。
        """
        if (self._robot is None
                or (now - self._yolo_t) > self.static_human_timeout
                or (now - self._visual_t) > self.static_human_timeout
                or not self._yolo_betas or not self._visual_boxes):
            return []
        cy, sy = math.cos(self._robot.yaw), math.sin(self._robot.yaw)
        # 每個 visual 框：車體前向分量、距離、相機系方位（右正=−車體左正）
        feats = []
        for (ox, oy, r) in self._visual_boxes:
            dx, dy = ox - self._robot.x, oy - self._robot.y
            xb = cy * dx + sy * dy               # 前向（>0=車頭前方）
            if xb <= 0.0:
                feats.append(None)               # 側後方 → 相機看不到，不可能是前方的人
                continue
            yb = -sy * dx + cy * dy              # 左正
            feats.append((ox, oy, r, math.hypot(dx, dy), -math.atan2(yb, xb)))
        chosen: list[int] = []
        for beta_cam in self._yolo_betas:
            best_i, best_d = -1, math.inf
            for i, f in enumerate(feats):
                if f is not None and abs(beta_cam - f[4]) <= self.static_human_tol and f[3] < best_d:
                    best_d, best_i = f[3], i
            if best_i >= 0 and best_i not in chosen:
                chosen.append(best_i)
        out: list[Obstacle] = []
        for i in chosen:
            ox, oy, r = feats[i][0], feats[i][1], feats[i][2]
            if any((ox - o.x) ** 2 + (oy - o.y) ** 2 < self.static_human_dedup ** 2
                   for o in dyn):
                continue                          # 同一人已在動態 track → 不重複加
            out.append(Obstacle(x=ox, y=oy, vx=0.0, vy=0.0, r=r))
        return out

    def _front_vo_obstacle_surface_dist(self) -> float | None:
        """前錐(±front_defer_cone)、engage_range 內，最近 VO 追蹤障礙的「表面距離」(車中心↔障礙心 − r)。

        用來判斷 front_m 是不是「VO 看得到的障礙(人)」造成：若某 tracked 障礙的表面距離 ≈ front_m，
        表示前方那顆 VO 有在管 → front-brake 讓開交給 VO 繞/停/退；否則(front_m 明顯更近)是 VO
        看不到的物(牆) → front-brake 自己硬管。無前錐障礙 → None。
        """
        if self._robot is None:
            return None
        sy, cy = math.sin(self._robot.yaw), math.cos(self._robot.yaw)
        rng2 = self.vo.engage_range * self.vo.engage_range
        best = None
        for ob in self._obstacles + self._human_obs:
            dx, dy = ob.x - self._robot.x, ob.y - self._robot.y
            if dx * dx + dy * dy > rng2:
                continue
            xb = cy * dx + sy * dy
            if xb <= 0.0:
                continue
            yb = -sy * dx + cy * dy
            if abs(math.atan2(yb, xb)) > self.front_defer_cone:
                continue
            surf = math.hypot(dx, dy) - ob.r
            if best is None or surf < best:
                best = surf
        return best

    def _cb_strict_toggle(self, msg) -> None:
        self._strict_front = not self._strict_front
        self._cone_age.clear()
        self.get_logger().warn(
            f"VO 嚴格前方模式 → {'ON（只處理前方±%.0f° 卡≥%.0fs 或直衝車端）' % (math.degrees(self.strict_cone), self.strict_stuck_s) if self._strict_front else 'OFF（全向）'}")

    def _strict_front_filter(self, obstacles: list, now: float) -> list:
        """嚴格前方模式：只保留「前方±cone° 且(卡住≥stuck_s 或 直衝車端)」的障礙。
        側/後方、路過橫越的人忽略。卡住=在錐內停滯≥stuck_s；直衝=接近率>closing 且橫向< lateral。"""
        if self._robot is None:
            return obstacles
        sy, cy = math.sin(self._robot.yaw), math.cos(self._robot.yaw)
        kept, seen = [], set()
        for ob in obstacles:
            dx, dy = ob.x - self._robot.x, ob.y - self._robot.y
            xb = cy * dx + sy * dy           # 前向
            if xb <= 0.3:                    # 不明確在前方 → 略過
                continue
            yb = -sy * dx + cy * dy          # 左
            if math.atan2(abs(yb), xb) > self.strict_cone:   # 超出 ±cone°
                continue
            key = (round(ob.x / 0.3), round(ob.y / 0.3))
            seen.add(key)
            rec = self._cone_age.setdefault(key, [now, now])
            rec[1] = now                      # 更新最後見到
            age = now - rec[0]
            # 接近/橫向相對速度（odom 系，2D）
            rng = math.hypot(dx, dy)
            rvx, rvy = ob.vx - self._robot.v, ob.vy - 0.0   # robot.v 是線速度(沿 heading)，近似
            # robot 速度向量（沿 heading）：v·(cos yaw, sin yaw)；較準的做法
            rvx = ob.vx - self._robot.v * cy
            rvy = ob.vy - self._robot.v * sy
            range_rate = (dx * rvx + dy * rvy) / rng        # >0 遠離、<0 接近
            closing = -range_rate
            phx, phy = dx / rng, dy / rng
            lat = math.hypot(rvx - range_rate * phx, rvy - range_rate * phy)
            stuck = (age >= self.strict_stuck_s
                     and math.hypot(ob.vx, ob.vy) < self.strict_stuck_vel)
            approaching = closing > self.strict_closing and lat < self.strict_lateral
            if stuck or approaching:
                kept.append(ob)
        # 清掉離開錐超過 1.5s 的 key
        self._cone_age = {k: v for k, v in self._cone_age.items()
                          if k in seen or (now - v[1]) < 1.5}
        return kept

    def _front_freeze_step(self, f: float, now: float):
        """兩段式前方安全狀態機。回傳 (v', w', 狀態字) 或 None（不介入）.

        第二段（近，優先）f ≤ hardstop_m(0.5)：
          - 持續硬停 (0,0,"stop")，車過不了此距離；
          - 前方持續 ≤hardstop_m 達 dwell_s 秒且後方淨空 → 觸發後退 (escape_reverse_v,0,"reverse")，
            退到 f ≥ trigger(0.7) 或後方不淨空才收。
        第一段（遠）hardstop_m < f ≤ trigger(0.7)：
          - freeze：v/w 全歸 0 停 hold_s(1s) → 交回 RL；f ≥ rearm(0.9) 才能再次觸發（遲滯）。
        ⚠ freeze 交回 RL 後於 0.5~0.7 區間不擋前進；真正的前向硬地板是 0.5m 那段（hardstop）。
        """
        rear_ok = (self._back_m is not None and self._back_m > self.rear_clear_min
                   and not self._rear_person_near())
        # 退到 rearm 外 → 重新武裝 freeze
        if not self._front_freeze_armed and f >= self.front_freeze_rearm_m:
            self._front_freeze_armed = True
        # ── 後退相位（一旦啟動，退到 f≥trigger 或後方不淨空才收）──
        if self._front_hardstop_reversing:
            if f < self.front_freeze_trigger_m and rear_ok:
                return self.escape_reverse_v, 0.0, "reverse"
            self._front_hardstop_reversing = False
            self._front_hardstop_since = None
        # ── 第二段：硬停 + dwell → 後退 ──
        if f <= self.front_hardstop_m:
            if self._front_hardstop_since is None:
                self._front_hardstop_since = now
            if (now - self._front_hardstop_since) >= self.front_hardstop_dwell_s and rear_ok:
                self._front_hardstop_reversing = True
                return self.escape_reverse_v, 0.0, "reverse"
            return 0.0, 0.0, "stop"              # 硬停（dwell 中 或 後方不淨空 → 只停）
        self._front_hardstop_since = None        # 離開 hardstop 區 → 清 dwell
        # ── 第一段：freeze（hardstop_m < f ≤ trigger）──
        if self._front_freeze_since is not None:
            if (now - self._front_freeze_since) < self.front_freeze_hold_s:
                return 0.0, 0.0, "freeze"
            self._front_freeze_since = None      # 時間到 → 交回 RL、解武裝
            self._front_freeze_armed = False
            return None
        if self._front_freeze_armed and f <= self.front_freeze_trigger_m:
            self._front_freeze_since = now       # 觸發 freeze
            return 0.0, 0.0, "freeze"
        return None

    def _apply_front_brake(self, v: float, w: float) -> tuple[float, float, str]:
        """前方 LiDAR 安全煞：吃 RL front_m（含牆任何物）. 回傳 (v', w', 狀態字).

        front_m ≤ stop → 禁止前進（v 鉗到 ≤0）＝必停；stop < front_m ≤ slow → 線性縮前進速度。
        front_brake_reverse=True 時：≤stop **先停住連續 reverse_delay_s（≈2s，同舊 stuck_escape
          的 stuck_timeout）** 且後方夠空 → 才直直倒車避讓（v=escape_reverse_v, w=0），退到
          front_m ≥ release 才收（遲滯防前後抖）；後方被擋/未知 → 不退、退回只停。
        """
        now = time.monotonic()
        sweep_ok = (self._front_m is not None
                    and (now - self._sweep_t) <= self.timeout_goal)
        # ── 定時硬凍結（opt-in，優先於 front_brake，獨立開關）──
        # 近障→v/w 全歸 0 凍結 hold_s→無條件交回 RL；解除後退到 rearm 外才重新武裝。
        if self.front_freeze_enable and sweep_ok:
            fr = self._front_freeze_step(self._front_m, now)
            if fr is not None:
                return fr
        elif not sweep_ok:
            self._front_freeze_since = None
            self._front_hardstop_since = None
            self._front_hardstop_reversing = False
            self._front_freeze_armed = True
        # 逾時（policy status 停發）→ 不亂煞，退回原輸出（VO desired 看門狗會另行處理 policy 掛掉）
        if not self.front_brake_enable or not sweep_ok:
            self._front_reversing = False
            self._front_stop_since = None
            return v, w, ""
        f = self._front_m
        # ── ★倒退鎖定（latch，2026-07-07 修「倒退時左右甩頭」）──
        # 已在倒退中 → 只認 front_m 遲滯（≥release 收）或後方不安全才收手；**繞過下面被
        # front_block_ratio 抖動觸發的「解除」早退**。否則 ratio 一抖 → 早退中斷倒退 → 空檔漏
        # RL 的亂甩 ±ω 出去 → 倒退段左右翻邊。鎖定期間 w 固定為 _escape_w 的 commit 側（≤escape_turn_w，
        # 不會是 RL 的 0.66），commit_until 續命避免中途過期翻側 → 乾淨的單側 K 轉倒退。
        if self.front_brake_reverse and self._front_reversing:
            rear_ok = (self._back_m is not None
                       and self._back_m > self.rear_clear_min + 0.15)
            if f < self.front_release_m and rear_ok:
                self._commit_until = now + self.commit_hold_s   # 倒退中持續鎖側，防 commit 過期翻邊
                return self.escape_reverse_v, self._escape_w(0.0), "reverse"
            self._front_reversing = False   # 到 release 或後方不安全 → 收，落入下面正常判斷
        # ── ★「全身堵滿」才停：±30° 前方扇區填滿率 ≥ thresh 才算「人正對堵死」→ 走停/退。
        #   填滿率低（人偏一側/有縫、或單根柱子）→ 不停，交給 RL/VO 繞過去。
        #   emergency 是絕對底線：任何物(不管填不填滿)到此距離一律停（安全網）。──
        blocked = (self._front_block_ratio is not None
                   and self._front_block_ratio >= self.front_block_ratio_thresh)
        if not blocked and f > self.front_emergency_m:
            self._front_reversing = False
            self._front_stop_since = None
            return v, w, ""      # 沒堵滿 → 不停，讓 RL/VO 繞
        # 直退避讓：≤stop 先停、停住滿 reverse_delay 且後方空才退（仿舊 stuck_escape 的 2s 才退）
        if self.front_brake_reverse:
            rear_ok = self._back_m is not None and self._back_m > self.rear_clear_min + 0.15
            if f > self.front_stop_m:
                self._front_stop_since = None   # 離開停車區 → 清停留計時
            if self._front_reversing:
                if f >= self.front_release_m or not rear_ok:
                    self._front_reversing = False
            elif f <= self.front_stop_m:
                if self._front_stop_since is None:
                    self._front_stop_since = now                  # 進停車區起算停留
                if (now - self._front_stop_since) >= self.front_reverse_delay and rear_ok:
                    if not self._front_reversing:
                        # ★開始退：以「左右哪邊空間多」鎖定繞行側（對齊人工示範 K 轉），
                        #   之後倒退＋前進 go-around 都沿此側連續畫弧，不橫跳。
                        ew0 = self._escape_w(0.0)
                        self._commit_side = 0 if ew0 == 0.0 else (1 if ew0 > 0.0 else -1)
                        self._commit_until = now + self.commit_hold_s
                    self._front_reversing = True                  # 停滿 delay 且後方空 → 開始退
            if self._front_reversing:
                # 朝較開側轉著退（非純直退）；escape_turn_w=0 時退化回直退（w=0）
                return self.escape_reverse_v, self._escape_w(0.0), "reverse"
        if f <= self.front_stop_m:
            return min(v, 0.0), w, "stop"
        if f <= self.front_slow_m and v > 0.0:
            frac = (f - self.front_stop_m) / max(1e-3, self.front_slow_m - self.front_stop_m)
            return v * frac, w, "slow"
        return v, w, ""

    # ── 20Hz 控制迴圈 ──
    def _tick_ctrl(self) -> None:
        now = time.monotonic()

        # 看門狗：RL 沉默 或 odom 掉線 → 立即發 0（fail-safe，不 slew，讓底盤盡快煞）
        if self._robot is None or (now - self._odom_t) > self.timeout_odom:
            self._publish_hard_zero()
            self._publish_status(fail="odom")
            return
        if (now - self._des_t) > self.timeout_desired:
            self._publish_hard_zero()
            self._publish_status(fail="desired")
            return

        # 障礙快取過期 → 視為無障礙（靜態仍由 RL sweep 擋）
        obs_stale = (now - self._obs_t) > self.timeout_obs
        obstacles = [] if obs_stale else list(self._obstacles)
        # 靜止人源（只滿血版）：YOLO×visual 關聯出的站著的人，當 v=0 障礙併入
        if self.static_human_enable:
            self._human_obs = self._static_human_obstacles(now, obstacles)
            obstacles = obstacles + self._human_obs
        else:
            self._human_obs = []
        # 嚴格前方模式（TUI 'f' 切換）：只留前方±cone° 且(卡≥stuck_s 或直衝車端)的障礙
        if self._strict_front:
            obstacles = self._strict_front_filter(obstacles, now)

        goal = self._goal_in_odom(now)   # odom-frame goal（None=無/逾時，退回純貼近 RL）
        self._goal_active = goal is not None and self.vo.w_goal > 0.0
        # commit 過期清零（過障礙後 engaged 不再續命 → 自然解鎖；見下方 escape 段續命）
        if self._commit_side != 0 and now > self._commit_until:
            self._commit_side = 0
        res = compute_safe_cmd(self._des_v, self._des_w, self._robot, obstacles,
                               self.vo, goal=goal, commit_side=self._commit_side)

        # 介入 = 有近障且 VO 解明顯偏離 RL 意圖（與下方 status/log 同一判據）
        intervening = res.engaged and (
            abs(res.v - self._des_v) > 0.05 or abs(res.w - self._des_w) > 0.1)

        # 介入時節流 log，方便現場觀察 VO 何時動作
        if res.blocked:
            self.get_logger().warn(
                f"VO 全堵死 → 停車（近障 {res.n_obstacles}）", throttle_duration_sec=1.0)
        elif intervening:
            self.get_logger().info(
                f"VO 介入：RL({self._des_v:+.2f},{self._des_w:+.2f}) → "
                f"目標({res.v:+.2f},{res.w:+.2f}) 近障{res.n_obstacles} 可行{res.n_feasible}",
                throttle_duration_sec=0.5)

        # 卡死逃脫（opt-in）：blocked+停住超過 stuck_timeout 且後方有空間 → 倒車+轉向「對準開闊側」，
        # 仿真人 K 轉：①還堵死→倒車開空間 ②有前進解但鼻子沒對準→停止後退、原地轉對準
        # ③鼻子對準(straight_feasible，接近直行就能前進)→退出，go-around 前進接手。
        # 後方淨空(back_m)掉到門檻下立刻中止倒車（防倒退撞牆，安全優先）；對準階段不倒車則不受此限。
        # ★另補：後方有 VO 追蹤到的人(rear_person)→ 一律不准退（觸發前擋、退到一半也中止）。
        rear_person = self._rear_person_near()
        rear_ok = (self._back_m is not None and self._back_m > self.rear_clear_min
                   and not rear_person)
        # 觸發門檻 rear_clear_start（純 LiDAR，牆/人都算）獨立於中止門檻 rear_clear_min：
        # 後方 back_m 需 > rear_clear_start 才敢啟動後退（後方 1.2m 內有任何東西就不退）；
        # start > min 天然提供遲滯，避免在門檻邊界抖動→退一拍就中止（實測 bug）。
        rear_ok_start = (self._back_m is not None
                         and self._back_m > self.rear_clear_start
                         and not rear_person)
        if self.stuck_escape_enable:
            if self._escaping:
                # 已後退距離（從起點 odom 量）
                rev_dist = math.hypot(self._robot.x - self._escape_start_xy[0],
                                      self._robot.y - self._escape_start_xy[1])
                if res.straight_feasible and self._forward_cone_clear():
                    # 鼻子已對準開闊側（接近直行可前進 + 前向錐內確實無近障）→ 退出，go-around 前進接手。
                    # ⚠ 兩條缺一不可：straight_feasible 只保證「往前 horizon 內不撞」，但倒車拉開距離後
                    #   它會自動成立（跟鼻子朝哪無關）→ 曾害車退遠後仍鼻子朝障礙就直行撞上（賭注的障礙）。
                    #   _forward_cone_clear 補「鼻子真的指向空檔」這一條，與退多遠解耦。
                    self._escaping = False
                    self._stuck_since = None
                elif (now - self._escape_start_t) >= self.escape_aim_timeout:
                    # 倒車+轉太久仍對不準開闊側 → 放棄、停著等障礙離開（防原地一直打轉/無限後退）
                    self._escaping = False
                    self._stuck_since = None
                    self._escape_exhausted = True
                    self.get_logger().warn(
                        f"VO 逃脫：{self.escape_aim_timeout:.0f}s 仍無法對準開闊側 → 放棄、停車等障礙離開")
                elif res.blocked and not rear_ok:    # 還需倒車但後方被擋/未知 → 中止 → 停
                    self._escaping = False
                    self._stuck_since = None
                    bm_str = "未知" if self._back_m is None else f"{self._back_m:.1f}m"
                    reason = ("後方偵測到人" if rear_person
                              else f"後方淨空 {bm_str}≤{self.rear_clear_min:.1f}m（接近牆/淨空未知）")
                    self.get_logger().warn(f"VO 逃脫中止：{reason} → 停車")
                elif res.blocked and rev_dist >= self.escape_max_dist:  # 退滿上限仍堵 → 放棄
                    self._escaping = False
                    self._stuck_since = None
                    self._escape_exhausted = True
                    self.get_logger().warn(
                        f"VO 逃脫：已後退 {rev_dist:.1f}m 仍堵死 → 放棄、停車等障礙離開")
            elif res.blocked and abs(self._out_v) < 0.05:
                if self._stuck_since is None:
                    self._stuck_since = now
                if ((now - self._stuck_since) >= self.stuck_timeout_s and rear_ok_start
                        and not self._escape_exhausted):
                    self._escaping = True
                    self._escape_start_xy = (self._robot.x, self._robot.y)
                    self._escape_start_t = now
                    # 鎖定繞行側：以 escape 當下選的轉向（跟 rollout 提示同源）為準，後續 going-around
                    # 沿用此側不橫跳（解「往右退→又往左轉」）。直退(ew=0)不鎖側。
                    ew = self._escape_w(res.best_turn_w)
                    self._commit_side = 0 if ew == 0.0 else (1 if ew > 0.0 else -1)
                    self.get_logger().warn(
                        f"VO 卡死逃脫：堵死 {self.stuck_timeout_s:.0f}s 且後方淨空 "
                        f"{self._back_m:.1f}m → 倒車+轉向對準開闊側（退≤{self.escape_max_dist:.1f}m）再繞")
            else:
                self._stuck_since = None
            # 前方一旦有解（障礙離開）→ 重置「放棄」狀態，允許日後再逃脫
            if not res.blocked:
                self._escape_exhausted = False

        # commit 續命：只要障礙仍 engaged（還在繞同一顆）就持續鎖側；障礙離開後 commit_hold_s
        # 過期 → _commit_side 自動清（見上方過期判斷）→ 解鎖、恢復正常選邊。
        if self._commit_side != 0 and res.engaged:
            self._commit_until = now + self.commit_hold_s

        # 輸出端 slew 限速：候選窗較寬，靠這裡把輸出每拍限在 accel×ctrl_dt 內，
        # 保證實際送底盤的 cmd 平滑可達（不會因 VO 切解而跳階）。
        if self._escaping:
            # 朝開闊側轉，把車頭對準到「接近直行就能前進」(straight_feasible) 才退出，仿真人
            # 「倒車打方向到對準才走」。倒車只在「還堵死(沒前進解)」時用來開空間；一旦有前進解但鼻子
            # 還沒對準，就停止後退、原地轉繼續對準（不必再往後吃空間）。
            # ⚠ 每拍把 _commit_side 同步成「實際轉向」：若原 commit 側中途被牆擋、_escape_w() 改轉
            # 另一側，commit 也跟著翻 → 避免「實際轉右、going-around 仍鎖左」的方向矛盾。
            ew = self._escape_w(res.best_turn_w)
            if ew != 0.0:
                self._commit_side = 1 if ew > 0.0 else -1
            tv, tw = (self.escape_reverse_v if res.blocked else 0.0), ew
        else:
            tv, tw = res.v, res.w
        # 前方 LiDAR 安全煞（只預設版）：吃 RL front_m，≤stop 禁前進、≤slow 減速（含牆任何物）。
        # 放最後、蓋過 VO/逃脫輸出 → 距離硬底線，不論障礙是人是牆一律生效。
        tv, tw, self._front_brake = self._apply_front_brake(tv, tw)
        # 前方煞狀態變化 → log（供盯 log 觀察；純觀察不影響控制）
        if self._front_brake != self._front_brake_prev:
            fm = "?" if self._front_m is None else f"{self._front_m:.2f}"
            if self._front_brake:
                self.get_logger().warn(f"前方煞 → {self._front_brake}（front_m={fm}m）")
            else:
                self.get_logger().info(f"前方煞 → 解除/讓VO繞（front_m={fm}m）")
            self._front_brake_prev = self._front_brake
        if self._front_brake in ("stop", "reverse", "freeze"):
            self._publish_brake(tv, tw)   # 安全煞：前進速度瞬間歸零（不 slew 慢收→不多衝 ~0.1m）
        else:
            self._publish_slew(tv, tw)
        self._publish_status(res=res, intervening=intervening, obs_stale=obs_stale)

    def _forward_cone_clear(self) -> bool:
        """前向 ±escape_aim_cone 錐內、engage_range 內是否無近障（鼻子是否真的對準空檔）.

        逃脫退出的對準閘：straight_feasible 只保證「往前 horizon 不撞」，倒車拉開距離後恆成立、
        與鼻子朝向脫鉤 → 需另補「前錐內確實沒障礙」才算真正對準開闊側，避免退遠後仍朝障礙直行。
        用 self._obstacles（不吃 vel_confidence 歸零，位置全採計）；無車姿/無障礙 → 視為對準(True)。
        """
        if self._robot is None:
            return True
        sy = math.sin(self._robot.yaw)
        cy = math.cos(self._robot.yaw)
        rng2 = self.vo.engage_range * self.vo.engage_range
        for ob in self._obstacles + self._human_obs:
            dx = ob.x - self._robot.x
            dy = ob.y - self._robot.y
            if dx * dx + dy * dy > rng2:
                continue
            x_body = cy * dx + sy * dy   # 前向分量（>0=車頭前方）
            if x_body <= 0.0:
                continue                 # 障礙在車身側後方 → 不擋前行
            y_body = -sy * dx + cy * dy  # 左正
            if abs(math.atan2(y_body, x_body)) <= self.escape_aim_cone:
                return False             # 前錐內有近障 → 鼻子尚未對準空檔
        return True

    def _rear_person_near(self) -> bool:
        """後方半平面 escape_rear_person_range 內是否有「VO 追蹤到的障礙(人)」→ 有則禁止後退.

        補足 back_m(LiDAR)只在 rear_clear_min≈0.85m 內才擋的盲區：人在 0.85m~range 之間時
        LiDAR back_m 放行、但車會退向他。用 tracked 障礙(self._obstacles + _human_obs，來自
        LV-DOT 動態人)在後向(車身 x_body<0)判定；靜態牆不在 tracked、仍由 back_m 管。
        無車姿/未啟用 → 視為無人(False，不擋)。
        """
        if self._robot is None or not self.escape_block_if_rear_person:
            return False
        sy = math.sin(self._robot.yaw)
        cy = math.cos(self._robot.yaw)
        rng2 = self.escape_rear_person_range * self.escape_rear_person_range
        for ob in self._obstacles + self._human_obs:
            dx = ob.x - self._robot.x
            dy = ob.y - self._robot.y
            if dx * dx + dy * dy > rng2:
                continue
            x_body = cy * dx + sy * dy   # 前向分量（<0=車身後方）
            if x_body < 0.0:
                return True              # 後方半平面 range 內有追蹤到的人 → 擋退
        return False

    def _escape_w(self, hint_w: float = 0.0) -> float:
        """後退逃脫時的轉向角速度（方向跟 rollout 同源 + 牆安全閘）.

        想轉哪側的優先序（★2026-07-07 改：對齊人工示範「往較開側 K 轉」，開側升為主依據）：
          1. **已 commit 側**（escape 一開始鎖定後就沿用，最高優先）——避免鎖定後左右橫跳
             （「往右退卻甩頭到左」）。鎖定後只認 commit，唯牆安全閘可翻邊。
          2. **LiDAR 較開側 left_m/right_m（未鎖定時的主要依據）**：哪邊空間多就往哪邊 K 轉，
             與人工示範一致（第一次往右、第二次往左，都朝當下較開側）。
          3. VO rollout 提示 best_turn_w（僅左右淨空都未知時墊底）。
        ⚠ VO 看不到靜態牆 → left_m/right_m 只當**安全閘**：欲轉側淨空 ≤ min_side（近牆）→ 不
          倒車轉進牆，改另一開側；兩側都窄 → 直直後退（純退讓）。
        left_m=+y/左→w>0(CCW)；right_m=−y/右→w<0。escape_turn_w=0 → 0（直退）。
        """
        if self.escape_turn_w <= 0.0:
            return 0.0
        lm = self._left_m
        rm = self._right_m
        # 牆安全閘：None（無資訊）→ 視為「不確定」，後面用提示但仍可轉（rollout 已把關動態障礙）
        left_open = lm is None or lm > self.escape_turn_min_side
        right_open = rm is None or rm > self.escape_turn_min_side
        if not left_open and not right_open:
            return 0.0   # 兩側都確定窄 → 直直退（純退讓）
        # 想轉方向：commit（鎖定側，最高優先）> 較開側（主依據，對齊人工示範）> rollout 提示（墊底）
        if self._commit_side != 0:
            want_left = self._commit_side > 0
        elif lm is not None and rm is not None:
            want_left = lm >= rm       # ★哪邊空間多往哪轉（left_m/right_m 較開側）
        elif abs(hint_w) > 1e-3:
            want_left = hint_w > 0.0    # 左右淨空未知才退回 rollout 提示
        else:
            return 0.0   # 無 commit、無左右資訊、無提示 → 不瞎轉，直退
        # 牆安全閘：想轉側近牆 → 改另一開側；另一側也窄 → 直退
        if want_left:
            if left_open:
                return self.escape_turn_w
            return -self.escape_turn_w if right_open else 0.0
        if right_open:
            return -self.escape_turn_w
        return self.escape_turn_w if left_open else 0.0

    def _lin_max_dv(self, target_v: float) -> float:
        """本拍線速度可變化量上限：★正在「建立倒退速度」(往更負走且目標<0)時用溫柔 escape_reverse_accel
        （對齊人工示範慢慢加速）；其餘（前進、從倒退收回 0）用底盤 accel_v。"""
        if target_v < self._out_v and target_v < 0.0:
            return self.escape_reverse_accel * self.vo.ctrl_dt
        return self.vo.accel_v * self.vo.ctrl_dt

    def _publish_slew(self, target_v: float, target_w: float) -> None:
        max_dv = self._lin_max_dv(target_v)
        max_dw = self.vo.accel_w * self.vo.ctrl_dt
        self._out_v += _clamp(target_v - self._out_v, -max_dv, max_dv)
        self._out_w += _clamp(target_w - self._out_w, -max_dw, max_dw)
        self._emit(self._out_v, self._out_w)

    def _publish_brake(self, target_v: float, target_w: float) -> None:
        """安全煞專用：前進速度**瞬間歸零**（不走 accel slew 慢收，那會多前衝 ~0.04–0.1m→撞）。
        反向（倒車避讓）用溫柔 escape_reverse_accel 慢慢建立（對齊人工示範，非猛倒）；轉向照常 slew。
        """
        max_dv = self._lin_max_dv(target_v)
        max_dw = self.vo.accel_w * self.vo.ctrl_dt
        if self._out_v > 0.0:
            self._out_v = 0.0                     # 立即殺前進（不創造前衝空間）
        # 反向目標（倒車）從 0 溫柔往負建立；停車目標=0 則維持 0
        self._out_v += _clamp(target_v - self._out_v, -max_dv, max_dv)
        self._out_w += _clamp(target_w - self._out_w, -max_dw, max_dw)
        self._emit(self._out_v, self._out_w)

    def _publish_hard_zero(self) -> None:
        self._out_v = 0.0
        self._out_w = 0.0
        self._escaping = False       # 看門狗發 0 → 取消逃脫
        self._stuck_since = None
        self._escape_exhausted = False
        self._commit_side = 0        # 看門狗發 0 → 解鎖 commit（重新開始）
        self._emit(0.0, 0.0)

    def _emit(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub_cmd.publish(msg)

    # ── 精簡狀態 JSON（供 status_tui；節流到 ~5Hz）──
    def _publish_status(self, res=None, intervening: bool = False,
                        obs_stale: bool = False, fail: str = "") -> None:
        self._status_tick += 1
        if self._status_tick % self._status_every:
            return
        d = {
            "fail": fail,                       # ""=正常；"odom"/"desired"=看門狗發 0
            "obs_stale": bool(obs_stale),       # 障礙源逾時 → VO 退化為放行
            "des_v": round(self._des_v, 3),     # RL 期望（VO 輸入）
            "des_w": round(self._des_w, 3),
            "out_v": round(self._out_v, 3),     # 實際送 mux 的 slew 後輸出
            "out_w": round(self._out_w, 3),
            "n_tracked": len(self._obstacles),  # 收到的追蹤障礙總數（engage 過濾前）
            # 參數狀況（供 TUI「VO參數」列）
            "w_max": self.vo.w_max, "horizon": self.vo.horizon,
            "engage_range": self.vo.engage_range, "margin": self.vo.margin,
            "window_time": self.vo.window_time,
            # goal 導向項：w_goal>0 且本拍有有效 goal → 繞行方向由 goal 主導
            "w_goal": self.vo.w_goal, "goal_active": bool(self._goal_active),
            # 卡死逃脫：是否正在後退 + 後方淨空（供 TUI/debug）
            "escaping": bool(self._escaping),
            "commit_side": int(self._commit_side),   # 鎖定繞行側：+1左/-1右/0未鎖
            "left_m": (None if self._left_m is None else round(self._left_m, 2)),
            "right_m": (None if self._right_m is None else round(self._right_m, 2)),
            "back_m": (None if self._back_m is None else round(self._back_m, 2)),
            # 前方安全煞（只預設版）+ 靜止人源（只滿血版）
            "front_m": (None if self._front_m is None else round(self._front_m, 2)),
            "front_brake": self._front_brake,   # ""/"slow"/"stop"/"reverse"/"freeze"
            "n_human": len(self._human_obs),    # 本拍補的靜止人源障礙數
            "strict_front": bool(self._strict_front),  # 嚴格前方模式（TUI 'f' 切換）
        }
        if res is not None:
            d.update({
                "engaged": bool(res.engaged),       # 近障進入 engage_range
                "blocked": bool(res.blocked),       # 全堵死 → 停車
                "intervening": bool(intervening),   # VO 解明顯偏離 RL 意圖
                "n_obs": int(res.n_obstacles),      # engage_range 內障礙數
                "n_feasible": int(res.n_feasible),
                "vo_v": round(res.v, 3), "vo_w": round(res.w, 3),  # VO 選定目標
                "best_turn_w": round(res.best_turn_w, 3),  # rollout 建議轉向（逃脫選邊用）
                "straight_feasible": bool(res.straight_feasible),  # 鼻子是否已對準（接近直行可前進）
                "min_ttc": (None if math.isinf(res.min_ttc) else round(res.min_ttc, 2)),
            })
        else:   # 看門狗 hard-zero：未跑 VO
            d.update({"engaged": False, "blocked": False, "intervening": False,
                      "n_obs": 0, "n_feasible": 0,
                      "vo_v": 0.0, "vo_w": 0.0, "min_ttc": None})
        self.pub_status.publish(String(data=json.dumps(d)))


def main() -> None:
    rclpy.init()
    node = VOSafetyNode()
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
