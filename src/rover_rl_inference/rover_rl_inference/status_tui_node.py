"""
status_tui_node — rover_rl 即時狀態儀表板（繁體中文 TUI）

目的：deploy_full 一次拉起十幾個節點，log 全擠在同一個 terminal 難以觀察。
此節點獨立一個 terminal 跑，訂閱 policy_node 發布的精簡狀態（/rover_rl_policy/status,
std_msgs/String 內含 JSON），用 curses 畫成乾淨的方框儀表板即時刷新。

特性：
  - 純訂閱、純渲染，不影響推論、不依賴 policy_node 內部
  - 顏色標記健康度（綠=正常 / 黃=注意 / 紅=危險/estop）
  - 收不到狀態時顯示「等待 policy_node…」

啟動：
  ros2 run rover_rl_inference status_tui
  ros2 run rover_rl_inference status_tui --ros-args -p topic_status:=/rover_rl_policy/status
  按 q 離開
"""
from __future__ import annotations

import curses
import json
import locale
import math
import threading
import time
import unicodedata

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Float32MultiArray, String
from visualization_msgs.msg import MarkerArray

# 到達判定距離（m）：與 diag_logger 的 auto_stop_goal_tol 對齊，goal_dist ≤ 此值即算到達
GOAL_REACHED_DIST_M = 0.6

# braille 點字 2×4 子格 → bit 對照（U+2800 基底）。
# 一個 braille 字元 = 2 欄 × 4 列共 8 點，可在純文字終端達到 4× 縱向解析度，
# 用來在右側畫極座標 LiDAR 雷達（不必把 matplotlib BEV 影像塞進 curses）。
BRAILLE_DOTS = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}

MODE_LABEL = {
    "nav": "NAV 自動導航",
    "idle": "IDLE 待命",
    "estop": "ESTOP 緊急停車",
    "manual": "MANUAL 外部接管",
    "paused": "PAUSED 暫停推論",
}


class StatusTuiNode(Node):
    """純訂閱端：把收到的狀態暫存在記憶體，渲染層用 snapshot() 取最新一份。

    用 _lock 是因為 rclpy.spin() 跑在背景執行緒（見 main），與 curses 渲染
    執行緒並行讀寫 self._status，需互斥避免讀到半更新的 dict。
    """

    def __init__(self) -> None:
        super().__init__("rover_rl_status_tui")
        self.declare_parameter("topic_status", "/rover_rl_policy/status")
        self.declare_parameter("topic_dynamic_bboxes", "/onboard_detector/dynamic_bboxes")
        self.declare_parameter("topic_diag_event", "/rover_rl/diag_event")
        self.declare_parameter("topic_route_stations", "/rover_rl/route_stations")
        # VO 安全層狀態（enable_vo:=true 才有此 topic；收不到→自動隱藏 VO 列）
        self.declare_parameter("topic_vo_status", "/vo_safety_node/status")
        # 右側 LiDAR 雷達（重畫 BEV 的極座標散點，不吃 bev_image 影像）
        self.declare_parameter("topic_sweep", "/rover_rl/lidar_sweep_72")
        # 往返測試（enable_pingpong:=true 才有此 topic；收不到→不顯示提示）：
        # 就緒時 TUI 跳提示，按空白鍵發 start 觸發開始往返
        self.declare_parameter("topic_pingpong_status", "/rover_rl/pingpong/status")
        self.declare_parameter("topic_pingpong_start", "/rover_rl/pingpong/start")
        self.declare_parameter("topic_pingpong_auto", "/rover_rl/pingpong/auto_toggle")
        self.declare_parameter("topic_pingpong_stop", "/rover_rl/pingpong/stop")
        self.declare_parameter("enable_radar", True)
        self.declare_parameter("radar_style", "braille")   # braille / dots，可熱切換
        self.declare_parameter("radar_range_m", 5.0)       # 雷達顯示半徑（>此距離貼邊）
        self.declare_parameter("r_max", 20.0)              # 與 preprocessor 正規化對齊
        self.declare_parameter("r_robot", 0.35)
        # 車頭前伸量：LiDAR 感測器到車頭最前緣的距離(m)。front_m 是「感測器→障礙」，
        # 扣掉此量才是「車頭→障礙」的真實餘裕。實測：撞牆瞬間 front_m≈0.5~0.6 → 預設 0.5
        self.declare_parameter("front_overhang_m", 0.5)
        # 碰撞/過近警示：前後左右任一方向感測器距離 < 此值 → 邊框粗閃紅
        self.declare_parameter("collision_warn_m", 0.5)
        topic = self.get_parameter("topic_status").get_parameter_value().string_value
        topic_dyn = self.get_parameter(
            "topic_dynamic_bboxes").get_parameter_value().string_value
        topic_diag = self.get_parameter("topic_diag_event").get_parameter_value().string_value
        topic_stations = self.get_parameter(
            "topic_route_stations").get_parameter_value().string_value
        topic_vo = self.get_parameter("topic_vo_status").get_parameter_value().string_value
        topic_sweep = self.get_parameter("topic_sweep").get_parameter_value().string_value
        topic_pp = self.get_parameter(
            "topic_pingpong_status").get_parameter_value().string_value
        topic_pp_start = self.get_parameter(
            "topic_pingpong_start").get_parameter_value().string_value
        topic_pp_auto = self.get_parameter(
            "topic_pingpong_auto").get_parameter_value().string_value
        topic_pp_stop = self.get_parameter(
            "topic_pingpong_stop").get_parameter_value().string_value
        self._radar_range = float(self.get_parameter("radar_range_m").value)
        self._r_max = float(self.get_parameter("r_max").value)
        self._r_robot = float(self.get_parameter("r_robot").value)
        self._front_overhang = float(self.get_parameter("front_overhang_m").value)
        self._collision_warn = float(self.get_parameter("collision_warn_m").value)
        self._lock = threading.Lock()
        self._status: dict | None = None
        self._last_t = 0.0
        self._lvdot_n: int | None = None     # 動態障礙框數；None=從未收到
        self._lvdot_t = 0.0
        self._diag_event: dict | None = None  # goal_reached 等事件
        self._diag_event_t = 0.0
        self._stations: dict | None = None    # routing 具名站序 {stations, wp_idx, n_wp}
        self._stations_t = 0.0
        self._vo: dict | None = None           # VO 安全層狀態；None=VO 未啟動/沒收到
        self._vo_t = 0.0
        self._sweep: list | None = None        # 72-bin 正規化 sweep（畫雷達用）
        self._sweep_t = 0.0
        self._pp: dict | None = None            # 往返測試狀態；None=未啟動/沒收到
        self._pp_t = 0.0
        self.create_subscription(String, topic, self._cb_status, 10)
        self.create_subscription(MarkerArray, topic_dyn, self._cb_dyn, 10)
        self.create_subscription(String, topic_diag, self._cb_diag_event, 10)
        self.create_subscription(String, topic_stations, self._cb_stations, 10)
        self.create_subscription(String, topic_vo, self._cb_vo, 10)
        self.create_subscription(Float32MultiArray, topic_sweep, self._cb_sweep, 10)
        self.create_subscription(String, topic_pp, self._cb_pp, 10)
        # 空白鍵 → 發 start 觸發往返測試開始
        self.pub_pp_start = self.create_publisher(Empty, topic_pp_start, 5)
        # 'a' 鍵 → 熱切換全自動模式
        self.pub_pp_auto = self.create_publisher(Empty, topic_pp_auto, 5)
        # 's' 鍵 → 主動中斷往返（running 中回待命、停車）
        self.pub_pp_stop = self.create_publisher(Empty, topic_pp_stop, 5)
        self._topic = topic

    def _cb_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._status = data
            self._last_t = time.monotonic()

    def _cb_dyn(self, msg: MarkerArray) -> None:
        # LV-DOT 偵測器用 MarkerArray 表示每個動態障礙框；marker.action==ADD(0)
        # 才是當前 frame 有效的框（DELETE/DELETEALL 不算），故只數 action==0。
        n = sum(1 for m in msg.markers if m.action == 0)
        with self._lock:
            self._lvdot_n = n
            self._lvdot_t = time.monotonic()

    def _cb_diag_event(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._diag_event = data
            self._diag_event_t = time.monotonic()

    def _cb_stations(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._stations = data
            self._stations_t = time.monotonic()

    def _cb_vo(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._vo = data
            self._vo_t = time.monotonic()

    def _cb_sweep(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self._sweep = list(msg.data)
            self._sweep_t = time.monotonic()

    def _cb_pp(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._pp = data
            self._pp_t = time.monotonic()

    def pingpong_ready(self) -> bool:
        """往返節點是否處於 ready（停在點上等空白鍵）且狀態新鮮（<2s）。"""
        with self._lock:
            if self._pp is None or (time.monotonic() - self._pp_t) > 2.0:
                return False
            return self._pp.get("state") == "ready"

    def request_pingpong_start(self) -> None:
        self.pub_pp_start.publish(Empty())

    def pingpong_active(self) -> bool:
        """往返節點是否在跑（狀態新鮮 <2s）——決定 'a' 鍵是否作用。"""
        with self._lock:
            return self._pp is not None and (time.monotonic() - self._pp_t) <= 2.0

    def toggle_pingpong_auto(self) -> None:
        self.pub_pp_auto.publish(Empty())

    def pingpong_running(self) -> bool:
        """往返節點是否在跑往返（running，狀態新鮮 <2s）——決定 's' 鍵是否作用。"""
        with self._lock:
            if self._pp is None or (time.monotonic() - self._pp_t) > 2.0:
                return False
            return self._pp.get("state") == "running"

    def request_pingpong_stop(self) -> None:
        self.pub_pp_stop.publish(Empty())

    def snapshot(self) -> tuple:
        with self._lock:
            return (self._status, self._last_t, self._lvdot_n, self._lvdot_t,
                    self._diag_event, self._diag_event_t,
                    self._stations, self._stations_t,
                    self._sweep, self._sweep_t,
                    self._vo, self._vo_t,
                    self._pp, self._pp_t)


def _fmt(val, fmt: str, default: str = "—") -> str:
    if val is None:
        return default
    try:
        return format(val, fmt)
    except (ValueError, TypeError):
        return str(val)


class Dashboard:
    """curses 渲染層；color pair: 1=綠 2=黃 3=紅 4=青 5=灰。"""

    def __init__(self, node: StatusTuiNode) -> None:
        self.node = node

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_WHITE, -1)

    def _c(self, pair: int) -> int:
        return curses.color_pair(pair) if curses.has_colors() else 0

    def run(self, stdscr) -> None:
        # 由 curses.wrapper 呼叫，stdscr 是已初始化的真實終端機畫面。
        # nodelay+timeout：getch 非阻塞、最多等 150ms，讓迴圈能定時重繪
        # 而不會卡在等鍵盤輸入（否則畫面不會自動刷新）。
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(150)
        self._init_colors()
        while rclpy.ok():
            try:
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q"), 27):  # q / ESC
                    break
                # 空白鍵：往返測試就緒時 → 發 start 觸發開始（其餘狀態忽略）
                if ch == ord(" ") and self.node.pingpong_ready():
                    self.node.request_pingpong_start()
                # 'a' 鍵：往返測試在跑時 → 熱切換全自動模式（停穩自動出發，免按空白）
                if ch in (ord("a"), ord("A")) and self.node.pingpong_active():
                    self.node.toggle_pingpong_auto()
                # 's' 鍵：往返測試 running 時 → 主動中斷（停車、回待命）
                if ch in (ord("s"), ord("S")) and self.node.pingpong_running():
                    self.node.request_pingpong_stop()
                self._render(stdscr)
            except curses.error:
                pass
            time.sleep(0.1)

    def _disp_w(self, text: str) -> int:
        """估算顯示寬度：CJK 與全形符號算 2 格。"""
        w = 0
        for ch in text:
            w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        return w

    @staticmethod
    def _current_station(stations: dict | None, path_i) -> tuple[list, int] | None:
        """由 path_i（policy 發的最近 waypoint index）對應目前在第幾站.

        stations.wp_idx[k] = 第 k 站的起始 waypoint index；目前站 = 最後一個
        wp_idx <= path_i 的站。回傳 (站名list, 目前index)；資料不足回傳 None。
        """
        if not stations:
            return None
        names = stations.get("stations") or []
        wp = stations.get("wp_idx") or []
        if len(names) < 2 or len(wp) != len(names):
            return None
        cur = 0
        if path_i is not None:
            for k, widx in enumerate(wp):
                if widx <= path_i:
                    cur = k
                else:
                    break
        return names, max(0, min(cur, len(names) - 1))

    def _render(self, stdscr) -> None:
        (status, last_t, lvdot_n, lvdot_t, diag_event, diag_event_t,
         stations, stations_t, sweep, sweep_t, vo, vo_t,
         pp, pp_t) = self.node.snapshot()
        now = time.monotonic()
        # age = 距上次收到狀態多久。policy_node 以 5 Hz 發，>1.5s 沒更新即視為
        # stale（policy_node 可能掛了），改顯示警告而非沿用過期數值誤導判讀。
        age = now - last_t if last_t else float("inf")
        lvdot_age = now - lvdot_t if lvdot_t else float("inf")
        vo_age = now - vo_t if vo_t else float("inf")
        stale = age > 1.5

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        box_w = min(max(52, w - 2), 70)

        # 先依狀態決定列數與是否畫地鐵式進度條（路徑導航才畫）→ 才能算框高
        station_info = None
        show_route = show_subway = False
        if status is not None and not stale:
            is_path = status.get("nav_type") == "path"
            pn = status.get("path_n")
            stations_fresh = stations_t and (now - stations_t) < 5.0
            if is_path and stations_fresh:
                station_info = self._current_station(stations, status.get("path_i"))
            # 到達目標後讓「目標」列改顯示黃字「已到達並已儲存diag」。判定二擇一：
            #   ① goal_dist ≤ 0.6m（與 diag auto_stop_goal_tol 對齊，即時反映）
            #   ② 收到 goal_reached event 後 10s 內（diag 確實已存檔的權威來源）
            # 新 goal 進來、距離又拉開即自動回復正常距離/方位顯示
            gd = status.get("goal_dist")
            goal_reached = (
                (gd is not None and gd <= GOAL_REACHED_DIST_M)
                or (diag_event is not None
                    and diag_event.get("event") == "goal_reached"
                    and (now - diag_event_t) < 10.0))
            rows = self._build_rows(status, lvdot_n, lvdot_age, station_info,
                                    goal_reached, vo, vo_age)
            # 有具名站序 → 畫站名地鐵線；否則退回幾何進度條（起→O→終）
            show_route = station_info is not None
            show_subway = (not show_route and is_path and pn and pn >= 2
                           and status.get("path_i") is not None)
        else:
            rows = []

        n = len(rows)
        line_y = 1 + n + 1             # rows 後空一列再畫站線/進度條
        show_line = show_route or show_subway
        box_h = (line_y + 2) if show_line else max(18, n + 3)
        # 框高超出終端機 → 先犧牲站線再縮；仍放不下才提示放大
        if show_line and h < box_h + 1:
            show_route = show_subway = show_line = False
            box_h = max(18, n + 3)
        if h < box_h + 1 or w < box_w:
            stdscr.addstr(0, 0, "終端機視窗太小，請放大…")
            stdscr.refresh()
            return

        # 障礙過近：前後左右任一方向感測器距離 < collision_warn_m → 邊框粗閃紅示警。
        # int(now*2)%2 以 2Hz 交替紅/正常，配合 ~150ms 刷新產生閃爍。
        collided = False
        if status is not None and not stale:
            warn = self.node._collision_warn
            collided = any(
                (d is not None and d < warn)
                for d in (status.get("front_m"), status.get("back_m"),
                          status.get("left_m"), status.get("right_m")))

        win = stdscr.derwin(box_h, box_w, 0, 0)
        if collided and int(now * 2) % 2 == 0:
            win.attron(self._c(3) | curses.A_BOLD)   # 粗紅框
            win.box()
            win.attroff(self._c(3) | curses.A_BOLD)
        else:
            win.box()
        title = "⚠ 障礙過近 ⚠" if collided else " rover_rl 即時狀態 "
        win.addstr(0, max(2, (box_w - self._disp_w(title)) // 2), title,
                   self._c(4) | curses.A_BOLD)

        if status is None or stale:
            msg = "⚠ 等待 policy_node 狀態…" if status is None else "⚠ 狀態逾時，policy_node 可能已停"
            win.addstr(box_h // 2, 3, msg, self._c(2) | curses.A_BOLD)
            win.addstr(box_h - 1, 2, " 按 q 離開 ", self._c(5))
            win.refresh()
            # 雷達只吃 sweep，與 policy 狀態解耦：狀態逾時仍可看 LiDAR
            self._maybe_draw_radar(stdscr, h, w, box_w, box_h,
                                   sweep, sweep_t, None, now)
            # 往返測試提示與 policy 狀態解耦，逾時時仍顯示
            if pp is not None and (now - pp_t) < 2.0:
                self._draw_pingpong(stdscr, h, w, pp)
            return

        val_col = 11  # 標籤欄 4 全形字 ≈ 8 格 + 邊距
        for i, (label, value, pair, bold) in enumerate(rows):
            y = 1 + i
            win.addstr(y, 2, label, self._c(5))
            attr = self._c(pair) | (curses.A_BOLD if bold else 0)
            # 依顯示寬度截斷，避免超出右框線
            avail = box_w - val_col - 2
            v = value
            while self._disp_w(v) > avail and v:
                v = v[:-1]
            win.addstr(y, val_col, v, attr)

        if show_route:
            self._draw_route_stations(win, line_y, box_w, *station_info)
        elif show_subway:
            self._draw_subway(win, line_y, box_w, status)

        foot = f" 更新 {age:.1f}s 前 · 按 q 離開 "
        win.addstr(box_h - 1, 2, foot, self._c(5))
        win.refresh()

        # 右側 LiDAR 雷達（重畫 BEV 極座標散點 + goal 箭頭）
        self._maybe_draw_radar(stdscr, h, w, box_w, box_h,
                               sweep, sweep_t, status, now)

        # 到達目標 banner（收到 goal_reached event 後顯示 6 秒）
        if diag_event is not None and diag_event.get("event") == "goal_reached":
            banner_age = now - diag_event_t
            if banner_age < 6.0:
                self._draw_goal_banner(stdscr, h, w, diag_event, 6.0 - banner_age)

        # 往返測試提示（就緒時跳大字提示按空白鍵；待命/來回中顯示底部一列）
        if pp is not None and (now - pp_t) < 2.0:
            self._draw_pingpong(stdscr, h, w, pp)

    def _draw_pingpong(self, stdscr, h: int, w: int, pp: dict) -> None:
        """往返測試提示：ready→大字提示按空白鍵；waiting/running→底部單列狀態。"""
        state = pp.get("state")
        a, b = pp.get("a"), pp.get("b")
        auto = bool(pp.get("auto"))
        auto_tag = "全自動:開" if auto else "全自動:關"
        if state == "ready":
            rp = pp.get("ready_point")
            other = b if rp == a else a
            banner_h = 5
            banner_w = min(max(50, w - 2), 72)
            by = min(h - banner_h - 1, 19)
            bx = max(0, (w - banner_w) // 2)
            if by < 0 or banner_w < 20:
                return
            try:
                bwin = stdscr.derwin(banner_h, banner_w, by, bx)
                bwin.erase()
                bwin.box()
                title = f" 往返測試就緒：車停在 {rp}　[{auto_tag}] "
                bwin.addstr(1, max(2, (banner_w - self._disp_w(title)) // 2), title,
                            self._c(1) | curses.A_BOLD)
                if auto:
                    hint = f"全自動模式：即將自動出發往 {other}（往返 {a} ↔ {b}）"
                else:
                    hint = f"按【空白鍵】開始往 {other}（往返 {a} ↔ {b}）"
                bwin.addstr(2, max(2, (banner_w - self._disp_w(hint)) // 2),
                            hint, self._c(2) | curses.A_BOLD)
                foot = " 空白=開始 · a=全自動切換 · q=離開 "
                bwin.addstr(banner_h - 1, 2, foot, self._c(5))
                bwin.refresh()
            except curses.error:
                pass
        else:
            # waiting / running → 螢幕最底一列
            if state == "running":
                txt = (f"↻ 往返測試進行中 {pp.get('from')}→{pp.get('target')}"
                       f" [{auto_tag}]（a=全自動切換 · s=中斷 · 切 manual/estop 亦中斷）")
                pair = 4
            elif not pp.get("nodes_loaded"):
                txt = "往返測試：等路由節點表載入（需 routing/NDT 在跑）…"
                pair = 2
            else:
                txt = (f"往返測試待命 [{auto_tag}]：把車開到 {a}/{b} 停穩 → "
                       f"就緒後{'自動出發' if auto else '按空白鍵開始'}（a=全自動切換）")
                pair = 2
            try:
                y = h - 1
                stdscr.addstr(y, 0, txt[:max(0, w - 1)], self._c(pair) | curses.A_BOLD)
            except curses.error:
                pass

    @staticmethod
    def _dist(m) -> str:
        if m is None:
            return "—"
        return ">20" if m >= 19.9 else f"{m:.1f}"

    def _draw_goal_banner(self, stdscr, h: int, w: int,
                          event: dict, remaining: float) -> None:
        """到達目標時在儀表板正下方疊加大字通知."""
        banner_h = 7
        banner_w = min(max(48, w - 2), 72)
        by = min(h - banner_h - 1, 19)  # 緊接在儀表板下方
        bx = max(0, (w - banner_w) // 2)
        if by < 0 or banner_w < 20:
            return
        try:
            bwin = stdscr.derwin(banner_h, banner_w, by, bx)
            bwin.erase()
            bwin.box()
            # 標題：大到達目標
            title = " ** 到達目標！** "
            bwin.addstr(1, max(2, (banner_w - self._disp_w(title)) // 2), title,
                        self._c(1) | curses.A_BOLD)
            # 距離
            dist_txt = f"距終點 {event.get('dist', '?')}m"
            bwin.addstr(3, max(2, (banner_w - self._disp_w(dist_txt)) // 2),
                        dist_txt, self._c(4))
            # log 路徑（截斷至可用寬度）
            csv = event.get("csv", "")
            if csv:
                max_csv = banner_w - 6
                short = csv if len(csv) <= max_csv else "..." + csv[-(max_csv - 3):]
                bwin.addstr(4, 3, short, self._c(5))
            # 倒數
            foot = f" {remaining:.0f}s 後消失 · 按 q 離開 "
            bwin.addstr(banner_h - 1, 2, foot, self._c(5))
            bwin.refresh()
        except curses.error:
            pass

    # ── 右側 LiDAR 雷達（把 BEV 的極座標散點重畫成文字，不吃 bev_image 影像） ──
    def _maybe_draw_radar(self, stdscr, h: int, w: int, box_w: int, box_h: int,
                          sweep, sweep_t: float, status, now: float) -> None:
        node = self.node
        try:
            if not node.get_parameter("enable_radar").value:
                return
            style = node.get_parameter("radar_style").get_parameter_value().string_value
        except Exception:
            style = "braille"
        rx = box_w + 1                 # 緊接主框右側
        avail_w = w - rx - 1
        rh = box_h
        if avail_w < 26 or h < rh + 1:  # 太窄/太矮放不下 → 跳過（窄螢幕自動退化）
            return
        # braille 點字物理上近正方，inner_cols ≈ 2×inner_rows 才會畫成圓
        desired_w = (rh - 2) * 2 + 2
        rw = min(avail_w, max(28, desired_w))
        try:
            win = stdscr.derwin(rh, rw, 0, rx)
        except curses.error:
            return
        win.erase()
        win.box()
        title = f" LiDAR 雷達 前↑ {node._radar_range:.0f}m "
        win.addstr(0, max(2, (rw - self._disp_w(title)) // 2), title,
                   self._c(4) | curses.A_BOLD)
        sweep_age = now - sweep_t if sweep_t else float("inf")
        if not sweep or sweep_age > 1.5:
            m = "⚠ 等待 sweep…" if not sweep else f"⚠ sweep 逾時 {sweep_age:.1f}s"
            win.addstr(rh // 2, max(2, (rw - self._disp_w(m)) // 2), m, self._c(2))
            win.refresh()
            return
        rows, cols = rh - 2, rw - 2
        if style == "dots":
            self._radar_dots(win, rows, cols, sweep, status)
        else:
            self._radar_braille(win, rows, cols, sweep, status)
        win.refresh()

    def _sev(self, real: float) -> int:
        # 與 BEV 同門檻：>=5m 安全(綠)、2~5m 注意(黃/橙)、<2m 危險(紅)。
        return 1 if real >= 5.0 else (2 if real >= 2.0 else 3)

    def _radar_braille(self, win, rows: int, cols: int, sweep, status) -> None:
        r_max, r_robot = self.node._r_max, self.node._r_robot
        rng = self.node._radar_range
        denom = r_max - r_robot
        n = len(sweep) or 1
        wd, hd = cols * 2, rows * 4          # 點字解析度：每 cell 2×4 點
        cx, cy = wd / 2.0, hd / 2.0
        scale = (min(wd, hd) / 2.0 - 2) / rng if rng > 0 else 1.0
        mask: dict = {}
        sev: dict = {}
        for i, norm in enumerate(sweep):
            real = norm * denom + r_robot
            if real >= r_max - 0.5:          # 貼近量程上限視為無回波 → 不畫（空曠）
                continue
            d = real if real < rng else rng  # 超出顯示半徑 → 貼邊
            ang = math.radians(-180.0 + i * (360.0 / n))
            dx = int(round(cx - d * math.sin(ang) * scale))   # 左(+y)畫左
            dy = int(round(cy - d * math.cos(ang) * scale))   # +上（前方）
            if not (0 <= dx < wd and 0 <= dy < hd):
                continue
            key = (dy // 4, dx // 2)
            mask[key] = mask.get(key, 0) | BRAILLE_DOTS[(dx % 2, dy % 4)]
            s = self._sev(real)
            if s > sev.get(key, 0):
                sev[key] = s
        for (row, col), bits in mask.items():
            try:
                win.addstr(1 + row, 1 + col, chr(0x2800 + bits),
                           self._c(sev[(row, col)]))
            except curses.error:
                pass
        self._radar_overlays(win, rows, cols, status,
                             lambda sx, sy: ((cy - sy * scale) // 4,
                                             (cx - sx * scale) // 2))

    def _radar_dots(self, win, rows: int, cols: int, sweep, status) -> None:
        r_max, r_robot = self.node._r_max, self.node._r_robot
        rng = self.node._radar_range
        denom = r_max - r_robot
        n = len(sweep) or 1
        hx, hy = (cols - 1) / 2.0, (rows - 1) / 2.0
        for i, norm in enumerate(sweep):
            real = norm * denom + r_robot
            if real >= r_max - 0.5:
                continue
            d = real if real < rng else rng
            ang = math.radians(-180.0 + i * (360.0 / n))
            col = int(round(hx - (d * math.sin(ang) / rng) * hx))
            row = int(round(hy - (d * math.cos(ang) / rng) * hy))
            if not (0 <= col < cols and 0 <= row < rows):
                continue
            try:
                win.addstr(1 + row, 1 + col, "●", self._c(self._sev(real)))
            except curses.error:
                pass
        self._radar_overlays(win, rows, cols, status,
                             lambda sx, sy: (hy - (sy / rng) * hy,
                                             hx - (sx / rng) * hx))

    def _radar_overlays(self, win, rows: int, cols: int, status, w2c) -> None:
        """疊上機器人（中心↑）與 goal（黃◆）。w2c: (sx,sy)世界→(row,col)格映射。"""
        try:
            win.addstr(1 + rows // 2, 1 + cols // 2, "↑",
                       self._c(5) | curses.A_BOLD)
        except curses.error:
            pass
        if not status:
            return
        gd, ga = status.get("goal_dist"), status.get("goal_ang_deg")
        if gd is None or ga is None:
            return
        rng = self.node._radar_range
        d = min(gd, rng)
        ang = math.radians(ga)               # 0=正前方，與 sweep 同慣例
        row, col = w2c(d * math.sin(ang), d * math.cos(ang))
        row = max(0, min(rows - 1, int(round(row))))
        col = max(0, min(cols - 1, int(round(col))))
        try:
            win.addstr(1 + row, 1 + col, "◆", self._c(2) | curses.A_BOLD)
        except curses.error:
            pass

    def _draw_subway(self, win, y: int, box_w: int, s: dict) -> None:
        """地鐵路線式進度條：起 ━━ 目前位置(O) ━ carrot(>) ━━ 終，附「第 i/N 點」。

        path 的 waypoint 是密集 pose（非命名站），故以「起→終」連續進度呈現：
        O = 車目前最近的 waypoint（path_i），> = lookahead carrot（往前的瞬時目標）。
        """
        pi = s.get("path_i") or 0
        pn = s.get("path_n") or 1
        ci = s.get("path_carrot")
        win.addstr(y, 2, "路線", self._c(5))
        x0 = 11
        avail = box_w - x0 - 2
        count_txt = f" {pi + 1}/{pn}"
        # 預留 起[ (3 格) + ]終 (3 格) + 計數文字
        reserve = 3 + 3 + self._disp_w(count_txt)
        bar_w = max(8, avail - reserve)
        frac = pi / (pn - 1) if pn > 1 else 0.0
        pos = max(0, min(bar_w - 1, int(round(frac * (bar_w - 1)))))
        track = ["="] * pos + ["O"] + ["-"] * (bar_w - pos - 1)
        cpos = None
        if ci is not None and pn > 1:
            cpos = max(0, min(bar_w - 1, int(round((ci / (pn - 1)) * (bar_w - 1)))))
            if cpos != pos:
                track[cpos] = ">"
        bar = "".join(track)
        bar_x = x0 + 3
        win.addstr(y, x0, "起[", self._c(5))
        win.addstr(y, bar_x, bar, self._c(5))
        try:
            win.addch(y, bar_x + pos, ord("O"), self._c(1) | curses.A_BOLD)
            if cpos is not None and cpos != pos:
                win.addch(y, bar_x + cpos, ord(">"), self._c(2))
        except curses.error:
            pass
        win.addstr(y, bar_x + bar_w, "]終", self._c(5))
        win.addstr(y, bar_x + bar_w + 3, count_txt, self._c(4))

    def _draw_route_stations(self, win, y: int, box_w: int,
                             names: list, cur: int) -> None:
        """地鐵路線式具名站線：c1 - c3 - e0，目前站綠色粗體；過寬時以 … 視窗化."""
        win.addstr(y, 2, "路線", self._c(5))
        x0 = 11
        avail = box_w - x0 - 2
        n = len(names)
        sep = " - "
        sep_w = len(sep)

        def span_w(lo: int, hi: int) -> int:
            w = sum(self._disp_w(names[i]) for i in range(lo, hi + 1))
            w += sep_w * (hi - lo)
            if lo > 0:
                w += 2   # 前置「… 」
            if hi < n - 1:
                w += 2   # 後置「 …」
            return w

        # 以目前站為中心，左右貪婪擴張到塞不下為止（視窗化避免站太多爆寬）
        lo = hi = max(0, min(cur, n - 1))
        while True:
            grew = False
            if lo > 0 and span_w(lo - 1, hi) <= avail:
                lo -= 1
                grew = True
            if hi < n - 1 and span_w(lo, hi + 1) <= avail:
                hi += 1
                grew = True
            if not grew:
                break

        x = x0
        if lo > 0:
            win.addstr(y, x, "… ", self._c(5))
            x += 2
        for i in range(lo, hi + 1):
            attr = (self._c(1) | curses.A_BOLD) if i == cur else self._c(5)
            win.addstr(y, x, names[i], attr)
            x += self._disp_w(names[i])
            if i < hi:
                win.addstr(y, x, sep, self._c(5))
                x += sep_w
        if hi < n - 1:
            win.addstr(y, x, " …", self._c(5))

    @staticmethod
    def _track_pair(sent, act) -> int:
        """送出 vs 實測 跟隨度顏色：底盤明顯跟不上→黃。

        |sent|<0.2 時數值太小、比值不可靠（含原地）→ 不評斷直接給灰。
        實測只達送出 6 成以下代表飽和/deadband/延遲（gap #2/#6），標黃提醒。
        """
        if sent is None or act is None or abs(sent) < 0.2:
            return 5
        ratio = abs(act) / max(abs(sent), 1e-3)
        return 2 if ratio < 0.6 else 5

    @staticmethod
    def _lag_text(s: dict) -> tuple[str, int]:
        lag = s.get("lag_ms")
        if lag is None:
            return "訊號平穩，暫無法判定（需移動中）", 5
        corr = s.get("lag_corr")
        ch = {"v": "線速", "w": "角速"}.get(s.get("lag_ch"), "")
        txt = f"{lag}ms  (相關 {_fmt(corr, '.2f')}, {ch}通道)"
        # CLAUDE.md gap #5：>200ms 可能振盪
        if lag > 400 or (corr is not None and corr < 0.4):
            return txt + " ⚠偏大", 3
        if lag > 200:
            return txt, 2
        return txt, 1

    @staticmethod
    def _dist_pair(m) -> int:
        # 最近障礙距離的健康度門檻：<0.6m 已近碰撞風險(紅)，0.6~1.2m 需注意(黃)，
        # 其餘安全(綠)。門檻對應車身半徑 0.35m + 安全餘裕。
        if m is None:
            return 5
        if m < 0.6:
            return 3   # 紅：過近
        if m < 1.2:
            return 2   # 黃：注意
        return 1       # 綠

    @staticmethod
    def _vo_rows(vo: dict) -> list[tuple[str, str, int, bool]]:
        """VO 安全層兩列：①介入狀態 ②參數狀況。enable_vo 才有資料，否則整段不畫。"""
        fail = vo.get("fail") or ""
        if fail:
            reason = {"odom": "odom 逾時", "desired": "RL 逾時"}.get(fail, fail)
            vo_txt, vo_pair, vo_bold = f"⚠ 看門狗發 0（{reason}）", 3, True
        elif vo.get("blocked"):
            vo_txt = f"⛔ 全堵死 → 停車（近障 {vo.get('n_obs', 0)}）"
            vo_pair, vo_bold = 3, True
        elif vo.get("intervening"):
            vo_txt = (
                f"介入中 RL({_fmt(vo.get('des_v'), '+.2f')},{_fmt(vo.get('des_w'), '+.2f')})"
                f"→({_fmt(vo.get('vo_v'), '+.2f')},{_fmt(vo.get('vo_w'), '+.2f')}) "
                f"近障{vo.get('n_obs', 0)} 可行{vo.get('n_feasible', 0)}")
            vo_pair, vo_bold = 2, True
        elif vo.get("obs_stale"):
            vo_txt, vo_pair, vo_bold = "放行（障礙源逾時/未偵測，僅 ω 限幅）", 2, False
        elif vo.get("engaged"):
            vo_txt, vo_pair, vo_bold = f"監看中（近障 {vo.get('n_obs', 0)}，未改寫放行）", 4, False
        else:
            vo_txt, vo_pair, vo_bold = "✓ 放行（無近障）", 1, False

        params_txt = (
            f"ω≤{_fmt(vo.get('w_max'), '.1f')} 預測{_fmt(vo.get('horizon'), '.1f')}s "
            f"觸發{_fmt(vo.get('engage_range'), '.0f')}m 餘裕{_fmt(vo.get('margin'), '.2f')}m "
            f"追蹤{vo.get('n_tracked', 0)}")
        return [
            ("VO", vo_txt, vo_pair, vo_bold),
            ("VO參數", params_txt, 5, False),
        ]

    def _build_rows(self, s: dict, lvdot_n=None, lvdot_age=float("inf"),
                    station_info=None, goal_reached=False,
                    vo=None, vo_age=float("inf")
                    ) -> list[tuple[str, str, int, bool]]:
        # 把 status JSON 整理成儀表板每一列 (標籤, 數值字串, 顏色pair, 粗體)。
        # 渲染層只負責畫，所有「該紅該黃」的健康度判斷都集中在這裡。
        mode = s.get("mode", "?")
        mode_pair = {"nav": 1, "idle": 2, "paused": 2,
                     "manual": 4, "estop": 3}.get(mode, 5)
        mode_txt = f"{MODE_LABEL.get(mode, mode)}    速率 {_fmt(s.get('speed_rate'), '.2f')}"

        # 三層速度：想要(RL) → 送出(濾波後) → 實測(odom)
        # 非 nav 模式不跑推論、cmd 強制 0，三層都會是 0 → 顯示灰字狀態文字避免誤會
        IDLE_SPEED_TXT = {
            "idle": "待命中（未啟動導航）", "paused": "暫停中（保留推論記憶）",
            "estop": "緊急停車中", "manual": "外部接管中（搖桿）",
        }
        w_over = bool(s.get("w_over"))
        if mode != "nav":
            placeholder = IDLE_SPEED_TXT.get(mode, "待命中")
            v_txt = w_txt = placeholder
            v_pair = w_pair = 5
            w_over = False
        else:
            v_txt = (f"想{_fmt(s.get('rl_v'), '+.2f')} 送{_fmt(s.get('sent_v'), '+.2f')} "
                     f"實{_fmt(s.get('act_v'), '+.2f')} m/s")
            w_txt = (f"想{_fmt(s.get('rl_w'), '+.2f')} 送{_fmt(s.get('sent_w'), '+.2f')} "
                     f"實{_fmt(s.get('act_w'), '+.2f')} rad/s")
            if w_over:
                w_txt += f" ⚠超{_fmt(s.get('chassis_w_max'), '.1f')}"
            v_pair = self._track_pair(s.get("sent_v"), s.get("act_v"))
            w_pair = 3 if w_over else self._track_pair(s.get("sent_w"), s.get("act_w"))

        lag_txt, lag_pair = self._lag_text(s)

        f, b, l, r = (s.get("front_m"), s.get("back_m"),
                      s.get("left_m"), s.get("right_m"))
        obst_txt = (f"前{self._dist(f)} 後{self._dist(b)} "
                    f"左{self._dist(l)} 右{self._dist(r)} m")
        obst_min = min([x for x in (f, b, l, r) if x is not None], default=None)
        obst_pair = self._dist_pair(obst_min)

        lidar_age = s.get("lidar_age")
        lidar_ok = lidar_age is not None and lidar_age < 0.5
        nearest = s.get("nearest_m")
        src = s.get("lidar_src", "")
        src_short = {"preprocessor_topic": "preproc", "inline": "inline",
                     "inline_fallback": "fallback", "none": "—"}.get(src, src)
        lidar_txt = (f"{'✓' if lidar_ok else '⚠'} {_fmt(lidar_age, '.2f')}s  "
                     f"最近 {self._dist(nearest)}m  ({src_short})")

        odom_age = s.get("odom_age")
        odom_ok = odom_age is not None and odom_age < 0.5
        odom_txt = f"{'✓' if odom_ok else '⚠'} {_fmt(odom_age, '.2f')}s"

        ndt_ok = bool(s.get("ndt_ok"))
        ndt_txt = f"{'✓ 穩定' if ndt_ok else '⚠ 未穩'}  {_fmt(s.get('ndt_age'), '.1f')}s"

        psrc = {"odom+offset": "NDT校正", "ndt_direct": "NDT直接",
                "odom_only": "純里程計"}.get(s.get("pose_src", ""), s.get("pose_src", ""))
        pose_txt = (f"({_fmt(s.get('pose_x'), '+.1f')}, {_fmt(s.get('pose_y'), '+.1f')})"
                    f"  ∠{_fmt(s.get('pose_yaw_deg'), '+.0f')}°  {psrc}")
        pose_pair = 1 if s.get("pose_src") != "odom_only" else 2

        off = (s.get("off_x"), s.get("off_y"), s.get("off_yaw_deg"))
        off_txt = ("—" if off[0] is None else
                   f"({_fmt(off[0], '+.2f')}, {_fmt(off[1], '+.2f')}, {_fmt(off[2], '+.1f')}°)")

        # 導航型態：方式 B（routing 多 waypoint）vs 方式 A（單一 goal_pose）
        nav_type = s.get("nav_type")
        if nav_type == "path":
            if station_info is not None:
                names, cur = station_info
                nav_txt = (f"路徑導航 (routing) · {names[cur]} "
                           f"(第 {cur + 1}/{len(names)} 站)")
            else:
                pi, pn = s.get("path_i"), s.get("path_n")
                nav_txt = ("路徑導航 (routing)" if pi is None or not pn
                           else f"路徑導航 (routing) · 第 {pi + 1}/{pn} 點")
            nav_pair = 4
        elif nav_type == "single":
            nav_txt, nav_pair = "單一 goal 導航", 4
        else:
            nav_txt, nav_pair = "待命（無目標）", 5

        gd = s.get("goal_dist")
        ga = s.get("goal_ang_deg")
        gsrc = s.get("goal_src") or ""
        goal_bold = False
        if goal_reached:
            goal_txt, goal_pair, goal_bold = "✓ 已到達並已儲存 diag", 2, True
        elif gd is None:
            goal_txt, goal_pair = "尚無 goal/path", 5
        else:
            tag = f"  ({gsrc.split('/')[0]})" if gsrc else ""
            goal_txt, goal_pair = f"距 {_fmt(gd, '.1f')}m  方位 {_fmt(ga, '+.0f')}°{tag}", 1

        model_txt = s.get("model") or "—"

        # 車頭距障：front_m 是「感測器→障礙」，扣車頭前伸量 = 車頭最前緣到障礙的真實餘裕。
        # ⚠ VLP-16 近場盲區 ~0.4m（感測器物理極限）：front_m 不會小於 ~0.4，故餘裕 ≤0 時
        # 顯示「⚠ 已進盲區/可能接觸」提醒——此時雷達已看不到、不能只信數字。
        oh = self.node._front_overhang
        if f is None:
            front_txt, front_pair = "—", 5
        else:
            clr = f - oh
            if clr <= 0.0:
                front_txt, front_pair = f"⚠ ≤0（已進盲區/可能接觸） 前緣 前{self._dist(f)}−{oh:.2f}", 3
            elif clr < 0.3:
                front_txt, front_pair = f"{clr:.2f}m ⚠近  (前{self._dist(f)}−車頭{oh:.2f})", 3
            elif clr < 0.6:
                front_txt, front_pair = f"{clr:.2f}m  (前{self._dist(f)}−車頭{oh:.2f})", 2
            else:
                front_txt, front_pair = f"{clr:.2f}m  (前{self._dist(f)}−車頭{oh:.2f})", 1

        # LV-DOT 動態障礙偵測（獨立節點，policy 不吃，純態勢感知）
        if lvdot_n is None:
            lvdot_txt, lvdot_pair = "未啟動 / 無資料", 5
        elif lvdot_age > 2.0:
            lvdot_txt, lvdot_pair = f"⚠ 逾時 {lvdot_age:.1f}s（偵測器可能死）", 2
        elif lvdot_n == 0:
            lvdot_txt, lvdot_pair = "✓ 無動態障礙", 1
        else:
            lvdot_txt, lvdot_pair = f"{lvdot_n} 個動態障礙", 2

        rows = [
            ("模式", mode_txt, mode_pair, True),
            ("速度v", v_txt, v_pair, False),
            ("速度ω", w_txt, w_pair, w_over),
            ("延遲", lag_txt, lag_pair, lag_pair == 3),
            ("障礙", obst_txt, obst_pair, obst_pair == 3),
            ("車頭距障", front_txt, front_pair, front_pair == 3),
            ("動態", lvdot_txt, lvdot_pair, False),
        ]
        # VO 安全層（enable_vo 才有 status；收不到/逾時就不畫，自動隱藏）
        if vo is not None and vo_age < 1.5:
            rows.extend(self._vo_rows(vo))
        rows += [
            ("LiDAR", lidar_txt, 3 if not lidar_ok else 1, False),
            ("里程計", odom_txt, 1 if odom_ok else 3, False),
            ("NDT", ndt_txt, 1 if ndt_ok else 2, False),
            ("位置", pose_txt, pose_pair, False),
            ("偏移", off_txt, 5, False),
            ("導航", nav_txt, nav_pair, nav_type == "path"),
            ("目標", goal_txt, goal_pair, goal_bold),
            ("模型", model_txt, 5, False),
        ]
        return rows


def main(args=None) -> None:
    # 設 locale 讓 curses 正確處理 UTF-8 中文寬字元，否則邊框/中文會錯位。
    locale.setlocale(locale.LC_ALL, "")
    rclpy.init(args=args)
    node = StatusTuiNode()
    # ROS spin 與 curses 渲染不能共用同一執行緒（兩者都是阻塞迴圈），
    # 故 spin 丟背景 daemon thread，主執行緒交給 curses。
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    dash = Dashboard(node)
    try:
        # curses.wrapper 負責 initscr/endwin 與例外時還原終端機；需真實 TTY，
        # 在 pipe / 非互動 shell 會崩潰，故此節點只由 deploy_rl_shell 前景啟動。
        curses.wrapper(dash.run)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("⏹ status_tui 已離開")


if __name__ == "__main__":
    main()
