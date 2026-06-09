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
import threading
import time
import unicodedata

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

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
        topic = self.get_parameter("topic_status").get_parameter_value().string_value
        topic_dyn = self.get_parameter(
            "topic_dynamic_bboxes").get_parameter_value().string_value
        topic_diag = self.get_parameter("topic_diag_event").get_parameter_value().string_value
        self._lock = threading.Lock()
        self._status: dict | None = None
        self._last_t = 0.0
        self._lvdot_n: int | None = None     # 動態障礙框數；None=從未收到
        self._lvdot_t = 0.0
        self._diag_event: dict | None = None  # goal_reached 等事件
        self._diag_event_t = 0.0
        self.create_subscription(String, topic, self._cb_status, 10)
        self.create_subscription(MarkerArray, topic_dyn, self._cb_dyn, 10)
        self.create_subscription(String, topic_diag, self._cb_diag_event, 10)
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

    def snapshot(self) -> tuple[dict | None, float, int | None, float, dict | None, float]:
        with self._lock:
            return (self._status, self._last_t, self._lvdot_n, self._lvdot_t,
                    self._diag_event, self._diag_event_t)


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

    def _render(self, stdscr) -> None:
        status, last_t, lvdot_n, lvdot_t, diag_event, diag_event_t = self.node.snapshot()
        now = time.monotonic()
        # age = 距上次收到狀態多久。policy_node 以 5 Hz 發，>1.5s 沒更新即視為
        # stale（policy_node 可能掛了），改顯示警告而非沿用過期數值誤導判讀。
        age = now - last_t if last_t else float("inf")
        lvdot_age = now - lvdot_t if lvdot_t else float("inf")
        stale = age > 1.5

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        box_w = min(max(52, w - 2), 70)

        # 先依狀態決定列數與是否畫地鐵式進度條（路徑導航才畫）→ 才能算框高
        if status is not None and not stale:
            rows = self._build_rows(status, lvdot_n, lvdot_age)
            pn = status.get("path_n")
            show_subway = (status.get("nav_type") == "path" and pn and pn >= 2
                           and status.get("path_i") is not None)
        else:
            rows = []
            show_subway = False

        n = len(rows)
        subway_y = 1 + n + 1            # rows 後空一列再畫進度條
        box_h = (subway_y + 2) if show_subway else max(18, n + 3)
        # 框高超出終端機 → 先犧牲進度條再縮；仍放不下才提示放大
        if show_subway and h < box_h + 1:
            show_subway = False
            box_h = max(18, n + 3)
        if h < box_h + 1 or w < box_w:
            stdscr.addstr(0, 0, "終端機視窗太小，請放大…")
            stdscr.refresh()
            return

        win = stdscr.derwin(box_h, box_w, 0, 0)
        win.box()
        title = " rover_rl 即時狀態 "
        win.addstr(0, max(2, (box_w - self._disp_w(title)) // 2), title,
                   self._c(4) | curses.A_BOLD)

        if status is None or stale:
            msg = "⚠ 等待 policy_node 狀態…" if status is None else "⚠ 狀態逾時，policy_node 可能已停"
            win.addstr(box_h // 2, 3, msg, self._c(2) | curses.A_BOLD)
            win.addstr(box_h - 1, 2, " 按 q 離開 ", self._c(5))
            win.refresh()
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

        if show_subway:
            self._draw_subway(win, subway_y, box_w, status)

        foot = f" 更新 {age:.1f}s 前 · 按 q 離開 "
        win.addstr(box_h - 1, 2, foot, self._c(5))
        win.refresh()

        # 到達目標 banner（收到 goal_reached event 後顯示 6 秒）
        if diag_event is not None and diag_event.get("event") == "goal_reached":
            banner_age = now - diag_event_t
            if banner_age < 6.0:
                self._draw_goal_banner(stdscr, h, w, diag_event, 6.0 - banner_age)

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

    def _build_rows(self, s: dict, lvdot_n=None, lvdot_age=float("inf")
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
            "idle": "待命中（未啟動導航）", "paused": "暫停中（保留 RNN 狀態）",
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
        if gd is None:
            goal_txt, goal_pair = "尚無 goal/path", 5
        else:
            tag = f"  ({gsrc.split('/')[0]})" if gsrc else ""
            goal_txt, goal_pair = f"距 {_fmt(gd, '.1f')}m  方位 {_fmt(ga, '+.0f')}°{tag}", 1

        model_txt = s.get("model") or "—"

        # RNN hidden state（episode 記憶）
        norm = s.get("rnn_norm")
        resets = s.get("rnn_resets")
        steps = s.get("rnn_steps")
        if norm is None:
            rnn_txt, rnn_pair = "—", 5
        elif norm < 0.01:
            rnn_txt, rnn_pair = f"待命/已重置  (累計重置 {_fmt(resets, 'd')} 次)", 5
        else:
            rnn_txt = (f"‖h‖={norm:.2f}  本段步數 {_fmt(steps, 'd')}  "
                       f"重置 {_fmt(resets, 'd')} 次")
            rnn_pair = 1

        # LV-DOT 動態障礙偵測（獨立節點，policy 不吃，純態勢感知）
        if lvdot_n is None:
            lvdot_txt, lvdot_pair = "未啟動 / 無資料", 5
        elif lvdot_age > 2.0:
            lvdot_txt, lvdot_pair = f"⚠ 逾時 {lvdot_age:.1f}s（偵測器可能死）", 2
        elif lvdot_n == 0:
            lvdot_txt, lvdot_pair = "✓ 無動態障礙", 1
        else:
            lvdot_txt, lvdot_pair = f"{lvdot_n} 個動態障礙", 2

        return [
            ("模式", mode_txt, mode_pair, True),
            ("速度v", v_txt, v_pair, False),
            ("速度ω", w_txt, w_pair, w_over),
            ("延遲", lag_txt, lag_pair, lag_pair == 3),
            ("RNN", rnn_txt, rnn_pair, False),
            ("障礙", obst_txt, obst_pair, obst_pair == 3),
            ("動態", lvdot_txt, lvdot_pair, False),
            ("LiDAR", lidar_txt, 3 if not lidar_ok else 1, False),
            ("里程計", odom_txt, 1 if odom_ok else 3, False),
            ("NDT", ndt_txt, 1 if ndt_ok else 2, False),
            ("位置", pose_txt, pose_pair, False),
            ("偏移", off_txt, 5, False),
            ("導航", nav_txt, nav_pair, nav_type == "path"),
            ("目標", goal_txt, goal_pair, False),
            ("模型", model_txt, 5, False),
        ]


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
