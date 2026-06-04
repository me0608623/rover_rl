"""Mode manager — 多模式 + hot-swap model 支援.

模式列表：
  - idle:    什麼都不做，發 zero cmd（待命）
  - nav:     正常 RL 推論（預設模式）
  - estop:   強制停車，鎖死 cmd=0，policy 不執行
  - manual:  cmd_vel 跳過 policy，由外部（搖桿）取代（rover_rl 不發 cmd）
  - paused:  暫停推論但保留 hidden state（短暫停止）

切換方式：
  - ROS service `~/set_mode` (std_srvs/SetBool 用 data=true 切到 nav, false 切到 idle)
    更彈性的方式：自定義 mode topic
  - ROS topic `~/mode` (std_msgs/String) — 訂閱式切換，方便整合 GUI / supervisor
  - launch param `initial_mode` — 啟動時設定

Hot-swap model：
  - ROS service `~/load_model` (rcl_interfaces 自訂；先用 SetString)
  - 切換 model_path → reload bundle → 重置 RNN hidden
  - 若 raw_obs_dim 不變可無縫切（policy 自己 reset），不變需告警
"""
from __future__ import annotations

from enum import Enum


class Mode(Enum):
    IDLE = "idle"
    NAV = "nav"
    ESTOP = "estop"
    MANUAL = "manual"
    PAUSED = "paused"

    @classmethod
    def parse(cls, s: str) -> "Mode":
        s = (s or "").strip().lower()
        for m in cls:
            if m.value == s:
                return m
        raise ValueError(f"unknown mode: {s!r}; expected {[m.value for m in cls]}")


class ModeManager:
    """簡單狀態機 + 切換 callback hook.

    本身不碰 ROS，只持有當前 mode 並在切換時觸發 callback；
    policy_node 透過 is_active / should_publish_cmd / force_zero_cmd
    三個查詢方法決定「要不要推論、要不要發 cmd_vel、要不要強制歸零」。
    把判斷邏輯收斂在這裡，避免散落在 node 各 callback 造成行為不一致。
    """

    def __init__(self, initial: Mode = Mode.NAV, on_change=None):
        self._mode = initial
        # on_change(old, new, reason)：切換時呼叫，供 node 做副作用
        # （如重置 RNN hidden、改 status badge），ModeManager 本身不做。
        self._on_change = on_change

    @property
    def mode(self) -> Mode:
        return self._mode

    def set(self, new_mode: Mode, reason: str = "") -> bool:
        # 同模式直接 no-op（回 False），避免重複觸發 on_change 把 RNN hidden 白白重置。
        if new_mode == self._mode:
            return False
        old = self._mode
        self._mode = new_mode
        if self._on_change is not None:
            # callback 失敗不能讓 mode 切換半途中斷（安全狀態優先），故吞例外。
            try:
                self._on_change(old, new_mode, reason)
            except Exception:
                pass
        return True

    def is_active(self) -> bool:
        """policy 是否該執行推論."""
        return self._mode == Mode.NAV

    def should_publish_cmd(self) -> bool:
        """rover_rl 是否該發 cmd_vel."""
        # 關鍵設計：唯獨 manual 模式「讓出 topic」完全不發，讓搖桿/外部
        # 節點獨占 cmd_vel topic（多發布者會在 mux 互搶）。
        # estop/idle/paused 反而「要主動發 0」——因為若靜默，底盤可能沿用
        # 上一筆 stale cmd 繼續動，主動發 0 才能確實覆蓋並煞停。
        return self._mode in (Mode.NAV, Mode.ESTOP, Mode.IDLE, Mode.PAUSED)

    def force_zero_cmd(self) -> bool:
        """是否強制 cmd=0（不論 policy 輸出什麼）."""
        # 這三種模式都不跑推論（is_active=False），但仍會發 cmd（見上），
        # 故需強制把內容壓成 0。nav 不在此列才會真正送出 policy 輸出。
        return self._mode in (Mode.ESTOP, Mode.IDLE, Mode.PAUSED)
