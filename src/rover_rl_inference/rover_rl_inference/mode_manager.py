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
    """簡單狀態機 + 切換 callback hook."""

    def __init__(self, initial: Mode = Mode.NAV, on_change=None):
        self._mode = initial
        self._on_change = on_change

    @property
    def mode(self) -> Mode:
        return self._mode

    def set(self, new_mode: Mode, reason: str = "") -> bool:
        if new_mode == self._mode:
            return False
        old = self._mode
        self._mode = new_mode
        if self._on_change is not None:
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
        # manual 模式讓外部接管；estop/idle 仍要發 0 以蓋過 stale cmd
        return self._mode in (Mode.NAV, Mode.ESTOP, Mode.IDLE, Mode.PAUSED)

    def force_zero_cmd(self) -> bool:
        """是否強制 cmd=0（不論 policy 輸出什麼）."""
        return self._mode in (Mode.ESTOP, Mode.IDLE, Mode.PAUSED)
