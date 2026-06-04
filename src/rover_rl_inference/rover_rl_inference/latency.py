"""
latency.py — 估計「送出 cmd」到「底盤實測速度」之間的時間落差（dead time）。

對應 CLAUDE.md sim-to-real gap #5：rover_rl 無 cmd_delay 補償，若底盤響應
延遲明顯（>0.2s）policy 可能振盪。此模組以滑動視窗正規化互相關估出落差秒數，
讓 TUI / diag_logger 能即時看到並記錄延遲。

原理：把最近一段「送出指令」與「實測速度」對齊，將實測往後平移 k 個取樣，
找使兩者相關係數最大的 k；lag = k * dt。訊號太平穩（站著不動）時回報 None。
"""
from __future__ import annotations

from collections import deque

import numpy as np


class LagEstimator:
    """單一通道（線速度或角速度）的延遲估計器。"""

    def __init__(self, dt: float, window_s: float = 4.0,
                 max_lag_s: float = 1.0, min_std: float = 0.04) -> None:
        self.dt = dt
        self.n = max(int(window_s / dt), 8)
        self.max_k = max(int(max_lag_s / dt), 1)
        self.min_std = min_std
        self.sent: deque[float] = deque(maxlen=self.n)
        self.act: deque[float] = deque(maxlen=self.n)

    def push(self, sent: float, act: float) -> None:
        self.sent.append(float(sent))
        self.act.append(float(act))

    def estimate(self) -> tuple[float | None, float | None]:
        """回傳 (lag_秒, 相關係數)；資料不足或訊號太平 → (None, None)。"""
        if len(self.sent) < self.max_k + 10:
            return None, None
        s = np.asarray(self.sent, dtype=np.float64)
        a = np.asarray(self.act, dtype=np.float64)
        if s.std() < self.min_std or a.std() < self.min_std:
            return None, None
        s = (s - s.mean()) / s.std()
        a = (a - a.mean()) / a.std()
        n = len(s)
        best_k, best_c = 0, -2.0
        for k in range(0, self.max_k + 1):
            if n - k <= 5:
                break
            c = float(np.mean(s[: n - k] * a[k:]))   # act 落後 sent k 步
            if c > best_c:
                best_c, best_k = c, k
        return best_k * self.dt, best_c
