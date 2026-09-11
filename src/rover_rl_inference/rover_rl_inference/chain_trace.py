"""逐拍完整命令鏈追蹤（handoff #8）。

為什麼要有這個：舞龍舞獅發生時，光看結果層（success/collision）無法判斷斷在哪一段。
把整條鏈用**同一個時間戳**記下來，就能一眼定位：

    logits 就在左右跳            → policy 本身
    logits 穩但 issued 在跳      → decode / slew
    issued 穩但 published 在跳   → VO 或發布層
    published 穩但 odom 在跳     → 底盤 / 馬達

⚠️ 設計要求：`hist` 欄位必須是**逐字複製當拍真正餵進網路的那 4 個值**，
不可事後重算。這次的 bug 正是「重算值」與「實際餵入值」不同 —— 若診斷也用
重算，就會跟著錯，而且看起來一切正常。

輸出 JSON Lines，一拍一行，可直接用 pandas 讀。旋轉式檔案避免長跑塞爆磁碟。
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict


def _f(x) -> float | None:
    """轉成 JSON 安全的 float（NaN/Inf → None）。"""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


@dataclass
class ChainSample:
    """一拍的完整命令鏈。欄位順序即資料流順序。"""

    t_wall: float                       # time.time()，跨節點對時用
    t_mono: float                       # time.monotonic()，算間隔用
    seq: int

    # ── policy 決策 ──
    idx_a: int | None = None
    idx_w: int | None = None
    logit_a_max: float | None = None    # argmax 的 logit 值（信心）
    logit_w_max: float | None = None
    logit_a_margin: float | None = None  # top1 - top2，越小代表越猶豫
    logit_w_margin: float | None = None

    # ── decode → issued（進入致動器延遲佇列的那一組）──
    decoded_v: float | None = None
    decoded_w: float | None = None
    issued_v: float | None = None
    issued_w: float | None = None
    issued_accel: float | None = None

    # ── 發布（可能被 VO/安全層改過）──
    published_v: float | None = None
    published_w: float | None = None
    safety_override: bool | None = None
    issued_vs_published_dv: float | None = None
    issued_vs_published_dw: float | None = None

    # ── 車子實際反應 ──
    odom_v: float | None = None
    odom_w: float | None = None

    # ── 定位健康度 ──
    map_x: float | None = None
    map_y: float | None = None
    map_yaw: float | None = None
    pose_dt: float | None = None
    pose_dpos: float | None = None
    pose_dyaw: float | None = None
    pose_source: str | None = None
    jump_guard_state: str | None = None   # ok / rejected / recovering
    jump_guard_reason: str | None = None

    # ── 餵進網路的 action history（逐字，非重算）──
    hist: list[float] = field(default_factory=list)

    # ── 餵進網路的 obs[78] 剩餘時間比例（逐字，非重算）──
    # episode_horizon 分母寫錯不會有任何徵兆（只是整段時間特徵偏移），
    # 且超過 horizon 後它會夾在 0 不動 —— 兩種情況都只能靠記下實際餵入值才看得出來。
    obs_time_rem: float | None = None

    # ── 其他 ──
    speed_rate: float | None = None
    infer_ms: float | None = None
    sweep_age_ms: float | None = None
    mode: str | None = None


class ChainTracer:
    """執行緒安全的 JSONL 逐拍記錄器。

    用法：
        tr = ChainTracer(path, enabled=True)
        s = tr.begin()                  # 取得本拍 sample
        s.idx_a = ...                   # 各段填值
        tr.commit(s)                    # 寫出

    刻意**不做**任何欄位推導 —— 所有值都由呼叫端從真實變數填入，
    診斷才不會與實際執行分家。
    """

    def __init__(self, path: str, enabled: bool = True,
                 max_bytes: int = 64 * 1024 * 1024, keep: int = 3):
        self.enabled = bool(enabled)
        self.path = path
        self.max_bytes = int(max_bytes)
        self.keep = int(keep)
        self._seq = 0
        self._lock = threading.Lock()
        self._fh = None
        if self.enabled:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", buffering=1, encoding="utf-8")

    # ── 生命週期 ──
    def begin(self) -> ChainSample:
        with self._lock:
            self._seq += 1
            seq = self._seq
        return ChainSample(t_wall=time.time(), t_mono=time.monotonic(), seq=seq)

    def commit(self, s: ChainSample) -> None:
        if not self.enabled or self._fh is None:
            return
        d = asdict(s)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = _f(v)
            elif isinstance(v, list):
                d[k] = [_f(x) for x in v]
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._maybe_rotate_locked()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    # ── 內部 ──
    def _maybe_rotate_locked(self) -> None:
        try:
            if self._fh.tell() < self.max_bytes:
                return
        except (OSError, ValueError):
            return
        self._fh.close()
        for i in range(self.keep - 1, 0, -1):
            src, dst = f"{self.path}.{i}", f"{self.path}.{i+1}"
            if os.path.exists(src):
                os.replace(src, dst)
        os.replace(self.path, f"{self.path}.1")
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")


def logit_stats(logits, num_bins: int = 19):
    """回傳 (idx_a, idx_w, max_a, max_w, margin_a, margin_w)。

    margin = top1 − top2：越接近 0 代表 policy 在兩個相鄰動作之間猶豫，
    是左右翻轉（舞龍舞獅）的早期訊號 —— 比只看最終 ω 更早看得出來。
    """
    import numpy as np

    la = np.asarray(logits[:num_bins], dtype=float)
    lw = np.asarray(logits[num_bins:], dtype=float)

    def _one(x):
        idx = int(np.argmax(x))
        srt = np.sort(x)[::-1]
        margin = float(srt[0] - srt[1]) if x.size >= 2 else float("nan")
        return idx, float(srt[0]), margin

    ia, ma, ga = _one(la)
    iw, mw, gw = _one(lw)
    return ia, iw, ma, mw, ga, gw
