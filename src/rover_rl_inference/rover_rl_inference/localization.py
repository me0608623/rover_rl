"""定位輔助 — 採 rover2_ws/spot_rl pattern 的 map→odom offset 緩存.

核心想法：
  - NDT (map→odom) 更新頻率低（1~10 Hz），用 tf_buffer 每幀查可能 stale。
  - 維護一個 cached offset：robot_in_map ≈ odom + offset（offset 在 NDT 穩定且機器人靜止時更新）。
  - tick 時用 cached offset + 即時 odom 算 robot pose，5 Hz 不阻塞。

收斂判定（rover2_ws P0-1）：
  - NDT 節點只在 is_converged=True 時發 /ndt_pose。
  - 「近 1 秒內持續收到，且累積 >= 5 次」= NDT 穩定。

Offset 更新閘門（rover2_ws P1-1）：
  - Gate 1: |v| < 0.1 m/s 才更新（移動中 NDT 不可靠）
  - Gate 2: 與舊 offset delta > 0.3 m 拒絕更新（NDT 飄移）

Fallback：
  - NDT 從未穩定 → 純 odom 模式（goal 假設也在 odom frame）
  - NDT 曾穩定但暫時失聯 → 用最後一次 offset 繼續跑
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


NDT_STABLE_AGE_S = 1.0       # 多久內收到 ndt_pose 才算 fresh
NDT_STABLE_COUNT = 5         # 累積至少這麼多次才算 settled
OFFSET_UPDATE_INTERVAL_S = 5.0
OFFSET_DELTA_REJECT_M = 0.3
SPEED_GATE_MPS = 0.1


@dataclass
class RobotPose:
    """機器人在 map frame 的位姿 + 來源標記."""
    x: float
    y: float
    yaw: float
    source: str   # 'odom+offset', 'ndt_direct', 'odom_only'


class MapOdomOffsetTracker:
    """追蹤 map→odom offset，給 rover_rl policy_node 用."""

    def __init__(self, logger=None):
        self._logger = logger
        # NDT 統計
        self._ndt_recv_count = 0
        self._last_ndt_recv_t = 0.0
        # 緩存
        self._offset: tuple[float, float] | None = None
        self._last_offset_update_t = 0.0
        # 同步 odom（在收到 ndt_pose 的當下記錄）
        self._sync_odom_xy: tuple[float, float] | None = None
        # 最新 NDT pose（fallback 用）
        self._last_ndt_xy: tuple[float, float] | None = None

    def on_ndt_pose(self, ndt_x: float, ndt_y: float,
                     odom_x_at_ndt: float, odom_y_at_ndt: float) -> None:
        """收到 /ndt_pose 時呼叫；odom_x/y 應為**當下**的 odom 位置.

        若沒有同步 odom 來源，可省略並退回 lookup_transform odom→base_link 方案。
        """
        now = time.monotonic()
        if self._last_ndt_recv_t == 0.0 or now - self._last_ndt_recv_t > NDT_STABLE_AGE_S:
            self._ndt_recv_count = 1
        else:
            self._ndt_recv_count = min(self._ndt_recv_count + 1, 1000)
        self._last_ndt_recv_t = now
        self._last_ndt_xy = (ndt_x, ndt_y)
        self._sync_odom_xy = (odom_x_at_ndt, odom_y_at_ndt)

    def is_ndt_stable(self) -> bool:
        if self._last_ndt_recv_t == 0.0:
            return False
        return (time.monotonic() - self._last_ndt_recv_t < NDT_STABLE_AGE_S
                and self._ndt_recv_count >= NDT_STABLE_COUNT)

    def try_update_offset(self, robot_speed_mps: float) -> bool:
        """嘗試更新 offset；回傳是否真的更新了."""
        if not self.is_ndt_stable():
            return False
        if self._sync_odom_xy is None or self._last_ndt_xy is None:
            return False
        now = time.monotonic()
        if (self._offset is not None
                and now - self._last_offset_update_t < OFFSET_UPDATE_INTERVAL_S):
            return False
        if abs(robot_speed_mps) > SPEED_GATE_MPS:
            return False

        ndt_x, ndt_y = self._last_ndt_xy
        sync_x, sync_y = self._sync_odom_xy
        new_offset = (ndt_x - sync_x, ndt_y - sync_y)

        if self._offset is None:
            self._log_info(
                f"[OFFSET] 初始化 map→odom=({new_offset[0]:.3f},{new_offset[1]:.3f})"
            )
        else:
            delta = math.hypot(
                new_offset[0] - self._offset[0],
                new_offset[1] - self._offset[1],
            )
            if delta > OFFSET_DELTA_REJECT_M:
                self._log_warn(
                    f"[OFFSET] delta={delta:.3f}m > {OFFSET_DELTA_REJECT_M}m，拒絕更新"
                )
                self._last_offset_update_t = now  # 冷卻
                return False
            if delta > 0.01:
                self._log_info(
                    f"[OFFSET] 更新=({new_offset[0]:.3f},{new_offset[1]:.3f}) "
                    f"delta={delta:.3f}m"
                )
        self._offset = new_offset
        self._last_offset_update_t = now
        return True

    def get_robot_pose_in_map(
        self, odom_x: float, odom_y: float, odom_yaw: float,
    ) -> RobotPose:
        """計算機器人在 map frame 的位姿，依優先順序選擇來源."""
        if self._offset is not None:
            ox, oy = self._offset
            return RobotPose(
                x=odom_x + ox, y=odom_y + oy, yaw=odom_yaw,
                source="odom+offset",
            )
        if self.is_ndt_stable() and self._last_ndt_xy is not None:
            return RobotPose(
                x=self._last_ndt_xy[0], y=self._last_ndt_xy[1], yaw=odom_yaw,
                source="ndt_direct",
            )
        return RobotPose(x=odom_x, y=odom_y, yaw=odom_yaw, source="odom_only")

    @property
    def offset(self) -> tuple[float, float] | None:
        return self._offset

    @property
    def ndt_age_s(self) -> float:
        if self._last_ndt_recv_t == 0.0:
            return float("inf")
        return time.monotonic() - self._last_ndt_recv_t

    def _log_info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warn(msg)


def world_to_body(
    world_x: float, world_y: float,
    robot_x: float, robot_y: float, robot_yaw: float,
) -> tuple[float, float]:
    """world frame → body frame（spot_rl pattern）.

    body_x = forward, body_y = left.
    """
    dx = world_x - robot_x
    dy = world_y - robot_y
    c = math.cos(-robot_yaw)
    s = math.sin(-robot_yaw)
    return c * dx - s * dy, s * dx + c * dy
