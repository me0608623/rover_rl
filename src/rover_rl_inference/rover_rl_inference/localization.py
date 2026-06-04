"""定位輔助 — 採 rover2_ws/spot_rl pattern 的 map→odom offset 緩存.

核心想法：
  - NDT (map→odom) 更新頻率低（1~10 Hz），用 tf_buffer 每幀查可能 stale。
  - 維護 cached offset (xy + yaw)：robot_in_map ≈ odom + offset（offset 在 NDT 穩定且機器人靜止時更新）。
  - tick 時用 cached offset + 即時 odom 算 robot pose，5 Hz 不阻塞。

收斂判定（rover2_ws P0-1）：
  - NDT 節點只在 is_converged=True 時發 /ndt_pose。
  - 「近 1 秒內持續收到，且累積 >= 5 次」= NDT 穩定。

Offset 更新閘門（rover2_ws P1-1）：
  - Gate 1: |v| < 0.1 m/s 才更新（移動中 NDT 不可靠）
  - Gate 2: xy delta > 0.3 m 或 yaw delta > 0.2 rad 拒絕更新（NDT 飄移）

Fallback：
  - NDT 從未穩定 → 純 odom 模式（goal 假設也在 odom frame）
  - NDT 曾穩定但暫時失聯 → 用最後一次 offset 繼續跑

注意：spot_rl 把 goal 轉到 odom frame 再用 odom_yaw 算 body frame；
rover_rl 反過來把 robot 轉到 map frame，需要 yaw_offset = ndt_yaw - odom_yaw。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


def _angle_diff(a: float, b: float) -> float:
    """計算 a - b 並正規化到 (-π, π]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d <= -math.pi:
        d += 2 * math.pi
    return d


NDT_STABLE_AGE_S = 1.0       # 多久內收到 ndt_pose 才算 fresh
NDT_STABLE_COUNT = 5         # 累積至少這麼多次才算 settled
OFFSET_UPDATE_INTERVAL_S = 5.0
OFFSET_DELTA_REJECT_M = 0.3
OFFSET_YAW_DELTA_REJECT_RAD = 0.2   # ~11°，NDT yaw 突跳超過此值拒絕
SPEED_GATE_MPS = 0.1


@dataclass
class RobotPose:
    """機器人在 map frame 的位姿 + 來源標記."""
    x: float
    y: float
    yaw: float
    source: str   # 'odom+offset', 'ndt_direct', 'odom_only'


class MapOdomOffsetTracker:
    """追蹤 map→odom offset，給 rover_rl policy_node 用.

    為何不直接呼叫 tf_buffer.transform()：NDT 更新慢（1-10 Hz），TF lookup 常拿到
    stale 或失敗的結果。改成自己維護一份 cached offset，即使 NDT 暫時失聯仍能用
    最後一次有效值繼續推算 robot pose（safer fallback），行為與 rover2_ws 既有
    rl_policy_node 一致。
    """

    def __init__(self, logger=None):
        self._logger = logger
        # NDT 統計
        self._ndt_recv_count = 0
        self._last_ndt_recv_t = 0.0
        # 緩存 xy offset
        self._offset: tuple[float, float] | None = None
        self._last_offset_update_t = 0.0
        # 緩存 yaw offset（ndt_yaw - odom_yaw，補正 odom 與 map 的朝向差）
        self._yaw_offset: float | None = None
        # 同步 odom（在收到 ndt_pose 的當下記錄）
        self._sync_odom_xy: tuple[float, float] | None = None
        self._sync_odom_yaw: float | None = None
        # 最新 NDT pose（fallback 用）
        self._last_ndt_xy: tuple[float, float] | None = None
        self._last_ndt_yaw: float | None = None

    def on_ndt_pose(
        self,
        ndt_x: float, ndt_y: float, ndt_yaw: float,
        odom_x_at_ndt: float, odom_y_at_ndt: float, odom_yaw_at_ndt: float,
    ) -> None:
        """收到 /ndt_pose 時呼叫；odom_x/y/yaw 應為**當下**的 odom 位姿.

        必須同步記錄收到當下的 odom，offset = ndt - odom 才對得上同一時刻；事後再
        補 odom 會因機器人已移動而算錯 offset。
        """
        now = time.monotonic()
        # 收到頻率判定：距上次 > NDT_STABLE_AGE_S 視為「中斷後重新開始」，計數歸 1；
        # 連續收到才累加，累積 >= NDT_STABLE_COUNT 才算 NDT 真正 settled（見 is_ndt_stable）。
        if self._last_ndt_recv_t == 0.0 or now - self._last_ndt_recv_t > NDT_STABLE_AGE_S:
            self._ndt_recv_count = 1
        else:
            self._ndt_recv_count = min(self._ndt_recv_count + 1, 1000)
        self._last_ndt_recv_t = now
        self._last_ndt_xy = (ndt_x, ndt_y)
        self._last_ndt_yaw = ndt_yaw
        self._sync_odom_xy = (odom_x_at_ndt, odom_y_at_ndt)
        self._sync_odom_yaw = odom_yaw_at_ndt

    def is_ndt_stable(self) -> bool:
        # 同時要求「近 1 秒內有收到（fresh）」與「累積次數夠（settled）」：
        # NDT 只在 is_converged 時才發 /ndt_pose，能持續發代表定位已穩定收斂。
        if self._last_ndt_recv_t == 0.0:
            return False
        return (time.monotonic() - self._last_ndt_recv_t < NDT_STABLE_AGE_S
                and self._ndt_recv_count >= NDT_STABLE_COUNT)

    def try_update_offset(self, robot_speed_mps: float) -> bool:
        """嘗試更新 xy + yaw offset；回傳是否真的更新了.

        多重閘門依序把關，任一不過就不更新（保留舊 offset）：
        NDT 穩定 → 有同步資料 → 距上次更新已過冷卻時間 → 機器人靜止 → delta 合理。
        """
        if not self.is_ndt_stable():
            return False
        if (self._sync_odom_xy is None or self._last_ndt_xy is None
                or self._sync_odom_yaw is None or self._last_ndt_yaw is None):
            return False
        now = time.monotonic()
        # 冷卻時間：已有 offset 就不必每幀重算，每 5 秒重抓一次即可（省力且避免抖動）。
        if (self._offset is not None
                and now - self._last_offset_update_t < OFFSET_UPDATE_INTERVAL_S):
            return False
        # 速度閘門：移動中 NDT 配準結果不可靠（motion blur / 配準延遲），只在靜止時取樣。
        if abs(robot_speed_mps) > SPEED_GATE_MPS:
            return False

        ndt_x, ndt_y = self._last_ndt_xy
        sync_x, sync_y = self._sync_odom_xy
        new_xy = (ndt_x - sync_x, ndt_y - sync_y)
        new_yaw = _angle_diff(self._last_ndt_yaw, self._sync_odom_yaw)

        if self._offset is None:
            self._log_info(
                f"[OFFSET] 初始化 xy=({new_xy[0]:.3f},{new_xy[1]:.3f}) "
                f"yaw={math.degrees(new_yaw):.1f}°"
            )
        else:
            # delta 閘門：新舊 offset 落差過大通常是 NDT 跳到錯誤定位（jump），
            # 寧可拒絕更新沿用舊 offset，也不要讓一次飄移污染 robot pose。
            xy_delta = math.hypot(new_xy[0] - self._offset[0], new_xy[1] - self._offset[1])
            if xy_delta > OFFSET_DELTA_REJECT_M:
                self._log_warn(
                    f"[OFFSET] xy_delta={xy_delta:.3f}m > {OFFSET_DELTA_REJECT_M}m，拒絕更新"
                )
                self._last_offset_update_t = now
                return False
            if self._yaw_offset is not None:
                yaw_delta = abs(_angle_diff(new_yaw, self._yaw_offset))
                if yaw_delta > OFFSET_YAW_DELTA_REJECT_RAD:
                    self._log_warn(
                        f"[OFFSET] yaw_delta={math.degrees(yaw_delta):.1f}° > "
                        f"{math.degrees(OFFSET_YAW_DELTA_REJECT_RAD):.1f}°，拒絕更新"
                    )
                    self._last_offset_update_t = now
                    return False
            if xy_delta > 0.01:
                self._log_info(
                    f"[OFFSET] 更新 xy=({new_xy[0]:.3f},{new_xy[1]:.3f}) "
                    f"yaw={math.degrees(new_yaw):.1f}° xy_delta={xy_delta:.3f}m"
                )

        self._offset = new_xy
        self._yaw_offset = new_yaw
        self._last_offset_update_t = now
        return True

    def get_robot_pose_in_map(
        self, odom_x: float, odom_y: float, odom_yaw: float,
    ) -> RobotPose:
        """計算機器人在 map frame 的位姿，依優先順序選擇來源.

        優先序：cached offset（最穩，NDT 暫失也能跑）> NDT 直讀（有穩定但還沒 cache
        到 offset 時的過渡）> 純 odom（NDT 從未穩定的 fallback，goal 亦假設在 odom frame）。
        """
        if self._offset is not None:
            ox, oy = self._offset
            yaw = odom_yaw + (self._yaw_offset or 0.0)
            return RobotPose(
                x=odom_x + ox, y=odom_y + oy, yaw=yaw,
                source="odom+offset",
            )
        if self.is_ndt_stable() and self._last_ndt_xy is not None:
            yaw = self._last_ndt_yaw if self._last_ndt_yaw is not None else odom_yaw
            return RobotPose(
                x=self._last_ndt_xy[0], y=self._last_ndt_xy[1], yaw=yaw,
                source="ndt_direct",
            )
        return RobotPose(x=odom_x, y=odom_y, yaw=odom_yaw, source="odom_only")

    @property
    def offset(self) -> tuple[float, float] | None:
        return self._offset

    @property
    def yaw_offset(self) -> float | None:
        return self._yaw_offset

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
