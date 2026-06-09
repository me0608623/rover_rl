"""SubgoalSelector — 從 Path 或 PoseStamped 選出當前 sub-goal.

支援來源：
  A. PoseStamped (/goal_pose)         — 單一目標，直接用
  B. nav_msgs/Path (/global_path)      — 多 waypoint，取 lookahead carrot

Lookahead carrot 策略（rover2_ws + 經典 pure-pursuit）：
  1. 找 path 上離機器人最近的 waypoint i*
  2. 從 i* 起往前走，找第一個距離 >= lookahead_m 的 waypoint 當 carrot
  3. 若 path 已到尾端仍 < lookahead，回傳最後一個（即 final goal）
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SubgoalChoice:
    x: float
    y: float
    frame: str
    source: str   # 'goal_pose', 'path_lookahead', 'path_final'
    index: int    # waypoint idx（path 來源時有意義；單 goal 為 -1）


class SubgoalSelector:
    def __init__(self, lookahead_m: float = 2.0):
        self.lookahead_m = float(lookahead_m)
        # 由 ROS 端餵入：list of (x, y) + frame
        self._path_xy: list[tuple[float, float]] = []
        self._path_frame: str = ""
        self._single_xy: tuple[float, float] | None = None
        self._single_frame: str = ""
        # path 來源優先還是 single goal 優先？預設 single_goal 優先（user manual goal 蓋過 planner）
        self.prefer_path: bool = False
        # 上次 path 選取的索引（給儀表板畫地鐵式進度條用）：
        # nearest = 車目前在 path 上最近的 waypoint；carrot = 往前 lookahead 的瞬時目標
        self.last_nearest_i: int = -1
        self.last_carrot_i: int = -1

    def path_len(self) -> int:
        return len(self._path_xy)

    def set_path(self, points: list[tuple[float, float]], frame: str) -> None:
        self._path_xy = list(points)
        self._path_frame = frame

    def clear_path(self) -> None:
        self._path_xy = []
        self._path_frame = ""
        self.last_nearest_i = -1
        self.last_carrot_i = -1

    def set_single_goal(self, x: float, y: float, frame: str) -> None:
        self._single_xy = (x, y)
        self._single_frame = frame

    def clear_single_goal(self) -> None:
        self._single_xy = None
        self._single_frame = ""

    def has_target(self) -> bool:
        return self._single_xy is not None or len(self._path_xy) > 0

    def select(self, robot_x: float, robot_y: float) -> SubgoalChoice | None:
        """機器人位置在 robot frame 對應的 path/goal frame。回傳當前 sub-goal."""
        # prefer_path（收到 path 時設 True）→ planner 路徑蓋過手動 single goal；
        # 否則只在沒有 single goal 時才退回吃 path
        prefer_path = self.prefer_path or self._single_xy is None
        if prefer_path and self._path_xy:
            return self._select_from_path(robot_x, robot_y)
        if self._single_xy is not None:
            return SubgoalChoice(
                x=self._single_xy[0], y=self._single_xy[1],
                frame=self._single_frame, source="goal_pose", index=-1,
            )
        if self._path_xy:
            return self._select_from_path(robot_x, robot_y)
        return None

    def _select_from_path(self, rx: float, ry: float) -> SubgoalChoice:
        # 1) 找最近 waypoint
        nearest_i = 0
        best_d = float("inf")
        for i, (px, py) in enumerate(self._path_xy):
            d = (px - rx) ** 2 + (py - ry) ** 2
            if d < best_d:
                best_d = d
                nearest_i = i
        self.last_nearest_i = nearest_i

        # 2) 從最近點往前找第一個距離 >= lookahead 的點當 carrot：
        #    給 policy 一個「夠遠、在路徑上」的瞬時目標，避免直奔終點抄近路撞牆，
        #    也避免目標太近導致原地抖動（pure-pursuit 的核心）。從 nearest_i 起算
        #    保證 carrot 永遠在前方、不會倒退選到已走過的 waypoint。
        for i in range(nearest_i, len(self._path_xy)):
            px, py = self._path_xy[i]
            if math.hypot(px - rx, py - ry) >= self.lookahead_m:
                self.last_carrot_i = i
                return SubgoalChoice(
                    x=px, y=py, frame=self._path_frame,
                    source="path_lookahead", index=i,
                )
        # 3) 走到底還沒達 lookahead → 取最後一點（final goal）
        self.last_carrot_i = len(self._path_xy) - 1
        px, py = self._path_xy[-1]
        return SubgoalChoice(
            x=px, y=py, frame=self._path_frame,
            source="path_final", index=len(self._path_xy) - 1,
        )
