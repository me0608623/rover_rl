"""RViz Marker 視覺化 — debug 用，看 policy 想做什麼.

發到 `/rover_rl/markers`（MarkerArray）：
  - 0: 機器人圓圈（綠）
  - 1: subgoal in body frame（紅球）
  - 2: policy 預期速度向量（藍箭頭）
  - 3: LiDAR 最近 bin（黃線）
  - 10+: subgoal source 文字（白）

座標系：base_footprint（body local），所有 marker 都在這個 frame。
"""
from __future__ import annotations

import math

from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r = r
    c.g = g
    c.b = b
    c.a = a
    return c


def build_marker_array(
    *,
    frame_id: str,
    stamp,
    robot_radius: float,
    subgoal_xy: tuple[float, float] | None,
    subgoal_source: str | None,
    cmd_v: float,
    cmd_w: float,
    nearest_lidar_bin: tuple[float, float] | None,  # (angle_rad, distance_m)
    mode: str,
) -> MarkerArray:
    arr = MarkerArray()

    # 0) 機器人圓圈
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "rover_rl"
    m.id = 0
    m.type = Marker.CYLINDER
    m.action = Marker.ADD
    m.pose.position.x = 0.0
    m.pose.position.y = 0.0
    m.pose.position.z = 0.0
    m.pose.orientation.w = 1.0
    m.scale.x = 2 * robot_radius
    m.scale.y = 2 * robot_radius
    m.scale.z = 0.05
    m.color = _color(0.2, 0.9, 0.2, 0.4)
    arr.markers.append(m)

    # 1) Subgoal
    if subgoal_xy is not None:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "rover_rl"
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(subgoal_xy[0])
        m.pose.position.y = float(subgoal_xy[1])
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = 0.25
        m.scale.y = 0.25
        m.scale.z = 0.25
        m.color = _color(0.9, 0.2, 0.2, 0.9)
        arr.markers.append(m)

    # 2) Policy 速度向量（箭頭從原點指向預期速度）
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "rover_rl"
    m.id = 2
    m.type = Marker.ARROW
    m.action = Marker.ADD
    p_start = Point(); p_start.x = 0.0; p_start.y = 0.0; p_start.z = 0.2
    # 速度向量在 body frame：方向是 cmd_w（轉向）+ cmd_v（前進）。簡單顯示前向長度
    p_end = Point()
    p_end.x = float(cmd_v) * 1.0
    p_end.y = 0.0
    p_end.z = 0.2
    m.points.append(p_start)
    m.points.append(p_end)
    m.scale.x = 0.05   # 桿
    m.scale.y = 0.12   # 箭頭頭部寬
    m.scale.z = 0.12
    m.color = _color(0.2, 0.4, 0.95, 0.95)
    arr.markers.append(m)

    # 3) 最近 LiDAR bin（黃線）
    if nearest_lidar_bin is not None:
        angle, dist = nearest_lidar_bin
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "rover_rl"
        m.id = 3
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        p0 = Point(); p0.x = 0.0; p0.y = 0.0; p0.z = 0.1
        p1 = Point()
        p1.x = math.cos(angle) * dist
        p1.y = math.sin(angle) * dist
        p1.z = 0.1
        m.points.append(p0)
        m.points.append(p1)
        m.scale.x = 0.03
        m.color = _color(0.95, 0.95, 0.2, 0.8)
        arr.markers.append(m)

    # 10) Mode + source 文字
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "rover_rl"
    m.id = 10
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = 0.0
    m.pose.position.y = 0.0
    m.pose.position.z = 1.0
    m.scale.z = 0.3
    m.color = _color(1.0, 1.0, 1.0, 1.0)
    src = subgoal_source or "—"
    m.text = f"mode={mode}\nsrc={src}\nv={cmd_v:+.2f} w={cmd_w:+.2f}"
    arr.markers.append(m)

    return arr
