"""VLP-16 點雲 → 72-bin sweep（與訓練側 wd_like_sweep_72 對齊）.

訓練側來源：source/.../obs_functions.py::wd_like_sweep_72
預設參數：r_max=20.0, r_robot=0.35, r_min=0.9, z_filter=0.5, num_bins=72
正規化：(d - r_robot) / (r_max - r_robot), clip [0, 1]
"""
from __future__ import annotations

import math

import numpy as np


def lidar_sweep_72_real(
    points_xyz: np.ndarray,
    r_max: float = 20.0,
    r_robot: float = 0.35,
    r_min: float = 0.9,
    z_filter: float = 0.5,
    num_bins: int = 72,
    yaw_offset: float = 0.0,
) -> np.ndarray:
    """轉點雲 → 72-bin normalized sweep.

    Args:
        points_xyz: [R, 3]，sensor frame (x=前, y=左, z=上)。NaN 自動過濾。
        r_max: 最遠距離 (m)
        r_robot: 機器人半徑 (m)，正規化時扣除
        r_min: 硬體盲區 (m)，VLP-16 實測 0.9
        z_filter: |z_hit| > z_filter 視為地板/天花板過濾
        num_bins: 角度 bin 數
        yaw_offset: sensor → base_link yaw 補償 (rad)
    """
    if points_xyz.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    pts = points_xyz.astype(np.float32, copy=False)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    if z_filter > 0:
        pts = pts[np.abs(pts[:, 2]) <= z_filter]
        if pts.size == 0:
            return np.ones(num_bins, dtype=np.float32)

    x, y = pts[:, 0], pts[:, 1]
    r = np.sqrt(x * x + y * y)
    mask = (r >= r_min) & (r <= r_max)
    if not mask.any():
        return np.ones(num_bins, dtype=np.float32)
    r = r[mask]
    x = x[mask]
    y = y[mask]

    theta = np.arctan2(y, x) + yaw_offset
    theta = np.mod(theta + math.pi, 2.0 * math.pi)
    bin_idx = np.minimum(
        (theta / (2.0 * math.pi) * num_bins).astype(np.int32),
        num_bins - 1,
    )

    sweep = np.full(num_bins, r_max, dtype=np.float32)
    for b in range(num_bins):
        m = bin_idx == b
        if m.any():
            sweep[b] = r[m].min()

    denom = max(r_max - r_robot, 1e-6)
    normalized = np.clip((sweep - r_robot) / denom, 0.0, 1.0)
    return normalized.astype(np.float32)


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → [R, 3] float32 numpy.

    純 numpy 解 PointCloud2，不依賴 sensor_msgs_py（部分 distro 沒裝）。
    """
    import struct

    fmt_map = {1: "b", 2: "B", 3: "h", 4: "H", 5: "i", 6: "I", 7: "f", 8: "d"}
    size_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}

    offsets = {}
    fmts = {}
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            offsets[f.name] = f.offset
            fmts[f.name] = fmt_map[f.datatype]

    if not all(k in offsets for k in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)

    point_step = msg.point_step
    n = msg.width * msg.height
    data = bytes(msg.data)
    endian = "<" if not msg.is_bigendian else ">"

    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        base = i * point_step
        out[i, 0] = struct.unpack_from(endian + fmts["x"], data, base + offsets["x"])[0]
        out[i, 1] = struct.unpack_from(endian + fmts["y"], data, base + offsets["y"])[0]
        out[i, 2] = struct.unpack_from(endian + fmts["z"], data, base + offsets["z"])[0]
    return out
