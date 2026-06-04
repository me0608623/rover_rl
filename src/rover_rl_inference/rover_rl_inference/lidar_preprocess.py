"""VLP-16 點雲 → 72-bin sweep（與訓練側 wd_like_sweep_72 對齊）.

本檔是「純函式」前處理邏輯（不含任何 ROS 依賴），供 lidar_preprocessor_node
與 policy_node inline fallback 共用。把 PointCloud2 解出的 xyz 點雲壓成 RL policy
直接吃的 72-bin 正規化距離向量。

⚠️ 為何公式要與訓練端逐步對齊：policy 是在 Isaac Sim 用 wd_like_sweep_72 產生的 obs
分布上訓練的；部署端只要任一步（z_filter / binning / min-pool / 正規化常數）和訓練端
不同，real obs 就會落在訓練分布外（sim-to-real obs mismatch），policy 容易原地震/亂走。
因此這裡每個常數與運算順序都刻意比照 obs_functions.py::wd_like_sweep_72。

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
    *,
    motion_compensation: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Args:
        ...
        motion_compensation: (dx, dy, dyaw) in body frame — 估算 LiDAR 掃描期間
            機器人位移；補償掃描畸變（spot_rl/spot_obs_process.cpp delay_pose）。
            約等於 odom.twist × scan_duration。None 不補償。
    """
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
    # 無點雲 → 回傳全 1（正規化後 1.0 = 最遠/無回波），等同訓練端「沒看到障礙物」
    if points_xyz.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    pts = points_xyz.astype(np.float32, copy=False)
    pts = pts[np.isfinite(pts).all(axis=1)]  # 濾掉 NaN/Inf 點（VLP-16 無回波會塞 NaN）
    if pts.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    # 步驟①：z 過濾。|z| > z_filter 視為地板/天花板，丟掉只留近水平面的點，
    # 對齊訓練端對 ray_hits 的 |z_hit|>0.5 過濾，避免地面點被當成障礙
    if z_filter > 0:
        pts = pts[np.abs(pts[:, 2]) <= z_filter]
        if pts.size == 0:
            return np.ones(num_bins, dtype=np.float32)

    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    # Motion compensation：把掃描期間機器人移過的距離扣回（讓點雲對齊 "scan-end pose"）。
    # VLP-16 一圈掃描需時間，移動中機器人會使點雲產生畸變；參照
    # spot_rl/spot_obs_process.cpp 的 delay_pose 思路做幾何補償，讓 obs 更貼近真實當下位姿
    if motion_compensation is not None:
        dx, dy, dyaw = motion_compensation
        # 簡單 0th-order：把點雲整體往機器人前進方向移動 -dx, -dy
        # 不處理 dyaw 因為 atan2 binning 對小角度旋轉容忍度高（< 1 bin = 5°）
        x = x - dx
        y = y - dy
    # 步驟②：投影到水平面取距離 r，並濾掉盲區(<r_min)與超遠(>r_max)的點。
    # r_min=0.9 是 VLP-16 實測盲區（太近的點不可信），與訓練端一致
    r = np.sqrt(x * x + y * y)
    mask = (r >= r_min) & (r <= r_max)
    if not mask.any():
        return np.ones(num_bins, dtype=np.float32)
    r = r[mask]
    x = x[mask]
    y = y[mask]

    # 步驟③：角度 binning。atan2(y,x) 取每個點的方位角，平移到 [0, 2π) 後均分成 num_bins 格。
    # 與訓練端用同樣的 atan2 公式，確保 bin 與真實方位的對應關係一致
    theta = np.arctan2(y, x) + yaw_offset
    theta = np.mod(theta + math.pi, 2.0 * math.pi)
    bin_idx = np.minimum(
        (theta / (2.0 * math.pi) * num_bins).astype(np.int32),
        num_bins - 1,  # clamp 邊界，避免 theta=2π 時溢位到 num_bins
    )

    # 步驟④：min-pool。每個角度 bin 取落在其中的最近距離（最保守＝最靠近的障礙物），
    # 空 bin 維持 r_max（視為無障礙）。與訓練端同 bin 取最小一致
    sweep = np.full(num_bins, r_max, dtype=np.float32)
    for b in range(num_bins):
        m = bin_idx == b
        if m.any():
            sweep[b] = r[m].min()

    # 步驟⑤：正規化到 [0,1]。扣掉 r_robot（機器人半徑，0=貼身障礙）後除以可用量程，
    # 公式與常數 (r_robot=0.35, r_max=20) 必須與訓練端逐字相同，否則 obs 尺度錯位
    denom = max(r_max - r_robot, 1e-6)
    normalized = np.clip((sweep - r_robot) / denom, 0.0, 1.0)
    return normalized.astype(np.float32)


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → [R, 3] float32 numpy.

    純 numpy 解 PointCloud2，不依賴 sensor_msgs_py（部分 distro 沒裝）。
    依 msg.fields 動態找 x/y/z 的 byte offset 與 datatype，再用 struct 逐點解包。
    """
    import struct

    # PointField.datatype 列舉 → struct 格式字元 / 位元組大小
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

    point_step = msg.point_step  # 每點佔的位元組數
    n = msg.width * msg.height
    data = bytes(msg.data)
    endian = "<" if not msg.is_bigendian else ">"  # 依 msg 標記決定位元組序

    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        base = i * point_step
        out[i, 0] = struct.unpack_from(endian + fmts["x"], data, base + offsets["x"])[0]
        out[i, 1] = struct.unpack_from(endian + fmts["y"], data, base + offsets["y"])[0]
        out[i, 2] = struct.unpack_from(endian + fmts["z"], data, base + offsets["z"])[0]
    return out
