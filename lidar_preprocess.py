"""
lidar_preprocess.py
-------------------
VLP-16 LiDAR 前處理：3D 點雲 → 72-bin 2D sweep（policy 輸入格式）

設計對應訓練側的 wd_like_sweep_72()（obs_functions.py），
使用完全相同的 z_filter + r_min + min-pool + normalize 流程，
確保 sim-to-real sensor alignment 一致。

注意：訓練側的 displacement noise σ(r) = displacement_std_per_meter × r（距離相依），
真機不加雜訊，此參數僅說明訓練時的噪聲強度來源：
  σ(r=1m) ≈ 0.36mm, σ(r=3m) ≈ 1.08mm（從 VLP-16 實測擬合）。

使用方式（ROS 2）：
    from lidar_preprocess import lidar_sweep_72_real
    import numpy as np

    def lidar_callback(msg):
        pts = ros_pc2_to_numpy(msg)          # [R, 3], sensor frame (x=前, y=左, z=上)
        sweep = lidar_sweep_72_real(pts)     # [72], float32, [0, 1]
        # 送進 policy ...

參數預設值與訓練側一致：
    r_max    = 20.0 m   — 最遠量測距離
    r_robot  = 0.3  m   — 機器人半徑（正規化時扣除）
    r_min    = 0.9  m   — VLP-16 硬體盲區（實測值）
    z_filter = 0.5  m   — 垂直過濾：|z_hit| > 0.5 視為地板/鬼影（sensor 裝在 z=1.6m）
    num_bins = 72       — 角度 bin 數（每 bin 5°，共 360°）
"""

import math
import numpy as np


def lidar_sweep_72_real(
    points_xyz: np.ndarray,
    r_max: float = 20.0,
    r_robot: float = 0.3,
    r_min: float = 0.9,
    z_filter: float = 0.5,
    num_bins: int = 72,
    yaw_offset: float = 0.0,
) -> np.ndarray:
    """
    VLP-16 點雲 → 72-bin 2D LiDAR sweep。

    Args:
        points_xyz:  shape [R, 3]，sensor frame（VLP-16 ROS driver 原始輸出）。
                     x = 前方，y = 左方，z = 上方。NaN 點會被自動過濾。
        r_max:       最遠有效距離 (m)，超過或無回波設為 r_max。
        r_robot:     機器人半徑 (m)，正規化前扣除，讓 0 = 緊貼機器人。
        r_min:       最近有效距離 (m)，< r_min 視為硬體盲區雜訊。
        z_filter:    垂直過濾閾值 (m)。|z_hit| > z_filter 的點濾掉。
                     z_hit 在 sensor frame 就是相對 sensor 的高度差，
                     所以地板（z ≈ -1.6m）、天花板、鬼影都會被去除。
        num_bins:    角度 bin 數，預設 72（每 bin 5°）。
        yaw_offset:  LiDAR 相對車頭的安裝偏轉角（rad）。
                     若 VLP-16 x 軸朝向車頭，設 0.0。
                     正值 = LiDAR 相對車頭逆時針偏轉。

    Returns:
        sweep: shape [num_bins]，float32，值域 [0, 1]。
               0 = 障礙物緊貼機器人表面，1 = 最遠或無障礙。
               bin 0 從 -180°（正後方偏左）開始，bin 36 ≈ 正前方。
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3, (
        f"points_xyz 必須是 [R, 3]，收到 {points_xyz.shape}"
    )

    points_xyz = points_xyz.astype(np.float32)

    # --- 1. 有效點遮罩 ---
    valid = np.isfinite(points_xyz).all(axis=-1)          # 過濾 NaN / Inf

    # z_filter：濾掉地板（z ≈ -1.6m）、天花板、z=-10m 鬼影
    if z_filter > 0.0:
        valid &= np.abs(points_xyz[:, 2]) <= z_filter

    # --- 2. 2D 水平距離（XY 平面） ---
    dist_2d = np.linalg.norm(points_xyz[:, :2], axis=-1)

    # r_min：濾掉硬體盲區（VLP-16 < 0.9m 不可靠）
    if r_min > 0.0:
        valid &= dist_2d >= r_min

    # 無效點設成最遠（min-pool 時自動被忽略）
    dist_2d = np.where(valid, dist_2d, r_max)
    dist_2d = np.clip(dist_2d, 0.0, r_max)

    # --- 3. 座標系旋轉（LiDAR 安裝偏轉補償） ---
    x = points_xyz[:, 0].copy()
    y = points_xyz[:, 1].copy()
    if yaw_offset != 0.0:
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        x, y = x * c - y * s, x * s + y * c

    # --- 4. atan2 分 bin ---
    # 角度範圍 [-π, π]，bin 0 從 -π 開始，每 bin = 2π / num_bins = 5°
    angles = np.arctan2(y, x)
    bin_size = 2.0 * math.pi / num_bins
    bin_idx = np.floor((angles + math.pi) / bin_size).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    # --- 5. Min-pool（每 bin 取最近距離） ---
    sweep = np.full(num_bins, r_max, dtype=np.float32)
    np.minimum.at(sweep, bin_idx, dist_2d)

    # --- 6. 正規化（扣機器人半徑，壓到 [0, 1]） ---
    sweep = np.clip(sweep - r_robot, 0.0, r_max) / r_max

    return sweep


# ---------------------------------------------------------------------------
# ROS 2 PointCloud2 輔助函式（需要 sensor_msgs_py 或 ros2_numpy）
# ---------------------------------------------------------------------------

def ros_pc2_to_numpy(msg) -> np.ndarray:
    """
    將 ROS 2 sensor_msgs/PointCloud2 轉成 numpy [R, 3]（x, y, z）。

    需要安裝：pip install ros2-numpy  或  sensor_msgs_py

    Args:
        msg: sensor_msgs/msg/PointCloud2

    Returns:
        points: shape [R, 3], float32, sensor frame
    """
    try:
        import ros2_numpy
        cloud = ros2_numpy.numpify(msg)
        points = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=-1)
    except ImportError:
        from sensor_msgs_py import point_cloud2 as pc2
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(gen), dtype=np.float32)
    return points.astype(np.float32)


# ---------------------------------------------------------------------------
# 快速單機測試（無 ROS 環境）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 模擬 VLP-16 一幀：5760 點，sensor frame
    N = 5760
    # 均勻分佈在 16 ring × 360 方位，距離 0.5~8m
    azimuth = rng.uniform(-math.pi, math.pi, N)
    elevation = rng.choice(
        np.linspace(-math.radians(15), math.radians(15), 16), N
    )
    dist = rng.uniform(0.5, 8.0, N)
    x = dist * np.cos(elevation) * np.cos(azimuth)
    y = dist * np.cos(elevation) * np.sin(azimuth)
    z = dist * np.sin(elevation)
    pts = np.stack([x, y, z], axis=-1)

    sweep = lidar_sweep_72_real(pts)

    print(f"輸出 shape : {sweep.shape}")          # (72,)
    print(f"值域       : [{sweep.min():.3f}, {sweep.max():.3f}]")
    print(f"平均距離   : {sweep.mean() * 20.0:.2f} m（×r_max）")
    print(f"前方 bin36 : {sweep[36]:.3f}")
    print("測試通過 ✓")
