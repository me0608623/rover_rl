# rover_rl_bringup 是純啟動 (bringup) package：只放 launch 檔與參數，無 Python 執行碼。
# 實際節點 (policy_node / lidar_preprocessor / bev_play ...) 都在 rover_rl_inference。
import os
from glob import glob

from setuptools import setup

package_name = "rover_rl_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    # data_files: 指定 colcon build 時要安裝到 share/ 的資源檔
    data_files=[
        # ament index 註冊（讓 ROS 2 找得到此 package）
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # 安裝 launch / config / rviz 三類資源（用 glob 收整個資料夾）
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Bringup for rover_rl",
    license="Proprietary",
    # 無 console_scripts：本 package 不提供可執行節點，只負責 launch 編排
    entry_points={"console_scripts": []},
)
