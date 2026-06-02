from setuptools import setup

package_name = "rover_rl_inference"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "torch"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="SA1_v2 RL policy inference for rover",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "policy_node = rover_rl_inference.policy_node:main",
            "lidar_preprocessor = rover_rl_inference.lidar_preprocessor_node:main",
            "bev_play = rover_rl_inference.bev_play_node:main",
            "ros_smoke_test = rover_rl_inference.ros_smoke_test:main",
            "export_policy = rover_rl_inference.export_policy:main",
            "routing_to_path = rover_rl_inference.routing_to_path:main",
            "routing_click_bridge = rover_rl_inference.routing_click_bridge:main",
            "diag_logger = rover_rl_inference.diag_logger_node:main",
            "analyze_diag = rover_rl_inference.analyze_diag:main",
        ],
    },
)
