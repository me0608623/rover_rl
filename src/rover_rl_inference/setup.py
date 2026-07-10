"""rover_rl_inference 的 ament_python package 設定。

entry_points 的 console_scripts 定義所有可 `ros2 run rover_rl_inference <name>`
啟動的可執行節點，是部署棧（deploy_*.launch.py）拉起各節點的依據。
"""
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
            # 核心推論：72-bin sweep → 79D obs → RNN policy → /input/nav_cmd_vel
            "policy_node = rover_rl_inference.policy_node:main",
            # LiDAR 前處理獨立節點：PointCloud2 → 72-bin 正規化 sweep（與 policy 解耦）
            "lidar_preprocessor = rover_rl_inference.lidar_preprocessor_node:main",
            # BEV 極座標可視化（純 debug，非 policy 輸入）→ /rover_rl/bev_image
            "bev_play = rover_rl_inference.bev_play_node:main",
            # ROS 通訊冒煙測試：上電前驗證 topic/型別接得通
            "ros_smoke_test = rover_rl_inference.ros_smoke_test:main",
            # 離線工具：把訓練 checkpoint 匯出成部署用 TorchScript（含 normalizer）
            "export_policy = rover_rl_inference.export_policy:main",
            # campusrover_routing service → /global_path topic 橋接（2Hz republish）
            "routing_to_path = rover_rl_inference.routing_to_path:main",
            # RViz「Publish Point」兩次點擊 → 呼叫 routing service 產生 path
            "routing_click_bridge = rover_rl_inference.routing_click_bridge:main",
            # 被動診斷記錄：訂閱各 topic 逐列寫 CSV，供事後分析晃動/不朝 goal
            "diag_logger = rover_rl_inference.diag_logger_node:main",
            # 離線分析 diag_logger 產生的 CSV（速度三層對比、延遲、晃動報告）
            "analyze_diag = rover_rl_inference.analyze_diag:main",
            # 彙整往返測試 session 指標 → SR/TO/CR 論文對比表（CSV/Markdown/LaTeX/wandb）
            "pingpong_report = rover_rl_inference.pingpong_report:main",
            # 繁中 curses 即時狀態儀表板（純訂閱純渲染，需真實 TTY）
            "status_tui = rover_rl_inference.status_tui_node:main",
            # VO 安全層：夾在 RL 與底盤間，用 LV-DOT 動態障礙做預測式避障濾波
            "vo_safety = rover_rl_inference.vo_safety_node:main",
            # Recovery Supervisor：RL 外層脫困 wrapper，保守接管 cmd_vel 後再交還 RL
            "recovery_supervisor = rover_rl_inference.recovery_supervisor_node:main",
            # 兩固定點往返避障測試：車停在 A/B 任一點即自動規劃往對向點，無限來回
            "pingpong_test = rover_rl_inference.pingpong_test_node:main",
        ],
    },
)
