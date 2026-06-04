#!/bin/bash
# deploy_all：一鍵啟動「完整棧 deploy_full(NDT + policy + routing + costmap + MOT + bev) + LV-DOT 動態偵測」
# 兩個 launch 都丟背景、log 導檔（Claude / 非互動 shell 也能跑）。
# 參數原樣轉給 deploy_full，例：
#   deploy_all initial_mode:=idle      # 靜態先確認（車不動）
#   deploy_all                         # 預設 nav（給完 initialpose、NDT 收斂後車會動）
source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash 2>/dev/null
source ~/rover_rl/install/setup.bash
source ~/rover_rl/setup_env.sh >/dev/null 2>&1
set -u   # ROS setup.bash 不相容 nounset (AMENT_TRACE_SETUP_FILES unbound)，故 source 完才開

mkdir -p ~/rover_rl/logs
TS=$(date +%Y%m%d_%H%M%S)
DEPLOY_LOG=~/rover_rl/logs/deploy_${TS}.log
LVDOT_LOG=~/rover_rl/logs/lvdot_${TS}.log

echo "[deploy_all] 啟動 deploy_full (NDT+policy+routing+costmap+MOT+bev) → $DEPLOY_LOG"
nohup ros2 launch rover_rl_bringup deploy_full.launch.py "$@" > "$DEPLOY_LOG" 2>&1 &
echo "  deploy_full PID $!"

echo "[deploy_all] 等 10s 讓 NDT / TF 起來..."
sleep 10

echo "[deploy_all] 啟動 LV-DOT (LiDAR+相機融合, YOLO 走 GPU) → $LVDOT_LOG"
nohup ros2 launch onboard_detector run_detector.launch.py use_yolo:=true > "$LVDOT_LOG" 2>&1 &
echo "  LV-DOT PID $!"

echo ""
echo "[deploy_all] ✅ 兩棧已背景啟動"
echo "  ⚠ deploy_full 預設 initial_mode=nav → 用 RViz 給 initialpose、NDT 收斂後『車會動』；要靜態先確認就用：deploy_all initial_mode:=idle"
echo "  RViz 給 initialpose（2D Pose Estimate）讓 NDT 收斂"
echo "  看 log : tail -f $DEPLOY_LOG   |   tail -f $LVDOT_LOG"
echo "  停 全部: deploy_all_stop"
