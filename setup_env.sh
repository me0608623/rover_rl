#!/bin/bash
# rover_rl 環境設定
# 本機 zenohd router 跑在 127.0.0.1:7447，連到 13 主機 router
# ROS 節點以 client 模式連到本機 router
#
# 使用方式：source ~/rover_rl/setup_env.sh
# ⚠️ 不要修改 RMW / ROS_DOMAIN_ID

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_DOMAIN_ID=55
export ZENOH_SESSION_CONFIG_URI=/home/aa/zenoh_session_local.json5

# ndt_ws 提供 ndt_localizer package
source /home/aa/ndt_ws/install/setup.bash 2>/dev/null

echo "✅ rover_rl 環境已載入 (Zenoh client → local router → 13)"
echo "   RMW=$RMW_IMPLEMENTATION  DOMAIN=$ROS_DOMAIN_ID"
