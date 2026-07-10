#!/usr/bin/env bash
# SessionStart hook：檢查 ROS MCP 前置條件，注入提示給 Claude
# 純偵測，不啟動任何服務、不阻擋 session

# 1. rosbridge WebSocket（ROS MCP 連線端點）是否在 9090 監聽
if (exec 3<>/dev/tcp/127.0.0.1/9090) 2>/dev/null; then
  exec 3>&- 3<&-
  ROSBRIDGE="up"
else
  ROSBRIDGE="down"
fi

# 2. zenoh router service 是否 active
if systemctl is-active --quiet zenoh-router.service 2>/dev/null; then
  ZENOH="active"
else
  ZENOH="inactive"
fi

# 組提示文字
if [ "$ROSBRIDGE" = "up" ]; then
  MSG="ROS MCP 前置檢查：rosbridge(9090)=在線, zenoh-router=${ZENOH}。請在開始 ROS 相關工作前，先用 ros-mcp 的 connect_to_robot(ip=\"127.0.0.1\", port=9090) 連線，再用 ping_robots 確認 ROS 2 通訊正常。"
else
  MSG="ROS MCP 前置檢查：rosbridge(9090)=未在線, zenoh-router=${ZENOH}。ROS MCP 目前無法連線。若需操作 ROS，請先啟動 rosbridge：source ~/rover_rl/setup_env.sh && ros2 launch rosbridge_server rosbridge_websocket_launch.xml &，再用 connect_to_robot/ping_robots 確認。"
fi

# 用 jq 安全輸出 JSON（避免特殊字元壞掉）
jq -nc --arg ctx "$MSG" \
  '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $ctx}}'
