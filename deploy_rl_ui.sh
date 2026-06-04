#!/bin/bash
# deploy_rl_ui.sh — 一鍵啟動 rover_rl 完整棧 + 同一終端機前景顯示繁中 TUI 儀表板
#
# 為何要這支腳本：ros2 launch 的子行程沒有 TTY，curses 不能塞進 launch 裡跑。
# 所以這裡把整個 launch 丟背景（log 導到檔案，保持畫面乾淨），前景直接跑 status_tui。
# 離開 TUI（按 q）或 Ctrl+C 時自動收掉整個棧。
#
# 用法：deploy_rl              （= 此腳本，含 UI）
#       deploy_rl initial_mode:=nav   （參數會原樣轉給 launch）
#       deploy_rl_raw          （舊行為：純 launch 滾動 log，無 UI）
# 注意：不可用 `set -u`，ROS 的 setup.bash 會引用未設定變數而報錯

source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash
source ~/rover_rl/install/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

mkdir -p ~/rover_rl/logs
LOG=~/rover_rl/logs/deploy_$(date +%Y%m%d_%H%M%S).log

cleanup() {
    trap '' INT TERM EXIT
    echo ""
    echo "[deploy_rl] 停止 rover_rl 棧…"
    bash ~/rover_rl_stop.sh >/dev/null 2>&1
    echo "[deploy_rl] 完整 log 保存於：$LOG"
}
trap cleanup EXIT INT TERM

echo "[deploy_rl] 啟動完整棧（log → $LOG）…"
ros2 launch rover_rl_bringup deploy_full.launch.py "$@" >"$LOG" 2>&1 &
LAUNCH_PID=$!

echo -n "[deploy_rl] 等待 policy_node 啟動"
for _ in $(seq 1 40); do
    if ros2 node list 2>/dev/null | grep -q rover_rl_policy; then
        echo " ✓"; break
    fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo ""; echo "[deploy_rl] ⚠ launch 提早結束，請看 log：$LOG"; exit 1
    fi
    echo -n "."; sleep 0.5
done

# 前景跑 TUI（curses 取得真實 TTY）；離開後觸發 cleanup
ros2 run rover_rl_inference status_tui
