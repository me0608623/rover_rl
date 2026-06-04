#!/bin/bash
# deploy_rl_shell.sh — 互動式：RL 棧背景 + 前景繁中 TUI 儀表板（給「人」在真實終端機用）
#
# 預設不啟 NDT / LV-DOT（enable_ndt:=false enable_lvdot:=false）——這兩個用 ndt / lv-dot alias 分開啟。
# 要一次全開（含 NDT+LV-DOT）請改用 deploy_all。
#
# 與 deploy_rl 的分工：
#   deploy_rl        = 純 ros2 launch（前景滾動 log），任何 shell 皆可，含 Claude 非互動環境
#   deploy_rl_shell  = 本腳本，背景 launch + 前景 curses TUI，需真實 TTY
#
# 為何要 TTY 守門：curses（status_tui）在非互動 shell（pipe / Claude Bash 工具）會卡住/亂碼。
# 故這裡先檢查 stdin/stdout 是否為終端機，不是就友善退出，不硬跑 curses。
#
# 用法：deploy_rl_shell                  （含 UI，按 q 離開即自動收棧）
#       deploy_rl_shell initial_mode:=nav  （參數原樣轉給 launch）

# TTY 守門：非互動環境直接退出，避免 curses 卡死
if [ ! -t 0 ] || [ ! -t 1 ]; then
    echo "[deploy_rl_shell] 這是互動式 UI，需在真實終端機（互動 shell）執行。"
    echo "  • 程式/Claude 中啟動完整棧 → 改用： deploy_rl"
    echo "  • 看即時狀態（JSON，可解析）→  ros2 topic echo /rover_rl_policy/status"
    echo "  • 停止整個棧 →                 deploy_rl_stop"
    exit 2
fi

source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash
source ~/rover_rl/install/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

mkdir -p ~/rover_rl/logs
LOG=~/rover_rl/logs/deploy_$(date +%Y%m%d_%H%M%S).log

cleanup() {
    trap - EXIT INT TERM          # 先解除自身 trap，避免重入（不用 '' 遮蔽，Ctrl+C 仍可中止收尾）
    echo ""
    echo "[deploy_rl_shell] 停止 rover_rl 棧…"
    bash ~/rover_rl_stop.sh >/dev/null 2>&1
    echo "[deploy_rl_shell] 完整 log 保存於：$LOG"
}
trap cleanup EXIT INT TERM

echo "[deploy_rl_shell] 啟動 RL 棧（不含 NDT/LV-DOT，log → $LOG）…"
# 預設 enable_ndt:=false enable_lvdot:=false → 只啟 RL 那層；NDT/LV-DOT 用 ndt + lv-dot alias 分開啟。
# 放在 "$@" 前面：user 傳的同名參數在後面會覆寫（ros2 launch 重複參數取最後值）。
ros2 launch rover_rl_bringup deploy_full.launch.py enable_ndt:=false enable_lvdot:=false "$@" >"$LOG" 2>&1 &
LAUNCH_PID=$!

echo -n "[deploy_rl_shell] 等待 policy_node 啟動"
for _ in $(seq 1 40); do
    if ros2 node list 2>/dev/null | grep -q rover_rl_policy; then
        echo " ✓"; break
    fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo ""; echo "[deploy_rl_shell] ⚠ launch 提早結束，請看 log：$LOG"; exit 1
    fi
    echo -n "."; sleep 0.5
done

# 前景跑 TUI（curses 取得真實 TTY）；離開後（q / Ctrl+C）觸發 cleanup
ros2 run rover_rl_inference status_tui
