#!/bin/bash
# rover_rl_stop.sh — 停止並清除所有 rover_rl 相關節點
# 對應 deploy_rl alias（deploy_full / deploy_with_bev）

echo "[rover_rl_stop] 開始停止 rover_rl 棧..."

# ── Step 1: 對 ros2 launch 發 SIGINT（觸發 launch 的 graceful shutdown）──
LAUNCH_PIDS=$(pgrep -f "ros2 launch rover_rl_bringup" 2>/dev/null)
if [ -n "$LAUNCH_PIDS" ]; then
    echo "[rover_rl_stop] 發 SIGINT 給 launch 進程: $LAUNCH_PIDS"
    kill -SIGINT $LAUNCH_PIDS 2>/dev/null
    sleep 3
fi

# ── Step 2: 直接殺 rover_rl 相關 Python 節點（launch 子進程）──
ROS_NODE_PATTERNS=(
    "policy_node"
    "vo_safety"
    "recovery_supervisor"  # 漏掉會變孤兒 10Hz 灌 /input/nav_cmd_vel → 車動動停停
    "orca_safety"
    "pingpong_test"
    "diag_logger"
    "lidar_preprocessor"
    "bev_play"
    "routing_to_path"
    "routing_click_bridge"
    "routing_engine_node"
    "mapinfo_db_handler"
    "routes_visualization"
    "campusrover_mot_node"
    "mot_marker_node"
    "local_costmap_node"
    "global_costmap_node"
    "simple_map_publisher"
    "mppi_planner"        # 抓 mppi_planner_node / mppi_planner_0520_node（含重複多開實例）
    "dwa_planner"          # 消融實驗 baseline（controller=dwa）
    "path_following"       # 消融實驗 baseline（controller=pid/pid_vo）
)

for pattern in "${ROS_NODE_PATTERNS[@]}"; do
    PIDS=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "[rover_rl_stop] 停止 $pattern (PID: $PIDS)"
        kill -SIGTERM $PIDS 2>/dev/null
    fi
done

sleep 2

# ── Step 3: 強制清除仍存活的節點 ──
for pattern in "${ROS_NODE_PATTERNS[@]}"; do
    PIDS=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "[rover_rl_stop] 強制終止 $pattern (PID: $PIDS)"
        kill -SIGKILL $PIDS 2>/dev/null
    fi
done

# ── Step 3.5: 清除卡住的 ros2 CLI 輔助進程（殘留 service call / topic pub / 掃描腳本）──
# 這些不是節點而是命令列工具：service 消失後會卡在等待 → 殘留並洗版「逾時」。
# 多 stack 疊跑、手動測試或路徑掃描留下的最常見（Step 2 的 routing_to_path pattern 只抓得到
# 呼叫 /routing_to_path/routing_call 的，抓不到呼叫 /generation_path 的，故在此補齊）。
# 用 RoutingPath/generation_path/working_floor 等 routing 特徵字串精準抓，不誤殺其他 ros2 指令。
STRAY_PATTERNS=(
    "ros2 service call.*RoutingPath"      # generation_path / routing_call（RoutingPath.srv）
    "ros2 service call.*generation_path"
    "ros2 service call.*/routing_"
    "ros2 topic pub.*working_floor"
    "ros2 topic pub.*global_path"
    "sweep_routes"                         # 路徑掃描腳本
    "bin/ros2 param dump"                  # deploy_rl_shell snapshot 的 param dump（zenoh 上會 hang → 卡住 orphan）
    "status_tui"                           # 前景 TUI（deploy_rl_shell 沒 q 乾淨會殘留）
    "baseline_status_line"                 # deploy_baseline_shell 的前景狀態行
)
for pattern in "${STRAY_PATTERNS[@]}"; do
    PIDS=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "[rover_rl_stop] 清除殘留 CLI ($pattern): $PIDS"
        kill -SIGKILL $PIDS 2>/dev/null
    fi
done

# ── Step 4: 清除殘留 launch 父進程（多 stack 會有多個，pgrep 一次抓全部）──
LAUNCH_PIDS=$(pgrep -f "ros2 launch rover_rl_bringup" 2>/dev/null)
if [ -n "$LAUNCH_PIDS" ]; then
    echo "[rover_rl_stop] 強制終止 launch 進程: $LAUNCH_PIDS"
    kill -SIGKILL $LAUNCH_PIDS 2>/dev/null
fi

# ── Step 4.5: 清除 orphan deploy_rl_shell wrapper（重複重啟 / snapshot 卡住會殘留）──
# ⚠ 排除「本 stop 腳本自己的祖先鏈」：若本腳本是被某個 deploy_rl_shell 的 q-cleanup 呼叫的，
#   那個 shell 正在正常收尾，不能殺它（否則 trap 收尾中斷）。standalone 執行時祖先無 shell → 全清。
SELF_CHAIN=" "
_p=$$
while [ "${_p:-0}" -gt 1 ]; do
    SELF_CHAIN="$SELF_CHAIN$_p "
    _p=$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ')
    [ -z "$_p" ] && break
done
for pid in $(pgrep -f "deploy_rl_shell.sh" 2>/dev/null); do
    case "$SELF_CHAIN" in *" $pid "*) continue ;; esac   # 跳過自己的祖先鏈（正在 q-cleanup 的 shell）
    echo "[rover_rl_stop] 清除 orphan deploy_rl_shell wrapper: $pid"
    kill -SIGKILL "$pid" 2>/dev/null
done

sleep 1

# ── Step 4.7: 清掉 RViz 殘留路徑（補發空 Path）──
# RViz Path display 會無限期保留最後一則訊息：publisher 死掉線也不會消失。
# 只清「rover_rl 自己擁有、停棧後就沒人發」的 topic：
#   /rover_rl/trail  policy_node 的已走軌跡
#   /global_path     routing_to_path 的規劃路徑（demo.rviz 的紅線）
# ⚠ 不清 /traj —— 那是 ndt_localizer_node 發的，NDT 不歸本腳本管（deploy_rl 也不啟 NDT）；
#   NDT 還活著時清了會被下一則蓋回來。要清 /traj 請用 ndt_stop / deploy_full_stop。
# 必須在殺完節點之後做，否則還活著的 node 會立刻蓋回去。
(
    source /opt/ros/humble/setup.bash 2>/dev/null
    source "$HOME/rover_rl/install/setup.bash" 2>/dev/null
    source "$HOME/rover_rl/setup_env.sh" >/dev/null 2>&1
    # -r 2 連發 ~3 秒（約 6 則），涵蓋 zenoh discovery 建連時間（RViz 常在別台）；
    # 兩個 topic 併發送，避免 3s×N 拉長 stop 時間；timeout 防 zenoh 卡住 stop 流程。
    for t in /rover_rl/trail /global_path; do
        timeout 3 ros2 topic pub -r 2 "$t" nav_msgs/msg/Path \
            "{header: {frame_id: 'odom'}, poses: []}" >/dev/null 2>&1 &
    done
    wait
)   # timeout 到期回傳 124 屬正常（連發滿 3 秒），故不用 && 串接訊息
echo "[rover_rl_stop] 已補發空 Path 清除 RViz 殘留路徑 (/rover_rl/trail, /global_path)"

# ── Step 5: 確認清除結果（含殘留節點 / CLI / 殭屍）──
# 殭屍（Z 狀態）是已死待父進程回收；上面殺掉 launch 父進程後，殘餘殭屍會被 init 回收。
ZOMBIES=$(ps -eo pid,stat,cmd 2>/dev/null | awk '$2 ~ /Z/ && /routing_engine|routing_to_path|policy_node|vo_safety|generation_path/ {print $1}')
if [ -n "$ZOMBIES" ]; then
    echo "[rover_rl_stop] 偵測到殭屍進程（待 init 回收，通常數秒內自動消失）: $ZOMBIES"
fi
REMAINING=$(pgrep -f "rover_rl_policy\|vo_safety\|recovery_supervisor\|orca_safety\|pingpong_test\|diag_logger\|rover_rl_lidar\|rover_rl_bev\|routing_to_path\|routing_click\|routing_engine_node\|mapinfo_db_handler\|mppi_planner\|dwa_planner\|path_following" 2>/dev/null)
STRAY=$(pgrep -f "ros2 service call.*RoutingPath\|ros2 service call.*generation_path\|sweep_routes\|bin/ros2 param dump\|status_tui\|baseline_status_line" 2>/dev/null)
# orphan wrapper（排除自己祖先鏈）也納入殘留判定
STRAY_SHELL=""
for pid in $(pgrep -f "deploy_rl_shell.sh" 2>/dev/null); do
    case "$SELF_CHAIN" in *" $pid "*) continue ;; esac
    STRAY_SHELL="$STRAY_SHELL $pid"
done
[ -n "$STRAY_SHELL" ] && STRAY="$STRAY $STRAY_SHELL"
if [ -z "$REMAINING" ] && [ -z "$STRAY" ]; then
    echo "[rover_rl_stop] ✅ 所有 rover_rl 節點與殘留 CLI 已清除"
else
    [ -n "$REMAINING" ] && echo "[rover_rl_stop] ⚠️  仍有殘留節點: $REMAINING"
    [ -n "$STRAY" ] && echo "[rover_rl_stop] ⚠️  仍有殘留 CLI: $STRAY"
fi
