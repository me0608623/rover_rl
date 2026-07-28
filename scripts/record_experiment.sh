#!/usr/bin/env bash
# record_experiment.sh — A/B 模型比較專用的「獨立」數據錄製系統
#
# 設計原則：與 deploy_rl / deploy_rl_shell / deploy_full 完全解耦。
#   - 平常跑 deploy_rl_shell：照舊由 diag_logger 收每趟 CSV（本腳本不碰）。
#   - 要做 noise+DR vs clean 的比較實驗時：在「另一個 terminal」單獨跑此腳本，
#     額外錄一份三層 LiDAR + tf 的 rosbag，供 E1/E2/E3 離線分析與 replay。
#   - 與 diag CSV 靠「牆鐘時間」對齊（diag CSV 有 t_wall，rosbag 也記 wall time），
#     不需要任何 IPC、不需要改動既有棧。
#
# 用法：
#   scripts/record_experiment.sh <label> [--no-raw]
#     <label>    本段實驗標籤（會清特殊字元），例：noisy_narrow_t1 / clean_open_t3
#     --no-raw   不錄 /velodyne_points（原始點雲很大）。導航多趟比較建議帶此旗標，
#                /rover_rl/lidar_sweep_72 已足夠做 E2/E3；只有 E1 靜態雜訊量測才需要 raw。
#
# 停止：Ctrl+C（會正常關閉 bag）。
set -euo pipefail

LABEL="${1:-trial}"
NO_RAW=0
for a in "$@"; do [ "$a" = "--no-raw" ] && NO_RAW=1; done

# 清 label（空白/斜線→底線，僅留安全字元）
SAFE_LABEL=$(printf '%s' "$LABEL" | tr ' /' '__' | tr -cd 'A-Za-z0-9_.-')
[ -z "$SAFE_LABEL" ] && SAFE_LABEL=trial

# --- 環境（與 deploy 同一 zenoh domain，否則看不到 topic）---
source /opt/ros/humble/setup.bash
[ -f "$HOME/rover_rl/install/setup.bash" ] && source "$HOME/rover_rl/install/setup.bash"
[ -f "$HOME/rover2_ws/install/setup.bash" ] && source "$HOME/rover2_ws/install/setup.bash"
source "$HOME/rover_rl/setup_env.sh"

# --- 輸出 ---
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$HOME/rover_rl/logs/experiments/${TS}_${SAFE_LABEL}"
mkdir -p "$OUTDIR"

# --- 要錄的 topic ---
TOPICS=(
  /rover_rl/lidar_sweep_72      # L1：policy 真正吃的 72-bin（E2/E3 核心）
  /rover_rl_policy/obs_debug    # L2：完整 obs（需 deploy 時 publish_obs_debug:=true 才有內容）
  /rover_rl_policy/status       # 三層速度/延遲/model 等 JSON
  /tf /tf_static               # 車姿合成
  /odom                        # 里程計
  /rover_rl/trail              # 已走軌跡(nav_msgs/Path, odom frame)：bag replay 可直接在 RViz 重播
  /goal_pose /global_path      # 導航目標
  /input/nav_cmd_vel           # 實際送底盤指令
)
[ "$NO_RAW" -eq 0 ] && TOPICS=(/velodyne_points "${TOPICS[@]}")  # L0：原始點雲（E1 才需要）

# --- metadata ---
cat > "$OUTDIR/meta.txt" <<EOF
label: $LABEL
started: $(date -Iseconds)
no_raw: $NO_RAW
ros_domain_id: ${ROS_DOMAIN_ID:-unset}
rmw: ${RMW_IMPLEMENTATION:-unset}
topics: ${TOPICS[*]}
EOF

echo "=== 實驗錄製（獨立系統，不影響 deploy）==="
echo "輸出：$OUTDIR/bag"
echo "topic：${TOPICS[*]}"
[ "$NO_RAW" -eq 1 ] && echo "（--no-raw：略過 /velodyne_points）"

# 連線健檢
if ! timeout 5 ros2 topic list >/dev/null 2>&1; then
  echo "✗ ros2 topic list 逾時 —— zenoh router 沒上線或環境沒對齊，先解決再錄。" >&2
  exit 1
fi
# obs_debug 提醒（沒 publisher 仍可錄，只是該 topic 空）
if ! timeout 3 ros2 topic info /rover_rl_policy/obs_debug 2>/dev/null | grep -qE "Publisher count: [1-9]"; then
  echo "⚠ /rover_rl_policy/obs_debug 無 publisher：要錄完整 obs，deploy 時加 publish_obs_debug:=true（不需改 deploy_rl_shell，當參數帶入即可）。"
fi

echo "（Ctrl+C 停止錄製）"
exec ros2 bag record -o "$OUTDIR/bag" "${TOPICS[@]}"
