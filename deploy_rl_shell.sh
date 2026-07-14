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

# ── Checkpoint 選單 + VO 詢問（抽到共用 deploy_select.sh，與 deploy_rl 同一套）──
# 設定 EXTRA_ARGS（model/config）與 VO_ARG（enable_vo:=…）。已在 TTY（本腳本上面已守門）。
# 命令列已帶 model_path:= / params_file:= / enable_vo:= 時各自跳過、尊重覆寫。
source ~/rover_rl/deploy_select.sh
DEPLOY_SELECT_TAG=deploy_rl_shell deploy_rl_select "$@"

# 預設只在 VO 模式啟用 Recovery Supervisor，取代 VO 內建倒退/脫困，
# 但保留 VO 其他避障/煞車行為。純 RL / ORCA 不自動串 recovery。
# 若使用者命令列已指定 enable_recovery:=...，尊重覆寫。
RECOVERY_ARG=()
if ! printf '%s\n' "$@" | grep -qE '^enable_recovery:='; then
    EFFECTIVE_VO_ARG="$VO_ARG"
    for arg in "$@"; do
        case "$arg" in
            enable_vo:=*) EFFECTIVE_VO_ARG="$arg" ;;
        esac
    done
    if [ "$EFFECTIVE_VO_ARG" = "enable_vo:=true" ]; then
        RECOVERY_ARG=("enable_recovery:=true")
        echo "[deploy_rl_shell] Recovery Supervisor：啟用（VO 模式，取代 VO 內建倒退/脫困）"
    else
        RECOVERY_ARG=("enable_recovery:=false")
        echo "[deploy_rl_shell] Recovery Supervisor：不啟用（非 VO，純 RL/ORCA 不改道）"
    fi
fi

# ── MPPI 靜態避障層詢問（RL 導航 + MPPI 靜態 + VO 動態，三層協作）──
# 啟用後 policy 輸出改道 → MPPI(static_guard，吃 local costmap 避靜態) → VO(動態) → mux。
# MPPI 吃 RL 意圖當 reference，只用 costmap 提早避靜態；動態仍交給 VO。
# 需 /campusrover_local_costmap 在發（enable_costmap 預設 true，需 velodyne 在跑）。
# 命令列已帶 enable_mppi:= 則跳過詢問、尊重覆寫。
MPPI_ARG=()
if ! printf '%s\n' "$@" | grep -qE '^enable_mppi:='; then
    echo "┌─ MPPI 靜態避障層（RL 導航 + MPPI 靜態 + VO 動態） ───────────"
    echo "│ 啟用後：policy → MPPI(靜態,吃 local costmap) → VO(動態) → mux。"
    echo "│ MPPI 吃 RL 意圖當 reference，只用 costmap 提早避靜態障礙；動態仍交給 VO。"
    echo "│ 需 /campusrover_local_costmap 在發（enable_costmap 預設開，需 velodyne 在跑）。"
    echo "│ 建議搭配 VO=啟用；首次請架空 + 低速測。"
    echo "└──────────────────────────────────────────────────────────────"
    read -rp "是否啟用 MPPI 靜態避障層？[y/N]（Enter=不啟用） " MPPI_SEL
    case "$MPPI_SEL" in
        [Yy]*)
            # MPPI 需 local costmap 才會輸出（gate 卡 get_costmap_data_）→ 明確一起帶上 enable_costmap:=true
            MPPI_ARG=("enable_mppi:=true" "enable_costmap:=true")
            echo "[deploy_rl_shell] MPPI 靜態避障層：啟用（RL→MPPI→VO 三層，一併啟 local costmap）"
            echo "[deploy_rl_shell] ⚠ MPPI 需 /campusrover_local_costmap → 確認 velodyne 在跑（costmap 吃點雲）"
            ;;
        *)
            echo "[deploy_rl_shell] MPPI 靜態避障層：不啟用"
            ;;
    esac
fi

# ── 往返測試詢問（兩固定點 A↔B 連續來回，測避障）──
# 把車手動開到 A/B 任一點停穩 → TUI 跳提示，按【空白鍵】開始往對向點來回。
# 中途切 manual/estop 即中斷，手動開回任一點停穩再按空白鍵重啟。
# 命令列已帶 enable_pingpong:= 則跳過此詢問、尊重覆寫。
PINGPONG_ARGS=()
if ! printf '%s\n' "$@" | grep -qE '^enable_pingpong:='; then
    echo "┌─ 兩固定點往返避障測試 ──────────────────────────────────────"
    echo "│ 車停在 A/B 任一點停穩 → TUI 提示按【空白鍵】開始往對向點，A↔B 來回。"
    echo "│ 需 NDT + routing 在跑（拓撲節點定位）；中途切 manual/estop 即中斷。"
    echo "└──────────────────────────────────────────────────────────────"
    read -rp "是否啟用往返測試？[Y/n]（Enter=啟用） " PP_SEL
    case "$PP_SEL" in
        [Nn]*) echo "[deploy_rl_shell] 往返測試：不啟用" ;;
        *)
            read -rp "  A 點節點名 [c24]： " PP_A; PP_A="${PP_A:-c24}"
            read -rp "  B 點節點名 [c27]： " PP_B; PP_B="${PP_B:-c27}"
            PINGPONG_ARGS=("enable_pingpong:=true" "pingpong_a:=$PP_A" "pingpong_b:=$PP_B")
            echo "[deploy_rl_shell] 往返測試：啟用（$PP_A ↔ $PP_B，到點後按空白鍵開始）"
            ;;
    esac
fi

# ── ros bag 錄製詢問（事後離線分析控制鏈/速度震盪用）──
# 只錄輕量控制鏈 + 狀態 topic（cmd_vel 各層 / status / odom / costmap / tf），
# 不錄點雲/影像 → 約數 MB/分。存到 ~/rover_rl/logs/bags/deploy_<時間>/。
RECORD_BAG=0
echo "┌─ 錄製 ros bag（事後分析速度震盪/控制鏈用）───────────────"
echo "│ 只錄控制鏈+狀態（cmd_vel 各層 / *_node/status / odom / costmap / tf），"
echo "│ 不錄點雲影像 → 輕量。存到 ~/rover_rl/logs/bags/。事後可交給 Claude 分析。"
echo "└──────────────────────────────────────────────────────────────"
read -rp "是否錄製 ros bag？[Y/n]（Enter=錄） " BAG_SEL
case "$BAG_SEL" in
    [Nn]*) echo "[deploy_rl_shell] ros bag：不錄" ;;
    *) RECORD_BAG=1; echo "[deploy_rl_shell] ros bag：錄製（事後可分析）" ;;
esac

source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash
source ~/rover_rl/install/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

mkdir -p ~/rover_rl/logs ~/rover_rl/logs/bags
TS=$(date +%Y%m%d_%H%M%S)
LOG=~/rover_rl/logs/deploy_$TS.log

cleanup() {
    trap - EXIT INT TERM          # 先解除自身 trap，避免重入（不用 '' 遮蔽，Ctrl+C 仍可中止收尾）
    echo ""
    # 先收尾 ros bag：SIGINT 讓 rosbag2 正常寫入 metadata.yaml 再退（直接 kill 會壞檔）
    if [ -n "$BAG_PID" ] && kill -0 "$BAG_PID" 2>/dev/null; then
        echo "[deploy_rl_shell] 收尾 ros bag…"
        kill -INT "$BAG_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8; do kill -0 "$BAG_PID" 2>/dev/null || break; sleep 0.5; done
        kill -9 "$BAG_PID" 2>/dev/null
    fi
    echo "[deploy_rl_shell] 停止 rover_rl 棧…"
    bash ~/rover_rl_stop.sh >/dev/null 2>&1
    echo "[deploy_rl_shell] 完整 log 保存於：$LOG"
    # 列出本次 session 建立的診斷 CSV（比 $LOG 更新的檔案 = 這次產生的）
    DIAG_CSVS=$(find ~/rover_rl/logs/diag/ -name "*.csv" -newer "$LOG" 2>/dev/null | sort)
    if [ -n "$DIAG_CSVS" ]; then
        echo "[deploy_rl_shell] 本次診斷記錄："
        echo "$DIAG_CSVS" | while read -r f; do echo "  • $f"; done
    fi
    if [ "$RECORD_BAG" = "1" ] && [ -d "$BAGDIR" ]; then
        echo "[deploy_rl_shell] ros bag 已存：$BAGDIR"
        [ -f "$PARAMS_SNAP" ] && echo "[deploy_rl_shell] 參數 snapshot：$PARAMS_SNAP（model/VO/MPPI/policy 全參數）"
        echo "    分析：ros2 bag info $BAGDIR   （把 bag + snapshot 路徑一起給 Claude）"
    fi
}
trap cleanup EXIT INT TERM

echo "[deploy_rl_shell] 啟動 RL 棧（不含 NDT/LV-DOT，log → $LOG）…"
# 預設 enable_ndt:=false enable_lvdot:=false → 只啟 RL 那層；NDT/LV-DOT 用 ndt + lv-dot alias 分開啟。
# $VO_ARG 來自上面互動詢問（預設 enable_vo:=true）；lv-dot 沒開時 VO 退化為放行+ω clamp，安全。
# $RECOVERY_ARG 只在有效 VO=true 時預設 enable_recovery:=true；純 RL 不啟動 recovery。
# 放在 "$@" 前面：user 傳的同名參數在後面會覆寫（ros2 launch 重複參數取最後值），如 enable_vo:=false。
ros2 launch rover_rl_bringup deploy_full.launch.py enable_ndt:=false enable_lvdot:=false "$VO_ARG" "$ORCA_ARG" "${RECOVERY_ARG[@]}" "${MPPI_ARG[@]}" rviz:=false "${EXTRA_ARGS[@]}" "${PINGPONG_ARGS[@]}" "$@" >"$LOG" 2>&1 &
LAUNCH_PID=$!

# ORCA 控制車模式：由 deploy_full.launch.py 的 enable_orca:=true 啟 orca_safety_node（在 RL 棧內），
# 不再背景跑旁觀 bystander。ORCA_ARG 由 deploy_select 設定（選 2 時 enable_orca:=true）。

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

# ── 起 ros bag（棧就緒後才錄，確保 topic 已存在）──
# 只錄輕量控制鏈 + 狀態 topic：能精確定位 (0.59,+1.1) 這種值在鏈的哪一段冒出、誰在搶。
# 刻意不錄 /velodyne_points、/camera/*（點雲影像太大）。
if [ "$RECORD_BAG" = "1" ]; then
    BAGDIR=~/rover_rl/logs/bags/deploy_$TS
    echo "[deploy_rl_shell] 開始錄 ros bag → $BAGDIR"
    ros2 bag record -o "$BAGDIR" \
        /input/nav_cmd_vel /rover_rl/cmd_vel_desired /rover_rl/cmd_vel_mppi \
        /rover_rl/cmd_vel_recovery_in /output/cmd_vel /cmd_vel \
        /rover_rl_policy/status /vo_safety_node/status /recovery_supervisor_node/status \
        /odom /rover_rl/lidar_sweep_72 /campusrover_local_costmap \
        /goal_pose /global_path /tf /tf_static \
        >"$LOG.bag.log" 2>&1 &
    BAG_PID=$!
    # ── 同時 dump 本次 run 的 model + policy/VO/MPPI/recovery 全參數 snapshot（事後對照用）──
    # 背景執行，不擋 TUI；ros2 param dump 需節點在線，故放在 policy 起來之後。
    PARAMS_SNAP=~/rover_rl/logs/bags/deploy_${TS}_params.yaml
    ( {
        echo "# ===== run snapshot $TS ====="
        echo "# 選單/命令列參數（含 model_path / config 選擇）:"
        echo "select_args: '${EXTRA_ARGS[*]}'"
        echo "vo_arg: '$VO_ARG'   mppi_arg: '${MPPI_ARG[*]}'   recovery_arg: '${RECOVERY_ARG[*]}'   static_avoid_arg: '${STATIC_AVOID_ARG[*]}'"
        for node in /rover_rl_policy /vo_safety_node /mppi_planner_node /recovery_supervisor_node; do
            echo ""
            echo "# ======================== $node ========================"
            # ⚠ ros2 param dump 在 zenoh 上可能 hang → 一律包 timeout，絕不卡死 subshell
            timeout 12 ros2 param dump "$node" 2>/dev/null || echo "  # (節點不存在 / dump 逾時)"
        done
      } > "$PARAMS_SNAP"
      echo "[deploy_rl_shell] 參數 snapshot 已存：$PARAMS_SNAP" ) &
fi

# 前景跑 TUI（curses 取得真實 TTY）；離開後（q / Ctrl+C）觸發 cleanup（含收尾 bag）
ros2 run rover_rl_inference status_tui
