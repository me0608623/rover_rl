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
    read -rp "是否啟用往返測試？[y/N]（Enter=不啟用） " PP_SEL
    case "$PP_SEL" in
        [Yy]*)
            read -rp "  A 點節點名 [c24]： " PP_A; PP_A="${PP_A:-c24}"
            read -rp "  B 點節點名 [c27]： " PP_B; PP_B="${PP_B:-c27}"
            PINGPONG_ARGS=("enable_pingpong:=true" "pingpong_a:=$PP_A" "pingpong_b:=$PP_B")
            echo "[deploy_rl_shell] 往返測試：啟用（$PP_A ↔ $PP_B，到點後按空白鍵開始）"
            ;;
        *) echo "[deploy_rl_shell] 往返測試：不啟用" ;;
    esac
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
    # 列出本次 session 建立的診斷 CSV（比 $LOG 更新的檔案 = 這次產生的）
    DIAG_CSVS=$(find ~/rover_rl/logs/diag/ -name "*.csv" -newer "$LOG" 2>/dev/null | sort)
    if [ -n "$DIAG_CSVS" ]; then
        echo "[deploy_rl_shell] 本次診斷記錄："
        echo "$DIAG_CSVS" | while read -r f; do echo "  • $f"; done
    fi
}
trap cleanup EXIT INT TERM

echo "[deploy_rl_shell] 啟動 RL 棧（不含 NDT/LV-DOT，log → $LOG）…"
# 預設 enable_ndt:=false enable_lvdot:=false → 只啟 RL 那層；NDT/LV-DOT 用 ndt + lv-dot alias 分開啟。
# $VO_ARG 來自上面互動詢問（預設 enable_vo:=true）；lv-dot 沒開時 VO 退化為放行+ω clamp，安全。
# $RECOVERY_ARG 只在有效 VO=true 時預設 enable_recovery:=true；純 RL 不啟動 recovery。
# 放在 "$@" 前面：user 傳的同名參數在後面會覆寫（ros2 launch 重複參數取最後值），如 enable_vo:=false。
ros2 launch rover_rl_bringup deploy_full.launch.py enable_ndt:=false enable_lvdot:=false "$VO_ARG" "$ORCA_ARG" "${RECOVERY_ARG[@]}" rviz:=false "${EXTRA_ARGS[@]}" "${PINGPONG_ARGS[@]}" "$@" >"$LOG" 2>&1 &
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

# 前景跑 TUI（curses 取得真實 TTY）；離開後（q / Ctrl+C）觸發 cleanup
ros2 run rover_rl_inference status_tui
