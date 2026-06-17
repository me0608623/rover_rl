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

# ── Checkpoint 互動選單（launch 前先選模型） ──────────────────────────
# 列出 ~/rover_rl/models/*.ts 供選擇；Enter = 沿用 policy_params.yaml 預設。
# 選到 *v3c* 模型會自動帶 v3c 專用 config（r_min=0.25 / ω_max=1.2 rad/s）。
# 選到 *v3e* 模型會自動帶 v3e 專用 config（r_min=0.25 / ω_max=1.2 rad/s / 83D，同 v3c 架構）。
# 註：ω_max 一律以 rad/s 顯示，v3c/v3e 皆 1.2（=訓練值；v3c 先前誤設 0.785=0.25π 已於 2026-06-17 修正）。
# user 已自帶 model_path:= 或 params_file:= 時跳過選單（尊重命令列覆寫）。
MODELS_DIR="$HOME/rover_rl/models"
CFG_DIR="$HOME/rover_rl/src/rover_rl_bringup/config"
EXTRA_ARGS=()
if ! printf '%s\n' "$@" | grep -qE '^(model_path|params_file|preprocessor_params_file):='; then
    mapfile -t TS_FILES < <(ls -1 "$MODELS_DIR"/*.ts 2>/dev/null | sort)
    if [ "${#TS_FILES[@]}" -gt 0 ]; then
        DEFAULT_TS=$(grep -oP 'model_path:\s*"\K[^"]+' "$CFG_DIR/policy_params.yaml" 2>/dev/null)
        echo "┌─ 選擇 RL checkpoint ─────────────────────────────────────────"
        i=1
        for f in "${TS_FILES[@]}"; do
            base=$(basename "$f"); tag=""
            [ "$f" = "$DEFAULT_TS" ] && tag=" ←預設"
            case "$base" in
                *v3c*) tag="$tag  [v3c：自動帶 r_min=0.25 / ω_max=1.2 config]";;
                *v3e*) tag="$tag  [v3e：自動帶 r_min=0.25 / ω_max=1.2 config]";;
            esac
            printf "│ %d) %s%s\n" "$i" "$base" "$tag"
            i=$((i+1))
        done
        echo "└──────────────────────────────────────────────────────────────"
        while :; do
            read -rp "輸入編號（Enter=預設）： " SEL
            [ -z "$SEL" ] && break
            if [[ "$SEL" =~ ^[0-9]+$ ]] && [ "$SEL" -ge 1 ] && [ "$SEL" -le "${#TS_FILES[@]}" ]; then
                CHOSEN="${TS_FILES[$((SEL-1))]}"
                EXTRA_ARGS+=("model_path:=$CHOSEN")
                CHOSEN_BASE="$(basename "$CHOSEN")"
                # ── 印此 checkpoint 的觀測維度語義（obs_spec.md，由 export_policy 產生）──
                # act_hist_mode 等語義隨模型走；policy_node 會自動讀同名 .obs_spec.json。
                SPEC_MD="${CHOSEN%.ts}.obs_spec.md"
                if [ -f "$SPEC_MD" ]; then
                    echo "┌─ 此 checkpoint 觀測維度語義 ──────────────────────────────────"
                    sed 's/^/│ /' "$SPEC_MD"
                    echo "└──────────────────────────────────────────────────────────────"
                else
                    echo "[deploy_rl_shell] （無 $CHOSEN_BASE 的 obs_spec.md；policy_node 假設 act_hist=raw）"
                fi
                # 偵測 variant（v3c / v3e 等同架構家族），自動帶對應 *_<variant>.yaml。
                # 兩者皆 83D；差別在 policy_params_<variant>.yaml（如 ω_max）。
                VARIANT=""
                case "$CHOSEN_BASE" in
                    *v3c*) VARIANT="v3c";;
                    *v3e*) VARIANT="v3e";;
                esac
                if [ -n "$VARIANT" ]; then
                    PP="$CFG_DIR/policy_params_$VARIANT.yaml"
                    LP="$CFG_DIR/lidar_preprocessor_params_$VARIANT.yaml"
                    if [ ! -f "$PP" ] || [ ! -f "$LP" ]; then
                        echo "[deploy_rl_shell] ✗ 缺 $VARIANT config（$CFG_DIR/*_$VARIANT.yaml），中止。"; exit 1
                    fi
                    EXTRA_ARGS+=("params_file:=$PP")
                    EXTRA_ARGS+=("preprocessor_params_file:=$LP")
                    echo "[deploy_rl_shell] 已選 $CHOSEN_BASE（$VARIANT：含專用 policy/preprocessor config）"
                else
                    echo "[deploy_rl_shell] 已選 $CHOSEN_BASE"
                fi
                break
            fi
            echo "  無效輸入，請輸入 1~${#TS_FILES[@]} 或直接 Enter。"
        done
    fi
fi

# ── VO 安全層互動詢問（launch 前先選）────────────────────────────────
# VO = 夾在 RL policy 與底盤之間的動態障礙預測式避障/煞停層。
# 啟用後 policy 輸出改道 /rover_rl/cmd_vel_desired → vo_safety → /input/nav_cmd_vel，
# 且 status_tui 會多顯示「VO / VO參數」兩列（介入狀態 + 參數）。
# user 已自帶 enable_vo:= 時跳過詢問（尊重命令列覆寫）。
VO_ARG="enable_vo:=true"
if ! printf '%s\n' "$@" | grep -qE '^enable_vo:='; then
    echo "┌─ VO 安全層（動態障礙預測式避障 / 煞停）─────────────────────"
    echo "│ 夾在 RL policy 與底盤之間，吃 vo_interface 動態障礙做短期碰撞預測。"
    echo "│ 需 lv-dot 在跑才有障礙來源（沒跑則退化為放行 + ω 限幅，安全）。"
    echo "│ 啟用後 TUI 會多顯示「VO / VO參數」兩列（介入狀態 + 參數）。"
    echo "└──────────────────────────────────────────────────────────────"
    read -rp "是否啟用 VO 安全層？[Y/n]（Enter=啟用） " VO_SEL
    case "$VO_SEL" in
        [Nn]*) VO_ARG="enable_vo:=false"; echo "[deploy_rl_shell] VO：不啟用（policy 直接送 mux）" ;;
        *)     VO_ARG="enable_vo:=true";  echo "[deploy_rl_shell] VO：啟用（RL→VO→mux，TUI 顯示 VO 列）" ;;
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
# 放在 "$@" 前面：user 傳的同名參數在後面會覆寫（ros2 launch 重複參數取最後值），如 enable_vo:=false。
ros2 launch rover_rl_bringup deploy_full.launch.py enable_ndt:=false enable_lvdot:=false "$VO_ARG" rviz:=false "${EXTRA_ARGS[@]}" "$@" >"$LOG" 2>&1 &
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
