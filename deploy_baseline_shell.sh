#!/bin/bash
# deploy_baseline_shell.sh — 互動式：消融實驗 baseline（DWA / PID / PID+VO）一鍵啟動
#
# 用途：論文消融實驗——把傳統演算法（DWA、PID 路徑跟蹤）跟 RL policy 在**同一套**
#       NDT / costmap / routing / LV-DOT / diag_logger / pingpong_test 基礎設施下比較，
#       只換「誰在發 /input/nav_cmd_vel」，確保比較公平。
#
# 與既有腳本的分工：
#   deploy_rl / deploy_rl_shell  = RL policy（含 checkpoint 選單 + curses TUI）
#   deploy_baseline_shell        = 本腳本，baseline 專用（不需要選 checkpoint）
#   deploy_all controller:=dwa   = 同樣的事，但要自己打完整參數（本腳本就是來免去這個）
#
# 為何不用 status_tui：它幾乎所有欄位都來自 /rover_rl_policy/status，baseline 沒有
# policy_node → 面板會卡在「等待 policy_node…」。改用 baseline_status_line.py（純文字，
# 不需要 TTY、pipe 也能跑）。
#
# 用法：deploy_baseline_shell                          （互動選單）
#       deploy_baseline_shell controller:=dwa          （命令列帶參數則跳過對應提問）
#       deploy_baseline_shell controller:=pid_vo enable_pingpong:=true pingpong_a:=c28 pingpong_b:=c3

IS_TTY=0
[ -t 0 ] && [ -t 1 ] && IS_TTY=1

has_arg() { printf '%s\n' "$@" | grep -qE "^$1:="; }

# ── 啟動前檢查：清掉前一次沒收乾淨的殘留節點（孤兒）──
# 與 deploy_rl_shell 同一套邏輯：孤兒不會自己死，會繼續搶發 /input/nav_cmd_vel → 車走走停停。
# ⚠ 比 deploy_rl_shell 多掃 campusrover_move/lib/（dwa_planner / path_following / mppi_planner
#   都在那），因為 baseline 的控制器就是從那裡起的。
# ⚠ 只掃 rover_rl 棧自己的東西；NDT / LV-DOT / velodyne / 底盤 driver 分開啟的不碰。
PREFLIGHT_PATHS=(
    "rover_rl_inference/lib/rover_rl_inference/"
    "orca_filter/lib/orca_filter/"
    "campusrover_move/lib/"
    "campusrover_routing/lib/"
    "campusrover_costmap_ros2/lib/"
    "campusrover_mot/lib/"
    "campusrover_demo/lib/campusrover_demo/simple_map_publisher"
    "ros2 launch rover_rl_bringup"
)
STALE_PIDS=""
for pat in "${PREFLIGHT_PATHS[@]}"; do
    for pid in $(pgrep -f "$pat" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        case " $STALE_PIDS " in *" $pid "*) continue ;; esac
        STALE_PIDS="$STALE_PIDS $pid"
    done
done
STALE_PIDS="${STALE_PIDS# }"

if [ -n "$STALE_PIDS" ]; then
    echo "┌─ ⚠️  啟動前檢查：偵測到 $(echo "$STALE_PIDS" | wc -w) 個殘留節點 ──────────"
    for pid in $STALE_PIDS; do
        pargs=$(ps -o args= -p "$pid" 2>/dev/null)
        [ -z "$pargs" ] && continue
        pname=$(echo "$pargs" | grep -oE "__node:=[^ ]+" | head -1 | cut -d= -f2)
        [ -z "$pname" ] && pname=$(echo "$pargs" | grep -oE "lib/[a-z0-9_]+/[a-zA-Z0-9_]+" | tail -1 | xargs basename 2>/dev/null)
        case "$pargs" in *"ros2 launch rover_rl_bringup"*) pname="ros2 launch（父進程）" ;; esac
        [ -z "$pname" ] && pname="(未知)"
        psec=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
        pppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        orphan=""; [ "$pppid" = "1" ] && orphan="  ← 孤兒(父進程已死)"
        printf "│ PID %-7s 已跑 %-9s %s%s\n" "$pid" "$([ -n "$psec" ] && printf '%dh%02dm' $((psec/3600)) $(((psec%3600)/60)))" "$pname" "$orphan"
    done
    echo "│ 不清會與新棧並存搶發 cmd_vel → 車走走停停、實驗數據作廢。"
    echo "└──────────────────────────────────────────────────────────────"
    if [ "$IS_TTY" = "1" ]; then
        read -rp "是否清除這些殘留？[Y/n]（Enter=清除，強烈建議） " STALE_SEL
    else
        STALE_SEL=""; echo "[baseline] 非互動環境 → 自動清除"
    fi
    case "$STALE_SEL" in
        [Nn]*) echo "[baseline] ⚠️  保留殘留繼續啟動——行為異常請先想到這裡" ;;
        *)
            kill -SIGTERM $STALE_PIDS 2>/dev/null; sleep 2
            LEFT=""
            for pid in $STALE_PIDS; do kill -0 "$pid" 2>/dev/null && LEFT="$LEFT $pid"; done
            [ -n "$LEFT" ] && { kill -SIGKILL $LEFT 2>/dev/null; sleep 1; }
            echo "[baseline] ✅ 殘留已清除"
            ;;
    esac
else
    echo "[baseline] ✅ 啟動前檢查：無殘留節點"
fi

# ── 1. controller 選擇 ──
CONTROLLER=""
if has_arg controller "$@"; then
    CONTROLLER=$(printf '%s\n' "$@" | grep -E '^controller:=' | tail -1 | cut -d= -f2 | tr -d ':')
    echo "[baseline] controller 由命令列指定：$CONTROLLER"
elif [ "$IS_TTY" = "1" ]; then
    echo "┌─ 選擇消融實驗 controller ────────────────────────────────────"
    echo "│ [1] dwa     軌跡取樣 DWA（campusrover_move/dwa_planner）"
    echo "│             吃 costmap 避障（含行人的當下位置），不預測速度"
    echo "│ [2] mppi    取樣式 MPC（campusrover_move/mppi_planner 原生 path 模式）"
    echo "│             230 條 rollout × 前瞻 + costmap 避障，比 DWA 現代的對照組"
    echo "│ [3] pid_vo  純路徑跟蹤 + VO 動態避障層"
    echo "│             反應式→預測式對照：VO 吃 LV-DOT 追蹤速度做預測避障"
    echo "└──────────────────────────────────────────────────────────────"
    echo "  （controller:=pid 純路徑跟蹤零避障組已於 2026-09-16 移出實驗矩陣，"
    echo "    程式碼仍在，需要時用命令列 deploy_baseline_shell controller:=pid 啟動）"
    read -rp "選擇 [1-3]（Enter=1 dwa）： " C_SEL
    case "$C_SEL" in
        2) CONTROLLER=mppi ;;
        3) CONTROLLER=pid_vo ;;
        *) CONTROLLER=dwa ;;
    esac
else
    CONTROLLER=dwa
    echo "[baseline] 非互動環境：未指定 controller → 預設 dwa"
fi
echo "[baseline] ▶ controller = $CONTROLLER"

# ── 1b. 速度上限：必須與 RL 組一致（消融實驗公平性的核心）──
# RL 的實體上限不是 yaml 的 act_max_*，而是 act_max × speed_rate
# （policy_node.py:117-118，speed_rate 同時縮線速度與角速度）。
# 各 checkpoint 的 speed_rate 從 0.35 到 1.0 都有，deploy_rl_shell 啟動時還會再問一次
# → 用「底盤上限 1.0/1.2」跑 baseline 會比 RL 快 43%~186%，路線/速度比較直接失效。
ALIGN_ARGS=()
if ! has_arg align_rl_config "$@" && ! has_arg baseline_max_v "$@" && [ "$IS_TTY" = "1" ]; then
    echo "┌─ 速度上限（⚠ 必須與 RL 組一致，否則這次數據不能用）──────────"
    echo "│ RL 實體上限 = act_max × speed_rate，隨 checkpoint 與啟動時選的 speed_rate 變動。"
    echo "│ [1] 對齊某組 RL config（自動換算，不會手算錯）  ← 推薦"
    echo "│ [2] 直接輸入數字（已經知道 RL 那邊跑多少）"
    echo "│ [3] 底盤真實上限 1.0 / 1.2（只有 RL 也跑 speed_rate=1.0 時才等價）"
    echo "└──────────────────────────────────────────────────────────────"
    read -rp "選擇 [1-3]（Enter=1 對齊）： " SPD_SEL
    case "$SPD_SEL" in
        2)
            read -rp "  線速度上限 v (m/s)： " MV
            read -rp "  角速度上限 ω (rad/s)： " MW
            ALIGN_ARGS=("baseline_max_v:=${MV}" "baseline_max_w:=${MW}")
            echo "[baseline] 速度上限：v=${MV} ω=${MW}（手動指定）"
            ;;
        3)
            ALIGN_ARGS=("baseline_max_v:=1.0" "baseline_max_w:=1.2")
            echo "[baseline] ⚠ 速度上限：v=1.0 ω=1.2（底盤真實上限）"
            echo "[baseline] ⚠ RL 組若不是 speed_rate=1.0，兩組速度不對等，比較結果會被質疑"
            ;;
        *)
            # 列出所有 RL config 與其換算後的實體上限，讓人直接挑
            mapfile -t CFGS < <(python3 - <<'PY'
import glob, os, yaml
for f in sorted(glob.glob(os.path.expanduser(
        "~/rover_rl/src/rover_rl_bringup/config/policy_params*.yaml"))):
    if ".bak" in f:
        continue
    try:
        p = list(yaml.safe_load(open(f, encoding="utf-8")).values())[0]["ros__parameters"]
        v = float(p.get("act_max_linear_velocity", 1.0))
        w = float(p.get("act_max_angular_velocity", 1.2))
        r = float(p.get("speed_rate", 1.0))
    except Exception:
        continue
    print(f"{os.path.basename(f)}\t{v*r:.2f}\t{w*r:.2f}\t{r}")
PY
)
            echo "┌─ 選擇 RL 組用的 config（baseline 會換算成同樣的實體上限）────"
            i=1
            for line in "${CFGS[@]}"; do
                name=$(echo "$line" | cut -f1); ev=$(echo "$line" | cut -f2)
                ew=$(echo "$line" | cut -f3); rt=$(echo "$line" | cut -f4)
                printf "│ [%d] %-30s 實體 v=%s ω=%s (rate %s)\n" "$i" "$name" "$ev" "$ew" "$rt"
                i=$((i+1))
            done
            echo "└──────────────────────────────────────────────────────────────"
            read -rp "選擇 [1-$((i-1))]： " CFG_SEL
            if [ -n "$CFG_SEL" ] && [ "$CFG_SEL" -ge 1 ] 2>/dev/null \
               && [ "$CFG_SEL" -le "${#CFGS[@]}" ]; then
                SEL_NAME=$(echo "${CFGS[$((CFG_SEL-1))]}" | cut -f1)
                SEL_RATE=$(echo "${CFGS[$((CFG_SEL-1))]}" | cut -f4)
                ALIGN_ARGS=("align_rl_config:=${SEL_NAME}")
                echo "[baseline] 對齊 ${SEL_NAME}（yaml speed_rate=${SEL_RATE}）"
                # deploy_rl_shell 啟動 RL 時會現場問 speed_rate 並覆寫 yaml，
                # 兩邊不一致的話這次比較就白跑了，所以一定要問。
                echo "  ⚠ 啟動 RL 那次，speed_rate 選單有沒有改成別的值？"
                read -rp "  有的話輸入該數字（Enter=就是 ${SEL_RATE}）： " RATE_OV
                if [ -n "$RATE_OV" ]; then
                    ALIGN_ARGS+=("align_speed_rate:=${RATE_OV}")
                    echo "[baseline] speed_rate 覆寫為 ${RATE_OV}"
                fi
            else
                echo "[baseline] ⚠ 未選擇 → 退回底盤上限 1.0/1.2（可能與 RL 不對等）"
            fi
            ;;
    esac
fi

# ── 2. experiment_tag（診斷分組標籤，論文分析用）──
EXP_TAG_ARG=()
if ! has_arg experiment_tag "$@" && [ "$IS_TTY" = "1" ]; then
    echo "┌─ experiment_tag（diag CSV / wandb 的分組標籤）────────────────"
    echo "│ 留空 = 只用 \"$CONTROLLER\"；輸入場景後綴會變成 \"${CONTROLLER}_<你輸入的>\""
    echo "│ 例：輸入 fixed_obstacle → ${CONTROLLER}_fixed_obstacle"
    echo "└──────────────────────────────────────────────────────────────"
    read -rp "場景後綴（Enter=不加）： " TAG_SEL
    if [ -n "$TAG_SEL" ]; then
        TAG_SEL=$(echo "$TAG_SEL" | tr -c '[:alnum:]_-' '_' | sed 's/_*$//')
        EXP_TAG_ARG=("experiment_tag:=${CONTROLLER}_${TAG_SEL}")
        echo "[baseline] experiment_tag = ${CONTROLLER}_${TAG_SEL}"
    fi
fi

# ── 3. 往返測試（兩固定點 A↔B 來回，中間可重擺障礙物）──
# baseline 一律全自動來回：空白鍵是 status_tui 攔截後發的，而 baseline 不跑 status_tui
# → 沒人能按空白鍵。deploy_full.launch.py 在 controller!=rl 時已把 auto_continue 預設為 true。
PINGPONG_ARGS=()
if ! has_arg enable_pingpong "$@" && [ "$IS_TTY" = "1" ]; then
    echo "┌─ 兩固定點往返避障測試 ──────────────────────────────────────"
    echo "│ 手動把車開到 A/B 任一點停穩 → 自動規劃往對向點 → 到點自動折返，如此來回。"
    echo "│ baseline 為全自動模式（不必按空白鍵），兩段之間即可重新擺放障礙物。"
    echo "│ 需 NDT + routing（下面會自動確認）。預設 c27 ↔ c28；長走廊用 c28 → c3。"
    echo "└──────────────────────────────────────────────────────────────"
    read -rp "是否啟用往返測試？[Y/n]（Enter=啟用） " PP_SEL
    case "$PP_SEL" in
        [Nn]*) echo "[baseline] 往返測試：不啟用（需自己發 goal 或用 RViz 兩點點擊）" ;;
        *)
            read -rp "  A 點節點名 [c27]（長走廊用 c28）： " PP_A; PP_A="${PP_A:-c27}"
            read -rp "  B 點節點名 [c28]（長走廊用 c3）： " PP_B; PP_B="${PP_B:-c28}"
            PINGPONG_ARGS=("enable_pingpong:=true" "pingpong_a:=$PP_A" "pingpong_b:=$PP_B")
            echo "[baseline] 往返測試：啟用（$PP_A ↔ $PP_B，全自動來回）"
            ;;
    esac
fi

# ── 4. ros bag 錄製 ──
RECORD_BAG=0
RECORD_CLOUD=0
if [ "$IS_TTY" = "1" ]; then
    echo "┌─ 錄製 ros bag（事後回放看路線 / 離線分析）───────────────────"
    echo "│ 預設只錄控制鏈 + 狀態 topic → 輕量，存 ~/rover_rl/logs/bags/。"
    echo "│ 回放：ros2 bag play <bagdir> + rviz2 -d ~/rover_rl/rviz/replay_baseline.rviz"
    echo "└──────────────────────────────────────────────────────────────"
    read -rp "是否錄製 ros bag？[Y/n]（Enter=錄） " BAG_SEL
    case "$BAG_SEL" in
        [Nn]*) echo "[baseline] ros bag：不錄" ;;
        *)
            RECORD_BAG=1; echo "[baseline] ros bag：錄製"
            # 論文要的「路線圖」通常需要障礙物當背景，但點雲很肥（VLP-16 約 1~2 GB/分）。
            # 故預設不錄，只有明確要做回放截圖時才開。
            read -rp "  是否一併錄點雲（回放看得到障礙物，但檔案大很多）？[y/N]（Enter=不錄） " CLOUD_SEL
            case "$CLOUD_SEL" in
                [Yy]*) RECORD_CLOUD=1; echo "[baseline]   點雲：錄（注意硬碟空間）" ;;
                *) echo "[baseline]   點雲：不錄（回放只有軌跡與 TF）" ;;
            esac
            ;;
    esac
fi

source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash
source ~/rover_rl/install/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# ── 5. 依現況決定要不要自己起 NDT / LV-DOT（避免與既有實例雙開）──
# NDT：baseline 靠 map frame 的 /global_path 導航、往返測試也靠 map 車姿判到點 → 必需。
if pgrep -f "ndt_localizer_node" >/dev/null 2>&1; then
    NDT_ARG="enable_ndt:=false"
    echo "[baseline] NDT：偵測到已在跑 → 沿用（不重複啟動）"
else
    NDT_ARG="enable_ndt:=true"
    echo "[baseline] NDT：未偵測到 → 由本次 launch 一併啟動"
fi

# LV-DOT：⚠ 關鍵——deploy_full 的 enable_lvdot 只起 detector + YOLO，**不含 vo_interface**，
# 而 vo_interface/tracked_obstacles 同時是：① pid_vo 的 VO 動態障礙來源
# ② diag_logger 的 dyn_obs_min_m（動態障礙距離）來源。要讓三種 controller 的指標一致，
# 應該用 lv-dot alias 那條完整路徑（含 vo_interface + 遠端相機）。
LVDOT_ARG="enable_lvdot:=false"
if pgrep -f "onboard_detector/lib/onboard_detector/detector_node" >/dev/null 2>&1; then
    echo "[baseline] LV-DOT：偵測到已在跑 → 沿用"
else
    echo "┌─ ⚠️  LV-DOT 未在跑 ──────────────────────────────────────────"
    echo "│ 影響：① diag 的動態障礙距離(dyn_obs_min_m)不會有值"
    if [ "$CONTROLLER" = "pid_vo" ]; then
        echo "│       ② pid_vo 的 VO 沒有動態障礙來源 → 退化成只剩前方 LiDAR 煞"
    fi
    echo "│ 建議讓本腳本一併啟動（等同 lv-dot alias，含 vo_interface + 遠端相機）。"
    echo "└──────────────────────────────────────────────────────────────"
    if [ "$IS_TTY" = "1" ]; then
        read -rp "是否一併啟動 LV-DOT？[Y/n]（Enter=啟動） " LV_SEL
    else
        LV_SEL=""
    fi
    case "$LV_SEL" in
        [Nn]*) echo "[baseline] LV-DOT：不啟動（動態障礙指標將為空）" ;;
        *) START_LVDOT=1 ;;
    esac
fi

mkdir -p ~/rover_rl/logs ~/rover_rl/logs/bags
TS=$(date +%Y%m%d_%H%M%S)
LOG=~/rover_rl/logs/deploy_baseline_${CONTROLLER}_$TS.log
LVLOG=~/rover_rl/logs/lvdot_$TS.log
BAGDIR=""
BAG_PID=""

cleanup() {
    trap - EXIT INT TERM       # 先解除自身 trap 避免重入（不用 '' 遮蔽，Ctrl+C 仍可中止收尾）
    echo ""
    if [ -n "$BAG_PID" ] && kill -0 "$BAG_PID" 2>/dev/null; then
        echo "[baseline] 收尾 ros bag…（SIGINT 讓 rosbag2 正常寫 metadata.yaml）"
        kill -INT "$BAG_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8; do kill -0 "$BAG_PID" 2>/dev/null || break; sleep 0.5; done
        kill -9 "$BAG_PID" 2>/dev/null
    fi
    echo "[baseline] 停止 baseline 棧…"
    bash ~/rover_rl_stop.sh >/dev/null 2>&1
    echo "[baseline] 完整 log：$LOG"
    DIAG_CSVS=$(find ~/rover_rl/logs/diag/ -name "*.csv" -newer "$LOG" 2>/dev/null | sort)
    if [ -n "$DIAG_CSVS" ]; then
        echo "[baseline] 本次診斷記錄（controller=$CONTROLLER）："
        echo "$DIAG_CSVS" | while read -r f; do echo "  • $f"; done
        echo "  分析：ros2 run rover_rl_inference analyze_diag <上面的 csv>"
    fi
    if [ -n "$BAGDIR" ] && [ -d "$BAGDIR" ]; then
        echo "[baseline] ros bag：$BAGDIR"
        echo "  回放看路線：ros2 bag play $BAGDIR"
        echo "              rviz2 -d ~/rover_rl/rviz/replay_baseline.rviz   （Fixed Frame=odom）"
    fi
    [ -n "$START_LVDOT" ] && echo "[baseline] ⚠ LV-DOT 是本腳本起的，要一併停請用：lv-dot_stop"
}
trap cleanup EXIT INT TERM

if [ -n "$START_LVDOT" ]; then
    echo "[baseline] 啟動 LV-DOT（含 vo_interface + 遠端相機）→ $LVLOG"
    nohup ros2 launch onboard_detector run_detector.launch.py use_yolo:=true \
        >"$LVLOG" 2>&1 &
fi

echo "[baseline] 啟動 baseline 棧（controller=$CONTROLLER）→ $LOG"
# 參數順序：本腳本算出的在前，user 的 "$@" 在最後 → ros2 launch 重複參數取最後值，命令列永遠贏。
ros2 launch rover_rl_bringup deploy_full.launch.py \
    controller:="$CONTROLLER" "$NDT_ARG" "$LVDOT_ARG" rviz:=false \
    "${ALIGN_ARGS[@]}" "${EXP_TAG_ARG[@]}" "${PINGPONG_ARGS[@]}" "$@" >"$LOG" 2>&1 &
LAUNCH_PID=$!

# ── 等待「這個 controller 對應的節點」起來（不是等 rover_rl_policy——baseline 沒有它）──
case "$CONTROLLER" in
    dwa)  WAIT_NODE="dwa_planner" ;;
    mppi) WAIT_NODE="mppi_planner_node" ;;
    *)    WAIT_NODE="path_following" ;;
esac
echo -n "[baseline] 等待 $WAIT_NODE 啟動"
for _ in $(seq 1 40); do
    if ros2 node list 2>/dev/null | grep -q "$WAIT_NODE"; then echo " ✓"; break; fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo ""; echo "[baseline] ⚠ launch 提早結束，請看 log：$LOG"; exit 1
    fi
    echo -n "."; sleep 0.5
done

if [ "$RECORD_BAG" = "1" ]; then
    BAGDIR=~/rover_rl/logs/bags/baseline_${CONTROLLER}_$TS
    echo "[baseline] 開始錄 ros bag → $BAGDIR"
    # 控制鏈各層 + 量測來源；pid_vo 多錄 VO 前的 baseline_desired 才追得出 VO 改了什麼。
    # /odom + /tf 是回放畫軌跡的最低需求（replay_baseline.rviz 用 Odometry display 的
    # Keep 畫整段路線，baseline 沒有 policy 發的 /rover_rl/trail）。
    BAG_TOPICS=(
        /input/nav_cmd_vel /rover_rl/cmd_vel_baseline_desired /output/cmd_vel /cmd_vel
        /vo_safety_node/status /rover_rl/pingpong/status
        /joy /input/joy_cmd_vel
        /odom /rover_rl/lidar_sweep_72 /campusrover_local_costmap
        /vo_interface/tracked_obstacles
        /goal_pose /global_path /tf /tf_static
    )
    [ "$RECORD_CLOUD" = "1" ] && BAG_TOPICS+=(/velodyne_points)
    ros2 bag record -o "$BAGDIR" "${BAG_TOPICS[@]}" >"$LOG.bag.log" 2>&1 &
    BAG_PID=$!
fi

echo "[baseline] ─────────────────────────────────────────────────────"
echo "[baseline] 就緒。Ctrl+C 離開 → 自動收棧（含收尾 bag）"
echo "[baseline] ─────────────────────────────────────────────────────"

# 前景狀態行（純文字，不需要 TTY；Ctrl+C 會一併觸發上面的 cleanup）
python3 ~/rover_rl/baseline_status_line.py --controller "$CONTROLLER"
