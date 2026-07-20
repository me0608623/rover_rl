#!/bin/bash
# deploy_select.sh — 共用互動選單：選 RL checkpoint + 問 VO 安全層。
#
# 由 deploy_rl_shell.sh / deploy_rl.sh source 後呼叫：
#     DEPLOY_SELECT_TAG=deploy_rl deploy_rl_select "$@"
# 設定兩個「全域」變數供 caller 用：
#     EXTRA_ARGS  陣列（model_path:= / params_file:= / preprocessor_params_file:=）
#     VO_ARG      字串（enable_vo:=true|false）
#
# ⚠ 呼叫前 caller 須自行確認在真實 TTY（本函式用 read 讀 stdin，非互動會卡）。
# 命令列已帶同名參數（model_path:= / params_file:= / preprocessor_params_file:= /
# enable_vo:=）會各自跳過該段、尊重覆寫。

deploy_rl_select() {
    local TAG="${DEPLOY_SELECT_TAG:-deploy_rl}"
    local MODELS_DIR="$HOME/rover_rl/models"
    local CFG_DIR="$HOME/rover_rl/src/rover_rl_bringup/config"
    # 全域（不加 local）：caller 要讀
    EXTRA_ARGS=()
    VO_ARG="enable_vo:=true"
    ORCA_ARG="enable_orca:=false"   # 選 ORCA 時改 true（由 launch 啟 orca_safety_node 控制車）

    # ── Checkpoint 互動選單（launch 前先選模型） ──────────────────────────
    # 列出 ~/rover_rl/models/*.ts；Enter = 沿用 policy_params.yaml 預設。
    # 選到 *v3c/v3e/v3f* 自動帶對應 *_<variant>.yaml（r_min=0.25 / ω_max=1.2 等）。
    if ! printf '%s\n' "$@" | grep -qE '^(model_path|params_file|preprocessor_params_file):='; then
        local TS_FILES
        mapfile -t TS_FILES < <(ls -1 "$MODELS_DIR"/*.ts 2>/dev/null | sort)
        if [ "${#TS_FILES[@]}" -gt 0 ]; then
            local DEFAULT_TS
            DEFAULT_TS=$(grep -oP 'model_path:\s*"\K[^"]+' "$CFG_DIR/policy_params.yaml" 2>/dev/null)
            echo "┌─ 選擇 RL checkpoint ─────────────────────────────────────────"
            local i=1 f base tag
            for f in "${TS_FILES[@]}"; do
                base=$(basename "$f"); tag=""
                [ "$f" = "$DEFAULT_TS" ] && tag=" ←預設"
                case "$base" in
                    *sa8_e2e*) tag="$tag  [★e2e clean-PPO｜SA8 k8 c89600（warp 14靜+6動,K8+future0.10+anti-spin0.15）⚠未畢業:crash-run iter700候選,四閘/情境待驗｜79D stateless｜8幀LiDAR CNN,RNN繞過｜r_min=0.5/ω_max=1.2｜自動帶 policy_params_e2e + lidar_preprocessor_params_e2e]";;
                    *e2e*) tag="$tag  [★e2e clean-PPO｜SA4 c89600 四閘全過(det SR97.6%)｜79D stateless｜4幀LiDAR CNN,RNN繞過｜r_min=0.5/ω_max=1.2｜自動帶 policy_params_e2e + lidar_preprocessor_params_e2e]";;
                    *tcadapt*) tag="$tag  [v3f 定版：79D TC1 走廊特化，帶 r_min=0.25 / ω_max=1.2 config]";;
                    *v3c*) tag="$tag  [v3c：自動帶 r_min=0.25 / ω_max=1.2 config]";;
                    *v3e*) tag="$tag  [v3e：自動帶 r_min=0.25 / ω_max=1.2 config]";;
                    *v3f*) tag="$tag  [v3f：79D／無 act_hist，自動帶 r_min=0.25 / ω_max=1.2 config]";;
                    *v3h*) tag="$tag  [v3h：SA6 clean baseline 對照臂 79D／無 act_hist，帶 r_min=0.25 / ω_max=1.2 config]";;
                esac
                printf "│ %d) %s%s\n" "$i" "$base" "$tag"
                i=$((i+1))
            done
            echo "└──────────────────────────────────────────────────────────────"
            local SEL CHOSEN CHOSEN_BASE SPEC_MD VARIANT PP LP
            while :; do
                read -rp "輸入編號（Enter=預設）： " SEL
                [ -z "$SEL" ] && break
                if [[ "$SEL" =~ ^[0-9]+$ ]] && [ "$SEL" -ge 1 ] && [ "$SEL" -le "${#TS_FILES[@]}" ]; then
                    CHOSEN="${TS_FILES[$((SEL-1))]}"
                    EXTRA_ARGS+=("model_path:=$CHOSEN")
                    CHOSEN_BASE="$(basename "$CHOSEN")"
                    # 印此 checkpoint 的觀測維度語義（obs_spec.md，由 export_policy 產生）
                    SPEC_MD="${CHOSEN%.ts}.obs_spec.md"
                    if [ -f "$SPEC_MD" ]; then
                        echo "┌─ 此 checkpoint 觀測維度語義 ──────────────────────────────────"
                        sed 's/^/│ /' "$SPEC_MD"
                        echo "└──────────────────────────────────────────────────────────────"
                    else
                        echo "[$TAG] （無 $CHOSEN_BASE 的 obs_spec.md；policy_node 假設 act_hist=raw）"
                    fi
                    # 偵測 variant（v3c / v3e / v3f 同架構家族），自動帶對應 *_<variant>.yaml
                    VARIANT=""
                    case "$CHOSEN_BASE" in
                        *e2e*) VARIANT="e2e";;
                        *v3c*) VARIANT="v3c";;
                        *v3e*) VARIANT="v3e";;
                        *v3f*) VARIANT="v3f";;
                        *v3h*) VARIANT="v3h";;
                    esac
                    if [ -n "$VARIANT" ]; then
                        PP="$CFG_DIR/policy_params_$VARIANT.yaml"
                        LP="$CFG_DIR/lidar_preprocessor_params_$VARIANT.yaml"
                        if [ ! -f "$PP" ] || [ ! -f "$LP" ]; then
                            echo "[$TAG] ✗ 缺 $VARIANT config（$CFG_DIR/*_$VARIANT.yaml），中止。"; exit 1
                        fi
                        EXTRA_ARGS+=("params_file:=$PP")
                        EXTRA_ARGS+=("preprocessor_params_file:=$LP")
                        echo "[$TAG] 已選 $CHOSEN_BASE（$VARIANT：含專用 policy/preprocessor config）"
                    else
                        echo "[$TAG] 已選 $CHOSEN_BASE"
                    fi
                    break
                fi
                echo "  無效輸入，請輸入 1~${#TS_FILES[@]} 或直接 Enter。"
            done
        fi
    fi

    # ── 安全層互動詢問（launch 前先選：VO 或 ORCA，皆控制車）──────
    # 1) VO = DWA 取樣+rollout 預測式避障/煞停層（控制車）。
    #    啟用後 policy 改道 /rover_rl/cmd_vel_desired → vo_safety → /input/nav_cmd_vel。
    # 2) ORCA = RVO2 half-plane 避讓層（non-cooperative，控制車）。
    #    policy 改道 /rover_rl/cmd_vel_desired → orca_safety → /input/nav_cmd_vel。
    # VO 與 ORCA 互斥：選 ORCA 時 enable_vo:=false enable_orca:=true。
    if ! printf '%s\n' "$@" | grep -qE '^enable_vo:='; then
        echo "┌─ 安全層選擇（VO 或 ORCA，皆控制車）──────────────────────"
        echo "│ 1) VO 安全層 — DWA 取樣+rollout 預測式避障/煞停（控制車）"
        echo "│ 2) ORCA 安全層 — RVO2 half-plane 避讓（non-cooperative，控制車）"
        echo "│ 3) 都不啟用 — policy 直接送 mux"
        echo "│ （兩者都需 lv-dot 在跑才有障礙來源；都只處理動態，靜態靠 RL sweep+front_brake）"
        echo "└──────────────────────────────────────────────────────────────"
        local SAFETY_SEL
        read -rp "選擇安全層[1=VO / 2=ORCA / 3=都不]（Enter=1 VO） " SAFETY_SEL
        case "$SAFETY_SEL" in
            2) VO_ARG="enable_vo:=false"; ORCA_ARG="enable_orca:=true"
               echo "[$TAG] ORCA 控制車：啟用 orca_safety_node（RL→ORCA→mux）" ;;
            3) VO_ARG="enable_vo:=false"; ORCA_ARG="enable_orca:=false"
               echo "[$TAG] 安全層：都不啟用（policy 直接送 mux）" ;;
            1|'') VO_ARG="enable_vo:=true";  ORCA_ARG="enable_orca:=false"
               echo "[$TAG] VO：啟用（RL→VO→mux，TUI 顯示 VO 列）" ;;
            *)    VO_ARG="enable_vo:=true";  ORCA_ARG="enable_orca:=false"
               echo "[$TAG] 無效輸入，預設啟用 VO" ;;
        esac
    fi

    return 0
}
