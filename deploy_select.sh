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
        # ── 選單隱藏清單（2026-07-28）──
        # 這些 .ts 仍在磁碟上、仍可用 model_path:=<絕對路徑> 明確載入，只是不列進互動選單，
        # 讓選單只留目前在用的模型。要恢復顯示就把該檔名從 HIDE_TS 拿掉即可。
        local HIDE_TS=(
            sa1_v3e_60000.ts
            sa2_v3c_90000.ts
            sa2_v3e_240000.ts
            sa3_v3c_240000.ts
            sa4_dense_v3f_30000.ts
            sa5_tc_g1_p30_270000.ts
            sa6_tc_dense_420k.ts
            sa1r1_c1700_83d_k8.ts
            sa4_v3f_240000.ts
            sa5_v3f_tcadapt_60000.ts
            sa6_v3h_nonoise_nodr_ck270000.ts
            sa7_tc_dense_300000.ts
            sa8_e2e_k8_cleanppo_89600.ts
            w1c10_k8_e2e_1280.ts
        )
        local TS_FILES ALL_TS tf tb th hidden
        mapfile -t ALL_TS < <(ls -1 "$MODELS_DIR"/*.ts 2>/dev/null | sort)
        TS_FILES=()
        for tf in "${ALL_TS[@]}"; do
            tb=$(basename "$tf"); hidden=0
            for th in "${HIDE_TS[@]}"; do [ "$tb" = "$th" ] && { hidden=1; break; }; done
            [ "$hidden" = "0" ] && TS_FILES+=("$tf")
        done
        if [ "${#TS_FILES[@]}" -gt 0 ]; then
            local DEFAULT_TS
            DEFAULT_TS=$(grep -oP 'model_path:\s*"\K[^"]+' "$CFG_DIR/policy_params.yaml" 2>/dev/null)
            echo "┌─ 選擇 RL checkpoint ─────────────────────────────────────────"
            # 每個項目分兩層呈現：主行（編號＋檔名＋短標籤/←預設）＋ 縮排詳述數行，
            # 避免長 tag 擠成一整行難讀。short=同行短標籤，desc=縮排詳述（以換行分行）。
            local i=1 f base short desc dline
            for f in "${TS_FILES[@]}"; do
                base=$(basename "$f"); short=""; desc=""
                case "$base" in
                    *sa1r1*)
                        short="★SA1-R1 c1700   ✓ parity 過 · 僅空曠"
                        desc="83D + action stacking ｜ 8 幀 LiDAR CNN（RNN 繞過）
訓練含實測 VLP-16 噪聲 + 致動延遲 U{0,1,2}=0/200/400ms
Nav20-clean 驗收 9/9（3 seeds × 3 延遲，SR≈100% / CR 0%）
車端 parity：bundle 5.2e-06 / runtime 5.7e-06 · mismatch 0
act_hist 分母 0.5 / 1.2（★不是 0.2 / π/15）
🔴 只驗過 20×20m 空曠；走廊無主張，不得放進有行人的走廊
r_min=0.5  ω_max=1.2  → policy_params_sa1r1 + lidar_preprocessor_params_sa1r1
首跑必須：架空 + estop 在手，看 rl_w 有無極限環指紋" ;;
                    *sa4_r3*)
                        short="SA4-R3 it50   ⚠ SA4 on HOLD · 診斷跑"
                        desc="83D + action stacking ｜ K=8 ｜ 契約與 sa1r1 逐字相同
訓練含噪聲 + 延遲 U{0,1,2}，但只 6400 steps（sa1r1 的 1/34）
實測(stage-4/seed818/d1=200ms，門檻 SR≥90 CR≤10)：
  空曠 94.06%/CR 5.94% ✅ ｜ 走廊橫穿 81.13%/18.87% ❌ ｜ 正面對衝 78.21%/21.73% ❌
車端 parity：bundle 5.7e-06 / runtime 5.7e-06 · mismatch 0
🔴 無任何 SA4 通過聯合閘門(accepted_parent=null)→這是診斷跑不是已驗證能力
🔴 走廊有人：橫向約每 5 個 episode 撞 1 次，只能當資料收集
→ policy_params_sa4r3 + lidar_preprocessor_params_sa4r3" ;;
                    *sa4_r2*)
                        short="SA4-R2 c6400   ⚠ 僅正面對衝走廊"
                        desc="83D + action stacking ｜ K=8 ｜ 契約與 sa1r1 逐字相同
訓練含噪聲 + 延遲 U{0,1,2}，但只 6400 steps（sa1r1 的 1/34）
實測(stage-4/seed818/d1=200ms)：
  正面對衝 98.47%/CR 1.53% ✅ ← 唯一通過走廊 cell 的 SA4
  走廊橫穿 83.05%/16.95% ❌ ｜ 空曠 **未量測**（無數據）
車端 parity：bundle 8.1e-06 / runtime 9.1e-06 · mismatch 0（門檻內，餘裕較薄）
🔴 SA4 on HOLD；用途很窄，驗延遲契約請改用 sa1r1
→ policy_params_sa4r2 + lidar_preprocessor_params_sa4r2" ;;
                    *sa8_e2e*)
                        short="★e2e SA8 k8   ⚠ 未畢業"
                        desc="clean-PPO ｜ 79D stateless ｜ 8 幀 LiDAR CNN（RNN 繞過）
warp 14靜+6動 · K8 · future 0.10 · anti-spin 0.15
crash-run iter700 候選，四閘/情境待驗
r_min=0.5  ω_max=1.2  → policy_params_e2e + lidar_preprocessor_params_e2e" ;;
                    *w1c10*)
                        short="🔴 W1-c10 k8   REAL-ROBOT FAIL"
                        desc="🔴 2026-07-28 判定實車失效：|rl_w|>=0.83 佔 33.7% · 57s 內 13 次滿舵翻轉
   延遲驅動極限環（delay/dt≈1.0）；此判定未被任何後續 handoff 推翻
SA6 血緣最乾淨 ｜ 83D + action stacking ｜ 8 幀 LiDAR CNN（RNN 繞過）
走廊四模式與 Gate2 皆優於 D0 · 窄縫 Gate5a 100%
⚠ 零致動延遲/馬達滯後/物理DR/LiDAR噪聲訓練 → 實車延遲未建模
r_min=0.5  ω_max=1.2  → policy_params_e2e + lidar_preprocessor_params_e2e" ;;
                    *sa5r2*)
                        short="SA5-R2 c250   ⚠ 未通過閘門（診斷用）"
                        desc="83D + action stacking ｜ K=8 ｜ e2e 8 幀 LiDAR CNN（RNN 繞過）
SA5-R2 訓練 300 輪，6 顆 checkpoint 全部 FAIL；c250 為相對最佳（1/6）
固定驗收 (seed818, d1=200ms, rate=1.0)：
  窄縫 SR 100% / CR 0%     低密度 0S1D SR 99.50%     空曠 SR 91.46%
  走廊 4S2D 最差題型 random_2d CR 67.44%   mixed CR 52.29%
🔴 hard gate FAIL、無 accepted parent → 診斷跑，不是已驗證能力
🔴 行人 0.9~1.1 m/s 時：即使 0 障礙 1 行人也只有 SR 46% / CR 53.8%
⚠ 本 config speed_rate=0.7；實測此血緣 0.7 差於 1.0（CR 38.82% → 58.77%）
→ policy_params_sa5r2c250 + lidar_preprocessor_params_sa5r2c250" ;;
                    *e2e*)
                        short="★e2e SA4   ✓ 四閘全過"
                        desc="clean-PPO ｜ 79D stateless ｜ 4 幀 LiDAR CNN（RNN 繞過）
det SR 97.6%
r_min=0.5  ω_max=1.2  → policy_params_e2e + lidar_preprocessor_params_e2e" ;;
                    *tcadapt*)
                        short="v3f 定版"
                        desc="79D TC1 走廊特化
r_min=0.25  ω_max=1.2" ;;
                    *v3c*)
                        short="v3c"
                        desc="r_min=0.25  ω_max=1.2" ;;
                    *v3e*)
                        short="v3e"
                        desc="r_min=0.25  ω_max=1.2" ;;
                    *v3f*)
                        short="v3f"
                        desc="79D ／ 無 act_hist
r_min=0.25  ω_max=1.2" ;;
                    *v3h*)
                        short="v3h"
                        desc="SA6 clean baseline 對照臂 · 79D ／ 無 act_hist
r_min=0.25  ω_max=1.2" ;;
                esac
                [ "$f" = "$DEFAULT_TS" ] && short="${short:+$short   }←預設"
                printf "│ %d) %-33s %s\n" "$i" "$base" "$short"
                if [ -n "$desc" ]; then
                    while IFS= read -r dline; do
                        printf "│      %s\n" "$dline"
                    done <<< "$desc"
                fi
                printf "│\n"
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
                        *sa1r1*) VARIANT="sa1r1";;
                        *sa4_r3*) VARIANT="sa4r3";;
                        *sa4_r2*) VARIANT="sa4r2";;
                        *w1c10*) VARIANT="w1c10";;
                        *sa5r2*) VARIANT="sa5r2c250";;
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
