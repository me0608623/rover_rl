#!/bin/bash
# deploy_rl.sh — 純前景 ros2 launch（滾動 log）。任何 shell 皆可，含 Claude 非互動環境。
#
# 與 deploy_rl_shell 的分工：
#   deploy_rl        = 本腳本，前景滾動 log（無 TUI）；非互動可用
#   deploy_rl_shell  = 背景 launch + 前景 curses TUI，必須真實 TTY
#
# ⭐ 真實終端機執行時：launch 前先互動「選 model + 問 VO」（與 deploy_rl_shell 同一套選單）。
#    非互動（pipe / Claude Bash 工具）：自動跳過選單，走預設（保留原 deploy_rl 純 launch 行為）。
#    任一情況都可用命令列參數覆寫：deploy_rl model_path:=… enable_vo:=false initial_mode:=nav
#
# 預設不啟 NDT / LV-DOT（用 ndt / lv-dot alias 分開啟）。要一次全開請用 deploy_all。

source ~/rover_rl/deploy_select.sh
EXTRA_ARGS=()
VO_ARG="enable_vo:=true"
ORCA_ARG="enable_orca:=false"

# TTY 守門：只有真實終端機才跑互動選單（read 在 pipe / 非互動會卡住）。
if [ -t 0 ] && [ -t 1 ]; then
    DEPLOY_SELECT_TAG=deploy_rl deploy_rl_select "$@"
else
    echo "[deploy_rl] 非互動環境：跳過選單，用預設 model + enable_vo:=true。"
    echo "[deploy_rl]   要選 model/VO → 在真實終端機跑，或用命令列參數："
    echo "[deploy_rl]   deploy_rl model_path:=/path/xxx.ts enable_vo:=false"
fi

source /opt/ros/humble/setup.bash
source ~/rover2_ws/install/setup.bash
source ~/rover_rl/install/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# 預設 enable_ndt/lvdot:=false；VO_ARG/EXTRA_ARGS 在前，user "$@" 接在後可覆寫
# （ros2 launch 重複參數取最後值）。exec 取代本 shell → 前景滾動 log，Ctrl+C 直接停 launch。
exec ros2 launch rover_rl_bringup deploy_full.launch.py \
    enable_ndt:=false enable_lvdot:=false "$VO_ARG" "$ORCA_ARG" rviz:=false \
    "${EXTRA_ARGS[@]}" "$@"
