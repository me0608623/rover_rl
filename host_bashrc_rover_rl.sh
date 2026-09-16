# host_bashrc_rover_rl.sh — ~/.bashrc 裡 rover_rl 相關區塊的版控副本（參考用，非自動生效）
#
# 為什麼有這個檔：這些 alias / 環境變數住在 ~/.bashrc，不在 repo 裡 → 換機器或重灌就消失。
# 但 ~/.bashrc 整份不能進 repo（含個人憑證），故只抽出 rover_rl 相關這段版控。
#
# 還原方式（擇一）：
#   a) 把本檔內容貼回 ~/.bashrc
#   b) 在 ~/.bashrc 末尾加：source ~/rover_rl/host_bashrc_rover_rl.sh
#
# ⚠ 這是快照，不會自動跟 ~/.bashrc 同步。改了 alias 記得回來更新這份。
# 快照時間：2026-09-14 17:02:55（來源 ~/.bashrc 第 270-363 行）

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export CYCLONEDDS_URI=file:///home/aa/cyclone.xml
export ZENOH_ROUTER_CONFIG_URI=/home/aa/zenoh_router.json5
# ZENOH_SESSION_CONFIG_URI 不設，用預設 peer 連 localhost:7447

# deploy_rl：純 ros2 launch（前景滾動 log）。任何 shell 皆可，含 Claude 非互動環境
#   ⭐ 真實終端機跑時：launch 前互動「選 model + 問 VO」（與 deploy_rl_shell 同一套，TTY 守門）；
#      非互動（pipe/Claude）自動跳過走預設（純 launch，原行為不變）。
#   預設 enable_ndt:=false enable_lvdot:=false → 只啟 RL 那層；NDT/LV-DOT 請用 ndt + lv-dot alias 分開啟。
#   enable_vo 預設 true（VO 常駐；lv-dot 沒開時退化放行+ω clamp，安全）；選單或命令列可覆寫。
#   （命令列同名參數接在後面會覆寫選單結果，例：deploy_rl model_path:=… enable_vo:=false initial_mode:=nav）
alias deploy_rl='bash ~/rover_rl/deploy_rl.sh'
# deploy_rl_shell：互動式 — 完整棧背景 + 前景繁中 TUI 儀表板（需真實 TTY，給人用，按 q 收棧）
alias deploy_rl_shell='bash ~/rover_rl/deploy_rl_shell.sh'
alias deploy_rl_stop='bash ~/rover_rl_stop.sh'
# deploy_baseline_shell：論文消融實驗用——傳統演算法 baseline（DWA / PID / PID+VO）互動式啟動。
#   互動問 controller / experiment_tag / 往返測試 A-B 點 / 錄 bag，並自動偵測 NDT、LV-DOT
#   是否已在跑（沒跑會問要不要一併啟）。前景是純文字狀態行（非 curses，pipe 也能跑）。
#   與 RL 共用同一套 NDT/costmap/routing/diag/pingpong，只換誰發 /input/nav_cmd_vel。
#   例：deploy_baseline_shell  /  deploy_baseline_shell controller:=pid_vo
#   停止：deploy_baseline_stop（＝ deploy_rl_stop，同一支 rover_rl_stop.sh；
#         已涵蓋 dwa_planner / path_following / baseline_status_line 與 orphan wrapper）
alias deploy_baseline_shell='bash ~/rover_rl/deploy_baseline_shell.sh'
alias deploy_baseline='deploy_baseline_shell'
alias deploy_baseline_stop='bash ~/rover_rl_stop.sh'
# record_teleop：純被動錄「人工示範操控速度」（模仿學習用）。標籤原樣轉給腳本。
#   例：record_teleop backup_demo_1  /  record_teleop backup_demo_1 --secs 60
#   輸出 ~/rover_rl/logs/teleop/teleop_<時間>_<標籤>/，Ctrl+C 停並印倒退段落分析。
alias record_teleop='source /opt/ros/humble/setup.bash && source ~/rover_rl/install/setup.bash >/dev/null 2>&1 && source ~/rover2_ws/install/setup.bash >/dev/null 2>&1 && source ~/rover_rl/setup_env.sh >/dev/null 2>&1 && python3 ~/rover_rl/scripts/record_teleop.py'
# rviz2_stop：清乾淨所有 rviz2 進程（含 launch 拉起的、手動開的）
alias rviz2_stop='pkill -9 -f "rviz2"; pkill -9 -f "rviz2_node"; echo "rviz2 已清乾淨"'
alias rviz_stop='rviz2_stop'

# ===== LV-DOT 動態障礙偵測（onboard_detector）=====
# lv-dot：啟動 LV-DOT（LiDAR+相機融合, YOLO 走 CUDA venv GPU）。前景滾動 log，任何 shell 皆可。
#   參數原樣轉給 launch，例：lv-dot use_yolo:=false / lv-dot use_rviz:=true
alias lv-dot='source /opt/ros/humble/setup.bash && source ~/rover_rl/install/setup.bash && source ~/rover_rl/setup_env.sh >/dev/null 2>&1 && ros2 launch onboard_detector run_detector.launch.py use_yolo:=true'
alias lvdot='lv-dot'
# lv-dot_stop：停掉 LV-DOT（detector + yolo + vo_interface 節點）
alias lv-dot_stop='pkill -9 -f "run_detector.launch.py"; pkill -9 -f "onboard_detector/lib/onboard_detector/detector_node"; pkill -9 -f "yolov11_detector_node"; pkill -9 -f "yolo_venv/bin/python"; pkill -9 -f "vo_interface/vo_interface_node"; echo "LV-DOT + vo 已停"'
alias lvdot_stop='lv-dot_stop'

# ===== NDT 定位（單獨啟動）=====
# ndt：只啟動 NDT 定位（map_loader + points_downsample + tf_static + ndt_localizer_node）
#   ⚠ 不要跟 deploy_all 同跑（deploy_all 內含 NDT，會雙啟衝突）
#   ✓ deploy_rl / deploy_rl_shell 預設不啟 NDT/LV-DOT，就是設計成搭配此 ndt + lv-dot 分開啟
alias ndt='source /opt/ros/humble/setup.bash && source ~/rover2_ws/install/setup.bash && source ~/rover_rl/setup_env.sh >/dev/null 2>&1 && ros2 launch ndt_localizer ndt_localizer_launch.py converged_param_transform_probability:=1.6'
alias ndt_stop='pkill -9 -f ndt_localizer_node; pkill -9 -f voxel_grid_filter; pkill -9 -f "ndt_localizer/lib"; pkill -9 -f "static_transform_publisher.*world.*map"; echo "NDT 已停"'

# ===== 一鍵全啟：完整棧(含 NDT) + LV-DOT =====
# deploy_all：deploy_full(NDT+policy+...) + LV-DOT 一起背景啟動。參數轉給 deploy_full。
#   deploy_all                    # 預設 nav（收斂後車會動）
#   deploy_all initial_mode:=idle # 靜態先確認
alias deploy_all='bash ~/rover_rl/deploy_all.sh'
# deploy_full_stop / deploy_all_stop：徹底停 deploy_full 棧 + NDT + LV-DOT + 清殭屍
alias deploy_full_stop='bash ~/rover_rl/deploy_full_stop.sh'
alias deploy_all_stop='bash ~/rover_rl/deploy_full_stop.sh'

# ── VLP-16 雜訊表徵引導腳本（需先 cd rover2_ws 觸發 auto-source）─────────────
alias guided_record='bash ~/rover2_ws/src/campusrover_sim_to_real/vlp16_measurement/guided_record.sh'
alias guided_replay='bash ~/rover2_ws/src/campusrover_sim_to_real/vlp16_measurement/guided_replay.sh'

# ── 自動 source workspace（依所在 workspace 切換時觸發一次）─────────────────
_AUTOSOURCE_LAST_WS=""
_autosource_ws() {
  local cwd ws=""
  cwd=$(pwd -P 2>/dev/null || pwd)
  case "$cwd" in
    /home/aa/rover_rl*)  ws=rover_rl ;;
    /home/aa/rover2_ws*) ws=rover2_ws ;;
    /home/aa/ndt_ws*)    ws=ndt_ws ;;
  esac
  # 只在「進入不同 workspace」時 source，避免同一 ws 內每次 cd 都重 source/洗版
  [[ "$ws" == "$_AUTOSOURCE_LAST_WS" ]] && return
  _AUTOSOURCE_LAST_WS="$ws"
  [[ -z "$ws" ]] && return

  source /opt/ros/humble/setup.bash 2>/dev/null
  case "$ws" in
    rover_rl)
      source /home/aa/rover_rl/install/setup.bash 2>/dev/null
      source /home/aa/rover_rl/setup_env.sh 2>/dev/null   # Zenoh RMW + ROS_DOMAIN_ID=55
      echo "✅ rover_rl sourced (Zenoh env, DOMAIN=55)"
      ;;
    rover2_ws)
      source /home/aa/rover2_ws/install/setup.bash 2>/dev/null
      source /home/aa/rover_rl/setup_env.sh 2>/dev/null   # Zenoh RMW + DOMAIN=55（看得到 /velodyne_points）
      echo "✅ rover2_ws sourced (Zenoh env, DOMAIN=55)"
      ;;
    ndt_ws)
      source /home/aa/ndt_ws/install/setup.bash 2>/dev/null
      echo "✅ ndt_ws sourced"
      ;;
  esac
}
PROMPT_COMMAND="${PROMPT_COMMAND:+$PROMPT_COMMAND; }_autosource_ws"
