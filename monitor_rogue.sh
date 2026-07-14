#!/bin/bash
# 背景監控速度震盪元兇：一直訂閱 /input/nav_cmd_vel，一偵測到「相鄰取樣暴跳」
# (|Δv|>0.30 或 |Δω|>0.50，＝兩來源交替的 rogue 簽名，不被正常快速導航誤觸)，
# 就「當場」抓下那一刻所有 mppi 進程 + 輸出 topic + 父進程，寫入 log。
# 用法：跑導航前先在另一個終端開著它 → 導航一段(含會震的段) → Ctrl+C 收工看摘要。
#   bash ~/rover_rl/monitor_rogue.sh
# 解讀：
#   尖峰事件當下若有 mppi → /input/nav_cmd_vel → 抓到 rogue（父進程=來源）。
#   尖峰事件當下 mppi 只有 →cmd_vel_mppi 一個 → 不是 rogue mppi，元兇另有其人（會印出當下所有 publisher 線索）。
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/rover_rl/install/setup.bash 2>/dev/null
source ~/rover_rl/setup_env.sh 2>/dev/null
LOG=~/rover_rl/logs/rogue_monitor_$(date +%Y%m%d_%H%M%S).log
echo "[monitor] 監控中… 尖峰事件會寫入 $LOG （Ctrl+C 結束看摘要）"
python3 - "$LOG" <<'PY'
import rclpy, time, sys, glob, os, subprocess, json
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
LOG=sys.argv[1]
def mppi_snapshot():
    lines=[]
    try:
        pids=subprocess.run("pgrep -x mppi_planner; pgrep -x mppi_planner_0520",shell=True,capture_output=True,text=True).stdout.split()
    except Exception: pids=[]
    for p in pids:
        try: c=open(f"/proc/{p}/cmdline").read().replace('\0',' ')
        except Exception: continue
        tgt=next((w for w in c.split() if w.startswith("cmd_vel:=")),"cmd_vel未remap")
        pp=subprocess.run(["ps","-o","ppid=","-p",p],capture_output=True,text=True).stdout.strip()
        pcmd=subprocess.run(["ps","-o","args=","-p",pp],capture_output=True,text=True).stdout.strip()[:70]
        rogue = "cmd_vel:=/input/nav_cmd_vel" in c
        lines.append(f"    mppi PID={p} {tgt} {'⚠ROGUE' if rogue else ''} 父={pp}[{pcmd}]")
    if not lines: lines=["    （當下無 mppi 進程）"]
    return lines
class M(Node):
    def __init__(s):
        super().__init__("rogue_monitor")
        s.n=0; s.spikes=0; s.last_ev=0.0; s.f=open(LOG,"a")
        s.pv=None; s.pw=None; s.rstate="?"   # recovery 當下狀態(分辨 rogue vs recovery自切)
        # 判據：相鄰取樣「暴跳」＝兩來源交替(rogue)。正常導航有 slew 限制、相鄰變化很小，
        # 不會誤觸；不看絕對值大小，故不會把「正常快速轉彎」當成 rogue。
        s.DV=0.30; s.DW=0.50   # 相鄰 |Δv|>0.30 或 |Δω|>0.50 rad/s 視為暴跳
        s.create_subscription(Twist,"/input/nav_cmd_vel",s.cb,50)
        s.create_subscription(String,"/recovery_supervisor_node/status",s.cb_rec,10)
    def cb_rec(s,m):
        try: s.rstate=json.loads(m.data).get("state","?")
        except Exception: pass
    def cb(s,m):
        s.n+=1
        v,w=m.linear.x,m.angular.z
        jump = (s.pv is not None and (abs(v-s.pv)>s.DV or abs(w-s.pw)>s.DW))
        s.pv, s.pw = v, w
        if jump:
            s.spikes+=1
            now=time.time()
            if now-s.last_ev>0.5:   # 每 0.5s 最多記一次事件(附進程快照)，避免洗版
                s.last_ev=now
                verdict = ("recovery=normal_rl → 純放行卻暴跳 ⇒ 必有第二來源(rogue)!"
                           if s.rstate=="normal_rl"
                           else f"recovery={s.rstate} → 可能是 recovery 自己的動作(非rogue)，看下面 mppi 快照佐證")
                s.f.write(f"[{time.strftime('%H:%M:%S')}] 暴跳 此拍(v={v:+.2f} ω={w:+.2f})  {verdict}\n")
                for ln in mppi_snapshot(): s.f.write(ln+"\n")
                s.f.flush()
                print(f"  ⚠ 暴跳! v={v:+.2f} ω={w:+.2f} recovery={s.rstate} → 已記快照")
rclpy.init(); n=M()
try:
    rclpy.spin(n)
except KeyboardInterrupt:
    pass
print(f"\n[monitor] 收到 {n.n} 筆 /input/nav_cmd_vel，其中 {n.spikes} 筆相鄰暴跳(rogue 簽名)")
if n.spikes:
    print(f"[monitor] 暴跳事件與當下 mppi 進程快照見: {LOG}")
    print("--- log 內容 ---"); os.system(f"cat {LOG}")
else:
    print("[monitor] 全程無暴跳 → 這段沒重現震盪(或震盪不是兩來源交替)，請確認有跑到會震的導航段")
n.destroy_node(); rclpy.shutdown()
PY
