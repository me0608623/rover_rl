#!/bin/bash
# 抓速度震盪元兇：一次同時查(a)mppi進程(b)/input/nav_cmd_vel當下有沒有 rogue 簽名尖峰。
# 用法：棧開著、最好正在看到速度震盪時跑。 bash ~/rover_rl/check_mppi.sh
# 結論對照：
#   有 rogue mppi(→/input/nav_cmd_vel) + topic 有尖峰 → 就是它，kill 該 PID、查父進程來源。
#   只有1個乾淨 mppi(→cmd_vel_mppi) 但 topic 仍有尖峰 → 不是 rogue mppi，需改追別的 publisher。
#   topic 無尖峰 → 這一刻沒在震，請在震的當下重跑。

echo "======== (a) 所有 mppi 進程 + 輸出 topic + 父進程 ========"
found=0
for p in $(pgrep -x mppi_planner; pgrep -x mppi_planner_0520); do
  found=1
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  t=$(echo "$c" | grep -oE 'cmd_vel:=[^ ]+' | head -1)
  cfg=$(echo "$c" | grep -oE '[^ /]+\.yaml' | head -1)
  pp=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
  pcmd=$(ps -o args= -p "$pp" 2>/dev/null | cut -c1-72)
  echo "mppi PID=$p  → ${t:-（cmd_vel 未 remap）}  cfg=${cfg:-?}"
  echo "     父進程 PID=$pp : $pcmd"
  if echo "$c" | grep -qE 'cmd_vel:=/input/nav_cmd_vel'; then
    echo "     ⚠⚠ ROGUE！直發 /input/nav_cmd_vel → 速度震盪元兇，kill $p；上面父進程=來源"
  elif echo "$c" | grep -qE 'mppi_planner\.yaml|mppi_planner_0520'; then
    echo "     ⚠ 舊全模式 config → 疑似 rogue"
  fi
done
[ "$found" = 0 ] && echo "（目前沒有 mppi 進程在跑）"

echo ""
echo "======== (b) /input/nav_cmd_vel 用真訂閱器抓 2.5 秒(~50筆@20Hz) ========"
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/rover_rl/install/setup.bash 2>/dev/null
source ~/rover_rl/setup_env.sh 2>/dev/null
python3 - <<'PY' 2>/dev/null || echo "（rclpy/環境讀不到，略過；改用 diag CSV 比對更準）"
import rclpy, time
from rclpy.node import Node
from geometry_msgs.msg import Twist
class S(Node):
    def __init__(s):
        super().__init__("check_mppi_probe")
        s.v=[]; s.w=[]
        s.create_subscription(Twist,"/input/nav_cmd_vel",s.cb,50)
    def cb(s,m): s.v.append(m.linear.x); s.w.append(m.angular.z)
rclpy.init(); n=S(); t0=time.time()
while time.time()-t0<2.5: rclpy.spin_once(n,timeout_sec=0.05)
v,w=n.v,n.w; n.destroy_node(); rclpy.shutdown()
if not v:
    print("  2.5s 內收不到 /input/nav_cmd_vel（棧沒起 / 沒在發）")
else:
    N=len(v); sp=[i for i in range(N) if v[i]>0.4 or abs(w[i])>0.9]
    print(f"  收到 {N} 筆  v範圍[{min(v):+.2f},{max(v):+.2f}]  ω範圍[{min(w):+.2f},{max(w):+.2f}]")
    if sp:
        print(f"  ⚠ {len(sp)}/{N} 筆 rogue 簽名尖峰(v>0.4或|ω|>0.9) → 正在震、有第二來源硬插")
        print(f"     尖峰值範例: v={v[sp[0]]:+.2f} ω={w[sp[0]]:+.2f}")
    else:
        print("  ✓ 無 rogue 尖峰 → 這一刻沒在震（要在震的當下重跑才抓得到）")
PY
