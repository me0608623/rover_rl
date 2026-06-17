#!/usr/bin/env python3
"""監測 /odom + /velodyne_points 斷流，抓間歇性 /odom dropout 並定性。

每次任一流斷檔 >GAP 秒就即時寫一行（含時間戳 + 是哪個流 + 是否同時斷=zenoh vs driver）。
輸出 /tmp/odom_monitor.log（逐行 flush，中途被殺也保留）。
"""
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

GAP = 0.5          # 斷檔門檻(s)
OUT = "/tmp/odom_monitor.log"


class Mon(Node):
    def __init__(self):
        super().__init__("stream_dropout_monitor")
        self.last = {"odom": None, "velo": None}
        self.events = []          # (t, stream, gap)
        self.counts = {"odom": 0, "velo": 0}
        self.t0 = time.time()
        self.f = open(OUT, "w")
        self._w(f"=== 監測開始 {time.strftime('%H:%M:%S')} (門檻 {GAP}s) ===")
        self.create_subscription(Odometry, "/odom", lambda m: self._tick("odom"), 50)
        self.create_subscription(PointCloud2, "/velodyne_points", lambda m: self._tick("velo"), 50)
        self.create_timer(30.0, self._heartbeat)

    def _w(self, s):
        self.f.write(s + "\n")
        self.f.flush()

    def _tick(self, name):
        now = time.time()
        self.counts[name] += 1
        prev = self.last[name]
        self.last[name] = now
        if prev is not None and (now - prev) > GAP:
            gap = now - prev
            # 另一個流在這段期間是否也斷了？(同時斷=zenoh；只一個斷=該流的 driver)
            other = "velo" if name == "odom" else "odom"
            other_last = self.last[other]
            other_also = (other_last is not None and (now - other_last) > GAP)
            tag = "【兩流同斷=疑 zenoh】" if other_also else f"【僅 {name} 斷=疑該 driver】"
            self._w(f"[{time.strftime('%H:%M:%S')} +{now-self.t0:.0f}s] {name} 斷檔 {gap:.2f}s {tag}")
            self.events.append((now, name, gap))

    def _heartbeat(self):
        el = time.time() - self.t0
        oh = self.counts["odom"] / max(el, 1)
        vh = self.counts["velo"] / max(el, 1)
        self._w(f"[心跳 +{el:.0f}s] odom~{oh:.0f}Hz velo~{vh:.0f}Hz 累積斷檔事件 {len(self.events)} 次")


def main():
    rclpy.init()
    n = Mon()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n._w(f"=== 監測結束，共 {len(n.events)} 次斷檔事件 ===")
        n.f.close()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
