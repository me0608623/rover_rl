#!/usr/bin/env python3
import os
import sys

# ensure sibling modules (yolov11_detector, utils, module) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from yolov11_detector import yolo_detector


def main():
    rclpy.init()
    node = yolo_detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
