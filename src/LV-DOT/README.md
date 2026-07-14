# LV-DOT: LiDAR-Visual Dynamic Obstacle Detection and Tracking for Autonomous Robots
[![ROS1](https://img.shields.io/badge/ROS1-Noetic-blue.svg)](https://wiki.ros.org/noetic)
[![Linux platform](https://img.shields.io/badge/platform-Ubuntu-27AE60.svg)](https://releases.ubuntu.com/20.04/)
[![license](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Linux platform](https://img.shields.io/badge/platform-linux--arm-brown.svg)](https://releases.ubuntu.com/20.04/)


This repository implements the LiDAR-visual Dynamic Obstacle Detection and Tracking (LV-DOT) framework which aims at detecting and tracking dynamic obstacles for robots with extremely constraint computational resources.

<table>
  <tr>
    <td><img src="media/LV-DOT-demo1.gif" style="width: 100%;"></td>
    <td><img src="media/LV-DOT-demo2.gif" style="width: 100%;"></td>
  </tr>
  <tr>
    <td><img src="media/LV-DOT-demo3.gif" style="width: 100%;"></td>
    <td><img src="media/LV-DOT-demo4.gif" style="width: 100%;"></td>
  </tr>
</table>

The LV-DOT framework supports dynamic obstacle detection and tracking with multiple sensor configurations:
 - Camera-only mode.
 - LiDAR-only mode.
 - Combined LiDAR and camera mode.


For additional details, please refer to the related paper available here:


Zhefan Xu\*, Haoyu Shen\*, Xinming Han, Hanyu Jin, Kanlong Ye, and Kenji Shimada, "LV-DOT: LiDAR-visual dynamic obstacle detection and tracking for autonomous robot navigation”, arXiv, 2025. [\[preprint\]](https://arxiv.org/pdf/2502.20607) [\[YouTube\]](https://youtu.be/rRvgTulWqvk) [\[BiliBili\]](https://www.bilibili.com/video/BV1qC9GY6EHj/?share_source=copy_web&vd_source=1333db331406abb1b5d4cece1e253427)

*The authors contributed equally.


## News
- **2025-02-28:** The GitHub code, video demos, and relavant papers for our LV-DOT framework are released. The authors will actively maintain and update this repo!

## Table of Contents
- [ROS 2 rover_rl Integration](#ros-2-rover_rl-integration)
- [Installation Guide](#I-Installation-Guide)
- [Run Demo](#II-Run-Demo)
    - [Run on dataset](#a-Run-on-dataset)
    - [Run on your device](#b-Run-on-your-device)
- [LV-DOT Framework and Results](#III-LV-DOT-Framework-and-Results)
- [Citation and Reference](#IV-Citation-and-Reference)
- [Acknowledgement](#V-Acknowledgement)


## ROS 2 `rover_rl` Integration

> This section documents the ROS 2 Humble version integrated in this workspace. The
> original upstream installation and demo instructions below describe ROS 1.

### Input and processing flow

The detector uses the topics configured in
`onboard_detector/cfg/detector_param.yaml`. The current rover configuration uses:

- LiDAR point cloud: `/velodyne_points`
- Robot odometry: `/odom`, or `/ndt_pose` when pose mode is selected
- Color image: `/camera/camera/color/image_raw`
- Optional depth image: `/camera/camera/depth/image_rect_raw`

LiDAR point clouds are synchronized with pose/odometry. The pose from the current
synchronized sample is applied before transforming that same cloud; pose matrices,
vectors, timestamps, sequence counters, and readiness flags are initialized before
use. Tracking and velocity estimation use the LiDAR measurement timestamp and the
actual sensor `dt` rather than assuming a fixed timer period.

The near-field LiDAR cutoff is currently:

```yaml
lidar_min_range: 0.5
```

Points with XY range below 0.5 m are rejected. Reducing the old 0.9 m cutoff improves
near-field coverage, but the robot body, wheels, and sensor mount must be checked for
self-reflections on the real platform.

### Formal detector output API

`MarkerArray` is visualization-only and must not be used as the detector data API.
Consumers shall subscribe to:

```text
/onboard_detector/dynamic_obstacles
```

Message type:

```text
onboard_detector/msg/DynamicObstacleArray
```

`DynamicObstacleArray.header.stamp` is the LiDAR measurement timestamp, not the
publication time. `header.frame_id` identifies the output coordinate frame.
`source_valid=false` means the synchronized LiDAR/pose source is stale and consumers
must immediately clear or stop coasting existing obstacles.

Each `DynamicObstacle` contains:

| Field | Meaning |
| --- | --- |
| `id` | Persistent track ID; it remains stable across matched frames but is reset after a detector restart. |
| `position` | 3D obstacle center in `header.frame_id`. |
| `velocity` | Estimated 3D velocity. |
| `acceleration` | Estimated 3D acceleration. |
| `size` | Bounding-box X/Y/Z dimensions. |
| `classification` | Currently `person` or `unknown`. |
| `source` | Currently `yolo_lidar` or `lidar`. |
| `confidence` | Detection/classification confidence in the range 0..1. |
| `is_moving` | Whether planar speed exceeds `dynamic_velocity_threshold`. |
| `position_covariance` | Flattened row-major 2x2 XY covariance: xx, xy, yx, yy. |
| `velocity_covariance` | Flattened row-major 2x2 XY velocity covariance. |

RViz/TUI marker topics remain available for debugging, but downstream navigation,
VO, logging, and safety nodes should use the timestamped message above.

### Stale-source and ghost-obstacle handling

The detector publishes `source_valid=false` when synchronized LiDAR/pose input exceeds
`detection_stale_timeout` (currently 0.5 s). The `vo_interface` has an independent
monotonic receive timeout, `source_timeout_s` (currently 0.5 s), covering the case
where the detector topic disappears completely. Either condition clears all tracks,
preventing the interface from publishing indefinitely extrapolated ghost obstacles.

### YOLO lifetime and LiDAR association

YOLO detections retain the original image timestamp and confidence. A result is used
only when all of the following conditions hold:

- receive age is at most `yolo_ttl` (currently 0.25 s);
- image-to-LiDAR timestamp error is at most `yolo_sync_tolerance` (currently 0.15 s);
- the projected 3D box and 2D YOLO box satisfy `yolo_bbox_iou_threshold` (currently 0.1).

This prevents a stale image detection from repeatedly labeling later LiDAR boxes.
YOLO only classifies objects inside the camera field of view; an obstacle outside that
view may still be detected by LiDAR but is normally reported as `unknown`/`lidar`.

### Tracking behavior

Frame-to-frame association uses global one-to-one Hungarian assignment with position
and size gates. A previous track can match at most one current detection, and vice
versa. New detections receive new persistent IDs; matched detections inherit the old
ID. When no detections remain, bounding boxes, histories, and Kalman filters are
cleared together. Kalman transition and process-noise matrices use the actual bounded
sensor `dt`.

This is deterministic hard assignment, not a full probabilistic JPDA implementation.
Crossing targets, prolonged occlusion, sparse point clouds, or abrupt shape changes
can still cause an ID switch.

### Can LV-DOT detect dynamic obstacles in every direction?

LiDAR geometric detection operates over every direction present in the received point
cloud. With a correctly mounted 360-degree LiDAR and valid calibration, it can detect
and track obstacles around the robot, including outside the camera view. This is not
an unconditional guarantee of detecting every dynamic object. Coverage is limited by:

- the physical LiDAR field of view, blind zones, minimum/maximum range, and mounting;
- occlusion and insufficient returns from small, thin, dark, low, or distant objects;
- ground/roof filtering, voxel downsampling, DBSCAN thresholds, and size constraints;
- pose/odometry timing and calibration accuracy;
- the track history required to distinguish genuine motion from point-cloud jitter.

Only the camera-covered region receives YOLO semantic confirmation, and the current
semantic classes are limited to `person` and `unknown`. The published covariance is
planar XY rather than a full 3D covariance.

### Build and verification

Build the ROS 2 integration with:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-up-to vo_interface --symlink-install \
  --allow-overriding onboard_detector vo_interface
source install/setup.bash
```

Inspect the generated API and live output with:

```bash
ros2 interface show onboard_detector/msg/DynamicObstacleArray
ros2 topic echo /onboard_detector/dynamic_obstacles
ros2 topic hz /onboard_detector/dynamic_obstacles
```

The current implementation has been compile-checked, Python syntax-checked, and
tested for immediate track clearing on `source_valid=false` and receive timeout. The
repository currently contains no package unit tests for these paths. Real-hardware
regression testing is still required, especially for 0.5 m self-reflections, crossing
targets, occlusion, ID stability, and sensor-drop recovery.


## I. Installation Guide
The system requirements for this repository are as follows. Please ensure your system meets these requirements:
- Ubuntu 18.04/20.04 LTS
- ROS Melodic/Noetic

This package has been tested on the following onboard computer:
- [NVIDIA Jetson Xavier NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/)
- [NVIDIA Jetson Orin NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) 
- [Intel NUC](https://www.intel.com/content/www/us/en/products/details/nuc.html)
 

Please follow the instructions below to install this package.
```
# This package needs ROS vision_msgs package
sudo apt install ros-noetic-vision-msgs

# Install YOLOv11 required package
pip install ultralytics

cd ~/catkin_ws/src
git clone https://github.com/Zhefan-Xu/LV-DOT.git
cd ..
catkin_make
```


## II. Run Demo
### a. Run on dataset
Please download the rosbag file from this [link](https://cmu.box.com/s/cucvje5b9xfpdpe57ilh0jx702b3ks2p):
```
rosbag play -l corridor_demo.bag
roslaunch onboard_detector run_detector.launch
```
The perception results can be visualized in Rviz as follows:

https://github.com/user-attachments/assets/e640edab-d4f3-40d6-88dc-9e5014430732


### b. Run on your device
Please adjust the configuration file under ```cfg/detector_param.yaml``` of your LiDAR and camera device. Also, change the color image topic name in ```scripts/yolo_detector/yolov11_detector.py```

From the parameter file, you can find that the algorithm expects the following data from the robot:
- LiDAR Point Cloud: ```/pointcloud```

- Depth image: ```/camera/depth/image_rect_raw```

- Color image: ```/camera/color/image_rect_raw```

- Robot pose: ```/mavros/local_position/pose```

- Robot odometry (alternative to robot pose): ```/mavros/local_position/odom```

Additionally, update the camera intrinsic parameters and the camera-LiDAR extrinsic parameters in the config file.

Run the following command to launch dynamic obstacle detection and tracking.
```
# Launch your sensor device first. Make sure it has the above data.
roslaunch onboard_detector run_detector.launch
```

The LV-DOT can be directly utilized to assist mobile robot navigation and collision avoidance in dynamic environments, as demonstrated below:

<table>
  <tr>
    <td><img src="media/navigation-demo.gif" style="width: 100%;"></td>
    <td><img src="media/avoidance-demo.gif" style="width: 100%;"></td>
  </tr>
</table>

## III. LV-DOT Framework and Results
The LV-DOT framework is shown below. Using onboard LiDAR, camera, and odometry inputs, the LiDAR and depth detection modules detect 3D obstacles, while the color detection module identifies 2D dynamic obstacles. The LiDAR-visual fusion module refines these detections, and the tracking module classifies obstacles as static or dynamic.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5352c7ae-341a-45c0-8ee1-253d9aed6078" width="90%">
</p>

Example qualitative perception results in various testing environments are shown below:
<p align="center">
  <img src="https://github.com/user-attachments/assets/054e3285-4c44-49e3-939d-74176c5d676e" width="90%">
</p>


## IV. Citation and Reference
If our work is useful to your research, please consider citing our paper.
```
@article{LV-DOT,
  title={LV-DOT: LiDAR-visual dynamic obstacle detection and tracking for autonomous robot navigation},
  author={Xu, Zhefan and Shen, Haoyu and Han, Xinming and Jin, Hanyu and Ye, Kanlong and Shimada, Kenji},
  journal={arXiv preprint arXiv:2502.20607},
  year={2025}
}
```
## V. Acknowledgement
The authors would like to express their sincere gratitude to Professor Kenji Shimada for his great support and all CERLAB UAV team members who contribute to the development of this research.
