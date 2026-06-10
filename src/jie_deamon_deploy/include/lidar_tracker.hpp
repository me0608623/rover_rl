/*
 * ============================================================================
 * 檔案：lidar_tracker.hpp
 * 說明：光達跟隨模組 — 處理 LaserScan 資料，計算目標質心，
 *       結合人工勢場法避障與卡爾曼濾波平滑，輸出跟隨速度指令
 *
 * 主要功能：
 *   1. 接收 sensor_msgs/LaserScan，轉換為 (x, y) 點雲
 *   2. 在目標搜尋半徑內計算質心，作為追蹤目標
 *   3. 人工勢場法 (APF) 計算障礙物排斥力
 *   4. 可選卡爾曼濾波平滑目標位置
 *   5. 差速機器人速度控制（僅 linear.x + angular.z）
 *   6. OpenCV 即時可視化（除錯用）
 *
 * [Humble 相容性]
 *   - sensor_msgs::msg::LaserScan 在 Humble/Jazzy 間 API 相同
 *   - geometry_msgs::msg::Twist 在 Humble/Jazzy 間 API 相同
 *   - OpenCV 為系統函式庫，與 ROS 版本無關
 *   - 注意：Humble 預設 QoS 為 KEEP_LAST(10) + RELIABLE，
 *     若光達驅動使用 BEST_EFFORT，需在訂閱端明確設定 QoS
 * ============================================================================
 */

#ifndef LIDAR_TRACKER_HPP
#define LIDAR_TRACKER_HPP

#include "common_types.hpp"                    // 共用常數與共享狀態
#include "kalman_filter.hpp"                   // 卡爾曼濾波器
#include <sensor_msgs/msg/laser_scan.hpp>      // 光達掃描訊息型別
#include <geometry_msgs/msg/twist.hpp>         // 速度指令訊息型別
#include <opencv2/opencv.hpp>                  // OpenCV 可視化函式庫
#include <cmath>                               // 數學函式 (sqrt, atan2, abs 等)
#include <algorithm>                           // std::clamp 等演算法
#include <functional>                          // std::function 回呼封裝

/*
 * 光達跟隨處理器
 * 負責從原始光達掃描中提取目標、計算避障速度、發布控制指令
 *
 * [Humble 相容] 本類別不繼承 rclcpp::Node，而是透過回呼函式
 *   與外部 ROS 節點互動，因此不受 Humble/Jazzy API 差異影響
 */
class LidarTracker {
public:
    using VelocityCallback = std::function<void(const geometry_msgs::msg::Twist&)>;  // 速度發布回呼型別
    using DataBroadcastCallback = std::function<void()>;                              // 資料廣播回呼型別

    LidarTracker(SharedState& state) : state_(state) {
        // 計算機器人在 OpenCV 視窗中的像素座標（視窗下方偏移）
        robot_center_pixel_ = cv::Point(
            WINDOW_SIZE / 2,                                                   // 水平置中
            WINDOW_SIZE - static_cast<int>(ROBOT_Y_OFFSET_M * METERS_TO_PIXELS)  // 垂直偏移
        );
    }

    /* 設定速度發布回呼：當計算完速度後透過此回呼發布 Twist 訊息 */
    void setVelocityCallback(VelocityCallback cb) {
        velocity_callback_ = std::move(cb);
    }

    /* 設定資料廣播回呼：每幀處理完後通知 Web/Android 端更新 */
    void setDataBroadcastCallback(DataBroadcastCallback cb) {
        data_broadcast_callback_ = std::move(cb);
    }

    /* 設定 OpenCV 可視化開關（除錯模式用，正式部署建議關閉以節省 CPU） */
    void setOpenCVEnabled(bool enabled) {
        enable_opencv_ = enabled;
        if (enabled) {
            cv::namedWindow("Follow", cv::WINDOW_AUTOSIZE);  // 建立固定大小的視窗
        }
    }

    /* 設定卡爾曼濾波開關 */
    void setKalmanEnabled(bool enabled) {
        enable_kalman_ = enabled;
        if (enabled) {
            kalman_.reset();  // 啟用時重置濾波器狀態
        }
    }

    /* 設定卡爾曼濾波參數 */
    void setKalmanParams(double process_noise, double measurement_noise) {
        kalman_.setProcessNoise(process_noise);          // 過程雜訊
        kalman_.setMeasurementNoise(measurement_noise);  // 觀測雜訊
    }

    /* 銷毀 OpenCV 視窗（節點關閉時呼叫） */
    void destroyWindows() {
        if (enable_opencv_) {
            cv::destroyAllWindows();
        }
    }

    /*
     * 處理 VLP-16 投影後的 2D 點雲（跟隨 + 避障）
     * 與 processScan 邏輯相同，但輸入是已投影的 2D 點（已排除機器人框架）。
     * 用於 Web 雷達跟隨模式：用戶點選目標 → 追蹤質心 → APF 避障 → 發布 cmd_vel。
     */
    void processCloudPoints(const std::vector<std::pair<double, double>>& points) {
        if (!state_.active.load()) return;

        double target_x, target_y;
        state_.getTarget(target_x, target_y);

        if (points.empty()) return;

        // 目標質心累積
        double centroid_x = 0.0, centroid_y = 0.0;
        int points_in_target = 0;

        // APF 排斥力
        double repulse_x = 0.0, repulse_y = 0.0;
        double min_obstacle_dist = 999.0;

        for (const auto& [px, py] : points) {
            double dist = std::sqrt(px * px + py * py);

            // 更新最近障礙距離
            if (dist < min_obstacle_dist) min_obstacle_dist = dist;

            // APF：只考慮前方近距離障礙
            if (dist < APF_INFLUENCE_DIST && px > -0.1) {
                double force = APF_REPULSE_GAIN * (1.0 / dist - 1.0 / APF_INFLUENCE_DIST)
                               / (dist * dist);
                repulse_x -= force * px / dist;
                repulse_y -= force * py / dist;
            }

            // 計算質心：在目標搜尋半徑內的點
            double dist_to_target = std::sqrt(std::pow(px - target_x, 2) + std::pow(py - target_y, 2));
            if (dist_to_target < TARGET_RADIUS) {
                centroid_x += px;
                centroid_y += py;
                points_in_target++;
            }
        }

        // 限制排斥力幅度
        double repulse_mag = std::sqrt(repulse_x * repulse_x + repulse_y * repulse_y);
        if (repulse_mag > 1.0) { repulse_x /= repulse_mag; repulse_y /= repulse_mag; }

        // 更新目標質心（卡爾曼濾波平滑）
        if (points_in_target > 0) {
            double raw_x = centroid_x / points_in_target;
            double raw_y = centroid_y / points_in_target;
            if (enable_kalman_) {
                double fx, fy;
                kalman_.update(raw_x, raw_y, fx, fy);
                state_.setTarget(fx, fy);
            } else {
                state_.setTarget(raw_x, raw_y);
            }
        }

        // 計算跟隨速度
        geometry_msgs::msg::Twist cmd;
        int mode = state_.control_mode.load();
        if (mode == MODE_FOLLOW) {
            calculateFollowVelocity(cmd, target_x, target_y, 0, 0, repulse_x, repulse_y, min_obstacle_dist);
        }
        state_.setVelocity(cmd.linear.x, cmd.linear.y, cmd.angular.z);
        publishVelocity(cmd, mode);

        if (data_broadcast_callback_) data_broadcast_callback_();
    }

    /*
     * 處理光達掃描資料（主要處理函式）
     * 適配二輪差速機器人：僅使用 linear.x + angular.z，不使用 linear.y
     *
     * 處理流程：
     *   1. 遍歷所有光達點，轉換為機器人座標系 (x, y)
     *   2. 排除機器人框架區域內的點
     *   3. 在目標半徑內計算質心
     *   4. 累積人工勢場排斥力
     *   5. 根據控制模式計算速度指令
     *   6. 發布速度、廣播資料、更新可視化
     */
    void processScan(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) {
        if (!state_.active.load()) {  // 系統未啟動時跳過
            return;
        }

        double target_x, target_y;
        state_.getTarget(target_x, target_y);  // 取得當前目標位置

        cv::Mat image;
        if (enable_opencv_) {
            image = cv::Mat::zeros(TOTAL_WINDOW_HEIGHT, WINDOW_SIZE, CV_8UC3);  // 建立黑色背景畫布
            drawBackground(image, target_x, target_y);                           // 繪製網格、目標圓等背景
        }

        if (scan_msg->ranges.empty()) {  // 空掃描資料則跳過
            return;
        }

        // 目標質心累積變數
        double centroid_x = 0.0, centroid_y = 0.0;
        int points_in_target_count = 0;  // 落在目標半徑內的點數

        // 目標方向向量（用於路徑矩形內障礙偵測）
        double target_vec_x = target_x;
        double target_vec_y = target_y;
        double target_vec_len = std::sqrt(target_vec_x * target_vec_x + target_vec_y * target_vec_y);  // 目標距離
        double left_y_min = -RECTANGLE_WIDTH / 2;   // 左側路徑邊界
        double right_y_min = RECTANGLE_WIDTH / 2;    // 右側路徑邊界

        // 人工勢場法：排斥力累積與最近障礙距離
        double repulse_x = 0.0, repulse_y = 0.0;  // 排斥力分量
        double min_obstacle_dist = 999.0;           // 最近障礙物距離（初始化為極大值）

        std::vector<std::pair<double, double>> points;
        points.reserve(scan_msg->ranges.size());  // 預分配記憶體避免頻繁 realloc

        for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
            float range = scan_msg->ranges[i];                              // 該光束的距離值
            if (std::isinf(range) || std::isnan(range)) continue;          // 跳過無效值（inf/nan）

            float angle = scan_msg->angle_min + i * scan_msg->angle_increment;  // 該光束的角度
            double point_x = -range * cos(angle);  // 轉換為機器人座標系 X（前方為正，取負號因光達座標系差異）
            double point_y = -range * sin(angle);  // 轉換為機器人座標系 Y（左方為正）

            double dist_to_robot = std::sqrt(point_x * point_x + point_y * point_y);  // 到機器人的距離

            // OpenCV 可視化：依距離著色
            if (enable_opencv_) {
                cv::Scalar color;
                if (dist_to_robot < APF_EMERGENCY_DIST) {
                    color = cv::Scalar(0, 0, 255);      // 紅色：危險區域
                } else if (dist_to_robot < APF_SLOWDOWN_DIST) {
                    color = cv::Scalar(0, 165, 255);    // 橙色：減速區域
                } else if (dist_to_robot < APF_INFLUENCE_DIST) {
                    color = cv::Scalar(0, 255, 255);    // 黃色：勢場影響範圍
                } else {
                    color = cv::Scalar(100, 100, 100);  // 灰色：安全區域
                }
                cv::circle(image, toPixel(point_x, point_y), 2, color, -1, cv::LINE_AA);  // 繪製點
            }

            // 判斷是否在機器人框架排除區域內（光達掃到自身結構）
            bool in_robot_frame = (point_x > -ROBOT_FRAME_BACK && point_x < ROBOT_FRAME_FRONT &&
                                   point_y > -ROBOT_FRAME_RIGHT && point_y < ROBOT_FRAME_LEFT);

            // 更新最近障礙距離（排除框架區域）
            if (!in_robot_frame && dist_to_robot < min_obstacle_dist) {
                min_obstacle_dist = dist_to_robot;
            }

            // 人工勢場法：計算排斥力（排除框架區域，僅考慮前方和側方障礙）
            if (!in_robot_frame && dist_to_robot < APF_INFLUENCE_DIST && point_x > -0.1) {
                double force = APF_REPULSE_GAIN * (1.0 / dist_to_robot - 1.0 / APF_INFLUENCE_DIST)
                               / (dist_to_robot * dist_to_robot);  // 反比平方排斥力
                repulse_x -= force * point_x / dist_to_robot;      // X 方向排斥分量
                repulse_y -= force * point_y / dist_to_robot;      // Y 方向排斥分量
            }

            // 排除機器人框架內的點：不參與目標質心、路徑障礙計算，也不廣播到 Web
            if (in_robot_frame) continue;

            // 非框架區域的點加入廣播列表
            points.emplace_back(point_x, point_y);

            // 計算該點到目標中心的距離
            double dist_to_target_center = std::sqrt(pow(point_x - target_x, 2) + pow(point_y - target_y, 2));

            // 落在目標搜尋半徑內 → 累加到質心計算
            if (dist_to_target_center < TARGET_RADIUS) {
                centroid_x += point_x;
                centroid_y += point_y;
                points_in_target_count++;
                continue;
            }

            // 路徑矩形內的障礙偵測（用於左右避障提示）
            if (target_vec_len > 1e-6) {
                // 將點投影到目標方向向量上
                double proj_x = (point_x * target_vec_x + point_y * target_vec_y) / target_vec_len;   // 縱向投影
                double proj_y = (point_x * -target_vec_y + point_y * target_vec_x) / target_vec_len;  // 橫向投影

                // 判斷是否在路徑矩形內
                if (proj_x >= 0 && proj_x <= target_vec_len && std::abs(proj_y) <= RECTANGLE_WIDTH / 2.0) {
                    if (enable_opencv_) {
                        if (proj_y > 0) {
                            cv::line(image, robot_center_pixel_, toPixel(point_x, point_y), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);  // 黃線：左側障礙
                        } else {
                            cv::line(image, robot_center_pixel_, toPixel(point_x, point_y), cv::Scalar(100, 100, 255), 1, cv::LINE_AA);  // 淡紅線：右側障礙
                        }
                    }

                    // 更新左右邊界最大侵入量
                    if (proj_y > 0 && proj_y > left_y_min) {
                        left_y_min = proj_y;      // 左側最大侵入
                    } else if (proj_y <= 0 && proj_y < right_y_min) {
                        right_y_min = proj_y;     // 右側最大侵入
                    }
                }
            }
        }

        // 限制排斥力幅度（防止極端值）
        double repulse_mag = std::sqrt(repulse_x * repulse_x + repulse_y * repulse_y);
        if (repulse_mag > 1.0) {
            repulse_x /= repulse_mag;  // 正規化為單位向量
            repulse_y /= repulse_mag;
        }

        // 更新共享狀態中的點雲快取（move 語義避免複製）
        state_.setPoints(std::move(points));

        // 更新目標位置為質心（可選卡爾曼濾波平滑）
        if (points_in_target_count > 0) {
            double raw_x = centroid_x / points_in_target_count;  // 質心 X
            double raw_y = centroid_y / points_in_target_count;  // 質心 Y

            if (enable_kalman_) {
                double filtered_x, filtered_y;
                kalman_.update(raw_x, raw_y, filtered_x, filtered_y);  // 卡爾曼濾波平滑
                state_.setTarget(filtered_x, filtered_y);               // 更新濾波後的目標
            } else {
                state_.setTarget(raw_x, raw_y);                         // 直接使用原始質心
            }
        }

        // 根據控制模式計算速度
        geometry_msgs::msg::Twist cmd_vel_msg;
        int mode = state_.control_mode.load();  // 讀取當前控制模式

        if (mode == MODE_DIRECT) {
            // 直接控制模式：讀取遙控指令
            double vx, vy, wz;
            state_.getDirectCmd(vx, vy, wz);
            cmd_vel_msg.linear.x = vx;
            cmd_vel_msg.linear.y = 0.0;   // 差速機器人不支援橫移
            cmd_vel_msg.angular.z = wz;
        }
        else if (mode == MODE_FOLLOW) {
            // 跟隨模式：計算跟隨速度（含勢場避障）
            calculateFollowVelocity(cmd_vel_msg, target_x, target_y,
                                    left_y_min, right_y_min,
                                    repulse_x, repulse_y, min_obstacle_dist);
        }

        // 快取計算後的速度（供 Web 廣播讀取）
        state_.setVelocity(cmd_vel_msg.linear.x, cmd_vel_msg.linear.y, cmd_vel_msg.angular.z);

        // 發布速度指令到 ROS topic
        publishVelocity(cmd_vel_msg, mode);

        // OpenCV 可視化更新
        if (enable_opencv_) {
            drawSpeedDisplay(image, cmd_vel_msg);  // 繪製底部速度顯示
            std::string status = state_.is_moving_enabled.load() ? "MOVING" : "STOPPED";
            cv::putText(image, status, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                       state_.is_moving_enabled.load() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);  // 狀態文字
            cv::imshow("Follow", image);   // 更新顯示
            cv::waitKey(1);                // 非阻塞等待鍵盤事件
        }

        // 觸發資料廣播回呼（通知 Web/Android 端）
        if (data_broadcast_callback_) {
            data_broadcast_callback_();
        }
    }

private:
    SharedState& state_;                           // 共享狀態參考
    cv::Point robot_center_pixel_;                 // 機器人在視窗中的像素座標
    bool enable_opencv_ = false;                   // OpenCV 可視化開關
    bool enable_kalman_ = false;                   // 卡爾曼濾波開關
    KalmanFilter2D kalman_;                        // 卡爾曼濾波器實例
    VelocityCallback velocity_callback_;           // 速度發布回呼
    DataBroadcastCallback data_broadcast_callback_; // 資料廣播回呼

    /*
     * 座標轉換：機器人座標系 (公尺) → OpenCV 像素座標
     * robot_x 為前方（畫面上方），robot_y 為左方（畫面右方）
     */
    cv::Point toPixel(double robot_x, double robot_y) const {
        int px = robot_center_pixel_.x - static_cast<int>(robot_y * METERS_TO_PIXELS);  // Y 軸反轉
        int py = robot_center_pixel_.y - static_cast<int>(robot_x * METERS_TO_PIXELS);  // X 軸反轉
        return cv::Point(px, py);
    }

    /*
     * 繪製可視化背景
     * 包含：路徑矩形、機器人位置、距離同心圓、目標圓圈與連線
     */
    void drawBackground(cv::Mat& image, double target_x, double target_y) {
        double dist_to_target = std::sqrt(target_x * target_x + target_y * target_y);  // 目標距離
        if (dist_to_target > 1e-6) {
            double angle_to_target = atan2(target_y, target_x);  // 目標方向角
            double half_width = RECTANGLE_WIDTH / 2.0;           // 路徑矩形半寬

            // 計算路徑矩形的四個角點（機器人座標系）
            cv::Point2f corners_robot[4];
            corners_robot[0] = cv::Point2f(0 - half_width * sin(angle_to_target), 0 + half_width * cos(angle_to_target));
            corners_robot[1] = cv::Point2f(0 + half_width * sin(angle_to_target), 0 - half_width * cos(angle_to_target));
            corners_robot[2] = cv::Point2f(target_x + half_width * sin(angle_to_target), target_y - half_width * cos(angle_to_target));
            corners_robot[3] = cv::Point2f(target_x - half_width * sin(angle_to_target), target_y + half_width * cos(angle_to_target));

            // 轉換為像素座標並繪製填充矩形
            std::vector<cv::Point> corners_pixel;
            for (int i = 0; i < 4; ++i) {
                corners_pixel.push_back(toPixel(corners_robot[i].x, corners_robot[i].y));
            }
            cv::fillConvexPoly(image, corners_pixel, cv::Scalar(50, 50, 50), cv::LINE_AA);  // 深灰色路徑區域
        }

        // 繪製機器人位置標記
        cv::circle(image, robot_center_pixel_, 8, cv::Scalar(255, 200, 200), -1, cv::LINE_AA);  // 淡粉色圓點

        // 繪製距離參考同心圓（1m, 2m, 3m, 4m）
        const std::vector<double> scales = {1.0, 2.0, 3.0, 4.0};
        for (double dist : scales) {
            int radius_px = static_cast<int>(dist * METERS_TO_PIXELS);
            cv::circle(image, robot_center_pixel_, radius_px, cv::Scalar(128, 128, 128), 1, cv::LINE_AA);  // 灰色圓圈
        }

        // 繪製目標搜尋圓圈與機器人到目標的連線
        cv::Point target_center_px = toPixel(target_x, target_y);
        int target_radius_px = static_cast<int>(TARGET_RADIUS * METERS_TO_PIXELS);
        cv::circle(image, target_center_px, target_radius_px, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);  // 紫色目標圓
        cv::line(image, robot_center_pixel_, target_center_px, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);   // 綠色連線
    }

    /*
     * 計算跟隨速度（帶勢場避障）
     * 適配差速機器人：linear.y 恆為 0，橫向偏差轉為角速度修正
     *
     * 控制策略：
     *   - 前後速度：比例控制 (dist_error × LINEAR_SCALE_FACTOR)，帶死區與最小速度
     *   - 角速度：比例控制 (angle_error × ANGULAR_SCALE_FACTOR)，帶死區
     *   - 融合勢場排斥力
     *   - 接近障礙物時線性減速
     */
    void calculateFollowVelocity(geometry_msgs::msg::Twist& cmd, double target_x, double target_y,
                                  double /*left_y_min*/, double /*right_y_min*/,
                                  double repulse_x, double repulse_y, double min_obstacle_dist) {
        // 緊急停止檢查：障礙物過近時立即停車
        if (min_obstacle_dist < APF_EMERGENCY_DIST) {
            cmd.linear.x = 0.0;
            cmd.linear.y = 0.0;
            cmd.angular.z = 0.0;
            return;
        }

        // 前後運動控制（帶 5cm 死區防止微振盪）
        double dist_error = target_x - FOLLOW_DIST;   // 距離誤差 = 目標距離 - 理想跟隨距離
        if (std::abs(dist_error) < 0.05) {
            cmd.linear.x = 0.0;                        // 死區內不動
        } else {
            cmd.linear.x = dist_error * LINEAR_SCALE_FACTOR;  // 比例控制
            if (cmd.linear.x < 0) cmd.linear.x *= 0.8;        // 後退時降速 20%（安全考量）

            double min_speed = 0.06;  // 最小速度閾值（防止馬達轉不動）
            if (std::abs(cmd.linear.x) < min_speed) {
                cmd.linear.x = (cmd.linear.x > 0) ? min_speed : -min_speed;
            }
        }

        // 旋轉運動控制（帶 0.1 rad 死區）
        double angle_error = atan2(target_y, target_x);  // 目標方位角
        if (std::abs(angle_error) < 0.1) {
            cmd.angular.z = 0.0;                           // 死區內不旋轉
        } else {
            cmd.angular.z = angle_error * ANGULAR_SCALE_FACTOR;  // 比例控制
        }

        // 差速機器人不支援橫移
        cmd.linear.y = 0.0;

        // 融合人工勢場排斥力（橫向排斥力轉為角速度修正）
        cmd.linear.x += repulse_x;   // 縱向排斥力直接疊加
        cmd.angular.z += repulse_y;   // 橫向排斥力轉為角速度

        // 接近障礙時線性減速
        if (min_obstacle_dist < APF_SLOWDOWN_DIST) {
            double slowdown_factor = (min_obstacle_dist - APF_EMERGENCY_DIST)
                                   / (APF_SLOWDOWN_DIST - APF_EMERGENCY_DIST);  // 線性插值減速因子
            slowdown_factor = std::clamp(slowdown_factor, 0.1, 1.0);             // 限制在 [0.1, 1.0]
            cmd.linear.x *= slowdown_factor;                                      // 套用減速
        }

        // 限制速度上下限
        cmd.linear.x = std::clamp(cmd.linear.x, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED);
        cmd.angular.z = std::clamp(cmd.angular.z, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED);
    }

    /*
     * 發布速度指令
     * 根據移動使能狀態與控制模式決定是否發送實際速度或零速度
     */
    void publishVelocity(const geometry_msgs::msg::Twist& cmd, int mode) {
        if (!velocity_callback_) return;  // 未設定回呼則跳過

        if (state_.is_moving_enabled.load() || mode == MODE_DIRECT) {
            if (mode == MODE_DIRECT) {
                velocity_callback_(cmd);                      // 直接控制模式：直接發送
            } else if (state_.is_moving_enabled.load()) {
                velocity_callback_(cmd);                      // 跟隨模式且已使能：發送計算速度
            } else {
                geometry_msgs::msg::Twist zero_vel;
                velocity_callback_(zero_vel);                 // 未使能：發送零速度
            }
        } else {
            geometry_msgs::msg::Twist zero_vel;
            velocity_callback_(zero_vel);                     // 其他情況：發送零速度（安全停車）
        }
    }

    /*
     * 繪製速度顯示面板（OpenCV 視窗底部區域）
     * 顯示：Vx 前後速度條、Vy 橫向速度條、Wz 角速度弧
     */
    void drawSpeedDisplay(cv::Mat& image, const geometry_msgs::msg::Twist& cmd_vel) {
        int center_x = WINDOW_SIZE / 2;                         // 速度顯示區水平中心
        int center_y = WINDOW_SIZE + SPEED_DISPLAY_HEIGHT / 2;  // 速度顯示區垂直中心

        // 繪製分隔線
        cv::line(image, cv::Point(0, WINDOW_SIZE), cv::Point(WINDOW_SIZE, WINDOW_SIZE),
                 cv::Scalar(80, 80, 80), 1);

        // 繪製十字基準線
        cv::line(image, cv::Point(center_x - SPEED_BAR_LENGTH - 10, center_y),
                 cv::Point(center_x + SPEED_BAR_LENGTH + 10, center_y),
                 cv::Scalar(60, 60, 60), 1);  // 水平基準線
        cv::line(image, cv::Point(center_x, center_y - SPEED_BAR_LENGTH - 10),
                 cv::Point(center_x, center_y + SPEED_BAR_LENGTH + 10),
                 cv::Scalar(60, 60, 60), 1);  // 垂直基準線

        // 限制速度值在顯示範圍內
        double vx = std::clamp(cmd_vel.linear.x, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED);
        double vy = std::clamp(cmd_vel.linear.y, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED);
        double wz = std::clamp(cmd_vel.angular.z, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED);

        // 計算速度條像素長度
        int bar_x = static_cast<int>((vx / MAX_LINEAR_SPEED) * SPEED_BAR_LENGTH * 4);  // Vx 條長度
        int bar_y = static_cast<int>((vy / MAX_LINEAR_SPEED) * SPEED_BAR_LENGTH * 4);  // Vy 條長度

        // 繪製 Vx 速度條（紅色，垂直方向）
        if (std::abs(bar_x) > 1) {
            cv::line(image, cv::Point(center_x, center_y),
                     cv::Point(center_x, center_y - bar_x),
                     cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
        }

        // 繪製 Vy 速度條（綠色，水平方向）
        if (std::abs(bar_y) > 1) {
            cv::line(image, cv::Point(center_x, center_y),
                     cv::Point(center_x - bar_y, center_y),
                     cv::Scalar(0, 255, 0), 4, cv::LINE_AA);
        }

        // 繪製中心點
        cv::circle(image, cv::Point(center_x, center_y), 5, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);

        // 繪製 Wz 角速度弧形（橙色）
        int arc_center_x = center_x + 120;  // 弧形偏右顯示
        cv::circle(image, cv::Point(arc_center_x, center_y), SPEED_ARC_RADIUS,
                   cv::Scalar(60, 60, 60), 1, cv::LINE_AA);  // 背景圓

        if (std::abs(wz) > 0.01) {
            double start_angle = -90;                                    // 起始角度（12 點鐘方向）
            double arc_angle = -(wz / MAX_ANGULAR_SPEED) * 180;        // 弧度映射
            double end_angle = start_angle + arc_angle;

            double draw_start = start_angle;
            double draw_end = end_angle;
            if (arc_angle < 0) {
                std::swap(draw_start, draw_end);  // 確保繪製方向正確
            }

            cv::ellipse(image, cv::Point(arc_center_x, center_y),
                       cv::Size(SPEED_ARC_RADIUS, SPEED_ARC_RADIUS),
                       0, draw_start, draw_end,
                       cv::Scalar(255, 150, 0), 4, cv::LINE_AA);  // 橙色弧形
        }

        // 繪製數值文字
        char buf[64];
        snprintf(buf, sizeof(buf), "Vx:%.2f", cmd_vel.linear.x);
        cv::putText(image, buf, cv::Point(10, WINDOW_SIZE + 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);     // Vx 紅色

        snprintf(buf, sizeof(buf), "Vy:%.2f", cmd_vel.linear.y);
        cv::putText(image, buf, cv::Point(10, WINDOW_SIZE + 50),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);     // Vy 綠色

        snprintf(buf, sizeof(buf), "Wz:%.2f", cmd_vel.angular.z);
        cv::putText(image, buf, cv::Point(10, WINDOW_SIZE + 75),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 150, 0), 1);   // Wz 橙色
    }
};

#endif // LIDAR_TRACKER_HPP
