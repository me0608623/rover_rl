/*
 * 檔案說明：robot_nexus.cpp — ROS 2 機器人中樞控制節點
 *
 * 功能概述：
 * - 訂閱雷射雷達 /scan 話題，追蹤目標
 * - 透過 UDP 將雷達資料傳送至 Android App
 * - 接收 Android App 的控制指令
 * - 提供 Web 視覺化介面（HTTP + WebSocket）
 * - 可選的 OpenCV 視覺化顯示（除錯用）
 * - 已改為差速驅動模式（Twist: linear.x + angular.z）
 *
 * Humble 相容性提示：
 * - create_publisher / create_subscription 在 Humble 與 Jazzy 中 API 語法相同
 * - 但 QoS 預設值可能不同：Humble 預設使用 rmw_qos_profile_default
 *   建議在 create_publisher / create_subscription 時明確指定 QoS 數值
 * - create_wall_timer 在 Humble 中語法一致，無需修改
 * - declare_parameter<T> 在 Humble 中需要顯式模板型別，此處已正確使用
 */

#include <rclcpp/rclcpp.hpp>                    // ROS 2 C++ 核心函式庫
#include <std_msgs/msg/int32.hpp>               // 底盤模式切換訊息型別
#include <sensor_msgs/msg/laser_scan.hpp>       // 雷射雷達掃描訊息型別
#include <sensor_msgs/msg/point_cloud2.hpp>     // 3D 點雲訊息型別（VLP-16）
#include <geometry_msgs/msg/twist.hpp>          // 速度指令訊息型別（linear + angular）
// std_msgs/String 保留供未來擴充用
#include <std_msgs/msg/string.hpp>              // 字串訊息型別（備用）
#include <geometry_msgs/msg/pose_stamped.hpp>   // 單點導航目標
#include <campusrover_msgs/srv/routing_path.hpp> // 路徑導航服務
#include <campusrover_msgs/srv/module_info.hpp>  // 拓撲節點查詢服務
#include <thread>                                // 標準執行緒支援
#include <chrono>                                // 時間工具（用於定時器週期）

#include "common_types.hpp"                      // 共用型別定義（SharedState 等）
#include "lidar_tracker.hpp"                     // 雷射雷達追蹤模組
#include "direct_control.hpp"                    // 直接速度控制模組
#include "web_comm.hpp"                          // Web 通訊管理模組（HTTP + WebSocket）
#include "android_comm.hpp"                      // Android UDP 通訊管理模組

/*
 * RobotNexusNode — 機器人中樞控制 ROS 2 節點
 *
 * 繼承 rclcpp::Node，整合以下子模組：
 * - LidarTracker：雷射雷達目標追蹤
 * - DirectController：手動速度控制（來自 Web / Android）
 * - WebCommManager：Web 前端通訊（HTTP API + WebSocket 廣播）
 * - AndroidCommManager：Android App UDP 雙向通訊
 *
 * 所有子模組共享 SharedState，透過原子變數與互斥鎖確保執行緒安全。
 */
class RobotNexusNode : public rclcpp::Node
{
public:
    /*
     * 建構函式 — 初始化節點名稱、子模組、參數、訂閱者與發布者
     * 所有子模組均接收 shared_state_ 的參考，以共享狀態資料。
     */
    RobotNexusNode()
        : Node("robot_nexus"),                   // 節點名稱設為 "robot_nexus"
          lidar_tracker_(shared_state_),          // 雷達追蹤器綁定共享狀態
          direct_controller_(shared_state_),      // 直接控制器綁定共享狀態
          web_comm_(shared_state_),               // Web 通訊綁定共享狀態
          android_comm_(shared_state_)            // Android 通訊綁定共享狀態
    {
        // 宣告 ROS 2 參數（可透過 launch 檔或命令列覆寫）
        this->declare_parameter<bool>("active", false);           // 是否啟用跟隨模式
        this->declare_parameter<bool>("enable_opencv", false);    // 是否啟用 OpenCV 視覺化
        this->declare_parameter<bool>("enable_web", true);        // 是否啟用 Web 介面
        this->declare_parameter<bool>("enable_kalman", false);    // 是否啟用卡爾曼濾波
        this->declare_parameter<std::string>("web_root", "");     // Web 靜態檔案根目錄路徑

        // 讀取參數值
        shared_state_.active.store(this->get_parameter("active").as_bool());  // 將 active 寫入共享原子變數
        bool enable_opencv = this->get_parameter("enable_opencv").as_bool();  // OpenCV 開關
        bool enable_web = this->get_parameter("enable_web").as_bool();        // Web 開關
        bool enable_kalman = this->get_parameter("enable_kalman").as_bool();  // 卡爾曼開關
        std::string web_root = this->get_parameter("web_root").as_string();   // Web 根目錄

        RCLCPP_INFO(this->get_logger(), "節點啟動 - 跟隨: %s, OpenCV: %s, Web: %s, 卡爾曼: %s",
                    shared_state_.active.load() ? "開啟" : "關閉",
                    enable_opencv ? "開啟" : "關閉",
                    enable_web ? "開啟" : "關閉",
                    enable_kalman ? "開啟" : "關閉");

        /*
         * 建立發布者與訂閱者
         * 【Humble 注意】QoS depth=1 在此明確指定，避免依賴預設值差異。
         *  若需更嚴格的 QoS，可改用 rclcpp::QoS(1).best_effort() 等。
         */
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 1);  // 發布速度指令到 /cmd_vel
        web_cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/input/web_cmd_vel", 1); // Web 專用通道
        mux_mode_pub_ = this->create_publisher<std_msgs::msg::Int32>("/mux_mode_cmd", 1); // 發布底盤模式切換指令
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 1,  // 訂閱 /scan 話題，QoS depth=1（僅保留最新一筆）
            std::bind(&RobotNexusNode::scanCallback, this, std::placeholders::_1)  // 綁定回呼函式
        );
        // VLP-16 3D 點雲 → 2D 投影（z 過濾），用於 Web 雷達顯示
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", rclcpp::QoS(5).best_effort(),
            std::bind(&RobotNexusNode::cloudCallback, this, std::placeholders::_1)
        );

        // 設定日誌回呼（讓子模組能透過 ROS 2 logger 輸出訊息）
        auto log_cb = [this](const std::string& msg) {
            RCLCPP_INFO(this->get_logger(), "%s", msg.c_str());  // 轉發到 ROS 2 日誌系統
        };

        // 設定速度發布回呼：Web 專用通道，不搶 joy 也不搶 nav
        auto vel_cb = [this](const geometry_msgs::msg::Twist& cmd) {
            web_cmd_pub_->publish(cmd);       // → /input/web_cmd_vel（Web 專用）
        };

        // 配置雷達追蹤器
        lidar_tracker_.setOpenCVEnabled(enable_opencv);   // 設定 OpenCV 視覺化開關
        lidar_tracker_.setKalmanEnabled(enable_kalman);   // 設定卡爾曼濾波開關
        lidar_tracker_.setVelocityCallback(vel_cb);       // 設定速度發布回呼
        lidar_tracker_.setDataBroadcastCallback([this]() {
            android_comm_.sendScanData();  // 每次掃描處理後，透過 UDP 傳送資料給 Android
            web_comm_.broadcastData();     // 同時透過 WebSocket 廣播給 Web 前端
        });

        // 配置直接控制器
        direct_controller_.setVelocityCallback(vel_cb);  // 設定速度發布回呼

        // 配置 Web 通訊模組
        if (enable_web) {
            web_comm_.setLogCallback(log_cb);  // 設定日誌回呼
            web_comm_.setWebRoot(web_root);     // 設定 Web 根目錄
            web_comm_.autoDetectWebRoot({       // 自動偵測 Web 靜態檔案路徑
                "./web",
                "../share/jie_deamon/web"
            });
            web_comm_.setDirectCmdCallback([this](double x, double y, double z) {
                direct_controller_.processDirectCmd(x, y, z);  // 處理 Web 前端的手動控制指令
                RCLCPP_INFO(this->get_logger(), "收到 direct_cmd: vx=%.2f, vy=%.2f, wz=%.2f", x, y, z);
            });
            // 差速機器人無需機械狗動作指令，忽略 action_cmd
            web_comm_.setActionCmdCallback([this](const std::string& action) {
                RCLCPP_WARN(this->get_logger(), "忽略動作指令 (差速機器人): %s", action.c_str());
            });
            // 底盤模式切換：Web → /mux_mode_cmd → lcr_cmd_vel_mux
            web_comm_.setMuxModeCallback([this](int mode) {
                auto msg = std_msgs::msg::Int32();
                msg.data = mode;
                mux_mode_pub_->publish(msg);
                const char* names[] = {"放鬆", "停止", "手動", "自動"};
                if (mode >= 0 && mode <= 3)
                    RCLCPP_INFO(this->get_logger(), "底盤模式切換: %d (%s)", mode, names[mode]);
            });
            web_comm_.start();  // 啟動 HTTP 與 WebSocket 伺服器

            std::string local_ip = web_comm_.getLocalIP();  // 取得本機 IP 位址
            RCLCPP_INFO(this->get_logger(), "Web 介面: http://%s:%d", local_ip.c_str(), HTTP_PORT);
        }

        // === 路徑導航 + 單點導航 ===
        goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 1);
        routing_client_ = this->create_client<campusrover_msgs::srv::RoutingPath>("/routing_to_path/routing_call");
        route_info_client_ = this->create_client<campusrover_msgs::srv::ModuleInfo>("/get_route_info");

        // 路徑導航回呼：Web → routing service → /global_path
        web_comm_.setRouteNavCallback([this](const std::string& origin, const std::string& dest) {
            if (!routing_client_->wait_for_service(std::chrono::seconds(1))) {
                RCLCPP_ERROR(this->get_logger(), "routing service 不可用");
                return;
            }
            auto req = std::make_shared<campusrover_msgs::srv::RoutingPath::Request>();
            req->origin = origin;
            req->destination = {dest};
            auto future = routing_client_->async_send_request(req,
                [this, origin, dest](rclcpp::Client<campusrover_msgs::srv::RoutingPath>::SharedFuture f) {
                    auto resp = f.get();
                    int n_pts = 0;
                    for (const auto& p : resp->routing) n_pts += (int)p.poses.size();
                    RCLCPP_INFO(this->get_logger(), "routing 完成: %s → %s  path points=%d",
                                origin.c_str(), dest.c_str(), n_pts);
                });
        });

        // 單點導航回呼：Web → /goal_pose
        web_comm_.setGoalNavCallback([this](double x, double y) {
            auto msg = geometry_msgs::msg::PoseStamped();
            msg.header.frame_id = "map";
            msg.header.stamp = this->now();
            msg.pose.position.x = x;
            msg.pose.position.y = y;
            msg.pose.orientation.w = 1.0;
            goal_pose_pub_->publish(msg);
            RCLCPP_INFO(this->get_logger(), "單點導航: (%.2f, %.2f)", x, y);
        });

        // 請求節點列表回呼
        web_comm_.setGetNodesCallback([this]() { sendRouteNodes(); });

        // 定時抓取拓撲節點（重試直到成功）
        fetch_nodes_timer_ = this->create_wall_timer(
            std::chrono::seconds(2), [this]() { fetchRouteNodes(); });

        // 配置 Android 通訊模組
        android_comm_.setLogCallback(log_cb);  // 設定日誌回呼
        android_comm_.start();                  // 啟動 UDP 接收執行緒

        /*
         * 建立直接控制定時器（10Hz = 每 100ms 觸發一次）
         * 【Humble 注意】create_wall_timer 語法在 Humble / Jazzy 中相同。
         */
        direct_control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),  // 每 100 毫秒觸發
            std::bind(&DirectController::timerCallback, &direct_controller_)  // 綁定控制器回呼
        );

        /*
         * Web 廣播定時器（10Hz）— 不依賴點雲 callback
         * 之前廣播只在 cloudCallback 觸發，若點雲斷流或 callback 未觸發，
         * Web 前端就完全收不到資料（畫面凍結）。改用獨立 timer 保底廣播，
         * 確保前端持續收到 scan_data（即使點雲暫時沒更新，也會送出最後狀態）。
         */
        broadcast_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),  // 10Hz
            [this]() { web_comm_.broadcastData(); }
        );

        RCLCPP_INFO(this->get_logger(), "機器人中樞節點 'robot_nexus' 已啟動 (差速驅動模式)");
    }

    /*
     * 解構函式 — 銷毀 OpenCV 視窗（若有開啟）
     */
    ~RobotNexusNode()
    {
        lidar_tracker_.destroyWindows();  // 關閉所有 OpenCV 視窗
    }

private:
    SharedState shared_state_;  // 共享狀態物件（所有子模組透過參考存取）

    // 功能子模組
    LidarTracker lidar_tracker_;            // 雷射雷達追蹤器
    DirectController direct_controller_;    // 手動直接控制器
    WebCommManager web_comm_;               // Web 通訊管理器
    AndroidCommManager android_comm_;       // Android 通訊管理器

    // ROS 2 成員
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;          // /cmd_vel 發布者（保留，未用）
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr web_cmd_pub_;         // /input/web_cmd_vel 發布者
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr mux_mode_pub_;              // /mux_mode_cmd 發布者
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;       // /scan 訂閱者
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;    // /velodyne_points 訂閱者
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_pub_; // /goal_pose 發布者
    rclcpp::Client<campusrover_msgs::srv::RoutingPath>::SharedPtr routing_client_;    // routing service client
    rclcpp::Client<campusrover_msgs::srv::ModuleInfo>::SharedPtr route_info_client_;  // 拓撲查詢 client
    rclcpp::TimerBase::SharedPtr direct_control_timer_;                            // 10Hz 定時器
    rclcpp::TimerBase::SharedPtr broadcast_timer_;                                 // 10Hz Web 廣播定時器
    rclcpp::TimerBase::SharedPtr fetch_nodes_timer_;                               // 拓撲節點抓取定時器

    // 路由節點列表（從 /get_route_info 取得）
    struct RouteNode { std::string name; double x, y; };
    std::vector<RouteNode> route_nodes_;
    std::mutex nodes_mutex_;
    bool nodes_loaded_ = false;

    /*
     * scanCallback — 雷射雷達掃描回呼函式
     * 每當 /scan 話題收到新訊息時觸發，將資料交給 LidarTracker 處理。
     */
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
    {
        lidar_tracker_.processScan(scan_msg);  // 將掃描資料送入追蹤器處理
    }

    /*
     * cloudCallback — VLP-16 3D 點雲回呼
     * 將 PointCloud2 投影到 2D（z 過濾），存入 SharedState.points 供 Web 顯示。
     * 跟隨模式啟用時，同時呼叫 LidarTracker 做目標追蹤。
     */
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
    {
        // 找 x, y, z 欄位偏移
        int x_offset = -1, y_offset = -1, z_offset = -1;
        uint32_t point_step = cloud_msg->point_step;
        for (const auto& field : cloud_msg->fields) {
            if (field.name == "x") x_offset = field.offset;
            else if (field.name == "y") y_offset = field.offset;
            else if (field.name == "z") z_offset = field.offset;
        }
        if (x_offset < 0 || y_offset < 0 || z_offset < 0) return;

        const uint8_t* data = cloud_msg->data.data();
        const size_t n = cloud_msg->width * cloud_msg->height;

        constexpr float Z_FILTER = 0.5f;       // 只取 |z| < 0.5m 的點（地面以上、天花板以下）
        constexpr float R_MIN = 0.2f;           // 盲區過濾
        constexpr float R_MAX = 20.0f;          // 最遠顯示距離

        std::vector<std::pair<double, double>> points;
        points.reserve(n / 4);  // 粗估四分之一的點通過過濾

        for (size_t i = 0; i < n; ++i) {
            const uint8_t* p = data + i * point_step;
            float x, y, z;
            memcpy(&x, p + x_offset, sizeof(float));
            memcpy(&y, p + y_offset, sizeof(float));
            memcpy(&z, p + z_offset, sizeof(float));

            if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
            if (std::fabs(z) > Z_FILTER) continue;  // 過濾地面/天花板

            float dist = std::sqrt(x * x + y * y);
            if (dist < R_MIN || dist > R_MAX) continue;

            // 機器人框架排除（與 lidar_tracker 同條件）
            if (x > -ROBOT_FRAME_BACK && x < ROBOT_FRAME_FRONT &&
                y > -ROBOT_FRAME_RIGHT && y < ROBOT_FRAME_LEFT) continue;

            points.emplace_back(static_cast<double>(x), static_cast<double>(y));
        }

        shared_state_.setPoints(std::move(points));

        // 跟隨模式：用投影後的 2D 點做追蹤 + 避障
        if (shared_state_.active.load()) {
            lidar_tracker_.processCloudPoints(
                shared_state_.getPoints());  // getPoints() 回傳 const ref
            // processCloudPoints 內部會呼叫 data_broadcast_callback_
            // 但若提前 return（空點雲）則不會，這裡補保底廣播
        }

        // 永遠廣播：確保 Web 前端即時收到點雲資料
        web_comm_.broadcastData();
    }

    /*
     * fetchRouteNodes — 定時嘗試從 /get_route_info 取得拓撲節點列表
     * 成功後停掉 timer，並立即廣播給前端。
     */
    void fetchRouteNodes() {
        if (nodes_loaded_) {
            fetch_nodes_timer_->cancel();
            return;
        }
        if (!route_info_client_->wait_for_service(std::chrono::milliseconds(100))) return;

        auto req = std::make_shared<campusrover_msgs::srv::ModuleInfo::Request>();
        req->building = "itc";
        req->floor = "3";
        auto future = route_info_client_->async_send_request(req,
            [this](rclcpp::Client<campusrover_msgs::srv::ModuleInfo>::SharedFuture f) {
                try {
                    auto resp = f.get();
                    std::lock_guard<std::mutex> lock(nodes_mutex_);
                    route_nodes_.clear();
                    for (const auto& n : resp->node) {
                        route_nodes_.push_back({n.name, n.pose.position.x, n.pose.position.y});
                    }
                    nodes_loaded_ = true;
                    RCLCPP_INFO(this->get_logger(), "載入 %zu 個路由節點", route_nodes_.size());
                    sendRouteNodes();
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "get_route_info 失敗: %s", e.what());
                }
            });
    }

    /*
     * sendRouteNodes — 把路由節點列表序列化成 JSON 廣播給 Web 前端
     */
    void sendRouteNodes() {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        if (route_nodes_.empty()) return;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "{\"type\":\"route_nodes\",\"nodes\":[";
        for (size_t i = 0; i < route_nodes_.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "{\"name\":\"" << route_nodes_[i].name << "\","
                << "\"x\":" << route_nodes_[i].x << ","
                << "\"y\":" << route_nodes_[i].y << "}";
        }
        oss << "]}";
        web_comm_.broadcastRouteNodes(oss.str());
    }
};

/*
 * main — 程式進入點
 * 初始化 ROS 2、建立節點、進入事件迴圈，直到收到關閉訊號。
 */
int main(int argc, char** argv)
{
    setlocale(LC_ALL, "");        // 設定語系（支援中文輸出）
    rclcpp::init(argc, argv);     // 初始化 ROS 2 客戶端函式庫

    auto node = std::make_shared<RobotNexusNode>();  // 建立中樞節點（共享指標）
    rclcpp::spin(node);           // 進入事件迴圈（阻塞，直到 shutdown）

    rclcpp::shutdown();           // 關閉 ROS 2
    return 0;
}
