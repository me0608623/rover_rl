/*
 * ============================================================================
 * 檔案：common_types.hpp
 * 說明：公共型別定義與常數 — 定義所有模組共用的常數、列舉、結構體與工具類別
 *
 * 包含內容：
 *   - 跟隨距離、搜尋半徑、速度比例等運動控制常數
 *   - 人工勢場法 (APF) 避障參數
 *   - 機器人框架排除區域（過濾光達掃到的自身結構點）
 *   - OpenCV 可視化除錯參數
 *   - 網路通訊埠號
 *   - 控制模式列舉 (直接控制 / 跟隨 / 導航)
 *   - 速度指令結構體
 *   - 共享狀態結構體 (執行緒安全)
 *   - 簡易 JSON 解析工具
 *
 * [Humble 相容性]
 *   此標頭檔為純 C++ 定義，不依賴任何 ROS API，
 *   Humble / Jazzy / Rolling 均可直接使用，無需修改。
 *   唯一注意：若上層 ROS 節點的 QoS 預設值在 Humble 與 Jazzy 間有差異，
 *   需在節點層級處理，此處不受影響。
 * ============================================================================
 */

#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

#include <mutex>       // std::mutex, std::lock_guard — 執行緒互斥鎖
#include <atomic>      // std::atomic — 無鎖原子操作
#include <string>      // std::string — 字串處理
#include <vector>      // std::vector — 動態陣列
#include <functional>  // std::function — 回呼函式封裝

// ----- 運動控制常數 -----
constexpr double FOLLOW_DIST = 0.7;           // 跟隨距離(公尺)：機器人與目標保持的理想距離
constexpr double TARGET_RADIUS = 0.3;         // 目標搜尋半徑(公尺)：在此半徑內的光達點視為目標
constexpr double LINEAR_SCALE_FACTOR = 0.5;   // 前後運動速度比例係數：距離誤差 × 此值 = 線速度
constexpr double ANGULAR_SCALE_FACTOR = 1.0;  // 旋轉運動速度比例係數：角度誤差 × 此值 = 角速度
constexpr double LINEAR_Y_SCALE_FACTOR = 0.0; // 橫移速度係數：差速機器人不支援橫移，保留供全向底盤切回使用
constexpr double RECTANGLE_WIDTH = 0.35;      // 路徑矩形寬度(公尺)：機器人到目標之間的通道寬度，用於路徑障礙偵測

// ----- 速度限制 -----
constexpr double MAX_LINEAR_SPEED = 1.0;      // 最大線速度(m/s)
constexpr double MAX_ANGULAR_SPEED = 1.0;     // 最大角速度(rad/s)

// ----- 人工勢場法 (APF) 避障參數 -----
constexpr double APF_INFLUENCE_DIST = 0.55;   // 障礙物影響距離(公尺)：此範圍內開始產生排斥力
constexpr double APF_REPULSE_GAIN = 0.01;     // 排斥力增益：數值越大排斥效果越強
constexpr double APF_EMERGENCY_DIST = 0.5;    // 緊急停止距離(公尺)：障礙物距離小於此值時立即停車
constexpr double APF_SLOWDOWN_DIST = 0.55;    // 減速距離(公尺)：障礙物距離小於此值時開始減速

// ----- 機器人框架排除區域（從前光達 ydlidar_front_link 中心量測） -----
// 根據 campusrover_chgh.xacro：前光達在 base_link 的 (x=0.2, y=0.08)
// 輪距 ±0.26m → 車寬約 0.52m，後輪 x=-0.381
constexpr double ROBOT_FRAME_FRONT = 0.10;    // 前方排除(公尺)：光達到車頭前緣
constexpr double ROBOT_FRAME_BACK = 0.55;     // 後方排除(公尺)：光達到車尾（0.2+0.381≈0.58，取 0.55 留餘量）
constexpr double ROBOT_FRAME_LEFT = 0.20;     // 左側排除(公尺)：半車寬 0.26 - 光達偏移 0.08 ≈ 0.18，取 0.20
constexpr double ROBOT_FRAME_RIGHT = 0.30;    // 右側排除(公尺)：半車寬 0.26 + 光達偏移 0.08 ≈ 0.34，取 0.30

// ----- OpenCV 可視化參數（除錯模式使用） -----
constexpr int WINDOW_SIZE = 800;                                          // 顯示視窗邊長(像素)
constexpr int SPEED_DISPLAY_HEIGHT = 150;                                 // 底部速度顯示區高度(像素)
constexpr int TOTAL_WINDOW_HEIGHT = WINDOW_SIZE + SPEED_DISPLAY_HEIGHT;   // 視窗總高度(像素)
constexpr double DISPLAY_WORLD_SIZE_M = 4.0;                              // 顯示的世界範圍(公尺)
constexpr double METERS_TO_PIXELS = WINDOW_SIZE / DISPLAY_WORLD_SIZE_M;   // 公尺轉像素比例
constexpr double ROBOT_Y_OFFSET_M = 0.5;                                  // 機器人在視窗中的 Y 偏移(公尺)，使機器人不在最底部

constexpr int SPEED_BAR_LENGTH = 50;           // 速度條長度(像素)
constexpr int SPEED_ARC_RADIUS = 40;           // 角速度弧形半徑(像素)

// ----- 網路通訊埠號 -----
constexpr int UDP_SEND_PORT = 8888;            // UDP 發送埠：向 Android 端傳送光達資料
constexpr int UDP_RECV_PORT = 8889;            // UDP 接收埠：接收 Android 端控制指令
constexpr int HTTP_PORT = 8080;                // HTTP 伺服器埠：提供 Web 界面
constexpr int WS_PORT = 8890;                  // WebSocket 埠：即時雙向資料傳輸

/*
 * 控制模式列舉
 * 定義機器人的三種操作模式，由上層節點或遠端指令切換
 *
 * [Humble 相容] 純 C++ 列舉，無 ROS 依賴
 */
enum ControlMode {
    MODE_DIRECT = 0,  // 直接控制模式：遙控器/手機直接操控速度
    MODE_FOLLOW = 1,  // 跟隨模式：光達追蹤目標自動跟隨
    MODE_NAV = 2      // 導航模式：由 Nav2 等導航系統接管
};

/*
 * 速度指令結構體
 * 封裝三軸速度分量，對應 geometry_msgs/Twist 的線速度與角速度
 *
 * [Humble 相容] 純 C++ 結構體，不依賴 ROS 訊息型別
 */
struct VelocityCmd {
    double vx = 0.0;  // 前後線速度(m/s)：正值前進，負值後退
    double vy = 0.0;  // 橫向線速度(m/s)：差速機器人通常為 0
    double wz = 0.0;  // 角速度(rad/s)：正值左轉，負值右轉
};

/*
 * 共享狀態結構體
 * 各模組間共享的執行緒安全資料容器
 * 使用 mutex 保護可變資料，atomic 處理狀態旗標
 *
 * [Humble 相容] 純 C++ 執行緒同步機制，
 *   不使用 ROS 特有的 callback group 或 executor，
 *   Humble/Jazzy 均適用。若未來改用 rclcpp 內建的
 *   MutuallyExclusiveCallbackGroup，需重構此處的鎖機制。
 */
struct SharedState {
    // --- 目標位置 (需要 mutex 保護) ---
    std::mutex target_mutex;                  // 目標位置互斥鎖
    double target_x = FOLLOW_DIST;            // 目標 X 座標(公尺)：初始值為跟隨距離
    double target_y = 0.0;                    // 目標 Y 座標(公尺)：初始值為正前方

    // --- 速度指令快取 (需要 mutex 保護) ---
    std::mutex velocity_mutex;                // 速度快取互斥鎖
    double cached_vx = 0.0;                   // 快取的前後線速度
    double cached_vy = 0.0;                   // 快取的橫向線速度
    double cached_wz = 0.0;                   // 快取的角速度

    // --- 光達點雲快取 (需要 mutex 保護) ---
    std::mutex scan_data_mutex;               // 點雲資料互斥鎖
    std::vector<std::pair<double, double>> cached_points;  // 快取的 (x, y) 點雲座標

    // --- 直接控制指令 (需要 mutex 保護) ---
    std::mutex direct_cmd_mutex;              // 直接控制指令互斥鎖
    double direct_vx = 0.0;                   // 直接控制前後線速度
    double direct_vy = 0.0;                   // 直接控制橫向線速度
    double direct_wz = 0.0;                   // 直接控制角速度

    // --- 原子狀態旗標 (無需 mutex，lock-free) ---
    std::atomic<bool> active{false};                    // 系統是否啟動
    std::atomic<bool> is_moving_enabled{false};         // 是否允許移動（安全開關）
    std::atomic<int> control_mode{MODE_FOLLOW};         // 當前控制模式

    /*
     * 取得目標位置（執行緒安全）
     * 透過 lock_guard 確保讀取時不會被其他執行緒修改
     */
    void getTarget(double& x, double& y) {
        std::lock_guard<std::mutex> lock(target_mutex);  // 自動加鎖/解鎖
        x = target_x;
        y = target_y;
    }

    /*
     * 設定目標位置（執行緒安全）
     * 由光達追蹤器或遠端指令更新
     */
    void setTarget(double x, double y) {
        std::lock_guard<std::mutex> lock(target_mutex);
        target_x = x;
        target_y = y;
    }

    /*
     * 取得速度快取（執行緒安全）
     * 供 Web 廣播等模組讀取當前速度
     */
    void getVelocity(double& vx, double& vy, double& wz) {
        std::lock_guard<std::mutex> lock(velocity_mutex);
        vx = cached_vx;
        vy = cached_vy;
        wz = cached_wz;
    }

    /*
     * 設定速度快取（執行緒安全）
     * 由控制模組在計算完速度後寫入
     */
    void setVelocity(double vx, double vy, double wz) {
        std::lock_guard<std::mutex> lock(velocity_mutex);
        cached_vx = vx;
        cached_vy = vy;
        cached_wz = wz;
    }

    /*
     * 取得直接控制指令（執行緒安全）
     * 供定時器回呼讀取遙控速度
     */
    void getDirectCmd(double& vx, double& vy, double& wz) {
        std::lock_guard<std::mutex> lock(direct_cmd_mutex);
        vx = direct_vx;
        vy = direct_vy;
        wz = direct_wz;
    }

    /*
     * 設定直接控制指令（執行緒安全）
     * 由通訊模組接收遙控指令後寫入
     */
    void setDirectCmd(double vx, double vy, double wz) {
        std::lock_guard<std::mutex> lock(direct_cmd_mutex);
        direct_vx = vx;
        direct_vy = vy;
        direct_wz = wz;
    }

    /*
     * 取得點雲資料（執行緒安全，回傳複本）
     * 回傳 vector 複本以避免外部持有期間被覆寫
     */
    std::vector<std::pair<double, double>> getPoints() {
        std::lock_guard<std::mutex> lock(scan_data_mutex);
        return cached_points;  // 回傳深拷貝
    }

    /*
     * 設定點雲資料（執行緒安全，使用 move 語義避免額外複製）
     * 由光達回呼處理完畢後寫入
     */
    void setPoints(std::vector<std::pair<double, double>>&& points) {
        std::lock_guard<std::mutex> lock(scan_data_mutex);
        cached_points = std::move(points);  // 移動語義：零複製成本
    }
};

/*
 * 簡易 JSON 解析工具
 * 手動字串搜尋實作，不依賴第三方 JSON 函式庫
 * 適合小型嵌入式場景，但不支援巢狀物件或跳脫字元
 *
 * [Humble 相容] 純 C++ 字串操作，無 ROS 依賴
 */
class JsonParser {
public:
    /*
     * 從 JSON 字串中提取指定 key 的數值
     * 例如：extractNumber("{\"x\":1.5}", "x") 回傳 1.5
     */
    static double extractNumber(const std::string& str, const std::string& key) {
        size_t pos = str.find("\"" + key + "\"");          // 找到 key 的位置
        if (pos == std::string::npos) return 0;             // 找不到則回傳 0
        pos = str.find(":", pos);                           // 找到冒號
        if (pos == std::string::npos) return 0;
        pos++;                                              // 跳過冒號
        while (pos < str.size() && (str[pos] == ' ' || str[pos] == '\t')) pos++;  // 跳過空白
        size_t end = pos;
        while (end < str.size() && (std::isdigit(str[end]) || str[end] == '.' || str[end] == '-')) end++;  // 掃描數字字元
        if (end > pos) {
            return std::stod(str.substr(pos, end - pos));   // 轉換為 double
        }
        return 0;
    }

    /*
     * 解析 JSON 中的 x, y 座標
     * 回傳 true 表示解析成功（目前總是回傳 true）
     */
    static bool parseXY(const std::string& json, double& x, double& y) {
        x = extractNumber(json, "x");  // 提取 x 值
        y = extractNumber(json, "y");  // 提取 y 值
        return true;
    }

    /*
     * 檢查 JSON 訊息是否包含指定的 type 欄位
     * 支援冒號後有無空格兩種格式
     */
    static bool hasType(const std::string& json, const std::string& type) {
        return json.find("\"type\":\"" + type + "\"") != std::string::npos ||     // 無空格格式
               json.find("\"type\": \"" + type + "\"") != std::string::npos;      // 有空格格式
    }

    /*
     * 檢查 JSON 中指定 key 的布林值是否為 true
     */
    static bool getBool(const std::string& json, const std::string& key) {
        return json.find("\"" + key + "\":true") != std::string::npos ||          // 無空格格式
               json.find("\"" + key + "\": true") != std::string::npos;           // 有空格格式
    }

    /*
     * 從 JSON 字串中提取指定 key 的字串值
     * 例如：extractString("{\"name\":\"robot\"}", "name") 回傳 "robot"
     */
    static std::string extractString(const std::string& str, const std::string& key) {
        size_t pos = str.find("\"" + key + "\"");           // 找到 key 的位置
        if (pos == std::string::npos) return "";
        pos = str.find(":", pos);                            // 找到冒號
        if (pos == std::string::npos) return "";
        pos = str.find("\"", pos);                           // 找到值的起始引號
        if (pos == std::string::npos) return "";
        size_t start = pos + 1;                              // 跳過引號
        size_t end = str.find("\"", start);                  // 找到值的結束引號
        if (end == std::string::npos) return "";
        return str.substr(start, end - start);               // 擷取字串值
    }
};

#endif // COMMON_TYPES_HPP
