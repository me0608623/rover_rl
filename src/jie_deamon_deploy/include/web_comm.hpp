/*
 * ============================================================================
 * 檔案：web_comm.hpp
 * 說明：Web 通訊模組 — 整合 HTTP 靜態檔案伺服器與 WebSocket 即時通訊
 *
 * 功能：
 *   - HTTP 伺服器：提供 Web 前端靜態檔案（使用 cpp-httplib）
 *   - WebSocket 伺服器：與瀏覽器建立即時雙向連線
 *   - 接收 WebSocket 控制指令（直接控制、動作指令等）
 *   - 廣播光達資料與速度狀態到所有已連接的 WebSocket 客戶端
 *   - 自動偵測 Web 前端根目錄
 *
 * 通訊埠：
 *   - HTTP：HTTP_PORT (8080)
 *   - WebSocket：WS_PORT (8890)
 *
 * [Humble 相容性]
 *   - cpp-httplib (httplib.h) 為純 C++ header-only 函式庫，與 ROS 版本無關
 *   - WebSocket 使用 POSIX socket 手動實作，不依賴 ROS 通訊
 *   - 注意：CPPHTTPLIB_NO_EXCEPTIONS 巨集已定義，異常處理改為回傳值檢查
 *   - 注意：CPPHTTPLIB_OPENSSL_SUPPORT 已取消定義（不使用 HTTPS）
 *   - Humble/Jazzy/Rolling 均可直接使用
 * ============================================================================
 */

#ifndef WEB_COMM_HPP
#define WEB_COMM_HPP

#include "common_types.hpp"        // 共用常數與共享狀態
#define CPPHTTPLIB_NO_EXCEPTIONS   // 停用 cpp-httplib 的例外機制
#undef CPPHTTPLIB_OPENSSL_SUPPORT  // 不使用 OpenSSL（僅 HTTP，不用 HTTPS）
#include "httplib.h"               // cpp-httplib：header-only HTTP 伺服器函式庫

#include <thread>                  // std::thread — HTTP/WebSocket 伺服器執行緒
#include <set>                     // std::set — WebSocket 客戶端 fd 集合
#include <sstream>                 // std::ostringstream — 字串格式化
#include <iomanip>                 // 數值格式化
#include <fstream>                 // std::ifstream — 檔案存在性檢測
#include <functional>              // std::function — 回呼封裝
#include <sys/socket.h>            // socket(), bind(), listen(), accept()
#include <netinet/in.h>            // sockaddr_in, htons()
#include <arpa/inet.h>             // inet_ntop()
#include <unistd.h>                // close()
#include <poll.h>                  // poll() — 非阻塞 I/O 多工
#include <ifaddrs.h>               // getifaddrs() — 取得本機 IP 位址

// ----- 跨平台 socket 相容巨集 -----
#ifndef SOCKET
#define SOCKET int                              // socket 描述符型別
#endif
#ifndef INVALID_SOCKET
#define INVALID_SOCKET -1                       // 無效 socket 標記
#endif
#define closesocket close                       // Linux 用 close() 關閉 socket

/*
 * Web 通訊管理器
 * 整合 HTTP 靜態檔案伺服器與 WebSocket 即時通訊伺服器
 * 管理所有 Web 相關的通訊生命週期
 *
 * [Humble 相容] 使用 cpp-httplib + POSIX socket，不依賴 ROS 通訊層
 */
class WebCommManager {
public:
    using LogCallback = std::function<void(const std::string&)>;              // 日誌回呼
    using DirectCmdCallback = std::function<void(double, double, double)>;    // 直接控制指令回呼 (vx, vy, wz)
    using ActionCmdCallback = std::function<void(const std::string&)>;        // 動作指令回呼 (JSON 字串)
    using MuxModeCallback = std::function<void(int)>;                         // 底盤模式切換回呼 (0=放鬆,1=停止,2=手動,3=自動)

    WebCommManager(SharedState& state) : state_(state) {}  // 建構函式：注入共享狀態
    ~WebCommManager() { stop(); }                           // 解構函式：確保資源釋放

    /* 設定日誌回呼 */
    void setLogCallback(LogCallback cb) { log_callback_ = std::move(cb); }
    /* 設定直接控制指令回呼 */
    void setDirectCmdCallback(DirectCmdCallback cb) { direct_cmd_callback_ = std::move(cb); }
    /* 設定動作指令回呼 */
    void setActionCmdCallback(ActionCmdCallback cb) { action_cmd_callback_ = std::move(cb); }
    /* 設定底盤模式切換回呼 */
    void setMuxModeCallback(MuxModeCallback cb) { mux_mode_callback_ = std::move(cb); }
    /* 手動設定 Web 前端根目錄 */
    void setWebRoot(const std::string& root) { web_root_ = root; }

    /*
     * 自動偵測 Web 前端根目錄
     * 依序嘗試提供的路徑列表，找到包含 index.html 的目錄即使用
     */
    void autoDetectWebRoot(const std::vector<std::string>& paths) {
        if (!web_root_.empty()) return;  // 已手動設定則跳過
        for (const auto& path : paths) {
            std::ifstream test(path + "/index.html");  // 檢查 index.html 是否存在
            if (test.good()) { web_root_ = path; break; }
        }
    }

    /* 取得本機 IP 位址（用於在日誌中顯示存取 URL） */
    std::string getLocalIP();
    /* 啟動 HTTP 與 WebSocket 伺服器 */
    void start();
    /* 停止所有伺服器並釋放資源 */
    void stop();
    /* 將光達資料與速度狀態廣播到所有 WebSocket 客戶端 */
    void broadcastData();

private:
    SharedState& state_;                                    // 共享狀態參考
    std::string web_root_;                                  // Web 前端靜態檔案根目錄
    std::atomic<bool> running_{false};                      // 伺服器執行狀態

    std::unique_ptr<httplib::Server> http_server_;          // cpp-httplib HTTP 伺服器實例
    std::thread http_thread_;                               // HTTP 伺服器執行緒
    SOCKET ws_server_socket_ = INVALID_SOCKET;              // WebSocket 伺服器 socket
    std::thread ws_thread_;                                 // WebSocket 接收迴圈執行緒
    std::mutex ws_clients_mutex_;                           // WebSocket 客戶端列表互斥鎖
    std::set<int> ws_clients_;                              // 已連接的 WebSocket 客戶端 fd 集合
    LogCallback log_callback_;                              // 日誌回呼
    DirectCmdCallback direct_cmd_callback_;                 // 直接控制指令回呼
    ActionCmdCallback action_cmd_callback_;                 // 動作指令回呼
    MuxModeCallback mux_mode_callback_;                     // 底盤模式切換回呼

    /* 內部日誌輸出 */
    void log(const std::string& msg) { if (log_callback_) log_callback_(msg); }

    /* 啟動 WebSocket 伺服器（建立 socket、綁定、監聽） */
    void startWebSocketServer();
    /* WebSocket 連線接受迴圈 */
    void wsAcceptLoop();
    /* 處理單一 WebSocket 客戶端連線（在獨立執行緒中執行） */
    void handleWebSocketClient(int fd);
    /* 解析並處理 WebSocket 訊息 */
    void handleWebSocketMessage(const std::string& message);
    /* 計算 WebSocket 握手 Accept Key（SHA-1 + Base64） */
    std::string computeWebSocketAcceptKey(const std::string& key);
    /* 從 WebSocket 連線讀取一則訊息 */
    bool recvWebSocketMessage(int fd, std::string& message);
    /* 向 WebSocket 連線發送一則訊息 */
    bool sendWebSocketMessage(int fd, const std::string& message);
    /* 向所有已連接的 WebSocket 客戶端廣播訊息 */
    void broadcastToWebSockets(const std::string& message);
};

#endif // WEB_COMM_HPP
