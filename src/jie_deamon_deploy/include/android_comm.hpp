/*
 * ============================================================================
 * 檔案：android_comm.hpp
 * 說明：Android UDP 通訊模組 — 透過 UDP 與 Android 手機 App 雙向通訊
 *
 * 功能：
 *   - UDP 接收迴圈：監聽 Android 端發送的控制指令 (JSON 格式)
 *   - UDP 發送：將光達掃描資料傳送到 Android 端進行可視化
 *   - 自動記錄客戶端 IP（首次收到封包時）
 *
 * 通訊協議：
 *   - 接收埠：UDP_RECV_PORT (8889)
 *   - 發送埠：UDP_SEND_PORT (8888)
 *   - 訊息格式：JSON 字串
 *
 * [Humble 相容性]
 *   此模組為純 POSIX socket 實作，不依賴任何 ROS API，
 *   Humble / Jazzy / Rolling 均可直接使用。
 *   使用 Linux 原生 socket API (sys/socket.h)，不支援 Windows。
 *   定義了 SOCKET / INVALID_SOCKET / closesocket 巨集以保持跨平台風格，
 *   但目前僅支援 Linux（close 取代 closesocket）。
 * ============================================================================
 */

#ifndef ANDROID_COMM_HPP
#define ANDROID_COMM_HPP

#include "common_types.hpp"      // 共用常數與共享狀態
#include <thread>                // std::thread — 接收迴圈執行緒
#include <sstream>               // std::ostringstream — 字串格式化
#include <iomanip>               // 數值格式化
#include <functional>            // std::function — 回呼封裝
#include <sys/socket.h>          // socket(), bind(), sendto(), recvfrom()
#include <netinet/in.h>          // sockaddr_in, htons()
#include <arpa/inet.h>           // inet_ntop(), inet_pton()
#include <unistd.h>              // close()

// ----- 跨平台 socket 相容巨集（目前僅支援 Linux） -----
#ifndef SOCKET
#define SOCKET int                             // socket 描述符型別
#endif
#ifndef INVALID_SOCKET
#define INVALID_SOCKET -1                      // 無效 socket 標記
#endif
#ifndef SOCKET_ERROR
#define SOCKET_ERROR -1                        // socket 錯誤標記
#endif
#define closesocket close                      // Linux 用 close() 關閉 socket

/*
 * Android 通訊管理器
 * 管理 UDP 收發 socket 的生命週期，在獨立執行緒中接收指令
 *
 * [Humble 相容] 純 POSIX socket + std::thread，不使用 ROS 通訊機制
 */
class AndroidCommManager {
public:
    using LogCallback = std::function<void(const std::string&)>;  // 日誌回呼型別

    AndroidCommManager(SharedState& state) : state_(state) {}  // 建構函式：注入共享狀態
    ~AndroidCommManager() { stop(); }                           // 解構函式：確保資源釋放

    /* 設定日誌回呼（通常綁定到 RCLCPP_INFO） */
    void setLogCallback(LogCallback cb) { log_callback_ = std::move(cb); }

    /* 啟動 UDP 通訊（建立 socket 並啟動接收執行緒） */
    void start();

    /* 停止 UDP 通訊（關閉 socket 並等待執行緒結束） */
    void stop();

    /* 將光達掃描資料透過 UDP 發送到 Android 端 */
    void sendScanData();

private:
    SharedState& state_;                                   // 共享狀態參考
    SOCKET udp_send_socket_ = INVALID_SOCKET;             // UDP 發送 socket
    SOCKET udp_recv_socket_ = INVALID_SOCKET;             // UDP 接收 socket
    std::thread recv_thread_;                              // 接收迴圈執行緒
    std::atomic<bool> running_{false};                     // 執行狀態旗標（原子操作）

    std::mutex client_mutex_;                              // 客戶端資訊互斥鎖
    std::string client_ip_;                                // 已連接的 Android 客戶端 IP
    bool client_connected_ = false;                        // 是否有客戶端已連接

    LogCallback log_callback_;                             // 日誌回呼函式

    /* 內部日誌輸出（若有回呼則呼叫，否則靜默） */
    void log(const std::string& msg) { if (log_callback_) log_callback_(msg); }

    /* UDP 接收迴圈：在獨立執行緒中持續監聽 Android 指令 */
    void receiveLoop();

    /* 解析收到的 JSON 控制指令並更新共享狀態 */
    void parseCommand(const std::string& json);
};

#endif // ANDROID_COMM_HPP
