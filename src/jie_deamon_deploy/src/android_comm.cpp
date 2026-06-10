/*
 * 檔案說明：android_comm.cpp — Android UDP 通訊模組實作
 *
 * 功能概述：
 * - 透過 UDP 接收 Android App 傳來的 JSON 控制指令
 * - 透過 UDP 將雷射雷達掃描資料傳送回 Android App
 * - 支援的指令：heartbeat, set_target, set_moving, set_active
 * - 採用非阻塞式設計：獨立接收執行緒 + 1 秒逾時
 *
 * Humble 相容性提示：
 * - 本檔案使用原生 POSIX socket API，完全不依賴 ROS 2 API
 * - 在 Humble / Jazzy / Rolling 中均可直接編譯，無需修改
 * - INVALID_SOCKET 與 SOCKET_ERROR 為自定義常數（定義於 android_comm.hpp）
 *   在 Linux 上分別對應 -1 和 -1，closesocket 對應 close
 */

#include "android_comm.hpp"  // Android 通訊管理器標頭檔

/*
 * start — 啟動 Android 通訊模組
 * 建立 UDP 傳送與接收 socket，綁定接收連接埠，設定逾時，並啟動接收執行緒。
 */
void AndroidCommManager::start() {
    if (running_) return;  // 防止重複啟動

    // 建立 UDP 傳送 socket
    udp_send_socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);  // SOCK_DGRAM = UDP
    if (udp_send_socket_ == INVALID_SOCKET) {
        log("建立傳送 socket 失敗");
    }

    // 建立 UDP 接收 socket
    udp_recv_socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);  // SOCK_DGRAM = UDP
    if (udp_recv_socket_ == INVALID_SOCKET) {
        log("建立接收 socket 失敗");
        return;  // 接收 socket 必要，失敗則放棄啟動
    }

    // 綁定接收連接埠
    struct sockaddr_in recv_addr{};          // 接收位址結構
    recv_addr.sin_family = AF_INET;          // IPv4
    recv_addr.sin_addr.s_addr = INADDR_ANY;  // 監聽所有介面
    recv_addr.sin_port = htons(UDP_RECV_PORT);  // 設定接收連接埠（網路位元組序）

    if (bind(udp_recv_socket_, (struct sockaddr*)&recv_addr, sizeof(recv_addr)) == SOCKET_ERROR) {
        log("綁定 UDP 接收連接埠失敗");
    } else {
        log("UDP 接收連接埠 " + std::to_string(UDP_RECV_PORT) + " 已綁定");
    }

    // 設定接收逾時（1 秒），避免 recvfrom 永久阻塞
    struct timeval tv;
    tv.tv_sec = 1;   // 1 秒逾時
    tv.tv_usec = 0;
    setsockopt(udp_recv_socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));  // 套用逾時設定

    running_ = true;  // 設定執行旗標
    recv_thread_ = std::thread(&AndroidCommManager::receiveLoop, this);  // 啟動接收迴圈執行緒
}

/*
 * stop — 停止 Android 通訊模組
 * 設定停止旗標，等待接收執行緒結束，關閉所有 socket。
 */
void AndroidCommManager::stop() {
    running_ = false;  // 設定停止旗標
    if (recv_thread_.joinable()) recv_thread_.join();  // 等待接收執行緒結束
    if (udp_send_socket_ != INVALID_SOCKET) { closesocket(udp_send_socket_); udp_send_socket_ = INVALID_SOCKET; }  // 關閉傳送 socket
    if (udp_recv_socket_ != INVALID_SOCKET) { closesocket(udp_recv_socket_); udp_recv_socket_ = INVALID_SOCKET; }  // 關閉接收 socket
}

/*
 * receiveLoop — UDP 接收迴圈（在獨立執行緒中執行）
 * 持續從 UDP 接收連接埠讀取資料，自動記錄客戶端 IP（用於回傳資料），
 * 然後將收到的 JSON 字串交給 parseCommand 解析。
 */
void AndroidCommManager::receiveLoop() {
    char buffer[4096];  // 接收緩衝區
    struct sockaddr_in client_addr{};  // 客戶端位址
    socklen_t client_len = sizeof(client_addr);

    while (running_) {  // 持續執行直到停止
        int recv_len = recvfrom(udp_recv_socket_, buffer, sizeof(buffer) - 1, 0,
                                (struct sockaddr*)&client_addr, &client_len);  // 接收 UDP 封包

        if (recv_len > 0) {  // 成功接收到資料
            buffer[recv_len] = '\0';  // 加入字串結尾

            char ip_str[INET_ADDRSTRLEN];  // IP 位址字串緩衝區
            inet_ntop(AF_INET, &(client_addr.sin_addr), ip_str, INET_ADDRSTRLEN);  // 轉換客戶端 IP 為可讀字串

            {
                std::lock_guard<std::mutex> lock(client_mutex_);  // 鎖定客戶端資訊
                client_ip_ = ip_str;       // 記錄客戶端 IP（用於 sendScanData 回傳）
                client_connected_ = true;  // 標記客戶端已連線
            }

            parseCommand(std::string(buffer));  // 解析收到的 JSON 指令
        }
    }
}

/*
 * parseCommand — 解析 Android App 傳來的 JSON 控制指令
 * 支援的指令類型：
 * - heartbeat：心跳封包，僅用於維持連線狀態（不處理）
 * - set_target：設定追蹤目標座標 (x, y)
 * - set_moving：設定運動致能（true/false）
 * - set_active：設定跟隨功能啟用（true/false）
 */
void AndroidCommManager::parseCommand(const std::string& json) {
    if (JsonParser::hasType(json, "heartbeat")) {
        // 心跳封包，僅確認連線存活，不做任何處理
    }
    else if (JsonParser::hasType(json, "set_target")) {  // 設定目標座標
        double x, y;
        if (JsonParser::parseXY(json, x, y)) {  // 解析 x, y
            state_.setTarget(x, y);  // 寫入共享狀態
            log("設定目標位置: (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        }
    }
    else if (JsonParser::hasType(json, "set_moving")) {  // 設定運動致能
        bool enabled = JsonParser::getBool(json, "enabled");  // 解析布林值
        state_.is_moving_enabled.store(enabled);  // 原子寫入
        log(std::string("運動致能: ") + (enabled ? "開啟" : "關閉"));
    }
    else if (JsonParser::hasType(json, "set_active")) {  // 設定跟隨功能
        bool active = JsonParser::getBool(json, "active");  // 解析布林值
        state_.active.store(active);  // 原子寫入
        log(std::string("跟隨功能: ") + (active ? "開啟" : "關閉"));
    }
}

/*
 * sendScanData — 透過 UDP 傳送雷射雷達掃描資料給 Android App
 * 將目標座標、運動致能、跟隨狀態、掃描點雲打包為 JSON，
 * 點雲會進行降採樣（最多 180 點）以減少 UDP 封包大小。
 * 僅在有客戶端連線時才傳送。
 */
void AndroidCommManager::sendScanData() {
    std::string client_ip;
    {
        std::lock_guard<std::mutex> lock(client_mutex_);  // 鎖定客戶端資訊
        if (!client_connected_) return;  // 無客戶端連線則直接返回
        client_ip = client_ip_;          // 複製客戶端 IP
    }

    double tx, ty;
    state_.getTarget(tx, ty);            // 取得目標座標
    auto points = state_.getPoints();    // 取得掃描點雲

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);  // 固定小數點 3 位
    oss << "{\"type\":\"scan_data\",";  // JSON 訊息類型
    oss << "\"target\":{\"x\":" << tx << ",\"y\":" << ty << "},";  // 目標座標
    oss << "\"is_moving_enabled\":" << (state_.is_moving_enabled.load() ? "true" : "false") << ",";  // 運動致能
    oss << "\"is_active\":" << (state_.active.load() ? "true" : "false") << ",";  // 跟隨啟用
    oss << "\"points\":[";  // 點雲陣列開始

    size_t max_points = 180;  // 最大傳輸點數
    size_t step = points.size() > max_points ? points.size() / max_points : 1;  // 降採樣步長
    bool first = true;  // 控制 JSON 逗號
    for (size_t i = 0; i < points.size(); i += step) {  // 以步長遍歷
        if (!first) oss << ",";
        oss << "{\"x\":" << points[i].first << ",\"y\":" << points[i].second << "}";  // 輸出點座標
        first = false;
    }
    oss << "]}";  // 點雲陣列結束

    std::string json = oss.str();  // 序列化完成的 JSON 字串

    struct sockaddr_in dest_addr{};          // 目的位址結構
    dest_addr.sin_family = AF_INET;          // IPv4
    dest_addr.sin_port = htons(UDP_SEND_PORT);  // 設定傳送連接埠
    inet_pton(AF_INET, client_ip.c_str(), &dest_addr.sin_addr);  // 將 IP 字串轉為二進位

    sendto(udp_send_socket_, json.c_str(), json.size(), 0,
           (struct sockaddr*)&dest_addr, sizeof(dest_addr));  // 透過 UDP 傳送
}
