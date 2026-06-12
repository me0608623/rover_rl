/*
 * 檔案說明：web_comm.cpp — Web 通訊模組實作
 *
 * 功能概述：
 * - 提供 HTTP REST API（使用 cpp-httplib 函式庫）
 * - 提供原生 WebSocket 伺服器（手動實作握手與幀解析）
 * - HTTP API 端點：/api/status, /api/set_target, /api/set_moving, /api/set_active
 * - WebSocket 支援即時雙向通訊：掃描資料廣播 + 指令接收
 *
 * Humble 相容性提示：
 * - 本檔案使用原生 POSIX socket，完全不依賴 ROS 2 API
 * - 在 Humble / Jazzy / Rolling 中均可直接編譯，無需修改
 * - WebSocket 握手透過 popen 呼叫 openssl 計算 SHA-1，需確保系統安裝 openssl
 */

#include "web_comm.hpp"  // Web 通訊管理器標頭檔

/*
 * getLocalIP — 取得本機非迴環 IP 位址
 * 優先回傳無線網卡（名稱以 'w' 開頭）的 IP 位址。
 * 若找不到非迴環 IP，則回傳 "localhost"。
 */
std::string WebCommManager::getLocalIP() {
    struct ifaddrs *ifAddrStruct = NULL;  // 網路介面鏈結串列指標
    void *tmpAddrPtr = NULL;               // 暫存位址指標
    std::string ip = "localhost";          // 預設回傳值

    getifaddrs(&ifAddrStruct);  // 取得所有網路介面資訊
    for (auto ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {  // 遍歷所有介面
        if (!ifa->ifa_addr) continue;  // 跳過無位址的介面
        if (ifa->ifa_addr->sa_family == AF_INET) {  // 只處理 IPv4
            tmpAddrPtr = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;  // 取出 IPv4 位址
            char addressBuffer[INET_ADDRSTRLEN];  // 位址字串緩衝區
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);  // 轉換為可讀字串
            std::string ipStr(addressBuffer);   // IP 位址字串
            std::string ifName(ifa->ifa_name);  // 介面名稱（如 eth0, wlan0）
            if (ipStr != "127.0.0.1" && !ipStr.empty()) {  // 排除迴環位址
                ip = ipStr;  // 記錄此 IP
                if (ifName.find("w") == 0) break;  // 若為無線網卡（wlan*），優先選用並跳出
            }
        }
    }
    if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);  // 釋放介面資訊記憶體
    return ip;
}

/*
 * start — 啟動 Web 通訊模組
 * 依序啟動 HTTP 伺服器（含 REST API 註冊）與 WebSocket 伺服器。
 */
void WebCommManager::start() {
    if (running_) return;  // 防止重複啟動
    running_ = true;       // 設定執行旗標

    http_server_ = std::make_unique<httplib::Server>();  // 建立 HTTP 伺服器實例
    if (!web_root_.empty()) http_server_->set_mount_point("/", web_root_);  // 掛載靜態檔案目錄

    // API: 取得目前狀態（GET /api/status）
    http_server_->Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);  // 固定小數點 3 位
        oss << "{\"is_moving_enabled\":" << (state_.is_moving_enabled.load() ? "true" : "false");  // 運動致能狀態
        oss << ",\"is_active\":" << (state_.active.load() ? "true" : "false");  // 跟隨功能狀態
        double tx, ty; state_.getTarget(tx, ty);  // 取得目前目標座標
        oss << ",\"target\":{\"x\":" << tx << ",\"y\":" << ty << "}}";  // 目標位置
        res.set_content(oss.str(), "application/json");  // 回傳 JSON
    });

    // API: 設定目標座標（POST /api/set_target）
    http_server_->Post("/api/set_target", [this](const httplib::Request& req, httplib::Response& res) {
        double x = 0, y = 0;
        if (JsonParser::parseXY(req.body, x, y)) {  // 從 JSON 中解析 x, y 座標
            state_.setTarget(x, y);  // 寫入共享狀態
            log("Web 設定目標: (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        }
        res.set_content("{\"ok\":true}", "application/json");  // 回傳成功
    });

    // API: 設定運動致能（POST /api/set_moving）
    http_server_->Post("/api/set_moving", [this](const httplib::Request& req, httplib::Response& res) {
        bool enabled = req.body.find("true") != std::string::npos;  // 簡易解析布林值
        state_.is_moving_enabled.store(enabled);  // 原子寫入運動致能旗標
        log(std::string("Web 運動致能: ") + (enabled ? "開啟" : "關閉"));
        res.set_content("{\"ok\":true}", "application/json");  // 回傳成功
    });

    // API: 設定跟隨啟用（POST /api/set_active）
    http_server_->Post("/api/set_active", [this](const httplib::Request& req, httplib::Response& res) {
        bool active = req.body.find("true") != std::string::npos;  // 簡易解析布林值
        state_.active.store(active);  // 原子寫入跟隨旗標
        log(std::string("Web 跟隨功能: ") + (active ? "開啟" : "關閉"));
        res.set_content("{\"ok\":true}", "application/json");  // 回傳成功
    });

    // 在獨立執行緒中啟動 HTTP 伺服器（listen 會阻塞）
    http_thread_ = std::thread([this]() {
        log("HTTP 伺服器啟動在連接埠 " + std::to_string(HTTP_PORT));
        http_server_->listen("0.0.0.0", HTTP_PORT);  // 監聽所有介面上的 HTTP_PORT
    });

    startWebSocketServer();  // 啟動 WebSocket 伺服器
}

/*
 * stop — 停止 Web 通訊模組
 * 關閉 HTTP 伺服器、WebSocket 伺服器，並清理所有客戶端連線與執行緒。
 */
void WebCommManager::stop() {
    running_ = false;  // 設定停止旗標
    if (http_server_) http_server_->stop();  // 停止 HTTP 伺服器
    if (ws_server_socket_ >= 0) { close(ws_server_socket_); ws_server_socket_ = -1; }  // 關閉 WebSocket 伺服器 socket
    {
        std::lock_guard<std::mutex> lock(ws_clients_mutex_);  // 鎖定客戶端集合
        for (int fd : ws_clients_) close(fd);  // 關閉所有 WebSocket 客戶端連線
        ws_clients_.clear();  // 清空客戶端集合
    }
    if (http_thread_.joinable()) http_thread_.join();  // 等待 HTTP 執行緒結束
    if (ws_thread_.joinable()) ws_thread_.join();      // 等待 WebSocket 執行緒結束
}

/*
 * broadcastData — 透過 WebSocket 廣播掃描資料給所有已連線客戶端
 * 包含目標座標、速度、控制模式、運動致能、掃描點雲等資訊。
 * 點雲資料會進行降採樣（最多 180 點）以減少傳輸量。
 */
void WebCommManager::broadcastData() {
    if (!running_) return;  // 未啟動則直接返回
    double tx, ty; state_.getTarget(tx, ty);          // 取得目標座標
    double vx, vy, wz; state_.getVelocity(vx, vy, wz);  // 取得目前速度
    auto points = state_.getPoints();                  // 取得掃描點雲資料

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);  // 固定小數點 3 位
    oss << "{\"type\":\"scan_data\",";  // JSON 訊息類型
    oss << "\"target\":{\"x\":" << tx << ",\"y\":" << ty << "},";  // 目標座標
    oss << "\"is_moving_enabled\":" << (state_.is_moving_enabled.load() ? "true" : "false") << ",";  // 運動致能
    oss << "\"is_active\":" << (state_.active.load() ? "true" : "false") << ",";  // 跟隨啟用
    oss << "\"mode\":" << state_.control_mode.load() << ",";  // 控制模式
    oss << "\"velocity\":{\"vx\":" << vx << ",\"vy\":" << vy << ",\"wz\":" << wz << "},";  // 速度三軸
    oss << "\"rectangle_width\":" << RECTANGLE_WIDTH << ",\"points\":[";  // 偵測矩形寬度 + 點雲陣列

    size_t max_points = 180;  // 最大傳輸點數（降採樣）
    size_t step = points.size() > max_points ? points.size() / max_points : 1;  // 計算步長
    bool first = true;  // 用於控制 JSON 陣列逗號
    for (size_t i = 0; i < points.size(); i += step) {  // 以步長遍歷點雲
        if (!first) oss << ",";
        oss << "{\"x\":" << points[i].first << ",\"y\":" << points[i].second << "}";  // 輸出每個點的 x, y
        first = false;
    }
    oss << "]}";
    broadcastToWebSockets(oss.str());  // 傳送給所有 WebSocket 客戶端
}

/*
 * broadcastRouteNodes — 廣播路由節點列表給所有 WebSocket 客戶端
 * nodes_json 已是完整 JSON 字串，直接廣播。
 */
void WebCommManager::broadcastRouteNodes(const std::string& nodes_json) {
    if (!running_) return;
    broadcastToWebSockets(nodes_json);
}

/*
 * startWebSocketServer — 建立並啟動 WebSocket 伺服器
 * 使用原生 POSIX socket，在 WS_PORT 上監聽連線。
 * 新連線由獨立執行緒處理（wsAcceptLoop）。
 */
void WebCommManager::startWebSocketServer() {
    ws_server_socket_ = socket(AF_INET, SOCK_STREAM, 0);  // 建立 TCP socket
    if (ws_server_socket_ < 0) { log("建立 WebSocket 伺服器 socket 失敗"); return; }

    int opt = 1;
    setsockopt(ws_server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));  // 允許位址重用

    struct sockaddr_in addr{};             // 伺服器位址結構
    addr.sin_family = AF_INET;             // IPv4
    addr.sin_addr.s_addr = INADDR_ANY;     // 監聽所有介面
    addr.sin_port = htons(WS_PORT);        // 設定 WebSocket 連接埠

    if (bind(ws_server_socket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {  // 綁定位址
        log("綁定 WebSocket 連接埠失敗"); close(ws_server_socket_); ws_server_socket_ = -1; return;
    }
    listen(ws_server_socket_, 10);  // 開始監聽，佇列長度 10
    log("WebSocket 伺服器啟動在連接埠 " + std::to_string(WS_PORT));
    ws_thread_ = std::thread(&WebCommManager::wsAcceptLoop, this);  // 啟動接受連線的執行緒
}

/*
 * wsAcceptLoop — WebSocket 連線接受迴圈
 * 使用 poll 等待新連線（100ms 逾時），避免忙碌等待。
 * 每個新客戶端由 handleWebSocketClient 在獨立 detached 執行緒中處理。
 */
void WebCommManager::wsAcceptLoop() {
    while (running_ && ws_server_socket_ >= 0) {  // 持續執行直到停止
        struct pollfd pfd = {ws_server_socket_, POLLIN, 0};  // 監控伺服器 socket 的可讀事件
        if (poll(&pfd, 1, 100) <= 0) continue;  // 100ms 逾時，無事件則繼續
        struct sockaddr_in client_addr{};         // 客戶端位址
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(ws_server_socket_, (struct sockaddr*)&client_addr, &client_len);  // 接受連線
        if (client_fd < 0) continue;  // 接受失敗則跳過
        std::thread(&WebCommManager::handleWebSocketClient, this, client_fd).detach();  // 分離執行緒處理客戶端
    }
}

/*
 * handleWebSocketClient — 處理單一 WebSocket 客戶端連線
 * 步驟：
 * 1. 接收 HTTP 升級請求，提取 Sec-WebSocket-Key
 * 2. 計算 Accept Key 並回傳 101 Switching Protocols
 * 3. 加入客戶端集合，開始接收 WebSocket 訊息
 * 4. 斷線後從集合中移除並關閉 socket
 */
void WebCommManager::handleWebSocketClient(int fd) {
    char buffer[4096];  // 接收緩衝區
    int n = recv(fd, buffer, sizeof(buffer) - 1, 0);  // 接收 HTTP 握手請求
    if (n <= 0) { close(fd); return; }  // 接收失敗則關閉
    buffer[n] = '\0';  // 字串結尾

    std::string request(buffer);  // 將請求轉為字串
    std::string ws_key;           // WebSocket 金鑰
    size_t key_pos = request.find("Sec-WebSocket-Key:");  // 尋找金鑰標頭
    if (key_pos != std::string::npos) {
        key_pos += 18;  // 跳過標頭名稱
        while (key_pos < request.size() && request[key_pos] == ' ') key_pos++;  // 跳過空白
        size_t end = request.find("\r\n", key_pos);  // 找到行尾
        if (end != std::string::npos) ws_key = request.substr(key_pos, end - key_pos);  // 提取金鑰值
    }
    if (ws_key.empty()) { close(fd); return; }  // 金鑰無效則關閉

    std::string accept_key = computeWebSocketAcceptKey(ws_key);  // 計算 Accept Key
    std::ostringstream oss;
    // 組裝 HTTP 101 回應標頭
    oss << "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " << accept_key << "\r\n\r\n";
    send(fd, oss.str().c_str(), oss.str().size(), 0);  // 傳送握手回應

    { std::lock_guard<std::mutex> lock(ws_clients_mutex_); ws_clients_.insert(fd); }  // 將客戶端加入集合
    log("WebSocket 客戶端已連線");

    while (running_) {  // 持續接收訊息
        struct pollfd pfd = {fd, POLLIN, 0};  // 監控客戶端 socket
        if (poll(&pfd, 1, 100) <= 0) continue;  // 100ms 逾時
        std::string message;
        if (!recvWebSocketMessage(fd, message)) break;  // 接收失敗（斷線或關閉幀）
        if (!message.empty()) handleWebSocketMessage(message);  // 處理收到的訊息
    }

    { std::lock_guard<std::mutex> lock(ws_clients_mutex_); ws_clients_.erase(fd); }  // 從集合中移除
    close(fd);  // 關閉客戶端 socket
    log("WebSocket 客戶端已斷線");
}

/*
 * handleWebSocketMessage — 解析並處理 WebSocket 收到的 JSON 訊息
 * 支援的訊息類型：
 * - set_target：設定追蹤目標座標
 * - set_moving：設定運動致能
 * - set_active：設定跟隨啟用
 * - switch_mode：切換控制模式
 * - direct_cmd：手動速度控制指令
 * - action_cmd：動作指令（差速機器人中已忽略）
 */
void WebCommManager::handleWebSocketMessage(const std::string& msg) {
    if (JsonParser::hasType(msg, "set_target")) {  // 設定目標
        double x, y; JsonParser::parseXY(msg, x, y);  // 解析 x, y 座標
        state_.setTarget(x, y);  // 寫入共享狀態
    } else if (JsonParser::hasType(msg, "set_moving")) {  // 設定運動致能
        state_.is_moving_enabled.store(JsonParser::getBool(msg, "enabled"));  // 原子寫入
    } else if (JsonParser::hasType(msg, "set_active")) {  // 設定跟隨啟用
        state_.active.store(JsonParser::getBool(msg, "active"));  // 原子寫入
    } else if (JsonParser::hasType(msg, "switch_mode")) {  // 切換控制模式
        size_t pos = msg.find("\"mode\"");  // 尋找 mode 欄位
        if (pos != std::string::npos) {
            pos = msg.find(":", pos);  // 找到冒號
            if (pos != std::string::npos) {
                int mode = std::stoi(msg.substr(pos + 1));  // 解析模式數值
                state_.control_mode.store(mode);  // 原子寫入控制模式
                if (mode == MODE_FOLLOW) state_.is_moving_enabled.store(false);  // 切回跟隨模式時，停用運動致能
            }
        }
    } else if (JsonParser::hasType(msg, "direct_cmd")) {  // 手動速度控制
        double x = JsonParser::extractNumber(msg, "x");  // 解析 x 分量
        double y = JsonParser::extractNumber(msg, "y");  // 解析 y 分量
        double z = JsonParser::extractNumber(msg, "z");  // 解析 z 分量（angular.z）
        if (direct_cmd_callback_) direct_cmd_callback_(x, y, z);  // 呼叫回呼函式
    } else if (JsonParser::hasType(msg, "action_cmd")) {  // 動作指令
        std::string action = JsonParser::extractString(msg, "action");  // 解析動作名稱
        if (action_cmd_callback_ && !action.empty()) action_cmd_callback_(action);  // 呼叫回呼函式
    } else if (JsonParser::hasType(msg, "set_mux_mode")) {  // 底盤模式切換
        size_t pos = msg.find("\"mode\"");
        if (pos != std::string::npos) {
            pos = msg.find(":", pos);
            if (pos != std::string::npos) {
                int mode = std::stoi(msg.substr(pos + 1));
                if (mux_mode_callback_) mux_mode_callback_(mode);
            }
        }
    } else if (JsonParser::hasType(msg, "route_nav")) {  // 路徑導航
        std::string origin = JsonParser::extractString(msg, "origin");
        std::string dest = JsonParser::extractString(msg, "destination");
        if (route_nav_callback_ && !origin.empty() && !dest.empty()) {
            route_nav_callback_(origin, dest);
        }
    } else if (JsonParser::hasType(msg, "goal_nav")) {  // 單點導航
        double x = JsonParser::extractNumber(msg, "x");
        double y = JsonParser::extractNumber(msg, "y");
        if (goal_nav_callback_) goal_nav_callback_(x, y);
    } else if (JsonParser::hasType(msg, "get_route_nodes")) {  // 請求節點列表
        if (get_nodes_callback_) get_nodes_callback_();
    }
}

/*
 * computeWebSocketAcceptKey — 計算 WebSocket 握手的 Accept Key
 * 依據 RFC 6455：將客戶端金鑰與魔術字串串接後，計算 SHA-1 再 Base64 編碼。
 * 此處透過 popen 呼叫 openssl 命令完成（需系統安裝 openssl）。
 */
std::string WebCommManager::computeWebSocketAcceptKey(const std::string& key) {
    std::string magic = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";  // 串接魔術 GUID
    std::string cmd = "echo -n '" + magic + "' | openssl sha1 -binary | base64";  // 組裝 shell 命令
    FILE* pipe = popen(cmd.c_str(), "r");  // 執行命令
    if (!pipe) return "";  // 開啟管線失敗
    char result[128];
    if (fgets(result, sizeof(result), pipe) != nullptr) {  // 讀取結果
        pclose(pipe);  // 關閉管線
        std::string ret(result);
        while (!ret.empty() && (ret.back() == '\n' || ret.back() == '\r')) ret.pop_back();  // 移除尾端換行
        return ret;
    }
    pclose(pipe);  // 關閉管線
    return "";
}

/*
 * recvWebSocketMessage — 接收一筆 WebSocket 訊息
 * 手動解析 WebSocket 幀格式（RFC 6455）：
 * - 2 位元組標頭（opcode + payload 長度）
 * - 擴展長度（若 payload_len == 126 或 127）
 * - 4 位元組遮罩金鑰（若 masked）
 * - 解碼後的酬載資料
 * 回傳 false 表示連線中斷或收到關閉幀。
 */
bool WebCommManager::recvWebSocketMessage(int fd, std::string& message) {
    unsigned char header[2];  // 2 位元組幀標頭
    if (recv(fd, header, 2, 0) != 2) return false;  // 接收失敗
    int opcode = header[0] & 0x0F;  // 提取 opcode（低 4 位元）
    if (opcode == 0x08) return false;  // 0x08 = 關閉幀，中斷連線
    bool masked = header[1] & 0x80;  // 檢查是否有遮罩（客戶端→伺服器必須有）
    uint64_t payload_len = header[1] & 0x7F;  // 提取 7 位元基本長度

    if (payload_len == 126) {  // 擴展 16 位元長度
        unsigned char ext[2];
        if (recv(fd, ext, 2, 0) != 2) return false;
        payload_len = (ext[0] << 8) | ext[1];  // 大端序組合
    } else if (payload_len == 127) {  // 擴展 64 位元長度
        unsigned char ext[8];
        if (recv(fd, ext, 8, 0) != 8) return false;
        payload_len = 0;
        for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | ext[i];  // 大端序組合
    }

    unsigned char mask[4] = {0};  // 4 位元組遮罩金鑰
    if (masked) { if (recv(fd, mask, 4, 0) != 4) return false; }  // 接收遮罩

    if (payload_len > 0 && payload_len < 65536) {  // 合理長度範圍內才處理
        std::vector<char> data(payload_len);  // 配置接收緩衝區
        size_t received = 0;
        while (received < payload_len) {  // 迴圈接收完整酬載
            int n = recv(fd, data.data() + received, payload_len - received, 0);
            if (n <= 0) return false;  // 接收失敗
            received += n;
        }
        if (masked) for (size_t i = 0; i < payload_len; i++) data[i] ^= mask[i % 4];  // XOR 解碼遮罩
        message.assign(data.begin(), data.end());  // 將酬載寫入輸出參數
    }
    return true;
}

/*
 * sendWebSocketMessage — 傳送一筆 WebSocket 訊息給指定客戶端
 * 組裝 WebSocket 幀（伺服器→客戶端不加遮罩）：
 * - 0x81 = FIN + Text 幀
 * - 長度欄位（7 位元 / 16 位元 / 64 位元）
 * - 酬載資料
 */
bool WebCommManager::sendWebSocketMessage(int fd, const std::string& message) {
    std::vector<unsigned char> frame;
    frame.push_back(0x81);  // FIN=1, opcode=0x01（Text 幀）
    size_t len = message.size();  // 酬載長度
    if (len < 126) frame.push_back(static_cast<unsigned char>(len));  // 7 位元長度
    else if (len < 65536) { frame.push_back(126); frame.push_back((len >> 8) & 0xFF); frame.push_back(len & 0xFF); }  // 16 位元長度
    else { frame.push_back(127); for (int i = 7; i >= 0; i--) frame.push_back((len >> (8 * i)) & 0xFF); }  // 64 位元長度
    frame.insert(frame.end(), message.begin(), message.end());  // 附加酬載資料
    return send(fd, frame.data(), frame.size(), MSG_NOSIGNAL) == (ssize_t)frame.size();  // 傳送，MSG_NOSIGNAL 避免 SIGPIPE
}

/*
 * broadcastToWebSockets — 將訊息廣播給所有已連線的 WebSocket 客戶端
 * 傳送失敗的客戶端會被標記為斷線並自動清除。
 */
void WebCommManager::broadcastToWebSockets(const std::string& message) {
    std::lock_guard<std::mutex> lock(ws_clients_mutex_);  // 鎖定客戶端集合
    std::vector<int> dead;  // 記錄已斷線的客戶端
    for (int fd : ws_clients_) if (!sendWebSocketMessage(fd, message)) dead.push_back(fd);  // 傳送失敗則標記
    for (int fd : dead) { ws_clients_.erase(fd); close(fd); }  // 移除並關閉已斷線客戶端
}
