/*
 * ============================================================================
 * 檔案：web_server.hpp
 * 說明：簡易 HTTP + WebSocket 伺服器實作 — 用於光達可視化 Web 界面
 *
 * 功能：
 *   - HTTP 靜態檔案伺服器：提供 HTML/CSS/JS/圖片等前端資源
 *   - WebSocket 伺服器：與瀏覽器建立即時雙向通訊
 *   - WebSocket 訊息廣播：將光達資料即時推送到所有客戶端
 *   - Base64 編碼 + SHA-1 雜湊：WebSocket 握手所需
 *
 * 依賴：
 *   - OpenSSL (openssl/sha.h)：僅用於 SHA-1 計算 WebSocket Accept Key
 *   - POSIX socket API：TCP 伺服器
 *   - 標準 C++ 函式庫
 *
 * [Humble 相容性]
 *   此模組為純 C++ 實作，不依賴任何 ROS API，
 *   Humble / Jazzy / Rolling 均可直接使用。
 *   OpenSSL 為系統函式庫依賴，需在 CMakeLists.txt 與 package.xml 中宣告：
 *     find_package(OpenSSL REQUIRED)
 *     target_link_libraries(... OpenSSL::SSL OpenSSL::Crypto)
 *   若切換 ROS 版本，需確認 OpenSSL 版本相容（Humble=3.0+, Jazzy=3.0+）。
 * ============================================================================
 */

#ifndef WEB_SERVER_HPP
#define WEB_SERVER_HPP

#include <string>       // std::string — 字串處理
#include <vector>       // std::vector — 動態陣列（客戶端列表、WebSocket 幀）
#include <map>          // std::map — 字典（保留，未使用但可能供路由擴充）
#include <functional>   // std::function — WebSocket 訊息處理回呼
#include <thread>       // std::thread — HTTP/WebSocket 伺服器執行緒
#include <mutex>        // std::mutex — 客戶端列表互斥鎖
#include <atomic>       // std::atomic — 伺服器運行狀態旗標
#include <memory>       // std::shared_ptr — 客戶端連線共享所有權
#include <sstream>      // std::ostringstream — HTTP 回應組裝
#include <fstream>      // std::ifstream — 讀取靜態檔案
#include <cstring>      // std::memset 等 C 字串工具
#include <algorithm>    // std::remove — 清理斷線客戶端

#include <sys/socket.h> // socket(), bind(), listen(), accept(), send(), recv()
#include <netinet/in.h> // sockaddr_in, htons(), INADDR_ANY
#include <arpa/inet.h>  // inet_ntop() — IP 位址轉字串
#include <unistd.h>     // close() — 關閉 socket
#include <fcntl.h>      // fcntl() — 設定非阻塞模式（保留）
#include <poll.h>       // poll() — 非阻塞 I/O 多工

#include <openssl/sha.h>  // SHA1() — WebSocket 握手所需的 SHA-1 雜湊

namespace webserver {

// Base64 編碼字元表
static const char* BASE64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/*
 * Base64 編碼函式
 * 將二進位資料編碼為 ASCII 字串，用於 WebSocket 握手回應
 *
 * @param data 輸入的二進位資料
 * @param len  資料長度
 * @return Base64 編碼後的字串
 */
inline std::string base64_encode(const unsigned char* data, size_t len) {
    std::string result;
    int i = 0;
    unsigned char arr3[3], arr4[4];  // 每 3 位元組轉換為 4 個 Base64 字元

    while (len--) {
        arr3[i++] = *(data++);
        if (i == 3) {
            arr4[0] = (arr3[0] & 0xfc) >> 2;                                      // 取前 6 位
            arr4[1] = ((arr3[0] & 0x03) << 4) + ((arr3[1] & 0xf0) >> 4);          // 跨位元組取 6 位
            arr4[2] = ((arr3[1] & 0x0f) << 2) + ((arr3[2] & 0xc0) >> 6);          // 跨位元組取 6 位
            arr4[3] = arr3[2] & 0x3f;                                               // 取後 6 位
            for (i = 0; i < 4; i++) result += BASE64_CHARS[arr4[i]];               // 查表輸出
            i = 0;
        }
    }

    // 處理剩餘的 1 或 2 位元組（補 '=' 填充）
    if (i) {
        for (int j = i; j < 3; j++) arr3[j] = '\0';                               // 不足 3 位元組補零
        arr4[0] = (arr3[0] & 0xfc) >> 2;
        arr4[1] = ((arr3[0] & 0x03) << 4) + ((arr3[1] & 0xf0) >> 4);
        arr4[2] = ((arr3[1] & 0x0f) << 2) + ((arr3[2] & 0xc0) >> 6);
        for (int j = 0; j < i + 1; j++) result += BASE64_CHARS[arr4[j]];
        while (i++ < 3) result += '=';                                              // 填充 '='
    }
    return result;
}

/*
 * 計算 WebSocket Accept Key
 * 根據 RFC 6455 規範：Accept = Base64(SHA1(key + GUID))
 * 其中 GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"（固定魔法字串）
 *
 * @param key 客戶端發送的 Sec-WebSocket-Key
 * @return 伺服器回應的 Sec-WebSocket-Accept 值
 */
inline std::string compute_accept_key(const std::string& key) {
    std::string magic = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";       // 拼接 GUID
    unsigned char hash[SHA_DIGEST_LENGTH];                                    // SHA-1 輸出 (20 位元組)
    SHA1(reinterpret_cast<const unsigned char*>(magic.c_str()), magic.size(), hash);  // 計算 SHA-1
    return base64_encode(hash, SHA_DIGEST_LENGTH);                            // Base64 編碼後回傳
}

/*
 * WebSocket 客戶端連線結構
 * 記錄每個已連接客戶端的 socket fd、IP 與連線狀態
 */
struct WSClient {
    int fd;            // socket 檔案描述符
    std::string ip;    // 客戶端 IP 位址
    bool connected;    // 連線是否仍存活

    WSClient(int f, const std::string& i) : fd(f), ip(i), connected(true) {}  // 建構時預設已連線
};

/*
 * 簡易 HTTP + WebSocket 伺服器
 * 在兩個獨立執行緒中分別運行 HTTP 與 WebSocket 服務
 *
 * [Humble 相容] 純 POSIX socket + std::thread，不依賴 ROS
 */
class SimpleWebServer {
public:
    using MessageHandler = std::function<void(const std::string&, std::shared_ptr<WSClient>)>;  // WebSocket 訊息處理回呼

    SimpleWebServer(int http_port = 8080, int ws_port = 8890)
        : http_port_(http_port), ws_port_(ws_port), running_(false) {}  // 建構函式：設定預設埠號

    ~SimpleWebServer() { stop(); }  // 解構函式：確保伺服器停止

    /* 設定靜態檔案根目錄（HTTP 伺服器從此目錄提供檔案） */
    void set_static_dir(const std::string& dir) { static_dir_ = dir; }

    /* 設定 WebSocket 訊息處理回呼 */
    void set_message_handler(MessageHandler handler) { message_handler_ = handler; }

    /*
     * 啟動伺服器
     * 建立 HTTP 與 WebSocket 的 TCP socket，綁定埠號並開始監聽
     * @return true 表示啟動成功，false 表示綁定失敗
     */
    bool start() {
        if (running_) return true;  // 已在運行則跳過

        // --- 建立 HTTP socket ---
        http_socket_ = socket(AF_INET, SOCK_STREAM, 0);     // IPv4 TCP socket
        if (http_socket_ < 0) return false;                   // 建立失敗

        int opt = 1;
        setsockopt(http_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));  // 允許位址重用（快速重啟）

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;                            // IPv4
        addr.sin_addr.s_addr = INADDR_ANY;                    // 監聽所有介面
        addr.sin_port = htons(http_port_);                    // HTTP 埠號

        if (bind(http_socket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(http_socket_);                               // 綁定失敗：釋放 socket
            return false;
        }

        listen(http_socket_, 10);                              // 開始監聽，佇列長度 10

        // --- 建立 WebSocket socket ---
        ws_socket_ = socket(AF_INET, SOCK_STREAM, 0);        // IPv4 TCP socket
        if (ws_socket_ < 0) {
            close(http_socket_);                               // 建立失敗：釋放 HTTP socket
            return false;
        }

        setsockopt(ws_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));  // 允許位址重用

        addr.sin_port = htons(ws_port_);                      // WebSocket 埠號
        if (bind(ws_socket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(http_socket_);                               // 綁定失敗：釋放所有 socket
            close(ws_socket_);
            return false;
        }

        listen(ws_socket_, 10);                                // 開始監聽

        running_ = true;
        http_thread_ = std::thread(&SimpleWebServer::http_loop, this);  // 啟動 HTTP 伺服迴圈
        ws_thread_ = std::thread(&SimpleWebServer::ws_loop, this);      // 啟動 WebSocket 伺服迴圈

        return true;
    }

    /*
     * 停止伺服器
     * 關閉所有 socket、斷開所有客戶端、等待執行緒結束
     */
    void stop() {
        running_ = false;

        if (http_socket_ >= 0) { close(http_socket_); http_socket_ = -1; }  // 關閉 HTTP 監聽 socket
        if (ws_socket_ >= 0) { close(ws_socket_); ws_socket_ = -1; }        // 關閉 WebSocket 監聽 socket

        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            for (auto& client : ws_clients_) {
                if (client->fd >= 0) close(client->fd);  // 關閉所有客戶端連線
            }
            ws_clients_.clear();                           // 清空客戶端列表
        }

        if (http_thread_.joinable()) http_thread_.join();    // 等待 HTTP 執行緒結束
        if (ws_thread_.joinable()) ws_thread_.join();        // 等待 WebSocket 執行緒結束
    }

    /*
     * 廣播訊息到所有已連接的 WebSocket 客戶端
     * 自動清理已斷線的客戶端
     *
     * @param message 要廣播的 JSON 字串
     */
    void broadcast(const std::string& message) {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        std::vector<std::shared_ptr<WSClient>> dead_clients;  // 待清理的斷線客戶端

        for (auto& client : ws_clients_) {
            if (!client->connected) {
                dead_clients.push_back(client);               // 已斷線：加入清理列表
                continue;
            }
            if (!send_ws_message(client->fd, message)) {
                client->connected = false;                    // 發送失敗：標記為斷線
                dead_clients.push_back(client);
            }
        }

        // 清理已斷線的連線
        for (auto& dead : dead_clients) {
            if (dead->fd >= 0) close(dead->fd);               // 關閉 socket
            ws_clients_.erase(
                std::remove(ws_clients_.begin(), ws_clients_.end(), dead),
                ws_clients_.end()
            );
        }
    }

    /* 取得當前已連接的 WebSocket 客戶端數量 */
    size_t client_count() const {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        return ws_clients_.size();
    }

private:
    int http_port_, ws_port_;                                    // HTTP 與 WebSocket 埠號
    int http_socket_ = -1, ws_socket_ = -1;                     // 監聽 socket fd
    std::atomic<bool> running_;                                  // 伺服器運行狀態
    std::thread http_thread_, ws_thread_;                        // 伺服器執行緒
    std::string static_dir_;                                     // 靜態檔案根目錄
    MessageHandler message_handler_;                             // WebSocket 訊息處理回呼

    mutable std::mutex clients_mutex_;                           // 客戶端列表互斥鎖（mutable 允許 const 方法加鎖）
    std::vector<std::shared_ptr<WSClient>> ws_clients_;          // WebSocket 客戶端列表

    /*
     * HTTP 伺服迴圈
     * 使用 poll() 以 100ms 超時非阻塞等待連線，避免忙碌等待
     */
    void http_loop() {
        while (running_) {
            struct pollfd pfd = {http_socket_, POLLIN, 0};       // 監聽可讀事件
            if (poll(&pfd, 1, 100) <= 0) continue;               // 100ms 超時或無事件則繼續

            struct sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(http_socket_, (struct sockaddr*)&client_addr, &client_len);  // 接受連線
            if (client_fd < 0) continue;                          // 接受失敗則跳過

            // 設定接收超時（防止慢客戶端阻塞伺服器）
            struct timeval tv = {5, 0};                           // 5 秒超時
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

            handle_http_request(client_fd);                       // 處理 HTTP 請求
            close(client_fd);                                     // 關閉連線（HTTP/1.0 風格）
        }
    }

    /*
     * 處理單一 HTTP 請求
     * 解析 GET 請求路徑，讀取對應靜態檔案並回傳
     */
    void handle_http_request(int fd) {
        char buffer[4096];
        int n = recv(fd, buffer, sizeof(buffer) - 1, 0);         // 接收請求
        if (n <= 0) return;
        buffer[n] = '\0';

        std::string request(buffer);
        std::string path = "/";

        // 解析請求路徑（僅支援 GET 方法）
        size_t start = request.find("GET ");
        if (start != std::string::npos) {
            start += 4;                                           // 跳過 "GET "
            size_t end = request.find(" ", start);                // 找到路徑結尾
            if (end != std::string::npos) {
                path = request.substr(start, end - start);
            }
        }

        if (path == "/") path = "/index.html";                   // 預設首頁

        // 安全檢查：防止目錄遍歷攻擊
        if (path.find("..") != std::string::npos) {
            send_http_response(fd, 403, "text/plain", "Forbidden");
            return;
        }

        // 讀取靜態檔案
        std::string filepath = static_dir_ + path;
        std::ifstream file(filepath, std::ios::binary);           // 二進位模式開啟
        if (!file) {
            send_http_response(fd, 404, "text/plain", "Not Found");
            return;
        }

        std::stringstream ss;
        ss << file.rdbuf();                                       // 讀取整個檔案
        std::string content = ss.str();

        // 根據副檔名判斷 MIME 類型
        std::string mime = "text/plain";
        if (path.find(".html") != std::string::npos) mime = "text/html; charset=utf-8";
        else if (path.find(".css") != std::string::npos) mime = "text/css; charset=utf-8";
        else if (path.find(".js") != std::string::npos) mime = "application/javascript; charset=utf-8";
        else if (path.find(".json") != std::string::npos) mime = "application/json";
        else if (path.find(".png") != std::string::npos) mime = "image/png";
        else if (path.find(".jpg") != std::string::npos) mime = "image/jpeg";
        else if (path.find(".svg") != std::string::npos) mime = "image/svg+xml";

        send_http_response(fd, 200, mime, content);               // 回傳檔案內容
    }

    /*
     * 發送 HTTP 回應
     * 組裝 HTTP/1.1 回應標頭與本體
     */
    void send_http_response(int fd, int code, const std::string& mime, const std::string& body) {
        std::string status = (code == 200) ? "OK" : (code == 404) ? "Not Found" : "Error";  // 狀態文字
        std::ostringstream oss;
        oss << "HTTP/1.1 " << code << " " << status << "\r\n"
            << "Content-Type: " << mime << "\r\n"                   // MIME 類型
            << "Content-Length: " << body.size() << "\r\n"          // 內容長度
            << "Access-Control-Allow-Origin: *\r\n"                 // CORS 允許所有來源
            << "Connection: close\r\n\r\n"                          // 回應後關閉連線
            << body;

        std::string response = oss.str();
        send(fd, response.c_str(), response.size(), 0);             // 發送回應
    }

    /*
     * WebSocket 伺服迴圈
     * 使用 poll() 等待新的 WebSocket 連線，每個客戶端開啟獨立執行緒處理
     */
    void ws_loop() {
        while (running_) {
            struct pollfd pfd = {ws_socket_, POLLIN, 0};           // 監聽可讀事件
            if (poll(&pfd, 1, 100) <= 0) continue;                 // 100ms 超時

            struct sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(ws_socket_, (struct sockaddr*)&client_addr, &client_len);  // 接受連線
            if (client_fd < 0) continue;

            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, ip, INET_ADDRSTRLEN);  // 取得客戶端 IP

            // 為每個客戶端啟動獨立的處理執行緒（detach：不等待結束）
            std::thread(&SimpleWebServer::handle_ws_client, this, client_fd, std::string(ip)).detach();
        }
    }

    /*
     * 處理單一 WebSocket 客戶端連線
     * 完成握手後進入訊息接收迴圈，直到連線斷開
     */
    void handle_ws_client(int fd, std::string ip) {
        // --- WebSocket 握手 ---
        char buffer[4096];
        int n = recv(fd, buffer, sizeof(buffer) - 1, 0);          // 接收 HTTP 升級請求
        if (n <= 0) { close(fd); return; }
        buffer[n] = '\0';

        std::string request(buffer);

        // 提取 Sec-WebSocket-Key 標頭
        std::string ws_key;
        size_t key_pos = request.find("Sec-WebSocket-Key:");
        if (key_pos != std::string::npos) {
            key_pos += 18;                                         // 跳過標頭名稱
            while (key_pos < request.size() && request[key_pos] == ' ') key_pos++;  // 跳過空白
            size_t end = request.find("\r\n", key_pos);
            if (end != std::string::npos) {
                ws_key = request.substr(key_pos, end - key_pos);   // 提取 Key 值
            }
        }

        if (ws_key.empty()) { close(fd); return; }                // 無 Key 表示非 WebSocket 請求

        // 組裝並發送握手回應（101 Switching Protocols）
        std::string accept_key = compute_accept_key(ws_key);       // 計算 Accept Key
        std::ostringstream oss;
        oss << "HTTP/1.1 101 Switching Protocols\r\n"
            << "Upgrade: websocket\r\n"                             // 升級為 WebSocket 協議
            << "Connection: Upgrade\r\n"
            << "Sec-WebSocket-Accept: " << accept_key << "\r\n\r\n";

        std::string response = oss.str();
        send(fd, response.c_str(), response.size(), 0);            // 發送握手回應

        // 將客戶端加入列表
        auto client = std::make_shared<WSClient>(fd, ip);
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            ws_clients_.push_back(client);                          // 記錄新客戶端
        }

        // --- 訊息接收迴圈 ---
        while (running_ && client->connected) {
            struct pollfd pfd = {fd, POLLIN, 0};                   // 監聽可讀事件
            if (poll(&pfd, 1, 100) <= 0) continue;                 // 100ms 超時

            std::string message;
            if (!recv_ws_message(fd, message)) {
                client->connected = false;                          // 接收失敗：標記斷線
                break;
            }

            if (!message.empty() && message_handler_) {
                message_handler_(message, client);                  // 呼叫訊息處理回呼
            }
        }

        // --- 清理 ---
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            ws_clients_.erase(
                std::remove(ws_clients_.begin(), ws_clients_.end(), client),
                ws_clients_.end()
            );
        }
        close(fd);                                                  // 關閉客戶端 socket
    }

    /*
     * 接收 WebSocket 訊息
     * 根據 RFC 6455 解析 WebSocket 幀格式
     *
     * 幀格式：[FIN+opcode][MASK+長度][擴充長度][掩碼金鑰][資料]
     *
     * @param fd       客戶端 socket fd
     * @param message  輸出：解碼後的訊息字串
     * @return true 表示成功接收，false 表示連線關閉或錯誤
     */
    bool recv_ws_message(int fd, std::string& message) {
        unsigned char header[2];
        if (recv(fd, header, 2, 0) != 2) return false;            // 讀取 2 位元組幀頭

        bool fin = header[0] & 0x80;                               // FIN 位元：是否為最後一幀
        int opcode = header[0] & 0x0F;                             // 操作碼：0x1=文字, 0x8=關閉
        bool masked = header[1] & 0x80;                            // MASK 位元：客戶端到伺服器必須 mask
        uint64_t payload_len = header[1] & 0x7F;                   // 基本長度（7 位元）

        if (opcode == 0x08) return false;                          // 關閉幀：回傳 false 結束迴圈

        // 擴充長度處理
        if (payload_len == 126) {
            unsigned char ext[2];                                   // 16 位元擴充長度
            if (recv(fd, ext, 2, 0) != 2) return false;
            payload_len = (ext[0] << 8) | ext[1];
        } else if (payload_len == 127) {
            unsigned char ext[8];                                   // 64 位元擴充長度
            if (recv(fd, ext, 8, 0) != 8) return false;
            payload_len = 0;
            for (int i = 0; i < 8; i++) {
                payload_len = (payload_len << 8) | ext[i];         // 大端序組裝
            }
        }

        // 讀取掩碼金鑰（4 位元組）
        unsigned char mask[4] = {0};
        if (masked) {
            if (recv(fd, mask, 4, 0) != 4) return false;
        }

        // 讀取資料承載（限制 64KB 防止記憶體爆掉）
        if (payload_len > 0 && payload_len < 65536) {
            std::vector<char> data(payload_len);
            size_t received = 0;
            while (received < payload_len) {
                int n = recv(fd, data.data() + received, payload_len - received, 0);  // 迴圈讀取直到完整
                if (n <= 0) return false;
                received += n;
            }

            // XOR 解碼（客戶端發送的資料必須經過掩碼）
            if (masked) {
                for (size_t i = 0; i < payload_len; i++) {
                    data[i] ^= mask[i % 4];                        // 循環 XOR 解碼
                }
            }

            message.assign(data.begin(), data.end());               // 組裝為字串
        }

        (void)fin;  // 暫不處理分片訊息（假設所有訊息都在單一幀內）
        return true;
    }

    /*
     * 發送 WebSocket 訊息
     * 組裝 WebSocket 幀並透過 socket 發送
     *
     * 伺服器到客戶端的幀不需要掩碼（RFC 6455 規範）
     *
     * @param fd       客戶端 socket fd
     * @param message  要發送的訊息字串
     * @return true 表示發送成功
     */
    bool send_ws_message(int fd, const std::string& message) {
        std::vector<unsigned char> frame;

        frame.push_back(0x81);                                      // FIN=1 + opcode=0x1（文字幀）

        // 組裝長度欄位
        size_t len = message.size();
        if (len < 126) {
            frame.push_back(static_cast<unsigned char>(len));       // 7 位元長度
        } else if (len < 65536) {
            frame.push_back(126);                                    // 16 位元擴充長度標記
            frame.push_back((len >> 8) & 0xFF);                     // 高位元組
            frame.push_back(len & 0xFF);                            // 低位元組
        } else {
            frame.push_back(127);                                    // 64 位元擴充長度標記
            for (int i = 7; i >= 0; i--) {
                frame.push_back((len >> (8 * i)) & 0xFF);          // 大端序寫入 8 位元組
            }
        }

        // 附加訊息資料
        frame.insert(frame.end(), message.begin(), message.end());

        // 發送完整幀（MSG_NOSIGNAL 防止 SIGPIPE 導致程式終止）
        return send(fd, frame.data(), frame.size(), MSG_NOSIGNAL) == (ssize_t)frame.size();
    }
};

} // namespace webserver

#endif // WEB_SERVER_HPP
