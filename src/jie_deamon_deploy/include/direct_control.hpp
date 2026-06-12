/*
 * ============================================================================
 * 檔案：direct_control.hpp
 * 說明：直接控制模組 — 處理來自遙控器或手機的即時速度指令
 *
 * 功能：
 *   - 接收外部直接速度指令 (vx, vy, wz)
 *   - 透過定時器回呼以固定頻率發布速度
 *   - 自動切換至直接控制模式
 *   - 適配差速機器人（linear.y 恆為 0）
 *
 * [Humble 相容性]
 *   - geometry_msgs::msg::Twist 在 Humble/Jazzy 間 API 相同
 *   - 本類別不繼承 rclcpp::Node，透過回呼與外部節點互動
 *   - 定時器由外部 ROS 節點建立，本類別僅提供 timerCallback()
 *   - 注意：Humble 的 create_wall_timer 回傳型別與 Jazzy 略有差異，
 *     但對回呼函式本身無影響
 * ============================================================================
 */

#ifndef DIRECT_CONTROL_HPP
#define DIRECT_CONTROL_HPP

#include "common_types.hpp"                // 共用常數與共享狀態
#include <geometry_msgs/msg/twist.hpp>     // 速度指令訊息型別
#include <functional>                      // std::function 回呼封裝

/*
 * 直接控制處理器
 * 將外部遙控指令轉發為 ROS Twist 訊息，支援模式自動切換
 *
 * [Humble 相容] 純 C++ 類別，不直接使用 rclcpp API，
 *   Humble/Jazzy 均適用
 */
class DirectController {
public:
    using VelocityCallback = std::function<void(const geometry_msgs::msg::Twist&)>;  // 速度發布回呼型別

    DirectController(SharedState& state) : state_(state) {}  // 建構函式：注入共享狀態

    /* 設定速度發布回呼 */
    void setVelocityCallback(VelocityCallback cb) {
        velocity_callback_ = std::move(cb);
    }

    /*
     * 定時器回呼 — 以固定頻率發布直接控制速度
     * 由外部 ROS 節點的 wall_timer 觸發
     * 僅在直接控制模式 (MODE_DIRECT) 下有效
     */
    void timerCallback() {
        // 非直接控制模式時跳過
        if (state_.control_mode.load() != MODE_DIRECT) {
            return;
        }

        geometry_msgs::msg::Twist cmd_vel_msg;
        double vx, vy, wz;
        state_.getDirectCmd(vx, vy, wz);       // 從共享狀態讀取遙控指令
        cmd_vel_msg.linear.x = vx;              // 前後線速度
        cmd_vel_msg.linear.y = 0.0;             // 差速機器人不支援橫移，強制為 0
        cmd_vel_msg.angular.z = wz;             // 角速度

        // 快取速度（供 Web 廣播等模組讀取）
        state_.setVelocity(vx, 0.0, wz);

        // 透過回呼發布速度到 ROS topic
        if (velocity_callback_) {
            velocity_callback_(cmd_vel_msg);
        }
    }

    /*
     * 處理直接控制指令
     * 由通訊模組（UDP/WebSocket）收到遙控指令後呼叫
     * 會自動切換到直接控制模式
     *
     * @param vx 前後線速度 (m/s)
     * @param vy 橫向線速度 (m/s)（差速機器人忽略此值）
     * @param wz 角速度 (rad/s)
     */
    void processDirectCmd(double vx, double vy, double wz) {
        state_.setDirectCmd(vx, vy, wz);  // 寫入共享狀態

        // 自動切換到直接控制模式（若尚未切換）
        if (state_.control_mode.load() != MODE_DIRECT) {
            state_.control_mode.store(MODE_DIRECT);
        }
    }

private:
    SharedState& state_;                   // 共享狀態參考
    VelocityCallback velocity_callback_;   // 速度發布回呼
};

#endif // DIRECT_CONTROL_HPP
