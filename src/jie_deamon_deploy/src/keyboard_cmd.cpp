/*
 * 檔案說明：keyboard_cmd.cpp — 鍵盤遙控節點（差速驅動）
 *
 * 功能概述：
 * - 透過終端機鍵盤輸入控制差速驅動機器人
 * - 發布 geometry_msgs/Twist 訊息到 /cmd_vel 話題
 * - 按鍵對應：
 *   W/S = 前進/後退（調整 linear.x）
 *   A/D = 左轉/右轉（調整 angular.z）
 *   Q/E = 微調左轉/右轉（調整 angular.z）
 *   空白鍵 = 煞車（歸零）
 *   X = 停車並退出
 * - 速度有上下限保護（k_vel 倍率限制）
 *
 * Humble 相容性提示：
 * - rclcpp::Node::make_shared 在 Humble 中語法相同
 * - create_publisher<T>(topic, qos) 在 Humble 中語法相同
 * - QoS depth=10 在此明確指定，Humble / Jazzy 行為一致
 * - 此節點不使用任何 Humble 特有 API，可直接跨版本編譯
 */

#include <rclcpp/rclcpp.hpp>                // ROS 2 C++ 核心函式庫
#include <geometry_msgs/msg/twist.hpp>      // 速度指令訊息型別
#include <stdio.h>                           // printf 標準 I/O
#include <termios.h>                         // 終端機屬性控制（非標準輸入模式）

static float linear_vel = 0.1;   // 線速度增量（每次按鍵增加的量，單位: m/s）
static float angular_vel = 0.1;  // 角速度增量（每次按鍵增加的量，單位: rad/s）
static int k_vel = 3;            // 速度倍率上限（最大速度 = vel * k_vel）

/*
 * GetCh — 從終端機讀取單一字元（無需按 Enter）
 * 暫時關閉終端機的 ICANON 模式（行緩衝），讀取一個字元後恢復原設定。
 * 注意：此函式使用 static 區域變數，非執行緒安全。
 */
int GetCh()
{
  static struct termios oldt, newt;            // 舊/新終端機設定
  tcgetattr(STDIN_FILENO, &oldt);              // 取得目前終端機設定
  newt = oldt;                                  // 複製設定
  newt.c_lflag &= ~(ICANON);                  // 關閉行緩衝（逐字元讀取）
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);    // 立即套用新設定
  int c = getchar();                            // 讀取單一字元
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);    // 恢復原始設定
  return c;
}

/*
 * main — 程式進入點
 * 初始化 ROS 2、建立節點與發布者，進入鍵盤控制迴圈。
 */
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);  // 初始化 ROS 2 客戶端函式庫

  // 顯示按鍵說明
  printf("差速驅動機器人鍵盤控制: \n");
  printf("w - 加速前進 \n");
  printf("s - 加速後退 \n");
  printf("a - 左轉 \n");
  printf("d - 右轉 \n");
  printf("q - 微調左轉 \n");
  printf("e - 微調右轉 \n");
  printf("空白鍵 - 煞車 \n");
  printf("x - 退出 \n");
  printf("------------- \n");

  auto node = rclcpp::Node::make_shared("keyboard_cmd");  // 建立節點（名稱: keyboard_cmd）

  auto cmd_vel_pub = node->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);  // 建立 /cmd_vel 發布者，QoS depth=10

  auto base_cmd = std::make_shared<geometry_msgs::msg::Twist>();  // 建立 Twist 訊息物件
  base_cmd->linear.x = 0;   // 初始線速度 = 0
  base_cmd->angular.z = 0;  // 初始角速度 = 0

  while (rclcpp::ok())  // 持續執行直到 ROS 2 關閉
  {
    int cKey = GetCh();  // 讀取鍵盤輸入
    if (cKey == 'w')  // W: 加速前進
    {
      base_cmd->linear.x += linear_vel;  // 增加線速度
      if (base_cmd->linear.x > linear_vel * k_vel)  // 上限保護
        base_cmd->linear.x = linear_vel * k_vel;
      cmd_vel_pub->publish(*base_cmd);  // 發布速度指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
    }
    else if (cKey == 's')  // S: 加速後退
    {
      base_cmd->linear.x += -linear_vel;  // 減少線速度（後退）

      if (base_cmd->linear.x < -linear_vel * k_vel)  // 下限保護
        base_cmd->linear.x = -linear_vel * k_vel;
      cmd_vel_pub->publish(*base_cmd);  // 發布速度指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
    }
    else if (cKey == 'a')  // A: 左轉（增加角速度，正值為逆時針）
    {
      base_cmd->angular.z += angular_vel;  // 增加角速度
      if (base_cmd->angular.z > angular_vel * k_vel)  // 上限保護
        base_cmd->angular.z = angular_vel * k_vel;
      cmd_vel_pub->publish(*base_cmd);  // 發布速度指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
    }
    else if (cKey == 'd')  // D: 右轉（減少角速度，負值為順時針）
    {
      base_cmd->angular.z += -angular_vel;  // 減少角速度
      if (base_cmd->angular.z < -angular_vel * k_vel)  // 下限保護
        base_cmd->angular.z = -angular_vel * k_vel;
      cmd_vel_pub->publish(*base_cmd);  // 發布速度指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
    }
    else if (cKey == 'q')  // Q: 微調左轉（與 A 相同邏輯）
    {
      base_cmd->angular.z += angular_vel;  // 增加角速度
      if (base_cmd->angular.z > angular_vel * k_vel)  // 上限保護
        base_cmd->angular.z = angular_vel * k_vel;
      cmd_vel_pub->publish(*base_cmd);  // 發布速度指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
    }
    else if (cKey == 'e')  // E: 微調右轉（與 D 相同邏輯）
    {
      base_cmd->angular.z += -angular_vel;  // 減少角速度
      if (base_cmd->angular.z < -angular_vel * k_vel)  // 下限保護
        base_cmd->angular.z = -angular_vel * k_vel;
      cmd_vel_pub->publish(*base_cmd);  // 發布速度指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
    }
    else if (cKey == ' ')  // 空白鍵: 煞車（速度歸零）
    {
      base_cmd->linear.x = 0;   // 線速度歸零
      base_cmd->angular.z = 0;  // 角速度歸零
      cmd_vel_pub->publish(*base_cmd);  // 發布停止指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
    }
    else if (cKey == 'x')  // X: 停車並退出程式
    {
      base_cmd->linear.x = 0;   // 線速度歸零
      base_cmd->angular.z = 0;  // 角速度歸零
      cmd_vel_pub->publish(*base_cmd);  // 發布停止指令
      printf(" - linear.x= %.2f angular.z= %.2f \n", base_cmd->linear.x, base_cmd->angular.z);
      printf("已退出！ \n");
      return 0;  // 結束程式
    }
    else  // 未定義的按鍵
    {
      printf(" - 未定義的指令 \n");
    }
  }

  rclcpp::shutdown();  // 關閉 ROS 2
  return 0;
}
