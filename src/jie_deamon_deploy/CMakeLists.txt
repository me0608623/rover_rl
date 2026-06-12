# ==============================================================================
# CMakeLists.txt — jie_deamon 套件建置設定
#
# 檔案說明:
#   定義 jie_deamon 套件的編譯規則、依賴項與安裝目標。
#   包含兩個可執行檔:
#     - robot_nexus: 主要後端服務節點（跟隨、Web 可視化、Android 通訊）
#     - keyboard_cmd: 鍵盤遙控節點（發布 geometry_msgs/Twist）
#
# Humble 相容性:
#   - ament_cmake 語法在 Humble 和 Jazzy 中完全相同
#   - find_package / ament_target_dependencies / install 指令無需修改
# ==============================================================================

# 要求 CMake 最低版本 3.8（ROS 2 Humble / Jazzy 皆滿足）
cmake_minimum_required(VERSION 3.8)

# 定義專案名稱（須與 package.xml 中的 <name> 一致）
project(jie_deamon)

# 設定 C++17 標準（若尚未設定）
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 啟用嚴格編譯警告（GCC / Clang）
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# === 查找依賴套件 ===
find_package(ament_cmake REQUIRED)     # ament 建置工具（ROS 2 CMake 基礎）
find_package(rclcpp REQUIRED)          # ROS 2 C++ 客戶端函式庫
find_package(sensor_msgs REQUIRED)     # 感測器訊息型別（LaserScan 等）
find_package(geometry_msgs REQUIRED)   # 幾何訊息型別（Twist、Pose 等）
find_package(std_msgs REQUIRED)        # 標準訊息型別（String、Bool 等）
find_package(OpenCV REQUIRED)          # OpenCV 電腦視覺函式庫
find_package(Threads REQUIRED)         # POSIX 執行緒（pthread）

# 包含標頭檔目錄（include/ 下的 .h / .hpp 檔案）
include_directories(include)

# === 編譯 robot_nexus 節點（模組化版本） ===
# 由三個原始檔組成:
#   - robot_nexus.cpp: 主節點邏輯（跟隨、感測器處理）
#   - web_comm.cpp:    Web 通訊模組（WebSocket / HTTP 伺服器）
#   - android_comm.cpp: Android 通訊模組
add_executable(robot_nexus
  src/robot_nexus.cpp
  src/web_comm.cpp
  src/android_comm.cpp
)
# 連結 ROS 2 訊息依賴（自動處理 include / link）
ament_target_dependencies(robot_nexus
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
)
# 連結 OpenCV 和執行緒函式庫
target_link_libraries(robot_nexus
  ${OpenCV_LIBS}
  Threads::Threads
)
# 加入 OpenCV 標頭檔搜尋路徑
target_include_directories(robot_nexus PUBLIC ${OpenCV_INCLUDE_DIRS})

# === 編譯 keyboard_cmd 節點 ===
# 鍵盤遙控節點，讀取鍵盤輸入並發布 Twist 指令
add_executable(keyboard_cmd src/keyboard_cmd.cpp)
ament_target_dependencies(keyboard_cmd
"rclcpp" "geometry_msgs"
)

# === 安裝可執行檔到 lib/<套件名> 目錄 ===
# ros2 run jie_deamon robot_nexus 會從此目錄尋找可執行檔
install(TARGETS
  robot_nexus
  keyboard_cmd
  DESTINATION lib/${PROJECT_NAME}
)

# === 安裝 launch 檔案到 share/<套件名>/ 目錄 ===
# ros2 launch jie_deamon start.launch.py 會從此目錄尋找 launch 檔
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# === 安裝 Web 靜態檔案到 share/<套件名>/ 目錄 ===
# robot_nexus 節點的 web_root 參數指向此目錄
install(DIRECTORY
  web
  DESTINATION share/${PROJECT_NAME}/
)

# === 測試設定 ===
# 啟用 BUILD_TESTING 時，使用 ament_lint_auto 自動尋找並執行 linter 測試
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# 產生 ament 套件索引（必須放在 CMakeLists.txt 最後）
ament_package()
