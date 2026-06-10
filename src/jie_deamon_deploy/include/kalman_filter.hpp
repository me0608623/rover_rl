/*
 * ============================================================================
 * 檔案：kalman_filter.hpp
 * 說明：二維卡爾曼濾波器 — 用於平滑目標追蹤位置，降低光達量測雜訊造成的抖動
 *
 * 採用等速運動模型 (Constant Velocity)，狀態向量為 [x, y, vx, vy]。
 * 不依賴定時器，透過光達資料的到達間隔自動推算 dt。
 *
 * 數學模型：
 *   狀態轉移：x' = x + vx·dt,  y' = y + vy·dt,  vx' = vx,  vy' = vy
 *   觀測模型：z = [x, y]（僅量測位置，不量測速度）
 *
 * [Humble 相容性]
 *   此模組為純 C++ 數學計算，不依賴任何 ROS API，
 *   Humble / Jazzy / Rolling 均可直接使用，無需修改。
 *   使用 std::chrono::steady_clock 計算時間差，不受 ROS 時鐘影響。
 * ============================================================================
 */

#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <array>    // std::array — 固定大小陣列
#include <cmath>    // std::abs — 數學函式
#include <chrono>   // std::chrono — 高精度時間計算

/*
 * 二維等速運動模型卡爾曼濾波器
 *
 * 狀態向量: [x, y, vx, vy]  (位置 + 速度)
 * 觀測向量: [x, y]          (僅位置)
 *
 * 狀態轉移矩陣 F:
 *   | 1  0  dt  0 |
 *   | 0  1  0  dt |
 *   | 0  0  1   0 |
 *   | 0  0  0   1 |
 *
 * 觀測矩陣 H:
 *   | 1  0  0  0 |
 *   | 0  1  0  0 |
 */
class KalmanFilter2D {
public:
    /*
     * 建構函式
     * @param process_noise    過程雜訊係數 q，越大 → 越信任觀測值（追蹤更靈敏但更抖）
     * @param measurement_noise 觀測雜訊係數 r，越大 → 越信任預測值（濾波更平滑但延遲更大）
     */
    KalmanFilter2D(double process_noise = 0.1, double measurement_noise = 0.05)
        : q_(process_noise), r_(measurement_noise) {
        reset();
    }

    /* 設定過程雜訊係數 */
    void setProcessNoise(double q) { q_ = q; }
    /* 設定觀測雜訊係數 */
    void setMeasurementNoise(double r) { r_ = r; }

    /*
     * 重置濾波器狀態
     * 清除所有狀態估計與協方差矩陣，回到未初始化狀態
     */
    void reset() {
        initialized_ = false;
        // 狀態向量: [x, y, vx, vy] 歸零
        x_ = {0.0, 0.0, 0.0, 0.0};
        // 協方差矩陣 P（4×4，用一維陣列展平，對角線初始化）
        P_.fill(0.0);
        P_[0]  = 1.0;   // P(0,0) = var(x)：位置 x 的初始不確定性
        P_[5]  = 1.0;   // P(1,1) = var(y)：位置 y 的初始不確定性
        P_[10] = 10.0;  // P(2,2) = var(vx)：速度 vx 的初始不確定性（較大，因為初始速度未知）
        P_[15] = 10.0;  // P(3,3) = var(vy)：速度 vy 的初始不確定性
    }

    /*
     * 用觀測值更新濾波器（預測 + 更新一體化）
     * 每次光達回呼到達時呼叫此函式
     *
     * @param meas_x      觀測到的 x 座標（光達質心）
     * @param meas_y      觀測到的 y 座標（光達質心）
     * @param[out] filtered_x  濾波後的 x 座標
     * @param[out] filtered_y  濾波後的 y 座標
     */
    void update(double meas_x, double meas_y,
                double& filtered_x, double& filtered_y) {
        auto now = std::chrono::steady_clock::now();  // 取得當前時間戳

        if (!initialized_) {
            // 首次呼叫：直接用觀測值初始化狀態，不做預測
            x_[0] = meas_x;   // 初始位置 x
            x_[1] = meas_y;   // 初始位置 y
            x_[2] = 0.0;      // 初始速度 vx = 0
            x_[3] = 0.0;      // 初始速度 vy = 0
            last_time_ = now;
            initialized_ = true;
            filtered_x = meas_x;
            filtered_y = meas_y;
            return;
        }

        // 計算 dt（秒）：兩次觀測之間的時間差
        double dt = std::chrono::duration<double>(now - last_time_).count();
        last_time_ = now;

        // 防護：dt 異常時使用預設值（對應約 10Hz 光達更新率）
        if (dt <= 0.0 || dt > 1.0) {
            dt = 0.1;
        }

        // === 預測步驟 (Predict) ===
        predict(dt);

        // === 更新步驟 (Correct / Update) ===
        correct(meas_x, meas_y);

        filtered_x = x_[0];  // 輸出濾波後的 x
        filtered_y = x_[1];  // 輸出濾波後的 y
    }

    /*
     * 僅執行預測（無觀測值時呼叫，例如目標暫時丟失）
     * 利用當前速度估計外推位置
     *
     * @param dt          時間間隔（秒）
     * @param[out] pred_x 預測的 x 座標
     * @param[out] pred_y 預測的 y 座標
     */
    void predictOnly(double dt, double& pred_x, double& pred_y) {
        if (!initialized_) {
            pred_x = 0.0;
            pred_y = 0.0;
            return;
        }
        predict(dt);
        pred_x = x_[0];  // 預測位置 x
        pred_y = x_[1];  // 預測位置 y
    }

    /*
     * 強制設定狀態（例如使用者手動指定目標位置時）
     * 速度歸零，協方差重置為中等不確定性
     */
    void setState(double x, double y) {
        x_[0] = x;        // 設定位置 x
        x_[1] = y;        // 設定位置 y
        x_[2] = 0.0;      // 重置速度 vx
        x_[3] = 0.0;      // 重置速度 vy
        // 重置協方差矩陣：位置不確定性較小，速度不確定性較大
        P_.fill(0.0);
        P_[0]  = 0.5;     // var(x)
        P_[5]  = 0.5;     // var(y)
        P_[10] = 10.0;    // var(vx)
        P_[15] = 10.0;    // var(vy)
        initialized_ = true;
        last_time_ = std::chrono::steady_clock::now();
    }

    bool isInitialized() const { return initialized_; }  // 檢查濾波器是否已初始化

    // 取得濾波後的速度估計
    double getVelocityX() const { return x_[2]; }  // 取得估計的 vx (m/s)
    double getVelocityY() const { return x_[3]; }  // 取得估計的 vy (m/s)

private:
    double q_;  // 過程雜訊係數：控制模型對動態變化的適應速度
    double r_;  // 觀測雜訊係數：控制對量測值的信任程度
    bool initialized_ = false;  // 濾波器是否已收到第一筆觀測

    // 狀態向量 [x, y, vx, vy]
    std::array<double, 4> x_;
    // 協方差矩陣 P (4×4 展平為一維陣列，row-major 排列)
    std::array<double, 16> P_;

    std::chrono::steady_clock::time_point last_time_;  // 上一次更新的時間戳

    /*
     * 預測步驟 (Predict)
     *
     * 狀態轉移: x' = F · x
     * F = | 1  0  dt  0 |
     *     | 0  1  0  dt |
     *     | 0  0  1   0 |
     *     | 0  0  0   1 |
     *
     * 過程雜訊矩陣 Q (基於等速模型的連續白噪聲積分):
     * Q = q · | dt³/3   0      dt²/2   0     |
     *         | 0       dt³/3  0       dt²/2 |
     *         | dt²/2   0      dt      0     |
     *         | 0       dt²/2  0       dt    |
     *
     * 協方差預測: P' = F · P · Fᵀ + Q
     */
    void predict(double dt) {
        // 狀態預測: x' = F · x
        x_[0] += x_[2] * dt;  // x' = x + vx·dt
        x_[1] += x_[3] * dt;  // y' = y + vy·dt
        // vx, vy 等速模型假設不變

        // 協方差預測: P' = F · P · Fᵀ + Q
        // 直接展開計算，避免引入矩陣運算函式庫
        double dt2 = dt * dt;    // dt²
        double dt3 = dt2 * dt;   // dt³

        // 先計算 F·P（暫存結果）
        std::array<double, 16> FP;
        // F·P 第 0 列: P[0]+dt·P[8], P[1]+dt·P[9], P[2]+dt·P[10], P[3]+dt·P[11]
        FP[0]  = P_[0]  + dt * P_[8];
        FP[1]  = P_[1]  + dt * P_[9];
        FP[2]  = P_[2]  + dt * P_[10];
        FP[3]  = P_[3]  + dt * P_[11];
        // F·P 第 1 列: P[4]+dt·P[12], P[5]+dt·P[13], P[6]+dt·P[14], P[7]+dt·P[15]
        FP[4]  = P_[4]  + dt * P_[12];
        FP[5]  = P_[5]  + dt * P_[13];
        FP[6]  = P_[6]  + dt * P_[14];
        FP[7]  = P_[7]  + dt * P_[15];
        // F·P 第 2 列: 速度列不受 F 影響，直接複製
        FP[8]  = P_[8];
        FP[9]  = P_[9];
        FP[10] = P_[10];
        FP[11] = P_[11];
        // F·P 第 3 列: 同上
        FP[12] = P_[12];
        FP[13] = P_[13];
        FP[14] = P_[14];
        FP[15] = P_[15];

        // P' = FP · Fᵀ + Q
        // Fᵀ 的行: [1,0,0,0], [0,1,0,0], [dt,0,1,0], [0,dt,0,1]
        P_[0]  = FP[0]  + FP[2]  * dt + q_ * dt3 / 3.0;   // var(x) + 過程雜訊
        P_[1]  = FP[1]  + FP[3]  * dt;                      // cov(x,y)
        P_[2]  = FP[2]  + q_ * dt2 / 2.0;                   // cov(x,vx)
        P_[3]  = FP[3];                                      // cov(x,vy)

        P_[4]  = FP[4]  + FP[6]  * dt;                      // cov(y,x)
        P_[5]  = FP[5]  + FP[7]  * dt + q_ * dt3 / 3.0;   // var(y) + 過程雜訊
        P_[6]  = FP[6]  + q_ * dt2 / 2.0;                   // cov(y,vx) — 注意：等速模型此處為 cov(y,vy)
        P_[7]  = FP[7];                                      // cov(y,vy)

        P_[8]  = FP[8]  + FP[10] * dt + q_ * dt2 / 2.0;   // cov(vx,x)
        P_[9]  = FP[9]  + FP[11] * dt;                      // cov(vx,y)
        P_[10] = FP[10] + q_ * dt;                           // var(vx) + 過程雜訊
        P_[11] = FP[11];                                      // cov(vx,vy)

        P_[12] = FP[12] + FP[14] * dt;                      // cov(vy,x)
        P_[13] = FP[13] + FP[15] * dt + q_ * dt2 / 2.0;   // cov(vy,y)
        P_[14] = FP[14] + q_ * dt;                           // cov(vy,vx) — 注意：此處為 var(vy) 方向的交叉項
        P_[15] = FP[15];                                      // var(vy)
    }

    /*
     * 更新步驟 (Correct / Measurement Update)
     *
     * 觀測矩陣: H = | 1  0  0  0 |
     *               | 0  1  0  0 |
     *
     * 新息 (Innovation):     y = z - H·x = [meas_x - x[0], meas_y - x[1]]
     * 新息協方差:             S = H·P·Hᵀ + R
     * 卡爾曼增益:             K = P·Hᵀ · S⁻¹
     * 狀態更新:               x = x + K·y
     * 協方差更新:             P = (I - K·H) · P
     *
     * 根據量測值修正狀態估計，降低追蹤抖動
     */
    void correct(double meas_x, double meas_y) {
        // 新息：觀測值與預測值的差
        double y0 = meas_x - x_[0];  // x 方向新息
        double y1 = meas_y - x_[1];  // y 方向新息

        // S = H·P·Hᵀ + R (2×2 矩陣)
        double s00 = P_[0]  + r_;     // S(0,0) = P(0,0) + r
        double s01 = P_[1];           // S(0,1) = P(0,1)
        double s10 = P_[4];           // S(1,0) = P(1,0)
        double s11 = P_[5]  + r_;     // S(1,1) = P(1,1) + r

        // S 的逆矩陣 (2×2 解析求逆)
        double det = s00 * s11 - s01 * s10;        // 行列式
        if (std::abs(det) < 1e-12) return;          // 奇異矩陣保護：避免除以零
        double inv_det = 1.0 / det;                  // 行列式的倒數
        double si00 =  s11 * inv_det;                // S⁻¹(0,0)
        double si01 = -s01 * inv_det;                // S⁻¹(0,1)
        double si10 = -s10 * inv_det;                // S⁻¹(1,0)
        double si11 =  s00 * inv_det;                // S⁻¹(1,1)

        // K = P·Hᵀ · S⁻¹  (4×2 卡爾曼增益矩陣)
        // P·Hᵀ = P 的前兩行: [P[0],P[1]; P[4],P[5]; P[8],P[9]; P[12],P[13]]
        double k00 = P_[0]  * si00 + P_[1]  * si10;  // K(0,0)：x 對 meas_x 的增益
        double k01 = P_[0]  * si01 + P_[1]  * si11;  // K(0,1)：x 對 meas_y 的增益
        double k10 = P_[4]  * si00 + P_[5]  * si10;  // K(1,0)：y 對 meas_x 的增益
        double k11 = P_[4]  * si01 + P_[5]  * si11;  // K(1,1)：y 對 meas_y 的增益
        double k20 = P_[8]  * si00 + P_[9]  * si10;  // K(2,0)：vx 對 meas_x 的增益
        double k21 = P_[8]  * si01 + P_[9]  * si11;  // K(2,1)：vx 對 meas_y 的增益
        double k30 = P_[12] * si00 + P_[13] * si10;  // K(3,0)：vy 對 meas_x 的增益
        double k31 = P_[12] * si01 + P_[13] * si11;  // K(3,1)：vy 對 meas_y 的增益

        // 狀態更新: x = x + K · y
        x_[0] += k00 * y0 + k01 * y1;  // 修正位置 x
        x_[1] += k10 * y0 + k11 * y1;  // 修正位置 y
        x_[2] += k20 * y0 + k21 * y1;  // 修正速度 vx
        x_[3] += k30 * y0 + k31 * y1;  // 修正速度 vy

        // 協方差更新: P = (I - K·H) · P
        // (I - K·H) 是 4×4 矩陣
        // (I-KH)[i][j] = I[i][j] - K[i][0]·H[0][j] - K[i][1]·H[1][j]
        // 其中 H[0][j] = (j==0)?1:0,  H[1][j] = (j==1)?1:0
        std::array<double, 16> P_new;
        auto P_old = P_;  // 保存舊 P 用於計算

        // 第 0 列
        P_new[0]  = (1.0 - k00) * P_old[0]  - k01 * P_old[4];
        P_new[1]  = (1.0 - k00) * P_old[1]  - k01 * P_old[5];
        P_new[2]  = (1.0 - k00) * P_old[2]  - k01 * P_old[6];
        P_new[3]  = (1.0 - k00) * P_old[3]  - k01 * P_old[7];
        // 第 1 列
        P_new[4]  = -k10 * P_old[0]  + (1.0 - k11) * P_old[4];
        P_new[5]  = -k10 * P_old[1]  + (1.0 - k11) * P_old[5];
        P_new[6]  = -k10 * P_old[2]  + (1.0 - k11) * P_old[6];
        P_new[7]  = -k10 * P_old[3]  + (1.0 - k11) * P_old[7];
        // 第 2 列
        P_new[8]  = -k20 * P_old[0]  - k21 * P_old[4]  + P_old[8];
        P_new[9]  = -k20 * P_old[1]  - k21 * P_old[5]  + P_old[9];
        P_new[10] = -k20 * P_old[2]  - k21 * P_old[6]  + P_old[10];
        P_new[11] = -k20 * P_old[3]  - k21 * P_old[7]  + P_old[11];
        // 第 3 列
        P_new[12] = -k30 * P_old[0]  - k31 * P_old[4]  + P_old[12];
        P_new[13] = -k30 * P_old[1]  - k31 * P_old[5]  + P_old[13];
        P_new[14] = -k30 * P_old[2]  - k31 * P_old[6]  + P_old[14];
        P_new[15] = -k30 * P_old[3]  - k31 * P_old[7]  + P_old[15];

        P_ = P_new;  // 將更新後的協方差寫回
    }
};

#endif // KALMAN_FILTER_HPP
