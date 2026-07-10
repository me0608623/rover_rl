# Obs spec — sa5_v3f_tcadapt_60000.ts

- raw_obs_dim: **79** → used: 79
- hidden_dim: 64, preprocess_dim: 12
- **act_hist_mode: `raw`**

| dims | name | 語義 |
|------|------|------|
| [0:4] | ego | [accel_norm, speed_norm, omega_norm, radius_norm] 機器人本體狀態 |
| [4:6] | goal | [goal_x_body, goal_y_body] 目標 body-frame 方向 |
| [6:78] | lidar | 72-bin sweep，r_min~r_max 正規化 [0,1] |
| [78] | time | episode 時間 ramp [0,1] |
