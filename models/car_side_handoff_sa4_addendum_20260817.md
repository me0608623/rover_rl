# Addendum: SA4 candidates added alongside SA1-R1 c1700

2026-08-17. Read `car_side_handoff_sa1r1_c1700_83d_k8_20260817.md` first — every
contract value, required fix and gate in it applies unchanged to these two.

## Contract is identical for all three checkpoints

```
obs 83D  |  rl_in 179D  |  K=8  |  feat_norm=False
action_history_accel_normalizer = 0.5
action_history_omega_normalizer = 1.2
```

So no car-side config change is needed to switch between them. Only the model
path and its golden change.

## Files

| checkpoint | file | SHA-256 | golden |
|---|---|---|---|
| SA4-R3 it50 | `sa4_r3_checkpoint_6400.pt` | `fa51b5a312d38b7aa1ee93c3464b087b9b9cb985a7137eefdc8fecf40fed166d` | `e2e_golden_sa4_r3_it50_83d_k8.npz` |
| SA4-R2 c6400 | `sa4_r2_checkpoint_6400.pt` | `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197` | `e2e_golden_sa4_r2_c6400_83d_k8.npz` |

## Measured capability (fixed screens, stage-4 geometry, seed818, d1=200 ms)

Threshold for a pass is `SR >= 90%, CR <= 10%, TO <= 5%`.

| checkpoint | nav_native | corridor lateral | corridor longitudinal |
|---|---|---|---|
| SA1-R1 c1700 | Nav20-clean **9/9 PASS** (SR>=98/CR<=1.5, 3 seeds x 3 delays) | not claimed | not claimed |
| SA4-R3 it50 | 94.06% / 5.94% **PASS** | 81.13% / 18.87% FAIL | 78.21% / 21.73% FAIL |
| SA4-R3 it25 | 95.09% / 4.91% **PASS** | 78.75% / 21.25% FAIL | 50.39% / 48.97% FAIL |
| SA4-R2 c6400 | not measured | 83.05% / 16.95% FAIL | **98.47% / 1.53% PASS** |
| SA4 pilot c100 | 92.67% / 7.33% **PASS** | 85.90% / 14.10% FAIL | 70.68% / 29.32% FAIL |

### How to read this

- **No SA4 checkpoint passes the joint gate.** `accepted_parent = null`; SA4 is
  on HOLD. Deploying one is a diagnostic run, not a validated capability.
- **SA4-R2 c6400 is the only SA4 checkpoint that passes any corridor cell**, and
  it passes longitudinal (head-on pedestrians) at CR 1.53%. Its lateral
  (pedestrians crossing) is 16.95% FAIL, and its native was never measured in
  that screen.
- **SA4-R3 it50 is the safest all-round SA4 choice**: native passes and it has
  the best worst-direction corridor CR (21.73%) of the R3 pair.
- SA1-R1 c1700 remains the only checkpoint with a formal pass across **three
  delays** (0/200/400 ms) and three seeds, which is why it is still the
  recommended first run for validating the delay contract itself.

### Suggested use

| what you want to test | checkpoint |
|---|---|
| delay contract + parity (first bring-up) | SA1-R1 c1700 |
| open room, stage-4 trained | SA4-R3 it50 |
| head-on pedestrian corridor | SA4-R2 c6400 |
| crossing pedestrian corridor | none — best measured is 14.10% CR |

## Same physical procedure applies

Elevated first, e-stop in hand, `speed_rate = 1.0`, >= 60 s continuous, watch for
the W1-c10 signature (`|rl_w| >= 0.83` fraction, sign flips, `map_yaw` jumps).
See section 5 of the main handoff.

Note that with wheels off the ground `odom_w` still reflects the commanded wheel
motion, so the command-to-drivetrain delay loop is observable, but `map_yaw` will
not move — the full closed-loop limit cycle cannot be reproduced elevated.

## Expect collisions in a corridor

If the ground test happens in a corridor with people, the measured rates above
say roughly 1 collision every 5 episodes for lateral traffic. Treat such a run as
data collection with a human on the e-stop, not as a capability check.
