# Car-Side Handoff: SA1-R1 c1700 (83D / K8) — first parity-gated candidate

Prepared 2026-08-17 on the training machine. Read this whole file before touching
the vehicle. **Nothing here authorises driving on the ground.**

---

## 0. What is being handed over

| item | value |
|---|---|
| checkpoint | `checkpoint_217600.pt` (iteration 1699, i.e. "c1700") |
| run | `sa1_sim2real_v1_ne1024_s42_r1` |
| SHA-256 | `8dcedfa7f94b9c60c8a2d9bcab6cff283cce10844e32a25e9c25106b22f460b0` |
| golden parity | `e2e_golden_sa1r1_c1700_83d_k8.npz` |
| destination | `/home/aa/rover_rl/models/` |

### Why this checkpoint

It is the only candidate that was **trained against the failure mode that killed
W1-c10**, and it passed a formal capability gate that included the real robot's
delay:

- trained with measured VLP-16 `full` noise (`lidar_no_noise=False`)
- trained with actuator delay `U{0,1,2}` steps = `0 / 200 / 400 ms`
- Nav20-clean acceptance: **9/9 blocking cells PASS** across
  3 seeds (515/616/717) x 3 fixed delays (d0/d1/d2);
  thresholds `SR>=0.98, CR<=0.015, TO<=0.005`; realised ~100% SR / 0% CR
  over ~20,000 episodes

### What it has NOT been shown to do

`scope = "sa1_hard_gate_nav20_clean_only"`.

- **Open-room navigation only** (20x20 m). No deployment-level claim for
  corridors or narrow gaps.
- Corridor capability is unsolved in this project line (SA4 is on HOLD; the most
  recent corridor screens fail at CR ~19-22%). **Do not put this policy in a
  corridor with pedestrians.**
- Its verdict is `deployable=false`, blocked solely by
  `python_torchscript_parity: "required_external_83d_k8_gate"` — i.e. by the gate
  in section 2 of this document, which had never been built until now.

---

## 1. Inference contract (all values read from source, not inferred)

```
obs_raw[83]  -- obs_normalizer(83D) -> clamp[-5,5] -->  obs_normed[83]
ext_in = cat(obs_normed[83], lidar_hist[(8-1)*72 = 504])          # 587D
feat   = LidarStateExtractor(ext_in, include_act_hist=True, K=8)  # 96D
lidar_hist <- cat(obs_normed[6:78], lidar_hist[:-72])             # prepend current, drop oldest
rl_in  = cat(obs_normed[83], feat[96])                            # 179D
logits = PolicyHead(rl_in)                                        # 38 = [19 linear | 19 angular]
action = [argmax(logits[:19]), argmax(logits[19:])]               # deterministic
```

`lidar_hist` is zeroed on episode reset (`PolicyRunner.reset()`).
`feat_norm = False`, so there is **no** feat_normalizer.

### 83D observation layout (authoritative: `modular_rnn_models.py:51-59`)

| slice | content |
|---|---|
| `[0:4]` | ego: accel, speed, omega, radius |
| `[4:6]` | goal (body frame) |
| `[6:78]` | LiDAR, 72 bins |
| `[78:79]` | remaining-time ratio |
| `[79:83]` | **act_hist**: past 2 issued `[linear_accel, omega]` pairs |

The frame-stack history is appended **after** act_hist, because the extractor
slices it as `obs[:, -(K-1)*72:]`.

### Action decoder (authoritative: `charge_env_cfg_vlp16.py:338-343`)

```
num_bins                = 19          (19 x 19 = 361)
max_linear_velocity     = 1.0  m/s
reverse_velocity_scale  = 0.2         -> reverse lower bound is -0.2 m/s
max_linear_accel        = 0.5  m/s^2
max_angular_vel         = 1.2  rad/s
max_angular_accel       = 3.0  rad/s^2   (command slew)
control_dt              = 0.2  s      (decimation=20, sim.dt=0.01)
```

Reverse scaling must be applied **inside the decoder**, not as a later
output-only clamp. W1-c10 produced `rl_v = -0.557 m/s` and a downstream filter
clamped `sent_v = -0.2`, which is not the same policy.

### LiDAR calibration (authoritative: `charge_env_cfg_vlp16.py:423-426`)

```
r_max     = 20.0 m
r_robot   = 0.35 m   (ROBOT_BODY_RADIUS)
r_min     = 0.5  m   <-- hits closer than this are treated as blind (set to r_max)
z_filter  = 0.5  m
```

`r_min = 0.5` is **not** 0.25. This is an independent calibration gate that the
network-parity test in section 2 **cannot** catch, because parity checks the
network graph, not observation construction.

---

## 2. REQUIRED FIX before parity can pass: act_hist normalisers

This is the highest-risk item in the handover.

`policy_params_w1c10.yaml` currently documents:

```
act_stack size=2,  a_max=0.2,  omega_max=pi/15   (~0.209)
```

The checkpoint stores what training actually used:

```
action_history_accel_normalizer = 0.5
action_history_omega_normalizer = 1.2
```

So each act_hist entry must be encoded as `[linear_accel / 0.5, omega / 1.2]`,
newest to oldest, finally clipped to `[-2, 2]`.

Concretely, for a real issued command of `omega = 0.6 rad/s`:

| divisor | value fed to the network |
|---|---|
| training (`1.2`) | **0.5** |
| current car config (`0.209`) | 2.87 -> clipped to **2.0** |

Same physical action, network sees 0.5 vs 2.0. These four inputs are precisely
what the policy uses to reason about the 200 ms actuator delay, so feeding them
on the wrong scale corrupts exactly the mechanism this checkpoint was trained to
exercise. The old `0.2 / pi/15` divisors belong only to legacy checkpoints.

**Read the divisors from the checkpoint; do not hardcode them.** Both values are
in `ckpt["args"]`.

### One issued-command layer

The act_hist must contain the two previous **post-decode / post-slew** controls
`[linear_accel, omega]` that were used to build the velocity commands entering
the actuator transport-delay queue. Decoder, history and diagnostics must all
refer to the same decode invocation. W1-c10's diagnostic recorded a different
layer from what was actually sent.

---

## 3. Parity gate (network graph)

```bash
# on the car, after exporting the bundle
python models/e2e_bundle_parity.py  --golden models/e2e_golden_sa1r1_c1700_83d_k8.npz
python models/e2e_runtime_parity.py --golden models/e2e_golden_sa1r1_c1700_83d_k8.npz
```

The golden npz contains `obs_seq` (RAW pre-normalisation, `[8, 2, 83]`),
`logits [8,2,38]`, `actions [8,2,2]`, `mean`, `var`, `frame_stack=8`,
`rl_in_dim=179`.

Pass condition, matching the 2026-07-16 precedent for the 79D/K4 bundle:

- `max|delta logits| <= 1e-5`
- **0** action mismatches

The previously deployed 79D/K4 bundle achieved `6.7e-6` (bundle) and `4.8e-6`
(runtime) with 0 mismatches. Anything materially worse means the port is wrong,
not noisy.

> Note: the oracle was extended today to support 83D/K8; it previously hard
> asserted `obs_dim == 79`. If the parity scripts also assume 79D, they need the
> same extension. The layout in section 1 is the authoritative reference.

---

## 4. Gates parity cannot catch — check these separately

1. **`r_min = 0.5`** applied to the LiDAR construction (section 1).
2. **act_hist divisors and issued-command layer** (section 2).
3. **Decoder bounds**, especially the `-0.2 m/s` reverse bound applied inside the
   decoder.
4. **Continuous 83D obs at dt = 0.2 s.** Frame stacking is what carries motion
   information; if observations arrive irregularly the stack is meaningless.

---

## 5. Physical bring-up — wheels off the ground first

Do **not** drive on the ground before this passes.

```
1. Wheels elevated, e-stop in hand.
2. Log rl_w (commanded angular velocity) at full rate.
3. Watch for the W1-c10 failure signature:
      |rl_w| >= 0.83        was 33.7% of the time
      left/right sign flips  was 13 in 57 s
      map_yaw jumping ~97 deg in ~0.05 s
4. If that signature appears, stop. It is a delay-induced limit cycle
   (delay/dt ~= 1.0) and it will not be fixed by tuning on the vehicle.
5. Only if the elevated run is clean: low-speed ground test, open space,
   no pedestrians, e-stop in hand.
```

`w1c10_k8_e2e_1280.ts` remains permanently classified REAL-ROBOT FAIL. W1 bridge
c10/c15 and SA7/SA7.1 are not deployment candidates. This handover does not
change those verdicts.

---

## 6. Provenance

- Golden produced by
  `scripts/reinforcement_learning/skrl/rnn_car_wdclean/e2e_parity_oracle.py`
  (extended 2026-08-17 for 83D/K8; fails closed if the derived `rl_in_dim`
  disagrees with the checkpoint's policy head).
- Acceptance evidence: `logs/gates/sa1_r1_nav20_acceptance/early_c1700_clean/`
- Contract sources: `modular_rnn_models.py:51-59`,
  `charge_env_cfg_vlp16.py:338-343` and `:423-426`
