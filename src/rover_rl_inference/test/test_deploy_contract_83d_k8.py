"""83D/K8 家族實檔部署契約測試（handoff §4：parity 抓不到的那幾項）。

涵蓋三顆 checkpoint —— SA1-R1 c1700 / SA4-R3 it50 / SA4-R2 c6400。
addendum 明載三者契約逐字相同，所以它們吃**同一組**斷言；
`test_contract_bodies_are_identical_across_models` 直接把那句話變成可執行檢查。

network parity 驗的是「網路圖」；觀測怎麼組、命令用什麼常數 decode，它一概驗不到。
W1-c10 的兩個實車失效都落在這個盲區：
    - act_hist 分母沿用舊模型的 0.2 / π/15（訓練實際是 0.5 / 1.2）
    - 倒車界在輸出層而不在 decoder 內

本測試不造假資料，直接讀**要上車的那幾個檔**：
    src/rover_rl_bringup/config/policy_params_<cfg>.yaml
    src/rover_rl_bringup/config/lidar_preprocessor_params_<cfg>.yaml
    models/<model>.obs_spec.json            (由 export_policy 從 checkpoint 產生)
    docs/freeze/sa1_action_contract_v1.json (訓練端凍結 fixture)
把 policy_node 開機時做的 manifest 檢查在 CI 先跑一遍，並補上 manifest 沒涵蓋的
LiDAR 校準同步（r_min/r_max/r_robot 兩個 yaml 必須一致）。
"""

import json
import pathlib
import unittest

import yaml

from rover_rl_inference.model_manifest import ManifestMismatch, verify_bundle

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CFG = _ROOT / "src/rover_rl_bringup/config"
_MODELS_DIR = _ROOT / "models"
_FIXTURE = _ROOT / "docs/freeze/sa1_action_contract_v1.json"

#: (顯示名, config 後綴, .ts 檔名)
MODELS = (
    ("SA1-R1 c1700", "sa1r1", "sa1r1_c1700_83d_k8.ts"),
    ("SA4-R3 it50", "sa4r3", "sa4_r3_it50_83d_k8.ts"),
    ("SA4-R2 c6400", "sa4r2", "sa4_r2_c6400_83d_k8.ts"),
)

#: 訓練 checkpoint args 的權威值（三顆相同，addendum §「Contract is identical」）：
#:   action_history_accel_normalizer = 0.5 / action_history_omega_normalizer = 1.2
#: 換 checkpoint 時要連同 yaml + sidecar 一起更新，不是放寬這裡。
WANT_ACT_STACK_A_MAX = 0.5
WANT_ACT_STACK_OMEGA_MAX = 1.2
#: 舊模型（v3c/v3e）的值。誤用會讓 ω=0.6 rad/s 從 0.5 變成 clip 後的 2.0。
LEGACY_A_MAX = 0.2
LEGACY_OMEGA_MAX = 0.20943951023931953   # π/15

#: 允許逐模型不同的參數；其餘每一項三顆必須完全相同。
PER_MODEL_KEYS = frozenset({"model_path", "manifest_expected_sha256"})


def _params(path: pathlib.Path, node: str) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))[node]["ros__parameters"]


class _MetaBundle:
    """從 .ts 的 _meta_dims 取值；torch 不在時退回 sidecar（仍能驗 yaml↔spec）。"""

    def __init__(self, spec: dict, model: pathlib.Path):
        self.source = "obs_spec"
        self.raw_obs_dim = int(spec["raw_obs_dim"])
        self.frame_stack = int(spec.get("frame_stack", 1))
        self.lidar_hist_dim = int(spec.get("lidar_hist_dim", 0))
        self.end_to_end = bool(spec.get("end_to_end_frame_stack", False))
        self.total_logits = 38
        if not model.is_file():
            return
        try:
            import torch
        except ImportError:
            return
        meta = torch.jit.load(str(model))._meta_dims.tolist()
        self.source = "ts_meta"
        self.raw_obs_dim = int(meta[0])
        self.lidar_hist_dim = int(meta[2])
        self.total_logits = int(meta[4])
        self.end_to_end = int(meta[5]) == 1
        self.frame_stack = int(meta[6])


class _Rec:
    def __init__(self, tag: str, cfg: str, model_file: str):
        self.tag = tag
        self.model = _MODELS_DIR / model_file
        self.spec_path = _MODELS_DIR / (model_file[:-3] + ".obs_spec.json")
        self.policy_yaml = _CFG / f"policy_params_{cfg}.yaml"
        self.preproc_yaml = _CFG / f"lidar_preprocessor_params_{cfg}.yaml"
        self.pol = _params(self.policy_yaml, "rover_rl_policy")
        self.pre = _params(self.preproc_yaml, "rover_rl_lidar_preprocessor")
        self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        self.bundle = _MetaBundle(self.spec, self.model)

    def runtime(self) -> dict:
        p = self.pol
        return {
            "act_max_linear_velocity": p["act_max_linear_velocity"],
            "act_max_angular_velocity": p["act_max_angular_velocity"],
            "act_max_linear_accel": p["act_max_linear_accel"],
            "act_max_angular_accel": p["act_max_angular_accel"],
            "control_dt": p["control_dt"],
            "reverse_velocity_scale": p["reverse_velocity_scale"],
            "speed_rate": p["speed_rate"],
            "act_stack_size": p["act_stack_size"],
            "act_stack_a_max": p["act_stack_a_max"],
            "act_stack_omega_max": p["act_stack_omega_max"],
            "cmd_passthrough": p["cmd_passthrough"],
        }


class DeployContract83dK8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _FIXTURE.is_file():
            raise unittest.SkipTest(f"缺 {_FIXTURE}")
        cls.fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        cls.recs = []
        missing = []
        for tag, cfg, model_file in MODELS:
            paths = (_CFG / f"policy_params_{cfg}.yaml",
                     _CFG / f"lidar_preprocessor_params_{cfg}.yaml",
                     _MODELS_DIR / (model_file[:-3] + ".obs_spec.json"))
            if all(p.is_file() for p in paths):
                cls.recs.append(_Rec(tag, cfg, model_file))
            else:
                missing.append(tag)
        if not cls.recs:
            raise unittest.SkipTest(f"三顆模型的部署檔都不在（{missing}）")
        cls.missing = missing

    def _each(self):
        for r in self.recs:
            with self.subTest(model=r.tag):
                yield r

    # ── 開機 manifest：把 policy_node 啟動時的檢查在此先跑 ──────────────
    def test_boot_manifest_passes_with_shipped_files(self):
        for r in self._each():
            if not r.model.is_file():
                self.skipTest(f"缺 {r.model}（先跑 export_policy.py）")
            rep = verify_bundle(
                model_path=str(r.model), bundle=r.bundle, obs_spec=r.spec,
                fixture=self.fx, runtime=r.runtime(),
                expected_sha256=r.pol["manifest_expected_sha256"] or None,
                strict=True,
            )
            self.assertTrue(rep.ok, f"[{r.tag}]\n{rep.text()}")

    def test_boot_manifest_rejects_legacy_act_stack_divisors(self):
        """把 yaml 改回舊分母 → 必須拒絕啟動（這就是 W1-c10 的 obs 污染路徑）。"""
        for r in self._each():
            rt = r.runtime()
            rt["act_stack_a_max"] = LEGACY_A_MAX
            rt["act_stack_omega_max"] = LEGACY_OMEGA_MAX
            with self.assertRaises(ManifestMismatch) as e:
                verify_bundle(
                    model_path=str(r.model), bundle=r.bundle, obs_spec=r.spec,
                    fixture=self.fx, runtime=rt, strict=True,
                )
            msg = str(e.exception)
            self.assertIn("act_stack_a_max", msg)
            self.assertIn("act_stack_omega_max", msg)

    def test_boot_manifest_rejects_swapped_model_sha(self):
        """拿 A 的 yaml 配 B 的 .ts（選錯模型/複製 config 忘改 sha）必須擋下。"""
        if len(self.recs) < 2:
            self.skipTest("需要至少兩顆模型才能測交叉配對")
        for i, r in enumerate(self.recs):
            other = self.recs[(i + 1) % len(self.recs)]
            with self.subTest(yaml=r.tag, ts=other.tag):
                if not other.model.is_file():
                    continue
                with self.assertRaises(ManifestMismatch) as e:
                    verify_bundle(
                        model_path=str(other.model), bundle=other.bundle,
                        obs_spec=other.spec, fixture=self.fx, runtime=r.runtime(),
                        expected_sha256=r.pol["manifest_expected_sha256"],
                        strict=True,
                    )
                self.assertIn("model_sha256", str(e.exception))

    # ── addendum：三顆契約逐字相同 ──────────────────────────────────────
    def test_contract_bodies_are_identical_across_models(self):
        """除 model_path / sha256 外，三份 policy yaml 必須每一項相同。

        addendum：「no car-side config change is needed to switch between them.
        Only the model path and its golden change.」—— 這條測試就是那句話。
        """
        if len(self.recs) < 2:
            self.skipTest("只有一顆模型，無從比較")
        base = self.recs[0]
        for r in self.recs[1:]:
            with self.subTest(model=r.tag, vs=base.tag):
                self.assertEqual(set(base.pol), set(r.pol), "參數鍵集合不同")
                diff = {k for k in base.pol
                        if k not in PER_MODEL_KEYS and base.pol[k] != r.pol[k]}
                self.assertFalse(diff, f"契約值與 {base.tag} 不同：{sorted(diff)}")
                self.assertEqual(base.pre, r.pre, "preprocessor 參數不同")
                # 反向：per-model 的兩項必須真的不同（否則就是複製忘了改）
                self.assertNotEqual(base.pol["model_path"], r.pol["model_path"])
                self.assertNotEqual(base.pol["manifest_expected_sha256"],
                                    r.pol["manifest_expected_sha256"])

    # ── act_hist 分母：sidecar 與 yaml 都必須是 checkpoint 的真值 ────────
    def test_sidecar_divisors_come_from_checkpoint(self):
        for r in self._each():
            self.assertAlmostEqual(r.spec["act_stack_a_max"],
                                   WANT_ACT_STACK_A_MAX, delta=1e-12)
            self.assertAlmostEqual(r.spec["act_stack_omega_max"],
                                   WANT_ACT_STACK_OMEGA_MAX, delta=1e-12)
            self.assertEqual(r.spec.get("act_stack_norm_source"), "checkpoint args",
                             "sidecar 的分母不是從 checkpoint args 讀出來的")

    def test_yaml_divisors_are_not_legacy_values(self):
        for r in self._each():
            self.assertAlmostEqual(r.pol["act_stack_a_max"],
                                   WANT_ACT_STACK_A_MAX, delta=1e-12)
            self.assertAlmostEqual(r.pol["act_stack_omega_max"],
                                   WANT_ACT_STACK_OMEGA_MAX, delta=1e-12)
            self.assertNotAlmostEqual(r.pol["act_stack_omega_max"],
                                      LEGACY_OMEGA_MAX, delta=1e-6)

    def test_act_hist_mode_is_raw_per_sidecar(self):
        """auto = 讀 sidecar；sidecar 必須明確是 raw（這批 checkpoint 非 action_error）。"""
        for r in self._each():
            self.assertEqual(r.spec["act_hist_mode"], "raw")
            self.assertIn(r.pol["act_hist_mode"], ("auto", "raw"))

    # ── LiDAR 校準：manifest 沒涵蓋，只能靠兩個 yaml 一致 ────────────────
    def test_lidar_calibration_matches_between_yamls(self):
        for r in self._each():
            self.assertAlmostEqual(r.pol["lidar_r_min_m"], r.pre["r_min"], delta=1e-12)
            self.assertAlmostEqual(r.pol["lidar_r_max_m"], r.pre["r_max"], delta=1e-12)
            self.assertAlmostEqual(r.pol["robot_radius_m"], r.pre["r_robot"], delta=1e-12)
            self.assertAlmostEqual(r.pol["lidar_z_filter_m"], r.pre["z_filter"], delta=1e-12)

    def test_lidar_r_min_is_training_value_not_025(self):
        """r_min=0.5（charge_env_cfg_vlp16.py:423-426）。0.25 是 v3c 家族的值。"""
        for r in self._each():
            self.assertAlmostEqual(r.pol["lidar_r_min_m"], 0.5, delta=1e-12)
            self.assertAlmostEqual(r.pol["lidar_r_max_m"], 20.0, delta=1e-12)

    def test_motion_compensation_off_on_both_sides(self):
        for r in self._each():
            self.assertFalse(r.pre["motion_compensation"])
            self.assertFalse(r.pol["lidar_motion_compensation"])

    # ── ego ObsTerm 分母 + episode 長度（權威：charge_env_cfg_vlp16.py，
    #    2026-08-17 PC 端補查後定案）──────────────────────────────────────
    def test_ego_obs_normalizers_match_training(self):
        """L389/L484、L394/L488、L399/L492。三個分母都不是可調參數。"""
        for r in self._each():
            self.assertAlmostEqual(r.pol["obs_max_acceleration"], 1.0, delta=1e-12)
            self.assertAlmostEqual(r.pol["obs_max_linear_velocity"], 1.0, delta=1e-12)
            self.assertAlmostEqual(r.pol["obs_max_angular_velocity"], 1.5, delta=1e-12)
            self.assertAlmostEqual(r.pol["robot_radius_m"], 0.35, delta=1e-12)

    def test_episode_horizon_is_60_not_45(self):
        """權威 = `wd_single_agent_v3.py:108` 的 stage 表：SA1~SA4 皆 **60s**。

            stage:      SA1 SA2 SA3 SA4 | SA5 SA6 SA7  SA8
            episode_s:   60  60  60  60 |  75  90 120  180

        此血緣的 task 是 Isaac-Navigation-Charge-VLP16-Curriculum-WD，
        `episode_length_s = float(scene["episode_s"])` 由 stage 表決定。
        ⚠ `charge_env_cfg_vlp16_curriculum.py:612` 的 45.0 只是 Phase 1 初始值
        （註解自己就寫「課程動態調整 45→90s」），2026-08-17 曾據此誤設成 45.0，同日更正。
        本測試同時擋回 45.0，避免又被那個初始值帶走。
        """
        for r in self._each():
            self.assertAlmostEqual(r.pol["episode_horizon_s"], 60.0, delta=1e-12)
            self.assertNotAlmostEqual(r.pol["episode_horizon_s"], 45.0, delta=1e-6)

    # ── 疊幀前提：連續等間隔 obs ────────────────────────────────────────
    def test_frame_stack_and_control_period(self):
        for r in self._each():
            self.assertEqual(r.bundle.frame_stack, 8)
            self.assertEqual(r.bundle.lidar_hist_dim, (8 - 1) * 72)
            self.assertEqual(r.spec["rl_input_dim"], 179)
            self.assertAlmostEqual(r.pol["control_dt"], 0.2, delta=1e-12)
            self.assertAlmostEqual(r.pre["publish_rate_hz"], 10.0, delta=1e-12)

    # ── 延遲：訓練已建模，車端不得再補一次 ──────────────────────────────
    def test_cmd_delay_compensation_disabled(self):
        for r in self._each():
            self.assertFalse(r.pol["cmd_delay_comp_enable"])
            self.assertAlmostEqual(r.pol["cmd_delay_comp_s"], 0.0, delta=1e-12)

    # ── 單一 issued-command 層 ─────────────────────────────────────────
    def test_single_issued_command_layer(self):
        sem = self.fx["semantics"]
        self.assertEqual(sem["reverse_bound_location"], "inside decoder")
        for r in self._each():
            self.assertTrue(r.pol["cmd_passthrough"],
                            "83D 必須 passthrough，否則 published ≠ issued，history 記錯層")
            self.assertAlmostEqual(r.pol["reverse_velocity_scale"], 0.2, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
