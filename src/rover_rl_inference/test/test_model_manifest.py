"""model_manifest 檢查測試（Mandatory test #6：啟動須拒絕錯誤規格）。"""

import json
import math
import os
import tempfile
import unittest

from rover_rl_inference.model_manifest import (
    ManifestMismatch, verify_bundle, sha256_file,
)

PI_15 = math.pi / 15.0


class _Bundle:
    def __init__(self, raw=83, e2e=True, fs=8, lh=504, logits=38):
        self.raw_obs_dim = raw
        self.end_to_end = e2e
        self.frame_stack = fs
        self.lidar_hist_dim = lh
        self.total_logits = logits


def _spec(raw=83, fs=8):
    return {
        "raw_obs_dim": raw,
        "frame_stack": fs,
        "end_to_end_frame_stack": True,
        "act_hist_mode": "raw",
        "act_stack_a_max": 0.2,
        "act_stack_omega_max": PI_15,
    }


def _fixture():
    return {
        "parameters": {
            "num_bins": 19,
            "max_linear_velocity_mps": 1.0,
            "max_angular_velocity_rad_s": 1.2,
            "max_linear_accel_mps2": 0.5,
            "max_angular_accel_rad_s2": 3.0,
            "control_dt_s": 0.2,
            "reverse_velocity_scale": 0.2,
        },
        "semantics": {
            "reverse_bound_location": "inside decoder",
            "issued_history": "two previous values after decode/slew and before actuator delay",
            "actuator_delay_location": "after decode/slew/history",
        },
    }


def _runtime(**over):
    d = {
        "act_max_linear_velocity": 1.0,
        "act_max_angular_velocity": 1.2,
        "act_max_linear_accel": 0.5,
        "act_max_angular_accel": 3.0,
        "control_dt": 0.2,
        "reverse_velocity_scale": 0.2,
        "speed_rate": 0.5,
        "act_stack_size": 2,
        "act_stack_a_max": 0.2,
        "act_stack_omega_max": PI_15,
        "cmd_passthrough": True,
    }
    d.update(over)
    return d


class ModelManifestTest(unittest.TestCase):
    def setUp(self):
        fd, self.model = tempfile.mkstemp(suffix=".ts")
        os.write(fd, b"dummy-model-bytes")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.model)

    def _verify(self, **over):
        return verify_bundle(
            model_path=self.model, bundle=over.pop("bundle", _Bundle()),
            obs_spec=over.pop("obs_spec", _spec()),
            fixture=over.pop("fixture", _fixture()),
            runtime=_runtime(**over.pop("runtime", {})),
            expected_sha256=over.pop("expected_sha256", None),
            strict=over.pop("strict", True),
        )

    # ── 正常路徑 ──
    def test_matching_configuration_passes(self):
        rep = self._verify()
        self.assertTrue(rep.ok, rep.text())

    def test_sha256_matches(self):
        rep = self._verify(expected_sha256=sha256_file(self.model))
        self.assertTrue(rep.ok, rep.text())

    # ── 每一種不符都必須擋下 ──
    def test_rejects_wrong_sha256(self):
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(expected_sha256="0" * 64)
        self.assertIn("model_sha256", str(e.exception))

    def test_rejects_v3c_omega_bug_0785(self):
        """最重要的一條：ω 被寫回 0.785（v3c 已知 bug）必須拒絕啟動。"""
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(runtime={"act_max_angular_velocity": 0.785})
        self.assertIn("act_max_angular_velocity", str(e.exception))

    def test_rejects_reverse_scale_mismatch(self):
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(runtime={"reverse_velocity_scale": 1.0})
        self.assertIn("reverse_velocity_scale", str(e.exception))

    def test_rejects_wrong_control_dt(self):
        with self.assertRaises(ManifestMismatch):
            self._verify(runtime={"control_dt": 0.1})

    def test_rejects_act_stack_norm_mismatch(self):
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(runtime={"act_stack_omega_max": 0.2})
        self.assertIn("act_stack_omega_max", str(e.exception))

    def test_rejects_lidar_hist_inconsistent_with_frame_stack(self):
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(bundle=_Bundle(fs=8, lh=216))   # 216 是 K=4 的值
        self.assertIn("lidar_hist_dim_consistent", str(e.exception))

    def test_rejects_obs_spec_bundle_dim_mismatch(self):
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(bundle=_Bundle(raw=83), obs_spec=_spec(raw=79))
        self.assertIn("obs_spec_matches_bundle", str(e.exception))

    def test_rejects_83d_without_passthrough(self):
        """83D 沒開 passthrough → published 會 ≠ issued，必須擋。"""
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(runtime={"cmd_passthrough": False})
        self.assertIn("cmd_passthrough_true_for_83d", str(e.exception))

    def test_rejects_missing_fixture(self):
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(fixture=None)
        self.assertIn("action_fixture_present", str(e.exception))

    def test_rejects_wrong_semantics(self):
        fx = _fixture()
        fx["semantics"]["reverse_bound_location"] = "output clamp"
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(fixture=fx)
        self.assertIn("reverse_bound_inside_decoder", str(e.exception))

    def test_reports_all_failures_not_just_first(self):
        """一次列出所有不符項，避免修一個跑一次。"""
        with self.assertRaises(ManifestMismatch) as e:
            self._verify(runtime={"act_max_angular_velocity": 0.785,
                                  "control_dt": 0.1})
        msg = str(e.exception)
        self.assertIn("act_max_angular_velocity", msg)
        self.assertIn("control_dt", msg)

    def test_non_strict_returns_report_instead_of_raising(self):
        rep = self._verify(runtime={"control_dt": 0.1}, strict=False)
        self.assertFalse(rep.ok)
        self.assertIn("control_dt", rep.text())

    def test_79d_does_not_require_passthrough(self):
        """既有 79D 部署不受新規則影響。"""
        rep = self._verify(
            bundle=_Bundle(raw=79, fs=4, lh=216),
            obs_spec=_spec(raw=79, fs=4),
            runtime={"cmd_passthrough": False},
            strict=False,
        )
        names = {n for n, ok, _ in rep.checks if not ok}
        self.assertNotIn("cmd_passthrough_true_for_83d", names)


if __name__ == "__main__":
    unittest.main()
