"""chain_trace JSONL 必須帶得走診斷需要的欄位。

PC 端判讀架空資料時要的三樣東西全靠這個檔案：`issued_w`（ω 意圖）、`hist`
（逐字餵進網路的 act_hist）、`obs_time_rem`（逐拍 obs[78]）。
欄位若在重構中消失，log 看起來仍然完全正常 —— 只是再也驗不出
「episode_horizon 分母寫錯」與「時間特徵長期夾在 0」這兩種無徵兆錯誤。
"""

import json
import pathlib
import tempfile
import unittest

from rover_rl_inference.chain_trace import ChainTracer

#: PC 端判讀必需欄位（缺一個就等於那項分析做不了）
REQUIRED = ("t_mono", "seq", "issued_w", "issued_v", "hist", "obs_time_rem",
            "decoded_w", "jump_guard_state", "speed_rate", "mode")


class ChainTraceFieldsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(pathlib.Path(self.tmp.name) / "ct.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, **fields) -> dict:
        tr = ChainTracer(self.path, enabled=True)
        s = tr.begin()
        for k, v in fields.items():
            setattr(s, k, v)
        tr.commit(s)
        tr.close()
        return json.loads(pathlib.Path(self.path).read_text(encoding="utf-8").strip())

    def test_required_fields_present(self):
        d = self._write(obs_time_rem=0.5, issued_w=-0.6, hist=[1.0, -0.5, 0.8, -0.5])
        for k in REQUIRED:
            self.assertIn(k, d, f"chain_trace 少了 {k}，該項診斷做不了")

    def test_obs_time_rem_round_trips(self):
        """45s 分母下 t=30s → 1−30/45 = 0.3333（用 60 會是 0.5，兩者可分辨）。"""
        d = self._write(obs_time_rem=1.0 - 30.0 / 45.0)
        self.assertAlmostEqual(d["obs_time_rem"], 0.33333333, delta=1e-6)

    def test_obs_time_rem_zero_is_recorded_not_dropped(self):
        """超過 horizon 夾成 0.0 必須記成 0.0，不能變 None —— 0 才是要看的訊號。"""
        d = self._write(obs_time_rem=0.0)
        self.assertIsNotNone(d["obs_time_rem"])
        self.assertEqual(d["obs_time_rem"], 0.0)

    def test_unset_obs_time_rem_is_null(self):
        """沒填（例如 79D 模型或推論還沒跑）→ null，不要偽造 0。"""
        self.assertIsNone(self._write()["obs_time_rem"])

    def test_nan_becomes_null_not_invalid_json(self):
        d = self._write(obs_time_rem=float("nan"), issued_w=float("inf"))
        self.assertIsNone(d["obs_time_rem"])
        self.assertIsNone(d["issued_w"])


if __name__ == "__main__":
    unittest.main()
