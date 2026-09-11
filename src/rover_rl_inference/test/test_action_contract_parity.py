"""車端 decoder 對訓練端凍結 fixture 的逐項 parity 測試。

fixture: docs/freeze/sa1_action_contract_v1.json（訓練 repo 產生）
    zero_state_all_361_actions  — 19×19 全枚舉，零初始狀態
    stateful_sequences          — reverse_saturation / forward_saturation /
                                  positive_omega_slew / omega_sign_flip

守的是實車已發生的失效：decoder 產出 rl_v=-0.557，輸出層才 clamp 成 -0.2，
於是 83D history 記到的值與實際送出的命令是兩層不同的東西。
fixture 的 semantics 明定：
    reverse_bound_location = "inside decoder"
    issued_history = "[linear_accel, omega] after decode/slew, before actuator delay"

⚠️ 消費 fixture 前必須驗它記錄的來源 hash（見 test_fixture_source_hashes_recorded）。
本測試在車端執行時無法存取訓練 repo，故只斷言 fixture 自帶 hash 欄位完整；
跨機驗證由訓練端 CI 負責，車端以 fixture 的 sha256 為信任錨。
"""

import json
import os
import pathlib
import unittest

import numpy as np

from rover_rl_inference.action_decoder import ActionParams, decode_logits_to_cmd

_FIXTURE = pathlib.Path(
    os.environ.get(
        "ACTION_CONTRACT_FIXTURE",
        "/home/aa/rover_rl/docs/freeze/sa1_action_contract_v1.json",
    )
)

#: 允差。decoder 全是 float64 基本運算，訓練端是 float32 tensor；
#: 1e-6 足以抓出任何語義差異，又不會被浮點尾數誤判。
TOL = 1e-6


def _logits_for(linear_index: int, angular_index: int, num_bins: int = 19) -> np.ndarray:
    """造出讓 argmax 落在指定 index 的 38 維 logits。"""
    logits = np.full(2 * num_bins, -10.0, dtype=np.float32)
    logits[linear_index] = 10.0
    logits[num_bins + angular_index] = 10.0
    return logits


class ActionContractParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _FIXTURE.is_file():
            raise unittest.SkipTest(f"fixture 不存在：{_FIXTURE}")
        cls.fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        p = cls.fx["parameters"]
        cls.params = ActionParams(
            num_bins=int(p["num_bins"]),
            max_linear_velocity=float(p["max_linear_velocity_mps"]),
            max_linear_accel=float(p["max_linear_accel_mps2"]),
            max_angular_velocity_action=float(p["max_angular_velocity_rad_s"]),
            dt=float(p["control_dt_s"]),
            max_angular_accel=float(p["max_angular_accel_rad_s2"]),
            reverse_velocity_scale=float(p["reverse_velocity_scale"]),
        )

    # ── fixture 完整性 ────────────────────────────────────────────────
    def test_fixture_source_hashes_recorded(self):
        """fixture 必須自帶來源檔與 sha256，否則無法判斷它對應哪一版訓練碼。"""
        src = self.fx["source"]
        for key in ("action_term", "action_term_sha256",
                    "decoder_helper", "decoder_helper_sha256"):
            self.assertIn(key, src, f"fixture source 缺 {key}")
            self.assertTrue(src[key], f"fixture source.{key} 是空的")
        for key in ("action_term_sha256", "decoder_helper_sha256"):
            self.assertEqual(len(src[key]), 64, f"{key} 不是 sha256")

    def test_fixture_semantics_match_this_implementation(self):
        """語義契約：倒車界在 decoder 內、history 是 post-decode/post-slew。"""
        sem = self.fx["semantics"]
        self.assertEqual(sem["reverse_bound_location"], "inside decoder")
        self.assertEqual(sem["actuator_delay_location"], "after decode/slew/history")
        self.assertIn("after decode/slew", sem["issued_history"])

    def test_parameters_match_training(self):
        p = self.fx["parameters"]
        self.assertEqual(p["reverse_velocity_scale"], 0.2)
        self.assertEqual(p["max_linear_velocity_mps"], 1.0)
        self.assertEqual(p["max_angular_velocity_rad_s"], 1.2)
        self.assertEqual(p["max_linear_accel_mps2"], 0.5)
        self.assertEqual(p["max_angular_accel_rad_s2"], 3.0)
        self.assertEqual(p["control_dt_s"], 0.2)
        self.assertEqual(p["num_bins"], 19)

    # ── 361 組全枚舉 ─────────────────────────────────────────────────
    def test_all_361_actions_match_fixture(self):
        rows = self.fx["zero_state_all_361_actions"]
        self.assertEqual(len(rows), 361, "fixture 不是完整 19×19")
        bad = []
        for r in rows:
            v, w, a = decode_logits_to_cmd(
                _logits_for(r["linear_index"], r["angular_index"],
                            self.params.num_bins),
                current_linear_vel=float(r["current_velocity_mps"]),
                params=self.params,
                deterministic=True,
                current_angular_vel=float(r["current_omega_rad_s"]),
            )
            for name, got, want in (
                ("v", v, r["issued_velocity_mps"]),
                ("w", w, r["issued_omega_rad_s"]),
                ("a", a, r["issued_linear_accel_mps2"]),
            ):
                if abs(got - want) > TOL:
                    bad.append(
                        f"idx=({r['linear_index']},{r['angular_index']}) "
                        f"{name}: got={got:.9f} want={want:.9f}"
                    )
        self.assertFalse(
            bad,
            f"{len(bad)}/{3*len(rows)} 項不符（前 8 筆）:\n" + "\n".join(bad[:8]),
        )

    # ── 有狀態序列 ───────────────────────────────────────────────────
    def test_stateful_sequences_match_fixture(self):
        for seq in self.fx["stateful_sequences"]:
            with self.subTest(sequence=seq["name"]):
                bad = []
                for st in seq["steps"]:
                    v, w, a = decode_logits_to_cmd(
                        _logits_for(st["linear_index"], st["angular_index"],
                                    self.params.num_bins),
                        current_linear_vel=float(st["current_velocity_mps"]),
                        params=self.params,
                        deterministic=True,
                        current_angular_vel=float(st["current_omega_rad_s"]),
                    )
                    for name, got, want in (
                        ("v", v, st["issued_velocity_mps"]),
                        ("w", w, st["issued_omega_rad_s"]),
                        ("a", a, st["issued_linear_accel_mps2"]),
                    ):
                        if abs(got - want) > TOL:
                            bad.append(
                                f"step={st['step']} {name}: "
                                f"got={got:.9f} want={want:.9f}"
                            )
                self.assertFalse(
                    bad, f"[{seq['name']}] 不符:\n" + "\n".join(bad[:8])
                )

    def test_issued_history_matches_fixture(self):
        """history 內容必須是 [issued_linear_accel, issued_omega]（post-decode/post-slew）。

        這條直接守實車失效：history 曾記到 decoder 未受界的 -0.557，
        而實際送出的是輸出層 clamp 後的 -0.2。

        ⚠️ 排序慣例：fixture 的 `history_after_step` 是 **oldest-first**
        （index 0 = a_{t-2}，index 1 = 本步 issued）。不要與 obs 佈局混淆，
        後者是 newest-first（見 test_obs_flatten_is_newest_first）。
        """
        for seq in self.fx["stateful_sequences"]:
            with self.subTest(sequence=seq["name"]):
                hist = [[0.0, 0.0], [0.0, 0.0]]   # oldest-first
                for st in seq["steps"]:
                    _v, w, a = decode_logits_to_cmd(
                        _logits_for(st["linear_index"], st["angular_index"],
                                    self.params.num_bins),
                        current_linear_vel=float(st["current_velocity_mps"]),
                        params=self.params,
                        deterministic=True,
                        current_angular_vel=float(st["current_omega_rad_s"]),
                    )
                    hist = [hist[1], [a, w]]      # 左移，新值放尾端
                    want = st["history_after_step"]
                    for i in range(2):
                        for j in range(2):
                            self.assertAlmostEqual(
                                hist[i][j], want[i][j], delta=TOL,
                                msg=(f"[{seq['name']}] step={st['step']} "
                                     f"history[{i}][{j}]"),
                            )

    def test_obs_flatten_is_newest_first_opposite_to_fixture(self):
        """obs[79:83] 與 fixture history 的排序**相反**，轉換必須顯式且被測到。

        fixture `history_after_step` : oldest-first  [a_{t-2}, a_{t-1}]
        obs_spec `[79:83]`           : newest-first  [a_{t-1}, ω_{t-1}, a_{t-2}, ω_{t-2}]

        車端 `_act_hist` 用 deque.appendleft → index 0 = 最新 → flatten 後
        自然是 newest-first，與 obs_spec 一致。但只要有人照 fixture 的排序
        去實作 obs，就會把 a_{t-1} 與 a_{t-2} 對調 —— 而且**兩者量綱相同、
        數值合法**，不會有任何錯誤訊息，只會讓 policy 讀到時序顛倒的歷史。
        """
        import collections

        seq = next(s for s in self.fx["stateful_sequences"]
                   if s["name"] == "reverse_saturation")
        dq = collections.deque([[0.0, 0.0], [0.0, 0.0]], maxlen=2)  # newest-first
        fixture_hist = [[0.0, 0.0], [0.0, 0.0]]                     # oldest-first

        for st in seq["steps"]:
            _v, w, a = decode_logits_to_cmd(
                _logits_for(st["linear_index"], st["angular_index"],
                            self.params.num_bins),
                current_linear_vel=float(st["current_velocity_mps"]),
                params=self.params, deterministic=True,
                current_angular_vel=float(st["current_omega_rad_s"]),
            )
            dq.appendleft([a, w])                     # 車端 _push_act_hist 的語義
            fixture_hist = [fixture_hist[1], [a, w]]  # fixture 的語義

            # 兩者必須恰為反序 —— 這就是唯一合法的轉換關係
            self.assertEqual(
                [list(x) for x in dq],
                list(reversed(fixture_hist)),
                f"step={st['step']}：obs 排序與 fixture 不是反序，"
                "代表某一端的時序被弄反了",
            )
            # 且 obs 的第一組必須是「本步 issued」
            self.assertAlmostEqual(dq[0][0], a, delta=TOL)
            self.assertAlmostEqual(dq[0][1], w, delta=TOL)

    # ── 邊界（Mandatory test #2）─────────────────────────────────────
    def test_reverse_bound_is_enforced_inside_decoder(self):
        """連續全力倒車不得越過 -0.2，且飽和後 issued accel 必須歸零。"""
        v = 0.0
        for _ in range(20):
            v, _w, a = decode_logits_to_cmd(
                _logits_for(0, 9, self.params.num_bins),
                current_linear_vel=v, params=self.params,
                deterministic=True, current_angular_vel=0.0,
            )
            self.assertGreaterEqual(
                v, -0.2 - TOL, "倒車越過訓練上限 -0.2（界不在 decoder 內？）"
            )
        self.assertAlmostEqual(v, -0.2, delta=TOL)
        self.assertAlmostEqual(a, 0.0, delta=TOL,
                               msg="飽和後 issued accel 應為 0（動態界收斂）")

    def test_forward_bound_is_unaffected_by_reverse_scale(self):
        v = 0.0
        for _ in range(20):
            v, _w, _a = decode_logits_to_cmd(
                _logits_for(18, 9, self.params.num_bins),
                current_linear_vel=v, params=self.params,
                deterministic=True, current_angular_vel=0.0,
            )
        self.assertAlmostEqual(v, 1.0, delta=TOL, msg="正向上限不應受倒車縮放影響")

    def test_angular_slew_limits_step_change(self):
        max_dw = self.params.max_angular_accel * self.params.dt
        w = 0.0
        for _ in range(10):
            _v, w_new, _a = decode_logits_to_cmd(
                _logits_for(9, 18, self.params.num_bins),
                current_linear_vel=0.0, params=self.params,
                deterministic=True, current_angular_vel=w,
            )
            self.assertLessEqual(abs(w_new - w), max_dw + TOL, "α slew 未生效")
            w = w_new
        self.assertAlmostEqual(w, 1.2, delta=TOL)

    def test_default_reverse_scale_preserves_legacy_symmetric_behaviour(self):
        """預設 1.0 必須維持舊行為（±v_max），否則既有 79D 部署會被改到。"""
        legacy = ActionParams(
            num_bins=19, max_linear_velocity=1.0, max_linear_accel=0.5,
            max_angular_velocity_action=1.2, dt=0.2, max_angular_accel=3.0,
        )
        self.assertEqual(legacy.reverse_velocity_scale, 1.0)
        v = 0.0
        for _ in range(20):
            v, _w, _a = decode_logits_to_cmd(
                _logits_for(0, 9), current_linear_vel=v, params=legacy,
                deterministic=True, current_angular_vel=0.0,
            )
        self.assertAlmostEqual(v, -1.0, delta=TOL)


if __name__ == "__main__":
    unittest.main()
