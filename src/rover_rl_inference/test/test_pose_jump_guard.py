"""PoseJumpGuard 行為測試（Mandatory test #4 的離線部分）。

主案例直接用實車失效數值：0.05 s 內 map_yaw 跳 97.4°。
"""

import math
import unittest

from rover_rl_inference.pose_jump_guard import (
    OdomDeadReckoner, PoseJumpGuard, STATE_OK, STATE_REJECTED,
    STATE_RECOVERING, STATE_SOFT,
)


class PoseJumpGuardTest(unittest.TestCase):
    def setUp(self):
        self.g = PoseJumpGuard(v_max=1.0, w_max=1.2, margin=1.5,
                               recover_samples=3)

    def test_first_sample_always_accepted(self):
        r = self.g.check(0.0, 0.0, 0.0, 100.0)
        self.assertTrue(r.ok)
        self.assertEqual(r.state, STATE_OK)

    def test_normal_motion_passes(self):
        """以 v_max/2、ω_max/2 前進 10 拍，全部必須放行。"""
        t, x, yaw = 100.0, 0.0, 0.0
        self.g.check(x, 0.0, yaw, t)
        for _ in range(10):
            t += 0.2
            x += 0.5 * 0.2
            yaw += 0.6 * 0.2
            r = self.g.check(x, 0.0, yaw, t)
            self.assertTrue(r.ok, r.reason)

    def test_real_failure_97deg_in_50ms_is_rejected(self):
        """實車失效：0.05 s 內 yaw 跳 97.4°。物理上限 3.4°，超標 28 倍。"""
        self.g.check(0.0, 0.0, 0.0, 100.0)
        r = self.g.check(0.0, 0.0, math.radians(97.4), 100.05)
        self.assertFalse(r.ok, "97.4°/0.05s 必須被拒絕")
        self.assertEqual(r.state, STATE_REJECTED)
        self.assertIn("角度跳變", r.reason)
        self.assertAlmostEqual(math.degrees(r.dyaw), 97.4, delta=0.1)

    def test_position_teleport_is_rejected(self):
        self.g.check(0.0, 0.0, 0.0, 100.0)
        r = self.g.check(5.0, 0.0, 0.0, 100.2)   # 0.2s 移動 5m，上限 0.3m
        self.assertFalse(r.ok)
        self.assertIn("位移跳變", r.reason)

    def test_recovery_requires_consecutive_good_samples(self):
        """拒絕後不可下一拍就放行 —— 必須連續 recover_samples 拍合理。"""
        self.g.check(0.0, 0.0, 0.0, 100.0)
        self.assertFalse(self.g.check(0.0, 0.0, math.radians(97.4), 100.05).ok)

        t, yaw = 100.05, math.radians(97.4)
        states = []
        for _ in range(3):
            t += 0.2
            yaw += 0.1 * 0.2
            r = self.g.check(0.0, 0.0, yaw, t)
            states.append((r.state, r.ok))
        self.assertEqual([s for s, _ in states],
                         [STATE_RECOVERING, STATE_RECOVERING, STATE_OK])
        self.assertEqual([o for _, o in states], [False, False, True])

    def test_angle_wrap_is_not_a_jump(self):
        """179° → -179° 實際只差 2°，不得誤判成 358° 跳變。"""
        self.g.check(0.0, 0.0, math.radians(179.0), 100.0)
        r = self.g.check(0.0, 0.0, math.radians(-179.0), 100.2)
        self.assertTrue(r.ok, f"±π 邊界被誤判：{r.reason}")
        self.assertAlmostEqual(math.degrees(r.dyaw), 2.0, delta=0.01)

    def test_long_gap_resets_instead_of_rejecting(self):
        """長時間沒資料（estop/剛啟動）後恢復，不得誤判為跳變。"""
        self.g.check(0.0, 0.0, 0.0, 100.0)
        r = self.g.check(3.0, 3.0, 2.0, 105.0)   # 隔 5 秒
        self.assertTrue(r.ok)
        self.assertEqual(r.reason, "gap_reset")

    def test_boundary_exactly_at_limit_passes(self):
        """恰好等於上限應放行（否則正常全速行駛會被連續誤判）。"""
        self.g.check(0.0, 0.0, 0.0, 100.0)
        dt = 0.2
        r = self.g.check(1.0 * dt * 1.5, 0.0, 0.0, 100.0 + dt)
        self.assertTrue(r.ok, r.reason)

    def test_reject_count_accumulates(self):
        self.g.check(0.0, 0.0, 0.0, 100.0)
        for i in range(3):
            self.g.check(0.0, 0.0, math.radians(90.0 * (i + 1)), 100.0 + 0.05 * (i + 1))
        self.assertGreaterEqual(self.g.reject_count, 1)

    def test_reset_clears_reference(self):
        self.g.check(0.0, 0.0, 0.0, 100.0)
        self.g.reset("mode change")
        r = self.g.check(99.0, 99.0, 3.0, 100.1)   # 巨大跳變但基準已清
        self.assertTrue(r.ok, "reset 後第一拍應無條件接受")


class PoseJumpGuardGradingTest(unittest.TestCase):
    """分級：NDT 常態噪聲走 soft（不停車），災難跳變才 fail-closed。

    實車數據（2026-08-21，7 段共 190 s）：12 次 guard 觸發的超標倍率
    11 次落在 1.01~1.32，只有 1 次是 2.2（NDT 連續 6 幀不收斂後的補跳）。
    門檻 2.0 正好把「該停」與「不該停」分開。
    """

    def setUp(self):
        self.g = PoseJumpGuard(v_max=1.0, w_max=1.2, margin=1.5,
                               recover_samples=3, hard_ratio=2.0,
                               soft_max_consecutive=10)

    def _pos_jump(self, dpos, dt=0.2, t0=100.0):
        self.g.check(0.0, 0.0, 0.0, t0)
        return self.g.check(dpos, 0.0, 0.0, t0 + dt)

    def test_real_ndt_noise_is_soft_not_stop(self):
        """實車那 11 次（超標 1.01~1.32 倍）必須走 soft，不得硬停。"""
        for dpos in (0.301, 0.307, 0.313, 0.341, 0.357, 0.381, 0.396):
            g = PoseJumpGuard(v_max=1.0, w_max=1.2, margin=1.5,
                              hard_ratio=2.0)
            g.check(0.0, 0.0, 0.0, 100.0)
            r = g.check(dpos, 0.0, 0.0, 100.2)     # 上限 0.300m
            self.assertEqual(r.state, STATE_SOFT, f"{dpos}m 應為 soft")
            self.assertTrue(r.use_extrapolation)
            self.assertFalse(r.ok, "soft 的 ok 必須是 False（呼叫端得換位姿）")
            self.assertLess(r.ratio, 2.0)

    def test_real_catastrophic_jump_still_hard_stops(self):
        """0.662m（超標 2.2 倍）與 97.4°（28 倍）都必須維持 fail-closed。"""
        r = self._pos_jump(0.662)
        self.assertEqual(r.state, STATE_REJECTED)
        self.assertFalse(r.use_extrapolation)
        self.assertGreaterEqual(r.ratio, 2.0)

        g = PoseJumpGuard(hard_ratio=2.0)
        g.check(0.0, 0.0, 0.0, 100.0)
        r = g.check(0.0, 0.0, math.radians(97.4), 100.05)
        self.assertEqual(r.state, STATE_REJECTED)

    def test_soft_does_not_poison_reference(self):
        """soft 那拍的髒位姿不得成為下一拍基準，否則跳變被默默吃下。

        第三拍 0.70 m 相對【真基準 (0,0)】超標（dt=0.4 → 上限 0.60 m）應判 soft；
        若基準被污染成 0.35，位移只剩 0.35 m 會被誤判成正常。這個數值能區分兩者。
        """
        self.g.check(0.0, 0.0, 0.0, 100.0)
        self.g.check(0.35, 0.0, 0.0, 100.2)          # soft，基準應仍是 (0,0)
        r = self.g.check(0.70, 0.0, 0.0, 100.4)
        self.assertEqual(r.state, STATE_SOFT,
                         "基準若被髒值污染，這拍會誤判成正常")

    def test_set_reference_updates_basis(self):
        """呼叫端算出遞推位姿後 set_reference，下一拍以它為準。"""
        self.g.check(0.0, 0.0, 0.0, 100.0)
        self.g.check(0.35, 0.0, 0.0, 100.2)
        self.g.set_reference(0.16, 0.0, 0.0, 100.2)  # odom 遞推值
        r = self.g.check(0.32, 0.0, 0.0, 100.4)      # 相對 0.16 只走 0.16m
        self.assertTrue(r.ok, r.reason)

    def test_soft_streak_escalates_to_hard(self):
        """連續 soft 超過上限 → 升級硬停（定位真的掛了不能一直遞推）。"""
        g = PoseJumpGuard(v_max=1.0, w_max=1.2, margin=1.5,
                          hard_ratio=2.0, soft_max_consecutive=3)
        g.check(0.0, 0.0, 0.0, 100.0)
        t = 100.0
        states = []
        for _ in range(4):
            t += 0.2
            r = g.check(0.35, 0.0, 0.0, t)     # 相對固定基準，每拍都 soft
            g.set_reference(0.0, 0.0, 0.0, t)  # 遞推值刻意維持原地
            states.append(r.state)
        self.assertEqual(states,
                         [STATE_SOFT, STATE_SOFT, STATE_SOFT, STATE_REJECTED])

    def test_soft_needs_no_recovery_period(self):
        """soft 沒停過車，下一拍正常就該直接放行，不必等 recover_samples。"""
        self.g.check(0.0, 0.0, 0.0, 100.0)
        self.assertEqual(self.g.check(0.35, 0.0, 0.0, 100.2).state, STATE_SOFT)
        self.g.set_reference(0.16, 0.0, 0.0, 100.2)
        r = self.g.check(0.30, 0.0, 0.0, 100.4)
        self.assertTrue(r.ok)
        self.assertEqual(r.state, STATE_OK)

    def test_hard_ratio_one_restores_legacy_behavior(self):
        """hard_ratio=1.0 = 關掉分級，任何超標都硬停（回到舊行為）。"""
        g = PoseJumpGuard(v_max=1.0, w_max=1.2, margin=1.5, hard_ratio=1.0)
        g.check(0.0, 0.0, 0.0, 100.0)
        r = g.check(0.301, 0.0, 0.0, 100.2)
        self.assertEqual(r.state, STATE_REJECTED)

    def test_normal_motion_reports_ratio_below_one(self):
        self.g.check(0.0, 0.0, 0.0, 100.0)
        r = self.g.check(0.1, 0.0, 0.0, 100.2)
        self.assertTrue(r.ok)
        self.assertLess(r.ratio, 1.0)

    def test_soft_and_hard_counters_are_separate(self):
        self.g.check(0.0, 0.0, 0.0, 100.0)
        self.g.check(0.35, 0.0, 0.0, 100.2)          # soft
        self.g.set_reference(0.16, 0.0, 0.0, 100.2)
        self.g.check(5.0, 0.0, 0.0, 100.4)           # hard
        self.assertEqual(self.g.soft_count, 1)
        self.assertEqual(self.g.reject_count, 1)


class OdomDeadReckonerTest(unittest.TestCase):
    """odom 遞推 = 暫時凍結 map→odom offset。"""

    def test_no_reference_returns_none(self):
        self.assertIsNone(OdomDeadReckoner().extrapolate(1.0, 2.0, 0.5))

    def test_straight_line_translation(self):
        r = OdomDeadReckoner()
        r.update(10.0, 5.0, 0.0, 0.0, 0.0, 0.0)      # map 與 odom 同向
        x, y, yaw = r.extrapolate(0.2, 0.0, 0.0)
        self.assertAlmostEqual(x, 10.2)
        self.assertAlmostEqual(y, 5.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_rotated_frames_translation_is_rotated(self):
        """map 與 odom 差 90°：odom 往 +x 走，map 上應往 +y 走。"""
        r = OdomDeadReckoner()
        r.update(0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0)
        x, y, yaw = r.extrapolate(0.5, 0.0, 0.0)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.5, places=6)
        self.assertAlmostEqual(yaw, math.pi / 2, places=6)

    def test_yaw_increment_is_added(self):
        r = OdomDeadReckoner()
        r.update(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        _, _, yaw = r.extrapolate(0.0, 0.0, 0.3)
        self.assertAlmostEqual(yaw, 1.3, places=6)

    def test_yaw_wraps_at_pi(self):
        r = OdomDeadReckoner()
        r.update(0.0, 0.0, math.radians(179.0), 0.0, 0.0, 0.0)
        _, _, yaw = r.extrapolate(0.0, 0.0, math.radians(3.0))
        self.assertAlmostEqual(math.degrees(yaw), -178.0, places=4)

    def test_reset_clears_reference(self):
        r = OdomDeadReckoner()
        r.update(1.0, 2.0, 0.0, 0.0, 0.0, 0.0)
        self.assertTrue(r.has_reference)
        r.reset()
        self.assertFalse(r.has_reference)
        self.assertIsNone(r.extrapolate(1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
