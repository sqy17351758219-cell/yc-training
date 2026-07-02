import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit.lambda_ctrl import LambdaController


class TestLambdaController(unittest.TestCase):
    def test_lam1_gives_half_ratio(self):
        c = LambdaController(lam=1.0)
        self.assertAlmostEqual(c.benign_ratio(), 0.5)

    def test_high_orr_pushes_more_benign(self):
        c = LambdaController(tau=0.04, eta=0.5, lam=1.0, ema=0.0)  # ema=0 -> no smoothing
        r0 = c.benign_ratio()
        for _ in range(20):
            c.update(0.30)  # way above tau
        self.assertGreater(c.benign_ratio(), r0)  # more benign states
        self.assertGreater(c.lam, 1.0)

    def test_low_orr_shrinks_benign(self):
        c = LambdaController(tau=0.04, eta=0.5, lam=1.0, ema=0.0)
        for _ in range(20):
            c.update(0.0)  # below tau
        self.assertLess(c.benign_ratio(), 0.5)
        self.assertLess(c.lam, 1.0)

    def test_ratio_clamped(self):
        c = LambdaController(tau=0.04, eta=5.0, lam=1.0, ema=0.0, pb_lo=0.2, pb_hi=0.8)
        for _ in range(100):
            c.update(1.0)
        self.assertLessEqual(c.benign_ratio(), 0.8)
        for _ in range(200):
            c.update(0.0)
        self.assertGreaterEqual(c.benign_ratio(), 0.2)

    def test_lam_clamped(self):
        c = LambdaController(eta=100.0, lam=1.0, lam_min=0.05, lam_max=20.0, ema=0.0)
        for _ in range(50):
            c.update(1.0)
        self.assertLessEqual(c.lam, 20.0)

    def test_split_counts_sum(self):
        c = LambdaController(lam=1.0)
        h, b = c.split_counts(512)
        self.assertEqual(h + b, 512)
        self.assertEqual(b, 256)

    def test_invalid_orr_raises(self):
        c = LambdaController()
        with self.assertRaises(ValueError):
            c.update(1.5)

    def test_history_recorded(self):
        c = LambdaController()
        c.update(0.1)
        c.update(0.1)
        self.assertEqual(len(c.history), 2)


if __name__ == "__main__":
    unittest.main()
