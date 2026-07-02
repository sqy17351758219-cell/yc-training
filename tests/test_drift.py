import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit import drift


class TestDrift(unittest.TestCase):
    def test_seed_pool_sizes(self):
        # blueprint: 17 harmful + 11 benign
        self.assertEqual(len(drift.HARMFUL_SEED_DRIFTS), 17)
        self.assertEqual(len(drift.BENIGN_SEED_DRIFTS), 11)

    def test_default_drifts_are_copies(self):
        a = drift.default_drifts()
        a[0].trials = 99
        b = drift.default_drifts()
        self.assertEqual(b[0].trials, 0)  # no state leak

    def test_split_steps(self):
        self.assertEqual(drift.split_steps("A. B! C?"), ["A.", "B!", "C?"])
        self.assertEqual(drift.split_steps(""), [])

    def test_splice_structure(self):
        s = drift.splice("PROMPT", "S1. S2. S3. S4.", "DRIFT", 0.5)
        self.assertIn("PROMPT", s)
        self.assertIn(drift.THINK_OPEN, s)
        self.assertTrue(s.endswith("DRIFT"))
        self.assertIn("S1.", s)  # kept prefix
        self.assertNotIn("S4.", s)  # dropped tail at z=0.5

    def test_splice_depth_bounds(self):
        with self.assertRaises(ValueError):
            drift.splice("p", "cot", "d", 0.0)
        with self.assertRaises(ValueError):
            drift.splice("p", "cot", "d", 1.5)

    def test_candidate_states_count_and_kind(self):
        drifts = drift.default_drifts("harmful")
        cands = drift.candidate_states(
            "p1", "PROMPT", "A. B. C. D.", "harmful", drifts,
            max_candidates=10, rng=random.Random(0),
        )
        self.assertLessEqual(len(cands), 10)
        self.assertTrue(all(c.kind == "harmful" for c in cands))
        self.assertTrue(all(c.state_text.startswith("PROMPT") for c in cands))

    def test_candidate_uses_sampler(self):
        drifts = drift.default_drifts("benign")
        called = {"n": 0}

        def sampler(pool, need):
            called["n"] += 1
            return pool[:need]

        drift.candidate_states(
            "p", "P", "A. B.", "benign", drifts, max_candidates=4,
            rng=random.Random(0), drift_sampler=sampler,
        )
        self.assertEqual(called["n"], 1)

    def test_drift_register_stats(self):
        d = drift.Drift("x", "t", "harmful")
        d.register(True)
        d.register(False)
        self.assertEqual(d.trials, 2)
        self.assertEqual(d.breaches, 1)
        self.assertAlmostEqual(d.breach_rate(), 0.5)


if __name__ == "__main__":
    unittest.main()
