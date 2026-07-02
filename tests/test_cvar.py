import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit import cvar


class TestCVaR(unittest.TestCase):
    def test_badness_orientation(self):
        # harmful: low safety -> high badness
        self.assertAlmostEqual(cvar.attacker_badness([0.1, 0.1], "harmful"), 0.9)
        # benign: high refusal -> high badness
        self.assertAlmostEqual(cvar.attacker_badness([1.0, 1.0], "benign"), 1.0)
        self.assertEqual(cvar.attacker_badness([], "harmful"), 0.0)

    def test_worst_tail_alpha_extremes(self):
        b = [0.1, 0.9, 0.5, 0.2]
        # alpha->0 keeps 1 (the worst)
        self.assertEqual(cvar.worst_tail_indices(b, 0.01), [1])
        # alpha=1 keeps all
        self.assertEqual(sorted(cvar.worst_tail_indices(b, 1.0)), [0, 1, 2, 3])

    def test_worst_tail_picks_highest(self):
        b = [0.1, 0.9, 0.5, 0.2]
        tail = cvar.worst_tail_indices(b, 0.5)  # ceil(0.5*4)=2 worst
        self.assertEqual(set(tail), {1, 2})

    def test_select_cvar_stays_in_tail(self):
        cands = list("abcd")
        b = [0.1, 0.9, 0.5, 0.2]
        rng = random.Random(0)
        picked = set()
        for _ in range(200):
            c, idx = cvar.select_cvar(cands, b, alpha=0.5, rng=rng)
            picked.add(idx)
        # only the two worst indices {1,2} may ever be selected
        self.assertTrue(picked.issubset({1, 2}))

    def test_select_cvar_favors_worst(self):
        cands = list("ab")
        b = [0.0, 1.0]
        rng = random.Random(1)
        counts = {0: 0, 1: 0}
        for _ in range(500):
            _, idx = cvar.select_cvar(cands, b, alpha=1.0, temp=0.5, rng=rng)
            counts[idx] += 1
        self.assertGreater(counts[1], counts[0])

    def test_softmax_normalizes(self):
        w = cvar.softmax_weights([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(w[2] > w[1] > w[0])

    def test_cvar_value(self):
        self.assertAlmostEqual(cvar.cvar_value([0.1, 0.9, 0.5, 0.2], 0.5), 0.7)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cvar.select_cvar(["a"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
