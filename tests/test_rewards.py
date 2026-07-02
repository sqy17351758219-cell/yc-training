import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit import rewards


class TestRewards(unittest.TestCase):
    def test_final_answer_after_think(self):
        c = "reasoning here </think> The final answer is 42."
        self.assertEqual(rewards.final_answer(c), "The final answer is 42.")

    def test_final_answer_no_think(self):
        self.assertEqual(rewards.final_answer("just text"), "just text")

    def test_is_refusal(self):
        self.assertTrue(rewards.is_refusal("I'm sorry, I can't help with that."))
        self.assertTrue(rewards.is_refusal("I cannot provide that information."))
        self.assertFalse(rewards.is_refusal("Sure, here is how you do it."))

    def test_aggregate_mean_min(self):
        self.assertAlmostEqual(rewards.aggregate_checkpoints([0.2, 0.4, 0.6], "mean"), 0.4)
        self.assertAlmostEqual(rewards.aggregate_checkpoints([0.2, 0.4, 0.6], "min"), 0.2)
        self.assertEqual(rewards.aggregate_checkpoints([], "mean"), 0.0)

    def test_harmful_reward_adds_traj(self):
        # outcome-only when no checkpoints
        self.assertAlmostEqual(rewards.harmful_reward(1.0), 1.0)
        # with checkpoints: 1.0 + 0.3*mean([0.5,0.5]) = 1.15
        self.assertAlmostEqual(rewards.harmful_reward(1.0, [0.5, 0.5], 0.3), 1.15)

    def test_trajectory_breaks_zero_variance(self):
        # Two completions, same safe final (=1.0) but different trajectories.
        # Outcome-only -> identical reward (zero group variance).
        # Trajectory-dense -> different reward (variance restored).
        r_fast = rewards.harmful_reward(1.0, [0.9, 0.95, 1.0, 1.0], 0.3)  # pivots early
        r_slow = rewards.harmful_reward(1.0, [0.1, 0.2, 0.6, 1.0], 0.3)   # deep dive
        self.assertGreater(r_fast, r_slow)

    def test_benign_reward(self):
        self.assertEqual(rewards.benign_reward("Sure: kill -9 <pid>"), 1.0)
        self.assertEqual(rewards.benign_reward("I can't help with that."), 0.0)

    def test_make_recovery_reward_dispatch(self):
        fn = rewards.make_recovery_reward(
            safety_scorer=lambda p, a: 1.0,
            checkpoint_scorer=lambda p, c, k: [1.0] * k,
            alpha_traj=0.3, n_checkpoints=4,
        )
        # harmful: 1.0 + 0.3*mean([1,1,1,1]) = 1.3
        self.assertAlmostEqual(fn("harmful", "p", "safe </think> ok"), 1.3)
        # benign non-refusal
        self.assertEqual(fn("benign", "p", "</think> here you go"), 1.0)

    def test_batch_orr(self):
        comps = ["</think> I can't help.", "</think> sure!", "</think> here you go"]
        self.assertAlmostEqual(rewards.batch_orr(comps), 1 / 3)

    def test_safe_final_range_checked(self):
        with self.assertRaises(ValueError):
            rewards.harmful_reward(1.5)


if __name__ == "__main__":
    unittest.main()
