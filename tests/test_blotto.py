"""Scoring checks for Colonel Blotto."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blotto


class TestBlotto(unittest.TestCase):
    def test_game1_score_against_itself_is_7_5(self) -> None:
        alloc = np.array([20, 20, 20, 20, 20], dtype=np.int16)
        score = blotto.game1_score(alloc, alloc.reshape(1, -1))
        self.assertAlmostEqual(score, 7.5)

    def test_allocations_sum_to_100(self) -> None:
        rng = np.random.default_rng(0)
        sample = blotto.random_allocations(rng, blotto.INITIAL_POOL)
        self.assertEqual(sample.shape, (blotto.INITIAL_POOL, blotto.N_CASTLES))
        self.assertTrue(np.all(sample >= 0))
        self.assertTrue(np.all(sample.sum(axis=1) == blotto.TOTAL_TROOPS))

    def test_game2_symmetric_pool_averages_3_75(self) -> None:
        identical = np.array([20, 20, 20, 20, 20], dtype=np.int16)
        pool = np.tile(identical, (6, 1))
        score = blotto.game2_score(
            identical,
            pool,
            n_groups=30,
            rng=np.random.default_rng(0),
        )
        self.assertAlmostEqual(score, 3.75)


if __name__ == "__main__":
    unittest.main()
