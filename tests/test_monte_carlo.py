import unittest

import numpy as np
import pandas as pd

from backtesting.validation import ValidationSuite


class MonteCarloEnhancementTests(unittest.TestCase):
    def test_paths_shape_and_start_normalized(self):
        returns = pd.Series([0.01, -0.005, 0.008, 0.002, -0.003, 0.006, 0.001])
        n_sims = 25
        horizon = 30
        result = ValidationSuite.monte_carlo_simulation(
            returns=returns,
            n_sims=n_sims,
            horizon_days=horizon,
            block_size=3,
            random_state=7,
            return_paths=True,
        )

        paths = result['paths']
        self.assertEqual(paths.shape, (n_sims, horizon + 1))
        self.assertTrue(np.allclose(paths[:, 0], 1.0))

    def test_path_stats_probability_and_drawdown_range(self):
        returns = pd.Series([0.003, -0.002, 0.004, -0.001, 0.002, -0.003, 0.005])
        mc = ValidationSuite.monte_carlo_simulation(
            returns=returns,
            n_sims=40,
            horizon_days=20,
            block_size=2,
            random_state=9,
            return_paths=True,
        )
        stats = ValidationSuite.monte_carlo_path_stats(mc['paths'])

        self.assertGreaterEqual(stats['p_ruin'], 0.0)
        self.assertLessEqual(stats['p_ruin'], 1.0)
        self.assertGreaterEqual(stats['avg_drawdown'], 0.0)

    def test_risk_of_ruin_is_one_for_large_negative_returns(self):
        returns = pd.Series([-0.5] * 30)
        mc = ValidationSuite.monte_carlo_simulation(
            returns=returns,
            n_sims=20,
            horizon_days=10,
            block_size=1,
            random_state=42,
            return_paths=True,
        )
        stats = ValidationSuite.monte_carlo_path_stats(mc['paths'], ruin_threshold=0.5)
        self.assertEqual(stats['p_ruin'], 1.0)

    def test_risk_of_ruin_is_zero_for_positive_returns(self):
        returns = pd.Series([0.01] * 30)
        mc = ValidationSuite.monte_carlo_simulation(
            returns=returns,
            n_sims=20,
            horizon_days=10,
            block_size=1,
            random_state=42,
            return_paths=True,
        )
        stats = ValidationSuite.monte_carlo_path_stats(mc['paths'], ruin_threshold=0.5)
        self.assertEqual(stats['p_ruin'], 0.0)


if __name__ == '__main__':
    unittest.main()
