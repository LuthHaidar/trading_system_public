import unittest

import numpy as np
import pandas as pd

from backtesting.validation import ValidationSuite


class ValidationRegressionTests(unittest.TestCase):
    def test_monte_carlo_block_bootstrap_returns_percentiles(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.012, -0.01, 0.02, -0.004])
        result = ValidationSuite.monte_carlo_simulation(
            returns,
            n_sims=50,
            horizon_days=30,
            block_size=5,
        )

        self.assertIn('p05_return', result)
        self.assertIn('p50_return', result)
        self.assertIn('p95_return', result)
        self.assertTrue(np.isfinite(result['p05_return']))
        self.assertTrue(np.isfinite(result['p50_return']))
        self.assertTrue(np.isfinite(result['p95_return']))

    def test_monte_carlo_seed_makes_results_deterministic(self):
        returns = pd.Series([0.01, -0.015, 0.008, 0.004, -0.006, 0.012, -0.003, 0.007])
        first = ValidationSuite.monte_carlo_simulation(
            returns,
            n_sims=100,
            horizon_days=40,
            block_size=4,
            random_state=123,
        )
        second = ValidationSuite.monte_carlo_simulation(
            returns,
            n_sims=100,
            horizon_days=40,
            block_size=4,
            random_state=123,
        )
        self.assertEqual(first, second)

    def test_benchmark_comparison_reports_active_return(self):
        strategy = pd.Series([0.01, 0.01, 0.01, 0.01])
        benchmark = pd.Series([0.02, 0.02, 0.02, 0.02])
        result = ValidationSuite.benchmark_comparison(strategy, benchmark, risk_free_rate=0.0)
        self.assertIn('active_return', result)
        self.assertLess(result['active_return'], 0.0)


if __name__ == '__main__':
    unittest.main()
