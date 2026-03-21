import unittest

import numpy as np
import pandas as pd

from research.efficient_frontier import calculate_efficient_frontier
from risk.optimizer import PortfolioOptimizer


class EfficientFrontierTests(unittest.TestCase):
    def _build_data(self):
        idx = pd.date_range('2022-01-01', periods=320, freq='D')
        rng = np.random.default_rng(123)

        # Asset A: lower volatility trend
        rets_a = rng.normal(0.0005, 0.008, len(idx))
        # Asset B: higher volatility trend
        rets_b = rng.normal(0.0007, 0.016, len(idx))
        # Asset C: medium vol / lower return
        rets_c = rng.normal(0.0003, 0.010, len(idx))

        close_a = 100 * np.cumprod(1 + rets_a)
        close_b = 90 * np.cumprod(1 + rets_b)
        close_c = 80 * np.cumprod(1 + rets_c)

        return {
            'A': pd.DataFrame({'Close': close_a}, index=idx),
            'B': pd.DataFrame({'Close': close_b}, index=idx),
            'C': pd.DataFrame({'Close': close_c}, index=idx),
        }

    def test_calculate_efficient_frontier_smoke(self):
        data = self._build_data()
        frontier = calculate_efficient_frontier(data, config={'optimizer_min_weight': 0.0, 'optimizer_max_weight': 1.0}, n_points=25)

        self.assertFalse(frontier.empty)
        self.assertEqual(list(frontier.columns), ['return', 'volatility', 'sharpe'])
        self.assertTrue(np.isfinite(frontier[['return', 'volatility', 'sharpe']].to_numpy()).all())
        self.assertTrue((frontier['volatility'] >= 0).all())

    def test_min_variance_portfolio_has_lowest_volatility_vs_individual_assets(self):
        data = self._build_data()
        optimizer = PortfolioOptimizer({'optimizer_min_weight': 0.0, 'optimizer_max_weight': 1.0})
        returns_df, _ = optimizer._prepare_returns({'A': 1.0, 'B': 1.0, 'C': 1.0}, data)
        cov_matrix = optimizer._estimate_covariance(returns_df)

        min_var_weights = optimizer._min_variance(cov_matrix)
        portfolio_vol = float(np.sqrt(np.dot(min_var_weights, np.dot(cov_matrix, min_var_weights))))

        individual_vols = np.sqrt(np.diag(cov_matrix.to_numpy()))
        self.assertLessEqual(portfolio_vol, float(np.min(individual_vols)) + 1e-10)

    def test_min_variance_respects_long_only_bounds(self):
        data = self._build_data()
        optimizer = PortfolioOptimizer({'optimizer_min_weight': 0.0, 'optimizer_max_weight': 1.0})
        returns_df, _ = optimizer._prepare_returns({'A': 1.0, 'B': 1.0, 'C': 1.0}, data)
        cov_matrix = optimizer._estimate_covariance(returns_df)

        weights = optimizer._min_variance(cov_matrix)

        self.assertTrue((weights >= -1e-10).all())
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=8)


if __name__ == '__main__':
    unittest.main()
