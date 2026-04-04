import unittest
import builtins
from unittest.mock import patch

import numpy as np
import pandas as pd

from risk.optimizer import PortfolioOptimizer

try:
    from sklearn.covariance import LedoitWolf  # noqa: F401
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class OptimizerRegressionTests(unittest.TestCase):
    def test_max_sharpe_handles_zero_volatility_without_nan(self):
        optimizer = PortfolioOptimizer(
            {'optimizer_min_weight': 0.0, 'optimizer_max_weight': 1.0, 'covariance_estimator': 'sample'}
        )
        mean_returns = pd.Series([0.10, 0.15], index=['AAA', 'BBB'])
        cov_matrix = pd.DataFrame(np.zeros((2, 2)), index=['AAA', 'BBB'], columns=['AAA', 'BBB'])

        weights = optimizer._max_sharpe(mean_returns, cov_matrix)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=8)

    def test_risk_parity_handles_zero_covariance_without_nan(self):
        optimizer = PortfolioOptimizer(
            {'optimizer_min_weight': 0.0, 'optimizer_max_weight': 1.0, 'covariance_estimator': 'sample'}
        )
        cov_matrix = pd.DataFrame(np.zeros((3, 3)), index=['A', 'B', 'C'], columns=['A', 'B', 'C'])

        weights = optimizer._risk_parity_optimize(cov_matrix)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=8)

    def test_min_variance_with_return_respects_configured_bounds(self):
        optimizer = PortfolioOptimizer(
            {'optimizer_min_weight': 0.0, 'optimizer_max_weight': 0.4, 'covariance_estimator': 'sample'}
        )
        mean_returns = pd.Series([0.08, 0.10, 0.12], index=['A', 'B', 'C'])
        cov_matrix = pd.DataFrame(
            np.diag([0.04, 0.05, 0.06]),
            index=['A', 'B', 'C'],
            columns=['A', 'B', 'C'],
        )

        weights = optimizer._min_variance_with_return(mean_returns, cov_matrix, target_return=0.10)
        self.assertIsNotNone(weights)
        self.assertTrue(np.all(np.asarray(weights) <= 0.400001))
        self.assertTrue(np.all(np.asarray(weights) >= -1e-9))

    def test_covariance_shrinkage_zeroes_off_diagonal_at_full_shrink(self):
        optimizer = PortfolioOptimizer(
            {
                'covariance_estimator': 'shrinkage',
                'covariance_shrinkage': 1.0,
                'covariance_jitter': 0.0,
            }
        )
        returns_df = pd.DataFrame(
            {
                'A': [0.01, 0.02, -0.01, 0.03, 0.00],
                'B': [0.01, 0.02, -0.01, 0.03, 0.00],  # perfectly collinear
            }
        )

        cov = optimizer._estimate_covariance(returns_df)
        self.assertAlmostEqual(float(cov.loc['A', 'B']), 0.0, places=10)
        self.assertGreater(float(cov.loc['A', 'A']), 0.0)
        self.assertGreater(float(cov.loc['B', 'B']), 0.0)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn unavailable in test environment")
    def test_ledoit_wolf_covariance_is_positive_definite(self):
        optimizer = PortfolioOptimizer(
            {
                'covariance_estimator': 'ledoit_wolf',
                'covariance_jitter': 0.0,
            }
        )
        returns_df = pd.DataFrame(
            {
                'A': [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.015, 0.012],
                'B': [-0.005, 0.01, 0.02, -0.01, 0.015, -0.004, 0.006, -0.002],
                'C': [0.02, 0.01, -0.01, 0.005, -0.02, 0.018, -0.009, 0.011],
            }
        )
        cov = optimizer._estimate_covariance(returns_df)
        eigvals = np.linalg.eigvals(cov.values)
        self.assertTrue(np.all(eigvals > 0))

    def test_ledoit_wolf_import_error_is_raised_at_startup(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith('sklearn'):
                raise ImportError('simulated missing sklearn')
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=guarded_import):
            with self.assertRaises(ImportError):
                PortfolioOptimizer({'covariance_estimator': 'ledoit_wolf'})

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn unavailable in test environment")
    def test_prepare_returns_drops_misaligned_nan_rows_for_ledoit_wolf(self):
        optimizer = PortfolioOptimizer({'covariance_estimator': 'ledoit_wolf'})
        dates_a = pd.date_range('2020-01-01', periods=400, freq='D')
        dates_b = pd.date_range('2020-01-05', periods=400, freq='D')
        data = {
            'AAA': pd.DataFrame({'Close': np.linspace(100, 140, len(dates_a))}, index=dates_a),
            'BBB': pd.DataFrame({'Close': np.linspace(50, 90, len(dates_b))}, index=dates_b),
        }
        signals = {'AAA': 1.0, 'BBB': 1.0}

        returns_df, valid_tickers = optimizer._prepare_returns(signals, data)

        self.assertEqual(valid_tickers, ['AAA', 'BBB'])
        self.assertFalse(returns_df.isna().any().any())
        cov = optimizer._estimate_covariance(returns_df)
        self.assertFalse(np.isnan(cov.values).any())


    def test_covariance_exponential_is_positive_semidefinite(self):
        optimizer = PortfolioOptimizer(
            {
                'covariance_estimator': 'exponential',
                'covariance_ew_halflife': 20,
                'covariance_jitter': 0.0,
            }
        )
        rng = np.random.default_rng(7)
        returns_df = pd.DataFrame(rng.normal(0.0, 0.01, size=(300, 4)), columns=list('ABCD'))
        cov = optimizer._estimate_covariance(returns_df)
        eigvals = np.linalg.eigvalsh(cov.values)
        self.assertTrue(np.all(eigvals >= -1e-10))

    def test_covariance_exponential_recent_bias(self):
        n = 300
        low_var = np.full(n // 2, 0.001)
        high_var = np.array([0.001, -0.001] * (n // 4))
        series = np.concatenate([low_var, high_var])
        returns_df = pd.DataFrame({'A': series, 'B': series * 0.5})

        sample = PortfolioOptimizer({'covariance_estimator': 'sample', 'covariance_jitter': 0.0})._estimate_covariance(returns_df)
        ew = PortfolioOptimizer(
            {'covariance_estimator': 'exponential', 'covariance_ew_halflife': 20, 'covariance_jitter': 0.0}
        )._estimate_covariance(returns_df)
        self.assertGreater(float(ew.loc['A', 'A']), float(sample.loc['A', 'A']))

    def test_covariance_exponential_shrinkage_full_shrink(self):
        optimizer = PortfolioOptimizer(
            {
                'covariance_estimator': 'exponential_shrinkage',
                'covariance_shrinkage': 1.0,
                'covariance_jitter': 0.0,
            }
        )
        returns_df = pd.DataFrame({'A': [0.01, 0.02, -0.01, 0.03], 'B': [0.01, 0.02, -0.01, 0.03]})
        cov = optimizer._estimate_covariance(returns_df)
        self.assertAlmostEqual(float(cov.loc['A', 'B']), 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
