import unittest

import pandas as pd

from research.efficient_frontier import calculate_efficient_frontier


class ResearchFrontierTests(unittest.TestCase):
    def test_calculate_efficient_frontier_returns_expected_columns(self):
        idx = pd.date_range('2023-01-01', periods=300, freq='D')
        data = {
            'AAA': pd.DataFrame({'Close': [100.0 + (i * 0.2) + ((-1) ** i) * 0.5 for i in range(len(idx))]}, index=idx),
            'BBB': pd.DataFrame({'Close': [80.0 + (i * 0.1) + ((-1) ** (i + 1)) * 0.3 for i in range(len(idx))]}, index=idx),
        }

        frontier = calculate_efficient_frontier(
            data,
            config={
                'covariance_estimator': 'sample',
                'optimizer_min_weight': 0.0,
                'optimizer_max_weight': 1.0,
            },
            n_points=10,
        )
        self.assertEqual(list(frontier.columns), ['return', 'volatility', 'sharpe'])
        self.assertGreaterEqual(len(frontier), 1)


if __name__ == '__main__':
    unittest.main()
