import unittest

import pandas as pd

from strategies.momentum import MomentumStrategy


class MomentumStrategyWeightingTests(unittest.TestCase):
    def _build_data(self) -> dict:
        idx = pd.bdate_range('2024-01-01', periods=260)

        def _series(start: float, step: float) -> pd.DataFrame:
            close = [start + i * step for i in range(len(idx))]
            return pd.DataFrame({'Close': close}, index=idx)

        return {
            'AAA': _series(100.0, 0.50),
            'BBB': _series(100.0, 0.30),
            'CCC': _series(100.0, 0.10),
        }

    def test_equal_weight_sums_to_one_and_equal(self):
        strategy = MomentumStrategy(
            {
                'lookback': 126,
                'skip_recent': 21,
                'n_positions': 2,
                'weight_method': 'equal',
                'min_momentum': 0.0,
                'rebalance_frequency': 1,
                'rebalance_threshold': 0.0,
            }
        )
        data = self._build_data()
        date = next(iter(data.values())).index[-1]

        weights = strategy.generate_signals(date, data, current_positions={})

        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=8)
        vals = list(weights.values())
        self.assertAlmostEqual(vals[0], vals[1], places=8)

    def test_proportional_weights_follow_momentum_order(self):
        strategy = MomentumStrategy(
            {
                'lookback': 126,
                'skip_recent': 21,
                'n_positions': 3,
                'weight_method': 'proportional',
                'min_momentum': 0.0,
                'rebalance_frequency': 1,
                'rebalance_threshold': 0.0,
            }
        )
        data = self._build_data()
        date = next(iter(data.values())).index[-1]

        weights = strategy.generate_signals(date, data, current_positions={})

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=8)
        self.assertGreater(weights['AAA'], weights['BBB'])
        self.assertGreater(weights['BBB'], weights['CCC'])

    def test_proportional_with_mixed_signs_shifts_to_positive(self):
        strategy = MomentumStrategy(
            {
                'lookback': 126,
                'skip_recent': 21,
                'n_positions': 3,
                'weight_method': 'proportional',
                'min_momentum': -1.0,
                'rebalance_frequency': 1,
                'rebalance_threshold': 0.0,
            }
        )
        idx = pd.bdate_range('2024-01-01', periods=260)
        data = {
            'POS': pd.DataFrame({'Close': [100 + i * 0.4 for i in range(len(idx))]}, index=idx),
            'FLAT': pd.DataFrame({'Close': [100 for _ in range(len(idx))]}, index=idx),
            'NEG': pd.DataFrame({'Close': [100 - i * 0.2 for i in range(len(idx))]}, index=idx),
        }
        date = idx[-1]

        weights = strategy.generate_signals(date, data, current_positions={})

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=8)
        for w in weights.values():
            self.assertGreater(w, 0.0)

    def test_min_momentum_filters_lower_scores(self):
        strategy = MomentumStrategy(
            {
                'lookback': 126,
                'skip_recent': 21,
                'n_positions': 3,
                'weight_method': 'equal',
                'min_momentum': 0.20,
                'rebalance_frequency': 1,
                'rebalance_threshold': 0.0,
            }
        )

        idx = pd.bdate_range('2024-01-01', periods=260)
        strong = pd.DataFrame({'Close': [100 + i * 0.8 for i in range(len(idx))]}, index=idx)
        weak = pd.DataFrame({'Close': [100 + i * 0.02 for i in range(len(idx))]}, index=idx)
        flat = pd.DataFrame({'Close': [100 for _ in range(len(idx))]}, index=idx)
        data = {'STRONG': strong, 'WEAK': weak, 'FLAT': flat}

        weights = strategy.generate_signals(idx[-1], data, current_positions={})

        self.assertIn('STRONG', weights)
        self.assertNotIn('WEAK', weights)
        self.assertNotIn('FLAT', weights)


if __name__ == '__main__':
    unittest.main()
