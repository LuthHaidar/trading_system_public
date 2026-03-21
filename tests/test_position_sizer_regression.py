import unittest

import pandas as pd

from risk.position_sizer import PositionSizer


class PositionSizerRegressionTests(unittest.TestCase):
    def test_kelly_uses_absolute_fractions_with_exposure_cap(self):
        idx = pd.date_range('2024-01-01', periods=5, freq='D')
        # Returns: +2%, +2%, +2%, -1%
        close_a = pd.Series([100.0, 102.0, 104.04, 106.1208, 105.059592], index=idx)
        close_b = pd.Series([50.0, 51.0, 52.02, 53.0604, 52.529796], index=idx)
        data = {
            'A': pd.DataFrame({'Close': close_a}, index=idx),
            'B': pd.DataFrame({'Close': close_b}, index=idx),
        }

        sizer = PositionSizer(
            method='kelly',
            config={
                'kelly_window': 4,
                'kelly_fraction': 0.5,
                'kelly_max_total_exposure': 0.60,
                'min_cash_reserve': 0.0,
            },
        )

        weights = sizer.size_positions({'A': 1.0, 'B': 1.0}, data, current_equity=1000.0)
        self.assertTrue(weights)
        self.assertAlmostEqual(sum(weights.values()), 0.60, places=8)
        self.assertAlmostEqual(weights['A'], 0.30, places=6)
        self.assertAlmostEqual(weights['B'], 0.30, places=6)

    def test_inverse_volatility_method_matches_legacy_risk_parity_alias(self):
        idx = pd.date_range('2024-01-01', periods=80, freq='D')
        # A has smoother path (lower vol), B has bumpier path (higher vol).
        close_a = pd.Series([100.0 + i * 0.2 for i in range(len(idx))], index=idx)
        close_b = pd.Series(
            [100.0 + i * 0.2 + ((-1) ** i) * 0.6 for i in range(len(idx))],
            index=idx,
        )
        data = {
            'A': pd.DataFrame({'Close': close_a}, index=idx),
            'B': pd.DataFrame({'Close': close_b}, index=idx),
        }
        signals = {'A': 1.0, 'B': 1.0}

        rp = PositionSizer(method='risk_parity', config={'volatility_window': 60})
        inv = PositionSizer(method='inverse_volatility', config={'volatility_window': 60})

        rp_weights = rp.size_positions(signals, data, current_equity=1000.0)
        inv_weights = inv.size_positions(signals, data, current_equity=1000.0)

        self.assertAlmostEqual(rp_weights['A'], inv_weights['A'], places=8)
        self.assertAlmostEqual(rp_weights['B'], inv_weights['B'], places=8)


if __name__ == '__main__':
    unittest.main()
