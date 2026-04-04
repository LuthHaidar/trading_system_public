import unittest

import numpy as np
import pandas as pd

from risk.position_sizer import RiskConstraints


class RiskConstraintsUnitTests(unittest.TestCase):
    def test_max_position_size_clamps_weight(self):
        rc = RiskConstraints({'max_position_size': 0.4, 'min_position_size': 0.0, 'min_cash_reserve': 0.0})
        constrained = rc.apply_constraints({'AAA': 0.8, 'BBB': 0.2})
        self.assertLessEqual(constrained['AAA'], 0.4 + 1e-12)

    def test_min_position_size_removes_small_weights(self):
        rc = RiskConstraints({'max_position_size': 1.0, 'min_position_size': 0.1, 'min_cash_reserve': 0.0})
        constrained = rc.apply_constraints({'AAA': 0.05, 'BBB': 0.25})
        self.assertNotIn('AAA', constrained)
        self.assertIn('BBB', constrained)

    def test_max_positions_drops_lowest_weight_assets(self):
        rc = RiskConstraints({'max_position_size': 1.0, 'min_position_size': 0.0, 'max_positions': 2, 'min_cash_reserve': 0.0})
        constrained = rc.apply_constraints({'AAA': 0.5, 'BBB': 0.3, 'CCC': 0.2})
        self.assertEqual(set(constrained.keys()), {'AAA', 'BBB'})

    def test_min_cash_reserve_reduces_total_exposure(self):
        rc = RiskConstraints({'max_position_size': 1.0, 'min_position_size': 0.0, 'min_cash_reserve': 0.1})
        constrained = rc.apply_constraints({'AAA': 0.7, 'BBB': 0.3})
        self.assertLessEqual(sum(constrained.values()), 0.9 + 1e-12)

    def test_correlation_limit_reduces_lower_weight_in_pair(self):
        dates = pd.date_range('2024-01-01', periods=80, freq='D')
        base = np.linspace(100, 120, len(dates))
        # nearly perfectly correlated by construction
        data = {
            'AAA': pd.DataFrame({'Close': base}, index=dates),
            'BBB': pd.DataFrame({'Close': base * 1.001}, index=dates),
        }
        rc = RiskConstraints(
            {
                'max_position_size': 1.0,
                'min_position_size': 0.0,
                'max_correlation': 0.5,
                'min_cash_reserve': 0.0,
            }
        )
        constrained = rc.apply_constraints({'AAA': 0.6, 'BBB': 0.4}, data=data)
        self.assertLessEqual(constrained.get('BBB', 0.0), 0.4)

    def test_constraints_applied_in_correct_order(self):
        rc = RiskConstraints(
            {
                'max_position_size': 0.5,
                'min_position_size': 0.1,
                'max_positions': 2,
                'min_cash_reserve': 0.1,
            }
        )
        constrained = rc.apply_constraints({'AAA': 0.8, 'BBB': 0.09, 'CCC': 0.5})
        self.assertLessEqual(sum(constrained.values()), 0.9 + 1e-12)
        self.assertLessEqual(len(constrained), 2)
        self.assertTrue(all(w >= 0.1 - 1e-12 for w in constrained.values()))
        self.assertTrue(all(w <= 0.5 + 1e-12 for w in constrained.values()))


if __name__ == '__main__':
    unittest.main()
