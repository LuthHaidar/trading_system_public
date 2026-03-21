import unittest

import numpy as np
import pandas as pd

from risk.position_sizer import RiskConstraints


class RiskConstraintsRegressionTests(unittest.TestCase):
    def test_max_positions_trim_renormalizes_weights(self):
        rc = RiskConstraints(
            {
                'max_position_size': 1.0,
                'min_position_size': 0.0,
                'max_positions': 2,
                'min_cash_reserve': 0.0,
            }
        )

        trimmed = rc._apply_max_positions_limit({'A': 0.6, 'B': 0.3, 'C': 0.1})
        self.assertEqual(set(trimmed.keys()), {'A', 'B'})
        self.assertAlmostEqual(sum(trimmed.values()), 1.0, places=8)
        self.assertAlmostEqual(trimmed['A'], 2.0 / 3.0, places=8)
        self.assertAlmostEqual(trimmed['B'], 1.0 / 3.0, places=8)

    def test_correlation_limits_scale_with_excess_correlation(self):
        rc = RiskConstraints(
            {
                'max_position_size': 1.0,
                'min_position_size': 0.0,
                'max_positions': None,
                'min_cash_reserve': 0.0,
                'max_correlation': 0.90,
            }
        )

        idx = pd.date_range('2024-01-01', periods=80, freq='D')
        close = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
        data = {
            'A': pd.DataFrame({'Close': close}, index=idx),
            'B': pd.DataFrame({'Close': close * 1.0}, index=idx),
        }

        constrained = rc._apply_correlation_limits({'A': 0.4, 'B': 0.6}, data)
        # Perfect correlation should fully scale the smaller leg at threshold=0.90.
        self.assertAlmostEqual(constrained['A'], 0.0, places=8)
        self.assertAlmostEqual(constrained['B'], 0.6, places=8)

    def test_position_cap_counts_are_tracked_for_periodic_warnings(self):
        rc = RiskConstraints(
            {
                'max_position_size': 0.20,
                'min_position_size': 0.0,
                'max_positions': None,
                'min_cash_reserve': 0.0,
                'capping_warning_every': 2,
            }
        )

        rc._apply_position_limits({'A': 0.50})
        rc._apply_position_limits({'A': 0.40})
        self.assertEqual(rc._capping_counts.get('A'), 2)


    def test_combined_constraint_pipeline_produces_valid_portfolio(self):
        rc = RiskConstraints(
            {
                'max_position_size': 0.50,
                'min_position_size': 0.05,
                'max_positions': 3,
                'max_sector_exposure': 0.60,
                'max_correlation': 0.80,
                'min_diversification_score': 0.20,
                'diversification_scale_floor': 0.60,
                'min_cash_reserve': 0.10,
            }
        )

        idx = pd.date_range('2024-01-01', periods=120, freq='D')
        r1 = pd.Series(np.sin(np.linspace(0, 10, len(idx))) * 0.01 + 0.001, index=idx)
        r2 = r1 * 0.98 + 0.0001  # strongly correlated with r1
        r3 = pd.Series(np.cos(np.linspace(0, 12, len(idx))) * 0.008 + 0.0008, index=idx)

        close_a = 100 * (1 + r1).cumprod()
        close_b = 80 * (1 + r2).cumprod()
        close_c = 60 * (1 + r3).cumprod()

        data = {
            'A': pd.DataFrame({'Close': close_a}, index=idx),
            'B': pd.DataFrame({'Close': close_b}, index=idx),
            'C': pd.DataFrame({'Close': close_c}, index=idx),
        }
        sector_map = {'A': 'Tech', 'B': 'Tech', 'C': 'Utilities'}
        raw_weights = {'A': 0.70, 'B': 0.55, 'C': 0.12, 'D': 0.03}

        constrained = rc.apply_constraints(raw_weights, data=data, sector_map=sector_map)

        self.assertTrue(constrained)
        self.assertLessEqual(len(constrained), 3)
        self.assertTrue(all(weight >= 0 for weight in constrained.values()))
        self.assertTrue(all(weight <= 0.50 + 1e-9 for weight in constrained.values()))
        self.assertLessEqual(sum(constrained.values()), 0.90 + 1e-9)

        tech_exposure = constrained.get('A', 0.0) + constrained.get('B', 0.0)
        self.assertLessEqual(tech_exposure, 0.60 + 1e-9)

        # Correlation-limited pair should not remain at their pre-constraint capped values.
        self.assertLess(constrained.get('A', 0.0), 0.50 + 1e-9)

    def test_diversification_floor_scales_down_when_score_is_too_low(self):
        rc = RiskConstraints(
            {
                'max_position_size': 1.0,
                'min_position_size': 0.0,
                'max_positions': None,
                'min_cash_reserve': 0.0,
                'max_correlation': 1.0,
                'min_diversification_score': 0.50,
                'diversification_scale_floor': 0.50,
            }
        )

        idx = pd.date_range('2024-01-01', periods=80, freq='D')
        close = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
        data = {
            'A': pd.DataFrame({'Close': close}, index=idx),
            'B': pd.DataFrame({'Close': close * 1.0}, index=idx),  # perfect correlation
        }

        constrained = rc.apply_constraints({'A': 0.4, 'B': 0.6}, data=data)
        self.assertAlmostEqual(sum(constrained.values()), 0.5, places=8)
        self.assertAlmostEqual(constrained['A'], 0.2, places=8)
        self.assertAlmostEqual(constrained['B'], 0.3, places=8)


    def test_combined_constraint_pipeline_produces_valid_portfolio(self):
        rc = RiskConstraints(
            {
                'max_position_size': 0.60,
                'min_position_size': 0.05,
                'max_positions': 3,
                'min_cash_reserve': 0.10,
                'max_sector_exposure': 0.50,
                'max_correlation': 0.85,
                'min_diversification_score': 0.30,
                'diversification_scale_floor': 0.60,
            }
        )

        idx = pd.date_range('2024-01-01', periods=100, freq='D')
        base = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
        noisy = base * (1 + 0.01 * np.sin(np.linspace(0, 8, len(idx))))
        diversifier = pd.Series(np.linspace(90, 105, len(idx)) + np.cos(np.linspace(0, 6, len(idx))), index=idx)
        data = {
            'A': pd.DataFrame({'Close': base}, index=idx),
            'B': pd.DataFrame({'Close': noisy}, index=idx),
            'C': pd.DataFrame({'Close': diversifier}, index=idx),
            'D': pd.DataFrame({'Close': diversifier * 0.99}, index=idx),
        }
        sector_map = {'A': 'Tech', 'B': 'Tech', 'C': 'Health', 'D': 'Health'}

        constrained = rc.apply_constraints(
            {'A': 0.70, 'B': 0.20, 'C': 0.08, 'D': 0.02},
            data=data,
            sector_map=sector_map,
        )

        self.assertTrue(constrained)
        self.assertLessEqual(len(constrained), 3)
        self.assertTrue(all(weight >= 0.0 for weight in constrained.values()))
        self.assertLessEqual(sum(constrained.values()), 0.90 + 1e-9)  # respects min_cash_reserve
        self.assertTrue(set(constrained.keys()).issubset({'A', 'B', 'C', 'D'}))


if __name__ == '__main__':
    unittest.main()
