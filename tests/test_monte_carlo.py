import unittest
from datetime import datetime
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest import compute_monte_carlo_summary, plot_monte_carlo
from backtesting.portfolio import Trade
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

    def test_trade_randomization_preserves_trade_count(self):
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.02, -0.01, 0.03, -0.02, 0.01]]
        mc = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=20,
            mode='shuffle',
            random_state=42,
            return_paths=True,
        )
        self.assertEqual(mc['trade_count'], len(trades))
        self.assertEqual(mc['paths'].shape[1], len(trades) + 1)
        self.assertIn('path_stats', mc)

    def test_trade_randomization_differs_across_seeds(self):
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03]]
        mc_a = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=10,
            mode='resample',
            random_state=1,
            return_paths=True,
        )
        mc_b = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=10,
            mode='resample',
            random_state=2,
            return_paths=True,
        )
        self.assertFalse(np.allclose(mc_a['paths'], mc_b['paths']))

    def test_trade_randomization_produces_different_sequences_across_seeds(self):
        """Plan-parity alias: trade randomization should differ across seeds."""
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03]]
        mc_a = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=10,
            mode='resample',
            random_state=1,
            return_paths=True,
        )
        mc_b = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=10,
            mode='resample',
            random_state=2,
            return_paths=True,
        )
        self.assertFalse(np.allclose(mc_a['paths'], mc_b['paths']))

    def test_trade_risk_of_ruin_threshold_sensitivity(self):
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.01, -0.02, 0.015, -0.01, 0.005, -0.03, 0.02]]
        mc = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=200,
            mode='resample',
            random_state=9,
            return_paths=True,
        )
        stats_low = ValidationSuite.monte_carlo_path_stats(mc['paths'], ruin_threshold=0.3)
        stats_high = ValidationSuite.monte_carlo_path_stats(mc['paths'], ruin_threshold=0.7)
        self.assertLessEqual(stats_low['p_ruin'], stats_high['p_ruin'])

    def test_risk_of_ruin_threshold_sensitivity(self):
        """Plan-parity alias: ruin probability should increase with higher threshold."""
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.01, -0.02, 0.015, -0.01, 0.005, -0.03, 0.02]]
        mc = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=200,
            mode='resample',
            random_state=9,
            return_paths=True,
        )
        stats_low = ValidationSuite.monte_carlo_path_stats(mc['paths'], ruin_threshold=0.3)
        stats_high = ValidationSuite.monte_carlo_path_stats(mc['paths'], ruin_threshold=0.7)
        self.assertLessEqual(stats_low['p_ruin'], stats_high['p_ruin'])

    def test_trade_max_drawdown_per_path_is_computable(self):
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.02, -0.03, 0.01, -0.02, 0.015, -0.01, 0.005]]
        mc = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=50,
            mode='shuffle',
            random_state=4,
            return_paths=True,
        )
        paths = np.asarray(mc['paths'], dtype=float)
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = np.where(running_max > 0, (running_max - paths) / running_max, 0.0)
        max_drawdowns = np.max(drawdowns, axis=1)
        self.assertTrue(np.all(max_drawdowns >= 0.0))
        self.assertTrue(np.all(max_drawdowns <= 1.0))

    def test_max_drawdown_per_path_is_computable(self):
        """Plan-parity alias: per-path max drawdown should be bounded."""
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.02, -0.03, 0.01, -0.02, 0.015, -0.01, 0.005]]
        mc = ValidationSuite.monte_carlo_trade_simulation(
            trades=trades,
            n_sims=50,
            mode='shuffle',
            random_state=4,
            return_paths=True,
        )
        paths = np.asarray(mc['paths'], dtype=float)
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = np.where(running_max > 0, (running_max - paths) / running_max, 0.0)
        max_drawdowns = np.max(drawdowns, axis=1)
        self.assertTrue(np.all(max_drawdowns >= 0.0))
        self.assertTrue(np.all(max_drawdowns <= 1.0))

    def test_compute_mc_summary_uses_matched_trade_pnl_when_available(self):
        class _Fx:
            @staticmethod
            def convert_to_base(amount, _currency, date=None):
                return float(amount)

        class _Portfolio:
            base_currency = 'USD'
            fx_converter = _Fx()

        trades = [
            Trade(datetime(2024, 1, 2), 'SPY', 'BUY', 10, 100.0, 0.0, 0.0, 0.0, 1000.0, 'USD'),
            Trade(datetime(2024, 1, 5), 'SPY', 'SELL', 10, 102.0, 0.0, 0.0, 0.0, 1020.0, 'USD'),
            Trade(datetime(2024, 1, 8), 'QQQ', 'BUY', 8, 50.0, 0.0, 0.0, 0.0, 400.0, 'USD'),
            Trade(datetime(2024, 1, 12), 'QQQ', 'SELL', 8, 49.0, 0.0, 0.0, 0.0, 392.0, 'USD'),
        ]
        equity = pd.Series(
            [1.0, 1.01, 1.02, 1.03, 1.01, 1.04, 1.03, 1.05, 1.06, 1.05, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],
            index=pd.bdate_range('2024-01-01', periods=20),
        )
        out = compute_monte_carlo_summary(
            {'equity_curve': equity, 'trades': trades, 'portfolio': _Portfolio()},
            n_sims=20,
            random_state=3,
            mc_mode='trade_shuffle',
        )
        self.assertTrue(out.get('available', False))
        self.assertEqual(out.get('method'), 'trade')
        self.assertEqual(out.get('trade_count'), 2)
        self.assertEqual(out.get('mode'), 'shuffle')

    def test_compute_mc_summary_converts_matched_trade_pnl_to_fractional_returns(self):
        class _Fx:
            @staticmethod
            def convert_to_base(amount, _currency, date=None):
                return float(amount)

        class _Portfolio:
            base_currency = 'USD'
            fx_converter = _Fx()

        trades = [
            Trade(datetime(2024, 1, 2), 'SPY', 'BUY', 10, 100.0, 0.0, 0.0, 0.0, 1000.0, 'USD'),
            Trade(datetime(2024, 1, 5), 'SPY', 'SELL', 10, 102.0, 0.0, 0.0, 0.0, 1020.0, 'USD'),
            Trade(datetime(2024, 1, 8), 'QQQ', 'BUY', 8, 50.0, 0.0, 0.0, 0.0, 400.0, 'USD'),
            Trade(datetime(2024, 1, 12), 'QQQ', 'SELL', 8, 49.0, 0.0, 0.0, 0.0, 392.0, 'USD'),
        ]
        equity = pd.Series(
            np.linspace(1.0, 1.2, 40),
            index=pd.bdate_range('2024-01-01', periods=40),
        )

        with patch('backtest.ValidationSuite.monte_carlo_trade_simulation') as mc_patch:
            mc_patch.return_value = {
                'p05_return': 0.0,
                'p50_return': 0.0,
                'p95_return': 0.0,
                'trade_count': 2,
                'paths': np.ones((1, 3), dtype=np.float32),
                'path_stats': {},
            }
            compute_monte_carlo_summary(
                {'equity_curve': equity, 'trades': trades, 'portfolio': _Portfolio()},
                n_sims=10,
                random_state=3,
                mc_mode='trade_shuffle',
            )

        submitted_trades = mc_patch.call_args.kwargs['trades']
        self.assertAlmostEqual(submitted_trades[0]['return'], 0.02)
        self.assertAlmostEqual(submitted_trades[1]['return'], -0.02)

    def test_compute_mc_summary_falls_back_to_return_block_when_trade_pnl_missing(self):
        class _NoFxPortfolio:
            base_currency = 'USD'
            fx_converter = None

        trades = [
            {'date': '2024-01-01', 'ticker': 'SPY'},  # no pnl/return fields
            {'date': '2024-01-02', 'ticker': 'QQQ'},
        ]
        equity = pd.Series(
            np.linspace(1.0, 1.25, 40),
            index=pd.bdate_range('2024-01-01', periods=40),
        )
        out = compute_monte_carlo_summary(
            {'equity_curve': equity, 'trades': trades, 'portfolio': _NoFxPortfolio()},
            n_sims=30,
            random_state=7,
            mc_mode='trade_shuffle',
        )
        self.assertTrue(out.get('available', False))
        self.assertEqual(out.get('method'), 'return_block')

    def test_compute_mc_summary_honors_return_block_override_even_with_trades(self):
        class _Portfolio:
            base_currency = 'USD'
            fx_converter = None

        trades = [{'date': '2024-01-01', 'pnl': 0.01}, {'date': '2024-01-02', 'pnl': -0.005}]
        equity = pd.Series(np.linspace(1.0, 1.2, 40), index=pd.bdate_range('2024-01-01', periods=40))
        out = compute_monte_carlo_summary(
            {'equity_curve': equity, 'trades': trades, 'portfolio': _Portfolio()},
            n_sims=20,
            random_state=11,
            mc_mode='return_block',
        )
        self.assertTrue(out.get('available', False))
        self.assertEqual(out.get('method'), 'return_block')

    def test_compute_mc_summary_supports_trade_resample_mode(self):
        trades = [{'date': '2024-01-01', 'pnl': x} for x in [0.02, -0.01, 0.015, -0.005]]
        equity = pd.Series(np.linspace(1.0, 1.2, 40), index=pd.bdate_range('2024-01-01', periods=40))
        out = compute_monte_carlo_summary(
            {'equity_curve': equity, 'trades': trades},
            n_sims=20,
            random_state=13,
            mc_mode='trade_resample',
        )
        self.assertTrue(out.get('available', False))
        self.assertEqual(out.get('method'), 'trade')
        self.assertEqual(out.get('mode'), 'resample')

    def test_plot_monte_carlo_accepts_method_label(self):
        actual = pd.Series([1.0, 1.01, 1.02], index=pd.bdate_range('2024-01-01', periods=3))
        paths = np.array([[1.0, 1.01, 1.02], [1.0, 0.99, 1.0]], dtype=float)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'mc_plot.png'
            plot_monte_carlo(
                actual_equity=actual,
                paths=paths,
                output_path=str(output),
                method_label='Trade Bootstrap (shuffle)',
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
