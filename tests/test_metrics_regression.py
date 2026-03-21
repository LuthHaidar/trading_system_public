import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

import backtesting.metrics as metrics_module
from backtesting.metrics import PerformanceMetrics
from backtesting.portfolio import Trade
from utils.currency import CurrencyConverter


class MetricsRegressionTests(unittest.TestCase):
    def test_metrics_uses_supplied_base_currency_for_fifo_matcher(self):
        captured = {}

        class DummyMatcher:
            def __init__(self, base_currency):
                captured['base_currency'] = base_currency

            def match_trades(self, trades, fx_converter):
                return []

            @staticmethod
            def summarize(matched_trades):
                return {
                    'realized_pnl': 0.0,
                    'closed_round_trips': 0,
                    'win_rate': 0.0,
                    'profit_factor': 1.0,
                }

        equity = pd.Series(
            [1000.0, 1010.0],
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )

        with patch.object(metrics_module, 'FIFOTradeMatcher', DummyMatcher):
            PerformanceMetrics.get_comprehensive_metrics(
                equity_curve=equity,
                trades=[],
                base_currency='USD',
            )

        self.assertEqual(captured.get('base_currency'), 'USD')

    def test_metrics_cost_totals_are_converted_to_base_currency(self):
        equity = pd.Series(
            [1000.0, 1000.0],
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        trades = [
            Trade(
                date=datetime(2024, 1, 2),
                ticker='AAA',
                action='BUY',
                shares=1,
                price=100.0,
                commission=1.0,
                slippage=1.0,
                fx_cost=0.5,
                value=100.0,
                currency='EUR',
            )
        ]

        def fake_convert(self, amount, from_currency, date=None):
            if from_currency == self.base_currency:
                return amount
            return amount * 2.0

        with patch.object(CurrencyConverter, 'convert_to_base', new=fake_convert):
            metrics = PerformanceMetrics.get_comprehensive_metrics(
                equity_curve=equity,
                trades=trades,
                base_currency='USD',
            )

        self.assertAlmostEqual(metrics['total_commission'], 2.0, places=8)
        self.assertAlmostEqual(metrics['total_slippage'], 2.0, places=8)
        self.assertAlmostEqual(metrics['total_fx_cost'], 0.5, places=8)
        self.assertAlmostEqual(metrics['total_cost'], 4.5, places=8)

    def test_sortino_uses_full_sample_downside_deviation(self):
        returns = pd.Series([0.01, -0.02, 0.03], dtype=float)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0)

        downside_dev = ((0.0 ** 2 + (-0.02) ** 2 + 0.0 ** 2) / 3.0) ** 0.5
        expected = (252 ** 0.5) * returns.mean() / downside_dev
        self.assertAlmostEqual(sortino, expected, places=10)

    def test_cagr_returns_nan_for_non_positive_terminal_equity_ratio(self):
        equity = pd.Series(
            [1000.0, 0.0],
            index=pd.to_datetime(['2024-01-02', '2025-01-02']),
        )
        cagr = PerformanceMetrics.calculate_cagr(equity)
        self.assertTrue(pd.isna(cagr))

    def test_mae_mfe_are_marked_unavailable_without_intratrade_path(self):
        equity = pd.Series(
            [1000.0, 1000.0],
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        metrics = PerformanceMetrics.get_comprehensive_metrics(
            equity_curve=equity,
            trades=[],
            base_currency='USD',
        )
        self.assertFalse(metrics['mae_mfe_available'])
        self.assertEqual(metrics['mae_mfe_reason'], 'intratrade_path_unavailable')
        self.assertIsNone(metrics['avg_mae_per_trade'])
        self.assertIsNone(metrics['avg_mfe_per_trade'])

    def test_recovery_time_is_anchored_to_max_drawdown_trough(self):
        equity = pd.Series(
            [100.0, 120.0, 90.0, 110.0, 130.0],
            index=pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06']),
        )
        recovery = PerformanceMetrics.calculate_time_to_recovery(equity)
        self.assertEqual(recovery['max_time_to_recovery_days'], 2)
        self.assertEqual(recovery['max_drawdown_recovery_days_from_trough'], 2)
        self.assertEqual(recovery['max_drawdown_recovery_days_from_peak'], 2)
        self.assertTrue(recovery['max_drawdown_recovered'])

    def test_comprehensive_metrics_include_rolling_sharpe_and_equity_labels(self):
        idx = pd.date_range('2024-01-02', periods=90, freq='B')
        equity = pd.Series(1000.0 + pd.Series(range(90), index=idx).astype(float), index=idx)
        metrics = PerformanceMetrics.get_comprehensive_metrics(
            equity_curve=equity,
            trades=[],
            base_currency='USD',
        )
        self.assertIn('rolling_sharpe_63', metrics)
        self.assertIn('rolling_sharpe_252', metrics)
        self.assertIn('rolling_sharpe_63_series', metrics)
        self.assertIn('rolling_sharpe_252_series', metrics)
        self.assertIn('rolling_sharpe_63_min', metrics)
        self.assertIn('rolling_sharpe_63_max', metrics)
        self.assertIn('rolling_sharpe_63_pct_positive', metrics)
        self.assertIn('initial_equity', metrics)
        self.assertIn('final_equity', metrics)




    def test_monthly_returns_fallbacks_when_me_alias_is_unavailable(self):
        idx = pd.date_range('2024-01-01', periods=90, freq='B')
        equity = pd.Series(1000.0 + pd.Series(range(90), index=idx).astype(float), index=idx)

        original_resample = pd.Series.resample

        def guarded_resample(self, rule, *args, **kwargs):
            if isinstance(rule, str) and rule == 'ME':
                raise ValueError('Invalid frequency: ME')
            return original_resample(self, rule, *args, **kwargs)

        with patch.object(pd.Series, 'resample', new=guarded_resample):
            monthly = PerformanceMetrics.calculate_monthly_returns(equity)

        self.assertIsInstance(monthly, pd.Series)

    def test_annual_returns_fallbacks_when_ye_alias_is_unavailable(self):
        idx = pd.date_range('2022-01-01', periods=500, freq='B')
        equity = pd.Series(1000.0 + pd.Series(range(500), index=idx).astype(float), index=idx)

        original_resample = pd.Series.resample

        def guarded_resample(self, rule, *args, **kwargs):
            if isinstance(rule, str) and rule == 'YE':
                raise ValueError('Invalid frequency: YE')
            return original_resample(self, rule, *args, **kwargs)

        with patch.object(pd.Series, 'resample', new=guarded_resample):
            annual = PerformanceMetrics.calculate_annual_returns(equity)

        self.assertIsInstance(annual, pd.Series)

    def test_comprehensive_metrics_include_periodic_return_series(self):
        idx = pd.date_range('2023-01-02', periods=400, freq='B')
        equity = pd.Series(1000.0 + pd.Series(range(400), index=idx).astype(float), index=idx)
        metrics = PerformanceMetrics.get_comprehensive_metrics(
            equity_curve=equity,
            trades=[],
            base_currency='USD',
        )

        self.assertIn('monthly_returns', metrics)
        self.assertIn('annual_returns', metrics)
        self.assertIsInstance(metrics['monthly_returns'], pd.Series)
        self.assertIsInstance(metrics['annual_returns'], pd.Series)


if __name__ == '__main__':
    unittest.main()
