import unittest

import pandas as pd

from backtesting.metrics import PerformanceMetrics
from strategies.mean_reversion import RSIMeanReversionStrategy


class StrategyAndMetricsCleanupTests(unittest.TestCase):
    @staticmethod
    def _manual_wilder_rsi(prices: pd.Series, period: int) -> float:
        delta = prices.diff().dropna()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)

        avg_gain = gains.iloc[:period].mean()
        avg_loss = losses.iloc[:period].mean()
        for idx in range(period, len(delta)):
            avg_gain = ((avg_gain * (period - 1)) + gains.iloc[idx]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses.iloc[idx]) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        if avg_gain == 0:
            return 0.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def test_rsi_uses_wilder_smoothing_not_simple_average(self):
        prices = pd.Series(
            [
                44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
                45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
                46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21,
            ],
            dtype=float,
        )
        period = 14
        strategy = RSIMeanReversionStrategy({})
        rsi = strategy.calculate_rsi(prices, period)

        expected_wilder = self._manual_wilder_rsi(prices, period)

        delta = prices.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        simple_rsi = 100 - (100 / (1 + (gains.tail(period).mean() / losses.tail(period).mean())))

        self.assertAlmostEqual(rsi, expected_wilder, places=8)
        self.assertGreater(abs(rsi - simple_rsi), 0.1)

    def test_drawdown_duration_and_time_to_recovery_are_consistent(self):
        equity = pd.Series(
            [100.0, 120.0, 90.0, 130.0, 80.0, 140.0],
            index=pd.to_datetime([
                '2024-01-01',
                '2024-01-02',
                '2024-01-03',
                '2024-01-04',
                '2024-01-05',
                '2024-01-06',
            ]),
        )
        drawdown = PerformanceMetrics.calculate_drawdown_series(equity)
        dd_metrics = PerformanceMetrics.calculate_drawdown_durations(drawdown)
        recovery_metrics = PerformanceMetrics.calculate_time_to_recovery(equity)

        self.assertEqual(
            dd_metrics['max_drawdown_duration_days'],
            recovery_metrics['max_time_to_recovery_days'],
        )
        self.assertAlmostEqual(
            dd_metrics['avg_drawdown_duration_days'],
            recovery_metrics['avg_time_to_recovery_days'],
            places=8,
        )


if __name__ == '__main__':
    unittest.main()
