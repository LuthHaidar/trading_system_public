import numpy as np
import pandas as pd

from backtesting.metrics import PerformanceMetrics


def test_regime_performance_structure_and_days_sum():
    idx = pd.date_range('2021-01-01', periods=120, freq='B')
    bull = np.linspace(100, 130, 40)
    neutral = np.linspace(130, 132, 40)
    bear = np.linspace(132, 110, 40)
    equity = pd.Series(np.concatenate([bull, neutral, bear]), index=idx)
    regimes = pd.Series(['bull'] * 40 + ['neutral'] * 40 + ['bear'] * 40, index=idx)

    perf = PerformanceMetrics.regime_performance(equity, regimes)

    assert set(perf.keys()) == {'bull', 'neutral', 'bear'}
    assert sum(v['days'] for v in perf.values()) == len(equity)
    for row in perf.values():
        for key in ['cagr', 'volatility', 'sharpe', 'max_drawdown', 'days', 'fraction']:
            assert key in row


def test_bull_cagr_higher_than_bear_for_trending_series():
    idx = pd.date_range('2022-01-01', periods=80, freq='B')
    equity = pd.Series(np.concatenate([np.linspace(100, 120, 40), np.linspace(120, 95, 40)]), index=idx)
    regimes = pd.Series(['bull'] * 40 + ['bear'] * 40, index=idx)
    perf = PerformanceMetrics.regime_performance(equity, regimes)
    assert perf['bull']['cagr'] > perf['bear']['cagr']
