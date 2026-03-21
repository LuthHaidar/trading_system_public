import pandas as pd

from backtesting.metrics import PerformanceMetrics


def test_drawdown_events_two_recovered_events():
    values = [100, 100, 90, 80, 85, 100, 100, 95, 90, 100, 105]
    idx = pd.date_range('2020-01-01', periods=len(values), freq='D')
    equity = pd.Series(values, index=idx)

    events = PerformanceMetrics.drawdown_events(equity, min_drawdown=0.02)
    assert len(events) == 2

    first = events[0]
    assert first['start_date'] == idx[2]
    assert first['trough_date'] == idx[3]
    assert first['recovery_date'] == idx[5]

    second = events[1]
    assert second['start_date'] == idx[7]
    assert second['trough_date'] == idx[8]
    assert second['recovery_date'] == idx[9]


def test_drawdown_events_unrecovered_and_filter():
    values = [100, 100, 99, 98, 97, 96]
    idx = pd.date_range('2020-02-01', periods=len(values), freq='D')
    equity = pd.Series(values, index=idx)

    events = PerformanceMetrics.drawdown_events(equity, min_drawdown=0.02)
    assert len(events) == 1
    assert events[0]['recovery_date'] is None

    shallow = PerformanceMetrics.drawdown_events(equity, min_drawdown=0.2)
    assert shallow == []
