import pandas as pd

from backtesting.walk_forward import _chain_oos_segments, run_walk_forward, walk_forward_splits


def test_walk_forward_split_count_and_non_overlap():
    index = pd.bdate_range('2020-01-01', periods=500)
    splits = walk_forward_splits(index, train_days=252, oos_days=63, step_days=63, anchored=False)

    expected = 3  # start positions: 252, 315, 378
    assert len(splits) == expected

    for left, right in zip(splits, splits[1:]):
        assert left.test_end < right.test_start


def test_walk_forward_train_windows_valid():
    index = pd.bdate_range('2021-01-01', periods=420)
    splits = walk_forward_splits(index, train_days=200, oos_days=50, step_days=25, anchored=True)
    assert splits
    for split in splits:
        assert split.train_start <= split.train_end < split.test_start <= split.test_end


def test_chain_oos_segments_is_continuous():
    idx1 = pd.bdate_range('2022-01-03', periods=5)
    idx2 = pd.bdate_range('2022-01-10', periods=5)
    seg1 = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx1)
    seg2 = pd.Series([1.0, 0.99, 1.01, 1.02, 1.03], index=idx2)

    chained = _chain_oos_segments([seg1, seg2])
    assert abs(chained.loc[idx2[0]] - chained.loc[idx1[-1]]) < 1e-12


def test_run_walk_forward_oos_chaining_with_stub_runner():
    index = pd.bdate_range('2020-01-01', periods=350)

    def stub_runner(**kwargs):
        start = pd.Timestamp(kwargs['start_date'])
        end = pd.Timestamp(kwargs['end_date'])
        idx = pd.bdate_range(start, end)
        eq = pd.Series([1.0 + (i * 0.001) for i in range(len(idx))], index=idx)
        return {
            'equity_curve': eq,
            'metrics': {'sharpe_ratio': 1.0, 'cagr': 0.10},
        }

    out = run_walk_forward(
        index=index,
        config_path='config/config.yaml',
        strategy_name='momentum',
        tickers=['SPY'],
        strategies_config_path='config/strategies.yaml',
        train_days=200,
        oos_days=40,
        step_days=40,
        backtest_runner=stub_runner,
    )

    assert out['splits']
    assert len(out['oos_equity_curve']) > 0
    assert out['aggregate_oos_metrics']['mean_oos_sharpe'] == 1.0
    assert out['aggregate_oos_metrics']['positive_oos_sharpe_fraction'] == 1.0
