import tempfile
from pathlib import Path

import pandas as pd

from backtesting.walk_forward import _chain_oos_segments, plot_walk_forward, run_walk_forward, walk_forward_splits


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


def test_walk_forward_produces_expected_number_of_splits():
    index = pd.bdate_range('2020-01-01', periods=500)
    splits = walk_forward_splits(index, train_days=200, oos_days=50, step_days=50, anchored=False)
    assert len(splits) == 6


def test_walk_forward_oos_windows_are_non_overlapping():
    index = pd.bdate_range('2020-01-01', periods=500)
    splits = walk_forward_splits(index, train_days=200, oos_days=50, step_days=50, anchored=False)
    for left, right in zip(splits, splits[1:]):
        assert left.test_end < right.test_start


def test_walk_forward_train_never_overlaps_oos_in_same_split():
    index = pd.bdate_range('2020-01-01', periods=500)
    splits = walk_forward_splits(index, train_days=200, oos_days=50, step_days=50, anchored=False)
    assert all(split.train_end < split.test_start for split in splits)


def test_walk_forward_anchored_expands_train_window():
    index = pd.bdate_range('2020-01-01', periods=500)
    splits = walk_forward_splits(index, train_days=200, oos_days=50, step_days=50, anchored=True)
    assert len(splits) >= 2
    first = splits[0]
    second = splits[1]
    assert first.train_start == second.train_start
    assert second.train_end > first.train_end


def test_walk_forward_single_split_degenerate_case():
    index = pd.bdate_range('2020-01-01', periods=250)
    splits = walk_forward_splits(index, train_days=200, oos_days=50, step_days=50, anchored=False)
    assert len(splits) == 1


def test_run_walk_forward_with_sweep_returns_best_overrides_and_stability():
    index = pd.bdate_range('2020-01-01', periods=320)

    def stub_runner(**kwargs):
        start = pd.Timestamp(kwargs['start_date'])
        end = pd.Timestamp(kwargs['end_date'])
        idx = pd.bdate_range(start, end)
        eq = pd.Series([1.0 + (i * 0.001) for i in range(len(idx))], index=idx)
        override = kwargs.get('strategy_override') or {}
        lookback = int(override.get('lookback', 0))
        return {
            'equity_curve': eq,
            'metrics': {'sharpe_ratio': float(lookback), 'cagr': 0.1},
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
        sweep_definitions=['lookback=63,126'],
        optimize_metric='sharpe_ratio',
        backtest_runner=stub_runner,
    )

    assert out['best_overrides_per_fold']
    assert all(item.get('lookback') == 126 for item in out['best_overrides_per_fold'])
    assert out['parameter_stability'] == 1.0
    assert out['parameter_stability_by_key'].get('lookback') == 1.0


def test_run_walk_forward_with_sweep_reports_per_key_stability():
    index = pd.bdate_range('2020-01-01', periods=360)

    def stub_runner(**kwargs):
        start = pd.Timestamp(kwargs['start_date'])
        end = pd.Timestamp(kwargs['end_date'])
        idx = pd.bdate_range(start, end)
        eq = pd.Series([1.0 + (i * 0.001) for i in range(len(idx))], index=idx)
        override = kwargs.get('strategy_override') or {}
        lookback = int(override.get('lookback', 0))
        skip_recent = int(override.get('skip_recent', 1))
        preferred_skip = 1 if (start.day % 2 == 0) else 2
        score = lookback + (100 if skip_recent == preferred_skip else 0)
        return {
            'equity_curve': eq,
            'metrics': {'sharpe_ratio': float(score), 'cagr': 0.1},
        }

    out = run_walk_forward(
        index=index,
        config_path='config/config.yaml',
        strategy_name='momentum',
        tickers=['SPY'],
        strategies_config_path='config/strategies.yaml',
        train_days=180,
        oos_days=40,
        step_days=40,
        sweep_definitions=['lookback=63,126', 'skip_recent=1,2'],
        optimize_metric='sharpe_ratio',
        backtest_runner=stub_runner,
    )

    assert out['best_overrides_per_fold']
    assert out['parameter_stability_by_key'].get('lookback') == 1.0
    skip_stability = out['parameter_stability_by_key'].get('skip_recent')
    assert skip_stability is not None
    assert 0.0 <= skip_stability < 1.0


def test_run_walk_forward_without_sweep_preserves_base_override_per_fold():
    index = pd.bdate_range('2020-01-01', periods=320)

    def stub_runner(**kwargs):
        start = pd.Timestamp(kwargs['start_date'])
        end = pd.Timestamp(kwargs['end_date'])
        idx = pd.bdate_range(start, end)
        eq = pd.Series([1.0 + (i * 0.001) for i in range(len(idx))], index=idx)
        return {
            'equity_curve': eq,
            'metrics': {'sharpe_ratio': 1.0, 'cagr': 0.1},
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
        strategy_override={'lookback': 126, 'top_n': 5},
        backtest_runner=stub_runner,
    )

    assert out['best_overrides_per_fold']
    assert all(item == {'lookback': 126, 'top_n': 5} for item in out['best_overrides_per_fold'])


def test_plot_walk_forward_accepts_fold_annotations():
    index = pd.bdate_range('2024-01-01', periods=12)
    full_equity = pd.Series([1.0 + (i * 0.01) for i in range(len(index))], index=index)
    oos_equity = full_equity.iloc[6:]
    splits = walk_forward_splits(index, train_days=6, oos_days=3, step_days=3, anchored=False)
    assert splits
    fold_annotations = [{'split': splits[0], 'best_override': {'lookback': 63, 'top_n': 3}}]

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / 'wf_plot.png'
        plot_walk_forward(
            full_equity=full_equity,
            oos_equity=oos_equity,
            split_boundaries=[splits[0].train_end, splits[0].test_start, splits[0].test_end],
            output_path=str(output),
            fold_annotations=fold_annotations,
        )
        assert output.exists()
        assert output.stat().st_size > 0


def test_plot_walk_forward_annotations_handle_nan_equity_values():
    index = pd.bdate_range('2024-01-01', periods=12)
    full_equity = pd.Series([1.0, 1.02, float('nan'), 1.04, 1.06, 1.05, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12], index=index)
    oos_equity = pd.Series([float('nan'), 1.01, 1.02, 1.03, 1.04, 1.05], index=index[6:])
    splits = walk_forward_splits(index, train_days=6, oos_days=3, step_days=3, anchored=False)
    assert splits
    fold_annotations = [{'split': splits[0], 'best_override': {'lookback': 63}}]

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / 'wf_plot_nans.png'
        plot_walk_forward(
            full_equity=full_equity,
            oos_equity=oos_equity,
            split_boundaries=[splits[0].train_end, splits[0].test_start, splits[0].test_end],
            output_path=str(output),
            fold_annotations=fold_annotations,
        )
        assert output.exists()
        assert output.stat().st_size > 0
