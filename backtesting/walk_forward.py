from __future__ import annotations

from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.validation import ValidationSuite, WindowSplit
from backtesting.engine import run_backtest_from_config


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_days: int,
    oos_days: int,
    step_days: int,
    anchored: bool = False,
) -> List[WindowSplit]:
    """Generate walk-forward train/OOS splits."""
    return ValidationSuite.time_series_cv(
        index=index,
        train_size=train_days,
        test_size=oos_days,
        step_size=step_days,
        anchored=anchored,
    )


def _chain_oos_segments(segments: List[pd.Series]) -> pd.Series:
    """Scale each OOS segment to start at previous segment's ending equity."""
    chained: List[pd.Series] = []
    prev_end: Optional[float] = None

    for segment in segments:
        if segment is None or len(segment) == 0:
            continue
        seg = segment.astype(float).copy()
        if prev_end is not None and float(seg.iloc[0]) != 0.0:
            seg = seg * (prev_end / float(seg.iloc[0]))
        prev_end = float(seg.iloc[-1])
        chained.append(seg)

    if not chained:
        return pd.Series(dtype=float)
    return pd.concat(chained).sort_index()


def run_walk_forward(
    index: pd.DatetimeIndex,
    *,
    config_path: str,
    strategy_name: str,
    tickers: List[str],
    strategies_config_path: str,
    train_days: int = 252,
    oos_days: int = 63,
    step_days: int = 63,
    anchored: bool = False,
    config_override: Optional[Dict] = None,
    strategy_override: Optional[Dict] = None,
    backtest_runner: Optional[Callable[..., Dict]] = None,
) -> Dict:
    """Run fixed-config walk-forward analysis and aggregate OOS statistics."""
    runner = backtest_runner or run_backtest_from_config
    splits = walk_forward_splits(index, train_days, oos_days, step_days, anchored=anchored)

    split_results: List[Dict] = []
    oos_segments: List[pd.Series] = []
    boundaries: List[pd.Timestamp] = []

    for split in splits:
        train_results = runner(
            config_path=config_path,
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=split.train_start.strftime('%Y-%m-%d'),
            end_date=split.train_end.strftime('%Y-%m-%d'),
            strategies_config_path=strategies_config_path,
            config_override=config_override,
            strategy_override=strategy_override,
        )
        oos_results = runner(
            config_path=config_path,
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=split.test_start.strftime('%Y-%m-%d'),
            end_date=split.test_end.strftime('%Y-%m-%d'),
            strategies_config_path=strategies_config_path,
            config_override=config_override,
            strategy_override=strategy_override,
        )

        split_results.append(
            {
                'split': split,
                'train_metrics': train_results.get('metrics', {}),
                'oos_metrics': oos_results.get('metrics', {}),
                'oos_equity_curve': oos_results.get('equity_curve', pd.Series(dtype=float)),
            }
        )
        oos_segments.append(oos_results.get('equity_curve', pd.Series(dtype=float)))
        boundaries.extend([split.train_end, split.test_start, split.test_end])

    oos_equity = _chain_oos_segments(oos_segments)
    sharpe_values = [
        float(item['oos_metrics'].get('sharpe_ratio', 0.0))
        for item in split_results
        if item['oos_metrics']
    ]
    cagr_values = [
        float(item['oos_metrics'].get('cagr', 0.0))
        for item in split_results
        if item['oos_metrics']
    ]

    aggregate = {
        'mean_oos_sharpe': float(np.mean(sharpe_values)) if sharpe_values else 0.0,
        'mean_oos_cagr': float(np.mean(cagr_values)) if cagr_values else 0.0,
        'positive_oos_sharpe_fraction': (
            float(np.mean([s > 0 for s in sharpe_values])) if sharpe_values else 0.0
        ),
    }

    return {
        'splits': splits,
        'split_results': split_results,
        'oos_equity_curve': oos_equity,
        'split_boundaries': sorted(set(boundaries)),
        'aggregate_oos_metrics': aggregate,
    }


def plot_walk_forward(
    full_equity: pd.Series,
    oos_equity: pd.Series,
    split_boundaries: List[pd.Timestamp],
    output_path: str,
) -> None:
    """Plot full equity and chained OOS equity with split boundaries."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if full_equity is not None and len(full_equity) > 0:
        ax.plot(full_equity.index, full_equity.values, label='Full backtest', linewidth=1.5)
    if oos_equity is not None and len(oos_equity) > 0:
        ax.plot(oos_equity.index, oos_equity.values, label='Walk-forward OOS (chained)', linewidth=2.0)

    for boundary in split_boundaries or []:
        ax.axvline(boundary, color='grey', alpha=0.2, linewidth=0.8)

    ax.set_title('Walk-Forward Analysis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
