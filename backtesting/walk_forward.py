from __future__ import annotations

from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.validation import ValidationSuite, WindowSplit
from backtesting.engine import run_backtest_from_config
from backtesting.tuner import build_strategy_overrides, rank_results




def _deep_merge_dict(base: Optional[Dict], override: Optional[Dict]) -> Dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    sweep_definitions: Optional[List[str]] = None,
    optimize_metric: str = 'sharpe_ratio',
    backtest_runner: Optional[Callable[..., Dict]] = None,
) -> Dict:
    """Run walk-forward analysis with optional per-fold train-sweep selection."""
    runner = backtest_runner or run_backtest_from_config
    splits = walk_forward_splits(index, train_days, oos_days, step_days, anchored=anchored)

    split_results: List[Dict] = []
    oos_segments: List[pd.Series] = []
    boundaries: List[pd.Timestamp] = []
    best_overrides_per_fold: List[Dict] = []

    use_sweep = bool(sweep_definitions)
    candidate_overrides = build_strategy_overrides(sweep_definitions or []) if use_sweep else [{}]

    for split in splits:
        selected_override = dict(strategy_override or {})

        if use_sweep:
            candidate_rows: List[Dict] = []
            for candidate in candidate_overrides:
                merged_override = _deep_merge_dict(strategy_override, candidate)
                train_results = runner(
                    config_path=config_path,
                    strategy_name=strategy_name,
                    tickers=tickers,
                    start_date=split.train_start.strftime('%Y-%m-%d'),
                    end_date=split.train_end.strftime('%Y-%m-%d'),
                    strategies_config_path=strategies_config_path,
                    config_override=config_override,
                    strategy_override=merged_override,
                )
                candidate_rows.append(
                    {
                        'strategy_override': merged_override,
                        'metrics': train_results.get('metrics', {}),
                    }
                )
            ranked = rank_results(candidate_rows, optimize_metric, top_n=1)
            if ranked:
                selected_override = dict(ranked[0].get('strategy_override', {}) or {})

        train_results = runner(
            config_path=config_path,
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=split.train_start.strftime('%Y-%m-%d'),
            end_date=split.train_end.strftime('%Y-%m-%d'),
            strategies_config_path=strategies_config_path,
            config_override=config_override,
            strategy_override=selected_override,
        )
        oos_results = runner(
            config_path=config_path,
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=split.test_start.strftime('%Y-%m-%d'),
            end_date=split.test_end.strftime('%Y-%m-%d'),
            strategies_config_path=strategies_config_path,
            config_override=config_override,
            strategy_override=selected_override,
        )

        split_results.append(
            {
                'split': split,
                'train_metrics': train_results.get('metrics', {}),
                'oos_metrics': oos_results.get('metrics', {}),
                'oos_equity_curve': oos_results.get('equity_curve', pd.Series(dtype=float)),
                'best_override': selected_override,
            }
        )
        best_overrides_per_fold.append(dict(selected_override or {}))
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

    matches_previous = 0
    comparisons = 0
    by_key_matches: Dict[str, int] = {}
    by_key_comparisons: Dict[str, int] = {}
    for prev, cur in zip(best_overrides_per_fold, best_overrides_per_fold[1:]):
        comparisons += 1
        if prev == cur:
            matches_previous += 1
        for key in set(prev.keys()) | set(cur.keys()):
            if key in prev and key in cur:
                by_key_comparisons[key] = by_key_comparisons.get(key, 0) + 1
                if prev.get(key) == cur.get(key):
                    by_key_matches[key] = by_key_matches.get(key, 0) + 1
    parameter_stability = float(matches_previous / comparisons) if comparisons > 0 else 1.0
    parameter_stability_by_key = {
        key: (
            float(by_key_matches.get(key, 0) / by_key_comparisons[key])
            if by_key_comparisons[key] > 0
            else 1.0
        )
        for key in sorted(by_key_comparisons.keys())
    }

    return {
        'splits': splits,
        'split_results': split_results,
        'oos_equity_curve': oos_equity,
        'split_boundaries': sorted(set(boundaries)),
        'aggregate_oos_metrics': aggregate,
        'best_overrides_per_fold': best_overrides_per_fold,
        'parameter_stability': parameter_stability,
        'parameter_stability_by_key': parameter_stability_by_key,
    }


def plot_walk_forward(
    full_equity: pd.Series,
    oos_equity: pd.Series,
    split_boundaries: List[pd.Timestamp],
    output_path: str,
    fold_annotations: Optional[List[Dict]] = None,
) -> None:
    """Plot full equity/OOS equity with split boundaries and optional fold-parameter annotations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if full_equity is not None and len(full_equity) > 0:
        ax.plot(full_equity.index, full_equity.values, label='Full backtest', linewidth=1.5)
    if oos_equity is not None and len(oos_equity) > 0:
        ax.plot(oos_equity.index, oos_equity.values, label='Walk-forward OOS (chained)', linewidth=2.0)

    for boundary in split_boundaries or []:
        ax.axvline(boundary, color='grey', alpha=0.2, linewidth=0.8)

    if fold_annotations:
        y_candidates = [1.0]
        if full_equity is not None and len(full_equity) > 0:
            full_max = float(np.nanmax(full_equity.values))
            if np.isfinite(full_max):
                y_candidates.append(full_max)
        if oos_equity is not None and len(oos_equity) > 0:
            oos_max = float(np.nanmax(oos_equity.values))
            if np.isfinite(oos_max):
                y_candidates.append(oos_max)
        y_anchor = float(max(y_candidates))
        for row in fold_annotations:
            split = row.get('split') if isinstance(row, dict) else None
            override = row.get('best_override') if isinstance(row, dict) else None
            if split is None or not override:
                continue
            label = ', '.join(f"{k}={v}" for k, v in sorted((override or {}).items()))
            if len(label) > 48:
                label = label[:45] + '...'
            x = getattr(split, 'test_start', None)
            if x is None:
                continue
            ax.annotate(
                label,
                xy=(x, y_anchor),
                xytext=(0, 6),
                textcoords='offset points',
                rotation=90,
                fontsize=7,
                alpha=0.7,
                va='bottom',
                ha='left',
            )

    ax.set_title('Walk-Forward Analysis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
