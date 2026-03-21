#!/usr/bin/env python3
"""
Command-line backtesting script

Usage:
    python backtest.py --strategy momentum --start 2020-01-01 --end 2023-12-31
    python backtest.py --strategy mean_reversion --tickers SPY QQQ IWM
"""

import argparse
import json
import os
import sys
import yaml
from pydantic import ValidationError
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.engine import run_backtest_from_config
from backtesting.tuner import build_strategy_overrides, rank_results
from backtesting.validation import ValidationSuite
from backtesting.walk_forward import run_walk_forward, plot_walk_forward
from strategies import get_available_strategies, create_strategy
from utils.data_platform import AuditStore
from utils.logger import setup_logger
from utils.config_schema import validate_main_config


def _format_pct(value):
    return f"{value:.2%}"


def _format_num(value):
    return f"{value:.2f}"


def _format_pct_or_na(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2%}"


def _format_num_or_na(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"


def print_degradation_summary(results):
    """Print train-vs-out-of-sample degradation summary when available."""
    degradation = results.get('degradation_analysis', {}) or {}
    if not degradation.get('available', False):
        reason = degradation.get('reason', 'not_available')
        print(f"Validation split: unavailable ({reason})")
        return

    split_ratio = float(degradation.get('split_ratio', 0.7))
    train_pct = max(0.0, min(1.0, split_ratio)) * 100.0
    oos_pct = 100.0 - train_pct

    print("\n" + "=" * 70)
    print("IN-SAMPLE vs OUT-OF-SAMPLE SPLIT (single 70/30 temporal split)")
    print("=" * 70)
    print("Note: this is not walk-forward optimisation. No parameters were varied.")
    print(f"Split: {train_pct:.0f}% train / {oos_pct:.0f}% out-of-sample")
    print(f"  {'Train Sharpe:':<28} {_format_num(degradation.get('train_sharpe', 0.0)):>12}")
    print(f"  {'OOS Sharpe:':<28} {_format_num(degradation.get('oos_sharpe', 0.0)):>12}")
    print(f"  {'Sharpe Drift:':<28} {_format_num(degradation.get('sharpe_drift', 0.0)):>12}")
    print(f"  {'Train CAGR:':<28} {_format_pct(degradation.get('train_cagr', 0.0)):>12}")
    print(f"  {'OOS CAGR:':<28} {_format_pct(degradation.get('oos_cagr', 0.0)):>12}")
    print(f"  {'CAGR Drift:':<28} {_format_pct(degradation.get('cagr_drift', 0.0)):>12}")


def compute_monte_carlo_summary(
    results,
    n_sims: int = 300,
    horizon_days: int = 252,
    block_size: int = 5,
    random_state: int = 42,
):
    """Compute block-bootstrap Monte Carlo summary, retaining simulated paths and path stats."""
    equity_curve = results.get('equity_curve')
    if equity_curve is None or len(equity_curve) < 20:
        return {'available': False, 'reason': 'insufficient_history'}

    returns = equity_curve.pct_change().dropna()
    if len(returns) < 20:
        return {'available': False, 'reason': 'insufficient_returns'}

    mc = ValidationSuite.monte_carlo_simulation(
        returns=returns,
        n_sims=n_sims,
        horizon_days=horizon_days,
        block_size=block_size,
        random_state=random_state,
        return_paths=True,
    )
    paths = mc.get('paths')
    summary = {
        'available': True,
        'n_sims': n_sims,
        'horizon_days': horizon_days,
        'block_size': block_size,
        'random_state': random_state,
        'paths': paths,
    }
    summary.update({k: v for k, v in mc.items() if k != 'paths'})
    summary['path_stats'] = ValidationSuite.monte_carlo_path_stats(paths)
    return summary


def print_monte_carlo_summary(results):
    """Print Monte Carlo validation summary when available."""
    mc = results.get('monte_carlo_validation', {}) or {}
    if not mc.get('available', False):
        reason = mc.get('reason', 'not_available')
        print(f"Monte Carlo validation: unavailable ({reason})")
        return

    print("\n" + "=" * 70)
    print("BOOTSTRAP RETURN DISTRIBUTION (block bootstrap, not overfitting test)")
    print("=" * 70)
    print(
        f"Simulations: {mc.get('n_sims', 0)}, "
        f"Horizon: {mc.get('horizon_days', 0)} days, "
        f"Block size: {mc.get('block_size', 0)}"
    )
    print(f"  {'P05 1-year return (worst 5%):':<36} {_format_pct(mc.get('p05_return', 0.0)):>12}")
    print(f"  {'P50 1-year return (median):':<36} {_format_pct(mc.get('p50_return', 0.0)):>12}")
    print(f"  {'P95 1-year return (best 5%):':<36} {_format_pct(mc.get('p95_return', 0.0)):>12}")

    path_stats = mc.get('path_stats', {}) or {}
    if path_stats:
        print("\n" + "-" * 70)
        print("MONTE CARLO PATH STATS")
        print("-" * 70)
        print(f"  {'Avg drawdown:':<36} {_format_pct(path_stats.get('avg_drawdown', 0.0)):>12}")
        print(
            f"  {'Drawdown P05 / P95:':<36} "
            f"{_format_pct(path_stats.get('p05_drawdown', 0.0))} / "
            f"{_format_pct(path_stats.get('p95_drawdown', 0.0))}"
        )
        print(f"  {'Avg Sharpe:':<36} {_format_num(path_stats.get('avg_sharpe', 0.0)):>12}")
        print(
            f"  {'Sharpe P05 / P95:':<36} "
            f"{_format_num(path_stats.get('p05_sharpe', 0.0))} / "
            f"{_format_num(path_stats.get('p95_sharpe', 0.0))}"
        )
        print(f"  {'Avg annualized vol:':<36} {_format_pct(path_stats.get('avg_volatility', 0.0)):>12}")
        print(f"  {'Risk of ruin:':<36} {_format_pct(path_stats.get('p_ruin', 0.0)):>12}")


def plot_monte_carlo(actual_equity, paths, output_path, path_stats=None):
    """Plot actual equity plus Monte Carlo forward paths."""
    if paths is None or len(paths) == 0:
        return

    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        return

    n_sims, points = paths.shape
    horizon_days = points - 1
    last_actual = float(actual_equity.iloc[-1])
    scaled_paths = paths * last_actual

    x_future = np.arange(horizon_days + 1)
    fig, ax = plt.subplots(figsize=(12, 6))

    for row in scaled_paths:
        ax.plot(x_future, row, color='steelblue', alpha=0.04, linewidth=1)

    p05 = np.percentile(scaled_paths, 5, axis=0)
    p50 = np.percentile(scaled_paths, 50, axis=0)
    p95 = np.percentile(scaled_paths, 95, axis=0)
    ax.fill_between(x_future, p05, p95, color='steelblue', alpha=0.15, label='P05-P95 envelope')
    ax.plot(x_future, p50, color='steelblue', linewidth=2.0, label='Median path (P50)')

    x_actual = np.arange(len(actual_equity)) - (len(actual_equity) - 1)
    ax.plot(x_actual, actual_equity.values, color='darkorange', linewidth=2.0, label='Actual')

    if path_stats:
        stats_text = (
            f"p_ruin={path_stats.get('p_ruin', 0.0):.1%}\n"
            f"avg drawdown={path_stats.get('avg_drawdown', 0.0):.1%}\n"
            f"avg Sharpe={path_stats.get('avg_sharpe', 0.0):.2f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va='top',
            fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
        )

    ax.set_title(f"Monte Carlo — Block Bootstrap ({n_sims} simulations, {horizon_days}d horizon)")
    ax.set_xlabel('Trading days relative to projection start')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Monte Carlo plot saved to: {output_path}")


def save_monte_carlo_data(mc_summary, paths, path_stats, output_path):
    """Persist Monte Carlo simulation outputs for downstream analysis."""
    summary_serializable = {
        k: v for k, v in (mc_summary or {}).items()
        if k not in {'paths', 'path_stats'}
    }
    np.savez_compressed(
        output_path,
        paths=np.asarray(paths, dtype=np.float32),
        summary=np.array(json.dumps(summary_serializable), dtype=object),
        path_stats=np.array(json.dumps(path_stats or {}), dtype=object),
    )
    print(f"Monte Carlo data saved to: {output_path}")


def print_benchmark_summary(results):
    """Print benchmark buy-and-hold and relative performance summary."""
    strategy_metrics = results.get('metrics', {}) or {}
    benchmark = results.get('benchmark_analysis', {})
    if not benchmark.get('available', False):
        reason = benchmark.get('reason', 'not_available')
        print(f"Benchmark comparison: unavailable ({reason})")
        return

    ticker = benchmark.get('ticker', 'SPY')
    bm_metrics = benchmark.get('metrics', {})
    rel = benchmark.get('relative_stats', {})

    print("\n" + "="*70)
    print(f"BENCHMARK COMPARISON (Buy & Hold {ticker})")
    print("="*70)
    print(f"  {'Benchmark Total Return:':<28} {_format_pct(bm_metrics.get('total_return', 0.0)):>12}")
    print(f"  {'Benchmark CAGR:':<28} {_format_pct(bm_metrics.get('cagr', 0.0)):>12}")
    print(f"  {'Benchmark Sharpe:':<28} {_format_num(bm_metrics.get('sharpe_ratio', 0.0)):>12}")
    print(f"  {'CAPM Alpha (annualized):':<28} {_format_pct(rel.get('alpha', 0.0)):>12}")
    print(f"  {'Active Return (annualized):':<28} {_format_pct(rel.get('active_return', 0.0)):>12}")
    print(f"  {'Beta:':<28} {_format_num(rel.get('beta', 0.0)):>12}")
    print(f"  {'Tracking Error:':<28} {_format_pct(rel.get('tracking_error', 0.0)):>12}")
    print(f"  {'Information Ratio:':<28} {_format_num(rel.get('information_ratio', 0.0)):>12}")
    print("  Note: CAPM alpha uses a non-standard market proxy; concentrated strategy beta/alpha can be unstable.")

    spy = benchmark.get('spy_analysis', {}) or {}
    spy_metrics = {}
    spy_rel = {}
    if spy.get('available', False) and spy.get('ticker') != ticker:
        spy_metrics = spy.get('metrics', {}) or {}
        spy_rel = spy.get('relative_stats', {}) or {}
        print(f"\n  {'CAPM Alpha vs SPY:':<28} {_format_pct(spy_rel.get('alpha', 0.0)):>12}")
        print(f"  {'Active Return vs SPY:':<28} {_format_pct(spy_rel.get('active_return', 0.0)):>12}")
        print(f"  {'Beta vs SPY:':<28} {_format_num(spy_rel.get('beta', 0.0)):>12}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<30}{'Strategy':>12}{ticker:>15}{'SPY':>12}")
    print(f"{'-' * 30}{'-' * 12}{'-' * 15}{'-' * 12}")
    print(
        f"{'Total Return':<30}"
        f"{_format_pct_or_na(strategy_metrics.get('total_return')):>12}"
        f"{_format_pct_or_na(bm_metrics.get('total_return')):>15}"
        f"{_format_pct_or_na(spy_metrics.get('total_return')):>12}"
    )
    print(
        f"{'CAGR':<30}"
        f"{_format_pct_or_na(strategy_metrics.get('cagr')):>12}"
        f"{_format_pct_or_na(bm_metrics.get('cagr')):>15}"
        f"{_format_pct_or_na(spy_metrics.get('cagr')):>12}"
    )
    print(
        f"{'Sharpe Ratio':<30}"
        f"{_format_num_or_na(strategy_metrics.get('sharpe_ratio')):>12}"
        f"{_format_num_or_na(bm_metrics.get('sharpe_ratio')):>15}"
        f"{_format_num_or_na(spy_metrics.get('sharpe_ratio')):>12}"
    )
    print(
        f"{'CAPM Alpha (annualized)':<30}"
        f"{_format_pct_or_na(rel.get('alpha')):>12}"
        f"{'--':>15}"
        f"{_format_pct_or_na(spy_rel.get('alpha')):>12}"
    )
    print(
        f"{'Active Return (annualized)':<30}"
        f"{_format_pct_or_na(rel.get('active_return')):>12}"
        f"{'--':>15}"
        f"{_format_pct_or_na(spy_rel.get('active_return')):>12}"
    )
    print(
        f"{'Beta':<30}"
        f"{_format_num_or_na(rel.get('beta')):>12}"
        f"{'--':>15}"
        f"{_format_num_or_na(spy_rel.get('beta')):>12}"
    )
    print(
        f"{'Tracking Error':<30}"
        f"{_format_pct_or_na(rel.get('tracking_error')):>12}"
        f"{'--':>15}"
        f"{_format_pct_or_na(spy_rel.get('tracking_error')):>12}"
    )
    print(
        f"{'Information Ratio':<30}"
        f"{_format_num_or_na(rel.get('information_ratio')):>12}"
        f"{'--':>15}"
        f"{_format_num_or_na(spy_rel.get('information_ratio')):>12}"
    )




def print_regime_performance_summary(results):
    """Print regime-conditional performance table when available."""
    metrics = results.get('metrics', {}) or {}
    regime_perf = metrics.get('regime_performance', {}) or {}
    if not regime_perf:
        print("Regime-conditional performance: unavailable")
        return

    print("\n" + "=" * 70)
    print("REGIME-CONDITIONAL PERFORMANCE")
    print("=" * 70)
    print(f"{'Regime':<10}{'Days':>8}{'Frac':>10}{'CAGR':>12}{'Vol':>12}{'Sharpe':>10}{'MaxDD':>12}")
    for regime in ['bull', 'neutral', 'bear']:
        row = regime_perf.get(regime)
        if not row:
            continue
        print(
            f"{regime:<10}{int(row.get('days', 0)):>8}{_format_pct(row.get('fraction', 0.0)):>10}"
            f"{_format_pct(row.get('cagr', 0.0)):>12}{_format_pct(row.get('volatility', 0.0)):>12}"
            f"{_format_num(row.get('sharpe', 0.0)):>10}{_format_pct(row.get('max_drawdown', 0.0)):>12}"
        )



def plot_calendar_heatmap(monthly_returns: pd.Series, output_path: str) -> None:
    """Plot a year x month heatmap of monthly returns."""
    if monthly_returns is None:
        return
    monthly = monthly_returns.dropna()
    if monthly.empty:
        return

    monthly.index = pd.to_datetime(monthly.index)
    years = sorted(monthly.index.year.unique())
    if not years:
        return

    grid = np.full((len(years), 12), np.nan, dtype=float)
    year_to_row = {year: i for i, year in enumerate(years)}
    for dt, value in monthly.items():
        row = year_to_row[int(dt.year)]
        col = int(dt.month) - 1
        grid[row, col] = float(value)

    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.5 * len(years) + 1.5)))
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 0.1
    vmax = max(vmax, 0.1)
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)

    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels([str(y) for y in years])

    for i in range(len(years)):
        for j in range(12):
            val = grid[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.1%}", ha='center', va='center', fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Monthly return')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Calendar heatmap saved to: {output_path}")

def plot_results(results, output_path='results/backtest_plot.png'):
    """Plot backtest results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    equity = results['equity_curve']
    base_currency = getattr(results.get('portfolio'), 'base_currency', 'BASE')
    axes[0].plot(equity.index, equity.values, label='Portfolio', linewidth=2)

    benchmark = results.get('benchmark_analysis', {})
    if benchmark.get('available', False):
        benchmark_equity = benchmark.get('equity_curve')
        ticker = benchmark.get('ticker', 'SPY')
        if benchmark_equity is not None and len(benchmark_equity) > 0:
            axes[0].plot(
                benchmark_equity.index,
                benchmark_equity.values,
                label=f'Buy & Hold {ticker}',
                linewidth=2,
                linestyle='--',
                alpha=0.85,
            )

    spy = benchmark.get('spy_analysis', {}) if isinstance(benchmark, dict) else {}
    if spy and spy.get('available', False):
        spy_equity = spy.get('equity_curve')
        if spy_equity is not None and len(spy_equity) > 0:
            axes[0].plot(
                spy_equity.index,
                spy_equity.values,
                label='Buy & Hold SPY',
                linewidth=1.8,
                linestyle=':',
                alpha=0.9,
            )

    axes[0].set_title(f"Equity Curve - {results['strategy']}", fontsize=14, fontweight='bold')
    axes[0].set_ylabel(f'Equity ({base_currency})', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    metrics = results.get('metrics', {}) or {}
    realized_equity = float(metrics.get('realized_equity', 0.0))
    unrealized_pnl = float(metrics.get('unrealized_pnl', 0.0))
    axes[0].text(
        0.01,
        0.02,
        f"Realized equity: {realized_equity:,.2f} | Unrealized PnL: {unrealized_pnl:,.2f}",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'},
    )

    from backtesting.metrics import PerformanceMetrics
    drawdown = PerformanceMetrics.calculate_drawdown_series(equity)
    axes[1].fill_between(drawdown.index, 0, drawdown.values * 100,
                         color='red', alpha=0.3, label='Drawdown')
    events = PerformanceMetrics.drawdown_events(equity, min_drawdown=0.02)
    if events:
        annotate_events = events
        if len(events) > 10:
            annotate_events = sorted(events, key=lambda e: abs(e.get('drawdown_pct', 0.0)), reverse=True)[:5]
        for idx, event in enumerate(events):
            start = event['start_date']
            end = event.get('recovery_date') or equity.index[-1]
            color = 'salmon' if idx % 2 == 0 else 'lightsalmon'
            axes[1].axvspan(start, end, color=color, alpha=0.10)
        for event in annotate_events:
            trough = event['trough_date']
            dd = float(event.get('drawdown_pct', 0.0))
            dur = event.get('duration_days')
            dur_label = f"{dur}d" if dur is not None else "open"
            axes[1].text(trough, dd * 100, f"{dd:.1%} / {dur_label}", fontsize=7)
    axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    rolling_63 = metrics.get('rolling_sharpe_63_series')
    rolling_252 = metrics.get('rolling_sharpe_252_series')
    if rolling_63 is None or len(rolling_63) == 0:
        rolling_63 = PerformanceMetrics.calculate_rolling_sharpe(
            equity.pct_change().dropna(), window=min(63, max(len(equity) - 1, 1))
        ).dropna()
    if rolling_252 is None or len(rolling_252) == 0:
        rolling_252 = PerformanceMetrics.calculate_rolling_sharpe(
            equity.pct_change().dropna(), window=min(252, max(len(equity) - 1, 1))
        ).dropna()

    if rolling_63 is not None and len(rolling_63) > 0:
        axes[2].plot(rolling_63.index, rolling_63.values, label='Rolling Sharpe 63d', linewidth=1.8)
    if rolling_252 is not None and len(rolling_252) > 0:
        axes[2].plot(rolling_252.index, rolling_252.values, label='Rolling Sharpe 252d', linewidth=1.8, linestyle='--')
    axes[2].axhline(0.0, color='black', linewidth=1.0, alpha=0.5)
    axes[2].set_title('Rolling Sharpe', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Sharpe', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def save_trades(results, output_path='results/trades.csv'):
    """Save trade log to CSV."""

    trades = results['trades']
    if not trades:
        print("No trades to save")
        return

    trade_data = []
    for trade in trades:
        trade_data.append({
            'run_id': getattr(trade, 'run_id', None),
            'date': trade.date,
            'signal_date': getattr(trade, 'signal_date', None),
            'ticker': trade.ticker,
            'action': trade.action,
            'decision': getattr(trade, 'decision', None),
            'decision_reason': getattr(trade, 'decision_reason', None),
            'signal_strength': getattr(trade, 'signal_strength', None),
            'signal_confidence': getattr(trade, 'signal_confidence', None),
            'weight_delta': getattr(trade, 'weight_delta', None),
            'current_weight': getattr(trade, 'current_weight', None),
            'target_weight': getattr(trade, 'target_weight', None),
            'current_shares': getattr(trade, 'current_shares', None),
            'target_shares': getattr(trade, 'target_shares', None),
            'executed_shares': getattr(trade, 'executed_shares', trade.shares),
            'shares': trade.shares,
            'price': trade.price,
            'value': trade.value,
            'commission': trade.commission,
            'slippage': trade.slippage,
            'fx_cost': trade.fx_cost,
            'total_cost': trade.total_cost(),
            'currency': trade.currency
        })

    df = pd.DataFrame(trade_data)
    df.to_csv(output_path, index=False)
    print(f"Trades saved to: {output_path}")


def save_metrics(results, output_path='results/metrics.txt'):
    """Save metrics to text file."""
    import io

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    from backtesting.metrics import PerformanceMetrics
    PerformanceMetrics.print_metrics(results['metrics'])

    output = buffer.getvalue()
    sys.stdout = old_stdout

    with open(output_path, 'w') as f:
        f.write(f"Strategy: {results['strategy']}\n")
        f.write(f"Tickers: {', '.join(results['tickers'])}\n")
        f.write(f"Period: {results['start_date']} to {results['end_date']}\n")
        f.write(output)

        benchmark = results.get('benchmark_analysis', {})
        f.write("\nBenchmark Comparison\n")
        f.write("-" * 30 + "\n")
        if benchmark.get('available', False):
            bm_metrics = benchmark.get('metrics', {})
            rel = benchmark.get('relative_stats', {})
            ticker = benchmark.get('ticker', 'SPY')
            f.write(f"Ticker: {ticker} (buy-and-hold)\n")
            f.write(f"Benchmark total return: {_format_pct(bm_metrics.get('total_return', 0.0))}\n")
            f.write(f"Benchmark CAGR: {_format_pct(bm_metrics.get('cagr', 0.0))}\n")
            f.write(f"CAPM alpha (annualized): {_format_pct(rel.get('alpha', 0.0))}\n")
            f.write(f"Active return (annualized): {_format_pct(rel.get('active_return', 0.0))}\n")
            f.write(f"Beta: {_format_num(rel.get('beta', 0.0))}\n")
            f.write(f"Tracking error: {_format_pct(rel.get('tracking_error', 0.0))}\n")
            f.write(f"Information ratio: {_format_num(rel.get('information_ratio', 0.0))}\n")
            f.write("Note: CAPM alpha uses a non-standard market proxy; beta/alpha can be unstable for concentrated portfolios.\n")
            spy = benchmark.get('spy_analysis', {}) or {}
            if spy.get('available', False) and spy.get('ticker') != ticker:
                spy_rel = spy.get('relative_stats', {}) or {}
                f.write(f"CAPM alpha vs SPY (annualized): {_format_pct(spy_rel.get('alpha', 0.0))}\n")
                f.write(f"Active return vs SPY (annualized): {_format_pct(spy_rel.get('active_return', 0.0))}\n")
        else:
            f.write(f"Unavailable: {benchmark.get('reason', 'not_available')}\n")

        degradation = results.get('degradation_analysis', {}) or {}
        f.write("\nIn-Sample vs Out-of-Sample Split (single 70/30 temporal split)\n")
        f.write("-" * 68 + "\n")
        f.write("Note: this is not walk-forward optimisation. No parameters were varied.\n")
        if degradation.get('available', False):
            split_ratio = float(degradation.get('split_ratio', 0.7))
            train_pct = max(0.0, min(1.0, split_ratio)) * 100.0
            oos_pct = 100.0 - train_pct
            f.write(f"Split: {train_pct:.0f}% train / {oos_pct:.0f}% out-of-sample\n")
            f.write(f"Train Sharpe: {_format_num(degradation.get('train_sharpe', 0.0))}\n")
            f.write(f"OOS Sharpe: {_format_num(degradation.get('oos_sharpe', 0.0))}\n")
            f.write(f"Sharpe Drift: {_format_num(degradation.get('sharpe_drift', 0.0))}\n")
            f.write(f"Train CAGR: {_format_pct(degradation.get('train_cagr', 0.0))}\n")
            f.write(f"OOS CAGR: {_format_pct(degradation.get('oos_cagr', 0.0))}\n")
            f.write(f"CAGR Drift: {_format_pct(degradation.get('cagr_drift', 0.0))}\n")
        else:
            f.write(f"Unavailable: {degradation.get('reason', 'not_available')}\n")

        regime_perf = (results.get('metrics', {}) or {}).get('regime_performance', {}) or {}
        f.write("\nRegime-Conditional Performance\n")
        f.write("-" * 30 + "\n")
        if regime_perf:
            for regime in ['bull', 'neutral', 'bear']:
                row = regime_perf.get(regime)
                if not row:
                    continue
                f.write(
                    f"{regime}: days={int(row.get('days', 0))}, frac={_format_pct(row.get('fraction', 0.0))}, "
                    f"CAGR={_format_pct(row.get('cagr', 0.0))}, vol={_format_pct(row.get('volatility', 0.0))}, "
                    f"Sharpe={_format_num(row.get('sharpe', 0.0))}, maxDD={_format_pct(row.get('max_drawdown', 0.0))}\n"
                )
        else:
            f.write("Unavailable\n")

        mc = results.get('monte_carlo_validation', {}) or {}
        f.write("\nBootstrap Return Distribution (block bootstrap, not overfitting test)\n")
        f.write("-" * 68 + "\n")
        if mc.get('available', False):
            f.write(
                f"Simulations: {mc.get('n_sims', 0)}, Horizon: {mc.get('horizon_days', 0)} days, "
                f"Block size: {mc.get('block_size', 0)}\n"
            )
            f.write(f"P05 1-year return (worst 5%): {_format_pct(mc.get('p05_return', 0.0))}\n")
            f.write(f"P50 1-year return (median): {_format_pct(mc.get('p50_return', 0.0))}\n")
            f.write(f"P95 1-year return (best 5%): {_format_pct(mc.get('p95_return', 0.0))}\n")
            path_stats = mc.get('path_stats', {}) or {}
            if path_stats:
                f.write("\nMonte Carlo Path Stats\n")
                f.write("-" * 30 + "\n")
                f.write(f"Avg drawdown: {_format_pct(path_stats.get('avg_drawdown', 0.0))}\n")
                f.write(
                    f"Drawdown P05 / P95: {_format_pct(path_stats.get('p05_drawdown', 0.0))} / "
                    f"{_format_pct(path_stats.get('p95_drawdown', 0.0))}\n"
                )
                f.write(f"Avg Sharpe: {_format_num(path_stats.get('avg_sharpe', 0.0))}\n")
                f.write(
                    f"Sharpe P05 / P95: {_format_num(path_stats.get('p05_sharpe', 0.0))} / "
                    f"{_format_num(path_stats.get('p95_sharpe', 0.0))}\n"
                )
                f.write(f"Avg annualized vol: {_format_pct(path_stats.get('avg_volatility', 0.0))}\n")
                f.write(f"Risk of ruin: {_format_pct(path_stats.get('p_ruin', 0.0))}\n")
        else:
            f.write(f"Unavailable: {mc.get('reason', 'not_available')}\n")

    print(f"Metrics saved to: {output_path}")


def _load_json_override(raw_value, label):
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON override: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} override must be a JSON object")
    return parsed




def _deep_merge(base, override):
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def _write_tuning_results(rows, output_path):
    import csv

    if not rows:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        'run_id',
        'db_run_id',
        'strategy_override',
        'total_return',
        'cagr',
        'sharpe_ratio',
        'rolling_sharpe_63',
        'rolling_sharpe_252',
        'max_drawdown',
        'volatility',
        'train_sharpe',
        'oos_sharpe',
        'sharpe_drift',
        'train_cagr',
        'oos_cagr',
        'cagr_drift',
    ]
    with open(output_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get('metrics', {})
            degradation = row.get('degradation_analysis', {}) or {}
            writer.writerow({
                'run_id': row.get('sweep_index', row.get('run_id')),
                'db_run_id': row.get('run_id'),
                'strategy_override': json.dumps(row.get('strategy_override', {}), sort_keys=True),
                'total_return': metrics.get('total_return'),
                'cagr': metrics.get('cagr'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'rolling_sharpe_63': metrics.get('rolling_sharpe_63'),
                'rolling_sharpe_252': metrics.get('rolling_sharpe_252'),
                'max_drawdown': metrics.get('max_drawdown'),
                'volatility': metrics.get('volatility'),
                'train_sharpe': degradation.get('train_sharpe'),
                'oos_sharpe': degradation.get('oos_sharpe'),
                'sharpe_drift': degradation.get('sharpe_drift'),
                'train_cagr': degradation.get('train_cagr'),
                'oos_cagr': degradation.get('oos_cagr'),
                'cagr_drift': degradation.get('cagr_drift'),
            })


def _build_param_grid_from_overrides(sweep_overrides):
    """Build scalar parameter grid from concrete sweep overrides."""
    grid = {}
    for override in sweep_overrides:
        for key, value in (override or {}).items():
            if isinstance(value, (dict, list, tuple, set)):
                continue
            grid.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in grid.items() if values}


def _resolve_audit_store(config_path: str, config_override: dict = None) -> AuditStore:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    if config_override:
        config = _deep_merge(config, config_override)
    try:
        validate_main_config(config)
    except (ValidationError, ValueError) as e:
        raise ValueError(f"Config validation failed: {e}") from e
    data_platform_cfg = config.get('data_platform', {}) or {}
    if not data_platform_cfg.get('enabled', False):
        raise ValueError("data_platform.enabled is false; enable it in config to query stored runs/journals.")
    return AuditStore(data_platform_cfg.get('sqlite_path', 'state/trading_audit.db'))


def _print_dataframe(df, index: bool = False):
    if df is None or df.empty:
        print("No records found.")
        return
    print(df.to_string(index=index))


def main():
    parser = argparse.ArgumentParser(
        description='Run backtesting for trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --strategy momentum --start 2020-01-01 --end 2023-12-31
  python backtest.py --strategy momentum --tickers SPY QQQ IWM
  python backtest.py --strategy momentum --sweep lookback=63,126 --sweep n_positions=3,5
  python backtest.py --strategy momentum --sweep lookback=63:252:63 --sweep min_momentum=0.0:0.2:0.1
        """
    )

    available_strategies = ', '.join(get_available_strategies())
    parser.add_argument('--strategy', type=str, default='momentum', help=f'Strategy name ({available_strategies})')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--strategies-config', type=str, default='config/strategies.yaml', help='Path to strategy config file')
    parser.add_argument('--tickers', nargs='+', default=None, help='List of tickers to trade')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD, defaults to today)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results to files')
    parser.add_argument('--mc-seed', type=int, default=42,
                        help='Random seed for Monte Carlo bootstrap (default: 42)')
    parser.add_argument('--no-mc-plot', action='store_true',
                        help='Skip Monte Carlo projection plot generation')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward analysis with fixed strategy config')
    parser.add_argument('--wf-train-days', type=int, default=252, help='Walk-forward train window (days)')
    parser.add_argument('--wf-oos-days', type=int, default=63, help='Walk-forward out-of-sample window (days)')
    parser.add_argument('--wf-step-days', type=int, default=63, help='Walk-forward step size (days)')
    parser.add_argument('--wf-anchored', action='store_true',
                        help='Use anchored expanding train window for walk-forward')
    parser.add_argument('--config-override', type=str, default=None, help='JSON object to override main config')
    parser.add_argument('--strategy-override', type=str, default=None, help='JSON object to override strategy config')
    parser.add_argument('--sweep', action='append', default=[],
                        help='Sweep definition key=v1,v2 or numeric range key=start:end[:step]')
    parser.add_argument('--optimize-metric', type=str, default='sharpe_ratio', help='Metric for ranking sweep runs')
    parser.add_argument('--validate-sweep', action='store_true',
                        help='Run advanced rolling/sensitivity validation in sweep mode (slower)')
    parser.add_argument('--list-runs', action='store_true', help='List persisted backtest runs from SQLite')
    parser.add_argument('--compare-runs', nargs='+', default=None, help='Compare persisted runs by run_id')
    parser.add_argument('--journal-run', type=str, default=None, help='Show trade decision journal for a run_id')
    parser.add_argument('--journal-limit', type=int, default=200, help='Max journal rows to display')
    parser.add_argument('--runs-limit', type=int, default=30, help='Max run rows to display')
    parser.add_argument('--runs-strategy', type=str, default=None, help='Optional strategy filter for --list-runs')

    args = parser.parse_args()

    end_ts = pd.Timestamp(args.end).tz_localize(None).normalize() if args.end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(args.start).tz_localize(None).normalize()
    if start_ts >= end_ts:
        print("ERROR: --start must precede --end")
        sys.exit(1)

    try:
        config_override = _load_json_override(args.config_override, 'config')
        base_strategy_override = _load_json_override(args.strategy_override, 'strategy') or {}

        with open(args.config, 'r') as f:
            merged_config = yaml.safe_load(f) or {}
        if config_override:
            merged_config = _deep_merge(merged_config, config_override)
        try:
            validate_main_config(merged_config)
        except (ValidationError, ValueError) as e:
            print("ERROR: config validation failed")
            print(e)
            sys.exit(1)

        if args.list_runs or args.compare_runs or args.journal_run:
            audit_store = _resolve_audit_store(args.config, config_override=config_override)
            if args.list_runs:
                print("=" * 70)
                print("PERSISTED BACKTEST RUNS")
                print("=" * 70)
                runs_df = audit_store.query_backtest_runs(
                    strategy=args.runs_strategy,
                    limit=max(1, int(args.runs_limit)),
                )
                _print_dataframe(runs_df)
            if args.compare_runs:
                print("\n" + "=" * 70)
                print("RUN COMPARISON")
                print("=" * 70)
                comparison_df = audit_store.compare_backtest_runs(args.compare_runs)
                _print_dataframe(comparison_df)
            if args.journal_run:
                print("\n" + "=" * 70)
                print(f"TRADE DECISION JOURNAL ({args.journal_run})")
                print("=" * 70)
                journal_df = audit_store.query_trade_decisions(
                    run_id=args.journal_run,
                    limit=max(1, int(args.journal_limit)),
                )
                _print_dataframe(journal_df)
            return

        setup_logger('backtest', f'{args.output_dir}/backtest.log', file_mode='w')

        print("="*70)
        print("BACKTESTING")
        print("="*70)
        print(f"Strategy: {args.strategy}")
        print(f"Period: {args.start} to {args.end or 'today'}")
        if args.tickers:
            print(f"Tickers: {', '.join(args.tickers)}")
        print("="*70)

        # Only run sweep mode when --sweep is explicitly provided.
        if args.sweep:
            sweep_overrides = build_strategy_overrides(args.sweep)
            all_results = []

            for run_idx, sweep_override in enumerate(sweep_overrides, start=1):
                strategy_override = _deep_merge(base_strategy_override, sweep_override)
                print(f"\nRun {run_idx}/{len(sweep_overrides)} strategy override: {strategy_override}")

                results = run_backtest_from_config(
                    config_path=args.config,
                    strategy_name=args.strategy,
                    tickers=args.tickers,
                    start_date=args.start,
                    end_date=args.end,
                    strategies_config_path=args.strategies_config,
                    config_override=config_override,
                    strategy_override=strategy_override or None,
                )
                results['sweep_index'] = run_idx
                results['strategy_override'] = strategy_override
                results['monte_carlo_validation'] = compute_monte_carlo_summary(
                    results,
                    random_state=args.mc_seed,
                )
                all_results.append(results)

            top = rank_results(all_results, metric=args.optimize_metric, top_n=min(5, len(all_results)))
            print("\n" + "=" * 70)
            print(f"SWEEP SUMMARY (ranked by {args.optimize_metric})")
            print("=" * 70)
            print("Note: ranking metric is in-sample over the full backtest window.")
            print("Use the OOS Sharpe/CAGR columns below to judge robustness.")
            for row in top:
                metric_value = row.get('metrics', {}).get(args.optimize_metric)
                degradation = row.get('degradation_analysis', {}) or {}
                oos_sharpe = degradation.get('oos_sharpe') if degradation.get('available', False) else None
                oos_cagr = degradation.get('oos_cagr') if degradation.get('available', False) else None

                metric_text = _format_num(metric_value) if isinstance(metric_value, (int, float)) else str(metric_value)
                oos_sharpe_text = _format_num(oos_sharpe) if isinstance(oos_sharpe, (int, float)) else "N/A"
                oos_cagr_text = _format_pct(oos_cagr) if isinstance(oos_cagr, (int, float)) else "N/A"
                display_run_id = row.get('sweep_index', row.get('run_id'))
                print(
                    f"Run {display_run_id}: in_sample={metric_text} "
                    f"oos_sharpe={oos_sharpe_text} oos_cagr={oos_cagr_text} "
                    f"override={row.get('strategy_override')}"
                )

            universe_metrics = {
                str(row.get('sweep_index', row.get('run_id'))): row.get('metrics', {})
                for row in all_results
            }
            diversity_summary = ValidationSuite.multi_universe_summary(universe_metrics)
            print("\nValidation summary across sweep runs:")
            print(
                f"  universes={diversity_summary.get('universes', 0)} "
                f"avg_sharpe={_format_num(diversity_summary.get('avg_sharpe', 0.0))} "
                f"sharpe_dispersion={_format_num(diversity_summary.get('sharpe_dispersion', 0.0))} "
                f"avg_total_return={_format_pct(diversity_summary.get('avg_total_return', 0.0))}"
            )

            all_param_keys = sorted({k for row in all_results for k in (row.get('strategy_override', {}) or {}).keys()})
            stability_input = []
            for row in all_results:
                params = {}
                override = row.get('strategy_override', {}) or {}
                for key in all_param_keys:
                    value = override.get(key)
                    if isinstance(value, (dict, list, tuple, set)):
                        value = json.dumps(value, sort_keys=True)
                    params[key] = value
                stability_input.append({'best_params': params})
            if stability_input:
                stability = ValidationSuite.parameter_stability(stability_input)
                print(f"  parameter_stability={_format_num(stability.get('stability', 0.0))}")

            if not args.no_save:
                _write_tuning_results(all_results, f'{args.output_dir}/tuning_results.csv')
                print(f"Tuning results saved to: {args.output_dir}/tuning_results.csv")

            if args.validate_sweep and all_results:
                param_grid = _build_param_grid_from_overrides(sweep_overrides)
                index = all_results[0].get('equity_curve', None)
                index = index.index if index is not None else []

                required_history = 0
                try:
                    strategy_for_warmup = create_strategy(args.strategy, {})
                    required_history = int(strategy_for_warmup.get_required_history())
                except Exception:
                    required_history = 0
                effective_observations = max(0, len(index) - required_history)

                if param_grid and effective_observations >= 120:
                    print("\nAdvanced sweep validation (rolling + sensitivity):")
                    train_size = max(60, int(len(index) * 0.6))
                    test_size = max(20, int(len(index) * 0.2))
                    step_size = test_size

                    def objective_fn(split, params):
                        split_override = _deep_merge(base_strategy_override, params)
                        split_results = run_backtest_from_config(
                            config_path=args.config,
                            strategy_name=args.strategy,
                            tickers=args.tickers,
                            start_date=str(split.train_start.date()),
                            end_date=str(split.train_end.date()),
                            strategies_config_path=args.strategies_config,
                            config_override=config_override,
                            strategy_override=split_override or None,
                        )
                        return float(split_results.get('metrics', {}).get(args.optimize_metric, 0.0) or 0.0)

                    try:
                        rolling = ValidationSuite.rolling_parameter_optimization(
                            index=index,
                            param_grid=param_grid,
                            objective_fn=objective_fn,
                            train_size=train_size,
                            test_size=test_size,
                            step_size=step_size,
                            anchored=False,
                        )
                        roll_stability = ValidationSuite.parameter_stability(rolling) if rolling else {'stability': 0.0}
                        print(
                            f"  rolling_windows={len(rolling)} "
                            f"rolling_parameter_stability={_format_num(roll_stability.get('stability', 0.0))}"
                        )
                    except Exception as exc:
                        print(f"  rolling validation skipped: {exc}")

                    try:
                        base_params = dict(top[0].get('strategy_override', {})) if top else {}

                        def evaluator(candidate):
                            candidate_override = _deep_merge(base_strategy_override, candidate)
                            candidate_results = run_backtest_from_config(
                                config_path=args.config,
                                strategy_name=args.strategy,
                                tickers=args.tickers,
                                start_date=args.start,
                                end_date=args.end,
                                strategies_config_path=args.strategies_config,
                                config_override=config_override,
                                strategy_override=candidate_override or None,
                            )
                            return float(candidate_results.get('metrics', {}).get(args.optimize_metric, 0.0) or 0.0)

                        sensitivity = ValidationSuite.sensitivity_analysis(
                            base_params=base_params,
                            perturbations=param_grid,
                            evaluator=evaluator,
                        )
                        if not sensitivity.empty:
                            best_score = float(sensitivity.iloc[0]['score'])
                            print(
                                f"  sensitivity_candidates={len(sensitivity)} "
                                f"best_{args.optimize_metric}={_format_num(best_score)}"
                            )
                    except Exception as exc:
                        print(f"  sensitivity validation skipped: {exc}")
                else:
                    print("\nAdvanced sweep validation skipped: need scalar sweep params and >=120 post-warmup observations")
        else:
            strategy_override = base_strategy_override or None
            results = run_backtest_from_config(
                config_path=args.config,
                strategy_name=args.strategy,
                tickers=args.tickers,
                start_date=args.start,
                end_date=args.end,
                strategies_config_path=args.strategies_config,
                config_override=config_override,
                strategy_override=strategy_override,
            )
            results['monte_carlo_validation'] = compute_monte_carlo_summary(
                results,
                random_state=args.mc_seed,
            )
            print(f"Run ID: {results.get('run_id', 'N/A')}")

            print_degradation_summary(results)
            print_monte_carlo_summary(results)
            print_regime_performance_summary(results)
            print_benchmark_summary(results)
            mc = results.get('monte_carlo_validation', {}) or {}
            mc_paths = mc.get('paths')
            mc_path_stats = mc.get('path_stats', {}) or {}

            if not args.no_save:
                os.makedirs(args.output_dir, exist_ok=True)
                save_trades(results, f'{args.output_dir}/trades.csv')
                save_metrics(results, f'{args.output_dir}/metrics.txt')
                equity_out = results['equity_curve'].to_frame(name='equity')
                regime_series = results.get('regime_series')
                if regime_series is not None and len(regime_series) > 0:
                    equity_out['regime'] = regime_series.reindex(equity_out.index)
                equity_out.to_csv(f'{args.output_dir}/equity_curve.csv')
                print(f"Equity curve saved to: {args.output_dir}/equity_curve.csv")
                positions_history = results.get('positions_history')
                if positions_history is not None and not positions_history.empty:
                    positions_history.to_csv(f'{args.output_dir}/positions_history.csv')
                    print(f"Positions history saved to: {args.output_dir}/positions_history.csv")
                monthly_returns = (results.get('metrics', {}) or {}).get('monthly_returns')
                if isinstance(monthly_returns, pd.Series) and not monthly_returns.dropna().empty:
                    monthly_returns.dropna().to_frame(name='monthly_return').to_csv(
                        f'{args.output_dir}/monthly_returns.csv',
                        index_label='date',
                    )
                    print(f"Monthly returns saved to: {args.output_dir}/monthly_returns.csv")
                if mc.get('available', False) and mc_paths is not None:
                    save_monte_carlo_data(
                        mc_summary=mc,
                        paths=mc_paths,
                        path_stats=mc_path_stats,
                        output_path=f'{args.output_dir}/monte_carlo_data.npz',
                    )

            if not args.no_plot:
                try:
                    plot_results(results, f'{args.output_dir}/backtest_plot.png')
                    monthly_returns = (results.get('metrics', {}) or {}).get('monthly_returns')
                    if isinstance(monthly_returns, pd.Series) and not monthly_returns.dropna().empty:
                        plot_calendar_heatmap(
                            monthly_returns=monthly_returns,
                            output_path=f'{args.output_dir}/calendar_heatmap.png',
                        )
                except Exception as e:
                    print(f"Warning: Could not create plot: {e}")

            if not args.no_mc_plot and mc.get('available', False) and mc_paths is not None:
                try:
                    os.makedirs(args.output_dir, exist_ok=True)
                    plot_monte_carlo(
                        actual_equity=results['equity_curve'],
                        paths=mc_paths,
                        output_path=f'{args.output_dir}/monte_carlo_plot.png',
                        path_stats=mc_path_stats,
                    )
                except Exception as e:
                    print(f"Warning: Could not create Monte Carlo plot: {e}")

            if args.walk_forward:
                try:
                    wf = run_walk_forward(
                        index=results['equity_curve'].index,
                        config_path=args.config,
                        strategy_name=args.strategy,
                        tickers=args.tickers or merged_config.get('data', {}).get('tickers', ['SPY', 'QQQ']),
                        strategies_config_path=args.strategies_config,
                        train_days=args.wf_train_days,
                        oos_days=args.wf_oos_days,
                        step_days=args.wf_step_days,
                        anchored=args.wf_anchored,
                        config_override=config_override,
                        strategy_override=strategy_override,
                    )
                    agg = wf.get('aggregate_oos_metrics', {})
                    print("\nWalk-forward aggregate OOS metrics:")
                    print(f"  Mean OOS Sharpe: {agg.get('mean_oos_sharpe', 0.0):.3f}")
                    print(f"  Mean OOS CAGR: {agg.get('mean_oos_cagr', 0.0):.2%}")
                    print(
                        f"  Positive OOS Sharpe fraction: "
                        f"{agg.get('positive_oos_sharpe_fraction', 0.0):.2%}"
                    )

                    if not args.no_plot:
                        os.makedirs(args.output_dir, exist_ok=True)
                        plot_walk_forward(
                            full_equity=results['equity_curve'],
                            oos_equity=wf.get('oos_equity_curve'),
                            split_boundaries=wf.get('split_boundaries', []),
                            output_path=f"{args.output_dir}/walk_forward_plot.png",
                        )
                        print(f"Walk-forward plot saved to: {args.output_dir}/walk_forward_plot.png")
                except Exception as e:
                    print(f"Warning: Walk-forward analysis failed: {e}")

        print("\n" + "="*70)
        print("BACKTEST COMPLETE")
        print("="*70)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
