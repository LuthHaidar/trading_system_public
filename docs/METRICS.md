# Backtest Results & Metrics Reference

This document describes the structure returned by `run_backtest_from_config(...)` and the
meaning/availability of each major field.

## Top-level result object

`run_backtest_from_config(...)` returns a `dict` with the following keys.

| Key | Type | Presence | Description |
|---|---|---|---|
| `run_id` | `str` | Always | Unique run identifier generated for the backtest. |
| `portfolio` | `Portfolio` | Always | Final portfolio object with positions, cash, and trade ledger. |
| `metrics` | `dict` | Always | Comprehensive performance and risk metrics for the run. |
| `equity_curve` | `pd.Series` | Always | Portfolio equity over time (warmup-aligned start). |
| `trades` | `List[Trade]` | Always | Executed trades in chronological order (may be empty). |
| `strategy` | `str` | Always | Strategy name used for the run. |
| `tickers` | `List[str]` | Always | Universe requested by the caller. |
| `start_date` | `str` | Always | Requested start date input. |
| `effective_start_date` | `str` | Always | Warmup-aligned start date actually used for equity/metrics. |
| `required_history_days` | `int` | Always | Strategy warmup requirement from `get_required_history()`. |
| `warmup_history_shortfall_days` | `int` | Always | Missing warmup bars count when history is insufficient. |
| `end_date` | `str` | Always | Requested/derived end date input. |
| `degradation_analysis` | `dict` | Always | Robustness diagnostics over the backtest horizon. |
| `benchmark_analysis` | `dict` | Always | Benchmark-relative analysis block and availability/reason info. |
| `cost_sensitivity` | `dict` | Always | Transaction-cost sensitivity diagnostics. |
| `execution_quality` | `dict` | Always | Simulation execution/TCA summary fields. |
| `regime_series` | `pd.Series` | Always | Retrospective (full-series) regime labels for visualization/export. |
| `regime_series_pit` | `pd.Series` | Always | Point-in-time regime labels for attribution-safe segmentation. |
| `per_ticker_regime_series` | `Dict[str, pd.Series]` | Conditional | Present as `{}` when disabled; populated when `risk.per_ticker_regime=true`. |
| `attribution` | `Dict[str, dict]` | Conditional | Non-empty for orchestration runs with attribution history. |
| `positions_history` | `pd.DataFrame` | Always | Position weights/shares history by date (may be empty in edge cases). |
| `asset_exclusions` | `List[AssetExclusionRecord]` | Always | Structured universe/data exclusion diagnostics. |

---

## Warmup-aligned timeline fields

### `required_history_days`
Bars requested by the strategy before signal generation can begin.

### `effective_start_date`
Requested `start_date` advanced on the available market index by `required_history_days`.
All portfolio return metrics are aligned to this date.

### `warmup_history_shortfall_days`
Positive when the available historical data cannot fully satisfy warmup requirements.
The engine still runs and reports this shortfall instead of failing hard.

---

## Regime outputs

Two regime series are produced:

1. **`regime_series`**: retrospective full-series decode (good for visualization).
2. **`regime_series_pit`**: point-in-time decode (good for attribution/performance conditioning).

### Regime performance blocks in `metrics`

- `metrics['regime_performance']`: computed against **PIT** regime labels.
- `metrics['regime_performance_retrospective']`: computed against retrospective labels.

Each regime block typically includes per-regime stats such as:

- `cagr`
- `volatility`
- `sharpe`
- `max_drawdown`
- `days`
- `fraction`

---

## Attribution block

`results['attribution']` is intended for `strategy_orchestration` runs and reports per-sub-strategy
contributions/characteristics. Expected sub-keys include metrics such as:

- `total_return`
- `annualized_return`
- `volatility`
- `max_drawdown`
- `sharpe_ratio`
- `weight_fraction`
- `contribution_to_portfolio_return`

If orchestration attribution is not available for a run, the block may be empty.

---

## Asset exclusions

`results['asset_exclusions']` is a list of structured records with fields:

- `ticker`
- `date`
- `reason` in `{'insufficient_history','stale_data','missing_file','data_error'}`
- `available_bars`
- `required_bars`
- `detail`

Use this list to distinguish expected warmup exclusions from true upstream data problems.

---

## Execution quality / TCA fields

Backtest outputs intentionally **do not** include live-fill observability fields:

- `fill_rate`
- `avg_fill_size`

Those fields are live/broker telemetry concepts and are omitted in simulation summaries.

Backtest execution-quality output retains simulation-relevant fields (for example slippage and
shortfall summaries) provided by `TransactionCostAnalysis.summarize(...)`.

---

## Practical interpretation notes

- Use `effective_start_date` (not requested `start_date`) when comparing runs for fairness.
- Use `regime_performance` (PIT) for attribution-safe regime analytics.
- Treat `regime_performance_retrospective` as a descriptive/visual lens, not live-tradable labels.
- Inspect `asset_exclusions` whenever realized universe size seems lower than expected.
