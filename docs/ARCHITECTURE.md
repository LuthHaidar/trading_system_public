# Architecture Overview

This document describes the current backtest/live architecture and how core modules interact.

---

## 1) Core modules

- **Entrypoints**
  - `backtest.py`: research/backtest CLI orchestration, walk-forward, sweep, reporting.
  - `main.py`: live/paper IBKR orchestration and scheduler.
- **Strategy layer (`strategies/`)**
  - Strategy registry/factory and concrete signal generators.
  - Optional ensemble strategy (`strategy_orchestration`) with attribution history.
- **Data layer (`data/`)**
  - `DataManager` CSV loading, freshness checks, optional yfinance fill.
- **Backtesting core (`backtesting/`)**
  - `engine.py`: simulation loop, warmup alignment, diagnostics, benchmark analytics.
  - `portfolio.py`: holdings/cash/trade accounting.
  - `execution.py`: simulation fills and transaction-cost components.
  - `metrics.py`: run-level performance statistics.
  - `validation.py`: Monte Carlo and validation helpers.
  - `walk_forward.py`: temporal split generation and per-fold evaluation.
- **Risk layer (`risk/`)**
  - `PositionSizer`, `RiskConstraints`, optional `PortfolioOptimizer`.
  - `HMMRegimeDetector` and heuristic regime helpers.
  - Live risk manager/circuit-breaker logic.
- **Utilities (`utils/`)**
  - Logging, FX conversion, config schema validation, persistence adapters.

---

## 2) Backtest execution architecture

```text
backtest.py
  -> run_backtest_from_config(...)
     -> validate_main_config / validate_strategies_config
     -> create strategy
     -> BacktestEngine.run(...)
        -> load data (+ warmup horizon)
        -> compute effective_start_date from strategy.get_required_history()
        -> iterate trading dates
           -> strategy.generate_signals(...)
           -> PositionSizer
           -> (optional) PortfolioOptimizer
           -> RiskConstraints
           -> ExecutionEngine rebalance
           -> Portfolio updates + equity snapshots
        -> benchmark analysis + metrics + validation summaries
        -> regime outputs (retrospective + PIT)
        -> optional per-ticker regimes
        -> optional orchestration attribution summary
```

### Warmup-aware timeline

`BacktestEngine` computes:

- `required_history_days`
- `effective_start_date`
- `warmup_history_shortfall_days`

All run-level performance stats are aligned to `effective_start_date`.

### Asset exclusion diagnostics

Filtered/unusable symbols are recorded as structured `AssetExclusionRecord` entries and returned in
`results['asset_exclusions']`.

---

## 3) Monte Carlo architecture

Two Monte Carlo paths are supported:

1. **Trade bootstrap** (default when usable trade PnL exists)
   - `ValidationSuite.monte_carlo_trade_simulation(...)`
   - modes: `shuffle` / `resample`
2. **Return block bootstrap**
   - `ValidationSuite.monte_carlo_simulation(...)`

CLI routing is controlled by `--mc-mode`:

- `trade_shuffle`
- `trade_resample`
- `return_block`

`compute_monte_carlo_summary(...)` chooses the path, then plotting/saving/reporting use a shared
normalized format and method labels.

---

## 4) Walk-forward architecture

Walk-forward supports both fixed-parameter and per-fold sweep selection.

```text
run_walk_forward(...)
  -> generate chronological train/OOS splits
  -> if sweep_definitions provided:
       - evaluate candidates on train block only
       - rank and select best override per fold
  -> run OOS with frozen fold override
  -> aggregate fold metrics
  -> compute parameter_stability + best_overrides_per_fold
```

CLI:

- `--walk-forward`
- `--wf-sweep ...` (mutually exclusive with global `--sweep`)

---

## 5) Regime architecture

The system separates descriptive and attribution-safe regime outputs:

- `regime_series`: retrospective full-series decode.
- `regime_series_pit`: point-in-time decode.

For orchestration and diagnostics:

- heuristic and HMM regimes are constrained to `bull|neutral|bear`.
- optional per-ticker decoding is available (`risk.per_ticker_regime=true`).

HMM robustness includes:

- refit by elapsed days (`refit_interval_days`),
- state-separation reliability checks,
- safe fallback behavior for unsupported/unreliable fits.

---

## 6) Ensemble attribution architecture

`StrategyOrchestrationStrategy` records per-date sub-strategy allocations.
`BacktestEngine` consumes this history post-run to produce `results['attribution']` with per
sub-strategy return/volatility/drawdown/sharpe/weight/contribution statistics.

---

## 7) Live architecture (IBKR)

```text
main.py (LiveTradingEngine)
  -> connect IBKR
  -> refresh/load data
  -> pull account + positions + prices
  -> strategy signals
  -> size / optimize / constrain
  -> convert target weights to shares
  -> risk/circuit-breaker gates
  -> execute or dry-run
  -> backup + optional SQLite audit/metrics
```

Safety controls include stale-data checks, order reject thresholds, drift detection, and scheduler
health/recovery loops.

---

## 8) Configuration model

Two validated YAML inputs:

1. `config/config.yaml`
   - portfolio/execution/risk/cost/data/live-risk/data-platform
2. `config/strategies.yaml`
   - per-strategy parameters and enable flags

`run_backtest_from_config` performs loading, override merge, and schema validation before engine
construction.

---

## 9) Extensibility points

- Add a strategy by registering in `strategies/__init__.py`.
- Add validation/analytics in `backtesting/validation.py` and report hooks in `backtest.py`.
- Extend optimizer/constraints via `risk/` modules and config schema.
- Add persistence sinks through utility adapters in `utils/`.
