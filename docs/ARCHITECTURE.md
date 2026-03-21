# Architecture Overview

This document explains how major packages collaborate during backtesting and live execution.

## High-level components

- **Entrypoints**
  - `backtest.py`: CLI orchestration for research and report generation.
  - `main.py`: CLI orchestration for IBKR-connected live/paper execution.
- **Strategies (`strategies/`)**
  - Concrete strategy classes output target portfolio weights by date.
  - Factory + registry in `strategies/__init__.py` keeps strategy names centralized.
- **Data (`data/`)**
  - `DataManager` loads ticker CSV data, normalizes schema, caches frames, and can fetch missing data.
- **Risk (`risk/`)**
  - Position sizing (`PositionSizer`), portfolio constraints (`RiskConstraints`), optimizer (`PortfolioOptimizer`), and live risk gates (`LiveRiskManager`).
- **Execution (`backtesting/execution.py`, `broker/ibkr_client.py`)**
  - Backtesting execution simulates fills and costs.
  - IBKR client handles real brokerage interactions.
- **Portfolio & metrics (`backtesting/`)**
  - Portfolio state, trade accounting, equity curve maintenance, and performance statistics.
- **Operational utilities (`utils/`)**
  - Logging, currency conversion, backup/drift guards, and optional SQLite persistence.

## Backtesting architecture

```text
backtest.py
  -> backtesting.engine.run_backtest_from_config(...)
      -> create strategy from registry
      -> load data with DataManager
      -> iterate rebalance dates
          -> strategy.generate_signals(...)
          -> PositionSizer.size_positions(...)
          -> RiskConstraints.apply_constraints(...)
          -> (optional) PortfolioOptimizer.optimize(...)
          -> ExecutionEngine.execute_order(...)
          -> Portfolio.update_positions(...)
      -> compute PerformanceMetrics
      -> optional: benchmark, TCA, validation summaries
```

### Key outputs

- Equity curve series
- Trade list with cost breakdown
- Summary metrics (return, CAGR, vol, Sharpe, drawdown, etc.)
- Optional benchmark-relative stats (alpha/beta/tracking error)

## Live architecture

```text
main.py (LiveTradingEngine)
  -> IBKRClient.connect()
  -> DataManager.update_all()/load_multiple()
  -> IBKR positions + market prices + account summary
  -> strategy.generate_signals(...)
  -> PositionSizer + RiskConstraints + (optional) optimizer
  -> weight->shares conversion (with FX handling)
  -> IBKRClient.reconcile_positions(...)
  -> LiveRiskManager checks + drift monitor
  -> IBKRClient.execute_rebalance(...) [or dry-run skip]
  -> backup + audit + metrics persistence
```

### Live safety and reliability controls

- Circuit-breaker style checks via `LiveRiskManager`.
- Duplicate-order prevention in IBKR adapter.
- Position-drift detection before and after recovery.
- Local state backups and optional SQLite snapshots/events.
- Scheduler loop auto-reconnect and health logging.

## Configuration model

Two YAML files drive behavior:

1. `config/config.yaml`
   - portfolio capital/currency
   - execution timing and rebalance cadence
   - cost/slippage model
   - risk sizing/constraints/optimizer
   - data and broker connectivity
   - live risk + operations + data platform paths
2. `config/strategies.yaml`
   - strategy parameters for each registered strategy

## Extensibility points

- Add new strategy class in `strategies/` and register it in `strategies/__init__.py`.
- Implement alternate metrics sink by extending `MetricsStoreAdapter`.
- Adjust slippage/cost realism in execution config without code changes.
- Add new risk checks in `risk/live_risk_manager.py` and wire into live engine.
