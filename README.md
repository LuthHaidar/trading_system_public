# Trading System

A Python trading platform for **strategy research (backtesting)** and **IBKR-connected live execution** with configurable risk controls, transaction-cost modeling, and optional SQLite-based metrics persistence.

## What this project does

- Runs end-to-end backtests from YAML config (`backtest.py` + `backtesting/` package).
- Connects to Interactive Brokers via `ib_insync` for scheduled live trading (`main.py` + `broker/`).
- Supports registered strategy families: momentum, dual momentum, mean reversion, RSI mean reversion, statistical arbitrage, factor model, volatility trading, and strategy orchestration.
- Applies position sizing, risk constraints, and optional portfolio optimization.
- Handles multi-currency portfolios with FX conversion support.
- Persists audit events, trade journals, state snapshots, and metrics to SQLite (optional).

---

## Repository layout

```text
trading_system/
├── main.py                  # Live trading entrypoint
├── backtest.py              # Backtesting CLI entrypoint
├── config/
│   ├── config.yaml          # System-wide runtime config
│   └── strategies.yaml      # Per-strategy parameters and enable flags
├── strategies/              # Signal generation logic + strategy registry
├── backtesting/             # Simulation engine, execution model, metrics, validation
├── broker/                  # IBKR adapter (connectivity + order/position operations)
├── risk/                    # Position sizing, constraints, optimizer, live risk manager
├── data/                    # CSV data management + yfinance scraper
├── utils/                   # Logging, currency helpers, ops utilities, data platform storage
├── tests/                   # Pytest suite (regressions + feature coverage)
└── requirements.txt
```

---

## Setup

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Verify local market data

The system expects ticker CSVs in `data/` (e.g., `SPY_daily_data.csv`).
If missing, `DataManager` will attempt to download data via yfinance on demand.

### 3) Configure runtime files

- `config/config.yaml` controls portfolio, execution, risk, cost, broker, operations, and data-platform settings.
- `config/strategies.yaml` controls strategy-specific parameters.

#### Data loading behavior (`data.use_adjusted_close`)

By default, `DataManager` substitutes `Adj Close` into `Close` when both columns are present
(`data.use_adjusted_close: true`). This keeps backtests closer to a total-return series by
accounting for splits/dividends reflected in adjusted prices.

If you want raw close prices instead, set:

```yaml
data:
  use_adjusted_close: false
```

This affects signal generation, returns, and all downstream performance metrics.

---

## Quickstart

## Backtest (recommended first step)

```bash
python backtest.py --strategy momentum --start 2020-01-01 --end 2023-12-31
```

Useful options:

- `--tickers SPY QQQ IWM`
- `--output-dir results`
- `--no-plot`
- `--no-save`
- `--mc-seed 42` (set deterministic Monte Carlo bootstrap seed)
- `--no-mc-plot` (skip Monte Carlo projection plot generation)
- `--walk-forward` (run fixed-config walk-forward analysis)
- `--wf-train-days 252 --wf-oos-days 63 --wf-step-days 63`
- `--wf-anchored` (anchored expanding train windows for walk-forward)
- `--strategies-config config/strategies.yaml`
- `--strategy-override '{"lookback": 126}'`
- `--config-override '{"risk": {"max_position_size": 0.15}}'`
- `--list-runs --runs-limit 20` (query persisted run metrics from SQLite)
- `--compare-runs <run_id_1> <run_id_2>` (side-by-side metrics comparison)
- `--journal-run <run_id> --journal-limit 200` (trade decision journal: entry/exit/increase/decrease + signal confidence)

### Parameter fine-tuning / sweeps

Use the built-in sweep mode to evaluate multiple strategy parameter sets in one run:

```bash
python backtest.py \
  --strategy momentum \
  --start 2021-01-01 \
  --end 2023-12-31 \
  --sweep lookback=63,126,252 \
  --sweep n_positions=3,5,8 \
  --optimize-metric sharpe_ratio
```

Sweep runs print a ranked leaderboard and save `results/tuning_results.csv` with each run's override + core metrics.

You can also use automatic numeric ranges in `--sweep` values via `start:end[:step]` syntax.
For example, `--sweep lookback=63:252:63` expands to `63,126,189,252`.

Artifacts produced (when saving is enabled):

- `results/trades.csv`
- `results/metrics.txt`
- `results/equity_curve.csv`
- `results/backtest_plot.png`
- `results/monte_carlo_plot.png`
- `results/monte_carlo_data.npz`
- `results/walk_forward_plot.png` (when `--walk-forward` and plotting enabled)
- `results/backtest.log`

The CLI prints a persisted `Run ID` for each backtest. Use that `run_id` for `--compare-runs` and `--journal-run`.

## Live run (dry-run safety first)

```bash
python main.py --paper-trading --strategy momentum --once --dry-run
```

Live-mode options:

- `--paper-trading` (IBKR port 7497)
- `--live` (IBKR port 7496; prompts for `CONFIRM` unless `--dry-run`)
- `--once` (single execution instead of scheduler)
- `--execution-time HH:MM` (for scheduled mode)
- `--strategies-config path/to/strategies.yaml` (custom strategy-config file for live runs)
- `--pre-flight-backtest` (run a short pre-flight backtest before IBKR connect)
- `--pre-flight-days N` (lookback window for pre-flight backtest; default: 90)

> Tip: start with paper + dry-run and inspect logs/audit tables before real execution.

---

## Core execution flow

### Backtesting flow

1. Load config and instantiate strategy from registry.
2. Load historical data via `DataManager`.
3. Generate target weights from strategy signals.
4. Apply position sizing and risk constraints.
5. Optionally optimize final portfolio weights.
6. Simulate fills and transaction costs with execution engine.
7. Record equity curve/trades and compute performance metrics.
8. Optionally run validation and transaction-cost analysis helpers.

### Live flow

1. Connect to IBKR and fetch account/position state.
2. Refresh local data and load historical windows for strategy features.
3. Generate target weights using current market context.
4. Apply sizing/constraints/optimizer.
5. Convert target weights to target share counts.
6. Reconcile current vs target positions and build order list.
7. Run circuit-breaker checks and drift checks.
8. Execute rebalance (or preview only with `--dry-run`).
9. Persist backups, audit events, and metrics (if enabled).

---

## Strategy system

Strategies are created through a registry in `strategies/__init__.py`.
Supported names include:

- `momentum`, `dual_momentum`
- `mean_reversion`
- `statistical_arbitrage`, `factor_model`, `volatility_trading`
- `strategy_orchestration` (ensemble of all `enabled: true` sub-strategies with regime/condition-aware weighting)

See `docs/STRATEGIES.md` for full per-strategy decision logic, indicators, and config explanations.
For research-only helpers (for example efficient frontier analysis), see `docs/RESEARCH.md`.

When using `strategy_orchestration`:

- Set `strategy_orchestration.enabled: true` and run `--strategy strategy_orchestration`.
- Enable sub-strategies in `config/strategies.yaml` via each strategy's `enabled` flag.
- `base_weights`, `allowed_conditions`, and `regime_multipliers` are applied at runtime.

To switch strategy:

1. Update or verify parameters in `config/strategies.yaml`.
2. Run with `--strategy <name>` in `backtest.py` or `main.py`.

---

## Risk, costs, and realism

### Risk controls

- Position sizing methods: `equal`, `volatility`, `inverse_volatility`, `kelly`, `target_vol` (legacy alias: `risk_parity`).
- Constraints include max/min position sizing, max positions, reserve cash, sector/correlation limits.
- Live risk manager supports stale-data checks, order-reject thresholds, daily loss limits, and trailing stop-loss behavior.

### Transaction cost model

`config.yaml` supports composable cost components:

- broker commissions (US and non-US)
- regulatory fees
- exchange fees
- market taxes by market suffix
- FX conversion costs
- spread + slippage heuristics/parameterized model

---

## Data platform (optional persistence)

When `data_platform.enabled: true`, the system writes to SQLite:

- audit events
- trade rows
- trade decision journal rows (signal date, decision type, signal strength, confidence, target/current weights/shares)
- equity snapshots
- state snapshots
- backtest run summaries (structured run-level metrics for cross-run comparison)
- metrics time series (via metrics store adapter)

Default DB paths:

- `state/trading_audit.db`
- `state/metrics.db`

---

## Testing and diagnostics

Run the pytest suite:

```bash
pytest -q
```

Focused smoke/regression subsets:

```bash
pytest -q tests/test_config_validation.py tests/test_immediate_bugs.py tests/test_ibkr_contracts.py
```
