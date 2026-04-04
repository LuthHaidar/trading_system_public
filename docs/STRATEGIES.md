# Strategy Reference

This document covers the strategies registered in `strategies/__init__.py` and configured via
`config/strategies.yaml`.

All strategies output long-only target weights. Final clipping/normalization and portfolio-level
risk controls are applied downstream by sizing/constraints/optimizer layers.

---

## Common strategy contract

Each concrete strategy implements:

- `generate_signals(date, data, current_positions) -> Dict[str, float]`
- `get_required_history() -> int`

`get_required_history()` is consumed by the backtest engine to compute warmup-aligned
`effective_start_date`.

---

## 1) `momentum`

**Class:** `MomentumStrategy` (`strategies/momentum.py`)

**Purpose:** Cross-sectional momentum ranking.

### Required config (with defaults)
- `lookback` (default `126`)
- `skip_recent` (default `21`)
- `n_positions` (default `5`)
- `rebalance_frequency` (default `21`)
- `min_momentum` (default `0.0`)
- `weight_method` (`equal` or `proportional`, default `equal`)

### Warmup
- `get_required_history() = lookback + skip_recent + 10`

### Notes / limitations
- Skips symbols without sufficient close history for the current date.
- `proportional` weighting uses positive momentum magnitudes.

---

## 2) `dual_momentum`

**Class:** `DualMomentumStrategy` (`strategies/momentum.py`)

**Purpose:** Relative + absolute momentum with cash fallback.

### Required config (with defaults)
- `lookback` (default `252`)
- `n_positions` (default `3`)
- `absolute_momentum_threshold` (default `0.0`)
- `cash_ticker` (default `SHY`)

### Warmup
- `get_required_history() = lookback + 10`

### Notes / limitations
- If no risky assets pass absolute momentum, allocates to `cash_ticker` when present in data.

---

## 3) `mean_reversion`

**Class:** `MeanReversionStrategy` (`strategies/mean_reversion.py`)

**Purpose:** Bollinger-band mean reversion.

### Required config (with defaults)
- `window` (default `20`)
- `num_std` (default `2`)
- `holding_period` (default `5`)
- `exit_zscore` (default `0`)
- `max_positions` (default `5`)

### Warmup
- `get_required_history() = window + 10`

### Notes / limitations
- Entry is oversold-only (`price < lower_band`) and long-only.

---

## 4) `rsi_mean_reversion`

**Class:** `RSIMeanReversionStrategy` (`strategies/mean_reversion.py`)

**Purpose:** RSI threshold mean reversion.

### Required config (with defaults)
- `rsi_period` (default `14`)
- `oversold_threshold` (default `30`)
- `overbought_threshold` (default `70`)
- `max_positions` (default `5`)

### Warmup
- `get_required_history() = rsi_period + 10`

### Notes / limitations
- Long-only implementation; exits on overbought threshold.

---

## 5) `statistical_arbitrage`

**Class:** `StatisticalArbitrageStrategy` (`strategies/alpha_expansion.py`)

**Purpose:** Cross-sectional short-term reversal score.

### Required config (with defaults)
- `lookback` (default `20`)
- `ranking_window` (default `5`)
- `n_positions` (default `3`)

### Warmup
- `get_required_history() = lookback + ranking_window`

### Notes / limitations
- Equal-weights selected symbols after ranking.

---

## 6) `factor_model`

**Class:** `FactorModelStrategy` (`strategies/alpha_expansion.py`)

**Purpose:** Composite rank from price-derived factor proxies.

### Required config (with defaults)
- `lookback` (default `126`)
- `n_positions` (default `5`)
- `factor_weights` mapping with keys:
  - `momentum` (default `0.5`)
  - `low_vol` (default `0.3`)
  - `quality_proxy` (default `0.2`)

### Warmup
- `get_required_history() = lookback`

### Notes / limitations
- Uses positive-score normalization; falls back to equal-weight if selected raw scores are non-positive.

---

## 7) `volatility_trading`

**Class:** `VolatilityTradingStrategy` (`strategies/alpha_expansion.py`)

**Purpose:** Risk-on / defensive basket allocation by realized volatility (optionally HMM).

### Required config (with defaults)
- `benchmark_ticker` (default `SPY`)
- `vol_window` (default `20`)
- `vol_threshold` (default `0.20`)
- `risk_on_tickers` (default `[]`)
- `defensive_tickers` (default `[]`)
- `use_hmm_regime` (default `false`)

### Warmup
- `get_required_history() = vol_window + 5`

### Notes / limitations
- If HMM is enabled and available, blends baskets via regime probabilities.
- If HMM inference is unavailable at runtime, falls back to threshold logic.

---

## 8) `strategy_orchestration`

**Class:** `StrategyOrchestrationStrategy` (`strategies/strategy_orchestration.py`)

**Purpose:** Ensemble of enabled sub-strategies with condition/regime-aware weighting.

### Required config (with defaults)
- `base_weights` (default `{}`; auto-equal if omitted)
- `allowed_conditions` (default `{}`)
- `regime_multipliers` (default `{}`)
- `benchmark_ticker` (default `SPY`)
- `performance_lookback` (default `20`)
- `min_weight_threshold` (default `0.0`)
- `regime_vol_threshold` (default `0.25`)
- `regime_bear_return_threshold` (default `0.0`)
- `hmm_n_states` (default `3`)
- `hmm_covariance_type` (default `full`)
- `use_per_ticker_regime` (optional, default behavior is disabled unless explicitly enabled)

### Warmup
- `get_required_history() = max(get_required_history() for each enabled sub-strategy)`
- Returns `0` when no sub-strategies are enabled.

### Notes / limitations
- Only strategies with `enabled: true` in `strategies.yaml` are included.
- Regime labels are constrained to `bull|neutral|bear`.
- Emits attribution history consumed by engine-level attribution summaries.
- Optional per-ticker regime gating/scaling is supported when provided by engine and enabled in config.

---

## Data requirements and practical caveats

- Strategies generally require a `Close` column in each ticker DataFrame.
- Universe members without sufficient history are skipped for that date; engine-level diagnostics are
  surfaced in `asset_exclusions`.
- Output signals are long-only target weights; execution realism (costs, fills, constraints) is
  handled by backtesting/live execution layers.
