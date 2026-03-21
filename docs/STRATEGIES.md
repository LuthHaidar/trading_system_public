# Strategy Reference

This document covers the strategies currently registered in `strategies/__init__.py` and configurable in `config/strategies.yaml`.

All strategies output long-only target weights. Final clipping/normalization behavior is handled by base strategy utilities and portfolio construction layers.

---

## 1) `momentum`

**Implementation:** `MomentumStrategy` (`strategies/momentum.py`)

### Inputs / features
- Trailing total return over `lookback`
- Optional skip window `skip_recent`

### Rules
1. Rebalance only when `rebalance_frequency` days elapsed.
2. Compute momentum per ticker.
3. Keep tickers where `momentum >= min_momentum`.
4. Rank descending and keep top `n_positions`.
5. Weight by `weight_method` (`equal` or `proportional`).

### Config keys
- `enabled`
- `lookback`
- `skip_recent`
- `n_positions`
- `rebalance_frequency`
- `min_momentum`
- `weight_method`

---

## 2) `dual_momentum`

**Implementation:** `DualMomentumStrategy` (`strategies/momentum.py`)

### Rules
1. Rank candidates by trailing return (`lookback`) excluding `cash_ticker`.
2. Apply absolute filter `>= absolute_momentum_threshold`.
3. Keep top `n_positions` and equal-weight.
4. If nothing passes, allocate to `cash_ticker` when available.

### Config keys
- `enabled`
- `lookback`
- `n_positions`
- `absolute_momentum_threshold`
- `cash_ticker`

---

## 3) `mean_reversion`

**Implementation:** `MeanReversionStrategy` (`strategies/mean_reversion.py`)

### Inputs / features
- Bollinger bands from `window` and `num_std`
- Z-score vs rolling mean

### Rules
- Enter long when oversold (`price < lower_band`) up to `max_positions`.
- Exit on mean reversion (`zscore >= exit_zscore`), upper-band touch, or `holding_period` expiry.

### Config keys
- `enabled`
- `window`
- `num_std`
- `holding_period`
- `exit_zscore`
- `max_positions`

---

## 4) `rsi_mean_reversion`

**Implementation:** `RSIMeanReversionStrategy` (`strategies/mean_reversion.py`)

### Inputs / features
- RSI over `rsi_period`

### Rules
- Entry: ticker RSI <= `oversold_threshold`.
- Exit: ticker RSI >= `overbought_threshold`.
- Position count constrained by `max_positions`.
- Active names are equally weighted.

### Config keys
- `enabled`
- `rsi_period`
- `oversold_threshold`
- `overbought_threshold`
- `max_positions`

---

## 5) `statistical_arbitrage`

**Implementation:** `StatisticalArbitrageStrategy` (`strategies/alpha_expansion.py`)

### Inputs / features
- Cross-sectional mean-reversion score from `lookback` returns and `ranking_window`

### Rules
1. Score tickers.
2. Rank descending.
3. Keep top `n_positions`.
4. Equal-weight selected names.

### Config keys
- `enabled`
- `lookback`
- `ranking_window`
- `n_positions`

---

## 6) `factor_model`

**Implementation:** `FactorModelStrategy` (`strategies/alpha_expansion.py`)

### Inputs / features
- Price-derived proxies (`momentum`, `low_vol`, `quality_proxy`) over `lookback`
- Weighted by `factor_weights`

### Rules
1. Score each ticker.
2. Keep top `n_positions`.
3. Use positive scores for weighting; fallback to equal-weight if all non-positive.

### Config keys
- `enabled`
- `lookback`
- `n_positions`
- `factor_weights.momentum`
- `factor_weights.low_vol`
- `factor_weights.quality_proxy`

---

## 7) `volatility_trading`

**Implementation:** `VolatilityTradingStrategy` (`strategies/alpha_expansion.py`)

### Inputs / features
- Realized annualized volatility of `benchmark_ticker` over `vol_window`

### Rules
- Default mode: if realized vol > `vol_threshold`, equal-weight `defensive_tickers`; else equal-weight `risk_on_tickers`.
- Optional HMM mode (`use_hmm_regime: true`): blend risk-on vs defensive baskets continuously from posterior probs.

### Config keys
- `enabled`
- `benchmark_ticker`
- `vol_window`
- `vol_threshold`
- `risk_on_tickers`
- `defensive_tickers`
- `use_hmm_regime`

---

## 8) `strategy_orchestration`

**Implementation:** `StrategyOrchestrationStrategy` (`strategies/strategy_orchestration.py`)

### Inputs / features
- Enabled sub-strategies from strategies config (excluding itself)
- Regime classification (`bull`/`neutral`/`bear`)
- Per-substrategy trailing performance weighting

### Rules
1. Generate sub-strategy target signals.
2. Compute dynamic strategy weights from `base_weights`, recent performance, and `regime_multipliers`.
3. Apply condition gating via `allowed_conditions`.
4. Ensemble and normalize ticker-level output.
5. Drop tiny weights under `min_weight_threshold`.

### Config keys
- `enabled`
- `base_weights`
- `allowed_conditions`
- `regime_multipliers`
- `benchmark_ticker`
- `performance_lookback`
- `min_weight_threshold`
- `hmm_n_states`
- `hmm_covariance_type`

---

## Data requirements

- Most strategies require at minimum a `Close` column per ticker.
- Tickers with insufficient lookback history are skipped for that date.
