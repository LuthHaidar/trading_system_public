# Risk Controls and Portfolio Construction

This document describes the runtime risk stack used by the backtest and live pathways.

## Components

The risk pipeline is implemented in `risk/` and runs in this order:

1. **Position sizing** (`PositionSizer`)
2. **Portfolio optimization** (`PortfolioOptimizer`, optional)
3. **Risk constraints** (`RiskConstraints`)
4. **Final target-vol scaling** (when `position_sizing_method: target_vol`)

See also:
- `risk/position_sizer.py`
- `risk/optimizer.py`
- `utils/config_schema.py` (`RiskConfig` validation)

---

## PositionSizer

`PositionSizer` converts raw strategy weights into sized weights.

Supported methods:

- `equal`
- `volatility`
- `inverse_volatility`
- `kelly`
- `target_vol`

### Kelly sizing

`kelly_fraction` is schema-validated in `(0.0, 1.0]` and defaults to `0.25` (quarter-Kelly).
This is a safety-oriented default designed to reduce over-sizing risk from noisy estimates.

Key knobs:

- `kelly_lookback`
- `kelly_max_weight`
- `kelly_min_trades` / estimator guards

### Target volatility sizing

When `position_sizing_method: target_vol`, the system scales exposure toward configured
`target_portfolio_volatility` subject to caps. A final scaling pass preserves the total exposure
limit implied by cash reserve and other constraints.

---

## PortfolioOptimizer

`PortfolioOptimizer` is optional and runs after sizing.

Optimization methods:

- `max_sharpe`
- `min_variance`
- `risk_parity`

Config controls include:

- `optimizer_max_weight`
- `optimizer_min_weight`
- `risk_free_rate`

### Covariance estimators

Supported estimators (`risk.covariance_estimator`):

- `sample`
- `ledoit_wolf` (`ledoitwolf`, `lw` aliases)
- `shrinkage` (`diagonal_shrinkage` alias)
- `exponential` (`ew` alias)
- `exponential_shrinkage` (`ew_shrinkage` alias)

Related parameters:

- `covariance_shrinkage`
- `covariance_ew_halflife`
- `covariance_jitter`

`exponential_shrinkage` computes EW covariance first, then applies diagonal shrinkage.

---

## RiskConstraints

`RiskConstraints` applies post-sizing safety constraints in this order:

1. Per-position min/max limits
2. Max position count
3. Correlation limits (if return history provided)
4. Diversification floor scaling (if configured)
5. Minimum cash reserve enforcement

Primary settings:

- `max_position_size`
- `min_position_size`
- `max_positions`
- `max_correlation`
- `min_diversification_score`
- `diversification_scale_floor`
- `min_cash_reserve`

This ordering is intentional so final weights remain compliant with portfolio-level limits.

---

## Configuration validation

`RiskConfig` in `utils/config_schema.py` validates risk fields and rejects unknown keys
(`extra='forbid'`) in the `risk` block.

Examples of validated fields include:

- `position_sizing_method`
- `max_position_size`
- `min_position_size`
- `kelly_fraction`
- `covariance_estimator`
- `covariance_ew_halflife`
- `per_ticker_regime`

Use `config/config.yaml` as the canonical template for supported options.
