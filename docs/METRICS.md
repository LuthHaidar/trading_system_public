# Metrics Reference

## Regime Labels in Backtests

`regime_series` is generated retrospectively at backtest end.

- Preferred path: full-series HMM decode (single fit/predict pass), which may differ from live streaming labels.
- Fallback path: expanding-window heuristic (`regime_based_scaler`) using information up to each date.

Use regime metrics for analytical segmentation, not as a claim of live point-in-time label parity.

## Regime-Conditional Performance

The `regime_performance` block reports, per available regime (`bull`, `neutral`, `bear`):

- `cagr`
- `volatility`
- `sharpe`
- `max_drawdown`
- `days`
- `fraction`

## Transaction Cost Fill Rate

In simulation mode, `TCA.fill_rate` is **not observed from real fills**.

- Backtests should emit `None` and render as `N/A (simulation)`.
- Live mode can report numeric fill-rate from broker execution outcomes.
