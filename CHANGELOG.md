# Changelog

## 2026-02-20

### Breaking Default Changes
- Changed `risk.covariance_estimator` default from `sample` to `ledoit_wolf` in `config/config.yaml`. This changes optimizer covariance estimates and can produce different backtest weights/results for the same data.
- Renamed `risk.position_sizing_method` preferred value from `risk_parity` to `inverse_volatility` for `PositionSizer`. The legacy `risk_parity` alias remains accepted but is deprecated and logs a warning.

### Added
- Added startup/runtime config validation with Pydantic models (`utils/config_schema.py`) and wired validation into backtest and live entrypoints.
- Added strategy-stanza shape validation for `config/strategies.yaml`.

### Docs
- Added `docs/CONFIGURATION.md` for validated configuration keys and behavior.
- Updated `README.md` testing section to use pytest-based workflows.
