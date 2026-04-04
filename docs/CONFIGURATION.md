# Configuration Reference

This project validates `config/config.yaml` at startup using Pydantic models in `utils/config_schema.py`.

## Top-level sections

- `portfolio`
- `execution`
- `risk`
- `data`
- `ibkr`
- Optional passthrough dict sections: `costs`, `logging`, `live_risk`, `operations`, `data_platform`

Unknown top-level sections are rejected.

## Core validated keys

### portfolio
- `initial_capital` (`float`, required)
- `currency` (`str`, required)

### execution
- `rebalance_timeframe` (`str`, required)
- `timing` (`str`, required)
- `progress_log_interval` (`int`, default `50`)
- `hold_weight_epsilon` (`float`, default `1e-4`)
- `invested_sleeve_drift_warn` (`float`, default `0.05`)
- `cash_drift_warn` (`float`, default `0.01`)
- `drift_warn_min_capital` (`float`, default `0.0`)
- `uninvested_cash_tolerance` (`float`, default `0.005`)
- `cost_aware_fee_bps` (`float`, default `0.0`)
- `cost_aware_spread_bps` (`float`, default `0.0`)

### risk
- `position_sizing_method` (`str`, required)
- `max_position_size` (`float`, required)
- `min_position_size` (`float`, required)
- `volatility_window` (`int`, default `60`)

### data
- `data_dir` (`str`, required)
- `max_stale_price_days` (`int`, default `5`)
- `fx_cache_staleness_days` (`int`, default `1`)
- `cache_size` (`int | null`, optional)

### ibkr
- `host` (`str`, required)
- `port` (`int`, required)
- `client_id` (`int`, required)
- `timeout` (`int`, default `60`)
- `strict_universe_validation` (`bool`, default `true`)
- `contract_overrides` (`dict[str, ContractOverride]`, default `{}`)
  - Optional explicit per-ticker IBKR contract mapping used by `IBKRClient.create_contract`.
  - Each override entry supports:
    - `symbol` (`str`, required)
    - `exchange` (`str`, required)
    - `currency` (`str`, required)
    - `primary_exchange` (`str`, optional)
  - Unknown fields are rejected by config validation (`extra='forbid'`).
  - Example:
    ```yaml
    ibkr:
      contract_overrides:
        SPY:
          symbol: SPY
          exchange: SMART
          currency: USD
          primary_exchange: ARCA
        HSBC.L:
          symbol: HSBC
          exchange: LSE
          currency: GBP
    ```

## Validation behavior

- `backtest.py`: validates after merging `--config-override`.
- `backtesting/engine.py`: validates in `run_backtest_from_config` and validates each `config/strategies.yaml` stanza is a mapping.
- `main.py`: validates before strategy execution and IBKR connection, and validates strategies YAML structure when loading strategies (default `config/strategies.yaml`, override via `--strategies-config`).

On validation failure, the process exits with a clear error message.


### Strategies-level contract definitions (optional)

`config/strategies.yaml` may define a top-level `contracts` mapping as a convenience for live IBKR routing:

```yaml
contracts:
  HSBC.L:
    symbol: HSBC
    exchange: LSE
    currency: GBP
  SPY:
    symbol: SPY
    exchange: SMART
    primary_exchange: ARCA
    currency: USD
```

At runtime, `main.py` loads this mapping and merges it into `ibkr.contract_overrides` before creating `IBKRClient`.
If the same ticker exists in both places, `ibkr.contract_overrides` wins.