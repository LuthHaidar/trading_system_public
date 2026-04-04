"""Pydantic schemas for runtime config validation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field




def _validate_model(model_cls: type[BaseModel], payload: Dict[str, Any]) -> BaseModel:
    """Validate payload across Pydantic v1/v2 APIs."""
    model_validate = getattr(model_cls, 'model_validate', None)
    if callable(model_validate):
        return model_validate(payload)

    parse_obj = getattr(model_cls, 'parse_obj', None)
    if callable(parse_obj):
        return parse_obj(payload)

    raise AttributeError(
        f"{model_cls.__name__} does not expose model_validate/parse_obj; unsupported Pydantic version"
    )

class PortfolioConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    initial_capital: float
    currency: str


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rebalance_timeframe: str
    timing: str
    progress_log_interval: int = 50
    hold_weight_epsilon: float = 1e-4
    min_rebalance_weight_delta: float = 0.02
    min_trade_value: float = 200.0
    invested_sleeve_drift_warn: float = 0.05
    cash_drift_warn: float = 0.01
    drift_warn_min_capital: float = 0.0
    uninvested_cash_tolerance: float = 0.005
    min_forward_data_days: int = 0
    cost_aware_fee_bps: float = 0.0
    cost_aware_spread_bps: float = 0.0


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    position_sizing_method: str
    max_position_size: float
    min_position_size: float
    max_positions: int = 10
    min_cash_reserve: float = 0.05
    max_correlation: float = 1.0
    min_diversification_score: float = 0.0
    diversification_scale_floor: float = 0.0
    target_volatility: float = 0.15
    volatility_window: int = 60
    kelly_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    kelly_window: int = 252
    kelly_max_total_exposure: float = 0.95
    capping_warning_every: int = 50
    use_optimizer: bool = False
    optimizer_method: str = 'max_sharpe'
    optimizer_max_weight: float = 1.0
    optimizer_min_weight: float = 0.0
    covariance_estimator: str = 'ledoit_wolf'
    covariance_shrinkage: float = 0.1
    covariance_ew_halflife: int = 63
    per_ticker_regime: bool = False
    risk_free_rate: float = 0.0


class DataConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    data_dir: str
    cache_size: int = 100
    max_stale_price_days: int = 5
    freshness_threshold_days: int = 3
    fx_cache_staleness_days: int = 1
    use_adjusted_close: bool = True


class BrokerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    host: str
    port: int
    client_id: int
    timeout: int = 60
    strict_universe_validation: bool = True
    contract_overrides: Dict[str, 'ContractOverrideConfig'] = Field(default_factory=dict)


class ContractOverrideConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    symbol: str
    exchange: str
    currency: str
    primary_exchange: Optional[str] = None


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    enabled: bool = True


class MainConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    portfolio: PortfolioConfig

    execution: ExecutionConfig
    risk: RiskConfig
    data: DataConfig
    ibkr: BrokerConfig

    costs: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    live_risk: Dict[str, Any] = Field(default_factory=dict)
    operations: Dict[str, Any] = Field(default_factory=dict)
    data_platform: Dict[str, Any] = Field(default_factory=dict)


def validate_main_config(config: Dict[str, Any]) -> MainConfig:
    """Validate a raw main config dictionary and return parsed model."""
    parsed = _validate_model(MainConfig, config)
    if parsed.execution.min_forward_data_days > 0 and parsed.execution.min_forward_data_days < parsed.data.max_stale_price_days:
        raise ValueError('execution.min_forward_data_days must be >= data.max_stale_price_days when enabled (>0)')
    return parsed


def validate_strategies_config(config: Mapping[str, Any]) -> Dict[str, StrategyConfig]:
    """Validate strategies config mapping and return parsed strategy config map."""
    parsed: Dict[str, StrategyConfig] = {}
    for name, value in (config or {}).items():
        if not isinstance(value, dict):
            raise TypeError(f"Strategy config for '{name}' must be a mapping")
        parsed[name] = _validate_model(StrategyConfig, value)
    return parsed
