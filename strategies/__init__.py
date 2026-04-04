"""Strategy registry and helpers.
"""

from importlib import import_module
from typing import Dict, List


STRATEGY_REGISTRY: Dict[str, str] = {
    'momentum': 'strategies.momentum:MomentumStrategy',
    'dual_momentum': 'strategies.momentum:DualMomentumStrategy',
    'mean_reversion': 'strategies.mean_reversion:MeanReversionStrategy',
    'rsi_mean_reversion': 'strategies.mean_reversion:RSIMeanReversionStrategy',
    'statistical_arbitrage': 'strategies.alpha_expansion:StatisticalArbitrageStrategy',
    'factor_model': 'strategies.alpha_expansion:FactorModelStrategy',
    'volatility_trading': 'strategies.alpha_expansion:VolatilityTradingStrategy',
    'strategy_orchestration': 'strategies.strategy_orchestration:StrategyOrchestrationStrategy',
}


def get_available_strategies() -> List[str]:
    """Return supported strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())


def create_strategy(strategy_name: str, strategy_config: Dict):
    """Instantiate a strategy by registry name."""
    strategy_path = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_path is None:
        available = ', '.join(get_available_strategies())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {available}")

    module_path, class_name = strategy_path.split(':', 1)
    module = import_module(module_path)
    strategy_cls = getattr(module, class_name)
    return strategy_cls(strategy_config)
