from __future__ import annotations

from typing import Dict, Iterable


class MultiStrategyOrchestrator:
    """Utilities for coordinating multiple strategies under one portfolio."""

    @staticmethod
    def dynamic_strategy_weights(
        base_weights: Dict[str, float],
        trailing_performance: Dict[str, float],
        regime: str,
        regime_multipliers: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        adjusted = {}
        regime_adjust = regime_multipliers.get(regime, {})

        for strategy, base in base_weights.items():
            perf = max(trailing_performance.get(strategy, 0.0), -1.0)
            perf_scale = 1.0 + perf
            regime_scale = regime_adjust.get(strategy, 1.0)
            adjusted[strategy] = max(base * perf_scale * regime_scale, 0.0)

        total = sum(adjusted.values())
        if total <= 0:
            equal = 1.0 / max(len(base_weights), 1)
            return {k: equal for k in base_weights}
        return {k: v / total for k, v in adjusted.items()}

    @staticmethod
    def condition_switches(
        enabled_strategies: Iterable[str],
        market_condition: str,
        allowed_conditions: Dict[str, Iterable[str]],
    ) -> Dict[str, bool]:
        status = {}
        for strategy in enabled_strategies:
            conditions = list(allowed_conditions.get(strategy, []))
            status[strategy] = not conditions or market_condition in conditions
        return status

    @staticmethod
    def ensemble_signals(
        strategy_signals: Dict[str, Dict[str, float]],
        strategy_weights: Dict[str, float],
        min_weight_threshold: float = 0.0,
    ) -> Dict[str, float]:
        combined: Dict[str, float] = {}
        for strategy_name, signal in strategy_signals.items():
            strategy_weight = strategy_weights.get(strategy_name, 0.0)
            if strategy_weight <= 0:
                continue

            for ticker, weight in signal.items():
                combined[ticker] = combined.get(ticker, 0.0) + strategy_weight * max(weight, 0.0)

        filtered = {k: v for k, v in combined.items() if v > min_weight_threshold}
        total = sum(filtered.values())
        if total <= 0:
            return {}

        return {k: v / total for k, v in filtered.items()}
