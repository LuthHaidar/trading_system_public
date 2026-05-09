from __future__ import annotations

from collections import deque
from importlib import import_module
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd

from risk.advanced_risk import AdvancedRiskAnalytics

try:
    from risk.hmm_regime import HMMRegimeDetector
except ImportError:
    HMMRegimeDetector = None
from strategies.base_strategy import BaseStrategy
from strategies.orchestration import MultiStrategyOrchestrator


class StrategyOrchestrationStrategy(BaseStrategy):
    """Combine multiple enabled strategies with regime/condition-aware weighting."""

    def __init__(self, config: Dict, name: str = 'strategy_orchestration'):
        super().__init__(config, name=name)
        self._all_configs = self.get_config_param('_all_strategies_config', {}) or {}
        self.base_weights = self.get_config_param('base_weights', {}) or {}
        self.allowed_conditions = self.get_config_param('allowed_conditions', {}) or {}
        self.regime_multipliers = self.get_config_param('regime_multipliers', {}) or {}
        self.benchmark_ticker = self.get_config_param('benchmark_ticker', 'SPY')
        self.performance_lookback = int(self.get_config_param('performance_lookback', 20))
        self.min_weight_threshold = float(self.get_config_param('min_weight_threshold', 0.0))
        self.min_strategy_weight_floor = float(
            self.get_config_param('min_strategy_weight_floor', 0.1)
        )
        self.max_attribution_records = int(
            self.get_config_param('max_attribution_records', 500)
        )
        self.regime_vol_threshold = float(self.get_config_param('regime_vol_threshold', 0.25))
        self.regime_bear_return_threshold = float(
            self.get_config_param('regime_bear_return_threshold', 0.0)
        )
        self.regime_bull_return_threshold = self.get_config_param('regime_bull_return_threshold', None)
        self.hmm_n_states = int(self.get_config_param('hmm_n_states', 3))
        self.hmm_covariance_type = str(self.get_config_param('hmm_covariance_type', 'full'))
        self.regime_detector = None
        if HMMRegimeDetector is not None:
            try:
                self.regime_detector = HMMRegimeDetector(
                    n_states=self.hmm_n_states,
                    covariance_type=self.hmm_covariance_type,
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                self.logger.warning(
                    "HMM detector unavailable, using heuristic regime detection: %s",
                    exc,
                )

        self.sub_strategies = self._build_enabled_sub_strategies()
        self.sub_strategy_names = list(self.sub_strategies.keys())

        if self.sub_strategy_names and not self.base_weights:
            equal = 1.0 / len(self.sub_strategy_names)
            self.base_weights = {name: equal for name in self.sub_strategy_names}

        self.performance_history: Dict[str, Deque[float]] = {
            name: deque(maxlen=self.performance_lookback) for name in self.sub_strategy_names
        }
        self.last_strategy_weights: Dict[str, float] = {}
        self.last_update_date: pd.Timestamp | None = None
        self._attribution_history: list[Dict[str, Any]] = []

    def _build_enabled_sub_strategies(self) -> Dict[str, BaseStrategy]:
        sub_strategies: Dict[str, BaseStrategy] = {}
        if not self._all_configs:
            return sub_strategies

        from strategies import STRATEGY_REGISTRY

        excluded = {'strategy_orchestration'}
        for strategy_name, cfg in self._all_configs.items():
            if strategy_name in excluded or strategy_name not in STRATEGY_REGISTRY:
                continue
            if not isinstance(cfg, dict) or not cfg.get('enabled', False):
                continue

            module_path, class_name = STRATEGY_REGISTRY[strategy_name].split(':', 1)
            module = import_module(module_path)
            strategy_cls = getattr(module, class_name)
            sub_strategies[strategy_name] = strategy_cls(cfg)

        return sub_strategies

    def get_required_history(self) -> int:
        if not self.sub_strategies:
            return 0
        return max(s.get_required_history() for s in self.sub_strategies.values())

    def _detect_market_condition(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> str:
        if self.benchmark_ticker not in data:
            return 'neutral'

        close = data[self.benchmark_ticker].loc[:date, 'Close'].dropna()
        returns = close.pct_change().dropna()
        if self.regime_detector is not None:
            try:
                regime = AdvancedRiskAnalytics.hmm_regime_scaler(
                    market_returns=returns,
                    detector=self.regime_detector,
                ).get('regime', 'unknown')
            except RuntimeError:
                self.logger.warning(
                    "HMM regime detector has insufficient history on %s; falling back to heuristic",
                    date.date(),
                )
                regime = AdvancedRiskAnalytics.regime_based_scaler(
                    returns,
                    vol_threshold=self.regime_vol_threshold,
                    bear_return_threshold=self.regime_bear_return_threshold,
                    bull_return_threshold=self.regime_bull_return_threshold,
                ).get('regime', 'unknown')
        else:
            regime = AdvancedRiskAnalytics.regime_based_scaler(
                returns,
                vol_threshold=self.regime_vol_threshold,
                bear_return_threshold=self.regime_bear_return_threshold,
                bull_return_threshold=self.regime_bull_return_threshold,
            ).get('regime', 'unknown')

        assert regime in {'bull', 'neutral', 'bear'}, f"Unrecognized regime label: {regime!r}"
        if regime == 'bull':
            return 'bull'
        if regime == 'bear':
            return 'bear'
        return 'neutral'

    def on_realized_portfolio_return(self, date: pd.Timestamp, portfolio_return: float,
                                     context: Optional[Dict] = None) -> None:
        """
        Update sub-strategy trailing performance from realized portfolio return attribution.

        Attribution uses the latest orchestration strategy weights applied at rebalance time.
        """
        if not self.last_strategy_weights:
            return

        realized = float(portfolio_return)
        for strategy_name, weight in self.last_strategy_weights.items():
            if strategy_name not in self.performance_history:
                continue
            contribution = float(weight) * realized
            self.performance_history[strategy_name].append(contribution)
        self.last_update_date = pd.Timestamp(date)

    @staticmethod
    def _renormalize(weights: Dict[str, float]) -> Dict[str, float]:
        positive = {k: max(0.0, float(v)) for k, v in weights.items()}
        total = sum(positive.values())
        if total <= 0:
            return {k: 0.0 for k in weights}
        return {k: v / total for k, v in positive.items()}

    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame], current_positions: Dict[str, float]) -> Dict[str, float]:
        self._last_signal_meta = {}
        if not self.sub_strategies:
            return {}

        strategy_signals: Dict[str, Dict[str, float]] = {}
        strategy_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for strategy_name, strategy in self.sub_strategies.items():
            try:
                strategy_signals[strategy_name] = strategy.generate_signals(date, data, current_positions)
                raw_meta = strategy.get_signal_metadata() if hasattr(strategy, 'get_signal_metadata') else {}
                if isinstance(raw_meta, dict):
                    strategy_meta[strategy_name] = {
                        str(ticker): dict(meta)
                        for ticker, meta in raw_meta.items()
                        if isinstance(meta, dict)
                    }
                else:
                    strategy_meta[strategy_name] = {}
            except (KeyError, IndexError, ValueError) as exc:
                self.logger.warning("Sub-strategy %s failed on %s: %s", strategy_name, date, exc)
                strategy_signals[strategy_name] = {}
                strategy_meta[strategy_name] = {}

        regime = self._detect_market_condition(date, data)
        trailing_perf = {
            name: float(np.mean(hist)) if len(hist) else 0.0
            for name, hist in self.performance_history.items()
        }

        base = {k: v for k, v in self.base_weights.items() if k in self.sub_strategy_names}
        if not base:
            equal = 1.0 / len(self.sub_strategy_names)
            base = {name: equal for name in self.sub_strategy_names}

        strategy_weights = MultiStrategyOrchestrator.dynamic_strategy_weights(
            base_weights=base,
            trailing_performance=trailing_perf,
            regime=regime,
            regime_multipliers=self.regime_multipliers,
            min_weight_floor=self.min_strategy_weight_floor,
        )

        switches = MultiStrategyOrchestrator.condition_switches(
            enabled_strategies=self.sub_strategy_names,
            market_condition=regime,
            allowed_conditions=self.allowed_conditions,
        )
        strategy_weights = {
            name: strategy_weights.get(name, 0.0) if switches.get(name, True) else 0.0
            for name in self.sub_strategy_names
        }
        strategy_weights = self._renormalize(strategy_weights)

        combined = MultiStrategyOrchestrator.ensemble_signals(
            strategy_signals=strategy_signals,
            strategy_weights=strategy_weights,
            min_weight_threshold=self.min_weight_threshold,
        )

        signal_meta: Dict[str, Dict[str, Any]] = {}
        all_tickers = set(combined.keys()) | set((current_positions or {}).keys())
        for ticker in all_tickers:
            contributors = []
            weighted_strength = 0.0
            strength_weight = 0.0
            participation = 0.0

            for strategy_name, strat_weight in strategy_weights.items():
                if strat_weight <= 0:
                    continue
                strat_signal = float(strategy_signals.get(strategy_name, {}).get(ticker, 0.0))
                if strat_signal <= 0:
                    continue
                participation += float(strat_weight)
                contributors.append((strategy_name, float(strat_weight), strat_signal))
                raw_strength = strategy_meta.get(strategy_name, {}).get(ticker, {}).get('signal_strength')
                if raw_strength is not None:
                    try:
                        weighted_strength += float(strat_weight) * float(raw_strength)
                        strength_weight += float(strat_weight)
                    except (TypeError, ValueError):
                        pass

            contributors.sort(key=lambda row: row[1], reverse=True)
            top = ",".join([f"{name}:{weight:.2f}" for name, weight, _ in contributors[:3]]) or "none"
            reason = (
                f"ensemble regime={regime} contributors={top}"
                if ticker in combined
                else f"ensemble_exit regime={regime} contributors={top}"
            )
            signal_meta[ticker] = {
                'reason': reason,
                'signal_strength': (weighted_strength / strength_weight) if strength_weight > 0 else None,
                'confidence': float(max(0.0, min(1.0, participation))),
            }

        self._last_signal_meta = signal_meta
        allocations_by_strategy: Dict[str, Dict[str, float]] = {}
        for strategy_name, strat_weight in strategy_weights.items():
            if strat_weight <= 0:
                continue
            ticker_allocations: Dict[str, float] = {}
            for ticker, signal_weight in strategy_signals.get(strategy_name, {}).items():
                w = float(strat_weight) * float(signal_weight)
                if w > 0:
                    ticker_allocations[str(ticker)] = w
            if ticker_allocations:
                allocations_by_strategy[strategy_name] = ticker_allocations
        self._attribution_history.append(
            {
                'date': pd.Timestamp(date),
                'strategy_allocations': allocations_by_strategy,
            }
        )
        # Cap to prevent unbounded memory growth on long backtests.
        if len(self._attribution_history) > self.max_attribution_records:
            excess = len(self._attribution_history) - self.max_attribution_records
            self._attribution_history = self._attribution_history[excess:]
        self.last_strategy_weights = dict(strategy_weights)
        return self.validate_signals(combined)

    def get_attribution_history(self) -> list[Dict[str, Any]]:
        return list(self._attribution_history)
