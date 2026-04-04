from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

try:
    from risk.hmm_regime import HMMRegimeDetector
except ImportError:
    HMMRegimeDetector = None


class StatisticalArbitrageStrategy(BaseStrategy):
    """Cross-sectional mean-reversion ranking strategy."""

    def get_required_history(self) -> int:
        return int(self.get_config_param('lookback', 20)) + int(self.get_config_param('ranking_window', 5))

    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame], current_positions: Dict[str, float]) -> Dict[str, float]:
        self._last_signal_meta = {}
        lookback = int(self.get_config_param('lookback', 20))
        ranking_window = int(self.get_config_param('ranking_window', 5))
        n_positions = int(self.get_config_param('n_positions', 3))

        scores = {}
        for ticker, df in data.items():
            close = df.loc[:date, 'Close'].dropna()
            if len(close) < lookback + ranking_window:
                continue

            returns = close.pct_change().dropna().tail(lookback)
            if len(returns) < ranking_window:
                continue

            short_term = returns.tail(ranking_window).mean()
            long_term = returns.mean()
            vol = returns.std() + 1e-9
            scores[ticker] = (long_term - short_term) / vol

        if not scores:
            return {}

        selected = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_positions]
        weight = 1.0 / len(selected)
        min_score = min((score for _, score in selected), default=0.0)
        max_score = max((score for _, score in selected), default=0.0)
        span = max(max_score - min_score, 0.0)
        self._last_signal_meta = {}
        for rank, (ticker, score) in enumerate(selected, start=1):
            confidence = ((float(score) - float(min_score)) / span) if span > 0 else 1.0
            self._last_signal_meta[ticker] = {
                'reason': f"stat_arb_rank={rank} zscore={score:.4f}",
                'signal_strength': float(score),
                'confidence': float(max(0.0, min(1.0, confidence))),
            }
        selected_tickers = {ticker for ticker, _ in selected}
        for ticker, current_weight in (current_positions or {}).items():
            if current_weight > 0 and ticker not in selected_tickers:
                held_score = scores.get(ticker)
                self._last_signal_meta[ticker] = {
                    'reason': (
                        f"stat_arb_exit_not_selected score={held_score:.4f}"
                        if held_score is not None
                        else "stat_arb_exit_no_score"
                    ),
                    'signal_strength': float(held_score) if held_score is not None else None,
                    'confidence': 0.0,
                }
        return self.validate_signals({ticker: weight for ticker, _ in selected})


class FactorModelStrategy(BaseStrategy):
    """Simple multi-factor model using price-derived proxies.

    Factors:
    - momentum (positive trend)
    - low_vol (inverse volatility)
    - quality_proxy (up-capture/downside ratio)
    """

    def get_required_history(self) -> int:
        return int(self.get_config_param('lookback', 126))

    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame], current_positions: Dict[str, float]) -> Dict[str, float]:
        self._last_signal_meta = {}
        lookback = int(self.get_config_param('lookback', 126))
        n_positions = int(self.get_config_param('n_positions', 5))
        factor_weights = self.get_config_param(
            'factor_weights', {'momentum': 0.5, 'low_vol': 0.3, 'quality_proxy': 0.2}
        )

        scores = {}
        for ticker, df in data.items():
            close = df.loc[:date, 'Close'].dropna()
            if len(close) < lookback:
                continue

            window = close.tail(lookback)
            returns = window.pct_change().dropna()
            if returns.empty:
                continue

            momentum = window.iloc[-1] / window.iloc[0] - 1
            low_vol = 1.0 / (returns.std() + 1e-6)

            up_days = returns[returns > 0].sum()
            down_days = abs(returns[returns < 0].sum()) + 1e-6
            quality_proxy = float(up_days / down_days)

            scores[ticker] = (
                factor_weights.get('momentum', 0.0) * momentum
                + factor_weights.get('low_vol', 0.0) * low_vol
                + factor_weights.get('quality_proxy', 0.0) * quality_proxy
            )

        if not scores:
            return {}

        selected = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_positions]
        raw = {ticker: float(max(score, 0.0)) for ticker, score in selected}
        min_score = min((score for _, score in selected), default=0.0)
        max_score = max((score for _, score in selected), default=0.0)
        span = max(max_score - min_score, 0.0)
        self._last_signal_meta = {}
        for rank, (ticker, score) in enumerate(selected, start=1):
            confidence = ((float(score) - float(min_score)) / span) if span > 0 else (1.0 if score > 0 else 0.0)
            self._last_signal_meta[ticker] = {
                'reason': f"factor_rank={rank} composite_score={score:.4f}",
                'signal_strength': float(score),
                'confidence': float(max(0.0, min(1.0, confidence))),
            }
        selected_tickers = {ticker for ticker, _ in selected}
        for ticker, current_weight in (current_positions or {}).items():
            if current_weight > 0 and ticker not in selected_tickers:
                held_score = scores.get(ticker)
                self._last_signal_meta[ticker] = {
                    'reason': (
                        f"factor_exit_not_selected score={held_score:.4f}"
                        if held_score is not None
                        else "factor_exit_no_score"
                    ),
                    'signal_strength': float(held_score) if held_score is not None else None,
                    'confidence': 0.0,
                }
        total_score = sum(raw.values())
        if total_score == 0:
            equal = 1.0 / len(selected)
            return self.validate_signals({ticker: equal for ticker, _ in selected})

        # Convert unbounded factor scores into portfolio weights.
        # This preserves relative ranking while avoiding very large pre-normalized
        # totals (e.g. from low-volatility / quality ratios on quiet windows).
        normalized = {ticker: score / total_score for ticker, score in raw.items()}
        return self.validate_signals(normalized)


class VolatilityTradingStrategy(BaseStrategy):
    """Regime-aware allocation between risk-on and defensive baskets."""

    def __init__(self, config: Dict, name: str = "volatility_trading"):
        super().__init__(config, name=name)
        self._hmm_detector = None
        if bool(self.get_config_param('use_hmm_regime', False)) and HMMRegimeDetector is not None:
            try:
                self._hmm_detector = HMMRegimeDetector()
            except (TypeError, ValueError, RuntimeError) as exc:
                self.logger.warning(
                    "Failed to initialize HMM detector, using hard-switch volatility regime: %s",
                    exc,
                )

    def get_required_history(self) -> int:
        return int(self.get_config_param('vol_window', 20)) + 5

    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame], current_positions: Dict[str, float]) -> Dict[str, float]:
        self._last_signal_meta = {}
        benchmark = self.get_config_param('benchmark_ticker', 'SPY')
        risk_on = self.get_config_param('risk_on_tickers', [])
        defensive = self.get_config_param('defensive_tickers', [])
        vol_window = int(self.get_config_param('vol_window', 20))
        vol_threshold = float(self.get_config_param('vol_threshold', 0.20))

        if benchmark not in data:
            return {}

        close = data[benchmark].loc[:date, 'Close'].dropna()
        if len(close) < vol_window + 1:
            return {}

        realized_vol = close.pct_change().dropna().tail(vol_window).std() * np.sqrt(252)
        use_hmm_regime = bool(self.get_config_param('use_hmm_regime', False))
        weights = {}
        regime = 'defensive' if realized_vol > vol_threshold else 'risk_on'
        confidence = float(max(0.0, min(1.0, abs(realized_vol - vol_threshold) / max(vol_threshold, 1e-9))))

        if use_hmm_regime and self._hmm_detector is not None:
            try:
                probs = self._hmm_detector.predict_proba(close.pct_change().dropna())
                p_bull = float(probs.get('bull', 0.0))
                p_neutral = float(probs.get('neutral', 0.0))
                p_bear = float(probs.get('bear', 0.0))
                weight_risk_on = p_bull + 0.5 * p_neutral
                weight_defensive = p_bear + 0.5 * p_neutral
                total_mix = max(weight_risk_on + weight_defensive, 1e-9)
                weight_risk_on /= total_mix
                weight_defensive /= total_mix

                risk_on_selected = [t for t in risk_on if t in data]
                defensive_selected = [t for t in defensive if t in data]
                if risk_on_selected:
                    each = weight_risk_on / len(risk_on_selected)
                    for t in risk_on_selected:
                        weights[t] = weights.get(t, 0.0) + each
                if defensive_selected:
                    each = weight_defensive / len(defensive_selected)
                    for t in defensive_selected:
                        weights[t] = weights.get(t, 0.0) + each
                regime = max({'bull': p_bull, 'neutral': p_neutral, 'bear': p_bear}, key=lambda k: {'bull': p_bull, 'neutral': p_neutral, 'bear': p_bear}[k])
                confidence = float(max(p_bull, p_neutral, p_bear))
            except RuntimeError:
                use_hmm_regime = False

        if not weights:
            selected = defensive if realized_vol > vol_threshold else risk_on
            selected = [t for t in selected if t in data]
            if not selected:
                return {}
            w = 1.0 / len(selected)
            weights = {t: w for t in selected}

        self._last_signal_meta = {
            ticker: {
                'reason': f"vol_regime={regime} realized_vol={realized_vol:.4f} threshold={vol_threshold:.4f}",
                'signal_strength': float(realized_vol - vol_threshold),
                'confidence': confidence,
            }
            for ticker in weights
        }
        for ticker, current_weight in (current_positions or {}).items():
            if current_weight > 0 and ticker not in weights:
                self._last_signal_meta[ticker] = {
                    'reason': f"vol_regime_exit regime={regime}",
                    'signal_strength': float(realized_vol - vol_threshold),
                    'confidence': 0.0,
                }
        return self.validate_signals(weights)
