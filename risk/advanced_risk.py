from typing import Dict

import numpy as np
import pandas as pd


class AdvancedRiskAnalytics:
    """Portfolio risk analytics for robust validation and sizing."""

    @staticmethod
    def var_cvar(returns: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
        if returns.empty:
            return {'var': 0.0, 'cvar': 0.0}
        q = np.quantile(returns, 1 - confidence)
        tail = returns[returns <= q]
        cvar = float(tail.mean()) if not tail.empty else float(q)
        return {'var': float(q), 'cvar': cvar}

    @staticmethod
    def volatility_adjusted_weights(signals: Dict[str, float], returns_df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
        if not signals:
            return {}
        vol = returns_df[list(signals.keys())].tail(lookback).std().replace(0, np.nan)
        inv = (1 / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            w = 1.0 / len(signals)
            return {k: w for k in signals}
        inv = inv / inv.sum()
        return {k: float(inv.get(k, 0.0)) for k in signals}

    @staticmethod
    def regime_based_scaler(market_returns: pd.Series,
                            bull_scale: float = 1.0,
                            bear_scale: float = 0.5,
                            high_vol_scale: float = 0.7,
                            vol_threshold: float = 0.25,
                            bear_return_threshold: float = 0.0) -> Dict[str, float]:
        """Deprecated heuristic regime scaler; prefer hmm_regime_scaler when available."""
        if len(market_returns) < 20:
            return {'regime': 'unknown', 'scale': 1.0}
        trend = market_returns.tail(60).mean()
        vol = market_returns.tail(20).std() * np.sqrt(252)
        if vol > vol_threshold:
            return {'regime': 'high_vol', 'scale': high_vol_scale}
        if trend < bear_return_threshold:
            return {'regime': 'bear', 'scale': bear_scale}
        return {'regime': 'bull', 'scale': bull_scale}

    @staticmethod
    def hmm_regime_scaler(market_returns: pd.Series,
                          detector,
                          bull_scale: float = 1.0,
                          bear_scale: float = 0.5,
                          high_vol_scale: float = 0.7) -> Dict[str, float]:
        """HMM-driven regime scaler returning a consistent {'regime', 'scale'} payload."""
        regime = detector.predict(market_returns)
        if regime == 'bull':
            scale = bull_scale
        elif regime == 'bear':
            scale = bear_scale
        else:
            scale = high_vol_scale
        return {'regime': regime, 'scale': float(scale)}

    @staticmethod
    def correlation_diversification_score(returns_df: pd.DataFrame) -> float:
        if returns_df.shape[1] < 2:
            return 1.0
        corr = returns_df.corr().values
        upper = corr[np.triu_indices_from(corr, k=1)]
        mean_abs_corr = np.mean(np.abs(upper)) if len(upper) else 0.0
        return float(max(0.0, 1.0 - mean_abs_corr))

    @staticmethod
    def enforce_factor_limits(weights: Dict[str, float],
                              factor_exposure: pd.DataFrame,
                              limits: Dict[str, float]) -> Dict[str, float]:
        """
        Deleverage portfolio globally when any factor exposure breaches its limit.

        Note: this scales all included positions, not only assets driving the breach.
        """
        if not weights:
            return {}

        ordered = [t for t in factor_exposure.index if t in weights]
        if not ordered:
            return weights

        w = pd.Series({t: weights[t] for t in ordered}, dtype=float)
        exposures = factor_exposure.loc[ordered]

        for factor, limit in limits.items():
            if factor not in exposures.columns:
                continue
            current = float((w * exposures[factor]).sum())
            if abs(current) <= limit:
                continue
            scale = limit / abs(current) if current != 0 else 1.0
            w = w * min(1.0, scale)

        total = w.sum()
        if total > 0:
            w = w / total
        updated = dict(weights)
        updated.update({k: float(v) for k, v in w.items()})
        return updated
