"""
Research utilities for efficient frontier analysis.

This module is intentionally separated from production optimizer paths.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from risk.optimizer import PortfolioOptimizer


def calculate_efficient_frontier(data: Dict[str, pd.DataFrame],
                                 config: Optional[Dict] = None,
                                 n_points: int = 50) -> pd.DataFrame:
    """
    Build efficient-frontier points using the configured optimizer model.

    Args:
        data: Historical OHLCV data keyed by ticker.
        config: Optional optimizer configuration (bounds, covariance settings, etc.).
        n_points: Number of target-return points across the return range.

    Returns:
        DataFrame with columns: return, volatility, sharpe.
    """
    optimizer = PortfolioOptimizer(config or {})
    tickers = list(data.keys())
    returns_df, _ = optimizer._prepare_returns({ticker: 1.0 for ticker in tickers}, data)
    if returns_df is None or returns_df.empty:
        return pd.DataFrame(columns=['return', 'volatility', 'sharpe'])

    mean_returns = returns_df.mean() * 252
    min_ret = float(mean_returns.min())
    max_ret = float(mean_returns.max())

    # Degenerate case: all assets have identical expected returns so every
    # frontier point collapses to the minimum-variance portfolio.
    if np.isclose(min_ret, max_ret):
        optimizer.logger.warning(
            "Efficient frontier: all %d assets have identical expected returns (%.6f); "
            "returning a single portfolio point",
            len(mean_returns),
            min_ret,
        )
        weights = optimizer._min_variance_with_return(mean_returns, cov_matrix, min_ret)
        if weights is None:
            return pd.DataFrame(columns=['return', 'volatility', 'sharpe'])
        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
        port_sharpe = (min_ret - optimizer.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        return pd.DataFrame(
            [{'return': min_ret, 'volatility': port_vol, 'sharpe': float(port_sharpe)}],
            columns=['return', 'volatility', 'sharpe'],
        )

    target_returns = np.linspace(min_ret, max_ret, int(n_points))

    rows = []
    for target_ret in target_returns:
        weights = optimizer._min_variance_with_return(mean_returns, cov_matrix, float(target_ret))
        if weights is None:
            continue

        port_return = float(np.dot(weights, mean_returns))
        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
        if port_vol <= 0:
            sharpe = 0.0
        else:
            sharpe = float((port_return - optimizer.risk_free_rate) / port_vol)

        rows.append({
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe,
        })

    return pd.DataFrame(rows, columns=['return', 'volatility', 'sharpe'])
