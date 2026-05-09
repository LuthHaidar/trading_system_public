import logging

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimization methods
    
    Implements:
    - Mean-variance optimization (Markowitz)
    - Minimum variance
    - Maximum Sharpe ratio
    - Constrained optimization
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config or {}
        self.logger = get_logger("portfolio_optimizer")
        
        # Optimization parameters
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        # Prefer explicit optimizer_* keys from config, with backward-compatible fallback.
        self.max_weight = self.config.get('optimizer_max_weight', self.config.get('max_weight', 0.30))
        self.min_weight = self.config.get('optimizer_min_weight', self.config.get('min_weight', 0.0))
        self.covariance_estimator = str(self.config.get('covariance_estimator', 'ledoit_wolf')).lower()
        self.covariance_shrinkage = float(self.config.get('covariance_shrinkage', 0.10))
        self.covariance_ew_halflife = int(self.config.get('covariance_ew_halflife', 63))
        self.covariance_jitter = float(self.config.get('covariance_jitter', 1e-8))
        self._warned_infeasible_bounds = set()
        self._warned_fallback_signatures = set()
        self._warned_unknown_cov_estimators = set()
        if self.covariance_estimator in {'ledoit_wolf', 'ledoitwolf', 'lw'}:
            try:
                from sklearn.covariance import LedoitWolf  # noqa: F401
            except Exception as exc:
                raise ImportError(
                    "scikit-learn with sklearn.covariance.LedoitWolf is required when "
                    "covariance_estimator=ledoit_wolf"
                ) from exc
        
    def optimize(self, signals: Dict[str, float],
                data: Dict[str, pd.DataFrame],
                method: str = 'max_sharpe') -> Dict[str, float]:
        """
        Optimize portfolio weights
        
        Args:
            signals: Strategy signals (used to filter universe)
            data: Historical data
            method: Optimization method
                - 'max_sharpe': Maximum Sharpe ratio
                - 'min_variance': Minimum variance
                - 'risk_parity': Risk parity (equal risk contribution)
                
        Returns:
            Optimized weights
        """
        if not signals:
            return {}
        
        # Get returns data
        returns_df, valid_tickers = self._prepare_returns(signals, data)
        
        if returns_df is None or len(valid_tickers) < 2:
            # Fall back to equal weight
            min_assets_required = 2
            lookback_window = 252
            valid_assets = len(valid_tickers)
            total_signals = len(signals)
            signature = (total_signals, valid_assets, lookback_window)
            if self.logger.isEnabledFor(logging.DEBUG) or self.logger.isEnabledFor(logging.INFO): # Use lazy formatting for logging calls
                message = (
                    "Optimizer fallback to equal weight: requires >=%d assets with >=%d return observations; "
                    "got signals=%d, valid_assets=%d"
                    % (min_assets_required, lookback_window, total_signals, valid_assets)
                )
                if signature not in self._warned_fallback_signatures:
                    self._warned_fallback_signatures.add(signature)
                    self.logger.warning(message)
                else:
                    self.logger.debug(message)
            n = len(signals)
            return {ticker: 1.0/n for ticker in signals}
        
        # Calculate statistics
        mean_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = self._estimate_covariance(returns_df)
        
        # Optimize
        if method == 'max_sharpe':
            weights = self._max_sharpe(mean_returns, cov_matrix)
        elif method == 'min_variance':
            weights = self._min_variance(cov_matrix)
        elif method == 'risk_parity':
            weights = self._risk_parity_optimize(cov_matrix)
        else:
            self.logger.warning("Unknown method %s, using equal weight", method)
            weights = np.ones(len(valid_tickers)) / len(valid_tickers)
        
        # Convert to dict
        weights_dict = {ticker: float(w) for ticker, w in zip(valid_tickers, weights) 
                       if w > 0.001}  # Filter out tiny weights
        
        # Normalize
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {ticker: w/total for ticker, w in weights_dict.items()}
        
        return weights_dict

    def _estimate_covariance(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate annualized covariance matrix with configurable regularization.

        Supported estimators:
            - sample
            - ledoit_wolf (default; falls back to shrinkage if sklearn unavailable)
            - shrinkage (diagonal shrinkage of sample covariance)
            - exponential / ew
            - exponential_shrinkage / ew_shrinkage
        """
        sample_cov = returns_df.cov()
        estimator = self.covariance_estimator

        if estimator == 'sample':
            cov = sample_cov
        elif estimator in {'ledoit_wolf', 'ledoitwolf', 'lw'}:
            from sklearn.covariance import LedoitWolf  # type: ignore

            lw = LedoitWolf().fit(returns_df.values)
            cov = pd.DataFrame(
                lw.covariance_,
                index=returns_df.columns,
                columns=returns_df.columns,
            )
        elif estimator in {'shrinkage', 'diagonal_shrinkage'}:
            cov = self._diagonal_shrinkage(sample_cov)
        elif estimator in {'exponential', 'ew'}:
            cov = self._estimate_exponential_covariance(returns_df)
        elif estimator in {'exponential_shrinkage', 'ew_shrinkage'}:
            cov = self._diagonal_shrinkage(self._estimate_exponential_covariance(returns_df))
        else:
            if estimator not in self._warned_unknown_cov_estimators:
                self._warned_unknown_cov_estimators.add(estimator)
                self.logger.warning(
                    "Unknown covariance_estimator '%s'; using sample covariance",
                    estimator,
                )
            cov = sample_cov

        cov = cov.astype(float)
        jitter = max(float(self.covariance_jitter), 0.0)
        if jitter > 0:
            cov = cov + np.eye(len(cov)) * jitter

        return cov * 252


    def _estimate_exponential_covariance(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate exponentially weighted covariance (non-annualized)."""
        halflife = max(int(self.covariance_ew_halflife), 1)
        alpha = 1.0 - np.exp(-np.log(2.0) / float(halflife))
        ewm_cov = returns_df.ewm(alpha=alpha, adjust=True).cov()
        last_ts = returns_df.index[-1]
        cov = ewm_cov.loc[last_ts]
        return cov.reindex(index=returns_df.columns, columns=returns_df.columns).astype(float)

    def _diagonal_shrinkage(self, sample_cov: pd.DataFrame) -> pd.DataFrame:
        """Shrink sample covariance toward its diagonal for numerical stability."""
        shrink = min(max(float(self.covariance_shrinkage), 0.0), 1.0)
        diagonal = pd.DataFrame(
            np.diag(np.diag(sample_cov.values)),
            index=sample_cov.index,
            columns=sample_cov.columns,
        )
        return ((1.0 - shrink) * sample_cov) + (shrink * diagonal)
    
    def _prepare_returns(self, signals: Dict[str, float],
                        data: Dict[str, pd.DataFrame],
                        window: int = 252) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Prepare returns DataFrame for optimization
        
        Args:
            signals: Strategy signals
            data: Historical data
            window: Lookback window
            
        Returns:
            (returns_df, valid_tickers)
        """
        returns_list = []
        valid_tickers = []
        
        for ticker in signals:
            if ticker not in data:
                continue
            
            try:
                prices = data[ticker]['Close']
                returns = prices.pct_change().dropna()
                
                if len(returns) >= window:
                    returns_list.append(returns.tail(window))
                    valid_tickers.append(ticker)
            except (KeyError, TypeError, ValueError) as e:
                self.logger.warning("Could not get returns for %s: %s", ticker, e)
                continue
        
        if len(returns_list) < 2:
            return None, []
        
        # Combine into DataFrame
        returns_df = pd.concat(returns_list, axis=1)
        returns_df.columns = valid_tickers

        # Align all assets to common timestamps and remove missing values. This
        # prevents downstream covariance estimators (e.g., LedoitWolf) from
        # receiving NaNs when one ticker has gaps relative to others.
        before_drop = len(returns_df)
        returns_df = returns_df.dropna(how='any')
        after_drop = len(returns_df)
        if before_drop > 0 and (before_drop - after_drop) / before_drop > 0.2:
            self.logger.warning(
                "dropna removed %d of %d observations (%.0f%%) from the covariance window "
                "due to staggered histories; the effective estimation window is only %d days",
                before_drop - after_drop,
                before_drop,
                ((before_drop - after_drop) / before_drop) * 100,
                after_drop,
            )

        if len(returns_df) < 2:
            return None, []
        
        return returns_df, valid_tickers

    def _get_feasible_bounds(self, n_assets: int) -> Tuple[Tuple[float, float], ...]:
        """Return feasible per-asset bounds for sum(weights)=1 constraint."""
        min_w = float(self.min_weight)
        max_w = float(self.max_weight)

        min_sum = n_assets * min_w
        max_sum = n_assets * max_w

        if min_sum <= 1.0 <= max_sum:
            return tuple((min_w, max_w) for _ in range(n_assets))

        adj_min = min_w
        adj_max = max_w

        if max_sum < 1.0:
            adj_max = 1.0 / n_assets
        elif min_sum > 1.0:
            adj_min = 1.0 / n_assets

        if adj_min > adj_max:
            target = 1.0 / n_assets
            adj_min = target
            adj_max = target

        key = (n_assets, round(min_w, 8), round(max_w, 8), round(adj_min, 8), round(adj_max, 8))
        if key not in self._warned_infeasible_bounds:
            self._warned_infeasible_bounds.add(key)
            self.logger.warning(
                "Infeasible optimizer bounds for %d assets (n*min=%.4f, n*max=%.4f); using adjusted bounds [%.4f, %.4f]",
                n_assets,
                min_sum,
                max_sum,
                adj_min,
                adj_max,
            )

        return tuple((adj_min, adj_max) for _ in range(n_assets))


    def _max_sharpe(self, mean_returns: pd.Series, 
                   cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Maximize Sharpe ratio
        
        Args:
            mean_returns: Expected returns
            cov_matrix: Covariance matrix
            
        Returns:
            Optimal weights
        """
        n_assets = len(mean_returns)
        
        # Objective: negative Sharpe ratio (minimize)
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_std < 1e-10:
                return 0.0
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            if not np.isfinite(sharpe):
                return 0.0
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = self._get_feasible_bounds(n_assets)
        
        # Initial guess: equal weight
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.logger.info(
                "Max Sharpe optimization successful: Sharpe = %.2f",
                -result.fun,
            )
            return result.x
        else:
            self.logger.warning("Optimization failed, using equal weight")
            return init_weights
    
    def _min_variance(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Minimize portfolio variance
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Optimal weights
        """
        n_assets = len(cov_matrix)
        
        # Objective: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = self._get_feasible_bounds(n_assets)
        
        # Initial guess
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            portfolio_vol = np.sqrt(result.fun)
            self.logger.info(
                "Min variance optimization successful: Vol = %.2f%%",
                portfolio_vol * 100.0,
            )
            return result.x
        else:
            self.logger.warning("Optimization failed, using equal weight")
            return init_weights
    
    def _risk_parity_optimize(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Risk parity optimization
        
        Each asset contributes equal risk to portfolio
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Risk parity weights
        """
        n_assets = len(cov_matrix)
        
        # Objective: minimize sum of squared differences in risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol < 1e-10:
                return 0.0

            # Marginal contribution to risk
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol

            # Risk contribution
            risk_contrib = weights * marginal_contrib

            # Equal-risk-contribution objective: pairwise contribution differences.
            pairwise_diffs = risk_contrib[:, None] - risk_contrib[None, :]
            return float(np.sum(np.triu(pairwise_diffs ** 2, k=1)))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = self._get_feasible_bounds(n_assets)
        
        # Initial guess: inverse volatility
        diag_vol = np.sqrt(np.diag(cov_matrix))
        diag_vol = np.where(diag_vol < 1e-10, 1e-10, diag_vol)
        init_weights = (1.0 / diag_vol) / np.sum(1.0 / diag_vol)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.logger.info("Risk parity optimization successful")
            return result.x
        else:
            self.logger.warning("Optimization failed, using inverse vol weights")
            return init_weights
    
    def _min_variance_with_return(self, mean_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  target_return: float) -> Optional[np.ndarray]:
        """Minimize variance subject to target return constraint"""
        n_assets = len(mean_returns)
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        ]
        
        bounds = self._get_feasible_bounds(n_assets)
        init_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else None
