import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from risk.advanced_risk import AdvancedRiskAnalytics
from utils.logger import get_logger

logger = get_logger(__name__)


class PositionSizer:
    """
    Position sizing algorithms for portfolio construction
    
    Implements multiple methods:
    - Equal weight
    - Volatility-based (inverse volatility)
    - Inverse volatility
    - Legacy "risk_parity" alias (implemented as inverse volatility)
    - Kelly criterion
    - Target volatility
    """
    
    def __init__(self, method: str = 'equal', config: Dict = None):
        """
        Initialize position sizer
        
        Args:
            method: Sizing method
                ('equal', 'volatility', 'inverse_volatility', 'risk_parity', 'kelly', 'target_vol')
            config: Configuration parameters
        """
        self.method = method
        self.config = config or {}
        self.logger = get_logger(f"position_sizer.{method}")
        
        self.logger.info(f"PositionSizer initialized: {method}")
    
    def size_positions(self, signals: Dict[str, float], 
                      data: Dict[str, pd.DataFrame],
                      current_equity: float = None) -> Dict[str, float]:
        """
        Size positions based on signals and method
        
        Args:
            signals: Dict of {ticker: signal_strength} from strategy
            data: Dict of {ticker: DataFrame} with historical data
            current_equity: Current portfolio equity (for some methods)
            
        Returns:
            Dict of {ticker: target_weight}
        """
        if not signals:
            return {}
        
        if self.method == 'equal':
            return self._equal_weight(signals)
        elif self.method == 'volatility':
            return self._volatility_weight(signals, data)
        elif self.method == 'inverse_volatility':
            return self._inverse_volatility_weight(signals, data)
        elif self.method == 'risk_parity':
            self.logger.warning(
                "position_sizing_method='risk_parity' is deprecated for PositionSizer; "
                "use 'inverse_volatility' instead. "
                "(True ERC risk parity remains in PortfolioOptimizer.)"
            )
            return self._inverse_volatility_weight(signals, data)
        elif self.method == 'kelly':
            return self._kelly_criterion(signals, data)
        elif self.method == 'target_vol':
            # Vol targeting is applied as a final post-constraint transform in engine.
            return self._equal_weight(signals)
        else:
            self.logger.warning(f"Unknown method {self.method}, using equal weight")
            return self._equal_weight(signals)
    
    def _equal_weight(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Equal weight allocation
        
        Args:
            signals: Signal strengths (just used to filter tickers)
            
        Returns:
            Equal weights for all tickers
        """
        n = len(signals)
        if n == 0:
            return {}
        
        weight = 1.0 / n
        return {ticker: weight for ticker in signals}
    
    def _volatility_weight(self, signals: Dict[str, float],
                          data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Inverse volatility weighting
        
        Lower volatility assets get higher weight
        
        Args:
            signals: Signal strengths
            data: Historical data
            
        Returns:
            Volatility-weighted allocations
        """
        window = self.config.get('volatility_window', 60)
        min_vol = self.config.get('min_volatility', 0.01)
        
        volatilities = {}
        for ticker in signals:
            if ticker not in data:
                continue
            
            try:
                # Calculate returns
                prices = data[ticker]['Close']
                returns = prices.pct_change().dropna()
                
                if len(returns) < window:
                    continue
                
                # Calculate annualized volatility
                vol = returns.tail(window).std() * np.sqrt(252)
                vol = max(vol, min_vol)  # Floor to avoid division by zero
                
                volatilities[ticker] = vol
                
            except Exception as e:
                self.logger.warning(f"Could not calculate volatility for {ticker}: {e}")
                continue
        
        if not volatilities:
            return self._equal_weight(signals)
        
        # Inverse volatility weights
        inv_vols = {ticker: 1.0 / vol for ticker, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {ticker: inv_vol / total_inv_vol 
                  for ticker, inv_vol in inv_vols.items()}
        
        self.logger.info(f"Volatility weights: {[f'{t}:{w:.2%}' for t, w in weights.items()]}")
        return weights
    
    def _inverse_volatility_weight(self, signals: Dict[str, float],
                                   data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Inverse volatility allocation.

        Note: this is kept as the implementation behind the legacy
        `risk_parity` method name for backward compatibility. True
        equal-risk-contribution optimization is implemented in
        `risk.optimizer.PortfolioOptimizer._risk_parity_optimize`.
        
        Args:
            signals: Signal strengths
            data: Historical data
            
        Returns:
            Risk parity weights
        """
        window = self.config.get('volatility_window', 60)
        
        # Calculate volatilities
        volatilities = {}
        for ticker in signals:
            if ticker not in data:
                continue
            
            try:
                prices = data[ticker]['Close']
                returns = prices.pct_change().dropna()
                
                if len(returns) < window:
                    continue
                
                vol = returns.tail(window).std() * np.sqrt(252)
                volatilities[ticker] = max(vol, 0.01)
                
            except Exception as e:
                self.logger.warning(f"Could not calculate volatility for {ticker}: {e}")
                continue
        
        if not volatilities:
            return self._equal_weight(signals)
        
        # Inverse-vol weights: weight = 1/vol / sum(1/vol)
        inv_vols = {ticker: 1.0 / vol for ticker, vol in volatilities.items()}
        total = sum(inv_vols.values())
        
        weights = {ticker: inv_vol / total for ticker, inv_vol in inv_vols.items()}

        self.logger.info(
            f"Inverse-vol weights (legacy risk_parity): "
            f"{[f'{t}:{w:.2%}' for t, w in weights.items()]}"
        )
        
        return weights

    # Backward-compatible alias for any internal/custom callers.
    def _kelly_criterion(self, signals: Dict[str, float],
                        data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Kelly criterion position sizing
        
        f* = (p*W - (1-p)*L) / W
        where p = win probability, W = avg win, L = avg loss
        
        Args:
            signals: Signal strengths (used as win probability estimate)
            data: Historical data
            
        Returns:
            Kelly-sized positions
        """
        window = self.config.get('kelly_window', 252)
        kelly_fraction = self.config.get('kelly_fraction', 0.5)  # Half-Kelly for safety
        
        kelly_sizes = {}
        
        for ticker in signals:
            if ticker not in data:
                continue
            
            try:
                prices = data[ticker]['Close']
                returns = prices.pct_change().dropna()
                
                if len(returns) < window:
                    continue
                
                # Use recent returns to estimate parameters
                recent_returns = returns.tail(window)
                
                # Win probability (proportion of positive returns)
                wins = recent_returns[recent_returns > 0]
                losses = recent_returns[recent_returns < 0]
                
                if len(losses) == 0:
                    kelly_sizes[ticker] = 0.25  # Default if no losses
                    continue
                
                p = len(wins) / len(recent_returns)
                
                # Average win and loss
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01

                if avg_win <= 0 or avg_loss <= 0:
                    kelly_sizes[ticker] = 0.0
                    continue

                # Kelly formula
                kelly_f = (p * avg_win - (1 - p) * avg_loss) / avg_win
                
                # Apply fraction and bounds
                kelly_f = kelly_f * kelly_fraction
                kelly_f = max(0, min(kelly_f, 0.5))  # Cap at 50%
                
                kelly_sizes[ticker] = kelly_f
                
            except Exception as e:
                self.logger.warning(f"Could not calculate Kelly for {ticker}: {e}")
                continue
        
        if not kelly_sizes:
            return self._equal_weight(signals)
        
        total_kelly = sum(kelly_sizes.values())
        if total_kelly <= 0:
            return self._equal_weight(signals)

        max_total_exposure_cfg = self.config.get('kelly_max_total_exposure')
        if max_total_exposure_cfg is None:
            max_total_exposure = 1.0 - float(self.config.get('min_cash_reserve', 0.0))
        else:
            max_total_exposure = float(max_total_exposure_cfg)
        max_total_exposure = min(max(max_total_exposure, 0.0), 1.0)

        if max_total_exposure <= 0:
            return {}

        scale = 1.0
        if total_kelly > max_total_exposure:
            scale = max_total_exposure / total_kelly

        weights = {
            ticker: float(size * scale)
            for ticker, size in kelly_sizes.items()
            if size > 0
        }

        gross_exposure = sum(weights.values())
        self.logger.info(
            f"Kelly weights (absolute fractions): {[f'{t}:{w:.2%}' for t, w in weights.items()]} "
            f"| gross_exposure={gross_exposure:.2%}"
        )
        return weights
    
    def apply_final_volatility_scaling(self,
                                       weights: Dict[str, float],
                                       data: Dict[str, pd.DataFrame],
                                       max_total_exposure: Optional[float] = None) -> Dict[str, float]:
        """
        Apply target-volatility scaling to final constrained weights.

        This is the sole implementation path for target-volatility scaling.
        It intentionally does not re-normalize weights back to 1.0,
        so a volatility scale is preserved rather than overwritten.
        """
        weights = {ticker: float(weight) for ticker, weight in (weights or {}).items() if float(weight) > 0}
        if not weights:
            return {}

        target_vol = float(self.config.get('target_volatility', 0.15))
        window = int(self.config.get('volatility_window', 60))

        returns_data = {}
        for ticker in weights:
            if ticker not in data:
                continue
            try:
                prices = data[ticker]['Close']
                returns = prices.pct_change().dropna()
                if len(returns) >= window:
                    returns_data[ticker] = returns.tail(window)
            except Exception as exc:
                self.logger.debug(f"Could not include {ticker} in final target-vol calc: {exc}")

        if len(returns_data) < 2:
            return weights

        returns_df = pd.DataFrame(returns_data).dropna()
        if returns_df.empty:
            return weights

        cov_matrix = returns_df.cov() * 252.0
        weights_array = np.array([weights.get(ticker, 0.0) for ticker in returns_df.columns], dtype=float)
        portfolio_var = float(np.dot(weights_array, np.dot(cov_matrix, weights_array)))
        portfolio_vol = float(np.sqrt(max(portfolio_var, 0.0)))
        if portfolio_vol <= 0:
            return weights

        scale_factor = target_vol / portfolio_vol

        if max_total_exposure is not None:
            max_total_exposure = max(float(max_total_exposure), 0.0)
            gross = float(sum(weights.values()))
            if gross > 0:
                scale_factor = min(scale_factor, max_total_exposure / gross)

        # Keep safety cap on leverage expansion.
        scale_factor = min(scale_factor, 2.0)
        scale_factor = max(scale_factor, 0.0)

        scaled_weights = {ticker: float(weight * scale_factor) for ticker, weight in weights.items()}
        self.logger.info(
            "Final target vol scaling: %.2fx (current: %.2f%%, target: %.2f%%)",
            scale_factor,
            portfolio_vol * 100.0,
            target_vol * 100.0,
        )
        return scaled_weights


class RiskConstraints:
    """
    Apply risk constraints to portfolio weights
    """
    
    def __init__(self, config: Dict):
        """
        Initialize risk constraints
        
        Args:
            config: Risk configuration
        """
        self.config = config
        self.logger = get_logger("risk_constraints")
        
        # Load constraints
        self.max_position_size = config.get('max_position_size', 0.20)
        self.max_positions = config.get('max_positions')
        self.min_position_size = config.get('min_position_size', 0.01)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.40)
        self.max_correlation = config.get('max_correlation', 0.90)
        self.min_diversification_score = config.get('min_diversification_score')
        self.diversification_scale_floor = float(config.get('diversification_scale_floor', 0.5))
        self.min_cash_reserve = config.get('min_cash_reserve', 0.05)
        self.capping_warning_every = int(config.get('capping_warning_every', 50))
        self._capping_counts = {}
        
        max_positions_text = self.max_positions if self.max_positions is not None else "none"
        self.logger.info(
            f"Risk constraints: max_pos={self.max_position_size:.1%}, "
            f"max_positions={max_positions_text}, min_cash={self.min_cash_reserve:.1%}"
        )
    
    def apply_constraints(self, weights: Dict[str, float],
                         data: Dict[str, pd.DataFrame] = None,
                         sector_map: Dict[str, str] = None) -> Dict[str, float]:
        """
        Apply all risk constraints to weights
        
        Args:
            weights: Target weights
            data: Historical data (for correlation checks)
            sector_map: Dict of {ticker: sector}
            
        Returns:
            Constrained weights
        """
        if not weights:
            return {}
        
        # Apply position size limits
        constrained = self._apply_position_limits(weights)

        # Apply portfolio-level max position count
        constrained = self._apply_max_positions_limit(constrained)
        
        # Apply sector limits if provided
        if sector_map:
            constrained = self._apply_sector_limits(constrained, sector_map)
        
        # Apply correlation limits if data provided
        if data:
            constrained = self._apply_correlation_limits(constrained, data)

        # Apply portfolio-level diversification floor when configured.
        if data:
            constrained = self._apply_diversification_floor(constrained, data)
        
        # Ensure cash reserve
        constrained = self._apply_cash_reserve(constrained)
        
        return constrained
    
    def _apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply per-position size limits"""
        constrained = {}
        
        for ticker, weight in weights.items():
            # Remove positions below minimum
            if weight < self.min_position_size:
                self.logger.debug(f"Removing {ticker}: {weight:.2%} < min {self.min_position_size:.2%}")
                continue
            
            # Cap positions above maximum
            if weight > self.max_position_size:
                count = int(self._capping_counts.get(ticker, 0)) + 1
                self._capping_counts[ticker] = count
                should_warn = (
                    count == 1
                    or (self.capping_warning_every > 0 and count % self.capping_warning_every == 0)
                )

                if should_warn:
                    self.logger.warning(
                        f"Capping {ticker}: {weight:.2%} -> {self.max_position_size:.2%} "
                        f"(occurrence #{count})"
                    )
                else:
                    self.logger.debug(f"Capping {ticker}: {weight:.2%} -> {self.max_position_size:.2%}")
                constrained[ticker] = self.max_position_size
            else:
                constrained[ticker] = weight
        
        return constrained

    def _apply_max_positions_limit(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Trim to max position count by keeping largest weights and renormalizing."""
        if not self.max_positions or self.max_positions <= 0:
            return weights

        if len(weights) <= self.max_positions:
            return weights

        # Deterministic ordering: largest weight first, then ticker for stable tie-break.
        sorted_weights = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        kept_items = sorted_weights[:self.max_positions]
        dropped = [ticker for ticker, _ in sorted_weights[self.max_positions:]]

        self.logger.warning(
            "Trimming positions from %s to max_positions=%s. Dropped: %s",
            len(weights),
            self.max_positions,
            dropped,
        )
        trimmed = dict(kept_items)
        total = sum(trimmed.values())
        if total > 0:
            trimmed = {ticker: weight / total for ticker, weight in trimmed.items()}
        return trimmed
    
    def _apply_sector_limits(self, weights: Dict[str, float],
                            sector_map: Dict[str, str]) -> Dict[str, float]:
        """Apply sector exposure limits"""
        # Calculate sector exposures
        sector_exposure = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        # Check for violations
        violations = {sector: exp for sector, exp in sector_exposure.items() 
                     if exp > self.max_sector_exposure}
        
        if not violations:
            return weights
        
        # Scale down overweight sectors
        constrained = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, 'Unknown')
            
            if sector in violations:
                scale = self.max_sector_exposure / violations[sector]
                new_weight = weight * scale
                self.logger.warning(f"Scaling {ticker} in {sector}: {weight:.2%} -> {new_weight:.2%}")
                constrained[ticker] = new_weight
            else:
                constrained[ticker] = weight
        
        return constrained
    
    def _apply_correlation_limits(self, weights: Dict[str, float],
                                  data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Apply correlation limits (reduce positions in highly correlated assets)
        """
        if len(weights) < 2:
            return weights
        
        window = 60
        tickers = list(weights.keys())
        
        # Calculate returns
        returns_list = []
        valid_tickers = []
        
        for ticker in tickers:
            if ticker not in data:
                continue
            
            try:
                prices = data[ticker]['Close']
                returns = prices.pct_change().dropna()
                
                if len(returns) >= window:
                    returns_list.append(returns.tail(window))
                    valid_tickers.append(ticker)
            except Exception as exc:
                self.logger.warning(f"Correlation check skipped for {ticker}: {exc}")
                continue
        
        if len(returns_list) < 2:
            return weights
        
        # Calculate correlation matrix
        returns_df = pd.concat(returns_list, axis=1)
        returns_df.columns = valid_tickers
        corr_matrix = returns_df.corr()
        
        # Find highly correlated pairs
        constrained = weights.copy()
        
        for i, ticker1 in enumerate(valid_tickers):
            for ticker2 in valid_tickers[i+1:]:
                corr = corr_matrix.loc[ticker1, ticker2]
                
                if abs(corr) > self.max_correlation:
                    # Reduce weight of smaller position
                    w1 = constrained.get(ticker1, 0.0)
                    w2 = constrained.get(ticker2, 0.0)
                    if w1 <= 0 and w2 <= 0:
                        continue

                    excess = (abs(corr) - self.max_correlation) / max(1.0 - self.max_correlation, 1e-12)
                    scale = max(0.0, 1.0 - excess)
                    
                    if w1 < w2:
                        constrained[ticker1] = w1 * scale
                        self.logger.warning(f"High correlation {ticker1}-{ticker2} "
                                          f"({corr:.2f}): reducing {ticker1} by {1.0-scale:.1%}")
                    else:
                        constrained[ticker2] = w2 * scale
                        self.logger.warning(f"High correlation {ticker1}-{ticker2} "
                                          f"({corr:.2f}): reducing {ticker2} by {1.0-scale:.1%}")
        
        return constrained

    def _apply_diversification_floor(self, weights: Dict[str, float],
                                     data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Scale portfolio exposure down when correlation diversification is too low.

        Uses AdvancedRiskAnalytics.correlation_diversification_score and only applies
        when `min_diversification_score` is configured.
        """
        if not weights:
            return weights
        if self.min_diversification_score is None:
            return weights

        try:
            min_score = float(self.min_diversification_score)
        except (TypeError, ValueError):
            self.logger.warning(
                "Ignoring invalid min_diversification_score=%s",
                self.min_diversification_score,
            )
            return weights

        if min_score <= 0:
            return weights

        tickers = [t for t in weights.keys() if t in data]
        if len(tickers) < 2:
            return weights

        window = 60
        returns = {}
        for ticker in tickers:
            try:
                r = data[ticker]['Close'].pct_change().dropna()
                if len(r) >= window:
                    returns[ticker] = r.tail(window)
            except Exception as exc:
                self.logger.debug(f"Skipping {ticker} in diversification score: {exc}")

        if len(returns) < 2:
            return weights

        returns_df = pd.DataFrame(returns).dropna()
        if returns_df.shape[1] < 2 or returns_df.empty:
            return weights

        score = AdvancedRiskAnalytics.correlation_diversification_score(returns_df)
        if score >= min_score:
            return weights

        floor = min(max(self.diversification_scale_floor, 0.0), 1.0)
        severity = min(max((min_score - score) / max(min_score, 1e-12), 0.0), 1.0)
        scale = 1.0 - severity * (1.0 - floor)
        constrained = {ticker: weight * scale for ticker, weight in weights.items()}
        self.logger.warning(
            "Diversification score %.3f below threshold %.3f; scaling portfolio exposure by %.3f",
            score,
            min_score,
            scale,
        )
        return constrained
    
    def _apply_cash_reserve(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure minimum cash reserve"""
        total_weight = sum(weights.values())
        max_invested = 1.0 - self.min_cash_reserve
        
        if total_weight <= max_invested:
            return weights
        
        # Scale down proportionally
        scale = max_invested / total_weight
        constrained = {ticker: weight * scale for ticker, weight in weights.items()}
        
        self.logger.info(f"Scaled positions by {scale:.2f}x to maintain "
                        f"{self.min_cash_reserve:.1%} cash reserve")
        
        return constrained
