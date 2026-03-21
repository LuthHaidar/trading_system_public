import pandas as pd
import numpy as np
from typing import Dict
from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy: Buy top N performers based on past returns
    
    Classic momentum anomaly:
    - Rank tickers by past performance
    - Go long top N performers
    - Hold for rebalance period
    - Skip recent period to avoid short-term reversal
    
    Config parameters:
    - lookback: Lookback period in days (default: 252 = 12 months)
    - skip_recent: Days to skip at end (default: 21 = 1 month)
    - n_positions: Number of positions to hold (default: 5)
    - rebalance_frequency: Days between rebalances (default: 21)
    - min_momentum: Minimum momentum threshold (default: 0.0)
    - weight_method: 'equal' or 'proportional' (default: 'equal')
    - rebalance_threshold: Min weight change to trigger rebalance (default: 0.0)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config, name='Momentum')
        
        self.lookback = self.get_config_param('lookback', 252)
        self.skip_recent = self.get_config_param('skip_recent', 21)
        self.n_positions = self.get_config_param('n_positions', 5)
        self.rebalance_frequency = self.get_config_param('rebalance_frequency', 21)
        self.min_momentum = self.get_config_param('min_momentum', 0.0)
        self.weight_method = self.get_config_param('weight_method', 'equal')
        self.rebalance_threshold = self.get_config_param('rebalance_threshold', 0.0)
        
        self.last_rebalance = None
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.lookback <= 0:
            raise ValueError(f"lookback must be positive, got {self.lookback}")
        if self.skip_recent < 0:
            raise ValueError(f"skip_recent must be non-negative, got {self.skip_recent}")
        if self.n_positions <= 0:
            raise ValueError(f"n_positions must be positive, got {self.n_positions}")
        if self.rebalance_frequency <= 0:
            raise ValueError(f"rebalance_frequency must be positive, got {self.rebalance_frequency}")
        if self.weight_method not in ['equal', 'proportional']:
            raise ValueError(f"weight_method must be 'equal' or 'proportional', got {self.weight_method}")
        if not 0.0 <= self.rebalance_threshold < 1.0:
            raise ValueError(f"rebalance_threshold must be in [0, 1), got {self.rebalance_threshold}")
    
    def get_required_history(self) -> int:
        """Need lookback + skip_recent days of data"""
        return self.lookback + self.skip_recent + 10
    
    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame],
                        current_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Generate momentum signals
        
        Strategy:
        1. Calculate momentum for each ticker
        2. Rank by momentum
        3. Select top N with positive momentum above threshold
        4. Assign weights (equal or proportional)
        5. Only rebalance if frequency met and positions changed significantly
        """
        self._last_signal_meta = {}

        # Validate inputs
        if not data:
            self.logger.warning(f"No data available on {date}")
            return current_positions if current_positions else {}
        
        # Check if we need to rebalance based on time
        if not self._should_rebalance(date):
            self._last_signal_meta = {}
            return current_positions
        
        # Calculate momentum for all tickers
        momentum_scores = {}
        min_required_length = self.lookback + self.skip_recent
        
        for ticker, df in data.items():
            try:
                # Validate data availability
                if len(df) < min_required_length:
                    self.logger.debug(f"Insufficient data for {ticker}: {len(df)} < {min_required_length}")
                    continue
                
                # Get prices up to current date
                prices = df.loc[:date, 'Close']
                
                if len(prices) < min_required_length:
                    self.logger.debug(f"Insufficient price history for {ticker} up to {date}")
                    continue
                
                # Calculate momentum
                momentum = self.calculate_momentum(prices, self.lookback, self.skip_recent)
                
                # Filter by minimum threshold
                if not np.isnan(momentum) and momentum >= self.min_momentum:
                    momentum_scores[ticker] = momentum
                    
            except Exception as e:
                self.logger.warning(f"Error calculating momentum for {ticker}: {e}")
                continue
        
        # Handle no valid momentum scores
        if not momentum_scores:
            self.logger.warning(f"No valid momentum scores on {date}")
            # Keep existing positions if we have them
            if current_positions:
                self._last_signal_meta = {}
                return current_positions
            return {}
        
        # Rank by momentum and select top N
        sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_tickers = sorted_tickers[:self.n_positions]
        
        self.logger.info(f"{date.date()}: Top momentum - {[f'{t}({m:.2%})' for t, m in top_tickers]}")
        
        # Assign weights
        if self.weight_method == 'equal':
            new_weights = self._equal_weight(top_tickers)
        else:  # proportional
            new_weights = self._proportional_weight(top_tickers)

        min_score = min((score for _, score in top_tickers), default=0.0)
        max_score = max((score for _, score in top_tickers), default=0.0)
        score_span = max(max_score - min_score, 0.0)
        self._last_signal_meta = {}
        for rank, (ticker, score) in enumerate(top_tickers, start=1):
            if score_span > 0:
                confidence = (float(score) - float(min_score)) / score_span
            else:
                confidence = 1.0 if score > 0 else 0.0
            self._last_signal_meta[ticker] = {
                'reason': f"momentum_rank={rank} score={score:.4f}",
                'signal_strength': float(score),
                'confidence': float(max(0.0, min(1.0, confidence))),
            }
        for ticker, current_weight in (current_positions or {}).items():
            if current_weight > 0 and ticker not in new_weights:
                held_score = momentum_scores.get(ticker)
                self._last_signal_meta[ticker] = {
                    'reason': (
                        f"momentum_exit_not_top_n score={held_score:.4f}"
                        if held_score is not None
                        else "momentum_exit_no_valid_score"
                    ),
                    'signal_strength': float(held_score) if held_score is not None else None,
                    'confidence': 0.0,
                }
        
        # Check if positions changed significantly enough to warrant rebalance
        if not self._positions_changed(current_positions, new_weights):
            self.logger.debug(f"{date.date()}: Positions unchanged, skipping rebalance")
            self._last_signal_meta = {}
            return current_positions
        
        # Update last rebalance date
        self.last_rebalance = date
        
        # Validate and return
        return self.validate_signals(new_weights)
    
    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        """Check if we should rebalance based on time frequency"""
        if self.last_rebalance is None:
            return True
        
        days_since = (date - self.last_rebalance).days
        return days_since >= self.rebalance_frequency
    
    def _positions_changed(self, current: Dict[str, float], new: Dict[str, float]) -> bool:
        """
        Check if positions changed significantly
        Returns True if positions are different enough to warrant rebalancing
        """
        if self.rebalance_threshold == 0.0:
            return True  # Always rebalance if threshold is 0
        
        # If different tickers, definitely changed
        current_tickers = set(current.keys())
        new_tickers = set(new.keys())
        
        if current_tickers != new_tickers:
            return True
        
        # Check if any weight changed by more than threshold
        for ticker in new_tickers:
            weight_diff = abs(current.get(ticker, 0.0) - new.get(ticker, 0.0))
            if weight_diff > self.rebalance_threshold:
                return True
        
        return False
    
    def _equal_weight(self, ranked_tickers: list) -> Dict[str, float]:
        """Assign equal weight to each position"""
        if not ranked_tickers:
            return {}
        
        weight = 1.0 / len(ranked_tickers)
        return {ticker: weight for ticker, _ in ranked_tickers}
    
    def _proportional_weight(self, ranked_tickers: list) -> Dict[str, float]:
        """
        Assign weights proportional to momentum
        Higher momentum gets higher weight
        """
        if not ranked_tickers:
            return {}
        
        # Get momentum values
        momentums = np.array([m for _, m in ranked_tickers])
        
        # Handle edge cases
        if len(momentums) == 0:
            return {}
        
        # For proportional weighting, momentums should already be positive due to min_momentum filter
        # However, handle edge case where all momentums are equal
        if np.all(momentums == momentums[0]):
            # Fall back to equal weighting
            return self._equal_weight(ranked_tickers)
        
        # If any negative values exist (shouldn't with proper min_momentum), shift to positive
        if np.any(momentums <= 0):
            momentums = momentums - momentums.min() + 0.01
        
        # Normalize to sum to 1
        weights_arr = momentums / momentums.sum()
        
        return {ticker: float(weight) for (ticker, _), weight in zip(ranked_tickers, weights_arr)}


class DualMomentumStrategy(BaseStrategy):
    """
    Dual Momentum: Combines relative and absolute momentum
    
    Strategy:
    1. Relative momentum: Rank assets by performance
    2. Absolute momentum: Only invest if trending up (> threshold)
    3. If no assets meet criteria, move to cash (or cash proxy)
    
    Config parameters:
    - lookback: Lookback period (default: 126 = 6 months)
    - n_positions: Number of positions (default: 3)
    - absolute_momentum_threshold: Minimum return to invest (default: 0.0)
    - cash_ticker: Ticker for cash proxy (default: 'SHY')
    """
    
    def __init__(self, config: Dict):
        super().__init__(config, name='DualMomentum')
        
        self.lookback = self.get_config_param('lookback', 126)
        self.n_positions = self.get_config_param('n_positions', 3)
        self.abs_threshold = self.get_config_param('absolute_momentum_threshold', 0.0)
        self.cash_ticker = self.get_config_param('cash_ticker', 'SHY')
    
    def get_required_history(self) -> int:
        return self.lookback + 10
    
    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame],
                        current_positions: Dict[str, float]) -> Dict[str, float]:
        """Generate dual momentum signals"""
        self._last_signal_meta = {}
        momentum_scores = {}
        
        for ticker, df in data.items():
            if ticker == self.cash_ticker:
                continue
            
            try:
                prices = df.loc[:date, 'Close']
                momentum = self.calculate_momentum(prices, self.lookback, skip_recent=0)
                
                # Apply absolute momentum filter
                if momentum >= self.abs_threshold:
                    momentum_scores[ticker] = momentum
                    
            except Exception as e:
                self.logger.warning(f"Could not calculate momentum for {ticker}: {e}")
                continue
        
        # If no assets pass absolute momentum test, go to cash
        if not momentum_scores:
            self.logger.info(f"{date.date()}: No assets with positive momentum, moving to cash")
            if self.cash_ticker in data:
                self._last_signal_meta = {
                    self.cash_ticker: {
                        'reason': "dual_momentum_cash_fallback",
                        'signal_strength': 0.0,
                        'confidence': 1.0,
                    }
                }
                return {self.cash_ticker: 1.0}
            else:
                return {}  # Stay in cash
        
        # Rank by relative momentum and select top N
        sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_tickers = sorted_tickers[:self.n_positions]
        
        self.logger.info(f"{date.date()}: Top dual momentum: {[f'{t}({m:.2%})' for t, m in top_tickers]}")
        
        # Equal weight
        weight = 1.0 / len(top_tickers)
        weights = {ticker: weight for ticker, _ in top_tickers}
        min_score = min((score for _, score in top_tickers), default=0.0)
        max_score = max((score for _, score in top_tickers), default=0.0)
        span = max(max_score - min_score, 0.0)
        self._last_signal_meta = {}
        for rank, (ticker, score) in enumerate(top_tickers, start=1):
            confidence = ((float(score) - float(min_score)) / span) if span > 0 else (1.0 if score > 0 else 0.0)
            self._last_signal_meta[ticker] = {
                'reason': f"dual_momentum_rank={rank} score={score:.4f}",
                'signal_strength': float(score),
                'confidence': float(max(0.0, min(1.0, confidence))),
            }
        for ticker, current_weight in (current_positions or {}).items():
            if current_weight > 0 and ticker not in weights:
                held_score = momentum_scores.get(ticker)
                self._last_signal_meta[ticker] = {
                    'reason': (
                        f"dual_momentum_exit_not_top_n score={held_score:.4f}"
                        if held_score is not None
                        else "dual_momentum_exit_no_valid_score"
                    ),
                    'signal_strength': float(held_score) if held_score is not None else None,
                    'confidence': 0.0,
                }
        
        return self.validate_signals(weights)
