from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    
    All strategies must implement:
    - generate_signals(): Returns target weights for each ticker
    - get_required_history(): Returns minimum days of historical data needed
    """
    
    def __init__(self, config: Dict, name: Optional[str] = None):
        """
        Initialize strategy
        
        Args:
            config: Strategy configuration dict
            name: Strategy name (defaults to class name)
        """
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"strategy.{self.name}")
        self._weight_normalization_warned = False
        self._last_signal_meta: Dict[str, Dict[str, Any]] = {}

        self.logger.info(f"Initialized {self.name} with config: {config}")
    
    @abstractmethod
    def generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame],
                        current_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Generate target weights for each ticker
        
        Args:
            date: Current date for signal generation
            data: Dict of {ticker: DataFrame} with historical data up to and including date
            current_positions: Dict of {ticker: current_weight}
            
        Returns:
            Dict of {ticker: target_weight} where:
            - Weights are in range [0, 1]
            - Sum of weights <= 1.0 (remainder is cash)
            - Weight of 0 means close position
            - Missing tickers from current_positions implies close them
        """
        pass
    
    @abstractmethod
    def get_required_history(self) -> int:
        """
        Return minimum number of days of historical data needed
        
        Returns:
            Number of days
        """
        pass

    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Return per-ticker signal metadata from the most recent generate_signals() call.

        Concrete strategies can populate this in generate_signals(), for example:
            {
                "NVDA": {
                    "reason": "momentum_rank=1 score=0.31",
                    "signal_strength": 0.31,
                    "confidence": 0.85,
                }
            }

        Returns:
            Mapping of ticker -> metadata dict. Empty by default.
        """
        return dict(self._last_signal_meta or {})
    
    def validate_signals(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and normalize signals
        
        Args:
            signals: Raw signals from strategy
            
        Returns:
            Validated signals
        """
        if not signals:
            return {}
        
        # Remove negative weights
        signals = {k: v for k, v in signals.items() if v > 0}
        
        # Check total weight
        total = sum(signals.values())
        
        if total > 1.0001:  # Allow small floating point error
            if not self._weight_normalization_warned:
                self._weight_normalization_warned = True
                self.logger.warning(
                    f"Total weight {total:.4f} > 1.0, normalizing "
                    f"(first occurrence; further normalizations logged at debug)"
                )
            else:
                self.logger.debug(f"Total weight {total:.4f} > 1.0, normalizing")
            signals = {k: v/total for k, v in signals.items()}
        
        # Round to avoid floating point issues
        signals = {k: round(v, 6) for k, v in signals.items()}
        
        return signals
    
    def get_config_param(self, param: str, default=None):
        """
        Get configuration parameter with default fallback
        
        Args:
            param: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value
        """
        return self.config.get(param, default)
    
    def calculate_momentum(self, prices: pd.Series, lookback: int,
                          skip_recent: int = 0) -> float:
        """
        Calculate momentum (total return over period)
        
        Args:
            prices: Price series
            lookback: Lookback period in days
            skip_recent: Days to skip at end (avoid short-term reversals)
            
        Returns:
            Momentum (return) over period
        """
        if len(prices) < lookback + skip_recent:
            return 0.0
        
        if skip_recent > 0:
            start_price = prices.iloc[-(lookback + skip_recent)]
            end_price = prices.iloc[-skip_recent]
        else:
            start_price = prices.iloc[-lookback]
            end_price = prices.iloc[-1]
        
        if start_price <= 0:
            return 0.0
        
        return (end_price / start_price) - 1.0
    
    def calculate_sma(self, prices: pd.Series, window: int) -> float:
        """
        Calculate simple moving average
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            SMA value
        """
        if len(prices) < window:
            return prices.iloc[-1]
        
        return prices.tail(window).mean()
    
    def calculate_volatility(self, returns: pd.Series, window: int) -> float:
        """
        Calculate rolling volatility (annualized)
        
        Args:
            returns: Return series
            window: Window size
            
        Returns:
            Annualized volatility
        """
        if len(returns) < window:
            return returns.std() * (252 ** 0.5)
        
        return returns.tail(window).std() * (252 ** 0.5)
    
    def calculate_zscore(self, value: float, series: pd.Series, window: int) -> float:
        """
        Calculate z-score of current value vs historical window
        
        Args:
            value: Current value
            series: Historical series
            window: Window size
            
        Returns:
            Z-score
        """
        if len(series) < window:
            return 0.0
        
        window_data = series.tail(window)
        mean = window_data.mean()
        std = window_data.std()
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    def __str__(self):
        return f"{self.name}({self.config})"
    
    def __repr__(self):
        return self.__str__()

    def on_realized_portfolio_return(self, date: pd.Timestamp, portfolio_return: float,
                                     context: Optional[Dict] = None) -> None:
        """
        Optional lifecycle hook invoked by engines with realized portfolio returns.

        Args:
            date: Timestamp for the realized return observation.
            portfolio_return: Net realized portfolio return for the period.
            context: Optional metadata payload from the engine.
        """
        return None
