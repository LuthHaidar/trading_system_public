import pandas as pd
import numpy as np
from typing import Dict
from strategies.base_strategy import BaseStrategy
from utils.types import TickerData, WeightMap

_WARMUP_BUFFER = 10


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion strategy using Bollinger Bands
    
    Strategy:
    1. Calculate Bollinger Bands (SMA ± N * std)
    2. Buy when price crosses below lower band (oversold)
    3. Sell when price crosses above upper band (overbought) or returns to mean
    4. Hold for maximum holding period
    
    Config parameters:
    - window: Moving average window (default: 20)
    - num_std: Number of standard deviations (default: 2)
    - holding_period: Maximum days to hold (default: 5)
    - exit_zscore: Z-score to exit position (default: 0)
    - max_positions: Maximum simultaneous positions (default: 5)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config, name='MeanReversion')
        
        self.window = self.get_config_param('window', 20)
        self.num_std = self.get_config_param('num_std', 2)
        self.holding_period = self.get_config_param('holding_period', 5)
        self.exit_zscore = self.get_config_param('exit_zscore', 0)
        self.max_positions = self.get_config_param('max_positions', 5)
        
        # Track entry dates for each position
        self.entry_dates = {}
    
    def get_required_history(self) -> int:
        return self.window + _WARMUP_BUFFER
    
    def generate_signals(self, date: pd.Timestamp, data: TickerData,
                        current_positions: WeightMap) -> WeightMap:
        """Generate mean reversion signals"""
        self._last_signal_meta = {}

        # Prune entry_dates for positions no longer held (e.g. closed externally
        # or engine restart with fresh state).
        held_tickers = {t for t, w in (current_positions or {}).items() if w > 0}
        stale = [t for t in self.entry_dates if t not in held_tickers]
        for ticker in stale:
            del self.entry_dates[ticker]

        new_positions = {}
        signal_meta = {}
        
        for ticker, df in data.items():
            try:
                prices = df.loc[:date, 'Close']
                
                # Not enough history
                if len(prices) < self.window:
                    continue
                
                # Calculate Bollinger Bands
                sma = prices.tail(self.window).mean()
                std = prices.tail(self.window).std()
                
                current_price = prices.iloc[-1]
                lower_band = sma - self.num_std * std
                upper_band = sma + self.num_std * std
                
                # Calculate z-score
                zscore = (current_price - sma) / std if std > 0 else 0
                
                # Check if currently holding this position
                is_holding = ticker in current_positions and current_positions[ticker] > 0
                
                if is_holding:
                    # Exit conditions
                    days_held = (date - self.entry_dates.get(ticker, date)).days
                    
                    # Exit if: price reverted to mean, reached upper band, or held too long
                    should_exit = (
                        zscore >= self.exit_zscore or  # Reverted to mean
                        current_price >= upper_band or  # Overbought
                        days_held >= self.holding_period  # Max holding period
                    )
                    
                    if should_exit:
                        self.logger.info(
                            "%s: Exit %s - zscore=%.2f, days=%d",
                            date.date(),
                            ticker,
                            zscore,
                            days_held,
                        )
                        signal_meta[ticker] = {
                            'reason': f"bollinger_exit zscore={zscore:.3f} days_held={days_held}",
                            'signal_strength': float(zscore),
                            'confidence': float(max(0.0, min(1.0, abs(zscore) / max(self.num_std, 1e-9)))),
                        }
                        # Remove from entry_dates
                        if ticker in self.entry_dates:
                            del self.entry_dates[ticker]
                    else:
                        # Keep holding
                        new_positions[ticker] = current_positions[ticker]
                        signal_meta[ticker] = {
                            'reason': f"bollinger_hold zscore={zscore:.3f} days_held={days_held}",
                            'signal_strength': float(zscore),
                            'confidence': float(max(0.0, min(1.0, abs(zscore) / max(self.num_std, 1e-9)))),
                        }
                
                else:
                    # Entry condition: price below lower band (oversold)
                    if current_price < lower_band and len(new_positions) < self.max_positions:
                        self.logger.info(
                            "%s: Enter %s - price=%.2f, lower_band=%.2f, zscore=%.2f",
                            date.date(),
                            ticker,
                            current_price,
                            lower_band,
                            zscore,
                        )
                        new_positions[ticker] = 1.0  # Will be normalized later
                        self.entry_dates[ticker] = date
                        signal_meta[ticker] = {
                            'reason': f"bollinger_entry zscore={zscore:.3f} lower_band_cross=1",
                            'signal_strength': float(-zscore),
                            'confidence': float(max(0.0, min(1.0, (-zscore) / max(self.num_std, 1e-9)))),
                        }
                        
            except (KeyError, IndexError, ValueError) as e:
                self.logger.warning("Error processing %s: %s", ticker, e)
                continue
        
        # Equal weight positions
        if new_positions:
            weight = 1.0 / len(new_positions)
            new_positions = {ticker: weight for ticker in new_positions}
        self._last_signal_meta = signal_meta
        return self.validate_signals(new_positions)


class RSIMeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion using RSI (Relative Strength Index)
    
    Strategy:
    1. Calculate RSI for each ticker
    2. Buy when RSI < oversold threshold (e.g., 30)
    3. Sell when RSI > overbought threshold (e.g., 70)
    4. Equal weight positions
    
    Config parameters:
    - rsi_period: RSI calculation period (default: 14)
    - oversold_threshold: RSI buy threshold (default: 30)
    - overbought_threshold: RSI sell threshold (default: 70)
    - max_positions: Maximum positions (default: 5)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config, name='RSIMeanReversion')
        
        self.rsi_period = self.get_config_param('rsi_period', 14)
        self.oversold = self.get_config_param('oversold_threshold', 30)
        self.overbought = self.get_config_param('overbought_threshold', 70)
        self.max_positions = self.get_config_param('max_positions', 5)
    
    def get_required_history(self) -> int:
        return self.rsi_period + _WARMUP_BUFFER
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index)
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        # Calculate price changes
        delta = prices.diff().dropna()
        
        # Separate gains and losses
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)

        # Wilder smoothing:
        # seed with simple averages over the first period, then recursively smooth.
        avg_gain = gains.iloc[:period].mean()
        avg_loss = losses.iloc[:period].mean()
        for idx in range(period, len(delta)):
            avg_gain = ((avg_gain * (period - 1)) + gains.iloc[idx]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses.iloc[idx]) / period
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        if avg_gain == 0:
            return 0.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, date: pd.Timestamp, data: TickerData,
                        current_positions: WeightMap) -> WeightMap:
        """Generate RSI-based mean reversion signals"""
        self._last_signal_meta = {}
        new_positions = {}
        signal_meta = {}
        
        for ticker, df in data.items():
            try:
                prices = df.loc[:date, 'Close']
                
                if len(prices) < self.rsi_period + 1:
                    continue
                
                # Calculate RSI
                rsi = self.calculate_rsi(prices, self.rsi_period)
                
                # Check if currently holding
                is_holding = ticker in current_positions and current_positions[ticker] > 0
                
                if is_holding:
                    # Exit if overbought
                    if rsi >= self.overbought:
                        self.logger.info(
                            "%s: Exit %s - RSI=%.1f (overbought)",
                            date.date(),
                            ticker,
                            rsi,
                        )
                        signal_meta[ticker] = {
                            'reason': f"rsi_exit rsi={rsi:.2f}",
                            'signal_strength': float(rsi),
                            'confidence': float(max(0.0, min(1.0, (rsi - self.overbought) / max(100.0 - self.overbought, 1e-9)))),
                        }
                    else:
                        # Keep holding
                        new_positions[ticker] = current_positions[ticker]
                        signal_meta[ticker] = {
                            'reason': f"rsi_hold rsi={rsi:.2f}",
                            'signal_strength': float(50.0 - rsi),
                            'confidence': float(max(0.0, min(1.0, abs(50.0 - rsi) / 50.0))),
                        }
                else:
                    # Enter if oversold
                    if rsi <= self.oversold and len(new_positions) < self.max_positions:
                        self.logger.info(
                            "%s: Enter %s - RSI=%.1f (oversold)",
                            date.date(),
                            ticker,
                            rsi,
                        )
                        new_positions[ticker] = 1.0
                        signal_meta[ticker] = {
                            'reason': f"rsi_entry rsi={rsi:.2f}",
                            'signal_strength': float(self.oversold - rsi),
                            'confidence': float(max(0.0, min(1.0, (self.oversold - rsi) / max(self.oversold, 1e-9)))),
                        }
                        
            except (KeyError, IndexError, ValueError) as e:
                self.logger.warning("Error processing %s: %s", ticker, e)
                continue
        
        # Equal weight
        if new_positions:
            weight = 1.0 / len(new_positions)
            new_positions = {ticker: weight for ticker in new_positions}
        self._last_signal_meta = signal_meta
        return self.validate_signals(new_positions)
