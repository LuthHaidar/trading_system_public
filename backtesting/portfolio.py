import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from utils.logger import get_logger
from utils.currency import CurrencyConverter

logger = get_logger(__name__)


@dataclass
class Trade:
    """Record of a single trade execution"""
    date: datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    commission: float
    slippage: float
    fx_cost: float  # Currency conversion cost
    value: float  # Total trade value
    currency: str  # Currency of the trade
    expected_price: Optional[float] = None
    benchmark_price: Optional[float] = None
    order_type: str = 'MARKET'
    parent_order_id: Optional[str] = None
    run_id: Optional[str] = None
    signal_date: Optional[datetime] = None
    decision: Optional[str] = None
    decision_reason: Optional[str] = None
    signal_strength: Optional[float] = None
    signal_confidence: Optional[float] = None
    weight_delta: Optional[float] = None
    current_weight: Optional[float] = None
    target_weight: Optional[float] = None
    current_shares: Optional[float] = None
    target_shares: Optional[float] = None
    executed_shares: Optional[float] = None
    
    def total_cost(self) -> float:
        """Total transaction cost"""
        return self.commission + self.slippage + self.fx_cost


class DataIntegrityError(RuntimeError):
    """Base class for integrity-critical data errors."""


class DataStalenessError(DataIntegrityError):
    """Raised when data staleness breaches configured halt policy."""


class Portfolio:
    """
    Manages portfolio state including positions, cash, and equity tracking
    Handles multi-currency positions with base currency (SGD) accounting
    """
    
    def __init__(self, initial_capital: float, base_currency: str = 'SGD',
                 max_stale_price_days: int = 5):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital in base currency
            base_currency: Portfolio base currency (default: SGD)
            max_stale_price_days: Max consecutive valuation sessions that can reuse last known price
        """
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        self.cash = initial_capital
        
        # Positions: {ticker: shares}
        self.positions = {}
        
        # Position currencies: {ticker: currency}
        self.position_currencies = {}
        
        # Trade history
        self.trades: List[Trade] = []
        
        # Equity curve
        self.equity_history = []
        self.date_history = []
        self.cash_history = []
        
        # Currency converter
        self.fx_converter = CurrencyConverter(base_currency)

        self.max_stale_price_days = int(max_stale_price_days)
        # Last known prices for stale-gap fallback
        self._last_prices: Dict[str, float] = {}
        self._last_price_dates: Dict[str, pd.Timestamp] = {}
        self._last_price_sessions: Dict[str, int] = {}
        self._valuation_sessions: Dict[pd.Timestamp, int] = {}
        self._next_valuation_session = 0
        
        logger.info("Portfolio initialized: %s %s", initial_capital, base_currency)

    def _get_valuation_session(self, valuation_date: Optional[pd.Timestamp]) -> Optional[int]:
        """Map each valuation date to a stable session counter."""
        if valuation_date is None:
            return None

        valuation_date = pd.Timestamp(valuation_date).tz_localize(None).normalize()
        session = self._valuation_sessions.get(valuation_date)
        if session is None:
            self._next_valuation_session += 1
            session = self._next_valuation_session
            self._valuation_sessions[valuation_date] = session
        return session
    
    def get_position_shares(self, ticker: str) -> float:
        """Get number of shares for a ticker"""
        return self.positions.get(ticker, 0)
    
    def get_position_value(self, ticker: str, price: float, currency: str = None,
                          date: datetime = None) -> float:
        """
        Calculate value of position in base currency
        
        Args:
            ticker: Ticker symbol
            price: Current price in ticker's currency
            currency: Currency of price (auto-detected if None)
            date: Valuation date for historical FX conversion
            
        Returns:
            Position value in base currency
        """
        shares = self.positions.get(ticker, 0)
        if shares == 0:
            return 0.0
        
        # Get currency
        if currency is None:
            currency = self.position_currencies.get(ticker)
            if currency is None:
                currency = self.fx_converter.get_ticker_currency(ticker)
        
        # Calculate value in ticker's currency
        value_foreign = shares * price
        
        # Convert to base currency
        if currency != self.base_currency:
            value_base = self.fx_converter.convert_to_base(value_foreign, currency, date=date)
        else:
            value_base = value_foreign
        
        return value_base
    
    def _resolve_effective_price(self, ticker: str, prices: Dict[str, float], valuation_date: Optional[pd.Timestamp]) -> Optional[float]:
        """Resolve valuation price using spot or stale fallback rules."""
        valuation_session = self._get_valuation_session(valuation_date)
        effective_price = prices.get(ticker)
        if effective_price is not None:
            self._last_prices[ticker] = float(effective_price)
            if valuation_date is not None:
                self._last_price_dates[ticker] = valuation_date
            if valuation_session is not None:
                self._last_price_sessions[ticker] = valuation_session
            return float(effective_price)

        last_price = self._last_prices.get(ticker)
        last_price_date = self._last_price_dates.get(ticker)
        last_price_session = self._last_price_sessions.get(ticker)
        if last_price is None or valuation_date is None or last_price_date is None or valuation_session is None or last_price_session is None:
            msg = f"Missing valuation price for held ticker {ticker} and no historical fallback available"
            logger.error(msg)
            raise DataStalenessError(msg)

        stale_sessions = int(valuation_session - last_price_session)
        if stale_sessions > self.max_stale_price_days:
            msg = (
                f"Stale valuation price for {ticker}: last seen {last_price_date.date()} "
                f"({stale_sessions} stale valuation session(s), max={self.max_stale_price_days})"
            )
            logger.error(msg)
            raise DataStalenessError(msg)

        logger.warning(
            "Using stale fallback price for %s (%d stale valuation session(s)); last date=%s",
            ticker,
            stale_sessions,
            last_price_date.date(),
        )
        return float(last_price)

    def get_total_equity(self, prices: Dict[str, float], 
                        currencies: Dict[str, str] = None,
                        date: datetime = None) -> float:
        """
        Calculate total portfolio equity in base currency
        
        Args:
            prices: Dict of {ticker: current_price}
            currencies: Dict of {ticker: currency} (optional)
            date: Valuation date for historical FX conversion
            
        Returns:
            Total equity in base currency
        """
        # Sum position values
        position_value = 0.0
        valuation_date = pd.Timestamp(date).tz_localize(None).normalize() if date is not None else None

        for ticker in list(self.positions.keys()):
            effective_price = self._resolve_effective_price(ticker, prices, valuation_date)
            if effective_price is None:
                continue

            currency = currencies.get(ticker) if currencies else None
            position_value += self.get_position_value(ticker, float(effective_price), currency, date=date)

        return self.cash + position_value
    
    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade and update portfolio state
        
        Args:
            trade: Trade object with execution details
            
        Returns:
            True if successful, False if insufficient cash
        """
        # Convert trade value to base currency
        if trade.currency != self.base_currency:
            trade_value_base = self.fx_converter.convert_to_base(
                trade.value, trade.currency, date=trade.date
            )
            cost_base = self.fx_converter.convert_to_base(
                trade.commission + trade.slippage, trade.currency, date=trade.date
            )
        else:
            trade_value_base = trade.value
            cost_base = trade.commission + trade.slippage
        
        if trade.action == 'BUY':
            # Check sufficient cash
            total_cost = trade_value_base + cost_base + trade.fx_cost
            if total_cost > self.cash:
                logger.warning(
                    "Insufficient cash for %s: need %.2f, have %.2f",
                    trade.ticker,
                    total_cost,
                    self.cash,
                )
                return False
            
            # Update cash
            self.cash -= total_cost
            
            # Update positions
            self.positions[trade.ticker] = self.positions.get(trade.ticker, 0) + trade.shares
            self.position_currencies[trade.ticker] = trade.currency
            
            logger.info(
                "BUY %.2f %s @ %.2f %s (cost: %.2f %s)",
                trade.shares,
                trade.ticker,
                trade.price,
                trade.currency,
                total_cost,
                self.base_currency,
            )
            
        else:  # SELL
            # Check sufficient shares
            current_shares = self.positions.get(trade.ticker, 0)
            if trade.shares > current_shares + 0.001:  # Allow small floating point error
                logger.warning(
                    "Insufficient shares for %s: need %s, have %s",
                    trade.ticker,
                    trade.shares,
                    current_shares,
                )
                return False
            
            # Update cash
            net_proceeds = trade_value_base - cost_base - trade.fx_cost
            self.cash += net_proceeds
            
            # Update positions
            new_shares = current_shares - trade.shares
            if abs(new_shares) < 0.001:  # Close to zero
                del self.positions[trade.ticker]
                if trade.ticker in self.position_currencies:
                    del self.position_currencies[trade.ticker]
            else:
                self.positions[trade.ticker] = new_shares
            
            logger.info(
                "SELL %.2f %s @ %.2f %s (proceeds: %.2f %s)",
                trade.shares,
                trade.ticker,
                trade.price,
                trade.currency,
                net_proceeds,
                self.base_currency,
            )
        
        # Record trade
        self.trades.append(trade)
        return True
    
    def record_equity(self, date: datetime, equity: float, cash: Optional[float] = None) -> None:
        """
        Record daily equity snapshot
        
        Args:
            date: Date
            equity: Total equity value
        """
        self.date_history.append(date)
        self.equity_history.append(equity)
        self.cash_history.append(float(self.cash if cash is None else cash))
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as pandas Series
        
        Returns:
            Series with dates as index and equity values
        """
        return pd.Series(self.equity_history, index=self.date_history)

    def get_cash_curve(self) -> pd.Series:
        """Get cash curve as pandas Series aligned to equity timestamps."""
        return pd.Series(self.cash_history, index=self.date_history)
    
    def get_returns(self) -> pd.Series:
        """
        Calculate daily returns
        
        Returns:
            Series of daily returns
        """
        equity = self.get_equity_curve()
        return equity.pct_change()
    
    def get_weights(self, prices: Dict[str, float],
                   currencies: Dict[str, str] = None,
                   date: datetime = None) -> Dict[str, float]:
        """
        Get current position weights
        
        Args:
            prices: Dict of {ticker: price}
            currencies: Dict of {ticker: currency}
            date: Valuation date for historical FX conversion
            
        Returns:
            Dict of {ticker: weight}
        """
        equity = self.get_total_equity(prices, currencies, date=date)
        
        if equity == 0:
            return {}
        
        valuation_date = pd.Timestamp(date).tz_localize(None).normalize() if date is not None else None

        weights = {}
        for ticker in self.positions:
            if self.positions[ticker] <= 0:
                continue
            effective_price = self._resolve_effective_price(ticker, prices, valuation_date)
            if effective_price is None:
                continue
            currency = currencies.get(ticker) if currencies else None
            value = self.get_position_value(ticker, effective_price, currency, date=date)
            weights[ticker] = value / equity
        
        return weights
    
    def get_cash_weight(self, prices: Dict[str, float],
                       currencies: Dict[str, str] = None,
                       date: datetime = None) -> float:
        """Get weight of cash in portfolio"""
        equity = self.get_total_equity(prices, currencies, date=date)
        return self.cash / equity if equity > 0 else 1.0
    
    def get_trade_summary(self) -> Dict:
        """
        Get summary statistics of trades
        
        Returns:
            Dict with trade statistics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'total_buys': 0,
                'total_sells': 0,
                'total_commission': 0,
                'total_slippage': 0,
                'total_fx_cost': 0,
                'total_cost': 0
            }
        
        buys = [t for t in self.trades if t.action == 'BUY']
        sells = [t for t in self.trades if t.action == 'SELL']
        
        return {
            'total_trades': len(self.trades),
            'total_buys': len(buys),
            'total_sells': len(sells),
            'total_commission': sum(t.commission for t in self.trades),
            'total_slippage': sum(t.slippage for t in self.trades),
            'total_fx_cost': sum(t.fx_cost for t in self.trades),
            'total_cost': sum(t.total_cost() for t in self.trades)
        }
    
    def get_position_summary(self, prices: Dict[str, float] = None,
                           currencies: Dict[str, str] = None,
                           date: datetime = None) -> pd.DataFrame:
        """
        Get summary of current positions
        
        Args:
            prices: Current prices (optional)
            currencies: Ticker currencies (optional)
            
        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for ticker, shares in self.positions.items():
            row = {
                'ticker': ticker,
                'shares': shares,
                'currency': self.position_currencies.get(ticker, 'USD')
            }
            
            if prices and ticker in prices:
                currency = currencies.get(ticker) if currencies else row['currency']
                value = self.get_position_value(ticker, prices[ticker], currency, date=date)
                row['price'] = prices[ticker]
                row['value'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_state_summary(self, prices: Dict[str, float] = None,
                         currencies: Dict[str, str] = None,
                         date: datetime = None) -> Dict:
        """
        Get complete portfolio state summary
        
        Args:
            prices: Current prices
            currencies: Ticker currencies
            
        Returns:
            Dict with portfolio state
        """
        equity = self.get_total_equity(prices, currencies, date=date) if prices else None
        
        return {
            'cash': self.cash,
            'num_positions': len(self.positions),
            'equity': equity,
            'cash_weight': self.cash / equity if equity else 1.0,
            'total_trades': len(self.trades),
            'base_currency': self.base_currency
        }
    
    def __repr__(self):
        return (f"Portfolio(cash={self.cash:.2f} {self.base_currency}, "
                f"positions={len(self.positions)}, trades={len(self.trades)})")
