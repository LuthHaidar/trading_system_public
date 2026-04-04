from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract, Order, Trade
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from utils.logger import get_logger
from utils.currency import CurrencyConverter

logger = get_logger(__name__)

EXCHANGE_TO_SUFFIX = {meta['exchange']: suffix for suffix, meta in CurrencyConverter.TICKER_MARKETS.items()}
US_EXCHANGES = {'SMART', 'NYSE', 'NASDAQ', 'ARCA', 'BATS', 'IEX', 'AMEX'}


class IBKRClient:
    """
    Interactive Brokers API client using ib_insync
    
    Handles:
    - Connection management
    - Order placement and monitoring
    - Position and account queries
    - Error handling and reconnection
    """
    
    def __init__(self, config: Dict):
        """
        Initialize IBKR client
        
        Args:
            config: IBKR configuration dict
        """
        self.config = config
        self.ib = IB()
        self.connected = False
        self.connection_time = None
        
        # Connection parameters
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497)  # 7497 for paper, 7496 for live
        self.client_id = config.get('client_id', 1)
        self.timeout = config.get('timeout', 60)
        
        # Paper trading flag
        self.paper_trading = (self.port == 7497)
        
        # Currency converter
        self.base_currency = config.get('base_currency', 'SGD')
        self.fx_converter = CurrencyConverter(self.base_currency)
        self._recent_orders = set()
        self._contract_cache: Dict[str, Contract] = {}
        self.contract_overrides = config.get('contract_overrides', {}) or {}
        
        logger.info("IBKR Client initialized: %s:%s (%s trading)", self.host, self.port, "PAPER" if self.paper_trading else "LIVE")
    
    def connect(self) -> bool:
        """
        Connect to IBKR TWS/Gateway
        
        Returns:
            True if successful
        """
        try:
            if self.connected:
                logger.warning("Already connected")
                return True
            
            logger.info("Connecting to IBKR at %s:%s...", self.host, self.port)
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout
            )
            
            self.connected = True
            self.connection_time = datetime.now()
            
            # Log account info
            account_info = self.get_account_summary()
            logger.info("Connected successfully to account: %s", account_info.get('account_id', 'Unknown'))
            logger.info("Account value: %,.2f %s", account_info.get('net_liquidation', 0), account_info.get('currency', 'USD')  )
            
            return True
            
        except Exception as e:
            logger.error("Connection failed: %s", e)
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from IBKR"""
        if self.connected:
            try:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IBKR")
            except Exception as e:
                logger.error("Error disconnecting: %s", e)
    
    def reconnect(self) -> bool:
        """Attempt to reconnect"""
        logger.info("Attempting to reconnect...")
        self.disconnect()
        time.sleep(2)
        return self.connect()
    
    def is_connected(self) -> bool:
        """Check if connected"""
        if not self.connected:
            return False
        
        try:
            # Verify connection is still alive
            self.ib.reqCurrentTime()
            return True
        except Exception as exc:
            logger.debug("IBKR connection heartbeat failed: %s", exc)
            self.connected = False
            return False
    
    def get_account_summary(self) -> Dict:
        """
        Get account summary
        
        Returns:
            Dict with account information
        """
        if not self.is_connected():
            logger.error("Not connected")
            return {}
        
        try:
            account_values = self.ib.accountSummary()
            
            summary = {}
            for item in account_values:
                if item.tag == 'AccountCode':
                    summary['account_id'] = item.value
                elif item.tag == 'NetLiquidation':
                    summary['net_liquidation'] = float(item.value)
                    summary['currency'] = item.currency
                elif item.tag == 'TotalCashValue':
                    summary['cash'] = float(item.value)
                elif item.tag == 'BuyingPower':
                    summary['buying_power'] = float(item.value)
                elif item.tag == 'GrossPositionValue':
                    summary['position_value'] = float(item.value)
            
            return summary
            
        except Exception as e:
            logger.error("Error getting account summary: %s", e)
            return {}
    
    def get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions
        
        Returns:
            Dict of {ticker: {'shares': float, 'avg_cost': float, 'market_value': float}}
        """
        if not self.is_connected():
            logger.error("Not connected")
            return {}
        
        try:
            positions = self.ib.positions()
            
            position_dict = {}
            for position in positions:
                exchange = self._resolve_contract_exchange(position.contract)
                ticker = self._ibkr_symbol_to_yahoo(
                    symbol=position.contract.symbol,
                    exchange=exchange,
                )
                position_dict[ticker] = {
                    'shares': position.position,
                    'avg_cost': position.avgCost,
                    'market_value': position.position * position.avgCost,
                    'contract': position.contract
                }
            
            logger.info("Current positions: %d tickers", len(position_dict))
            return position_dict
            
        except Exception as e:
            logger.error("Error getting positions: %s", e)
            return {}
    
    def create_contract(self, ticker: str, exchange: str = 'SMART',
                       currency: str = None) -> Contract:
        """
        Create (and cache) an IBKR Stock contract for a Yahoo-style ticker.
        """
        cached = self._contract_cache.get(ticker)
        if cached is not None:
            return cached

        contract_cfg = self.contract_overrides.get(ticker, {}) if isinstance(self.contract_overrides, dict) else {}

        symbol = ticker
        contract_exchange = exchange
        contract_currency = currency or self.fx_converter.get_ticker_currency(ticker)

        if contract_cfg:
            symbol = str(contract_cfg.get('symbol', symbol))
            contract_exchange = str(contract_cfg.get('exchange', contract_exchange))
            contract_currency = str(contract_cfg.get('currency', contract_currency))
        elif '.' in ticker:
            base_symbol, suffix = ticker.rsplit('.', 1)
            market = CurrencyConverter.TICKER_MARKETS.get(suffix)
            if market:
                symbol = base_symbol
                contract_exchange = market['exchange']
                contract_currency = market['currency']

        contract = Stock(symbol, contract_exchange, contract_currency)

        primary_exchange = contract_cfg.get('primary_exchange') if contract_cfg else None
        if primary_exchange is None:
            if contract_exchange == 'SMART':
                primary_exchange = 'SMART'
            else:
                primary_exchange = contract_exchange
        contract.primaryExchange = primary_exchange

        self._contract_cache[ticker] = contract
        return contract

    @staticmethod
    def _resolve_contract_exchange(contract: Contract) -> str:
        """Resolve exchange for ticker reconstruction.

        IBKR positions can be SMART-routed while carrying the venue in
        ``primaryExchange``. Prefer that venue when available so non-US
        positions map back to the correct Yahoo-style suffix.
        """
        exchange = getattr(contract, 'exchange', None)
        primary_exchange = getattr(contract, 'primaryExchange', None)
        if exchange in US_EXCHANGES and primary_exchange:
            return primary_exchange
        return exchange

    @staticmethod
    def _ibkr_symbol_to_yahoo(symbol: str, exchange: str) -> str:
        """Map IBKR symbol+exchange back to internal Yahoo-style ticker key."""
        if exchange in US_EXCHANGES:
            return symbol
        suffix = EXCHANGE_TO_SUFFIX.get(exchange)
        if suffix is None:
            logger.warning(
                "Unknown IBKR exchange '%s' for symbol '%s'; returning bare symbol",
                exchange,
                symbol,
            )
            return symbol
        return f"{symbol}.{suffix}"

    def validate_universe(self, tickers: List[str]) -> Dict[str, bool]:
        """Qualify every ticker once before trading starts and cache successful contracts."""
        results: Dict[str, bool] = {}
        for ticker in tickers:
            try:
                self._ensure_contract_qualified(ticker)
                results[ticker] = True
            except Exception as exc:
                logger.error("Universe validation failed for %s: %s", ticker, exc)
                results[ticker] = False
        return results

    def _ensure_contract_qualified(self, ticker: str) -> Contract:
        """Qualify a contract if needed and cache the resolved IBKR contract."""
        contract = self.create_contract(ticker)
        if getattr(contract, 'conId', 0):
            return contract

        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Unable to qualify contract for {ticker}")
        self._contract_cache[ticker] = qualified[0]
        return qualified[0]

    def place_market_order(self, ticker: str, quantity: int,
                          action: str = 'BUY') -> Optional[Trade]:
        """
        Place market order
        
        Args:
            ticker: Ticker symbol
            quantity: Number of shares (positive)
            action: 'BUY' or 'SELL'
            
        Returns:
            Order object or None
        """
        if not self.is_connected():
            logger.error("Not connected")
            return None
        
        try:
            contract = self._ensure_contract_qualified(ticker)
            
            # Create order
            order = MarketOrder(action, abs(quantity))
            
            # Duplicate trade prevention against currently open orders / recent intents
            order_key = (ticker, abs(quantity), action)
            if order_key in self._recent_orders:
                logger.warning("Duplicate order prevented: %s", order_key)
                return None

            open_orders = self.get_open_orders()
            for open_trade in open_orders:
                if (open_trade.contract.symbol == ticker and
                        open_trade.order.action == action and
                        int(open_trade.order.totalQuantity) == int(abs(quantity))):
                    logger.warning("Duplicate open order detected, skipping: %s", order_key)
                    return None

            # Place order
            trade = self.ib.placeOrder(contract, order)
            self._recent_orders.add(order_key)
            
            logger.info("Placed %s order: %d %s", action, quantity, ticker)
            logger.debug("Order ID: %s", trade.order.orderId)
            
            return trade
            
        except Exception as e:
            logger.error("Error placing market order for %s: %s", ticker, e)
            return None
    
    def place_limit_order(self, ticker: str, quantity: int,
                         limit_price: float, action: str = 'BUY') -> Optional[Trade]:
        """
        Place limit order
        
        Args:
            ticker: Ticker symbol
            quantity: Number of shares
            limit_price: Limit price
            action: 'BUY' or 'SELL'
            
        Returns:
            Order object or None
        """
        if not self.is_connected():
            logger.error("Not connected")
            return None
        
        try:
            contract = self._ensure_contract_qualified(ticker)

            order = LimitOrder(action, abs(quantity), limit_price)
            trade = self.ib.placeOrder(contract, order)
            
            logger.info("Placed %s limit order: %d %s @ %f", action, quantity, ticker, limit_price)
            return trade
            
        except Exception as e:
            logger.error("Error placing limit order for %s: %s", ticker, e)
            return None
    
    def cancel_order(self, order) -> bool:
        """
        Cancel an order
        
        Args:
            order: Order object from place_order
            
        Returns:
            True if successful
        """
        try:
            self.ib.cancelOrder(order.order)
            logger.info("Cancelled order %s", order.order.orderId)
            return True
        except Exception as e:
            logger.error("Error cancelling order: %s", e)
            return False
    
    def get_open_orders(self) -> List:
        """Get list of open orders"""
        if not self.is_connected():
            return []
        
        try:
            trades = self.ib.openTrades()
            logger.info("Open orders: %d", len(trades))
            return trades
        except Exception as e:
            logger.error("Error getting open orders: %s", e)
            return []
    
    def wait_for_fills(self, timeout: int = 60) -> bool:
        """
        Wait for all orders to fill
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if all filled
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            open_orders = self.get_open_orders()
            
            if len(open_orders) == 0:
                logger.info("All orders filled")
                return True
            
            logger.info("Waiting for %d orders to fill...", len(open_orders))
            time.sleep(2)
        
        logger.warning("Timeout waiting for orders to fill")
        return False
    
    def get_market_price(self, ticker: str) -> Optional[float]:
        """
        Get current market price for ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Current price or None
        """
        if not self.is_connected():
            return None
        
        try:
            contract = self._ensure_contract_qualified(ticker)

            # Request market data
            ticker_data = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data
            
            # Get last price
            if ticker_data.last and ticker_data.last > 0:
                price = ticker_data.last
            elif ticker_data.close and ticker_data.close > 0:
                price = ticker_data.close
            else:
                price = None
            
            # Cancel market data subscription
            self.ib.cancelMktData(contract)
            
            if price:
                logger.debug("%s price: %f", ticker, price)
            
            return price
            
        except Exception as e:
            logger.error("Error getting market price for %s: %s", ticker, e)
            return None
    
    def get_market_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get market prices for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict of {ticker: price}
        """
        if not self.is_connected():
            return {}

        prices: Dict[str, float] = {}
        market_data_handles: List[Tuple[str, Contract, object]] = []

        # Request all subscriptions up-front so IBKR can populate them in parallel.
        for ticker in tickers:
            try:
                contract = self._ensure_contract_qualified(ticker)
                market_data_handles.append((ticker, contract, self.ib.reqMktData(contract)))
            except Exception as exc:
                logger.error("Error requesting market data for %s: %s", ticker, exc)

        # Single wait after issuing all reqMktData calls keeps wall time near O(1)
        # for the batch rather than O(n) sequential waits.
        if market_data_handles:
            self.ib.sleep(2)

        for ticker, contract, ticker_data in market_data_handles:
            try:
                if ticker_data is None:
                    price = None
                elif getattr(ticker_data, 'last', None) and ticker_data.last > 0:
                    price = float(ticker_data.last)
                elif getattr(ticker_data, 'close', None) and ticker_data.close > 0:
                    price = float(ticker_data.close)
                else:
                    price = None

                if price:
                    prices[ticker] = price
                    logger.debug("%s price: %f", ticker, price)
            except Exception as exc:
                logger.error("Error parsing market data for %s: %s", ticker, exc)
            finally:
                try:
                    self.ib.cancelMktData(contract)
                except Exception as cancel_exc:
                    logger.warning("Error cancelling market data for %s: %s", ticker, cancel_exc)

        return prices
    
    def reconcile_positions(self, target_positions: Dict[str, int]) -> List[Tuple[str, int, str]]:
        """
        Calculate orders needed to reach target positions
        
        Args:
            target_positions: Dict of {ticker: target_shares}
            
        Returns:
            List of (ticker, quantity, action) tuples
        """
        current_positions = self.get_positions()
        
        orders = []
        
        # Check positions to close
        for ticker in current_positions:
            current_shares = current_positions[ticker]['shares']
            target_shares = target_positions.get(ticker, 0)
            
            delta = target_shares - current_shares
            
            if abs(delta) >= 1:  # Only trade whole shares
                action = 'BUY' if delta > 0 else 'SELL'
                orders.append((ticker, int(abs(delta)), action))
        
        # Check positions to open
        for ticker in target_positions:
            if ticker not in current_positions and target_positions[ticker] > 0:
                orders.append((ticker, int(target_positions[ticker]), 'BUY'))
        
        invalid_targets = [t for t, qty in target_positions.items() if qty < 0]
        if invalid_targets:
            logger.error("Reconciliation check failed: negative targets for %s", invalid_targets)

        logger.info("Reconciliation: %d orders needed", len(orders))
        return orders
    
    def execute_rebalance(self, target_positions: Dict[str, int]) -> bool:
        """
        Execute orders to reach target positions
        
        Args:
            target_positions: Dict of {ticker: target_shares}
            
        Returns:
            True if successful
        """
        # Reset in-memory dedupe for each rebalance attempt so failed/partial
        # runs do not block legitimate retries.
        self._recent_orders.clear()

        if not self.is_connected():
            logger.error("Not connected")
            return False
        
        # Calculate orders
        orders = self.reconcile_positions(target_positions)
        
        if not orders:
            logger.info("No rebalancing needed")
            return True
        
        # Place orders
        trades = []
        for ticker, quantity, action in orders:
            trade = self.place_market_order(ticker, quantity, action)
            if trade:
                trades.append(trade)
            else:
                logger.error("Failed to place order: %s %d %s", action, quantity, ticker)
        
        # Wait for fills
        if trades:
            success = self.wait_for_fills(timeout=120)
            
            if success:
                logger.info("Rebalancing complete")
            else:
                logger.warning("Some orders may not have filled")
            
            if success:
                self._recent_orders.clear()
            return success
        
        return False
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
