import pandas as pd
from typing import Any, Dict, Optional
from datetime import datetime
from backtesting.portfolio import Trade
from utils.logger import get_logger
from utils.currency import CurrencyConverter

logger = get_logger(__name__)


class ExecutionEngine:
    """
    Simulates realistic order execution with transaction costs
    Models IBKR Pro pricing structure
    """
    
    def __init__(self, config: Dict, base_currency: str = 'SGD'):
        """
        Initialize execution engine
        
        Args:
            config: Configuration dict with cost parameters
            base_currency: Portfolio base currency
        """
        self.config = config
        self.base_currency = base_currency
        
        # IBKR Pro commission structure
        self.commission_per_share = config.get('commission_per_share', 0.0035)
        self.commission_min = config.get('commission_min', 0.35)
        self.commission_max_pct = config.get('commission_max_pct', 0.01)
        
        # Non-US stocks
        self.non_us_commission_rate = config.get('non_us_commission_rate', 0.001)
        self.non_us_commission_min = config.get('non_us_commission_min', 4.0)

        # Composable fee components
        self.cost_components = config.get('components', {
            'broker': {
                'enabled': True,
                'us_commission_per_share': self.commission_per_share,
                'us_commission_min': self.commission_min,
                'us_commission_max_pct': self.commission_max_pct,
                'non_us_commission_rate': self.non_us_commission_rate,
                'non_us_commission_min': self.non_us_commission_min,
            },
            'regulatory': {
                'enabled': True,
                'us_sec_rate': 0.000008,
                'us_taf_per_share': 0.000145,
                'us_taf_min': 0.01,
                'us_taf_max': 7.27,
            },
            'exchange': {
                'enabled': True,
                'per_share': 0.0,
                'ad_valorem_rate': 0.0,
            },
            'market_taxes': {
                'enabled': True,
                'by_market': {}
            }
        })
        
        # FX conversion
        self.fx_conversion_rate = config.get('fx_conversion_rate', 0.00002)  # 2 bps
        
        # Slippage
        self.slippage_bps = config.get('slippage_bps', {
            'high_liquidity': 2,
            'medium_liquidity': 5,
            'low_liquidity': 10
        })
        self.slippage_model = config.get('slippage_model', {})
        self.slippage_mode = self.slippage_model.get('mode', 'heuristic')
        logger.info(f"Slippage model mode: {self.slippage_mode}")

        # Simple spread simulation
        self.spread_bps = config.get('spread_bps', {
            'high_liquidity': 1.0,
            'medium_liquidity': 3.0,
            'low_liquidity': 8.0
        })
        
        # Currency converter
        self.fx_converter = CurrencyConverter(base_currency)
        
        logger.info("ExecutionEngine initialized with IBKR Pro cost structure")
    
    def calculate_commission(self, shares: float, price: float, ticker: str,
                             action: str = 'BUY') -> float:
        """
        Calculate total commission and fee stack as composable components.

        Args:
            shares: Number of shares
            price: Price per share
            ticker: Ticker symbol
            action: BUY or SELL

        Returns:
            Total transaction fee cost
        """
        trade_value = abs(shares * price)
        market_info = self._get_market_info(ticker)

        broker_commission = self._calculate_broker_commission(shares, trade_value, market_info)
        regulatory_fees = self._calculate_regulatory_fees(shares, trade_value, action, market_info)
        exchange_fees = self._calculate_exchange_fees(shares, trade_value, market_info)
        market_taxes = self._calculate_market_taxes(trade_value, action, market_info)

        return broker_commission + regulatory_fees + exchange_fees + market_taxes

    def _get_market_info(self, ticker: str) -> Dict:
        """Get market metadata for ticker."""
        return self.fx_converter.get_ticker_market(ticker)

    def _calculate_broker_commission(self, shares: float, trade_value: float, market_info: Dict) -> float:
        """Calculate broker commission component."""
        broker_cfg = self.cost_components.get('broker', {})
        if not broker_cfg.get('enabled', True):
            return 0.0

        if market_info.get('is_us', False):
            comm = abs(shares) * broker_cfg.get('us_commission_per_share', self.commission_per_share)
            comm = max(comm, broker_cfg.get('us_commission_min', self.commission_min))
            comm = min(comm, trade_value * broker_cfg.get('us_commission_max_pct', self.commission_max_pct))
            return comm

        non_us_rate = broker_cfg.get('non_us_commission_rate', self.non_us_commission_rate)
        non_us_min = broker_cfg.get('non_us_commission_min', self.non_us_commission_min)
        return max(non_us_min, trade_value * non_us_rate)

    def _calculate_regulatory_fees(self, shares: float, trade_value: float, action: str, market_info: Dict) -> float:
        """Calculate regulatory fee component (e.g., SEC/TAF)."""
        reg_cfg = self.cost_components.get('regulatory', {})
        if not reg_cfg.get('enabled', True):
            return 0.0

        fees = 0.0
        if market_info.get('country') == 'US' and action == 'SELL':
            sec_rate = reg_cfg.get('us_sec_rate', 0.0)
            taf_per_share = reg_cfg.get('us_taf_per_share', 0.0)
            taf_min = reg_cfg.get('us_taf_min', 0.0)
            taf_max = reg_cfg.get('us_taf_max', float('inf'))

            sec_fee = trade_value * sec_rate
            taf_fee = abs(shares) * taf_per_share
            if taf_fee > 0:
                taf_fee = min(max(taf_fee, taf_min), taf_max)
            fees += sec_fee + taf_fee

        return fees

    def _calculate_exchange_fees(self, shares: float, trade_value: float, market_info: Dict) -> float:
        """Calculate exchange/clearing/pass-through fee component."""
        exch_cfg = self.cost_components.get('exchange', {})
        if not exch_cfg.get('enabled', True):
            return 0.0

        per_share = exch_cfg.get('per_share', 0.0)
        ad_valorem = exch_cfg.get('ad_valorem_rate', 0.0)
        return (abs(shares) * per_share) + (trade_value * ad_valorem)

    def _calculate_market_taxes(self, trade_value: float, action: str, market_info: Dict) -> float:
        """Calculate market-specific taxes/stamp duty."""
        tax_cfg = self.cost_components.get('market_taxes', {})
        if not tax_cfg.get('enabled', True):
            return 0.0

        taxes_cfg = tax_cfg.get('by_market', {})
        market_key = market_info.get('suffix') or market_info.get('country') or market_info.get('market')
        market_tax = taxes_cfg.get(market_key, {})
        if not market_tax and market_info.get('country'):
            market_tax = taxes_cfg.get(market_info['country'], {})

        if not market_tax:
            return 0.0

        if market_tax.get('side', 'both').upper() not in ('BOTH', action.upper()):
            return 0.0

        return trade_value * market_tax.get('rate', 0.0)

    def calculate_slippage(self, shares: float, price: float,
                          avg_volume: float, ticker: str,
                          volatility: Optional[float] = None,
                          spread_proxy_bps: Optional[float] = None,
                          market_bucket: Optional[str] = None,
                          timing: Optional[str] = None) -> float:
        """
        Calculate slippage based on liquidity and order size
        
        Args:
            shares: Number of shares to trade
            price: Current price
            avg_volume: Average daily volume
            ticker: Ticker symbol
            
        Returns:
            Slippage cost
        """
        if self.slippage_mode == 'parameterized':
            base_bps = self._calculate_parameterized_slippage_bps(
                shares=shares,
                avg_volume=avg_volume,
                ticker=ticker,
                volatility=volatility,
                spread_proxy_bps=spread_proxy_bps,
                market_bucket=market_bucket
            )
        else:
            base_bps = self._calculate_heuristic_slippage_bps(
                shares=shares,
                avg_volume=avg_volume,
                ticker=ticker
            )
        
        total_bps = base_bps

        # Calculate slippage in price terms
        slippage_cost = abs(shares) * price * (total_bps / 10000)

        return slippage_cost
    
    def _calculate_heuristic_slippage_bps(self, shares: float,
                                          avg_volume: Optional[float],
                                          ticker: str) -> float:
        """Current heuristic fallback model for slippage in basis points."""
        if avg_volume == 0 or avg_volume is None:
            base_bps = self.slippage_bps['medium_liquidity']
        else:
            base_bps = self._infer_liquidity_bucket_base_bps(ticker)
            volume_pct = abs(shares) / avg_volume
            impact_multiplier = 1.0 + (volume_pct * 10)  # 10x sensitivity
            base_bps = base_bps * min(impact_multiplier, 5.0)  # Cap at 5x

        return base_bps

    def _calculate_parameterized_slippage_bps(self, shares: float,
                                              avg_volume: Optional[float],
                                              ticker: str,
                                              volatility: Optional[float] = None,
                                              spread_proxy_bps: Optional[float] = None,
                                              market_bucket: Optional[str] = None) -> float:
        """Parameterized slippage model in basis points."""
        model_cfg = self.slippage_model
        default_bucket = model_cfg.get('default_bucket', 'medium_liquidity')
        bucket = market_bucket or model_cfg.get('bucket_overrides', {}).get(ticker) or default_bucket

        bucket_base_bps = model_cfg.get('bucket_base_bps', {})
        inferred_base_bps = self._infer_liquidity_bucket_base_bps(ticker)
        base_bps = bucket_base_bps.get(bucket, inferred_base_bps)

        spread_weight = model_cfg.get('spread_weight', 0.5)
        vol_weight = model_cfg.get('volatility_weight', 25.0)
        participation_weight = model_cfg.get('participation_weight', 100.0)
        participation_exponent = model_cfg.get('participation_exponent', 0.6)
        max_participation = model_cfg.get('max_participation_rate', 1.0)
        max_slippage_bps = model_cfg.get('max_slippage_bps', 100.0)

        effective_spread = spread_proxy_bps
        if effective_spread is None:
            bucket_spread_proxy = model_cfg.get('bucket_spread_proxy_bps', {})
            effective_spread = bucket_spread_proxy.get(bucket, 0.0)

        effective_volatility = max(volatility or 0.0, 0.0)

        if avg_volume and avg_volume > 0:
            participation_rate = min(abs(shares) / avg_volume, max_participation)
        else:
            participation_rate = max_participation

        bps = (
            base_bps
            + (spread_weight * effective_spread)
            + (vol_weight * effective_volatility)
            + (participation_weight * (participation_rate ** participation_exponent))
        )

        return min(bps, max_slippage_bps)

    def _infer_liquidity_bucket_base_bps(self, ticker: str) -> float:
        """
        Get base slippage in basis points by ticker
        
        High liquidity: SPY, QQQ, IWM, EEM, etc.
        Medium liquidity: Most stocks and ETFs
        Low liquidity: Small cap, international
        """
        # High liquidity tickers
        high_liquidity = ['SPY', 'QQQ', 'IWM', 'EEM', 'VTI', 'VOO', 
                         'GLD', 'TLT', 'AGG', 'DIA']
        
        if ticker in high_liquidity:
            return self.slippage_bps['high_liquidity']
        
        # Low liquidity indicators
        if '.' in ticker and ticker.split('.')[-1] not in ['L', 'TO']:
            return self.slippage_bps['low_liquidity']
        
        # Default medium liquidity
        return self.slippage_bps['medium_liquidity']
    

    def _infer_spread_bps(self, ticker: str) -> float:
        base_bps = self._infer_liquidity_bucket_base_bps(ticker)
        if base_bps <= self.slippage_bps.get('high_liquidity', 2):
            return self.spread_bps.get('high_liquidity', 1.0)
        if base_bps >= self.slippage_bps.get('low_liquidity', 10):
            return self.spread_bps.get('low_liquidity', 8.0)
        return self.spread_bps.get('medium_liquidity', 3.0)

    def _build_trade(self, ticker: str, action: str, shares: int, execution_price: float,
                     reference_price: float, avg_volume: float, date: datetime,
                     currency: Optional[str], order_type: str,
                     parent_order_id: Optional[str] = None,
                     run_id: Optional[str] = None,
                     signal_date: Optional[datetime] = None,
                     decision: Optional[str] = None,
                     decision_reason: Optional[str] = None,
                     signal_strength: Optional[float] = None,
                     signal_confidence: Optional[float] = None,
                     weight_delta: Optional[float] = None,
                     current_weight: Optional[float] = None,
                     target_weight: Optional[float] = None,
                     current_shares: Optional[float] = None,
                     target_shares: Optional[float] = None) -> Trade:
        if currency is None:
            currency = self.fx_converter.get_ticker_currency(ticker)

        commission = self.calculate_commission(shares, execution_price, ticker, action=action)
        model_slippage = self.calculate_slippage(
            shares,
            execution_price,
            avg_volume,
            ticker,
        )
        spread_cost = shares * abs(execution_price - reference_price)
        slippage = model_slippage + spread_cost
        trade_value = shares * execution_price
        fx_cost = self.calculate_fx_cost(trade_value, currency, date=date)

        return Trade(
            date=date,
            ticker=ticker,
            action=action,
            shares=shares,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            fx_cost=fx_cost,
            value=trade_value,
            currency=currency,
            expected_price=reference_price,
            benchmark_price=reference_price,
            order_type=order_type,
            parent_order_id=parent_order_id,
            run_id=run_id,
            signal_date=signal_date,
            decision=decision,
            decision_reason=decision_reason,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
            weight_delta=weight_delta,
            current_weight=current_weight,
            target_weight=target_weight,
            current_shares=current_shares,
            target_shares=target_shares,
            executed_shares=float(shares),
        )

    def calculate_fx_cost(self, trade_value: float, from_currency: str,
                         date: datetime = None) -> float:
        """
        Calculate FX conversion cost
        
        Args:
            trade_value: Trade value in foreign currency
            from_currency: Source currency
            date: Trade/bar date for historical FX conversion
            
        Returns:
            FX conversion cost in base currency
        """
        if from_currency == self.base_currency:
            return 0.0
        
        # Convert to base currency
        value_base = self.fx_converter.convert_to_base(trade_value, from_currency, date=date)
        
        # Apply conversion fee (2 bps)
        fx_cost = value_base * self.fx_conversion_rate
        
        return fx_cost
    
    def execute_order(self, ticker: str, target_shares: float, 
                     current_shares: float, price: float,
                     avg_volume: float, date: datetime,
                     currency: str = None,
                     order_type: str = 'MARKET',
                     timing: str = None,
                     run_id: Optional[str] = None,
                     signal_date: Optional[datetime] = None,
                     decision: Optional[str] = None,
                     decision_reason: Optional[str] = None,
                     signal_strength: Optional[float] = None,
                     signal_confidence: Optional[float] = None,
                     weight_delta: Optional[float] = None,
                     current_weight: Optional[float] = None,
                     target_weight: Optional[float] = None) -> Optional[Trade]:
        """
        Execute an order to move from current to target shares
        
        Args:
            ticker: Ticker symbol
            target_shares: Desired number of shares
            current_shares: Current number of shares
            price: Current price
            avg_volume: Average daily volume
            date: Execution date
            currency: Currency of price (auto-detect if None)
            
        Returns:
            Trade object or None if no trade needed
        """
        # Calculate shares to trade
        shares_to_trade = target_shares - current_shares
        
        # Don't trade fractional shares (IBKR requires whole shares for most stocks)
        if abs(shares_to_trade) < 1:
            return None
        
        shares_to_trade = int(round(shares_to_trade))
        
        if shares_to_trade == 0:
            return None
        
        # Determine action
        action = 'BUY' if shares_to_trade > 0 else 'SELL'
        normalized_order_type = 'MARKET'
        if (order_type or 'MARKET').upper() != 'MARKET':
            logger.warning("Order type '%s' is not supported for retail mode; using MARKET", order_type)

        filled_shares = abs(shares_to_trade)

        spread_bps = self._infer_spread_bps(ticker)
        half_spread = price * (spread_bps / 10000.0) / 2.0
        execution_price = price + half_spread if action == 'BUY' else max(price - half_spread, 0.0)

        return self._build_trade(
            ticker=ticker,
            action=action,
            shares=filled_shares,
            execution_price=execution_price,
            reference_price=price,
            avg_volume=avg_volume,
            date=date,
            currency=currency,
            order_type=normalized_order_type,
            run_id=run_id,
            signal_date=signal_date,
            decision=decision,
            decision_reason=decision_reason,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
            weight_delta=weight_delta,
            current_weight=current_weight,
            target_weight=target_weight,
            current_shares=current_shares,
            target_shares=target_shares,
        )

    @staticmethod
    def infer_decision_label(current_shares: float, target_shares: float) -> str:
        """Classify position intent for journaling and trade metadata."""
        if target_shares > current_shares and current_shares <= 0:
            return 'ENTRY'
        if target_shares < current_shares and target_shares <= 0:
            return 'EXIT'
        if target_shares > current_shares:
            return 'INCREASE'
        if target_shares < current_shares:
            return 'DECREASE'
        return 'HOLD'
    
    def execute_rebalance(self, target_positions: Dict[str, int],
                         current_positions: Dict[str, float],
                         prices: Dict[str, float],
                         volumes: Dict[str, float],
                         currencies: Dict[str, str],
                         date: datetime,
                         timing: str = None,
                         target_weights: Optional[Dict[str, float]] = None,
                         current_weights: Optional[Dict[str, float]] = None,
                         signal_date: Optional[datetime] = None,
                         confidence_by_ticker: Optional[Dict[str, float]] = None,
                         signal_meta: Optional[Dict[str, Dict[str, Any]]] = None,
                         decision_reason: Optional[str] = None,
                         run_id: Optional[str] = None) -> list:
        """
        Execute multiple orders to rebalance portfolio
        
        Args:
            target_positions: Dict of {ticker: target_shares}
            current_positions: Dict of {ticker: current_shares}
            prices: Dict of {ticker: price}
            volumes: Dict of {ticker: avg_volume}
            currencies: Dict of {ticker: currency}
            date: Execution date
            
        Returns:
            List of Trade objects
        """
        if timing is None:
            timing = execution_session

        trades = []
        
        # Close positions not in targets
        for ticker in current_positions:
            if ticker not in target_positions:
                target_positions[ticker] = 0
        
        # Execute each order
        for ticker, target_shares in target_positions.items():
            if ticker not in prices:
                logger.warning(f"No price available for {ticker}, skipping")
                continue
            
            current_shares = current_positions.get(ticker, 0)
            
            if target_shares == current_shares:
                continue

            current_weight = (
                float(current_weights.get(ticker, 0.0))
                if current_weights is not None
                else 0.0
            )
            target_weight = (
                float(target_weights.get(ticker, 0.0))
                if target_weights is not None
                else 0.0
            )
            weight_delta = (
                float(target_weight - current_weight)
                if target_weights is not None and current_weights is not None
                else None
            )
            ticker_meta = (signal_meta or {}).get(ticker, {}) or {}
            resolved_reason = ticker_meta.get('reason') or decision_reason or 'strategy_rebalance'
            resolved_signal_strength = ticker_meta.get('signal_strength')
            if resolved_signal_strength is not None:
                try:
                    resolved_signal_strength = float(resolved_signal_strength)
                except (TypeError, ValueError):
                    resolved_signal_strength = None
            resolved_confidence = ticker_meta.get('confidence')
            if resolved_confidence is None and confidence_by_ticker and ticker in confidence_by_ticker:
                resolved_confidence = confidence_by_ticker.get(ticker)
            if resolved_confidence is not None:
                try:
                    resolved_confidence = float(resolved_confidence)
                except (TypeError, ValueError):
                    resolved_confidence = None
            
            trade = self.execute_order(
                ticker=ticker,
                target_shares=target_shares,
                current_shares=current_shares,
                price=prices[ticker],
                avg_volume=volumes.get(ticker, 0),
                date=date,
                currency=currencies.get(ticker),
                timing=timing,
                run_id=run_id,
                signal_date=signal_date,
                decision=self.infer_decision_label(current_shares, target_shares),
                decision_reason=resolved_reason,
                signal_strength=resolved_signal_strength,
                signal_confidence=resolved_confidence,
                weight_delta=weight_delta,
                current_weight=(current_weight if current_weights is not None else None),
                target_weight=(target_weight if target_weights is not None else None),
            )
            
            if trade:
                trades.append(trade)
        
        return trades
    
    def get_cost_summary(self, trades: list) -> Dict:
        """
        Get summary of transaction costs
        
        Args:
            trades: List of Trade objects
            
        Returns:
            Dict with cost breakdown
        """
        if not trades:
            return {
                'total_commission': 0,
                'total_slippage': 0,
                'total_fx_cost': 0,
                'total_cost': 0,
                'num_trades': 0
            }
        
        return {
            'total_commission': sum(t.commission for t in trades),
            'total_slippage': sum(t.slippage for t in trades),
            'total_fx_cost': sum(t.fx_cost for t in trades),
            'total_cost': sum(t.total_cost() for t in trades),
            'num_trades': len(trades)
        }
