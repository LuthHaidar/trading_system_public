#!/usr/bin/env python3
"""
Live Trading Script

Executes trading strategy live with IBKR integration

Usage:
    python main.py --paper-trading
    python main.py --live --strategy momentum
    python main.py --dry-run  # Preview only, no execution
"""

import argparse
import sys
import time
import yaml
from pydantic import ValidationError
import schedule
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from broker.ibkr_client import IBKRClient
from strategies import create_strategy, get_available_strategies
from data.data_manager import DataManager
from risk.position_sizer import PositionSizer, RiskConstraints
from risk.optimizer import PortfolioOptimizer
from utils.logger import setup_logger, get_logger
from utils.config_schema import validate_main_config, validate_strategies_config
from backtesting.engine import run_backtest_from_config
from utils.currency import CurrencyConverter
from risk.live_risk_manager import LiveRiskManager
from utils.operations import StateBackupManager, PositionDriftMonitor
from utils.data_platform import AuditStore, create_metrics_store

logger = get_logger(__name__)

def _load_contract_overrides_from_strategies(strategies_config_path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Load optional top-level `contracts` overrides from strategies config YAML."""
    if not strategies_config_path:
        return {}

    path = str(strategies_config_path)
    try:
        with open(path, 'r') as f:
            strategies_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Strategies config not found at %s; skipping contracts overrides", path)
        return {}
    except Exception as exc:
        logger.warning("Failed reading strategies config %s (%s); skipping contracts overrides", path, exc)
        return {}

    raw_contracts = strategies_cfg.get('contracts', {}) if isinstance(strategies_cfg, dict) else {}
    if not isinstance(raw_contracts, dict):
        logger.warning("strategies contracts section must be a mapping; got %s", type(raw_contracts).__name__)
        return {}

    parsed: Dict[str, Dict[str, str]] = {}
    for ticker, spec in raw_contracts.items():
        if not isinstance(spec, dict):
            logger.warning("contracts[%s] must be a mapping; skipping", ticker)
            continue

        symbol = spec.get('symbol')
        exchange = spec.get('exchange')
        currency = spec.get('currency')
        if not symbol or not exchange or not currency:
            logger.warning("contracts[%s] missing required symbol/exchange/currency; skipping", ticker)
            continue

        parsed[str(ticker)] = {
            'symbol': str(symbol),
            'exchange': str(exchange),
            'currency': str(currency),
        }
        primary_exchange = spec.get('primary_exchange')
        if primary_exchange:
            parsed[str(ticker)]['primary_exchange'] = str(primary_exchange)

    return parsed




class LiveTradingEngine:
    """
    Live trading engine that executes strategies with IBKR
    """
    
    def __init__(self, config: Dict, strategy, dry_run: bool = False):
        """
        Initialize live trading engine
        
        Args:
            config: System configuration
            strategy: Trading strategy instance
            dry_run: If True, preview only without execution
        """
        self.config = config
        self.strategy = strategy
        self.dry_run = dry_run
        
        # Initialize components
        self.data_manager = DataManager(
            data_dir=self.config['data']['data_dir'],
            use_adjusted_close=bool(self.config.get('data', {}).get('use_adjusted_close', True)),
            freshness_threshold_days=int(self.config.get('data', {}).get('freshness_threshold_days', 3)),
        )
        
        # IBKR client
        ibkr_config = self.config['ibkr'].copy()
        ibkr_config['base_currency'] = self.config['portfolio']['currency']
        strategy_contract_overrides = _load_contract_overrides_from_strategies(
            self.config.get('strategies_config_path')
        )
        existing_overrides = ibkr_config.get('contract_overrides', {})
        if not isinstance(existing_overrides, dict):
            logger.warning("ibkr.contract_overrides must be a mapping; ignoring invalid value")
            existing_overrides = {}
        # Strategies-level explicit contracts provide defaults; ibkr overrides keep highest precedence.
        ibkr_config['contract_overrides'] = {**strategy_contract_overrides, **existing_overrides}
        self.ibkr = IBKRClient(ibkr_config)
        
        # Risk management
        risk_config = self.config.get('risk', {})
        
        sizing_method = risk_config.get('position_sizing_method', 'equal')
        self.position_sizer = PositionSizer(method=sizing_method, config=risk_config)
        
        self.risk_constraints = RiskConstraints(risk_config)
        
        self.use_optimizer = risk_config.get('use_optimizer', False)
        if self.use_optimizer:
            self.optimizer = PortfolioOptimizer(risk_config)
        else:
            self.optimizer = None
        
        # Trading schedule
        execution_cfg = self.config.get('execution', {})

        # ``timing`` is canonical for execution semantics (e.g. open/close in
        # backtests). Live scheduling still needs an HH:MM clock time.
        raw_timing = str(execution_cfg.get('timing', '')).strip().lower()
        legacy_time = str(execution_cfg.get('time', '16:00')).strip()
        if raw_timing in {'open', 'market_open'}:
            self.execution_time = '09:30'
        elif raw_timing in {'close', 'market_close'}:
            self.execution_time = '16:00'
        elif raw_timing and len(raw_timing) == 5 and raw_timing[2] == ':':
            self.execution_time = raw_timing
        else:
            self.execution_time = legacy_time
            if raw_timing:
                logger.warning(
                    "Unsupported execution.timing value '%s' for live scheduling; "
                    "falling back to execution.time=%s",
                    raw_timing,
                    legacy_time,
                )

        self.hold_weight_epsilon = float(execution_cfg.get('hold_weight_epsilon', 1e-4))
        
        # Currency converter
        self.fx_converter = CurrencyConverter(self.config['portfolio']['currency'])
        
        # Phase 0 operational/risk guards
        self.live_risk_manager = LiveRiskManager(self.config.get('live_risk', {}))
        self.backup_manager = StateBackupManager(self.config.get('operations', {}).get('backup_dir', 'backups'))
        self.drift_monitor = PositionDriftMonitor(self.config.get('operations', {}).get('max_position_drift_pct', 0.02))
        operations_cfg = self.config.get('operations', {})
        self.audit_retention_days = int(operations_cfg.get('audit_retention_days', 90))
        self.maintenance_day = str(operations_cfg.get('maintenance_day', 'sunday')).lower()
        self.maintenance_time = str(operations_cfg.get('maintenance_time', '03:00'))
        self._live_risk_reset_date = None
        self._position_high_water = {}
        self._last_observed_equity = None

        # Phase 2 data platform (optional)
        data_platform_cfg = self.config.get('data_platform', {})
        self.audit_store = None
        self.metrics_store = None
        if data_platform_cfg.get('enabled', False):
            self.audit_store = AuditStore(data_platform_cfg.get('sqlite_path', 'state/trading_audit.db'))
            self.metrics_store = create_metrics_store(data_platform_cfg.get('metrics_store', {}))

        logger.info(f"Live Trading Engine initialized")
        logger.info(f"Strategy: {strategy.name}")
        logger.info(f"Position sizing: {sizing_method}")
        logger.info(f"Execution time: {self.execution_time}")
        if dry_run:
            logger.warning("DRY RUN MODE - No orders will be executed")
    
    def connect(self) -> bool:
        """Connect to IBKR"""
        return self.ibkr.connect()

    def validate_universe(self, tickers: list) -> list:
        """Validate tickers and return the active universe for this session."""
        validation = self.ibkr.validate_universe(tickers)
        failures = [ticker for ticker, ok in validation.items() if not ok]
        if not failures:
            logger.info("Universe validation passed for %d tickers", len(tickers))
            return list(tickers)

        strict_validation = bool(self.config.get('ibkr', {}).get('strict_universe_validation', True))
        if strict_validation:
            logger.error("Universe validation failed for tickers: %s", failures)
            return []

        active_tickers = [ticker for ticker in tickers if validation.get(ticker, False)]
        logger.warning(
            "Universe validation failed for %s, continuing due to strict_universe_validation=false; active universe=%s",
            failures,
            active_tickers,
        )
        return active_tickers
    
    def disconnect(self) -> None:
        """Disconnect from IBKR"""
        self.ibkr.disconnect()
    
    def update_data(self, tickers: list) -> None:
        """Update market data for tickers"""
        logger.info(f"Updating market data for {len(tickers)} tickers...")
        results = self.data_manager.update_all(tickers)
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Updated {success_count}/{len(tickers)} tickers successfully")

    def _validate_data_freshness(self, data: Dict[str, pd.DataFrame], tickers: list) -> None:
        """
        Fail fast when signal generation data is stale.

        This prevents live orders from being generated off out-of-date CSV history.
        """
        max_age_days = int(self.config.get('live_risk', {}).get('max_data_staleness_days', 1))
        today = pd.Timestamp(datetime.now()).normalize()
        stale = {}

        for ticker in tickers:
            df = data.get(ticker)
            if df is None or df.empty:
                stale[ticker] = "missing_data"
                continue
            latest = pd.Timestamp(df.index[-1]).tz_localize(None).normalize()
            age_days = int((today - latest).days)
            if age_days > max_age_days:
                stale[ticker] = f"latest={latest.date()} age_days={age_days}"

        if stale:
            raise RuntimeError(f"Stale market data detected: {stale}")

    def _ensure_daily_risk_reset(self) -> None:
        """Reset live daily risk baselines at start of each trading day."""
        today = datetime.now().date()
        if self._live_risk_reset_date == today:
            return

        account_summary = self.ibkr.get_account_summary()
        equity = float(account_summary.get('net_liquidation', 0.0))
        if equity <= 0:
            logger.warning("Could not establish positive equity for daily risk reset; skipping")
            return

        self.live_risk_manager.reset_daily(equity, reset_date=today)
        self._live_risk_reset_date = today
        logger.info("Live risk baseline reset for %s using equity %.2f", today, equity)
        self._persist_runtime_state(extra={'reason': 'daily_reset', 'equity': equity})

    def _runtime_snapshot_payload(self, extra: Dict = None) -> Dict:
        if hasattr(self.live_risk_manager, 'snapshot_state'):
            risk_state = self.live_risk_manager.snapshot_state()
        else:
            reset_date = getattr(self.live_risk_manager, 'last_reset_date', None)
            risk_state = {
                'reject_count': int(getattr(self.live_risk_manager, 'reject_count', 0)),
                'kill_switch': bool(getattr(self.live_risk_manager, 'kill_switch', False)),
                'kill_switch_reason': getattr(self.live_risk_manager, 'kill_switch_reason', None),
                'last_reset_date': str(reset_date) if reset_date is not None else None,
            }
        payload = {
            'timestamp': datetime.now(),
            'risk_state': risk_state,
            'position_high_water': dict(getattr(self, '_position_high_water', {})),
        }
        if extra:
            payload['context'] = extra
        return payload

    def _persist_runtime_state(self, extra: Dict = None) -> None:
        """Persist current runtime risk/ops state to audit snapshot storage."""
        payload = self._runtime_snapshot_payload(extra=extra)
        persisted = False

        store = getattr(self, 'audit_store', None)
        if store and hasattr(store, 'save_state_snapshot'):
            try:
                store.save_state_snapshot(
                    state=payload,
                    source='live',
                    strategy=self.strategy.name,
                )
                persisted = True
            except Exception as exc:
                logger.warning("Audit-store runtime persistence failed: %s", exc)

        backup_manager = getattr(self, 'backup_manager', None)
        if backup_manager is not None:
            try:
                if hasattr(backup_manager, 'backup_runtime_state'):
                    backup_manager.backup_runtime_state(payload)
                    persisted = True
                elif hasattr(backup_manager, 'backup'):
                    backup_manager.backup(payload)
                    persisted = True
            except Exception as exc:
                logger.warning("Backup-manager runtime persistence failed: %s", exc)

        if not persisted:
            logger.warning("Runtime state persistence unavailable: no writable audit/backup sink")

    def _evaluate_position_level_stops(self,
                                       live_positions: Dict[str, Dict],
                                       current_prices: Dict[str, float]) -> Dict[str, Dict]:
        """
        Detect per-position hard and trailing stop breaches.

        Returns:
            Dict of {ticker: {'reason': str, ...}} for positions to force-exit.
        """
        config = self.config.get('live_risk', {})
        hard_stop = config.get('position_stop_loss_pct')
        trailing_stop = config.get('position_trailing_stop_pct')
        if hard_stop is None and trailing_stop is None:
            return {}

        hard_stop = abs(float(hard_stop)) if hard_stop is not None else None
        trailing_stop = abs(float(trailing_stop)) if trailing_stop is not None else None

        forced_exits = {}
        open_tickers = set()

        for ticker, payload in (live_positions or {}).items():
            shares = float(payload.get('shares', 0.0))
            if shares <= 0:
                continue
            open_tickers.add(ticker)

            price = current_prices.get(ticker)
            if price is None or price <= 0:
                continue

            peak = max(float(self._position_high_water.get(ticker, price)), float(price))
            self._position_high_water[ticker] = peak

            avg_cost = float(payload.get('avg_cost', 0.0) or 0.0)
            if hard_stop is not None and avg_cost > 0:
                pnl_pct = (float(price) / avg_cost) - 1.0
                if pnl_pct <= -hard_stop:
                    forced_exits[ticker] = {
                        'reason': 'position_stop_loss',
                        'pnl_pct': pnl_pct,
                        'threshold': -hard_stop,
                        'avg_cost': avg_cost,
                        'price': float(price),
                    }
                    continue

            if trailing_stop is not None and peak > 0:
                drawdown = 1.0 - (float(price) / peak)
                if drawdown >= trailing_stop:
                    forced_exits[ticker] = {
                        'reason': 'position_trailing_stop',
                        'drawdown_from_peak': drawdown,
                        'threshold': trailing_stop,
                        'peak_price': peak,
                        'price': float(price),
                    }

        # Drop stale high-water marks for closed positions.
        for ticker in list(self._position_high_water.keys()):
            if ticker not in open_tickers:
                self._position_high_water.pop(ticker, None)

        return forced_exits
    
    def execute_strategy(self, tickers: list) -> None:
        """
        Execute trading strategy
        
        Args:
            tickers: List of tickers in universe
        """
        logger.info("="*70)
        logger.info(f"EXECUTING STRATEGY: {self.strategy.name}")
        logger.info(f"Time: {datetime.now()}")
        logger.info("="*70)
        
        try:
            if not hasattr(self, '_position_high_water'):
                self._position_high_water = {}

            # Update data
            self.update_data(tickers)
            
            # Load historical data
            logger.info("Loading historical data...")
            data = self.data_manager.load_multiple(tickers)
            
            if not data:
                logger.error("No data available")
                return

            self._validate_data_freshness(data, tickers)
            
            # Get current positions from IBKR
            ibkr_positions = self.ibkr.get_positions()
            current_shares = {ticker: int(pos['shares']) 
                            for ticker, pos in ibkr_positions.items()}
            
            logger.info(f"Current positions: {current_shares}")
            if self.audit_store:
                self.audit_store.log_event(
                    event_type='positions_snapshot',
                    source='live',
                    strategy=self.strategy.name,
                    details={'positions': current_shares},
                )
            
            # Get current prices
            current_prices = self.ibkr.get_market_prices(tickers)
            if not current_prices:
                logger.error("Could not get market prices")
                return
            
            logger.info(f"Current prices: {current_prices}")
            self.live_risk_manager.mark_data_heartbeat(datetime.now())

            forced_exits = self._evaluate_position_level_stops(
                live_positions=ibkr_positions,
                current_prices=current_prices,
            )
            if forced_exits:
                logger.warning("Position-level stops triggered: %s", forced_exits)
                if self.audit_store:
                    self.audit_store.log_event(
                        event_type='position_level_stop_triggered',
                        source='live',
                        strategy=self.strategy.name,
                        details={'forced_exits': forced_exits},
                    )
            
            # Calculate current equity
            account_summary = self.ibkr.get_account_summary()
            equity = account_summary.get('net_liquidation', 0)
            
            logger.info(f"Account equity: {equity:,.2f} {account_summary.get('currency', 'USD')}")

            risk_event = self.live_risk_manager.evaluate(equity, datetime.now())
            if risk_event.triggered:
                logger.error(f"Circuit breaker triggered: {risk_event.reason} metadata={risk_event.metadata}")
                if self.audit_store:
                    self.audit_store.log_event(
                        event_type='circuit_breaker_triggered',
                        source='live',
                        strategy=self.strategy.name,
                        details={'reason': risk_event.reason, 'metadata': risk_event.metadata},
                    )
                self._persist_runtime_state(extra={'reason': risk_event.reason, 'triggered': True})
                return

            previous_equity = getattr(self, '_last_observed_equity', None)
            if previous_equity is not None and float(previous_equity) > 0:
                realized_return = (float(equity) / float(previous_equity)) - 1.0
                if hasattr(self.strategy, 'on_realized_portfolio_return'):
                    try:
                        self.strategy.on_realized_portfolio_return(
                            pd.Timestamp(datetime.now()),
                            float(realized_return),
                            context={
                                'engine': 'live',
                                'equity': float(equity),
                                'previous_equity': float(previous_equity),
                            },
                        )
                    except Exception as exc:
                        logger.warning("Strategy realized-return hook failed: %s", exc)
            self._last_observed_equity = float(equity)
            
            # Calculate current weights
            currencies = {ticker: self.fx_converter.get_ticker_currency(ticker) 
                         for ticker in tickers}
            
            position_values = {}
            for ticker, shares in current_shares.items():
                if ticker in current_prices:
                    value = shares * current_prices[ticker]
                    currency = currencies.get(ticker, 'USD')
                    if currency != self.config['portfolio']['currency']:
                        value = self.fx_converter.convert_to_base(value, currency)
                    position_values[ticker] = value
            
            current_weights = {ticker: value / equity 
                             for ticker, value in position_values.items()}
            
            logger.info(f"Current weights: {current_weights}")
            
            # Generate signals from strategy
            logger.info("Generating strategy signals...")
            date = datetime.now()
            
            target_weights = self.strategy.generate_signals(
                pd.Timestamp(date),
                data,
                current_weights
            )
            
            logger.info(f"Strategy signals: {target_weights}")
            
            if not target_weights:
                if not forced_exits:
                    logger.info("No signals generated, maintaining current positions")
                    return
                logger.info("No strategy signals, but forced exits are active")
                target_weights = {}
            else:
                hold_weight_epsilon = float(getattr(self, 'hold_weight_epsilon', 1e-4))
                max_weight_delta = self._calculate_max_weight_deviation(target_weights, current_weights)
                if max_weight_delta <= hold_weight_epsilon and not forced_exits:
                    logger.debug(
                        "Skipping live rebalance: max target/current deviation %.8f <= epsilon %.8f",
                        max_weight_delta,
                        hold_weight_epsilon,
                    )
                    return

                # Apply position sizing
                logger.info("Applying position sizing...")
                target_weights = self.position_sizer.size_positions(
                    target_weights,
                    data,
                    equity
                )
                
                logger.info(f"Sized weights: {target_weights}")
                
                # Portfolio optimization (optional): run before final constraints.
                if self.use_optimizer and self.optimizer and target_weights:
                    logger.info("Optimizing portfolio...")
                    try:
                        optimizer_method = self.config['risk'].get('optimizer_method', 'max_sharpe')
                        target_weights = self.optimizer.optimize(
                            target_weights,
                            data,
                            method=optimizer_method
                        )
                        logger.info(f"Optimized weights: {target_weights}")
                    except Exception as e:
                        logger.error(f"Optimization failed: {e}")
                        halt_on_failure = bool(
                            self.config.get('live_risk', {}).get('halt_on_optimization_failure', True)
                        )
                        if halt_on_failure:
                            self.live_risk_manager.kill_switch = True
                            self.live_risk_manager.kill_switch_reason = 'optimization_failure'
                            self.live_risk_manager.record_order_reject()
                            self._persist_runtime_state(
                                extra={'reason': 'optimization_failure', 'error': str(e)}
                            )
                            if self.audit_store:
                                self.audit_store.log_event(
                                    event_type='optimization_failed',
                                    source='live',
                                    strategy=self.strategy.name,
                                    details={'error': str(e), 'halted': True},
                                )
                            logger.error("Halting live execution due to optimization failure")
                            return

                logger.info("Applying risk constraints...")
                target_weights = self.risk_constraints.apply_constraints(
                    target_weights,
                    data
                )
                if target_weights and self.position_sizer.method == 'target_vol':
                    target_weights = self.position_sizer.apply_final_volatility_scaling(
                        target_weights,
                        data,
                        max_total_exposure=max(0.0, 1.0 - float(self.risk_constraints.min_cash_reserve)),
                    )
                logger.info(f"Constrained weights: {target_weights}")
            
            # Calculate target positions in shares
            logger.info("Calculating target positions...")
            target_positions = {}

            if target_weights:
                for ticker, weight in target_weights.items():
                    if ticker not in current_prices:
                        continue
                    
                    # Calculate target value in base currency
                    target_value_base = equity * weight
                    
                    # Convert to ticker's currency
                    currency = currencies.get(ticker, 'USD')
                    if currency != self.config['portfolio']['currency']:
                        fx_rate = self.fx_converter.get_fx_rate(
                            self.config['portfolio']['currency'], currency
                        )
                        target_value_foreign = target_value_base * fx_rate
                    else:
                        target_value_foreign = target_value_base
                    
                    # Calculate shares
                    price = current_prices[ticker]
                    target_shares = int(target_value_foreign / price)
                    
                    if target_shares > 0:
                        target_positions[ticker] = target_shares
            else:
                # Forced-exit-only flow: hold current positions except exited names.
                target_positions = {ticker: int(shares) for ticker, shares in current_shares.items() if int(shares) > 0}

            for ticker in forced_exits.keys():
                if ticker in target_positions:
                    logger.warning("Forcing exit for %s due to %s", ticker, forced_exits[ticker].get('reason'))
                    target_positions.pop(ticker, None)
            
            logger.info(f"Target positions: {target_positions}")
            
            # Calculate orders needed
            orders = self.ibkr.reconcile_positions(target_positions)
            
            if not orders:
                logger.info("No rebalancing needed")
                return
            
            logger.info(f"Orders to execute: {len(orders)}")
            for ticker, quantity, action in orders:
                logger.info(f"  {action} {quantity} {ticker}")

            if self.audit_store:
                self.audit_store.log_event(
                    event_type='orders_generated',
                    source='live',
                    strategy=self.strategy.name,
                    details={'orders': orders, 'target_positions': target_positions},
                )

            drift = self.drift_monitor.check(target_positions, ibkr_positions)
            if drift:
                logger.warning(f"Position drift detected before execution: {drift}")
            # Execute orders (unless dry run)
            if self.dry_run:
                logger.warning("DRY RUN - Orders not executed")
            else:
                logger.info("Executing orders...")
                execution_success = self.ibkr.execute_rebalance(target_positions)
                positions_verified = False
                if execution_success:
                    max_retries = int(
                        self.config.get('live_risk', {}).get('partial_fill_retries', 1)
                    )
                    positions_verified = self._verify_post_execution_positions(
                        target_positions=target_positions,
                        max_retries=max_retries
                    )

                if self.audit_store:
                    self.audit_store.log_event(
                        event_type='orders_executed' if execution_success and positions_verified else 'orders_failed',
                        source='live',
                        strategy=self.strategy.name,
                        details={
                            'target_positions': target_positions,
                            'orders': orders,
                            'execution_success': execution_success,
                            'positions_verified': positions_verified,
                        },
                    )

                if execution_success and positions_verified:
                    logger.info("Strategy execution complete")
                else:
                    if not execution_success:
                        self.live_risk_manager.record_order_reject()
                    logger.error("Strategy execution failed or left residual position drift")

            backup_payload = {
                'timestamp': datetime.now(),
                'target_positions': target_positions,
                'current_positions': current_shares,
                'orders': orders,
                'risk_state': self._runtime_snapshot_payload().get('risk_state', {}),
                'position_high_water': dict(self._position_high_water),
            }
            backup_path = self.backup_manager.backup(backup_payload)
            logger.info(f"State backup created: {backup_path}")

            if self.audit_store:
                self.audit_store.save_state_snapshot(
                    state=backup_payload,
                    source='live',
                    strategy=self.strategy.name,
                )

            if self.metrics_store:
                self.metrics_store.write_metric(
                    'live.equity',
                    float(equity),
                    datetime.now(timezone.utc),
                    tags={'strategy': self.strategy.name},
                )

            self._persist_runtime_state(extra={'execution_complete': True})
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _extract_position_shares(live_positions: Dict[str, Dict]) -> Dict[str, int]:
        """Normalize broker position payload into whole-share map."""
        shares = {}
        for ticker, payload in (live_positions or {}).items():
            raw_shares = payload.get('shares', 0)
            shares[ticker] = int(round(float(raw_shares)))
        return shares

    @staticmethod
    def _calculate_max_weight_deviation(target_weights: Dict[str, float],
                                        current_weights: Dict[str, float]) -> float:
        """Maximum absolute per-ticker deviation across target/current weight vectors."""
        tickers = set((target_weights or {}).keys()) | set((current_weights or {}).keys())
        if not tickers:
            return 0.0
        return max(
            abs(float((target_weights or {}).get(ticker, 0.0)) - float((current_weights or {}).get(ticker, 0.0)))
            for ticker in tickers
        )

    @staticmethod
    def _compute_share_drift(target_positions: Dict[str, int],
                             live_shares: Dict[str, int]) -> Dict[str, int]:
        """Return per-ticker target-minus-live share drift."""
        drift = {}
        tickers = set(target_positions.keys()) | set(live_shares.keys())
        for ticker in tickers:
            target = int(target_positions.get(ticker, 0))
            live = int(live_shares.get(ticker, 0))
            delta = target - live
            if delta != 0:
                drift[ticker] = delta
        return drift

    def _verify_post_execution_positions(self, target_positions: Dict[str, int],
                                         max_retries: int = 1) -> bool:
        """
        Verify post-trade positions and retry once if residual drift remains.

        Residual drift commonly indicates partial fills or unfilled orders.
        """
        retries_used = 0
        retry_budget = max(0, int(max_retries))

        while True:
            live_positions = self.ibkr.get_positions()
            live_shares = self._extract_position_shares(live_positions)
            drift = self._compute_share_drift(target_positions, live_shares)
            if not drift:
                return True

            logger.error(
                "Post-execution position drift detected (possible partial fills): %s",
                drift,
            )
            self.live_risk_manager.record_order_reject()
            self._persist_runtime_state(extra={'reason': 'post_execution_drift', 'drift': drift})
            if self.audit_store:
                self.audit_store.log_event(
                    event_type='post_execution_drift',
                    source='live',
                    strategy=self.strategy.name,
                    details={
                        'drift_shares': drift,
                        'target_positions': target_positions,
                        'live_positions': live_shares,
                        'retries_used': retries_used,
                    },
                )

            if retries_used >= retry_budget:
                return False

            retries_used += 1
            logger.warning(
                "Attempting immediate reconciliation retry (%d/%d)",
                retries_used,
                retry_budget,
            )
            retry_success = self.ibkr.execute_rebalance(target_positions)
            if not retry_success:
                logger.error("Reconciliation retry failed before drift was resolved")
                return False


    def recover_from_persisted_state(self) -> None:
        """Recover and inspect latest persisted state snapshot before execution."""
        snapshot = None
        if self.audit_store:
            snapshot = self.audit_store.load_latest_state(source='live', strategy=self.strategy.name)
        backup_manager = getattr(self, 'backup_manager', None)
        if snapshot is None and backup_manager is not None and hasattr(backup_manager, 'load_latest_runtime_state'):
            try:
                snapshot = backup_manager.load_latest_runtime_state()
            except Exception as exc:
                logger.warning("Could not load latest runtime backup snapshot: %s", exc)

        if not snapshot:
            logger.info('No persisted live state snapshot available for recovery')
            return

        logger.info(f"Recovered persisted snapshot timestamp={snapshot.get('timestamp')}")
        risk_state = snapshot.get('risk_state', {}) or {}
        if risk_state:
            try:
                reset_date_raw = risk_state.get('last_reset_date')
                reset_date = pd.Timestamp(reset_date_raw).date() if reset_date_raw else None
            except Exception:
                reset_date = None

            today = datetime.now().date()
            if reset_date == today:
                if hasattr(self.live_risk_manager, 'restore_state'):
                    self.live_risk_manager.restore_state(risk_state)
                else:
                    # Backward-compatible fallback for lightweight test doubles.
                    self.live_risk_manager.reject_count = int(risk_state.get('reject_count', 0))
                    self.live_risk_manager.kill_switch = bool(risk_state.get('kill_switch', False))
                    self.live_risk_manager.last_reset_date = reset_date
                    if hasattr(self.live_risk_manager, 'kill_switch_reason'):
                        self.live_risk_manager.kill_switch_reason = risk_state.get('kill_switch_reason')
                logger.info(
                    "Restored live risk state from snapshot: reject_count=%s kill_switch=%s reason=%s",
                    self.live_risk_manager.reject_count,
                    self.live_risk_manager.kill_switch,
                    getattr(self.live_risk_manager, 'kill_switch_reason', None),
                )

        restored_high_water = snapshot.get('position_high_water', {}) or {}
        if isinstance(restored_high_water, dict):
            self._position_high_water = {
                str(ticker): float(price)
                for ticker, price in restored_high_water.items()
                if price is not None
            }

        previous_targets = snapshot.get('target_positions', {})
        if previous_targets and self.audit_store:
            current_positions = self.ibkr.get_positions()
            drift = self.drift_monitor.check(previous_targets, current_positions)
            if drift:
                logger.warning(f"Recovered-state drift detected: {drift}")
                self.audit_store.log_event(
                    event_type='recovery_drift_detected',
                    source='live',
                    strategy=self.strategy.name,
                    details={'drift': drift, 'previous_targets': previous_targets},
                )

    def run_once(self, tickers: list) -> None:
        """Run strategy execution once"""
        if not self.ibkr.is_connected():
            if not self.connect():
                logger.error("Failed to connect to IBKR")
                return

        self._ensure_daily_risk_reset()
        self.recover_from_persisted_state()
        self.execute_strategy(tickers)

    def _run_scheduled_maintenance(self) -> None:
        """Run periodic maintenance tasks configured for the live scheduler."""
        if not self.audit_store or not hasattr(self.audit_store, 'prune_audit_events'):
            logger.debug("Maintenance skipped: audit pruning not available")
            return
        try:
            deleted = int(self.audit_store.prune_audit_events(self.audit_retention_days))
            logger.info(
                "Maintenance: pruned %d audit events older than %d days",
                deleted,
                self.audit_retention_days,
            )
            self.audit_store.log_event(
                event_type='maintenance_prune_completed',
                source='live',
                strategy=self.strategy.name,
                details={
                    'deleted_rows': deleted,
                    'older_than_days': self.audit_retention_days,
                },
            )
        except Exception as exc:
            logger.warning("Maintenance prune failed: %s", exc)
    
    def run_scheduled(self, tickers: list) -> None:
        """Run on schedule (daily at specified time)"""
        logger.info(f"Scheduling daily execution at {self.execution_time}")
        
        # Schedule daily execution
        schedule.every().day.at(self.execution_time).do(
            self.run_once, tickers=tickers
        )
        weekly_builder = schedule.every()
        if hasattr(weekly_builder, 'week'):
            weekly_builder = weekly_builder.week
        elif hasattr(weekly_builder, 'weeks'):
            weekly_builder = weekly_builder.weeks
        elif hasattr(weekly_builder, 'day'):
            weekly_builder = weekly_builder.day

        day_builder = getattr(weekly_builder, self.maintenance_day, None) or weekly_builder
        if hasattr(day_builder, 'at') and hasattr(day_builder, 'do'):
            day_builder.at(self.maintenance_time).do(self._run_scheduled_maintenance)
            logger.info(
                "Scheduling weekly maintenance on %s at %s (audit retention: %d days)",
                self.maintenance_day,
                self.maintenance_time,
                self.audit_retention_days,
            )
        
        logger.info("Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                try:
                    schedule.run_pending()
                except Exception as e:
                    logger.error(f"Scheduler health check failure: {e}; attempting automatic restart")
                    time.sleep(5)
                time.sleep(60)  # Check every minute

                # Reconnect if needed
                if not self.ibkr.is_connected():
                    logger.warning("Connection lost, attempting to reconnect...")
                    self.ibkr.reconnect()

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self.disconnect()




def _resolve_strategy_config(strategy_name: str, strategies_config: Dict, strategies_config_path: str = 'config/strategies.yaml') -> Dict:
    """Validate strategy registry + YAML presence and return selected strategy config."""
    try:
        validate_strategies_config(strategies_config)
    except (ValidationError, TypeError) as e:
        raise ValueError(f"Strategy config validation failed: {e}") from e

    available_strategies = get_available_strategies()
    if strategy_name not in available_strategies:
        available = ', '.join(available_strategies)
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {available}")

    if strategy_name not in strategies_config:
        raise ValueError(
            f"Strategy {strategy_name} is supported but missing in {strategies_config_path}"
        )

    strategy_config = dict(strategies_config[strategy_name] or {})
    if strategy_name == 'strategy_orchestration':
        strategy_config['_all_strategies_config'] = strategies_config
    return strategy_config

def load_strategy(strategy_name: str, config: Dict):
    """Load strategy from configuration"""
    # Load strategy config
    strategies_config_path = str((config or {}).get('strategies_config_path', 'config/strategies.yaml'))
    with open(strategies_config_path, 'r') as f:
        strategies_config = yaml.safe_load(f) or {}

    strategy_config = _resolve_strategy_config(
        strategy_name,
        strategies_config,
        strategies_config_path=strategies_config_path,
    )
    return create_strategy(strategy_name, strategy_config)


def run_pre_flight_backtest(config_path: str,
                            strategy_name: str,
                            tickers: list,
                            days: int = 90,
                            config_override: Dict | None = None) -> Dict:
    """Run a short pre-flight backtest and return summary metrics."""
    end_ts = pd.Timestamp(datetime.now()).normalize()
    start_ts = end_ts - pd.Timedelta(days=max(1, int(days)))

    results = run_backtest_from_config(
        config_path=config_path,
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=str(start_ts.date()),
        end_date=str(end_ts.date()),
        config_override=config_override,
    )
    metrics = results.get('metrics', {}) or {}
    trades = results.get('trades', []) or []

    metric_trade_count = metrics.get('num_trades')
    try:
        metric_trade_count = int(metric_trade_count) if metric_trade_count is not None else 0
    except (TypeError, ValueError):
        metric_trade_count = 0

    trades_trade_count = len(trades)
    # Prefer explicit trade objects when present, otherwise fall back to metrics.
    num_trades = trades_trade_count if trades_trade_count > 0 else metric_trade_count

    return {
        'num_trades': int(num_trades),
        'sharpe_ratio': float(metrics.get('sharpe_ratio', 0.0) or 0.0),
        'start_date': str(start_ts.date()),
        'end_date': str(end_ts.date()),
    }


def evaluate_pre_flight_summary(preflight: Dict) -> tuple[bool, str]:
    """Evaluate pre-flight summary against live-start safety checks."""
    raw_num_trades = preflight.get('num_trades', 0)
    try:
        num_trades = int(raw_num_trades or 0)
    except (TypeError, ValueError):
        return False, f"invalid num_trades value: {raw_num_trades!r}"

    sharpe = preflight.get('sharpe_ratio', 0.0)
    try:
        sharpe = float(sharpe)
    except (TypeError, ValueError):
        return False, f"invalid Sharpe value: {sharpe!r}"

    if num_trades <= 0:
        return False, "no trades generated"
    if pd.isna(sharpe):
        return False, "Sharpe is NaN"
    if sharpe < 0:
        return False, f"negative Sharpe {sharpe:.4f}"

    return True, "passed"


def main():
    parser = argparse.ArgumentParser(
        description='Live trading with IBKR',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--strategy', type=str, default='momentum',
                       help='Strategy name')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--strategies-config', type=str, default='config/strategies.yaml',
                       help='Strategy config file path')
    parser.add_argument('--tickers', nargs='+', default=None,
                       help='List of tickers to trade')
    parser.add_argument('--paper-trading', action='store_true',
                       help='Use paper trading account (port 7497)')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading account (port 7496)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview only, do not execute orders')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no scheduling)')
    parser.add_argument('--execution-time', type=str, default=None,
                       help='Daily execution time (HH:MM)')
    parser.add_argument('--pre-flight-backtest', action='store_true',
                       help='Run a 90-day pre-flight backtest before connecting to IBKR')
    parser.add_argument('--pre-flight-days', type=int, default=90,
                       help='Lookback window (days) for pre-flight backtest')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Update IBKR port based on arguments
    if args.paper_trading:
        config['ibkr']['port'] = 7497
    elif args.live:
        config['ibkr']['port'] = 7496
    
    # Update execution time
    if args.execution_time:
        config['execution']['timing'] = args.execution_time

    try:
        validate_main_config(config)
    except (ValidationError, ValueError) as e:
        print("Config validation failed:")
        print(e)
        return 1

    # Runtime-only path used by live engine for optional strategies contracts.
    config['strategies_config_path'] = args.strategies_config
    
    # Get tickers
    tickers = args.tickers or config['data'].get('tickers', ['SPY', 'QQQ'])
    
    # Setup logging
    log_file = 'logs/live_trading.log'
    setup_logger('live_trading', log_file)
    
    # Load strategy
    strategy_load_config = dict(config)
    strategy_load_config['strategies_config_path'] = args.strategies_config
    strategy = load_strategy(args.strategy, strategy_load_config)
    
    # Confirm live trading
    if args.live and not args.dry_run:
        logger.warning("="*70)
        logger.warning("LIVE TRADING MODE")
        logger.warning("Real money will be used!")
        logger.warning("="*70)
        response = input("Type 'CONFIRM' to proceed: ")
        if response != 'CONFIRM':
            print("Aborted")
            return 1
    
    config_override = None
    if args.paper_trading:
        config_override = {'ibkr': {'port': 7497}}
    elif args.live:
        config_override = {'ibkr': {'port': 7496}}

    if args.pre_flight_backtest:
        logger.info("Running pre-flight backtest for %d days before IBKR connection", int(args.pre_flight_days))
        preflight = run_pre_flight_backtest(
            config_path=args.config,
            strategy_name=args.strategy,
            tickers=tickers,
            days=args.pre_flight_days,
            config_override=config_override,
        )
        logger.info(
            "Pre-flight backtest %s -> %s | trades=%d sharpe=%.4f",
            preflight['start_date'],
            preflight['end_date'],
            preflight['num_trades'],
            preflight['sharpe_ratio'],
        )
        passed, reason = evaluate_pre_flight_summary(preflight)
        if not passed:
            logger.error("Pre-flight backtest failed: %s", reason)
            return 1

    # Create trading engine
    engine = LiveTradingEngine(
        config=config,
        strategy=strategy,
        dry_run=args.dry_run
    )
    
    try:
        # Connect to IBKR
        if not engine.connect():
            logger.error("Failed to connect to IBKR")
            return 1

        active_tickers = engine.validate_universe(tickers)
        if not active_tickers:
            raise SystemExit(1)

        # Run once or scheduled
        if args.once:
            engine.run_once(active_tickers)
        else:
            engine.run_scheduled(active_tickers)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        engine.disconnect()


if __name__ == '__main__':
    sys.exit(main())
