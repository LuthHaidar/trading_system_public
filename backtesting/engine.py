import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from numbers import Real
from datetime import datetime
from datetime import timedelta
from uuid import uuid4
from dataclasses import dataclass
import yaml
from pydantic import ValidationError

from strategies.base_strategy import BaseStrategy
from strategies import create_strategy, get_available_strategies
from data.data_manager import DataManager
from backtesting.portfolio import DataIntegrityError, Portfolio
from backtesting.execution import ExecutionEngine
from backtesting.metrics import PerformanceMetrics
from backtesting.validation import ValidationSuite
from backtesting.tca import TransactionCostAnalysis
from risk.advanced_risk import AdvancedRiskAnalytics

try:
    from risk.hmm_regime import HMMRegimeDetector
except ImportError:
    HMMRegimeDetector = None
from risk.position_sizer import PositionSizer, RiskConstraints
from risk.optimizer import PortfolioOptimizer
from utils.logger import get_logger
from utils.currency import CurrencyConverter
from utils.data_platform import AuditStore, create_metrics_store
from utils.config_schema import validate_main_config, validate_strategies_config

logger = get_logger(__name__)


@dataclass
class AssetExclusionRecord:
    ticker: str
    date: pd.Timestamp
    reason: str
    available_bars: int
    required_bars: int
    detail: str


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the simulation
    """
    
    def __init__(self, strategy: BaseStrategy, data_manager: DataManager, config: Dict):
        """
        Initialize backtest engine.

        Contract:
            `config` must be a fully materialized and validated dictionary.
            Callers are responsible for all file loading, YAML parsing, and
            override merging before constructing `BacktestEngine`.
            This constructor performs no configuration I/O or merge logic.
        
        Args:
            strategy: Trading strategy instance
            data_manager: Data manager for loading market data
            config: System configuration dict
        """
        self.strategy = strategy
        self.data_manager = data_manager
        self.config = config

        execution_cfg = self.config.get('execution', {})
        self.rebalance_timeframe = execution_cfg.get('rebalance_timeframe', 'daily').lower()
        self.timing = str(execution_cfg.get('timing', 'close')).upper()
        self.progress_log_interval = int(execution_cfg.get('progress_log_interval', 50))
        self.hold_weight_epsilon = float(execution_cfg.get('hold_weight_epsilon', 1e-4))
        self.min_rebalance_weight_delta = float(execution_cfg.get('min_rebalance_weight_delta', 0.02))
        self.min_trade_value = float(execution_cfg.get('min_trade_value', 200.0))
        self.invested_sleeve_drift_warn = float(execution_cfg.get('invested_sleeve_drift_warn', 0.05))
        self.cash_drift_warn = float(execution_cfg.get('cash_drift_warn', 0.01))
        self.drift_warn_min_capital = float(execution_cfg.get('drift_warn_min_capital', 0.0))
        self._consecutive_drift_warnings = 0
        self.uninvested_cash_tolerance = float(execution_cfg.get('uninvested_cash_tolerance', 0.005))
        self.min_forward_data_days = int(execution_cfg.get('min_forward_data_days', 0))
        self.cost_aware_fee_bps = float(execution_cfg.get('cost_aware_fee_bps', 0.0))
        self.cost_aware_spread_bps = float(execution_cfg.get('cost_aware_spread_bps', 0.0))
        self.missing_volume_fallback = float(execution_cfg.get('missing_volume_fallback', 1_000_000.0))
        
        # Initialize components
        data_cfg = self.config.get('data', {})
        max_stale_days = int(data_cfg.get('max_stale_price_days', 5))
        if self.min_forward_data_days > 0 and self.min_forward_data_days < max_stale_days:
            raise ValueError(
                f"Invalid configuration: min_forward_data_days ({self.min_forward_data_days}) must be >= max_stale_price_days ({max_stale_days}) when enabled (>0)"
            )

        self.portfolio = Portfolio(
            initial_capital=self.config['portfolio']['initial_capital'],
            base_currency=self.config['portfolio']['currency'],
            max_stale_price_days=max_stale_days,
        )
        
        self.execution_engine = ExecutionEngine(
            self.config['costs'],
            base_currency=self.config['portfolio']['currency']
        )

        fx_data_dir = os.path.join(str(data_cfg.get('data_dir', './data')), 'fx')
        self.fx_converter = CurrencyConverter(
            self.config['portfolio']['currency'],
            fx_cache_staleness_days=int(data_cfg.get('fx_cache_staleness_days', 1)),
            fx_data_dir=fx_data_dir,
        )

        # Phase 2 data platform (optional)
        data_platform_cfg = self.config.get('data_platform', {})
        self.audit_store = None
        self.metrics_store = None
        if data_platform_cfg.get('enabled', False):
            self.audit_store = AuditStore(data_platform_cfg.get('sqlite_path', 'state/trading_audit.db'))
            self.metrics_store = create_metrics_store(data_platform_cfg.get('metrics_store', {}))
            logger.info("Data platform persistence enabled")
        
        # Risk management
        risk_config = self.config.get('risk', {})
        
        # Position sizer
        sizing_method = risk_config.get('position_sizing_method', 'equal')
        self.position_sizer = PositionSizer(
            method=sizing_method,
            config=risk_config
        )
        
        # Risk constraints
        self.risk_constraints = RiskConstraints(risk_config)
        
        # Portfolio optimizer (optional)
        self.use_optimizer = risk_config.get('use_optimizer', False)
        if self.use_optimizer:
            self.optimizer = PortfolioOptimizer(risk_config)
            optimizer_method = risk_config.get('optimizer_method', 'max_sharpe')
            logger.info("Portfolio optimization enabled: %s", optimizer_method)
        else:
            self.optimizer = None

        logger.info("BacktestEngine initialized with strategy: %s", strategy.name)
        logger.info("Position sizing: %s", sizing_method)
        logger.info(
            "Risk constraints: max_pos=%.1f%%, min_cash=%.1f%%",
            self.risk_constraints.max_position_size * 100.0,
            self.risk_constraints.min_cash_reserve * 100.0,
        )
        self.current_run_id: Optional[str] = None
        self._last_rebalance_date: Optional[pd.Timestamp] = None
        self.positions_history: List[Dict[str, Any]] = []
        self.asset_exclusions: List[AssetExclusionRecord] = []
        self.warmup_history_shortfall_days: int = 0
    
    def run(self, tickers: List[str], start_date: str, end_date: str,
            initial_positions: Dict[str, float] = None) -> Dict:
        """
        Run backtest simulation
        
        Args:
            tickers: List of ticker symbols to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_positions: Initial positions dict (optional)
            
        Returns:
            Dict with results including portfolio, metrics, trades
        """
        logger.info("Starting backtest: %s to %s", start_date, end_date)
        if not tickers:
            raise ValueError("tickers list is empty")
        logger.info("Tickers: %s", ', '.join(tickers))
        self.current_run_id = f"bt_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
        self.positions_history = []
        self.asset_exclusions = []
        self.warmup_history_shortfall_days = 0
        self._last_rebalance_date = None
        if self.audit_store:
            self.audit_store.log_event(
                event_type='backtest_started',
                source='backtest',
                strategy=self.strategy.name,
                details={'start_date': start_date, 'end_date': end_date, 'tickers': tickers},
                run_id=self.current_run_id,
            )
        
        required_history_days = int(self.strategy.get_required_history())
        warmup_load_start = (
            pd.Timestamp(start_date) - pd.offsets.BDay(max(required_history_days, 0) + 5)
        ).strftime('%Y-%m-%d')

        # Load data for all tickers (including warmup lookback window)
        logger.info("Loading market data...")
        data = {}
        for ticker in tickers:
            try:
                df = self.data_manager.load_ticker(
                    ticker,
                    start_date=warmup_load_start,
                    end_date=end_date,
                )
                data[ticker] = df
            except FileNotFoundError as e:
                self._record_asset_exclusion(
                    ticker=ticker,
                    date=pd.Timestamp(start_date),
                    reason='missing_file',
                    available_bars=0,
                    required_bars=required_history_days,
                    detail=str(e),
                )
                logger.warning("Missing data file for %s: %s", ticker, e)
            except (pd.errors.EmptyDataError, OSError) as e:
                self._record_asset_exclusion(
                    ticker=ticker,
                    date=pd.Timestamp(start_date),
                    reason='data_error',
                    available_bars=0,
                    required_bars=required_history_days,
                    detail=str(e),
                )
                logger.warning("Failed to load %s due to recoverable data error: %s", ticker, e)
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading {ticker}") from e
        
        if not data:
            raise ValueError("No data loaded for any ticker")
        
        logger.info("Loaded data for %s tickers", len(data))
        
        # Get unified date range (union across all tickers)
        trading_dates_all = self._get_trading_dates(data, warmup_load_start, end_date)
        effective_start_date = self._compute_effective_start_date(
            trading_dates_all=trading_dates_all,
            requested_start_date=start_date,
            required_history_days=required_history_days,
        )
        trading_dates = trading_dates_all[trading_dates_all >= effective_start_date]
        logger.info(
            "Backtest warmup alignment: requested_start=%s required_history_days=%d effective_start_date=%s shortfall_days=%d",
            pd.Timestamp(start_date).date(),
            required_history_days,
            effective_start_date.date(),
            self.warmup_history_shortfall_days,
        )
        logger.info("Backtest period: %s trading days", len(trading_dates))

        if len(trading_dates) == 0:
            raise ValueError(
                f"No overlapping trading dates found for {start_date} to {end_date}. "
                "Check date range and data availability for selected tickers."
            )
        
        # Get currencies for each ticker
        currencies = {ticker: self.fx_converter.get_ticker_currency(ticker) 
                     for ticker in tickers}

        # Preload FX series once for all foreign currencies to avoid repeated
        # point-in-time downloads during valuation and execution checks.
        base_ccy = self.config['portfolio']['currency']
        unique_foreign_currencies = sorted({ccy for ccy in currencies.values() if ccy != base_ccy})
        if unique_foreign_currencies:
            preload_start = (effective_start_date - timedelta(days=30)).strftime('%Y-%m-%d')
            for foreign_ccy in unique_foreign_currencies:
                self.fx_converter.preload_pair(foreign_ccy, base_ccy, preload_start, end_date)
        
        # Initialize with starting positions if provided
        if initial_positions:
            self._initialize_starting_positions(
                initial_positions=initial_positions,
                first_date=trading_dates[0],
                data=data,
                currencies=currencies,
            )
        
        # Main backtest loop
        logger.info("Running backtest simulation...")
        print(f"Running backtest simulation over {len(trading_dates)} trading days...", flush=True)
        for i, date in enumerate(trading_dates):
            try:
                signal_date = trading_dates[i - 1] if i > 0 else None
                self._simulate_day(date, data, currencies, signal_date=signal_date)

                # Surface periodic progress to stdout so long runs don't appear stuck.
                if self.progress_log_interval > 0 and (i + 1) % self.progress_log_interval == 0:
                    pct = ((i + 1) / len(trading_dates)) * 100
                    print(
                        f"Progress: {i + 1}/{len(trading_dates)} days ({pct:.1f}%) - {date.date()}",
                        flush=True,
                    )
                    logger.info("Processed %s/%s days", i + 1, len(trading_dates))

            except DataIntegrityError as e:
                logger.error("Critical data integrity error on %s: %s", date, e)
                raise
            except (ValueError, KeyError) as e:
                # Treat value/key errors as recoverable per-day data/strategy issues.
                # Unexpected runtime/type/attribute errors should propagate.
                logger.error("Recoverable day-level error on %s: %s", date, e)
                continue
        
        # Calculate final metrics
        logger.info("Calculating performance metrics...")
        equity_curve = self.portfolio.get_equity_curve()

        benchmark_analysis = self._calculate_benchmark_analysis(
            equity_curve=equity_curve,
            start_date=effective_start_date.strftime('%Y-%m-%d'),
            end_date=end_date,
            tickers=tickers,
        )
        benchmark_returns = None
        if benchmark_analysis.get('available', False):
            benchmark_equity = benchmark_analysis.get('equity_curve')
            if benchmark_equity is not None and len(benchmark_equity) > 1:
                benchmark_returns = benchmark_equity.pct_change().dropna()

        metrics = PerformanceMetrics.get_comprehensive_metrics(
            equity_curve,
            self.portfolio.trades,
            risk_free_rate=0.02,
            benchmark_returns=benchmark_returns,
            base_currency=self.portfolio.base_currency,
        )
        metrics['beta_reference'] = benchmark_analysis.get('ticker', 'N/A')
        if not metrics.get('beta_correlation_available', False):
            metrics['beta_correlation_reason'] = benchmark_analysis.get(
                'reason',
                metrics.get('beta_correlation_reason', 'benchmark_returns_unavailable')
            )
        if benchmark_analysis.get('available', False):
            relative = benchmark_analysis.get('relative_stats', {}) or {}
            metrics['benchmark_alpha'] = float(relative.get('alpha', 0.0))
            metrics['benchmark_beta'] = float(relative.get('beta', 0.0))
            metrics['benchmark_tracking_error'] = float(relative.get('tracking_error', 0.0))
            metrics['benchmark_information_ratio'] = float(relative.get('information_ratio', 0.0))
            metrics['benchmark_active_return'] = float(relative.get('active_return', 0.0))
            metrics['benchmark_alpha_reference'] = benchmark_analysis.get('ticker', 'Benchmark')
        spy_analysis = benchmark_analysis.get('spy_analysis', {}) or {}
        if spy_analysis.get('available', False):
            spy_relative = spy_analysis.get('relative_stats', {}) or {}
            metrics['spy_alpha'] = float(spy_relative.get('alpha', 0.0))
            metrics['spy_beta'] = float(spy_relative.get('beta', 0.0))
            metrics['spy_tracking_error'] = float(spy_relative.get('tracking_error', 0.0))
            metrics['spy_information_ratio'] = float(spy_relative.get('information_ratio', 0.0))
            metrics['spy_active_return'] = float(spy_relative.get('active_return', 0.0))

        degradation_analysis = self._calculate_degradation_analysis(equity_curve)
        cost_sensitivity = self._run_transaction_cost_sensitivity(
            tickers=tickers,
            start_date=effective_start_date.strftime('%Y-%m-%d'),
            end_date=end_date
        )
        execution_quality = TransactionCostAnalysis.summarize(self.portfolio.trades)
        metrics.update(execution_quality)

        regime_series, regime_series_pit = self._build_regime_series_pair(equity_curve)
        regime_perf_pit = PerformanceMetrics.regime_performance(equity_curve, regime_series_pit)
        if regime_perf_pit:
            metrics['regime_performance'] = regime_perf_pit
        regime_perf_retro = PerformanceMetrics.regime_performance(equity_curve, regime_series)
        if regime_perf_retro:
            metrics['regime_performance_retrospective'] = regime_perf_retro
        per_ticker_regime_series = (
            self._build_per_ticker_regime_series(
                market_data=data,
                start_date=effective_start_date.strftime('%Y-%m-%d'),
                end_date=end_date
            )
            if bool(self.config.get('risk', {}).get('per_ticker_regime', False))
            else {}
        )
        attribution = self._calculate_strategy_attribution(equity_curve)
        
        # Compile results
        positions_history_df = pd.DataFrame(self.positions_history)
        if not positions_history_df.empty:
            positions_history_df = positions_history_df.set_index('date').sort_index()
            positions_history_df.index = pd.to_datetime(positions_history_df.index)

        results = {
            'run_id': self.current_run_id,
            'portfolio': self.portfolio,
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trades': self.portfolio.trades,
            'strategy': self.strategy.name,
            'tickers': tickers,
            'start_date': start_date,
            'effective_start_date': effective_start_date.strftime('%Y-%m-%d'),
            'required_history_days': required_history_days,
            'warmup_history_shortfall_days': self.warmup_history_shortfall_days,
            'end_date': end_date,
            'degradation_analysis': degradation_analysis,
            'benchmark_analysis': benchmark_analysis,
            'cost_sensitivity': cost_sensitivity,
            'execution_quality': execution_quality,
            'regime_series': regime_series,
            'regime_series_pit': regime_series_pit,
            'per_ticker_regime_series': per_ticker_regime_series,
            'attribution': attribution,
            'positions_history': positions_history_df,
            'asset_exclusions': list(self.asset_exclusions),
        }
        
        logger.info("Backtest complete!")
        PerformanceMetrics.print_metrics(metrics)

        if self.audit_store:
            trade_count = self.audit_store.record_trades(
                self.portfolio.trades,
                strategy=self.strategy.name,
                source='backtest',
                run_id=self.current_run_id,
            )
            points = self.audit_store.record_equity_curve(
                equity_curve,
                strategy=self.strategy.name,
                source='backtest',
                cash_curve=self.portfolio.get_cash_curve(),
                run_id=self.current_run_id,
            )
            self.audit_store.save_state_snapshot(
                {
                    'cash': self.portfolio.cash,
                    'positions': self.portfolio.positions,
                    'base_currency': self.portfolio.base_currency,
                    'run_id': self.current_run_id,
                    'run': {'start_date': start_date, 'end_date': end_date, 'tickers': tickers},
                },
                source='backtest',
                strategy=self.strategy.name,
                run_id=self.current_run_id,
            )
            self.audit_store.record_backtest_run(
                run_id=self.current_run_id,
                strategy=self.strategy.name,
                start_date=start_date,
                end_date=end_date,
                tickers=tickers,
                metrics=metrics,
                benchmark_analysis=benchmark_analysis,
                degradation_analysis=degradation_analysis,
                cost_sensitivity=cost_sensitivity,
                source='backtest',
            )
            self.audit_store.log_event(
                event_type='backtest_completed',
                source='backtest',
                strategy=self.strategy.name,
                details={
                    'trade_count': trade_count,
                    'equity_points': points,
                    'metrics': metrics,
                    'required_history_days': required_history_days,
                    'effective_start_date': effective_start_date.strftime('%Y-%m-%d'),
                    'warmup_history_shortfall_days': self.warmup_history_shortfall_days,
                },
                run_id=self.current_run_id,
            )

        if self.metrics_store:
            tags = {'strategy': self.strategy.name, 'run_id': self.current_run_id or ''}
            metric_ts = datetime.utcnow()
            for metric_name, value in (metrics or {}).items():
                if isinstance(value, bool):
                    numeric_value = float(value)
                elif isinstance(value, Real):
                    numeric_value = float(value)
                else:
                    continue
                self.metrics_store.write_metric(
                    f'backtest.{metric_name}',
                    numeric_value,
                    metric_ts,
                    tags=tags,
                )

            run_id = (self.current_run_id or '').strip()
            for metric_name, value in (metrics or {}).items():
                if not isinstance(value, pd.Series) or value.empty:
                    continue
                series_key = f"backtest.{metric_name}.{run_id}" if run_id else f"backtest.{metric_name}"
                clean_series = value.dropna()
                for ts, point in clean_series.items():
                    if isinstance(point, bool):
                        point_val = float(point)
                    elif isinstance(point, Real):
                        point_val = float(point)
                    else:
                        continue
                    point_ts = pd.Timestamp(ts).to_pydatetime()
                    self.metrics_store.write_metric(
                        series_key,
                        point_val,
                        point_ts,
                        tags=tags,
                    )

        return results
    
    def _simulate_day(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame],
                     currencies: Dict[str, str],
                     signal_date: Optional[pd.Timestamp] = None) -> None:
        """
        Simulate a single trading day
        
        Args:
            date: Current date
            data: Dict of {ticker: DataFrame}
            currencies: Dict of {ticker: currency}
            signal_date: Date used for signal generation (must be prior to execution date)
        """
        # No prior bar means no eligible signal yet; only mark-to-market.
        if signal_date is None:
            close_prices = self._get_prices_for_date(data, date, field='Close')
            equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
            self._record_equity_and_notify(date, equity)
            return

        if not self._should_rebalance_on_date(date):
            # No rebalance scheduled for this date; still mark-to-market equity.
            close_prices = self._get_prices_for_date(data, date, field='Close')
            equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
            self._record_equity_and_notify(date, equity)
            return

        # Build signal-history data strictly up to the prior signal date,
        # recording explicit exclusions for tickers that cannot be evaluated.
        historical_data = {}
        required_bars = int(max(self.strategy.get_required_history(), 0))
        for ticker, df in data.items():
            if df is None or df.empty:
                self._record_asset_exclusion(
                    ticker=ticker,
                    date=pd.Timestamp(signal_date),
                    reason='data_error',
                    available_bars=0,
                    required_bars=required_bars,
                    detail='empty dataframe',
                )
                continue

            ticker_history = df.loc[:signal_date]
            available_bars = int(len(ticker_history))
            if available_bars < required_bars:
                self._record_asset_exclusion(
                    ticker=ticker,
                    date=pd.Timestamp(signal_date),
                    reason='insufficient_history',
                    available_bars=available_bars,
                    required_bars=required_bars,
                    detail='history shorter than strategy requirement',
                )
                continue

            if pd.Timestamp(signal_date) not in ticker_history.index:
                self._record_asset_exclusion(
                    ticker=ticker,
                    date=pd.Timestamp(signal_date),
                    reason='stale_data',
                    available_bars=available_bars,
                    required_bars=required_bars,
                    detail='missing bar on signal date',
                )
                continue

            historical_data[ticker] = ticker_history

        # Execute at current day open; value end-of-day equity at close.
        execution_prices = self._get_prices_for_date(data, date, field='Open')
        close_prices = self._get_prices_for_date(data, date, field='Close')
        volumes = self._get_avg_volumes(historical_data)
        
        # Calculate current equity and weights at execution price snapshot.
        equity = self.portfolio.get_total_equity(execution_prices, currencies, date=date)
        current_weights = self.portfolio.get_weights(execution_prices, currencies, date=date)
        
        # Generate signals from strategy
        signal_meta: Dict[str, Dict[str, Any]] = {}
        try:
            target_weights = self.strategy.generate_signals(
                signal_date, historical_data, current_weights
            )
            raw_signal_meta = (
                self.strategy.get_signal_metadata()
                if hasattr(self.strategy, 'get_signal_metadata')
                else {}
            )
            if isinstance(raw_signal_meta, dict):
                signal_meta = {
                    str(ticker): dict(meta)
                    for ticker, meta in raw_signal_meta.items()
                    if isinstance(meta, dict)
                }
            elif raw_signal_meta is not None:
                logger.warning(
                    "Ignoring non-dict signal metadata from strategy %s: %s",
                    self.strategy.name,
                    type(raw_signal_meta).__name__,
                )
        except (ValueError, KeyError) as e:
            # Recoverable strategy-data mismatch (e.g., missing key / malformed slice):
            # hold current weights and continue simulation.
            logger.error("Recoverable strategy error on %s: %s", date, e)
            target_weights = current_weights
            signal_meta = {}
        target_weights = target_weights or {}
        signal_universe = set(target_weights.keys()) | set((current_weights or {}).keys())
        if signal_universe:
            signal_meta = {
                ticker: meta
                for ticker, meta in signal_meta.items()
                if ticker in signal_universe
            }
        else:
            signal_meta = {}

        max_weight_delta = self._calculate_max_weight_deviation(target_weights, current_weights)
        if max_weight_delta <= self.hold_weight_epsilon:
            logger.debug(
                "Skipping rebalance on %s: max target/current deviation %.8f <= epsilon %.8f",
                date.date(),
                max_weight_delta,
                self.hold_weight_epsilon,
            )
            final_equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
            self._record_equity_and_notify(date, final_equity)
            return

        total_weight_drift_l1 = self._calculate_weight_drift_l1(target_weights, current_weights)
        if total_weight_drift_l1 < self.min_rebalance_weight_delta:
            logger.debug(
                "Skipping rebalance on %s: L1 target/current drift %.6f < min_rebalance_weight_delta %.6f",
                date.date(),
                total_weight_drift_l1,
                self.min_rebalance_weight_delta,
            )
            final_equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
            self._record_equity_and_notify(date, final_equity)
            return

        # Apply position sizing
        if target_weights:
            target_weights = self.position_sizer.size_positions(
                target_weights,
                historical_data,
                equity
            )

        # Portfolio optimization (optional) before final constraints pass.
        if self.use_optimizer and self.optimizer and target_weights:
            try:
                optimizer_method = self.config['risk'].get('optimizer_method', 'max_sharpe')
                target_weights = self.optimizer.optimize(
                    target_weights,
                    historical_data,
                    method=optimizer_method
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                logger.warning("Optimization failed on %s: %s", date, e)
                # Keep sized weights and continue to final constraints pass.

        # Final risk-constraint pass so executed targets remain compliant.
        if target_weights:
            target_weights = self.risk_constraints.apply_constraints(
                target_weights,
                historical_data
            )

        if target_weights and self.position_sizer.method == 'target_vol':
            target_weights = self.position_sizer.apply_final_volatility_scaling(
                target_weights,
                historical_data,
                max_total_exposure=max(0.0, 1.0 - float(self.risk_constraints.min_cash_reserve)),
            )
        
        # Calculate target positions in shares
        target_positions = self._calculate_target_positions(
            target_weights, equity, execution_prices, currencies, date=date
        )
        current_positions = dict(self.portfolio.positions)
        
        # Execute trades to reach target positions
        trades = self.execution_engine.execute_rebalance(
            target_positions=target_positions,
            current_positions=current_positions,
            prices=execution_prices,
            volumes=volumes,
            currencies=currencies,
            date=date,
            timing=self.timing,
            target_weights=target_weights,
            current_weights=current_weights,
            signal_date=signal_date,
            signal_meta=signal_meta,
            run_id=self.current_run_id,
        )

        # Execute sells first to release cash, then buys
        trades = sorted(trades, key=lambda t: 0 if t.action == 'SELL' else 1)
        executed_trades = []
        
        # Update portfolio with trades
        for trade in trades:
            if trade.action == 'BUY':
                if not self._has_sufficient_forward_horizon(trade.ticker, date, data):
                    continue
                # If buy is not affordable after FX/fees, scale down shares
                if not self.portfolio.execute_trade(trade):
                    resized_trade = self._resize_buy_trade_to_cash(
                        trade=trade,
                        price=execution_prices[trade.ticker],
                        avg_volume=volumes.get(trade.ticker, 0),
                        current_shares=self.portfolio.get_position_shares(trade.ticker),
                        available_cash=self.portfolio.cash
                    )

                    if resized_trade:
                        if self.portfolio.execute_trade(resized_trade):
                            executed_trades.append(resized_trade)
                else:
                    executed_trades.append(trade)
            else:
                if self.portfolio.execute_trade(trade):
                    executed_trades.append(trade)

        decision_rows = self._build_trade_decision_rows(
            date=date,
            signal_date=signal_date,
            target_positions=target_positions,
            current_positions=current_positions,
            target_weights=target_weights,
            current_weights=current_weights,
            executed_trades=executed_trades,
            signal_meta=signal_meta,
        )
        if self.audit_store and decision_rows:
            self.audit_store.record_trade_decisions(
                decision_rows,
                strategy=self.strategy.name,
                source='backtest',
                run_id=self.current_run_id,
            )

        if target_weights:
            run_drift_check = True
            if self.drift_warn_min_capital > 0:
                drift_check_equity = float(
                    self.portfolio.get_total_equity(execution_prices, currencies, date=date)
                )
                if drift_check_equity < self.drift_warn_min_capital:
                    run_drift_check = False
                    self._consecutive_drift_warnings = 0

            if not run_drift_check:
                final_equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
                self._record_equity_and_notify(date, final_equity)
                return

            realized_invested_weights = self._get_realized_invested_weights(
                execution_prices, currencies, date=date
            )
            target_invested_weights = self._normalize_to_invested_sleeve(target_weights)
            invested_sleeve_drift_l1 = self._calculate_weight_drift_l1(
                target_invested_weights, realized_invested_weights
            )
            target_cash_weight = max(0.0, 1.0 - float(sum(target_weights.values())))
            realized_cash_weight = self.portfolio.get_cash_weight(
                execution_prices, currencies, date=date
            )
            cash_drift = abs(float(realized_cash_weight) - float(target_cash_weight))

            breached = (
                invested_sleeve_drift_l1 > self.invested_sleeve_drift_warn
                or cash_drift > self.cash_drift_warn
            )

            if breached:
                log_fn = logger.warning if self._consecutive_drift_warnings == 0 else logger.debug
                log_fn(
                    "Rebalance execution drift on %s: invested_sleeve_drift_l1=%.4f "
                    "(threshold=%.4f), cash_drift=%.4f (threshold=%.4f). "
                    "target_invested=%s realized_invested=%s target_cash=%.4f realized_cash=%.4f",
                    date.date(),
                    invested_sleeve_drift_l1,
                    self.invested_sleeve_drift_warn,
                    cash_drift,
                    self.cash_drift_warn,
                    target_invested_weights,
                    realized_invested_weights,
                    target_cash_weight,
                    realized_cash_weight,
                )
                self._consecutive_drift_warnings += 1
                if self.audit_store:
                    self.audit_store.log_event(
                        event_type='rebalance_weight_drift',
                        source='backtest',
                        strategy=self.strategy.name,
                        details={
                            'date': str(date.date()),
                            'invested_sleeve_drift_l1': invested_sleeve_drift_l1,
                            'invested_sleeve_threshold': self.invested_sleeve_drift_warn,
                            'cash_drift': cash_drift,
                            'cash_drift_threshold': self.cash_drift_warn,
                            'target_weights': target_weights,
                            'target_invested_weights': target_invested_weights,
                            'realized_invested_weights': realized_invested_weights,
                            'target_cash_weight': target_cash_weight,
                            'realized_cash_weight': realized_cash_weight,
                        },
                        run_id=self.current_run_id,
                    )
            else:
                self._consecutive_drift_warnings = 0
        else:
            self._consecutive_drift_warnings = 0
        
        self._last_rebalance_date = date

        realized_weights_close = self.portfolio.get_weights(close_prices, currencies, date=date)
        history_row = {'date': pd.Timestamp(date)}
        history_row.update({ticker: float(weight) for ticker, weight in realized_weights_close.items()})
        self.positions_history.append(history_row)

        # Record final equity for the day
        final_equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
        self._record_equity_and_notify(date, final_equity)

    def _initialize_starting_positions(
        self,
        initial_positions: Dict[str, float],
        first_date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        currencies: Dict[str, str],
    ) -> None:
        """Seed portfolio holdings from target starting weights on first trading date."""
        positive_weights = {
            ticker: float(weight)
            for ticker, weight in (initial_positions or {}).items()
            if float(weight) > 0
        }
        if not positive_weights:
            return

        unknown_tickers = sorted(set(positive_weights) - set(data.keys()))
        if unknown_tickers:
            raise ValueError(f"initial_positions contains unknown tickers: {unknown_tickers}")

        total_weight = float(sum(positive_weights.values()))
        if total_weight > 1.0 + 1e-9:
            raise ValueError(
                f"initial_positions weights sum to {total_weight:.4f}; expected <= 1.0"
            )

        open_prices = self._get_prices_for_date(data, first_date, field='Open')
        if not open_prices:
            open_prices = self._get_prices_for_date(data, first_date, field='Close')

        missing_prices = sorted([ticker for ticker in positive_weights if ticker not in open_prices])
        if missing_prices:
            raise ValueError(
                f"Missing first-bar prices for initial_positions tickers: {missing_prices}"
            )

        equity = float(self.portfolio.cash)
        target_positions = self._calculate_target_positions(
            weights=positive_weights,
            equity=equity,
            prices=open_prices,
            currencies=currencies,
            date=first_date,
        )
        if not target_positions:
            logger.warning("initial_positions produced no executable shares on %s", first_date.date())
            return

        estimated_total_cost = self._estimate_total_target_cost_base(
            target_positions=target_positions,
            prices=open_prices,
            currencies=currencies,
            date=first_date,
        )

        self.portfolio.positions = {ticker: float(shares) for ticker, shares in target_positions.items()}
        self.portfolio.position_currencies.update(
            {ticker: currencies.get(ticker, self.portfolio.base_currency) for ticker in target_positions}
        )
        self.portfolio.cash = max(0.0, equity - float(estimated_total_cost))

        logger.info(
            "Initialized starting positions on %s: %s (cash %.2f)",
            first_date.date(),
            self.portfolio.positions,
            self.portfolio.cash,
        )
    
    def _should_rebalance_on_date(self, date: pd.Timestamp) -> bool:
        """Determine whether to rebalance on this date based on configured timeframe."""
        tf = self.rebalance_timeframe
        if tf == 'daily':
            return True
        if tf == 'weekly':
            # Friday rebalance, with a month-end fallback for shortened weeks.
            # Guard with a >=5-day minimum interval to prevent duplicate week-boundary rebalances.
            is_weekly_rebalance_day = (
                (date.weekday() == 4) or
                ((date + pd.tseries.offsets.BDay(1)).month != date.month and date.weekday() >= 3)
            )
            if not is_weekly_rebalance_day:
                return False
            if self._last_rebalance_date is None:
                return True
            return (date - self._last_rebalance_date).days >= 5
        if tf == 'monthly':
            return (date + pd.tseries.offsets.BDay(1)).month != date.month
        logger.warning("Unknown rebalance timeframe '%s', defaulting to daily", tf)
        return True

    def _has_sufficient_forward_horizon(self, ticker: str, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> bool:
        """Return True when ticker has enough bars after execution date for new risk."""
        if self.min_forward_data_days <= 0:
            return True

        df = data.get(ticker)
        if df is None or df.empty:
            return False

        forward_bars = int((df.index > date).sum())
        if forward_bars < self.min_forward_data_days:
            data_end = pd.Timestamp(df.index.max()).date()
            logger.warning(
                "Blocking BUY/ENTRY/INCREASE for %s on %s: forward bars=%d, required=%d, data_end=%s",
                ticker,
                pd.Timestamp(date).date(),
                forward_bars,
                self.min_forward_data_days,
                data_end,
            )
            return False
        return True

    def _get_trading_dates(self, data: Dict[str, pd.DataFrame],
                          start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Get union trading dates across the configured universe.

        Missing ticker prices on a union date are handled by the portfolio's stale-price
        carry-forward logic (subject to max_stale_price_days).
        """
        date_union = pd.DatetimeIndex([])
        for _, df in data.items():
            ticker_dates = df.loc[start_date:end_date].index
            date_union = date_union.union(ticker_dates)

        if len(date_union) == 0:
            return pd.DatetimeIndex([])
        return date_union.sort_values()

    def _compute_effective_start_date(self,
                                      trading_dates_all: pd.DatetimeIndex,
                                      requested_start_date: str,
                                      required_history_days: int) -> pd.Timestamp:
        """Compute effective simulation start date after warmup history alignment."""
        if len(trading_dates_all) == 0:
            raise ValueError("No trading dates available to compute effective start")

        requested_ts = pd.Timestamp(requested_start_date)
        eligible_dates = trading_dates_all[trading_dates_all >= requested_ts]
        if len(eligible_dates) == 0:
            raise ValueError(
                f"No trading dates available on/after requested start {requested_start_date}"
            )

        base_date = pd.Timestamp(eligible_dates[0])
        if required_history_days <= 0:
            self.warmup_history_shortfall_days = 0
            return base_date

        base_idx = int(trading_dates_all.get_indexer([base_date])[0])
        effective_idx = base_idx + int(required_history_days)
        if effective_idx >= len(trading_dates_all):
            shortfall_days = int(effective_idx - (len(trading_dates_all) - 1))
            self.warmup_history_shortfall_days = shortfall_days
            logger.warning(
                "Insufficient trading history to satisfy warmup requirement; "
                "falling back to requested start. required_history_days=%d available_dates=%d shortfall_days=%d",
                required_history_days,
                len(trading_dates_all),
                shortfall_days,
            )
            return base_date
        self.warmup_history_shortfall_days = 0
        return pd.Timestamp(trading_dates_all[effective_idx])

    def _record_asset_exclusion(self,
                                ticker: str,
                                date: pd.Timestamp,
                                reason: str,
                                available_bars: int,
                                required_bars: int,
                                detail: str) -> None:
        """Record structured asset exclusion for backtest diagnostics."""
        record = AssetExclusionRecord(
            ticker=str(ticker),
            date=pd.Timestamp(date),
            reason=str(reason),
            available_bars=int(max(available_bars, 0)),
            required_bars=int(max(required_bars, 0)),
            detail=str(detail),
        )
        self.asset_exclusions.append(record)
        logger.debug(
            "Asset exclusion ticker=%s date=%s reason=%s available_bars=%d required_bars=%d detail=%s",
            record.ticker,
            record.date.date(),
            record.reason,
            record.available_bars,
            record.required_bars,
            record.detail,
        )

    def _get_prices_for_date(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp,
                             field: str = 'Close') -> Dict[str, float]:
        """Get strict per-ticker prices for a specific date/field."""
        prices = {}
        for ticker, df in data.items():
            if field not in df.columns:
                logger.warning("%s missing '%s' column on %s; skipping", ticker, field, date.date())
                continue
            if date not in df.index:
                logger.warning("%s missing %s %s bar; skipping", ticker, date.date(), field)
                continue
            prices[ticker] = float(df.loc[date, field])
        return prices
    
    def _get_avg_volumes(self, data: Dict[str, pd.DataFrame],
                        window: int = 20) -> Dict[str, float]:
        """Get average volumes for all tickers"""
        volumes = {}
        for ticker, df in data.items():
            volume_series = df.get('Volume', pd.Series(dtype=float))
            if len(volume_series) >= window:
                mean_volume = float(volume_series.tail(window).mean())
            elif len(volume_series) > 0:
                mean_volume = float(volume_series.mean())
            else:
                mean_volume = 0.0

            volumes[ticker] = mean_volume if mean_volume > 0 else self.missing_volume_fallback
        return volumes

    @staticmethod
    def _calculate_weight_drift_l1(target_weights: Dict[str, float],
                                   realized_weights: Dict[str, float]) -> float:
        """Compute L1 distance between target and realized portfolio weights."""
        tickers = set(target_weights.keys()) | set(realized_weights.keys())
        return float(
            sum(
                abs(float(target_weights.get(ticker, 0.0)) - float(realized_weights.get(ticker, 0.0)))
                for ticker in tickers
            )
        )

    @staticmethod
    def _calculate_max_weight_deviation(target_weights: Dict[str, float],
                                        current_weights: Dict[str, float]) -> float:
        """Return max absolute per-ticker deviation across target/current vectors."""
        tickers = set((target_weights or {}).keys()) | set((current_weights or {}).keys())
        if not tickers:
            return 0.0
        return max(
            abs(float((target_weights or {}).get(ticker, 0.0)) - float((current_weights or {}).get(ticker, 0.0)))
            for ticker in tickers
        )

    @staticmethod
    def _normalize_to_invested_sleeve(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize position weights to the invested sleeve (exclude target cash)."""
        weights = weights or {}
        invested_total = float(sum(max(float(w), 0.0) for w in weights.values()))
        if invested_total <= 0:
            return {}
        return {
            ticker: float(max(weight, 0.0) / invested_total)
            for ticker, weight in weights.items()
            if float(weight) > 0
        }

    def _get_realized_invested_weights(self,
                                       prices: Dict[str, float],
                                       currencies: Dict[str, str],
                                       date: Optional[datetime] = None) -> Dict[str, float]:
        """Compute realized ex-cash position weights."""
        portfolio_weights = self.portfolio.get_weights(prices, currencies, date=date)
        invested_weight = float(sum(portfolio_weights.values()))
        if invested_weight <= 0:
            return {}

        return {
            ticker: float(weight / invested_weight)
            for ticker, weight in portfolio_weights.items()
            if float(weight) > 0
        }

    def _build_trade_decision_rows(self,
                                   date: pd.Timestamp,
                                   signal_date: Optional[pd.Timestamp],
                                   target_positions: Dict[str, int],
                                   current_positions: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   current_weights: Dict[str, float],
                                   executed_trades: List[Any],
                                   signal_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Build decision-level journal rows across traded and held tickers."""
        target_weights = target_weights or {}
        current_weights = current_weights or {}
        tickers = set(target_positions.keys()) | set(current_positions.keys())
        tickers |= set(target_weights.keys()) | set(current_weights.keys())
        if not tickers:
            return []

        executed_by_ticker: Dict[str, float] = {}
        for trade in executed_trades or []:
            signed = float(trade.shares) if trade.action == 'BUY' else -float(trade.shares)
            executed_by_ticker[trade.ticker] = executed_by_ticker.get(trade.ticker, 0.0) + signed

        rows: List[Dict[str, Any]] = []
        for ticker in sorted(tickers):
            current_shares = float(current_positions.get(ticker, 0.0))
            target_shares = float(target_positions.get(ticker, 0.0))
            current_weight = float(current_weights.get(ticker, 0.0))
            target_weight = float(target_weights.get(ticker, 0.0))
            weight_delta = float(target_weight - current_weight)
            meta = (signal_meta or {}).get(ticker, {}) or {}
            raw_signal_strength = meta.get('signal_strength')
            if raw_signal_strength is not None:
                try:
                    signal_strength = float(raw_signal_strength)
                except (TypeError, ValueError):
                    signal_strength = None
            else:
                signal_strength = None
            decision = self.execution_engine.infer_decision_label(current_shares, target_shares)
            raw_confidence = meta.get('confidence')
            if raw_confidence is not None:
                try:
                    confidence = float(raw_confidence)
                except (TypeError, ValueError):
                    confidence = None
            else:
                confidence = None
            decision_reason = str(meta.get('reason') or 'strategy_rebalance')

            rows.append({
                'ts': pd.Timestamp(date),
                'signal_ts': pd.Timestamp(signal_date) if signal_date is not None else None,
                'ticker': ticker,
                'decision': decision,
                'decision_reason': decision_reason,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'signal_strength': signal_strength,
                'weight_delta': weight_delta,
                'confidence': confidence,
                'current_shares': current_shares,
                'target_shares': target_shares,
                'executed_shares': float(executed_by_ticker.get(ticker, 0.0)),
                'run_id': self.current_run_id,
            })
        return rows

    def _record_equity_and_notify(self, date: pd.Timestamp, equity: float) -> None:
        """
        Record equity and notify strategy of realized portfolio return when available.
        """
        previous_equity = (
            float(self.portfolio.equity_history[-1])
            if self.portfolio.equity_history
            else None
        )
        self.portfolio.record_equity(date, equity, cash=self.portfolio.cash)

        if previous_equity is None or previous_equity <= 0:
            return

        realized_return = (float(equity) / previous_equity) - 1.0
        if hasattr(self.strategy, 'on_realized_portfolio_return'):
            try:
                self.strategy.on_realized_portfolio_return(
                    pd.Timestamp(date),
                    float(realized_return),
                    context={
                        'engine': 'backtest',
                        'equity': float(equity),
                        'previous_equity': previous_equity,
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Strategy on_realized_portfolio_return hook failed on %s: %s",
                    pd.Timestamp(date).date(),
                    exc,
                )
    
    def _calculate_target_positions(self, weights: Dict[str, float], equity: float,
                                   prices: Dict[str, float],
                                   currencies: Dict[str, str],
                                   date: datetime = None) -> Dict[str, int]:
        """
        Convert target weights to target share quantities
        
        Args:
            weights: Target weights
            equity: Current portfolio equity
            prices: Current prices
            currencies: Ticker currencies
            
        Returns:
            Dict of {ticker: target_shares}
        """
        target_positions: Dict[str, int] = {}
        weights = weights or {}
        if not weights or equity <= 0:
            return target_positions

        ranked_weights = sorted(
            ((ticker, float(weight)) for ticker, weight in weights.items() if float(weight) > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        total_investable_budget_base = 0.0

        for ticker, weight in ranked_weights:
            if ticker not in prices:
                continue
            
            # Calculate target value in base currency
            target_value_base = equity * weight
            
            # Get ticker's currency
            currency = currencies.get(ticker, 'USD')
            
            # Convert to ticker's currency
            if currency != self.portfolio.base_currency:
                # Get FX rate
                fx_rate = self.fx_converter.get_fx_rate(
                    self.portfolio.base_currency, currency, date=date
                )
                target_value_foreign = target_value_base * fx_rate
            else:
                target_value_foreign = target_value_base

            expected_cost_base = self._estimate_expected_trade_cost_base(
                target_value_base=target_value_base,
                currency=currency,
            )
            expected_cost_foreign = (
                expected_cost_base * fx_rate
                if currency != self.portfolio.base_currency
                else expected_cost_base
            )

            investable_value_foreign = max(target_value_foreign - expected_cost_foreign, 0.0)
            total_investable_budget_base += max(target_value_base - expected_cost_base, 0.0)

            # Calculate shares (round down to whole shares)
            price = prices[ticker]
            target_shares = int(investable_value_foreign / price)
            
            if target_shares > 0:
                target_positions[ticker] = target_shares

        target_positions = self._apply_min_trade_value_filter(
            target_positions=target_positions,
            prices=prices,
            currencies=currencies,
            equity=equity,
            date=date,
        )

        # Greedy residual allocation by priority to reduce idle cash.
        estimated_total_cost = self._estimate_total_target_cost_base(
            target_positions=target_positions,
            prices=prices,
            currencies=currencies,
            date=date,
        )
        residual_budget = total_investable_budget_base - estimated_total_cost
        if residual_budget > 0 and ranked_weights:
            max_iters = max(1, len(ranked_weights) * 1000)
            iters = 0
            made_progress = True
            while residual_budget > 0 and made_progress and iters < max_iters:
                made_progress = False
                for ticker, _ in ranked_weights:
                    if ticker not in prices:
                        continue
                    incremental_cost = self._estimate_incremental_share_cost_base(
                        ticker=ticker,
                        price=prices[ticker],
                        currency=currencies.get(ticker, 'USD'),
                        date=date,
                    )
                    if incremental_cost <= residual_budget + 1e-10:
                        target_positions[ticker] = int(target_positions.get(ticker, 0)) + 1
                        residual_budget -= incremental_cost
                        made_progress = True
                iters += 1

        # Strict affordability check against NAV before emitting orders.
        estimated_total_cost = self._estimate_total_target_cost_base(
            target_positions=target_positions,
            prices=prices,
            currencies=currencies,
            date=date,
        )
        max_allowed_cost = min(equity, total_investable_budget_base)
        if estimated_total_cost > max_allowed_cost + 1e-8:
            for ticker, _ in sorted(ranked_weights, key=lambda item: item[1]):
                while target_positions.get(ticker, 0) > 0 and estimated_total_cost > max_allowed_cost + 1e-8:
                    target_positions[ticker] -= 1
                    if target_positions[ticker] <= 0:
                        target_positions.pop(ticker, None)
                    estimated_total_cost = self._estimate_total_target_cost_base(
                        target_positions=target_positions,
                        prices=prices,
                        currencies=currencies,
                        date=date,
                    )
                if estimated_total_cost <= max_allowed_cost + 1e-8:
                    break

        expected_cash_reserve = max(equity - total_investable_budget_base, 0.0)
        residual_cash_above_target = max((equity - estimated_total_cost) - expected_cash_reserve, 0.0)
        if equity > 0 and (residual_cash_above_target / equity) > self.uninvested_cash_tolerance:
            logger.warning(
                "Estimated residual cash %.4f exceeds tolerance %.4f after cost-aware sizing",
                residual_cash_above_target / equity,
                self.uninvested_cash_tolerance,
            )
        
        return target_positions


    def _position_value_base(self, ticker: str, shares: int, prices: Dict[str, float],
                             currencies: Dict[str, str], date: Optional[datetime] = None) -> float:
        """Return base-currency notional for a position at current price snapshot."""
        if shares <= 0 or ticker not in prices:
            return 0.0
        notional_foreign = float(shares) * float(prices[ticker])
        currency = currencies.get(ticker, 'USD')
        if currency != self.portfolio.base_currency:
            return float(self.fx_converter.convert_to_base(notional_foreign, currency, date=date))
        return float(notional_foreign)

    def _apply_min_trade_value_filter(self,
                                      target_positions: Dict[str, int],
                                      prices: Dict[str, float],
                                      currencies: Dict[str, str],
                                      equity: float,
                                      date: Optional[datetime] = None) -> Dict[str, int]:
        """Remove tiny target positions and redistribute freed budget within max-position caps."""
        if not target_positions:
            return {}

        min_trade_value = max(0.0, float(self.min_trade_value))
        if min_trade_value <= 0:
            return dict(target_positions)

        positions = dict(target_positions)
        max_position_size = max(0.0, min(1.0, float(self.risk_constraints.max_position_size)))
        max_position_value = float(equity) * max_position_size if equity > 0 else 0.0

        residual_budget = 0.0
        for ticker in list(positions.keys()):
            position_value = self._position_value_base(ticker, int(positions[ticker]), prices, currencies, date=date)
            if position_value < min_trade_value:
                residual_budget += position_value
                positions.pop(ticker, None)

        if residual_budget <= 1e-9 or not positions:
            return positions

        max_iterations = 10
        for _ in range(max_iterations):
            if residual_budget <= (1e-6 * max(float(equity), 1.0)):
                break

            remaining = []
            total_remaining_value = 0.0
            for ticker, shares in positions.items():
                current_value = self._position_value_base(ticker, int(shares), prices, currencies, date=date)
                capacity = max(0.0, max_position_value - current_value)
                if capacity <= 0:
                    continue
                remaining.append((ticker, current_value, capacity))
                total_remaining_value += max(current_value, 0.0)

            if not remaining:
                break

            made_progress = False
            weights_denominator = total_remaining_value if total_remaining_value > 0 else float(len(remaining))
            for ticker, current_value, capacity in remaining:
                base_price = self._position_value_base(ticker, 1, prices, currencies, date=date)
                if base_price <= 0:
                    continue
                alloc_weight = (current_value / weights_denominator) if total_remaining_value > 0 else (1.0 / len(remaining))
                alloc_budget = min(residual_budget * alloc_weight, capacity)
                add_shares = int(alloc_budget / base_price)
                if add_shares <= 0:
                    continue
                positions[ticker] = int(positions.get(ticker, 0)) + add_shares
                spent = add_shares * base_price
                residual_budget = max(0.0, residual_budget - spent)
                made_progress = True

            if not made_progress:
                break

        if residual_budget > (1e-6 * max(float(equity), 1.0)):
            logger.debug(
                "Residual budget %.4f left after min_trade_value redistribution; retained as cash",
                residual_budget,
            )

        return positions

    def _estimate_expected_trade_cost_base(self, target_value_base: float, currency: str) -> float:
        """Estimate conservative per-asset transaction costs in base currency."""
        notional = max(float(target_value_base), 0.0)
        if notional <= 0:
            return 0.0
        fee_cost = notional * (self.cost_aware_fee_bps / 10000.0)
        spread_cost = notional * (self.cost_aware_spread_bps / 10000.0)
        fx_cost = (
            notional * float(self.execution_engine.fx_conversion_rate)
            if currency != self.portfolio.base_currency
            else 0.0
        )
        return fee_cost + spread_cost + fx_cost

    def _estimate_incremental_share_cost_base(self, ticker: str, price: float, currency: str,
                                              date: Optional[datetime] = None) -> float:
        """Estimate total base-currency cost of adding one share."""
        if currency != self.portfolio.base_currency:
            notional_base = self.fx_converter.convert_to_base(float(price), currency, date=date)
        else:
            notional_base = float(price)
        return notional_base + self._estimate_expected_trade_cost_base(notional_base, currency)

    def _estimate_total_target_cost_base(self,
                                         target_positions: Dict[str, int],
                                         prices: Dict[str, float],
                                         currencies: Dict[str, str],
                                         date: Optional[datetime] = None) -> float:
        """Estimate total base-currency notional + expected costs for target positions."""
        total = 0.0
        for ticker, shares in (target_positions or {}).items():
            if shares <= 0 or ticker not in prices:
                continue
            currency = currencies.get(ticker, 'USD')
            notional_foreign = float(shares) * float(prices[ticker])
            if currency != self.portfolio.base_currency:
                notional_base = self.fx_converter.convert_to_base(notional_foreign, currency, date=date)
            else:
                notional_base = notional_foreign
            total += notional_base + self._estimate_expected_trade_cost_base(notional_base, currency)
        return total


    def _build_regime_series_pair(self, equity_curve: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Build retrospective and point-in-time regime labels."""
        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            empty = pd.Series(dtype='object')
            return empty, empty

        # Prefer HMM full-series decode for stable retrospective labels.
        strategy_cfg = (self.strategy.config or {}) if hasattr(self.strategy, 'config') else {}
        hmm_n_states = int(strategy_cfg.get('hmm_n_states', 3))
        hmm_covariance_type = str(strategy_cfg.get('hmm_covariance_type', 'full'))

        if HMMRegimeDetector is not None:
            try:
                detector = HMMRegimeDetector(
                    n_states=hmm_n_states,
                    covariance_type=hmm_covariance_type,
                    min_fit_observations=60,
                )
                retrospective = detector.decode_full_series(returns)
                pit = detector.decode_point_in_time(returns, min_history=detector.min_fit_observations)
                return retrospective, pit
            except (ValueError, RuntimeError) as exc:
                logger.warning("HMM retrospective decode unavailable, falling back to heuristic: %s", exc)

        labels = {}
        for dt in returns.index:
            hist = returns.loc[:dt]
            regime = AdvancedRiskAnalytics.regime_based_scaler(hist).get('regime', 'neutral')
            if regime not in {'bull', 'neutral', 'bear'}:
                regime = 'neutral'
            labels[dt] = regime
        heuristic = pd.Series(labels, dtype='object')
        return heuristic, heuristic.copy()

    def _build_regime_series(self, equity_curve: pd.Series) -> pd.Series:
        """Backward-compatible accessor for retrospective regime labels."""
        retrospective, _ = self._build_regime_series_pair(equity_curve)
        return retrospective

    def _build_per_ticker_regime_series(self, market_data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Build retrospective regime labels per ticker when enabled."""
        out: Dict[str, pd.Series] = {}
        if HMMRegimeDetector is None:
            return out

        strategy_cfg = (self.strategy.config or {}) if hasattr(self.strategy, 'config') else {}
        hmm_n_states = int(strategy_cfg.get('hmm_n_states', 3))
        hmm_covariance_type = str(strategy_cfg.get('hmm_covariance_type', 'full'))
        for ticker, frame in (market_data or {}).items():
            returns = pd.Series(dtype=float)
            try:
                # Slice to the effective simulation window before regime labelling.
                sliced = frame.loc[start_date:end_date]
                close = sliced['Close'].astype(float).dropna()
                returns = close.pct_change().dropna()
                if len(returns) < 60:
                    continue
                detector = HMMRegimeDetector(
                    n_states=hmm_n_states,
                    covariance_type=hmm_covariance_type,
                    min_fit_observations=60,
                )
                out[ticker] = detector.decode_full_series(returns)
            except (KeyError, ValueError, RuntimeError) as exc:
                labels = {}
                for dt in returns.index:
                    hist = returns.loc[:dt]
                    regime = AdvancedRiskAnalytics.regime_based_scaler(hist).get('regime', 'neutral')
                    if regime not in {'bull', 'neutral', 'bear'}:
                        regime = 'neutral'
                    labels[dt] = regime
                if labels:
                    out[ticker] = pd.Series(labels, dtype='object')
                logger.debug("Per-ticker regime decode fell back to heuristic for %s: %s", ticker, exc)
        return out

    def _calculate_strategy_attribution(self, equity_curve: pd.Series) -> Dict[str, Dict[str, float]]:
        """Estimate per-sub-strategy contribution metrics for orchestration strategy runs."""
        if not hasattr(self.strategy, 'get_attribution_history'):
            return {}
        try:
            history = self.strategy.get_attribution_history()
        except (AttributeError, TypeError, ValueError):
            return {}
        if not history:
            return {}

        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            return {}

        weights_by_strategy: Dict[str, list[float]] = {}
        contribution_by_strategy: Dict[str, list[float]] = {}

        return_index = returns.index
        for row in history:
            signal_dt = pd.Timestamp(row.get('date')) if isinstance(row, dict) and row.get('date') is not None else None
            alloc = row.get('strategy_allocations', {}) if isinstance(row, dict) else {}
            if signal_dt is None or not isinstance(alloc, dict):
                continue

            exec_pos = return_index.searchsorted(signal_dt, side='right')
            if exec_pos >= len(return_index):
                continue

            portfolio_ret = float(returns.iloc[exec_pos])
            for strategy_name, ticker_alloc in alloc.items():
                total_weight = float(sum((ticker_alloc or {}).values())) if isinstance(ticker_alloc, dict) else 0.0
                weights_by_strategy.setdefault(strategy_name, []).append(total_weight)
                contribution_by_strategy.setdefault(strategy_name, []).append(total_weight * portfolio_ret)

        out: Dict[str, Dict[str, float]] = {}
        for strategy_name, contrib_values in contribution_by_strategy.items():
            contrib = pd.Series(contrib_values, dtype=float)
            if contrib.empty:
                continue
            eq = (1.0 + contrib).cumprod()
            running_max = eq.cummax()
            drawdown = ((eq - running_max) / running_max).min() if len(eq) else 0.0
            vol = float(contrib.std() * np.sqrt(252))
            ann = float(contrib.mean() * 252)
            sharpe = float(ann / vol) if vol > 0 else 0.0
            out[strategy_name] = {
                'total_return': float(eq.iloc[-1] - 1.0),
                'annualized_return': ann,
                'volatility': vol,
                'max_drawdown': float(abs(drawdown)),
                'sharpe_ratio': sharpe,
                'weight_fraction': float(np.mean(weights_by_strategy.get(strategy_name, [0.0]))),
                'contribution_to_portfolio_return': float(contrib.sum()),
            }
        return out

    def _calculate_benchmark_analysis(self, equity_curve: pd.Series,
                                      start_date: str, end_date: str,
                                      tickers: Optional[List[str]] = None,
                                      benchmark_ticker: str = 'SPY') -> Dict:
        """Calculate benchmark performance and relative stats.

        Primary benchmark: equal-weight buy-and-hold basket of all portfolio tickers.
        Fallback benchmark: single benchmark_ticker (default SPY).
        """
        if len(equity_curve) < 2:
            return {
                'available': False,
                'reason': 'insufficient_equity_history',
                'spy_analysis': {'available': False, 'reason': 'insufficient_equity_history'},
            }

        benchmark_close = None
        benchmark_label = benchmark_ticker

        # Prefer an equal-weight baseline built from the portfolio universe.
        if tickers:
            close_series = []
            for ticker in tickers:
                try:
                    ticker_data = self.data_manager.load_ticker(ticker, start_date, end_date)
                except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, OSError, ValueError) as exc:
                    logger.warning("Skipping %s in equal-weight benchmark: %s", ticker, exc)
                    continue

                if ticker_data.empty or 'Close' not in ticker_data.columns:
                    continue

                close = ticker_data['Close'].dropna()
                if len(close) < 2:
                    continue
                close_series.append(close.rename(ticker))

            if close_series:
                close_frame = pd.concat(close_series, axis=1, join='inner').dropna()
                if len(close_frame) >= 2:
                    # Equal initial weights across all tickers with available data.
                    normalized = close_frame.div(close_frame.iloc[0])
                    benchmark_close = normalized.mean(axis=1)
                    benchmark_label = f"EqualWeight({','.join(close_frame.columns)})"

        # Fall back to SPY (or configured benchmark_ticker) when equal-weight cannot be built.
        if benchmark_close is None:
            try:
                benchmark_data = self.data_manager.load_ticker(benchmark_ticker, start_date, end_date)
            except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, OSError, ValueError) as exc:
                logger.warning("Benchmark %s unavailable: %s", benchmark_ticker, exc)
                primary = {'available': False, 'reason': f'benchmark_load_failed: {exc}'}
            else:
                if benchmark_data.empty or 'Close' not in benchmark_data.columns:
                    primary = {'available': False, 'reason': 'benchmark_missing_close'}
                else:
                    benchmark_close = benchmark_data['Close'].dropna()
                    primary = self._build_benchmark_payload(
                        equity_curve=equity_curve,
                        benchmark_close=benchmark_close,
                        benchmark_label=benchmark_label,
                    )
        else:
            primary = self._build_benchmark_payload(
                equity_curve=equity_curve,
                benchmark_close=benchmark_close,
                benchmark_label=benchmark_label,
            )

        spy_analysis = {'available': False, 'reason': 'spy_unavailable'}
        if str(benchmark_label).upper() == 'SPY' and primary.get('available', False):
            spy_analysis = dict(primary)
        else:
            try:
                spy_data = self.data_manager.load_ticker('SPY', start_date, end_date)
                if not spy_data.empty and 'Close' in spy_data.columns:
                    spy_analysis = self._build_benchmark_payload(
                        equity_curve=equity_curve,
                        benchmark_close=spy_data['Close'].dropna(),
                        benchmark_label='SPY',
                    )
                else:
                    spy_analysis = {'available': False, 'reason': 'spy_missing_close'}
            except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, OSError, ValueError) as exc:
                spy_analysis = {'available': False, 'reason': f'spy_load_failed: {exc}'}

        primary['spy_analysis'] = spy_analysis
        return primary

    def _build_benchmark_payload(self,
                                 equity_curve: pd.Series,
                                 benchmark_close: pd.Series,
                                 benchmark_label: str) -> Dict[str, Any]:
        """Build benchmark equity/metrics payload for a specific close series."""
        aligned = pd.concat([equity_curve, benchmark_close], axis=1, join='inner').dropna()
        if len(aligned) < 2:
            return {'available': False, 'reason': 'insufficient_overlap'}

        aligned.columns = ['strategy_equity', 'benchmark_close']
        initial_equity = float(aligned['strategy_equity'].iloc[0])
        benchmark_norm = aligned['benchmark_close'] / aligned['benchmark_close'].iloc[0]
        benchmark_equity = benchmark_norm * initial_equity

        strategy_returns = aligned['strategy_equity'].pct_change().dropna()
        benchmark_returns = benchmark_equity.pct_change().dropna()

        benchmark_metrics = PerformanceMetrics.get_comprehensive_metrics(
            benchmark_equity,
            trades=[],
            risk_free_rate=0.02,
            base_currency=self.portfolio.base_currency,
        )
        relative_stats = ValidationSuite.benchmark_comparison(strategy_returns, benchmark_returns)

        return {
            'available': True,
            'ticker': benchmark_label,
            'equity_curve': benchmark_equity,
            'metrics': benchmark_metrics,
            'relative_stats': relative_stats,
        }

    def _calculate_degradation_analysis(self, equity_curve: pd.Series) -> Dict:
        """Baseline train-vs-OOS degradation metrics hook."""
        returns = equity_curve.pct_change().dropna()
        if len(returns) < 20:
            return {'available': False, 'reason': 'insufficient_history'}

        split_ratio = 0.7
        train_returns, oos_returns = ValidationSuite.train_oos_split(returns, split_ratio=split_ratio)
        if len(train_returns) < 2 or len(oos_returns) < 2:
            return {'available': False, 'reason': 'insufficient_train_oos_samples'}

        degradation = ValidationSuite.degradation_report(train_returns, oos_returns)

        return {
            'available': True,
            'split_ratio': split_ratio,
            **degradation,
        }

    def _run_transaction_cost_sensitivity(self, tickers: List[str], start_date: str, end_date: str) -> Dict:
        """Stress-test cost assumptions with simple multipliers."""
        base_costs = self.config.get('costs', {})
        base_commission = base_costs.get('commission_per_share', 0.0)
        base_slippage = base_costs.get('slippage_bps', {}).get('medium_liquidity', 0.0)

        scenarios = {
            'base': 1.0,
            'costs_x0_5': 0.5,
            'costs_x1_5': 1.5,
            'costs_x2_0': 2.0,
        }

        baseline_return = PerformanceMetrics.calculate_total_return(self.portfolio.get_equity_curve())
        initial_capital = self.config['portfolio'].get('initial_capital', 0)
        observed_total_cost = sum(t.total_cost() for t in self.portfolio.trades)

        output = {}
        for name, mult in scenarios.items():
            est_cost = observed_total_cost * mult
            output[name] = {
                'commission_per_share': base_commission * mult,
                'slippage_bps_medium': base_slippage * mult,
                'estimated_total_cost': est_cost,
                'estimated_total_return_net': baseline_return - (est_cost / initial_capital) if initial_capital else baseline_return,
                'tickers': tickers,
                'start_date': start_date,
                'end_date': end_date,
            }

        return output

    def _get_buy_total_cost_base(self, trade, date: datetime = None) -> float:
        """Convert BUY trade value and fees to base currency total cost."""
        if trade.currency != self.portfolio.base_currency:
            trade_value_base = self.fx_converter.convert_to_base(trade.value, trade.currency, date=date)
            fees_base = self.fx_converter.convert_to_base(
                trade.commission + trade.slippage,
                trade.currency,
                date=date
            )
        else:
            trade_value_base = trade.value
            fees_base = trade.commission + trade.slippage

        return trade_value_base + fees_base + trade.fx_cost

    def _resize_buy_trade_to_cash(self, trade, price: float, avg_volume: float,
                                  current_shares: float, available_cash: float):
        """Find the largest affordable BUY trade size (whole shares)."""
        max_shares = int(trade.shares)
        if max_shares <= 0:
            return None

        low, high = 1, max_shares
        best_trade = None

        while low <= high:
            mid = (low + high) // 2
            candidate = self.execution_engine.execute_order(
                ticker=trade.ticker,
                target_shares=current_shares + mid,
                current_shares=current_shares,
                price=price,
                avg_volume=avg_volume,
                date=trade.date,
                currency=trade.currency,
                run_id=getattr(trade, 'run_id', None),
                signal_date=getattr(trade, 'signal_date', None),
                decision=getattr(trade, 'decision', None),
                decision_reason=getattr(trade, 'decision_reason', None),
                signal_strength=getattr(trade, 'signal_strength', None),
                signal_confidence=getattr(trade, 'signal_confidence', None),
                weight_delta=getattr(trade, 'weight_delta', None),
                current_weight=getattr(trade, 'current_weight', None),
                target_weight=getattr(trade, 'target_weight', None),
            )

            if candidate is None:
                break

            candidate_cost = self._get_buy_total_cost_base(candidate, date=trade.date)

            if candidate_cost <= available_cash:
                best_trade = candidate
                low = mid + 1
            else:
                high = mid - 1

        if best_trade is None:
            logger.warning(
                f"Skipping BUY {trade.ticker}: no affordable size with cash {available_cash:.2f}"
            )
            return None

        logger.info(
            f"Resized BUY {trade.ticker} from {trade.shares} to {best_trade.shares} shares "
            f"to fit available cash {available_cash:.2f} {self.portfolio.base_currency}"
        )
        return best_trade


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries and return a new merged dictionary."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_backtest_from_config(config_path: str = 'config/config.yaml',
                             strategy_name: str = 'momentum',
                             tickers: List[str] = None,
                             start_date: str = '2020-01-01',
                             end_date: str = None,
                             strategies_config_path: str = 'config/strategies.yaml',
                             config_override: Optional[Dict[str, Any]] = None,
                             strategy_override: Optional[Dict[str, Any]] = None) -> Dict:
    """
    Convenience function to run backtest from config files
    
    Args:
        config_path: Path to main config file
        strategy_name: Name of strategy in strategies.yaml
        tickers: List of tickers (defaults to config)
        start_date: Start date
        end_date: End date (defaults to today)
        strategies_config_path: Path to strategy configs
        config_override: Optional nested override for main config
        strategy_override: Optional nested override for selected strategy config
        
    Returns:
        Backtest results dict
    """
    # Load configs
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    if config_override:
        config = _deep_merge_dict(config, config_override)

    try:
        validate_main_config(config)
    except (ValidationError, ValueError) as e:
        logger.error("Config validation failed:\n%s", e)
        raise SystemExit(1) from e

    with open(strategies_config_path, 'r') as f:
        strategies_config = yaml.safe_load(f) or {}

    try:
        validate_strategies_config(strategies_config)
    except (ValidationError, TypeError) as e:
        logger.error("Strategy config validation failed: %s", e)
        raise SystemExit(1) from e
    
    available_strategies = get_available_strategies()
    unknown_in_config = sorted(set(strategies_config.keys()) - set(available_strategies))
    if unknown_in_config:
        logger.warning(
            "Ignoring unsupported entries in config/strategies.yaml: %s",
            ', '.join(unknown_in_config),
        )

    if strategy_name not in available_strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Supported strategies: {', '.join(available_strategies)}"
        )

    # Get strategy config
    if strategy_name not in strategies_config:
        raise ValueError(
            f"Strategy {strategy_name} is supported but not found in config/strategies.yaml. "
            f"Configured strategies: {', '.join(sorted(strategies_config.keys()))}"
        )

    strategy_config = dict(strategies_config[strategy_name] or {})
    if strategy_override:
        strategy_config = _deep_merge_dict(strategy_config, strategy_override)
    if strategy_name == 'strategy_orchestration':
        strategy_config['_all_strategies_config'] = strategies_config
    strategy = create_strategy(strategy_name, strategy_config)
    
    # Get tickers
    if tickers is None:
        tickers = config['data'].get('tickers', ['SPY', 'QQQ'])
    
    # Set end date
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize data manager
    data_manager = DataManager(
        data_dir=config['data']['data_dir'],
        use_adjusted_close=bool(config.get('data', {}).get('use_adjusted_close', True)),
        freshness_threshold_days=int(config.get('data', {}).get('freshness_threshold_days', 3)),
    )
    
    # Create and run backtest
    engine = BacktestEngine(strategy, data_manager, config)
    results = engine.run(tickers, start_date, end_date)
    
    return results
