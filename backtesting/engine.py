import os
from datetime import datetime, timedelta
from numbers import Real
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError

from backtesting._engine.analysis import BacktestAnalysisMixin
from backtesting._engine.data import AssetExclusionRecord, BacktestDataMixin
from backtesting._engine.positioning import BacktestPositioningMixin
from backtesting.execution import ExecutionEngine
from backtesting.metrics import PerformanceMetrics
from backtesting.portfolio import DataIntegrityError, Portfolio
from backtesting.tca import TransactionCostAnalysis
from data.data_manager import DataManager
from risk.optimizer import PortfolioOptimizer
from risk.position_sizer import PositionSizer, RiskConstraints
from strategies import create_strategy, get_available_strategies
from strategies.base_strategy import BaseStrategy
from utils.config_schema import validate_main_config, validate_strategies_config
from utils.currency import CurrencyConverter
from utils.data_platform import AuditStore, create_metrics_store
from utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine(BacktestAnalysisMixin, BacktestPositioningMixin, BacktestDataMixin):
    """Main backtesting engine that orchestrates the simulation."""

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

        data_platform_cfg = self.config.get('data_platform', {})
        self.audit_store = None
        self.metrics_store = None
        if data_platform_cfg.get('enabled', False):
            self.audit_store = AuditStore(data_platform_cfg.get('sqlite_path', 'state/trading_audit.db'))
            self.metrics_store = create_metrics_store(data_platform_cfg.get('metrics_store', {}))
            logger.info("Data platform persistence enabled")

        risk_config = self.config.get('risk', {})

        sizing_method = risk_config.get('position_sizing_method', 'equal')
        self.position_sizer = PositionSizer(
            method=sizing_method,
            config=risk_config
        )
        self.risk_constraints = RiskConstraints(risk_config)

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
        Run backtest simulation.

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
            except FileNotFoundError as exc:
                self._record_asset_exclusion(
                    ticker=ticker,
                    date=pd.Timestamp(start_date),
                    reason='missing_file',
                    available_bars=0,
                    required_bars=required_history_days,
                    detail=str(exc),
                )
                logger.warning("Missing data file for %s: %s", ticker, exc)
            except (pd.errors.EmptyDataError, OSError) as exc:
                self._record_asset_exclusion(
                    ticker=ticker,
                    date=pd.Timestamp(start_date),
                    reason='data_error',
                    available_bars=0,
                    required_bars=required_history_days,
                    detail=str(exc),
                )
                logger.warning("Failed to load %s due to recoverable data error: %s", ticker, exc)
            except Exception as exc:
                raise RuntimeError(f"Unexpected error loading {ticker}") from exc

        if not data:
            raise ValueError("No data loaded for any ticker")

        logger.info("Loaded data for %s tickers", len(data))

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

        currencies = {ticker: self.fx_converter.get_ticker_currency(ticker) for ticker in tickers}

        base_ccy = self.config['portfolio']['currency']
        unique_foreign_currencies = sorted({ccy for ccy in currencies.values() if ccy != base_ccy})
        if unique_foreign_currencies:
            preload_start = (effective_start_date - timedelta(days=30)).strftime('%Y-%m-%d')
            for foreign_ccy in unique_foreign_currencies:
                self.fx_converter.preload_pair(foreign_ccy, base_ccy, preload_start, end_date)

        if initial_positions:
            self._initialize_starting_positions(
                initial_positions=initial_positions,
                first_date=trading_dates[0],
                data=data,
                currencies=currencies,
            )

        logger.info("Running backtest simulation...")
        print(f"Running backtest simulation over {len(trading_dates)} trading days...", flush=True)
        for i, date in enumerate(trading_dates):
            try:
                signal_date = trading_dates[i - 1] if i > 0 else None
                self._simulate_day(date, data, currencies, signal_date=signal_date)

                if self.progress_log_interval > 0 and (i + 1) % self.progress_log_interval == 0:
                    pct = ((i + 1) / len(trading_dates)) * 100
                    print(
                        f"Progress: {i + 1}/{len(trading_dates)} days ({pct:.1f}%) - {date.date()}",
                        flush=True,
                    )
                    logger.info("Processed %s/%s days", i + 1, len(trading_dates))

            except DataIntegrityError as exc:
                logger.error("Critical data integrity error on %s: %s", date, exc)
                raise
            except (ValueError, KeyError) as exc:
                logger.error("Recoverable day-level error on %s: %s", date, exc)
                continue

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
        Simulate a single trading day.

        Args:
            date: Current date
            data: Dict of {ticker: DataFrame}
            currencies: Dict of {ticker: currency}
            signal_date: Date used for signal generation (must be prior to execution date).
                When None, the engine has no prior in-window bar to form a
                lookahead-free signal from, so the day is valuation-only.
        """
        if signal_date is None:
            close_prices = self._get_prices_for_date(data, date, field='Close')
            equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
            self._record_equity_and_notify(date, equity)
            return

        if not self._should_rebalance_on_date(date):
            close_prices = self._get_prices_for_date(data, date, field='Close')
            equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
            self._record_equity_and_notify(date, equity)
            return

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

        execution_prices = self._get_prices_for_date(data, date, field='Open')
        close_prices = self._get_prices_for_date(data, date, field='Close')
        volumes = self._get_avg_volumes(historical_data)

        equity = self.portfolio.get_total_equity(execution_prices, currencies, date=date)
        current_weights = self.portfolio.get_weights(execution_prices, currencies, date=date)

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
        except (ValueError, KeyError) as exc:
            logger.error("Recoverable strategy error on %s: %s", date, exc)
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

        if target_weights:
            target_weights = self.position_sizer.size_positions(
                target_weights,
                historical_data,
                equity
            )

        if self.use_optimizer and self.optimizer and target_weights:
            try:
                optimizer_method = self.config['risk'].get('optimizer_method', 'max_sharpe')
                target_weights = self.optimizer.optimize(
                    target_weights,
                    historical_data,
                    method=optimizer_method
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                logger.warning("Optimization failed on %s: %s", date, exc)

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

        target_positions = self._calculate_target_positions(
            target_weights, equity, execution_prices, currencies, date=date
        )
        current_positions = dict(self.portfolio.positions)

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

        trades = sorted(trades, key=lambda t: 0 if t.action == 'SELL' else 1)
        executed_trades = []

        for trade in trades:
            if trade.action == 'BUY':
                if not self._has_sufficient_forward_horizon(trade.ticker, date, data):
                    continue
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

        final_equity = self.portfolio.get_total_equity(close_prices, currencies, date=date)
        self._record_equity_and_notify(date, final_equity)


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
    Convenience function to run backtest from config files.

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
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle) or {}

    if config_override:
        config = _deep_merge_dict(config, config_override)

    try:
        validate_main_config(config)
    except (ValidationError, ValueError) as exc:
        logger.error("Config validation failed:\n%s", exc)
        raise SystemExit(1) from exc

    with open(strategies_config_path, 'r') as handle:
        strategies_config = yaml.safe_load(handle) or {}

    try:
        validate_strategies_config(strategies_config)
    except (ValidationError, TypeError) as exc:
        logger.error("Strategy config validation failed: %s", exc)
        raise SystemExit(1) from exc

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

    if tickers is None:
        tickers = config['data'].get('tickers', ['SPY', 'QQQ'])

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    data_manager = DataManager(
        data_dir=config['data']['data_dir'],
        use_adjusted_close=bool(config.get('data', {}).get('use_adjusted_close', True)),
        freshness_threshold_days=int(config.get('data', {}).get('freshness_threshold_days', 3)),
    )

    engine = BacktestEngine(strategy, data_manager, config)
    return engine.run(tickers, start_date, end_date)


__all__ = ['AssetExclusionRecord', 'BacktestEngine', 'run_backtest_from_config']
