from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.metrics import PerformanceMetrics
from backtesting.validation import ValidationSuite
from risk.advanced_risk import AdvancedRiskAnalytics
from utils.logger import get_logger

logger = get_logger("backtesting.engine")

try:
    from risk.hmm_regime import HMMRegimeDetector
except ImportError:
    HMMRegimeDetector = None


class BacktestAnalysisMixin:
    def _build_regime_series_pair(self, equity_curve: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Build retrospective and point-in-time regime labels."""
        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            empty = pd.Series(dtype='object')
            return empty, empty

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

        weights_by_strategy: Dict[str, List[float]] = {}
        contribution_by_strategy: Dict[str, List[float]] = {}

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
        """Calculate benchmark performance and relative stats."""
        if len(equity_curve) < 2:
            return {
                'available': False,
                'reason': 'insufficient_equity_history',
                'spy_analysis': {'available': False, 'reason': 'insufficient_equity_history'},
            }

        benchmark_close = None
        benchmark_label = benchmark_ticker

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
                    normalized = close_frame.div(close_frame.iloc[0])
                    benchmark_close = normalized.mean(axis=1)
                    benchmark_label = f"EqualWeight({','.join(close_frame.columns)})"

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
        """Stress-test observed transaction costs using base-currency totals."""
        scenarios = {
            'base': 1.0,
            'costs_x0_5': 0.5,
            'costs_x1_5': 1.5,
            'costs_x2_0': 2.0,
        }

        baseline_return = PerformanceMetrics.calculate_total_return(self.portfolio.get_equity_curve())
        initial_capital = self.config['portfolio'].get('initial_capital', 0)
        base_currency = self.portfolio.base_currency
        observed_commission_cost_base = PerformanceMetrics._sum_trade_component_in_base(
            self.portfolio.trades,
            'commission',
            self.fx_converter,
        )
        observed_slippage_cost_base = PerformanceMetrics._sum_trade_component_in_base(
            self.portfolio.trades,
            'slippage',
            self.fx_converter,
        )
        observed_fx_cost_base = PerformanceMetrics._sum_trade_component_in_base(
            self.portfolio.trades,
            'fx_cost',
            self.fx_converter,
        )
        observed_total_cost_base = (
            observed_commission_cost_base +
            observed_slippage_cost_base +
            observed_fx_cost_base
        )

        output = {}
        for name, mult in scenarios.items():
            est_commission_cost_base = observed_commission_cost_base * mult
            est_slippage_cost_base = observed_slippage_cost_base * mult
            est_fx_cost_base = observed_fx_cost_base * mult
            est_cost_base = observed_total_cost_base * mult
            output[name] = {
                'base_currency': base_currency,
                'cost_multiplier': mult,
                'estimated_commission_cost_base': est_commission_cost_base,
                'estimated_slippage_cost_base': est_slippage_cost_base,
                'estimated_fx_cost_base': est_fx_cost_base,
                'estimated_total_cost_base': est_cost_base,
                'estimated_total_cost': est_cost_base,
                'estimated_total_return_net': baseline_return - (est_cost_base / initial_capital) if initial_capital else baseline_return,
                'tickers': tickers,
                'start_date': start_date,
                'end_date': end_date,
            }

        return output
