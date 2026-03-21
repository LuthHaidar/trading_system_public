import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from backtesting.portfolio import Trade
from backtesting.trade_accounting import FIFOTradeMatcher
from utils.currency import CurrencyConverter
from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for backtesting
    """
    
    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        """Calculate total return over period"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    
    @staticmethod
    def calculate_cagr(equity_curve: pd.Series) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            equity_curve: Series of equity values with dates
            
        Returns:
            Annualized return
        """
        if len(equity_curve) < 2:
            return 0.0
        
        total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if total_days == 0:
            return 0.0
        
        n_years = total_days / 365.25
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        
        if total_return <= 0:
            logger.warning(
                "CAGR undefined for non-positive terminal equity ratio (final=%.4f, initial=%.4f, ratio=%.6f); returning NaN",
                float(equity_curve.iloc[-1]),
                float(equity_curve.iloc[0]),
                float(total_return),
            )
            return float('nan')
        
        cagr = (total_return ** (1/n_years)) - 1.0
        return cagr
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility of returns
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize (multiply by sqrt(252))
            
        Returns:
            Volatility
        """
        if len(returns) < 2:
            return 0.0
        
        vol = returns.std()
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (annualized)
        
        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Convert annual risk-free rate to daily
        daily_rf = risk_free_rate / 252
        
        # Calculate excess returns
        excess_returns = returns - daily_rf
        
        # Sharpe = mean(excess) / std(returns) * sqrt(252)
        std = excess_returns.std()
        if np.isclose(std, 0.0):
            return 0.0

        sharpe = np.sqrt(252) * excess_returns.mean() / std
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (uses downside deviation)
        
        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        # Downside deviation over the full sample:
        # negative excess returns contribute squared downside, non-negative contribute 0.
        downside_dev = np.sqrt(np.mean(np.minimum(excess_returns, 0.0) ** 2))
        if np.isclose(downside_dev, 0.0):
            return 0.0

        sortino = np.sqrt(252) * excess_returns.mean() / downside_dev
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Maximum drawdown (negative value)
        """
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max
        
        # Return maximum drawdown (most negative)
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (CAGR / abs(MaxDD))
        
        Args:
            cagr: Compound annual growth rate
            max_drawdown: Maximum drawdown (negative)
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0
        
        return cagr / abs(max_drawdown)
    
    @staticmethod
    def _sum_trade_component_in_base(trades: List[Trade], component: str,
                                     fx_converter: CurrencyConverter) -> float:
        """
        Sum trade cost component in base currency.

        commission/slippage are stored in trade currency; fx_cost is already in base.
        """
        total = 0.0
        for trade in trades:
            value = float(getattr(trade, component, 0.0))
            if component == 'fx_cost' or trade.currency == fx_converter.base_currency:
                total += value
            else:
                total += float(
                    fx_converter.convert_to_base(value, trade.currency, date=trade.date)
                )
        return float(total)

    @staticmethod
    def _sum_trade_total_cost_in_base(trades: List[Trade],
                                      fx_converter: CurrencyConverter) -> float:
        """Sum total transaction costs in base currency."""
        total = 0.0
        for trade in trades:
            if trade.currency == fx_converter.base_currency:
                total += float(trade.total_cost())
                continue

            trading_costs = float(trade.commission + trade.slippage)
            trading_costs_base = fx_converter.convert_to_base(
                trading_costs, trade.currency, date=trade.date
            )
            total += float(trading_costs_base + trade.fx_cost)
        return float(total)
    

    @staticmethod
    def _resample_last_with_fallback(series: pd.Series, primary_freq: str, fallback_freq) -> pd.Series:
        """Resample with a compatibility fallback for pandas frequency aliases."""
        try:
            return series.resample(primary_freq).last()
        except ValueError as exc:
            if 'Invalid frequency' not in str(exc):
                raise
            return series.resample(fallback_freq).last()

    @staticmethod
    def calculate_monthly_returns(equity_curve: pd.Series) -> pd.Series:
        """
        Calculate monthly returns
        
        Args:
            equity_curve: Daily equity curve
            
        Returns:
            Series of monthly returns
        """
        # Resample to month end
        monthly = PerformanceMetrics._resample_last_with_fallback(equity_curve, 'ME', pd.offsets.MonthEnd(1))
        return monthly.pct_change()
    
    @staticmethod
    def calculate_annual_returns(equity_curve: pd.Series) -> pd.Series:
        """
        Calculate annual returns
        
        Args:
            equity_curve: Daily equity curve
            
        Returns:
            Series of annual returns
        """
        # Resample to year end
        annual = PerformanceMetrics._resample_last_with_fallback(equity_curve, 'YE', pd.offsets.YearEnd(1))
        return annual.pct_change()
    
    @staticmethod
    def calculate_rolling_sharpe(returns: pd.Series, window: int = 252,
                                 risk_free_rate: float = 0.02) -> pd.Series:
        """
        Calculate rolling Sharpe ratio
        
        Args:
            returns: Daily returns
            window: Rolling window size (default: 252 = 1 year)
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Series of rolling Sharpe ratios
        """
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        
        rolling_sharpe = (np.sqrt(252) * rolling_mean / rolling_std).replace([np.inf, -np.inf], np.nan)
        return rolling_sharpe
    
    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """
        Calculate drawdown at each point in time
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Series of drawdown percentages
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown

    @staticmethod
    def drawdown_events(equity_curve: pd.Series, min_drawdown: float = 0.02) -> List[Dict[str, Any]]:
        """Extract contiguous drawdown events with start/trough/recovery timestamps."""
        if equity_curve is None or len(equity_curve) < 2:
            return []

        ath = equity_curve.cummax()
        in_drawdown = (equity_curve < ath).astype(int)
        changes = in_drawdown.diff().fillna(0)

        starts = list(equity_curve.index[changes == 1])
        ends = list(equity_curve.index[changes == -1])
        if in_drawdown.iloc[0] == 1:
            starts = [equity_curve.index[0]] + starts
        if len(ends) < len(starts):
            ends.append(equity_curve.index[-1])

        events: List[Dict[str, Any]] = []
        for start_date, end_date in zip(starts, ends):
            window = equity_curve.loc[start_date:end_date]
            if window.empty:
                continue
            trough_date = window.idxmin()
            trough_value = float(window.min())
            ath_at_start = float(ath.loc[start_date])
            if ath_at_start <= 0:
                continue
            drawdown_pct = (trough_value - ath_at_start) / ath_at_start
            if abs(drawdown_pct) < float(min_drawdown):
                continue

            recovery_date = None
            post = equity_curve.loc[end_date:]
            recovered = post[post >= ath_at_start]
            if not recovered.empty:
                recovery_date = recovered.index[0]

            duration_days = None
            if recovery_date is not None:
                duration_days = int((recovery_date - start_date).days)

            events.append({
                'start_date': start_date,
                'trough_date': trough_date,
                'trough_value': trough_value,
                'recovery_date': recovery_date,
                'drawdown_pct': float(drawdown_pct),
                'duration_days': duration_days,
            })

        events.sort(key=lambda e: e['start_date'])
        return events

    @staticmethod
    def regime_performance(equity_curve: pd.Series, regime_series: pd.Series) -> Dict[str, Dict[str, float]]:
        """Compute per-regime metrics for bull/neutral/bear labels."""
        if equity_curve is None or regime_series is None or len(equity_curve) < 2:
            return {}

        aligned = pd.concat([equity_curve, regime_series], axis=1, join='inner').dropna()
        if aligned.empty:
            return {}
        aligned.columns = ['equity', 'regime']
        total_days = len(aligned)

        payload: Dict[str, Dict[str, float]] = {}
        for regime in ['bull', 'neutral', 'bear']:
            sub = aligned[aligned['regime'] == regime]
            if sub.empty:
                continue
            sub_equity = sub['equity']
            sub_returns = sub_equity.pct_change().dropna()
            payload[regime] = {
                'cagr': float(PerformanceMetrics.calculate_cagr(sub_equity)),
                'volatility': float(PerformanceMetrics.calculate_volatility(sub_returns)),
                'sharpe': float(PerformanceMetrics.calculate_sharpe_ratio(sub_returns, risk_free_rate=0.0)),
                'max_drawdown': float(PerformanceMetrics.calculate_max_drawdown(sub_equity)),
                'days': int(len(sub)),
                'fraction': float(len(sub) / total_days),
            }

        return payload

    @staticmethod
    def calculate_drawdown_durations(drawdown_series: pd.Series) -> Dict[str, float]:
        """Calculate max/average drawdown duration in days."""
        episodes = PerformanceMetrics._extract_drawdown_episodes(drawdown_series)
        durations = [int(ep['start_to_recovery_days']) for ep in episodes]
        if not durations:
            return {
                'max_drawdown_duration_days': 0,
                'avg_drawdown_duration_days': 0.0,
            }
        return {
            'max_drawdown_duration_days': int(max(durations)),
            'avg_drawdown_duration_days': float(np.mean(durations)),
        }

    @staticmethod
    def _extract_drawdown_episodes(drawdown_series: pd.Series) -> List[Dict[str, Any]]:
        """Extract drawdown episodes with start/trough/recovery timing."""
        if drawdown_series.empty:
            return []

        episodes: List[Dict[str, Any]] = []
        in_drawdown = False
        start_date = None
        trough_date = None
        trough_drawdown = 0.0

        for dt, dd in drawdown_series.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = dt
                trough_date = dt
                trough_drawdown = float(dd)
            elif dd < 0 and in_drawdown:
                if float(dd) < trough_drawdown:
                    trough_drawdown = float(dd)
                    trough_date = dt
            elif dd >= 0 and in_drawdown:
                episodes.append({
                    'start_date': start_date,
                    'trough_date': trough_date,
                    'recovery_date': dt,
                    'recovered': True,
                    'min_drawdown': float(trough_drawdown),
                    'start_to_recovery_days': int((dt - start_date).days),
                    'trough_to_recovery_days': int((dt - trough_date).days),
                })
                in_drawdown = False
                start_date = None
                trough_date = None
                trough_drawdown = 0.0

        if in_drawdown and start_date is not None:
            terminal = drawdown_series.index[-1]
            episodes.append({
                'start_date': start_date,
                'trough_date': trough_date,
                'recovery_date': None,
                'recovered': False,
                'min_drawdown': float(trough_drawdown),
                'start_to_recovery_days': int((terminal - start_date).days),
                'trough_to_recovery_days': int((terminal - trough_date).days) if trough_date is not None else 0,
            })

        return episodes

    @staticmethod
    def _calculate_trade_derived_metrics(matched_trades, trades: List[Trade], equity_curve: pd.Series,
                                         fx_converter: CurrencyConverter) -> Dict[str, Any]:
        """Compute detailed trade and exposure/turnover metrics."""
        ticker_pnl = {}
        trade_count_by_ticker = {}
        closed_trade_count_by_ticker = {}
        win_count_by_ticker = {}
        holding_days_by_ticker = {}
        hit_rate_by_holding_bucket = {
            '0_1d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
            '2_5d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
            '6_20d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
            '21+d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
        }

        if matched_trades:
            pnls = [t.realized_pnl_base for t in matched_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            reward_to_risk = (avg_win / abs(avg_loss)) if avg_loss < 0 else (float('inf') if avg_win > 0 else 0.0)

            holding_days = [(t.exit_date - t.entry_date).days for t in matched_trades]
            avg_holding_period = float(np.mean(holding_days)) if holding_days else 0.0
            median_holding_period = float(np.median(holding_days)) if holding_days else 0.0
            expectancy = float(np.mean(pnls)) if pnls else 0.0
            largest_loss = float(min(pnls)) if pnls else 0.0

            pnl_series = pd.Series(pnls)
            payoff_skewness = float(pnl_series.skew()) if len(pnl_series) > 2 else 0.0
            payoff_kurtosis = float(pnl_series.kurtosis()) if len(pnl_series) > 3 else 0.0

            # True MAE/MFE require intratrade mark-to-market path data.
            avg_mae = None
            avg_mfe = None

            consecutive_losses = 0
            max_consecutive_losses = 0
            for pnl in pnls:
                if pnl < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0

            for mt in matched_trades:
                ticker_pnl[mt.ticker] = ticker_pnl.get(mt.ticker, 0.0) + mt.realized_pnl_base
                closed_trade_count_by_ticker[mt.ticker] = closed_trade_count_by_ticker.get(mt.ticker, 0) + 1

                hold = (mt.exit_date - mt.entry_date).days
                holding_days_by_ticker.setdefault(mt.ticker, []).append(int(hold))
                if mt.realized_pnl_base > 0:
                    win_count_by_ticker[mt.ticker] = win_count_by_ticker.get(mt.ticker, 0) + 1
                if hold <= 1:
                    bucket = '0_1d'
                elif hold <= 5:
                    bucket = '2_5d'
                elif hold <= 20:
                    bucket = '6_20d'
                else:
                    bucket = '21+d'

                hit_rate_by_holding_bucket[bucket]['total'] += 1
                if mt.realized_pnl_base > 0:
                    hit_rate_by_holding_bucket[bucket]['wins'] += 1

            for bucket in hit_rate_by_holding_bucket.values():
                total = bucket['total']
                bucket['hit_rate'] = (bucket['wins'] / total) if total > 0 else 0.0
        else:
            avg_win = 0.0
            avg_loss = 0.0
            reward_to_risk = 0.0
            avg_holding_period = 0.0
            median_holding_period = 0.0
            expectancy = 0.0
            largest_loss = 0.0
            payoff_skewness = 0.0
            payoff_kurtosis = 0.0
            avg_mae = None
            avg_mfe = None
            max_consecutive_losses = 0

        for t in trades:
            trade_count_by_ticker[t.ticker] = trade_count_by_ticker.get(t.ticker, 0) + 1

        if len(equity_curve) > 0:
            avg_equity = float(equity_curve.mean())
            num_days = len(equity_curve)
        else:
            avg_equity = 0.0
            num_days = 0

        turnover_ratio = 0.0
        annualized_turnover = 0.0
        avg_exposure = 0.0
        max_exposure = 0.0
        min_exposure = 0.0
        exposure_concentration_hhi = 0.0

        if trades and avg_equity > 0 and num_days > 0:
            sorted_trades = sorted(trades, key=lambda t: t.date)
            trade_values_base = []
            exposure_events = {}
            ticker_exposure_events = {}

            for t in sorted_trades:
                value_base = fx_converter.convert_to_base(t.value, t.currency, date=t.date)
                trade_values_base.append(abs(value_base))
                signed = value_base if t.action == 'BUY' else -value_base
                exposure_events.setdefault(t.date, 0.0)
                exposure_events[t.date] += signed

                if t.ticker not in ticker_exposure_events:
                    ticker_exposure_events[t.ticker] = {}
                ticker_exposure_events[t.ticker].setdefault(t.date, 0.0)
                ticker_exposure_events[t.ticker][t.date] += signed

            gross_traded_value = sum(trade_values_base)
            turnover_ratio = gross_traded_value / avg_equity
            annualized_turnover = turnover_ratio * (252 / num_days)

            # Approximate daily capital exposure from cumulative net invested notional
            event_series = pd.Series(exposure_events).sort_index().cumsum()
            exposure_notional = event_series.reindex(equity_curve.index).ffill().fillna(0.0).abs()
            safe_equity = equity_curve.replace(0, np.nan)
            exposure_ratio = (exposure_notional / safe_equity).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            avg_exposure = float(exposure_ratio.mean())
            max_exposure = float(exposure_ratio.max()) if not exposure_ratio.empty else 0.0
            min_exposure = float(exposure_ratio.min()) if not exposure_ratio.empty else 0.0

            # Concentration using HHI of per-ticker absolute exposure weights.
            ticker_abs_exposure = []
            for ticker, events in ticker_exposure_events.items():
                t_series = pd.Series(events).sort_index().cumsum()
                ticker_abs_exposure.append(t_series.reindex(equity_curve.index).ffill().fillna(0.0).abs().rename(ticker))
            if ticker_abs_exposure:
                ticker_exposure_df = pd.concat(ticker_abs_exposure, axis=1)
                gross_abs = ticker_exposure_df.sum(axis=1)
                gross_abs_safe = gross_abs.replace(0, np.nan)
                gross_exposure_ratio = (
                    gross_abs_safe / safe_equity
                ).replace([np.inf, -np.inf], np.nan)
                valid_exposure_mask = gross_exposure_ratio >= 0.01
                weights = ticker_exposure_df.div(gross_abs_safe, axis=0).fillna(0.0)
                hhi_series = (weights ** 2).sum(axis=1)
                hhi_series = hhi_series[valid_exposure_mask].dropna()
                exposure_concentration_hhi = float(hhi_series.mean()) if not hhi_series.empty else 0.0

        pnl_contribution_by_ticker = {}
        total_realized = sum(ticker_pnl.values())
        if total_realized != 0:
            pnl_contribution_by_ticker = {
                ticker: pnl / total_realized for ticker, pnl in ticker_pnl.items()
            }

        win_rate_by_ticker = {}
        for ticker, total_closed in closed_trade_count_by_ticker.items():
            wins = win_count_by_ticker.get(ticker, 0)
            win_rate_by_ticker[ticker] = (wins / total_closed) if total_closed > 0 else 0.0

        avg_holding_period_by_ticker = {}
        for ticker, holding_days in holding_days_by_ticker.items():
            avg_holding_period_by_ticker[ticker] = float(np.mean(holding_days)) if holding_days else 0.0

        return {
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'reward_to_risk_ratio': reward_to_risk,
            'expectancy_per_trade': expectancy,
            'avg_holding_period_days': avg_holding_period,
            'median_holding_period_days': median_holding_period,
            'turnover_ratio': turnover_ratio,
            'annualized_turnover_ratio': annualized_turnover,
            'avg_exposure': avg_exposure,
            'max_exposure': max_exposure,
            'min_exposure': min_exposure,
            'largest_loss': largest_loss,
            'payoff_skewness': payoff_skewness,
            'payoff_kurtosis': payoff_kurtosis,
            'avg_mae_per_trade': avg_mae,
            'avg_mfe_per_trade': avg_mfe,
            'mae_mfe_available': False,
            'mae_mfe_reason': 'intratrade_path_unavailable',
            'max_consecutive_losses': max_consecutive_losses,
            'trade_count_by_ticker': trade_count_by_ticker,
            'pnl_contribution_by_ticker': pnl_contribution_by_ticker,
            'win_rate_by_ticker': win_rate_by_ticker,
            'avg_holding_period_by_ticker': avg_holding_period_by_ticker,
            'hit_rate_by_holding_bucket': hit_rate_by_holding_bucket,
            'exposure_concentration_hhi': exposure_concentration_hhi,
        }    

    @staticmethod
    def calculate_time_to_recovery(equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate recovery timing, anchored to the max-drawdown episode trough."""
        if equity_curve.empty:
            return {
                'max_time_to_recovery_days': 0,
                'avg_time_to_recovery_days': 0.0,
                'max_drawdown_recovery_days_from_peak': 0,
                'max_drawdown_recovery_days_from_trough': 0,
                'max_drawdown_recovered': False,
            }

        drawdown_series = PerformanceMetrics.calculate_drawdown_series(equity_curve)
        episodes = PerformanceMetrics._extract_drawdown_episodes(drawdown_series)
        if not episodes:
            return {
                'max_time_to_recovery_days': 0,
                'avg_time_to_recovery_days': 0.0,
                'max_drawdown_recovery_days_from_peak': 0,
                'max_drawdown_recovery_days_from_trough': 0,
                'max_drawdown_recovered': True,
            }

        trough_recovery_durations = [int(ep['trough_to_recovery_days']) for ep in episodes]
        max_dd_episode = min(episodes, key=lambda ep: float(ep.get('min_drawdown', 0.0)))
        max_dd_trough_to_recovery = int(max_dd_episode.get('trough_to_recovery_days', 0))
        max_dd_peak_to_recovery = int(max_dd_episode.get('start_to_recovery_days', 0))

        return {
            'max_time_to_recovery_days': max_dd_trough_to_recovery,
            'avg_time_to_recovery_days': float(np.mean(trough_recovery_durations)),
            'max_drawdown_recovery_days_from_peak': max_dd_peak_to_recovery,
            'max_drawdown_recovery_days_from_trough': max_dd_trough_to_recovery,
            'max_drawdown_recovered': bool(max_dd_episode.get('recovered', False)),
        }

    @staticmethod
    def calculate_beta_correlation_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series],
                                           rolling_window: int = 63) -> Dict[str, float]:
        """Calculate static and rolling beta/correlation vs benchmark returns."""
        defaults = {
            'beta': 0.0,
            'correlation': 0.0,
            'rolling_beta': 0.0,
            'rolling_correlation': 0.0,
            'beta_correlation_available': False,
            'beta_correlation_reason': 'benchmark_returns_unavailable',
        }
        if benchmark_returns is None or benchmark_returns.empty or returns.empty:
            return defaults

        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if len(aligned) < 2:
            defaults['beta_correlation_reason'] = 'insufficient_overlap'
            return defaults

        aligned.columns = ['strategy', 'benchmark']
        var_bench = aligned['benchmark'].var()
        beta = float(aligned['strategy'].cov(aligned['benchmark']) / var_bench) if not np.isclose(var_bench, 0.0) else 0.0
        corr = float(aligned['strategy'].corr(aligned['benchmark'])) if len(aligned) > 1 else 0.0

        window = min(rolling_window, len(aligned))
        rolling_cov = aligned['strategy'].rolling(window).cov(aligned['benchmark'])
        rolling_var = aligned['benchmark'].rolling(window).var()
        rolling_beta_series = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan).dropna()
        rolling_corr_series = aligned['strategy'].rolling(window).corr(aligned['benchmark']).dropna()

        return {
            'beta': beta,
            'correlation': corr if not np.isnan(corr) else 0.0,
            'rolling_beta': float(rolling_beta_series.iloc[-1]) if not rolling_beta_series.empty else 0.0,
            'rolling_correlation': float(rolling_corr_series.iloc[-1]) if not rolling_corr_series.empty else 0.0,
            'beta_correlation_available': True,
            'beta_correlation_reason': 'ok',
        }

    @staticmethod
    def get_comprehensive_metrics(equity_curve: pd.Series, trades: List[Trade],
                                  risk_free_rate: float = 0.02,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  base_currency: str = 'SGD') -> Dict:
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: Daily equity curve
            trades: List of executed trades
            risk_free_rate: Annual risk-free rate
            benchmark_returns: Optional benchmark return series for beta/correlation
            base_currency: Portfolio base currency for cost/PnL accounting
            
        Returns:
            Dict with all metrics
        """
        base_currency = base_currency or 'SGD'
        fx_converter = CurrencyConverter(base_currency)

        if len(equity_curve) < 2:
            return PerformanceMetrics._build_default_metrics(
                equity_curve,
                trades,
                base_currency=base_currency,
            )
        
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        drawdown_series = PerformanceMetrics.calculate_drawdown_series(equity_curve)
        drawdown_duration_metrics = PerformanceMetrics.calculate_drawdown_durations(drawdown_series)
        cagr = PerformanceMetrics.calculate_cagr(equity_curve)
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        recovery_metrics = PerformanceMetrics.calculate_time_to_recovery(equity_curve)
        beta_corr_metrics = PerformanceMetrics.calculate_beta_correlation_metrics(returns, benchmark_returns)
        rolling_window_63 = min(63, len(returns))
        rolling_window_252 = min(252, len(returns))
        rolling_sharpe_63 = PerformanceMetrics.calculate_rolling_sharpe(
            returns,
            window=rolling_window_63,
            risk_free_rate=risk_free_rate,
        ).dropna()
        rolling_sharpe_252 = PerformanceMetrics.calculate_rolling_sharpe(
            returns,
            window=rolling_window_252,
            risk_free_rate=risk_free_rate,
        ).dropna()
        rolling_sharpe_63_latest = float(rolling_sharpe_63.iloc[-1]) if not rolling_sharpe_63.empty else 0.0
        rolling_sharpe_252_latest = float(rolling_sharpe_252.iloc[-1]) if not rolling_sharpe_252.empty else 0.0

        metrics = {
            # Returns
            'total_return': PerformanceMetrics.calculate_total_return(equity_curve),
            'cagr': cagr,
            
            # Risk
            'volatility': PerformanceMetrics.calculate_volatility(returns),
            'max_drawdown': max_drawdown,
            **drawdown_duration_metrics,
            **recovery_metrics,
            
            # Risk-adjusted returns
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate),
            'rolling_sharpe_63_series': rolling_sharpe_63,
            'rolling_sharpe_252_series': rolling_sharpe_252,
            'rolling_sharpe_63': rolling_sharpe_63_latest,
            'rolling_sharpe_252': rolling_sharpe_252_latest,
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(cagr, max_drawdown),
            **beta_corr_metrics,

            # Periodic return breakdowns
            'monthly_returns': PerformanceMetrics.calculate_monthly_returns(equity_curve),
            'annual_returns': PerformanceMetrics.calculate_annual_returns(equity_curve),
            
            # Trade statistics
            'num_trades': len(trades),
            'total_commission': PerformanceMetrics._sum_trade_component_in_base(
                trades, 'commission', fx_converter
            ),
            'total_slippage': PerformanceMetrics._sum_trade_component_in_base(
                trades, 'slippage', fx_converter
            ),
            'total_fx_cost': PerformanceMetrics._sum_trade_component_in_base(
                trades, 'fx_cost', fx_converter
            ),
            'total_cost': PerformanceMetrics._sum_trade_total_cost_in_base(
                trades, fx_converter
            ),
            
            # Period
            'start_date': equity_curve.index[0],
            'end_date': equity_curve.index[-1],
            'num_days': len(equity_curve),
            
            # Final values
            'initial_equity': float(equity_curve.iloc[0]),
            'final_equity': float(equity_curve.iloc[-1]),
            # Backward-compatible aliases.
            'initial_capital': float(equity_curve.iloc[0]),
            'final_capital': float(equity_curve.iloc[-1]),
            'base_currency': base_currency,
        }

        for window, series in (('63', rolling_sharpe_63), ('252', rolling_sharpe_252)):
            if series is None or series.empty:
                metrics[f'rolling_sharpe_{window}_min'] = 0.0
                metrics[f'rolling_sharpe_{window}_mean'] = 0.0
                metrics[f'rolling_sharpe_{window}_max'] = 0.0
                metrics[f'rolling_sharpe_{window}_pct_positive'] = 0.0
                continue
            metrics[f'rolling_sharpe_{window}_min'] = float(series.min())
            metrics[f'rolling_sharpe_{window}_mean'] = float(series.mean())
            metrics[f'rolling_sharpe_{window}_max'] = float(series.max())
            metrics[f'rolling_sharpe_{window}_pct_positive'] = float((series > 0).mean())
        

        # Deterministic trade-level FIFO realized PnL metrics
        fifo_matcher = FIFOTradeMatcher(base_currency=base_currency)
        matched_trades = fifo_matcher.match_trades(trades, fx_converter)
        trade_stats = fifo_matcher.summarize(matched_trades)
        metrics.update(trade_stats)
        metrics.update(
            PerformanceMetrics._calculate_trade_derived_metrics(
                matched_trades=matched_trades,
                trades=trades,
                equity_curve=equity_curve,
                fx_converter=fx_converter,
            )
        )

        metrics['realized_equity'] = float(metrics.get('initial_equity', 0.0) + metrics.get('realized_pnl', 0.0))
        metrics['unrealized_pnl'] = float(metrics.get('final_equity', 0.0) - metrics.get('realized_equity', 0.0))

        # Add cost as percentage of initial capital
        if equity_curve.iloc[0] > 0:
            metrics['cost_pct'] = metrics['total_cost'] / equity_curve.iloc[0]
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict) -> None:
        """
        Print formatted metrics
        
        Args:
            metrics: Dict of metrics from get_comprehensive_metrics
        """
        if not metrics:
            print("\nNo performance metrics available.")
            return

        start_date = metrics.get('start_date')
        end_date = metrics.get('end_date')
        period_str = "N/A"
        if start_date is not None and end_date is not None:
            period_str = f"{start_date.date()} to {end_date.date()}"

        print("\n" + "="*70)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*70)
        
        print(f"\nPeriod: {period_str}")
        print(f"Trading Days: {metrics.get('num_days', 0)}")
        
        print(f"\n{'Returns:':<30}")
        print(f"  {'Initial Equity:':<28} {metrics.get('initial_equity', metrics.get('initial_capital', 0)):>12,.2f}")
        print(f"  {'Final Equity:':<28} {metrics.get('final_equity', metrics.get('final_capital', 0)):>12,.2f}")
        print(f"  {'Realized Equity:':<28} {metrics.get('realized_equity', 0):>12,.2f}")
        print(f"  {'Unrealized PnL:':<28} {metrics.get('unrealized_pnl', 0):>12,.2f}")
        print(f"  {'Total Return:':<28} {metrics.get('total_return', 0):>11.2%}")
        print(f"  {'CAGR:':<28} {metrics.get('cagr', 0):>11.2%}")
        
        print(f"\n{'Risk:':<30}")
        print(f"  {'Volatility (annual):':<28} {metrics.get('volatility', 0):>11.2%}")
        print(f"  {'Maximum Drawdown:':<28} {metrics.get('max_drawdown', 0):>11.2%}")
        print(f"  {'Max Drawdown Duration:':<28} {metrics.get('max_drawdown_duration_days', 0):>12} d")
        print(f"  {'Avg Drawdown Duration:':<28} {metrics.get('avg_drawdown_duration_days', 0):>12.1f} d")
        print(f"  {'Max DD Recovery (peak):':<28} {metrics.get('max_drawdown_recovery_days_from_peak', metrics.get('max_time_to_recovery_days', 0)):>12} d")
        print(f"  {'Max DD Recovery (trough):':<28} {metrics.get('max_drawdown_recovery_days_from_trough', metrics.get('max_time_to_recovery_days', 0)):>12} d")
        print(f"  {'Avg Time to Recovery:':<28} {metrics.get('avg_time_to_recovery_days', 0):>12.1f} d")
        
        benchmark_label = metrics.get('beta_reference', 'Benchmark')
        benchmark_alpha_ref = metrics.get('benchmark_alpha_reference', benchmark_label)
        base_currency = str(metrics.get('base_currency', 'SGD'))

        print(f"\n{'Risk-Adjusted Returns:':<30}")
        print(f"  {'Sharpe Ratio:':<28} {metrics.get('sharpe_ratio', 0):>12.2f}")
        print(
            f"  {'Rolling Sharpe (63d):':<28} "
            f"min={metrics.get('rolling_sharpe_63_min', 0.0):>6.2f} "
            f"mean={metrics.get('rolling_sharpe_63_mean', 0.0):>6.2f} "
            f"max={metrics.get('rolling_sharpe_63_max', 0.0):>6.2f} "
            f"current={metrics.get('rolling_sharpe_63', 0.0):>6.2f} "
            f"(+ve: {metrics.get('rolling_sharpe_63_pct_positive', 0.0):>6.1%})"
        )
        print(
            f"  {'Rolling Sharpe (252d):':<28} "
            f"min={metrics.get('rolling_sharpe_252_min', 0.0):>6.2f} "
            f"mean={metrics.get('rolling_sharpe_252_mean', 0.0):>6.2f} "
            f"max={metrics.get('rolling_sharpe_252_max', 0.0):>6.2f} "
            f"current={metrics.get('rolling_sharpe_252', 0.0):>6.2f} "
            f"(+ve: {metrics.get('rolling_sharpe_252_pct_positive', 0.0):>6.1%})"
        )
        print(f"  {'Sortino Ratio:':<28} {metrics.get('sortino_ratio', 0):>12.2f}")
        print(f"  {'Calmar Ratio:':<28} {metrics.get('calmar_ratio', 0):>12.2f}")
        if metrics.get('beta_correlation_available', False):
            print(f"  {f'Beta vs {benchmark_label}:':<28} {metrics.get('beta', 0):>12.2f}")
            print(f"  {f'Correlation vs {benchmark_label}:':<28} {metrics.get('correlation', 0):>12.2f}")
            print(f"  {'Rolling Beta:':<28} {metrics.get('rolling_beta', 0):>12.2f}")
            print(f"  {'Rolling Correlation:':<28} {metrics.get('rolling_correlation', 0):>12.2f}")
        else:
            reason = metrics.get('beta_correlation_reason', 'benchmark_returns_unavailable')
            print(f"  {f'Beta vs {benchmark_label}:':<28} {'N/A':>12}")
            print(f"  {f'Correlation vs {benchmark_label}:':<28} {'N/A':>12}")
            print(f"  {'Rolling Beta:':<28} {'N/A':>12}")
            print(f"  {'Rolling Correlation:':<28} {'N/A':>12}")
            print(f"  {'Benchmark linkage:':<28} {reason:>12}")
        if 'benchmark_tracking_error' in metrics or 'benchmark_information_ratio' in metrics:
            print(f"  {f'CAPM Alpha vs {benchmark_alpha_ref}:':<28} {metrics.get('benchmark_alpha', 0):>11.2%}")
            print(f"  {f'Active Return vs {benchmark_alpha_ref}:':<28} {metrics.get('benchmark_active_return', 0):>11.2%}")
            print(f"  {'Tracking Error:':<28} {metrics.get('benchmark_tracking_error', 0):>11.2%}")
            print(f"  {'Information Ratio:':<28} {metrics.get('benchmark_information_ratio', 0):>12.2f}")
            print("  Note: CAPM alpha uses a non-standard market proxy; concentrated strategy beta/alpha can be unstable.")
        if 'spy_alpha' in metrics:
            print(f"  {'Alpha vs SPY:':<28} {metrics.get('spy_alpha', 0):>11.2%}")
            print(f"  {'Active Return vs SPY:':<28} {metrics.get('spy_active_return', 0):>11.2%}")
        
        print(f"\n{'Trading Activity:':<30}")
        print(f"  {'Total Trades:':<28} {metrics.get('num_trades', 0):>12}")
        print(f"  {'Closed Round Trips:':<28} {metrics.get('closed_round_trips', 0):>12}")
        print(f"  {'Win Rate:':<28} {metrics.get('win_rate', 0):>11.2%}")
        print(f"  {'Profit Factor:':<28} {metrics.get('profit_factor', 0):>12.2f}")
        print(f"  {'Average Win:':<28} {metrics.get('avg_win', 0):>12.2f}")
        print(f"  {'Average Loss:':<28} {metrics.get('avg_loss', 0):>12.2f}")
        print(f"  {'Reward-to-Risk Ratio:':<28} {metrics.get('reward_to_risk_ratio', 0):>12.2f}")
        print(f"  {f'Expectancy per Closed Trade ({base_currency}):':<28} {metrics.get('expectancy_per_trade', 0):>12.2f}")
        print(f"  {'Avg Holding Period:':<28} {metrics.get('avg_holding_period_days', 0):>12.1f} d")
        print(f"  {'Median Holding Period:':<28} {metrics.get('median_holding_period_days', 0):>12.1f} d")
        print(f"  {'Turnover (x avg equity):':<28} {metrics.get('turnover_ratio', 0):>12.2f}")
        print(f"  {'Annualized Turnover (x avg equity):':<28} {metrics.get('annualized_turnover_ratio', 0):>12.2f}")
        print(f"  {'Avg Exposure:':<28} {metrics.get('avg_exposure', 0):>11.2%}")
        print(f"  {'Max Exposure:':<28} {metrics.get('max_exposure', 0):>11.2%}")
        print(f"  {'Exposure Concentration:':<28} {metrics.get('exposure_concentration_hhi', 0):>12.3f}")
        print(f"  {'Largest Loss:':<28} {metrics.get('largest_loss', 0):>12.2f}")
        if metrics.get('mae_mfe_available', False):
            print(
                f"  {'Avg MAE / MFE:':<28} "
                f"{metrics.get('avg_mae_per_trade', 0):>6.2f} / "
                f"{metrics.get('avg_mfe_per_trade', 0):<6.2f}"
            )
        else:
            reason = metrics.get('mae_mfe_reason', 'intratrade_path_unavailable')
            print(f"  {'Avg MAE / MFE:':<28} {'N/A':>12} ({reason})")
        print(f"  {'Payoff Skew / Kurtosis:':<28} {metrics.get('payoff_skewness', 0):>6.2f} / {metrics.get('payoff_kurtosis', 0):<6.2f}")
        print(f"  {'Max Consecutive Losses:':<28} {metrics.get('max_consecutive_losses', 0):>12}")
        print(f"  {'Realized PnL (FIFO):':<28} {metrics.get('realized_pnl', 0):>12.2f}")
        print(f"  {'Total Commission:':<28} {metrics.get('total_commission', 0):>12.2f}")
        print(f"  {'Total Slippage:':<28} {metrics.get('total_slippage', 0):>12.2f}")
        print(f"  {'Total FX Cost:':<28} {metrics.get('total_fx_cost', 0):>12.2f}")
        print(f"  {'Total Costs:':<28} {metrics.get('total_cost', 0):>12.2f}")
        print(f"  {'Costs (% of initial equity):':<28} {metrics.get('cost_pct', 0):>11.2%}")
        fill_rate = metrics.get('fill_rate')
        if fill_rate is None:
            print(f"  {'Fill Rate:':<28} {'N/A (simulation)':>12}")
        else:
            print(f"  {'Fill Rate:':<28} {fill_rate:>11.2%}")

        trade_count_by_ticker = metrics.get('trade_count_by_ticker', {}) or {}
        pnl_contribution_by_ticker = metrics.get('pnl_contribution_by_ticker', {}) or {}
        win_rate_by_ticker = metrics.get('win_rate_by_ticker', {}) or {}
        avg_holding_by_ticker = metrics.get('avg_holding_period_by_ticker', {}) or {}
        annual_returns = metrics.get('annual_returns')
        if isinstance(annual_returns, pd.Series) and not annual_returns.dropna().empty:
            print(f"\n{'Annual Returns:':<30}")
            print("  Year      Return")
            print("  ----      ------")
            for dt, value in annual_returns.dropna().items():
                print(f"  {pd.Timestamp(dt).year:<8} {float(value):>7.2%}")

        if trade_count_by_ticker or pnl_contribution_by_ticker:
            tickers = set(trade_count_by_ticker.keys()) | set(pnl_contribution_by_ticker.keys())
            print(f"\n{'Per-Ticker Breakdown:':<30}")
            print("  Ticker    Trades    PnL Contrib    Win Rate (closed)    Avg Hold (d)")
            print("  ------    ------    -----------    -----------------    ------------")
            ranked = sorted(
                tickers,
                key=lambda t: abs(float(pnl_contribution_by_ticker.get(t, 0.0))),
                reverse=True,
            )
            for ticker in ranked:
                trades_count = int(trade_count_by_ticker.get(ticker, 0))
                pnl_contrib = float(pnl_contribution_by_ticker.get(ticker, 0.0))
                win_rate = win_rate_by_ticker.get(ticker)
                avg_hold = avg_holding_by_ticker.get(ticker)
                win_text = f"{float(win_rate):.1%}" if win_rate is not None else "N/A"
                hold_text = f"{float(avg_hold):.1f}" if avg_hold is not None else "N/A"
                print(
                    f"  {ticker:<8}  {trades_count:>6}    {pnl_contrib:>11.1%}    "
                    f"{win_text:>17}    {hold_text:>12}"
                )
        
        print("\n" + "="*70)

    @staticmethod
    def _build_default_metrics(equity_curve: pd.Series, trades: List[Trade],
                               base_currency: str = 'SGD') -> Dict:
        """Build default metrics payload for short/empty backtests."""
        has_data = len(equity_curve) > 0
        initial_capital = float(equity_curve.iloc[0]) if has_data else 0.0
        final_capital = float(equity_curve.iloc[-1]) if has_data else 0.0
        fx_converter = CurrencyConverter(base_currency or 'SGD')
        total_commission = PerformanceMetrics._sum_trade_component_in_base(
            trades, 'commission', fx_converter
        )
        total_slippage = PerformanceMetrics._sum_trade_component_in_base(
            trades, 'slippage', fx_converter
        )
        total_fx_cost = PerformanceMetrics._sum_trade_component_in_base(
            trades, 'fx_cost', fx_converter
        )
        total_cost = PerformanceMetrics._sum_trade_total_cost_in_base(trades, fx_converter)

        metrics = {
            'total_return': 0.0,
            'cagr': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration_days': 0,
            'avg_drawdown_duration_days': 0.0,
            'max_time_to_recovery_days': 0,
            'avg_time_to_recovery_days': 0.0,
            'max_drawdown_recovery_days_from_peak': 0,
            'max_drawdown_recovery_days_from_trough': 0,
            'max_drawdown_recovered': False,
            'beta': 0.0,
            'correlation': 0.0,
            'rolling_beta': 0.0,
            'rolling_correlation': 0.0,
            'beta_correlation_available': False,
            'beta_correlation_reason': 'insufficient_equity_history',
            'sharpe_ratio': 0.0,
            'rolling_sharpe_63_series': pd.Series(dtype=float),
            'rolling_sharpe_252_series': pd.Series(dtype=float),
            'rolling_sharpe_63': 0.0,
            'rolling_sharpe_252': 0.0,
            'rolling_sharpe_63_min': 0.0,
            'rolling_sharpe_63_mean': 0.0,
            'rolling_sharpe_63_max': 0.0,
            'rolling_sharpe_63_pct_positive': 0.0,
            'rolling_sharpe_252_min': 0.0,
            'rolling_sharpe_252_mean': 0.0,
            'rolling_sharpe_252_max': 0.0,
            'rolling_sharpe_252_pct_positive': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'monthly_returns': pd.Series(dtype=float),
            'annual_returns': pd.Series(dtype=float),
            'num_trades': len(trades),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_fx_cost': total_fx_cost,
            'total_cost': total_cost,
            'start_date': equity_curve.index[0] if has_data else None,
            'end_date': equity_curve.index[-1] if has_data else None,
            'num_days': len(equity_curve),
            'initial_equity': initial_capital,
            'final_equity': final_capital,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'base_currency': base_currency or 'SGD',
            'realized_pnl': 0.0,
            'realized_equity': initial_capital,
            'unrealized_pnl': final_capital - initial_capital,
            'closed_round_trips': 0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'reward_to_risk_ratio': 0.0,
            'expectancy_per_trade': 0.0,
            'avg_holding_period_days': 0.0,
            'median_holding_period_days': 0.0,
            'turnover_ratio': 0.0,
            'annualized_turnover_ratio': 0.0,
            'avg_exposure': 0.0,
            'max_exposure': 0.0,
            'min_exposure': 0.0,
            'largest_loss': 0.0,
            'avg_mae_per_trade': None,
            'avg_mfe_per_trade': None,
            'mae_mfe_available': False,
            'mae_mfe_reason': 'intratrade_path_unavailable',
            'payoff_skewness': 0.0,
            'payoff_kurtosis': 0.0,
            'max_consecutive_losses': 0,
            'trade_count_by_ticker': {},
            'pnl_contribution_by_ticker': {},
            'win_rate_by_ticker': {},
            'avg_holding_period_by_ticker': {},
            'hit_rate_by_holding_bucket': {
                '0_1d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
                '2_5d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
                '6_20d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
                '21+d': {'wins': 0, 'total': 0, 'hit_rate': 0.0},
            },
            'exposure_concentration_hhi': 0.0,
            'avg_expected_slippage_bps': 0.0,
            'avg_realized_slippage_bps': 0.0,
            'implementation_shortfall': 0.0,
            'implementation_shortfall_bps': 0.0,
            'fill_rate': None,
            'avg_fill_size': 0.0,
            'best_execution_score': 1.0,
            'order_type_breakdown': {},
        }

        if initial_capital > 0:
            metrics['cost_pct'] = total_cost / initial_capital

        return metrics
