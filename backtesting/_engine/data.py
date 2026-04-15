from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("backtesting.engine")


@dataclass
class AssetExclusionRecord:
    ticker: str
    date: pd.Timestamp
    reason: str
    available_bars: int
    required_bars: int
    detail: str


class BacktestDataMixin:
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
        """Compute effective simulation start date after warmup history alignment.

        This only controls when the simulation window begins after satisfying the
        strategy's history requirement. It does not remove the separate one-bar
        signal/execution lag used by the daily simulation loop.

        If the requested start date does not have sufficient prior history within the available
        trading dates to satisfy the strategy's requirement, the engine will fall back to the
        requested start date and record the shortfall in warmup_history_shortfall_days for diagnostics.
        """
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
        """Get average volumes for all tickers."""
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
        """Record equity and notify strategy of realized portfolio return when available."""
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
