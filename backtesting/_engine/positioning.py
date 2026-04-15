from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger("backtesting.engine")


class BacktestPositioningMixin:
    def _initialize_starting_positions(
        self,
        initial_positions: Dict[str, float],
        first_date: datetime,
        data,
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

    def _calculate_target_positions(self, weights: Dict[str, float], equity: float,
                                   prices: Dict[str, float],
                                   currencies: Dict[str, str],
                                   date: datetime = None) -> Dict[str, int]:
        """Convert target weights to target share quantities."""
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

            target_value_base = equity * weight
            currency = currencies.get(ticker, 'USD')

            if currency != self.portfolio.base_currency:
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
