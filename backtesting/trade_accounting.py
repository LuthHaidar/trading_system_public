from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

from backtesting.portfolio import Trade


@dataclass
class Lot:
    ticker: str
    shares: float
    unit_cost_base: float
    date: datetime


@dataclass
class MatchedTrade:
    ticker: str
    entry_date: datetime
    exit_date: datetime
    shares: float
    entry_price: float
    exit_price: float
    realized_pnl_base: float


class FIFOTradeMatcher:
    """Deterministic FIFO lot matcher for realized trade-level PnL in base currency."""

    def __init__(self, base_currency: str):
        self.base_currency = base_currency

    def _costs_in_base(self, trade: Trade, fx_converter) -> float:
        """Return total transaction costs in matcher base currency."""
        if trade.currency == self.base_currency:
            return trade.total_cost()

        # Commission/slippage are recorded in trade currency; fx_cost is already base.
        trading_costs = trade.commission + trade.slippage
        trading_costs_base = fx_converter.convert_to_base(trading_costs, trade.currency, date=trade.date)
        return trading_costs_base + trade.fx_cost

    def match_trades(self, trades: List[Trade], fx_converter) -> List[MatchedTrade]:
        lots: Dict[str, Deque[Lot]] = defaultdict(deque)
        matched: List[MatchedTrade] = []

        for trade in sorted(trades, key=lambda t: t.date):
            if trade.shares <= 0:
                continue

            if trade.action == 'BUY':
                gross_value_base = fx_converter.convert_to_base(trade.value, trade.currency, date=trade.date)
                total_cost_base = gross_value_base + self._costs_in_base(trade, fx_converter)
                unit_cost_base = total_cost_base / trade.shares
                lots[trade.ticker].append(
                    Lot(
                        ticker=trade.ticker,
                        shares=trade.shares,
                        unit_cost_base=unit_cost_base,
                        date=trade.date,
                    )
                )
                continue

            # SELL: consume FIFO buy lots and realize PnL including sell-side costs
            remaining_sell_shares = trade.shares
            gross_proceeds_base = fx_converter.convert_to_base(trade.value, trade.currency, date=trade.date)
            sell_total_cost_base = self._costs_in_base(trade, fx_converter)
            sell_cost_per_share = sell_total_cost_base / trade.shares
            exit_unit_proceeds_base = (gross_proceeds_base / trade.shares) - sell_cost_per_share

            while remaining_sell_shares > 0 and lots[trade.ticker]:
                lot = lots[trade.ticker][0]
                matched_shares = min(remaining_sell_shares, lot.shares)

                realized_pnl = (exit_unit_proceeds_base - lot.unit_cost_base) * matched_shares
                matched.append(
                    MatchedTrade(
                        ticker=trade.ticker,
                        entry_date=lot.date,
                        exit_date=trade.date,
                        shares=matched_shares,
                        entry_price=lot.unit_cost_base,
                        exit_price=exit_unit_proceeds_base,
                        realized_pnl_base=realized_pnl,
                    )
                )

                lot.shares -= matched_shares
                remaining_sell_shares -= matched_shares

                if lot.shares <= 0:
                    lots[trade.ticker].popleft()

        return matched

    @staticmethod
    def summarize(matched_trades: List[MatchedTrade]) -> Dict[str, Optional[float]]:
        if not matched_trades:
            return {
                'realized_pnl': 0.0,
                'closed_round_trips': 0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
            }

        pnls = [t.realized_pnl_base for t in matched_trades]
        gross_profit = sum(v for v in pnls if v > 0)
        gross_loss = abs(sum(v for v in pnls if v < 0))

        return {
            'realized_pnl': sum(pnls),
            'closed_round_trips': len(matched_trades),
            'win_rate': sum(1 for v in pnls if v > 0) / len(pnls),
            'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit > 0 else 1.0),
        }
