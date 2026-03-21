from typing import Dict, List

from backtesting.portfolio import Trade


class TransactionCostAnalysis:
    """Post-trade execution quality and transaction cost analysis helpers."""

    @staticmethod
    def summarize(trades: List[Trade]) -> Dict:
        if not trades:
            return {
                'avg_expected_slippage_bps': 0.0,
                'avg_realized_slippage_bps': 0.0,
                'implementation_shortfall': 0.0,
                'implementation_shortfall_bps': 0.0,
                'fill_rate': 0.0,
                'avg_fill_size': 0.0,
                'best_execution_score': 1.0,
                'order_type_breakdown': {},
            }

        notional = sum(abs(t.value) for t in trades)
        expected_notional = sum(abs((t.expected_price or t.price) * t.shares) for t in trades)
        gross_slippage = sum(t.slippage for t in trades)
        expected_vs_actual = sum(abs((t.price - (t.expected_price or t.price)) * t.shares) for t in trades)
        expected_slippage_notional = max(notional - gross_slippage, 0.0)

        avg_expected_slippage_bps = (gross_slippage / expected_slippage_notional * 10000.0) if expected_slippage_notional > 0 else 0.0
        avg_realized_slippage_bps = (expected_vs_actual / expected_notional * 10000.0) if expected_notional > 0 else 0.0

        total_cost = sum(t.total_cost() for t in trades)
        implementation_shortfall = total_cost + expected_vs_actual
        implementation_shortfall_bps = (implementation_shortfall / expected_notional * 10000.0) if expected_notional > 0 else 0.0

        fills = [t.shares for t in trades if t.shares > 0]
        fill_rate = min(sum(fills) / max(sum(fills), 1), 1.0)
        avg_fill_size = sum(fills) / len(fills) if fills else 0.0

        # 1.0 is best, degrade score with execution drag (capped at 0)
        best_execution_score = max(0.0, 1.0 - implementation_shortfall_bps / 100.0)

        order_type_breakdown = {}
        for t in trades:
            key = getattr(t, 'order_type', 'MARKET') or 'MARKET'
            order_type_breakdown[key] = order_type_breakdown.get(key, 0) + 1

        return {
            'avg_expected_slippage_bps': avg_expected_slippage_bps,
            'avg_realized_slippage_bps': avg_realized_slippage_bps,
            'implementation_shortfall': implementation_shortfall,
            'implementation_shortfall_bps': implementation_shortfall_bps,
            'fill_rate': fill_rate,
            'avg_fill_size': avg_fill_size,
            'best_execution_score': best_execution_score,
            'order_type_breakdown': order_type_breakdown,
        }
