import unittest
from datetime import datetime

from backtesting.portfolio import Trade
from backtesting.trade_accounting import FIFOTradeMatcher


class DummyFXConverter:
    def convert_to_base(self, amount, currency, date=None):
        if currency == 'USD':
            return amount * 2.0
        return amount


class TradeAccountingRegressionTests(unittest.TestCase):
    def test_fifo_matcher_converts_sell_costs_to_base_currency(self):
        matcher = FIFOTradeMatcher(base_currency='SGD')
        fx = DummyFXConverter()

        trades = [
            Trade(
                date=datetime(2024, 1, 2),
                ticker='AAA',
                action='BUY',
                shares=10,
                price=10.0,
                commission=1.0,
                slippage=1.0,
                fx_cost=0.5,
                value=100.0,
                currency='USD',
            ),
            Trade(
                date=datetime(2024, 1, 3),
                ticker='AAA',
                action='SELL',
                shares=10,
                price=12.0,
                commission=1.0,
                slippage=1.0,
                fx_cost=0.5,
                value=120.0,
                currency='USD',
            ),
        ]

        matched = matcher.match_trades(trades, fx)
        self.assertEqual(len(matched), 1)

        # Expected with proper base conversion:
        # buy unit cost = (100*2 + (1+1)*2 + 0.5)/10 = 20.45
        # sell unit proceeds = (120*2)/10 - ((1+1)*2 + 0.5)/10 = 23.55
        # pnl = (23.55 - 20.45) * 10 = 31.0
        self.assertAlmostEqual(matched[0].realized_pnl_base, 31.0, places=8)


if __name__ == '__main__':
    unittest.main()
