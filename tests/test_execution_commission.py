import unittest

from backtesting.execution import ExecutionEngine


class NonUSCommissionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            'commission_per_share': 0.0035,
            'commission_min': 0.35,
            'commission_max_pct': 0.01,
            'non_us_commission_rate': 0.001,
            'non_us_commission_min': 4.0,
            'fx_conversion_rate': 0.0,
            'slippage_bps': {'high_liquidity': 0.0, 'medium_liquidity': 0.0, 'low_liquidity': 0.0},
            'spread_bps': {'high_liquidity': 0.0, 'medium_liquidity': 0.0, 'low_liquidity': 0.0},
            'slippage_model': {'mode': 'heuristic'},
        }
        self.engine = ExecutionEngine(self.config, base_currency='USD')

    def test_lse_ticker_uses_non_us_rate_when_above_minimum(self):
        shares = 1000
        price = 10.0
        trade_value = shares * price

        commission = self.engine.calculate_commission(shares=shares, price=price, ticker='HSBA.L', action='BUY')

        self.assertAlmostEqual(commission, trade_value * self.config['non_us_commission_rate'])

    def test_sgx_ticker_uses_non_us_minimum_when_trade_is_small(self):
        shares = 10
        price = 10.0

        commission = self.engine.calculate_commission(shares=shares, price=price, ticker='D05.SI', action='BUY')

        self.assertAlmostEqual(commission, self.config['non_us_commission_min'])


if __name__ == '__main__':
    unittest.main()
