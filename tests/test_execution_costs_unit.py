import unittest
from datetime import datetime
from unittest.mock import Mock

from backtesting.execution import ExecutionEngine


class ExecutionCostsUnitTests(unittest.TestCase):
    def _base_config(self):
        return {
            'commission_per_share': 0.0035,
            'commission_min': 0.35,
            'commission_max_pct': 0.01,
            'non_us_commission_rate': 0.001,
            'non_us_commission_min': 4.0,
            'fx_conversion_rate': 0.0002,
            'slippage_bps': {'high_liquidity': 2.0, 'medium_liquidity': 5.0, 'low_liquidity': 10.0},
            'spread_bps': {'high_liquidity': 1.0, 'medium_liquidity': 3.0, 'low_liquidity': 8.0},
            'slippage_model': {'mode': 'heuristic'},
            'components': {
                'broker': {
                    'enabled': True,
                    'us_commission_per_share': 0.0035,
                    'us_commission_min': 0.35,
                    'us_commission_max_pct': 0.01,
                    'non_us_commission_rate': 0.001,
                    'non_us_commission_min': 4.0,
                },
                'regulatory': {'enabled': False},
                'exchange': {'enabled': False},
                'market_taxes': {
                    'enabled': True,
                    'by_market': {
                        'L': {'rate': 0.005, 'side': 'BUY'},
                        'HK': {'rate': 0.001, 'side': 'BOTH'},
                    },
                },
            },
        }

    def test_us_stock_commission_per_share(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        shares = 1000
        price = 10.0
        commission = engine.calculate_commission(shares=shares, price=price, ticker='SPY', action='BUY')
        self.assertAlmostEqual(commission, 3.5, places=8)

    def test_commission_min_floor_applied(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        commission = engine.calculate_commission(shares=10, price=10.0, ticker='SPY', action='BUY')
        self.assertAlmostEqual(commission, 0.35, places=8)

    def test_commission_max_pct_ceiling_applied(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        commission = engine.calculate_commission(shares=1_000_000, price=0.1, ticker='SPY', action='BUY')
        # 1% max of trade value (100_000 * 1%) = 1_000
        self.assertAlmostEqual(commission, 1000.0, places=8)

    def test_non_us_stock_uses_rate_not_per_share(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        commission = engine.calculate_commission(shares=1000, price=10.0, ticker='HSBA.L', action='BUY')
        # non-US commission 0.1% of notional + UK BUY stamp duty 0.5%
        self.assertAlmostEqual(commission, 10.0 + 50.0, places=8)

    def test_uk_stamp_duty_applied_on_buy(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        buy_commission = engine.calculate_commission(shares=100, price=100.0, ticker='VOD.L', action='BUY')
        # non-US broker 0.1% of notional + 0.5% stamp duty
        self.assertAlmostEqual(buy_commission, 10.0 + 50.0, places=8)

    def test_uk_stamp_duty_not_applied_on_sell(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        sell_commission = engine.calculate_commission(shares=100, price=100.0, ticker='VOD.L', action='SELL')
        self.assertAlmostEqual(sell_commission, 10.0, places=8)

    def test_hk_stamp_duty_applied_on_both_sides(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        buy = engine.calculate_commission(shares=1000, price=10.0, ticker='0700.HK', action='BUY')
        sell = engine.calculate_commission(shares=1000, price=10.0, ticker='0700.HK', action='SELL')
        self.assertAlmostEqual(buy, 10.0 + 10.0, places=8)
        self.assertAlmostEqual(sell, 10.0 + 10.0, places=8)

    def test_fx_conversion_rate_applied_to_non_usd(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        engine.fx_converter.convert_to_base = lambda value, from_currency, date=None: value * 1.25
        fx_cost = engine.calculate_fx_cost(trade_value=1000.0, from_currency='GBP', date=datetime(2024, 1, 2))
        self.assertAlmostEqual(fx_cost, 1000.0 * 1.25 * 0.0002, places=8)

    def test_slippage_heuristic_mode_applies_bps(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        # SPY -> high liquidity bps=2.0 and tiny participation impact
        slippage = engine.calculate_slippage(shares=100, price=100.0, avg_volume=10_000_000, ticker='SPY')
        self.assertAlmostEqual(slippage, 100 * 100.0 * (2.0 / 10000.0) * (1.0 + (100 / 10_000_000) * 10), places=6)

    def test_total_cost_is_sum_of_components(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        engine.fx_converter.convert_to_base = lambda value, from_currency, date=None: value * 1.10

        ticker = 'HSBA.L'
        action = 'BUY'
        shares = 200
        reference_price = 10.0
        spread_bps = engine._infer_spread_bps(ticker)
        half_spread = reference_price * (spread_bps / 10000.0) / 2.0
        execution_price = reference_price + half_spread
        avg_volume = 1_000_000

        commission = engine.calculate_commission(shares, execution_price, ticker, action=action)
        model_slippage = engine.calculate_slippage(shares, execution_price, avg_volume, ticker)
        fx_cost = engine.calculate_fx_cost(shares * execution_price, 'GBP', date=datetime(2024, 1, 2))

        trade = engine._build_trade(
            ticker=ticker,
            action=action,
            shares=shares,
            execution_price=execution_price,
            reference_price=reference_price,
            avg_volume=avg_volume,
            date=datetime(2024, 1, 2),
            currency='GBP',
            order_type='MARKET',
        )

        self.assertAlmostEqual(trade.commission, commission, places=8)
        self.assertAlmostEqual(trade.slippage, model_slippage, places=8)
        self.assertAlmostEqual(trade.fx_cost, fx_cost, places=8)
        self.assertAlmostEqual(trade.total_cost(), commission + model_slippage + fx_cost, places=8)

    def test_parameterized_mode_does_not_add_explicit_spread_cost(self):
        config = self._base_config()
        config['slippage_model'] = {
            'mode': 'parameterized',
            'default_bucket': 'medium_liquidity',
            'spread_weight': 0.5,
            'volatility_weight': 0.0,
            'participation_weight': 0.0,
            'bucket_base_bps': {'medium_liquidity': 0.0},
            'bucket_spread_proxy_bps': {'medium_liquidity': 3.0},
            'bucket_overrides': {'SPY': 'medium_liquidity'},
        }

        engine = ExecutionEngine(config, base_currency='USD')
        trade = engine.execute_order(
            ticker='SPY',
            target_shares=100,
            current_shares=0,
            price=100.0,
            avg_volume=1_000_000,
            date=datetime(2024, 1, 2),
            currency='USD',
        )

        self.assertIsNotNone(trade)
        expected_model_slippage = engine.calculate_slippage(
            shares=100,
            price=trade.price,
            avg_volume=1_000_000,
            ticker='SPY',
        )
        realized_spread_cost = trade.shares * abs(trade.price - 100.0)

        self.assertAlmostEqual(trade.slippage, expected_model_slippage, places=8)
        self.assertNotAlmostEqual(
            trade.slippage,
            expected_model_slippage + realized_spread_cost,
            places=8,
        )

    def test_execute_rebalance_does_not_mutate_target_positions_when_closing_missing_holdings(self):
        engine = ExecutionEngine(self._base_config(), base_currency='USD')
        target_positions = {'SPY': 10}
        current_positions = {'SPY': 5, 'QQQ': 7}
        recorded_targets = []

        engine.execute_order = Mock(
            side_effect=lambda **kwargs: recorded_targets.append(
                (kwargs['ticker'], kwargs['target_shares'], kwargs['current_shares'])
            ) or None
        )

        engine.execute_rebalance(
            target_positions=target_positions,
            current_positions=current_positions,
            prices={'SPY': 100.0, 'QQQ': 200.0},
            volumes={'SPY': 1_000_000, 'QQQ': 1_000_000},
            currencies={'SPY': 'USD', 'QQQ': 'USD'},
            date=datetime(2024, 1, 2),
        )

        self.assertEqual(target_positions, {'SPY': 10})
        self.assertIn(('QQQ', 0, 7), recorded_targets)


if __name__ == '__main__':
    unittest.main()
