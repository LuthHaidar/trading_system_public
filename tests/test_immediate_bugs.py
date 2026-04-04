import unittest
from types import SimpleNamespace

import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.portfolio import DataStalenessError, Portfolio
try:
    from broker.ibkr_client import IBKRClient
except ModuleNotFoundError:
    IBKRClient = None


class SelectiveStrategy:
    name = 'Selective'

    def get_required_history(self):
        return 100

    def generate_signals(self, date, data, current_positions):
        return {'AAA': 1.0}


class MultiTickerDataManager:
    def __init__(self, frames):
        self.frames = frames

    def load_ticker(self, ticker, start_date=None, end_date=None):
        return self.frames[ticker].copy()


class ImmediateBugRegressionTests(unittest.TestCase):
    def _base_config(self):
        return {
            'portfolio': {'initial_capital': 1000.0, 'currency': 'USD'},
            'execution': {'rebalance_timeframe': 'daily', 'timing': 'close', 'progress_log_interval': 0, 'min_forward_data_days': 3},
            'costs': {
                'commission_per_share': 0.0,
                'commission_min': 0.0,
                'commission_max_pct': 0.0,
                'non_us_commission_rate': 0.0,
                'non_us_commission_min': 0.0,
                'fx_conversion_rate': 0.0,
                'slippage_bps': {'high_liquidity': 0.0, 'medium_liquidity': 0.0, 'low_liquidity': 0.0},
                'spread_bps': {'high_liquidity': 0.0, 'medium_liquidity': 0.0, 'low_liquidity': 0.0},
                'slippage_model': {'mode': 'heuristic'},
            },
            'risk': {
                'position_sizing_method': 'equal',
                'max_position_size': 1.0,
                'min_position_size': 0.0,
                'min_cash_reserve': 0.0,
                'use_optimizer': False,
                'covariance_estimator': 'sample',
            },
            'data': {'max_stale_price_days': 3},
            'data_platform': {'enabled': False},
        }

    def test_b1_does_not_skip_day_when_only_one_ticker_has_short_history(self):
        long_idx = pd.bdate_range('2024-01-01', periods=120)
        short_idx = pd.bdate_range('2024-04-01', periods=5)

        aaa = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=long_idx)
        bbb = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=short_idx)

        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': aaa, 'BBB': bbb}),
            config=self._base_config(),
        )

        results = engine.run(['AAA', 'BBB'], '2024-01-01', '2024-06-30')
        self.assertGreater(len(results['trades']), 0)
        self.assertEqual(results['trades'][0].ticker, 'AAA')

    def test_b2_carry_forward_and_stale_price_threshold(self):
        portfolio = Portfolio(initial_capital=1000.0, base_currency='USD', max_stale_price_days=2)
        portfolio.positions = {'AAA': 1.0}

        # Day 1 explicit price
        day1 = pd.Timestamp('2024-01-01')
        eq1 = portfolio.get_total_equity({'AAA': 100.0}, {'AAA': 'USD'}, date=day1)
        self.assertAlmostEqual(eq1, 1100.0)

        # Day 2/3 missing => stale fallback allowed
        day2 = pd.Timestamp('2024-01-02')
        self.assertAlmostEqual(portfolio.get_total_equity({}, {'AAA': 'USD'}, date=day2), 1100.0)
        day3 = pd.Timestamp('2024-01-03')
        self.assertAlmostEqual(portfolio.get_total_equity({}, {'AAA': 'USD'}, date=day3), 1100.0)

        # Day 4 exceeds stale threshold and raises.
        with self.assertRaises(DataStalenessError):
            portfolio.get_total_equity({}, {'AAA': 'USD'}, date=pd.Timestamp('2024-01-04'))

    def test_b2_stale_threshold_uses_valuation_sessions_not_calendar_days(self):
        portfolio = Portfolio(initial_capital=1000.0, base_currency='USD', max_stale_price_days=1)
        portfolio.positions = {'AAA': 1.0}
        portfolio.position_currencies = {'AAA': 'USD'}

        friday = pd.Timestamp('2024-01-05')
        monday = pd.Timestamp('2024-01-08')
        tuesday = pd.Timestamp('2024-01-09')

        self.assertAlmostEqual(
            portfolio.get_total_equity({'AAA': 100.0}, {'AAA': 'USD'}, date=friday),
            1100.0,
        )
        self.assertAlmostEqual(
            portfolio.get_total_equity({}, {'AAA': 'USD'}, date=monday),
            1100.0,
        )
        with self.assertRaises(DataStalenessError):
            portfolio.get_total_equity({}, {'AAA': 'USD'}, date=tuesday)

    def test_b4_initial_positions_are_applied_on_first_bar(self):
        idx = pd.bdate_range('2024-01-01', periods=10)
        frame = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=idx)
        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': frame}),
            config=self._base_config(),
        )

        results = engine.run(['AAA'], '2024-01-01', '2024-01-31', initial_positions={'AAA': 0.5})

        self.assertIn('AAA', engine.portfolio.positions)
        self.assertGreater(engine.portfolio.positions['AAA'], 0)
        self.assertLess(results['equity_curve'].iloc[0], 100_000.0)
        self.assertEqual(results['effective_start_date'], '2024-01-01')
        self.assertGreater(results.get('warmup_history_shortfall_days', 0), 0)

    def test_b4_initial_positions_reject_unknown_ticker(self):
        idx = pd.bdate_range('2024-01-01', periods=5)
        frame = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=idx)
        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': frame}),
            config=self._base_config(),
        )

        with self.assertRaises(ValueError):
            engine.run(['AAA'], '2024-01-01', '2024-01-31', initial_positions={'BBB': 0.5})

    def test_b4_initial_positions_reject_overweight_sum(self):
        idx = pd.bdate_range('2024-01-01', periods=5)
        frame = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=idx)
        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': frame}),
            config=self._base_config(),
        )

        with self.assertRaises(ValueError):
            engine.run(['AAA'], '2024-01-01', '2024-01-31', initial_positions={'AAA': 1.2})

    def test_b10_trading_dates_use_union_calendar(self):
        aaa = pd.DataFrame(
            {'Open': [100.0, 101.0], 'High': [100.0, 101.0], 'Low': [100.0, 101.0], 'Close': [100.0, 101.0], 'Volume': [1_000_000, 1_000_000]},
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        bbb = pd.DataFrame(
            {'Open': [200.0], 'High': [200.0], 'Low': [200.0], 'Close': [200.0], 'Volume': [1_000_000]},
            index=pd.to_datetime(['2024-01-02']),
        )

        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': aaa, 'BBB': bbb}),
            config=self._base_config(),
        )

        trading_dates = engine._get_trading_dates(
            {'AAA': aaa, 'BBB': bbb},
            start_date='2024-01-01',
            end_date='2024-01-31',
        )
        self.assertEqual(list(trading_dates), [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')])

    def test_b10_current_weights_include_stale_held_positions(self):
        portfolio = Portfolio(initial_capital=0.0, base_currency='USD', max_stale_price_days=3)
        portfolio.positions = {'AAA': 1.0, 'BBB': 1.0}

        day1 = pd.Timestamp('2024-01-02')
        weights_day1 = portfolio.get_weights({'AAA': 100.0, 'BBB': 100.0}, {'AAA': 'USD', 'BBB': 'USD'}, date=day1)
        self.assertAlmostEqual(weights_day1['AAA'], 0.5)
        self.assertAlmostEqual(weights_day1['BBB'], 0.5)

        # Day 2: BBB market closed/missing bar; weight should still include stale BBB valuation.
        day2 = pd.Timestamp('2024-01-03')
        weights_day2 = portfolio.get_weights({'AAA': 120.0}, {'AAA': 'USD', 'BBB': 'USD'}, date=day2)
        self.assertIn('BBB', weights_day2)
        self.assertAlmostEqual(weights_day2['AAA'], 120.0 / 220.0)
        self.assertAlmostEqual(weights_day2['BBB'], 100.0 / 220.0)





    def test_stale_breach_raises_without_policy_switches(self):
        portfolio = Portfolio(
            initial_capital=1000.0,
            base_currency='USD',
            max_stale_price_days=1,
        )
        portfolio.positions = {'AAA': 1.0}

        portfolio.get_total_equity({'AAA': 100.0}, {'AAA': 'USD'}, date=pd.Timestamp('2024-01-01'))
        portfolio.get_total_equity({}, {'AAA': 'USD'}, date=pd.Timestamp('2024-01-03'))
        with self.assertRaises(DataStalenessError):
            portfolio.get_total_equity({}, {'AAA': 'USD'}, date=pd.Timestamp('2024-01-04'))

    def test_stale_breach_does_not_liquidate_position(self):
        portfolio = Portfolio(initial_capital=1000.0, base_currency='USD', max_stale_price_days=1)
        portfolio.positions = {'AAA': 2.0}
        portfolio.position_currencies = {'AAA': 'USD'}

        portfolio.get_total_equity({'AAA': 50.0}, {'AAA': 'USD'}, date=pd.Timestamp('2024-01-01'))
        portfolio.get_total_equity({}, {'AAA': 'USD'}, date=pd.Timestamp('2024-01-03'))
        with self.assertRaises(DataStalenessError):
            portfolio.get_total_equity({}, {'AAA': 'USD'}, date=pd.Timestamp('2024-01-04'))
        self.assertIn('AAA', portfolio.positions)
        self.assertEqual(len(portfolio.trades), 0)



    def test_forward_horizon_guard_is_opt_in_for_legacy_configs(self):
        idx = pd.bdate_range('2024-01-01', periods=2)
        frame = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=idx)
        cfg = self._base_config()
        cfg['execution'].pop('min_forward_data_days', None)

        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': frame}),
            config=cfg,
        )

        results = engine.run(['AAA'], '2024-01-01', '2024-01-05')
        self.assertGreaterEqual(len(results['trades']), 1)

    def test_engine_blocks_buy_when_forward_horizon_insufficient(self):
        idx = pd.bdate_range('2024-01-01', periods=6)
        frame = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=idx)
        cfg = self._base_config()
        cfg['execution']['min_forward_data_days'] = 10
        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': frame}),
            config=cfg,
        )

        results = engine.run(['AAA'], '2024-01-01', '2024-01-10')
        self.assertEqual(len(results['trades']), 0)

    def test_engine_config_rejects_min_forward_less_than_max_stale(self):
        cfg = self._base_config()
        cfg['execution']['min_forward_data_days'] = 1
        cfg['data']['max_stale_price_days'] = 3

        idx = pd.bdate_range('2024-01-01', periods=10)
        frame = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=idx)

        with self.assertRaises(ValueError):
            BacktestEngine(
                strategy=SelectiveStrategy(),
                data_manager=MultiTickerDataManager({'AAA': frame}),
                config=cfg,
            )



    def test_halt_policy_stops_run_on_stale_breach_and_no_metrics_returned(self):
        cfg = self._base_config()
        cfg['execution']['min_forward_data_days'] = 1
        cfg['data']['max_stale_price_days'] = 1

        aaa_idx = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        bbb_idx = pd.bdate_range('2024-01-01', periods=8)
        aaa = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=aaa_idx)
        bbb = pd.DataFrame({'Open': 200.0, 'High': 200.0, 'Low': 200.0, 'Close': 200.0, 'Volume': 1_000_000}, index=bbb_idx)

        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': aaa, 'BBB': bbb}),
            config=cfg,
        )

        with self.assertRaises(DataStalenessError):
            engine.run(['AAA', 'BBB'], '2024-01-01', '2024-01-15')

    def test_engine_stale_breach_halts_without_force_close_path(self):
        cfg = self._base_config()
        cfg['execution']['min_forward_data_days'] = 1
        cfg['data']['max_stale_price_days'] = 1

        aaa_idx = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        bbb_idx = pd.bdate_range('2024-01-01', periods=8)
        aaa = pd.DataFrame({'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1_000_000}, index=aaa_idx)
        bbb = pd.DataFrame({'Open': 200.0, 'High': 200.0, 'Low': 200.0, 'Close': 200.0, 'Volume': 1_000_000}, index=bbb_idx)

        engine = BacktestEngine(
            strategy=SelectiveStrategy(),
            data_manager=MultiTickerDataManager({'AAA': aaa, 'BBB': bbb}),
            config=cfg,
        )

        with self.assertRaises(DataStalenessError):
            engine.run(['AAA', 'BBB'], '2024-01-01', '2024-01-15')

    @unittest.skipUnless(IBKRClient is not None, "ib_insync not installed")
    def test_b15_get_positions_reconstructs_exchange_suffix(self):
        client = IBKRClient({'host': '127.0.0.1', 'port': 7497, 'client_id': 1})
        client.connected = True
        client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            positions=lambda: [
                SimpleNamespace(
                    contract=SimpleNamespace(symbol='HSBC', exchange='LSE'),
                    position=10,
                    avgCost=50.0,
                )
            ],
        )

        positions = client.get_positions()
        self.assertIn('HSBC.L', positions)


if __name__ == '__main__':
    unittest.main()
