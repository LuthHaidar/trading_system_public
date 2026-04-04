import unittest

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics


class AlwaysLongStrategy:
    name = 'AlwaysLong'

    def get_required_history(self):
        return 0

    def generate_signals(self, date, data, current_positions):
        return {'AAA': 1.0}

    def on_realized_portfolio_return(self, date, portfolio_return, context=None):
        return None


class HoldCurrentWeightsStrategy:
    name = 'HoldCurrent'

    def get_required_history(self):
        return 0

    def generate_signals(self, date, data, current_positions):
        return dict(current_positions or {})


class EqualTwoAssetStrategy:
    name = 'EqualTwo'

    def get_required_history(self):
        return 0

    def generate_signals(self, date, data, current_positions):
        return {'AAA': 1.0, 'BBB': 1.0}



class TwoStepDriftStrategy:
    name = 'TwoStepDrift'

    def get_required_history(self):
        return 0

    def generate_signals(self, date, data, current_positions):
        if pd.Timestamp(date) <= pd.Timestamp('2024-01-02'):
            return {'AAA': 0.5, 'BBB': 0.5}
        return {'AAA': 0.49, 'BBB': 0.51}




class TinyTailWeightStrategy:
    name = 'TinyTail'

    def get_required_history(self):
        return 0

    def generate_signals(self, date, data, current_positions):
        return {'AAA': 0.90, 'BBB': 0.09, 'CCC': 0.01}



class WarmupStrategy:
    name = 'Warmup'

    def get_required_history(self):
        return 2

    def generate_signals(self, date, data, current_positions):
        return {'AAA': 1.0}


class MetadataLongStrategy:
    name = 'MetadataLong'

    def __init__(self):
        self._last_meta = {}

    def get_required_history(self):
        return 0

    def generate_signals(self, date, data, current_positions):
        self._last_meta = {
            'AAA': {
                'reason': 'momentum_rank=1 score=0.5000',
                'signal_strength': 0.5,
                'confidence': 0.8,
            }
        }
        return {'AAA': 1.0}

    def get_signal_metadata(self):
        return dict(self._last_meta)


class AttributionStubStrategy:
    name = 'strategy_orchestration'

    def __init__(self):
        self._history = []

    def get_required_history(self):
        return 0

    def generate_signals(self, date, data, current_positions):
        self._history.append(
            {
                'date': pd.Timestamp(date),
                'strategy_allocations': {'sub_a': {'AAA': 0.6}, 'sub_b': {'AAA': 0.4}},
            }
        )
        return {'AAA': 1.0}

    def get_attribution_history(self):
        return list(self._history)


class FakeDataManager:
    def __init__(self, frame):
        self.frame = frame

    def load_ticker(self, ticker, start_date=None, end_date=None):
        return self.frame.copy()


class MultiFrameDataManager:
    def __init__(self, frames):
        self.frames = frames

    def load_ticker(self, ticker, start_date=None, end_date=None):
        return self.frames[ticker].copy()


class BacktestEngineRegressionTests(unittest.TestCase):
    def _base_config(self):
        return {
            'portfolio': {
                'initial_capital': 1000.0,
                'currency': 'USD',
            },
            'execution': {
                'rebalance_timeframe': 'daily',
                'timing': 'close',
                'progress_log_interval': 0,
            },
            'costs': {
                'commission_per_share': 0.0,
                'commission_min': 0.0,
                'commission_max_pct': 0.0,
                'non_us_commission_rate': 0.0,
                'non_us_commission_min': 0.0,
                'fx_conversion_rate': 0.0,
                'slippage_bps': {
                    'high_liquidity': 0.0,
                    'medium_liquidity': 0.0,
                    'low_liquidity': 0.0,
                },
                'spread_bps': {
                    'high_liquidity': 0.0,
                    'medium_liquidity': 0.0,
                    'low_liquidity': 0.0,
                },
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
            'data_platform': {
                'enabled': False,
            },
        }

    def test_backtest_uses_prior_bar_signal_and_current_open_execution(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 200.0],
                'High': [100.0, 200.0],
                'Low': [100.0, 200.0],
                'Close': [100.0, 50.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )

        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )

        results = engine.run(['AAA'], '2024-01-02', '2024-06-30')
        trades = results['trades']
        self.assertEqual(len(trades), 1)

        trade = trades[0]
        self.assertEqual(pd.Timestamp(trade.date), pd.Timestamp('2024-01-03'))
        self.assertAlmostEqual(trade.price, 200.0, places=8)
        self.assertNotAlmostEqual(trade.price, 50.0, places=8)

    def test_optimizer_output_is_reconstrained_before_execution(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0],
                'High': [100.0, 100.0],
                'Low': [100.0, 100.0],
                'Close': [100.0, 100.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )

        config = self._base_config()
        config['risk']['use_optimizer'] = True
        config['risk']['optimizer_method'] = 'max_sharpe'
        config['risk']['max_position_size'] = 0.20

        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=config,
        )

        # Force optimizer to emit a violating weight.
        engine.optimizer.optimize = lambda signals, data, method='max_sharpe': {'AAA': 1.0}

        results = engine.run(['AAA'], '2024-01-02', '2024-01-03')
        trades = results['trades']
        self.assertEqual(len(trades), 1)

        # With 1000 equity and 100 execution price, 20% cap => 2 shares.
        self.assertEqual(int(trades[0].shares), 2)

    def test_weight_drift_helper_calculates_l1_distance(self):
        target = {'AAA': 0.50, 'BBB': 0.40}
        realized = {'AAA': 0.45, 'BBB': 0.35, 'CCC': 0.10}
        drift = BacktestEngine._calculate_weight_drift_l1(target, realized)
        self.assertAlmostEqual(drift, 0.20, places=8)

    def test_backtest_notifies_strategy_with_realized_returns(self):
        class HookedStrategy(AlwaysLongStrategy):
            def __init__(self):
                self.calls = []

            def on_realized_portfolio_return(self, date, portfolio_return, context=None):
                self.calls.append((date, portfolio_return, context or {}))

        frame = pd.DataFrame(
            {
                'Open': [100.0, 200.0],
                'High': [100.0, 200.0],
                'Low': [100.0, 200.0],
                'Close': [100.0, 50.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )

        strategy = HookedStrategy()
        engine = BacktestEngine(
            strategy=strategy,
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )
        engine.run(['AAA'], '2024-01-02', '2024-01-03')

        self.assertEqual(len(strategy.calls), 1)
        realized = strategy.calls[0][1]
        self.assertAlmostEqual(realized, -0.75, places=8)

    def test_hold_signal_skips_rebalance_and_emits_debug_log(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 101.0],
                'High': [100.0, 101.0],
                'Low': [100.0, 101.0],
                'Close': [100.0, 101.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        engine = BacktestEngine(
            strategy=HoldCurrentWeightsStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )

        with self.assertLogs('backtesting.engine', level='DEBUG') as logs:
            results = engine.run(['AAA'], '2024-01-02', '2024-01-03')

        self.assertEqual(results['trades'], [])
        self.assertTrue(any('Skipping rebalance' in line for line in logs.output))

    def test_drift_decomposition_is_cash_aware(self):
        target_weights = {'AAA': 0.50, 'BBB': 0.50}
        realized_total_weights = {'AAA': 0.4975, 'BBB': 0.4975}

        target_invested = BacktestEngine._normalize_to_invested_sleeve(target_weights)
        realized_invested = BacktestEngine._normalize_to_invested_sleeve(realized_total_weights)
        invested_drift = BacktestEngine._calculate_weight_drift_l1(target_invested, realized_invested)

        target_cash = max(0.0, 1.0 - sum(target_weights.values()))
        realized_cash = max(0.0, 1.0 - sum(realized_total_weights.values()))
        cash_drift = abs(target_cash - realized_cash)

        self.assertAlmostEqual(invested_drift, 0.0, places=8)
        self.assertAlmostEqual(cash_drift, 0.005, places=8)

    def test_final_vol_scaler_applies_after_constraints_pipeline(self):
        days = 90
        dates = pd.date_range('2024-01-01', periods=days, freq='B')
        returns = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(days)])
        prices = 100.0 * np.cumprod(1.0 + returns)
        frame = pd.DataFrame(
            {
                'Open': prices,
                'High': prices,
                'Low': prices,
                'Close': prices,
                'Volume': np.full(days, 1_000_000),
            },
            index=dates,
        )

        config = self._base_config()
        config['risk']['position_sizing_method'] = 'target_vol'
        config['risk']['target_volatility'] = 0.10
        config['risk']['volatility_window'] = 20
        config['risk']['min_cash_reserve'] = 0.0
        config['risk']['max_position_size'] = 1.0
        config['risk']['max_correlation'] = 1.0
        config['execution']['rebalance_timeframe'] = 'daily'

        engine = BacktestEngine(
            strategy=EqualTwoAssetStrategy(),
            data_manager=FakeDataManager(frame),
            config=config,
        )

        captured = {'weights': None}
        original_execute_rebalance = engine.execution_engine.execute_rebalance

        def capture_execute_rebalance(*args, **kwargs):
            captured['weights'] = dict(kwargs.get('target_weights') or {})
            return original_execute_rebalance(*args, **kwargs)

        engine.execution_engine.execute_rebalance = capture_execute_rebalance
        engine.run(['AAA', 'BBB'], str(dates[0].date()), str(dates[-1].date()))

        final_weights = captured['weights']
        self.assertIsNotNone(final_weights)
        self.assertTrue(sum(final_weights.values()) > 0)

        returns_df = pd.DataFrame({'AAA': frame['Close'].pct_change(), 'BBB': frame['Close'].pct_change()}).dropna()
        cov = returns_df.tail(20).cov() * 252.0
        cols = [ticker for ticker in cov.columns if final_weights.get(ticker, 0.0) > 0]
        w = np.array([final_weights.get(ticker, 0.0) for ticker in cols], dtype=float)
        sub_cov = cov.loc[cols, cols].values
        vol_est = float(np.sqrt(np.dot(w, np.dot(sub_cov, w))))
        self.assertAlmostEqual(vol_est, 0.10, delta=0.03)


    def test_engine_passes_timing_keyword_to_execution_rebalance(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 101.0],
                'High': [100.0, 101.0],
                'Low': [100.0, 101.0],
                'Close': [100.0, 101.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )

        config = self._base_config()
        config['execution']['timing'] = 'open'

        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=config,
        )

        captured = {'kwargs': None}
        original_execute_rebalance = engine.execution_engine.execute_rebalance

        def capture_execute_rebalance(*args, **kwargs):
            captured['kwargs'] = dict(kwargs)
            return original_execute_rebalance(*args, **kwargs)

        engine.execution_engine.execute_rebalance = capture_execute_rebalance
        engine.run(['AAA'], '2024-01-02', '2024-01-03')

        self.assertIsNotNone(captured['kwargs'])
        self.assertIn('timing', captured['kwargs'])
        self.assertNotIn('execution_session', captured['kwargs'])
        self.assertEqual(captured['kwargs']['timing'], 'OPEN')

    def test_trade_metadata_defaults_without_strategy_signal_metadata(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0],
                'High': [100.0, 100.0],
                'Low': [100.0, 100.0],
                'Close': [100.0, 100.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )

        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )
        results = engine.run(['AAA'], '2024-01-02', '2024-01-03')
        trade = results['trades'][0]

        self.assertEqual(trade.decision_reason, 'strategy_rebalance')
        self.assertIsNone(trade.signal_confidence)
        self.assertIsNone(trade.signal_strength)
        self.assertAlmostEqual(float(trade.weight_delta), 1.0, places=8)

    def test_trade_metadata_uses_strategy_signal_metadata(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0],
                'High': [100.0, 100.0],
                'Low': [100.0, 100.0],
                'Close': [100.0, 100.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )

        engine = BacktestEngine(
            strategy=MetadataLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )
        results = engine.run(['AAA'], '2024-01-02', '2024-01-03')
        trade = results['trades'][0]

        self.assertEqual(trade.decision_reason, 'momentum_rank=1 score=0.5000')
        self.assertAlmostEqual(float(trade.signal_strength), 0.5, places=8)
        self.assertAlmostEqual(float(trade.signal_confidence), 0.8, places=8)
        self.assertAlmostEqual(float(trade.weight_delta), 1.0, places=8)

    def test_trade_decision_rows_keep_weight_delta_separate(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0],
                'High': [100.0, 100.0],
                'Low': [100.0, 100.0],
                'Close': [100.0, 100.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )

        rows = engine._build_trade_decision_rows(
            date=pd.Timestamp('2024-01-03'),
            signal_date=pd.Timestamp('2024-01-02'),
            target_positions={'AAA': 10},
            current_positions={},
            target_weights={'AAA': 0.2},
            current_weights={'AAA': 0.0},
            executed_trades=[],
            signal_meta={
                'AAA': {
                    'reason': 'custom_reason',
                    'signal_strength': 0.77,
                    'confidence': 0.91,
                }
            },
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['decision_reason'], 'custom_reason')
        self.assertAlmostEqual(float(row['signal_strength']), 0.77, places=8)
        self.assertAlmostEqual(float(row['weight_delta']), 0.2, places=8)
        self.assertAlmostEqual(float(row['confidence']), 0.91, places=8)



    def test_min_trade_value_filters_tiny_positions_and_redistributes(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0],
                'High': [100.0, 100.0],
                'Low': [100.0, 100.0],
                'Close': [100.0, 100.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        config = self._base_config()
        config['execution']['min_trade_value'] = 200.0

        engine = BacktestEngine(
            strategy=TinyTailWeightStrategy(),
            data_manager=FakeDataManager(frame),
            config=config,
        )

        filtered = engine._apply_min_trade_value_filter(
            target_positions={'AAA': 90, 'BBB': 9, 'CCC': 1},
            prices={'AAA': 100.0, 'BBB': 100.0, 'CCC': 100.0},
            currencies={'AAA': 'USD', 'BBB': 'USD', 'CCC': 'USD'},
            equity=10_000.0,
            date=pd.Timestamp('2024-01-03'),
        )

        self.assertNotIn('CCC', filtered)
        self.assertGreaterEqual(filtered.get('AAA', 0), 90)
        self.assertGreaterEqual(filtered.get('BBB', 0), 9)


    def test_min_trade_value_redistribution_respects_max_position_cap(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0],
                'High': [100.0],
                'Low': [100.0],
                'Close': [100.0],
                'Volume': [1_000_000],
            },
            index=pd.to_datetime(['2024-01-03']),
        )
        config = self._base_config()
        config['execution']['min_trade_value'] = 350.0
        config['risk']['max_position_size'] = 0.5

        engine = BacktestEngine(
            strategy=TinyTailWeightStrategy(),
            data_manager=FakeDataManager(frame),
            config=config,
        )

        filtered = engine._apply_min_trade_value_filter(
            target_positions={'AAA': 4, 'BBB': 4, 'CCC': 3},
            prices={'AAA': 100.0, 'BBB': 100.0, 'CCC': 100.0},
            currencies={'AAA': 'USD', 'BBB': 'USD', 'CCC': 'USD'},
            equity=1_000.0,
            date=pd.Timestamp('2024-01-03'),
        )

        # CCC is below min_trade_value and removed.
        self.assertNotIn('CCC', filtered)
        # Redistribution should not push any surviving position above max_position_size cap.
        max_shares_at_cap = int((1_000.0 * 0.5) / 100.0)
        self.assertLessEqual(filtered.get('AAA', 0), max_shares_at_cap)
        self.assertLessEqual(filtered.get('BBB', 0), max_shares_at_cap)

    def test_positions_history_records_rebalance_weights(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0],
                'High': [100.0, 100.0],
                'Low': [100.0, 100.0],
                'Close': [100.0, 100.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )

        results = engine.run(['AAA'], '2024-01-02', '2024-01-03')

        history = results.get('positions_history')
        self.assertIsInstance(history, pd.DataFrame)
        self.assertIn(pd.Timestamp('2024-01-03'), history.index)
        self.assertAlmostEqual(float(history.loc[pd.Timestamp('2024-01-03'), 'AAA']), 1.0, places=8)

    def test_min_rebalance_weight_delta_skips_small_l1_drift(self):
        dates = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04'])
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0, 100.0],
                'High': [100.0, 100.0, 100.0],
                'Low': [100.0, 100.0, 100.0],
                'Close': [100.0, 100.0, 100.0],
                'Volume': [1_000_000, 1_000_000, 1_000_000],
            },
            index=dates,
        )
        config = self._base_config()
        config['execution']['min_rebalance_weight_delta'] = 0.03

        engine = BacktestEngine(
            strategy=TwoStepDriftStrategy(),
            data_manager=FakeDataManager(frame),
            config=config,
        )

        results = engine.run(['AAA', 'BBB'], '2024-01-02', '2024-01-04')

        # First rebalance buys two positions. Day-3 drift (L1=0.02) is below threshold and skipped.
        self.assertEqual(len(results['trades']), 2)


    def test_fill_rate_fields_absent_in_backtest_metrics(self):
        frame = pd.DataFrame(
            {
                'Open': [100.0, 100.0],
                'High': [100.0, 100.0],
                'Low': [100.0, 100.0],
                'Close': [100.0, 100.0],
                'Volume': [1_000_000, 1_000_000],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )

        results = engine.run(['AAA'], '2024-01-02', '2024-01-03')

        self.assertNotIn('fill_rate', results['metrics'])
        self.assertNotIn('avg_fill_size', results['metrics'])
    def test_required_history_aligns_effective_start_for_equity_and_benchmark(self):
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08'])
        frame = pd.DataFrame(
            {
                'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                'High': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                'Low': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                'Close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                'Volume': [1_000_000] * 6,
            },
            index=dates,
        )

        engine = BacktestEngine(
            strategy=WarmupStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )

        results = engine.run(['AAA'], '2024-01-02', '2024-01-08')
        self.assertEqual(results['effective_start_date'], '2024-01-04')
        self.assertEqual(results['required_history_days'], 2)
        self.assertEqual(results['warmup_history_shortfall_days'], 0)
        self.assertEqual(pd.Timestamp(results['equity_curve'].index[0]), pd.Timestamp('2024-01-04'))

        benchmark_equity = results['benchmark_analysis'].get('equity_curve')
        self.assertIsNotNone(benchmark_equity)
        self.assertEqual(pd.Timestamp(benchmark_equity.index[0]), pd.Timestamp('2024-01-04'))

    def test_asset_exclusions_capture_insufficient_history(self):
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
        long_frame = pd.DataFrame(
            {
                'Open': [100.0, 101.0, 102.0, 103.0],
                'High': [100.0, 101.0, 102.0, 103.0],
                'Low': [100.0, 101.0, 102.0, 103.0],
                'Close': [100.0, 101.0, 102.0, 103.0],
                'Volume': [1_000_000] * 4,
            },
            index=dates,
        )
        short_frame = long_frame.iloc[2:].copy()

        engine = BacktestEngine(
            strategy=WarmupStrategy(),
            data_manager=MultiFrameDataManager({'AAA': long_frame, 'BBB': short_frame}),
            config=self._base_config(),
        )

        results = engine.run(['AAA', 'BBB'], '2024-01-01', '2024-01-04')
        exclusions = results.get('asset_exclusions', [])
        self.assertTrue(exclusions)
        self.assertTrue(any(ex.ticker == 'BBB' and ex.reason == 'insufficient_history' for ex in exclusions))


    def test_metrics_store_writes_all_numeric_scalar_metrics(self):
        class MemoryMetricsStore:
            def __init__(self):
                self.calls = []

            def write_metric(self, name, value, timestamp, tags=None):
                self.calls.append((name, value, tags or {}))

        dates = pd.date_range('2024-01-02', periods=120, freq='B')
        close = pd.Series(range(120), index=dates, dtype=float) + 100.0
        frame = pd.DataFrame(
            {
                'Open': close.values,
                'High': close.values,
                'Low': close.values,
                'Close': close.values,
                'Volume': [1_000_000] * len(dates),
            },
            index=dates,
        )
        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )
        store = MemoryMetricsStore()
        engine.metrics_store = store

        results = engine.run(['AAA'], '2024-01-02', '2024-06-30')

        written_names = {name for name, _, _ in store.calls}
        self.assertIn('backtest.total_return', written_names)
        self.assertIn('backtest.sharpe_ratio', written_names)
        self.assertIn('backtest.num_trades', written_names)
        self.assertNotIn('backtest.regime_performance', written_names)
        self.assertNotIn('backtest.fill_rate', written_names)

        numeric_metric_count = sum(
            1 for value in results['metrics'].values()
            if isinstance(value, (int, float, bool)) and not isinstance(value, bool)
        )
        self.assertGreaterEqual(len(store.calls), numeric_metric_count - 1)

        series_metric_writes = [
            (name, tags) for name, _, tags in store.calls
            if name.startswith('backtest.rolling_sharpe_63_series.')
        ]
        self.assertTrue(series_metric_writes)

        run_id = engine.current_run_id
        self.assertTrue(run_id)
        for metric_name, tags in series_metric_writes[:5]:
            self.assertTrue(metric_name.endswith(run_id))
            self.assertEqual(tags.get('run_id'), run_id)
            self.assertEqual(tags.get('strategy'), engine.strategy.name)

    def test_results_include_pit_and_retrospective_regime_outputs(self):
        dates = pd.date_range('2024-01-02', periods=180, freq='B')
        close = pd.Series(range(180), index=dates, dtype=float) + 100.0
        frame = pd.DataFrame(
            {
                'Open': close.values,
                'High': close.values,
                'Low': close.values,
                'Close': close.values,
                'Volume': [1_000_000] * len(dates),
            },
            index=dates,
        )
        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )

        results = engine.run(['AAA'], '2024-01-02', '2024-09-30')

        self.assertIn('regime_series', results)
        self.assertIn('regime_series_pit', results)
        self.assertIsInstance(results['regime_series_pit'], pd.Series)
        self.assertIn('regime_performance', results['metrics'])
        self.assertIn('regime_performance_retrospective', results['metrics'])

    def test_per_ticker_regime_series_present_when_enabled(self):
        dates = pd.date_range('2024-01-02', periods=180, freq='B')
        rng = np.random.default_rng(123)
        rets_a = rng.normal(0.0008, 0.01, len(dates))
        rets_b = rng.normal(0.0006, 0.012, len(dates))
        close_a = pd.Series(100.0 * np.cumprod(1.0 + rets_a), index=dates, dtype=float)
        close_b = pd.Series(90.0 * np.cumprod(1.0 + rets_b), index=dates, dtype=float)
        frame_a = pd.DataFrame(
            {
                'Open': close_a.values,
                'High': close_a.values,
                'Low': close_a.values,
                'Close': close_a.values,
                'Volume': [1_000_000] * len(dates),
            },
            index=dates,
        )
        frame_b = pd.DataFrame(
            {
                'Open': close_b.values,
                'High': close_b.values,
                'Low': close_b.values,
                'Close': close_b.values,
                'Volume': [1_000_000] * len(dates),
            },
            index=dates,
        )
        config = self._base_config()
        config['risk']['per_ticker_regime'] = True
        engine = BacktestEngine(
            strategy=AlwaysLongStrategy(),
            data_manager=MultiFrameDataManager({'AAA': frame_a, 'BBB': frame_b}),
            config=config,
        )
        results = engine.run(['AAA', 'BBB'], '2024-01-02', '2024-09-30')

        self.assertIn('per_ticker_regime_series', results)
        regime_map = results['per_ticker_regime_series']
        self.assertIn('AAA', regime_map)
        self.assertIsInstance(regime_map['AAA'], pd.Series)

    def test_strategy_attribution_block_present_for_orchestration_like_strategy(self):
        dates = pd.date_range('2024-01-02', periods=80, freq='B')
        close = pd.Series(100.0 * np.cumprod(1.0 + np.random.default_rng(55).normal(0.001, 0.01, len(dates))), index=dates)
        frame = pd.DataFrame(
            {
                'Open': close.values,
                'High': close.values,
                'Low': close.values,
                'Close': close.values,
                'Volume': [1_000_000] * len(dates),
            },
            index=dates,
        )
        engine = BacktestEngine(
            strategy=AttributionStubStrategy(),
            data_manager=FakeDataManager(frame),
            config=self._base_config(),
        )
        results = engine.run(['AAA'], '2024-01-02', '2024-04-30')

        self.assertIn('attribution', results)
        self.assertIn('sub_a', results['attribution'])
        self.assertIn('weight_fraction', results['attribution']['sub_a'])


if __name__ == '__main__':
    unittest.main()
