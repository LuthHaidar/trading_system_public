import sys
import types
import unittest
import importlib
from unittest.mock import patch
from datetime import datetime

import pandas as pd


def _install_schedule_stub():
    if 'schedule' in sys.modules:
        return

    schedule_module = types.ModuleType('schedule')

    class _Every:
        @property
        def day(self):
            return self

        def at(self, _time):
            return self

        def do(self, *args, **kwargs):
            return None

    schedule_module.every = lambda: _Every()
    schedule_module.run_pending = lambda: None
    sys.modules['schedule'] = schedule_module


def _install_ibkr_stub():
    if 'broker.ibkr_client' in sys.modules:
        return

    # Prefer the real module when available so this test file does not poison
    # global module state for other tests that require the actual IBKR client.
    try:
        importlib.import_module('broker.ibkr_client')
        return
    except Exception:
        pass

    broker_module = types.ModuleType('broker.ibkr_client')

    class IBKRClient:  # pragma: no cover - stub for import only
        pass

    broker_module.IBKRClient = IBKRClient
    sys.modules['broker.ibkr_client'] = broker_module


_install_schedule_stub()
_install_ibkr_stub()

from main import (
    LiveTradingEngine,
    _load_contract_overrides_from_strategies,
    _resolve_strategy_config,
    evaluate_pre_flight_summary,
    load_strategy,
    run_pre_flight_backtest,
)
import main as main_module


class DummyRiskManager:
    def __init__(self):
        self.reject_count = 0
        self.reset_calls = []
        self.kill_switch = False
        self.kill_switch_reason = None
        self.last_reset_date = None

    def record_order_reject(self):
        self.reject_count += 1

    def reset_daily(self, equity, reset_date=None, clear_kill_switch=True):
        self.reset_calls.append((equity, reset_date, clear_kill_switch))
        self.last_reset_date = reset_date

    def snapshot_state(self):
        return {
            'reject_count': self.reject_count,
            'kill_switch': self.kill_switch,
            'kill_switch_reason': self.kill_switch_reason,
            'last_reset_date': str(self.last_reset_date) if self.last_reset_date else None,
        }

    def restore_state(self, state):
        self.reject_count = int(state.get('reject_count', 0))
        self.kill_switch = bool(state.get('kill_switch', False))
        self.kill_switch_reason = state.get('kill_switch_reason')
        raw = state.get('last_reset_date')
        self.last_reset_date = pd.Timestamp(raw).date() if raw else None


class DummyLiveEvalRiskManager(DummyRiskManager):
    def mark_data_heartbeat(self, _timestamp):
        return None

    def evaluate(self, _equity, _now):
        return types.SimpleNamespace(triggered=False, reason=None, metadata=None)


class DummyIBKR:
    def __init__(self):
        self._position_call = 0
        self.rebalance_calls = 0
        self.connected = True

    def get_positions(self):
        self._position_call += 1
        if self._position_call == 1:
            return {'AAA': {'shares': 8}}
        return {'AAA': {'shares': 10}}

    def execute_rebalance(self, target_positions):
        self.rebalance_calls += 1
        return True

    def get_account_summary(self):
        return {'net_liquidation': 1000.0}

    def is_connected(self):
        return self.connected


class DummyAuditStore:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.events = []

    def load_latest_state(self, source=None, strategy=None):
        return self.snapshot

    def log_event(self, **kwargs):
        self.events.append(kwargs)


class LiveExecutionRegressionTests(unittest.TestCase):

    def test_live_engine_prefers_execution_timing_key(self):
        import unittest.mock as mock

        config = {
            'data': {'data_dir': 'data'},
            'ibkr': {},
            'portfolio': {'currency': 'USD'},
            'risk': {},
            'execution': {'timing': '15:45', 'time': '09:30'},
            'operations': {},
            'live_risk': {},
            'data_platform': {'enabled': False},
        }

        strategy = types.SimpleNamespace(name='test_strategy')

        with (
            mock.patch.object(main_module, 'DataManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'IBKRClient', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'PositionSizer', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'RiskConstraints', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'CurrencyConverter', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'LiveRiskManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'StateBackupManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'PositionDriftMonitor', return_value=types.SimpleNamespace()),
        ):
            engine = LiveTradingEngine(config=config, strategy=strategy, dry_run=True)

        self.assertEqual(engine.execution_time, '15:45')

    def test_live_engine_maps_close_timing_to_market_close_time(self):
        import unittest.mock as mock

        config = {
            'data': {'data_dir': 'data'},
            'ibkr': {},
            'portfolio': {'currency': 'USD'},
            'risk': {},
            'execution': {'timing': 'close'},
            'operations': {},
            'live_risk': {},
            'data_platform': {'enabled': False},
        }

        strategy = types.SimpleNamespace(name='test_strategy')

        with (
            mock.patch.object(main_module, 'DataManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'IBKRClient', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'PositionSizer', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'RiskConstraints', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'CurrencyConverter', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'LiveRiskManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'StateBackupManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'PositionDriftMonitor', return_value=types.SimpleNamespace()),
        ):
            engine = LiveTradingEngine(config=config, strategy=strategy, dry_run=True)

        self.assertEqual(engine.execution_time, '16:00')

    def test_live_engine_falls_back_to_legacy_time_for_unsupported_timing(self):
        import unittest.mock as mock

        config = {
            'data': {'data_dir': 'data'},
            'ibkr': {},
            'portfolio': {'currency': 'USD'},
            'risk': {},
            'execution': {'timing': 'nonsense', 'time': '14:20'},
            'operations': {},
            'live_risk': {},
            'data_platform': {'enabled': False},
        }

        strategy = types.SimpleNamespace(name='test_strategy')

        with (
            mock.patch.object(main_module, 'DataManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'IBKRClient', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'PositionSizer', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'RiskConstraints', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'CurrencyConverter', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'LiveRiskManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'StateBackupManager', return_value=types.SimpleNamespace()),
            mock.patch.object(main_module, 'PositionDriftMonitor', return_value=types.SimpleNamespace()),
        ):
            engine = LiveTradingEngine(config=config, strategy=strategy, dry_run=True)

        self.assertEqual(engine.execution_time, '14:20')

    def test_scheduled_maintenance_prunes_audit_events_when_available(self):
        captured = {'days': None, 'events': []}

        class _AuditStore:
            def prune_audit_events(self, older_than_days=90):
                captured['days'] = older_than_days
                return 7

            def log_event(self, **kwargs):
                captured['events'].append(kwargs)

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.audit_store = _AuditStore()
        engine.audit_retention_days = 45
        engine.strategy = types.SimpleNamespace(name='test_strategy')

        engine._run_scheduled_maintenance()

        self.assertEqual(captured['days'], 45)
        self.assertTrue(captured['events'])
        self.assertEqual(captured['events'][0].get('event_type'), 'maintenance_prune_completed')

    def test_scheduled_maintenance_noop_without_audit_store(self):
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.audit_store = None
        engine.audit_retention_days = 90
        engine.strategy = types.SimpleNamespace(name='test_strategy')

        engine._run_scheduled_maintenance()

    def test_partial_fill_drift_triggers_retry_and_recovers(self):
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.ibkr = DummyIBKR()
        engine.live_risk_manager = DummyRiskManager()
        engine.audit_store = None
        engine.strategy = types.SimpleNamespace(name='test_strategy')

        ok = engine._verify_post_execution_positions({'AAA': 10}, max_retries=1)

        self.assertTrue(ok)
        self.assertEqual(engine.live_risk_manager.reject_count, 1)
        self.assertEqual(engine.ibkr.rebalance_calls, 1)

    def test_validate_data_freshness_rejects_stale_history(self):
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.config = {'live_risk': {'max_data_staleness_days': 1}}

        stale_df = pd.DataFrame(
            {'Close': [100.0]},
            index=pd.to_datetime(['2020-01-01']),
        )
        with self.assertRaises(RuntimeError):
            engine._validate_data_freshness({'AAA': stale_df}, ['AAA'])

    def test_run_once_resets_daily_risk_before_execution(self):
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.ibkr = DummyIBKR()
        engine.live_risk_manager = DummyRiskManager()
        engine._live_risk_reset_date = None

        calls = []
        engine.recover_from_persisted_state = lambda: calls.append('recover')
        engine.execute_strategy = lambda tickers: calls.append(('execute', tuple(tickers)))
        engine.connect = lambda: True

        engine.run_once(['AAA'])

        self.assertEqual(len(engine.live_risk_manager.reset_calls), 1)
        self.assertEqual(calls[0], 'recover')
        self.assertEqual(calls[1], ('execute', ('AAA',)))

    def test_recover_from_snapshot_restores_same_day_risk_state(self):
        today = datetime.now().date().isoformat()
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'risk_state': {
                'reject_count': 2,
                'kill_switch': True,
                'last_reset_date': today,
            },
            'target_positions': {},
        }

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.audit_store = DummyAuditStore(snapshot)
        engine.live_risk_manager = DummyRiskManager()
        engine.strategy = types.SimpleNamespace(name='test_strategy')
        engine.ibkr = DummyIBKR()
        engine.drift_monitor = types.SimpleNamespace(check=lambda a, b: {})

        engine.recover_from_persisted_state()

        self.assertEqual(engine.live_risk_manager.reject_count, 2)
        self.assertTrue(engine.live_risk_manager.kill_switch)

    def test_optimizer_failure_halts_live_execution_when_configured(self):
        today = pd.Timestamp(datetime.now().date())
        history = pd.DataFrame(
            {'Close': [100.0, 101.0]},
            index=pd.to_datetime([today - pd.Timedelta(days=1), today]),
        )
        events = []
        rebalance_calls = {'count': 0}

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.config = {
            'portfolio': {'currency': 'USD'},
            'risk': {'optimizer_method': 'max_sharpe'},
            'live_risk': {'halt_on_optimization_failure': True, 'max_data_staleness_days': 1},
        }
        engine.strategy = types.SimpleNamespace(
            name='test_strategy',
            generate_signals=lambda date, data, current_positions: {'AAA': 1.0},
        )
        engine.dry_run = False
        engine.data_manager = types.SimpleNamespace(
            update_all=lambda tickers: {t: True for t in tickers},
            load_multiple=lambda tickers: {'AAA': history},
        )
        engine.ibkr = types.SimpleNamespace(
            get_positions=lambda: {},
            get_market_prices=lambda tickers: {'AAA': 100.0},
            get_account_summary=lambda: {'net_liquidation': 1000.0, 'currency': 'USD'},
            execute_rebalance=lambda target_positions: rebalance_calls.__setitem__('count', rebalance_calls['count'] + 1) or True,
        )
        engine.live_risk_manager = DummyLiveEvalRiskManager()
        engine.fx_converter = types.SimpleNamespace(
            get_ticker_currency=lambda ticker: 'USD',
            convert_to_base=lambda value, currency: value,
            get_fx_rate=lambda from_ccy, to_ccy: 1.0,
        )
        engine.position_sizer = types.SimpleNamespace(
            size_positions=lambda signals, data, equity: signals
        )
        engine.risk_constraints = types.SimpleNamespace(
            apply_constraints=lambda weights, data: weights
        )
        engine.use_optimizer = True
        engine.optimizer = types.SimpleNamespace(
            optimize=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('optimizer boom'))
        )
        engine.audit_store = types.SimpleNamespace(log_event=lambda **kwargs: events.append(kwargs))
        engine.metrics_store = None
        engine.backup_manager = types.SimpleNamespace(backup=lambda payload: 'noop')
        engine.drift_monitor = types.SimpleNamespace(check=lambda target, live: {})

        engine.execute_strategy(['AAA'])

        self.assertTrue(engine.live_risk_manager.kill_switch)
        self.assertEqual(engine.live_risk_manager.reject_count, 1)
        self.assertEqual(rebalance_calls['count'], 0)
        self.assertTrue(any(e.get('event_type') == 'optimization_failed' for e in events))

    def test_position_level_stop_forces_exit_even_without_signals(self):
        today = pd.Timestamp(datetime.now().date())
        history = pd.DataFrame(
            {'Close': [100.0, 100.0]},
            index=pd.to_datetime([today - pd.Timedelta(days=1), today]),
        )

        captured = {'target_positions': None, 'orders': None}

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.config = {
            'portfolio': {'currency': 'USD'},
            'risk': {},
            'live_risk': {
                'max_data_staleness_days': 1,
                'position_stop_loss_pct': 0.10,
            },
        }
        engine.strategy = types.SimpleNamespace(
            name='test_strategy',
            generate_signals=lambda date, data, current_positions: {},
        )
        engine.dry_run = True
        engine.data_manager = types.SimpleNamespace(
            update_all=lambda tickers: {t: True for t in tickers},
            load_multiple=lambda tickers: {'AAA': history, 'BBB': history},
        )
        engine.ibkr = types.SimpleNamespace(
            get_positions=lambda: {
                'AAA': {'shares': 10, 'avg_cost': 100.0},
                'BBB': {'shares': 5, 'avg_cost': 100.0},
            },
            get_market_prices=lambda tickers: {'AAA': 85.0, 'BBB': 101.0},
            get_account_summary=lambda: {'net_liquidation': 1000.0, 'currency': 'USD'},
            reconcile_positions=lambda target_positions: captured.update({'target_positions': dict(target_positions)}) or [('AAA', 10, 'SELL')],
        )
        engine.live_risk_manager = DummyLiveEvalRiskManager()
        engine.fx_converter = types.SimpleNamespace(
            get_ticker_currency=lambda ticker: 'USD',
            convert_to_base=lambda value, currency: value,
            get_fx_rate=lambda from_ccy, to_ccy: 1.0,
        )
        engine.position_sizer = types.SimpleNamespace(
            size_positions=lambda signals, data, equity: signals
        )
        engine.risk_constraints = types.SimpleNamespace(
            apply_constraints=lambda weights, data: weights
        )
        engine.use_optimizer = False
        engine.optimizer = None
        engine.audit_store = None
        engine.metrics_store = None
        engine.backup_manager = types.SimpleNamespace(backup=lambda payload: 'noop')
        engine.drift_monitor = types.SimpleNamespace(check=lambda target, live: {})
        engine._position_high_water = {}

        engine.execute_strategy(['AAA', 'BBB'])

        self.assertIsNotNone(captured['target_positions'])
        self.assertNotIn('AAA', captured['target_positions'])
        self.assertEqual(captured['target_positions'].get('BBB'), 5)

    def test_persist_runtime_state_uses_backup_manager_without_audit_store(self):
        captured = {'payload': None}

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.strategy = types.SimpleNamespace(name='test_strategy')
        engine.live_risk_manager = DummyRiskManager()
        engine.audit_store = None
        engine.backup_manager = types.SimpleNamespace(
            backup_runtime_state=lambda payload: captured.__setitem__('payload', payload) or 'runtime.json'
        )
        engine._position_high_water = {'AAA': 123.0}

        engine._persist_runtime_state(extra={'reason': 'unit_test'})

        self.assertIsNotNone(captured['payload'])
        self.assertIn('risk_state', captured['payload'])
        self.assertIn('position_high_water', captured['payload'])
        self.assertEqual(captured['payload']['context']['reason'], 'unit_test')

    def test_recover_from_backup_runtime_state_without_audit_store(self):
        today = datetime.now().date().isoformat()
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'risk_state': {
                'reject_count': 3,
                'kill_switch': True,
                'kill_switch_reason': 'reject_count_breaker',
                'last_reset_date': today,
            },
            'position_high_water': {'AAA': 150.0},
        }

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.audit_store = None
        engine.backup_manager = types.SimpleNamespace(load_latest_runtime_state=lambda: snapshot)
        engine.live_risk_manager = DummyRiskManager()
        engine.strategy = types.SimpleNamespace(name='test_strategy')
        engine.ibkr = DummyIBKR()
        engine.drift_monitor = types.SimpleNamespace(check=lambda a, b: {})

        engine.recover_from_persisted_state()

        self.assertEqual(engine.live_risk_manager.reject_count, 3)
        self.assertTrue(engine.live_risk_manager.kill_switch)
        self.assertEqual(engine.live_risk_manager.kill_switch_reason, 'reject_count_breaker')
        self.assertEqual(engine._position_high_water.get('AAA'), 150.0)

    def test_pre_flight_backtest_fails_when_no_trades(self):
        with patch('main.run_backtest_from_config', return_value={'trades': [], 'metrics': {'sharpe_ratio': 1.0}}):
            ok = run_pre_flight_backtest(
                config_path='config/config.yaml',
                strategy_name='momentum',
                tickers=['AAA'],
            )
        self.assertEqual(ok['num_trades'], 0)

    def test_pre_flight_backtest_fails_when_sharpe_negative(self):
        mock_results = {
            'trades': [object()],
            'metrics': {'sharpe_ratio': -0.1},
        }
        with patch('main.run_backtest_from_config', return_value=mock_results):
            ok = run_pre_flight_backtest(
                config_path='config/config.yaml',
                strategy_name='momentum',
                tickers=['AAA'],
            )
        self.assertLess(ok['sharpe_ratio'], 0)

    def test_pre_flight_backtest_passes_with_trades_and_non_negative_sharpe(self):
        mock_results = {
            'trades': [object()],
            'metrics': {'sharpe_ratio': 0.2},
        }
        with patch('main.run_backtest_from_config', return_value=mock_results):
            ok = run_pre_flight_backtest(
                config_path='config/config.yaml',
                strategy_name='momentum',
                tickers=['AAA'],
            )
        self.assertGreater(ok['num_trades'], 0)
        self.assertGreaterEqual(ok['sharpe_ratio'], 0)

    def test_pre_flight_backtest_prefers_trade_list_over_metric_count(self):
        mock_results = {
            'trades': [object(), object()],
            'metrics': {'num_trades': 0, 'sharpe_ratio': 0.5},
        }
        with patch('main.run_backtest_from_config', return_value=mock_results):
            ok = run_pre_flight_backtest(
                config_path='config/config.yaml',
                strategy_name='momentum',
                tickers=['AAA'],
            )
        self.assertEqual(ok['num_trades'], 2)

    def test_evaluate_pre_flight_summary_rejects_nan_sharpe(self):
        passed, reason = evaluate_pre_flight_summary({'num_trades': 2, 'sharpe_ratio': float('nan')})
        self.assertFalse(passed)
        self.assertIn('NaN', reason)

    def test_evaluate_pre_flight_summary_rejects_invalid_trade_count(self):
        passed, reason = evaluate_pre_flight_summary({'num_trades': 'many', 'sharpe_ratio': 0.1})
        self.assertFalse(passed)
        self.assertIn('num_trades', reason)

    def test_evaluate_pre_flight_summary_rejects_non_numeric_sharpe(self):
        passed, reason = evaluate_pre_flight_summary({'num_trades': 2, 'sharpe_ratio': 'bad'})
        self.assertFalse(passed)
        self.assertIn('Sharpe', reason)



    def test_validate_universe_non_strict_filters_failed_tickers(self):
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.ibkr = types.SimpleNamespace(validate_universe=lambda tickers: {'AAA': True, 'BBB': False, 'CCC': True})
        engine.config = {'ibkr': {'strict_universe_validation': False}}

        active = engine.validate_universe(['AAA', 'BBB', 'CCC'])

        self.assertEqual(active, ['AAA', 'CCC'])

    def test_validate_universe_strict_returns_empty_on_failures(self):
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.ibkr = types.SimpleNamespace(validate_universe=lambda tickers: {'AAA': True, 'BBB': False})
        engine.config = {'ibkr': {'strict_universe_validation': True}}

        active = engine.validate_universe(['AAA', 'BBB'])

        self.assertEqual(active, [])




    def test_load_contract_overrides_from_strategies_honors_contracts_section(self):
        import tempfile
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/strategies.yaml"
            with open(path, 'w') as f:
                yaml.safe_dump({
                    'momentum': {'enabled': True},
                    'contracts': {
                        'HSBC.L': {'symbol': 'HSBC', 'exchange': 'LSE', 'currency': 'GBP'},
                        'SPY': {'symbol': 'SPY', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'ARCA'},
                    },
                }, f)

            overrides = _load_contract_overrides_from_strategies(path)

        self.assertEqual(overrides['HSBC.L']['exchange'], 'LSE')
        self.assertEqual(overrides['SPY']['primary_exchange'], 'ARCA')

    def test_live_engine_merges_strategy_contracts_with_ibkr_overrides(self):
        import tempfile
        import yaml
        import unittest.mock as mock

        config = {
            'data': {'data_dir': 'data'},
            'ibkr': {
                'contract_overrides': {
                    'SPY': {'symbol': 'SPY', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'NYSE'}
                }
            },
            'portfolio': {'currency': 'USD'},
            'risk': {},
            'execution': {'timing': '15:45'},
            'operations': {},
            'live_risk': {},
            'data_platform': {'enabled': False},
        }
        strategy = types.SimpleNamespace(name='test_strategy')

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/strategies.yaml"
            with open(path, 'w') as f:
                yaml.safe_dump({
                    'momentum': {'enabled': True},
                    'contracts': {
                        'HSBC.L': {'symbol': 'HSBC', 'exchange': 'LSE', 'currency': 'GBP'},
                        'SPY': {'symbol': 'SPY', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'ARCA'},
                    },
                }, f)
            config['strategies_config_path'] = path

            captured = {}
            def _ibkr_ctor(cfg):
                captured['cfg'] = cfg
                return types.SimpleNamespace()

            with (
                mock.patch.object(main_module, 'DataManager', return_value=types.SimpleNamespace()),
                mock.patch.object(main_module, 'IBKRClient', side_effect=_ibkr_ctor),
                mock.patch.object(main_module, 'PositionSizer', return_value=types.SimpleNamespace()),
                mock.patch.object(main_module, 'RiskConstraints', return_value=types.SimpleNamespace()),
                mock.patch.object(main_module, 'CurrencyConverter', return_value=types.SimpleNamespace()),
                mock.patch.object(main_module, 'LiveRiskManager', return_value=types.SimpleNamespace()),
                mock.patch.object(main_module, 'StateBackupManager', return_value=types.SimpleNamespace()),
                mock.patch.object(main_module, 'PositionDriftMonitor', return_value=types.SimpleNamespace()),
            ):
                LiveTradingEngine(config=config, strategy=strategy, dry_run=True)

        merged = captured['cfg']['contract_overrides']
        self.assertIn('HSBC.L', merged)
        self.assertEqual(merged['HSBC.L']['exchange'], 'LSE')
        # ibkr.contract_overrides wins precedence over strategies contracts for duplicate tickers
        self.assertEqual(merged['SPY']['primary_exchange'], 'NYSE')

    def test_load_strategy_uses_configured_strategies_config_path(self):
        import tempfile
        import yaml
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/strategies.yaml"
            with open(path, 'w') as f:
                yaml.safe_dump({'momentum': {'enabled': True, 'lookback': 63}}, f)

            with (
                mock.patch.object(main_module, 'create_strategy', return_value='dummy_strategy') as create_mock,
                mock.patch.object(main_module, 'get_available_strategies', return_value=['momentum']),
            ):
                strategy = load_strategy('momentum', {'strategies_config_path': path})

            self.assertEqual(strategy, 'dummy_strategy')
            create_mock.assert_called_once()

    def test_resolve_strategy_config_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve_strategy_config('definitely_unknown', {'momentum': {'enabled': True}})
        self.assertIn('Unknown strategy', str(ctx.exception))

    def test_resolve_strategy_config_rejects_missing_yaml_stanza(self):
        # use a known strategy name but omit it from strategies config
        with self.assertRaises(ValueError) as ctx:
            _resolve_strategy_config('momentum', {'mean_reversion': {'enabled': True}})
        self.assertIn('missing in config/strategies.yaml', str(ctx.exception))


    def test_resolve_strategy_config_missing_message_uses_supplied_path(self):
        custom_path = '/tmp/custom_strategies.yaml'
        with self.assertRaises(ValueError) as ctx:
            _resolve_strategy_config(
                'momentum',
                {'mean_reversion': {'enabled': True}},
                strategies_config_path=custom_path,
            )
        self.assertIn(custom_path, str(ctx.exception))

    def test_resolve_strategy_config_injects_orchestration_configs(self):
        cfg = {
            'strategy_orchestration': {'enabled': True, 'base_weights': {'momentum': 1.0}},
            'momentum': {'enabled': True, 'lookback': 63},
        }
        resolved = _resolve_strategy_config('strategy_orchestration', cfg)
        self.assertIn('_all_strategies_config', resolved)
        self.assertEqual(resolved['_all_strategies_config'], cfg)

    def test_main_preflight_backtest_blocks_startup_on_failure(self):
        import unittest.mock as mock

        argv = ['main.py', '--once', '--dry-run', '--pre-flight-backtest']

        class DummyEngine:
            def __init__(self, *args, **kwargs):
                raise AssertionError('Engine should not be constructed on pre-flight failure')

        with (
            mock.patch.object(sys, 'argv', argv),
            mock.patch.object(
                main_module,
                'run_pre_flight_backtest',
                return_value={
                    'num_trades': 0,
                    'sharpe_ratio': 1.0,
                    'start_date': '2024-01-01',
                    'end_date': '2024-04-01',
                },
            ),
            mock.patch.object(main_module, 'LiveTradingEngine', DummyEngine),
        ):
            rc = main_module.main()

        self.assertEqual(rc, 1)

    def test_main_preflight_backtest_allows_startup_on_pass(self):
        import unittest.mock as mock

        argv = ['main.py', '--once', '--dry-run', '--pre-flight-backtest']
        calls = {'connect': 0, 'run_once': 0, 'disconnect': 0}

        class DummyEngine:
            def __init__(self, *args, **kwargs):
                return None

            def connect(self):
                calls['connect'] += 1
                return True

            def validate_universe(self, tickers):
                return list(tickers)

            def run_once(self, tickers):
                calls['run_once'] += 1

            def disconnect(self):
                calls['disconnect'] += 1

        with (
            mock.patch.object(sys, 'argv', argv),
            mock.patch.object(
                main_module,
                'run_pre_flight_backtest',
                return_value={
                    'num_trades': 3,
                    'sharpe_ratio': 0.5,
                    'start_date': '2024-01-01',
                    'end_date': '2024-04-01',
                },
            ),
            mock.patch.object(main_module, 'LiveTradingEngine', DummyEngine),
        ):
            rc = main_module.main()

        self.assertEqual(rc, 0)
        self.assertEqual(calls['connect'], 1)
        self.assertEqual(calls['run_once'], 1)
        self.assertEqual(calls['disconnect'], 1)


if __name__ == '__main__':
    unittest.main()
