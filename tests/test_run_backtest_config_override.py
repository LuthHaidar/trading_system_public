import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from backtesting.engine import run_backtest_from_config


class RunBacktestFromConfigOverrideTests(unittest.TestCase):
    def test_config_override_is_deep_merged_before_engine_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / 'config.yaml'
            strategies_path = root / 'strategies.yaml'

            config = {
                'portfolio': {'initial_capital': 1000.0, 'currency': 'USD'},
                'execution': {'rebalance_timeframe': 'daily', 'timing': 'close'},
                'risk': {
                    'position_sizing_method': 'equal',
                    'max_position_size': 1.0,
                    'min_position_size': 0.0,
                },
                'data': {'data_dir': './data'},
                'ibkr': {'host': '127.0.0.1', 'port': 7497, 'client_id': 1},
            }
            strategies = {'momentum': {'lookback': 63, 'top_n': 2, 'skip_recent': 5}}
            config_path.write_text(yaml.safe_dump(config))
            strategies_path.write_text(yaml.safe_dump(strategies))

            captured = {}

            class DummyEngine:
                def __init__(self, strategy, data_manager, cfg):
                    captured['config'] = cfg

                def run(self, tickers, start_date, end_date):
                    return {'ok': True, 'tickers': tickers, 'start_date': start_date, 'end_date': end_date}

            with patch('backtesting.engine.BacktestEngine', DummyEngine):
                out = run_backtest_from_config(
                    config_path=str(config_path),
                    strategies_config_path=str(strategies_path),
                    strategy_name='momentum',
                    tickers=['SPY'],
                    start_date='2024-01-01',
                    end_date='2024-01-10',
                    config_override={'risk': {'max_position_size': 0.25}},
                )

            self.assertTrue(out['ok'])
            self.assertEqual(captured['config']['risk']['max_position_size'], 0.25)
            # Ensure deep merge preserved siblings under risk.
            self.assertEqual(captured['config']['risk']['min_position_size'], 0.0)


if __name__ == '__main__':
    unittest.main()
