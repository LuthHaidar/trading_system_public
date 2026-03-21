import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from data.data_manager import DataManager


class DataManagerAdjustedCloseTests(unittest.TestCase):
    def _write_sample_csv(self, root: Path, ticker: str = 'AAA') -> None:
        frame = pd.DataFrame(
            {
                'Date': ['2024-01-02', '2024-01-03'],
                'Open': [100.0, 100.0],
                'High': [101.0, 101.0],
                'Low': [99.0, 99.0],
                'Close': [100.0, 105.0],
                'Adj Close': [98.0, 103.0],
                'Volume': [1_000_000, 1_000_000],
            }
        )
        frame.to_csv(root / f'{ticker}_daily_data.csv', index=False)

    def test_load_ticker_uses_adjusted_close_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_csv(root)
            manager = DataManager(data_dir=str(root), use_adjusted_close=True)

            loaded = manager.load_ticker('AAA', use_cache=False)

            self.assertAlmostEqual(float(loaded.iloc[0]['Close']), 98.0, places=8)
            self.assertAlmostEqual(float(loaded.iloc[1]['Close']), 103.0, places=8)

    def test_load_ticker_preserves_raw_close_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_csv(root)
            manager = DataManager(data_dir=str(root), use_adjusted_close=False)

            loaded = manager.load_ticker('AAA', use_cache=False)

            self.assertAlmostEqual(float(loaded.iloc[0]['Close']), 100.0, places=8)
            self.assertAlmostEqual(float(loaded.iloc[1]['Close']), 105.0, places=8)


class DataManagerCacheInvalidationTests(unittest.TestCase):
    def test_invalidate_cache_is_idempotent_for_missing_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = DataManager(data_dir=str(Path(tmp)))

            # Should not raise when ticker is not cached.
            manager._invalidate_cache('MISSING')
            manager._invalidate_cache('MISSING')

            self.assertEqual(manager.cache, {})
            self.assertEqual(manager.last_update, {})

    def test_update_all_invalidates_entries_without_key_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame(
                {
                    'Date': ['2024-01-02'],
                    'Open': [100.0],
                    'High': [101.0],
                    'Low': [99.0],
                    'Close': [100.0],
                    'Adj Close': [100.0],
                    'Volume': [1_000_000],
                }
            )
            frame.to_csv(root / 'AAA_daily_data.csv', index=False)

            manager = DataManager(data_dir=str(root), use_adjusted_close=True)
            manager.load_ticker('AAA', use_cache=True)
            self.assertIn('AAA', manager.cache)

            with patch('data.data_manager.update_all_tickers', return_value={'AAA': True, 'BBB': False}):
                results = manager.update_all(['AAA', 'BBB'])

            self.assertEqual(results, {'AAA': True, 'BBB': False})
            self.assertNotIn('AAA', manager.cache)
            self.assertNotIn('AAA', manager.last_update)
            self.assertNotIn('BBB', manager.cache)
            self.assertNotIn('BBB', manager.last_update)


if __name__ == '__main__':
    unittest.main()


class DataManagerFreshnessTests(unittest.TestCase):
    def _write_stale_csv(self, root: Path, ticker: str = 'AAA', end_date: str = '2024-01-03') -> None:
        frame = pd.DataFrame(
            {
                'Date': ['2024-01-02', end_date],
                'Open': [100.0, 101.0],
                'High': [101.0, 102.0],
                'Low': [99.0, 100.0],
                'Close': [100.0, 101.0],
                'Adj Close': [100.0, 101.0],
                'Volume': [1_000_000, 1_000_000],
            }
        )
        frame.to_csv(root / f'{ticker}_daily_data.csv', index=False)

    def test_stale_cache_triggers_forward_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_stale_csv(root)
            manager = DataManager(data_dir=str(root), freshness_threshold_days=1)

            updated = pd.DataFrame(
                {
                    'Open': [100.0, 101.0, 102.0],
                    'High': [101.0, 102.0, 103.0],
                    'Low': [99.0, 100.0, 101.0],
                    'Close': [100.0, 101.0, 102.0],
                    'Adj Close': [100.0, 101.0, 102.0],
                    'Volume': [1_000_000, 1_000_000, 1_000_000],
                },
                index=pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04'])
            )
            with patch('data.data_manager.forward_update', return_value=updated) as mock_forward:
                loaded = manager.load_ticker('AAA', end_date='2024-01-10', use_cache=False)

            self.assertEqual(len(loaded), 3)
            mock_forward.assert_called_once()

    def test_fresh_cache_skips_forward_update_and_logs_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_stale_csv(root, end_date='2024-01-09')
            manager = DataManager(data_dir=str(root), freshness_threshold_days=3)

            with patch('data.data_manager.forward_update') as mock_forward:
                with self.assertLogs('data.data_manager', level='INFO') as logs:
                    loaded = manager.load_ticker('AAA', end_date='2024-01-10', use_cache=False)

            self.assertEqual(len(loaded), 2)
            mock_forward.assert_not_called()
            self.assertTrue(any('cache ends' in line and 'delta=' in line for line in logs.output))
