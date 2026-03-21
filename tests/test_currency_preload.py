import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from utils.currency import CurrencyConverter


class CurrencyPreloadTests(unittest.TestCase):
    def _fx_frame(self):
        idx = pd.date_range('2024-01-01', periods=5, freq='D')
        return pd.DataFrame({'Close': [1.10, 1.11, 1.12, 1.13, 1.14]}, index=idx)

    def test_preload_pair_stores_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CurrencyConverter(base_currency='USD', fx_data_dir=tmpdir)
            with patch('utils.currency.YFINANCE_AVAILABLE', True), \
                 patch('utils.currency.yf.download', return_value=self._fx_frame()):
                converter.preload_pair('EUR', 'USD', '2024-01-01', '2024-01-10')

        self.assertIn('EURUSD', converter.fx_history)
        self.assertEqual(len(converter.fx_history['EURUSD']), 5)

    def test_preload_pair_uses_fresh_csv_cache_without_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CurrencyConverter(
                base_currency='USD',
                fx_cache_staleness_days=1,
                fx_data_dir=tmpdir,
            )
            csv_path = Path(tmpdir) / 'EURUSD_daily.csv'
            pd.DataFrame(
                {'Close': [1.20, 1.25]},
                index=pd.to_datetime(['2024-01-09', '2024-01-10'])
            ).to_csv(csv_path)

            with patch('utils.currency.YFINANCE_AVAILABLE', True), \
                 patch('utils.currency.yf.download') as download_mock:
                converter.preload_pair('EUR', 'USD', '2024-01-01', '2024-01-10')

        self.assertIn('EURUSD', converter.fx_history)
        self.assertAlmostEqual(float(converter.fx_history['EURUSD'].iloc[-1]), 1.25)
        download_mock.assert_not_called()

    def test_get_fx_rate_uses_preloaded_history_without_download(self):
        converter = CurrencyConverter(base_currency='USD')
        idx = pd.date_range('2024-01-01', periods=3, freq='D')
        converter.fx_history['EURUSD'] = pd.Series([1.1, 1.2, 1.3], index=idx)

        with patch('utils.currency.YFINANCE_AVAILABLE', True), \
             patch('utils.currency.yf.download') as download_mock:
            rate = converter.get_fx_rate('EUR', 'USD', pd.Timestamp('2024-01-02'))

        self.assertAlmostEqual(rate, 1.2)
        download_mock.assert_not_called()

    def test_get_fx_rate_before_preloaded_start_falls_back(self):
        converter = CurrencyConverter(base_currency='USD')
        idx = pd.date_range('2024-01-10', periods=2, freq='D')
        converter.fx_history['EURUSD'] = pd.Series([1.1, 1.2], index=idx)

        with patch('utils.currency.YFINANCE_AVAILABLE', True), \
             patch.object(converter, '_fetch_fx_with_retry', return_value=1.05) as fetch_mock:
            rate = converter.get_fx_rate('EUR', 'USD', pd.Timestamp('2024-01-01'))

        self.assertAlmostEqual(rate, 1.05)
        fetch_mock.assert_called_once()


    def test_preload_pair_handles_dataframe_close_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CurrencyConverter(base_currency='USD', fx_data_dir=tmpdir)
            idx = pd.date_range('2024-01-01', periods=3, freq='D')
            columns = pd.MultiIndex.from_product([['Close'], ['EURUSD=X']])
            downloaded = pd.DataFrame([[1.10], [1.20], [1.30]], index=idx, columns=columns)

            with patch('utils.currency.YFINANCE_AVAILABLE', True), \
                 patch('utils.currency.yf.download', return_value=downloaded):
                converter.preload_pair('EUR', 'USD', '2024-01-01', '2024-01-10')

        rate = converter.get_fx_rate('EUR', 'USD', pd.Timestamp('2024-01-03'))
        self.assertAlmostEqual(rate, 1.30)

    def test_get_fx_rate_handles_dataframe_history_entry(self):
        converter = CurrencyConverter(base_currency='USD')
        idx = pd.date_range('2024-01-01', periods=2, freq='D')
        converter.fx_history['EURUSD'] = pd.DataFrame({'close': [1.1, 1.2]}, index=idx)

        rate = converter.get_fx_rate('EUR', 'USD', pd.Timestamp('2024-01-02'))

        self.assertAlmostEqual(rate, 1.2)
        self.assertIsInstance(converter.fx_history['EURUSD'], pd.Series)


    def test_get_fx_rate_drops_empty_preloaded_history(self):
        converter = CurrencyConverter(base_currency='USD')
        converter.fx_history['EURUSD'] = pd.DataFrame(index=pd.DatetimeIndex([]))

        with patch('utils.currency.YFINANCE_AVAILABLE', True), \
             patch.object(converter, '_fetch_fx_with_retry', return_value=1.07):
            rate = converter.get_fx_rate('EUR', 'USD', pd.Timestamp('2024-01-01'))

        self.assertAlmostEqual(rate, 1.07)
        self.assertNotIn('EURUSD', converter.fx_history)

    def test_historical_rate_handles_multiindex_close_payload(self):
        converter = CurrencyConverter(base_currency='USD')
        idx = pd.date_range('2024-01-01', periods=3, freq='D')
        columns = pd.MultiIndex.from_product([['Close'], ['EURUSD=X']])
        downloaded = pd.DataFrame([[1.10], [1.20], [1.30]], index=idx, columns=columns)

        with patch('utils.currency.yf.download', return_value=downloaded):
            rate = converter._get_historical_or_previous_rate('EURUSD=X', pd.Timestamp('2024-01-03'))

        self.assertAlmostEqual(rate, 1.30)

    def test_convert_price_series_uses_per_date_rates(self):
        converter = CurrencyConverter(base_currency='USD')
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        prices = pd.Series([10.0, 10.0, 10.0], index=dates)
        converter.fx_history['EURUSD'] = pd.Series([1.0, 2.0, 3.0], index=dates)

        converted = converter.convert_price_series(prices, 'EUR')

        self.assertListEqual(converted.round(6).tolist(), [10.0, 20.0, 30.0])


if __name__ == '__main__':
    unittest.main()
