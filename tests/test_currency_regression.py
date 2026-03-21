import unittest
from unittest.mock import patch

from utils.currency import CurrencyConverter


class CurrencyRegressionTests(unittest.TestCase):
    def test_fx_latest_cache_hit_avoids_refetch(self):
        converter = CurrencyConverter(base_currency='USD')
        converter.fx_cache_ttl_seconds = 3600

        with patch('utils.currency.YFINANCE_AVAILABLE', True):
            with patch.object(converter, '_get_latest_fx_rate', return_value=1.25) as fetch_latest:
                first = converter.get_fx_rate('EUR', 'USD')
                second = converter.get_fx_rate('EUR', 'USD')

        self.assertAlmostEqual(first, 1.25, places=8)
        self.assertAlmostEqual(second, 1.25, places=8)
        self.assertEqual(fetch_latest.call_count, 1)

    def test_fx_retry_exhaustion_raises_explicit_exception(self):
        converter = CurrencyConverter(base_currency='USD')
        converter.fx_retry_attempts = 3
        converter.fx_retry_backoff_seconds = 0.01

        with patch('utils.currency.YFINANCE_AVAILABLE', True):
            with patch.object(converter, '_get_latest_fx_rate', side_effect=TimeoutError('timeout')):
                with patch('utils.currency.time.sleep') as sleep_mock:
                    with self.assertRaises(RuntimeError):
                        converter.get_fx_rate('EUR', 'USD')

        # 3 attempts => sleep between attempt 1->2 and 2->3
        self.assertEqual(sleep_mock.call_count, 2)


if __name__ == '__main__':
    unittest.main()
