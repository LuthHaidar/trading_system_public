import unittest
from types import SimpleNamespace

from broker.ibkr_client import IBKRClient
from utils.currency import CurrencyConverter


class _FakeIBForBatchQuotes:
    def __init__(self):
        self.sleep_calls = []
        self.req_calls = []
        self.cancel_calls = []

    def reqCurrentTime(self):
        return None

    def reqMktData(self, contract):
        self.req_calls.append(contract.symbol)
        if contract.symbol == 'AAA':
            return SimpleNamespace(last=101.25, close=100.5)
        if contract.symbol == 'BBB':
            return SimpleNamespace(last=None, close=55.0)
        return SimpleNamespace(last=None, close=None)

    def cancelMktData(self, contract):
        self.cancel_calls.append(contract.symbol)

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)


class _FakeIBForBatchQuotesWithRequestFailure(_FakeIBForBatchQuotes):
    def reqMktData(self, contract):
        self.req_calls.append(contract.symbol)
        if contract.symbol == 'BAD':
            raise RuntimeError('market data failure')
        return super().reqMktData(contract)


class IBKRContractTests(unittest.TestCase):
    def setUp(self):
        self.client = IBKRClient({'host': '127.0.0.1', 'port': 7497, 'client_id': 1})

    def test_create_contract_suffix_mappings(self):
        cases = {
            'HSBC.L': ('HSBC', 'LSE', 'GBP'),
            '7203.T': ('7203', 'TSEJ', 'JPY'),
            'CBA.AX': ('CBA', 'ASX', 'AUD'),
            '0005.HK': ('0005', 'SEHK', 'HKD'),
            'NOVN.SW': ('NOVN', 'EBS', 'CHF'),
            'SAP.DE': ('SAP', 'XETRA', 'EUR'),
            'AIR.PA': ('AIR', 'SBF', 'EUR'),
            'ASML.AS': ('ASML', 'AEB', 'EUR'),
            'D.TO': ('D', 'TSX', 'CAD'),
            'SPY': ('SPY', 'SMART', 'USD'),
        }

        for ticker, (symbol, exchange, currency) in cases.items():
            contract = self.client.create_contract(ticker)
            self.assertEqual(contract.symbol, symbol)
            self.assertEqual(contract.exchange, exchange)
            self.assertEqual(contract.currency, currency)

    def test_get_positions_maps_us_and_non_us_exchanges(self):
        self.client.connected = True
        self.client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            positions=lambda: [
                SimpleNamespace(contract=SimpleNamespace(symbol='HSBC', exchange='LSE'), position=1, avgCost=100),
                SimpleNamespace(contract=SimpleNamespace(symbol='SPY', exchange='SMART'), position=2, avgCost=200),
                SimpleNamespace(contract=SimpleNamespace(symbol='QQQ', exchange='NASDAQ'), position=3, avgCost=300),
                SimpleNamespace(contract=SimpleNamespace(symbol='DIA', exchange='NYSE'), position=4, avgCost=250),
                SimpleNamespace(contract=SimpleNamespace(symbol='SPTM', exchange='ARCA'), position=5, avgCost=50),
                SimpleNamespace(contract=SimpleNamespace(symbol='VTI', exchange='BATS'), position=6, avgCost=180),
            ],
        )

        positions = self.client.get_positions()
        self.assertIn('HSBC.L', positions)
        self.assertIn('SPY', positions)
        self.assertIn('QQQ', positions)
        self.assertIn('DIA', positions)
        self.assertIn('SPTM', positions)
        self.assertIn('VTI', positions)

    def test_create_contract_sets_primary_exchange(self):
        us_contract = self.client.create_contract('SPY')
        lse_contract = self.client.create_contract('HSBC.L')

        self.assertEqual(us_contract.primaryExchange, 'SMART')
        self.assertEqual(lse_contract.primaryExchange, 'LSE')

    def test_create_contract_applies_contract_overrides(self):
        client = IBKRClient({
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1,
            'contract_overrides': {
                'SPY': {
                    'symbol': 'SPY',
                    'exchange': 'SMART',
                    'currency': 'USD',
                    'primary_exchange': 'ARCA',
                }
            },
        })

        contract = client.create_contract('SPY')
        self.assertEqual(contract.symbol, 'SPY')
        self.assertEqual(contract.exchange, 'SMART')
        self.assertEqual(contract.currency, 'USD')
        self.assertEqual(contract.primaryExchange, 'ARCA')

    def test_get_positions_uses_primary_exchange_when_smart_routed(self):
        self.client.connected = True
        self.client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            positions=lambda: [
                SimpleNamespace(
                    contract=SimpleNamespace(symbol='HSBC', exchange='SMART', primaryExchange='LSE'),
                    position=1,
                    avgCost=100,
                ),
            ],
        )

        positions = self.client.get_positions()
        self.assertIn('HSBC.L', positions)

    def test_get_positions_unknown_exchange_warns_and_returns_bare_symbol(self):
        self.client.connected = True
        self.client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            positions=lambda: [
                SimpleNamespace(contract=SimpleNamespace(symbol='FOO', exchange='UNKNOWNX'), position=1, avgCost=10),
            ],
        )

        with self.assertLogs('broker.ibkr_client', level='WARNING') as logs:
            positions = self.client.get_positions()

        self.assertIn('FOO', positions)
        self.assertTrue(any('Unknown IBKR exchange' in entry for entry in logs.output))

    def test_get_market_prices_batches_requests_and_cancels_all(self):
        self.client.connected = True
        req_calls = []
        cancel_calls = []
        sleeps = []

        def qualify(contract):
            contract.conId = 123
            return [contract]

        def req_mkt_data(contract):
            req_calls.append(contract.symbol)
            if contract.symbol == 'SPY':
                return SimpleNamespace(last=100.0, close=99.0)
            return SimpleNamespace(last=None, close=50.0)

        self.client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            qualifyContracts=qualify,
            reqMktData=req_mkt_data,
            cancelMktData=lambda contract: cancel_calls.append(contract.symbol),
            sleep=lambda seconds: sleeps.append(seconds),
        )

        prices = self.client.get_market_prices(['SPY', 'HSBC.L'])

        self.assertEqual(prices, {'SPY': 100.0, 'HSBC.L': 50.0})
        self.assertEqual(req_calls, ['SPY', 'HSBC'])
        self.assertEqual(cancel_calls, ['SPY', 'HSBC'])
        self.assertEqual(sleeps, [2])


    def test_get_market_prices_skips_unparseable_handles_without_aborting_batch(self):
        self.client.connected = True
        cancel_calls = []

        def qualify(contract):
            contract.conId = 321
            return [contract]

        def req_mkt_data(contract):
            if contract.symbol == 'SPY':
                return None
            if contract.symbol == 'QQQ':
                return SimpleNamespace(last='bad', close='also_bad')
            return SimpleNamespace(last=25.0, close=24.0)

        self.client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            qualifyContracts=qualify,
            reqMktData=req_mkt_data,
            cancelMktData=lambda contract: cancel_calls.append(contract.symbol),
            sleep=lambda _seconds: None,
        )

        prices = self.client.get_market_prices(['SPY', 'QQQ', 'HSBC.L'])

        self.assertEqual(prices, {'HSBC.L': 25.0})
        self.assertEqual(cancel_calls, ['SPY', 'QQQ', 'HSBC'])


    def test_get_market_prices_handles_duplicate_tickers_and_cancels_each_subscription(self):
        self.client.connected = True
        req_calls = []
        cancel_calls = []

        def qualify(contract):
            contract.conId = 777
            return [contract]

        def req_mkt_data(contract):
            req_calls.append(contract.symbol)
            return SimpleNamespace(last=10.0, close=9.0)

        self.client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            qualifyContracts=qualify,
            reqMktData=req_mkt_data,
            cancelMktData=lambda contract: cancel_calls.append(contract.symbol),
            sleep=lambda _seconds: None,
        )

        prices = self.client.get_market_prices(['SPY', 'SPY'])

        self.assertEqual(prices, {'SPY': 10.0})
        self.assertEqual(req_calls, ['SPY', 'SPY'])
        self.assertEqual(cancel_calls, ['SPY', 'SPY'])

    def test_get_market_prices_continues_when_cancel_fails(self):
        self.client.connected = True
        cancel_calls = []

        def qualify(contract):
            contract.conId = 888
            return [contract]

        def req_mkt_data(contract):
            if contract.symbol == 'SPY':
                return SimpleNamespace(last=101.0, close=100.0)
            return SimpleNamespace(last=51.0, close=50.0)

        def cancel(contract):
            cancel_calls.append(contract.symbol)
            if contract.symbol == 'SPY':
                raise RuntimeError('disconnect during cancel')

        self.client.ib = SimpleNamespace(
            reqCurrentTime=lambda: None,
            qualifyContracts=qualify,
            reqMktData=req_mkt_data,
            cancelMktData=cancel,
            sleep=lambda _seconds: None,
        )

        prices = self.client.get_market_prices(['SPY', 'QQQ'])

        self.assertEqual(prices, {'SPY': 101.0, 'QQQ': 51.0})
        self.assertEqual(cancel_calls, ['SPY', 'QQQ'])

    def test_ticker_market_exchange_codes_are_unique(self):
        exchanges = [meta['exchange'] for meta in CurrencyConverter.TICKER_MARKETS.values()]
        self.assertEqual(len(exchanges), len(set(exchanges)))


    def test_get_market_prices_batches_requests_single_wait(self):
        self.client.connected = True
        fake_ib = _FakeIBForBatchQuotes()
        self.client.ib = fake_ib
        self.client._contract_cache = {
            'AAA': SimpleNamespace(symbol='AAA', exchange='SMART', currency='USD', conId=1),
            'BBB': SimpleNamespace(symbol='BBB', exchange='SMART', currency='USD', conId=1),
        }

        prices = self.client.get_market_prices(['AAA', 'BBB'])

        self.assertEqual(prices, {'AAA': 101.25, 'BBB': 55.0})
        self.assertEqual(fake_ib.req_calls, ['AAA', 'BBB'])
        self.assertEqual(fake_ib.sleep_calls, [2])
        self.assertEqual(fake_ib.cancel_calls, ['AAA', 'BBB'])

    def test_get_market_prices_continues_when_one_request_fails(self):
        self.client.connected = True
        fake_ib = _FakeIBForBatchQuotesWithRequestFailure()
        self.client.ib = fake_ib
        self.client._contract_cache = {
            'BAD': SimpleNamespace(symbol='BAD', exchange='SMART', currency='USD', conId=1),
            'AAA': SimpleNamespace(symbol='AAA', exchange='SMART', currency='USD', conId=1),
        }

        with self.assertLogs('broker.ibkr_client', level='ERROR') as logs:
            prices = self.client.get_market_prices(['BAD', 'AAA'])

        self.assertEqual(prices, {'AAA': 101.25})
        self.assertEqual(fake_ib.sleep_calls, [2])
        self.assertEqual(fake_ib.cancel_calls, ['AAA'])
        self.assertTrue(any('Error requesting market data for BAD' in entry for entry in logs.output))


if __name__ == '__main__':
    unittest.main()
