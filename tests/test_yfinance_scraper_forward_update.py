import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.yfinance_scraper import forward_update


def _write_cache(path: Path):
    frame = pd.DataFrame(
        {
            'Date': ['2024-01-02', '2024-01-03'],
            'Open': [100.0, 101.0],
            'High': [101.0, 102.0],
            'Low': [99.0, 100.0],
            'Close': [100.0, 101.0],
            'Adj Close': [100.0, 101.0],
            'Volume': [1000, 1000],
        }
    )
    frame.to_csv(path, index=False)


def test_forward_update_appends_and_deduplicates():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fp = root / 'AAA_daily_data.csv'
        _write_cache(fp)

        fetched = pd.DataFrame(
            {
                'Open': [111.0, 112.0],
                'High': [112.0, 113.0],
                'Low': [110.0, 111.0],
                'Close': [111.0, 112.0],
                'Adj Close': [111.0, 112.0],
                'Volume': [1000, 1000],
            },
            index=pd.to_datetime(['2024-01-03', '2024-01-04'])
        )

        with patch('data.yfinance_scraper.yf.download', return_value=fetched):
            combined = forward_update('AAA', str(root), from_date=pd.Timestamp('2024-01-02').date(), to_date=pd.Timestamp('2024-01-04').date())

        assert len(combined) == 3
        assert pd.Timestamp('2024-01-04') in combined.index
        assert float(combined.loc[pd.Timestamp('2024-01-03'), 'Close']) == 111.0


def test_forward_update_no_new_data_returns_existing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fp = root / 'AAA_daily_data.csv'
        _write_cache(fp)

        with patch('data.yfinance_scraper.yf.download', return_value=pd.DataFrame()):
            combined = forward_update('AAA', str(root), from_date=pd.Timestamp('2024-01-03').date(), to_date=pd.Timestamp('2024-01-05').date())

        assert len(combined) == 2


def test_forward_update_persists_overlap_only_revisions_to_disk():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fp = root / 'AAA_daily_data.csv'
        _write_cache(fp)

        fetched = pd.DataFrame(
            {
                'Open': [111.0],
                'High': [112.0],
                'Low': [110.0],
                'Close': [111.0],
                'Adj Close': [111.0],
                'Volume': [1000],
            },
            index=pd.to_datetime(['2024-01-03'])
        )

        with patch('data.yfinance_scraper.yf.download', return_value=fetched):
            combined = forward_update(
                'AAA',
                str(root),
                from_date=pd.Timestamp('2024-01-02').date(),
                to_date=pd.Timestamp('2024-01-03').date(),
            )

        assert len(combined) == 2
        assert float(combined.loc[pd.Timestamp('2024-01-03'), 'Close']) == 111.0

        persisted = pd.read_csv(fp)
        persisted['Date'] = pd.to_datetime(persisted['Date'])
        row = persisted.loc[persisted['Date'] == pd.Timestamp('2024-01-03')].iloc[0]
        assert float(row['Close']) == 111.0
