import os
from datetime import datetime, timedelta
from datetime import date as date_type

import pandas as pd
from utils.logger import get_logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

logger = get_logger(__name__)


def load_existing_data(filepath):
    """Load existing CSV data if available."""
    if not os.path.exists(filepath):
        return None, None, None

    try:
        df = pd.read_csv(filepath)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'Date' not in df.columns:
            if df.index.name == 'Date' or 'date' in str(df.index.name).lower():
                df.reset_index(inplace=True)
            else:
                logger.warning(f"No Date column found in {filepath}")
                return None, None, None

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        if len(df) == 0:
            return None, None, None

        return df, df.index[-1], df.index[0]
    except Exception as e:
        logger.warning(f"Could not read existing file {filepath}: {e}")
        return None, None, None


def download_ticker_data(ticker, data_dir, force_full=False, required_start_date=None):
    """Download historical data for a ticker, appending to existing data if available."""
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not available - cannot download data")
        return False

    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"{ticker.replace('.', '_')}_daily_data.csv")

    existing_data, last_date, first_date = load_existing_data(filepath)

    if force_full or existing_data is None:
        logger.info(f"Downloading full history for {ticker}...")
        data = yf.download(ticker, interval="1d", period="max", progress=False)
        if len(data) == 0:
            logger.error(f"No data returned for {ticker}")
            return False

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.reset_index(inplace=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Saved {len(data)} rows to {filepath}")
        return True

    if required_start_date and first_date is not None:
        requested = pd.to_datetime(required_start_date)
        if requested < first_date:
            logger.info(
                f"Backfilling {ticker} from {requested.date()} to {(first_date - timedelta(days=1)).date()}..."
            )
            older_data = yf.download(
                ticker,
                start=requested.strftime('%Y-%m-%d'),
                end=first_date.strftime('%Y-%m-%d'),
                interval="1d",
                progress=False,
            )

            if len(older_data) > 0:
                if isinstance(older_data.columns, pd.MultiIndex):
                    older_data.columns = older_data.columns.get_level_values(0)
                older_data.reset_index(inplace=True)

                existing_data_reset = existing_data.reset_index()
                combined = pd.concat([older_data, existing_data_reset], ignore_index=True)
                combined.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                combined.sort_values('Date', inplace=True)
                combined.reset_index(drop=True, inplace=True)
                combined.to_csv(filepath, index=False)
                existing_data = combined.set_index('Date')
                logger.info(f"Backfill complete for {ticker}: added {len(older_data)} rows")
            else:
                logger.warning(
                    f"No backfill data returned for {ticker} from {requested.date()} to {first_date.date()}"
                )

    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')

    if start_date >= today:
        logger.info(f"{ticker}: Data is up to date (last date: {last_date.date()})")
        return True

    logger.info(f"Updating {ticker} from {start_date} to {today}...")
    new_data = yf.download(ticker, start=start_date, end=today, interval="1d", progress=False)

    if len(new_data) == 0:
        logger.info(f"{ticker}: No new data available")
        return True

    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = new_data.columns.get_level_values(0)

    new_data.reset_index(inplace=True)
    existing_data_reset = existing_data.reset_index()
    combined = pd.concat([existing_data_reset, new_data], ignore_index=True)
    combined.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    combined.sort_values('Date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(filepath, index=False)
    logger.info(f"Added {len(new_data)} new rows to {filepath} (total: {len(combined)})")
    return True


def forward_update(ticker: str, data_dir: str, from_date: date_type, to_date: date_type = None) -> pd.DataFrame:
    """Fetch and append forward-window data for an existing ticker cache."""
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not available - cannot forward update data")
    if from_date is None:
        raise ValueError("from_date is required for forward_update")

    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"{ticker.replace('.', '_')}_daily_data.csv")
    existing_data, _, _ = load_existing_data(filepath)
    if existing_data is None:
        raise FileNotFoundError(f"No existing cache found for {ticker} at {filepath}")

    start_dt = pd.Timestamp(from_date).normalize() + timedelta(days=1)
    target_end = pd.Timestamp(to_date if to_date is not None else datetime.now().date()).normalize()

    existing_reset = existing_data.reset_index()
    existing_rows = len(existing_reset)

    if start_dt > target_end:
        logger.info(
            "%s: forward_update already up to date (start=%s, end=%s)",
            ticker,
            start_dt.date(),
            target_end.date(),
        )
        return existing_data.sort_index()

    fetch_end_exclusive = target_end + timedelta(days=1)
    logger.info("%s: forward update window %s -> %s", ticker, start_dt.date(), target_end.date())
    new_data = yf.download(
        ticker,
        start=start_dt.strftime('%Y-%m-%d'),
        end=fetch_end_exclusive.strftime('%Y-%m-%d'),
        interval="1d",
        progress=False,
    )

    if len(new_data) == 0:
        logger.info("%s: no new data in forward window %s -> %s", ticker, start_dt.date(), target_end.date())
        return existing_data.sort_index()

    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = new_data.columns.get_level_values(0)
    new_data.reset_index(inplace=True)
    if 'Date' not in new_data.columns and 'index' in new_data.columns:
        new_data.rename(columns={'index': 'Date'}, inplace=True)

    combined = pd.concat([existing_reset, new_data], ignore_index=True)
    combined.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    combined.sort_values('Date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    existing_cmp = existing_reset.copy()
    existing_cmp.sort_values('Date', inplace=True)
    existing_cmp.reset_index(drop=True, inplace=True)

    # Detect both appended rows and overlap-only OHLCV revisions.
    common_cols = sorted(set(existing_cmp.columns).intersection(set(combined.columns)))
    existing_aligned = existing_cmp[common_cols].copy()
    combined_aligned = combined[common_cols].copy()
    if 'Date' in common_cols:
        existing_aligned['Date'] = pd.to_datetime(existing_aligned['Date'])
        combined_aligned['Date'] = pd.to_datetime(combined_aligned['Date'])
    data_changed = not combined_aligned.equals(existing_aligned)

    rows_added = len(combined) - existing_rows
    if data_changed:
        combined.to_csv(filepath, index=False)
        if rows_added > 0:
            logger.info("%s: forward update added %d rows", ticker, rows_added)
        else:
            logger.info("%s: forward update applied overlap revisions; cache refreshed", ticker)
    else:
        logger.info("%s: forward update fetched overlap only; cache unchanged", ticker)

    combined['Date'] = pd.to_datetime(combined['Date'])
    combined.set_index('Date', inplace=True)
    return combined.sort_index()


def update_all_tickers(tickers, data_dir):
    """Update data for all tickers."""
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = download_ticker_data(ticker, data_dir)
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            results[ticker] = False

    return results
