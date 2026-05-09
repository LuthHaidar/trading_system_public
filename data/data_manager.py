import os
from collections import OrderedDict
from threading import RLock
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger
from data.yfinance_scraper import download_ticker_data, forward_update, update_all_tickers

logger = get_logger(__name__)


class DataManager:
    """
    Manages loading, caching, and updating of market data
    """
    
    def __init__(self, data_dir: str = './data', cache_size: int = 100, use_adjusted_close: bool = True,
                 freshness_threshold_days: int = 3):
        """
        Initialize DataManager
        
        Args:
            data_dir: Directory containing CSV data files
            cache_size: Maximum number of dataframes to cache
            use_adjusted_close: If True, replace Close with Adj Close when available
            freshness_threshold_days: Max allowed age of cache tail before forward refresh
        """
        self.data_dir = data_dir
        self.cache_size = cache_size
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.use_adjusted_close = bool(use_adjusted_close)
        self.freshness_threshold_days = int(freshness_threshold_days)
        self.last_update = {}  # {ticker: datetime}
        self._cache_lock = RLock()
        
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"DataManager initialized with data_dir: {data_dir}")
    
    def _get_filepath(self, ticker: str) -> str:
        """Get filepath for ticker CSV"""
        clean_ticker = ticker.replace('.', '_')
        return os.path.join(self.data_dir, f"{clean_ticker}_daily_data.csv")
    
    def _manage_cache(self):
        """Remove least-recently-used item if cache is full (caller must hold cache lock)."""
        if len(self.cache) >= self.cache_size:
            lru_ticker, _ = self.cache.popitem(last=False)
            self.last_update.pop(lru_ticker, None)
            logger.debug(f"Removed {lru_ticker} from cache (LRU)")

    def _touch_cache(self, ticker: str) -> None:
        """Move ticker to end of LRU order (caller must hold cache lock)."""
        if ticker in self.cache:
            self.cache.move_to_end(ticker)

    def _invalidate_cache(self, ticker: str) -> None:
        """Invalidate cache entry safely (idempotent, lock-protected)."""
        with self._cache_lock:
            self.cache.pop(ticker, None)
            self.last_update.pop(ticker, None)
    

    def _read_ticker_csv(self, filepath: str) -> pd.DataFrame:
        """Read and normalize a ticker CSV into canonical in-memory form."""
        df = pd.read_csv(filepath)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'Date' not in df.columns:
            if 'date' in df.columns:
                df.rename(columns={'date': 'Date'}, inplace=True)
            else:
                df.reset_index(inplace=True)

        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        col_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
            'adj close': 'Adj Close', 'Adj close': 'Adj Close'
        }
        df.rename(columns=col_mapping, inplace=True)

        if self.use_adjusted_close and 'Adj Close' in df.columns and 'Close' in df.columns:
            df['Close'] = df['Adj Close']

        return df

    def _ensure_forward_freshness(self, ticker: str, df: pd.DataFrame, end_date: Optional[str]) -> pd.DataFrame:
        """Ensure cache tail freshness by forward-updating stale files."""
        if df.empty:
            return df

        anchor_date = pd.Timestamp(end_date).date() if end_date else date.today()
        cache_end = df.index[-1].date()
        delta_days = int((anchor_date - cache_end).days)
        logger.info("%s: cache ends %s, today is %s, delta=%d days", ticker, cache_end, anchor_date, delta_days)

        if delta_days > self.freshness_threshold_days:
            logger.info(
                "%s: cache stale by %d days (threshold=%d); running forward update",
                ticker,
                delta_days,
                self.freshness_threshold_days,
            )
            combined = forward_update(
                ticker=ticker,
                data_dir=self.data_dir,
                from_date=cache_end,
                to_date=anchor_date,
            )
            combined = combined.copy()
            combined.index = pd.to_datetime(combined.index).tz_localize(None).normalize()
            combined.sort_index(inplace=True)
            combined = combined[~combined.index.duplicated(keep='last')]
            if self.use_adjusted_close and 'Adj Close' in combined.columns and 'Close' in combined.columns:
                combined['Close'] = combined['Adj Close']
            df = combined

        return df

    def load_ticker(self, ticker: str, start_date: str = None, 
                   end_date: str = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Load data for a ticker
        
        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data indexed by Date
        """
        # Check cache
        with self._cache_lock:
            cached_df = self.cache.get(ticker) if use_cache else None
            if cached_df is not None:
                self.cache.move_to_end(ticker)
        if cached_df is not None:
            df = cached_df
            logger.debug(f"Loaded {ticker} from cache")
            df = self._ensure_forward_freshness(ticker, df, end_date=end_date)
            if use_cache:
                with self._cache_lock:
                    self.cache[ticker] = df.copy()
                    self.last_update[ticker] = datetime.now()
        else:
            # Load from file
            filepath = self._get_filepath(ticker)
            
            if not os.path.exists(filepath):
                logger.warning(f"No data file found for {ticker} at {filepath}")
                logger.info(f"Attempting to download {ticker}...")
                if not download_ticker_data(ticker, self.data_dir, required_start_date=start_date):
                    raise FileNotFoundError(f"Could not download data for {ticker}")
            
            try:
                df = self._read_ticker_csv(filepath)

                # Cache the data
                if use_cache:
                    with self._cache_lock:
                        self._manage_cache()
                        self.cache[ticker] = df.copy()
                        self.last_update[ticker] = datetime.now()

                logger.info(f"Loaded {ticker}: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

                # If caller requested older history than local cache contains, backfill it.
                if start_date and len(df) > 0 and pd.to_datetime(start_date) < df.index[0]:
                    logger.info(
                        f"{ticker}: earliest local row is {df.index[0].date()}, backfilling to {start_date}..."
                    )
                    if download_ticker_data(ticker, self.data_dir, required_start_date=start_date):
                        df = self._read_ticker_csv(filepath)

                df = self._ensure_forward_freshness(ticker, df, end_date=end_date)
                if use_cache:
                    with self._cache_lock:
                        self.cache[ticker] = df.copy()
                        self.last_update[ticker] = datetime.now()

            except Exception as e:
                logger.error(f"Error loading {ticker}: {e}")
                raise
        
        # Filter by date range
        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]
        
        return df.copy()


    def load_multiple(self, tickers: List[str], start_date: str = None,
                     end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = self.load_ticker(ticker, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")
        
        return data
    
    def update_ticker(self, ticker: str) -> bool:
        """
        Download latest data for ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            True if successful
        """
        success = download_ticker_data(ticker, self.data_dir)
        
        # Invalidate cache
        self._invalidate_cache(ticker)
        
        return success
    
    def update_all(self, tickers: List[str]) -> Dict[str, bool]:
        """
        Update data for all tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of {ticker: success}
        """
        results = update_all_tickers(tickers, self.data_dir)
        
        # Clear cache for updated tickers
        for ticker in results:
            self._invalidate_cache(ticker)
        
        return results
    
    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get most recent close prices for tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of {ticker: price}
        """
        prices = {}
        for ticker in tickers:
            try:
                df = self.load_ticker(ticker)
                prices[ticker] = float(df['Close'].iloc[-1])
            except Exception as e:
                logger.error(f"Could not get price for {ticker}: {e}")
                prices[ticker] = None
        
        return prices
    
    def get_latest_date(self, ticker: str) -> Optional[datetime]:
        """
        Get the latest date available for ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Latest date or None
        """
        try:
            df = self.load_ticker(ticker)
            return df.index[-1]
        except Exception:
            return None
    
    def compute_returns(self, ticker: str, periods: int = 1) -> pd.Series:
        """
        Calculate returns for a ticker
        
        Args:
            ticker: Ticker symbol
            periods: Number of periods for return calculation
            
        Returns:
            Series of returns
        """
        df = self.load_ticker(ticker)
        return df['Close'].pct_change(periods=periods)
    
    def compute_volatility(self, ticker: str, window: int = 20,
                          annualize: bool = True) -> pd.Series:
        """
        Calculate rolling volatility
        
        Args:
            ticker: Ticker symbol
            window: Rolling window size
            annualize: Whether to annualize (multiply by sqrt(252))
            
        Returns:
            Series of volatility
        """
        returns = self.compute_returns(ticker, periods=1)
        vol = returns.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    def get_volume(self, ticker: str, date: str = None) -> float:
        """
        Get trading volume for a ticker on a specific date
        
        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD). If None, returns latest.
            
        Returns:
            Volume
        """
        df = self.load_ticker(ticker)
        
        if date is None:
            return float(df['Volume'].iloc[-1])
        else:
            return float(df.loc[date, 'Volume'])
    
    def get_avg_volume(self, ticker: str, window: int = 20,
                       end_date: str = None) -> float:
        """
        Calculate average volume over window
        
        Args:
            ticker: Ticker symbol
            window: Number of days for average
            end_date: End date (defaults to latest)
            
        Returns:
            Average volume
        """
        df = self.load_ticker(ticker)
        
        if end_date:
            df = df.loc[:end_date]
        
        return float(df['Volume'].tail(window).mean())
    
    def validate_data(self, ticker: str) -> Tuple[bool, List[str]]:
        """
        Validate data quality for a ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            df = self.load_ticker(ticker)
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
            
            # Check for negative prices
            if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
                issues.append("Negative prices detected")
            
            # Check for zero volume
            zero_vol = (df['Volume'] == 0).sum()
            if zero_vol > len(df) * 0.1:  # More than 10% zero volume
                issues.append(f"High number of zero volume days: {zero_vol}")
            
            # Check for data gaps
            date_diff = df.index.to_series().diff()
            large_gaps = (date_diff > pd.Timedelta(days=7)).sum()
            if large_gaps > 0:
                issues.append(f"Data gaps detected: {large_gaps} gaps > 7 days")
            
            # Check High >= Low
            if (df['High'] < df['Low']).any():
                issues.append("High < Low detected")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.info(f"{ticker}: Data validation passed")
            else:
                logger.warning(f"{ticker}: Data validation issues - {', '.join(issues)}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating {ticker}: {e}")
            return False, [str(e)]
    
    def get_data_summary(self, ticker: str) -> Dict:
        """
        Get summary statistics for ticker data
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary of summary stats
        """
        df = self.load_ticker(ticker)
        
        return {
            'ticker': ticker,
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'num_days': len(df),
            'avg_close': float(df['Close'].mean()),
            'avg_volume': float(df['Volume'].mean()),
            'latest_close': float(df['Close'].iloc[-1]),
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.last_update.clear()
        logger.info("Cache cleared")
