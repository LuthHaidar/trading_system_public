try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None
    
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
import time
from utils.logger import get_logger

logger = get_logger(__name__)


class CurrencyConverter:
    """
    Handle currency conversions for multi-currency portfolios
    """
    
    # Common ticker currency mappings
    TICKER_CURRENCIES = {
        'L': 'GBP',  # London Stock Exchange
        'TO': 'CAD',  # Toronto
        'AX': 'AUD',  # Australia
        'HK': 'HKD',  # Hong Kong
        'T': 'JPY',   # Tokyo
        'SW': 'CHF',  # Switzerland
        'SI': 'SGD',  # Singapore
        'DE': 'EUR',  # Germany
        'PA': 'EUR',  # Paris
        'AS': 'EUR',  # Amsterdam
    }

    # Market metadata by exchange suffix
    TICKER_MARKETS = {
        'L': {'exchange': 'LSE', 'country': 'GB', 'currency': 'GBP'},
        'TO': {'exchange': 'TSX', 'country': 'CA', 'currency': 'CAD'},
        'AX': {'exchange': 'ASX', 'country': 'AU', 'currency': 'AUD'},
        'HK': {'exchange': 'SEHK', 'country': 'HK', 'currency': 'HKD'},
        'T': {'exchange': 'TSEJ', 'country': 'JP', 'currency': 'JPY'},
        'SW': {'exchange': 'EBS', 'country': 'CH', 'currency': 'CHF'},
        'SI': {'exchange': 'SGX', 'country': 'SG', 'currency': 'SGD'},
        'DE': {'exchange': 'XETRA', 'country': 'DE', 'currency': 'EUR'},
        'PA': {'exchange': 'SBF', 'country': 'FR', 'currency': 'EUR'},
        'AS': {'exchange': 'AEB', 'country': 'NL', 'currency': 'EUR'},
    }

    _EXCHANGES = [meta['exchange'] for meta in TICKER_MARKETS.values()]
    _EXCHANGE_COUNTS: Dict[str, int] = {}
    for _ex in _EXCHANGES:
        _EXCHANGE_COUNTS[_ex] = _EXCHANGE_COUNTS.get(_ex, 0) + 1
    _DUPLICATE_EXCHANGES = sorted([_ex for _ex, _count in _EXCHANGE_COUNTS.items() if _count > 1])
    assert len(_EXCHANGES) == len(set(_EXCHANGES)), (
        "TICKER_MARKETS contains duplicate exchange codes — EXCHANGE_TO_SUFFIX would be lossy. "
        f"Duplicates: {_DUPLICATE_EXCHANGES}"
    )
    
    def __init__(
        self,
        base_currency: str = 'SGD',
        fx_cache_staleness_days: int = 1,
        fx_data_dir: Union[str, Path] = 'data/fx',
    ):
        """
        Initialize currency converter
        
        Args:
            base_currency: Portfolio base currency (default: SGD)
        """
        self.base_currency = base_currency
        self.fx_cache_staleness_days = max(int(fx_cache_staleness_days), 0)
        self.fx_data_dir = Path(fx_data_dir)
        self.fx_cache: Dict[str, Dict[str, Union[float, datetime, bool]]] = {}
        self.fx_history: Dict[str, pd.Series] = {}
        self.fx_cache_ttl_seconds = 3600
        self.fx_retry_attempts = 3
        self.fx_retry_backoff_seconds = 1.0
        logger.info("CurrencyConverter initialized with base currency: %s", base_currency)

    def preload_pair(
        self,
        from_currency: str,
        to_currency: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
    ) -> None:
        """Preload daily FX close history for a currency pair.

        Data is persisted to ``data/fx/{pair}_daily.csv`` and reused when it is
        sufficiently fresh relative to ``end_date``.
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance unavailable; cannot preload FX pair %s%s", from_currency, to_currency)
            return

        pair = f"{from_currency}{to_currency}"
        csv_path = self.fx_data_dir / f'{pair}_daily.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        normalized_start = self._normalize_date(start_date)
        normalized_end = self._normalize_date(end_date)
        if normalized_start is None or normalized_end is None:
            logger.warning("Invalid preload range for FX pair %s", pair)
            return

        if csv_path.exists():
            try:
                cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if 'Close' in cached.columns:
                    series = self._coerce_close_series(cached['Close'])
                else:
                    series = self._coerce_close_series(cached.squeeze('columns'))
                last_date = series.index.max()
                if pd.notna(last_date) and (normalized_end - last_date).days <= self.fx_cache_staleness_days:
                    self.fx_history[pair] = series.sort_index()
                    logger.info("Loaded FX history cache for %s from %s", pair, csv_path)
                    return
            except Exception as exc:
                logger.warning("Failed reading FX cache %s (%s); re-downloading", csv_path, exc)

        fx_ticker = f"{pair}=X"
        try:
            downloaded = yf.download(
                fx_ticker,
                start=normalized_start.strftime('%Y-%m-%d'),
                end=(normalized_end + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False,
            )
            if len(downloaded) == 0 or 'Close' not in downloaded.columns:
                logger.warning("No FX close data downloaded for %s", fx_ticker)
                return

            series = self._coerce_close_series(downloaded['Close'])
            self.fx_history[pair] = series
            series.to_frame(name='Close').to_csv(csv_path)
            logger.info("Preloaded FX history for %s (%d rows)", pair, len(series))
        except Exception as exc:
            logger.warning("Failed to preload FX history for %s: %s", fx_ticker, exc)
    
    def get_ticker_currency(self, ticker: str) -> str:
        """
        Determine the currency of a ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Currency code (e.g., 'USD', 'GBP', 'SGD')
        """
        # Check for exchange suffix
        if '.' in ticker:
            suffix = ticker.split('.')[-1]
            currency = self.TICKER_CURRENCIES.get(suffix, 'USD')
            return currency
        
        # Default to USD for US tickers
        return 'USD'

    def get_ticker_market(self, ticker: str) -> dict:
        """
        Determine market metadata for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dict with market, country, currency, suffix, and is_us fields.
        """
        suffix = None
        if '.' in ticker:
            suffix = ticker.split('.')[-1]
            market_info = self.TICKER_MARKETS.get(suffix)
            if market_info:
                return {
                    **market_info,
                    'market': market_info['exchange'],
                    'suffix': suffix,
                    'is_us': False
                }

            return {
                'market': 'INTL',
                'country': 'INTL',
                'currency': self.get_ticker_currency(ticker),
                'suffix': suffix,
                'is_us': False
            }

        return {
            'market': 'US',
            'country': 'US',
            'currency': 'USD',
            'suffix': suffix,
            'is_us': True
        }
    
    def _normalize_date(self, date: Optional[Union[str, datetime, pd.Timestamp]]) -> Optional[pd.Timestamp]:
        """Normalize date-like inputs to a timezone-naive pandas Timestamp."""
        if date is None:
            return None

        normalized = pd.Timestamp(date)
        if normalized.tzinfo is not None:
            normalized = normalized.tz_localize(None)
        return normalized

    def get_fx_rate(self, from_currency: str, to_currency: str = None,
                   date: Optional[Union[str, datetime, pd.Timestamp]] = None) -> float:
        """
        Get exchange rate from one currency to another
        
        Args:
            from_currency: Source currency
            to_currency: Target currency (defaults to base_currency)
            date: Date for historical rate (None = latest)
            
        Returns:
            Exchange rate
        """
        if to_currency is None:
            to_currency = self.base_currency
        
        # Same currency = 1.0
        if from_currency == to_currency:
            return 1.0
        
        if not YFINANCE_AVAILABLE:
            raise RuntimeError(
                f"yfinance is required for FX conversion {from_currency}/{to_currency}"
            )
        
        normalized_date = self._normalize_date(date)
        pair = f"{from_currency}{to_currency}"
        preloaded_history = self.fx_history.get(pair)
        if preloaded_history is not None and len(preloaded_history) == 0:
            self.fx_history.pop(pair, None)
            preloaded_history = None

        if preloaded_history is not None and len(preloaded_history) > 0:
            preloaded_history = self._coerce_close_series(preloaded_history)
            if len(preloaded_history) == 0:
                self.fx_history.pop(pair, None)
            else:
                self.fx_history[pair] = preloaded_history
                lookup_date = normalized_date if normalized_date is not None else preloaded_history.index.max()
                asof_rate = preloaded_history.asof(lookup_date)
                if isinstance(asof_rate, pd.Series):
                    asof_rate = asof_rate.iloc[0] if len(asof_rate) > 0 else float('nan')
                if pd.notna(asof_rate):
                    return float(asof_rate)

        cache_date_key = normalized_date.strftime('%Y-%m-%d') if normalized_date is not None else 'latest'

        # Check cache
        cache_key = f"{from_currency}{to_currency}:{cache_date_key}"
        cache_entry = self.fx_cache.get(cache_key)
        if cache_entry is not None:
            if bool(cache_entry.get('is_historical', False)):
                logger.debug("FX cache hit for %s", cache_key)
                return float(cache_entry['rate'])
            fetched_at = cache_entry.get('fetched_at')
            if isinstance(fetched_at, datetime):
                age_seconds = (datetime.utcnow() - fetched_at).total_seconds()
                if age_seconds <= float(self.fx_cache_ttl_seconds):
                    logger.debug("FX cache hit for %s (age=%.1fs)", cache_key, age_seconds)
                    return float(cache_entry['rate'])
            logger.debug("FX cache stale for %s; refreshing", cache_key)
        else:
            logger.debug("FX cache miss for %s", cache_key)
        
        # Use yfinance FX pairs (e.g., USDSGD=X)
        fx_ticker = f"{from_currency}{to_currency}=X"
        rate = self._fetch_fx_with_retry(fx_ticker, normalized_date)

        self.fx_cache[cache_key] = {
            'rate': float(rate),
            'fetched_at': datetime.utcnow(),
            'is_historical': normalized_date is not None,
        }
        return float(rate)

    def _fetch_fx_with_retry(self, fx_ticker: str, normalized_date: Optional[pd.Timestamp]) -> float:
        """Fetch FX rate with bounded retry policy and explicit failure."""
        last_exc: Optional[Exception] = None
        attempts = max(int(self.fx_retry_attempts), 1)
        backoff = max(float(self.fx_retry_backoff_seconds), 0.0)

        for attempt in range(1, attempts + 1):
            try:
                if normalized_date is not None:
                    return self._get_historical_or_previous_rate(fx_ticker, normalized_date)
                return self._get_latest_fx_rate(fx_ticker)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                sleep_seconds = backoff * (2 ** (attempt - 1))
                logger.debug(
                    "FX fetch retry %d/%d for %s after error: %s (sleep %.1fs)",
                    attempt,
                    attempts,
                    fx_ticker,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        logger.warning(
            "FX fetch failed after %d attempts for %s; raising explicit exception",
            attempts,
            fx_ticker,
        )
        raise RuntimeError(f"Failed to fetch FX rate for {fx_ticker}") from last_exc


    def _extract_close_value(self, close_data, index: int = -1) -> float:
        """Extract a scalar close value from yfinance close output."""
        value = close_data.iloc[index]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return float(value)

    @staticmethod
    def _coerce_close_series(close_data) -> pd.Series:
        """Coerce yfinance close payloads into a single float Series indexed by date."""
        if isinstance(close_data, pd.DataFrame):
            if close_data.shape[1] == 0:
                return pd.Series(dtype=float)
            series = close_data.iloc[:, 0]
        elif isinstance(close_data, pd.Series):
            series = close_data
        else:
            return pd.Series(dtype=float)

        series = series.astype(float)
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return series.sort_index()

    def _get_historical_or_previous_rate(self, fx_ticker: str, date: pd.Timestamp) -> float:
        """Get historical FX close on date, or closest previous available close."""
        start = (date - timedelta(days=14)).strftime('%Y-%m-%d')
        end = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        data = yf.download(fx_ticker, start=start, end=end, progress=False)

        if len(data) == 0:
            logger.warning("No FX data around %s, using latest rate", date.date())
            return self._get_latest_fx_rate(fx_ticker)

        close_series = self._coerce_close_series(data['Close'])
        if len(close_series) == 0:
            logger.warning("No FX close history for %s, using latest rate", fx_ticker)
            return self._get_latest_fx_rate(fx_ticker)

        rate = close_series.asof(date)
        if pd.isna(rate):
            logger.warning(
                "No FX history up to %s for %s, using latest rate",
                date.date(),
                fx_ticker,
            )
            return self._get_latest_fx_rate(fx_ticker)

        last_observed_date = close_series.loc[:date].index.max()
        if pd.notna(last_observed_date) and last_observed_date.date() != date.date():
            logger.warning(
                f"No FX close for {date.date()} ({fx_ticker}); "
                f"using previous close from {last_observed_date.date()}"
            )
        return float(rate)
    
    def _get_latest_fx_rate(self, fx_ticker: str) -> float:
        """Get latest FX rate for a ticker"""
        ticker = yf.Ticker(fx_ticker)
        data = ticker.history(period="1d")
        if len(data) > 0:
            return self._extract_close_value(data['Close'], -1)
        raise ValueError(f"No data for {fx_ticker}")
    
    def convert_to_base(self, amount: float, from_currency: str,
                       date: Optional[Union[str, datetime, pd.Timestamp]] = None) -> float:
        """
        Convert amount to base currency
        
        Args:
            amount: Amount in source currency
            from_currency: Source currency
            date: Date for historical conversion
            
        Returns:
            Amount in base currency
        """
        if from_currency == self.base_currency:
            return amount
        
        rate = self.get_fx_rate(from_currency, self.base_currency, date)
        return amount * rate
    
    def convert_price_series(self, prices: pd.Series, from_currency: str) -> pd.Series:
        """
        Convert a price series to base currency
        
        Args:
            prices: Price series with DatetimeIndex
            from_currency: Currency of prices
            
        Returns:
            Converted price series
        """
        if from_currency == self.base_currency:
            return prices
        
        converted = prices.copy().astype(float)
        pair = f"{from_currency}{self.base_currency}"
        history = self.fx_history.get(pair)

        if history is not None and len(history) > 0:
            history = self._coerce_close_series(history)
            fx_series = history.reindex(converted.index, method='ffill')
            missing_idx = fx_series[fx_series.isna()].index
            for idx in missing_idx:
                fx_series.loc[idx] = self.get_fx_rate(from_currency, self.base_currency, date=idx)
        else:
            fx_rates = [self.get_fx_rate(from_currency, self.base_currency, date=idx) for idx in converted.index]
            fx_series = pd.Series(fx_rates, index=converted.index, dtype=float)

        logger.info("Converting prices from %s to %s using per-date FX rates", from_currency, self.base_currency)
        return converted * fx_series.astype(float)
    
    def get_conversion_cost(self, amount: float, rate: float = 0.00002) -> float:
        """
        Calculate FX conversion cost (IBKR charges ~2 bps)
        
        Args:
            amount: Amount being converted
            rate: Conversion fee rate (default: 0.00002 = 2 bps)
            
        Returns:
            Conversion cost
        """
        return amount * rate
    
    def clear_cache(self):
        """Clear FX rate and preloaded-history caches."""
        self.fx_cache.clear()
        self.fx_history.clear()
        logger.info("FX rate cache cleared")
