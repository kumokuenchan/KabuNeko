"""
Data fetching module for stock market data using yfinance.

This module provides functions to fetch stock data with caching,
error handling, and retry logic to avoid API rate limits.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List, Dict, Optional
from tqdm import tqdm
import time


class StockDataFetcher:
    """
    A class to fetch stock market data with caching support.

    Attributes:
        cache_dir (str): Directory path for caching downloaded data
        use_cache (bool): Whether to use cached data by default
    """

    def __init__(self, cache_dir: str = "data/cache", use_cache: bool = True):
        """
        Initialize the StockDataFetcher.

        Args:
            cache_dir (str): Directory to store cached data
            use_cache (bool): Enable/disable caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    def _get_cache_filename(self, ticker: str, start: str, end: str, interval: str) -> Path:
        """Generate cache filename based on parameters."""
        return self.cache_dir / f"{ticker}_{start}_{end}_{interval}.csv"

    def _is_cache_valid(self, cache_file: Path, max_age_hours: int = 24) -> bool:
        """
        Check if cache file exists and is fresh enough.

        Args:
            cache_file (Path): Path to cache file
            max_age_hours (int): Maximum age of cache in hours

        Returns:
            bool: True if cache is valid, False otherwise
        """
        if not cache_file.exists():
            return False

        # Check file age
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - file_time

        return age.total_seconds() < (max_age_hours * 3600)

    def get_stock_data(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = '1d',
        use_cache: Optional[bool] = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a single ticker.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start (str): Start date in 'YYYY-MM-DD' format (default: 5 years ago)
            end (str): End date in 'YYYY-MM-DD' format (default: today)
            interval (str): Data interval ('1d', '1wk', '1mo', '1h', etc.)
            use_cache (bool): Override default cache setting
            max_retries (int): Maximum number of retry attempts

        Returns:
            pd.DataFrame: Stock data with OHLCV columns

        Example:
            >>> fetcher = StockDataFetcher()
            >>> df = fetcher.get_stock_data('AAPL', start='2020-01-01', end='2023-12-31')
        """
        # Set defaults
        if start is None:
            start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        if use_cache is None:
            use_cache = self.use_cache

        ticker = ticker.upper()

        # Check cache
        cache_file = self._get_cache_filename(ticker, start, end, interval)
        if use_cache and self._is_cache_valid(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                print(f"Loaded {ticker} from cache")
                return df
            except Exception as e:
                print(f"Cache read error: {e}. Fetching fresh data...")

        # Fetch from yfinance with retries
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start, end=end, interval=interval)

                if df.empty:
                    raise ValueError(f"No data found for {ticker}")

                # Save to cache
                if use_cache:
                    df.to_csv(cache_file)
                    print(f"Cached {ticker} data")

                print(f"Successfully fetched {ticker} ({len(df)} records)")
                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch {ticker} after {max_retries} attempts: {e}")
                    raise

    def get_multiple_stocks(
        self,
        tickers: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = '1d',
        use_cache: Optional[bool] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks.

        Args:
            tickers (List[str]): List of ticker symbols
            start (str): Start date in 'YYYY-MM-DD' format
            end (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval
            use_cache (bool): Whether to use cache

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping tickers to their data

        Example:
            >>> fetcher = StockDataFetcher()
            >>> data = fetcher.get_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'])
        """
        results = {}

        for ticker in tqdm(tickers, desc="Fetching stocks"):
            try:
                df = self.get_stock_data(ticker, start, end, interval, use_cache)
                results[ticker] = df
                # Small delay to be respectful to API
                time.sleep(0.5)
            except Exception as e:
                print(f"Skipping {ticker}: {e}")

        return results

    def get_fundamental_data(self, ticker: str) -> Dict:
        """
        Fetch fundamental data for a stock.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Dict: Dictionary containing fundamental data

        Example:
            >>> fetcher = StockDataFetcher()
            >>> info = fetcher.get_fundamental_data('AAPL')
            >>> print(info['marketCap'], info['trailingPE'])
        """
        try:
            stock = yf.Ticker(ticker.upper())

            # Get various fundamental data
            info = stock.info

            # Additional financial statements
            fundamentals = {
                'info': info,
                'balance_sheet': stock.balance_sheet,
                'income_stmt': stock.income_stmt,
                'cash_flow': stock.cash_flow,
                'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                'quarterly_income_stmt': stock.quarterly_income_stmt,
                'quarterly_cash_flow': stock.quarterly_cash_flow,
                'financials': stock.financials,
                'quarterly_financials': stock.quarterly_financials,
            }

            print(f"Fetched fundamental data for {ticker}")
            return fundamentals

        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            raise

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get basic information about a stock.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Dict: Basic stock information
        """
        try:
            stock = yf.Ticker(ticker.upper())
            return stock.info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            raise

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cached data.

        Args:
            ticker (str): Specific ticker to clear, or None to clear all
        """
        if ticker:
            # Clear specific ticker
            for file in self.cache_dir.glob(f"{ticker.upper()}_*.csv"):
                file.unlink()
                print(f"Cleared cache for {ticker}")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*.csv"):
                file.unlink()
            print("Cleared all cache")

    def update_cache(self, tickers: Optional[List[str]] = None):
        """
        Update cache for specified tickers or all cached tickers.

        Args:
            tickers (List[str]): List of tickers to update, or None for all
        """
        if tickers is None:
            # Get all cached tickers
            tickers = set()
            for file in self.cache_dir.glob("*.csv"):
                ticker = file.stem.split('_')[0]
                tickers.add(ticker)
            tickers = list(tickers)

        print(f"Updating cache for {len(tickers)} tickers...")
        for ticker in tqdm(tickers):
            try:
                # Force refresh by clearing cache first
                self.clear_cache(ticker)
                self.get_stock_data(ticker)
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed to update {ticker}: {e}")


# Convenience functions for quick access
def get_stock_data(ticker: str, start: str = None, end: str = None,
                   interval: str = '1d', use_cache: bool = True) -> pd.DataFrame:
    """
    Quick function to fetch stock data.

    Args:
        ticker (str): Stock ticker symbol
        start (str): Start date
        end (str): End date
        interval (str): Data interval
        use_cache (bool): Whether to use cache

    Returns:
        pd.DataFrame: Stock data
    """
    fetcher = StockDataFetcher(use_cache=use_cache)
    return fetcher.get_stock_data(ticker, start, end, interval)


def get_multiple_stocks(tickers: List[str], start: str = None, end: str = None,
                       interval: str = '1d', use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Quick function to fetch multiple stocks.

    Args:
        tickers (List[str]): List of ticker symbols
        start (str): Start date
        end (str): End date
        interval (str): Data interval
        use_cache (bool): Whether to use cache

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of stock data
    """
    fetcher = StockDataFetcher(use_cache=use_cache)
    return fetcher.get_multiple_stocks(tickers, start, end, interval)


if __name__ == "__main__":
    # Example usage
    fetcher = StockDataFetcher()

    # Fetch single stock
    aapl = fetcher.get_stock_data('AAPL', start='2023-01-01')
    print(f"\nAAPL data shape: {aapl.shape}")
    print(aapl.head())

    # Fetch multiple stocks
    stocks = fetcher.get_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'], start='2023-01-01')
    print(f"\nFetched {len(stocks)} stocks")
