"""
Advanced Market Screener

Find stocks matching specific criteria including gap-ups/downs, unusual volume,
momentum plays, value stocks, and other trading opportunities.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class MarketScreener:
    """Advanced market screener with multiple preset strategies"""

    # Popular stock universes
    SP500_TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
        'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV', 'MRK',
        'AVGO', 'KO', 'PEP', 'COST', 'ADBE', 'WMT', 'MCD', 'CSCO', 'ACN', 'TMO',
        'ABT', 'CRM', 'NFLX', 'NKE', 'DIS', 'VZ', 'INTC', 'AMD', 'CMCSA', 'TXN',
        'QCOM', 'DHR', 'UNP', 'PM', 'NEE', 'RTX', 'HON', 'BMY', 'UPS', 'SBUX'
    ]

    TECH_STOCKS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE', 'CRM', 'ORCL',
        'INTC', 'AMD', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'NFLX', 'PYPL', 'SQ', 'SHOP',
        'UBER', 'LYFT', 'SNAP', 'PINS', 'TWLO', 'ZM', 'DOCU', 'CRWD', 'SNOW', 'PLTR'
    ]

    @staticmethod
    def fetch_stock_info(ticker: str) -> Optional[Dict]:
        """
        Fetch current stock information

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock data or None if error
        """
        try:
            stock = yf.Ticker(ticker)

            # Get current price and previous close
            hist = stock.history(period='5d')

            if hist.empty or len(hist) < 2:
                return None

            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]

            # Get info
            info = stock.info

            # Calculate metrics
            gap_pct = ((current_price - prev_close) / prev_close) * 100

            # Volume analysis
            current_volume = hist['Volume'].iloc[-1]
            avg_volume_20d = hist['Volume'].tail(20).mean() if len(hist) >= 20 else hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0

            return {
                'ticker': ticker,
                'name': info.get('shortName', ticker),
                'price': current_price,
                'prev_close': prev_close,
                'gap_pct': gap_pct,
                'volume': current_volume,
                'avg_volume': avg_volume_20d,
                'volume_ratio': volume_ratio,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 0),
                'sector': info.get('sector', 'Unknown'),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
            }

        except Exception as e:
            return None

    @classmethod
    def scan_gap_ups(cls, tickers: List[str], min_gap: float = 3.0, max_workers: int = 10) -> pd.DataFrame:
        """
        Find stocks with significant gap ups

        Args:
            tickers: List of tickers to scan
            min_gap: Minimum gap percentage
            max_workers: Number of parallel workers

        Returns:
            DataFrame with gap up stocks
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_stock_info, ticker): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                data = future.result()
                if data and data['gap_pct'] >= min_gap:
                    results.append(data)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('gap_pct', ascending=False)

        return df

    @classmethod
    def scan_gap_downs(cls, tickers: List[str], min_gap: float = -3.0, max_workers: int = 10) -> pd.DataFrame:
        """
        Find stocks with significant gap downs

        Args:
            tickers: List of tickers to scan
            min_gap: Minimum gap percentage (negative)
            max_workers: Number of parallel workers

        Returns:
            DataFrame with gap down stocks
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_stock_info, ticker): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                data = future.result()
                if data and data['gap_pct'] <= min_gap:
                    results.append(data)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('gap_pct', ascending=True)

        return df

    @classmethod
    def scan_unusual_volume(cls, tickers: List[str], min_ratio: float = 2.0, max_workers: int = 10) -> pd.DataFrame:
        """
        Find stocks with unusual volume

        Args:
            tickers: List of tickers to scan
            min_ratio: Minimum volume ratio vs average
            max_workers: Number of parallel workers

        Returns:
            DataFrame with unusual volume stocks
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_stock_info, ticker): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                data = future.result()
                if data and data['volume_ratio'] >= min_ratio:
                    results.append(data)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('volume_ratio', ascending=False)

        return df

    @classmethod
    def scan_momentum(cls, tickers: List[str], min_gap: float = 2.0, min_volume_ratio: float = 1.5, max_workers: int = 10) -> pd.DataFrame:
        """
        Find momentum stocks (gap up + high volume)

        Args:
            tickers: List of tickers to scan
            min_gap: Minimum gap up percentage
            min_volume_ratio: Minimum volume ratio
            max_workers: Number of parallel workers

        Returns:
            DataFrame with momentum stocks
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_stock_info, ticker): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                data = future.result()
                if data and data['gap_pct'] >= min_gap and data['volume_ratio'] >= min_volume_ratio:
                    # Calculate momentum score
                    data['momentum_score'] = (data['gap_pct'] * 0.6) + (data['volume_ratio'] * 10 * 0.4)
                    results.append(data)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('momentum_score', ascending=False)

        return df

    @classmethod
    def scan_value_stocks(cls, tickers: List[str], max_pe: float = 20, min_div_yield: float = 2.0, max_workers: int = 10) -> pd.DataFrame:
        """
        Find undervalued stocks

        Args:
            tickers: List of tickers to scan
            max_pe: Maximum P/E ratio
            min_div_yield: Minimum dividend yield
            max_workers: Number of parallel workers

        Returns:
            DataFrame with value stocks
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_stock_info, ticker): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                data = future.result()
                if data:
                    pe = data['pe_ratio'] if data['pe_ratio'] else 999
                    div = data['dividend_yield'] if data['dividend_yield'] else 0

                    if pe > 0 and pe <= max_pe and div >= min_div_yield:
                        results.append(data)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('pe_ratio', ascending=True)

        return df

    @classmethod
    def scan_high_beta(cls, tickers: List[str], min_beta: float = 1.5, max_workers: int = 10) -> pd.DataFrame:
        """
        Find high beta (volatile) stocks

        Args:
            tickers: List of tickers to scan
            min_beta: Minimum beta value
            max_workers: Number of parallel workers

        Returns:
            DataFrame with high beta stocks
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_stock_info, ticker): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                data = future.result()
                if data and data['beta'] >= min_beta:
                    results.append(data)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('beta', ascending=False)

        return df

    @classmethod
    def scan_near_52w_high(cls, tickers: List[str], threshold: float = 5.0, max_workers: int = 10) -> pd.DataFrame:
        """
        Find stocks near 52-week high

        Args:
            tickers: List of tickers to scan
            threshold: How close to 52w high (percentage)
            max_workers: Number of parallel workers

        Returns:
            DataFrame with stocks near 52w high
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_stock_info, ticker): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                data = future.result()
                if data and data['52w_high'] > 0:
                    distance_from_high = ((data['52w_high'] - data['price']) / data['52w_high']) * 100

                    if distance_from_high <= threshold:
                        data['distance_from_52w_high'] = distance_from_high
                        results.append(data)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('distance_from_52w_high', ascending=True)

        return df
