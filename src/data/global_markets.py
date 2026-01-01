"""
Global Markets Data Fetcher

Track international indices, currencies, commodities, and global market status.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pytz


class GlobalMarketsFetcher:
    """Fetch global market data including indices, currencies, and commodities"""

    # Major Global Indices
    GLOBAL_INDICES = {
        # US Markets
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT',

        # European Markets
        'FTSE 100 (UK)': '^FTSE',
        'DAX (Germany)': '^GDAXI',
        'CAC 40 (France)': '^FCHI',
        'IBEX 35 (Spain)': '^IBEX',
        'FTSE MIB (Italy)': 'FTSEMIB.MI',

        # Asian Markets
        'Nikkei 225 (Japan)': '^N225',
        'Hang Seng (Hong Kong)': '^HSI',
        'Shanghai Composite': '000001.SS',
        'KOSPI (South Korea)': '^KS11',
        'Sensex (India)': '^BSESN',
        'Nifty 50 (India)': '^NSEI',

        # Other Regions
        'ASX 200 (Australia)': '^AXJO',
        'TSX (Canada)': '^GSPTSE',
        'Bovespa (Brazil)': '^BVSP',
    }

    # Major Currency Pairs
    CURRENCIES = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'JPY=X',
        'USD/CNY': 'CNY=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CAD': 'CAD=X',
        'USD/CHF': 'CHF=X',
        'NZD/USD': 'NZDUSD=X',
    }

    # Commodities
    COMMODITIES = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude Oil (WTI)': 'CL=F',
        'Brent Oil': 'BZ=F',
        'Natural Gas': 'NG=F',
        'Copper': 'HG=F',
        'Platinum': 'PL=F',
        'Corn': 'ZC=F',
        'Wheat': 'ZW=F',
    }

    # Crypto (for reference)
    CRYPTO = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
    }

    # Market Trading Hours (in local timezone)
    MARKET_HOURS = {
        'US (NYSE)': {'open': '09:30', 'close': '16:00', 'timezone': 'America/New_York'},
        'UK (LSE)': {'open': '08:00', 'close': '16:30', 'timezone': 'Europe/London'},
        'Germany (XETRA)': {'open': '09:00', 'close': '17:30', 'timezone': 'Europe/Berlin'},
        'Japan (TSE)': {'open': '09:00', 'close': '15:00', 'timezone': 'Asia/Tokyo'},
        'Hong Kong (HKEX)': {'open': '09:30', 'close': '16:00', 'timezone': 'Asia/Hong_Kong'},
        'Shanghai (SSE)': {'open': '09:30', 'close': '15:00', 'timezone': 'Asia/Shanghai'},
        'India (NSE)': {'open': '09:15', 'close': '15:30', 'timezone': 'Asia/Kolkata'},
        'Australia (ASX)': {'open': '10:00', 'close': '16:00', 'timezone': 'Australia/Sydney'},
    }

    @staticmethod
    def fetch_market_data(symbols: Dict[str, str]) -> pd.DataFrame:
        """
        Fetch current data for multiple symbols

        Args:
            symbols: Dictionary of name: ticker pairs

        Returns:
            DataFrame with market data
        """
        results = []

        for name, ticker in symbols.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')

                if hist.empty or len(hist) < 2:
                    continue

                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

                # Calculate 52-week range if available
                hist_1y = stock.history(period='1y')
                week_52_high = hist_1y['High'].max() if not hist_1y.empty else current_price
                week_52_low = hist_1y['Low'].min() if not hist_1y.empty else current_price

                results.append({
                    'name': name,
                    'ticker': ticker,
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    '52w_high': week_52_high,
                    '52w_low': week_52_low,
                })

            except Exception as e:
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    @classmethod
    def get_global_indices(cls) -> pd.DataFrame:
        """Get current data for all global indices"""
        return cls.fetch_market_data(cls.GLOBAL_INDICES)

    @classmethod
    def get_currencies(cls) -> pd.DataFrame:
        """Get current data for major currency pairs"""
        return cls.fetch_market_data(cls.CURRENCIES)

    @classmethod
    def get_commodities(cls) -> pd.DataFrame:
        """Get current data for commodities"""
        return cls.fetch_market_data(cls.COMMODITIES)

    @classmethod
    def get_crypto(cls) -> pd.DataFrame:
        """Get current data for major cryptocurrencies"""
        return cls.fetch_market_data(cls.CRYPTO)

    @staticmethod
    def is_market_open(market_name: str, market_hours: Dict) -> bool:
        """
        Check if a market is currently open

        Args:
            market_name: Name of the market
            market_hours: Dictionary with open, close times and timezone

        Returns:
            True if market is open, False otherwise
        """
        try:
            tz = pytz.timezone(market_hours['timezone'])
            now = datetime.now(tz)

            # Get current time in market timezone
            current_time = now.time()

            # Parse open and close times
            open_time = datetime.strptime(market_hours['open'], '%H:%M').time()
            close_time = datetime.strptime(market_hours['close'], '%H:%M').time()

            # Check if current time is between open and close
            # Also check if it's a weekday (Mon-Fri)
            is_weekday = now.weekday() < 5

            return is_weekday and open_time <= current_time <= close_time

        except Exception:
            return False

    @classmethod
    def get_market_status(cls) -> List[Dict]:
        """
        Get status of all major markets

        Returns:
            List of dictionaries with market status
        """
        status_list = []

        for market_name, hours in cls.MARKET_HOURS.items():
            is_open = cls.is_market_open(market_name, hours)

            # Get current time in market timezone
            try:
                tz = pytz.timezone(hours['timezone'])
                local_time = datetime.now(tz).strftime('%H:%M')
            except:
                local_time = "N/A"

            status_list.append({
                'market': market_name,
                'status': 'OPEN' if is_open else 'CLOSED',
                'local_time': local_time,
                'open': hours['open'],
                'close': hours['close'],
            })

        return status_list

    @staticmethod
    def calculate_correlation(ticker1: str, ticker2: str, period: str = '3mo') -> Optional[float]:
        """
        Calculate correlation between two assets

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            period: Historical period for calculation

        Returns:
            Correlation coefficient or None if error
        """
        try:
            data1 = yf.Ticker(ticker1).history(period=period)['Close']
            data2 = yf.Ticker(ticker2).history(period=period)['Close']

            if data1.empty or data2.empty:
                return None

            # Align the data
            df = pd.DataFrame({'asset1': data1, 'asset2': data2})
            df = df.dropna()

            if len(df) < 10:
                return None

            correlation = df['asset1'].corr(df['asset2'])

            return correlation

        except Exception:
            return None
