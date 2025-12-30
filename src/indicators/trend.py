"""
Trend indicators for technical analysis.

This module provides both pandas-ta wrappers and custom implementations
of common trend indicators like SMA, EMA, MACD, and ADX.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta not installed. Using custom implementations only.")


class TrendIndicators:
    """
    A class containing trend indicator calculations.
    """

    @staticmethod
    def sma(df: pd.DataFrame, column: str = 'Close', period: int = 20,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate SMA on
            period (int): Period for SMA
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: SMA values

        Example:
            >>> sma_20 = TrendIndicators.sma(df, period=20)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.sma(close=df[column], length=period)
        else:
            # Custom implementation
            return df[column].rolling(window=period).mean()

    @staticmethod
    def sma_custom(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Custom SMA implementation for educational purposes.

        Simple Moving Average is the average price over a specified period.

        Formula: SMA = Sum of closing prices over n periods / n

        Args:
            prices (pd.Series): Price series
            period (int): Period for SMA

        Returns:
            pd.Series: SMA values

        Example:
            >>> sma = TrendIndicators.sma_custom(df['Close'], period=20)
        """
        sma = prices.rolling(window=period).mean()
        return sma

    @staticmethod
    def ema(df: pd.DataFrame, column: str = 'Close', period: int = 20,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate EMA on
            period (int): Period for EMA
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: EMA values

        Example:
            >>> ema_20 = TrendIndicators.ema(df, period=20)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.ema(close=df[column], length=period)
        else:
            # Custom implementation
            return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def ema_custom(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Custom EMA implementation for educational purposes.

        Exponential Moving Average gives more weight to recent prices.

        Formula:
            EMA(t) = Price(t) * k + EMA(t-1) * (1 - k)
            where k = 2 / (period + 1)

        Args:
            prices (pd.Series): Price series
            period (int): Period for EMA

        Returns:
            pd.Series: EMA values

        Example:
            >>> ema = TrendIndicators.ema_custom(df['Close'], period=12)
        """
        ema = prices.ewm(span=period, adjust=False).mean()
        return ema

    @staticmethod
    def macd(df: pd.DataFrame, column: str = 'Close',
             fast: int = 12, slow: int = 26, signal: int = 9,
             use_library: bool = True) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate MACD on
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.DataFrame: DataFrame with MACD, Signal, and Histogram columns

        Example:
            >>> macd_df = TrendIndicators.macd(df)
        """
        if use_library and HAS_PANDAS_TA:
            macd_result = df.ta.macd(close=df[column], fast=fast, slow=slow, signal=signal)
            return macd_result
        else:
            # Custom implementation
            exp1 = df[column].ewm(span=fast, adjust=False).mean()
            exp2 = df[column].ewm(span=slow, adjust=False).mean()

            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            result = pd.DataFrame({
                f'MACD_{fast}_{slow}_{signal}': macd_line,
                f'MACDs_{fast}_{slow}_{signal}': signal_line,
                f'MACDh_{fast}_{slow}_{signal}': histogram
            }, index=df.index)

            return result

    @staticmethod
    def macd_custom(prices: pd.Series, fast: int = 12, slow: int = 26,
                    signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Custom MACD implementation for educational purposes.

        MACD shows the relationship between two moving averages.

        Formula:
            MACD Line = 12-day EMA - 26-day EMA
            Signal Line = 9-day EMA of MACD Line
            Histogram = MACD Line - Signal Line

        Args:
            prices (pd.Series): Price series
            fast (int): Fast EMA period (default 12)
            slow (int): Slow EMA period (default 26)
            signal (int): Signal EMA period (default 9)

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, Signal line, Histogram

        Example:
            >>> macd, signal, hist = TrendIndicators.macd_custom(df['Close'])
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14, use_library: bool = True) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            period (int): Period for ADX
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.DataFrame: DataFrame with ADX, +DI, -DI columns

        Example:
            >>> adx_df = TrendIndicators.adx(df, period=14)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.adx(high=df['High'], low=df['Low'],
                            close=df['Close'], length=period)
        else:
            # Custom implementation
            high = df['High']
            low = df['Low']
            close = df['Close']

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate directional movement
            plus_dm = high - high.shift(1)
            minus_dm = low.shift(1) - low

            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            # Smooth the values
            tr_smooth = tr.rolling(window=period).sum()
            plus_dm_smooth = plus_dm.rolling(window=period).sum()
            minus_dm_smooth = minus_dm.rolling(window=period).sum()

            # Calculate directional indicators
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)

            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()

            result = pd.DataFrame({
                f'ADX_{period}': adx,
                f'DMP_{period}': plus_di,
                f'DMN_{period}': minus_di
            }, index=df.index)

            return result

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0,
                   use_library: bool = True) -> pd.DataFrame:
        """
        Calculate Supertrend indicator.

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            period (int): ATR period
            multiplier (float): Multiplier for ATR
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.DataFrame: DataFrame with Supertrend and direction columns

        Example:
            >>> st = TrendIndicators.supertrend(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.supertrend(high=df['High'], low=df['Low'],
                                   close=df['Close'], length=period,
                                   multiplier=multiplier)
        else:
            # Basic custom implementation
            high = df['High']
            low = df['Low']
            close = df['Close']

            # Calculate ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()

            # Calculate basic bands
            hl_avg = (high + low) / 2
            upper_band = hl_avg + (multiplier * atr)
            lower_band = hl_avg - (multiplier * atr)

            result = pd.DataFrame({
                f'SUPERT_{period}_{multiplier}': upper_band,  # Simplified
                f'SUPERTd_{period}_{multiplier}': 1  # Direction placeholder
            }, index=df.index)

            return result

    @staticmethod
    def detect_golden_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
        """
        Detect golden cross (fast MA crosses above slow MA).

        Args:
            df (pd.DataFrame): Input data
            fast (int): Fast MA period
            slow (int): Slow MA period

        Returns:
            pd.Series: Boolean series indicating golden cross

        Example:
            >>> crosses = TrendIndicators.detect_golden_cross(df)
        """
        sma_fast = TrendIndicators.sma(df, period=fast)
        sma_slow = TrendIndicators.sma(df, period=slow)

        # Golden cross: fast crosses above slow
        cross = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))

        return cross

    @staticmethod
    def detect_death_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
        """
        Detect death cross (fast MA crosses below slow MA).

        Args:
            df (pd.DataFrame): Input data
            fast (int): Fast MA period
            slow (int): Slow MA period

        Returns:
            pd.Series: Boolean series indicating death cross

        Example:
            >>> crosses = TrendIndicators.detect_death_cross(df)
        """
        sma_fast = TrendIndicators.sma(df, period=fast)
        sma_slow = TrendIndicators.sma(df, period=slow)

        # Death cross: fast crosses below slow
        cross = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

        return cross


# Convenience functions
def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
    """Quick function to calculate SMA."""
    return TrendIndicators.sma(df, column=column, period=period)


def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
    """Quick function to calculate EMA."""
    return TrendIndicators.ema(df, column=column, period=period)


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> pd.DataFrame:
    """Quick function to calculate MACD."""
    return TrendIndicators.macd(df, fast=fast, slow=slow, signal=signal)


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Fetch sample data
    df = get_stock_data('AAPL', start='2023-01-01')

    # Calculate indicators
    df['SMA_20'] = TrendIndicators.sma(df, period=20)
    df['SMA_50'] = TrendIndicators.sma(df, period=50)
    df['EMA_12'] = TrendIndicators.ema(df, period=12)

    # MACD
    macd_df = TrendIndicators.macd(df)
    df = pd.concat([df, macd_df], axis=1)

    # Detect crosses
    df['Golden_Cross'] = TrendIndicators.detect_golden_cross(df)
    df['Death_Cross'] = TrendIndicators.detect_death_cross(df)

    print("\nData with trend indicators:")
    print(df[['Close', 'SMA_20', 'SMA_50', 'EMA_12']].tail())

    # Show golden/death crosses
    golden = df[df['Golden_Cross']]
    death = df[df['Death_Cross']]

    print(f"\nGolden crosses: {len(golden)}")
    print(f"Death crosses: {len(death)}")
