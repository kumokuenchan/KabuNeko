"""
Volatility indicators for technical analysis.

This module provides volatility-based indicators like Bollinger Bands,
ATR (Average True Range), Keltner Channels, and Donchian Channels.
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


class VolatilityIndicators:
    """
    A class containing volatility indicator calculations.
    """

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, column: str = 'Close',
                       period: int = 20, std_dev: float = 2.0,
                       use_library: bool = True) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate bands on
            period (int): Period for moving average (default 20)
            std_dev (float): Number of standard deviations (default 2.0)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.DataFrame: DataFrame with Lower, Middle, Upper bands and %B

        Example:
            >>> bb = VolatilityIndicators.bollinger_bands(df, period=20)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.bbands(close=df[column], length=period, std=std_dev)
        else:
            # Custom implementation
            sma = df[column].rolling(window=period).mean()
            std = df[column].rolling(window=period).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            # Calculate %B (position within bands)
            percent_b = (df[column] - lower_band) / (upper_band - lower_band)

            # Bandwidth (measure of volatility)
            bandwidth = (upper_band - lower_band) / sma

            result = pd.DataFrame({
                f'BBL_{period}_{std_dev}': lower_band,
                f'BBM_{period}_{std_dev}': sma,
                f'BBU_{period}_{std_dev}': upper_band,
                f'BBB_{period}_{std_dev}': percent_b,
                f'BBW_{period}_{std_dev}': bandwidth
            }, index=df.index)

            return result

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            period (int): Period for ATR (default 14)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: ATR values

        Example:
            >>> atr = VolatilityIndicators.atr(df, period=14)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.atr(high=df['High'], low=df['Low'],
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

            # Calculate ATR using Wilder's smoothing
            atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

            return atr

    @staticmethod
    def keltner_channels(df: pd.DataFrame, period: int = 20,
                        atr_period: int = 10, multiplier: float = 2.0,
                        use_library: bool = True) -> pd.DataFrame:
        """
        Calculate Keltner Channels.

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            period (int): Period for EMA (default 20)
            atr_period (int): Period for ATR (default 10)
            multiplier (float): ATR multiplier (default 2.0)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.DataFrame: DataFrame with Lower, Middle, Upper channels

        Example:
            >>> kc = VolatilityIndicators.keltner_channels(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.kc(high=df['High'], low=df['Low'],
                           close=df['Close'], length=period,
                           scalar=multiplier, mamode='ema')
        else:
            # Custom implementation
            # Middle line (EMA of close)
            middle = df['Close'].ewm(span=period, adjust=False).mean()

            # Calculate ATR
            atr = VolatilityIndicators.atr(df, period=atr_period, use_library=False)

            # Upper and lower bands
            upper = middle + (multiplier * atr)
            lower = middle - (multiplier * atr)

            result = pd.DataFrame({
                f'KCL_{period}_{atr_period}_{multiplier}': lower,
                f'KCM_{period}_{atr_period}_{multiplier}': middle,
                f'KCU_{period}_{atr_period}_{multiplier}': upper
            }, index=df.index)

            return result

    @staticmethod
    def donchian_channels(df: pd.DataFrame, period: int = 20,
                         use_library: bool = True) -> pd.DataFrame:
        """
        Calculate Donchian Channels.

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            period (int): Period for channels (default 20)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.DataFrame: DataFrame with Lower, Middle, Upper channels

        Example:
            >>> dc = VolatilityIndicators.donchian_channels(df, period=20)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.donchian(high=df['High'], low=df['Low'],
                                 close=df['Close'], lower_length=period,
                                 upper_length=period)
        else:
            # Custom implementation
            upper = df['High'].rolling(window=period).max()
            lower = df['Low'].rolling(window=period).min()
            middle = (upper + lower) / 2

            result = pd.DataFrame({
                f'DCL_{period}': lower,
                f'DCM_{period}': middle,
                f'DCU_{period}': upper
            }, index=df.index)

            return result

    @staticmethod
    def historical_volatility(df: pd.DataFrame, column: str = 'Close',
                             period: int = 20, annualize: bool = True) -> pd.Series:
        """
        Calculate historical volatility (standard deviation of returns).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate volatility on
            period (int): Period for volatility calculation
            annualize (bool): Whether to annualize the volatility

        Returns:
            pd.Series: Historical volatility values

        Example:
            >>> hv = VolatilityIndicators.historical_volatility(df, period=20)
        """
        # Calculate log returns
        log_returns = np.log(df[column] / df[column].shift(1))

        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=period).std()

        # Annualize if requested (assuming 252 trading days)
        if annualize:
            volatility = volatility * np.sqrt(252)

        return volatility

    @staticmethod
    def parkinson_volatility(df: pd.DataFrame, period: int = 20,
                            annualize: bool = True) -> pd.Series:
        """
        Calculate Parkinson's volatility (uses high and low prices).

        Args:
            df (pd.DataFrame): Input data with High and Low
            period (int): Period for volatility calculation
            annualize (bool): Whether to annualize the volatility

        Returns:
            pd.Series: Parkinson's volatility values

        Example:
            >>> pv = VolatilityIndicators.parkinson_volatility(df, period=20)
        """
        hl_ratio = np.log(df['High'] / df['Low'])
        volatility = np.sqrt((hl_ratio ** 2).rolling(window=period).mean() / (4 * np.log(2)))

        if annualize:
            volatility = volatility * np.sqrt(252)

        return volatility

    @staticmethod
    def detect_bollinger_squeeze(bb_df: pd.DataFrame,
                                 threshold: float = 0.02) -> pd.Series:
        """
        Detect Bollinger Band squeeze (low volatility period).

        Args:
            bb_df (pd.DataFrame): Bollinger Bands data
            threshold (float): Bandwidth threshold for squeeze

        Returns:
            pd.Series: Boolean series indicating squeeze periods

        Example:
            >>> bb = VolatilityIndicators.bollinger_bands(df)
            >>> squeeze = VolatilityIndicators.detect_bollinger_squeeze(bb)
        """
        # Get bandwidth column
        bw_col = [col for col in bb_df.columns if 'BBW_' in col][0]
        bandwidth = bb_df[bw_col]

        # Squeeze when bandwidth is below threshold
        squeeze = bandwidth < threshold

        return squeeze

    @staticmethod
    def detect_bollinger_breakout(df: pd.DataFrame, bb_df: pd.DataFrame,
                                  column: str = 'Close') -> Tuple[pd.Series, pd.Series]:
        """
        Detect Bollinger Band breakouts.

        Args:
            df (pd.DataFrame): Original price data
            bb_df (pd.DataFrame): Bollinger Bands data
            column (str): Price column to check

        Returns:
            Tuple[pd.Series, pd.Series]: Upper breakout, Lower breakout signals

        Example:
            >>> bb = VolatilityIndicators.bollinger_bands(df)
            >>> upper_break, lower_break = VolatilityIndicators.detect_bollinger_breakout(df, bb)
        """
        # Get band columns
        bbu_col = [col for col in bb_df.columns if 'BBU_' in col][0]
        bbl_col = [col for col in bb_df.columns if 'BBL_' in col][0]

        upper_band = bb_df[bbu_col]
        lower_band = bb_df[bbl_col]

        # Breakout signals
        upper_breakout = df[column] > upper_band
        lower_breakout = df[column] < lower_band

        return upper_breakout, lower_breakout

    @staticmethod
    def ulcer_index(df: pd.DataFrame, column: str = 'Close',
                   period: int = 14) -> pd.Series:
        """
        Calculate Ulcer Index (downside volatility measure).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate on
            period (int): Period for calculation

        Returns:
            pd.Series: Ulcer Index values

        Example:
            >>> ui = VolatilityIndicators.ulcer_index(df, period=14)
        """
        # Calculate percentage drawdown from highest high
        highest = df[column].rolling(window=period).max()
        drawdown = ((df[column] - highest) / highest) * 100

        # Square the drawdowns
        squared_drawdown = drawdown ** 2

        # Calculate average and take square root
        ulcer = np.sqrt(squared_drawdown.rolling(window=period).mean())

        return ulcer

    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range.

        Args:
            df (pd.DataFrame): Input data with High, Low, Close

        Returns:
            pd.Series: True Range values

        Example:
            >>> tr = VolatilityIndicators.true_range(df)
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr

    @staticmethod
    def mass_index(df: pd.DataFrame, fast: int = 9, slow: int = 25) -> pd.Series:
        """
        Calculate Mass Index (trend reversal indicator).

        Args:
            df (pd.DataFrame): Input data with High and Low
            fast (int): Fast EMA period (default 9)
            slow (int): Slow EMA period (default 25)

        Returns:
            pd.Series: Mass Index values

        Example:
            >>> mi = VolatilityIndicators.mass_index(df)
        """
        high_low = df['High'] - df['Low']
        ema1 = high_low.ewm(span=fast, adjust=False).mean()
        ema2 = ema1.ewm(span=fast, adjust=False).mean()

        ratio = ema1 / ema2
        mass = ratio.rolling(window=slow).sum()

        return mass


# Convenience functions
def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20,
                             std_dev: float = 2.0) -> pd.DataFrame:
    """Quick function to calculate Bollinger Bands."""
    return VolatilityIndicators.bollinger_bands(df, period=period, std_dev=std_dev)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Quick function to calculate ATR."""
    return VolatilityIndicators.atr(df, period=period)


def calculate_keltner_channels(df: pd.DataFrame, period: int = 20,
                               atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """Quick function to calculate Keltner Channels."""
    return VolatilityIndicators.keltner_channels(df, period=period,
                                                 atr_period=atr_period, multiplier=multiplier)


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Fetch sample data
    df = get_stock_data('AAPL', start='2023-01-01')

    # Calculate volatility indicators
    bb = VolatilityIndicators.bollinger_bands(df, period=20)
    df = pd.concat([df, bb], axis=1)

    df['ATR'] = VolatilityIndicators.atr(df, period=14)

    kc = VolatilityIndicators.keltner_channels(df)
    df = pd.concat([df, kc], axis=1)

    df['Hist_Vol'] = VolatilityIndicators.historical_volatility(df, period=20)

    # Detect squeeze
    squeeze = VolatilityIndicators.detect_bollinger_squeeze(bb)

    # Detect breakouts
    upper_break, lower_break = VolatilityIndicators.detect_bollinger_breakout(df, bb)

    print("\nData with volatility indicators:")
    bb_cols = [col for col in df.columns if 'BB' in col]
    print(df[['Close'] + bb_cols + ['ATR', 'Hist_Vol']].tail())

    print(f"\nBollinger squeeze periods: {squeeze.sum()}")
    print(f"Upper band breakouts: {upper_break.sum()}")
    print(f"Lower band breakouts: {lower_break.sum()}")
