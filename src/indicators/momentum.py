"""
Momentum indicators for technical analysis.

This module provides momentum-based indicators like RSI, Stochastic,
Williams %R, and Rate of Change.
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


class MomentumIndicators:
    """
    A class containing momentum indicator calculations.
    """

    @staticmethod
    def rsi(df: pd.DataFrame, column: str = 'Close', period: int = 14,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate RSI on
            period (int): Period for RSI (default 14)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: RSI values (0-100)

        Example:
            >>> rsi = MomentumIndicators.rsi(df, period=14)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.rsi(close=df[column], length=period)
        else:
            # Custom implementation
            return MomentumIndicators.rsi_custom(df[column], period)

    @staticmethod
    def rsi_custom(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Custom RSI implementation for educational purposes.

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions.

        Formula:
            RSI = 100 - (100 / (1 + RS))
            where RS = Average Gain / Average Loss

        Args:
            prices (pd.Series): Price series
            period (int): Period for RSI (default 14)

        Returns:
            pd.Series: RSI values (0-100)

        Example:
            >>> rsi = MomentumIndicators.rsi_custom(df['Close'], period=14)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss using Wilder's smoothing
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                   use_library: bool = True) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            k_period (int): %K period (default 14)
            d_period (int): %D period (default 3)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.DataFrame: DataFrame with %K and %D columns

        Example:
            >>> stoch = MomentumIndicators.stochastic(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.stoch(high=df['High'], low=df['Low'],
                              close=df['Close'], k=k_period, d=d_period)
        else:
            # Custom implementation
            high = df['High']
            low = df['Low']
            close = df['Close']

            # Calculate %K
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()

            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))

            # Calculate %D (moving average of %K)
            d = k.rolling(window=d_period).mean()

            result = pd.DataFrame({
                f'STOCHk_{k_period}_{d_period}': k,
                f'STOCHd_{k_period}_{d_period}': d
            }, index=df.index)

            return result

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14,
                   use_library: bool = True) -> pd.Series:
        """
        Calculate Williams %R.

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            period (int): Period for Williams %R (default 14)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: Williams %R values (-100 to 0)

        Example:
            >>> wr = MomentumIndicators.williams_r(df, period=14)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.willr(high=df['High'], low=df['Low'],
                              close=df['Close'], length=period)
        else:
            # Custom implementation
            high = df['High']
            low = df['Low']
            close = df['Close']

            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()

            wr = -100 * ((highest_high - close) / (highest_high - lowest_low))

            return wr

    @staticmethod
    def roc(df: pd.DataFrame, column: str = 'Close', period: int = 12,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Rate of Change (ROC).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate ROC on
            period (int): Period for ROC (default 12)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: ROC values (percentage)

        Example:
            >>> roc = MomentumIndicators.roc(df, period=12)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.roc(close=df[column], length=period)
        else:
            # Custom implementation
            roc = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
            return roc

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).

        Args:
            df (pd.DataFrame): Input data with High, Low, Close
            period (int): Period for CCI (default 20)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: CCI values

        Example:
            >>> cci = MomentumIndicators.cci(df, period=20)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.cci(high=df['High'], low=df['Low'],
                            close=df['Close'], length=period)
        else:
            # Custom implementation
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )

            cci = (typical_price - sma) / (0.015 * mad)
            return cci

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).

        Args:
            df (pd.DataFrame): Input data with High, Low, Close, Volume
            period (int): Period for MFI (default 14)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: MFI values (0-100)

        Example:
            >>> mfi = MomentumIndicators.mfi(df, period=14)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.mfi(high=df['High'], low=df['Low'],
                            close=df['Close'], volume=df['Volume'], length=period)
        else:
            # Custom implementation
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']

            # Positive and negative money flow
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

            # Sum over period
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()

            # Money Flow Ratio and Index
            mfr = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + mfr))

            return mfi

    @staticmethod
    def detect_rsi_signals(rsi: pd.Series, overbought: float = 70,
                          oversold: float = 30) -> Tuple[pd.Series, pd.Series]:
        """
        Detect RSI overbought and oversold signals.

        Args:
            rsi (pd.Series): RSI values
            overbought (float): Overbought threshold (default 70)
            oversold (float): Oversold threshold (default 30)

        Returns:
            Tuple[pd.Series, pd.Series]: Oversold signals, Overbought signals

        Example:
            >>> oversold, overbought = MomentumIndicators.detect_rsi_signals(rsi)
        """
        oversold_signal = rsi < oversold
        overbought_signal = rsi > overbought

        return oversold_signal, overbought_signal

    @staticmethod
    def detect_stochastic_signals(stoch_df: pd.DataFrame,
                                  overbought: float = 80,
                                  oversold: float = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Stochastic oscillator signals.

        Args:
            stoch_df (pd.DataFrame): Stochastic data with %K and %D
            overbought (float): Overbought threshold (default 80)
            oversold (float): Oversold threshold (default 20)

        Returns:
            Tuple[pd.Series, pd.Series]: Buy signals, Sell signals

        Example:
            >>> buy, sell = MomentumIndicators.detect_stochastic_signals(stoch)
        """
        # Get column names (they vary based on parameters)
        k_col = [col for col in stoch_df.columns if 'STOCHk' in col][0]
        d_col = [col for col in stoch_df.columns if 'STOCHd' in col][0]

        k = stoch_df[k_col]
        d = stoch_df[d_col]

        # Buy: %K crosses above %D in oversold territory
        buy_signal = (k > d) & (k.shift(1) <= d.shift(1)) & (k < oversold)

        # Sell: %K crosses below %D in overbought territory
        sell_signal = (k < d) & (k.shift(1) >= d.shift(1)) & (k > overbought)

        return buy_signal, sell_signal

    @staticmethod
    def momentum(df: pd.DataFrame, column: str = 'Close',
                 period: int = 10) -> pd.Series:
        """
        Calculate simple momentum (rate of change over period).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate momentum on
            period (int): Period for momentum

        Returns:
            pd.Series: Momentum values

        Example:
            >>> mom = MomentumIndicators.momentum(df, period=10)
        """
        return df[column] - df[column].shift(period)

    @staticmethod
    def trix(df: pd.DataFrame, column: str = 'Close', period: int = 15,
             use_library: bool = True) -> pd.Series:
        """
        Calculate TRIX (Triple Exponential Average).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate TRIX on
            period (int): Period for TRIX
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: TRIX values

        Example:
            >>> trix = MomentumIndicators.trix(df, period=15)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.trix(close=df[column], length=period)
        else:
            # Custom implementation
            ema1 = df[column].ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()

            trix = ema3.pct_change() * 100
            return trix


# Convenience functions
def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
    """Quick function to calculate RSI."""
    return MomentumIndicators.rsi(df, column=column, period=period)


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14,
                        d_period: int = 3) -> pd.DataFrame:
    """Quick function to calculate Stochastic."""
    return MomentumIndicators.stochastic(df, k_period=k_period, d_period=d_period)


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Quick function to calculate Williams %R."""
    return MomentumIndicators.williams_r(df, period=period)


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Fetch sample data
    df = get_stock_data('AAPL', start='2023-01-01')

    # Calculate momentum indicators
    df['RSI'] = MomentumIndicators.rsi(df, period=14)
    df['Williams_R'] = MomentumIndicators.williams_r(df, period=14)
    df['ROC'] = MomentumIndicators.roc(df, period=12)
    df['MFI'] = MomentumIndicators.mfi(df, period=14)

    # Stochastic
    stoch = MomentumIndicators.stochastic(df)
    df = pd.concat([df, stoch], axis=1)

    # Detect RSI signals
    oversold, overbought = MomentumIndicators.detect_rsi_signals(df['RSI'])

    print("\nData with momentum indicators:")
    print(df[['Close', 'RSI', 'Williams_R', 'ROC', 'MFI']].tail())

    print(f"\nRSI oversold signals: {oversold.sum()}")
    print(f"RSI overbought signals: {overbought.sum()}")

    # Show when RSI is oversold
    oversold_dates = df[oversold].index
    if len(oversold_dates) > 0:
        print(f"\nRecent oversold dates:")
        print(oversold_dates[-5:].tolist())
