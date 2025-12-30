"""
Volume indicators for technical analysis.

This module provides volume-based indicators like OBV, VWAP,
Volume Moving Average, and other volume analysis tools.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta not installed. Using custom implementations only.")


class VolumeIndicators:
    """
    A class containing volume indicator calculations.
    """

    @staticmethod
    def obv(df: pd.DataFrame, use_library: bool = True) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        Args:
            df (pd.DataFrame): Input data with Close and Volume
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: OBV values

        Example:
            >>> obv = VolumeIndicators.obv(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.obv(close=df['Close'], volume=df['Volume'])
        else:
            # Custom implementation
            obv = pd.Series(index=df.index, dtype='float64')
            obv.iloc[0] = df['Volume'].iloc[0]

            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            return obv

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        Args:
            df (pd.DataFrame): Input data with High, Low, Close, Volume

        Returns:
            pd.Series: VWAP values

        Example:
            >>> vwap = VolumeIndicators.vwap(df)
        """
        # Typical price
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        # Cumulative typical price * volume
        cum_vol_price = (typical_price * df['Volume']).cumsum()

        # Cumulative volume
        cum_volume = df['Volume'].cumsum()

        # VWAP
        vwap = cum_vol_price / cum_volume

        return vwap

    @staticmethod
    def vwap_intraday(df: pd.DataFrame, reset_daily: bool = True) -> pd.Series:
        """
        Calculate intraday VWAP (resets each day).

        Args:
            df (pd.DataFrame): Input data with High, Low, Close, Volume
            reset_daily (bool): Whether to reset VWAP daily

        Returns:
            pd.Series: VWAP values

        Example:
            >>> vwap = VolumeIndicators.vwap_intraday(df)
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        if reset_daily:
            # Group by date and calculate VWAP for each day
            df_copy = df.copy()
            df_copy['Date'] = df_copy.index.date
            df_copy['TypicalPrice'] = typical_price
            df_copy['TP_Volume'] = typical_price * df['Volume']

            vwap = df_copy.groupby('Date').apply(
                lambda x: x['TP_Volume'].cumsum() / x['Volume'].cumsum()
            ).reset_index(level=0, drop=True)
        else:
            vwap = VolumeIndicators.vwap(df)

        return vwap

    @staticmethod
    def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Volume Simple Moving Average.

        Args:
            df (pd.DataFrame): Input data with Volume
            period (int): Period for SMA

        Returns:
            pd.Series: Volume SMA values

        Example:
            >>> vol_sma = VolumeIndicators.volume_sma(df, period=20)
        """
        return df['Volume'].rolling(window=period).mean()

    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Volume Ratio (current volume / average volume).

        Args:
            df (pd.DataFrame): Input data with Volume
            period (int): Period for average volume

        Returns:
            pd.Series: Volume ratio values

        Example:
            >>> vol_ratio = VolumeIndicators.volume_ratio(df, period=20)
        """
        avg_volume = df['Volume'].rolling(window=period).mean()
        ratio = df['Volume'] / avg_volume

        return ratio

    @staticmethod
    def ad_line(df: pd.DataFrame, use_library: bool = True) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.

        Args:
            df (pd.DataFrame): Input data with High, Low, Close, Volume
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: A/D Line values

        Example:
            >>> ad = VolumeIndicators.ad_line(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.ad(high=df['High'], low=df['Low'],
                           close=df['Close'], volume=df['Volume'])
        else:
            # Custom implementation
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            clv = clv.fillna(0)  # Handle division by zero

            ad = (clv * df['Volume']).cumsum()

            return ad

    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).

        Args:
            df (pd.DataFrame): Input data with High, Low, Close, Volume
            period (int): Period for CMF (default 20)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: CMF values

        Example:
            >>> cmf = VolumeIndicators.cmf(df, period=20)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.cmf(high=df['High'], low=df['Low'],
                            close=df['Close'], volume=df['Volume'], length=period)
        else:
            # Custom implementation
            mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mf_multiplier = mf_multiplier.fillna(0)

            mf_volume = mf_multiplier * df['Volume']

            cmf = mf_volume.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()

            return cmf

    @staticmethod
    def eom(df: pd.DataFrame, period: int = 14,
            use_library: bool = True) -> pd.Series:
        """
        Calculate Ease of Movement (EOM).

        Args:
            df (pd.DataFrame): Input data with High, Low, Volume
            period (int): Period for smoothing (default 14)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: EOM values

        Example:
            >>> eom = VolumeIndicators.eom(df, period=14)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.eom(high=df['High'], low=df['Low'],
                            close=df['Close'], volume=df['Volume'], length=period)
        else:
            # Custom implementation
            distance = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
            box_ratio = (df['Volume'] / 1000000) / (df['High'] - df['Low'])

            eom = distance / box_ratio
            eom_ma = eom.rolling(window=period).mean()

            return eom_ma

    @staticmethod
    def force_index(df: pd.DataFrame, period: int = 13,
                   use_library: bool = True) -> pd.Series:
        """
        Calculate Force Index.

        Args:
            df (pd.DataFrame): Input data with Close and Volume
            period (int): Period for EMA smoothing (default 13)
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: Force Index values

        Example:
            >>> fi = VolumeIndicators.force_index(df, period=13)
        """
        # Raw force index
        force = (df['Close'] - df['Close'].shift(1)) * df['Volume']

        # Smooth with EMA
        force_ema = force.ewm(span=period, adjust=False).mean()

        return force_ema

    @staticmethod
    def vpt(df: pd.DataFrame, use_library: bool = True) -> pd.Series:
        """
        Calculate Volume Price Trend (VPT).

        Args:
            df (pd.DataFrame): Input data with Close and Volume
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: VPT values

        Example:
            >>> vpt = VolumeIndicators.vpt(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.pvt(close=df['Close'], volume=df['Volume'])
        else:
            # Custom implementation
            price_change_pct = df['Close'].pct_change()
            vpt = (price_change_pct * df['Volume']).cumsum()

            return vpt

    @staticmethod
    def nvi(df: pd.DataFrame, use_library: bool = True) -> pd.Series:
        """
        Calculate Negative Volume Index (NVI).

        Args:
            df (pd.DataFrame): Input data with Close and Volume
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: NVI values

        Example:
            >>> nvi = VolumeIndicators.nvi(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.nvi(close=df['Close'], volume=df['Volume'])
        else:
            # Custom implementation
            nvi = pd.Series(index=df.index, dtype='float64')
            nvi.iloc[0] = 1000  # Starting value

            for i in range(1, len(df)):
                if df['Volume'].iloc[i] < df['Volume'].iloc[i-1]:
                    price_change = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                    nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change)
                else:
                    nvi.iloc[i] = nvi.iloc[i-1]

            return nvi

    @staticmethod
    def pvi(df: pd.DataFrame, use_library: bool = True) -> pd.Series:
        """
        Calculate Positive Volume Index (PVI).

        Args:
            df (pd.DataFrame): Input data with Close and Volume
            use_library (bool): Use pandas-ta library if available

        Returns:
            pd.Series: PVI values

        Example:
            >>> pvi = VolumeIndicators.pvi(df)
        """
        if use_library and HAS_PANDAS_TA:
            return df.ta.pvi(close=df['Close'], volume=df['Volume'])
        else:
            # Custom implementation
            pvi = pd.Series(index=df.index, dtype='float64')
            pvi.iloc[0] = 1000  # Starting value

            for i in range(1, len(df)):
                if df['Volume'].iloc[i] > df['Volume'].iloc[i-1]:
                    price_change = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                    pvi.iloc[i] = pvi.iloc[i-1] * (1 + price_change)
                else:
                    pvi.iloc[i] = pvi.iloc[i-1]

            return pvi

    @staticmethod
    def detect_volume_spike(df: pd.DataFrame, threshold: float = 2.0,
                           period: int = 20) -> pd.Series:
        """
        Detect volume spikes (volume significantly above average).

        Args:
            df (pd.DataFrame): Input data with Volume
            threshold (float): Multiplier for average volume (default 2.0)
            period (int): Period for average volume calculation

        Returns:
            pd.Series: Boolean series indicating volume spikes

        Example:
            >>> spikes = VolumeIndicators.detect_volume_spike(df, threshold=2.0)
        """
        avg_volume = df['Volume'].rolling(window=period).mean()
        volume_spike = df['Volume'] > (avg_volume * threshold)

        return volume_spike

    @staticmethod
    def detect_unusual_volume(df: pd.DataFrame, std_dev: float = 2.0,
                             period: int = 20) -> pd.Series:
        """
        Detect unusual volume using standard deviation.

        Args:
            df (pd.DataFrame): Input data with Volume
            std_dev (float): Number of standard deviations (default 2.0)
            period (int): Period for calculation

        Returns:
            pd.Series: Boolean series indicating unusual volume

        Example:
            >>> unusual = VolumeIndicators.detect_unusual_volume(df)
        """
        vol_mean = df['Volume'].rolling(window=period).mean()
        vol_std = df['Volume'].rolling(window=period).std()

        upper_threshold = vol_mean + (std_dev * vol_std)

        unusual_volume = df['Volume'] > upper_threshold

        return unusual_volume


# Convenience functions
def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Quick function to calculate OBV."""
    return VolumeIndicators.obv(df)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Quick function to calculate VWAP."""
    return VolumeIndicators.vwap(df)


def calculate_ad_line(df: pd.DataFrame) -> pd.Series:
    """Quick function to calculate A/D Line."""
    return VolumeIndicators.ad_line(df)


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Fetch sample data
    df = get_stock_data('AAPL', start='2023-01-01')

    # Calculate volume indicators
    df['OBV'] = VolumeIndicators.obv(df)
    df['VWAP'] = VolumeIndicators.vwap(df)
    df['AD_Line'] = VolumeIndicators.ad_line(df)
    df['CMF'] = VolumeIndicators.cmf(df, period=20)
    df['Force_Index'] = VolumeIndicators.force_index(df, period=13)
    df['Volume_SMA'] = VolumeIndicators.volume_sma(df, period=20)

    # Detect volume anomalies
    volume_spikes = VolumeIndicators.detect_volume_spike(df, threshold=2.0)
    unusual_volume = VolumeIndicators.detect_unusual_volume(df)

    print("\nData with volume indicators:")
    print(df[['Close', 'Volume', 'OBV', 'VWAP', 'CMF', 'Force_Index']].tail())

    print(f"\nVolume spikes detected: {volume_spikes.sum()}")
    print(f"Unusual volume periods: {unusual_volume.sum()}")

    # Show dates with volume spikes
    spike_dates = df[volume_spikes].index
    if len(spike_dates) > 0:
        print(f"\nRecent volume spike dates:")
        print(spike_dates[-5:].tolist())
