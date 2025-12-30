"""
Trading Strategies for Backtesting

This module provides pre-built trading strategies for backtesting
using the backtesting.py library.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover
from typing import Optional


class SMACrossover(Strategy):
    """
    Simple Moving Average Crossover Strategy.

    Buy when fast SMA crosses above slow SMA (golden cross).
    Sell when fast SMA crosses below slow SMA (death cross).
    """

    # Strategy parameters
    fast_period = 50
    slow_period = 200

    def init(self):
        """Initialize indicators."""
        close = self.data.Close
        self.fast_sma = self.I(self._sma, close, self.fast_period)
        self.slow_sma = self.I(self._sma, close, self.slow_period)

    def next(self):
        """Execute trading logic."""
        # Golden cross - buy signal
        if crossover(self.fast_sma, self.slow_sma):
            if not self.position:
                self.buy()

        # Death cross - sell signal
        elif crossover(self.slow_sma, self.fast_sma):
            if self.position:
                self.position.close()

    @staticmethod
    def _sma(values, period):
        """Calculate Simple Moving Average."""
        return pd.Series(values).rolling(period).mean()


class RSIMeanReversion(Strategy):
    """
    RSI Mean Reversion Strategy.

    Buy when RSI is oversold (< lower threshold).
    Sell when RSI is overbought (> upper threshold).
    """

    # Strategy parameters
    rsi_period = 14
    rsi_lower = 30
    rsi_upper = 70

    def init(self):
        """Initialize indicators."""
        close = self.data.Close
        self.rsi = self.I(self._rsi, close, self.rsi_period)

    def next(self):
        """Execute trading logic."""
        # Oversold - buy signal
        if self.rsi[-1] < self.rsi_lower:
            if not self.position:
                self.buy()

        # Overbought - sell signal
        elif self.rsi[-1] > self.rsi_upper:
            if self.position:
                self.position.close()

    @staticmethod
    def _rsi(values, period):
        """Calculate RSI."""
        delta = pd.Series(values).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Buy when MACD line crosses above signal line.
    Sell when MACD line crosses below signal line.
    """

    # Strategy parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        """Initialize indicators."""
        close = self.data.Close
        self.macd_line, self.signal_line = self.I(
            self._macd, close, self.fast_period,
            self.slow_period, self.signal_period
        )

    def next(self):
        """Execute trading logic."""
        # MACD crosses above signal - buy
        if crossover(self.macd_line, self.signal_line):
            if not self.position:
                self.buy()

        # MACD crosses below signal - sell
        elif crossover(self.signal_line, self.macd_line):
            if self.position:
                self.position.close()

    @staticmethod
    def _macd(values, fast, slow, signal):
        """Calculate MACD and Signal line."""
        prices = pd.Series(values)
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line


class BollingerBandStrategy(Strategy):
    """
    Bollinger Band Mean Reversion Strategy.

    Buy when price touches lower band (oversold).
    Sell when price touches upper band (overbought).
    """

    # Strategy parameters
    bb_period = 20
    bb_std = 2

    def init(self):
        """Initialize indicators."""
        close = self.data.Close
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            self._bollinger_bands, close, self.bb_period, self.bb_std
        )

    def next(self):
        """Execute trading logic."""
        price = self.data.Close[-1]

        # Price touches lower band - buy signal
        if price <= self.bb_lower[-1]:
            if not self.position:
                self.buy()

        # Price touches upper band - sell signal
        elif price >= self.bb_upper[-1]:
            if self.position:
                self.position.close()

        # Exit at middle band (optional)
        elif self.position and price >= self.bb_middle[-1]:
            self.position.close()

    @staticmethod
    def _bollinger_bands(values, period, std_dev):
        """Calculate Bollinger Bands."""
        prices = pd.Series(values)
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower


class TrendFollowing(Strategy):
    """
    Combined Trend Following Strategy.

    Uses multiple indicators:
    - SMA for trend direction
    - ADX for trend strength
    - ATR for position sizing
    """

    # Strategy parameters
    sma_period = 50
    adx_period = 14
    adx_threshold = 25
    atr_period = 14

    def init(self):
        """Initialize indicators."""
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.sma = self.I(self._sma, close, self.sma_period)
        self.adx = self.I(self._adx, high, low, close, self.adx_period)
        self.atr = self.I(self._atr, high, low, close, self.atr_period)

    def next(self):
        """Execute trading logic."""
        price = self.data.Close[-1]

        # Strong uptrend - buy signal
        if (price > self.sma[-1] and
            self.adx[-1] > self.adx_threshold and
            not self.position):
            self.buy()

        # Trend reversal or weakness - sell signal
        elif self.position and (price < self.sma[-1] or
                                self.adx[-1] < self.adx_threshold):
            self.position.close()

    @staticmethod
    def _sma(values, period):
        """Calculate Simple Moving Average."""
        return pd.Series(values).rolling(period).mean()

    @staticmethod
    def _adx(high, low, close, period):
        """Calculate ADX (Average Directional Index)."""
        df = pd.DataFrame({'High': high, 'Low': low, 'Close': close})

        # Calculate +DM and -DM
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        df['DMplus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            df['High'] - df['High'].shift(1), 0
        )
        df['DMplus'] = np.where(df['DMplus'] < 0, 0, df['DMplus'])

        df['DMminus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            df['Low'].shift(1) - df['Low'], 0
        )
        df['DMminus'] = np.where(df['DMminus'] < 0, 0, df['DMminus'])

        # Calculate smoothed TR, +DM, -DM
        TRn = df['TR'].rolling(window=period).sum()
        DMplusN = df['DMplus'].rolling(window=period).sum()
        DMminusN = df['DMminus'].rolling(window=period).sum()

        # Calculate +DI and -DI
        df['DIplus'] = 100 * (DMplusN / TRn)
        df['DIminus'] = 100 * (DMminusN / TRn)

        # Calculate DX and ADX
        df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
        adx = df['DX'].rolling(window=period).mean()

        return adx

    @staticmethod
    def _atr(high, low, close, period):
        """Calculate ATR (Average True Range)."""
        df = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = df['TR'].rolling(window=period).mean()
        return atr


class MLPredictionStrategy(Strategy):
    """
    Machine Learning Based Strategy.

    Uses pre-trained ML model predictions to generate signals.
    Requires a model that predicts next-day returns.
    """

    # Strategy parameters
    prediction_threshold = 0.01  # 1% predicted return to trigger trade

    def init(self):
        """
        Initialize strategy.

        Note: This strategy expects the data to have a 'Prediction' column
        with ML model predictions already calculated.
        """
        if 'Prediction' not in self.data.df.columns:
            raise ValueError("Data must include 'Prediction' column with ML forecasts")

        self.predictions = self.data.df['Prediction'].values

    def next(self):
        """Execute trading logic based on ML predictions."""
        current_idx = len(self.data) - 1

        if current_idx >= len(self.predictions):
            return

        predicted_return = self.predictions[current_idx]

        # Positive prediction above threshold - buy
        if predicted_return > self.prediction_threshold and not self.position:
            self.buy()

        # Negative prediction or below threshold - sell
        elif predicted_return < 0 and self.position:
            self.position.close()


class MultiStrategyCombo(Strategy):
    """
    Combined Multi-Indicator Strategy.

    Uses multiple signals:
    - Trend: SMA crossover
    - Momentum: RSI
    - Volume: Volume confirmation
    """

    # Strategy parameters
    fast_sma = 20
    slow_sma = 50
    rsi_period = 14
    rsi_lower = 30
    rsi_upper = 70
    volume_ma = 20

    def init(self):
        """Initialize indicators."""
        close = self.data.Close
        volume = self.data.Volume

        self.fast_sma_ind = self.I(self._sma, close, self.fast_sma)
        self.slow_sma_ind = self.I(self._sma, close, self.slow_sma)
        self.rsi_ind = self.I(self._rsi, close, self.rsi_period)
        self.volume_ma_ind = self.I(self._sma, volume, self.volume_ma)

    def next(self):
        """Execute trading logic with multiple confirmations."""
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # Buy conditions (all must be true)
        trend_up = self.fast_sma_ind[-1] > self.slow_sma_ind[-1]
        oversold = self.rsi_ind[-1] < self.rsi_lower
        volume_confirm = volume > self.volume_ma_ind[-1]

        if trend_up and oversold and volume_confirm and not self.position:
            self.buy()

        # Sell conditions
        trend_down = self.fast_sma_ind[-1] < self.slow_sma_ind[-1]
        overbought = self.rsi_ind[-1] > self.rsi_upper

        if self.position and (trend_down or overbought):
            self.position.close()

    @staticmethod
    def _sma(values, period):
        """Calculate Simple Moving Average."""
        return pd.Series(values).rolling(period).mean()

    @staticmethod
    def _rsi(values, period):
        """Calculate RSI."""
        delta = pd.Series(values).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


if __name__ == "__main__":
    # Example usage
    from backtesting import Backtest
    from src.data.fetcher import get_stock_data

    # Fetch data
    df = get_stock_data('AAPL', start='2020-01-01', end='2023-12-31')

    # Ensure required columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Run backtest with SMA Crossover
    bt = Backtest(df, SMACrossover, cash=10000, commission=0.002)
    stats = bt.run()

    print("=== SMA Crossover Strategy ===")
    print(stats)
    print(f"\nReturn: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

    # Plot results
    bt.plot()
