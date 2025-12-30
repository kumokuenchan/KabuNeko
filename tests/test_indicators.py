"""
Unit tests for technical indicators
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.indicators.trend import TrendIndicators
from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators


class TestTrendIndicators:
    """Tests for trend indicators"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        data = {
            'Open': np.random.uniform(100, 150, 200),
            'High': np.random.uniform(105, 155, 200),
            'Low': np.random.uniform(95, 145, 200),
            'Close': np.random.uniform(100, 150, 200),
            'Volume': np.random.uniform(1000000, 10000000, 200)
        }
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        return df

    def test_sma_calculation(self, sample_data):
        """Test Simple Moving Average"""
        sma = TrendIndicators.calculate_sma(sample_data, period=20)

        assert sma is not None
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_data)

        # First 19 values should be NaN
        assert sma.iloc[:19].isna().all()
        # Rest should have values
        assert sma.iloc[19:].notna().all()

        # SMA should be within reasonable range
        assert (sma.dropna() >= sample_data['Close'].min()).all()
        assert (sma.dropna() <= sample_data['Close'].max()).all()

    def test_ema_calculation(self, sample_data):
        """Test Exponential Moving Average"""
        ema = TrendIndicators.calculate_ema(sample_data, period=20)

        assert ema is not None
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)

        # EMA should have values after period-1
        assert ema.iloc[19:].notna().all()

    def test_macd_calculation(self, sample_data):
        """Test MACD"""
        macd_df = TrendIndicators.calculate_macd(sample_data)

        assert macd_df is not None
        assert isinstance(macd_df, pd.DataFrame)
        assert 'MACD' in macd_df.columns
        assert 'Signal' in macd_df.columns
        assert 'MACD_Histogram' in macd_df.columns
        assert len(macd_df) == len(sample_data)

    def test_adx_calculation(self, sample_data):
        """Test ADX"""
        adx = TrendIndicators.calculate_adx(sample_data, period=14)

        assert adx is not None
        assert isinstance(adx, pd.Series)

        # ADX should be between 0 and 100
        assert (adx.dropna() >= 0).all()
        assert (adx.dropna() <= 100).all()

    def test_golden_cross_detection(self, sample_data):
        """Test golden cross detection"""
        signals = TrendIndicators.detect_golden_cross(sample_data)

        assert signals is not None
        assert isinstance(signals, pd.Series)
        assert signals.dtype == bool


class TestMomentumIndicators:
    """Tests for momentum indicators"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        data = {
            'Open': np.random.uniform(100, 150, 200),
            'High': np.random.uniform(105, 155, 200),
            'Low': np.random.uniform(95, 145, 200),
            'Close': np.random.uniform(100, 150, 200),
            'Volume': np.random.uniform(1000000, 10000000, 200)
        }
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        return df

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation"""
        rsi = MomentumIndicators.calculate_rsi(sample_data, period=14)

        assert rsi is not None
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)

        # RSI should be between 0 and 100
        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()

    def test_stochastic_calculation(self, sample_data):
        """Test Stochastic Oscillator"""
        stoch_df = MomentumIndicators.calculate_stochastic(sample_data)

        assert stoch_df is not None
        assert isinstance(stoch_df, pd.DataFrame)
        assert 'Stoch_K' in stoch_df.columns
        assert 'Stoch_D' in stoch_df.columns

        # Values should be between 0 and 100
        assert (stoch_df['Stoch_K'].dropna() >= 0).all()
        assert (stoch_df['Stoch_K'].dropna() <= 100).all()

    def test_williams_r_calculation(self, sample_data):
        """Test Williams %R"""
        williams = MomentumIndicators.calculate_williams_r(sample_data, period=14)

        assert williams is not None
        # Williams %R should be between -100 and 0
        assert (williams.dropna() >= -100).all()
        assert (williams.dropna() <= 0).all()

    def test_roc_calculation(self, sample_data):
        """Test Rate of Change"""
        roc = MomentumIndicators.calculate_roc(sample_data, period=12)

        assert roc is not None
        assert isinstance(roc, pd.Series)


class TestVolatilityIndicators:
    """Tests for volatility indicators"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        data = {
            'Open': np.random.uniform(100, 150, 200),
            'High': np.random.uniform(105, 155, 200),
            'Low': np.random.uniform(95, 145, 200),
            'Close': np.random.uniform(100, 150, 200),
            'Volume': np.random.uniform(1000000, 10000000, 200)
        }
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        return df

    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands"""
        bb_df = VolatilityIndicators.calculate_bollinger_bands(sample_data, period=20)

        assert bb_df is not None
        assert isinstance(bb_df, pd.DataFrame)
        assert 'BB_Upper' in bb_df.columns
        assert 'BB_Middle' in bb_df.columns
        assert 'BB_Lower' in bb_df.columns

        # Upper should be greater than middle, middle greater than lower
        valid_rows = bb_df.dropna()
        assert (valid_rows['BB_Upper'] >= valid_rows['BB_Middle']).all()
        assert (valid_rows['BB_Middle'] >= valid_rows['BB_Lower']).all()

    def test_atr_calculation(self, sample_data):
        """Test Average True Range"""
        atr = VolatilityIndicators.calculate_atr(sample_data, period=14)

        assert atr is not None
        assert isinstance(atr, pd.Series)

        # ATR should be positive
        assert (atr.dropna() >= 0).all()

    def test_keltner_channels(self, sample_data):
        """Test Keltner Channels"""
        kc_df = VolatilityIndicators.calculate_keltner_channels(sample_data)

        assert kc_df is not None
        assert 'KC_Upper' in kc_df.columns
        assert 'KC_Middle' in kc_df.columns
        assert 'KC_Lower' in kc_df.columns


class TestVolumeIndicators:
    """Tests for volume indicators"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        close_prices = 100 + np.cumsum(np.random.randn(200) * 2)  # Random walk
        data = {
            'Open': close_prices - np.random.uniform(0, 2, 200),
            'High': close_prices + np.random.uniform(0, 3, 200),
            'Low': close_prices - np.random.uniform(0, 3, 200),
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 10000000, 200)
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_obv_calculation(self, sample_data):
        """Test On-Balance Volume"""
        obv = VolumeIndicators.calculate_obv(sample_data)

        assert obv is not None
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(sample_data)

    def test_vwap_calculation(self, sample_data):
        """Test VWAP"""
        vwap = VolumeIndicators.calculate_vwap(sample_data)

        assert vwap is not None
        assert isinstance(vwap, pd.Series)

        # VWAP should be within reasonable range
        assert (vwap.dropna() > 0).all()

    def test_ad_line_calculation(self, sample_data):
        """Test Accumulation/Distribution Line"""
        ad_line = VolumeIndicators.calculate_ad_line(sample_data)

        assert ad_line is not None
        assert isinstance(ad_line, pd.Series)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
