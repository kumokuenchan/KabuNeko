"""
Unit tests for data modules (fetcher and preprocessor)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fetcher import StockDataFetcher, get_stock_data, get_multiple_stocks
from src.data.preprocessor import StockDataPreprocessor


class TestStockDataFetcher:
    """Tests for StockDataFetcher class"""

    def test_fetcher_initialization(self):
        """Test fetcher can be initialized"""
        fetcher = StockDataFetcher(cache_dir='data/cache')
        assert fetcher is not None
        assert fetcher.cache_dir == 'data/cache'

    def test_get_stock_data_basic(self):
        """Test basic stock data fetching"""
        df = get_stock_data('AAPL', start='2023-01-01', end='2023-12-31')

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in df.columns

        # Check data integrity
        assert df['High'].min() >= 0
        assert (df['High'] >= df['Low']).all()
        assert (df['High'] >= df['Open']).all()
        assert (df['High'] >= df['Close']).all()

    def test_get_stock_data_invalid_ticker(self):
        """Test handling of invalid ticker"""
        # This should either return None or raise an exception
        try:
            df = get_stock_data('INVALIDTICKER999', start='2023-01-01', end='2023-01-31')
            # If it doesn't raise, it should return empty or None
            assert df is None or len(df) == 0
        except:
            # Exception is acceptable
            pass

    def test_get_multiple_stocks(self):
        """Test fetching multiple stocks"""
        tickers = ['AAPL', 'MSFT']
        stocks_data = get_multiple_stocks(tickers, start='2023-01-01', end='2023-12-31')

        assert stocks_data is not None
        assert isinstance(stocks_data, dict)
        assert len(stocks_data) == 2

        for ticker in tickers:
            assert ticker in stocks_data
            assert isinstance(stocks_data[ticker], pd.DataFrame)
            assert len(stocks_data[ticker]) > 0

    def test_data_date_range(self):
        """Test data is within requested date range"""
        start_date = '2023-06-01'
        end_date = '2023-12-31'

        df = get_stock_data('AAPL', start=start_date, end=end_date)

        assert df.index[0] >= pd.to_datetime(start_date)
        assert df.index[-1] <= pd.to_datetime(end_date)


class TestStockDataPreprocessor:
    """Tests for StockDataPreprocessor class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = {
            'Open': np.random.uniform(100, 150, 100),
            'High': np.random.uniform(105, 155, 100),
            'Low': np.random.uniform(95, 145, 100),
            'Close': np.random.uniform(100, 150, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        # Ensure data integrity
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        return df

    def test_clean_data(self, sample_data):
        """Test data cleaning"""
        # Add some NaN values
        sample_data.iloc[10, 0] = np.nan
        sample_data.iloc[20, 1] = np.nan

        cleaned = StockDataPreprocessor.clean_data(sample_data)

        assert cleaned is not None
        assert isinstance(cleaned, pd.DataFrame)
        # NaN values should be handled
        assert cleaned.isnull().sum().sum() == 0

    def test_add_returns(self, sample_data):
        """Test adding returns columns"""
        result = StockDataPreprocessor.add_returns(sample_data)

        assert 'Daily_Return' in result.columns
        assert 'Log_Return' in result.columns
        assert len(result) == len(sample_data)

        # Check returns are reasonable
        assert result['Daily_Return'].abs().max() < 1.0  # Less than 100% daily move

    def test_normalize_data(self, sample_data):
        """Test data normalization"""
        normalized = StockDataPreprocessor.normalize_data(
            sample_data,
            columns=['Close'],
            method='minmax'
        )

        assert 'Close' in normalized.columns
        # MinMax should scale to 0-1
        assert normalized['Close'].min() >= 0
        assert normalized['Close'].max() <= 1

    def test_split_data(self, sample_data):
        """Test train/test split"""
        train, test = StockDataPreprocessor.split_data(sample_data, train_size=0.8)

        assert len(train) + len(test) == len(sample_data)
        assert len(train) == int(len(sample_data) * 0.8)
        # Train should come before test (time series)
        assert train.index[-1] < test.index[0]

    def test_create_sequences(self, sample_data):
        """Test sequence creation for LSTM"""
        X, y = StockDataPreprocessor.create_sequences(
            sample_data['Close'].values,
            lookback=10
        )

        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert X.shape[1] == 10  # lookback period
        assert len(X) == len(sample_data) - 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
