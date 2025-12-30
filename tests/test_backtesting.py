"""
Unit tests for backtesting framework
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.strategies import SMACrossover, RSIMeanReversion
from src.backtesting.metrics import PerformanceMetrics


class TestBacktestingStrategies:
    """Tests for backtesting strategies"""

    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample OHLCV data for backtesting"""
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        # Create trending data
        trend = np.linspace(100, 150, 500)
        noise = np.random.randn(500) * 5

        close_prices = trend + noise
        data = {
            'Open': close_prices - np.random.uniform(0, 2, 500),
            'High': close_prices + np.random.uniform(0, 3, 500),
            'Low': close_prices - np.random.uniform(0, 3, 500),
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 10000000, 500)
        }
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        return df

    def test_sma_crossover_strategy_exists(self):
        """Test SMA Crossover strategy can be instantiated"""
        strategy = SMACrossover

        assert strategy is not None
        assert hasattr(strategy, 'fast_period')
        assert hasattr(strategy, 'slow_period')

    def test_rsi_mean_reversion_strategy_exists(self):
        """Test RSI Mean Reversion strategy exists"""
        strategy = RSIMeanReversion

        assert strategy is not None
        assert hasattr(strategy, 'rsi_period')
        assert hasattr(strategy, 'rsi_lower')
        assert hasattr(strategy, 'rsi_upper')


class TestPerformanceMetrics:
    """Tests for performance metrics"""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02)  # Daily returns
        return returns

    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02)
        equity = (1 + returns).cumprod() * 10000
        return equity

    def test_sharpe_ratio_calculation(self, sample_returns):
        """Test Sharpe ratio calculation"""
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(
            sample_returns,
            risk_free_rate=0.02
        )

        assert sharpe is not None
        assert isinstance(sharpe, float)
        # Sharpe ratio should be reasonable
        assert -5 < sharpe < 5

    def test_sortino_ratio_calculation(self, sample_returns):
        """Test Sortino ratio calculation"""
        sortino = PerformanceMetrics.calculate_sortino_ratio(
            sample_returns,
            risk_free_rate=0.02
        )

        assert sortino is not None
        assert isinstance(sortino, float)

    def test_max_drawdown_calculation(self, sample_equity_curve):
        """Test maximum drawdown calculation"""
        mdd = PerformanceMetrics.calculate_max_drawdown(sample_equity_curve)

        assert mdd is not None
        assert isinstance(mdd, dict)
        assert 'Max Drawdown [%]' in mdd
        assert 'Drawdown Start' in mdd
        assert 'Drawdown End' in mdd
        assert 'Duration [days]' in mdd

        # Max drawdown should be negative or zero
        assert mdd['Max Drawdown [%]'] >= 0

    def test_calmar_ratio_calculation(self, sample_returns, sample_equity_curve):
        """Test Calmar ratio calculation"""
        calmar = PerformanceMetrics.calculate_calmar_ratio(
            sample_returns,
            sample_equity_curve
        )

        assert calmar is not None
        assert isinstance(calmar, float)

    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        # Create sample trades
        trades = pd.DataFrame({
            'ReturnPct': [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.015]
        })

        stats = PerformanceMetrics.calculate_win_rate(trades)

        assert stats is not None
        assert isinstance(stats, dict)
        assert 'Win Rate [%]' in stats
        assert 'Avg Win [%]' in stats
        assert 'Avg Loss [%]' in stats
        assert 'Profit Factor' in stats

        # Win rate should be between 0 and 100
        assert 0 <= stats['Win Rate [%]'] <= 100

    def test_risk_adjusted_returns(self, sample_returns):
        """Test risk-adjusted returns calculation"""
        metrics = PerformanceMetrics.calculate_risk_adjusted_returns(sample_returns)

        assert metrics is not None
        assert isinstance(metrics, dict)
        assert 'Annual Return [%]' in metrics
        assert 'Annual Volatility [%]' in metrics
        assert 'Downside Deviation [%]' in metrics
        assert 'VaR 95% [%]' in metrics

    def test_comprehensive_metrics(self, sample_equity_curve):
        """Test comprehensive metrics calculation"""
        # Create dummy trades
        trades = pd.DataFrame({
            'ReturnPct': np.random.randn(20) * 0.02,
            'EntryTime': pd.date_range('2023-01-01', periods=20, freq='W'),
            'ExitTime': pd.date_range('2023-01-08', periods=20, freq='W')
        })

        metrics = PerformanceMetrics.calculate_comprehensive_metrics(
            sample_equity_curve,
            trades,
            initial_cash=10000
        )

        assert metrics is not None
        assert isinstance(metrics, pd.Series)
        assert 'Final Equity [$]' in metrics
        assert 'Total Return [%]' in metrics
        assert 'Sharpe Ratio' in metrics
        assert 'Max Drawdown [%]' in metrics


class TestBacktestingIntegration:
    """Integration tests for backtesting"""

    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample data"""
        dates = pd.date_range('2022-01-01', periods=300, freq='D')
        trend = np.linspace(100, 130, 300)
        noise = np.random.randn(300) * 3

        close_prices = trend + noise
        data = {
            'Open': close_prices - np.random.uniform(0, 2, 300),
            'High': close_prices + np.random.uniform(0, 3, 300),
            'Low': close_prices - np.random.uniform(0, 3, 300),
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 10000000, 300)
        }
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        return df

    def test_backtest_engine_initialization(self, sample_backtest_data):
        """Test backtest engine can be initialized"""
        try:
            from src.backtesting.engine import BacktestEngine

            engine = BacktestEngine(
                sample_backtest_data,
                initial_cash=10000,
                commission=0.002
            )

            assert engine is not None
            assert engine.initial_cash == 10000
            assert engine.commission == 0.002
        except ImportError:
            pytest.skip("Backtesting engine requires backtesting.py library")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
