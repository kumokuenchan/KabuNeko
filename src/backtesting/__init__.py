"""
Backtesting Module

This module provides backtesting capabilities including:
- Pre-built trading strategies
- Backtesting engine wrapper
- Performance metrics and analysis
"""

from .strategies import (
    SMACrossover,
    RSIMeanReversion,
    MACDStrategy,
    BollingerBandStrategy,
    TrendFollowing,
    MLPredictionStrategy,
    MultiStrategyCombo
)

from .engine import (
    BacktestEngine,
    quick_backtest
)

from .metrics import (
    PerformanceMetrics,
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns
)

__all__ = [
    # Strategies
    'SMACrossover',
    'RSIMeanReversion',
    'MACDStrategy',
    'BollingerBandStrategy',
    'TrendFollowing',
    'MLPredictionStrategy',
    'MultiStrategyCombo',

    # Engine
    'BacktestEngine',
    'quick_backtest',

    # Metrics
    'PerformanceMetrics',
    'plot_equity_curve',
    'plot_drawdown',
    'plot_monthly_returns',
]
