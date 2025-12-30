"""
Backtesting Engine

This module provides a wrapper around the backtesting.py library
with utilities for running and comparing multiple strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type, Any
from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy
import matplotlib.pyplot as plt


class BacktestEngine:
    """
    Wrapper for backtesting.py library with convenience methods.
    """

    def __init__(self, data: pd.DataFrame,
                 initial_cash: float = 10000,
                 commission: float = 0.002,
                 margin: float = 1.0,
                 trade_on_close: bool = False,
                 hedging: bool = False,
                 exclusive_orders: bool = False):
        """
        Initialize backtesting engine.

        Args:
            data (pd.DataFrame): OHLCV data with DatetimeIndex
            initial_cash (float): Starting capital
            commission (float): Commission as fraction (0.002 = 0.2%)
            margin (float): Margin requirement (1.0 = no leverage)
            trade_on_close (bool): Execute trades on close price
            hedging (bool): Allow hedging (multiple positions)
            exclusive_orders (bool): Cancel pending orders on new signal

        Example:
            >>> engine = BacktestEngine(df, initial_cash=10000)
            >>> results = engine.run_backtest(SMACrossover)
        """
        # Validate data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        self.data = data[required_columns].copy()
        self.initial_cash = initial_cash
        self.commission = commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.hedging = hedging
        self.exclusive_orders = exclusive_orders

        self.backtest = None
        self.results = None

    def run_backtest(self, strategy: Type[Strategy],
                    strategy_params: Optional[Dict[str, Any]] = None,
                    optimize: bool = False,
                    optimize_params: Optional[Dict] = None,
                    constraint: Optional[callable] = None,
                    maximize: str = 'Equity Final [$]',
                    max_tries: Optional[int] = None,
                    return_heatmap: bool = False) -> pd.Series:
        """
        Run backtest with specified strategy.

        Args:
            strategy (Type[Strategy]): Strategy class to backtest
            strategy_params (dict): Strategy parameters
            optimize (bool): Whether to optimize parameters
            optimize_params (dict): Parameters grid for optimization
            constraint (callable): Optimization constraint function
            maximize (str): Metric to maximize during optimization
            max_tries (int): Max optimization iterations
            return_heatmap (bool): Return optimization heatmap

        Returns:
            pd.Series: Backtest statistics

        Example:
            >>> results = engine.run_backtest(SMACrossover,
            ...                               strategy_params={'fast_period': 50})
        """
        # Create backtest instance
        self.backtest = Backtest(
            self.data,
            strategy,
            cash=self.initial_cash,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            hedging=self.hedging,
            exclusive_orders=self.exclusive_orders
        )

        # Run backtest
        if optimize:
            if optimize_params is None:
                raise ValueError("optimize_params required when optimize=True")

            self.results = self.backtest.optimize(
                **optimize_params,
                constraint=constraint,
                maximize=maximize,
                max_tries=max_tries,
                return_heatmap=return_heatmap
            )
        else:
            # Set strategy parameters if provided
            if strategy_params:
                for key, value in strategy_params.items():
                    setattr(strategy, key, value)

            self.results = self.backtest.run()

        return self.results

    def plot_results(self, **plot_kwargs):
        """
        Plot backtest results.

        Args:
            **plot_kwargs: Additional arguments for plot

        Example:
            >>> engine.plot_results()
        """
        if self.backtest is None:
            raise ValueError("Must run backtest first")

        self.backtest.plot(**plot_kwargs)

    def get_trades(self) -> pd.DataFrame:
        """
        Get detailed trade log.

        Returns:
            pd.DataFrame: Trade history

        Example:
            >>> trades = engine.get_trades()
        """
        if self.results is None:
            raise ValueError("Must run backtest first")

        return self.results['_trades']

    def compare_strategies(self, strategies: Dict[str, Type[Strategy]],
                          strategy_params: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            strategies (dict): Dictionary of {name: Strategy class}
            strategy_params (dict): Dictionary of {name: params dict}

        Returns:
            pd.DataFrame: Comparison table

        Example:
            >>> comparison = engine.compare_strategies({
            ...     'SMA': SMACrossover,
            ...     'RSI': RSIMeanReversion
            ... })
        """
        results = []

        for name, strategy in strategies.items():
            # Get params for this strategy
            params = strategy_params.get(name, {}) if strategy_params else {}

            # Run backtest
            stats = self.run_backtest(strategy, strategy_params=params)

            # Extract key metrics
            results.append({
                'Strategy': name,
                'Return [%]': stats['Return [%]'],
                'Sharpe Ratio': stats['Sharpe Ratio'],
                'Max Drawdown [%]': stats['Max. Drawdown [%]'],
                'Win Rate [%]': stats['Win Rate [%]'],
                'Trades': stats['# Trades'],
                'Avg Trade [%]': stats['Avg. Trade [%]'],
                'Best Trade [%]': stats['Best Trade [%]'],
                'Worst Trade [%]': stats['Worst Trade [%]'],
                'Calmar Ratio': stats.get('Calmar Ratio', np.nan)
            })

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('Strategy')

        return comparison_df

    def walk_forward_analysis(self, strategy: Type[Strategy],
                             train_period: int,
                             test_period: int,
                             optimize_params: Optional[Dict] = None,
                             maximize: str = 'Sharpe Ratio') -> pd.DataFrame:
        """
        Perform walk-forward analysis.

        Walk-forward analysis trains on a window, tests on the next window,
        then moves forward. This prevents overfitting.

        Args:
            strategy (Type[Strategy]): Strategy to test
            train_period (int): Training window in days
            test_period (int): Test window in days
            optimize_params (dict): Parameters to optimize
            maximize (str): Metric to maximize

        Returns:
            pd.DataFrame: Walk-forward results

        Example:
            >>> wf_results = engine.walk_forward_analysis(
            ...     SMACrossover, train_period=252, test_period=63
            ... )
        """
        results = []
        data_length = len(self.data)
        current_idx = train_period

        while current_idx + test_period <= data_length:
            # Split data
            train_data = self.data.iloc[current_idx - train_period:current_idx]
            test_data = self.data.iloc[current_idx:current_idx + test_period]

            # Optimize on training data
            if optimize_params:
                train_engine = BacktestEngine(
                    train_data,
                    initial_cash=self.initial_cash,
                    commission=self.commission
                )
                train_stats = train_engine.run_backtest(
                    strategy,
                    optimize=True,
                    optimize_params=optimize_params,
                    maximize=maximize
                )

                # Extract best parameters
                best_params = {}
                for key in optimize_params.keys():
                    best_params[key] = train_stats._strategy.__dict__[key]
            else:
                best_params = {}

            # Test on out-of-sample data
            test_engine = BacktestEngine(
                test_data,
                initial_cash=self.initial_cash,
                commission=self.commission
            )
            test_stats = test_engine.run_backtest(
                strategy,
                strategy_params=best_params
            )

            # Store results
            results.append({
                'Period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
                'Return [%]': test_stats['Return [%]'],
                'Sharpe Ratio': test_stats['Sharpe Ratio'],
                'Max Drawdown [%]': test_stats['Max. Drawdown [%]'],
                'Trades': test_stats['# Trades'],
                **{f'Param_{k}': v for k, v in best_params.items()}
            })

            # Move forward
            current_idx += test_period

        return pd.DataFrame(results)


def quick_backtest(data: pd.DataFrame, strategy: Type[Strategy],
                   initial_cash: float = 10000,
                   commission: float = 0.002,
                   strategy_params: Optional[Dict] = None,
                   plot: bool = True) -> pd.Series:
    """
    Quick function to run a backtest.

    Args:
        data (pd.DataFrame): OHLCV data
        strategy (Type[Strategy]): Strategy class
        initial_cash (float): Starting capital
        commission (float): Commission rate
        strategy_params (dict): Strategy parameters
        plot (bool): Whether to plot results

    Returns:
        pd.Series: Backtest statistics

    Example:
        >>> stats = quick_backtest(df, SMACrossover, plot=True)
    """
    engine = BacktestEngine(data, initial_cash=initial_cash, commission=commission)
    stats = engine.run_backtest(strategy, strategy_params=strategy_params)

    print("=== Backtest Results ===")
    print(f"Strategy: {strategy.__name__}")
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"\nPerformance:")
    print(f"  Return: {stats['Return [%]']:.2f}%")
    print(f"  Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Calmar Ratio: {stats.get('Calmar Ratio', 'N/A')}")
    print(f"\nTrades:")
    print(f"  Total Trades: {stats['# Trades']}")
    print(f"  Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"  Avg Trade: {stats['Avg. Trade [%]']:.2f}%")
    print(f"  Best Trade: {stats['Best Trade [%]']:.2f}%")
    print(f"  Worst Trade: {stats['Worst Trade [%]']:.2f}%")

    if plot:
        engine.plot_results()

    return stats


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data
    from src.backtesting.strategies import SMACrossover, RSIMeanReversion

    # Fetch data
    df = get_stock_data('AAPL', start='2020-01-01', end='2023-12-31')

    # Initialize engine
    engine = BacktestEngine(df, initial_cash=10000, commission=0.002)

    # Run single backtest
    print("Running SMA Crossover backtest...")
    results = engine.run_backtest(SMACrossover)
    print(results)

    # Compare strategies
    print("\n" + "="*50)
    print("Comparing multiple strategies...")
    comparison = engine.compare_strategies({
        'SMA Crossover': SMACrossover,
        'RSI Mean Reversion': RSIMeanReversion
    })
    print(comparison)

    # Plot
    engine.plot_results()
