"""
Performance Metrics for Backtesting

This module provides additional performance metrics and analysis
tools for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt


class PerformanceMetrics:
    """
    Calculate advanced performance metrics for backtesting.
    """

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series,
                               risk_free_rate: float = 0.0,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio.

        Args:
            returns (pd.Series): Return series
            risk_free_rate (float): Annual risk-free rate
            periods_per_year (int): Trading periods per year

        Returns:
            float: Sharpe ratio

        Example:
            >>> sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        return sharpe

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                                risk_free_rate: float = 0.0,
                                periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation only).

        Args:
            returns (pd.Series): Return series
            risk_free_rate (float): Annual risk-free rate
            periods_per_year (int): Trading periods per year

        Returns:
            float: Sortino ratio

        Example:
            >>> sortino = PerformanceMetrics.calculate_sortino_ratio(returns)
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
        return sortino

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, Union[float, pd.Timestamp]]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            equity_curve (pd.Series): Equity curve series

        Returns:
            Dict: Max drawdown %, start date, end date, duration

        Example:
            >>> mdd = PerformanceMetrics.calculate_max_drawdown(equity)
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max * 100

        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()

        # Find start of drawdown (last peak before max drawdown)
        max_dd_start = equity_curve[:max_dd_end].idxmax()

        # Calculate duration
        if isinstance(max_dd_start, pd.Timestamp) and isinstance(max_dd_end, pd.Timestamp):
            duration = (max_dd_end - max_dd_start).days
        else:
            duration = max_dd_end - max_dd_start

        return {
            'Max Drawdown [%]': abs(max_dd),
            'Drawdown Start': max_dd_start,
            'Drawdown End': max_dd_end,
            'Duration [days]': duration
        }

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series,
                               equity_curve: pd.Series,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio (Annual Return / Max Drawdown).

        Args:
            returns (pd.Series): Return series
            equity_curve (pd.Series): Equity curve
            periods_per_year (int): Trading periods per year

        Returns:
            float: Calmar ratio

        Example:
            >>> calmar = PerformanceMetrics.calculate_calmar_ratio(returns, equity)
        """
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * periods_per_year
        max_dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        max_dd = max_dd_info['Max Drawdown [%]']

        if max_dd == 0:
            return 0.0

        return annual_return / max_dd

    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate win rate and related statistics.

        Args:
            trades (pd.DataFrame): Trade log from backtest

        Returns:
            Dict: Win rate, avg win, avg loss, profit factor

        Example:
            >>> win_stats = PerformanceMetrics.calculate_win_rate(trades)
        """
        if len(trades) == 0:
            return {
                'Win Rate [%]': 0.0,
                'Avg Win [%]': 0.0,
                'Avg Loss [%]': 0.0,
                'Profit Factor': 0.0,
                'Largest Win [%]': 0.0,
                'Largest Loss [%]': 0.0
            }

        # Calculate returns
        trades['Return [%]'] = (trades['ReturnPct'] * 100) if 'ReturnPct' in trades.columns else 0

        winning_trades = trades[trades['Return [%]'] > 0]
        losing_trades = trades[trades['Return [%]'] < 0]

        # Win rate
        win_rate = (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0

        # Average win/loss
        avg_win = winning_trades['Return [%]'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['Return [%]'].mean()) if len(losing_trades) > 0 else 0

        # Profit factor
        total_wins = winning_trades['Return [%]'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['Return [%]'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Largest win/loss
        largest_win = winning_trades['Return [%]'].max() if len(winning_trades) > 0 else 0
        largest_loss = abs(losing_trades['Return [%]'].min()) if len(losing_trades) > 0 else 0

        return {
            'Win Rate [%]': win_rate,
            'Avg Win [%]': avg_win,
            'Avg Loss [%]': avg_loss,
            'Profit Factor': profit_factor,
            'Largest Win [%]': largest_win,
            'Largest Loss [%]': largest_loss,
            'Total Wins': len(winning_trades),
            'Total Losses': len(losing_trades)
        }

    @staticmethod
    def calculate_consecutive_stats(trades: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate consecutive wins/losses.

        Args:
            trades (pd.DataFrame): Trade log

        Returns:
            Dict: Max consecutive wins, losses

        Example:
            >>> consec = PerformanceMetrics.calculate_consecutive_stats(trades)
        """
        if len(trades) == 0:
            return {
                'Max Consecutive Wins': 0,
                'Max Consecutive Losses': 0
            }

        # Determine if trade is win or loss
        trades['Win'] = trades['ReturnPct'] > 0 if 'ReturnPct' in trades.columns else False

        # Count consecutive
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for win in trades['Win']:
            if win:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return {
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses
        }

    @staticmethod
    def calculate_risk_adjusted_returns(returns: pd.Series,
                                       periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate various risk-adjusted return metrics.

        Args:
            returns (pd.Series): Return series
            periods_per_year (int): Trading periods per year

        Returns:
            Dict: Various risk metrics

        Example:
            >>> risk_metrics = PerformanceMetrics.calculate_risk_adjusted_returns(returns)
        """
        if len(returns) == 0:
            return {}

        # Annualized return
        annual_return = returns.mean() * periods_per_year

        # Annualized volatility
        annual_volatility = returns.std() * np.sqrt(periods_per_year)

        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()

        return {
            'Annual Return [%]': annual_return * 100,
            'Annual Volatility [%]': annual_volatility * 100,
            'Downside Deviation [%]': downside_deviation * 100,
            'VaR 95% [%]': var_95 * 100,
            'CVaR 95% [%]': cvar_95 * 100
        }

    @staticmethod
    def calculate_comprehensive_metrics(equity_curve: pd.Series,
                                       trades: pd.DataFrame,
                                       initial_cash: float = 10000,
                                       risk_free_rate: float = 0.0) -> pd.Series:
        """
        Calculate comprehensive performance metrics.

        Args:
            equity_curve (pd.Series): Equity curve
            trades (pd.DataFrame): Trade log
            initial_cash (float): Starting capital
            risk_free_rate (float): Risk-free rate

        Returns:
            pd.Series: All metrics

        Example:
            >>> metrics = PerformanceMetrics.calculate_comprehensive_metrics(
            ...     equity, trades
            ... )
        """
        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        metrics = {}

        # Basic metrics
        metrics['Initial Cash [$]'] = initial_cash
        metrics['Final Equity [$]'] = equity_curve.iloc[-1]
        metrics['Total Return [%]'] = ((equity_curve.iloc[-1] - initial_cash) / initial_cash) * 100

        # Risk-adjusted returns
        risk_metrics = PerformanceMetrics.calculate_risk_adjusted_returns(returns)
        metrics.update(risk_metrics)

        # Sharpe and Sortino
        metrics['Sharpe Ratio'] = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['Sortino Ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate)

        # Drawdown
        dd_metrics = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        metrics.update(dd_metrics)

        # Calmar
        metrics['Calmar Ratio'] = PerformanceMetrics.calculate_calmar_ratio(returns, equity_curve)

        # Trade statistics
        if len(trades) > 0:
            win_stats = PerformanceMetrics.calculate_win_rate(trades)
            metrics.update(win_stats)

            consec_stats = PerformanceMetrics.calculate_consecutive_stats(trades)
            metrics.update(consec_stats)

            metrics['Total Trades'] = len(trades)
            metrics['Avg Trade Duration [days]'] = (
                (trades['ExitTime'] - trades['EntryTime']).dt.days.mean()
                if 'EntryTime' in trades.columns and 'ExitTime' in trades.columns
                else 0
            )

        return pd.Series(metrics)


def plot_equity_curve(equity_curve: pd.Series,
                     benchmark: Optional[pd.Series] = None,
                     title: str = 'Equity Curve'):
    """
    Plot equity curve with optional benchmark.

    Args:
        equity_curve (pd.Series): Strategy equity curve
        benchmark (pd.Series): Benchmark equity curve
        title (str): Plot title

    Example:
        >>> plot_equity_curve(equity, benchmark=spy_equity)
    """
    plt.figure(figsize=(14, 7))

    plt.plot(equity_curve.index, equity_curve.values,
             label='Strategy', linewidth=2, color='#2E86AB')

    if benchmark is not None:
        plt.plot(benchmark.index, benchmark.values,
                label='Benchmark', linewidth=2, color='#A23B72', alpha=0.7)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_drawdown(equity_curve: pd.Series, title: str = 'Drawdown'):
    """
    Plot drawdown over time.

    Args:
        equity_curve (pd.Series): Equity curve
        title (str): Plot title

    Example:
        >>> plot_drawdown(equity)
    """
    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100

    plt.figure(figsize=(14, 5))
    plt.fill_between(drawdown.index, drawdown.values, 0,
                     color='red', alpha=0.3, label='Drawdown')
    plt.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_monthly_returns(returns: pd.Series, title: str = 'Monthly Returns'):
    """
    Plot monthly returns heatmap.

    Args:
        returns (pd.Series): Daily returns
        title (str): Plot title

    Example:
        >>> plot_monthly_returns(returns)
    """
    # Resample to monthly
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

    # Create pivot table
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not pd.isna(pivot.iloc[i, j]):
                text = ax.text(j, i, f"{pivot.iloc[i, j]:.1f}%",
                             ha="center", va="center", color="black", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, label='Return (%)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Fetch data
    df = get_stock_data('AAPL', start='2020-01-01', end='2023-12-31')

    # Simulate equity curve
    returns = df['Close'].pct_change().dropna()
    equity = (1 + returns).cumprod() * 10000

    # Calculate metrics
    print("=== Performance Metrics ===")
    metrics = PerformanceMetrics.calculate_comprehensive_metrics(
        equity,
        pd.DataFrame(),  # Empty trades for example
        initial_cash=10000
    )
    print(metrics)

    # Plot equity curve
    plot_equity_curve(equity, title='AAPL Buy & Hold')

    # Plot drawdown
    plot_drawdown(equity)

    # Plot monthly returns
    plot_monthly_returns(returns)
