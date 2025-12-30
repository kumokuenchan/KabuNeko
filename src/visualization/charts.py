"""
Visualization module for creating stock charts and plots.

This module provides functions for creating candlestick charts,
line charts, and other financial visualizations using matplotlib
and plotly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Dict, List
import warnings

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False
    warnings.warn("mplfinance not installed. Some chart features unavailable.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not installed. Interactive charts unavailable.")


class StockCharts:
    """
    A class for creating stock market visualizations.
    """

    @staticmethod
    def plot_candlestick(
        df: pd.DataFrame,
        title: str = "Stock Price",
        volume: bool = True,
        indicators: Optional[Dict[str, pd.Series]] = None,
        style: str = 'yahoo',
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None
    ):
        """
        Create a candlestick chart using mplfinance.

        Args:
            df (pd.DataFrame): Stock data with OHLCV columns
            title (str): Chart title
            volume (bool): Whether to show volume
            indicators (Dict): Dictionary of indicator name -> Series to overlay
            style (str): Chart style ('yahoo', 'charles', 'classic', etc.)
            figsize (tuple): Figure size
            save_path (str): Path to save the chart

        Example:
            >>> indicators = {'SMA_20': df['SMA_20'], 'SMA_50': df['SMA_50']}
            >>> StockCharts.plot_candlestick(df, indicators=indicators)
        """
        if not HAS_MPLFINANCE:
            print("mplfinance not installed. Using simple line plot instead.")
            StockCharts.plot_price_line(df, title=title)
            return

        # Prepare additional plots
        add_plots = []
        if indicators:
            for name, series in indicators.items():
                add_plots.append(mpf.make_addplot(series, label=name))

        # Create the plot
        kwargs = {
            'type': 'candle',
            'style': style,
            'title': title,
            'volume': volume,
            'figsize': figsize,
            'ylabel': 'Price',
            'ylabel_lower': 'Volume' if volume else '',
        }

        if add_plots:
            kwargs['addplot'] = add_plots

        if save_path:
            kwargs['savefig'] = save_path

        mpf.plot(df, **kwargs)

    @staticmethod
    def plot_price_line(
        df: pd.DataFrame,
        title: str = "Stock Price",
        columns: List[str] = None,
        figsize: tuple = (14, 6)
    ):
        """
        Create a simple line chart of stock prices.

        Args:
            df (pd.DataFrame): Stock data
            title (str): Chart title
            columns (List[str]): Columns to plot (default: ['Close'])
            figsize (tuple): Figure size

        Example:
            >>> StockCharts.plot_price_line(df, columns=['Close', 'SMA_20', 'SMA_50'])
        """
        if columns is None:
            columns = ['Close']

        plt.figure(figsize=figsize)

        for col in columns:
            if col in df.columns:
                plt.plot(df.index, df[col], label=col, linewidth=2)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_with_volume(
        df: pd.DataFrame,
        title: str = "Stock Price and Volume",
        figsize: tuple = (14, 8)
    ):
        """
        Create a price chart with volume subplot.

        Args:
            df (pd.DataFrame): Stock data with Close and Volume
            title (str): Chart title
            figsize (tuple): Figure size

        Example:
            >>> StockCharts.plot_with_volume(df)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={'height_ratios': [3, 1]})

        # Price chart
        ax1.plot(df.index, df['Close'], label='Close Price', color='#2E86AB', linewidth=2)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Volume chart
        colors = ['green' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else 'red'
                  for i in range(len(df))]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_indicators(
        df: pd.DataFrame,
        price_cols: List[str],
        oscillators: Optional[Dict[str, pd.Series]] = None,
        title: str = "Technical Analysis",
        figsize: tuple = (14, 10)
    ):
        """
        Create a multi-panel chart with price and oscillator indicators.

        Args:
            df (pd.DataFrame): Stock data
            price_cols (List[str]): Columns to plot on price chart
            oscillators (Dict): Dictionary of oscillator indicators with ranges
                               e.g., {'RSI': (df['RSI'], (0, 100), [30, 70])}
            title (str): Chart title
            figsize (tuple): Figure size

        Example:
            >>> oscillators = {'RSI': (df['RSI'], (0, 100), [30, 70])}
            >>> StockCharts.plot_indicators(df, ['Close', 'SMA_20'], oscillators)
        """
        n_plots = 2 if oscillators else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize,
                                 gridspec_kw={'height_ratios': [3, 1] if n_plots == 2 else [1]})

        if n_plots == 1:
            axes = [axes]

        # Price chart
        ax_price = axes[0]
        for col in price_cols:
            if col in df.columns:
                ax_price.plot(df.index, df[col], label=col, linewidth=2)

        ax_price.set_title(title, fontsize=16, fontweight='bold')
        ax_price.set_ylabel('Price ($)', fontsize=12)
        ax_price.legend(fontsize=10)
        ax_price.grid(True, alpha=0.3)

        # Oscillators
        if oscillators:
            ax_osc = axes[1]
            for name, (series, y_range, levels) in oscillators.items():
                ax_osc.plot(df.index, series, label=name, linewidth=2)

                # Add reference levels
                if levels:
                    for level in levels:
                        ax_osc.axhline(y=level, color='gray', linestyle='--',
                                      linewidth=1, alpha=0.5)

                if y_range:
                    ax_osc.set_ylim(y_range)

            ax_osc.set_ylabel('Oscillator Value', fontsize=12)
            ax_osc.set_xlabel('Date', fontsize=12)
            ax_osc.legend(fontsize=10)
            ax_osc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_interactive_chart(
        df: pd.DataFrame,
        title: str = "Stock Price",
        indicators: Optional[Dict[str, pd.Series]] = None,
        volume: bool = True
    ):
        """
        Create an interactive candlestick chart using Plotly.

        Args:
            df (pd.DataFrame): Stock data with OHLCV columns
            title (str): Chart title
            indicators (Dict): Dictionary of indicator name -> Series
            volume (bool): Whether to show volume

        Example:
            >>> indicators = {'SMA_20': df['SMA_20'], 'SMA_50': df['SMA_50']}
            >>> StockCharts.create_interactive_chart(df, indicators=indicators)
        """
        if not HAS_PLOTLY:
            print("Plotly not installed. Cannot create interactive chart.")
            return

        # Create subplots
        rows = 2 if volume else 1
        row_heights = [0.7, 0.3] if volume else [1.0]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=(title, 'Volume') if volume else (title,)
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add indicators
        if indicators:
            for name, series in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=series,
                        mode='lines',
                        name=name,
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )

        # Volume bar chart
        if volume:
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i]
                     else 'green' for i in range(len(df))]

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True,
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        fig.show()

    @staticmethod
    def plot_comparison(
        stocks_dict: Dict[str, pd.DataFrame],
        column: str = 'Close',
        normalize: bool = True,
        title: str = "Stock Comparison",
        figsize: tuple = (14, 6)
    ):
        """
        Compare multiple stocks on the same chart.

        Args:
            stocks_dict (Dict[str, pd.DataFrame]): Dictionary of ticker -> DataFrame
            column (str): Column to compare
            normalize (bool): Whether to normalize to starting value
            title (str): Chart title
            figsize (tuple): Figure size

        Example:
            >>> stocks = {'AAPL': df_aapl, 'MSFT': df_msft, 'GOOGL': df_googl}
            >>> StockCharts.plot_comparison(stocks, normalize=True)
        """
        plt.figure(figsize=figsize)

        for ticker, df in stocks_dict.items():
            if column in df.columns:
                series = df[column]

                if normalize:
                    # Normalize to starting value (percentage change)
                    series = (series / series.iloc[0] - 1) * 100

                plt.plot(series.index, series, label=ticker, linewidth=2)

        ylabel = "Return (%)" if normalize else "Price ($)"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_returns_distribution(
        df: pd.DataFrame,
        column: str = 'Return_1d',
        bins: int = 50,
        title: str = "Returns Distribution",
        figsize: tuple = (12, 5)
    ):
        """
        Plot the distribution of returns.

        Args:
            df (pd.DataFrame): Stock data with returns
            column (str): Returns column to plot
            bins (int): Number of histogram bins
            title (str): Chart title
            figsize (tuple): Figure size

        Example:
            >>> StockCharts.plot_returns_distribution(df, column='Return_1d')
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        ax1.set_title('Histogram', fontsize=14)
        ax1.set_xlabel('Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(df[column].dropna())
        ax2.set_title('Box Plot', fontsize=14)
        ax2.set_ylabel('Return', fontsize=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Convenience functions
def plot_stock(df: pd.DataFrame, title: str = "Stock Price", interactive: bool = False):
    """
    Quick function to plot stock data.

    Args:
        df (pd.DataFrame): Stock data
        title (str): Chart title
        interactive (bool): Use interactive plotly chart
    """
    if interactive:
        StockCharts.create_interactive_chart(df, title=title)
    else:
        StockCharts.plot_candlestick(df, title=title)


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data
    from src.indicators.trend import calculate_sma

    # Fetch data
    df = get_stock_data('AAPL', start='2023-01-01')

    # Add indicators
    df['SMA_20'] = calculate_sma(df, period=20)
    df['SMA_50'] = calculate_sma(df, period=50)

    # Plot
    print("Creating candlestick chart...")
    indicators = {'SMA_20': df['SMA_20'], 'SMA_50': df['SMA_50']}
    StockCharts.plot_candlestick(df, indicators=indicators)

    print("\nCreating interactive chart...")
    StockCharts.create_interactive_chart(df, indicators=indicators)
