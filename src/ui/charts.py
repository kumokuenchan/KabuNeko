"""
Chart Utilities Module

Provides reusable chart creation functions for consistent styling
across the dashboard.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Dict


def create_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    height: int = 500
) -> go.Figure:
    """
    Create a candlestick chart for stock prices.

    Args:
        df: DataFrame with OHLC columns
        ticker: Stock ticker symbol
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=height,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_line_chart(
    df: pd.DataFrame,
    y_column: str,
    title: str,
    y_title: str = 'Value',
    x_title: str = 'Date',
    height: int = 400,
    color: str = 'blue'
) -> go.Figure:
    """
    Create a simple line chart.

    Args:
        df: DataFrame with data
        y_column: Column name for y-axis
        title: Chart title
        y_title: Y-axis label
        x_title: X-axis label
        height: Chart height in pixels
        color: Line color

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[y_column],
        mode='lines',
        name=y_column,
        line=dict(color=color)
    ))

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title=x_title,
        template='plotly_white',
        height=height
    )

    return fig


def create_multi_line_chart(
    df: pd.DataFrame,
    y_columns: List[str],
    title: str,
    y_title: str = 'Value',
    height: int = 400
) -> go.Figure:
    """
    Create a multi-line chart for comparing multiple series.

    Args:
        df: DataFrame with data
        y_columns: List of column names to plot
        title: Chart title
        y_title: Y-axis label
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for col in y_columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col
        ))

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title='Date',
        template='plotly_white',
        height=height,
        hovermode='x unified'
    )

    return fig


def create_volume_chart(
    df: pd.DataFrame,
    ticker: str,
    height: int = 300
) -> go.Figure:
    """
    Create a volume bar chart.

    Args:
        df: DataFrame with Volume column
        ticker: Stock ticker symbol
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title=f'{ticker} Trading Volume',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_white',
        height=height
    )

    return fig


def create_bar_chart(
    x_values: List,
    y_values: List,
    title: str,
    x_title: str = 'Category',
    y_title: str = 'Value',
    height: int = 400,
    colors: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a bar chart.

    Args:
        x_values: X-axis values
        y_values: Y-axis values
        title: Chart title
        x_title: X-axis label
        y_title: Y-axis label
        height: Chart height in pixels
        colors: Optional list of bar colors

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    marker_config = {'color': colors} if colors else {}

    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        marker=marker_config
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template='plotly_white',
        height=height
    )

    return fig


def create_heatmap(
    data: pd.DataFrame,
    title: str,
    height: int = 400,
    colorscale: str = 'RdYlGn'
) -> go.Figure:
    """
    Create a heatmap (typically for correlation matrices).

    Args:
        data: DataFrame with numeric data
        title: Chart title
        height: Chart height in pixels
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.columns,
        colorscale=colorscale,
        zmid=0
    ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=height
    )

    return fig


def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str = '',
    colors: Optional[List[str]] = None,
    height: int = 300,
    hole: float = 0.0
) -> go.Figure:
    """
    Create a pie chart or donut chart.

    Args:
        labels: Slice labels
        values: Slice values
        title: Chart title
        colors: Optional slice colors
        height: Chart height in pixels
        hole: Hole size for donut chart (0-1)

    Returns:
        Plotly Figure object
    """
    marker_config = {'colors': colors} if colors else {}

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=marker_config,
        hole=hole
    )])

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=height
    )

    return fig


def create_scatter_with_fill(
    x_values: List,
    y_values: List,
    title: str,
    y_title: str = 'Value',
    color: str = 'green',
    height: int = 400
) -> go.Figure:
    """
    Create a scatter plot with area fill (e.g., for cumulative P&L).

    Args:
        x_values: X-axis values
        y_values: Y-axis values
        title: Chart title
        y_title: Y-axis label
        color: Fill color
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        name='Value',
        line=dict(color=color, width=3),
        fill='tozeroy'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_title,
        height=height,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig
