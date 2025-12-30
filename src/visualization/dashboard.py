"""
Interactive Dashboard Components

This module provides interactive dashboard components for Jupyter notebooks
using ipywidgets and plotly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output


class StockDashboard:
    """
    Interactive stock analysis dashboard.
    """

    def __init__(self, tickers: List[str], start_date: str = '2020-01-01'):
        """
        Initialize dashboard.

        Args:
            tickers (list): List of stock tickers
            start_date (str): Start date for data

        Example:
            >>> dashboard = StockDashboard(['AAPL', 'MSFT', 'GOOGL'])
            >>> dashboard.show()
        """
        self.tickers = tickers
        self.start_date = start_date
        self.data_cache = {}
        self.current_ticker = tickers[0] if tickers else None

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch data with caching.

        Args:
            ticker (str): Stock ticker

        Returns:
            pd.DataFrame: Stock data
        """
        if ticker not in self.data_cache:
            from src.data.fetcher import get_stock_data
            self.data_cache[ticker] = get_stock_data(ticker, start=self.start_date)
        return self.data_cache[ticker]

    def create_price_chart(self, ticker: str,
                          show_volume: bool = True,
                          indicators: List[str] = None) -> go.Figure:
        """
        Create interactive price chart.

        Args:
            ticker (str): Stock ticker
            show_volume (bool): Show volume subplot
            indicators (list): List of indicators to plot

        Returns:
            go.Figure: Plotly figure
        """
        df = self.fetch_data(ticker)

        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{ticker} Price', 'Volume')
            )
        else:
            fig = go.Figure()

        # Candlestick chart
        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )

        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)

        # Add indicators
        if indicators:
            from src.indicators import TrendIndicators, MomentumIndicators

            if 'SMA20' in indicators:
                sma20 = TrendIndicators.calculate_sma(df, period=20)
                fig.add_trace(go.Scatter(
                    x=df.index, y=sma20,
                    name='SMA 20', line=dict(color='orange', width=1)
                ), row=1, col=1)

            if 'SMA50' in indicators:
                sma50 = TrendIndicators.calculate_sma(df, period=50)
                fig.add_trace(go.Scatter(
                    x=df.index, y=sma50,
                    name='SMA 50', line=dict(color='blue', width=1)
                ), row=1, col=1)

            if 'SMA200' in indicators:
                sma200 = TrendIndicators.calculate_sma(df, period=200)
                fig.add_trace(go.Scatter(
                    x=df.index, y=sma200,
                    name='SMA 200', line=dict(color='red', width=1)
                ), row=1, col=1)

        # Volume
        if show_volume:
            colors = ['red' if close < open else 'green'
                     for close, open in zip(df['Close'], df['Open'])]

            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ), row=2, col=1)

        # Layout
        fig.update_layout(
            title=f'{ticker} Stock Price',
            xaxis_rangeslider_visible=False,
            height=600 if show_volume else 500,
            template='plotly_white'
        )

        return fig

    def create_technical_panel(self, ticker: str) -> widgets.VBox:
        """
        Create technical analysis panel.

        Args:
            ticker (str): Stock ticker

        Returns:
            widgets.VBox: Widget panel
        """
        df = self.fetch_data(ticker)

        from src.indicators import TrendIndicators, MomentumIndicators, VolatilityIndicators

        # Calculate indicators
        current_price = df['Close'].iloc[-1]
        sma20 = TrendIndicators.calculate_sma(df, 20).iloc[-1]
        sma50 = TrendIndicators.calculate_sma(df, 50).iloc[-1]
        sma200 = TrendIndicators.calculate_sma(df, 200).iloc[-1]
        rsi = MomentumIndicators.calculate_rsi(df, 14).iloc[-1]

        # Create display
        html_content = f"""
        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px;">
            <h3 style="margin-top: 0;">{ticker} Technical Indicators</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px;"><b>Current Price:</b></td>
                    <td style="padding: 8px;">${current_price:.2f}</td>
                </tr>
                <tr style="background-color: white;">
                    <td style="padding: 8px;"><b>SMA 20:</b></td>
                    <td style="padding: 8px;">${sma20:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><b>SMA 50:</b></td>
                    <td style="padding: 8px;">${sma50:.2f}</td>
                </tr>
                <tr style="background-color: white;">
                    <td style="padding: 8px;"><b>SMA 200:</b></td>
                    <td style="padding: 8px;">${sma200:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><b>RSI (14):</b></td>
                    <td style="padding: 8px;">{rsi:.2f}</td>
                </tr>
            </table>

            <h4>Trend Analysis:</h4>
            <p style="margin: 5px 0;">
                {'✅ <b>Bullish</b> - Price above SMA 50' if current_price > sma50 else '❌ <b>Bearish</b> - Price below SMA 50'}
            </p>
            <p style="margin: 5px 0;">
                {'✅ <b>Golden Cross</b> - SMA 50 > SMA 200' if sma50 > sma200 else '❌ <b>Death Cross</b> - SMA 50 < SMA 200'}
            </p>
            <p style="margin: 5px 0;">
                {'⚠️ <b>Overbought</b> - RSI > 70' if rsi > 70 else ('⚠️ <b>Oversold</b> - RSI < 30' if rsi < 30 else '✅ <b>Neutral</b> - RSI in normal range')}
            </p>
        </div>
        """

        return widgets.VBox([widgets.HTML(html_content)])

    def create_fundamental_panel(self, ticker: str) -> widgets.VBox:
        """
        Create fundamental analysis panel.

        Args:
            ticker (str): Stock ticker

        Returns:
            widgets.VBox: Widget panel
        """
        try:
            from src.fundamental import FinancialRatios

            ratios_df = FinancialRatios.get_all_ratios(ticker)

            if ratios_df is not None and not ratios_df.empty:
                # Get latest values
                latest = ratios_df.iloc[-1]

                html_content = f"""
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px;">
                    <h3 style="margin-top: 0;">{ticker} Fundamental Ratios</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px;"><b>P/E Ratio:</b></td>
                            <td style="padding: 8px;">{latest.get('PE_Ratio', 'N/A')}</td>
                        </tr>
                        <tr style="background-color: white;">
                            <td style="padding: 8px;"><b>P/B Ratio:</b></td>
                            <td style="padding: 8px;">{latest.get('PB_Ratio', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><b>ROE:</b></td>
                            <td style="padding: 8px;">{latest.get('ROE', 'N/A')}</td>
                        </tr>
                        <tr style="background-color: white;">
                            <td style="padding: 8px;"><b>Debt/Equity:</b></td>
                            <td style="padding: 8px;">{latest.get('Debt_to_Equity', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><b>Current Ratio:</b></td>
                            <td style="padding: 8px;">{latest.get('Current_Ratio', 'N/A')}</td>
                        </tr>
                    </table>
                </div>
                """
            else:
                html_content = f"""
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffc107;">
                    <h3 style="margin-top: 0;">{ticker} Fundamental Data</h3>
                    <p>⚠️ Fundamental data not available for this ticker.</p>
                </div>
                """

        except Exception as e:
            html_content = f"""
            <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                <h3 style="margin-top: 0;">Error</h3>
                <p>Unable to fetch fundamental data: {str(e)}</p>
            </div>
            """

        return widgets.VBox([widgets.HTML(html_content)])

    def show(self):
        """
        Display the interactive dashboard.
        """
        # Stock selector
        ticker_dropdown = widgets.Dropdown(
            options=self.tickers,
            value=self.current_ticker,
            description='Ticker:',
            style={'description_width': '100px'}
        )

        # Indicator checkboxes
        indicator_checks = widgets.SelectMultiple(
            options=['SMA20', 'SMA50', 'SMA200'],
            value=['SMA50'],
            description='Indicators:',
            style={'description_width': '100px'}
        )

        # Volume toggle
        volume_check = widgets.Checkbox(
            value=True,
            description='Show Volume',
            style={'description_width': '100px'}
        )

        # Output area
        output = widgets.Output()

        def update_dashboard(change=None):
            """Update dashboard when selections change."""
            with output:
                clear_output(wait=True)

                ticker = ticker_dropdown.value
                indicators = list(indicator_checks.value)
                show_volume = volume_check.value

                # Create chart
                fig = self.create_price_chart(ticker, show_volume, indicators)
                fig.show()

                # Create panels
                tech_panel = self.create_technical_panel(ticker)
                fund_panel = self.create_fundamental_panel(ticker)

                # Display panels
                display(widgets.HBox([tech_panel, fund_panel]))

        # Attach event handlers
        ticker_dropdown.observe(update_dashboard, names='value')
        indicator_checks.observe(update_dashboard, names='value')
        volume_check.observe(update_dashboard, names='value')

        # Controls
        controls = widgets.VBox([
            ticker_dropdown,
            indicator_checks,
            volume_check
        ])

        # Initial display
        update_dashboard()

        # Display dashboard
        display(widgets.VBox([
            widgets.HTML("<h2>📊 Stock Analysis Dashboard</h2>"),
            controls,
            output
        ]))


class PortfolioDashboard:
    """
    Portfolio analysis dashboard.
    """

    def __init__(self, portfolio: Dict[str, float], start_date: str = '2020-01-01'):
        """
        Initialize portfolio dashboard.

        Args:
            portfolio (dict): {ticker: shares} dictionary
            start_date (str): Start date for analysis

        Example:
            >>> portfolio = {'AAPL': 10, 'MSFT': 5, 'GOOGL': 3}
            >>> dashboard = PortfolioDashboard(portfolio)
            >>> dashboard.show()
        """
        self.portfolio = portfolio
        self.start_date = start_date

    def calculate_portfolio_value(self) -> pd.DataFrame:
        """
        Calculate portfolio value over time.

        Returns:
            pd.DataFrame: Portfolio value history
        """
        from src.data.fetcher import get_stock_data

        portfolio_values = []

        for ticker, shares in self.portfolio.items():
            df = get_stock_data(ticker, start=self.start_date)
            portfolio_values.append(df['Close'] * shares)

        # Combine all stocks
        portfolio_df = pd.concat(portfolio_values, axis=1)
        portfolio_df.columns = list(self.portfolio.keys())
        portfolio_df['Total'] = portfolio_df.sum(axis=1)

        return portfolio_df

    def create_portfolio_chart(self) -> go.Figure:
        """
        Create portfolio value chart.

        Returns:
            go.Figure: Plotly figure
        """
        portfolio_df = self.calculate_portfolio_value()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value Over Time', 'Individual Holdings'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )

        # Total portfolio value
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['Total'],
            name='Total Portfolio',
            line=dict(color='darkblue', width=3),
            fill='tozeroy'
        ), row=1, col=1)

        # Individual holdings
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        for i, ticker in enumerate(self.portfolio.keys()):
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df[ticker],
                name=ticker,
                line=dict(color=colors[i % len(colors)], width=2)
            ), row=2, col=1)

        fig.update_layout(
            height=700,
            template='plotly_white',
            showlegend=True
        )

        return fig

    def show(self):
        """
        Display portfolio dashboard.
        """
        portfolio_df = self.calculate_portfolio_value()

        # Current values
        current_values = portfolio_df.iloc[-1]
        total_value = current_values['Total']
        initial_value = portfolio_df.iloc[0]['Total']
        total_return = ((total_value - initial_value) / initial_value) * 100

        # Create summary
        html_content = f"""
        <div style="background-color: #e7f3ff; padding: 20px; border-radius: 10px; border: 2px solid #2E86AB;">
            <h2 style="margin-top: 0;">💼 Portfolio Summary</h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
                <tr>
                    <td style="padding: 10px;"><b>Total Value:</b></td>
                    <td style="padding: 10px; text-align: right;"><b>${total_value:,.2f}</b></td>
                </tr>
                <tr style="background-color: white;">
                    <td style="padding: 10px;"><b>Total Return:</b></td>
                    <td style="padding: 10px; text-align: right; color: {'green' if total_return > 0 else 'red'};">
                        <b>{total_return:+.2f}%</b>
                    </td>
                </tr>
            </table>

            <h3>Holdings:</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #2E86AB; color: white;">
                    <th style="padding: 8px; text-align: left;">Ticker</th>
                    <th style="padding: 8px; text-align: right;">Shares</th>
                    <th style="padding: 8px; text-align: right;">Value</th>
                    <th style="padding: 8px; text-align: right;">% of Portfolio</th>
                </tr>
        """

        for ticker, shares in self.portfolio.items():
            value = current_values[ticker]
            percentage = (value / total_value) * 100
            html_content += f"""
                <tr style="background-color: {'white' if list(self.portfolio.keys()).index(ticker) % 2 else '#f8f9fa'};">
                    <td style="padding: 8px;"><b>{ticker}</b></td>
                    <td style="padding: 8px; text-align: right;">{shares}</td>
                    <td style="padding: 8px; text-align: right;">${value:,.2f}</td>
                    <td style="padding: 8px; text-align: right;">{percentage:.1f}%</td>
                </tr>
            """

        html_content += """
            </table>
        </div>
        """

        # Display
        display(widgets.HTML(html_content))

        # Show chart
        fig = self.create_portfolio_chart()
        fig.show()


if __name__ == "__main__":
    # Example usage
    print("Stock Dashboard Example:")
    dashboard = StockDashboard(['AAPL', 'MSFT', 'GOOGL'])
    dashboard.show()
