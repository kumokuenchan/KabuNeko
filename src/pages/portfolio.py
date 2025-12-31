"""
Page: Portfolio Analysis

Track and analyze multiple stocks as a portfolio with performance metrics,
correlation analysis, and comparison charts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data.fetcher import get_multiple_stocks


def render():
    """Portfolio analysis page"""

    st.title("💼 Portfolio Analysis")

    st.markdown("""
    Track and analyze multiple stocks as a portfolio.
    """)

    # Portfolio input
    st.markdown("### Add Stocks to Portfolio")

    col1, col2 = st.columns(2)

    with col1:
        portfolio_tickers = st.text_area(
            "Enter stock tickers (one per line)",
            value="\n".join(st.session_state.get('default_stocks', ['AAPL', 'MSFT', 'GOOGL'])),
            height=150
        )

    with col2:
        period = st.selectbox("Analysis Period", ["1 Month", "3 Months", "6 Months", "1 Year"], index=3)

        days_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        days = days_map[period]

    if st.button("📊 Analyze Portfolio", type="primary"):
        tickers = [t.strip().upper() for t in portfolio_tickers.split('\n') if t.strip()]

        if not tickers:
            st.error("Please enter at least one ticker symbol")
            return

        with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
            try:
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')

                stocks_data = get_multiple_stocks(tickers, start=start_date, end=end_date)

                if not stocks_data:
                    st.error("Failed to fetch stock data")
                    return

                st.success(f"✅ Successfully loaded data for {len(stocks_data)} stocks")

                # Portfolio metrics
                st.markdown("### Portfolio Performance")

                # Calculate returns for each stock
                returns_data = {}
                performance_data = []

                for ticker, df in stocks_data.items():
                    if df is not None and len(df) > 0:
                        start_price = df['Close'].iloc[0]
                        end_price = df['Close'].iloc[-1]
                        total_return = ((end_price / start_price) - 1) * 100

                        returns_data[ticker] = total_return
                        performance_data.append({
                            'Ticker': ticker,
                            'Start Price': start_price,
                            'Current Price': end_price,
                            'Return (%)': total_return,
                            'Change ($)': end_price - start_price
                        })

                # Display performance table
                perf_df = pd.DataFrame(performance_data)
                perf_df = perf_df.sort_values('Return (%)', ascending=False)

                st.dataframe(perf_df.style.format({
                    'Start Price': '${:.2f}',
                    'Current Price': '${:.2f}',
                    'Return (%)': '{:.2f}%',
                    'Change ($)': '${:.2f}'
                }).background_gradient(subset=['Return (%)'], cmap='RdYlGn'), use_container_width=True)

                # Portfolio composition
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Returns Distribution")
                    fig_bar = go.Figure(data=[
                        go.Bar(x=list(returns_data.keys()), y=list(returns_data.values()),
                               marker_color=['green' if v > 0 else 'red' for v in returns_data.values()])
                    ])
                    fig_bar.update_layout(
                        yaxis_title='Return (%)',
                        xaxis_title='Stock',
                        template='plotly_white',
                        height=300
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col2:
                    st.markdown("### Portfolio Summary")
                    avg_return = sum(returns_data.values()) / len(returns_data)
                    best_stock = max(returns_data, key=returns_data.get)
                    worst_stock = min(returns_data, key=returns_data.get)

                    st.metric("Average Return", f"{avg_return:.2f}%")
                    st.metric("Best Performer", f"{best_stock} ({returns_data[best_stock]:.2f}%)")
                    st.metric("Worst Performer", f"{worst_stock} ({returns_data[worst_stock]:.2f}%)")

                # Price comparison chart
                st.markdown("### Price Comparison (Normalized)")

                fig_compare = go.Figure()

                for ticker, df in stocks_data.items():
                    if df is not None and len(df) > 0:
                        # Normalize to 100 at start
                        normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                        fig_compare.add_trace(go.Scatter(
                            x=df.index,
                            y=normalized,
                            name=ticker,
                            mode='lines'
                        ))

                fig_compare.update_layout(
                    title='Portfolio Stocks - Normalized Price Comparison',
                    yaxis_title='Normalized Price (Start = 100)',
                    xaxis_title='Date',
                    template='plotly_white',
                    height=400
                )

                st.plotly_chart(fig_compare, use_container_width=True)

                # Correlation matrix
                st.markdown("### Correlation Matrix")
                st.info("Shows how stocks move together. 1 = perfect correlation, -1 = inverse correlation")

                # Create returns dataframe
                returns_df = pd.DataFrame()
                for ticker, df in stocks_data.items():
                    if df is not None and len(df) > 0:
                        returns_df[ticker] = df['Close'].pct_change()

                correlation_matrix = returns_df.corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10}
                ))

                fig_corr.update_layout(
                    title='Stock Correlation Matrix',
                    height=400
                )

                st.plotly_chart(fig_corr, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error analyzing portfolio: {str(e)}")

    else:
        st.info("👆 Enter stock tickers and click 'Analyze Portfolio' to get started!")
