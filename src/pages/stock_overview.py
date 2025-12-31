"""
Page: Stock Overview

Real-time stock price charts, key metrics, and trading volume visualization.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data.fetcher import get_stock_data


def render():
    """Render the stock overview page"""

    st.title("📊 Stock Overview")

    # Input controls
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value=st.session_state.get('default_stocks', ['AAPL'])[0],
            help="Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()

    with col2:
        period = st.selectbox(
            "Time Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Custom"],
            index=3
        )

    with col3:
        if period == "Custom":
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", value=datetime.now())
        else:
            days_map = {
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365,
                "2 Years": 730,
                "5 Years": 1825
            }
            start_date = datetime.now() - timedelta(days=days_map[period])
            end_date = datetime.now()
            st.write(f"From: {start_date.strftime('%Y-%m-%d')}")

    if st.button("📈 Load Stock Data", type="primary"):
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                # Fetch stock data
                df = get_stock_data(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                if df is not None and len(df) > 0:
                    st.session_state['current_stock'] = ticker
                    st.session_state['current_data'] = df
                    st.success(f"✅ Successfully loaded {len(df)} days of data for {ticker}")
                else:
                    st.error(f"❌ No data found for {ticker}. Please check the ticker symbol.")
                    return

            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return

    # Display data if available
    if 'current_data' in st.session_state and st.session_state['current_data'] is not None:
        df = st.session_state['current_data']
        ticker = st.session_state.get('current_stock', 'Stock')

        # Key metrics
        st.markdown("### Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")

        with col2:
            high_52week = df['High'].tail(min(252, len(df))).max()
            st.metric("52-Week High", f"${high_52week:.2f}")

        with col3:
            low_52week = df['Low'].tail(min(252, len(df))).min()
            st.metric("52-Week Low", f"${low_52week:.2f}")

        with col4:
            avg_volume = df['Volume'].tail(20).mean()
            st.metric("Avg Volume (20d)", f"{avg_volume/1e6:.2f}M")

        with col5:
            total_return = ((current_price / df['Close'].iloc[0]) - 1) * 100
            st.metric("Total Return", f"{total_return:.2f}%")

        # Price chart
        st.markdown("### Price Chart")

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
            height=500,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, width="stretch")

        # Volume chart
        st.markdown("### Volume")

        fig_vol = go.Figure()

        fig_vol.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))

        fig_vol.update_layout(
            title=f'{ticker} Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig_vol, width="stretch")

        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.tail(100), width="stretch")

    else:
        st.info("👆 Enter a stock ticker and click 'Load Stock Data' to get started!")
