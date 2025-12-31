"""
Page: Home

Welcome page with introduction, quick market stats, and getting started guide.
"""

import streamlit as st
from datetime import datetime, timedelta
from src.data.fetcher import get_stock_data


def render():
    """Render the home page"""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## Welcome to Your Personal Stock Analysis Tool! 👋")
        st.markdown("""
        This dashboard helps you analyze stocks like a professional trader,
        without needing to write any code!

        ### What You Can Do:

        📊 **Stock Overview** - View real-time stock prices, charts, and basic information

        💡 **Investment Advice** - Get smart BUY/SELL/HOLD recommendations with AI-powered analysis

        📉 **Technical Analysis** - Explore advanced indicators like RSI, MACD, Moving Averages

        🤖 **Price Prediction** - Use AI to predict future stock prices

        ⚡ **Backtesting** - Test trading strategies on historical data

        💼 **Portfolio Analysis** - Track and analyze your stock portfolio

        ### How to Get Started:

        1. **Select a page** from the sidebar navigation
        2. **Choose a stock** using the ticker input (e.g., AAPL for Apple)
        3. **Pick your date range** to analyze
        4. **Explore** the interactive charts and data!

        ---

        ### Need Help?

        - **Stock Ticker**: Use standard symbols like AAPL (Apple), MSFT (Microsoft), GOOGL (Google)
        - **Date Range**: Select how far back you want to analyze (default: 1 year)
        - **Indicators**: Hover over charts for detailed information
        """)

    with col2:
        st.markdown("## Quick Stats")

        # Show some quick market stats
        try:
            # Fetch major indices
            sp500 = get_stock_data('^GSPC', start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            dow = get_stock_data('^DJI', start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            nasdaq = get_stock_data('^IXIC', start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))

            if sp500 is not None and len(sp500) >= 2:
                sp_change = ((sp500['Close'].iloc[-1] / sp500['Close'].iloc[-2]) - 1) * 100
                st.metric("S&P 500", f"${sp500['Close'].iloc[-1]:.2f}", f"{sp_change:.2f}%")

            if dow is not None and len(dow) >= 2:
                dow_change = ((dow['Close'].iloc[-1] / dow['Close'].iloc[-2]) - 1) * 100
                st.metric("Dow Jones", f"${dow['Close'].iloc[-1]:.2f}", f"{dow_change:.2f}%")

            if nasdaq is not None and len(nasdaq) >= 2:
                nq_change = ((nasdaq['Close'].iloc[-1] / nasdaq['Close'].iloc[-2]) - 1) * 100
                st.metric("NASDAQ", f"${nasdaq['Close'].iloc[-1]:.2f}", f"{nq_change:.2f}%")

        except Exception as e:
            st.info("Market data will load here")

        st.markdown("---")
        st.markdown("### Popular Stocks")
        st.markdown("""
        - 🍎 **AAPL** - Apple Inc.
        - Ⓜ️ **MSFT** - Microsoft
        - 🔍 **GOOGL** - Google
        - 📦 **AMZN** - Amazon
        - 👍 **META** - Meta (Facebook)
        - ⚡ **TSLA** - Tesla
        - 🎮 **NVDA** - NVIDIA
        """)

    st.markdown("---")
    st.info("💡 **Tip**: Start with the 'Stock Overview' page to view basic stock information!")
