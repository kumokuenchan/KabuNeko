"""
Stock Analysis Toolkit - Simple Web Interface
==============================================

A user-friendly dashboard for stock market analysis.
No coding required - just select stocks and explore!

How to run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.fetcher import get_stock_data, get_multiple_stocks
from src.indicators.trend import TrendIndicators
from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators
from src.fundamental.ratios import FinancialRatios
from src.models.random_forest import RandomForestPredictor
from src.models.feature_engineering import FeatureEngineer
from src.backtesting.strategies import SMACrossover, RSIMeanReversion

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application"""

    # Header
    st.markdown('<p class="main-header">📈 Stock Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar - Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/line-chart.png", width=100)
        st.title("Navigation")

        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "📊 Stock Overview", "📉 Technical Analysis",
             "🤖 Price Prediction", "⚡ Backtesting", "💼 Portfolio Analysis"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Quick Settings")

        # Popular stock watchlists
        watchlist = st.selectbox(
            "Stock Watchlist",
            ["Custom", "Tech Giants", "Dow Jones 30", "S&P 500 Top 10"]
        )

        if watchlist == "Tech Giants":
            default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        elif watchlist == "Dow Jones 30":
            default_stocks = ["AAPL", "MSFT", "JPM", "V", "UNH"]
        elif watchlist == "S&P 500 Top 10":
            default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        else:
            default_stocks = ["AAPL"]

        st.session_state['default_stocks'] = default_stocks

    # Route to different pages
    if "🏠 Home" in page:
        show_home_page()
    elif "📊 Stock Overview" in page:
        show_stock_overview_page()
    elif "📉 Technical Analysis" in page:
        show_technical_analysis_page()
    elif "🤖 Price Prediction" in page:
        show_prediction_page()
    elif "⚡ Backtesting" in page:
        show_backtesting_page()
    elif "💼 Portfolio Analysis" in page:
        show_portfolio_page()


def show_home_page():
    """Home page with introduction and quick start"""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## Welcome to Your Personal Stock Analysis Tool! 👋")
        st.markdown("""
        This dashboard helps you analyze stocks like a professional trader,
        without needing to write any code!

        ### What You Can Do:

        📊 **Stock Overview** - View real-time stock prices, charts, and basic information

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


def show_stock_overview_page():
    """Stock overview page with price charts and basic info"""

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

        st.plotly_chart(fig, use_container_width=True)

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

        st.plotly_chart(fig_vol, use_container_width=True)

        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.tail(100), use_container_width=True)

    else:
        st.info("👆 Enter a stock ticker and click 'Load Stock Data' to get started!")


def show_technical_analysis_page():
    """Technical analysis page with indicators"""

    st.title("📉 Technical Analysis")

    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("⚠️ Please load stock data first from the 'Stock Overview' page!")
        return

    df = st.session_state['current_data'].copy()
    ticker = st.session_state.get('current_stock', 'Stock')

    st.markdown(f"### Analyzing: **{ticker}**")

    # Indicator selection
    st.markdown("#### Select Indicators to Display")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_sma = st.checkbox("Moving Averages (SMA)", value=True)
        show_rsi = st.checkbox("RSI", value=True)

    with col2:
        show_macd = st.checkbox("MACD", value=False)
        show_bb = st.checkbox("Bollinger Bands", value=False)

    with col3:
        show_stoch = st.checkbox("Stochastic", value=False)
        show_atr = st.checkbox("ATR", value=False)

    with col4:
        show_obv = st.checkbox("OBV", value=False)
        show_vwap = st.checkbox("VWAP", value=False)

    # Indicator classes (all methods are static)
    # No need to instantiate, but keeping for clarity

    # Price chart with indicators
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

    # Add selected indicators
    if show_sma:
        sma_20 = TrendIndicators.sma(df, period=20)
        sma_50 = TrendIndicators.sma(df, period=50)
        fig.add_trace(go.Scatter(x=df.index, y=sma_20, name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=sma_50, name='SMA 50', line=dict(color='blue')))

    if show_bb:
        bb_df = VolatilityIndicators.bollinger_bands(df)
        # Extract columns (they're named BBL_20_2.0, BBM_20_2.0, BBU_20_2.0)
        bb_lower = bb_df.iloc[:, 0]  # First column is lower
        bb_middle = bb_df.iloc[:, 1]  # Second column is middle
        bb_upper = bb_df.iloc[:, 2]  # Third column is upper
        fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name='BB Upper', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name='BB Lower', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=bb_middle, name='BB Middle', line=dict(color='purple')))

    if show_vwap:
        vwap = VolumeIndicators.vwap(df)
        fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='green', dash='dot')))

    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Additional indicator charts
    if show_rsi:
        st.markdown("#### RSI (Relative Strength Index)")
        rsi = MomentumIndicators.rsi(df)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")

        fig_rsi.update_layout(
            yaxis_title='RSI',
            xaxis_title='Date',
            template='plotly_white',
            height=250,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig_rsi, use_container_width=True)

        # RSI interpretation
        current_rsi = rsi.iloc[-1]
        if current_rsi > 70:
            st.warning(f"📊 RSI is {current_rsi:.1f} - Stock may be **overbought** (potential sell signal)")
        elif current_rsi < 30:
            st.success(f"📊 RSI is {current_rsi:.1f} - Stock may be **oversold** (potential buy signal)")
        else:
            st.info(f"📊 RSI is {current_rsi:.1f} - Stock is in **neutral** territory")

    if show_macd:
        st.markdown("#### MACD (Moving Average Convergence Divergence)")
        macd_df = TrendIndicators.macd(df)
        # Extract columns (MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9)
        macd = macd_df.iloc[:, 0]  # MACD line
        signal = macd_df.iloc[:, 1]  # Signal line
        histogram = macd_df.iloc[:, 2]  # Histogram

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram', marker_color='lightgray'))

        fig_macd.update_layout(
            yaxis_title='MACD',
            xaxis_title='Date',
            template='plotly_white',
            height=250
        )

        st.plotly_chart(fig_macd, use_container_width=True)

    if show_stoch:
        st.markdown("#### Stochastic Oscillator")
        stoch_df = MomentumIndicators.stochastic(df)
        # Extract columns (STOCHk_14_3, STOCHd_14_3)
        stoch_k = stoch_df.iloc[:, 0]  # %K
        stoch_d = stoch_df.iloc[:, 1]  # %D

        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=df.index, y=stoch_k, name='%K', line=dict(color='blue')))
        fig_stoch.add_trace(go.Scatter(x=df.index, y=stoch_d, name='%D', line=dict(color='red')))
        fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")

        fig_stoch.update_layout(
            yaxis_title='Stochastic',
            xaxis_title='Date',
            template='plotly_white',
            height=250,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig_stoch, use_container_width=True)

    if show_atr:
        st.markdown("#### ATR (Average True Range) - Volatility")
        atr = VolatilityIndicators.atr(df)

        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df.index, y=atr, name='ATR', line=dict(color='orange')))

        fig_atr.update_layout(
            yaxis_title='ATR',
            xaxis_title='Date',
            template='plotly_white',
            height=250
        )

        st.plotly_chart(fig_atr, use_container_width=True)

    if show_obv:
        st.markdown("#### OBV (On-Balance Volume)")
        obv = VolumeIndicators.obv(df)

        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(x=df.index, y=obv, name='OBV', line=dict(color='teal')))

        fig_obv.update_layout(
            yaxis_title='OBV',
            xaxis_title='Date',
            template='plotly_white',
            height=250
        )

        st.plotly_chart(fig_obv, use_container_width=True)


def show_prediction_page():
    """Price prediction page using ML"""

    st.title("🤖 AI Price Prediction")

    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("⚠️ Please load stock data first from the 'Stock Overview' page!")
        return

    df = st.session_state['current_data'].copy()
    ticker = st.session_state.get('current_stock', 'Stock')

    st.markdown(f"### Predicting: **{ticker}**")

    st.info("""
    📊 **How it works**: This AI model uses historical price patterns, technical indicators,
    and statistical features to predict future prices. Remember, predictions are estimates and
    should not be used as the sole basis for investment decisions!
    """)

    col1, col2 = st.columns(2)

    with col1:
        prediction_days = st.slider("Predict how many days ahead?", 1, 30, 5)

    with col2:
        model_type = st.selectbox("AI Model", ["Random Forest (Recommended)", "Linear Regression"])

    if st.button("🚀 Generate Prediction", type="primary"):
        with st.spinner("Training AI model... This may take a minute..."):
            try:
                # Prepare features and dataset (all-in-one)
                engineer = FeatureEngineer()
                X, y, scaler = engineer.prepare_ml_dataset(
                    df,
                    target_column='Close',
                    forecast_horizon=1,
                    target_type='price'
                )

                # Train model
                model = RandomForestPredictor(n_estimators=100, random_state=42)
                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                model.fit(X_train, y_train)

                # Make predictions
                predictions = model.predict(X_test)

                # Calculate metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                import numpy as np

                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)

                st.success("✅ Model trained successfully!")

                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                with col2:
                    st.metric("Root Mean Squared Error", f"${rmse:.2f}")
                with col3:
                    st.metric("R² Score", f"{r2:.3f}")

                # Plot predictions vs actual
                st.markdown("### Prediction vs Actual Prices")

                # Calculate proper test dates (account for NaN rows dropped during feature creation)
                test_dates = df.index[-(len(X) - train_size):][:len(y_test)]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=y_test, name='Actual Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_dates, y=predictions, name='Predicted Price', line=dict(color='red', dash='dash')))

                fig.update_layout(
                    title=f'{ticker} - AI Predictions vs Actual',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    template='plotly_white',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Future prediction
                st.markdown(f"### {prediction_days}-Day Future Forecast")

                # Use last known values for future prediction
                last_features = X_test.iloc[-1:].values
                future_predictions = []

                for i in range(prediction_days):
                    pred = model.predict(last_features)[0]
                    future_predictions.append(pred)

                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)

                # Plot future forecast
                fig_future = go.Figure()

                # Historical prices (last 30 days)
                fig_future.add_trace(go.Scatter(
                    x=df.index[-30:],
                    y=df['Close'].tail(30),
                    name='Historical Price',
                    line=dict(color='blue')
                ))

                # Future predictions
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    name='Forecast',
                    line=dict(color='red', dash='dash'),
                    mode='lines+markers'
                ))

                fig_future.update_layout(
                    title=f'{ticker} - {prediction_days}-Day Price Forecast',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    template='plotly_white',
                    height=400
                )

                st.plotly_chart(fig_future, use_container_width=True)

                # Show prediction values
                st.markdown("#### Predicted Prices")
                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })
                prediction_df['Change from Today'] = prediction_df['Predicted Price'] - df['Close'].iloc[-1]
                prediction_df['% Change'] = (prediction_df['Change from Today'] / df['Close'].iloc[-1]) * 100

                st.dataframe(prediction_df.style.format({
                    'Predicted Price': '${:.2f}',
                    'Change from Today': '${:.2f}',
                    '% Change': '{:.2f}%'
                }), use_container_width=True)

                # Feature importance
                with st.expander("📊 Feature Importance - What the AI Looks At"):
                    importance = model.get_feature_importance()
                    if importance is not None:
                        st.bar_chart(importance.head(10))

                st.warning("""
                ⚠️ **Disclaimer**: These predictions are for educational purposes only.
                Stock markets are inherently unpredictable. Always do your own research and
                consult with a financial advisor before making investment decisions.
                """)

            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("💡 Tip: Make sure you have at least 60 days of historical data loaded.")


def show_backtesting_page():
    """Backtesting page for trading strategies"""

    st.title("⚡ Strategy Backtesting")

    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("⚠️ Please load stock data first from the 'Stock Overview' page!")
        return

    df = st.session_state['current_data'].copy()
    ticker = st.session_state.get('current_stock', 'Stock')

    st.markdown(f"### Backtesting: **{ticker}**")

    st.info("""
    🎯 **Backtesting** lets you test trading strategies on historical data to see how they would have performed.
    This helps you evaluate strategies before risking real money!
    """)

    # Strategy selection
    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Select Trading Strategy",
            ["SMA Crossover", "RSI Mean Reversion", "Both (Compare)"]
        )

    with col2:
        initial_cash = st.number_input("Initial Investment ($)", value=10000, min_value=1000, step=1000)

    # Strategy parameters
    if strategy in ["SMA Crossover", "Both (Compare)"]:
        st.markdown("#### SMA Crossover Parameters")
        col1, col2 = st.columns(2)
        with col1:
            sma_short = st.slider("Short-term SMA", 5, 50, 20)
        with col2:
            sma_long = st.slider("Long-term SMA", 20, 200, 50)

    if strategy in ["RSI Mean Reversion", "Both (Compare)"]:
        st.markdown("#### RSI Mean Reversion Parameters")
        col1, col2 = st.columns(2)
        with col1:
            rsi_oversold = st.slider("RSI Oversold Level", 20, 40, 30)
        with col2:
            rsi_overbought = st.slider("RSI Overbought Level", 60, 80, 70)

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                from backtesting import Backtest
                from src.backtesting.metrics import PerformanceMetrics

                # Prepare data for backtesting library
                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                results_list = []

                # Run SMA Crossover
                if strategy in ["SMA Crossover", "Both (Compare)"]:
                    # Create dynamic strategy class
                    class SMACrossoverCustom(SMACrossover):
                        n1 = sma_short
                        n2 = sma_long

                    bt_sma = Backtest(data, SMACrossoverCustom, cash=initial_cash, commission=.002)
                    stats_sma = bt_sma.run()
                    results_list.append(("SMA Crossover", stats_sma))

                # Run RSI Mean Reversion
                if strategy in ["RSI Mean Reversion", "Both (Compare)"]:
                    # Create dynamic strategy class
                    class RSIMeanReversionCustom(RSIMeanReversion):
                        rsi_lower = rsi_oversold
                        rsi_upper = rsi_overbought

                    bt_rsi = Backtest(data, RSIMeanReversionCustom, cash=initial_cash, commission=.002)
                    stats_rsi = bt_rsi.run()
                    results_list.append(("RSI Mean Reversion", stats_rsi))

                # Display results
                st.success("✅ Backtest completed!")

                for strategy_name, stats in results_list:
                    st.markdown(f"### {strategy_name} Results")

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        final_value = stats['Equity Final [$]']
                        st.metric("Final Portfolio Value", f"${final_value:,.0f}")

                    with col2:
                        total_return = stats['Return [%]']
                        st.metric("Total Return", f"{total_return:.2f}%")

                    with col3:
                        sharpe = stats.get('Sharpe Ratio', 0)
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                    with col4:
                        max_dd = stats['Max. Drawdown [%]']
                        st.metric("Max Drawdown", f"{max_dd:.2f}%")

                    # Additional metrics
                    with st.expander(f"📊 Detailed Metrics - {strategy_name}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Number of Trades**: {stats['# Trades']}")
                            st.write(f"**Win Rate**: {stats['Win Rate [%]']:.2f}%")
                            st.write(f"**Best Trade**: {stats['Best Trade [%]']:.2f}%")
                            st.write(f"**Worst Trade**: {stats['Worst Trade [%]']:.2f}%")

                        with col2:
                            st.write(f"**Avg Trade**: {stats['Avg. Trade [%]']:.2f}%")
                            st.write(f"**Max Trade Duration**: {stats['Max. Trade Duration']}")
                            st.write(f"**Avg Trade Duration**: {stats['Avg. Trade Duration']}")
                            st.write(f"**Exposure Time**: {stats['Exposure Time [%]']:.2f}%")

                    # Equity curve
                    st.markdown(f"#### Equity Curve - {strategy_name}")
                    equity_curve = stats['_equity_curve']

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve['Equity'],
                        name='Portfolio Value',
                        fill='tozeroy',
                        line=dict(color='green')
                    ))

                    fig.update_layout(
                        title=f'{strategy_name} - Portfolio Value Over Time',
                        yaxis_title='Portfolio Value ($)',
                        xaxis_title='Date',
                        template='plotly_white',
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")

                # Compare strategies if both were run
                if len(results_list) == 2:
                    st.markdown("### Strategy Comparison")

                    comparison_df = pd.DataFrame({
                        'Metric': ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', '# Trades'],
                        results_list[0][0]: [
                            results_list[0][1]['Return [%]'],
                            results_list[0][1].get('Sharpe Ratio', 0),
                            results_list[0][1]['Max. Drawdown [%]'],
                            results_list[0][1]['Win Rate [%]'],
                            results_list[0][1]['# Trades']
                        ],
                        results_list[1][0]: [
                            results_list[1][1]['Return [%]'],
                            results_list[1][1].get('Sharpe Ratio', 0),
                            results_list[1][1]['Max. Drawdown [%]'],
                            results_list[1][1]['Win Rate [%]'],
                            results_list[1][1]['# Trades']
                        ]
                    })

                    st.dataframe(comparison_df, use_container_width=True)

                st.info("""
                💡 **How to interpret**:
                - **Total Return**: Higher is better
                - **Sharpe Ratio**: Risk-adjusted return. Above 1 is good, above 2 is excellent
                - **Max Drawdown**: Largest peak-to-valley decline. Lower is better
                - **Win Rate**: Percentage of profitable trades
                """)

            except Exception as e:
                st.error(f"❌ Error running backtest: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def show_portfolio_page():
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


if __name__ == "__main__":
    main()
