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
import json
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
from src.analysis.investment_recommendation import get_investment_recommendation

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme CSS Generator
def get_theme_css(dark_mode=False):
    """Generate CSS based on current theme"""
    if dark_mode:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #4dabf7;
                text-align: center;
                padding: 1rem 0;
            }
            .metric-card {
                background-color: #2d3748;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                color: #e2e8f0;
            }
            .stAlert {
                margin-top: 1rem;
            }
            /* Dark mode overrides */
            .stApp {
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .stMarkdown {
                color: #e2e8f0;
            }
            .stDataFrame {
                background-color: #2d3748;
            }
        </style>
        """
    else:
        return """
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
        """


# ============================================================================
# PERSISTENT DATA MANAGER
# ============================================================================

USER_DATA_DIR = Path("data/user_data")
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json_data(filename: str, default: dict) -> dict:
    """Load JSON data from user_data directory"""
    file_path = USER_DATA_DIR / filename
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading {filename}: {e}")
            return default
    return default


def save_json_data(filename: str, data: dict) -> bool:
    """Save JSON data to user_data directory"""
    file_path = USER_DATA_DIR / filename
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving {filename}: {e}")
        return False


def initialize_user_data():
    """Initialize all user data in session state on app start"""
    if 'user_watchlists' not in st.session_state:
        st.session_state['user_watchlists'] = load_json_data(
            'watchlists.json',
            {'watchlists': {}, 'last_updated': None}
        )

    if 'user_alerts' not in st.session_state:
        st.session_state['user_alerts'] = load_json_data(
            'alerts.json',
            {'alerts': [], 'last_checked': None}
        )

    if 'performance_data' not in st.session_state:
        st.session_state['performance_data'] = load_json_data(
            'performance_tracker.json',
            {'trades': [], 'statistics': {}}
        )

    if 'dark_mode' not in st.session_state:
        preferences = load_json_data('preferences.json', {'dark_mode': False})
        st.session_state['dark_mode'] = preferences.get('dark_mode', False)


def check_price_alerts():
    """Check active alerts and show triggered ones in sidebar"""
    alerts_list = st.session_state.get('user_alerts', {}).get('alerts', [])
    if not alerts_list:
        return

    active_alerts = [a for a in alerts_list if a.get('active', False) and not a.get('triggered_at')]

    if not active_alerts:
        return

    triggered_count = 0

    for alert in active_alerts:
        try:
            # Fetch recent data (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df = get_stock_data(
                alert['ticker'],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )

            if df is None or len(df) == 0:
                continue

            triggered = False

            if alert['condition'] == 'price_above':
                if df['Close'].iloc[-1] > alert['threshold']:
                    triggered = True

            elif alert['condition'] == 'price_below':
                if df['Close'].iloc[-1] < alert['threshold']:
                    triggered = True

            elif alert['condition'] == 'rsi_oversold':
                df_features = FeatureEngineer.prepare_features(df)
                if 'RSI' in df_features.columns:
                    rsi = df_features['RSI'].iloc[-1]
                    if rsi < alert['threshold']:
                        triggered = True

            elif alert['condition'] == 'rsi_overbought':
                df_features = FeatureEngineer.prepare_features(df)
                if 'RSI' in df_features.columns:
                    rsi = df_features['RSI'].iloc[-1]
                    if rsi > alert['threshold']:
                        triggered = True

            elif alert['condition'] == 'macd_bullish_cross':
                macd_df = TrendIndicators.macd(df)
                if macd_df.iloc[-1, 0] > macd_df.iloc[-1, 1]:  # MACD > Signal
                    triggered = True

            if triggered:
                alert['triggered_at'] = datetime.now().isoformat()
                alert['active'] = False  # Deactivate after triggering
                triggered_count += 1

        except Exception as e:
            # Skip alerts that error
            continue

    if triggered_count > 0:
        save_json_data('alerts.json', st.session_state['user_alerts'])
        st.sidebar.success(f"🔔 {triggered_count} Alert(s) Triggered! Check Price Alerts page.")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""

    # Initialize user data (watchlists, alerts, performance tracker)
    initialize_user_data()

    # Apply theme CSS
    dark_mode = st.session_state.get('dark_mode', False)
    st.markdown(get_theme_css(dark_mode), unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">📈 Stock Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar - Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/line-chart.png", width=100)
        st.title("Navigation")

        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "📊 Stock Overview", "💡 Investment Advice", "📉 Technical Analysis",
             "🤖 Price Prediction", "⚡ Backtesting", "💼 Portfolio Analysis",
             "📋 Watchlist Manager", "🔄 Stock Comparison", "🔍 Stock Screener", "🔔 Price Alerts", "💹 Performance Tracker"],
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

        # Dark Mode Toggle
        dark_mode_toggle = st.checkbox(
            "🌙 Dark Mode",
            value=st.session_state.get('dark_mode', False),
            key="dark_mode_checkbox"
        )

        # Save dark mode preference if changed
        if dark_mode_toggle != st.session_state.get('dark_mode', False):
            st.session_state['dark_mode'] = dark_mode_toggle
            save_json_data('preferences.json', {'dark_mode': dark_mode_toggle})
            st.rerun()

        # Check for triggered price alerts
        st.markdown("---")
        check_price_alerts()

    # Route to different pages
    if "🏠 Home" in page:
        show_home_page()
    elif "📊 Stock Overview" in page:
        show_stock_overview_page()
    elif "💡 Investment Advice" in page:
        show_investment_advice_page()
    elif "📉 Technical Analysis" in page:
        show_technical_analysis_page()
    elif "🤖 Price Prediction" in page:
        show_prediction_page()
    elif "⚡ Backtesting" in page:
        show_backtesting_page()
    elif "💼 Portfolio Analysis" in page:
        show_portfolio_page()
    elif "📋 Watchlist Manager" in page:
        show_watchlist_manager_page()
    elif "🔄 Stock Comparison" in page:
        show_stock_comparison_page()
    elif "🔍 Stock Screener" in page:
        show_stock_screener_page()
    elif "🔔 Price Alerts" in page:
        show_price_alerts_page()
    elif "💹 Performance Tracker" in page:
        show_performance_tracker_page()


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


def show_investment_advice_page():
    """Investment recommendation page with AI-powered analysis"""

    st.title("💡 Investment Advice - Should You Buy?")

    st.markdown("""
    Get an intelligent recommendation based on:
    - 📊 Technical indicators (trend, momentum, volatility)
    - 🤖 AI price predictions
    - ⚠️ Risk assessment
    - 🎯 Entry/exit price targets
    """)

    # Input controls
    col1, col2 = st.columns([2, 2])

    with col1:
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value=st.session_state.get('current_stock', 'AAPL'),
            help="Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()

    with col2:
        use_ml = st.checkbox(
            "Include AI Prediction",
            value=True,
            help="Use machine learning to improve recommendation accuracy"
        )

    if st.button("🔍 Analyze Stock", type="primary"):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Fetch stock data (1 year for ML training)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)

                df = get_stock_data(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                if df is None or len(df) < 60:
                    st.error(f"❌ Insufficient data for {ticker}. Need at least 60 days of history.")
                    return

                # Prepare features
                engineer = FeatureEngineer()
                df_features = engineer.prepare_features(df)

                # ML prediction (optional)
                ml_prediction = None
                ml_confidence = None

                if use_ml and len(df_features) >= 100:
                    try:
                        with st.spinner("Training AI model..."):
                            # Prepare ML dataset
                            X, y, scaler = engineer.prepare_ml_dataset(
                                df,
                                target_column='Close',
                                forecast_horizon=1,
                                target_type='price'
                            )

                            if len(X) > 50:
                                # Train model
                                model = RandomForestPredictor()
                                model.train(X, y)

                                # Get metrics
                                metrics = model.get_metrics(X, y)
                                ml_confidence = metrics.get('r2', 0)

                                # Predict next day
                                ml_prediction = model.predict(X.tail(1))[0]

                                st.success(f"✅ AI model trained (R² = {ml_confidence:.2%})")
                    except Exception as e:
                        st.warning(f"⚠️ Could not run AI prediction: {str(e)}")

                # Get investment recommendation
                recommendation = get_investment_recommendation(
                    df_features,
                    ml_prediction=ml_prediction,
                    ml_confidence=ml_confidence
                )

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                # Main recommendation card
                rec_color_map = {
                    'STRONG BUY': '#00ff00',
                    'BUY': '#90ee90',
                    'HOLD': '#ffff00',
                    'SELL': '#ffcccb',
                    'STRONG SELL': '#ff0000'
                }

                rec_color = rec_color_map.get(recommendation['recommendation'], '#cccccc')

                st.markdown(f"""
                <div style="background-color: {rec_color}; padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0;">
                    <h1 style="margin: 0; color: #000;">{recommendation['color']} {recommendation['recommendation']}</h1>
                    <h2 style="margin: 0.5rem 0; color: #000;">Score: {recommendation['overall_score']}/100</h2>
                    <p style="font-size: 1.2rem; margin: 0; color: #000;">{recommendation['action']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Details in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🎯 Price Targets")
                    targets = recommendation['price_targets']

                    st.metric("Current Price", f"${targets['current_price']}")
                    st.metric("Recommended Entry", f"${targets['entry_price']}",
                             delta=f"{((targets['entry_price']/targets['current_price']-1)*100):.1f}%")
                    st.metric("Stop Loss", f"${targets['stop_loss']}",
                             delta=f"{((targets['stop_loss']/targets['current_price']-1)*100):.1f}%")
                    st.metric("Price Target", f"${targets['target_price']}",
                             delta=f"{((targets['target_price']/targets['current_price']-1)*100):.1f}%")

                    st.markdown("---")
                    st.markdown(f"**Support Level:** ${targets['support']}")
                    st.markdown(f"**Resistance Level:** ${targets['resistance']}")

                with col2:
                    st.markdown("### ⚠️ Risk Assessment")
                    risk = recommendation['risk_metrics']

                    risk_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    st.markdown(f"**Risk Level:** :{risk_color[risk['risk_level']]}[{risk['risk_level']}]")
                    st.markdown(f"**Volatility (Annual):** {risk['volatility']:.1f}%")
                    st.markdown(f"**Max Drawdown:** {risk['max_drawdown']:.1f}%")

                    st.markdown("---")

                    st.markdown("### 📈 Technical Score")
                    st.progress(recommendation['technical_score'] / 100)
                    st.markdown(f"**{recommendation['technical_score']}/100**")

                # Detailed signals
                st.markdown("### 📋 Detailed Analysis")

                for detail in recommendation['details']:
                    if '✅' in detail:
                        st.success(detail)
                    elif '❌' in detail:
                        st.error(detail)
                    elif '⚠️' in detail:
                        st.warning(detail)
                    else:
                        st.info(detail)

                # Disclaimer
                st.markdown("---")
                st.warning(recommendation['disclaimer'])

                st.caption(f"Analysis generated at: {recommendation['timestamp']}")

            except Exception as e:
                st.error(f"❌ Error analyzing {ticker}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    else:
        # Show example when no analysis run yet
        st.info("👆 Enter a stock ticker and click 'Analyze Stock' to get investment advice!")

        st.markdown("### How It Works:")
        st.markdown("""
        1. **Technical Analysis**: Examines trend (SMA), momentum (RSI, MACD), volatility (Bollinger Bands), and volume
        2. **AI Prediction**: Machine learning model predicts next-day price movement
        3. **Risk Assessment**: Calculates volatility and maximum drawdown
        4. **Smart Scoring**: Combines all factors into a 0-100 score
        5. **Actionable Advice**: Provides clear BUY/SELL/HOLD recommendation with price targets

        **Scoring Breakdown:**
        - 70-100: STRONG BUY - Multiple positive signals
        - 55-69: BUY - Good opportunity
        - 45-54: HOLD - Wait for clarity
        - 30-44: SELL - Negative signals present
        - 0-29: STRONG SELL - Avoid or exit
        """)


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


def show_price_alerts_page():
    """Manage price alerts"""
    st.title("🔔 Price Alerts")

    st.markdown("""
    Set alerts to get notified when stocks hit your target prices or technical conditions.
    Alerts are checked when you visit the dashboard.
    """)

    # Create alert form
    st.markdown("### ➕ Create New Alert")
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        alert_ticker = st.text_input("Ticker", key="alert_ticker", placeholder="e.g., AAPL").upper()
    with col2:
        alert_type = st.selectbox("Condition", [
            "Price Above",
            "Price Below",
            "RSI Oversold",
            "RSI Overbought",
            "MACD Bullish Cross"
        ], key="alert_type")
    with col3:
        if "Price" in alert_type:
            alert_value = st.number_input("Price Target", min_value=0.0, value=100.0, key="alert_value")
        else:
            alert_value = st.number_input("Threshold", min_value=0, max_value=100, value=30, key="alert_threshold")
    with col4:
        st.write("")  # Spacing
        st.write("")  # More spacing
        if st.button("Create Alert", type="primary"):
            if not alert_ticker:
                st.error("Please enter a ticker symbol")
            else:
                alerts_list = st.session_state['user_alerts']['alerts']
                alerts_list.append({
                    'id': f"alert_{len(alerts_list) + 1}_{int(datetime.now().timestamp())}",
                    'ticker': alert_ticker,
                    'condition': alert_type.lower().replace(' ', '_'),
                    'threshold': alert_value,
                    'active': True,
                    'created_at': datetime.now().isoformat(),
                    'triggered_at': None
                })
                save_json_data('alerts.json', st.session_state['user_alerts'])
                st.success(f"✅ Alert created for {alert_ticker}")
                st.rerun()

    st.markdown("---")

    # Display active alerts
    st.markdown("### 📋 Active Alerts")
    alerts_list = st.session_state['user_alerts']['alerts']
    active_alerts = [a for a in alerts_list if a['active']]

    if active_alerts:
        for alert in active_alerts:
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col1:
                st.write(f"**{alert['ticker']}**")
            with col2:
                condition_display = alert['condition'].replace('_', ' ').title()
                st.write(f"{condition_display}")
            with col3:
                st.write(f"Threshold: {alert['threshold']}")
            with col4:
                if st.button("🗑️", key=f"del_{alert['id']}"):
                    alerts_list.remove(alert)
                    save_json_data('alerts.json', st.session_state['user_alerts'])
                    st.success("Alert deleted")
                    st.rerun()

    else:
        st.info("📭 No active alerts. Create one above!")

    # Show triggered alerts (in last 24 hours)
    triggered_alerts = [a for a in alerts_list if a.get('triggered_at')]
    if triggered_alerts:
        # Filter to last 24 hours
        recent_triggered = []
        for alert in triggered_alerts:
            try:
                triggered_time = datetime.fromisoformat(alert['triggered_at'])
                if (datetime.now() - triggered_time).total_seconds() < 86400:  # 24 hours
                    recent_triggered.append(alert)
            except:
                pass

        if recent_triggered:
            st.markdown("---")
            st.markdown("### 🔔 Recently Triggered (Last 24h)")
            for alert in recent_triggered:
                triggered_time = datetime.fromisoformat(alert['triggered_at'])
                time_ago = datetime.now() - triggered_time
                hours_ago = int(time_ago.total_seconds() / 3600)

                condition_display = alert['condition'].replace('_', ' ').title()
                st.success(f"✅ **{alert['ticker']}**: {condition_display} (threshold: {alert['threshold']}) - {hours_ago}h ago")

    st.markdown("---")
    st.markdown("### 💡 How Alerts Work")
    st.markdown("""
    - Alerts are checked automatically when you visit the dashboard
    - Active alerts show in the sidebar when triggered
    - Triggered alerts are marked and shown for 24 hours
    - Set multiple alerts for different conditions on the same stock
    """)


def show_performance_tracker_page():
    """Track paper trading performance based on recommendations"""
    st.title("💹 Performance Tracker")

    st.markdown("""
    Track your paper trading performance based on Investment Advice recommendations.
    Log entry and exit points, and analyze your trading results.
    """)

    performance_data = st.session_state['performance_data']
    trades = performance_data.get('trades', [])

    # Add new trade section
    st.markdown("### ➕ Log New Trade")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        trade_ticker = st.text_input("Ticker", key="trade_ticker").upper()
    with col2:
        trade_type = st.selectbox("Type", ["BUY", "SELL"], key="trade_type")
    with col3:
        trade_price = st.number_input("Price", min_value=0.0, value=100.0, step=0.01, key="trade_price")
    with col4:
        trade_shares = st.number_input("Shares", min_value=1, value=100, step=1, key="trade_shares")
    with col5:
        st.write("")  # Spacing
        st.write("")  # More spacing
        if st.button("Log Trade", type="primary"):
            if not trade_ticker:
                st.error("Please enter a ticker symbol")
            else:
                new_trade = {
                    'id': f"trade_{len(trades) + 1}_{int(datetime.now().timestamp())}",
                    'ticker': trade_ticker,
                    'type': trade_type,
                    'price': trade_price,
                    'shares': trade_shares,
                    'total_value': trade_price * trade_shares,
                    'date': datetime.now().isoformat(),
                    'status': 'OPEN' if trade_type == 'BUY' else 'CLOSED'
                }

                # If SELL, try to match with open BUY
                if trade_type == 'SELL':
                    open_buys = [t for t in trades if t['ticker'] == trade_ticker and t['type'] == 'BUY' and t.get('status') == 'OPEN']
                    if open_buys:
                        # Match with first open buy
                        buy_trade = open_buys[0]
                        buy_trade['status'] = 'CLOSED'
                        buy_trade['exit_price'] = trade_price
                        buy_trade['exit_date'] = datetime.now().isoformat()

                        # Calculate P&L
                        profit_loss = (trade_price - buy_trade['price']) * min(trade_shares, buy_trade['shares'])
                        profit_loss_pct = ((trade_price - buy_trade['price']) / buy_trade['price']) * 100

                        buy_trade['profit_loss'] = profit_loss
                        buy_trade['profit_loss_pct'] = profit_loss_pct
                        new_trade['matched_buy'] = buy_trade['id']
                        new_trade['profit_loss'] = profit_loss
                        new_trade['profit_loss_pct'] = profit_loss_pct

                trades.append(new_trade)
                performance_data['trades'] = trades
                save_json_data('performance_tracker.json', performance_data)
                st.success(f"✅ {trade_type} trade logged for {trade_ticker}")
                st.rerun()

    st.markdown("---")

    if not trades:
        st.info("📊 No trades logged yet. Start tracking your performance by logging trades above!")
        return

    # Performance Statistics
    st.markdown("### 📈 Performance Statistics")

    # Calculate statistics
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED' and t.get('profit_loss') is not None]

    if closed_trades:
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['profit_loss'] > 0]
        losing_trades = [t for t in closed_trades if t['profit_loss'] < 0]

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(t['profit_loss'] for t in winning_trades)
        total_loss = sum(t['profit_loss'] for t in losing_trades)
        net_profit = total_profit + total_loss

        avg_win = (total_profit / len(winning_trades)) if winning_trades else 0
        avg_loss = (total_loss / len(losing_trades)) if losing_trades else 0

        # Display stats in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with col2:
            st.metric("Winning Trades", len(winning_trades))
            st.metric("Losing Trades", len(losing_trades))

        with col3:
            profit_color = "normal" if net_profit >= 0 else "inverse"
            st.metric("Net P&L", f"${net_profit:,.2f}", delta=f"{net_profit:+,.2f}")
            st.metric("Total Profit", f"${total_profit:,.2f}")

        with col4:
            st.metric("Total Loss", f"${total_loss:,.2f}")
            st.metric("Avg Win/Loss", f"${avg_win:.2f} / ${avg_loss:.2f}")

        # P&L Chart
        st.markdown("### 📊 Cumulative P&L")

        # Sort by date
        sorted_trades = sorted(closed_trades, key=lambda x: x.get('exit_date', x['date']))

        cumulative_pl = 0
        cumulative_data = []
        dates = []

        for trade in sorted_trades:
            cumulative_pl += trade['profit_loss']
            cumulative_data.append(cumulative_pl)
            try:
                trade_date = datetime.fromisoformat(trade.get('exit_date', trade['date']))
                dates.append(trade_date.strftime('%Y-%m-%d'))
            except:
                dates.append('N/A')

        fig_pl = go.Figure()
        fig_pl.add_trace(go.Scatter(
            x=dates,
            y=cumulative_data,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='green' if cumulative_pl >= 0 else 'red', width=3),
            fill='tozeroy'
        ))

        fig_pl.update_layout(
            xaxis_title='Date',
            yaxis_title='Cumulative P&L ($)',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )

        st.plotly_chart(fig_pl, use_container_width=True)

        # Win/Loss Distribution
        st.markdown("### 📊 Win/Loss Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # Win/Loss pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[len(winning_trades), len(losing_trades)],
                marker=dict(colors=['#00cc66', '#ff4444']),
                hole=0.4
            )])
            fig_pie.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # P&L by ticker
            ticker_pl = {}
            for trade in closed_trades:
                ticker = trade['ticker']
                if ticker not in ticker_pl:
                    ticker_pl[ticker] = 0
                ticker_pl[ticker] += trade['profit_loss']

            fig_bar = go.Figure(data=[go.Bar(
                x=list(ticker_pl.keys()),
                y=list(ticker_pl.values()),
                marker=dict(color=['green' if v >= 0 else 'red' for v in ticker_pl.values()])
            )])
            fig_bar.update_layout(
                xaxis_title='Ticker',
                yaxis_title='P&L ($)',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("💡 Close some trades (log SELL orders) to see performance statistics")

    # Trade History
    st.markdown("---")
    st.markdown("### 📋 Trade History")

    # Display all trades in table
    trade_display = []
    for trade in reversed(trades):  # Most recent first
        try:
            trade_date = datetime.fromisoformat(trade['date']).strftime('%Y-%m-%d %H:%M')
        except:
            trade_date = trade['date']

        display_row = {
            'Date': trade_date,
            'Ticker': trade['ticker'],
            'Type': trade['type'],
            'Price': f"${trade['price']:.2f}",
            'Shares': trade['shares'],
            'Total': f"${trade['total_value']:.2f}",
            'Status': trade.get('status', 'N/A')
        }

        if trade.get('profit_loss') is not None:
            display_row['P&L'] = f"${trade['profit_loss']:+,.2f} ({trade['profit_loss_pct']:+.2f}%)"

        trade_display.append(display_row)

    df_trades = pd.DataFrame(trade_display)
    st.dataframe(df_trades, use_container_width=True, hide_index=True)

    # Export trades
    if trades:
        csv = df_trades.to_csv(index=False)
        st.download_button(
            label="📥 Export Trades to CSV",
            data=csv,
            file_name=f"performance_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

    # Clear all trades button
    st.markdown("---")
    if st.button("🗑️ Clear All Trades", type="secondary"):
        if st.checkbox("⚠️ Confirm deletion of all trades"):
            performance_data['trades'] = []
            save_json_data('performance_tracker.json', performance_data)
            st.success("All trades cleared")
            st.rerun()


def show_stock_screener_page():
    """Screen stocks based on technical criteria"""
    st.title("🔍 Stock Screener")

    st.markdown("""
    Find stocks matching your technical criteria. Screen by RSI, MACD, volume, price levels, and more.
    """)

    # Stock universe selection
    st.markdown("### 📋 Select Stock Universe")
    source = st.radio("Source", ["From Watchlist", "Popular Stocks", "Custom List"], horizontal=True)

    if source == "From Watchlist":
        watchlists = st.session_state['user_watchlists']['watchlists']
        if watchlists:
            selected_wl = st.selectbox("Select Watchlist", list(watchlists.keys()))
            tickers = watchlists[selected_wl]
        else:
            st.warning("⚠️ No watchlists found! Create one in Watchlist Manager first.")
            return
    elif source == "Popular Stocks":
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT",
                   "JNJ", "PG", "MA", "HD", "DIS", "NFLX", "PYPL", "INTC", "CSCO", "PEP"]
    else:
        tickers_input = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA")
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if not tickers:
        st.warning("⚠️ Please enter at least one ticker symbol")
        return

    # Screening criteria
    st.markdown("### 🎯 Screening Criteria")
    st.markdown("*Select one or more criteria. Stocks must match ALL selected criteria.*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Bullish Signals")
        check_rsi_oversold = st.checkbox("RSI Oversold (Buy Signal)")
        if check_rsi_oversold:
            rsi_threshold = st.slider("RSI below", 20, 40, 30, key="rsi_low")

        check_macd_bullish = st.checkbox("MACD Bullish Crossover")

        check_volume_spike = st.checkbox("Volume Spike")
        if check_volume_spike:
            volume_multiplier = st.slider("Volume > X times average", 1.5, 5.0, 2.0, 0.5, key="vol_spike")

        check_price_low = st.checkbox("Near 52-Week Low")
        if check_price_low:
            low_threshold = st.slider("Within % of 52W low", 1, 20, 10, key="price_low")

    with col2:
        st.markdown("#### 📉 Bearish Signals")
        check_rsi_overbought = st.checkbox("RSI Overbought (Sell Signal)")
        if check_rsi_overbought:
            rsi_high_threshold = st.slider("RSI above", 60, 80, 70, key="rsi_high")

        check_price_high = st.checkbox("Near 52-Week High")
        if check_price_high:
            high_threshold = st.slider("Within % of 52W high", 1, 20, 10, key="price_high")

        st.markdown("#### 🎯 General Filters")
        check_tech_score = st.checkbox("Minimum Technical Score")
        if check_tech_score:
            min_score = st.slider("Score above", 0, 100, 60, key="tech_score")

    if st.button("🔍 Run Screener", type="primary"):
        # Count selected criteria
        criteria_count = sum([
            check_rsi_oversold, check_rsi_overbought, check_macd_bullish,
            check_volume_spike, check_price_low, check_price_high, check_tech_score
        ])

        if criteria_count == 0:
            st.warning("⚠️ Please select at least one screening criterion")
            return

        with st.spinner(f"Screening {len(tickers)} stocks against {criteria_count} criteria..."):
            try:
                results = []
                progress_bar = st.progress(0)

                for idx, ticker in enumerate(tickers):
                    # Update progress
                    progress_bar.progress((idx + 1) / len(tickers))

                    # Fetch data (1 year for indicators)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)

                    try:
                        df = get_stock_data(
                            ticker,
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d')
                        )

                        if df is None or len(df) < 60:
                            continue

                        # Prepare features
                        df_features = FeatureEngineer.prepare_features(df)
                        rec = get_investment_recommendation(df_features)

                        # Check criteria
                        matches = []
                        passes = True

                        if check_rsi_oversold:
                            rsi = df_features['RSI'].iloc[-1]
                            if rsi < rsi_threshold:
                                matches.append(f"RSI {rsi:.1f}")
                            else:
                                passes = False

                        if check_rsi_overbought:
                            rsi = df_features['RSI'].iloc[-1]
                            if rsi > rsi_high_threshold:
                                matches.append(f"RSI {rsi:.1f}")
                            else:
                                passes = False

                        if check_macd_bullish:
                            macd_df = TrendIndicators.macd(df)
                            if macd_df.iloc[-1, 0] > macd_df.iloc[-1, 1]:  # MACD > Signal
                                matches.append("MACD Bullish")
                            else:
                                passes = False

                        if check_volume_spike:
                            if 'Volume_ratio' in df_features.columns:
                                vol_ratio = df_features['Volume_ratio'].iloc[-1]
                                if pd.notna(vol_ratio) and vol_ratio > volume_multiplier:
                                    matches.append(f"Vol {vol_ratio:.1f}x")
                                else:
                                    passes = False
                            else:
                                passes = False

                        if check_price_low:
                            current = df['Close'].iloc[-1]
                            low_52w = df['Low'].tail(min(252, len(df))).min()
                            distance = ((current - low_52w) / low_52w) * 100
                            if distance <= low_threshold:
                                matches.append(f"{distance:.1f}% from low")
                            else:
                                passes = False

                        if check_price_high:
                            current = df['Close'].iloc[-1]
                            high_52w = df['High'].tail(min(252, len(df))).max()
                            distance = ((high_52w - current) / high_52w) * 100
                            if distance <= high_threshold:
                                matches.append(f"{distance:.1f}% from high")
                            else:
                                passes = False

                        if check_tech_score:
                            if rec['technical_score'] >= min_score:
                                matches.append(f"Score {rec['technical_score']}")
                            else:
                                passes = False

                        if passes:
                            current_price = df['Close'].iloc[-1]
                            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                            change_pct = ((current_price - prev_price) / prev_price) * 100

                            results.append({
                                'Ticker': ticker,
                                'Price': f"${current_price:.2f}",
                                'Change %': f"{change_pct:+.2f}%",
                                'RSI': f"{df_features['RSI'].iloc[-1]:.1f}" if 'RSI' in df_features.columns else 'N/A',
                                'Tech Score': rec['technical_score'],
                                'Recommendation': rec['recommendation'],
                                'Signals': ', '.join(matches)
                            })

                    except Exception as e:
                        # Skip stocks with errors
                        continue

                progress_bar.empty()

                # Display results
                st.markdown("---")
                st.markdown(f"### 🎯 Screening Results: {len(results)} matches out of {len(tickers)} stocks")

                if results:
                    df_results = pd.DataFrame(results)

                    # Sort by technical score (descending)
                    df_results_sorted = df_results.sort_values('Tech Score', ascending=False)

                    st.dataframe(df_results_sorted, use_container_width=True, hide_index=True)

                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Matches Found", len(results))
                    with col2:
                        avg_score = df_results['Tech Score'].mean()
                        st.metric("Avg Tech Score", f"{avg_score:.1f}")
                    with col3:
                        buy_recs = len([r for r in results if 'BUY' in r['Recommendation']])
                        st.metric("Buy Recommendations", buy_recs)

                    # Export results
                    st.markdown("---")
                    csv = df_results_sorted.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Screening Results",
                        data=csv,
                        file_name=f"stock_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime='text/csv'
                    )

                else:
                    st.info("📭 No stocks matched all the selected criteria. Try adjusting your filters.")

            except Exception as e:
                st.error(f"❌ Screening error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    else:
        st.info("👆 Select criteria and click 'Run Screener' to find matching stocks!")

        st.markdown("### 💡 Screening Examples:")
        st.markdown("""
        **Find Oversold Stocks:**
        - ✓ RSI Oversold (< 30)
        - ✓ Minimum Technical Score (> 50)

        **Find Breakout Candidates:**
        - ✓ Near 52-Week High
        - ✓ Volume Spike (> 2x)
        - ✓ MACD Bullish Crossover

        **Find Value Opportunities:**
        - ✓ Near 52-Week Low
        - ✓ Minimum Technical Score (> 60)
        """)


def show_stock_comparison_page():
    """Compare multiple stocks side-by-side"""
    st.title("📊 Stock Comparison")

    st.markdown("""
    Compare multiple stocks to identify the best investment opportunities.
    See technical indicators, performance metrics, and correlation analysis side-by-side.
    """)

    # Input: Select stocks
    col1, col2 = st.columns([3, 1])

    with col1:
        source = st.radio("Stock Source", ["Manual Entry", "From Watchlist"], horizontal=True)

        if source == "Manual Entry":
            tickers_input = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL")
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        else:
            watchlists = st.session_state['user_watchlists']['watchlists']
            if watchlists:
                selected_wl = st.selectbox("Select Watchlist", list(watchlists.keys()))
                tickers = watchlists[selected_wl]
            else:
                st.warning("⚠️ No watchlists found! Create one in Watchlist Manager first.")
                return

    with col2:
        period = st.selectbox("Period", ["1 Month", "3 Months", "6 Months", "1 Year"], index=2)
        days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
        days = days_map[period]

    if st.button("📊 Compare Stocks", type="primary"):
        if len(tickers) < 2:
            st.error("❌ Select at least 2 stocks to compare")
            return

        with st.spinner(f"Loading comparison data for {len(tickers)} stocks..."):
            try:
                # Fetch data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                stocks_data = get_multiple_stocks(
                    tickers,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                # Calculate metrics for each stock
                comparison_data = []
                for ticker in tickers:
                    if ticker not in stocks_data or stocks_data[ticker] is None or len(stocks_data[ticker]) < 20:
                        st.warning(f"⚠️ Skipping {ticker}: Insufficient data")
                        continue

                    df = stocks_data[ticker]

                    # Prepare features
                    df_features = FeatureEngineer.prepare_features(df)

                    # Get recommendation
                    rec = get_investment_recommendation(df_features)

                    current = df['Close'].iloc[-1]
                    start_price = df['Close'].iloc[0]
                    total_return = ((current - start_price) / start_price) * 100

                    comparison_data.append({
                        'Ticker': ticker,
                        'Price': f"${current:.2f}",
                        'Return %': f"{total_return:+.2f}%",
                        'RSI': f"{df_features['RSI'].iloc[-1]:.1f}" if 'RSI' in df_features else 'N/A',
                        'Volatility': f"{rec['risk_metrics']['volatility']:.1f}%",
                        'Tech Score': rec['technical_score'],
                        'Recommendation': rec['recommendation']
                    })

                if not comparison_data:
                    st.error("❌ No valid stock data to compare")
                    return

                # Display comparison table
                st.markdown("### 📋 Comparison Table")
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)

                # Normalized price chart
                st.markdown("### 📈 Price Performance (Normalized to 100)")
                fig = go.Figure()

                for ticker in tickers:
                    if ticker in stocks_data and stocks_data[ticker] is not None and len(stocks_data[ticker]) > 0:
                        df = stocks_data[ticker]
                        normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=normalized,
                            mode='lines',
                            name=ticker,
                            line=dict(width=2)
                        ))

                fig.update_layout(
                    yaxis_title='Normalized Price (Start = 100)',
                    xaxis_title='Date',
                    height=500,
                    template='plotly_white',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Two columns for correlation and technical scores
                col1, col2 = st.columns(2)

                with col1:
                    # Correlation heatmap
                    st.markdown("### 🔗 Correlation Matrix")
                    returns_data = {}
                    for ticker in tickers:
                        if ticker in stocks_data and stocks_data[ticker] is not None and len(stocks_data[ticker]) > 0:
                            returns_data[ticker] = stocks_data[ticker]['Close'].pct_change()

                    if len(returns_data) >= 2:
                        df_returns = pd.DataFrame(returns_data).dropna()
                        correlation = df_returns.corr()

                        fig_corr = go.Figure(data=go.Heatmap(
                            z=correlation.values,
                            x=correlation.columns,
                            y=correlation.columns,
                            colorscale='RdYlGn',
                            zmid=0,
                            text=correlation.values,
                            texttemplate='%{text:.2f}',
                            textfont={"size": 10},
                            colorbar=dict(title="Correlation")
                        ))
                        fig_corr.update_layout(
                            height=400,
                            template='plotly_white',
                            xaxis=dict(side="bottom")
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                with col2:
                    # Technical scores bar chart
                    st.markdown("### 🎯 Technical Scores")
                    scores = {item['Ticker']: item['Tech Score'] for item in comparison_data}

                    fig_scores = go.Figure(data=[
                        go.Bar(
                            x=list(scores.keys()),
                            y=list(scores.values()),
                            marker=dict(
                                color=list(scores.values()),
                                colorscale='RdYlGn',
                                cmin=0,
                                cmax=100,
                                colorbar=dict(title="Score")
                            ),
                            text=list(scores.values()),
                            textposition='auto'
                        )
                    ])
                    fig_scores.update_layout(
                        yaxis_title='Technical Score (0-100)',
                        xaxis_title='Stock',
                        height=400,
                        template='plotly_white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)

                # Export comparison
                st.markdown("---")
                csv = df_comparison.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison as CSV",
                    data=csv,
                    file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"❌ Error comparing stocks: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    else:
        st.info("👆 Select stocks and click 'Compare Stocks' to get started!")

        st.markdown("### 💡 Comparison Features:")
        st.markdown("""
        - **Side-by-Side Metrics**: Compare price, returns, RSI, volatility, and technical scores
        - **Normalized Performance**: See which stock performed best over the period
        - **Correlation Matrix**: Understand how stocks move together
        - **Technical Scores**: Quick visual comparison of technical strength
        - **Export Data**: Download comparison table as CSV
        """)


def show_watchlist_manager_page():
    """Watchlist management page"""
    st.title("📋 Watchlist Manager")

    st.markdown("""
    Organize your favorite stocks into custom watchlists for quick access and comparison.
    Create multiple watchlists (e.g., "Tech Stocks", "Dividend Stocks", "Growth Stocks")
    """)

    watchlists = st.session_state['user_watchlists']['watchlists']

    # Create new watchlist
    st.markdown("### Create New Watchlist")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_name = st.text_input("Watchlist Name", key="new_watchlist", placeholder="e.g., Tech Giants")
    with col2:
        st.write("")  # Spacing
        if st.button("➕ Create", type="primary"):
            if new_name and new_name not in watchlists:
                watchlists[new_name] = []
                st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                save_json_data('watchlists.json', st.session_state['user_watchlists'])
                st.success(f"✅ Created watchlist: {new_name}")
                st.rerun()
            elif new_name in watchlists:
                st.error(f"Watchlist '{new_name}' already exists!")
            else:
                st.error("Please enter a watchlist name")

    st.markdown("---")

    # Display each watchlist in tabs
    if watchlists:
        st.markdown("### My Watchlists")
        tabs = st.tabs(list(watchlists.keys()))

        for i, (name, tickers) in enumerate(watchlists.items()):
            with tabs[i]:
                # Add ticker to this watchlist
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    new_ticker = st.text_input(
                        "Add Stock Ticker",
                        key=f"add_{name}",
                        placeholder="e.g., AAPL"
                    ).upper()
                with col2:
                    if st.button("➕ Add Stock", key=f"btn_{name}"):
                        if new_ticker and new_ticker not in tickers:
                            tickers.append(new_ticker)
                            st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                            save_json_data('watchlists.json', st.session_state['user_watchlists'])
                            st.success(f"Added {new_ticker}")
                            st.rerun()
                        elif new_ticker in tickers:
                            st.warning(f"{new_ticker} already in watchlist")
                        else:
                            st.error("Enter a ticker symbol")
                with col3:
                    if st.button("🗑️ Delete Watchlist", key=f"del_wl_{name}"):
                        del watchlists[name]
                        st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                        save_json_data('watchlists.json', st.session_state['user_watchlists'])
                        st.success(f"Deleted {name}")
                        st.rerun()

                # Show tickers
                if tickers:
                    st.markdown(f"**{len(tickers)} stocks in this watchlist:**")

                    # Fetch current prices
                    with st.spinner("Loading current prices..."):
                        try:
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=5)
                            stocks_data = get_multiple_stocks(
                                tickers,
                                start=start_date.strftime('%Y-%m-%d'),
                                end=end_date.strftime('%Y-%m-%d')
                            )

                            # Create summary table
                            summary = []
                            for ticker in tickers:
                                if ticker in stocks_data:
                                    df = stocks_data[ticker]
                                    if df is not None and len(df) > 0:
                                        current = df['Close'].iloc[-1]
                                        prev = df['Close'].iloc[-2] if len(df) > 1 else current
                                        change = current - prev
                                        change_pct = (change / prev) * 100
                                        volume = df['Volume'].iloc[-1]

                                        summary.append({
                                            'Ticker': ticker,
                                            'Price': f"${current:.2f}",
                                            'Change': f"${change:+.2f}",
                                            'Change %': f"{change_pct:+.2f}%",
                                            'Volume': f"{volume/1e6:.1f}M"
                                        })
                                    else:
                                        summary.append({
                                            'Ticker': ticker,
                                            'Price': 'N/A',
                                            'Change': 'N/A',
                                            'Change %': 'N/A',
                                            'Volume': 'N/A'
                                        })
                                else:
                                    summary.append({
                                        'Ticker': ticker,
                                        'Price': 'N/A',
                                        'Change': 'N/A',
                                        'Change %': 'N/A',
                                        'Volume': 'N/A'
                                    })

                            if summary:
                                df_summary = pd.DataFrame(summary)
                                st.dataframe(df_summary, use_container_width=True, hide_index=True)

                                # Quick actions
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"📊 Compare All {len(tickers)} Stocks", key=f"compare_{name}"):
                                        st.info("💡 Go to 'Stock Comparison' page to compare these stocks")
                                with col2:
                                    # Export watchlist
                                    csv = df_summary.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Export as CSV",
                                        data=csv,
                                        file_name=f"watchlist_{name.replace(' ', '_')}.csv",
                                        mime='text/csv',
                                        key=f"export_{name}"
                                    )

                        except Exception as e:
                            st.error(f"Error loading prices: {e}")

                    # Remove individual stocks
                    st.markdown("**Remove stocks:**")
                    cols = st.columns(min(len(tickers), 4))
                    for idx, ticker in enumerate(tickers):
                        with cols[idx % 4]:
                            if st.button(f"✖️ {ticker}", key=f"remove_{name}_{ticker}"):
                                tickers.remove(ticker)
                                st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                                save_json_data('watchlists.json', st.session_state['user_watchlists'])
                                st.rerun()

                else:
                    st.info(f"This watchlist is empty. Add stocks using the form above.")

    else:
        st.info("📝 Create your first watchlist to get started!")


if __name__ == "__main__":
    main()
