"""
Stock Analysis Toolkit - Simple Web Interface
==============================================

A user-friendly dashboard for stock market analysis.
No coding required - just select stocks and explore!

How to run:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import utilities
from src.data.persistence import initialize_user_data, save_json_data
from src.ui.themes import get_theme_css
from src.alerts.checker import check_price_alerts

# Import page renderers
from src.pages import (
    render_home,
    render_stock_overview,
    render_investment_advice,
    render_technical_analysis,
    render_price_prediction,
    render_backtesting,
    render_portfolio,
    render_alerts,
    render_performance_tracker,
    render_stock_screener,
    render_stock_comparison,
    render_watchlist_manager,
    render_crypto_analysis,
    render_news_sentiment,
    render_insider_trading,
    render_earnings_calendar,
    render_pattern_scanner,
    render_market_screener,
    render_global_markets,
    render_etf_explorer,
)

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
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
            [
                "🏠 Home",
                "📊 Stock Overview",
                "💡 Investment Advice",
                "📉 Technical Analysis",
                "🤖 Price Prediction",
                "⚡ Backtesting",
                "💼 Portfolio Analysis",
                "📋 Watchlist Manager",
                "🔄 Stock Comparison",
                "🔍 Stock Screener",
                "🔔 Price Alerts",
                "💹 Performance Tracker",
                "₿ Crypto Analysis",
                "📰 News Sentiment",
                "💼 Insider Trading",
                "📊 Earnings Calendar",
                "🔍 Pattern Scanner",
                "🎯 Market Screener",
                "🌐 Global Markets",
                "📦 ETF Explorer"
            ],
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

    # Route to pages using dictionary mapping
    page_routes = {
        "🏠 Home": render_home,
        "📊 Stock Overview": render_stock_overview,
        "💡 Investment Advice": render_investment_advice,
        "📉 Technical Analysis": render_technical_analysis,
        "🤖 Price Prediction": render_price_prediction,
        "⚡ Backtesting": render_backtesting,
        "💼 Portfolio Analysis": render_portfolio,
        "📋 Watchlist Manager": render_watchlist_manager,
        "🔄 Stock Comparison": render_stock_comparison,
        "🔍 Stock Screener": render_stock_screener,
        "🔔 Price Alerts": render_alerts,
        "💹 Performance Tracker": render_performance_tracker,
        "₿ Crypto Analysis": render_crypto_analysis,
        "📰 News Sentiment": render_news_sentiment,
        "💼 Insider Trading": render_insider_trading,
        "📊 Earnings Calendar": render_earnings_calendar,
        "🔍 Pattern Scanner": render_pattern_scanner,
        "🎯 Market Screener": render_market_screener,
        "🌐 Global Markets": render_global_markets,
        "📦 ETF Explorer": render_etf_explorer,
    }

    # Find and render the selected page
    for page_name, render_func in page_routes.items():
        if page_name in page:
            render_func()
            break


if __name__ == "__main__":
    main()
