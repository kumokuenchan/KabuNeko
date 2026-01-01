"""
Stock Analysis Dashboard - Page Modules

This module contains all page rendering functions for the dashboard.
Each page is in its own file with a render() function.
"""

from .home import render as render_home
from .stock_overview import render as render_stock_overview
from .investment_advice import render as render_investment_advice
from .technical_analysis import render as render_technical_analysis
from .price_prediction import render as render_price_prediction
from .backtesting import render as render_backtesting
from .portfolio import render as render_portfolio
from .alerts import render as render_alerts
from .performance_tracker import render as render_performance_tracker
from .stock_screener import render as render_stock_screener
from .stock_comparison import render as render_stock_comparison
from .watchlist_manager import render as render_watchlist_manager
from .crypto_analysis import render as render_crypto_analysis
from .news_sentiment import render as render_news_sentiment
from .insider_trading import render as render_insider_trading
from .earnings_calendar import render as render_earnings_calendar
from .pattern_scanner import render as render_pattern_scanner

__all__ = [
    'render_home',
    'render_stock_overview',
    'render_investment_advice',
    'render_technical_analysis',
    'render_price_prediction',
    'render_backtesting',
    'render_portfolio',
    'render_alerts',
    'render_performance_tracker',
    'render_stock_screener',
    'render_stock_comparison',
    'render_watchlist_manager',
    'render_crypto_analysis',
    'render_news_sentiment',
    'render_insider_trading',
    'render_earnings_calendar',
    'render_pattern_scanner',
]
