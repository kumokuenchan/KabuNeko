"""
Data Loader Module

Provides standardized data fetching with error handling and user feedback.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
from src.data.fetcher import get_stock_data, get_multiple_stocks


def load_stock_with_spinner(
    ticker: str,
    start_date: str,
    end_date: str,
    min_days: int = 1
) -> Optional[pd.DataFrame]:
    """
    Load stock data with loading spinner and error handling.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_days: Minimum required days of data

    Returns:
        DataFrame or None if error/insufficient data
    """
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            df = get_stock_data(ticker, start=start_date, end=end_date)

            if df is None or len(df) == 0:
                st.error(f"❌ No data found for {ticker}. Please check the ticker symbol.")
                return None

            if len(df) < min_days:
                st.error(f"❌ Insufficient data for {ticker}. Need at least {min_days} days of history (got {len(df)}).")
                return None

            st.success(f"✅ Successfully loaded {len(df)} days of data for {ticker}")
            return df

        except Exception as e:
            st.error(f"❌ Error loading data for {ticker}: {str(e)}")
            return None


def load_stock_quiet(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Load stock data quietly without user feedback (for background operations).

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame or None if error
    """
    try:
        return get_stock_data(ticker, start=start_date, end=end_date)
    except:
        return None


def load_recent_stock(
    ticker: str,
    days: int = 365
) -> Optional[pd.DataFrame]:
    """
    Load recent stock data for specified number of days.

    Args:
        ticker: Stock ticker symbol
        days: Number of days to fetch

    Returns:
        DataFrame or None if error
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    return load_stock_with_spinner(
        ticker,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


def load_multiple_stocks_with_progress(
    tickers: list,
    start_date: str,
    end_date: str
) -> dict:
    """
    Load multiple stocks with progress indicator.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping tickers to DataFrames
    """
    with st.spinner(f"Loading {len(tickers)} stocks..."):
        try:
            stocks_data = get_multiple_stocks(tickers, start=start_date, end=end_date)

            # Filter out failed loads
            successful = {k: v for k, v in stocks_data.items() if v is not None and len(v) > 0}
            failed = len(tickers) - len(successful)

            if failed > 0:
                st.warning(f"⚠️ Could not load {failed} stocks")

            if successful:
                st.success(f"✅ Successfully loaded {len(successful)} stocks")

            return successful

        except Exception as e:
            st.error(f"❌ Error loading stocks: {str(e)}")
            return {}


def validate_date_range(
    start_date: str,
    end_date: str
) -> Tuple[bool, str]:
    """
    Validate date range for stock data fetching.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        if start >= end:
            return False, "Start date must be before end date"

        if end > datetime.now():
            return False, "End date cannot be in the future"

        days_diff = (end - start).days
        if days_diff < 1:
            return False, "Date range must be at least 1 day"

        if days_diff > 7300:  # 20 years
            return False, "Date range cannot exceed 20 years"

        return True, ""

    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD"


def get_date_range_from_period(period: str) -> Tuple[str, str]:
    """
    Convert period selection to date range.

    Args:
        period: Period string (e.g., "1 Month", "1 Year")

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    days_map = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825
    }

    days = days_map.get(period, 365)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    return (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


def save_to_session_state(
    ticker: str,
    df: pd.DataFrame
) -> None:
    """
    Save stock data to session state for cross-page access.

    Args:
        ticker: Stock ticker symbol
        df: Stock data DataFrame
    """
    st.session_state['current_stock'] = ticker
    st.session_state['current_data'] = df


def get_from_session_state() -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Retrieve stock data from session state.

    Returns:
        Tuple of (ticker, DataFrame) or (None, None)
    """
    ticker = st.session_state.get('current_stock')
    df = st.session_state.get('current_data')

    if ticker and df is not None:
        return ticker, df

    return None, None
