"""
Insider Trading Data Module

Fetches and processes insider trading transactions for stocks.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class InsiderDataFetcher:
    """Fetch and process insider trading data"""

    @staticmethod
    def fetch_insider_transactions(ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch insider trading transactions for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with insider transactions or None if error
        """
        try:
            stock = yf.Ticker(ticker)

            # Get insider transactions
            insider_txns = stock.insider_transactions

            if insider_txns is None or insider_txns.empty:
                return None

            # Reset index to make date a column
            insider_txns = insider_txns.reset_index()

            return insider_txns

        except Exception as e:
            print(f"Error fetching insider data: {e}")
            return None

    @staticmethod
    def fetch_insider_roster(ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch current insider roster (holders).

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with insider roster or None
        """
        try:
            stock = yf.Ticker(ticker)

            # Get insider roster
            insider_roster = stock.insider_roster_holders

            if insider_roster is None or insider_roster.empty:
                return None

            return insider_roster

        except Exception as e:
            print(f"Error fetching insider roster: {e}")
            return None

    @classmethod
    def analyze_insider_activity(cls, ticker: str, days: int = 180) -> Dict:
        """
        Analyze insider trading activity for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to analyze

        Returns:
            Dictionary with analysis results
        """
        txns = cls.fetch_insider_transactions(ticker)

        if txns is None or txns.empty:
            return {
                'total_transactions': 0,
                'total_buys': 0,
                'total_sells': 0,
                'net_activity': 0,
                'buy_value': 0,
                'sell_value': 0,
                'net_value': 0,
                'signal': 'No Data',
                'signal_emoji': '⚪',
                'recent_transactions': []
            }

        # Filter to recent transactions
        cutoff_date = datetime.now() - timedelta(days=days)

        # Parse dates - handle different column names
        date_col = None
        for col in ['Start Date', 'Date', 'date', 'start_date']:
            if col in txns.columns:
                date_col = col
                break

        if date_col:
            txns[date_col] = pd.to_datetime(txns[date_col], errors='coerce')
            recent_txns = txns[txns[date_col] >= cutoff_date].copy()
        else:
            recent_txns = txns.copy()

        # Identify buy/sell transactions
        # Look for 'Transaction' column or similar
        trans_col = None
        for col in ['Transaction', 'transaction', 'Type', 'type']:
            if col in recent_txns.columns:
                trans_col = col
                break

        if trans_col:
            buys = recent_txns[recent_txns[trans_col].str.contains('Purchase|Buy', case=False, na=False)]
            sells = recent_txns[recent_txns[trans_col].str.contains('Sale|Sell', case=False, na=False)]
        else:
            # Fallback: try to infer from shares or value
            buys = pd.DataFrame()
            sells = pd.DataFrame()

        # Calculate values
        total_buys = len(buys)
        total_sells = len(sells)
        total_transactions = len(recent_txns)

        # Try to calculate monetary values
        buy_value = 0
        sell_value = 0

        value_col = None
        for col in ['Value', 'value', 'Amount', 'amount']:
            if col in recent_txns.columns:
                value_col = col
                break

        if value_col:
            buy_value = buys[value_col].sum() if not buys.empty else 0
            sell_value = sells[value_col].sum() if not sells.empty else 0

        net_value = buy_value - sell_value
        net_activity = total_buys - total_sells

        # Determine signal
        if net_activity > 5:
            signal = "Strong Buy Signal"
            signal_emoji = "🟢🟢"
        elif net_activity > 2:
            signal = "Buy Signal"
            signal_emoji = "🟢"
        elif net_activity < -5:
            signal = "Strong Sell Signal"
            signal_emoji = "🔴🔴"
        elif net_activity < -2:
            signal = "Sell Signal"
            signal_emoji = "🔴"
        else:
            signal = "Neutral"
            signal_emoji = "⚪"

        # Get recent transactions for display
        recent_list = recent_txns.head(20).to_dict('records') if not recent_txns.empty else []

        return {
            'total_transactions': total_transactions,
            'total_buys': total_buys,
            'total_sells': total_sells,
            'net_activity': net_activity,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'net_value': net_value,
            'signal': signal,
            'signal_emoji': signal_emoji,
            'recent_transactions': recent_list,
            'dataframe': recent_txns
        }

    @classmethod
    def get_insider_summary(cls, ticker: str) -> Dict:
        """
        Get a summary of insider activity with multiple timeframes.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with multi-timeframe analysis
        """
        analysis_30d = cls.analyze_insider_activity(ticker, days=30)
        analysis_90d = cls.analyze_insider_activity(ticker, days=90)
        analysis_180d = cls.analyze_insider_activity(ticker, days=180)

        return {
            '30_days': analysis_30d,
            '90_days': analysis_90d,
            '180_days': analysis_180d
        }
