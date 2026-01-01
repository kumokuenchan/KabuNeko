"""
Earnings Data Fetcher

Fetch and analyze earnings data including upcoming earnings dates,
historical earnings performance, and earnings surprises.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List


class EarningsDataFetcher:
    """Fetch and analyze earnings data for stocks"""

    @staticmethod
    def fetch_earnings_dates(ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch upcoming and historical earnings dates

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with earnings dates and estimates, or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            earnings_dates = stock.earnings_dates

            if earnings_dates is None or earnings_dates.empty:
                return None

            return earnings_dates

        except Exception as e:
            print(f"Error fetching earnings dates for {ticker}: {e}")
            return None

    @staticmethod
    def fetch_earnings_history(ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical quarterly earnings data

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with historical earnings, or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.quarterly_earnings

            if earnings is None or earnings.empty:
                return None

            return earnings

        except Exception as e:
            print(f"Error fetching earnings history for {ticker}: {e}")
            return None

    @staticmethod
    def fetch_financials(ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch quarterly financial statements

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with financial data, or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            financials = stock.quarterly_financials

            if financials is None or financials.empty:
                return None

            return financials

        except Exception as e:
            print(f"Error fetching financials for {ticker}: {e}")
            return None

    @classmethod
    def analyze_earnings_surprises(cls, ticker: str) -> Dict:
        """
        Analyze earnings surprises (actual vs expected)

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with earnings surprise analysis
        """
        earnings_dates = cls.fetch_earnings_dates(ticker)

        if earnings_dates is None or earnings_dates.empty:
            return {
                'has_data': False,
                'total_reports': 0,
                'beat_count': 0,
                'miss_count': 0,
                'meet_count': 0,
                'avg_surprise_pct': 0,
                'recent_surprises': []
            }

        # Filter for reported earnings (not future estimates)
        df = earnings_dates.copy()
        df = df[df.index <= pd.Timestamp.now()]

        # Calculate surprises
        surprises = []
        beat_count = 0
        miss_count = 0
        meet_count = 0

        for idx, row in df.iterrows():
            eps_estimate = row.get('EPS Estimate')
            eps_actual = row.get('Reported EPS')

            if pd.notna(eps_estimate) and pd.notna(eps_actual):
                surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100 if eps_estimate != 0 else 0

                surprises.append({
                    'date': idx,
                    'estimate': eps_estimate,
                    'actual': eps_actual,
                    'surprise_pct': surprise_pct
                })

                if surprise_pct > 1:
                    beat_count += 1
                elif surprise_pct < -1:
                    miss_count += 1
                else:
                    meet_count += 1

        avg_surprise = sum(s['surprise_pct'] for s in surprises) / len(surprises) if surprises else 0

        return {
            'has_data': True,
            'total_reports': len(surprises),
            'beat_count': beat_count,
            'miss_count': miss_count,
            'meet_count': meet_count,
            'avg_surprise_pct': avg_surprise,
            'recent_surprises': sorted(surprises, key=lambda x: x['date'], reverse=True)[:8]
        }

    @classmethod
    def get_next_earnings_date(cls, ticker: str) -> Optional[Dict]:
        """
        Get the next upcoming earnings date

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with next earnings info, or None
        """
        earnings_dates = cls.fetch_earnings_dates(ticker)

        if earnings_dates is None or earnings_dates.empty:
            return None

        # Filter for future dates
        future_earnings = earnings_dates[earnings_dates.index > pd.Timestamp.now()]

        if future_earnings.empty:
            return None

        # Get the nearest future date
        next_date = future_earnings.index[0]
        next_row = future_earnings.iloc[0]

        return {
            'date': next_date,
            'eps_estimate': next_row.get('EPS Estimate'),
            'revenue_estimate': next_row.get('Revenue Estimate'),
            'days_until': (next_date - pd.Timestamp.now()).days
        }

    @classmethod
    def get_earnings_summary(cls, ticker: str) -> Dict:
        """
        Get comprehensive earnings summary

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with complete earnings analysis
        """
        next_earnings = cls.get_next_earnings_date(ticker)
        surprise_analysis = cls.analyze_earnings_surprises(ticker)
        earnings_history = cls.fetch_earnings_history(ticker)

        return {
            'next_earnings': next_earnings,
            'surprise_analysis': surprise_analysis,
            'earnings_history': earnings_history
        }
