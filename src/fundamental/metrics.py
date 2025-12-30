"""
Growth and Earnings Metrics Module

This module provides functions to calculate growth metrics,
earnings trends, and other fundamental analysis metrics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Tuple
from datetime import datetime


class GrowthMetrics:
    """
    A class for calculating growth and earnings metrics.
    """

    @staticmethod
    def get_financials(ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements for a stock.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Dict: Dictionary with income statement, balance sheet, cash flow

        Example:
            >>> financials = GrowthMetrics.get_financials('AAPL')
        """
        try:
            stock = yf.Ticker(ticker)

            return {
                'income_stmt': stock.income_stmt,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cash_flow,
                'quarterly_income': stock.quarterly_income_stmt,
                'quarterly_balance': stock.quarterly_balance_sheet,
                'quarterly_cashflow': stock.quarterly_cash_flow
            }
        except Exception as e:
            print(f"Error fetching financials for {ticker}: {e}")
            return {}

    @staticmethod
    def revenue_growth(ticker: str, quarterly: bool = False) -> Optional[pd.Series]:
        """
        Calculate revenue growth rate (Year-over-Year).

        Args:
            ticker (str): Stock ticker symbol
            quarterly (bool): Use quarterly data if True

        Returns:
            Optional[pd.Series]: Revenue growth rates

        Example:
            >>> growth = GrowthMetrics.revenue_growth('AAPL')
        """
        financials = GrowthMetrics.get_financials(ticker)

        if quarterly:
            income = financials.get('quarterly_income')
        else:
            income = financials.get('income_stmt')

        if income is None or income.empty:
            return None

        # Get total revenue row
        if 'Total Revenue' in income.index:
            revenue = income.loc['Total Revenue']
        elif 'TotalRevenue' in income.index:
            revenue = income.loc['TotalRevenue']
        else:
            print("Revenue data not found")
            return None

        # Calculate growth rate
        growth = revenue.pct_change(periods=-1) * 100  # Negative because dates are descending

        return growth

    @staticmethod
    def earnings_growth(ticker: str, quarterly: bool = False) -> Optional[pd.Series]:
        """
        Calculate earnings (net income) growth rate.

        Args:
            ticker (str): Stock ticker symbol
            quarterly (bool): Use quarterly data if True

        Returns:
            Optional[pd.Series]: Earnings growth rates

        Example:
            >>> growth = GrowthMetrics.earnings_growth('AAPL')
        """
        financials = GrowthMetrics.get_financials(ticker)

        if quarterly:
            income = financials.get('quarterly_income')
        else:
            income = financials.get('income_stmt')

        if income is None or income.empty:
            return None

        # Get net income row
        if 'Net Income' in income.index:
            earnings = income.loc['Net Income']
        elif 'NetIncome' in income.index:
            earnings = income.loc['NetIncome']
        else:
            print("Net Income data not found")
            return None

        # Calculate growth rate
        growth = earnings.pct_change(periods=-1) * 100

        return growth

    @staticmethod
    def eps_trend(ticker: str) -> Optional[pd.DataFrame]:
        """
        Get EPS (Earnings Per Share) trend.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[pd.DataFrame]: EPS historical data

        Example:
            >>> eps = GrowthMetrics.eps_trend('AAPL')
        """
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings

            if earnings is not None and not earnings.empty:
                return earnings

            return None
        except Exception as e:
            print(f"Error fetching EPS for {ticker}: {e}")
            return None

    @staticmethod
    def quarterly_earnings_trend(ticker: str) -> Optional[pd.DataFrame]:
        """
        Get quarterly earnings trend.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[pd.DataFrame]: Quarterly earnings data

        Example:
            >>> qtr_earnings = GrowthMetrics.quarterly_earnings_trend('AAPL')
        """
        try:
            stock = yf.Ticker(ticker)
            quarterly_earnings = stock.quarterly_earnings

            if quarterly_earnings is not None and not quarterly_earnings.empty:
                return quarterly_earnings

            return None
        except Exception as e:
            print(f"Error fetching quarterly earnings for {ticker}: {e}")
            return None

    @staticmethod
    def calculate_cagr(ticker: str, years: int = 5,
                      metric: str = 'revenue') -> Optional[float]:
        """
        Calculate Compound Annual Growth Rate (CAGR).

        Args:
            ticker (str): Stock ticker symbol
            years (int): Number of years for CAGR calculation
            metric (str): 'revenue' or 'earnings'

        Returns:
            Optional[float]: CAGR percentage

        Example:
            >>> cagr = GrowthMetrics.calculate_cagr('AAPL', years=5)
        """
        financials = GrowthMetrics.get_financials(ticker)
        income = financials.get('income_stmt')

        if income is None or income.empty:
            return None

        # Get appropriate metric
        if metric == 'revenue':
            if 'Total Revenue' in income.index:
                data = income.loc['Total Revenue']
            elif 'TotalRevenue' in income.index:
                data = income.loc['TotalRevenue']
            else:
                return None
        elif metric == 'earnings':
            if 'Net Income' in income.index:
                data = income.loc['Net Income']
            elif 'NetIncome' in income.index:
                data = income.loc['NetIncome']
            else:
                return None
        else:
            return None

        # Sort by date (ascending)
        data = data.sort_index()

        # Get beginning and ending values
        if len(data) < 2:
            return None

        # Try to get values for the specified number of years
        if len(data) >= years + 1:
            beginning_value = data.iloc[0]
            ending_value = data.iloc[years]
            actual_years = years
        else:
            # Use all available data
            beginning_value = data.iloc[0]
            ending_value = data.iloc[-1]
            actual_years = len(data) - 1

        if beginning_value <= 0 or ending_value <= 0:
            return None

        # Calculate CAGR
        cagr = (((ending_value / beginning_value) ** (1 / actual_years)) - 1) * 100

        return cagr

    @staticmethod
    def get_growth_summary(ticker: str) -> pd.DataFrame:
        """
        Get a summary of growth metrics.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            pd.DataFrame: Growth metrics summary

        Example:
            >>> summary = GrowthMetrics.get_growth_summary('AAPL')
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get growth rates from info
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        revenue_per_share = info.get('revenuePerShare')

        # Calculate CAGRs
        revenue_cagr_5y = GrowthMetrics.calculate_cagr(ticker, years=5, metric='revenue')
        earnings_cagr_5y = GrowthMetrics.calculate_cagr(ticker, years=5, metric='earnings')

        metrics = {
            'Revenue Growth (YoY)': revenue_growth * 100 if revenue_growth else None,
            'Earnings Growth (YoY)': earnings_growth * 100 if earnings_growth else None,
            'Revenue CAGR (5Y)': revenue_cagr_5y,
            'Earnings CAGR (5Y)': earnings_cagr_5y,
            'Revenue Per Share': revenue_per_share,
        }

        df = pd.DataFrame({
            'Metric': metrics.keys(),
            'Value': metrics.values()
        })

        return df

    @staticmethod
    def earnings_surprise(ticker: str) -> Optional[pd.DataFrame]:
        """
        Get earnings surprise history.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[pd.DataFrame]: Earnings surprise data

        Example:
            >>> surprise = GrowthMetrics.earnings_surprise('AAPL')
        """
        try:
            quarterly_earnings = GrowthMetrics.quarterly_earnings_trend(ticker)

            if quarterly_earnings is None:
                return None

            # Calculate surprise percentage
            if 'Revenue' in quarterly_earnings.columns and 'Earnings' in quarterly_earnings.columns:
                quarterly_earnings['Earnings_Surprise_%'] = \
                    ((quarterly_earnings['Earnings'] - quarterly_earnings['Earnings'].shift(1)) /
                     quarterly_earnings['Earnings'].shift(1).abs()) * 100

            return quarterly_earnings

        except Exception as e:
            print(f"Error calculating earnings surprise for {ticker}: {e}")
            return None

    @staticmethod
    def cash_flow_metrics(ticker: str) -> pd.DataFrame:
        """
        Calculate cash flow metrics.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            pd.DataFrame: Cash flow metrics

        Example:
            >>> cf_metrics = GrowthMetrics.cash_flow_metrics('AAPL')
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        metrics = {
            'Operating Cash Flow': info.get('operatingCashflow'),
            'Free Cash Flow': info.get('freeCashflow'),
            'Operating Margin (%)': info.get('operatingMargins', 0) * 100,
            'EBITDA': info.get('ebitda'),
            'EBITDA Margins (%)': info.get('ebitdaMargins', 0) * 100,
        }

        df = pd.DataFrame({
            'Metric': metrics.keys(),
            'Value': metrics.values()
        })

        return df

    @staticmethod
    def compare_growth_metrics(tickers: list) -> pd.DataFrame:
        """
        Compare growth metrics across multiple stocks.

        Args:
            tickers (list): List of ticker symbols

        Returns:
            pd.DataFrame: Comparison DataFrame

        Example:
            >>> comp = GrowthMetrics.compare_growth_metrics(['AAPL', 'MSFT', 'GOOGL'])
        """
        comparison = {}

        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info

            revenue_growth = info.get('revenueGrowth')
            earnings_growth = info.get('earningsGrowth')
            revenue_cagr = GrowthMetrics.calculate_cagr(ticker, years=5, metric='revenue')

            comparison[ticker] = {
                'Revenue Growth (%)': revenue_growth * 100 if revenue_growth else None,
                'Earnings Growth (%)': earnings_growth * 100 if earnings_growth else None,
                'Revenue CAGR 5Y (%)': revenue_cagr,
                'Market Cap': info.get('marketCap'),
                'Enterprise Value': info.get('enterpriseValue'),
            }

        df = pd.DataFrame(comparison).T
        return df

    @staticmethod
    def assess_growth(ticker: str) -> str:
        """
        Provide a growth assessment.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            str: Growth assessment

        Example:
            >>> assessment = GrowthMetrics.assess_growth('AAPL')
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')

        assessment = []

        if revenue_growth is not None:
            rev_pct = revenue_growth * 100
            if rev_pct > 20:
                assessment.append("Strong revenue growth (>20%)")
            elif rev_pct > 10:
                assessment.append("Moderate revenue growth (10-20%)")
            elif rev_pct > 0:
                assessment.append("Slow revenue growth (<10%)")
            else:
                assessment.append("Revenue declining")

        if earnings_growth is not None:
            earn_pct = earnings_growth * 100
            if earn_pct > 20:
                assessment.append("Strong earnings growth (>20%)")
            elif earn_pct > 10:
                assessment.append("Moderate earnings growth (10-20%)")
            elif earn_pct > 0:
                assessment.append("Slow earnings growth (<10%)")
            else:
                assessment.append("Earnings declining")

        return "; ".join(assessment) if assessment else "Insufficient growth data"


# Convenience functions
def get_revenue_growth(ticker: str, quarterly: bool = False) -> Optional[pd.Series]:
    """Quick function to get revenue growth."""
    return GrowthMetrics.revenue_growth(ticker, quarterly)


def get_earnings_growth(ticker: str, quarterly: bool = False) -> Optional[pd.Series]:
    """Quick function to get earnings growth."""
    return GrowthMetrics.earnings_growth(ticker, quarterly)


def get_growth_summary(ticker: str) -> pd.DataFrame:
    """Quick function to get growth summary."""
    return GrowthMetrics.get_growth_summary(ticker)


if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'

    print(f"\n=== Growth Metrics for {ticker} ===\n")

    # Revenue growth
    print("Revenue Growth (Annual):")
    rev_growth = GrowthMetrics.revenue_growth(ticker)
    if rev_growth is not None:
        print(rev_growth.tail())

    # CAGR
    print(f"\nRevenue CAGR (5Y): {GrowthMetrics.calculate_cagr(ticker, years=5, metric='revenue'):.2f}%")

    # Growth summary
    print("\n=== Growth Summary ===")
    summary = GrowthMetrics.get_growth_summary(ticker)
    print(summary.to_string(index=False))

    # Growth assessment
    print(f"\n=== Growth Assessment ===")
    print(GrowthMetrics.assess_growth(ticker))

    # Compare growth metrics
    print("\n=== Growth Comparison ===")
    comparison = GrowthMetrics.compare_growth_metrics(['AAPL', 'MSFT', 'GOOGL'])
    print(comparison)
