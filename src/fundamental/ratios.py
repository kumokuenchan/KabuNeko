"""
Financial Ratios Module

This module provides functions to calculate various financial ratios
for fundamental analysis of stocks.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Union


class FinancialRatios:
    """
    A class for calculating financial ratios and metrics.
    """

    @staticmethod
    def get_stock_info(ticker: str) -> Dict:
        """
        Get basic stock information from yfinance.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Dict: Stock information

        Example:
            >>> info = FinancialRatios.get_stock_info('AAPL')
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {}

    @staticmethod
    def pe_ratio(ticker: str, use_forward: bool = False) -> Optional[float]:
        """
        Calculate Price-to-Earnings (P/E) ratio.

        Args:
            ticker (str): Stock ticker symbol
            use_forward (bool): Use forward P/E if available

        Returns:
            Optional[float]: P/E ratio

        Example:
            >>> pe = FinancialRatios.pe_ratio('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)

        if use_forward and 'forwardPE' in info:
            return info.get('forwardPE')
        else:
            return info.get('trailingPE')

    @staticmethod
    def pb_ratio(ticker: str) -> Optional[float]:
        """
        Calculate Price-to-Book (P/B) ratio.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: P/B ratio

        Example:
            >>> pb = FinancialRatios.pb_ratio('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('priceToBook')

    @staticmethod
    def ps_ratio(ticker: str) -> Optional[float]:
        """
        Calculate Price-to-Sales (P/S) ratio.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: P/S ratio

        Example:
            >>> ps = FinancialRatios.ps_ratio('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('priceToSalesTrailing12Months')

    @staticmethod
    def peg_ratio(ticker: str) -> Optional[float]:
        """
        Calculate PEG (Price/Earnings to Growth) ratio.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: PEG ratio

        Example:
            >>> peg = FinancialRatios.peg_ratio('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('pegRatio')

    @staticmethod
    def ev_ebitda(ticker: str) -> Optional[float]:
        """
        Calculate EV/EBITDA ratio.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: EV/EBITDA ratio

        Example:
            >>> ev_ebitda = FinancialRatios.ev_ebitda('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('enterpriseToEbitda')

    @staticmethod
    def roe(ticker: str) -> Optional[float]:
        """
        Calculate Return on Equity (ROE).

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: ROE as percentage

        Example:
            >>> roe = FinancialRatios.roe('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        roe_value = info.get('returnOnEquity')

        if roe_value is not None:
            return roe_value * 100  # Convert to percentage
        return None

    @staticmethod
    def roa(ticker: str) -> Optional[float]:
        """
        Calculate Return on Assets (ROA).

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: ROA as percentage

        Example:
            >>> roa = FinancialRatios.roa('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        roa_value = info.get('returnOnAssets')

        if roa_value is not None:
            return roa_value * 100  # Convert to percentage
        return None

    @staticmethod
    def debt_to_equity(ticker: str) -> Optional[float]:
        """
        Calculate Debt-to-Equity ratio.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: Debt-to-Equity ratio

        Example:
            >>> de = FinancialRatios.debt_to_equity('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('debtToEquity')

    @staticmethod
    def current_ratio(ticker: str) -> Optional[float]:
        """
        Calculate Current Ratio.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: Current ratio

        Example:
            >>> cr = FinancialRatios.current_ratio('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('currentRatio')

    @staticmethod
    def quick_ratio(ticker: str) -> Optional[float]:
        """
        Calculate Quick Ratio (Acid Test).

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: Quick ratio

        Example:
            >>> qr = FinancialRatios.quick_ratio('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('quickRatio')

    @staticmethod
    def profit_margins(ticker: str) -> Dict[str, Optional[float]]:
        """
        Get various profit margins.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Dict: Gross, operating, and net profit margins

        Example:
            >>> margins = FinancialRatios.profit_margins('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)

        gross = info.get('grossMargins')
        operating = info.get('operatingMargins')
        net = info.get('profitMargins')

        return {
            'gross_margin': gross * 100 if gross else None,
            'operating_margin': operating * 100 if operating else None,
            'net_margin': net * 100 if net else None
        }

    @staticmethod
    def dividend_yield(ticker: str) -> Optional[float]:
        """
        Get dividend yield.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: Dividend yield as percentage

        Example:
            >>> dy = FinancialRatios.dividend_yield('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        div_yield = info.get('dividendYield')

        if div_yield is not None:
            return div_yield * 100  # Convert to percentage
        return None

    @staticmethod
    def payout_ratio(ticker: str) -> Optional[float]:
        """
        Get dividend payout ratio.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: Payout ratio as percentage

        Example:
            >>> pr = FinancialRatios.payout_ratio('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        payout = info.get('payoutRatio')

        if payout is not None:
            return payout * 100  # Convert to percentage
        return None

    @staticmethod
    def beta(ticker: str) -> Optional[float]:
        """
        Get stock beta (volatility vs market).

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Optional[float]: Beta value

        Example:
            >>> beta = FinancialRatios.beta('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)
        return info.get('beta')

    @staticmethod
    def get_all_ratios(ticker: str) -> pd.DataFrame:
        """
        Get all major financial ratios for a stock.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            pd.DataFrame: DataFrame with all ratios

        Example:
            >>> ratios = FinancialRatios.get_all_ratios('AAPL')
        """
        info = FinancialRatios.get_stock_info(ticker)

        if not info:
            print(f"No data available for {ticker}")
            return pd.DataFrame()

        # Valuation ratios
        valuation = {
            'P/E Ratio (Trailing)': info.get('trailingPE'),
            'P/E Ratio (Forward)': info.get('forwardPE'),
            'P/B Ratio': info.get('priceToBook'),
            'P/S Ratio': info.get('priceToSalesTrailing12Months'),
            'PEG Ratio': info.get('pegRatio'),
            'EV/EBITDA': info.get('enterpriseToEbitda'),
        }

        # Profitability ratios
        gross_margin = info.get('grossMargins')
        operating_margin = info.get('operatingMargins')
        net_margin = info.get('profitMargins')
        roe_val = info.get('returnOnEquity')
        roa_val = info.get('returnOnAssets')

        profitability = {
            'Gross Margin (%)': gross_margin * 100 if gross_margin else None,
            'Operating Margin (%)': operating_margin * 100 if operating_margin else None,
            'Net Margin (%)': net_margin * 100 if net_margin else None,
            'ROE (%)': roe_val * 100 if roe_val else None,
            'ROA (%)': roa_val * 100 if roa_val else None,
        }

        # Liquidity ratios
        liquidity = {
            'Current Ratio': info.get('currentRatio'),
            'Quick Ratio': info.get('quickRatio'),
        }

        # Leverage ratios
        leverage = {
            'Debt-to-Equity': info.get('debtToEquity'),
        }

        # Dividend ratios
        div_yield = info.get('dividendYield')
        payout = info.get('payoutRatio')

        dividend = {
            'Dividend Yield (%)': div_yield * 100 if div_yield else None,
            'Payout Ratio (%)': payout * 100 if payout else None,
        }

        # Risk metrics
        risk = {
            'Beta': info.get('beta'),
        }

        # Combine all ratios
        all_ratios = {
            **valuation,
            **profitability,
            **liquidity,
            **leverage,
            **dividend,
            **risk
        }

        # Create DataFrame
        df = pd.DataFrame({
            'Metric': all_ratios.keys(),
            'Value': all_ratios.values()
        })

        return df

    @staticmethod
    def compare_ratios(tickers: list) -> pd.DataFrame:
        """
        Compare financial ratios across multiple stocks.

        Args:
            tickers (list): List of ticker symbols

        Returns:
            pd.DataFrame: Comparison DataFrame

        Example:
            >>> comp = FinancialRatios.compare_ratios(['AAPL', 'MSFT', 'GOOGL'])
        """
        comparison = {}

        for ticker in tickers:
            info = FinancialRatios.get_stock_info(ticker)

            if not info:
                continue

            # Extract key ratios
            gross_margin = info.get('grossMargins')
            net_margin = info.get('profitMargins')
            roe_val = info.get('returnOnEquity')

            comparison[ticker] = {
                'P/E': info.get('trailingPE'),
                'P/B': info.get('priceToBook'),
                'P/S': info.get('priceToSalesTrailing12Months'),
                'PEG': info.get('pegRatio'),
                'ROE (%)': roe_val * 100 if roe_val else None,
                'Gross Margin (%)': gross_margin * 100 if gross_margin else None,
                'Net Margin (%)': net_margin * 100 if net_margin else None,
                'Debt/Equity': info.get('debtToEquity'),
                'Current Ratio': info.get('currentRatio'),
                'Beta': info.get('beta'),
            }

        df = pd.DataFrame(comparison).T
        return df

    @staticmethod
    def assess_valuation(ticker: str) -> str:
        """
        Provide a simple valuation assessment.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            str: Valuation assessment

        Example:
            >>> assessment = FinancialRatios.assess_valuation('AAPL')
        """
        pe = FinancialRatios.pe_ratio(ticker)
        pb = FinancialRatios.pb_ratio(ticker)
        peg = FinancialRatios.peg_ratio(ticker)

        assessment = []

        if pe is not None:
            if pe < 15:
                assessment.append("P/E suggests undervalued")
            elif pe > 30:
                assessment.append("P/E suggests overvalued")
            else:
                assessment.append("P/E suggests fairly valued")

        if pb is not None:
            if pb < 1:
                assessment.append("P/B suggests undervalued")
            elif pb > 5:
                assessment.append("P/B suggests overvalued")
            else:
                assessment.append("P/B suggests fairly valued")

        if peg is not None:
            if peg < 1:
                assessment.append("PEG suggests undervalued relative to growth")
            elif peg > 2:
                assessment.append("PEG suggests overvalued relative to growth")
            else:
                assessment.append("PEG suggests fairly valued relative to growth")

        return "; ".join(assessment) if assessment else "Insufficient data for assessment"


# Convenience functions
def get_pe_ratio(ticker: str) -> Optional[float]:
    """Quick function to get P/E ratio."""
    return FinancialRatios.pe_ratio(ticker)


def get_all_ratios(ticker: str) -> pd.DataFrame:
    """Quick function to get all ratios."""
    return FinancialRatios.get_all_ratios(ticker)


def compare_stocks(tickers: list) -> pd.DataFrame:
    """Quick function to compare stocks."""
    return FinancialRatios.compare_ratios(tickers)


if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'

    print(f"\n=== Financial Ratios for {ticker} ===\n")

    # Individual ratios
    print(f"P/E Ratio: {FinancialRatios.pe_ratio(ticker)}")
    print(f"P/B Ratio: {FinancialRatios.pb_ratio(ticker)}")
    print(f"ROE: {FinancialRatios.roe(ticker):.2f}%" if FinancialRatios.roe(ticker) else "ROE: N/A")
    print(f"Debt-to-Equity: {FinancialRatios.debt_to_equity(ticker)}")

    # All ratios
    print("\n=== All Ratios ===")
    all_ratios = FinancialRatios.get_all_ratios(ticker)
    print(all_ratios.to_string(index=False))

    # Valuation assessment
    print(f"\n=== Valuation Assessment ===")
    print(FinancialRatios.assess_valuation(ticker))

    # Compare multiple stocks
    print("\n=== Stock Comparison ===")
    comparison = FinancialRatios.compare_ratios(['AAPL', 'MSFT', 'GOOGL'])
    print(comparison)
