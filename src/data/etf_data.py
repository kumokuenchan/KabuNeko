"""
ETF Data Fetcher

Fetch and analyze ETF holdings, sector allocation, and fund information.
Uses web scraping for reliable holdings data.
"""

import yfinance as yf
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from streamlit import cache_data
import time


class ETFDataFetcher:
    """Fetch ETF holdings and fund information via web scraping"""

    # Popular ETFs for quick access
    POPULAR_ETFS = {
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ (NASDAQ-100)',
        'VOO': 'Vanguard S&P 500 ETF',
        'VTI': 'Vanguard Total Stock Market',
        'IVV': 'iShares Core S&P 500',
        'VEA': 'Vanguard FTSE Developed Markets',
        'IEMG': 'iShares Core MSCI Emerging Markets',
        'VWO': 'Vanguard FTSE Emerging Markets',
        'AGG': 'iShares Core U.S. Aggregate Bond',
        'BND': 'Vanguard Total Bond Market',
        'GLD': 'SPDR Gold Shares',
        'VNQ': 'Vanguard Real Estate',
        'XLF': 'Financial Select Sector SPDR',
        'XLE': 'Energy Select Sector SPDR',
        'XLK': 'Technology Select Sector SPDR',
        'XLV': 'Health Care Select Sector SPDR',
        'XLI': 'Industrial Select Sector SPDR',
        'XLY': 'Consumer Discretionary Select Sector SPDR',
        'XLP': 'Consumer Staples Select Sector SPDR',
        'ARKK': 'ARK Innovation ETF',
        'ARKG': 'ARK Genomic Revolution ETF',
    }

    @staticmethod
    def get_etf_info(ticker: str) -> Optional[Dict]:
        """
        Get basic ETF information

        Args:
            ticker: ETF ticker symbol

        Returns:
            Dictionary with ETF info or None if error
        """
        try:
            etf = yf.Ticker(ticker)
            info = etf.info

            # Get current price
            hist = etf.history(period='5d')
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
            prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0

            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'price': current_price,
                'change_pct': change_pct,
                'expense_ratio': info.get('expenseRatio', 0) * 100 if info.get('expenseRatio') else None,
                'yield': info.get('yield', 0) * 100 if info.get('yield') else None,
                'aum': info.get('totalAssets', 0),
                'inception_date': info.get('fundInceptionDate', 'N/A'),
                'category': info.get('category', 'N/A'),
                'fund_family': info.get('fundFamily', 'N/A'),
                'description': info.get('longBusinessSummary', 'No description available'),
            }

        except Exception as e:
            return None

    @staticmethod
    @cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def scrape_etfdb_holdings(ticker: str, limit: int = 15) -> Optional[pd.DataFrame]:
        """
        Scrape ETF holdings from etfdb.com

        Args:
            ticker: ETF ticker symbol
            limit: Maximum number of holdings to return

        Returns:
            DataFrame with holdings or None if error
        """
        try:
            # etfdb.com URL structure
            url = f"https://etfdb.com/etf/{ticker}/#holdings"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'lxml')

            # Find holdings table by looking for specific headers
            holdings_data = []

            tables = soup.find_all('table')

            for table in tables:
                # Check if this is the holdings table by examining headers
                header_row = table.find('tr')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]

                    # Look for holdings table headers (they might be duplicated like "SymbolSymbol")
                    is_holdings_table = False

                    if headers:
                        headers_text = ' '.join(headers).lower()
                        if ('symbol' in headers_text or 'ticker' in headers_text) and \
                           ('holding' in headers_text or 'name' in headers_text) and \
                           ('asset' in headers_text or 'weight' in headers_text or '%' in headers_text):
                            is_holdings_table = True

                    if is_holdings_table:
                        # Found the holdings table, extract data
                        rows = table.find_all('tr')[1:]  # Skip header

                        for row in rows[:limit]:
                            cols = row.find_all('td')

                            if len(cols) >= 3:
                                # Extract data
                                symbol = cols[0].get_text(strip=True)
                                name = cols[1].get_text(strip=True)
                                weight_text = cols[2].get_text(strip=True)

                                # Parse weight percentage
                                weight = 0.0
                                try:
                                    weight = float(weight_text.replace('%', '').strip())
                                except:
                                    pass

                                if symbol and name and symbol != 'Symbol':  # Skip any header rows
                                    holdings_data.append({
                                        'Symbol': symbol,
                                        'Holding': name,
                                        'Weight (%)': weight
                                    })

                        # If we found data, break
                        if holdings_data:
                            break

            if holdings_data:
                df = pd.DataFrame(holdings_data)
                return df.head(limit)

            return None

        except Exception as e:
            # Log error for debugging
            print(f"ETF scraping error for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def get_etf_holdings(ticker: str, limit: int = 15) -> Optional[pd.DataFrame]:
        """
        Get ETF top holdings using web scraping

        Args:
            ticker: ETF ticker symbol
            limit: Maximum number of holdings to return

        Returns:
            DataFrame with holdings or None if error
        """
        # Try web scraping first
        holdings = ETFDataFetcher.scrape_etfdb_holdings(ticker, limit)

        if holdings is not None and not holdings.empty:
            return holdings

        # Fallback to yfinance
        try:
            etf = yf.Ticker(ticker)

            # Try funds data
            try:
                funds_data = etf.funds_data
                if funds_data and 'holdings' in funds_data:
                    holdings_list = funds_data['holdings']
                    if holdings_list:
                        return pd.DataFrame(holdings_list[:limit])
            except:
                pass

            return None

        except Exception as e:
            return None

    @staticmethod
    def get_sector_allocation(ticker: str) -> Optional[Dict]:
        """
        Get ETF sector allocation

        Args:
            ticker: ETF ticker symbol

        Returns:
            Dictionary with sector allocations or None if error
        """
        try:
            etf = yf.Ticker(ticker)

            # Try to get sector weights
            sector_weights = None

            # Method 1: Check if ETF has sector breakdown in info
            info = etf.info
            if 'sectorWeightings' in info and info['sectorWeightings']:
                sector_weights = info['sectorWeightings']

            # Method 2: Try funds_data
            if not sector_weights:
                try:
                    funds_data = etf.funds_data
                    if funds_data and 'sector_weightings' in funds_data:
                        sector_weights = funds_data['sector_weightings']
                except:
                    pass

            # If we have sector weights, format them
            if sector_weights:
                # Convert to dictionary if needed
                if isinstance(sector_weights, list) and len(sector_weights) > 0:
                    sector_dict = {}
                    for item in sector_weights:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                sector_dict[key] = value * 100  # Convert to percentage
                    return sector_dict
                elif isinstance(sector_weights, dict):
                    return {k: v * 100 for k, v in sector_weights.items()}

            return None

        except Exception as e:
            return None

    @staticmethod
    def calculate_holdings_concentration(holdings_df: pd.DataFrame) -> Dict:
        """
        Calculate concentration metrics from holdings

        Args:
            holdings_df: DataFrame with holdings data

        Returns:
            Dictionary with concentration metrics
        """
        if holdings_df is None or holdings_df.empty:
            return {'top_10_weight': 0, 'concentration': 'N/A'}

        # Try to find weight column
        weight_col = None
        for col in ['Weight (%)', 'Weight', 'weight', '% of Total', 'pct_of_total']:
            if col in holdings_df.columns:
                weight_col = col
                break

        if weight_col:
            top_10_weight = holdings_df[weight_col].head(10).sum()

            # Determine concentration level
            if top_10_weight >= 70:
                concentration = 'Highly Concentrated'
            elif top_10_weight >= 50:
                concentration = 'Moderately Concentrated'
            else:
                concentration = 'Well Diversified'

            return {
                'top_10_weight': top_10_weight,
                'concentration': concentration
            }

        return {'top_10_weight': 0, 'concentration': 'N/A'}
