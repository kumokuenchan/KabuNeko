"""
Fundamental Analysis Module

This module provides comprehensive fundamental analysis tools including:
- Financial ratios (P/E, P/B, ROE, ROA, Debt ratios, etc.)
- Growth metrics (Revenue growth, Earnings growth, CAGR)
- Earnings trends and analysis
- Cash flow metrics
- Company comparison tools
"""

from .ratios import (
    FinancialRatios,
    get_pe_ratio,
    get_all_ratios,
    compare_stocks
)

from .metrics import (
    GrowthMetrics,
    get_revenue_growth,
    get_earnings_growth,
    get_growth_summary
)

__all__ = [
    # Classes
    'FinancialRatios',
    'GrowthMetrics',

    # Ratio functions
    'get_pe_ratio',
    'get_all_ratios',
    'compare_stocks',

    # Growth functions
    'get_revenue_growth',
    'get_earnings_growth',
    'get_growth_summary',
]
