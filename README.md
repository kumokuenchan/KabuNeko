# 📈 Stock Analysis Dashboard

A professional-grade stock analysis web application built with Streamlit, featuring AI-powered recommendations, technical indicators, backtesting, and portfolio tracking.

## 🌟 Features

- **📊 Stock Overview** - Real-time price charts, candlestick patterns, volume analysis
- **💡 Investment Advice** - AI-powered BUY/SELL/HOLD recommendations with risk assessment
- **📉 Technical Analysis** - 8+ indicators (RSI, MACD, Bollinger Bands, SMA, etc.)
- **🤖 Price Prediction** - Machine learning price forecasting using Random Forest
- **⚡ Backtesting** - Test trading strategies on historical data
- **💼 Portfolio Analysis** - Track multiple stocks, correlation analysis
- **📋 Watchlist Manager** - Create and manage custom stock watchlists
- **🔄 Stock Comparison** - Compare multiple stocks side-by-side
- **🔍 Stock Screener** - Find stocks matching technical criteria
- **🔔 Price Alerts** - Set alerts for price targets and indicators
- **💹 Performance Tracker** - Track paper trading performance with P&L analysis
- **₿ Crypto Analysis** - Dedicated cryptocurrency analysis with 24/7 market data and volatility metrics
- **📰 AI News Sentiment** - AI-powered analysis of news headlines with sentiment scoring and trend detection
- **💼 Insider Trading Tracker** - Monitor executive and insider transactions with buy/sell signals and multi-timeframe analysis
- **📊 Earnings Calendar** - Track upcoming earnings dates, analyze historical earnings performance, and monitor beat/miss rates
- **🔍 Chart Pattern Scanner** - AI-powered detection of technical patterns (head & shoulders, double tops/bottoms, triangles) with trading signals
- **🎯 Advanced Market Screener** - 7 preset screeners to find gap-ups/downs, unusual volume, momentum stocks, value plays, high beta, and 52-week high breakouts
- **🌐 Global Markets Dashboard** - Track 20 international indices (Asia, Europe, Americas including Malaysia KLCI), currencies (including MYR), commodities, and correlations with US markets in real-time
- **📦 ETF Holdings Explorer** - Discover what stocks are inside any ETF, view top holdings with weights, analyze sector allocation, and understand fund concentration
- **🌙 Dark Mode** - Toggle between light and dark themes

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure (Refactored!)

The codebase has been refactored from a single 2,428-line file into a clean, modular structure:

```
stock/
├── app.py                          # Main entry point (152 lines - 94% reduction!)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── DEVELOPER_GUIDE.md              # Developer documentation
│
├── data/
│   ├── cache/                      # Cached stock data
│   └── user_data/                  # User preferences & data
│       ├── watchlists.json
│       ├── alerts.json
│       ├── performance_tracker.json
│       └── preferences.json
│
└── src/
    ├── data/
    │   ├── fetcher.py              # Stock data fetching (yfinance)
    │   ├── persistence.py          # User data persistence
    │   ├── loader.py               # Standardized data loading utilities
    │   ├── insider_data.py         # Insider trading data
    │   ├── earnings_data.py        # Earnings calendar & analysis
    │   ├── global_markets.py       # Global markets data
    │   └── etf_data.py             # ETF holdings & info ⭐ NEW
    │
    ├── indicators/
    │   ├── trend.py                # Trend indicators (SMA, EMA, MACD)
    │   ├── momentum.py             # Momentum indicators (RSI, Stochastic)
    │   ├── volatility.py           # Volatility indicators (Bollinger Bands, ATR)
    │   └── volume.py               # Volume indicators (OBV, VWAP)
    │
    ├── models/
    │   ├── random_forest.py        # ML price prediction
    │   └── feature_engineering.py  # Feature creation for ML
    │
    ├── backtesting/
    │   └── strategies.py           # Trading strategies (SMA, RSI)
    │
    ├── analysis/
    │   ├── investment_recommendation.py  # AI recommendation engine
    │   ├── sentiment_analyzer.py   # News sentiment analysis
    │   ├── pattern_detector.py     # Chart pattern detection
    │   └── market_screener.py      # Advanced market screener ⭐ NEW
    │
    ├── fundamental/
    │   └── ratios.py               # Financial ratios
    │
    ├── ui/
    │   ├── themes.py               # Apple-inspired CSS themes ⭐ UPDATED
    │   ├── charts.py               # Reusable chart functions
    │   └── components.py           # UI component library ⭐ NEW
    │
    ├── alerts/
    │   └── checker.py              # Price alert monitoring
    │
    └── pages/                      # 20 page modules
        ├── __init__.py
        ├── home.py
        ├── stock_overview.py
        ├── investment_advice.py
        ├── technical_analysis.py
        ├── price_prediction.py
        ├── backtesting.py
        ├── portfolio.py
        ├── alerts.py
        ├── performance_tracker.py
        ├── stock_screener.py
        ├── stock_comparison.py
        ├── watchlist_manager.py
        ├── crypto_analysis.py
        ├── news_sentiment.py          # AI news sentiment
        ├── insider_trading.py         # Insider trading tracker
        ├── earnings_calendar.py       # Earnings calendar & analysis
        ├── pattern_scanner.py         # Chart pattern detection
        ├── market_screener.py         # Advanced market screener
        ├── global_markets.py          # Global markets dashboard
        └── etf_explorer.py            # ETF holdings explorer ⭐ NEW
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data**: yfinance, pandas, numpy
- **Visualization**: Plotly, mplfinance
- **Machine Learning**: scikit-learn, Random Forest
- **Technical Indicators**: pandas_ta, custom implementations
- **Backtesting**: backtesting.py library

## 📊 Usage Examples

### Analyze a Stock

1. Navigate to **Stock Overview**
2. Enter a ticker symbol (e.g., AAPL)
3. Select date range
4. Click "Load Stock Data"

### Get Investment Advice

1. Go to **Investment Advice**
2. Enter ticker symbol
3. Enable "Include AI Prediction"
4. Click "Analyze Stock"
5. Review BUY/SELL/HOLD recommendation with price targets

### Create a Watchlist

1. Open **Watchlist Manager**
2. Create a new watchlist
3. Add stocks to the watchlist
4. View real-time prices

### Screen for Opportunities

1. Navigate to **Stock Screener**
2. Select stock universe (watchlist or popular stocks)
3. Set screening criteria (RSI, MACD, volume, etc.)
4. Run screener to find matches

## 🎯 Recent Improvements

### Code Quality Enhancements (December 2025)

✅ **Fixed Deprecation Warnings**
- Replaced `use_container_width=True` with `width="stretch"` (28 instances)
- Code now compatible with Streamlit 1.40+

✅ **Extracted Duplicate Code**
- Created `src/ui/charts.py` with 9 reusable chart functions
- Created `src/data/loader.py` with 9 standardized data loading utilities
- Reduced code duplication by ~500 lines

✅ **Added Type Hints**
- Type annotations added to all utility modules
- Improved IDE support and code clarity
- Better error detection

✅ **Modular Architecture**
- Refactored from 2,428-line monolithic file to 16 organized modules
- app.py reduced to 152 lines (94% smaller!)
- Each page in its own file for easy maintenance

## 🔧 Configuration

### Default Settings

- Default stock list: Tech Giants (AAPL, MSFT, GOOGL, AMZN, META)
- Cache duration: 1 hour
- Dark mode: Off (toggle in sidebar)

### Customization

Edit watchlist options in `app.py`:

```python
if watchlist == "Custom Group":
    default_stocks = ["TICKER1", "TICKER2", "TICKER3"]
```

## 📝 Development

### Adding a New Page

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for detailed instructions.

Quick steps:
1. Create `src/pages/new_page.py` with a `render()` function
2. Add import to `src/pages/__init__.py`
3. Add page name to navigation list in `app.py`
4. Add routing entry in `page_routes` dictionary

### Code Quality Tools

The codebase follows these principles:
- **Modular structure** - Each page is a separate module
- **Type hints** - Functions have type annotations
- **DRY principle** - Reusable utilities for common operations
- **Consistent styling** - Standardized chart creation and data loading

## 🐛 Troubleshooting

### "No data found" errors
- Check internet connection
- Verify ticker symbol is correct
- Try a different date range

### Slow performance
- Reduce date range for analysis
- Clear cache: delete `data/cache/` directory
- Disable AI prediction for faster results

### Import errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: 3.8+

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It is NOT financial advice. Always:
- Do your own research
- Consult a licensed financial advisor
- Understand the risks of trading/investing
- Never invest more than you can afford to lose

---

**Made with ❤️ using Streamlit**
