# Stock Analysis Toolkit

A comprehensive Python-based toolkit for stock market analysis featuring technical analysis, fundamental analysis, machine learning price prediction, interactive visualizations, and backtesting capabilities.

## 🎯 Two Ways to Use This Toolkit

### For Non-Technical Users: 🌐 Web Dashboard (Recommended!)

**No coding required!** Use the beautiful web interface:

```bash
# After installation, just run:
streamlit run app.py
```

Your browser will open with an easy-to-use dashboard featuring:
- 📊 Real-time stock price charts
- 📉 Technical analysis indicators (point-and-click)
- 🤖 AI price predictions
- ⚡ Strategy backtesting
- 💼 Portfolio tracking

👉 **See [USER_GUIDE.md](USER_GUIDE.md) for step-by-step instructions**

### For Technical Users: 📓 Jupyter Notebooks

Use the included notebooks for full control and customization:
- 8 interactive notebooks covering all features
- Full access to Python code
- Customize and extend functionality

## Features

### 1. Technical Analysis
- 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, etc.)
- Signal detection (Golden Cross, Death Cross, RSI overbought/oversold)
- Customizable indicator parameters
- Both library-based and educational custom implementations

### 2. Fundamental Analysis
- Financial ratio calculations (P/E, P/B, ROE, Debt-to-Equity, etc.)
- Financial statement analysis (Income Statement, Balance Sheet, Cash Flow)
- Earnings and growth metrics
- Peer comparison tools

### 3. Machine Learning Price Prediction
- Random Forest Regressor
- XGBoost
- LSTM (Long Short-Term Memory) neural networks
- Feature engineering utilities
- Model evaluation and comparison tools

### 4. Interactive Visualizations
- Candlestick charts
- Interactive Plotly charts with zoom/pan
- Technical indicator overlays
- Multi-stock comparison
- Returns distribution analysis

### 5. Backtesting
- Strategy backtesting framework using backtesting.py
- Pre-built strategies (SMA crossover, RSI mean reversion, MACD, etc.)
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Parameter optimization

### 6. Data Management
- yfinance integration for stock data
- Intelligent caching system
- Support for multiple time intervals
- Historical and real-time data access

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd stock
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Fetch Stock Data

```python
from src.data.fetcher import get_stock_data

# Fetch Apple stock data
df = get_stock_data('AAPL', start='2023-01-01')
print(df.head())
```

### 2. Calculate Technical Indicators

```python
from src.indicators.trend import calculate_sma, calculate_macd

# Add moving averages
df['SMA_20'] = calculate_sma(df, period=20)
df['SMA_50'] = calculate_sma(df, period=50)

# Calculate MACD
macd_df = calculate_macd(df)
```

### 3. Visualize Stock Data

```python
from src.visualization.charts import plot_stock

# Create an interactive chart
plot_stock(df, title='AAPL Stock Price', interactive=True)
```

### 4. Run Jupyter Notebooks

```bash
jupyter notebook
```

Then open `notebooks/01_data_collection.ipynb` to get started.

## Project Structure

```
stock/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_collection.ipynb     # Data fetching and exploration
│   ├── 02_technical_analysis.ipynb  # Technical indicators
│   ├── 03_fundamental_analysis.ipynb # Fundamental analysis
│   ├── 04_ml_price_prediction.ipynb # ML models
│   ├── 05_backtesting.ipynb         # Strategy backtesting
│   ├── 06_dashboard.ipynb           # Interactive dashboard
│   └── examples/                    # Example analyses
│       ├── example_aapl_analysis.ipynb      # Complete AAPL analysis
│       └── example_portfolio_analysis.ipynb # Portfolio tracking
│
├── src/                              # Reusable Python modules
│   ├── data/
│   │   ├── fetcher.py              # Data fetching with yfinance
│   │   └── preprocessor.py         # Data cleaning and preparation
│   │
│   ├── indicators/
│   │   ├── trend.py                # Trend indicators (SMA, EMA, MACD, ADX)
│   │   ├── momentum.py             # Momentum indicators (RSI, Stochastic)
│   │   ├── volatility.py           # Volatility indicators (BB, ATR)
│   │   └── volume.py               # Volume indicators (OBV, VWAP)
│   │
│   ├── fundamental/
│   │   ├── ratios.py               # Financial ratios
│   │   └── metrics.py              # Growth metrics
│   │
│   ├── models/
│   │   ├── lstm_model.py           # LSTM implementation
│   │   ├── random_forest.py        # Random Forest
│   │   ├── feature_engineering.py  # Feature creation
│   │   └── evaluation.py           # Model evaluation
│   │
│   ├── backtesting/
│   │   ├── engine.py               # Backtesting wrapper
│   │   ├── strategies.py           # Trading strategies
│   │   └── metrics.py              # Performance metrics
│   │
│   └── visualization/
│       ├── charts.py               # Charting utilities
│       └── dashboard.py            # Dashboard components
│
├── data/                            # Data storage (git-ignored)
│   ├── raw/                        # Raw downloaded data
│   ├── processed/                  # Processed data
│   └── cache/                      # Cached API responses
│
├── models/                          # Saved ML models (git-ignored)
│   └── saved_models/
│
├── config/
│   ├── config.yaml                 # Configuration settings
│   └── stocks.yaml                 # Stock watchlists
│
└── tests/                           # Unit tests
```

## Notebooks Guide

All notebooks are complete and ready to use! Start with notebook 01 and progress through to 06.

### 01_data_collection.ipynb
Introduction to fetching and exploring stock market data.

**Topics covered:**
- Fetching historical data with yfinance
- Understanding OHLCV data structure
- Multiple time intervals (daily, weekly, monthly)
- Fundamental data access
- Data caching
- Basic visualization

### 02_technical_analysis.ipynb
Complete guide to technical indicators and signal detection.

**Topics covered:**
- Trend indicators (SMA, EMA, MACD, ADX)
- Momentum indicators (RSI, Stochastic, Williams %R)
- Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
- Volume indicators (OBV, VWAP, A/D Line)
- Signal detection (golden cross, overbought/oversold)
- Interactive candlestick charts
- Multi-stock technical comparison

### 03_fundamental_analysis.ipynb
Company fundamental analysis and valuation.

**Topics covered:**
- Financial statement analysis
- 20+ financial ratio calculations (P/E, P/B, ROE, ROA, etc.)
- Profitability, liquidity, and leverage metrics
- Growth metrics and CAGR calculations
- Earnings trends and surprise analysis
- Peer comparison
- Investment scoring system

### 04_ml_price_prediction.ipynb
Machine learning for stock price prediction with comprehensive warnings.

**Topics covered:**
- Feature engineering (50+ features from OHLCV data)
- Random Forest regression
- LSTM neural networks (optional, requires TensorFlow)
- Model comparison and evaluation
- Directional accuracy metrics
- Trading simulation with P/L
- Extensive disclaimers about prediction limitations

### 05_backtesting.ipynb
Strategy development, backtesting, and performance analysis.

**Topics covered:**
- Introduction to backtesting concepts
- 7 pre-built strategies (SMA crossover, RSI, MACD, Bollinger Bands, etc.)
- Strategy comparison framework
- Parameter optimization (with overfitting warnings)
- Walk-forward analysis
- Comprehensive performance metrics
- Best practices and common pitfalls

### 06_dashboard.ipynb
Interactive analysis dashboard with widgets and real-time updates.

**Topics covered:**
- Interactive stock selector
- Real-time price charts with Plotly
- Technical indicator toggles
- Fundamental metrics display
- Custom analysis panel
- Multi-stock comparison
- Sector rotation analysis
- Portfolio tracking

### Example Notebooks

#### example_aapl_analysis.ipynb
Complete end-to-end analysis of Apple Inc. (AAPL) demonstrating the full toolkit.

**Includes:**
- Data collection and price overview
- Complete technical analysis with 15+ indicators
- Fundamental ratio analysis
- ML model training and prediction
- Strategy backtesting
- Investment recommendation system with scoring

#### example_portfolio_analysis.ipynb
Portfolio-level analysis and optimization.

**Includes:**
- Portfolio construction and tracking
- Performance metrics by holding
- Correlation analysis and heatmaps
- Risk metrics (volatility, Sharpe ratio, max drawdown)
- Diversification scoring
- Rebalancing recommendations
- Interactive portfolio dashboard

## Usage Examples

### Example 1: Fetch and Analyze Multiple Stocks

```python
from src.data.fetcher import StockDataFetcher

fetcher = StockDataFetcher()

# Fetch tech stocks
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
stocks_data = fetcher.get_multiple_stocks(tech_stocks, start='2023-01-01')

# Compare performance
for ticker, df in stocks_data.items():
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    print(f"{ticker}: {total_return:.2f}% return")
```

### Example 2: Technical Analysis

```python
from src.data.fetcher import get_stock_data
from src.indicators.trend import TrendIndicators

# Fetch data
df = get_stock_data('AAPL', start='2023-01-01')

# Calculate indicators
df['SMA_20'] = TrendIndicators.sma(df, period=20)
df['SMA_50'] = TrendIndicators.sma(df, period=50)

# Detect golden cross
golden_cross = TrendIndicators.detect_golden_cross(df)
print(f"Golden crosses detected: {golden_cross.sum()}")
```

### Example 3: Data Preprocessing

```python
from src.data.preprocessor import StockDataPreprocessor

# Clean data
df_clean = StockDataPreprocessor.clean_data(df)

# Add features
df_clean = StockDataPreprocessor.add_returns(df_clean)
df_clean = StockDataPreprocessor.add_volatility(df_clean)

# Split for ML
train_df, test_df = StockDataPreprocessor.split_data(df_clean, train_size=0.8)
```

## Configuration

Edit `config/config.yaml` to customize:
- Data cache settings
- Model parameters
- Backtesting settings
- Visualization preferences
- Default indicator parameters

Edit `config/stocks.yaml` to manage:
- Stock watchlists (tech, finance, healthcare, etc.)
- Portfolio configurations

## Important Notes

### Data Source
This toolkit uses **yfinance** to fetch stock data from Yahoo Finance. Please note:
- Data is free but subject to Yahoo Finance's terms of service
- No API key required
- Rate limiting may apply for excessive requests
- Use caching to minimize API calls

### Market Prediction Disclaimer

**This toolkit is for educational and research purposes only.**

- Stock market prediction is inherently uncertain
- Past performance does not guarantee future results
- Machine learning models may not accurately predict stock prices
- Do not use this toolkit as the sole basis for investment decisions
- Always consult with a qualified financial advisor
- The authors are not responsible for any financial losses

### Backtesting Cautions

- Backtesting results may not reflect real trading performance
- Beware of overfitting and look-ahead bias
- Transaction costs and slippage affect real-world results
- Market conditions change over time

## Development Status

🎉 **All phases complete!** The Stock Analysis Toolkit is fully functional.

### Phase 1: Foundation ✅
- [x] Project structure
- [x] Data fetching (yfinance)
- [x] Data preprocessing
- [x] Basic trend indicators
- [x] Visualization utilities
- [x] First notebook (data collection)

### Phase 2: Technical Analysis ✅
- [x] Momentum indicators (RSI, Stochastic, Williams %R, ROC, CCI, MFI)
- [x] Volatility indicators (Bollinger Bands, ATR, Keltner Channels, Donchian)
- [x] Volume indicators (OBV, VWAP, A/D Line, CMF, Force Index)
- [x] Technical analysis notebook with interactive charts

### Phase 3: Fundamental Analysis ✅
- [x] Financial ratio calculations (20+ ratios)
- [x] Growth metrics and CAGR
- [x] Earnings trends and analysis
- [x] Fundamental analysis notebook with investment scoring

### Phase 4: Machine Learning ✅
- [x] Feature engineering (50+ features)
- [x] Random Forest implementation
- [x] LSTM implementation (optional TensorFlow)
- [x] Model evaluation tools
- [x] ML prediction notebook with comprehensive disclaimers

### Phase 5: Backtesting ✅
- [x] Backtesting engine wrapper
- [x] 7 pre-built strategies (SMA, RSI, MACD, BB, Trend Following, ML, Multi-Strategy)
- [x] Advanced performance metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
- [x] Parameter optimization and walk-forward analysis
- [x] Backtesting notebook with best practices

### Phase 6: Integration ✅
- [x] Interactive dashboard components
- [x] Dashboard notebook with widgets
- [x] Complete AAPL analysis example
- [x] Portfolio analysis example
- [x] Complete documentation

## Future Enhancements

Potential additions for community contributions:
- Sentiment analysis (news, social media APIs)
- Portfolio optimization algorithms (Modern Portfolio Theory)
- Real-time data streaming
- Advanced ML models (Transformers, Reinforcement Learning)
- Web application deployment (Streamlit/Dash)
- Options and derivatives analysis
- Automated email/SMS alerts
- Multi-timeframe analysis
- Custom indicator development framework

## What's Included

This toolkit provides a complete stock analysis framework:

**📊 Data & Analysis**
- 6 comprehensive Jupyter notebooks (250+ pages of tutorials)
- 2 complete example analyses (AAPL stock + Portfolio tracking)
- 20+ Python modules with 3,000+ lines of production-quality code
- 50+ technical indicators across 4 categories
- 20+ fundamental financial ratios

**🤖 Machine Learning**
- Feature engineering with 50+ automatically generated features
- Random Forest and LSTM model implementations
- Comprehensive evaluation metrics
- Trading simulation with P/L calculation

**📈 Backtesting**
- 7 pre-built trading strategies
- Walk-forward analysis to prevent overfitting
- 15+ performance metrics (Sharpe, Sortino, Calmar, VaR, etc.)
- Strategy comparison and optimization framework

**🎯 Interactive Tools**
- Stock analysis dashboard with real-time updates
- Portfolio tracking and rebalancing recommendations
- Correlation and diversification analysis
- Multi-stock comparison tools
- Sector rotation analysis

## Getting Started - Quick Tutorial

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Your First Analysis
```python
from src.data.fetcher import get_stock_data
from src.indicators import TrendIndicators, MomentumIndicators
from src.visualization.charts import plot_candlestick

# Fetch Apple stock data
df = get_stock_data('AAPL', start='2023-01-01')

# Add technical indicators
df['SMA_50'] = TrendIndicators.calculate_sma(df, 50)
df['RSI'] = MomentumIndicators.calculate_rsi(df, 14)

# Visualize
plot_candlestick(df.tail(90), title='AAPL Analysis',
                indicators=['SMA_50'])

print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
print(f"RSI: {df['RSI'].iloc[-1]:.2f}")
```

### Step 3: Explore the Notebooks
Start with `notebooks/01_data_collection.ipynb` and progress through all 6 notebooks to learn the complete toolkit.

## Contributing

Contributions are welcome! Here's how you can help:

1. **Bug Reports**: Open an issue describing the bug
2. **Feature Requests**: Suggest new features or enhancements
3. **Code Contributions**: Fork the repo and submit a pull request
4. **Documentation**: Improve tutorials and examples
5. **Testing**: Add unit tests for existing modules

**Potential Enhancement Areas:**
- Sentiment analysis (news, social media)
- Portfolio optimization (Markowitz, Black-Litterman)
- Real-time data streaming
- Advanced ML models (Transformers, RL)
- Web application deployment (Streamlit/Dash)
- Options and derivatives analysis
- Automated alerts and notifications

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check the notebooks for examples
2. Review the module documentation
3. Examine the configuration files

## Acknowledgments

Built with:
- [yfinance](https://github.com/ranaroussi/yfinance) - Stock data
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical analysis
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [TensorFlow](https://www.tensorflow.org/) - Deep learning
- [Plotly](https://plotly.com/) - Interactive visualizations
- [backtesting.py](https://kernc.github.io/backtesting.py/) - Strategy backtesting

---

**Happy analyzing! Remember: Always do your own research and never invest more than you can afford to lose.**
