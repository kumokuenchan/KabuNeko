# 📈 Stock Analysis Dashboard - User Guide

**Simple Web Interface - No Coding Required!**

---

## 🚀 Quick Start (3 Easy Steps)

### Step 1: Open Terminal

On your Mac:
1. Press `Command + Space` to open Spotlight
2. Type "Terminal" and press Enter

### Step 2: Navigate to the Project

In Terminal, type these commands (press Enter after each):

```bash
cd /Users/kuen/stock
source venv/bin/activate
```

You should see `(venv)` appear in your terminal - this means you're ready!

### Step 3: Launch the Dashboard

Type this command and press Enter:

```bash
streamlit run app.py
```

🎉 **That's it!** Your web browser will automatically open with the dashboard!

If it doesn't open automatically, look for a message in Terminal that says:
```
Local URL: http://localhost:8501
```

Copy that URL and paste it into your web browser.

---

## 📱 Using the Dashboard

### Home Page - Your Starting Point

When you first open the dashboard, you'll see:
- **Welcome message** with instructions
- **Quick market stats** (S&P 500, Dow Jones, NASDAQ)
- **Popular stocks list** for easy reference

### Navigation Sidebar (Left Side)

Click any of these options to explore:

#### 🏠 **Home**
- Welcome page with instructions
- Market overview
- Quick links to popular stocks

#### 📊 **Stock Overview**
- View real-time stock prices
- See beautiful candlestick charts
- Check trading volume
- Get key metrics like 52-week high/low

**How to use:**
1. Enter a stock ticker (like `AAPL` for Apple)
2. Choose your time period (1 month, 1 year, etc.)
3. Click "Load Stock Data"
4. Explore the charts!

#### 📉 **Technical Analysis**
- Add moving averages to charts
- View RSI (shows if stock is overbought/oversold)
- Display MACD, Bollinger Bands, and more
- Get buy/sell signal interpretations

**How to use:**
1. First, load a stock from "Stock Overview"
2. Check the boxes for indicators you want to see
3. Charts update automatically
4. Read the interpretations below each chart

#### 🤖 **Price Prediction**
- Use AI to predict future stock prices
- See how accurate the model is
- Get 1-30 day forecasts
- Understand what features the AI looks at

**How to use:**
1. First, load a stock from "Stock Overview"
2. Choose how many days ahead to predict (1-30)
3. Click "Generate Prediction"
4. Wait while the AI trains (about 30 seconds)
5. View the forecast chart and predictions table

**Important**: Predictions are estimates only! Don't rely on them alone for investing.

#### ⚡ **Backtesting**
- Test trading strategies on historical data
- See what would have happened if you traded in the past
- Compare different strategies
- Understand win rates and returns

**How to use:**
1. First, load a stock from "Stock Overview"
2. Choose a strategy (SMA Crossover or RSI Mean Reversion)
3. Set your parameters with the sliders
4. Enter your initial investment amount
5. Click "Run Backtest"
6. Review the results and equity curve

**Available Strategies:**
- **SMA Crossover**: Buy when short-term average crosses above long-term
- **RSI Mean Reversion**: Buy when oversold, sell when overbought

#### 💼 **Portfolio Analysis**
- Track multiple stocks at once
- Compare performance across stocks
- See which stocks move together (correlation)
- Identify your best and worst performers

**How to use:**
1. Enter stock tickers (one per line)
2. Choose your time period
3. Click "Analyze Portfolio"
4. View performance table and charts
5. Check the correlation matrix

---

## 💡 Tips for Non-Technical Users

### What is a Stock Ticker?

A ticker is a short code for a company's stock. Examples:
- `AAPL` = Apple
- `MSFT` = Microsoft
- `GOOGL` = Google
- `AMZN` = Amazon
- `TSLA` = Tesla

**Where to find tickers:**
- Google the company name + "stock ticker"
- Visit Yahoo Finance or Google Finance

### Understanding the Charts

**Candlestick Chart:**
- Green/white candle = price went up that day
- Red/black candle = price went down that day
- Top of candle = highest price
- Bottom of candle = lowest price

**Moving Averages (Lines on chart):**
- Shows the average price over time
- When lines cross, it might signal a buy or sell opportunity

**RSI (0-100 scale):**
- Above 70 = Stock might be "overbought" (could go down soon)
- Below 30 = Stock might be "oversold" (could go up soon)
- 30-70 = Neutral zone

### Common Terms Explained

**Backtesting**: Testing a strategy on past data to see if it would have worked

**Sharpe Ratio**: Measures return vs risk. Higher is better. Above 1 is good.

**Max Drawdown**: Biggest drop from peak. Lower is better. -20% means the portfolio fell 20% at its worst.

**Win Rate**: Percentage of profitable trades. 60% means 6 out of 10 trades made money.

**Correlation**: How stocks move together. 1 = move together perfectly, -1 = move in opposite directions.

---

## ❓ Troubleshooting

### Dashboard won't load?

1. Make sure you're in the right folder:
   ```bash
   cd /Users/kuen/stock
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Try launching again:
   ```bash
   streamlit run app.py
   ```

### "No data found for ticker"?

- Double-check the ticker symbol spelling
- Make sure the stock exists and trades publicly
- Try a common stock like `AAPL` to test

### Charts look weird?

- Try selecting a different time period
- Make sure you have enough historical data (at least 60 days recommended)
- Refresh the page (press F5)

### Browser doesn't open automatically?

Look in Terminal for a line that says:
```
Local URL: http://localhost:8501
```

Copy that URL and paste it into Chrome, Safari, or your preferred browser.

### Error messages?

- Most errors happen when there's not enough data
- Try loading 6 months or 1 year of data
- For predictions, you need at least 60 days of data
- For backtesting, you need at least 3 months of data

---

## 🛑 Stopping the Dashboard

When you're done:

1. Go to the Terminal window
2. Press `Control + C` (hold Control, then press C)
3. Type `deactivate` and press Enter
4. Close the Terminal

---

## 📞 Getting Help

### Need More Data?

The dashboard needs internet to fetch stock data. Make sure you're connected!

### Want to Learn More?

Check out these files in the project folder:
- `README.md` - Full technical documentation
- `QUICKSTART.md` - Quick setup guide
- `INSTALLATION.md` - Installation help

### Still Stuck?

Common issues:
1. **"Command not found"** - Make sure you activated the virtual environment
2. **"No module named..."** - Run the setup script: `./setup.sh`
3. **Slow loading** - First-time fetching can be slow, be patient!

---

## 🎓 Learning Resources

### Want to understand more?

**Books:**
- "A Random Walk Down Wall Street" by Burton Malkiel
- "The Intelligent Investor" by Benjamin Graham

**Websites:**
- Investopedia.com - Free stock market education
- Yahoo Finance - Real-time stock data and news

**YouTube Channels:**
- The Plain Bagel
- Financial Education
- Andrei Jikh

---

## ⚠️ Important Disclaimer

**This tool is for educational purposes only!**

- Past performance does not guarantee future results
- Stock market investing carries risk
- AI predictions are estimates, not guarantees
- Always do your own research
- Consider consulting a licensed financial advisor
- Never invest money you can't afford to lose

---

## 🎉 Enjoy!

You now have a powerful stock analysis tool at your fingertips!

Remember:
- Start with the "Stock Overview" page
- Load a stock before using other features
- Explore the different indicators and tools
- Have fun learning about the stock market!

**Happy investing! 📈📊💰**
