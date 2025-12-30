# 🚀 Getting Started in 2 Minutes!

## For Non-Technical Users (No Coding!)

### Method 1: Use the Simple Launcher (Easiest!)

1. **Double-click** `launch_dashboard.sh` in your file browser
   - Or in Terminal: `./launch_dashboard.sh`

2. **Wait** for your browser to open (about 5-10 seconds)

3. **Start analyzing!** The dashboard will be ready at `http://localhost:8501`

### Method 2: Manual Launch

Open Terminal and type:

```bash
cd /Users/kuen/stock
source venv/bin/activate
streamlit run app.py
```

That's it! Your browser will open with the dashboard.

---

## What You Can Do

### 📊 **Stock Overview** (Start Here!)
1. Enter a stock ticker like `AAPL`
2. Click "Load Stock Data"
3. See beautiful charts and metrics

### 📉 **Technical Analysis**
1. First load a stock (from Stock Overview)
2. Check boxes for indicators you want
3. Get buy/sell signals

### 🤖 **Price Prediction**
1. First load a stock
2. Choose prediction days (1-30)
3. Click "Generate Prediction"
4. See AI forecast

### ⚡ **Backtesting**
1. First load a stock
2. Choose a strategy
3. Click "Run Backtest"
4. See what would have happened

### 💼 **Portfolio**
1. Enter multiple stock tickers
2. Click "Analyze Portfolio"
3. Compare performance

---

## Popular Stock Tickers to Try

- **AAPL** - Apple
- **MSFT** - Microsoft
- **GOOGL** - Google
- **AMZN** - Amazon
- **TSLA** - Tesla
- **NVDA** - NVIDIA
- **META** - Meta (Facebook)

---

## Need Help?

📖 **Detailed Guide**: See [USER_GUIDE.md](USER_GUIDE.md)

🔧 **Installation Issues**: See [INSTALLATION.md](INSTALLATION.md)

❓ **Common Problems**:

**Dashboard won't start?**
```bash
cd /Users/kuen/stock
source venv/bin/activate
streamlit run app.py
```

**No data for ticker?**
- Check ticker spelling (use Yahoo Finance to verify)
- Try a popular stock like AAPL first

**Charts not loading?**
- Make sure you loaded a stock from "Stock Overview" first
- Select at least 60 days of data

---

## 🛑 To Stop the Dashboard

1. Go to Terminal window
2. Press **Control + C**
3. Type `deactivate`

---

## 📚 Quick Tips

✅ **Do**:
- Start with "Stock Overview" page
- Load at least 6 months of data
- Experiment with different indicators
- Compare multiple stocks

❌ **Don't**:
- Rely solely on predictions for investing
- Forget to check the date range
- Use real money without research
- Ignore the risk warnings

---

## ⚠️ Important Reminder

This tool is for **education only**. Stock predictions are not guaranteed. Always do your own research and consult a financial advisor before investing real money.

---

## 🎉 You're Ready!

**To start now:**

```bash
cd /Users/kuen/stock
./launch_dashboard.sh
```

**Enjoy your stock analysis dashboard!** 📈
