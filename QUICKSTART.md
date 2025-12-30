# Quick Start Guide

Get started with the Stock Analysis Toolkit in 5 minutes!

## For macOS Users (Your System)

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
./setup.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Verify the installation
4. Show you next steps

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python verify_installation.py
```

---

## Important Notes for macOS

⚠️ On macOS, use these commands:
- `python3` instead of `python`
- `pip3` instead of `pip` (or just `pip` inside virtual environment)

Example:
```bash
# ❌ Wrong
python verify_installation.py

# ✅ Correct
python3 verify_installation.py

# ✅ Or activate venv first
source venv/bin/activate
python verify_installation.py  # Now this works!
```

---

## Next Steps After Installation

### 1. Start Jupyter Notebook

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Launch Jupyter
jupyter notebook
```

Your browser will open automatically.

### 2. Open First Tutorial

Navigate to: `notebooks/01_data_collection.ipynb`

### 3. Run Your First Analysis

Try this in a Jupyter cell:

```python
from src.data.fetcher import get_stock_data

# Fetch Apple stock data
df = get_stock_data('AAPL', start='2024-01-01')

# Display first few rows
print(df.head())

# Show current price
print(f"Current price: ${df['Close'].iloc[-1]:.2f}")
```

---

## Quick Reference Commands

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Run Tests
```bash
source venv/bin/activate
python run_tests.py
```

### Verify Installation
```bash
source venv/bin/activate
python verify_installation.py
```

### Launch Jupyter
```bash
source venv/bin/activate
jupyter notebook
```

---

## Troubleshooting

### "command not found: python"
Use `python3` instead:
```bash
python3 verify_installation.py
```

### "Permission denied"
Make script executable:
```bash
chmod +x setup.sh
./setup.sh
```

### "No module named 'pandas'"
Activate virtual environment first:
```bash
source venv/bin/activate
```

### Dependencies won't install
Try updating pip:
```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## File Structure Overview

```
stock/
├── setup.sh                    # 👈 Run this first!
├── QUICKSTART.md              # 👈 You are here
├── README.md                  # Full documentation
├── requirements.txt           # Dependencies
│
├── notebooks/                 # 👈 Start here after setup
│   ├── 01_data_collection.ipynb
│   ├── 02_technical_analysis.ipynb
│   └── ...
│
├── src/                       # Python modules
│   ├── data/
│   ├── indicators/
│   ├── models/
│   └── ...
│
└── tests/                     # Test files
```

---

## Learning Path

**Beginner:**
1. `notebooks/01_data_collection.ipynb` - Learn to fetch data
2. `notebooks/02_technical_analysis.ipynb` - Technical indicators
3. `notebooks/03_fundamental_analysis.ipynb` - Company analysis

**Intermediate:**
4. `notebooks/04_ml_price_prediction.ipynb` - Machine learning
5. `notebooks/05_backtesting.ipynb` - Strategy testing
6. `notebooks/06_dashboard.ipynb` - Interactive tools

**Advanced:**
7. `notebooks/examples/example_aapl_analysis.ipynb` - Complete analysis
8. `notebooks/examples/example_portfolio_analysis.ipynb` - Portfolio tracking

---

## Common Tasks

### Analyze a Stock
```python
from src.data.fetcher import get_stock_data
from src.indicators import TrendIndicators

df = get_stock_data('AAPL', start='2023-01-01')
df['SMA_50'] = TrendIndicators.calculate_sma(df, 50)
print(df.tail())
```

### Compare Multiple Stocks
```python
from src.data.fetcher import get_multiple_stocks

stocks = get_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'], start='2023-01-01')

for ticker, data in stocks.items():
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    print(f"{ticker}: {total_return:.2f}% return")
```

### Run a Backtest
```python
from src.data.fetcher import get_stock_data
from src.backtesting import BacktestEngine, SMACrossover

df = get_stock_data('AAPL', start='2022-01-01')
data = df[['Open', 'High', 'Low', 'Close', 'Volume']]

engine = BacktestEngine(data, initial_cash=10000)
results = engine.run_backtest(SMACrossover)
print(results)
```

---

## Getting Help

1. **Documentation**: Check `README.md` and `INSTALLATION.md`
2. **Testing Guide**: See `TESTING.md`
3. **Examples**: Review notebooks in `notebooks/examples/`
4. **Verify Setup**: Run `python verify_installation.py`

---

## Success Checklist

- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] `verify_installation.py` passes all checks
- [ ] Jupyter notebook launches
- [ ] Can fetch stock data
- [ ] First notebook runs successfully

---

**You're ready to start!** 🚀

Run `./setup.sh` to begin, then `jupyter notebook` to explore!
