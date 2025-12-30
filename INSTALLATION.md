# Installation and Setup Guide

Complete guide to setting up the Stock Analysis Toolkit on your system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)
5. [Optional Components](#optional-components)

---

## System Requirements

### Required

- **Python**: 3.8 or higher
- **pip**: Package installer for Python
- **Operating System**: Windows, macOS, or Linux

### Recommended

- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space for data caching
- **Internet**: For downloading stock data

---

## Installation Steps

### Step 1: Verify Python Installation

```bash
# Check Python version
python --version
# or
python3 --version

# Should show Python 3.8 or higher
```

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3` or download from python.org
- **Linux**: `sudo apt-get install python3 python3-pip`

### Step 2: Clone or Download Project

```bash
# If using git
git clone <repository-url>
cd stock

# Or download and extract ZIP file
unzip stock.zip
cd stock
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will install:
# - yfinance (stock data)
# - pandas, numpy (data manipulation)
# - matplotlib, plotly (visualization)
# - scikit-learn (machine learning)
# - backtesting (strategy testing)
# - jupyter (notebooks)
# - and more...
```

**Note**: TensorFlow (for LSTM models) is optional. If you want LSTM support:

```bash
# For CPU-only version
pip install tensorflow

# For GPU version (requires CUDA)
pip install tensorflow-gpu
```

### Step 5: Verify Installation

```bash
# Run verification tests
python verify_installation.py

# Expected output:
# ✅ All core modules installed successfully!
```

---

## Verification

### Quick Verification Test

Create a file `test_install.py`:

```python
#!/usr/bin/env python3
"""Quick installation test"""

def test_imports():
    """Test all critical imports"""
    try:
        # Core dependencies
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import yfinance as yf

        # Project modules
        from src.data.fetcher import get_stock_data
        from src.indicators import TrendIndicators
        from src.models import FeatureEngineer

        print("✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_data_fetch():
    """Test data fetching"""
    try:
        from src.data.fetcher import get_stock_data

        # Fetch small amount of data
        df = get_stock_data('AAPL', start='2024-01-01', end='2024-01-31')

        if df is not None and len(df) > 0:
            print("✅ Data fetching works!")
            return True
        else:
            print("❌ Data fetch returned empty")
            return False

    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Stock Analysis Toolkit Installation")
    print("=" * 60)

    # Test imports
    imports_ok = test_imports()

    # Test data fetch
    data_ok = test_data_fetch()

    # Summary
    print("\n" + "=" * 60)
    if imports_ok and data_ok:
        print("✅ Installation verified successfully!")
        print("\nNext steps:")
        print("1. Launch Jupyter: jupyter notebook")
        print("2. Open: notebooks/01_data_collection.ipynb")
    else:
        print("❌ Some tests failed. See errors above.")
    print("=" * 60)
```

Run it:

```bash
python test_install.py
```

### Run Test Suite

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests
python run_tests.py

# Or
python -m pytest tests/ -v
```

### Verify Notebooks

```bash
# Verify notebooks are valid
python verify_notebooks.py

# Launch Jupyter
jupyter notebook

# Open notebooks/01_data_collection.ipynb
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Solution 1: Reinstall dependencies
pip install -r requirements.txt

# Solution 2: Install specific package
pip install <package-name>

# Solution 3: Upgrade pip and retry
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Permission Denied

```bash
# Solution 1: Use --user flag
pip install --user -r requirements.txt

# Solution 2: Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: TensorFlow Installation Fails

TensorFlow is optional. If it fails:

```bash
# Option 1: Skip TensorFlow
# LSTM models won't work, but everything else will

# Option 2: Install CPU-only version
pip install tensorflow-cpu

# Option 3: Use conda
conda install tensorflow
```

### Issue: Jupyter Not Found

```bash
# Install Jupyter
pip install jupyter notebook

# Or use JupyterLab
pip install jupyterlab
```

### Issue: SSL Certificate Error (yfinance)

```bash
# macOS: Install certificates
/Applications/Python\ 3.X/Install\ Certificates.command

# Linux/Windows: Update certifi
pip install --upgrade certifi
```

### Issue: "No module named 'src'"

```bash
# Solution: Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

---

## Optional Components

### Development Tools

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint

# Type checking
pip install mypy
```

### Enhanced Testing

```bash
# Coverage reporting
pip install pytest-cov

# Parallel testing
pip install pytest-xdist

# Notebook testing
pip install nbval
```

### Additional Visualization

```bash
# Interactive plotting
pip install plotly-express

# Financial charts
pip install mplfinance

# Seaborn for statistical plots
pip install seaborn
```

---

## Post-Installation

### Configure Cache Directory

Edit `config/config.yaml`:

```yaml
data:
  cache_dir: "data/cache"  # Change if needed
  default_start: "2020-01-01"
  default_interval: "1d"
```

### Set Up Watchlists

Edit `config/stocks.yaml`:

```yaml
watchlists:
  my_stocks:
    - AAPL
    - MSFT
    - GOOGL
  # Add your own watchlists
```

### Create Data Directories

```bash
# Create cache directories
mkdir -p data/cache
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/saved_models
```

---

## Quick Start After Installation

```bash
# 1. Activate virtual environment (if using one)
source venv/bin/activate

# 2. Launch Jupyter
jupyter notebook

# 3. Open first notebook
# Navigate to notebooks/01_data_collection.ipynb

# 4. Run cells and start learning!
```

---

## Updating

To update the toolkit:

```bash
# Pull latest changes (if using git)
git pull

# Update dependencies
pip install --upgrade -r requirements.txt

# Verify update
python verify_installation.py
```

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Delete virtual environment
rm -rf venv/

# Delete cache data (optional)
rm -rf data/cache/

# Delete saved models (optional)
rm -rf models/saved_models/
```

---

## Getting Help

If you encounter issues:

1. Check this documentation
2. Review [TESTING.md](TESTING.md)
3. Check [README.md](README.md)
4. Run diagnostic: `python test_install.py`
5. Open an issue on GitHub

---

## System-Specific Notes

### macOS

```bash
# If using Apple Silicon (M1/M2)
# TensorFlow installation may require:
pip install tensorflow-macos
pip install tensorflow-metal  # For GPU acceleration
```

### Windows

```bash
# Use Command Prompt or PowerShell
# Activate virtual environment:
venv\Scripts\activate.bat  # CMD
venv\Scripts\Activate.ps1  # PowerShell

# If PowerShell execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux

```bash
# Install Python development headers
sudo apt-get install python3-dev

# For plotting backends
sudo apt-get install python3-tk
```

---

## Success Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] All requirements installed (`pip list`)
- [ ] Test script passes (`python test_install.py`)
- [ ] Can fetch data (`from src.data.fetcher import get_stock_data`)
- [ ] Jupyter launches (`jupyter notebook`)
- [ ] Notebooks open successfully
- [ ] Tests pass (`python run_tests.py`)

---

**Installation complete!** Start exploring with `jupyter notebook` 🚀
