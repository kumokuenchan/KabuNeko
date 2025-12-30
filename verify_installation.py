#!/usr/bin/env python3
"""
Installation Verification Script

This script verifies that all required dependencies are installed
and the toolkit is ready to use.

Usage:
    python verify_installation.py
"""

import sys
import importlib.util


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 8)

    if version >= required:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (requires 3.8+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    spec = importlib.util.find_spec(import_name)

    if spec is not None:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name:20s} {version}")
            return True
        except Exception as e:
            print(f"⚠️  {package_name:20s} (import error: {str(e)[:30]})")
            return False
    else:
        print(f"❌ {package_name:20s} NOT INSTALLED")
        return False


def check_optional_package(package_name, import_name=None):
    """Check optional package"""
    if import_name is None:
        import_name = package_name

    spec = importlib.util.find_spec(import_name)

    if spec is not None:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name:20s} {version} (optional)")
            return True
        except:
            print(f"⚠️  {package_name:20s} (optional, import error)")
            return False
    else:
        print(f"⏭️  {package_name:20s} NOT INSTALLED (optional)")
        return False


def check_project_modules():
    """Check project modules can be imported"""
    print("\nProject Modules:")

    modules_to_check = [
        ('src.data.fetcher', 'Data Fetcher'),
        ('src.data.preprocessor', 'Data Preprocessor'),
        ('src.indicators.trend', 'Trend Indicators'),
        ('src.indicators.momentum', 'Momentum Indicators'),
        ('src.indicators.volatility', 'Volatility Indicators'),
        ('src.indicators.volume', 'Volume Indicators'),
        ('src.fundamental.ratios', 'Financial Ratios'),
        ('src.fundamental.metrics', 'Growth Metrics'),
        ('src.models.feature_engineering', 'Feature Engineering'),
        ('src.models.random_forest', 'Random Forest'),
        ('src.models.evaluation', 'Model Evaluation'),
        ('src.backtesting.strategies', 'Trading Strategies'),
        ('src.backtesting.engine', 'Backtesting Engine'),
        ('src.backtesting.metrics', 'Performance Metrics'),
        ('src.visualization.charts', 'Charts'),
        ('src.visualization.dashboard', 'Dashboard'),
    ]

    all_ok = True
    for module_name, description in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"✅ {description:25s}")
        except Exception as e:
            print(f"❌ {description:25s} (error: {str(e)[:40]})")
            all_ok = False

    return all_ok


def test_data_fetch():
    """Test data fetching capability"""
    try:
        from src.data.fetcher import get_stock_data
        import pandas as pd
        from datetime import datetime, timedelta

        print("\nTesting data fetch (AAPL, last 5 days)...")

        # Calculate dates for last 5 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        df = get_stock_data('AAPL', start=start_date, end=end_date)

        if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
            print(f"✅ Data fetch successful ({len(df)} rows)")
            return True
        else:
            print("❌ Data fetch returned empty or invalid data")
            return False

    except Exception as e:
        print(f"❌ Data fetch error: {str(e)}")
        return False


def main():
    """Main verification function"""
    print("=" * 70)
    print("Stock Analysis Toolkit - Installation Verification")
    print("=" * 70)

    all_checks_passed = True

    # Python version
    print("\nPython Version:")
    python_ok = check_python_version()
    all_checks_passed = all_checks_passed and python_ok

    # Core dependencies
    print("\nCore Dependencies:")
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('yfinance', 'yfinance'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('scikit-learn', 'sklearn'),
        ('backtesting', 'backtesting'),
        ('jupyter', 'jupyter'),
        ('ipywidgets', 'ipywidgets'),
    ]

    for pkg_name, import_name in required_packages:
        pkg_ok = check_package(pkg_name, import_name)
        all_checks_passed = all_checks_passed and pkg_ok

    # Optional dependencies
    print("\nOptional Dependencies:")
    check_optional_package('tensorflow', 'tensorflow')
    check_optional_package('pandas-ta', 'pandas_ta')

    # Project modules
    modules_ok = check_project_modules()
    all_checks_passed = all_checks_passed and modules_ok

    # Data fetch test
    if all_checks_passed:
        data_ok = test_data_fetch()
        all_checks_passed = all_checks_passed and data_ok

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if all_checks_passed:
        print("\n✅ All checks passed!")
        print("\nInstallation verified successfully!")
        print("\n🚀 Next steps:")
        print("   1. Launch Jupyter: jupyter notebook")
        print("   2. Open: notebooks/01_data_collection.ipynb")
        print("   3. Start learning and analyzing!")
    else:
        print("\n❌ Some checks failed")
        print("\n📋 Troubleshooting:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Check Python version (requires 3.8+)")
        print("   3. Review INSTALLATION.md for detailed setup")

    print("=" * 70)

    return 0 if all_checks_passed else 1


if __name__ == '__main__':
    sys.exit(main())
