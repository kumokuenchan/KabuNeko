# Testing Guide

This document provides comprehensive information about testing the Stock Analysis Toolkit.

## Table of Contents

1. [Setup](#setup)
2. [Running Tests](#running-tests)
3. [Test Structure](#test-structure)
4. [Test Coverage](#test-coverage)
5. [Writing Tests](#writing-tests)
6. [Continuous Integration](#continuous-integration)

---

## Setup

### Prerequisites

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Install project dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run all tests
python run_tests.py

# Or use pytest directly
python -m pytest tests/ -v
```

---

## Running Tests

### Using the Test Runner Script

The `run_tests.py` script provides an easy interface for running tests:

```bash
# Run all tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file test_data.py

# Run tests matching a pattern
python run_tests.py --test sma

# Stop on first failure
python run_tests.py -x

# Show print statements
python run_tests.py -s

# Extra verbose
python run_tests.py --verbose
```

### Using pytest Directly

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data.py -v

# Run specific test class
python -m pytest tests/test_indicators.py::TestTrendIndicators -v

# Run specific test
python -m pytest tests/test_indicators.py::TestTrendIndicators::test_sma_calculation -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Stop on first failure
python -m pytest tests/ -x

# Show print output
python -m pytest tests/ -s

# Run in parallel (requires pytest-xdist)
python -m pytest tests/ -n auto
```

---

## Test Structure

### Test Files

```
tests/
├── __init__.py                # Test package initialization
├── test_data.py               # Data fetching and preprocessing tests
├── test_indicators.py         # Technical indicators tests
├── test_models.py             # ML models tests
└── test_backtesting.py        # Backtesting framework tests
```

### Test Organization

Each test file contains:

1. **Imports** - Required modules and fixtures
2. **Test Classes** - Organized by functionality
3. **Fixtures** - Reusable test data using `@pytest.fixture`
4. **Test Methods** - Individual test cases with `test_` prefix

Example structure:

```python
class TestFeatureName:
    """Tests for specific feature"""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        return create_sample_data()

    def test_basic_functionality(self, sample_data):
        """Test basic functionality"""
        result = function_under_test(sample_data)
        assert result is not None

    def test_edge_case(self, sample_data):
        """Test edge case"""
        # Test implementation
        pass
```

---

## Test Coverage

### Current Test Coverage

The test suite covers:

✅ **Data Module (test_data.py)**
- Stock data fetching (single and multiple stocks)
- Invalid ticker handling
- Data date range validation
- Data cleaning and preprocessing
- Returns calculation
- Normalization
- Train/test splitting
- Sequence creation for LSTM

✅ **Indicators Module (test_indicators.py)**
- Trend indicators (SMA, EMA, MACD, ADX)
- Momentum indicators (RSI, Stochastic, Williams %R, ROC)
- Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
- Volume indicators (OBV, VWAP, A/D Line)
- Signal detection

✅ **Models Module (test_models.py)**
- Feature engineering (lagged, rolling, technical features)
- ML dataset preparation
- Random Forest model (initialization, fitting, prediction, evaluation)
- Feature importance extraction
- Model evaluation metrics

✅ **Backtesting Module (test_backtesting.py)**
- Trading strategies (SMA Crossover, RSI Mean Reversion)
- Performance metrics (Sharpe, Sortino, Max Drawdown, Calmar)
- Win rate and profit factor calculations
- Risk-adjusted returns

### Generating Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows

# Terminal coverage report
python -m pytest tests/ --cov=src --cov-report=term

# Show missing lines
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Writing Tests

### Best Practices

1. **Use Descriptive Names**
   ```python
   def test_sma_returns_correct_values():
       # Good: Clear what is being tested
       pass

   def test_1():
       # Bad: Unclear purpose
       pass
   ```

2. **One Assertion Per Test** (when possible)
   ```python
   def test_data_has_required_columns():
       df = get_stock_data('AAPL')
       assert 'Close' in df.columns

   def test_data_has_datetime_index():
       df = get_stock_data('AAPL')
       assert isinstance(df.index, pd.DatetimeIndex)
   ```

3. **Use Fixtures for Common Data**
   ```python
   @pytest.fixture
   def sample_stock_data():
       dates = pd.date_range('2023-01-01', periods=100)
       return pd.DataFrame({'Close': range(100)}, index=dates)

   def test_something(sample_stock_data):
       result = process_data(sample_stock_data)
       assert result is not None
   ```

4. **Test Edge Cases**
   ```python
   def test_empty_dataframe():
       df = pd.DataFrame()
       result = calculate_sma(df, 20)
       assert result.empty

   def test_single_row():
       df = pd.DataFrame({'Close': [100]})
       result = calculate_sma(df, 20)
       assert result.isna().all()
   ```

5. **Test Error Handling**
   ```python
   def test_invalid_ticker_raises_error():
       with pytest.raises(ValueError):
           get_stock_data('INVALID999')
   ```

### Adding New Tests

To add a new test:

1. Choose appropriate test file or create new one
2. Create test class if needed
3. Add fixture for test data if reusable
4. Write test method with `test_` prefix
5. Use descriptive assertion messages

Example:

```python
def test_new_feature():
    """Test new feature description"""
    # Arrange
    data = create_test_data()

    # Act
    result = new_feature(data)

    # Assert
    assert result is not None, "Result should not be None"
    assert len(result) > 0, "Result should not be empty"
```

---

## Verifying Notebooks

All Jupyter notebooks can be verified for structural validity:

```bash
# Verify all notebooks
python verify_notebooks.py
```

This checks:
- Valid JSON structure
- Required notebook fields
- Cell count and types

To actually run notebooks (requires kernel):

```bash
# Install jupyter
pip install jupyter

# Run notebook
jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb
```

---

## Continuous Integration

### GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=src

    - name: Verify notebooks
      run: |
        python verify_notebooks.py
```

---

## Troubleshooting

### Common Issues

**Issue: Module not found**
```bash
# Solution: Add project to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
python -m pytest tests/
```

**Issue: Network tests failing**
```bash
# Solution: Skip network-dependent tests
python -m pytest tests/ -m "not requires_network"
```

**Issue: Slow tests**
```bash
# Solution: Run fast tests only
python run_tests.py --fast
```

**Issue: Import errors in tests**
```bash
# Solution: Install package in development mode
pip install -e .
```

---

## Testing Checklist

Before committing code:

- [ ] All tests pass
- [ ] New features have tests
- [ ] Edge cases are tested
- [ ] Coverage hasn't decreased
- [ ] Notebooks are valid
- [ ] Documentation updated

---

## Test Metrics

Current test statistics (as of project completion):

- **Total Test Files**: 4
- **Total Test Cases**: 50+
- **Notebooks Verified**: 8/8 ✅
- **Test Coverage**: ~75% (core modules)

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

## Support

For testing issues:
1. Check this documentation
2. Review existing test examples
3. Consult pytest documentation
4. Open an issue on GitHub

---

**Happy Testing!** 🧪
