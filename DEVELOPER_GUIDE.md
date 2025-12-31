# 🛠️ Developer Guide

Complete guide for developers working on the Stock Analysis Dashboard.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Adding a New Page](#adding-a-new-page)
3. [Using Utility Modules](#using-utility-modules)
4. [Code Style Guide](#code-style-guide)
5. [Testing](#testing)

## 🏗️ Architecture Overview

### Modular Design

The app follows a clean modular architecture:

```
app.py (Main Controller)
    ↓
src/pages/* (Page Modules)
    ↓
src/ui/* + src/data/* (Utilities)
    ↓
src/indicators/* + src/models/* (Business Logic)
```

### Key Principles

1. **Separation of Concerns** - Each page is self-contained
2. **DRY (Don't Repeat Yourself)** - Common functions in utilities
3. **Type Safety** - Type hints on all public functions
4. **Consistent UX** - Reusable chart and data loading functions

## 📄 Adding a New Page

### Step 1: Create the Page Module

Create `src/pages/my_new_page.py`:

```python
"""
Page: My New Feature

Description of what this page does.
"""

import streamlit as st
import pandas as pd
from typing import Optional
from src.data.loader import load_stock_with_spinner
from src.ui.charts import create_line_chart


def render() -> None:
    """Render the my new feature page"""

    st.title("🎯 My New Feature")

    # Your page logic here
    st.markdown("This is a new page!")

    # Example: Load stock data
    ticker = st.text_input("Enter ticker", value="AAPL")

    if st.button("Analyze"):
        df = load_stock_with_spinner(
            ticker=ticker,
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        if df is not None:
            # Create a chart
            fig = create_line_chart(
                df=df,
                y_column='Close',
                title=f'{ticker} Price',
                y_title='Price ($)'
            )
            st.plotly_chart(fig, width="stretch")
```

### Step 2: Register the Page

Add import to `src/pages/__init__.py`:

```python
from .my_new_page import render as render_my_new_page

__all__ = [
    # ... existing imports
    'render_my_new_page',
]
```

### Step 3: Add Navigation

In `app.py`, update the navigation list:

```python
page = st.radio(
    "Choose a page:",
    [
        "🏠 Home",
        # ... existing pages
        "🎯 My New Feature",  # Add here
    ],
    label_visibility="collapsed"
)
```

### Step 4: Add Routing

In `app.py`, update the `page_routes` dictionary:

```python
page_routes = {
    "🏠 Home": render_home,
    # ... existing routes
    "🎯 My New Feature": render_my_new_page,
}
```

### Done!

Your new page is now accessible from the sidebar navigation.

## 🔧 Using Utility Modules

### Chart Creation (`src/ui/charts.py`)

Available chart functions:

```python
from src.ui.charts import (
    create_candlestick_chart,
    create_line_chart,
    create_multi_line_chart,
    create_volume_chart,
    create_bar_chart,
    create_heatmap,
    create_pie_chart,
    create_scatter_with_fill
)

# Candlestick chart
fig = create_candlestick_chart(df, ticker="AAPL", height=500)
st.plotly_chart(fig, width="stretch")

# Line chart
fig = create_line_chart(
    df=df,
    y_column='Close',
    title='Stock Price',
    y_title='Price ($)',
    color='blue'
)

# Multi-line chart
fig = create_multi_line_chart(
    df=df,
    y_columns=['SMA_20', 'SMA_50'],
    title='Moving Averages',
    height=400
)

# Heatmap (for correlation)
fig = create_heatmap(
    data=correlation_matrix,
    title='Correlation Matrix',
    colorscale='RdYlGn'
)
```

### Data Loading (`src/data/loader.py`)

Standardized data loading functions:

```python
from src.data.loader import (
    load_stock_with_spinner,        # With loading feedback
    load_stock_quiet,                # Silent (no UI feedback)
    load_recent_stock,               # Last N days
    load_multiple_stocks_with_progress,
    get_date_range_from_period,
    save_to_session_state,
    get_from_session_state
)

# Load with user feedback
df = load_stock_with_spinner(
    ticker="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
    min_days=60  # Minimum required days
)

# Load last 365 days
df = load_recent_stock(ticker="AAPL", days=365)

# Load multiple stocks with progress bar
stocks_data = load_multiple_stocks_with_progress(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Convert period to date range
start_date, end_date = get_date_range_from_period("1 Year")

# Save to session state (for cross-page access)
save_to_session_state(ticker="AAPL", df=df)

# Retrieve from session state
ticker, df = get_from_session_state()
```

### Data Persistence (`src/data/persistence.py`)

Save and load user data:

```python
from src.data.persistence import (
    load_json_data,
    save_json_data,
    initialize_user_data
)

# Load data
watchlists = load_json_data('watchlists.json', default={'lists': []})

# Save data
save_json_data('watchlists.json', watchlists)

# Session state is automatically initialized on app start
# Access via:
st.session_state['user_watchlists']
st.session_state['user_alerts']
st.session_state['performance_data']
st.session_state['dark_mode']
```

## 📐 Code Style Guide

### Type Hints

All public functions should have type hints:

```python
from typing import Optional, List, Dict, Tuple
import pandas as pd

def my_function(
    ticker: str,
    start_date: str,
    min_days: int = 60
) -> Optional[pd.DataFrame]:
    """
    Function description.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        min_days: Minimum required days

    Returns:
        DataFrame or None if error
    """
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate key metrics from stock data.

    Args:
        df: DataFrame with OHLC columns

    Returns:
        Dictionary of metric name to value

    Example:
        >>> metrics = calculate_metrics(df)
        >>> print(metrics['volatility'])
        0.15
    """
    pass
```

### Error Handling

Use try-except with user-friendly messages:

```python
try:
    df = get_stock_data(ticker)
    if df is None or len(df) == 0:
        st.error(f"❌ No data found for {ticker}")
        return

except Exception as e:
    st.error(f"❌ Error loading {ticker}: {str(e)}")
    return
```

### Constants

Use uppercase for constants:

```python
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL"]
CACHE_DURATION = 3600  # 1 hour
MIN_DATA_POINTS = 60
```

## 🧪 Testing

### Manual Testing Checklist

When adding a new feature, test:

- ✅ Page loads without errors
- ✅ Invalid inputs handled gracefully
- ✅ Data loads successfully
- ✅ Charts render correctly
- ✅ Export/download features work
- ✅ Dark mode compatible
- ✅ Session state preserved across page navigation

### Example Test Cases

```python
# Test data loading with invalid ticker
df = load_stock_with_spinner(ticker="INVALID123", ...)
assert df is None  # Should return None

# Test date validation
valid, msg = validate_date_range("2024-01-01", "2024-12-31")
assert valid == True

# Test chart creation
fig = create_line_chart(df, y_column='Close', title='Test')
assert fig is not None
assert len(fig.data) > 0
```

## 🎨 UI Best Practices

### Layout Patterns

Use columns for side-by-side controls:

```python
col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Ticker")

with col2:
    period = st.selectbox("Period", ["1 Month", "1 Year"])

with col3:
    st.write("")  # Spacing
    if st.button("Analyze"):
        # Action
```

### Loading States

Always show loading feedback:

```python
with st.spinner("Loading data..."):
    df = get_stock_data(ticker)

st.success("✅ Data loaded successfully!")
```

### Error Messages

Use consistent emoji prefixes:

- ✅ Success: `st.success("✅ Operation successful")`
- ❌ Error: `st.error("❌ Something went wrong")`
- ⚠️ Warning: `st.warning("⚠️ Please note...")`
- 💡 Info: `st.info("💡 Tip: Try this...")`

### Charts

Always use `width="stretch"` for responsive charts:

```python
st.plotly_chart(fig, width="stretch")
```

## 🚀 Performance Tips

1. **Cache expensive operations**:
```python
@st.cache_data(ttl=3600)
def expensive_calculation(ticker):
    # Cached for 1 hour
    return result
```

2. **Lazy load data** - Only fetch when user clicks button
3. **Use spinners** for operations > 1 second
4. **Minimize session state** - Only store essential data

## 📝 Commit Guidelines

Use conventional commits:

```bash
feat: Add stock comparison page
fix: Resolve deprecation warnings
docs: Update developer guide
refactor: Extract chart utilities
test: Add data loader tests
chore: Update dependencies
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following this guide
4. Test thoroughly
5. Commit your changes
6. Push to your fork
7. Open a Pull Request

## 💬 Questions?

Open an issue on GitHub or check existing documentation in the codebase.

---

**Happy Coding! 🚀**
