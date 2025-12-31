"""
Page: Watchlist Manager

Organize favorite stocks into custom watchlists for quick access and comparison.
Create multiple watchlists for different categories.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from src.data.fetcher import get_multiple_stocks


# User data directory setup
USER_DATA_DIR = Path("data/user_data")
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json_data(filename: str, default: dict) -> dict:
    """Load JSON data from user_data directory"""
    file_path = USER_DATA_DIR / filename
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading {filename}: {e}")
            return default
    return default


def save_json_data(filename: str, data: dict) -> bool:
    """Save JSON data to user_data directory"""
    file_path = USER_DATA_DIR / filename
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving {filename}: {e}")
        return False


def render():
    """Watchlist management page"""
    st.title("📋 Watchlist Manager")

    st.markdown("""
    Organize your favorite stocks into custom watchlists for quick access and comparison.
    Create multiple watchlists (e.g., "Tech Stocks", "Dividend Stocks", "Growth Stocks")
    """)

    watchlists = st.session_state['user_watchlists']['watchlists']

    # Create new watchlist
    st.markdown("### Create New Watchlist")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_name = st.text_input("Watchlist Name", key="new_watchlist", placeholder="e.g., Tech Giants")
    with col2:
        st.write("")  # Spacing
        if st.button("➕ Create", type="primary"):
            if new_name and new_name not in watchlists:
                watchlists[new_name] = []
                st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                save_json_data('watchlists.json', st.session_state['user_watchlists'])
                st.success(f"✅ Created watchlist: {new_name}")
                st.rerun()
            elif new_name in watchlists:
                st.error(f"Watchlist '{new_name}' already exists!")
            else:
                st.error("Please enter a watchlist name")

    st.markdown("---")

    # Display each watchlist in tabs
    if watchlists:
        st.markdown("### My Watchlists")
        tabs = st.tabs(list(watchlists.keys()))

        for i, (name, tickers) in enumerate(watchlists.items()):
            with tabs[i]:
                # Add ticker to this watchlist
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    new_ticker = st.text_input(
                        "Add Stock Ticker",
                        key=f"add_{name}",
                        placeholder="e.g., AAPL"
                    ).upper()
                with col2:
                    if st.button("➕ Add Stock", key=f"btn_{name}"):
                        if new_ticker and new_ticker not in tickers:
                            tickers.append(new_ticker)
                            st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                            save_json_data('watchlists.json', st.session_state['user_watchlists'])
                            st.success(f"Added {new_ticker}")
                            st.rerun()
                        elif new_ticker in tickers:
                            st.warning(f"{new_ticker} already in watchlist")
                        else:
                            st.error("Enter a ticker symbol")
                with col3:
                    if st.button("🗑️ Delete Watchlist", key=f"del_wl_{name}"):
                        del watchlists[name]
                        st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                        save_json_data('watchlists.json', st.session_state['user_watchlists'])
                        st.success(f"Deleted {name}")
                        st.rerun()

                # Show tickers
                if tickers:
                    st.markdown(f"**{len(tickers)} stocks in this watchlist:**")

                    # Fetch current prices
                    with st.spinner("Loading current prices..."):
                        try:
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=5)
                            stocks_data = get_multiple_stocks(
                                tickers,
                                start=start_date.strftime('%Y-%m-%d'),
                                end=end_date.strftime('%Y-%m-%d')
                            )

                            # Create summary table
                            summary = []
                            for ticker in tickers:
                                if ticker in stocks_data:
                                    df = stocks_data[ticker]
                                    if df is not None and len(df) > 0:
                                        current = df['Close'].iloc[-1]
                                        prev = df['Close'].iloc[-2] if len(df) > 1 else current
                                        change = current - prev
                                        change_pct = (change / prev) * 100
                                        volume = df['Volume'].iloc[-1]

                                        summary.append({
                                            'Ticker': ticker,
                                            'Price': f"${current:.2f}",
                                            'Change': f"${change:+.2f}",
                                            'Change %': f"{change_pct:+.2f}%",
                                            'Volume': f"{volume/1e6:.1f}M"
                                        })
                                    else:
                                        summary.append({
                                            'Ticker': ticker,
                                            'Price': 'N/A',
                                            'Change': 'N/A',
                                            'Change %': 'N/A',
                                            'Volume': 'N/A'
                                        })
                                else:
                                    summary.append({
                                        'Ticker': ticker,
                                        'Price': 'N/A',
                                        'Change': 'N/A',
                                        'Change %': 'N/A',
                                        'Volume': 'N/A'
                                    })

                            if summary:
                                df_summary = pd.DataFrame(summary)
                                st.dataframe(df_summary, use_container_width=True, hide_index=True)

                                # Quick actions
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"📊 Compare All {len(tickers)} Stocks", key=f"compare_{name}"):
                                        st.info("💡 Go to 'Stock Comparison' page to compare these stocks")
                                with col2:
                                    # Export watchlist
                                    csv = df_summary.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Export as CSV",
                                        data=csv,
                                        file_name=f"watchlist_{name.replace(' ', '_')}.csv",
                                        mime='text/csv',
                                        key=f"export_{name}"
                                    )

                        except Exception as e:
                            st.error(f"Error loading prices: {e}")

                    # Remove individual stocks
                    st.markdown("**Remove stocks:**")
                    cols = st.columns(min(len(tickers), 4))
                    for idx, ticker in enumerate(tickers):
                        with cols[idx % 4]:
                            if st.button(f"✖️ {ticker}", key=f"remove_{name}_{ticker}"):
                                tickers.remove(ticker)
                                st.session_state['user_watchlists']['last_updated'] = datetime.now().isoformat()
                                save_json_data('watchlists.json', st.session_state['user_watchlists'])
                                st.rerun()

                else:
                    st.info(f"This watchlist is empty. Add stocks using the form above.")

    else:
        st.info("📝 No watchlists yet. Create your first watchlist above!")
