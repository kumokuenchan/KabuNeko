"""
Page: Performance Tracker

Track paper trading performance based on recommendations. Log entry and exit points,
and analyze trading results with P&L tracking.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path


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
    """Track paper trading performance based on recommendations"""
    st.title("💹 Performance Tracker")

    st.markdown("""
    Track your paper trading performance based on Investment Advice recommendations.
    Log entry and exit points, and analyze your trading results.
    """)

    performance_data = st.session_state['performance_data']
    trades = performance_data.get('trades', [])

    # Add new trade section
    st.markdown("### ➕ Log New Trade")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        trade_ticker = st.text_input("Ticker", key="trade_ticker").upper()
    with col2:
        trade_type = st.selectbox("Type", ["BUY", "SELL"], key="trade_type")
    with col3:
        trade_price = st.number_input("Price", min_value=0.0, value=100.0, step=0.01, key="trade_price")
    with col4:
        trade_shares = st.number_input("Shares", min_value=1, value=100, step=1, key="trade_shares")
    with col5:
        st.write("")  # Spacing
        st.write("")  # More spacing
        if st.button("Log Trade", type="primary"):
            if not trade_ticker:
                st.error("Please enter a ticker symbol")
            else:
                new_trade = {
                    'id': f"trade_{len(trades) + 1}_{int(datetime.now().timestamp())}",
                    'ticker': trade_ticker,
                    'type': trade_type,
                    'price': trade_price,
                    'shares': trade_shares,
                    'total_value': trade_price * trade_shares,
                    'date': datetime.now().isoformat(),
                    'status': 'OPEN' if trade_type == 'BUY' else 'CLOSED'
                }

                # If SELL, try to match with open BUY
                if trade_type == 'SELL':
                    open_buys = [t for t in trades if t['ticker'] == trade_ticker and t['type'] == 'BUY' and t.get('status') == 'OPEN']
                    if open_buys:
                        # Match with first open buy
                        buy_trade = open_buys[0]
                        buy_trade['status'] = 'CLOSED'
                        buy_trade['exit_price'] = trade_price
                        buy_trade['exit_date'] = datetime.now().isoformat()

                        # Calculate P&L
                        profit_loss = (trade_price - buy_trade['price']) * min(trade_shares, buy_trade['shares'])
                        profit_loss_pct = ((trade_price - buy_trade['price']) / buy_trade['price']) * 100

                        buy_trade['profit_loss'] = profit_loss
                        buy_trade['profit_loss_pct'] = profit_loss_pct
                        new_trade['matched_buy'] = buy_trade['id']
                        new_trade['profit_loss'] = profit_loss
                        new_trade['profit_loss_pct'] = profit_loss_pct

                trades.append(new_trade)
                performance_data['trades'] = trades
                save_json_data('performance_tracker.json', performance_data)
                st.success(f"✅ {trade_type} trade logged for {trade_ticker}")
                st.rerun()

    st.markdown("---")

    if not trades:
        st.info("📊 No trades logged yet. Start tracking your performance by logging trades above!")
        return

    # Performance Statistics
    st.markdown("### 📈 Performance Statistics")

    # Calculate statistics
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED' and t.get('profit_loss') is not None]

    if closed_trades:
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['profit_loss'] > 0]
        losing_trades = [t for t in closed_trades if t['profit_loss'] < 0]

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(t['profit_loss'] for t in winning_trades)
        total_loss = sum(t['profit_loss'] for t in losing_trades)
        net_profit = total_profit + total_loss

        avg_win = (total_profit / len(winning_trades)) if winning_trades else 0
        avg_loss = (total_loss / len(losing_trades)) if losing_trades else 0

        # Display stats in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with col2:
            st.metric("Winning Trades", len(winning_trades))
            st.metric("Losing Trades", len(losing_trades))

        with col3:
            profit_color = "normal" if net_profit >= 0 else "inverse"
            st.metric("Net P&L", f"${net_profit:,.2f}", delta=f"{net_profit:+,.2f}")
            st.metric("Total Profit", f"${total_profit:,.2f}")

        with col4:
            st.metric("Total Loss", f"${total_loss:,.2f}")
            st.metric("Avg Win/Loss", f"${avg_win:.2f} / ${avg_loss:.2f}")

        # P&L Chart
        st.markdown("### 📊 Cumulative P&L")

        # Sort by date
        sorted_trades = sorted(closed_trades, key=lambda x: x.get('exit_date', x['date']))

        cumulative_pl = 0
        cumulative_data = []
        dates = []

        for trade in sorted_trades:
            cumulative_pl += trade['profit_loss']
            cumulative_data.append(cumulative_pl)
            try:
                trade_date = datetime.fromisoformat(trade.get('exit_date', trade['date']))
                dates.append(trade_date.strftime('%Y-%m-%d'))
            except:
                dates.append('N/A')

        fig_pl = go.Figure()
        fig_pl.add_trace(go.Scatter(
            x=dates,
            y=cumulative_data,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='green' if cumulative_pl >= 0 else 'red', width=3),
            fill='tozeroy'
        ))

        fig_pl.update_layout(
            xaxis_title='Date',
            yaxis_title='Cumulative P&L ($)',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )

        st.plotly_chart(fig_pl, use_container_width=True)

        # Win/Loss Distribution
        st.markdown("### 📊 Win/Loss Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # Win/Loss pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[len(winning_trades), len(losing_trades)],
                marker=dict(colors=['#00cc66', '#ff4444']),
                hole=0.4
            )])
            fig_pie.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # P&L by ticker
            ticker_pl = {}
            for trade in closed_trades:
                ticker = trade['ticker']
                if ticker not in ticker_pl:
                    ticker_pl[ticker] = 0
                ticker_pl[ticker] += trade['profit_loss']

            fig_bar = go.Figure(data=[go.Bar(
                x=list(ticker_pl.keys()),
                y=list(ticker_pl.values()),
                marker=dict(color=['green' if v >= 0 else 'red' for v in ticker_pl.values()])
            )])
            fig_bar.update_layout(
                xaxis_title='Ticker',
                yaxis_title='P&L ($)',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("💡 Close some trades (log SELL orders) to see performance statistics")

    # Trade History
    st.markdown("---")
    st.markdown("### 📋 Trade History")

    # Display all trades in table
    trade_display = []
    for trade in reversed(trades):  # Most recent first
        try:
            trade_date = datetime.fromisoformat(trade['date']).strftime('%Y-%m-%d %H:%M')
        except:
            trade_date = trade['date']

        display_row = {
            'Date': trade_date,
            'Ticker': trade['ticker'],
            'Type': trade['type'],
            'Price': f"${trade['price']:.2f}",
            'Shares': trade['shares'],
            'Total': f"${trade['total_value']:.2f}",
            'Status': trade.get('status', 'N/A')
        }

        if trade.get('profit_loss') is not None:
            display_row['P&L'] = f"${trade['profit_loss']:+,.2f} ({trade['profit_loss_pct']:+.2f}%)"

        trade_display.append(display_row)

    df_trades = pd.DataFrame(trade_display)
    st.dataframe(df_trades, use_container_width=True, hide_index=True)

    # Export trades
    if trades:
        csv = df_trades.to_csv(index=False)
        st.download_button(
            label="📥 Export Trades to CSV",
            data=csv,
            file_name=f"performance_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

    # Clear all trades button
    st.markdown("---")
    if st.button("🗑️ Clear All Trades", type="secondary"):
        if st.checkbox("⚠️ Confirm deletion of all trades"):
            performance_data['trades'] = []
            save_json_data('performance_tracker.json', performance_data)
            st.success("All trades cleared")
            st.rerun()
