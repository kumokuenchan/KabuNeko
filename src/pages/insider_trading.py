"""
Page: Insider Trading Tracker

Track insider buying and selling activity. When executives put their money where
their mouth is, it's often a strong signal for investors.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional
from src.data.insider_data import InsiderDataFetcher


def create_insider_timeline(df: pd.DataFrame) -> go.Figure:
    """Create timeline chart of insider transactions"""

    if df is None or df.empty:
        return go.Figure()

    # Find date column
    date_col = None
    for col in ['Start Date', 'Date', 'date', 'start_date']:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        return go.Figure()

    # Find transaction type column
    trans_col = None
    for col in ['Transaction', 'transaction', 'Type', 'type']:
        if col in df.columns:
            trans_col = col
            break

    if not trans_col:
        return go.Figure()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    # Categorize transactions
    df['Transaction_Type'] = 'Other'
    df.loc[df[trans_col].str.contains('Purchase|Buy', case=False, na=False), 'Transaction_Type'] = 'Buy'
    df.loc[df[trans_col].str.contains('Sale|Sell', case=False, na=False), 'Transaction_Type'] = 'Sell'

    # Create figure
    fig = go.Figure()

    # Add buys
    buys = df[df['Transaction_Type'] == 'Buy']
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys[date_col],
            y=[1] * len(buys),
            mode='markers',
            name='Insider Buys',
            marker=dict(size=12, color='#34C759', symbol='triangle-up'),
            text=buys.get('Insider', [''] * len(buys)),
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Buy Transaction<extra></extra>'
        ))

    # Add sells
    sells = df[df['Transaction_Type'] == 'Sell']
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells[date_col],
            y=[1] * len(sells),
            mode='markers',
            name='Insider Sells',
            marker=dict(size=12, color='#FF3B30', symbol='triangle-down'),
            text=sells.get('Insider', [''] * len(sells)),
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Sell Transaction<extra></extra>'
        ))

    fig.update_layout(
        title="Insider Transaction Timeline",
        xaxis_title="Date",
        yaxis_visible=False,
        template='plotly_white',
        height=300,
        showlegend=True,
        hovermode='closest'
    )

    return fig


def create_insider_summary_chart(analysis: dict) -> go.Figure:
    """Create bar chart showing buy vs sell activity"""

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Buys',
        x=['30 Days', '90 Days', '180 Days'],
        y=[
            analysis['30_days']['total_buys'],
            analysis['90_days']['total_buys'],
            analysis['180_days']['total_buys']
        ],
        marker_color='#34C759'
    ))

    fig.add_trace(go.Bar(
        name='Sells',
        x=['30 Days', '90 Days', '180 Days'],
        y=[
            analysis['30_days']['total_sells'],
            analysis['90_days']['total_sells'],
            analysis['180_days']['total_sells']
        ],
        marker_color='#FF3B30'
    ))

    fig.update_layout(
        title="Insider Trading Activity by Timeframe",
        barmode='group',
        template='plotly_white',
        height=400,
        xaxis_title="Period",
        yaxis_title="Number of Transactions"
    )

    return fig


def render() -> None:
    """Render the insider trading tracker page"""

    st.title("💼 Insider Trading Tracker")

    st.markdown("""
    Track when company insiders (executives, directors, major shareholders) buy or sell stock.
    **Insider buying is often bullish** - they're putting their money where their mouth is!
    """)

    # Stock input
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", placeholder="e.g., AAPL, TSLA, NVDA").upper()

    with col2:
        timeframe = st.selectbox("Analysis Period", ["30 Days", "90 Days", "180 Days"], index=1)

    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Track Insiders", type="primary")

    if analyze_btn:
        with st.spinner(f"Fetching insider trading data for {ticker}..."):

            # Get insider summary
            summary = InsiderDataFetcher.get_insider_summary(ticker)

            # Get detailed transactions
            timeframe_map = {
                "30 Days": '30_days',
                "90 Days": '90_days',
                "180 Days": '180_days'
            }

            current_analysis = summary[timeframe_map[timeframe]]

            if current_analysis['total_transactions'] == 0:
                st.warning(f"⚠️ No insider transactions found for **{ticker}** in the selected period")
                st.info("""
                **This could mean:**
                - No insider activity in this timeframe
                - Data not available for this ticker
                - The company may have trading blackout periods

                **Try:**
                - A different timeframe (e.g., 180 days)
                - A well-known ticker (AAPL, MSFT, GOOGL, META)
                - Check if this is a valid US stock ticker
                """)
                return

            st.success(f"✅ Found {current_analysis['total_transactions']} insider transactions for {ticker}")

            # === INSIDER SIGNAL ===
            st.markdown("### 🎯 Insider Trading Signal")

            col1, col2 = st.columns([1, 2])

            with col1:
                # Signal indicator
                signal_color = '#34C759' if 'Buy' in current_analysis['signal'] else '#FF3B30' if 'Sell' in current_analysis['signal'] else '#8E8E93'

                st.markdown(f"""
                <div class="metric-card" style="border-left: 6px solid {signal_color}; text-align: center;">
                    <h1 style="font-size: 3rem; margin: 0;">{current_analysis['signal_emoji']}</h1>
                    <h2 style="margin: 0.5rem 0; color: {signal_color};">{current_analysis['signal']}</h2>
                    <p style="color: #86868B; margin: 0;">Based on {timeframe} activity</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Key metrics
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("📈 Insider Buys", current_analysis['total_buys'])

                with col_b:
                    st.metric("📉 Insider Sells", current_analysis['total_sells'])

                with col_c:
                    net = current_analysis['net_activity']
                    st.metric("⚖️ Net Activity", f"{net:+d}",
                             delta_color="normal" if net > 0 else "inverse")

                st.markdown("---")

                # Interpretation
                if current_analysis['net_activity'] > 3:
                    st.success(f"""
                    **🟢 Bullish Signal**: Insiders are **buying heavily** ({current_analysis['total_buys']} buys vs {current_analysis['total_sells']} sells).
                    When executives buy their own stock, they often know something positive about the company's future.
                    """)
                elif current_analysis['net_activity'] < -3:
                    st.error(f"""
                    **🔴 Bearish Signal**: Insiders are **selling heavily** ({current_analysis['total_sells']} sells vs {current_analysis['total_buys']} buys).
                    Heavy insider selling can indicate concerns about valuation or future prospects.
                    """)
                else:
                    st.info(f"""
                    **⚪ Neutral Signal**: Insider activity is balanced ({current_analysis['total_buys']} buys, {current_analysis['total_sells']} sells).
                    No clear directional signal from insider trading.
                    """)

            # === MULTI-TIMEFRAME ANALYSIS ===
            st.markdown("### 📊 Multi-Timeframe Analysis")

            fig_summary = create_insider_summary_chart(summary)
            st.plotly_chart(fig_summary, use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 30 Days")
                net_30 = summary['30_days']['net_activity']
                st.metric("Net Activity", f"{net_30:+d}")
                st.caption(f"{summary['30_days']['signal_emoji']} {summary['30_days']['signal']}")

            with col2:
                st.markdown("#### 90 Days")
                net_90 = summary['90_days']['net_activity']
                st.metric("Net Activity", f"{net_90:+d}")
                st.caption(f"{summary['90_days']['signal_emoji']} {summary['90_days']['signal']}")

            with col3:
                st.markdown("#### 180 Days")
                net_180 = summary['180_days']['net_activity']
                st.metric("Net Activity", f"{net_180:+d}")
                st.caption(f"{summary['180_days']['signal_emoji']} {summary['180_days']['signal']}")

            # === TRANSACTION TIMELINE ===
            st.markdown("### 📅 Transaction Timeline")

            if current_analysis['dataframe'] is not None and not current_analysis['dataframe'].empty:
                timeline_fig = create_insider_timeline(current_analysis['dataframe'])
                st.plotly_chart(timeline_fig, use_container_width=True)

            # === RECENT TRANSACTIONS TABLE ===
            st.markdown(f"### 📋 Recent Insider Transactions ({timeframe})")

            # Transaction type legend
            col_legend1, col_legend2, col_legend3, col_legend4, col_legend5 = st.columns(5)
            with col_legend1:
                st.markdown("🟢 **Buy** - Purchase")
            with col_legend2:
                st.markdown("🔴 **Sell** - Sale")
            with col_legend3:
                st.markdown("🏆 **Award** - Stock grant")
            with col_legend4:
                st.markdown("🎁 **Gift** - Stock gift")
            with col_legend5:
                st.markdown("⚙️ **Option** - Exercise")

            if current_analysis['dataframe'] is not None and not current_analysis['dataframe'].empty:
                df_display = current_analysis['dataframe'].copy()

                # Find transaction type column - check both Transaction and Text columns
                trans_col = None
                text_col = None

                for col in ['Transaction', 'transaction', 'Type', 'type']:
                    if col in df_display.columns:
                        trans_col = col
                        break

                for col in ['Text', 'text', 'Description', 'description']:
                    if col in df_display.columns:
                        text_col = col
                        break

                # Add visual indicators for buy/sell
                def categorize_transaction(row):
                    # Check Transaction column first (if it's not empty)
                    if trans_col and not pd.isna(row.get(trans_col)):
                        trans_str = str(row[trans_col]).lower().strip()
                        if trans_str:  # Only process if not empty string
                            if 'purchase' in trans_str or 'buy' in trans_str or 'acquisition' in trans_str:
                                return '🟢 Buy'
                            elif 'sale' in trans_str or 'sell' in trans_str or 'disposition' in trans_str:
                                return '🔴 Sell'
                            elif 'award' in trans_str or 'grant' in trans_str:
                                return '🏆 Award'
                            elif 'gift' in trans_str:
                                return '🎁 Gift'
                            elif 'conversion' in trans_str or 'exercise' in trans_str:
                                return '⚙️ Exercise'

                    # Check Text column if Transaction is empty or didn't match
                    if text_col and not pd.isna(row.get(text_col)):
                        text_str = str(row[text_col]).lower()
                        if 'purchase' in text_str or 'buy' in text_str or 'acquisition' in text_str:
                            return '🟢 Buy'
                        elif 'sale' in text_str or 'sell' in text_str:
                            return '🔴 Sell'
                        elif 'award' in text_str or 'grant' in text_str:
                            return '🏆 Award'
                        elif 'gift' in text_str:
                            return '🎁 Gift'
                        elif 'conversion' in text_str or 'exercise' in text_str or 'option' in text_str:
                            return '⚙️ Exercise'

                    return '⚪ Other'

                df_display['Type'] = df_display.apply(categorize_transaction, axis=1)

                # Select relevant columns for display
                display_cols = []

                # Date column
                for col in ['Start Date', 'Date', 'date']:
                    if col in df_display.columns:
                        display_cols.append(col)
                        break

                # Add Type column (our new visual indicator)
                if 'Type' in df_display.columns:
                    display_cols.append('Type')

                # Insider name
                for col in ['Insider', 'insider', 'Name', 'name']:
                    if col in df_display.columns:
                        display_cols.append(col)
                        break

                # Transaction details (skip original transaction column if we have Type)
                if trans_col and trans_col != 'Type' and trans_col not in display_cols:
                    display_cols.append(trans_col)

                # Shares
                for col in ['Shares', 'shares', 'Amount', 'amount']:
                    if col in df_display.columns and col not in display_cols:
                        display_cols.append(col)
                        break

                # Value
                for col in ['Value', 'value']:
                    if col in df_display.columns and col not in display_cols:
                        display_cols.append(col)
                        break

                if display_cols:
                    df_display = df_display[display_cols].head(20)

                    # Show summary of visible transactions
                    if 'Type' in df_display.columns:
                        buy_count = (df_display['Type'] == '🟢 Buy').sum()
                        sell_count = (df_display['Type'] == '🔴 Sell').sum()
                        award_count = (df_display['Type'] == '🏆 Award').sum()
                        gift_count = (df_display['Type'] == '🎁 Gift').sum()
                        option_count = (df_display['Type'] == '⚙️ Option').sum()
                        other_count = (df_display['Type'] == '⚪ Other').sum()

                        summary_parts = [
                            f"<strong style='color: #34C759;'>🟢 {buy_count} Buys</strong>" if buy_count > 0 else "",
                            f"<strong style='color: #FF3B30;'>🔴 {sell_count} Sells</strong>" if sell_count > 0 else "",
                            f"<strong style='color: #FFD700;'>🏆 {award_count} Awards</strong>" if award_count > 0 else "",
                            f"<strong style='color: #FF9500;'>🎁 {gift_count} Gifts</strong>" if gift_count > 0 else "",
                            f"<strong style='color: #007AFF;'>⚙️ {option_count} Options</strong>" if option_count > 0 else "",
                            f"<strong style='color: #8E8E93;'>⚪ {other_count} Other</strong>" if other_count > 0 else ""
                        ]
                        summary_text = " | ".join([p for p in summary_parts if p])

                        st.markdown(f"""
                        <div style="background: #F5F5F7; padding: 1rem; border-radius: 12px; margin: 1rem 0;">
                            Showing {len(df_display)} transactions: {summary_text}
                        </div>
                        """, unsafe_allow_html=True)

                    # Display with custom HTML styling
                    st.markdown("""
                    <style>
                    .transaction-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 1rem 0;
                    }
                    .transaction-table th {
                        background: #F5F5F7;
                        padding: 0.75rem;
                        text-align: left;
                        font-weight: 600;
                        border-bottom: 2px solid #E5E5E7;
                    }
                    .transaction-table td {
                        padding: 0.75rem;
                        border-bottom: 1px solid #E5E5E7;
                    }
                    .transaction-table tr:hover {
                        background: #F5F5F7;
                    }
                    .buy-badge {
                        background: #E8F5E9;
                        color: #34C759;
                        padding: 0.25rem 0.75rem;
                        border-radius: 12px;
                        font-weight: 600;
                    }
                    .sell-badge {
                        background: #FFE5E5;
                        color: #FF3B30;
                        padding: 0.25rem 0.75rem;
                        border-radius: 12px;
                        font-weight: 600;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download button
                    csv = df_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Transactions (CSV)",
                        data=csv,
                        file_name=f"{ticker}_insider_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No transaction details available")

            # === INSIGHTS ===
            st.markdown("### 💡 Insider Trading Insights")

            st.info("""
            **📚 Understanding Insider Trading:**

            **Why Insider Trading Matters:**
            - Insiders have better knowledge of company prospects
            - Insider buying often precedes positive news
            - Multiple insiders buying is a stronger signal than one

            **Key Patterns:**
            - **🟢 Cluster Buying**: Multiple insiders buying within short period = Very bullish
            - **🟢 CEO/Director Buys**: Top executives buying = Strong confidence signal
            - **🔴 Heavy Selling**: Large insider sales might indicate overvaluation
            - **⚪ Planned Sales**: Many sells are pre-scheduled (10b5-1 plans) and less meaningful

            **What to Watch:**
            - Size of transactions (bigger = more significant)
            - Frequency (repeated transactions = stronger signal)
            - Who is trading (CEO/CFO more important than other insiders)
            - Market context (buying during market dips can be extra bullish)
            """)

            # === DISCLAIMER ===
            st.warning("""
            ⚠️ **Legal Disclaimer:**
            - Insider trading data is publicly disclosed through SEC Form 4 filings
            - Insiders must file within 2 business days of transaction
            - Some transactions may be delayed in appearing
            - This data is for informational purposes only and not investment advice
            - Always combine insider data with other analysis before making decisions
            """)

    else:
        # Show info when no analysis yet
        st.info("👆 Enter a stock ticker and click 'Track Insiders' to get started!")

        st.markdown("""
        ### 🎯 What This Tool Does:

        - **📊 Track Insider Activity** - See when executives buy or sell their stock
        - **🎯 Generate Signals** - Automated buy/sell signals based on insider activity
        - **📈 Multi-Timeframe Analysis** - View activity across 30, 90, and 180-day periods
        - **📅 Transaction Timeline** - Visual timeline of all insider trades
        - **📋 Detailed Reports** - Full transaction history with dates, amounts, insiders

        ### 🟢 Why Insider Buying is Bullish:

        1. **Information Advantage** - Insiders know the company better than anyone
        2. **Skin in the Game** - They're risking their own money
        3. **Confidence Signal** - They believe stock is undervalued
        4. **Leading Indicator** - Often precedes positive news or earnings

        ### 🔴 Why Insider Selling Doesn't Always Mean Bearish:

        - Many sells are pre-planned (10b5-1 plans) for diversification
        - Insiders may sell for personal reasons (taxes, house, etc.)
        - Only heavy cluster selling by multiple insiders is concerning
        - Focus more on **buying** than selling as a signal

        ### 💡 Best Tickers to Track:

        - **AAPL** (Apple) - High insider activity
        - **MSFT** (Microsoft) - Frequent transactions
        - **GOOGL** (Alphabet) - Good data availability
        - **META** (Meta) - Active insider trading
        """)
