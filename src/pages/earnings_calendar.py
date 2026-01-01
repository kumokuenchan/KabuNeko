"""
Page: Earnings Calendar & Analysis

Track upcoming earnings dates and analyze historical earnings performance.
Earnings reports are critical events that can cause significant stock price movements.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional
from src.data.earnings_data import EarningsDataFetcher


def create_earnings_surprise_chart(surprises: list) -> go.Figure:
    """Create bar chart showing earnings surprises"""

    if not surprises:
        return go.Figure()

    df = pd.DataFrame(surprises)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # Color based on beat/miss
    colors = ['#34C759' if s > 0 else '#FF3B30' if s < 0 else '#8E8E93' for s in df['surprise_pct']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['date_str'],
        y=df['surprise_pct'],
        marker_color=colors,
        text=[f"{s:+.1f}%" for s in df['surprise_pct']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Surprise: %{y:.2f}%<br><extra></extra>'
    ))

    fig.update_layout(
        title="Earnings Surprise History (Actual vs Estimate)",
        xaxis_title="Earnings Date",
        yaxis_title="Surprise %",
        template='plotly_white',
        height=400,
        showlegend=False,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#E5E5E7')
    )

    return fig


def create_eps_trend_chart(earnings_history: pd.DataFrame) -> go.Figure:
    """Create line chart showing EPS trend over time"""

    if earnings_history is None or earnings_history.empty:
        return go.Figure()

    df = earnings_history.copy()

    # Ensure we have Revenue and Earnings columns
    if 'Revenue' not in df.columns and 'Earnings' not in df.columns:
        return go.Figure()

    # Sort by date (index)
    df = df.sort_index()

    fig = go.Figure()

    # Add EPS trace if available
    if 'Earnings' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Earnings'],
            mode='lines+markers',
            name='Earnings (EPS)',
            line=dict(color='#007AFF', width=3),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title="Quarterly Earnings Per Share (EPS) Trend",
        xaxis_title="Quarter",
        yaxis_title="EPS ($)",
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )

    return fig


def create_revenue_chart(earnings_history: pd.DataFrame) -> go.Figure:
    """Create bar chart showing revenue trend"""

    if earnings_history is None or earnings_history.empty:
        return go.Figure()

    df = earnings_history.copy()

    if 'Revenue' not in df.columns:
        return go.Figure()

    # Sort by date
    df = df.sort_index()

    # Convert revenue to billions for better readability
    df['Revenue_B'] = df['Revenue'] / 1_000_000_000

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Revenue_B'],
        marker_color='#34C759',
        text=[f"${r:.2f}B" for r in df['Revenue_B']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:.2f}B<br><extra></extra>'
    ))

    fig.update_layout(
        title="Quarterly Revenue Trend",
        xaxis_title="Quarter",
        yaxis_title="Revenue (Billions $)",
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def render() -> None:
    """Render the earnings calendar & analysis page"""

    st.title("📊 Earnings Calendar & Analysis")

    st.markdown("""
    Track upcoming earnings dates and analyze historical earnings performance.
    **Earnings reports can cause major price swings** - know when they're coming!
    """)

    # Stock input
    col1, col2 = st.columns([3, 1])

    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").upper()

    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("📊 Analyze Earnings", type="primary")

    if analyze_btn:
        with st.spinner(f"Fetching earnings data for {ticker}..."):

            # Get comprehensive earnings summary
            summary = EarningsDataFetcher.get_earnings_summary(ticker)

            next_earnings = summary['next_earnings']
            surprise_analysis = summary['surprise_analysis']
            earnings_history = summary['earnings_history']

            # === NEXT EARNINGS DATE ===
            st.markdown("### 📅 Next Earnings Report")

            if next_earnings:
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Countdown card
                    days_until = next_earnings['days_until']
                    date_str = next_earnings['date'].strftime('%B %d, %Y')

                    countdown_color = '#FF3B30' if days_until <= 7 else '#FF9500' if days_until <= 30 else '#007AFF'

                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 6px solid {countdown_color}; text-align: center;">
                        <h1 style="font-size: 4rem; margin: 0; color: {countdown_color};">{days_until}</h1>
                        <h3 style="margin: 0.5rem 0;">Days Until Earnings</h3>
                        <p style="color: #86868B; margin: 0;">{date_str}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # Estimates
                    st.markdown("#### 📈 Analyst Estimates")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        eps_estimate = next_earnings.get('eps_estimate')
                        if eps_estimate and not pd.isna(eps_estimate):
                            st.metric("EPS Estimate", f"${eps_estimate:.2f}")
                        else:
                            st.metric("EPS Estimate", "N/A")

                    with col_b:
                        rev_estimate = next_earnings.get('revenue_estimate')
                        if rev_estimate and not pd.isna(rev_estimate):
                            # Convert to billions if large number
                            if rev_estimate > 1_000_000_000:
                                rev_b = rev_estimate / 1_000_000_000
                                st.metric("Revenue Estimate", f"${rev_b:.2f}B")
                            else:
                                st.metric("Revenue Estimate", f"${rev_estimate:,.0f}")
                        else:
                            st.metric("Revenue Estimate", "N/A")

                    st.markdown("---")

                    # Calendar alert
                    if days_until <= 7:
                        st.error(f"""
                        🚨 **Earnings Alert**: {ticker} reports earnings in **{days_until} days**!
                        Expect increased volatility and potential significant price movements.
                        """)
                    elif days_until <= 30:
                        st.warning(f"""
                        ⏰ **Upcoming Earnings**: {ticker} reports in **{days_until} days**.
                        Plan your trades accordingly.
                        """)
                    else:
                        st.info(f"""
                        📆 **Next Earnings**: {ticker} reports in **{days_until} days** ({date_str}).
                        """)

            else:
                st.info("""
                ⚠️ **No upcoming earnings date found**

                This could mean:
                - Earnings date not yet announced
                - Data not available for this ticker
                - Try a different well-known ticker (AAPL, MSFT, GOOGL)
                """)

            st.markdown("---")

            # === EARNINGS SURPRISE ANALYSIS ===
            st.markdown("### 🎯 Earnings Surprise Track Record")

            if surprise_analysis['has_data'] and surprise_analysis['total_reports'] > 0:

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total = surprise_analysis['total_reports']
                    st.metric("📋 Total Reports", total)

                with col2:
                    beats = surprise_analysis['beat_count']
                    beat_rate = (beats / total * 100) if total > 0 else 0
                    st.metric("🟢 Beats", f"{beats} ({beat_rate:.0f}%)")

                with col3:
                    misses = surprise_analysis['miss_count']
                    miss_rate = (misses / total * 100) if total > 0 else 0
                    st.metric("🔴 Misses", f"{misses} ({miss_rate:.0f}%)")

                with col4:
                    avg_surprise = surprise_analysis['avg_surprise_pct']
                    st.metric("📊 Avg Surprise", f"{avg_surprise:+.1f}%",
                             delta_color="normal" if avg_surprise > 0 else "inverse")

                st.markdown("")

                # Interpretation
                if beat_rate >= 70:
                    st.success(f"""
                    **🟢 Consistent Performer**: {ticker} has beaten earnings estimates **{beat_rate:.0f}% of the time**.
                    This consistency often indicates strong management and reliable execution.
                    """)
                elif miss_rate >= 50:
                    st.error(f"""
                    **🔴 Inconsistent Performance**: {ticker} has missed estimates **{miss_rate:.0f}% of the time**.
                    This could indicate challenges in forecasting or execution issues.
                    """)
                else:
                    st.info(f"""
                    **⚪ Mixed Track Record**: {ticker} has a balanced earnings history with **{beat_rate:.0f}% beats** and **{miss_rate:.0f}% misses**.
                    """)

                # Surprise chart
                if surprise_analysis['recent_surprises']:
                    st.markdown("#### Recent Earnings Surprises")
                    surprise_chart = create_earnings_surprise_chart(surprise_analysis['recent_surprises'])
                    st.plotly_chart(surprise_chart, use_container_width=True)

            else:
                st.info("📊 No earnings surprise data available for this ticker")

            st.markdown("---")

            # === HISTORICAL EARNINGS ===
            st.markdown("### 📈 Historical Earnings Performance")

            if earnings_history is not None and not earnings_history.empty:

                # EPS trend chart
                eps_chart = create_eps_trend_chart(earnings_history)
                st.plotly_chart(eps_chart, use_container_width=True)

                # Revenue trend chart
                revenue_chart = create_revenue_chart(earnings_history)
                st.plotly_chart(revenue_chart, use_container_width=True)

                # Earnings table
                st.markdown("#### 📋 Quarterly Earnings Details")

                df_display = earnings_history.copy()
                df_display = df_display.sort_index(ascending=False)

                # Format revenue to billions
                if 'Revenue' in df_display.columns:
                    df_display['Revenue (B)'] = (df_display['Revenue'] / 1_000_000_000).round(2)
                    df_display = df_display.drop('Revenue', axis=1)

                # Display table
                st.dataframe(
                    df_display.head(12),
                    use_container_width=True
                )

                # Download button
                csv = df_display.to_csv()
                st.download_button(
                    label="📥 Download Earnings History (CSV)",
                    data=csv,
                    file_name=f"{ticker}_earnings_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            else:
                st.info("📊 No historical earnings data available for this ticker")

            # === INSIGHTS ===
            st.markdown("### 💡 Earnings Trading Insights")

            st.info("""
            **📚 Understanding Earnings Reports:**

            **Key Metrics to Watch:**
            - **EPS (Earnings Per Share)**: Profit divided by shares outstanding
            - **Revenue**: Total sales - shows business growth
            - **Guidance**: Company's forecast for next quarter (often more important than current results!)
            - **Earnings Beat/Miss**: Actual vs analyst estimates

            **Trading Around Earnings:**
            - **Before Earnings**: IV (implied volatility) typically rises → option premiums increase
            - **Earnings Day**: Expect 5-20% price swings (or more for growth stocks)
            - **After Earnings**: Price can gap up/down based on results + guidance

            **Earnings Strategies:**
            - **🟢 Momentum Play**: Buy stocks with consistent beats before earnings
            - **🔴 Avoid Risk**: Close positions before earnings to avoid volatility
            - **📊 Straddle/Strangle**: Options strategy to profit from volatility regardless of direction
            - **⏰ Wait & See**: Wait for post-earnings volatility to settle before entering

            **Red Flags:**
            - Multiple consecutive earnings misses
            - Lowered guidance
            - Declining revenue despite growing EPS (buybacks masking issues)
            - Management turnover around earnings
            """)

            # === DISCLAIMER ===
            st.warning("""
            ⚠️ **Earnings Disclaimer:**
            - Earnings data is historical and not predictive of future results
            - Analyst estimates can be inaccurate
            - Earnings dates can change - always verify with official sources
            - This is for informational purposes only, not investment advice
            - Earnings volatility creates both opportunities and risks
            """)

    else:
        # Show info when no analysis yet
        st.info("👆 Enter a stock ticker and click 'Analyze Earnings' to get started!")

        st.markdown("""
        ### 🎯 What This Tool Does:

        - **📅 Next Earnings Date** - Countdown to upcoming earnings report
        - **🎯 Earnings Surprises** - Track beat/miss record vs analyst estimates
        - **📈 EPS Trends** - Visualize earnings growth over time
        - **💰 Revenue Analysis** - Monitor quarterly revenue trends
        - **📊 Historical Data** - Complete earnings history with downloadable data

        ### 💡 Why Earnings Matter:

        1. **Price Catalysts** - Earnings often cause 5-20% single-day moves
        2. **Company Health** - Shows if business is growing or struggling
        3. **Beat the Street** - Consistent beats = strong execution
        4. **Guidance Matters** - Future outlook often more important than current results
        5. **Volatility Spike** - Options become expensive before earnings

        ### 🟢 Best Tickers to Analyze:

        - **AAPL** (Apple) - Quarterly earnings powerhouse
        - **TSLA** (Tesla) - High volatility around earnings
        - **NVDA** (NVIDIA) - Consistent beats, major moves
        - **MSFT** (Microsoft) - Reliable performer
        - **AMZN** (Amazon) - Revenue growth leader
        """)
