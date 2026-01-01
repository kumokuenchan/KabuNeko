"""
Page: ETF Holdings Explorer

Explore ETF holdings, sector allocation, and fund characteristics.
Understand what you're investing in when you buy an ETF.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from src.data.etf_data import ETFDataFetcher


def create_sector_pie_chart(sector_data: dict) -> go.Figure:
    """Create pie chart for sector allocation"""

    if not sector_data:
        return go.Figure()

    # Prepare data
    sectors = list(sector_data.keys())
    weights = list(sector_data.values())

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=weights,
        hole=0.4,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        ),
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
    )])

    fig.update_layout(
        title="Sector Allocation",
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )

    return fig


def create_holdings_bar_chart(holdings_df: pd.DataFrame, limit: int = 10) -> go.Figure:
    """Create bar chart for top holdings"""

    if holdings_df is None or holdings_df.empty:
        return go.Figure()

    # Find weight column
    weight_col = None
    for col in ['Weight (%)', 'Weight', 'weight', '% of Total', 'pct_of_total']:
        if col in holdings_df.columns:
            weight_col = col
            break

    if not weight_col:
        return go.Figure()

    # Get top holdings
    top_holdings = holdings_df.head(limit).copy()

    # Find holding name column
    name_col = None
    for col in ['Holding', 'holding', 'Symbol', 'symbol', 'Name', 'name']:
        if col in top_holdings.columns:
            name_col = col
            break

    if not name_col:
        return go.Figure()

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_holdings[name_col],
        x=top_holdings[weight_col],
        orientation='h',
        marker_color='#007AFF',
        text=[f"{w:.2f}%" for w in top_holdings[weight_col]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Weight: %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=f"Top {limit} Holdings by Weight",
        xaxis_title="Weight (%)",
        yaxis_title="",
        template='plotly_white',
        height=max(350, limit * 35),
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )

    return fig


def render() -> None:
    """Render the ETF holdings explorer page"""

    st.title("🔍 ETF Holdings Explorer")

    st.markdown("""
    Discover what's inside any ETF - view holdings, sector allocation, and fund details.
    **Know what you're investing in** before you buy!
    """)

    # ETF input
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        ticker = st.text_input(
            "Enter ETF Ticker",
            value="SPY",
            placeholder="e.g., SPY, QQQ, VOO",
            help="Enter an ETF ticker symbol"
        ).upper()

    with col2:
        # Popular ETFs dropdown
        popular = st.selectbox(
            "Or select popular ETF",
            ["Custom"] + list(ETFDataFetcher.POPULAR_ETFS.keys()),
            format_func=lambda x: f"{x} - {ETFDataFetcher.POPULAR_ETFS[x]}" if x != "Custom" else x
        )

        if popular != "Custom":
            ticker = popular

    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Explore ETF", type="primary")

    # Clear cache button
    if st.button("🔄 Clear Cache", help="Clear cached ETF data and re-fetch from source"):
        st.cache_data.clear()
        st.success("✅ Cache cleared! Try exploring an ETF again.")
        st.rerun()

    if analyze_btn:
        with st.spinner(f"Analyzing {ticker}..."):

            # Get ETF info
            etf_info = ETFDataFetcher.get_etf_info(ticker)

            if not etf_info:
                st.error(f"""
                ❌ Could not fetch data for **{ticker}**.

                **Possible reasons:**
                - Not a valid ETF ticker
                - Ticker might be a stock, not an ETF
                - Data temporarily unavailable

                **Try:**
                - Check the ticker symbol
                - Use a popular ETF from the dropdown
                - Examples: SPY, QQQ, VOO, VTI
                """)
                return

            st.success(f"✅ Successfully loaded data for {etf_info['name']}")

            # === ETF OVERVIEW ===
            st.markdown("### 📊 ETF Overview")

            # Key metrics in cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Current Price",
                    f"${etf_info['price']:.2f}",
                    f"{etf_info['change_pct']:+.2f}%"
                )

            with col2:
                if etf_info['expense_ratio']:
                    st.metric("Expense Ratio", f"{etf_info['expense_ratio']:.2f}%")
                else:
                    st.metric("Expense Ratio", "N/A")

            with col3:
                if etf_info['yield']:
                    st.metric("Dividend Yield", f"{etf_info['yield']:.2f}%")
                else:
                    st.metric("Dividend Yield", "N/A")

            with col4:
                if etf_info['aum'] and etf_info['aum'] > 0:
                    aum_b = etf_info['aum'] / 1_000_000_000
                    st.metric("AUM", f"${aum_b:.2f}B")
                else:
                    st.metric("AUM", "N/A")

            # ETF Details card
            st.markdown("#### 📋 Fund Details")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                <div style="background: #F5F5F7; padding: 1.5rem; border-radius: 12px;">
                    <p style="margin: 0.5rem 0;"><strong>Fund Family:</strong> {etf_info['fund_family']}</p>
                    <p style="margin: 0.5rem 0;"><strong>Category:</strong> {etf_info['category']}</p>
                    <p style="margin: 0.5rem 0;"><strong>Inception Date:</strong> {etf_info['inception_date']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Quick stats
                expense_ratio = etf_info['expense_ratio']

                if expense_ratio is not None:
                    expense_color = "#34C759" if expense_ratio < 0.20 else "#FF9500"
                    expense_text = f"{expense_ratio:.2f}%"
                    cost_label = 'Low Cost ✓' if expense_ratio < 0.20 else 'Higher Cost'
                else:
                    expense_color = "#8E8E93"
                    expense_text = "N/A"
                    cost_label = 'Data not available'

                st.markdown(f"""
                <div style="background: #F5F5F7; padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.85rem; color: #86868B;">Expense Ratio</div>
                    <div style="font-size: 2rem; font-weight: 600; color: {expense_color};">
                        {expense_text}
                    </div>
                    <div style="font-size: 0.75rem; color: #86868B;">
                        {cost_label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Description
            if etf_info['description']:
                with st.expander("📖 ETF Description"):
                    st.info(etf_info['description'])

            st.markdown("---")

            # === TOP HOLDINGS ===
            st.markdown("### 🏆 Top Holdings")

            holdings = ETFDataFetcher.get_etf_holdings(ticker, limit=15)

            if holdings is not None and not holdings.empty:

                # Holdings concentration
                concentration = ETFDataFetcher.calculate_holdings_concentration(holdings)

                if concentration['top_10_weight'] > 0:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Bar chart
                        holdings_chart = create_holdings_bar_chart(holdings, limit=10)
                        st.plotly_chart(holdings_chart, use_container_width=True)

                    with col2:
                        # Concentration metrics
                        conc_color = "#FF3B30" if concentration['top_10_weight'] >= 70 else "#FF9500" if concentration['top_10_weight'] >= 50 else "#34C759"

                        st.markdown(f"""
                        <div style="background: #F5F5F7; padding: 1.5rem; border-radius: 12px; margin-top: 2rem;">
                            <h4 style="margin-top: 0;">Concentration Analysis</h4>
                            <div style="text-align: center; margin: 1.5rem 0;">
                                <div style="font-size: 0.85rem; color: #86868B;">Top 10 Holdings</div>
                                <div style="font-size: 3rem; font-weight: 600; color: {conc_color};">
                                    {concentration['top_10_weight']:.1f}%
                                </div>
                                <div style="font-size: 0.9rem; font-weight: 600; color: {conc_color};">
                                    {concentration['concentration']}
                                </div>
                            </div>
                            <p style="font-size: 0.85rem; color: #86868B; margin: 0;">
                                The top 10 holdings represent {concentration['top_10_weight']:.1f}% of total assets.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Holdings table
                st.markdown("#### 📋 Complete Holdings List")

                # Display holdings table
                display_df = holdings.copy()

                # Format columns if they exist
                if 'Weight (%)' in display_df.columns:
                    display_df['Weight (%)'] = display_df['Weight (%)'].apply(lambda x: f"{x:.2f}%")

                if 'Shares' in display_df.columns:
                    display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")

                if 'Value' in display_df.columns:
                    display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Download button
                csv = holdings.to_csv(index=False)
                st.download_button(
                    label="📥 Download Holdings (CSV)",
                    data=csv,
                    file_name=f"{ticker}_holdings_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            else:
                st.warning(f"""
                ⚠️ **Holdings data not available for {ticker}**

                **We attempted to scrape data from:**
                - etfdb.com (web scraping)
                - yfinance API (fallback)

                **Possible reasons:**
                - Website structure changed
                - Ticker not found on etfdb.com
                - Network/connectivity issues
                - Anti-scraping protection

                **Alternative options:**
                - Visit [etfdb.com/{ticker}](https://etfdb.com/etf/{ticker}/) directly
                - Check the fund provider's website
                - Use ETF.com or Morningstar
                - Try popular ETFs: SPY, QQQ, VOO, IVV, VTI

                **Note:** The dashboard will automatically retry web scraping on next refresh.
                """)

            st.markdown("---")

            # === SECTOR ALLOCATION ===
            st.markdown("### 🏭 Sector Allocation")

            sector_data = ETFDataFetcher.get_sector_allocation(ticker)

            if sector_data:
                # Sector pie chart
                sector_chart = create_sector_pie_chart(sector_data)
                st.plotly_chart(sector_chart, use_container_width=True)

                # Sector breakdown table
                with st.expander("📊 View Sector Breakdown Table"):
                    sector_df = pd.DataFrame(list(sector_data.items()), columns=['Sector', 'Weight (%)'])
                    sector_df = sector_df.sort_values('Weight (%)', ascending=False)
                    sector_df['Weight (%)'] = sector_df['Weight (%)'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(sector_df, use_container_width=True, hide_index=True)

            else:
                st.info("""
                📊 **Sector allocation data not available**

                Sector breakdowns are typically available on:
                - Fund provider websites
                - ETF.com
                - Morningstar
                """)

            st.markdown("---")

            # === INSIGHTS ===
            st.markdown("### 💡 ETF Investment Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.info("""
                **📚 Understanding ETF Holdings:**

                **What to Look For:**
                - **Top 10 Weight**: Higher = more concentrated risk
                  - <40%: Well diversified
                  - 40-70%: Moderate concentration
                  - >70%: Highly concentrated (check top holdings!)

                **Expense Ratios:**
                - <0.10%: Excellent (ultra-low cost)
                  - 0.10-0.25%: Good (low cost)
                - 0.25-0.50%: Average
                - >0.50%: High (better have good performance!)

                **Sector Concentration:**
                - Balanced: No sector >25%
                - Sector-focused: One sector >50%
                - Know what you're betting on!

                **Holdings Overlap:**
                - Many ETFs hold the same stocks
                - Check overlap if buying multiple ETFs
                - Example: SPY and VOO are 99% the same!
                """)

            with col2:
                st.info("""
                **🎯 ETF Selection Tips:**

                **For Core Holdings:**
                - Low expense ratios (<0.15%)
                - High AUM (>$1B)
                - Broad diversification
                - Examples: SPY, VOO, VTI

                **Red Flags:**
                - Very high expense ratios (>1%)
                - Low AUM (<$50M) - liquidity risk
                - Extreme concentration
                - New/unproven fund family

                **Compare Before Buying:**
                - Same index, different expense ratios
                - SPY (0.09%) vs VOO (0.03%) - same S&P 500!
                - IVV (0.03%) - also S&P 500, even cheaper
                - Small % saves big over time

                **Tax Efficiency:**
                - ETFs are generally tax-efficient
                - Check dividend yield if in taxable account
                - Municipal bond ETFs for high earners
                """)

            # === DISCLAIMER ===
            st.warning("""
            ⚠️ **ETF Explorer Disclaimer:**
            - Holdings data may be delayed or limited due to API restrictions
            - Always verify holdings on official fund provider websites
            - Past performance doesn't guarantee future results
            - Expense ratios and holdings can change
            - This is for educational purposes only, not investment advice
            - Consider consulting a financial advisor for personalized advice
            """)

    else:
        # Show info when no analysis yet
        st.info("👆 Enter an ETF ticker or select from popular ETFs and click 'Explore ETF' to get started!")

        st.markdown("""
        ### 🎯 What This Tool Does:

        - **🔍 View Holdings** - See what stocks/bonds are inside the ETF
        - **📊 Sector Breakdown** - Understand sector exposure and concentration
        - **💰 Fund Details** - Expense ratio, AUM, dividend yield, inception date
        - **📈 Concentration Analysis** - How diversified or concentrated is the fund?
        - **📥 Export Data** - Download holdings for further analysis

        ### 💡 Popular ETFs to Explore:

        **📈 Broad Market:**
        - **SPY** - S&P 500 (most traded ETF)
        - **QQQ** - NASDAQ-100 (tech-heavy)
        - **VOO** - S&P 500 (low cost alternative to SPY)
        - **VTI** - Total US Stock Market

        **🌍 International:**
        - **VEA** - Developed Markets (Europe, Japan)
        - **VWO** - Emerging Markets
        - **IEMG** - Core Emerging Markets

        **🏢 Sector:**
        - **XLK** - Technology
        - **XLF** - Financials
        - **XLE** - Energy
        - **XLV** - Healthcare

        **🚀 Thematic:**
        - **ARKK** - Innovation (ARK)
        - **ARKG** - Genomics (ARK)
        - **GLD** - Gold
        - **VNQ** - Real Estate

        ### 🟢 Why This Matters:

        - Know exactly what you're buying
        - Avoid overlap between ETFs
        - Understand concentration risk
        - Compare expense ratios
        - Make informed investment decisions
        """)
