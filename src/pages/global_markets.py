"""
Page: Global Markets Dashboard

Track international stock indices, currencies, commodities, and global market status.
Monitor worldwide markets and their correlation with US stocks.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from src.data.global_markets import GlobalMarketsFetcher


def create_index_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of global indices performance"""

    if df.empty:
        return go.Figure()

    # Group by region
    regions = {
        'Americas': ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000', 'TSX (Canada)', 'Bovespa (Brazil)'],
        'Europe': ['FTSE 100 (UK)', 'DAX (Germany)', 'CAC 40 (France)', 'IBEX 35 (Spain)', 'FTSE MIB (Italy)'],
        'Asia-Pacific': ['Nikkei 225 (Japan)', 'Hang Seng (Hong Kong)', 'Shanghai Composite', 'KOSPI (South Korea)',
                        'Sensex (India)', 'Nifty 50 (India)', 'ASX 200 (Australia)'],
    }

    fig = go.Figure()

    y_pos = 0
    for region, indices in regions.items():
        region_data = df[df['name'].isin(indices)]

        if region_data.empty:
            continue

        for _, row in region_data.iterrows():
            color = '#34C759' if row['change_pct'] >= 0 else '#FF3B30'

            fig.add_trace(go.Bar(
                x=[row['change_pct']],
                y=[row['name']],
                orientation='h',
                marker_color=color,
                text=f"{row['change_pct']:+.2f}%",
                textposition='outside',
                hovertemplate=f"<b>{row['name']}</b><br>Price: {row['price']:.2f}<br>Change: {row['change_pct']:+.2f}%<extra></extra>",
                showlegend=False
            ))

    fig.update_layout(
        title="Global Market Performance Today",
        xaxis_title="Change %",
        yaxis_title="",
        template='plotly_white',
        height=max(400, len(df) * 25),
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#E5E5E7')
    )

    return fig


def create_currency_chart(df: pd.DataFrame) -> go.Figure:
    """Create bar chart for currency pairs"""

    if df.empty:
        return go.Figure()

    # Sort by change percentage
    df_sorted = df.sort_values('change_pct', ascending=True)

    colors = ['#34C759' if x >= 0 else '#FF3B30' for x in df_sorted['change_pct']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_sorted['change_pct'],
        y=df_sorted['name'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}%" for x in df_sorted['change_pct']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Currency Pairs Performance",
        xaxis_title="Change %",
        yaxis_title="",
        template='plotly_white',
        height=350,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#E5E5E7')
    )

    return fig


def create_commodity_chart(df: pd.DataFrame) -> go.Figure:
    """Create bar chart for commodities"""

    if df.empty:
        return go.Figure()

    # Sort by change percentage
    df_sorted = df.sort_values('change_pct', ascending=True)

    colors = ['#34C759' if x >= 0 else '#FF3B30' for x in df_sorted['change_pct']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_sorted['change_pct'],
        y=df_sorted['name'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}%" for x in df_sorted['change_pct']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Price: $%{customdata:.2f}<br>Change: %{x:.2f}%<extra></extra>',
        customdata=df_sorted['price']
    ))

    fig.update_layout(
        title="Commodities Performance",
        xaxis_title="Change %",
        yaxis_title="",
        template='plotly_white',
        height=400,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#E5E5E7')
    )

    return fig


def render() -> None:
    """Render the global markets dashboard page"""

    st.title("🌐 Global Markets Dashboard")

    st.markdown("""
    Track worldwide stock markets, currencies, and commodities in real-time.
    **Monitor global trends** and correlations with US markets!
    """)

    # Refresh button
    col1, col2 = st.columns([6, 1])

    with col2:
        refresh_btn = st.button("🔄 Refresh", type="primary")

    # === MARKET STATUS ===
    st.markdown("### ⏰ Global Market Status")

    market_status = GlobalMarketsFetcher.get_market_status()

    # Display in columns
    cols = st.columns(4)

    for i, status_info in enumerate(market_status):
        col_idx = i % 4
        with cols[col_idx]:
            status_emoji = "🟢" if status_info['status'] == 'OPEN' else "🔴"
            status_color = "#34C759" if status_info['status'] == 'OPEN' else "#8E8E93"

            st.markdown(f"""
            <div style="background: #F5F5F7; padding: 0.75rem; border-radius: 8px; border-left: 4px solid {status_color};">
                <div style="font-size: 0.85rem; font-weight: 600;">{status_emoji} {status_info['market']}</div>
                <div style="font-size: 0.75rem; color: #86868B; margin-top: 0.25rem;">
                    {status_info['status']} • {status_info['local_time']} local
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")  # Spacing

    st.markdown("---")

    # Fetch data with spinner
    with st.spinner("🌍 Loading global market data..."):
        indices_df = GlobalMarketsFetcher.get_global_indices()
        currencies_df = GlobalMarketsFetcher.get_currencies()
        commodities_df = GlobalMarketsFetcher.get_commodities()
        crypto_df = GlobalMarketsFetcher.get_crypto()

    # === GLOBAL INDICES ===
    st.markdown("### 📊 Global Stock Indices")

    if not indices_df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gainers = (indices_df['change_pct'] > 0).sum()
            st.metric("Markets Up", gainers, f"of {len(indices_df)}")

        with col2:
            losers = (indices_df['change_pct'] < 0).sum()
            st.metric("Markets Down", losers, f"of {len(indices_df)}")

        with col3:
            avg_change = indices_df['change_pct'].mean()
            st.metric("Avg Change", f"{avg_change:+.2f}%")

        with col4:
            top_performer = indices_df.loc[indices_df['change_pct'].idxmax()]
            st.metric("Top Performer", top_performer['name'][:15], f"{top_performer['change_pct']:+.1f}%")

        # Heatmap
        heatmap = create_index_heatmap(indices_df)
        st.plotly_chart(heatmap, use_container_width=True)

        # Detailed table
        with st.expander("📋 View Detailed Index Data"):
            display_df = indices_df.copy()
            display_df['price'] = display_df['price'].apply(lambda x: f"{x:,.2f}")
            display_df['change'] = display_df['change'].apply(lambda x: f"{x:+.2f}")
            display_df['change_pct'] = display_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
            display_df['52w_high'] = display_df['52w_high'].apply(lambda x: f"{x:,.2f}")
            display_df['52w_low'] = display_df['52w_low'].apply(lambda x: f"{x:,.2f}")

            display_df = display_df.rename(columns={
                'name': 'Index',
                'price': 'Price',
                'change': 'Change',
                'change_pct': 'Change %',
                '52w_high': '52W High',
                '52w_low': '52W Low'
            })

            st.dataframe(display_df[['Index', 'Price', 'Change', 'Change %', '52W High', '52W Low']],
                        use_container_width=True, hide_index=True)

    else:
        st.warning("⚠️ Unable to load global indices data")

    st.markdown("---")

    # === CURRENCIES ===
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💱 Currency Pairs")

        if not currencies_df.empty:
            currency_chart = create_currency_chart(currencies_df)
            st.plotly_chart(currency_chart, use_container_width=True)

            # Key metrics
            strongest = currencies_df.loc[currencies_df['change_pct'].idxmax()]
            weakest = currencies_df.loc[currencies_df['change_pct'].idxmin()]

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Strongest", strongest['name'], f"{strongest['change_pct']:+.2f}%")
            with col_b:
                st.metric("Weakest", weakest['name'], f"{weakest['change_pct']:+.2f}%")

        else:
            st.warning("⚠️ Unable to load currency data")

    with col2:
        st.markdown("### 💰 Cryptocurrencies")

        if not crypto_df.empty:
            for _, row in crypto_df.iterrows():
                change_color = "#34C759" if row['change_pct'] >= 0 else "#FF3B30"

                st.markdown(f"""
                <div style="background: #F5F5F7; padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; font-size: 1.1rem;">{row['name']}</div>
                            <div style="color: #86868B; font-size: 0.85rem;">${row['price']:,.2f}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {change_color}; font-weight: 600; font-size: 1.1rem;">{row['change_pct']:+.2f}%</div>
                            <div style="color: #86868B; font-size: 0.85rem;">${row['change']:+,.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ Unable to load crypto data")

    st.markdown("---")

    # === COMMODITIES ===
    st.markdown("### 🛢️ Commodities")

    if not commodities_df.empty:
        commodity_chart = create_commodity_chart(commodities_df)
        st.plotly_chart(commodity_chart, use_container_width=True)

        # Key commodities cards
        col1, col2, col3, col4 = st.columns(4)

        commodity_highlights = ['Gold', 'Crude Oil (WTI)', 'Silver', 'Natural Gas']

        for i, commodity_name in enumerate(commodity_highlights):
            commodity_data = commodities_df[commodities_df['name'] == commodity_name]

            if not commodity_data.empty:
                row = commodity_data.iloc[0]
                change_color = "#34C759" if row['change_pct'] >= 0 else "#FF3B30"

                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div style="background: #F5F5F7; padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 0.85rem; color: #86868B;">{row['name']}</div>
                        <div style="font-size: 1.5rem; font-weight: 600;">${row['price']:.2f}</div>
                        <div style="color: {change_color}; font-weight: 600;">{row['change_pct']:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Unable to load commodities data")

    st.markdown("---")

    # === CORRELATIONS ===
    st.markdown("### 🔗 Market Correlations with S&P 500")

    with st.expander("📊 View Correlation Analysis"):
        st.info("""
        **Understanding Correlations:**
        - **+1.0** = Perfect positive correlation (move together)
        - **0.0** = No correlation
        - **-1.0** = Perfect negative correlation (move opposite)
        - **> 0.7** = Strong positive correlation
        - **< -0.7** = Strong negative correlation
        """)

        with st.spinner("Calculating correlations..."):
            # Calculate correlations with S&P 500
            sp500_ticker = '^GSPC'

            corr_indices = ['Nikkei 225 (Japan)', 'DAX (Germany)', 'FTSE 100 (UK)', 'Shanghai Composite']
            corr_results = []

            for index_name in corr_indices:
                index_data = indices_df[indices_df['name'] == index_name]

                if not index_data.empty:
                    ticker = index_data.iloc[0]['ticker']
                    corr = GlobalMarketsFetcher.calculate_correlation(sp500_ticker, ticker)

                    if corr is not None:
                        corr_results.append({
                            'Index': index_name,
                            'Correlation': corr,
                            'Correlation %': f"{corr * 100:.1f}%",
                            'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'
                        })

            if corr_results:
                corr_df = pd.DataFrame(corr_results)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Unable to calculate correlations")

    # === INSIGHTS ===
    st.markdown("### 💡 Global Market Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **📚 Reading Global Markets:**

        **Market Open/Close:**
        - Markets trade 24/5 (Mon-Fri) across timezones
        - Asian markets open first, then Europe, then Americas
        - Weekend gaps can be significant after major news

        **Regional Correlations:**
        - **Europe & US**: Highly correlated (~0.8)
        - **Asia & US**: Moderately correlated (~0.5-0.6)
        - **Emerging Markets**: Lower correlation, more volatile

        **Leading Indicators:**
        - Asian markets can signal US market direction
        - European markets trade during US pre-market
        - Watch Shanghai for China-related news impact
        - Nikkei often leads Asian sentiment

        **Currency Impact:**
        - Strong USD = US stocks up, international down
        - Weak USD = commodities up, gold up
        - EUR/USD affects European markets
        """)

    with col2:
        st.info("""
        **🎯 Trading with Global Markets:**

        **Morning Routine:**
        1. Check Asian close (happened overnight)
        2. Monitor European markets (open during US pre-market)
        3. Identify global trends before US open

        **Risk-On vs Risk-Off:**
        - **Risk-On**: Stocks up, USD down, commodities up
        - **Risk-Off**: Stocks down, USD up, gold up, bonds up

        **Commodity Signals:**
        - **Gold Up**: Fear, inflation concerns
        - **Oil Up**: Economic growth expectations
        - **Copper Up**: Industrial demand rising

        **Currency Signals:**
        - **USD Strength**: Flight to safety
        - **EUR/USD Up**: European growth optimism
        - **JPY Strength**: Risk-off sentiment

        **Global Events:**
        - Watch for Central Bank meetings (Fed, ECB, BOJ)
        - Monitor geopolitical tensions
        - Track major economic data releases
        """)

    # === DISCLAIMER ===
    st.warning("""
    ⚠️ **Global Markets Disclaimer:**
    - Market data may be delayed 15-20 minutes
    - Market hours shown in local timezone
    - Correlations are historical and can change
    - Global events can cause sudden decoupling
    - This is for educational purposes only, not investment advice
    - Always verify market status before trading
    """)


if __name__ == "__main__":
    render()
