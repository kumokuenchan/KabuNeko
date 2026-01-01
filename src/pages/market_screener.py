"""
Page: Advanced Market Screener

Find trading opportunities with preset screeners including gap scanners,
unusual volume detector, momentum plays, value stocks, and more.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.analysis.market_screener import MarketScreener


def render() -> None:
    """Render the advanced market screener page"""

    st.title("🔍 Advanced Market Screener")

    st.markdown("""
    Find trading opportunities with powerful preset screeners.
    **Scan 50+ popular stocks** for gaps, unusual volume, momentum, and more!
    """)

    # Screener selection
    col1, col2 = st.columns([3, 1])

    with col1:
        screener_type = st.selectbox(
            "Select Screening Strategy",
            [
                "📈 Gap Ups (Pre-Market Movers)",
                "📉 Gap Downs (Dip Opportunities)",
                "🔊 Unusual Volume (Breakout Potential)",
                "⚡ Momentum Stocks (Gap + Volume)",
                "💎 Value Stocks (Low P/E + Dividend)",
                "🎢 High Beta (Volatile Stocks)",
                "🏆 Near 52-Week High (Strength)"
            ]
        )

    with col2:
        st.write("")
        st.write("")
        scan_btn = st.button("🔍 Run Screener", type="primary")

    # Stock universe selection
    with st.expander("⚙️ Advanced Settings"):
        universe = st.radio(
            "Stock Universe",
            ["Top 50 S&P 500", "Tech Stocks (30)", "Full S&P 500 (takes longer)"],
            index=0
        )

        if "Gap Up" in screener_type:
            min_gap = st.slider("Minimum Gap %", 1.0, 10.0, 3.0, 0.5)
        elif "Gap Down" in screener_type:
            min_gap = st.slider("Minimum Gap %", -10.0, -1.0, -3.0, 0.5)
        elif "Unusual Volume" in screener_type:
            min_volume_ratio = st.slider("Minimum Volume Ratio", 1.5, 5.0, 2.0, 0.25)
        elif "Momentum" in screener_type:
            col_a, col_b = st.columns(2)
            with col_a:
                min_gap = st.slider("Minimum Gap %", 1.0, 10.0, 2.0, 0.5)
            with col_b:
                min_volume_ratio = st.slider("Min Volume Ratio", 1.0, 5.0, 1.5, 0.25)
        elif "Value" in screener_type:
            col_a, col_b = st.columns(2)
            with col_a:
                max_pe = st.slider("Maximum P/E Ratio", 5, 30, 20, 1)
            with col_b:
                min_div_yield = st.slider("Min Dividend Yield %", 0.5, 5.0, 2.0, 0.25)
        elif "High Beta" in screener_type:
            min_beta = st.slider("Minimum Beta", 1.0, 3.0, 1.5, 0.1)
        elif "52-Week High" in screener_type:
            threshold = st.slider("Distance from 52W High %", 1.0, 10.0, 5.0, 0.5)

    if scan_btn:
        # Determine ticker universe
        if universe == "Tech Stocks (30)":
            tickers = MarketScreener.TECH_STOCKS
        elif universe == "Full S&P 500 (takes longer)":
            tickers = MarketScreener.SP500_TICKERS
            st.warning("⏳ Scanning full S&P 500 - this may take 30-60 seconds...")
        else:
            tickers = MarketScreener.SP500_TICKERS[:50]

        with st.spinner(f"🔍 Scanning {len(tickers)} stocks..."):

            # Run appropriate screener
            if "Gap Up" in screener_type:
                results = MarketScreener.scan_gap_ups(tickers, min_gap=min_gap if 'min_gap' in locals() else 3.0)
                sort_column = 'gap_pct'
                title = f"📈 Gap Ups (≥{min_gap if 'min_gap' in locals() else 3.0}%)"

            elif "Gap Down" in screener_type:
                results = MarketScreener.scan_gap_downs(tickers, min_gap=min_gap if 'min_gap' in locals() else -3.0)
                sort_column = 'gap_pct'
                title = f"📉 Gap Downs (≤{min_gap if 'min_gap' in locals() else -3.0}%)"

            elif "Unusual Volume" in screener_type:
                results = MarketScreener.scan_unusual_volume(tickers, min_ratio=min_volume_ratio if 'min_volume_ratio' in locals() else 2.0)
                sort_column = 'volume_ratio'
                title = f"🔊 Unusual Volume (≥{min_volume_ratio if 'min_volume_ratio' in locals() else 2.0}x avg)"

            elif "Momentum" in screener_type:
                results = MarketScreener.scan_momentum(
                    tickers,
                    min_gap=min_gap if 'min_gap' in locals() else 2.0,
                    min_volume_ratio=min_volume_ratio if 'min_volume_ratio' in locals() else 1.5
                )
                sort_column = 'momentum_score'
                title = "⚡ Momentum Stocks"

            elif "Value" in screener_type:
                results = MarketScreener.scan_value_stocks(
                    tickers,
                    max_pe=max_pe if 'max_pe' in locals() else 20,
                    min_div_yield=min_div_yield if 'min_div_yield' in locals() else 2.0
                )
                sort_column = 'pe_ratio'
                title = "💎 Value Stocks"

            elif "High Beta" in screener_type:
                results = MarketScreener.scan_high_beta(tickers, min_beta=min_beta if 'min_beta' in locals() else 1.5)
                sort_column = 'beta'
                title = f"🎢 High Beta Stocks (≥{min_beta if 'min_beta' in locals() else 1.5})"

            else:  # Near 52-Week High
                results = MarketScreener.scan_near_52w_high(tickers, threshold=threshold if 'threshold' in locals() else 5.0)
                sort_column = 'distance_from_52w_high'
                title = "🏆 Near 52-Week High"

        # Display results
        if results.empty:
            st.warning(f"""
            ⚠️ **No stocks found** matching your criteria.

            Try:
            - Adjusting the filters (lower gap %, volume ratio, etc.)
            - Selecting a different stock universe
            - Trying a different screener
            - Checking back later (market conditions change throughout the day)
            """)
        else:
            st.success(f"✅ Found **{len(results)} stocks** matching criteria!")

            st.markdown(f"### {title}")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Stocks Found", len(results))

            with col2:
                if 'gap_pct' in results.columns:
                    avg_gap = results['gap_pct'].mean()
                    st.metric("Avg Gap", f"{avg_gap:+.2f}%")
                elif 'volume_ratio' in results.columns:
                    avg_vol = results['volume_ratio'].mean()
                    st.metric("Avg Vol Ratio", f"{avg_vol:.2f}x")

            with col3:
                if 'sector' in results.columns:
                    top_sector = results['sector'].mode()[0] if len(results['sector'].mode()) > 0 else "N/A"
                    st.metric("Top Sector", top_sector)

            with col4:
                if 'market_cap' in results.columns:
                    total_mcap = results['market_cap'].sum() / 1e12
                    st.metric("Total Market Cap", f"${total_mcap:.2f}T")

            st.markdown("---")

            # Display table
            st.markdown("#### 📋 Scan Results")

            # Format dataframe for display
            display_df = results.copy()

            # Select columns based on screener type
            base_cols = ['ticker', 'name', 'price', 'gap_pct']

            if 'momentum_score' in display_df.columns:
                display_cols = base_cols + ['volume_ratio', 'momentum_score', 'sector']
            elif 'volume_ratio' in display_df.columns:
                display_cols = base_cols + ['volume', 'volume_ratio', 'sector']
            elif 'pe_ratio' in display_df.columns and 'dividend_yield' in display_df.columns:
                display_cols = ['ticker', 'name', 'price', 'pe_ratio', 'dividend_yield', 'sector']
            elif 'beta' in display_df.columns:
                display_cols = base_cols + ['beta', 'sector']
            elif 'distance_from_52w_high' in display_df.columns:
                display_cols = ['ticker', 'name', 'price', '52w_high', 'distance_from_52w_high', 'sector']
            else:
                display_cols = base_cols + ['sector']

            # Filter to available columns
            display_cols = [col for col in display_cols if col in display_df.columns]
            display_df = display_df[display_cols]

            # Format numbers
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
            if 'gap_pct' in display_df.columns:
                display_df['gap_pct'] = display_df['gap_pct'].apply(lambda x: f"{x:+.2f}%")
            if 'volume_ratio' in display_df.columns:
                display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.2f}x")
            if 'momentum_score' in display_df.columns:
                display_df['momentum_score'] = display_df['momentum_score'].apply(lambda x: f"{x:.1f}")
            if 'pe_ratio' in display_df.columns:
                display_df['pe_ratio'] = display_df['pe_ratio'].apply(lambda x: f"{x:.1f}" if x > 0 else "N/A")
            if 'dividend_yield' in display_df.columns:
                display_df['dividend_yield'] = display_df['dividend_yield'].apply(lambda x: f"{x:.2f}%")
            if 'beta' in display_df.columns:
                display_df['beta'] = display_df['beta'].apply(lambda x: f"{x:.2f}")
            if '52w_high' in display_df.columns:
                display_df['52w_high'] = display_df['52w_high'].apply(lambda x: f"${x:.2f}")
            if 'distance_from_52w_high' in display_df.columns:
                display_df['distance_from_52w_high'] = display_df['distance_from_52w_high'].apply(lambda x: f"{x:.1f}%")

            # Rename columns for display
            display_df = display_df.rename(columns={
                'ticker': 'Ticker',
                'name': 'Name',
                'price': 'Price',
                'gap_pct': 'Gap %',
                'volume': 'Volume',
                'volume_ratio': 'Vol Ratio',
                'momentum_score': 'Momentum',
                'pe_ratio': 'P/E',
                'dividend_yield': 'Div Yield',
                'beta': 'Beta',
                'sector': 'Sector',
                '52w_high': '52W High',
                'distance_from_52w_high': 'Distance'
            })

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            st.markdown("---")

            # === INSIGHTS ===
            st.markdown("### 💡 Screener Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.info("""
                **📚 How to Use Screener Results:**

                **Gap Scanners:**
                - **Gap Ups**: Look for follow-through with high volume
                - **Gap Downs**: Potential dip-buying opportunities
                - Check news catalyst for the gap
                - Watch for gap fill or continuation

                **Volume Scanners:**
                - Unusual volume = unusual interest
                - Check for news, earnings, or rumors
                - High volume confirms breakouts
                - Can signal institutional accumulation

                **Momentum:**
                - Combines gap and volume
                - Best for short-term trades
                - Use tight stops (3-5%)
                - Take profits quickly

                **Value:**
                - Long-term investment ideas
                - Low P/E + dividends = stable returns
                - Do fundamental analysis before buying
                - Hold for quarters/years, not days
                """)

            with col2:
                st.info("""
                **🎯 Trading Strategies by Screener:**

                **Gap Ups (>3%):**
                - Wait for first 30min to settle
                - Look for break of opening high
                - Target: Previous resistance or +5-10%

                **Gap Downs (>3%):**
                - Look for bounce at support
                - Check if oversold (RSI < 30)
                - Risk: Gap may continue lower

                **Unusual Volume (>2x):**
                - Investigate the cause
                - Watch for breakout confirmation
                - Combine with pattern analysis

                **High Beta (>1.5):**
                - Higher risk, higher reward
                - Use smaller position sizes
                - Good for day trading
                - Avoid during low volatility

                **Near 52W High:**
                - Strong momentum signal
                - Breakout to new highs = bullish
                - Use tight trailing stops
                - Momentum can continue for weeks
                """)

            # === DISCLAIMER ===
            st.warning("""
            ⚠️ **Screener Disclaimer:**
            - Screeners find potential opportunities, not guaranteed winners
            - Always do your own research before trading
            - Check news and earnings dates
            - Use proper position sizing and stop-losses
            - Past performance doesn't guarantee future results
            - Market data may be delayed 15-20 minutes
            - This is for educational purposes, not investment advice
            """)

    else:
        # Show info when no scan yet
        st.info("👆 Select a screening strategy and click 'Run Screener' to find trading opportunities!")

        st.markdown("""
        ### 🎯 Available Screeners:

        **📈 Gap Ups** - Stocks gapping up in pre-market/opening
        - Find momentum plays
        - Catch breakouts early
        - Identify potential runners

        **📉 Gap Downs** - Stocks gapping down (dip opportunities)
        - Find oversold bounces
        - Identify value entry points
        - Catch reversals

        **🔊 Unusual Volume** - Stocks with volume spikes
        - Detect institutional activity
        - Find breakout candidates
        - Spot news-driven moves

        **⚡ Momentum Stocks** - Gap up + high volume combo
        - Strongest movers
        - High-probability setups
        - Best for day/swing trading

        **💎 Value Stocks** - Low P/E + dividend payers
        - Long-term investments
        - Stable dividend income
        - Undervalued opportunities

        **🎢 High Beta** - Volatile stocks (Beta ≥ 1.5)
        - Day trading candidates
        - Options plays
        - High risk/reward

        **🏆 Near 52-Week High** - Stocks at/near yearly highs
        - Breakout candidates
        - Momentum leaders
        - Strength indicators

        ### 🟢 Pro Tips:

        - **Combine Multiple Screeners**: Cross-reference results
        - **Check Volume**: Breakouts need volume confirmation
        - **Verify News**: Always check why a stock is moving
        - **Use Alerts**: Set price alerts on screener results
        - **Backtest**: Check how stocks historically behave after gaps/volume spikes
        """)
