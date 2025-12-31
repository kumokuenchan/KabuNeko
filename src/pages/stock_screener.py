"""
Page: Stock Screener

Screen stocks based on technical criteria including RSI, MACD, volume spikes,
52-week highs/lows, and technical scores.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.data.fetcher import get_stock_data
from src.models.feature_engineering import FeatureEngineer
from src.analysis.investment_recommendation import get_investment_recommendation
from src.indicators.trend import TrendIndicators


def render():
    """Screen stocks based on technical criteria"""
    st.title("🔍 Stock Screener")

    st.markdown("""
    Find stocks matching your technical criteria. Screen by RSI, MACD, volume, price levels, and more.
    """)

    # Stock universe selection
    st.markdown("### 📋 Select Stock Universe")
    source = st.radio("Source", ["From Watchlist", "Popular Stocks", "Custom List"], horizontal=True)

    if source == "From Watchlist":
        watchlists = st.session_state['user_watchlists']['watchlists']
        if watchlists:
            selected_wl = st.selectbox("Select Watchlist", list(watchlists.keys()))
            tickers = watchlists[selected_wl]
        else:
            st.warning("⚠️ No watchlists found! Create one in Watchlist Manager first.")
            return
    elif source == "Popular Stocks":
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT",
                   "JNJ", "PG", "MA", "HD", "DIS", "NFLX", "PYPL", "INTC", "CSCO", "PEP"]
    else:
        tickers_input = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA")
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if not tickers:
        st.warning("⚠️ Please enter at least one ticker symbol")
        return

    # Screening criteria
    st.markdown("### 🎯 Screening Criteria")
    st.markdown("*Select one or more criteria. Stocks must match ALL selected criteria.*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Bullish Signals")
        check_rsi_oversold = st.checkbox("RSI Oversold (Buy Signal)")
        if check_rsi_oversold:
            rsi_threshold = st.slider("RSI below", 20, 40, 30, key="rsi_low")

        check_macd_bullish = st.checkbox("MACD Bullish Crossover")

        check_volume_spike = st.checkbox("Volume Spike")
        if check_volume_spike:
            volume_multiplier = st.slider("Volume > X times average", 1.5, 5.0, 2.0, 0.5, key="vol_spike")

        check_price_low = st.checkbox("Near 52-Week Low")
        if check_price_low:
            low_threshold = st.slider("Within % of 52W low", 1, 20, 10, key="price_low")

    with col2:
        st.markdown("#### 📉 Bearish Signals")
        check_rsi_overbought = st.checkbox("RSI Overbought (Sell Signal)")
        if check_rsi_overbought:
            rsi_high_threshold = st.slider("RSI above", 60, 80, 70, key="rsi_high")

        check_price_high = st.checkbox("Near 52-Week High")
        if check_price_high:
            high_threshold = st.slider("Within % of 52W high", 1, 20, 10, key="price_high")

        st.markdown("#### 🎯 General Filters")
        check_tech_score = st.checkbox("Minimum Technical Score")
        if check_tech_score:
            min_score = st.slider("Score above", 0, 100, 60, key="tech_score")

    if st.button("🔍 Run Screener", type="primary"):
        # Count selected criteria
        criteria_count = sum([
            check_rsi_oversold, check_rsi_overbought, check_macd_bullish,
            check_volume_spike, check_price_low, check_price_high, check_tech_score
        ])

        if criteria_count == 0:
            st.warning("⚠️ Please select at least one screening criterion")
            return

        with st.spinner(f"Screening {len(tickers)} stocks against {criteria_count} criteria..."):
            try:
                results = []
                progress_bar = st.progress(0)

                for idx, ticker in enumerate(tickers):
                    # Update progress
                    progress_bar.progress((idx + 1) / len(tickers))

                    # Fetch data (1 year for indicators)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)

                    try:
                        df = get_stock_data(
                            ticker,
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d')
                        )

                        if df is None or len(df) < 60:
                            continue

                        # Prepare features
                        df_features = FeatureEngineer.prepare_features(df)
                        rec = get_investment_recommendation(df_features)

                        # Check criteria
                        matches = []
                        passes = True

                        if check_rsi_oversold:
                            rsi = df_features['RSI'].iloc[-1]
                            if rsi < rsi_threshold:
                                matches.append(f"RSI {rsi:.1f}")
                            else:
                                passes = False

                        if check_rsi_overbought:
                            rsi = df_features['RSI'].iloc[-1]
                            if rsi > rsi_high_threshold:
                                matches.append(f"RSI {rsi:.1f}")
                            else:
                                passes = False

                        if check_macd_bullish:
                            macd_df = TrendIndicators.macd(df)
                            if macd_df.iloc[-1, 0] > macd_df.iloc[-1, 1]:  # MACD > Signal
                                matches.append("MACD Bullish")
                            else:
                                passes = False

                        if check_volume_spike:
                            if 'Volume_ratio' in df_features.columns:
                                vol_ratio = df_features['Volume_ratio'].iloc[-1]
                                if pd.notna(vol_ratio) and vol_ratio > volume_multiplier:
                                    matches.append(f"Vol {vol_ratio:.1f}x")
                                else:
                                    passes = False
                            else:
                                passes = False

                        if check_price_low:
                            current = df['Close'].iloc[-1]
                            low_52w = df['Low'].tail(min(252, len(df))).min()
                            distance = ((current - low_52w) / low_52w) * 100
                            if distance <= low_threshold:
                                matches.append(f"{distance:.1f}% from low")
                            else:
                                passes = False

                        if check_price_high:
                            current = df['Close'].iloc[-1]
                            high_52w = df['High'].tail(min(252, len(df))).max()
                            distance = ((high_52w - current) / high_52w) * 100
                            if distance <= high_threshold:
                                matches.append(f"{distance:.1f}% from high")
                            else:
                                passes = False

                        if check_tech_score:
                            if rec['technical_score'] >= min_score:
                                matches.append(f"Score {rec['technical_score']}")
                            else:
                                passes = False

                        if passes:
                            current_price = df['Close'].iloc[-1]
                            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                            change_pct = ((current_price - prev_price) / prev_price) * 100

                            results.append({
                                'Ticker': ticker,
                                'Price': f"${current_price:.2f}",
                                'Change %': f"{change_pct:+.2f}%",
                                'RSI': f"{df_features['RSI'].iloc[-1]:.1f}" if 'RSI' in df_features.columns else 'N/A',
                                'Tech Score': rec['technical_score'],
                                'Recommendation': rec['recommendation'],
                                'Signals': ', '.join(matches)
                            })

                    except Exception as e:
                        # Skip stocks with errors
                        continue

                progress_bar.empty()

                # Display results
                st.markdown("---")
                st.markdown(f"### 🎯 Screening Results: {len(results)} matches out of {len(tickers)} stocks")

                if results:
                    df_results = pd.DataFrame(results)

                    # Sort by technical score (descending)
                    df_results_sorted = df_results.sort_values('Tech Score', ascending=False)

                    st.dataframe(df_results_sorted, use_container_width=True, hide_index=True)

                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Matches Found", len(results))
                    with col2:
                        avg_score = df_results['Tech Score'].mean()
                        st.metric("Avg Tech Score", f"{avg_score:.1f}")
                    with col3:
                        buy_recs = len([r for r in results if 'BUY' in r['Recommendation']])
                        st.metric("Buy Recommendations", buy_recs)

                    # Export results
                    st.markdown("---")
                    csv = df_results_sorted.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Screening Results",
                        data=csv,
                        file_name=f"stock_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime='text/csv'
                    )

                else:
                    st.info("📭 No stocks matched all the selected criteria. Try adjusting your filters.")

            except Exception as e:
                st.error(f"❌ Screening error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    else:
        st.info("👆 Select criteria and click 'Run Screener' to find matching stocks!")

        st.markdown("### 💡 Screening Examples:")
        st.markdown("""
        **Find Oversold Stocks:**
        - ✓ RSI Oversold (< 30)
        - ✓ Minimum Technical Score (> 50)

        **Find Breakout Candidates:**
        - ✓ Near 52-Week High
        - ✓ Volume Spike (> 2x)
        - ✓ MACD Bullish Crossover

        **Find Value Opportunities:**
        - ✓ Near 52-Week Low
        - ✓ Minimum Technical Score (> 60)
        """)
