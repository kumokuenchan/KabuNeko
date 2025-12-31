"""
Page: Cryptocurrency Analysis

Dedicated page for analyzing Bitcoin and other cryptocurrencies with crypto-specific
features including 24/7 market data, volatility analysis, and multi-crypto comparison.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional
from src.data.fetcher import get_stock_data, get_multiple_stocks
from src.models.feature_engineering import FeatureEngineer
from src.analysis.investment_recommendation import get_investment_recommendation
from src.ui.charts import create_candlestick_chart, create_line_chart, create_bar_chart


# Popular cryptocurrencies
CRYPTO_LIST = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'XRP-USD': 'Ripple',
    'ADA-USD': 'Cardano',
    'SOL-USD': 'Solana',
    'DOGE-USD': 'Dogecoin',
    'DOT-USD': 'Polkadot',
    'MATIC-USD': 'Polygon',
    'LTC-USD': 'Litecoin',
    'AVAX-USD': 'Avalanche',
    'LINK-USD': 'Chainlink',
}


def render() -> None:
    """Render the cryptocurrency analysis page"""

    st.title("₿ Cryptocurrency Analysis")

    st.markdown("""
    Analyze Bitcoin and other cryptocurrencies with real-time data, technical indicators,
    and volatility metrics. **Note:** Crypto markets trade 24/7!
    """)

    # Crypto selector
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        crypto_symbol = st.selectbox(
            "Select Cryptocurrency",
            options=list(CRYPTO_LIST.keys()),
            format_func=lambda x: f"{CRYPTO_LIST[x]} ({x})",
            index=0
        )
        crypto_name = CRYPTO_LIST[crypto_symbol]

    with col2:
        period = st.selectbox(
            "Time Period",
            ["24 Hours", "7 Days", "30 Days", "90 Days", "1 Year", "All Time"],
            index=4
        )

    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("📊 Analyze Crypto", type="primary")

    # Period mapping
    period_map = {
        "24 Hours": 1,
        "7 Days": 7,
        "30 Days": 30,
        "90 Days": 90,
        "1 Year": 365,
        "All Time": 1825  # ~5 years
    }

    if analyze_btn:
        days = period_map[period]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        with st.spinner(f"Fetching {crypto_name} data..."):
            try:
                df = get_stock_data(
                    crypto_symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                if df is None or len(df) == 0:
                    st.error(f"❌ Could not fetch data for {crypto_name}")
                    return

                st.success(f"✅ Loaded {len(df)} data points for {crypto_name}")

                # === KEY METRICS ===
                st.markdown("### 💰 Key Metrics")

                current_price = df['Close'].iloc[-1]
                prev_price_1d = df['Close'].iloc[-2] if len(df) > 1 else current_price
                prev_price_7d = df['Close'].iloc[-7] if len(df) > 7 else current_price
                prev_price_30d = df['Close'].iloc[-30] if len(df) > 30 else current_price

                change_1d = ((current_price - prev_price_1d) / prev_price_1d) * 100
                change_7d = ((current_price - prev_price_7d) / prev_price_7d) * 100
                change_30d = ((current_price - prev_price_30d) / prev_price_30d) * 100

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Current Price", f"${current_price:,.2f}", f"{change_1d:+.2f}%")

                with col2:
                    st.metric("24h Change", f"{change_1d:+.2f}%",
                             delta_color="normal" if change_1d >= 0 else "inverse")

                with col3:
                    st.metric("7d Change", f"{change_7d:+.2f}%",
                             delta_color="normal" if change_7d >= 0 else "inverse")

                with col4:
                    st.metric("30d Change", f"{change_30d:+.2f}%",
                             delta_color="normal" if change_30d >= 0 else "inverse")

                with col5:
                    avg_volume = df['Volume'].tail(7).mean()
                    st.metric("Avg Volume (7d)", f"${avg_volume/1e9:.2f}B")

                # === VOLATILITY ANALYSIS ===
                st.markdown("### 📈 Volatility Analysis")

                # Calculate daily returns
                returns = df['Close'].pct_change().dropna()

                # Annualized volatility (crypto trades 24/7)
                volatility_daily = returns.std() * 100
                volatility_annual = returns.std() * (365 ** 0.5) * 100

                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min() * 100)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Daily Volatility", f"{volatility_daily:.2f}%")

                with col2:
                    st.metric("Annual Volatility", f"{volatility_annual:.1f}%")

                with col3:
                    st.metric("Max Drawdown", f"{max_drawdown:.1f}%")

                with col4:
                    sharpe_ratio = (returns.mean() / returns.std()) * (365 ** 0.5) if returns.std() > 0 else 0
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                # === PRICE CHART ===
                st.markdown("### 📊 Price Chart")

                fig = create_candlestick_chart(df, ticker=crypto_name, height=500)
                st.plotly_chart(fig, width="stretch")

                # === VOLUME CHART ===
                st.markdown("### 📊 Trading Volume")

                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))

                fig_vol.update_layout(
                    title=f'{crypto_name} Trading Volume',
                    yaxis_title='Volume (USD)',
                    xaxis_title='Date',
                    template='plotly_white',
                    height=300
                )

                st.plotly_chart(fig_vol, width="stretch")

                # === TECHNICAL INDICATORS ===
                if len(df) >= 50:
                    st.markdown("### 📉 Technical Indicators")

                    # Prepare features
                    df_features = FeatureEngineer.prepare_features(df)

                    col1, col2 = st.columns(2)

                    with col1:
                        # RSI
                        if 'RSI' in df_features.columns:
                            rsi = df_features['RSI'].iloc[-1]
                            st.markdown(f"**RSI (14):** {rsi:.1f}")

                            if rsi > 70:
                                st.warning("⚠️ Overbought - Consider taking profits")
                            elif rsi < 30:
                                st.success("✅ Oversold - Potential buying opportunity")
                            else:
                                st.info("📊 Neutral zone")

                            # RSI Chart
                            fig_rsi = create_line_chart(
                                df_features.tail(100),
                                y_column='RSI',
                                title='RSI (14)',
                                y_title='RSI',
                                color='purple',
                                height=300
                            )
                            # Add overbought/oversold lines
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                            st.plotly_chart(fig_rsi, width="stretch")

                    with col2:
                        # MACD
                        if 'MACD' in df_features.columns and 'MACD_signal' in df_features.columns:
                            macd = df_features['MACD'].iloc[-1]
                            signal = df_features['MACD_signal'].iloc[-1]

                            st.markdown(f"**MACD:** {macd:.2f}")
                            st.markdown(f"**Signal:** {signal:.2f}")

                            if macd > signal:
                                st.success("✅ Bullish signal")
                            else:
                                st.warning("⚠️ Bearish signal")

                            # MACD Chart
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(
                                x=df_features.tail(100).index,
                                y=df_features.tail(100)['MACD'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue')
                            ))
                            fig_macd.add_trace(go.Scatter(
                                x=df_features.tail(100).index,
                                y=df_features.tail(100)['MACD_signal'],
                                mode='lines',
                                name='Signal',
                                line=dict(color='red')
                            ))
                            fig_macd.update_layout(
                                title='MACD',
                                template='plotly_white',
                                height=300
                            )
                            st.plotly_chart(fig_macd, width="stretch")

                    # === INVESTMENT RECOMMENDATION ===
                    st.markdown("### 💡 AI Recommendation")

                    try:
                        recommendation = get_investment_recommendation(df_features)

                        rec_color_map = {
                            'STRONG BUY': '#00ff00',
                            'BUY': '#90ee90',
                            'HOLD': '#ffff00',
                            'SELL': '#ffcccb',
                            'STRONG SELL': '#ff0000'
                        }

                        rec_color = rec_color_map.get(recommendation['recommendation'], '#cccccc')

                        st.markdown(f"""
                        <div style="background-color: {rec_color}; padding: 1.5rem; border-radius: 1rem; text-align: center;">
                            <h2 style="margin: 0; color: #000;">{recommendation['color']} {recommendation['recommendation']}</h2>
                            <p style="margin: 0.5rem 0; color: #000;">Score: {recommendation['overall_score']}/100</p>
                            <p style="margin: 0; color: #000;">{recommendation['action']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("---")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Price Targets:**")
                            targets = recommendation['price_targets']
                            st.markdown(f"- Current: ${targets['current_price']:,.2f}")
                            st.markdown(f"- Entry: ${targets['entry_price']:,.2f}")
                            st.markdown(f"- Stop Loss: ${targets['stop_loss']:,.2f}")
                            st.markdown(f"- Target: ${targets['target_price']:,.2f}")

                        with col2:
                            st.markdown("**Risk Assessment:**")
                            risk = recommendation['risk_metrics']
                            st.markdown(f"- Risk Level: **{risk['risk_level']}**")
                            st.markdown(f"- Volatility: {risk['volatility']:.1f}%")
                            st.markdown(f"- Max Drawdown: {risk['max_drawdown']:.1f}%")

                    except Exception as e:
                        st.warning(f"Could not generate recommendation: {str(e)}")

                # === PRICE DISTRIBUTION ===
                st.markdown("### 📊 Price Distribution (Last 30 Days)")

                recent_prices = df['Close'].tail(30)
                fig_dist = go.Figure(data=[go.Histogram(
                    x=recent_prices,
                    nbinsx=20,
                    marker_color='lightgreen'
                )])

                fig_dist.update_layout(
                    title='Price Frequency Distribution',
                    xaxis_title='Price (USD)',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=300
                )

                st.plotly_chart(fig_dist, width="stretch")

                # === CRYPTO INFO ===
                st.markdown("---")
                st.markdown("### ℹ️ About Cryptocurrency Trading")
                st.info("""
                **Key Differences from Stocks:**
                - 🕐 **24/7 Trading** - Crypto markets never close
                - 📈 **Higher Volatility** - Expect larger price swings
                - 🌍 **Global Market** - Trades across all time zones
                - ⚡ **Instant Settlement** - Faster than traditional stocks
                - 🔒 **Decentralized** - Not controlled by any single entity

                **Investment Tips:**
                - Only invest what you can afford to lose
                - Crypto is highly volatile - use stop losses
                - Consider dollar-cost averaging (DCA)
                - Store crypto securely in a wallet
                - Stay updated on regulation news
                """)

                # === DISCLAIMER ===
                st.warning("""
                ⚠️ **Cryptocurrency Disclaimer:**
                Cryptocurrencies are highly volatile and speculative investments. Prices can fluctuate
                dramatically in short periods. This analysis is for educational purposes only and is
                NOT financial advice. Always do your own research (DYOR) and consult a financial
                advisor before investing in cryptocurrencies.
                """)

            except Exception as e:
                st.error(f"❌ Error analyzing {crypto_name}: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())

    else:
        # Show crypto overview when no analysis run yet
        st.info("👆 Select a cryptocurrency and click 'Analyze Crypto' to get started!")

        st.markdown("### 🪙 Popular Cryptocurrencies")

        # Show quick price overview for top cryptos
        with st.spinner("Loading top cryptos..."):
            try:
                top_cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
                end_date = datetime.now()
                start_date = end_date - timedelta(days=2)

                crypto_data = get_multiple_stocks(
                    top_cryptos,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                quick_data = []
                for symbol, df in crypto_data.items():
                    if df is not None and len(df) >= 2:
                        current = df['Close'].iloc[-1]
                        prev = df['Close'].iloc[-2]
                        change = ((current - prev) / prev) * 100

                        quick_data.append({
                            'Crypto': CRYPTO_LIST.get(symbol, symbol),
                            'Symbol': symbol,
                            'Price': f"${current:,.2f}",
                            '24h Change': f"{change:+.2f}%"
                        })

                if quick_data:
                    df_quick = pd.DataFrame(quick_data)
                    st.dataframe(df_quick, width="stretch", hide_index=True)

            except Exception as e:
                st.warning("Could not load quick crypto overview")

        st.markdown("""
        ### 💡 What You Can Analyze:

        - **Real-time Prices** - Current crypto prices in USD
        - **Volatility Metrics** - Daily and annual volatility, Sharpe ratio
        - **Technical Indicators** - RSI, MACD, Moving Averages
        - **Trading Volume** - Market activity and liquidity
        - **AI Recommendations** - BUY/SELL/HOLD signals
        - **Risk Assessment** - Maximum drawdown and risk levels

        ### 🎯 Supported Cryptocurrencies:

        Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB), Ripple (XRP),
        Cardano (ADA), Solana (SOL), Dogecoin (DOGE), Polkadot (DOT),
        Polygon (MATIC), Litecoin (LTC), Avalanche (AVAX), Chainlink (LINK)
        """)
