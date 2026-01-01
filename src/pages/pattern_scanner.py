"""
Page: Chart Pattern Scanner

Automatically detect technical chart patterns and generate trading signals.
Uses AI to identify head & shoulders, double tops/bottoms, triangles, and more.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional
from src.data.fetcher import get_stock_data
from src.analysis.pattern_detector import PatternDetector


def create_pattern_chart(df: pd.DataFrame, ticker: str, pattern: Optional[dict] = None) -> go.Figure:
    """Create candlestick chart with pattern highlighting"""

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker,
        increasing_line_color='#34C759',
        decreasing_line_color='#FF3B30'
    ))

    # Add pattern indicators if detected
    if pattern:
        current_price = pattern['current_price']

        # Add resistance line if available
        if 'resistance' in pattern:
            fig.add_hline(
                y=pattern['resistance'],
                line_dash="dash",
                line_color="#FF3B30",
                annotation_text=f"Resistance: ${pattern['resistance']:.2f}",
                annotation_position="right"
            )

        # Add support line if available
        if 'support' in pattern:
            fig.add_hline(
                y=pattern['support'],
                line_dash="dash",
                line_color="#34C759",
                annotation_text=f"Support: ${pattern['support']:.2f}",
                annotation_position="right"
            )

        # Add neckline if available
        if 'neckline' in pattern:
            fig.add_hline(
                y=pattern['neckline'],
                line_dash="dot",
                line_color="#007AFF",
                annotation_text=f"Neckline: ${pattern['neckline']:.2f}",
                annotation_position="right"
            )

        # Add target price
        if 'target_price' in pattern:
            fig.add_hline(
                y=pattern['target_price'],
                line_dash="dash",
                line_color="#FFD700",
                annotation_text=f"Target: ${pattern['target_price']:.2f}",
                annotation_position="right"
            )

    fig.update_layout(
        title=f"{ticker} Price Chart with Pattern Detection",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template='plotly_white',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    return fig


def create_support_resistance_chart(df: pd.DataFrame, ticker: str, levels: dict) -> go.Figure:
    """Create chart showing support and resistance levels"""

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#007AFF', width=2)
    ))

    # Current price
    current = levels['current_price']

    # Add resistance levels
    for i, (key, value) in enumerate([('resistance_1', levels['resistance_1']),
                                       ('resistance_2', levels['resistance_2']),
                                       ('resistance_3', levels['resistance_3'])]):
        alpha = 1.0 - (i * 0.25)
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=f"rgba(255, 59, 48, {alpha})",
            annotation_text=f"R{i+1}: ${value:.2f}",
            annotation_position="right"
        )

    # Add support levels
    for i, (key, value) in enumerate([('support_1', levels['support_1']),
                                       ('support_2', levels['support_2']),
                                       ('support_3', levels['support_3'])]):
        alpha = 1.0 - (i * 0.25)
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=f"rgba(52, 199, 89, {alpha})",
            annotation_text=f"S{i+1}: ${value:.2f}",
            annotation_position="right"
        )

    # Add pivot point
    fig.add_hline(
        y=levels['pivot'],
        line_dash="dot",
        line_color="#8E8E93",
        annotation_text=f"Pivot: ${levels['pivot']:.2f}",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"{ticker} Support & Resistance Levels",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    return fig


def render() -> None:
    """Render the chart pattern scanner page"""

    st.title("🔍 Chart Pattern Scanner")

    st.markdown("""
    Automatically detect technical chart patterns with AI-powered analysis.
    **Patterns can signal major price moves** - catch them early!
    """)

    # Stock input
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", placeholder="e.g., AAPL, TSLA, NVDA").upper()

    with col2:
        timeframe = st.selectbox("Analysis Timeframe", ["1 Month", "3 Months", "6 Months", "1 Year"], index=1)

    with col3:
        st.write("")
        st.write("")
        scan_btn = st.button("🔍 Scan Patterns", type="primary")

    if scan_btn:
        with st.spinner(f"Scanning {ticker} for chart patterns..."):

            # Map timeframe to days
            timeframe_map = {
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365
            }

            days = timeframe_map[timeframe]

            # Fetch data with start/end dates
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            df = get_stock_data(ticker, start=start_date, end=end_date)

            if df is None or df.empty:
                st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                return

            st.success(f"✅ Analyzing {len(df)} days of price data for {ticker}")

            # Detect patterns
            patterns = PatternDetector.detect_all_patterns(df)

            # Calculate support/resistance
            levels = PatternDetector.calculate_support_resistance(df)

            # === DETECTED PATTERNS ===
            st.markdown("### 🎯 Detected Patterns")

            if patterns:
                st.info(f"🎉 **Found {len(patterns)} pattern(s)** in {ticker}")

                for i, pattern in enumerate(patterns):
                    with st.expander(f"{pattern['emoji']} {pattern['pattern']} - {pattern['type']} ({pattern['confidence']:.0f}% confidence)", expanded=(i == 0)):

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            # Pattern details card
                            signal_color = '#34C759' if pattern['signal'] == 'BUY' else '#FF3B30'

                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 6px solid {signal_color};">
                                <h2 style="margin: 0; color: {signal_color};">{pattern['signal']} Signal</h2>
                                <p style="color: #86868B; margin: 0.5rem 0;">Confidence: {pattern['confidence']:.0f}%</p>
                                <hr style="border: none; border-top: 1px solid #E5E5E7; margin: 1rem 0;">
                                <p style="margin: 0;"><strong>Current Price:</strong> ${pattern['current_price']:.2f}</p>
                                <p style="margin: 0;"><strong>Target Price:</strong> ${pattern['target_price']:.2f}</p>
                                <p style="margin: 0;"><strong>Potential Move:</strong> {((pattern['target_price'] - pattern['current_price']) / pattern['current_price'] * 100):+.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Progress bar for confidence
                            st.progress(pattern['confidence'] / 100)

                        with col2:
                            # Pattern description
                            st.markdown(f"**📖 Pattern Description:**")
                            st.info(pattern['description'])

                            # Trading recommendation
                            if pattern['signal'] == 'BUY':
                                st.success(f"""
                                **🟢 Bullish Signal**

                                This pattern suggests potential upside. Consider:
                                - Entry: Current price ${pattern['current_price']:.2f}
                                - Target: ${pattern['target_price']:.2f} ({((pattern['target_price'] - pattern['current_price']) / pattern['current_price'] * 100):+.1f}%)
                                - Stop Loss: Below recent support
                                """)
                            else:
                                st.error(f"""
                                **🔴 Bearish Signal**

                                This pattern suggests potential downside. Consider:
                                - Current price: ${pattern['current_price']:.2f}
                                - Target: ${pattern['target_price']:.2f} ({((pattern['target_price'] - pattern['current_price']) / pattern['current_price'] * 100):+.1f}%)
                                - Protect positions or consider short
                                """)

                        st.markdown("---")

                        # Chart with pattern highlighted
                        st.markdown("**📈 Pattern Visualization:**")
                        pattern_chart = create_pattern_chart(df.tail(60), ticker, pattern)
                        st.plotly_chart(pattern_chart, use_container_width=True)

            else:
                st.warning(f"""
                ⚠️ **No classic patterns detected** in {ticker} for the selected timeframe.

                This could mean:
                - Price is in a consolidation phase
                - Pattern hasn't fully formed yet
                - Try a different timeframe
                - Check back tomorrow as patterns develop over time
                """)

            st.markdown("---")

            # === SUPPORT & RESISTANCE LEVELS ===
            st.markdown("### 📊 Support & Resistance Levels")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🔴 Resistance Levels")
                current = levels['current_price']

                for i in range(1, 4):
                    r_level = levels[f'resistance_{i}']
                    distance = ((r_level - current) / current) * 100
                    st.metric(f"R{i}", f"${r_level:.2f}", f"{distance:+.1f}%")

            with col2:
                st.markdown("#### ⚪ Pivot Point")
                pivot = levels['pivot']
                distance = ((pivot - current) / current) * 100
                st.metric("Pivot", f"${pivot:.2f}", f"{distance:+.1f}%")
                st.caption("Center reference point")

            with col3:
                st.markdown("#### 🟢 Support Levels")

                for i in range(1, 4):
                    s_level = levels[f'support_{i}']
                    distance = ((s_level - current) / current) * 100
                    st.metric(f"S{i}", f"${s_level:.2f}", f"{distance:+.1f}%")

            # Support/Resistance chart
            sr_chart = create_support_resistance_chart(df.tail(60), ticker, levels)
            st.plotly_chart(sr_chart, use_container_width=True)

            st.markdown("---")

            # === PATTERN INSIGHTS ===
            st.markdown("### 💡 Pattern Trading Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.info("""
                **📚 How to Use Chart Patterns:**

                **Pattern Confirmation:**
                - Wait for pattern to fully form
                - Look for volume confirmation
                - Check multiple timeframes
                - Combine with other indicators

                **Entry & Exit:**
                - **Reversal Patterns**: Enter on breakout/breakdown
                - **Continuation Patterns**: Enter on breakout in trend direction
                - Set stop-loss below support (long) or above resistance (short)
                - Take profit at target or trail stop

                **Risk Management:**
                - Never risk more than 2% of portfolio per trade
                - Pattern failures happen - use stops
                - Patterns work better in trending markets
                - Higher timeframes = more reliable patterns
                """)

            with col2:
                st.info("""
                **🎯 Pattern Success Rates (Historical):**

                **Reversal Patterns:**
                - Head & Shoulders: 83% success
                - Double Top/Bottom: 78% success
                - Inverse H&S: 81% success

                **Continuation Patterns:**
                - Ascending Triangle: 72% success
                - Descending Triangle: 64% success
                - Flags & Pennants: 68% success

                **Important Notes:**
                - Success = reaching target price
                - Requires proper confirmation
                - Volume is critical
                - Market conditions matter
                - Use with other analysis tools

                ⚠️ Past performance doesn't guarantee future results
                """)

            # === DISCLAIMER ===
            st.warning("""
            ⚠️ **Pattern Detection Disclaimer:**
            - Pattern detection is algorithmic and may produce false signals
            - Not all patterns lead to the expected price movement
            - Always confirm patterns with volume and other indicators
            - Use proper risk management and position sizing
            - This is for educational purposes only, not investment advice
            - Patterns are more reliable on higher timeframes and liquid stocks
            """)

    else:
        # Show info when no scan yet
        st.info("👆 Enter a stock ticker and click 'Scan Patterns' to get started!")

        st.markdown("""
        ### 🎯 What This Tool Does:

        - **🔍 Pattern Detection** - Automatically finds classic chart patterns
        - **📊 6 Pattern Types** - Head & Shoulders, Double Top/Bottom, Triangles
        - **🎯 Trading Signals** - BUY/SELL recommendations with confidence scores
        - **💰 Target Prices** - Calculate profit targets based on pattern geometry
        - **📈 Visual Charts** - See patterns highlighted on price charts
        - **🔴🟢 Support & Resistance** - Identify key price levels for trading

        ### 📈 Patterns We Detect:

        **Reversal Patterns (Trend Change):**
        1. **Head and Shoulders** ⬇️ - Bearish reversal after uptrend
        2. **Inverse Head and Shoulders** ⬆️ - Bullish reversal after downtrend
        3. **Double Top** ⬇️ - Bearish reversal (two peaks)
        4. **Double Bottom** ⬆️ - Bullish reversal (two troughs)

        **Continuation Patterns (Trend Resumes):**
        5. **Ascending Triangle** ⬆️ - Bullish continuation
        6. **Descending Triangle** ⬇️ - Bearish continuation

        ### 🟢 Best Practices:

        - **Use Multiple Timeframes** - Patterns on daily/weekly charts are stronger
        - **Confirm with Volume** - Breakouts need volume to be valid
        - **Wait for Confirmation** - Don't trade on incomplete patterns
        - **Combine with Indicators** - Use RSI, MACD to confirm signals
        - **Manage Risk** - Always use stop-losses

        ### 💡 Recommended Tickers:

        - **AAPL** (Apple) - Liquid, clean patterns
        - **TSLA** (Tesla) - High volatility, clear patterns
        - **NVDA** (NVIDIA) - Strong trends, good patterns
        - **SPY** (S&P 500 ETF) - Market-wide patterns
        """)
