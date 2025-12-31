"""
Page: Technical Analysis

Technical analysis page with indicators including SMA, RSI, MACD, Bollinger Bands,
Stochastic, ATR, OBV, and VWAP.
"""

import streamlit as st
import plotly.graph_objects as go
from src.indicators.trend import TrendIndicators
from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators


def render():
    """Technical analysis page with indicators"""

    st.title("📉 Technical Analysis")

    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("⚠️ Please load stock data first from the 'Stock Overview' page!")
        return

    df = st.session_state['current_data'].copy()
    ticker = st.session_state.get('current_stock', 'Stock')

    st.markdown(f"### Analyzing: **{ticker}**")

    # Indicator selection
    st.markdown("#### Select Indicators to Display")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_sma = st.checkbox("Moving Averages (SMA)", value=True)
        show_rsi = st.checkbox("RSI", value=True)

    with col2:
        show_macd = st.checkbox("MACD", value=False)
        show_bb = st.checkbox("Bollinger Bands", value=False)

    with col3:
        show_stoch = st.checkbox("Stochastic", value=False)
        show_atr = st.checkbox("ATR", value=False)

    with col4:
        show_obv = st.checkbox("OBV", value=False)
        show_vwap = st.checkbox("VWAP", value=False)

    # Indicator classes (all methods are static)
    # No need to instantiate, but keeping for clarity

    # Price chart with indicators
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

    # Add selected indicators
    if show_sma:
        sma_20 = TrendIndicators.sma(df, period=20)
        sma_50 = TrendIndicators.sma(df, period=50)
        fig.add_trace(go.Scatter(x=df.index, y=sma_20, name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=sma_50, name='SMA 50', line=dict(color='blue')))

    if show_bb:
        bb_df = VolatilityIndicators.bollinger_bands(df)
        # Extract columns (they're named BBL_20_2.0, BBM_20_2.0, BBU_20_2.0)
        bb_lower = bb_df.iloc[:, 0]  # First column is lower
        bb_middle = bb_df.iloc[:, 1]  # Second column is middle
        bb_upper = bb_df.iloc[:, 2]  # Third column is upper
        fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name='BB Upper', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name='BB Lower', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=bb_middle, name='BB Middle', line=dict(color='purple')))

    if show_vwap:
        vwap = VolumeIndicators.vwap(df)
        fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='green', dash='dot')))

    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Additional indicator charts
    if show_rsi:
        st.markdown("#### RSI (Relative Strength Index)")
        rsi = MomentumIndicators.rsi(df)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")

        fig_rsi.update_layout(
            yaxis_title='RSI',
            xaxis_title='Date',
            template='plotly_white',
            height=250,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig_rsi, use_container_width=True)

        # RSI interpretation
        current_rsi = rsi.iloc[-1]
        if current_rsi > 70:
            st.warning(f"📊 RSI is {current_rsi:.1f} - Stock may be **overbought** (potential sell signal)")
        elif current_rsi < 30:
            st.success(f"📊 RSI is {current_rsi:.1f} - Stock may be **oversold** (potential buy signal)")
        else:
            st.info(f"📊 RSI is {current_rsi:.1f} - Stock is in **neutral** territory")

    if show_macd:
        st.markdown("#### MACD (Moving Average Convergence Divergence)")
        macd_df = TrendIndicators.macd(df)
        # Extract columns (MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9)
        macd = macd_df.iloc[:, 0]  # MACD line
        signal = macd_df.iloc[:, 1]  # Signal line
        histogram = macd_df.iloc[:, 2]  # Histogram

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram', marker_color='lightgray'))

        fig_macd.update_layout(
            yaxis_title='MACD',
            xaxis_title='Date',
            template='plotly_white',
            height=250
        )

        st.plotly_chart(fig_macd, use_container_width=True)

    if show_stoch:
        st.markdown("#### Stochastic Oscillator")
        stoch_df = MomentumIndicators.stochastic(df)
        # Extract columns (STOCHk_14_3, STOCHd_14_3)
        stoch_k = stoch_df.iloc[:, 0]  # %K
        stoch_d = stoch_df.iloc[:, 1]  # %D

        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=df.index, y=stoch_k, name='%K', line=dict(color='blue')))
        fig_stoch.add_trace(go.Scatter(x=df.index, y=stoch_d, name='%D', line=dict(color='red')))
        fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")

        fig_stoch.update_layout(
            yaxis_title='Stochastic',
            xaxis_title='Date',
            template='plotly_white',
            height=250,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig_stoch, use_container_width=True)

    if show_atr:
        st.markdown("#### ATR (Average True Range) - Volatility")
        atr = VolatilityIndicators.atr(df)

        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df.index, y=atr, name='ATR', line=dict(color='orange')))

        fig_atr.update_layout(
            yaxis_title='ATR',
            xaxis_title='Date',
            template='plotly_white',
            height=250
        )

        st.plotly_chart(fig_atr, use_container_width=True)

    if show_obv:
        st.markdown("#### OBV (On-Balance Volume)")
        obv = VolumeIndicators.obv(df)

        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(x=df.index, y=obv, name='OBV', line=dict(color='teal')))

        fig_obv.update_layout(
            yaxis_title='OBV',
            xaxis_title='Date',
            template='plotly_white',
            height=250
        )

        st.plotly_chart(fig_obv, use_container_width=True)
