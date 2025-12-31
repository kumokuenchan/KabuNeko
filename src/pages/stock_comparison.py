"""
Page: Stock Comparison

Compare multiple stocks side-by-side with technical indicators, performance metrics,
and correlation analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data.fetcher import get_multiple_stocks
from src.models.feature_engineering import FeatureEngineer
from src.analysis.investment_recommendation import get_investment_recommendation


def render():
    """Compare multiple stocks side-by-side"""
    st.title("📊 Stock Comparison")

    st.markdown("""
    Compare multiple stocks to identify the best investment opportunities.
    See technical indicators, performance metrics, and correlation analysis side-by-side.
    """)

    # Input: Select stocks
    col1, col2 = st.columns([3, 1])

    with col1:
        source = st.radio("Stock Source", ["Manual Entry", "From Watchlist"], horizontal=True)

        if source == "Manual Entry":
            tickers_input = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL")
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        else:
            watchlists = st.session_state['user_watchlists']['watchlists']
            if watchlists:
                selected_wl = st.selectbox("Select Watchlist", list(watchlists.keys()))
                tickers = watchlists[selected_wl]
            else:
                st.warning("⚠️ No watchlists found! Create one in Watchlist Manager first.")
                return

    with col2:
        period = st.selectbox("Period", ["1 Month", "3 Months", "6 Months", "1 Year"], index=2)
        days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
        days = days_map[period]

    if st.button("📊 Compare Stocks", type="primary"):
        if len(tickers) < 2:
            st.error("❌ Select at least 2 stocks to compare")
            return

        with st.spinner(f"Loading comparison data for {len(tickers)} stocks..."):
            try:
                # Fetch data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                stocks_data = get_multiple_stocks(
                    tickers,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                # Calculate metrics for each stock
                comparison_data = []
                for ticker in tickers:
                    if ticker not in stocks_data or stocks_data[ticker] is None or len(stocks_data[ticker]) < 20:
                        st.warning(f"⚠️ Skipping {ticker}: Insufficient data")
                        continue

                    df = stocks_data[ticker]

                    # Prepare features
                    df_features = FeatureEngineer.prepare_features(df)

                    # Get recommendation
                    rec = get_investment_recommendation(df_features)

                    current = df['Close'].iloc[-1]
                    start_price = df['Close'].iloc[0]
                    total_return = ((current - start_price) / start_price) * 100

                    comparison_data.append({
                        'Ticker': ticker,
                        'Price': f"${current:.2f}",
                        'Return %': f"{total_return:+.2f}%",
                        'RSI': f"{df_features['RSI'].iloc[-1]:.1f}" if 'RSI' in df_features else 'N/A',
                        'Volatility': f"{rec['risk_metrics']['volatility']:.1f}%",
                        'Tech Score': rec['technical_score'],
                        'Recommendation': rec['recommendation']
                    })

                if not comparison_data:
                    st.error("❌ No valid stock data to compare")
                    return

                # Display comparison table
                st.markdown("### 📋 Comparison Table")
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, width="stretch", hide_index=True)

                # Normalized price chart
                st.markdown("### 📈 Price Performance (Normalized to 100)")
                fig = go.Figure()

                for ticker in tickers:
                    if ticker in stocks_data and stocks_data[ticker] is not None and len(stocks_data[ticker]) > 0:
                        df = stocks_data[ticker]
                        normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=normalized,
                            mode='lines',
                            name=ticker,
                            line=dict(width=2)
                        ))

                fig.update_layout(
                    yaxis_title='Normalized Price (Start = 100)',
                    xaxis_title='Date',
                    height=500,
                    template='plotly_white',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, width="stretch")

                # Two columns for correlation and technical scores
                col1, col2 = st.columns(2)

                with col1:
                    # Correlation heatmap
                    st.markdown("### 🔗 Correlation Matrix")
                    returns_data = {}
                    for ticker in tickers:
                        if ticker in stocks_data and stocks_data[ticker] is not None and len(stocks_data[ticker]) > 0:
                            returns_data[ticker] = stocks_data[ticker]['Close'].pct_change()

                    if len(returns_data) >= 2:
                        df_returns = pd.DataFrame(returns_data).dropna()
                        correlation = df_returns.corr()

                        fig_corr = go.Figure(data=go.Heatmap(
                            z=correlation.values,
                            x=correlation.columns,
                            y=correlation.columns,
                            colorscale='RdYlGn',
                            zmid=0,
                            text=correlation.values,
                            texttemplate='%{text:.2f}',
                            textfont={"size": 10},
                            colorbar=dict(title="Correlation")
                        ))
                        fig_corr.update_layout(
                            height=400,
                            template='plotly_white',
                            xaxis=dict(side="bottom")
                        )
                        st.plotly_chart(fig_corr, width="stretch")

                with col2:
                    # Technical scores bar chart
                    st.markdown("### 🎯 Technical Scores")
                    scores = {item['Ticker']: item['Tech Score'] for item in comparison_data}

                    fig_scores = go.Figure(data=[
                        go.Bar(
                            x=list(scores.keys()),
                            y=list(scores.values()),
                            marker=dict(
                                color=list(scores.values()),
                                colorscale='RdYlGn',
                                cmin=0,
                                cmax=100,
                                colorbar=dict(title="Score")
                            ),
                            text=list(scores.values()),
                            textposition='auto'
                        )
                    ])
                    fig_scores.update_layout(
                        yaxis_title='Technical Score (0-100)',
                        xaxis_title='Stock',
                        height=400,
                        template='plotly_white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_scores, width="stretch")

                # Export comparison
                st.markdown("---")
                csv = df_comparison.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison as CSV",
                    data=csv,
                    file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"❌ Error comparing stocks: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    else:
        st.info("👆 Select stocks and click 'Compare Stocks' to get started!")

        st.markdown("### 💡 Comparison Features:")
        st.markdown("""
        - **Side-by-Side Metrics**: Compare price, returns, RSI, volatility, and technical scores
        - **Normalized Performance**: See which stock performed best over the period
        - **Correlation Matrix**: Understand how stocks move together
        - **Technical Scores**: Quick visual comparison of technical strength
        - **Export Data**: Download comparison table as CSV
        """)
