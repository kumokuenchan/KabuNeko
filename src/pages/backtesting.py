"""
Page: Backtesting

Strategy backtesting page for testing trading strategies on historical data.
Supports SMA Crossover and RSI Mean Reversion strategies.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.backtesting.strategies import SMACrossover, RSIMeanReversion
from src.indicators.trend import TrendIndicators
from src.models.feature_engineering import FeatureEngineer


def render():
    """Backtesting page for trading strategies"""

    st.title("⚡ Strategy Backtesting")

    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("⚠️ Please load stock data first from the 'Stock Overview' page!")
        return

    df = st.session_state['current_data'].copy()
    ticker = st.session_state.get('current_stock', 'Stock')

    st.markdown(f"### Backtesting: **{ticker}**")

    st.info("""
    🎯 **Backtesting** lets you test trading strategies on historical data to see how they would have performed.
    This helps you evaluate strategies before risking real money!
    """)

    # Strategy selection
    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Select Trading Strategy",
            ["SMA Crossover", "RSI Mean Reversion", "Both (Compare)"]
        )

    with col2:
        initial_cash = st.number_input("Initial Investment ($)", value=10000, min_value=1000, step=1000)

    # Strategy parameters
    if strategy in ["SMA Crossover", "Both (Compare)"]:
        st.markdown("#### SMA Crossover Parameters")
        col1, col2 = st.columns(2)
        with col1:
            sma_short = st.slider("Short-term SMA", 5, 50, 20)
        with col2:
            sma_long = st.slider("Long-term SMA", 20, 200, 50)

    if strategy in ["RSI Mean Reversion", "Both (Compare)"]:
        st.markdown("#### RSI Mean Reversion Parameters")
        col1, col2 = st.columns(2)
        with col1:
            rsi_oversold = st.slider("RSI Oversold Level", 20, 40, 30)
        with col2:
            rsi_overbought = st.slider("RSI Overbought Level", 60, 80, 70)

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                from backtesting import Backtest
                from src.backtesting.metrics import PerformanceMetrics

                # Prepare data for backtesting library
                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                results_list = []

                # Run SMA Crossover
                if strategy in ["SMA Crossover", "Both (Compare)"]:
                    # Create dynamic strategy class
                    class SMACrossoverCustom(SMACrossover):
                        n1 = sma_short
                        n2 = sma_long

                    bt_sma = Backtest(data, SMACrossoverCustom, cash=initial_cash, commission=.002)
                    stats_sma = bt_sma.run()
                    results_list.append(("SMA Crossover", stats_sma))

                # Run RSI Mean Reversion
                if strategy in ["RSI Mean Reversion", "Both (Compare)"]:
                    # Create dynamic strategy class
                    class RSIMeanReversionCustom(RSIMeanReversion):
                        rsi_lower = rsi_oversold
                        rsi_upper = rsi_overbought

                    bt_rsi = Backtest(data, RSIMeanReversionCustom, cash=initial_cash, commission=.002)
                    stats_rsi = bt_rsi.run()
                    results_list.append(("RSI Mean Reversion", stats_rsi))

                # Display results
                st.success("✅ Backtest completed!")

                for strategy_name, stats in results_list:
                    st.markdown(f"### {strategy_name} Results")

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        final_value = stats['Equity Final [$]']
                        st.metric("Final Portfolio Value", f"${final_value:,.0f}")

                    with col2:
                        total_return = stats['Return [%]']
                        st.metric("Total Return", f"{total_return:.2f}%")

                    with col3:
                        sharpe = stats.get('Sharpe Ratio', 0)
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                    with col4:
                        max_dd = stats['Max. Drawdown [%]']
                        st.metric("Max Drawdown", f"{max_dd:.2f}%")

                    # Additional metrics
                    with st.expander(f"📊 Detailed Metrics - {strategy_name}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Number of Trades**: {stats['# Trades']}")
                            st.write(f"**Win Rate**: {stats['Win Rate [%]']:.2f}%")
                            st.write(f"**Best Trade**: {stats['Best Trade [%]']:.2f}%")
                            st.write(f"**Worst Trade**: {stats['Worst Trade [%]']:.2f}%")

                        with col2:
                            st.write(f"**Avg Trade**: {stats['Avg. Trade [%]']:.2f}%")
                            st.write(f"**Max Trade Duration**: {stats['Max. Trade Duration']}")
                            st.write(f"**Avg Trade Duration**: {stats['Avg. Trade Duration']}")
                            st.write(f"**Exposure Time**: {stats['Exposure Time [%]']:.2f}%")

                    # Equity curve
                    st.markdown(f"#### Equity Curve - {strategy_name}")
                    equity_curve = stats['_equity_curve']

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve['Equity'],
                        name='Portfolio Value',
                        fill='tozeroy',
                        line=dict(color='green')
                    ))

                    fig.update_layout(
                        title=f'{strategy_name} - Portfolio Value Over Time',
                        yaxis_title='Portfolio Value ($)',
                        xaxis_title='Date',
                        template='plotly_white',
                        height=400
                    )

                    st.plotly_chart(fig, width="stretch")

                    st.markdown("---")

                # Compare strategies if both were run
                if len(results_list) == 2:
                    st.markdown("### Strategy Comparison")

                    comparison_df = pd.DataFrame({
                        'Metric': ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', '# Trades'],
                        results_list[0][0]: [
                            results_list[0][1]['Return [%]'],
                            results_list[0][1].get('Sharpe Ratio', 0),
                            results_list[0][1]['Max. Drawdown [%]'],
                            results_list[0][1]['Win Rate [%]'],
                            results_list[0][1]['# Trades']
                        ],
                        results_list[1][0]: [
                            results_list[1][1]['Return [%]'],
                            results_list[1][1].get('Sharpe Ratio', 0),
                            results_list[1][1]['Max. Drawdown [%]'],
                            results_list[1][1]['Win Rate [%]'],
                            results_list[1][1]['# Trades']
                        ]
                    })

                    st.dataframe(comparison_df, width="stretch")

                st.info("""
                💡 **How to interpret**:
                - **Total Return**: Higher is better
                - **Sharpe Ratio**: Risk-adjusted return. Above 1 is good, above 2 is excellent
                - **Max Drawdown**: Largest peak-to-valley decline. Lower is better
                - **Win Rate**: Percentage of profitable trades
                """)

            except Exception as e:
                st.error(f"❌ Error running backtest: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
