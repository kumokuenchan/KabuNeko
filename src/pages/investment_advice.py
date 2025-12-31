"""
Page: Investment Advice

AI-powered investment recommendations with technical analysis, risk assessment,
and price targets.
"""

import streamlit as st
from datetime import datetime, timedelta
from src.data.fetcher import get_stock_data
from src.models.feature_engineering import FeatureEngineer
from src.models.random_forest import RandomForestPredictor
from src.analysis.investment_recommendation import get_investment_recommendation


def render():
    """Render the investment advice page"""

    st.title("💡 Investment Advice - Should You Buy?")

    st.markdown("""
    Get an intelligent recommendation based on:
    - 📊 Technical indicators (trend, momentum, volatility)
    - 🤖 AI price predictions
    - ⚠️ Risk assessment
    - 🎯 Entry/exit price targets
    """)

    # Input controls
    col1, col2 = st.columns([2, 2])

    with col1:
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value=st.session_state.get('current_stock', 'AAPL'),
            help="Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()

    with col2:
        use_ml = st.checkbox(
            "Include AI Prediction",
            value=True,
            help="Use machine learning to improve recommendation accuracy"
        )

    if st.button("🔍 Analyze Stock", type="primary"):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Fetch stock data (1 year for ML training)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)

                df = get_stock_data(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                if df is None or len(df) < 60:
                    st.error(f"❌ Insufficient data for {ticker}. Need at least 60 days of history.")
                    return

                # Prepare features
                engineer = FeatureEngineer()
                df_features = engineer.prepare_features(df)

                # ML prediction (optional)
                ml_prediction = None
                ml_confidence = None

                if use_ml and len(df_features) >= 100:
                    try:
                        with st.spinner("Training AI model..."):
                            # Prepare ML dataset
                            X, y, scaler = engineer.prepare_ml_dataset(
                                df,
                                target_column='Close',
                                forecast_horizon=1,
                                target_type='price'
                            )

                            if len(X) > 50:
                                # Train model
                                model = RandomForestPredictor()
                                model.train(X, y)

                                # Get metrics
                                metrics = model.get_metrics(X, y)
                                ml_confidence = metrics.get('r2', 0)

                                # Predict next day
                                ml_prediction = model.predict(X.tail(1))[0]

                                st.success(f"✅ AI model trained (R² = {ml_confidence:.2%})")
                    except Exception as e:
                        st.warning(f"⚠️ Could not run AI prediction: {str(e)}")

                # Get investment recommendation
                recommendation = get_investment_recommendation(
                    df_features,
                    ml_prediction=ml_prediction,
                    ml_confidence=ml_confidence
                )

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                # Main recommendation card
                rec_color_map = {
                    'STRONG BUY': '#00ff00',
                    'BUY': '#90ee90',
                    'HOLD': '#ffff00',
                    'SELL': '#ffcccb',
                    'STRONG SELL': '#ff0000'
                }

                rec_color = rec_color_map.get(recommendation['recommendation'], '#cccccc')

                st.markdown(f"""
                <div style="background-color: {rec_color}; padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0;">
                    <h1 style="margin: 0; color: #000;">{recommendation['color']} {recommendation['recommendation']}</h1>
                    <h2 style="margin: 0.5rem 0; color: #000;">Score: {recommendation['overall_score']}/100</h2>
                    <p style="font-size: 1.2rem; margin: 0; color: #000;">{recommendation['action']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Details in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🎯 Price Targets")
                    targets = recommendation['price_targets']

                    st.metric("Current Price", f"${targets['current_price']}")
                    st.metric("Recommended Entry", f"${targets['entry_price']}",
                             delta=f"{((targets['entry_price']/targets['current_price']-1)*100):.1f}%")
                    st.metric("Stop Loss", f"${targets['stop_loss']}",
                             delta=f"{((targets['stop_loss']/targets['current_price']-1)*100):.1f}%")
                    st.metric("Price Target", f"${targets['target_price']}",
                             delta=f"{((targets['target_price']/targets['current_price']-1)*100):.1f}%")

                    st.markdown("---")
                    st.markdown(f"**Support Level:** ${targets['support']}")
                    st.markdown(f"**Resistance Level:** ${targets['resistance']}")

                with col2:
                    st.markdown("### ⚠️ Risk Assessment")
                    risk = recommendation['risk_metrics']

                    risk_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    st.markdown(f"**Risk Level:** :{risk_color[risk['risk_level']]}[{risk['risk_level']}]")
                    st.markdown(f"**Volatility (Annual):** {risk['volatility']:.1f}%")
                    st.markdown(f"**Max Drawdown:** {risk['max_drawdown']:.1f}%")

                    st.markdown("---")

                    st.markdown("### 📈 Technical Score")
                    st.progress(recommendation['technical_score'] / 100)
                    st.markdown(f"**{recommendation['technical_score']}/100**")

                # Detailed signals
                st.markdown("### 📋 Detailed Analysis")

                for detail in recommendation['details']:
                    if '✅' in detail:
                        st.success(detail)
                    elif '❌' in detail:
                        st.error(detail)
                    elif '⚠️' in detail:
                        st.warning(detail)
                    else:
                        st.info(detail)

                # Disclaimer
                st.markdown("---")
                st.warning(recommendation['disclaimer'])

                st.caption(f"Analysis generated at: {recommendation['timestamp']}")

            except Exception as e:
                st.error(f"❌ Error analyzing {ticker}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    else:
        # Show example when no analysis run yet
        st.info("👆 Enter a stock ticker and click 'Analyze Stock' to get investment advice!")

        st.markdown("### How It Works:")
        st.markdown("""
        1. **Technical Analysis**: Examines trend (SMA), momentum (RSI, MACD), volatility (Bollinger Bands), and volume
        2. **AI Prediction**: Machine learning model predicts next-day price movement
        3. **Risk Assessment**: Calculates volatility and maximum drawdown
        4. **Smart Scoring**: Combines all factors into a 0-100 score
        5. **Actionable Advice**: Provides clear BUY/SELL/HOLD recommendation with price targets

        **Scoring Breakdown:**
        - 70-100: STRONG BUY - Multiple positive signals
        - 55-69: BUY - Good opportunity
        - 45-54: HOLD - Wait for clarity
        - 30-44: SELL - Negative signals present
        - 0-29: STRONG SELL - Avoid or exit
        """)
