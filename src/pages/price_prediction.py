"""
Page: Price Prediction

AI-powered price prediction using machine learning models. Trains on historical data
to forecast future prices.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from src.models.random_forest import RandomForestPredictor
from src.models.feature_engineering import FeatureEngineer


def render():
    """Price prediction page using ML"""

    st.title("🤖 AI Price Prediction")

    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("⚠️ Please load stock data first from the 'Stock Overview' page!")
        return

    df = st.session_state['current_data'].copy()
    ticker = st.session_state.get('current_stock', 'Stock')

    st.markdown(f"### Predicting: **{ticker}**")

    st.info("""
    📊 **How it works**: This AI model uses historical price patterns, technical indicators,
    and statistical features to predict future prices. Remember, predictions are estimates and
    should not be used as the sole basis for investment decisions!
    """)

    col1, col2 = st.columns(2)

    with col1:
        prediction_days = st.slider("Predict how many days ahead?", 1, 30, 5)

    with col2:
        model_type = st.selectbox("AI Model", ["Random Forest (Recommended)", "Linear Regression"])

    if st.button("🚀 Generate Prediction", type="primary"):
        with st.spinner("Training AI model... This may take a minute..."):
            try:
                # Prepare features and dataset (all-in-one)
                engineer = FeatureEngineer()
                X, y, scaler = engineer.prepare_ml_dataset(
                    df,
                    target_column='Close',
                    forecast_horizon=1,
                    target_type='price'
                )

                # Train model
                model = RandomForestPredictor(n_estimators=100, random_state=42)
                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                model.fit(X_train, y_train)

                # Make predictions
                predictions = model.predict(X_test)

                # Calculate metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                import numpy as np

                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)

                st.success("✅ Model trained successfully!")

                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                with col2:
                    st.metric("Root Mean Squared Error", f"${rmse:.2f}")
                with col3:
                    st.metric("R² Score", f"{r2:.3f}")

                # Plot predictions vs actual
                st.markdown("### Prediction vs Actual Prices")

                # Calculate proper test dates (account for NaN rows dropped during feature creation)
                test_dates = df.index[-(len(X) - train_size):][:len(y_test)]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=y_test, name='Actual Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_dates, y=predictions, name='Predicted Price', line=dict(color='red', dash='dash')))

                fig.update_layout(
                    title=f'{ticker} - AI Predictions vs Actual',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    template='plotly_white',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Future prediction
                st.markdown(f"### {prediction_days}-Day Future Forecast")

                # Use last known values for future prediction
                last_features = X_test.iloc[-1:].values
                future_predictions = []

                for i in range(prediction_days):
                    pred = model.predict(last_features)[0]
                    future_predictions.append(pred)

                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)

                # Plot future forecast
                fig_future = go.Figure()

                # Historical prices (last 30 days)
                fig_future.add_trace(go.Scatter(
                    x=df.index[-30:],
                    y=df['Close'].tail(30),
                    name='Historical Price',
                    line=dict(color='blue')
                ))

                # Future predictions
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    name='Forecast',
                    line=dict(color='red', dash='dash'),
                    mode='lines+markers'
                ))

                fig_future.update_layout(
                    title=f'{ticker} - {prediction_days}-Day Price Forecast',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    template='plotly_white',
                    height=400
                )

                st.plotly_chart(fig_future, use_container_width=True)

                # Show prediction values
                st.markdown("#### Predicted Prices")
                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })
                prediction_df['Change from Today'] = prediction_df['Predicted Price'] - df['Close'].iloc[-1]
                prediction_df['% Change'] = (prediction_df['Change from Today'] / df['Close'].iloc[-1]) * 100

                st.dataframe(prediction_df.style.format({
                    'Predicted Price': '${:.2f}',
                    'Change from Today': '${:.2f}',
                    '% Change': '{:.2f}%'
                }), use_container_width=True)

                # Feature importance
                with st.expander("📊 Feature Importance - What the AI Looks At"):
                    importance = model.get_feature_importance()
                    if importance is not None:
                        st.bar_chart(importance.head(10))

                st.warning("""
                ⚠️ **Disclaimer**: These predictions are for educational purposes only.
                Stock markets are inherently unpredictable. Always do your own research and
                consult with a financial advisor before making investment decisions.
                """)

            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("💡 Tip: Make sure you have at least 60 days of historical data loaded.")
