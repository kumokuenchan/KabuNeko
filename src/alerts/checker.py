"""
Price Alerts Checker Module

Checks active price alerts and triggers notifications when conditions are met.
"""

import streamlit as st
from datetime import datetime, timedelta
from src.data.fetcher import get_stock_data
from src.data.persistence import save_json_data
from src.models.feature_engineering import FeatureEngineer
from src.indicators.trend import TrendIndicators


def check_price_alerts():
    """Check active alerts and show triggered ones in sidebar"""
    alerts_list = st.session_state.get('user_alerts', {}).get('alerts', [])
    if not alerts_list:
        return

    active_alerts = [a for a in alerts_list if a.get('active', False) and not a.get('triggered_at')]

    if not active_alerts:
        return

    triggered_count = 0

    for alert in active_alerts:
        try:
            # Fetch recent data (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df = get_stock_data(
                alert['ticker'],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )

            if df is None or len(df) == 0:
                continue

            triggered = False

            if alert['condition'] == 'price_above':
                if df['Close'].iloc[-1] > alert['threshold']:
                    triggered = True

            elif alert['condition'] == 'price_below':
                if df['Close'].iloc[-1] < alert['threshold']:
                    triggered = True

            elif alert['condition'] == 'rsi_oversold':
                df_features = FeatureEngineer.prepare_features(df)
                if 'RSI' in df_features.columns:
                    rsi = df_features['RSI'].iloc[-1]
                    if rsi < alert['threshold']:
                        triggered = True

            elif alert['condition'] == 'rsi_overbought':
                df_features = FeatureEngineer.prepare_features(df)
                if 'RSI' in df_features.columns:
                    rsi = df_features['RSI'].iloc[-1]
                    if rsi > alert['threshold']:
                        triggered = True

            elif alert['condition'] == 'macd_bullish_cross':
                macd_df = TrendIndicators.macd(df)
                if macd_df.iloc[-1, 0] > macd_df.iloc[-1, 1]:  # MACD > Signal
                    triggered = True

            if triggered:
                alert['triggered_at'] = datetime.now().isoformat()
                alert['active'] = False  # Deactivate after triggering
                triggered_count += 1

        except Exception as e:
            # Skip alerts that error
            continue

    if triggered_count > 0:
        save_json_data('alerts.json', st.session_state['user_alerts'])
        st.sidebar.success(f"🔔 {triggered_count} Alert(s) Triggered! Check Price Alerts page.")
