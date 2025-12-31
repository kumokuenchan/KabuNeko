"""
Page: Price Alerts

Manage price alerts for stocks with conditions like price above/below,
RSI oversold/overbought, and MACD crossovers.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path


# User data directory setup
USER_DATA_DIR = Path("data/user_data")
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json_data(filename: str, default: dict) -> dict:
    """Load JSON data from user_data directory"""
    file_path = USER_DATA_DIR / filename
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading {filename}: {e}")
            return default
    return default


def save_json_data(filename: str, data: dict) -> bool:
    """Save JSON data to user_data directory"""
    file_path = USER_DATA_DIR / filename
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving {filename}: {e}")
        return False


def render():
    """Manage price alerts"""
    st.title("🔔 Price Alerts")

    st.markdown("""
    Set alerts to get notified when stocks hit your target prices or technical conditions.
    Alerts are checked when you visit the dashboard.
    """)

    # Create alert form
    st.markdown("### ➕ Create New Alert")
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        alert_ticker = st.text_input("Ticker", key="alert_ticker", placeholder="e.g., AAPL").upper()
    with col2:
        alert_type = st.selectbox("Condition", [
            "Price Above",
            "Price Below",
            "RSI Oversold",
            "RSI Overbought",
            "MACD Bullish Cross"
        ], key="alert_type")
    with col3:
        if "Price" in alert_type:
            alert_value = st.number_input("Price Target", min_value=0.0, value=100.0, key="alert_value")
        else:
            alert_value = st.number_input("Threshold", min_value=0, max_value=100, value=30, key="alert_threshold")
    with col4:
        st.write("")  # Spacing
        st.write("")  # More spacing
        if st.button("Create Alert", type="primary"):
            if not alert_ticker:
                st.error("Please enter a ticker symbol")
            else:
                alerts_list = st.session_state['user_alerts']['alerts']
                alerts_list.append({
                    'id': f"alert_{len(alerts_list) + 1}_{int(datetime.now().timestamp())}",
                    'ticker': alert_ticker,
                    'condition': alert_type.lower().replace(' ', '_'),
                    'threshold': alert_value,
                    'active': True,
                    'created_at': datetime.now().isoformat(),
                    'triggered_at': None
                })
                save_json_data('alerts.json', st.session_state['user_alerts'])
                st.success(f"✅ Alert created for {alert_ticker}")
                st.rerun()

    st.markdown("---")

    # Display active alerts
    st.markdown("### 📋 Active Alerts")
    alerts_list = st.session_state['user_alerts']['alerts']
    active_alerts = [a for a in alerts_list if a['active']]

    if active_alerts:
        for alert in active_alerts:
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col1:
                st.write(f"**{alert['ticker']}**")
            with col2:
                condition_display = alert['condition'].replace('_', ' ').title()
                st.write(f"{condition_display}")
            with col3:
                st.write(f"Threshold: {alert['threshold']}")
            with col4:
                if st.button("🗑️", key=f"del_{alert['id']}"):
                    alerts_list.remove(alert)
                    save_json_data('alerts.json', st.session_state['user_alerts'])
                    st.success("Alert deleted")
                    st.rerun()

    else:
        st.info("📭 No active alerts. Create one above!")

    # Show triggered alerts (in last 24 hours)
    triggered_alerts = [a for a in alerts_list if a.get('triggered_at')]
    if triggered_alerts:
        # Filter to last 24 hours
        recent_triggered = []
        for alert in triggered_alerts:
            try:
                triggered_time = datetime.fromisoformat(alert['triggered_at'])
                if (datetime.now() - triggered_time).total_seconds() < 86400:  # 24 hours
                    recent_triggered.append(alert)
            except:
                pass

        if recent_triggered:
            st.markdown("---")
            st.markdown("### 🔔 Recently Triggered (Last 24h)")
            for alert in recent_triggered:
                triggered_time = datetime.fromisoformat(alert['triggered_at'])
                time_ago = datetime.now() - triggered_time
                hours_ago = int(time_ago.total_seconds() / 3600)

                condition_display = alert['condition'].replace('_', ' ').title()
                st.success(f"✅ **{alert['ticker']}**: {condition_display} (threshold: {alert['threshold']}) - {hours_ago}h ago")

    st.markdown("---")
    st.markdown("### 💡 How Alerts Work")
    st.markdown("""
    - Alerts are checked automatically when you visit the dashboard
    - Active alerts show in the sidebar when triggered
    - Triggered alerts are marked and shown for 24 hours
    - Set multiple alerts for different conditions on the same stock
    """)
