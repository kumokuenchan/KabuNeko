"""
Data Persistence Module

Handles loading and saving user data (watchlists, alerts, performance tracking)
to JSON files in the data/user_data directory.
"""

import streamlit as st
import json
from pathlib import Path

# User data directory
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


def initialize_user_data():
    """Initialize all user data in session state on app start"""
    if 'user_watchlists' not in st.session_state:
        st.session_state['user_watchlists'] = load_json_data(
            'watchlists.json',
            {'watchlists': {}, 'last_updated': None}
        )

    if 'user_alerts' not in st.session_state:
        st.session_state['user_alerts'] = load_json_data(
            'alerts.json',
            {'alerts': [], 'last_checked': None}
        )

    if 'performance_data' not in st.session_state:
        st.session_state['performance_data'] = load_json_data(
            'performance_tracker.json',
            {'trades': [], 'statistics': {}}
        )

    if 'dark_mode' not in st.session_state:
        preferences = load_json_data('preferences.json', {'dark_mode': False})
        st.session_state['dark_mode'] = preferences.get('dark_mode', False)
