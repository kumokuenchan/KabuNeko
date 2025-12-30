#!/bin/bash
# Simple launcher for Stock Analysis Dashboard
# Double-click this file to start the web interface!

echo "======================================================================"
echo "Stock Analysis Dashboard - Launcher"
echo "======================================================================"
echo ""
echo "Starting the web dashboard..."
echo "Your browser will open automatically in a few seconds."
echo ""
echo "When you see 'You can now view your Streamlit app in your browser'"
echo "the dashboard is ready!"
echo ""
echo "To stop the dashboard: Press Control+C in this window"
echo "======================================================================"
echo ""

# Change to the script's directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found!"
    echo "Please run setup.sh first:"
    echo "    ./setup.sh"
    exit 1
fi

# Launch Streamlit
echo ""
echo "🚀 Launching dashboard..."
echo ""

streamlit run app.py

# Deactivate when done
deactivate
