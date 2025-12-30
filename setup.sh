#!/bin/bash
# Setup script for Stock Analysis Toolkit (macOS/Linux)

set -e

echo "======================================================================"
echo "Stock Analysis Toolkit - Setup Script"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 1: Creating virtual environment..."
echo "======================================================================"

# Create virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✅ Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

echo ""
echo "======================================================================"
echo "Step 2: Activating virtual environment..."
echo "======================================================================"

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

echo ""
echo "======================================================================"
echo "Step 3: Upgrading pip..."
echo "======================================================================"

# Upgrade pip
pip install --upgrade pip
echo "✅ pip upgraded"

echo ""
echo "======================================================================"
echo "Step 4: Installing dependencies..."
echo "======================================================================"
echo "This may take several minutes..."
echo ""

# Install requirements
if pip install -r requirements.txt; then
    echo ""
    echo "✅ Core dependencies installed!"
else
    echo ""
    echo "❌ Some dependencies failed to install"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Optional: Installing TensorFlow (for LSTM models)..."
echo "======================================================================"
echo "Note: TensorFlow may not support Python 3.14+"
echo "Attempting installation..."
echo ""

# Try to install optional dependencies
if pip install -r requirements-optional.txt 2>/dev/null; then
    echo "✅ Optional dependencies (TensorFlow) installed!"
else
    echo "⚠️  TensorFlow installation skipped (not compatible with your Python version)"
    echo "   LSTM models won't work, but everything else will!"
    echo "   Random Forest models are still available."
fi

echo ""
echo "======================================================================"
echo "Step 5: Verifying installation..."
echo "======================================================================"
echo ""

# Run verification
python verify_installation.py

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "To use the toolkit:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Launch Jupyter: jupyter notebook"
echo "  3. Open: notebooks/01_data_collection.ipynb"
echo ""
echo "To run tests:"
echo "  python run_tests.py"
echo ""
echo "======================================================================"
