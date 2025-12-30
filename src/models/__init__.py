"""
Machine Learning Models Module

This module provides machine learning tools for stock price prediction including:
- Feature engineering for ML models
- Random Forest regressor
- LSTM neural network
- XGBoost (to be implemented)
- Model evaluation and comparison tools
"""

from .feature_engineering import (
    FeatureEngineer,
    prepare_features,
    prepare_ml_dataset
)

from .random_forest import (
    RandomForestPredictor,
    train_random_forest
)

try:
    from .lstm_model import (
        LSTMPredictor,
        train_lstm
    )
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False
    print("Warning: TensorFlow not available. LSTM model unavailable.")

from .evaluation import (
    ModelEvaluator,
    evaluate_model
)

__all__ = [
    # Feature Engineering
    'FeatureEngineer',
    'prepare_features',
    'prepare_ml_dataset',

    # Models
    'RandomForestPredictor',
    'train_random_forest',

    # Evaluation
    'ModelEvaluator',
    'evaluate_model',
]

# Add LSTM to exports if available
if HAS_LSTM:
    __all__.extend(['LSTMPredictor', 'train_lstm'])
