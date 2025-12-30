"""
Unit tests for ML models
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.feature_engineering import FeatureEngineer
from src.models.random_forest import RandomForestPredictor
from src.models.evaluation import ModelEvaluator


class TestFeatureEngineering:
    """Tests for feature engineering"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        close_prices = 100 + np.cumsum(np.random.randn(300) * 2)
        data = {
            'Open': close_prices - np.random.uniform(0, 2, 300),
            'High': close_prices + np.random.uniform(0, 3, 300),
            'Low': close_prices - np.random.uniform(0, 3, 300),
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 10000000, 300)
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_create_lagged_features(self, sample_data):
        """Test lagged feature creation"""
        result = FeatureEngineer.create_lagged_features(
            sample_data,
            column='Close',
            lags=[1, 2, 3, 5]
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)

        # Check lagged columns exist
        for lag in [1, 2, 3, 5]:
            assert f'Close_Lag_{lag}' in result.columns

    def test_create_rolling_features(self, sample_data):
        """Test rolling statistical features"""
        result = FeatureEngineer.create_rolling_features(
            sample_data,
            column='Close',
            windows=[5, 10, 20]
        )

        assert result is not None

        # Check for rolling mean columns
        for window in [5, 10, 20]:
            assert f'Close_MA_{window}' in result.columns
            assert f'Close_STD_{window}' in result.columns

    def test_create_technical_features(self, sample_data):
        """Test technical indicator features"""
        result = FeatureEngineer.create_technical_features(sample_data)

        assert result is not None

        # Should have various technical indicators
        assert 'SMA_20' in result.columns or 'RSI' in result.columns

    def test_prepare_ml_dataset(self, sample_data):
        """Test complete ML dataset preparation"""
        X, y, scaler = FeatureEngineer.prepare_ml_dataset(
            sample_data,
            target_column='Close',
            forecast_horizon=1,
            target_type='price'
        )

        assert X is not None
        assert y is not None
        assert scaler is not None

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        # Should have features
        assert X.shape[1] > 0
        # X and y should have same length
        assert len(X) == len(y)


class TestRandomForestPredictor:
    """Tests for Random Forest model"""

    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML dataset"""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Create target with some correlation to features
        y = pd.Series(
            X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
        )

        return X, y

    def test_model_initialization(self):
        """Test model can be initialized"""
        model = RandomForestPredictor(n_estimators=10, random_state=42)

        assert model is not None
        assert model.n_estimators == 10
        assert model.random_state == 42

    def test_model_fitting(self, sample_ml_data):
        """Test model can be fitted"""
        X, y = sample_ml_data

        model = RandomForestPredictor(n_estimators=10, random_state=42)
        model.fit(X, y)

        assert model.is_fitted
        assert model.feature_names is not None

    def test_model_prediction(self, sample_ml_data):
        """Test model can make predictions"""
        X, y = sample_ml_data

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train and predict
        model = RandomForestPredictor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_feature_importance(self, sample_ml_data):
        """Test feature importance extraction"""
        X, y = sample_ml_data

        model = RandomForestPredictor(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == X.shape[1]
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

    def test_model_evaluation(self, sample_ml_data):
        """Test model evaluation"""
        X, y = sample_ml_data

        model = RandomForestPredictor(n_estimators=10, random_state=42)
        model.fit(X, y)

        metrics = model.evaluate(X, y)

        assert metrics is not None
        assert isinstance(metrics, dict)
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics


class TestModelEvaluator:
    """Tests for model evaluation tools"""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions"""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.randn(n_samples).cumsum() + 100
        y_pred = y_true + np.random.randn(n_samples) * 2

        return y_true, y_pred

    def test_regression_metrics(self, sample_predictions):
        """Test regression metrics calculation"""
        y_true, y_pred = sample_predictions

        metrics = ModelEvaluator.calculate_regression_metrics(y_true, y_pred)

        assert metrics is not None
        assert isinstance(metrics, dict)
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert 'MAPE' in metrics

        # MAE and RMSE should be positive
        assert metrics['MAE'] > 0
        assert metrics['RMSE'] > 0
        # R2 should be between -inf and 1 (typically)
        assert metrics['R2'] <= 1

    def test_directional_accuracy(self, sample_predictions):
        """Test directional accuracy calculation"""
        y_true, y_pred = sample_predictions

        accuracy = ModelEvaluator.calculate_directional_accuracy(y_true, y_pred)

        assert accuracy is not None
        assert isinstance(accuracy, float)
        # Accuracy should be between 0 and 100
        assert 0 <= accuracy <= 100

    def test_profit_loss_calculation(self, sample_predictions):
        """Test P/L calculation"""
        y_true, y_pred = sample_predictions

        pl_metrics = ModelEvaluator.calculate_profit_loss(
            y_true, y_pred,
            initial_capital=10000,
            commission=0.002
        )

        assert pl_metrics is not None
        assert isinstance(pl_metrics, dict)
        assert 'Final_Capital' in pl_metrics
        assert 'Total_Return_%' in pl_metrics
        assert 'Trades' in pl_metrics

        # Final capital should be positive
        assert pl_metrics['Final_Capital'] > 0

    def test_compare_models(self, sample_predictions):
        """Test model comparison"""
        y_true, y_pred = sample_predictions

        # Create slightly different prediction
        y_pred2 = y_true + np.random.randn(len(y_true)) * 1.5

        comparison = ModelEvaluator.compare_models({
            'Model1': (y_true, y_pred),
            'Model2': (y_true, y_pred2)
        })

        assert comparison is not None
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'MAE' in comparison.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
