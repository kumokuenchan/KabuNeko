"""
Random Forest Model for Stock Price Prediction

This module provides a Random Forest implementation for predicting
stock prices with hyperparameter tuning and feature importance analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


class RandomForestPredictor:
    """
    Random Forest model for stock price prediction.
    """

    def __init__(self, task_type: str = 'regression',
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42):
        """
        Initialize Random Forest model.

        Args:
            task_type (str): 'regression' or 'classification'
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples to split
            min_samples_leaf (int): Minimum samples per leaf
            random_state (int): Random seed

        Example:
            >>> model = RandomForestPredictor(n_estimators=100)
        """
        self.task_type = task_type
        self.random_state = random_state

        if task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
        elif task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestPredictor':
        """
        Train the Random Forest model.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable

        Returns:
            self: Fitted model

        Example:
            >>> model.fit(X_train, y_train)
        """
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.ndarray: Predictions

        Example:
            >>> predictions = model.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n (int): Return top N features (None = all)

        Returns:
            pd.DataFrame: Feature importance scores

        Example:
            >>> importance = model.get_feature_importance(top_n=10)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                            param_grid: Optional[Dict] = None,
                            cv_splits: int = 5) -> Dict:
        """
        Tune hyperparameters using Grid Search with Time Series Cross-Validation.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            param_grid (dict): Parameter grid for search
            cv_splits (int): Number of CV splits

        Returns:
            Dict: Best parameters

        Example:
            >>> best_params = model.tune_hyperparameters(X_train, y_train)
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        # Use TimeSeriesSplit for time-series data
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error' if self.task_type == 'regression' else 'accuracy',
            n_jobs=-1,
            verbose=1
        )

        print("Starting hyperparameter tuning...")
        grid_search.fit(X, y)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best score: {-grid_search.best_score_:.4f}" if self.task_type == 'regression'
              else f"Best accuracy: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.feature_names = X.columns.tolist()
        self.is_fitted = True

        return grid_search.best_params_

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True values

        Returns:
            Dict: Evaluation metrics

        Example:
            >>> metrics = model.evaluate(X_test, y_test)
        """
        predictions = self.predict(X)

        if self.task_type == 'regression':
            metrics = {
                'MAE': mean_absolute_error(y, predictions),
                'RMSE': np.sqrt(mean_squared_error(y, predictions)),
                'R2': r2_score(y, predictions),
                'MAPE': np.mean(np.abs((y - predictions) / y)) * 100
            }
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            metrics = {
                'Accuracy': accuracy_score(y, predictions),
                'Precision': precision_score(y, predictions, average='weighted'),
                'Recall': recall_score(y, predictions, average='weighted'),
                'F1': f1_score(y, predictions, average='weighted')
            }

        return metrics

    def save_model(self, filepath: str):
        """
        Save model to disk.

        Args:
            filepath (str): Path to save model

        Example:
            >>> model.save_model('models/saved_models/rf_model.pkl')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'task_type': self.task_type
        }, filepath)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath (str): Path to saved model

        Example:
            >>> model.load_model('models/saved_models/rf_model.pkl')
        """
        saved_data = joblib.load(filepath)

        self.model = saved_data['model']
        self.feature_names = saved_data['feature_names']
        self.task_type = saved_data['task_type']
        self.is_fitted = True

        print(f"Model loaded from {filepath}")

    def get_prediction_intervals(self, X: pd.DataFrame,
                                 percentile: float = 95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction intervals using quantile regression with trees.

        Args:
            X (pd.DataFrame): Features
            percentile (float): Confidence percentile (e.g., 95)

        Returns:
            Tuple: Predictions, lower bound, upper bound

        Example:
            >>> pred, lower, upper = model.get_prediction_intervals(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])

        # Calculate prediction and intervals
        predictions = np.mean(all_predictions, axis=0)
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile

        lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)

        return predictions, lower_bound, upper_bound


# Convenience functions
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        task_type: str = 'regression',
                        tune: bool = False) -> RandomForestPredictor:
    """
    Quick function to train and evaluate Random Forest model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        task_type (str): 'regression' or 'classification'
        tune (bool): Whether to tune hyperparameters

    Returns:
        RandomForestPredictor: Trained model

    Example:
        >>> model = train_random_forest(X_train, y_train, X_test, y_test)
    """
    model = RandomForestPredictor(task_type=task_type)

    if tune:
        model.tune_hyperparameters(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    print("\n=== Model Performance ===")
    print("\nTraining Set:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTest Set:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Feature importance
    importance = model.get_feature_importance(top_n=10)
    print("\n=== Top 10 Important Features ===")
    print(importance.to_string(index=False))

    return model


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data
    from src.models.feature_engineering import FeatureEngineer
    from src.data.preprocessor import StockDataPreprocessor

    # Fetch and prepare data
    df = get_stock_data('AAPL', start='2022-01-01')

    # Prepare ML dataset
    X, y, scaler = FeatureEngineer.prepare_ml_dataset(
        df, target_column='Close', forecast_horizon=1, target_type='price'
    )

    # Split data
    train_df, test_df = StockDataPreprocessor.split_data(
        pd.concat([X, y], axis=1), train_size=0.8
    )

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train model
    model = train_random_forest(X_train, y_train, X_test, y_test, tune=False)

    # Make predictions
    predictions = model.predict(X_test)
    print(f"\nSample predictions vs actual:")
    comparison = pd.DataFrame({
        'Actual': y_test.values[:5],
        'Predicted': predictions[:5]
    })
    print(comparison)
