"""
Model Evaluation Module

This module provides tools for evaluating and comparing
machine learning models for stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt


class ModelEvaluator:
    """
    Tools for evaluating ML models.
    """

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray,
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict: Regression metrics

        Example:
            >>> metrics = ModelEvaluator.calculate_regression_metrics(y_test, predictions)
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        return metrics

    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray,
                                       y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (predicting up/down correctly).

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            float: Directional accuracy percentage

        Example:
            >>> acc = ModelEvaluator.calculate_directional_accuracy(y_test, predictions)
        """
        # Calculate actual and predicted directions
        actual_direction = np.diff(y_true) > 0
        predicted_direction = np.diff(y_pred) > 0

        # Calculate accuracy
        correct = np.sum(actual_direction == predicted_direction)
        accuracy = (correct / len(actual_direction)) * 100

        return accuracy

    @staticmethod
    def calculate_profit_loss(y_true: np.ndarray, y_pred: np.ndarray,
                             initial_capital: float = 10000,
                             commission: float = 0.001) -> Dict[str, float]:
        """
        Simulate trading based on predictions and calculate P/L.

        Args:
            y_true (np.ndarray): True prices
            y_pred (np.ndarray): Predicted prices
            initial_capital (float): Starting capital
            commission (float): Trading commission (as fraction)

        Returns:
            Dict: Profit/loss metrics

        Example:
            >>> pl = ModelEvaluator.calculate_profit_loss(y_test, predictions)
        """
        capital = initial_capital
        shares = 0
        trades = 0

        for i in range(len(y_pred) - 1):
            current_price = y_true[i]
            predicted_next = y_pred[i + 1]

            # Buy signal: predicted price increase
            if predicted_next > current_price and shares == 0:
                shares = capital / current_price
                capital = 0
                trades += 1
                shares *= (1 - commission)  # Commission on buy

            # Sell signal: predicted price decrease
            elif predicted_next < current_price and shares > 0:
                capital = shares * current_price
                shares = 0
                trades += 1
                capital *= (1 - commission)  # Commission on sell

        # Close position at end
        if shares > 0:
            capital = shares * y_true[-1]
            capital *= (1 - commission)

        total_return = ((capital - initial_capital) / initial_capital) * 100

        metrics = {
            'Final_Capital': capital,
            'Total_Return_%': total_return,
            'Trades': trades
        }

        return metrics

    @staticmethod
    def compare_models(models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models_dict (dict): Dictionary of {model_name: (y_true, y_pred)}

        Returns:
            pd.DataFrame: Comparison table

        Example:
            >>> comparison = ModelEvaluator.compare_models({
            ...     'RF': (y_test, rf_pred),
            ...     'LSTM': (y_test, lstm_pred)
            ... })
        """
        results = []

        for model_name, (y_true, y_pred) in models_dict.items():
            metrics = ModelEvaluator.calculate_regression_metrics(y_true, y_pred)
            dir_acc = ModelEvaluator.calculate_directional_accuracy(y_true, y_pred)

            results.append({
                'Model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE'],
                'Dir_Accuracy_%': dir_acc
            })

        df = pd.DataFrame(results)
        df = df.set_index('Model')

        return df

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = 'Predictions vs Actual',
                        dates: Optional[pd.DatetimeIndex] = None):
        """
        Plot predictions vs actual values.

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
            dates (pd.DatetimeIndex): Optional dates for x-axis

        Example:
            >>> ModelEvaluator.plot_predictions(y_test, predictions)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        x_axis = dates if dates is not None else range(len(y_true))

        # Time series plot
        ax1.plot(x_axis, y_true, label='Actual', linewidth=2, alpha=0.7)
        ax1.plot(x_axis, y_pred, label='Predicted', linewidth=2, alpha=0.7)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Price', fontsize=12)
        ax2.set_ylabel('Predicted Price', fontsize=12)
        ax2.set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_model_comparison(models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                             dates: Optional[pd.DatetimeIndex] = None):
        """
        Plot comparison of multiple models.

        Args:
            models_dict (dict): Dictionary of {model_name: (y_true, y_pred)}
            dates (pd.DatetimeIndex): Optional dates for x-axis

        Example:
            >>> ModelEvaluator.plot_model_comparison({
            ...     'RF': (y_test, rf_pred),
            ...     'LSTM': (y_test, lstm_pred)
            ... })
        """
        plt.figure(figsize=(14, 8))

        x_axis = dates if dates is not None else range(len(list(models_dict.values())[0][0]))

        # Plot actual values
        y_true = list(models_dict.values())[0][0]
        plt.plot(x_axis, y_true, label='Actual', linewidth=3, alpha=0.7, color='black')

        # Plot predictions from each model
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        for i, (model_name, (_, y_pred)) in enumerate(models_dict.items()):
            plt.plot(x_axis, y_pred, label=model_name, linewidth=2,
                    alpha=0.7, color=colors[i % len(colors)])

        plt.title('Model Comparison - Predictions vs Actual', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = 'Residual Analysis'):
        """
        Plot residual analysis.

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title

        Example:
            >>> ModelEvaluator.plot_residuals(y_test, predictions)
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Residuals over time
        axes[0, 0].plot(residuals, linewidth=1, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Residual Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Q-Q plot (simplified)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals vs Predictions
        axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Residuals vs Predictions', fontweight='bold')
        axes[1, 1].set_xlabel('Predicted Value')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Plot error distribution analysis.

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Example:
            >>> ModelEvaluator.plot_error_distribution(y_test, predictions)
        """
        errors = y_true - y_pred
        percent_errors = (errors / y_true) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute error distribution
        axes[0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Percentage error distribution
        axes[1].hist(percent_errors, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

        print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.4f}")
        print(f"Mean Percentage Error: {np.mean(percent_errors):.2f}%")
        print(f"Median Percentage Error: {np.median(percent_errors):.2f}%")


# Convenience functions
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   plot: bool = True) -> Dict[str, float]:
    """
    Quick function to evaluate a model.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        plot (bool): Whether to create plots

    Returns:
        Dict: Evaluation metrics

    Example:
        >>> metrics = evaluate_model(y_test, predictions)
    """
    metrics = ModelEvaluator.calculate_regression_metrics(y_true, y_pred)
    dir_acc = ModelEvaluator.calculate_directional_accuracy(y_true, y_pred)
    pl_metrics = ModelEvaluator.calculate_profit_loss(y_true, y_pred)

    print("=== Model Evaluation ===")
    print("\nRegression Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nDirectional Accuracy: {dir_acc:.2f}%")

    print("\nTrading Simulation:")
    for metric, value in pl_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")

    if plot:
        ModelEvaluator.plot_predictions(y_true, y_pred)
        ModelEvaluator.plot_residuals(y_true, y_pred)

    return {**metrics, 'Directional_Accuracy': dir_acc, **pl_metrics}


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Create sample data
    np.random.seed(42)
    y_true = np.random.randn(100).cumsum() + 100
    y_pred = y_true + np.random.randn(100) * 2  # Add some noise

    print("Evaluating sample predictions...")
    metrics = evaluate_model(y_true, y_pred, plot=True)

    # Compare multiple models
    y_pred2 = y_true + np.random.randn(100) * 1.5  # Better model

    comparison = ModelEvaluator.compare_models({
        'Model_1': (y_true, y_pred),
        'Model_2': (y_true, y_pred2)
    })

    print("\n=== Model Comparison ===")
    print(comparison)

    ModelEvaluator.plot_model_comparison({
        'Model_1': (y_true, y_pred),
        'Model_2': (y_true, y_pred2)
    })
