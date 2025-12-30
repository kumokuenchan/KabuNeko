"""
LSTM Model for Stock Price Prediction

This module provides an LSTM (Long Short-Term Memory) neural network
implementation for time-series stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. LSTM model unavailable.")


class LSTMPredictor:
    """
    LSTM neural network for stock price prediction.
    """

    def __init__(self, lookback: int = 60,
                 lstm_units: List[int] = [50, 50],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize LSTM model.

        Args:
            lookback (int): Number of time steps to look back
            lstm_units (list): Number of units in each LSTM layer
            dropout (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random seed

        Example:
            >>> model = LSTMPredictor(lookback=60, lstm_units=[50, 50])
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for LSTM model")

        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.model = None
        self.history = None
        self.is_fitted = False

    def build_model(self, input_shape: Tuple[int, int]):
        """
        Build LSTM model architecture.

        Args:
            input_shape (tuple): Shape of input (lookback, n_features)

        Example:
            >>> model.build_model(input_shape=(60, 1))
        """
        self.model = Sequential()

        # First LSTM layer
        self.model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=input_shape
        ))
        self.model.add(Dropout(self.dropout))

        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout))

        # Dense layers
        self.model.add(Dense(units=25, activation='relu'))
        self.model.add(Dense(units=1))

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        print("Model architecture:")
        self.model.summary()

    def create_sequences(self, data: np.ndarray,
                        target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.

        Args:
            data (np.ndarray): Input data
            target (np.ndarray): Target data (optional)

        Returns:
            Tuple: X sequences, y targets

        Example:
            >>> X, y = model.create_sequences(data, target)
        """
        X = []
        y = [] if target is not None else None

        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            if target is not None:
                y.append(target[i])

        X = np.array(X)
        if y is not None:
            y = np.array(y)

        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.2,
            epochs: int = 50,
            batch_size: int = 32,
            early_stopping: bool = True,
            patience: int = 10,
            verbose: int = 1) -> 'LSTMPredictor':
        """
        Train the LSTM model.

        Args:
            X (np.ndarray): Training sequences (samples, lookback, features)
            y (np.ndarray): Target values
            validation_split (float): Fraction for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            early_stopping (bool): Use early stopping
            patience (int): Early stopping patience
            verbose (int): Verbosity level

        Returns:
            self: Fitted model

        Example:
            >>> model.fit(X_train, y_train, epochs=50)
        """
        # Build model if not already built
        if self.model is None:
            input_shape = (X.shape[1], X.shape[2])
            self.build_model(input_shape)

        # Callbacks
        callbacks = []

        if early_stopping:
            es = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(es)

        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X (np.ndarray): Input sequences

        Returns:
            np.ndarray: Predictions

        Example:
            >>> predictions = model.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance.

        Args:
            X (np.ndarray): Test sequences
            y (np.ndarray): True values

        Returns:
            dict: Evaluation metrics

        Example:
            >>> metrics = model.evaluate(X_test, y_test)
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        predictions = self.predict(X)

        metrics = {
            'MAE': mean_absolute_error(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions)),
            'R2': r2_score(y, predictions),
            'MAPE': np.mean(np.abs((y - predictions) / y)) * 100
        }

        return metrics

    def save_model(self, filepath: str):
        """
        Save model to disk.

        Args:
            filepath (str): Path to save model

        Example:
            >>> model.save_model('models/saved_models/lstm_model.h5')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath (str): Path to saved model

        Example:
            >>> model.load_model('models/saved_models/lstm_model.h5')
        """
        self.model = keras.models.load_model(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")

    def plot_training_history(self):
        """
        Plot training history.

        Example:
            >>> model.plot_training_history()
        """
        if self.history is None:
            print("No training history available")
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE
        if 'mae' in self.history.history:
            ax2.plot(self.history.history['mae'], label='Training MAE')
            if 'val_mae' in self.history.history:
                ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_future(self, last_sequence: np.ndarray,
                      steps: int = 1) -> np.ndarray:
        """
        Predict multiple steps into the future.

        Args:
            last_sequence (np.ndarray): Last known sequence
            steps (int): Number of steps to predict

        Returns:
            np.ndarray: Future predictions

        Example:
            >>> future_pred = model.predict_future(last_sequence, steps=5)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(steps):
            # Predict next value
            pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
            predictions.append(pred[0, 0])

            # Update sequence (shift and append prediction)
            if current_sequence.shape[1] == 1:
                # Single feature
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, 0] = pred[0, 0]
            else:
                # Multiple features - only update first feature (price)
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, 0] = pred[0, 0]

        return np.array(predictions)


# Convenience function
def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray,
               lookback: int = 60,
               epochs: int = 50,
               batch_size: int = 32) -> LSTMPredictor:
    """
    Quick function to train and evaluate LSTM model.

    Args:
        X_train (np.ndarray): Training sequences
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Test sequences
        y_test (np.ndarray): Test targets
        lookback (int): Lookback period
        epochs (int): Number of epochs
        batch_size (int): Batch size

    Returns:
        LSTMPredictor: Trained model

    Example:
        >>> model = train_lstm(X_train, y_train, X_test, y_test)
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow not available. Cannot train LSTM model.")
        return None

    model = LSTMPredictor(lookback=lookback)

    print("Training LSTM model...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

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

    # Plot training history
    model.plot_training_history()

    return model


if __name__ == "__main__":
    if not HAS_TENSORFLOW:
        print("TensorFlow is required to run this example")
    else:
        # Example usage
        from src.data.fetcher import get_stock_data
        from sklearn.preprocessing import MinMaxScaler

        # Fetch data
        df = get_stock_data('AAPL', start='2020-01-01')

        # Prepare data
        data = df['Close'].values.reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Create LSTM instance
        lookback = 60
        model = LSTMPredictor(lookback=lookback)

        # Create sequences
        X, y = model.create_sequences(data_scaled, data_scaled)

        print(f"Sequences shape: X={X.shape}, y={y.shape}")

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model.fit(X_train, y_train, epochs=20, batch_size=32)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Make predictions
        predictions = model.predict(X_test[:10])
        predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test[:10].reshape(-1, 1))

        print("\nSample predictions vs actual:")
        comparison = pd.DataFrame({
            'Actual': y_test_actual.flatten(),
            'Predicted': predictions_actual.flatten()
        })
        print(comparison)
