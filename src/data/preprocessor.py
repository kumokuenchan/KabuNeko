"""
Data preprocessing module for cleaning and preparing stock data.

This module provides functions for data cleaning, feature engineering,
normalization, and train/test splitting for time-series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class StockDataPreprocessor:
    """
    A class to preprocess stock market data for analysis and machine learning.
    """

    @staticmethod
    def clean_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Clean stock data by handling missing values and outliers.

        Args:
            df (pd.DataFrame): Input stock data
            method (str): Method for handling missing values
                         ('ffill', 'bfill', 'interpolate', 'drop')

        Returns:
            pd.DataFrame: Cleaned data

        Example:
            >>> df_clean = StockDataPreprocessor.clean_data(df, method='ffill')
        """
        df = df.copy()

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values. Applying {method}...")

            if method == 'ffill':
                df = df.fillna(method='ffill')
            elif method == 'bfill':
                df = df.fillna(method='bfill')
            elif method == 'interpolate':
                df = df.interpolate(method='linear')
            elif method == 'drop':
                df = df.dropna()
            else:
                raise ValueError(f"Unknown method: {method}")

            # Forward fill any remaining NaN at the start
            df = df.fillna(method='bfill')

        # Remove any duplicate indices
        if df.index.duplicated().any():
            print("Removing duplicate indices...")
            df = df[~df.index.duplicated(keep='first')]

        # Sort by date
        df = df.sort_index()

        print(f"Cleaned data: {len(df)} records, {df.columns.tolist()}")
        return df

    @staticmethod
    def add_returns(df: pd.DataFrame, periods: list = [1, 5, 20]) -> pd.DataFrame:
        """
        Add return columns for different periods.

        Args:
            df (pd.DataFrame): Input stock data with 'Close' column
            periods (list): List of periods for return calculation

        Returns:
            pd.DataFrame: Data with added return columns

        Example:
            >>> df = StockDataPreprocessor.add_returns(df, periods=[1, 5, 10])
        """
        df = df.copy()

        # Simple returns
        for period in periods:
            df[f'Return_{period}d'] = df['Close'].pct_change(periods=period)

        # Log returns (useful for some ML models)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Cumulative returns
        df['Cumulative_Return'] = (1 + df['Return_1d']).cumprod() - 1

        print(f"Added return columns for periods: {periods}")
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, windows: list = [5, 20, 60]) -> pd.DataFrame:
        """
        Add rolling volatility (standard deviation) columns.

        Args:
            df (pd.DataFrame): Input stock data with 'Close' column
            windows (list): List of window sizes for volatility calculation

        Returns:
            pd.DataFrame: Data with added volatility columns

        Example:
            >>> df = StockDataPreprocessor.add_volatility(df, windows=[5, 20])
        """
        df = df.copy()

        # Calculate returns first if not present
        if 'Return_1d' not in df.columns:
            df['Return_1d'] = df['Close'].pct_change()

        # Rolling volatility
        for window in windows:
            df[f'Volatility_{window}d'] = df['Return_1d'].rolling(window=window).std()

        print(f"Added volatility columns for windows: {windows}")
        return df

    @staticmethod
    def normalize_data(
        df: pd.DataFrame,
        columns: Optional[list] = None,
        method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
        """
        Normalize specified columns in the dataframe.

        Args:
            df (pd.DataFrame): Input data
            columns (list): Columns to normalize (default: all numeric columns)
            method (str): Normalization method ('minmax' or 'standard')
            feature_range (tuple): Range for MinMaxScaler

        Returns:
            Tuple[pd.DataFrame, Scaler]: Normalized data and fitted scaler

        Example:
            >>> df_norm, scaler = StockDataPreprocessor.normalize_data(df, method='minmax')
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'minmax' or 'standard'")

        df[columns] = scaler.fit_transform(df[columns])

        print(f"Normalized {len(columns)} columns using {method} scaling")
        return df, scaler

    @staticmethod
    def create_sequences(
        data: np.ndarray,
        lookback: int,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/RNN models.

        Args:
            data (np.ndarray): Input data array
            lookback (int): Number of time steps to look back
            forecast_horizon (int): Number of steps ahead to predict

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (sequences) and y (targets)

        Example:
            >>> X, y = StockDataPreprocessor.create_sequences(data, lookback=60)
        """
        X, y = [], []

        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            y.append(data[i + forecast_horizon - 1])

        X = np.array(X)
        y = np.array(y)

        print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        train_size: float = 0.8,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-series aware).

        Args:
            df (pd.DataFrame): Input data
            train_size (float): Proportion of data for training (0 to 1)
            shuffle (bool): Whether to shuffle (should be False for time-series)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes

        Example:
            >>> train_df, test_df = StockDataPreprocessor.split_data(df, train_size=0.8)
        """
        if shuffle:
            print("Warning: Shuffling time-series data can lead to look-ahead bias!")

        split_idx = int(len(df) * train_size)

        if shuffle:
            df = df.sample(frac=1, random_state=42)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"Split data: Train {len(train_df)} records, Test {len(test_df)} records")
        print(f"Train period: {train_df.index[0]} to {train_df.index[-1]}")
        print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")

        return train_df, test_df

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in a specific column.

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to check for outliers
            method (str): Detection method ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection

        Returns:
            pd.Series: Boolean series indicating outliers

        Example:
            >>> outliers = StockDataPreprocessor.detect_outliers(df, 'Close', method='iqr')
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_count = outliers.sum()
        print(f"Detected {outlier_count} outliers in {column} using {method} method")

        return outliers

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from the data.

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to check for outliers
            method (str): Detection method ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection

        Returns:
            pd.DataFrame: Data with outliers removed

        Example:
            >>> df_clean = StockDataPreprocessor.remove_outliers(df, 'Volume')
        """
        outliers = StockDataPreprocessor.detect_outliers(df, column, method, threshold)
        df_clean = df[~outliers].copy()

        print(f"Removed {outliers.sum()} outliers. Remaining: {len(df_clean)} records")
        return df_clean

    @staticmethod
    def add_lagged_features(
        df: pd.DataFrame,
        columns: list,
        lags: list = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Add lagged features for machine learning.

        Args:
            df (pd.DataFrame): Input data
            columns (list): Columns to create lagged features for
            lags (list): List of lag periods

        Returns:
            pd.DataFrame: Data with lagged features

        Example:
            >>> df = StockDataPreprocessor.add_lagged_features(df, ['Close'], lags=[1, 5, 10])
        """
        df = df.copy()

        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # Drop rows with NaN created by lagging
        df = df.dropna()

        print(f"Added lagged features for {columns} with lags {lags}")
        return df

    @staticmethod
    def prepare_ml_features(
        df: pd.DataFrame,
        target_col: str = 'Close',
        forecast_horizon: int = 1,
        add_lags: bool = True,
        add_returns: bool = True,
        add_vol: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for machine learning.

        Args:
            df (pd.DataFrame): Input stock data
            target_col (str): Target column name
            forecast_horizon (int): Steps ahead to predict
            add_lags (bool): Whether to add lagged features
            add_returns (bool): Whether to add return features
            add_vol (bool): Whether to add volatility features

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)

        Example:
            >>> X, y = StockDataPreprocessor.prepare_ml_features(df)
        """
        df = df.copy()

        # Add return features
        if add_returns:
            df = StockDataPreprocessor.add_returns(df, periods=[1, 5, 10, 20])

        # Add volatility features
        if add_vol:
            df = StockDataPreprocessor.add_volatility(df, windows=[5, 10, 20])

        # Add lagged features
        if add_lags:
            df = StockDataPreprocessor.add_lagged_features(
                df,
                columns=['Close', 'Volume'],
                lags=[1, 2, 3, 5, 10]
            )

        # Create target
        df['Target'] = df[target_col].shift(-forecast_horizon)

        # Drop NaN values
        df = df.dropna()

        # Separate features and target
        y = df['Target']
        X = df.drop(columns=['Target'])

        print(f"Prepared ML dataset: X shape {X.shape}, y shape {y.shape}")

        return X, y


# Convenience functions
def clean_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """Quick function to clean data."""
    return StockDataPreprocessor.clean_data(df, method)


def split_data(df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quick function to split data."""
    return StockDataPreprocessor.split_data(df, train_size)


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Fetch sample data
    df = get_stock_data('AAPL', start='2023-01-01')

    # Clean data
    df_clean = StockDataPreprocessor.clean_data(df)

    # Add returns and volatility
    df_clean = StockDataPreprocessor.add_returns(df_clean)
    df_clean = StockDataPreprocessor.add_volatility(df_clean)

    # Split data
    train, test = StockDataPreprocessor.split_data(df_clean)

    print("\nPreprocessed data sample:")
    print(df_clean.tail())
