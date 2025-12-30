"""
Feature Engineering Module for Machine Learning

This module provides functions to create features from stock data
for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureEngineer:
    """
    A class for creating features from stock data for ML models.
    """

    @staticmethod
    def create_lagged_features(df: pd.DataFrame, columns: List[str],
                               lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features for specified columns.

        Args:
            df (pd.DataFrame): Input data
            columns (list): Columns to create lags for
            lags (list): List of lag periods

        Returns:
            pd.DataFrame: Data with lagged features

        Example:
            >>> df = FeatureEngineer.create_lagged_features(df, ['Close'], [1, 5, 10])
        """
        df = df.copy()

        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    @staticmethod
    def create_rolling_features(df: pd.DataFrame, column: str = 'Close',
                               windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling window features (mean, std, min, max).

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate rolling features on
            windows (list): List of window sizes

        Returns:
            pd.DataFrame: Data with rolling features

        Example:
            >>> df = FeatureEngineer.create_rolling_features(df, 'Close', [5, 20])
        """
        df = df.copy()

        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()

        return df

    @staticmethod
    def create_return_features(df: pd.DataFrame, column: str = 'Close',
                              periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create return features for different periods.

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate returns on
            periods (list): List of periods for returns

        Returns:
            pd.DataFrame: Data with return features

        Example:
            >>> df = FeatureEngineer.create_return_features(df)
        """
        df = df.copy()

        for period in periods:
            df[f'return_{period}d'] = df[column].pct_change(periods=period)

        # Log returns
        df['log_return'] = np.log(df[column] / df[column].shift(1))

        return df

    @staticmethod
    def create_volatility_features(df: pd.DataFrame, column: str = 'Close',
                                   windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create volatility features.

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to calculate volatility on
            windows (list): List of window sizes

        Returns:
            pd.DataFrame: Data with volatility features

        Example:
            >>> df = FeatureEngineer.create_volatility_features(df)
        """
        df = df.copy()

        # Calculate returns if not present
        if 'return_1d' not in df.columns:
            df['return_1d'] = df[column].pct_change()

        for window in windows:
            df[f'volatility_{window}d'] = df['return_1d'].rolling(window=window).std()

        return df

    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features.

        Args:
            df (pd.DataFrame): Input data with OHLCV

        Returns:
            pd.DataFrame: Data with technical features

        Example:
            >>> df = FeatureEngineer.create_technical_features(df)
        """
        df = df.copy()

        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in [12, 26]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_upper'] = sma_20 + (std_20 * 2)
        df['BB_lower'] = sma_20 - (std_20 * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # Volume features
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']

        return df

    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features (day of week, month, quarter).

        Args:
            df (pd.DataFrame): Input data with datetime index

        Returns:
            pd.DataFrame: Data with temporal features

        Example:
            >>> df = FeatureEngineer.create_temporal_features(df)
        """
        df = df.copy()

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert if possible
            try:
                df.index = pd.to_datetime(df.index)
            except:
                # If conversion fails, skip temporal features
                return df

        # Extract time features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Cyclical encoding for day of week (0-6)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Cyclical encoding for month (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    @staticmethod
    def create_target(df: pd.DataFrame, column: str = 'Close',
                     horizon: int = 1, target_type: str = 'price') -> pd.Series:
        """
        Create target variable for prediction.

        Args:
            df (pd.DataFrame): Input data
            column (str): Column to create target from
            horizon (int): Steps ahead to predict
            target_type (str): 'price', 'return', or 'direction'

        Returns:
            pd.Series: Target variable

        Example:
            >>> target = FeatureEngineer.create_target(df, horizon=1, target_type='return')
        """
        if target_type == 'price':
            # Predict future price
            target = df[column].shift(-horizon)

        elif target_type == 'return':
            # Predict future return
            target = df[column].pct_change(periods=horizon).shift(-horizon)

        elif target_type == 'direction':
            # Predict if price will go up (1) or down (0)
            future_price = df[column].shift(-horizon)
            target = (future_price > df[column]).astype(int)

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return target

    @staticmethod
    def prepare_features(df: pd.DataFrame,
                        include_lagged: bool = True,
                        include_rolling: bool = True,
                        include_returns: bool = True,
                        include_volatility: bool = True,
                        include_technical: bool = True,
                        include_temporal: bool = True) -> pd.DataFrame:
        """
        Prepare all features in one go.

        Args:
            df (pd.DataFrame): Input data
            include_lagged (bool): Include lagged features
            include_rolling (bool): Include rolling features
            include_returns (bool): Include return features
            include_volatility (bool): Include volatility features
            include_technical (bool): Include technical indicators
            include_temporal (bool): Include temporal features

        Returns:
            pd.DataFrame: Data with all features

        Example:
            >>> df_features = FeatureEngineer.prepare_features(df)
        """
        df = df.copy()

        if include_returns:
            df = FeatureEngineer.create_return_features(df)

        if include_volatility:
            df = FeatureEngineer.create_volatility_features(df)

        if include_rolling:
            df = FeatureEngineer.create_rolling_features(df)

        if include_lagged:
            df = FeatureEngineer.create_lagged_features(
                df,
                columns=['Close', 'Volume'],
                lags=[1, 2, 3, 5, 10]
            )

        if include_technical:
            df = FeatureEngineer.create_technical_features(df)

        if include_temporal:
            df = FeatureEngineer.create_temporal_features(df)

        return df

    @staticmethod
    def prepare_ml_dataset(df: pd.DataFrame,
                          target_column: str = 'Close',
                          forecast_horizon: int = 1,
                          target_type: str = 'price',
                          feature_columns: Optional[List[str]] = None,
                          scale_features: bool = True,
                          scaler_type: str = 'minmax') -> Tuple[pd.DataFrame, pd.Series, object]:
        """
        Prepare complete ML dataset with features and target.

        Args:
            df (pd.DataFrame): Input data
            target_column (str): Column to predict
            forecast_horizon (int): Steps ahead to predict
            target_type (str): 'price', 'return', or 'direction'
            feature_columns (list): Specific columns to use (None = all numeric)
            scale_features (bool): Whether to scale features
            scaler_type (str): 'minmax' or 'standard'

        Returns:
            Tuple[pd.DataFrame, pd.Series, Scaler]: Features, target, fitted scaler

        Example:
            >>> X, y, scaler = FeatureEngineer.prepare_ml_dataset(df)
        """
        # Create all features
        df_features = FeatureEngineer.prepare_features(df)

        # Create target
        df_features['target'] = FeatureEngineer.create_target(
            df, target_column, forecast_horizon, target_type
        )

        # Drop NaN values
        df_clean = df_features.dropna()

        # Separate features and target
        y = df_clean['target']
        X = df_clean.drop(columns=['target'])

        # Select feature columns
        if feature_columns is not None:
            X = X[feature_columns]
        else:
            # Use all numeric columns except OHLCV
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                          'Dividends', 'Stock Splits']
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            X = X[feature_cols]

        # Scale features
        scaler = None
        if scale_features:
            if scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {scaler_type}")

            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X, y, scaler

    @staticmethod
    def get_feature_importance_names(df: pd.DataFrame) -> List[str]:
        """
        Get descriptive names for features.

        Args:
            df (pd.DataFrame): Feature DataFrame

        Returns:
            List[str]: Feature names

        Example:
            >>> names = FeatureEngineer.get_feature_importance_names(X)
        """
        return df.columns.tolist()


# Convenience functions
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick function to prepare all features."""
    return FeatureEngineer.prepare_features(df)


def prepare_ml_dataset(df: pd.DataFrame, target_column: str = 'Close',
                       forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series, object]:
    """Quick function to prepare ML dataset."""
    return FeatureEngineer.prepare_ml_dataset(
        df, target_column=target_column, forecast_horizon=forecast_horizon
    )


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data

    # Fetch data
    df = get_stock_data('AAPL', start='2022-01-01')

    print("Original data shape:", df.shape)

    # Create features
    df_features = FeatureEngineer.prepare_features(df)
    print("\nWith features shape:", df_features.shape)
    print("\nFeature columns:")
    print(df_features.columns.tolist())

    # Prepare ML dataset
    X, y, scaler = FeatureEngineer.prepare_ml_dataset(
        df, target_column='Close', forecast_horizon=1, target_type='price'
    )

    print(f"\nML Dataset:")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns[:10]):
        print(f"  {i+1}. {col}")
    if len(X.columns) > 10:
        print(f"  ... and {len(X.columns) - 10} more features")

    print(f"\nSample features:")
    print(X.head())
    print(f"\nSample target:")
    print(y.head())
