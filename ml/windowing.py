import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_sequences(data: np.ndarray, window_size: int,
                     target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM.

    Args:
        data: Input data array of shape (n_samples, n_features)
        window_size: Number of timesteps in each sequence
        target_col_idx: Index of target column

    Returns:
        Tuple of (X, y) where X shape is (n_samples, window_size, n_features)
        and y shape is (n_samples,)
    """
    n_samples = len(data) - window_size
    if n_samples <= 0:
        raise ValueError(f"Data length ({len(data)}) must be greater than window_size ({window_size})")

    n_features = data.shape[1]
    X = np.zeros((n_samples, window_size, n_features))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = data[i:i + window_size]
        y[i] = data[i + window_size, target_col_idx]

    logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    return X, y


def create_sequences_multivariate(data: np.ndarray, window_size: int,
                                   target_col_idx: int = 0,
                                   horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for multivariate forecasting.

    Args:
        data: Input data array of shape (n_samples, n_features)
        window_size: Number of timesteps in each sequence
        target_col_idx: Index of target column
        horizon: Forecast horizon (steps ahead)

    Returns:
        Tuple of (X, y) where X shape is (n_samples, window_size, n_features)
        and y shape is (n_samples, horizon)
    """
    n_samples = len(data) - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError(f"Data length ({len(data)}) must be greater than window_size + horizon - 1")

    n_features = data.shape[1]
    X = np.zeros((n_samples, window_size, n_features))
    y = np.zeros((n_samples, horizon))

    for i in range(n_samples):
        X[i] = data[i:i + window_size]
        y[i] = data[i + window_size:i + window_size + horizon, target_col_idx]

    logger.info(f"Created multivariate sequences: X shape {X.shape}, y shape {y.shape}")
    return X, y


def split_sequences_by_time(X: np.ndarray, y: np.ndarray,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences chronologically.

    Args:
        X: Input sequences
        y: Target values
        train_ratio: Training ratio
        val_ratio: Validation ratio

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    logger.info(f"Split sequences: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_lstm_data(df: pd.DataFrame, target_col: str, window_size: int = 24,
                       feature_cols: list = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare data for LSTM from DataFrame.

    Args:
        df: DataFrame with timestamp index
        target_col: Target column name
        window_size: Window size for sequences
        feature_cols: List of feature columns (default: all numeric)

    Returns:
        Tuple of (X, y, feature_names)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col not in feature_cols:
        feature_cols = [target_col] + [c for c in feature_cols if c != target_col]

    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)

    X, y = create_sequences(data, window_size, target_idx)
    return X, y, feature_cols


def validate_sequence_shapes(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Validate sequence shapes for consistency.

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'shapes': {
            'X_train': X_train.shape,
            'y_train': y_train.shape,
            'X_val': X_val.shape,
            'y_val': y_val.shape,
            'X_test': X_test.shape,
            'y_test': y_test.shape
        }
    }

    # Check feature consistency
    n_features = X_train.shape[2]
    for name, arr in [('X_val', X_val), ('X_test', X_test)]:
        if arr.shape[2] != n_features:
            results['valid'] = False
            results['errors'].append(f"{name} has {arr.shape[2]} features, expected {n_features}")

    # Check sample count consistency
    for X, y, name in [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]:
        if len(X) != len(y):
            results['valid'] = False
            results['errors'].append(f"{name}: X has {len(X)} samples, y has {len(y)} samples")

    return results