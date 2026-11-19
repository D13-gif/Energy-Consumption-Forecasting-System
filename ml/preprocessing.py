import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PreprocessingConfig:
    def __init__(self, aggregation_freq: str = 'h', missing_strategy: str = 'interpolate',
                 train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                 window_size: int = 24, scaler_type: str = 'minmax'):
        self.aggregation_freq = aggregation_freq
        self.missing_strategy = missing_strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.window_size = window_size
        self.scaler_type = scaler_type

    def to_dict(self) -> dict:
        return {
            'aggregation_freq': self.aggregation_freq,
            'missing_strategy': self.missing_strategy,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'window_size': self.window_size,
            'scaler_type': self.scaler_type
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            aggregation_freq=data.get('aggregation_freq', 'h'),
            missing_strategy=data.get('missing_strategy', 'interpolate'),
            train_ratio=data.get('train_ratio', 0.7),
            val_ratio=data.get('val_ratio', 0.15),
            test_ratio=data.get('test_ratio', 0.15),
            window_size=data.get('window_size', 24),
            scaler_type=data.get('scaler_type', 'minmax')
        )


def handle_missing_values(df: pd.DataFrame, target_col: str, strategy: str = 'interpolate',
                          max_gap: int = 6) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values in the target column.

    Args:
        df: DataFrame with timestamp index
        target_col: Target column name
        strategy: 'interpolate', 'drop', or 'ffill'
        max_gap: Maximum gap for interpolation (in timesteps)

    Returns:
        Tuple of (processed_df, info_dict)
    """
    df = df.copy()
    info = {'strategy': strategy, 'missing_before': int(df[target_col].isna().sum())}

    if strategy == 'interpolate':
        # Interpolate with limit
        df[target_col] = df[target_col].interpolate(
            method='time', limit=max_gap, limit_direction='both'
        )
    elif strategy == 'drop':
        df = df.dropna(subset=[target_col])
    elif strategy == 'ffill':
        df[target_col] = df[target_col].ffill()

    info['missing_after'] = int(df[target_col].isna().sum())
    info['filled_count'] = info['missing_before'] - info['missing_after']

    return df, info


def aggregate_hourly(df: pd.DataFrame, target_col: str, freq: str = 'h') -> Tuple[pd.DataFrame, Dict]:
    """
    Aggregate minute-level data to hourly frequency.

    Args:
        df: DataFrame with timestamp index
        target_col: Target column name
        freq: Resampling frequency

    Returns:
        Tuple of (aggregated_df, info_dict)
    """
    df = df.copy()
    original_len = len(df)

    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Resample to hourly frequency using mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_hourly = df[numeric_cols].resample(freq).mean()

    # Handle empty hourly buckets
    empty_buckets = df_hourly[target_col].isna().sum()
    if empty_buckets > 0:
        # Forward fill empty buckets (short gaps)
        df_hourly[target_col] = df_hourly[target_col].interpolate(method='time', limit=3)

    info = {
        'original_frequency': 'minute',
        'target_frequency': freq,
        'original_rows': original_len,
        'aggregated_rows': len(df_hourly),
        'empty_buckets_filled': int(empty_buckets)
    }

    return df_hourly.reset_index(), info


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.7,
                        val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split data chronologically into train/validation/test.

    Args:
        df: DataFrame sorted by timestamp
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Tuple of (train_df, val_df, test_df, info_dict)
    """
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    info = {
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'train_range': {
            'start': train_df['timestamp'].min().isoformat() if 'timestamp' in train_df.columns else str(train_df.index.min()),
            'end': train_df['timestamp'].max().isoformat() if 'timestamp' in train_df.columns else str(train_df.index.max())
        },
        'val_range': {
            'start': val_df['timestamp'].min().isoformat() if 'timestamp' in val_df.columns else str(val_df.index.min()),
            'end': val_df['timestamp'].max().isoformat() if 'timestamp' in val_df.columns else str(val_df.index.max())
        },
        'test_range': {
            'start': test_df['timestamp'].min().isoformat() if 'timestamp' in test_df.columns else str(test_df.index.min()),
            'end': test_df['timestamp'].max().isoformat() if 'timestamp' in test_df.columns else str(test_df.index.max())
        }
    }

    return train_df, val_df, test_df, info


def get_scaler(scaler_type: str = 'minmax'):
    """
    Get scaler instance by type.

    Args:
        scaler_type: 'minmax', 'standard', or 'robust'

    Returns:
        Scaler instance
    """
    scalers = {
        'minmax': MinMaxScaler(feature_range=(0, 1)),
        'standard': StandardScaler(),
        'robust': RobustScaler()
    }
    return scalers.get(scaler_type, MinMaxScaler(feature_range=(0, 1)))


def fit_scaler(train_data: np.ndarray, scaler_type: str = 'minmax'):
    """
    Fit scaler on training data only.

    Args:
        train_data: Training data array
        scaler_type: Type of scaler

    Returns:
        Fitted scaler
    """
    scaler = get_scaler(scaler_type)
    scaler.fit(train_data)
    return scaler


def transform_data(data: np.ndarray, scaler) -> np.ndarray:
    """
    Transform data using fitted scaler.

    Args:
        data: Data to transform
        scaler: Fitted scaler

    Returns:
        Transformed data
    """
    return scaler.transform(data)


def inverse_transform_data(data: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse transform data to original scale.

    Args:
        data: Scaled data
        scaler: Fitted scaler

    Returns:
        Data in original scale
    """
    return scaler.inverse_transform(data)


def save_scaler(scaler, filepath: Path):
    """Save scaler to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)
    logger.info(f"Scaler saved to {filepath}")


def load_scaler(filepath: Path):
    """Load scaler from file."""
    return joblib.load(filepath)


def save_preprocessing_metadata(metadata: dict, filepath: Path):
    """Save preprocessing metadata to JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Preprocessing metadata saved to {filepath}")


def load_preprocessing_metadata(filepath: Path) -> dict:
    """Load preprocessing metadata from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)