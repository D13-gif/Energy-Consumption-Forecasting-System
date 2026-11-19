import numpy as np
import pandas as pd
from tensorflow.keras import models
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def recursive_forecast(model: models.Model, last_sequence: np.ndarray,
                       horizon: int, scaler=None, target_col_idx: int = 0) -> np.ndarray:
    """
    Generate recursive multi-step forecast.

    Args:
        model: Trained Keras model
        last_sequence: Last known sequence of shape (window_size, n_features)
        horizon: Number of steps to forecast
        scaler: Fitted scaler for inverse transform
        target_col_idx: Target column index

    Returns:
        Array of forecasted values (in original scale if scaler provided)
    """
    window_size = last_sequence.shape[0]
    n_features = last_sequence.shape[1]
    current_seq = last_sequence.copy()
    forecasts_scaled = []

    for _ in range(horizon):
        # Predict next step
        pred = model.predict(current_seq[np.newaxis, ...], verbose=0)[0, 0]
        forecasts_scaled.append(pred)

        # Update sequence: roll and append prediction
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, target_col_idx] = pred

    forecasts_scaled = np.array(forecasts_scaled)

    # Inverse transform if scaler provided
    if scaler is not None:
        n_features = scaler.n_features_in_
        dummy = np.zeros((horizon, n_features))
        dummy[:, target_col_idx] = forecasts_scaled
        forecasts = scaler.inverse_transform(dummy)[:, target_col_idx]
    else:
        forecasts = forecasts_scaled

    return forecasts


def generate_future_timestamps(last_timestamp: pd.Timestamp, horizon: int,
                                freq: str = 'h') -> List[pd.Timestamp]:
    """
    Generate future timestamps for forecast.

    Args:
        last_timestamp: Last known timestamp
        horizon: Number of steps
        freq: Frequency string

    Returns:
        List of future timestamps
    """
    return pd.date_range(
        start=last_timestamp + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq
    ).tolist()


def prepare_forecast_input(df: pd.DataFrame, window_size: int,
                            feature_cols: list, target_col: str,
                            scaler) -> np.ndarray:
    """
    Prepare the last sequence for forecasting.

    Args:
        df: DataFrame with processed data
        window_size: Window size
        feature_cols: Feature column names
        target_col: Target column name
        scaler: Fitted scaler

    Returns:
        Last sequence array of shape (window_size, n_features)
    """
    # Get last window_size rows
    last_data = df[feature_cols].tail(window_size).values

    # Scale
    last_scaled = scaler.transform(last_data)

    return last_scaled


def run_forecast(model: models.Model, df: pd.DataFrame, horizon: int,
                 window_size: int, feature_cols: list, target_col: str,
                 scaler, freq: str = 'h') -> Dict:
    """
    Run complete forecasting pipeline.

    Args:
        model: Trained model
        df: Processed DataFrame with timestamp index
        horizon: Forecast horizon
        window_size: Window size
        feature_cols: Feature column names
        target_col: Target column name
        scaler: Fitted scaler
        freq: Data frequency

    Returns:
        Dictionary with forecast results
    """
    # Prepare input
    last_sequence = prepare_forecast_input(df, window_size, feature_cols, target_col, scaler)

    # Get target column index
    target_col_idx = feature_cols.index(target_col)

    # Generate forecast
    forecast_values = recursive_forecast(model, last_sequence, horizon, scaler, target_col_idx)

    # Generate timestamps
    last_timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'].iloc[-1]
    forecast_timestamps = generate_future_timestamps(last_timestamp, horizon, freq)

    # Also generate predictions on test set for comparison
    # (This would be handled separately in the service layer)

    return {
        'forecast': {
            'timestamps': [ts.isoformat() for ts in forecast_timestamps],
            'values': forecast_values.tolist(),
            'horizon': horizon,
            'freq': freq,
            'generated_at': datetime.now().isoformat()
        }
    }


def evaluate_and_forecast(model: models.Model, X_test: np.ndarray, y_test: np.ndarray,
                           scaler, target_col_idx: int = 0) -> Dict:
    """
    Evaluate on test set and prepare data for visualization.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets (scaled)
        scaler: Fitted scaler
        target_col_idx: Target column index

    Returns:
        Dictionary with evaluation predictions
    """
    # Predict on test set
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform
    n_features = scaler.n_features_in_
    dummy_true = np.zeros((len(y_test), n_features))
    dummy_pred = np.zeros((len(y_pred_scaled), n_features))

    dummy_true[:, target_col_idx] = y_test.flatten()
    dummy_pred[:, target_col_idx] = y_pred_scaled

    y_true = scaler.inverse_transform(dummy_true)[:, target_col_idx]
    y_pred = scaler.inverse_transform(dummy_pred)[:, target_col_idx]

    return {
        'evaluation': {
            'actual': y_true.tolist(),
            'predicted': y_pred.tolist(),
            'timestamps': []  # To be filled by service layer
        }
    }