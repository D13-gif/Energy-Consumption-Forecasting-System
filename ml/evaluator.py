import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with metrics
    """
    # Ensure same shape
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true {len(y_true)} vs y_pred {len(y_pred)}")

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mse': np.nan, 'r2': np.nan, 'mape': np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100

    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'correlation': float(correlation),
        'n_samples': len(y_true)
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   scaler=None, target_col_idx: int = 0) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test targets (scaled)
        scaler: Fitted scaler for inverse transform
        target_col_idx: Target column index in scaler

    Returns:
        Dictionary with evaluation results
    """
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform if scaler provided
    if scaler is not None:
        # Create dummy arrays for inverse transform
        n_features = scaler.n_features_in_
        dummy_true = np.zeros((len(y_test), n_features))
        dummy_pred = np.zeros((len(y_pred_scaled), n_features))

        dummy_true[:, target_col_idx] = y_test.flatten()
        dummy_pred[:, target_col_idx] = y_pred_scaled

        y_true = scaler.inverse_transform(dummy_true)[:, target_col_idx]
        y_pred = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
    else:
        y_true = y_test.flatten()
        y_pred = y_pred_scaled

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    return {
        'metrics': metrics,
        'actual': y_true.tolist(),
        'predicted': y_pred.tolist(),
        'residuals': (y_true - y_pred).tolist()
    }


def evaluate_on_splits(model, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        scaler=None, target_col_idx: int = 0) -> Dict:
    """
    Evaluate model on all splits.

    Returns:
        Dictionary with metrics for each split
    """
    results = {}

    for name, X, y in [('train', X_train, y_train),
                        ('val', X_val, y_val),
                        ('test', X_test, y_test)]:
        if len(X) > 0:
            results[name] = evaluate_model(model, X, y, scaler, target_col_idx)
        else:
            results[name] = {'metrics': {'mae': np.nan, 'rmse': np.nan, 'mse': np.nan}}

    return results


def calculate_test_details(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """
    Calculate detailed test set statistics.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Dictionary with detailed statistics
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {}

    residuals = actual - predicted

    return {
        'mean_actual': float(np.mean(actual)),
        'mean_predicted': float(np.mean(predicted)),
        'std_actual': float(np.std(actual)),
        'std_predicted': float(np.std(predicted)),
        'min_actual': float(np.min(actual)),
        'max_actual': float(np.max(actual)),
        'min_predicted': float(np.min(predicted)),
        'max_predicted': float(np.max(predicted)),
        'correlation': float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else np.nan,
        'mape': float(np.mean(np.abs(residuals / np.where(actual != 0, actual, 1e-8))) * 100),
        'r2': float(r2_score(actual, predicted)),
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
        'residual_min': float(np.min(residuals)),
        'residual_max': float(np.max(residuals))
    }