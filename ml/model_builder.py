import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def build_lstm_model(input_shape: Tuple[int, int],
                     lstm_units: int = 32,
                     dropout_rate: float = 0.1,
                     loss: str = 'mse',
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001) -> models.Model:
    """
    Build LSTM model for time series forecasting.

    Args:
        input_shape: Tuple of (window_size, n_features)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        loss: Loss function
        optimizer: Optimizer name
        learning_rate: Learning rate

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(dropout_rate) if dropout_rate > 0 else layers.Lambda(lambda x: x),
        layers.Dense(1)
    ])

    # Configure optimizer
    opt_map = {
        'adam': optimizers.Adam(learning_rate=learning_rate),
        'rmsprop': optimizers.RMSprop(learning_rate=learning_rate),
        'sgd': optimizers.SGD(learning_rate=learning_rate)
    }
    opt = opt_map.get(optimizer.lower(), optimizers.Adam(learning_rate=learning_rate))

    model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    logger.info(f"Model built: input_shape={input_shape}, lstm_units={lstm_units}, "
                f"dropout={dropout_rate}, loss={loss}, optimizer={optimizer}, lr={learning_rate}")

    return model


def build_lstm_model_deep(input_shape: Tuple[int, int],
                           lstm_units: list = [32, 16],
                           dropout_rate: float = 0.1,
                           loss: str = 'mse',
                           optimizer: str = 'adam',
                           learning_rate: float = 0.001) -> models.Model:
    """
    Build deeper LSTM model with multiple LSTM layers.

    Args:
        input_shape: Tuple of (window_size, n_features)
        lstm_units: List of units per LSTM layer
        dropout_rate: Dropout rate
        loss: Loss function
        optimizer: Optimizer name
        learning_rate: Learning rate

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    opt_map = {
        'adam': optimizers.Adam(learning_rate=learning_rate),
        'rmsprop': optimizers.RMSprop(learning_rate=learning_rate),
        'sgd': optimizers.SGD(learning_rate=learning_rate)
    }
    opt = opt_map.get(optimizer.lower(), optimizers.Adam(learning_rate=learning_rate))

    model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    logger.info(f"Deep model built: input_shape={input_shape}, lstm_units={lstm_units}, "
                f"dropout={dropout_rate}, loss={loss}, optimizer={optimizer}, lr={learning_rate}")

    return model


def get_model_summary(model: models.Model) -> str:
    """Get model summary as string."""
    from io import StringIO
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


def get_callbacks(patience: int = 5, monitor: str = 'val_loss',
                  mode: str = 'min', restore_best_weights: bool = True,
                  verbose: int = 1) -> list:
    """
    Get training callbacks.

    Args:
        patience: Early stopping patience
        monitor: Metric to monitor
        mode: 'min' or 'max'
        restore_best_weights: Whether to restore best weights
        verbose: Verbosity level

    Returns:
        List of callbacks
    """
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=restore_best_weights,
            verbose=verbose
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(1, patience // 2),
            mode=mode,
            min_lr=1e-6,
            verbose=verbose
        )
    ]


def save_model(model: models.Model, filepath: str):
    """Save model to file."""
    model.save(filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> models.Model:
    """Load model from file."""
    model = models.load_model(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def get_model_info(model: models.Model) -> Dict:
    """
    Extract model information.

    Returns:
        Dictionary with model info
    """
    return {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'trainable_params': sum(p.shape.num_elements() for p in model.trainable_weights),
        'non_trainable_params': sum(p.shape.num_elements() for p in model.non_trainable_weights),
        'layers': [
            {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': layer.output_shape,
                'params': layer.count_params()
            }
            for layer in model.layers
        ]
    }