import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
import time
import logging
from datetime import datetime

from ml.model_builder import get_callbacks, get_model_info, save_model

logger = logging.getLogger(__name__)


class TrainingState:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = 'pending'
        self.current_epoch = 0
        self.total_epochs = 0
        self.overall_progress = 0.0
        self.train_loss = None
        self.val_loss = None
        self.elapsed_seconds = 0.0
        self.history = {'loss': [], 'val_loss': []}
        self.best_epoch = None
        self.best_val_loss = None
        self.final_train_loss = None
        self.started_at = None
        self.completed_at = None
        self.duration_seconds = None
        self.error = None
        self.config = {}
        self.artifacts = {}

    def to_dict(self) -> Dict:
        return {
            'run_id': self.run_id,
            'status': self.status,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'overall_progress': self.overall_progress,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'elapsed_seconds': self.elapsed_seconds,
            'history': self.history,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.final_train_loss,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'error': self.error,
            'config': self.config,
            'artifacts': self.artifacts
        }


class TrainingManager:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.models_dir = artifacts_dir / 'models'
        self.scalers_dir = artifacts_dir / 'scalers'
        self.metadata_dir = artifacts_dir / 'metadata'
        self.current_state: Optional[TrainingState] = None

        for d in [self.models_dir, self.scalers_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def start_training(self, config: Dict, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       scaler, feature_names: list) -> TrainingState:
        """
        Start model training.

        Args:
            config: Training configuration
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            scaler: Fitted scaler
            feature_names: List of feature names

        Returns:
            TrainingState object
        """
        run_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        state = TrainingState(run_id)
        state.config = config
        state.status = 'running'
        state.started_at = datetime.now()
        state.total_epochs = config.get('epochs', 20)
        self.current_state = state

        logger.info(f"Starting training run: {run_id}")

        try:
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self._build_model_from_config(input_shape, config)

            # Get callbacks
            callbacks_list = get_callbacks(
                patience=config.get('patience', 5),
                monitor='val_loss',
                mode='min',
                restore_best_weights=True
            )

            # Add custom callback for progress tracking
            callbacks_list.append(ProgressCallback(state))

            # Train
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.get('epochs', 20),
                batch_size=config.get('batch_size', 32),
                callbacks=callbacks_list,
                verbose=1
            )
            elapsed = time.time() - start_time

            # Update state
            state.status = 'completed'
            state.completed_at = datetime.now()
            state.duration_seconds = elapsed
            state.history = {
                'loss': [float(v) for v in history.history.get('loss', [])],
                'val_loss': [float(v) for v in history.history.get('val_loss', [])]
            }
            state.final_train_loss = float(history.history['loss'][-1]) if history.history.get('loss') else None

            # Find best epoch
            if history.history.get('val_loss'):
                val_losses = history.history['val_loss']
                best_idx = np.argmin(val_losses)
                state.best_epoch = best_idx + 1
                state.best_val_loss = float(val_losses[best_idx])

            # Save artifacts
            self._save_artifacts(state, model, scaler, feature_names, config)

            logger.info(f"Training completed: {run_id}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            state.status = 'failed'
            state.error = str(e)
            state.completed_at = datetime.now()
            if state.started_at:
                state.duration_seconds = (state.completed_at - state.started_at).total_seconds()

        return state

    def _build_model_from_config(self, input_shape: Tuple[int, int], config: Dict) -> models.Model:
        """Build model from configuration."""
        from ml.model_builder import build_lstm_model
        return build_lstm_model(
            input_shape=input_shape,
            lstm_units=config.get('lstm_units', 32),
            dropout_rate=config.get('dropout', 0.1),
            loss=config.get('loss', 'mse'),
            optimizer=config.get('optimizer', 'adam'),
            learning_rate=config.get('learning_rate', 0.001)
        )

    def _save_artifacts(self, state: TrainingState, model: models.Model,
                        scaler, feature_names: list, config: Dict):
        """Save model artifacts."""
        # Save model
        model_path = self.models_dir / f"{state.run_id}.keras"
        save_model(model, str(model_path))

        # Save scaler
        import joblib
        scaler_path = self.scalers_dir / f"{state.run_id}_scaler.pkl"
        joblib.dump(scaler, scaler_path)

        # Save config
        config_path = self.metadata_dir / f"{state.run_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save model info
        model_info = get_model_info(model)
        model_info_path = self.metadata_dir / f"{state.run_id}_model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)

        # Save history
        history_path = self.metadata_dir / f"{state.run_id}_history.json"
        with open(history_path, 'w') as f:
            json.dump(state.history, f, indent=2)

        # Save feature names
        features_path = self.metadata_dir / f"{state.run_id}_features.json"
        with open(features_path, 'w') as f:
            json.dump(feature_names, f, indent=2)

        state.artifacts = {
            'model_path': str(model_path.relative_to(self.artifacts_dir)),
            'scaler_path': str(scaler_path.relative_to(self.artifacts_dir)),
            'config_path': str(config_path.relative_to(self.artifacts_dir)),
            'model_info_path': str(model_info_path.relative_to(self.artifacts_dir)),
            'history_path': str(history_path.relative_to(self.artifacts_dir)),
            'features_path': str(features_path.relative_to(self.artifacts_dir))
        }

    def get_latest_run(self) -> Optional[TrainingState]:
        """Get the latest training run state."""
        if self.current_state:
            return self.current_state

        # Try to load from metadata
        history_files = list(self.metadata_dir.glob("*_history.json"))
        if not history_files:
            return None

        latest = max(history_files, key=lambda f: f.stat().st_mtime)
        run_id = latest.stem.replace('_history', '')

        # Try to reconstruct state
        state = TrainingState(run_id)
        try:
            with open(latest, 'r') as f:
                state.history = json.load(f)
            state.status = 'completed'
        except:
            pass

        return state


class ProgressCallback(tf.keras.callbacks.Callback):
    """Callback to track training progress."""

    def __init__(self, state: TrainingState):
        super().__init__()
        self.state = state
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.state.current_epoch = epoch + 1
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.state.train_loss = float(logs.get('loss', 0))
        self.state.val_loss = float(logs.get('val_loss', 0)) if 'val_loss' in logs else None
        self.state.history['loss'].append(self.state.train_loss)
        if self.state.val_loss is not None:
            self.state.history['val_loss'].append(self.state.val_loss)

        # Update progress
        if self.state.total_epochs > 0:
            self.state.overall_progress = (self.state.current_epoch / self.state.total_epochs) * 100

        if self.epoch_start_time:
            self.state.elapsed_seconds += time.time() - self.epoch_start_time

    def on_train_end(self, logs=None):
        pass