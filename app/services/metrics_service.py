import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import json

from ml.evaluator import calculate_metrics, evaluate_model, calculate_test_details

logger = logging.getLogger(__name__)


class MetricsService:
    def __init__(self, data_service, artifacts_dir: Path):
        self.data_service = data_service
        self.config = data_service.config
        self.artifacts_dir = artifacts_dir
        self.models_dir = artifacts_dir / 'models'
        self.scalers_dir = artifacts_dir / 'scalers'
        self.metadata_dir = artifacts_dir / 'metadata'

    def get_metrics(self) -> Dict:
        """Get model metrics on test set."""
        # Find latest model
        model_files = list(self.models_dir.glob("*.keras"))
        if not model_files:
            return {'success': False, 'message': 'Tidak ada model tersedia'}

        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        run_id = latest_model.stem

        # Load model
        from tensorflow.keras import models
        model = models.load_model(str(latest_model))

        # Load scaler
        import joblib
        scaler_path = self.scalers_dir / f"{run_id}_scaler.pkl"
        if not scaler_path.exists():
            scaler_path = self.scalers_dir / 'scaler.pkl'
        scaler = joblib.load(scaler_path)

        # Load sequences
        sequences_path = self.config.PROCESSED_DATA_DIR / 'sequences.npz'
        if not sequences_path.exists():
            return {'success': False, 'message': 'Test sequences tidak ditemukan'}

        sequences = np.load(sequences_path)
        X_test = sequences['X_test']
        y_test = sequences['y_test']

        if len(X_test) == 0:
            return {'success': False, 'message': 'Test set kosong'}

        # Load feature names and target index
        features_path = self.metadata_dir / f"{run_id}_features.json"
        if not features_path.exists():
            features_path = self.metadata_dir / 'preprocessing_metadata.json'
        with open(features_path, 'r') as f:
            if 'preprocessing_metadata' in str(features_path):
                metadata = json.load(f)
                feature_names = metadata.get('feature_names', [])
            else:
                feature_names = json.load(f)

        target_col = self.data_service.target_col
        target_idx = feature_names.index(target_col) if target_col in feature_names else 0

        # Evaluate
        eval_result = evaluate_model(model, X_test, y_test, scaler, target_idx)

        # Calculate detailed test statistics
        test_details = calculate_test_details(
            np.array(eval_result['actual']),
            np.array(eval_result['predicted'])
        )

        # Load training history for loss curves
        history_path = self.metadata_dir / f"{run_id}_history.json"
        history = {}
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)

        # Load split info
        metadata_path = self.metadata_dir / 'preprocessing_metadata.json'
        split_info = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                split_info = metadata.get('split_info', {})

        # Load config
        config_path = self.metadata_dir / f"{run_id}_config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Prepare response
        metrics = eval_result['metrics']

        return {
            'success': True,
            'data': {
                'mae': metrics.get('mae'),
                'rmse': metrics.get('rmse'),
                'mse': metrics.get('mse'),
                'r2': metrics.get('r2'),
                'mape': metrics.get('mape'),
                'correlation': metrics.get('correlation'),
                'test_samples': metrics.get('n_samples'),
                'predictions': eval_result['predicted'],
                'actual': eval_result['actual'],
                'residuals': eval_result['residuals'],
                'test_details': test_details,
                'history': history,
                'split_info': split_info,
                'config': config
            }
        }

    def get_model_info(self) -> Dict:
        """Get model architecture and metadata."""
        model_files = list(self.models_dir.glob("*.keras"))
        if not model_files:
            return {'success': False, 'message': 'Tidak ada model tersedia'}

        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        run_id = latest_model.stem

        # Load model
        from tensorflow.keras import models
        model = models.load_model(str(latest_model))

        # Load model info
        info_path = self.metadata_dir / f"{run_id}_model_info.json"
        model_info = {}
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)

        # Load config
        config_path = self.metadata_dir / f"{run_id}_config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Load preprocessing metadata
        metadata_path = self.metadata_dir / 'preprocessing_metadata.json'
        preprocessing = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                preprocessing = metadata.get('config', {})

        # Load scaler params
        import joblib
        scaler_path = self.scalers_dir / f"{run_id}_scaler.pkl"
        if not scaler_path.exists():
            scaler_path = self.scalers_dir / 'scaler.pkl'
        scaler = joblib.load(scaler_path)
        scaler_params = {
            'type': type(scaler).__name__,
            'n_features': scaler.n_features_in_
        }
        if hasattr(scaler, 'data_min_'):
            scaler_params['data_min'] = scaler.data_min_.tolist()
            scaler_params['data_max'] = scaler.data_max_.tolist()
        if hasattr(scaler, 'mean_'):
            scaler_params['mean'] = scaler.mean_.tolist()
            scaler_params['scale'] = scaler.scale_.tolist()
        if hasattr(scaler, 'center_'):
            scaler_params['center'] = scaler.center_.tolist()
            scaler_params['scale'] = scaler.scale_.tolist()

        return {
            'success': True,
            'data': {
                'architecture': model_info,
                'hyperparams': config,
                'io_info': {
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape,
                    'n_features': model.input_shape[2] if model.input_shape else None,
                    'window_size': model.input_shape[1] if model.input_shape else None,
                    'feature_names': model_info.get('feature_names', []),
                    'target_name': self.data_service.target_col,
                    'target_unit': 'kW'
                },
                'preprocessing': preprocessing,
                'metadata': {
                    'model_version': f"v{len(model_files)}",
                    'trained_at': datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat(),
                    'duration_seconds': model_info.get('duration_seconds'),
                    'best_epoch': model_info.get('best_epoch'),
                    'best_val_loss': model_info.get('best_val_loss'),
                    'final_train_loss': model_info.get('final_train_loss'),
                    'dataset_hash': None,
                    'artifacts_path': str(self.artifacts_dir.relative_to(self.artifacts_dir.parent))
                }
            }
        }