import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import json
from datetime import datetime

from tensorflow.keras import models
import joblib

from ml.forecaster import run_forecast, evaluate_and_forecast, generate_future_timestamps

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, data_service, artifacts_dir: Path):
        self.data_service = data_service
        self.config = data_service.config
        self.artifacts_dir = artifacts_dir
        self.models_dir = artifacts_dir / 'models'
        self.scalers_dir = artifacts_dir / 'scalers'
        self.metadata_dir = artifacts_dir / 'metadata'

    def get_model_status(self) -> Dict:
        """Check if model is ready for forecasting."""
        # Find latest model
        model_files = list(self.models_dir.glob("*.keras"))
        if not model_files:
            return {
                'success': True,
                'data': {
                    'model_ready': False,
                    'message': 'Tidak ada model yang tersedia. Latih model terlebih dahulu.'
                }
            }

        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        run_id = latest_model.stem

        # Load model info
        info_path = self.metadata_dir / f"{run_id}_model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        else:
            model_info = {}

        # Load config
        config_path = self.metadata_dir / f"{run_id}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                train_config = json.load(f)
        else:
            train_config = {}

        return {
            'success': True,
            'data': {
                'model_ready': True,
                'model_id': run_id,
                'model_version': f"v{len(model_files)}",
                'trained_at': datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat(),
                'input_shape': model_info.get('input_shape'),
                'window_size': train_config.get('window_size'),
                'features': train_config.get('lstm_units', 1),
                'mae': model_info.get('mae'),
                'rmse': model_info.get('rmse')
            }
        }

    def generate_forecast(self, horizon: int, model_id: str = None) -> Dict:
        """Generate forecast for given horizon."""
        # Determine model to use
        if model_id is None:
            model_files = list(self.models_dir.glob("*.keras"))
            if not model_files:
                return {'success': False, 'message': 'Tidak ada model tersedia'}
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            model_id = latest_model.stem
        else:
            latest_model = self.models_dir / f"{model_id}.keras"
            if not latest_model.exists():
                return {'success': False, 'message': f'Model {model_id} tidak ditemukan'}

        # Load model
        model = models.load_model(str(latest_model))

        # Load scaler
        scaler_path = self.scalers_dir / f"{model_id}_scaler.pkl"
        if not scaler_path.exists():
            # Try default scaler
            scaler_path = self.scalers_dir / 'scaler.pkl'
        scaler = joblib.load(scaler_path)

        # Load feature names
        features_path = self.metadata_dir / f"{model_id}_features.json"
        if not features_path.exists():
            features_path = self.metadata_dir / 'preprocessing_metadata.json'
        with open(features_path, 'r') as f:
            if 'preprocessing_metadata' in str(features_path):
                metadata = json.load(f)
                feature_names = metadata.get('feature_names', [])
            else:
                feature_names = json.load(f)

        # Load config
        config_path = self.metadata_dir / f"{model_id}_config.json"
        if not config_path.exists():
            config_path = self.metadata_dir / 'preprocessing_metadata.json'
        with open(config_path, 'r') as f:
            if 'preprocessing_metadata' in str(config_path):
                metadata = json.load(f)
                train_config = metadata.get('config', {})
            else:
                train_config = json.load(f)

        window_size = train_config.get('window_size', 24)
        target_col = self.data_service.target_col
        aggregation_freq = train_config.get('aggregation_freq', 'h')

        # Ensure processed data is loaded
        if self.data_service.processed_data is None:
            if not self.data_service.load_preprocessing_artifacts():
                return {'success': False, 'message': 'Data preprocessing tidak ditemukan'}

        df = self.data_service.processed_data.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            pass
        elif 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            return {'success': False, 'message': 'Data tidak memiliki timestamp index'}

        # Run forecast
        result = run_forecast(
            model=model,
            df=df,
            horizon=horizon,
            window_size=window_size,
            feature_cols=feature_names,
            target_col=target_col,
            scaler=scaler,
            freq=aggregation_freq
        )

        # Also get evaluation predictions for comparison
        sequences_path = self.config.PROCESSED_DATA_DIR / 'sequences.npz'
        if sequences_path.exists():
            sequences = np.load(sequences_path)
            X_test = sequences['X_test']
            y_test = sequences['y_test']

            if len(X_test) > 0:
                # Get test timestamps
                test_df = self.data_service.test_data.copy()
                if 'timestamp' in test_df.columns:
                    test_timestamps = test_df['timestamp'].iloc[window_size:window_size+len(X_test)].tolist()
                else:
                    test_timestamps = test_df.index[window_size:window_size+len(X_test)].tolist()

                eval_result = evaluate_and_forecast(model, X_test, y_test, scaler)
                eval_result['evaluation']['timestamps'] = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in test_timestamps]

                result['evaluation'] = eval_result['evaluation']

        return {
            'success': True,
            'data': result
        }

    def get_latest_forecast(self) -> Dict:
        """Get latest saved forecast."""
        forecast_files = list(self.metadata_dir.glob("*_forecast.json"))
        if not forecast_files:
            return {'success': False, 'message': 'Tidak ada forecast tersimpan'}

        latest = max(forecast_files, key=lambda f: f.stat().st_mtime)
        with open(latest, 'r') as f:
            forecast = json.load(f)

        return {'success': True, 'data': forecast}