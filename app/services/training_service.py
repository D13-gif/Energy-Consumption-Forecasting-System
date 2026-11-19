import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import json

from ml.trainer import TrainingManager, TrainingState
from ml.model_builder import build_lstm_model

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, data_service, artifacts_dir: Path):
        self.data_service = data_service
        self.config = data_service.config
        self.training_manager = TrainingManager(artifacts_dir)
        self.current_run_id: Optional[str] = None

    def start_training(self, config: Dict) -> Dict:
        """Start model training."""
        # Ensure preprocessing artifacts are loaded
        if self.data_service.processed_data is None:
            if not self.data_service.load_preprocessing_artifacts():
                return {'success': False, 'message': 'Preprocessing artifacts not found. Run preprocessing first.'}

        # Load sequences
        sequences_path = self.config.PROCESSED_DATA_DIR / 'sequences.npz'
        if not sequences_path.exists():
            return {'success': False, 'message': 'Training sequences not found. Run preprocessing first.'}

        sequences = np.load(sequences_path)
        X_train = sequences['X_train']
        y_train = sequences['y_train']
        X_val = sequences['X_val']
        y_val = sequences['y_val']

        # Load scaler
        import joblib
        scaler_path = self.config.SCALERS_DIR / 'scaler.pkl'
        if not scaler_path.exists():
            return {'success': False, 'message': 'Scaler not found. Run preprocessing first.'}
        scaler = joblib.load(scaler_path)

        # Load feature names
        features_path = self.config.METADATA_DIR / 'preprocessing_metadata.json'
        with open(features_path, 'r') as f:
            metadata = json.load(f)
        feature_names = metadata.get('feature_names', [])

        # Start training
        state = self.training_manager.start_training(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scaler=scaler,
            feature_names=feature_names
        )

        self.current_run_id = state.run_id

        return {
            'success': True,
            'data': {
                'run_id': state.run_id,
                'status': state.status
            }
        }

    def get_training_status(self, run_id: str = None) -> Dict:
        """Get training status."""
        if run_id is None:
            run_id = self.current_run_id

        if run_id is None:
            # Try to get latest
            state = self.training_manager.get_latest_run()
            if state is None:
                return {'success': False, 'message': 'No training runs found'}
            run_id = state.run_id
        else:
            # Load specific run status from metadata
            metadata_path = self.config.METADATA_DIR / f"{run_id}_history.json"
            if not metadata_path.exists():
                return {'success': False, 'message': f'Run {run_id} not found'}

            state = TrainingState(run_id)
            with open(metadata_path, 'r') as f:
                state.history = json.load(f)
            state.status = 'completed'

        return {'success': True, 'data': state.to_dict()}

    def get_latest_status(self) -> Dict:
        """Get latest training run status."""
        state = self.training_manager.get_latest_run()
        if state is None:
            return {'success': False, 'message': 'No training runs found'}

        return {'success': True, 'data': state.to_dict()}

    def stop_training(self) -> Dict:
        """Stop current training (not fully implemented - would need threading)."""
        # In a real implementation, this would signal the training thread to stop
        # For now, we just return a message
        return {'success': False, 'message': 'Stop training not implemented in this version'}