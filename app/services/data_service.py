import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

from ml.data_loader import load_raw_dataset, validate_schema, parse_timestamp, convert_numeric_columns, get_missing_summary
from ml.preprocessing import (PreprocessingConfig, handle_missing_values, aggregate_hourly,
                               chronological_split, fit_scaler, transform_data,
                               save_scaler, save_preprocessing_metadata)
from ml.windowing import create_sequences, prepare_lstm_data, validate_sequence_shapes

logger = logging.getLogger(__name__)


class DataService:
    def __init__(self, config):
        self.config = config
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.scaler = None
        self.preprocessing_config: Optional[PreprocessingConfig] = None
        self.feature_names: List[str] = []
        self.target_col = config.TARGET_COLUMN

    def load_dataset(self, filepath: Path = None) -> Dict:
        """Load and validate raw dataset."""
        if filepath is None:
            filepath = self.config.DATASET_PATH

        self.raw_data = load_raw_dataset(filepath)
        is_valid, errors = validate_schema(self.raw_data)

        return {
            'success': is_valid,
            'errors': errors,
            'shape': self.raw_data.shape,
            'columns': self.raw_data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in self.raw_data.dtypes.items()}
        }

    def get_data_summary(self) -> Dict:
        """Get dataset summary."""
        if self.raw_data is None:
            self.load_dataset()

        df = self.raw_data
        missing_summary = get_missing_summary(df)

        # Get date range after parsing
        df_parsed = parse_timestamp(df.copy())
        date_range = None
        if 'timestamp' in df_parsed.columns:
            valid_ts = df_parsed['timestamp'].dropna()
            if not valid_ts.empty:
                date_range = {
                    'start': valid_ts.min().isoformat(),
                    'end': valid_ts.max().isoformat()
                }

        return {
            'total_records': int(len(df)),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'missing_summary': missing_summary,
            'date_range': date_range,
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }

    def get_data_preview(self, page: int = 1, page_size: int = 25,
                          start: str = None, end: str = None) -> Dict:
        """Get paginated data preview."""
        if self.raw_data is None:
            self.load_dataset()

        df = self.raw_data.copy()

        # Parse timestamps for filtering
        df = parse_timestamp(df)
        df = convert_numeric_columns(df)

        # Filter by date range
        if start:
            df = df[df['timestamp'] >= pd.Timestamp(start)]
        if end:
            df = df[df['timestamp'] <= pd.Timestamp(end)]

        total = len(df)
        total_pages = (total + page_size - 1) // page_size
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        page_data = df.iloc[start_idx:end_idx]

        return {
            'columns': page_data.columns.tolist(),
            'rows': page_data.to_dict('records'),
            'pagination': {
                'current_page': page,
                'page_size': page_size,
                'total_pages': total_pages,
                'total_records': total,
                'showing_from': start_idx + 1 if total > 0 else 0,
                'showing_to': min(end_idx, total)
            }
        }

    def get_validation_summary(self) -> Dict:
        """Get data validation summary."""
        if self.raw_data is None:
            self.load_dataset()

        df = self.raw_data.copy()
        df = parse_timestamp(df)
        df = convert_numeric_columns(df)

        # Check duplicates
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
        else:
            duplicate_timestamps = 0

        missing_summary = get_missing_summary(df)
        total_missing = sum(v['missing_count'] for v in missing_summary.values())
        total_cells = df.size
        missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0

        # Valid rows (no missing in target)
        valid_rows = df[self.target_col].notna().sum() if self.target_col in df.columns else len(df)

        return {
            'total_rows': int(len(df)),
            'total_columns': len(df.columns),
            'valid_rows': int(valid_rows),
            'invalid_rows': int(len(df) - valid_rows),
            'duplicate_timestamps': int(duplicate_timestamps),
            'missing_count': int(total_missing),
            'missing_percentage': round(missing_percentage, 2),
            'missing_summary': missing_summary
        }

    def run_preprocessing(self, config: PreprocessingConfig) -> Dict:
        """Run full preprocessing pipeline."""
        if self.raw_data is None:
            self.load_dataset()

        df = self.raw_data.copy()
        steps = []

        # Step 1: Parse timestamp
        logger.info("Step 1: Parsing timestamps")
        df = parse_timestamp(df)
        steps.append({'name': 'parse_timestamp', 'status': 'completed', 'details': 'Date + Time combined to timestamp'})

        # Step 2: Convert numeric
        logger.info("Step 2: Converting numeric columns")
        df = convert_numeric_columns(df)
        steps.append({'name': 'convert_numeric', 'status': 'completed', 'details': 'Non-numeric values coerced to NaN'})

        # Step 3: Handle missing values
        logger.info("Step 3: Handling missing values")
        df, missing_info = handle_missing_values(
            df.set_index('timestamp') if 'timestamp' in df.columns else df,
            self.target_col,
            strategy=config.missing_strategy
        )
        steps.append({'name': 'handle_missing', 'status': 'completed', 'details': f"Strategy: {config.missing_strategy}, {missing_info}"})

        # Step 4: Sort by timestamp
        logger.info("Step 4: Sorting by timestamp")
        df = df.sort_index()
        steps.append({'name': 'sort', 'status': 'completed', 'details': 'Data sorted chronologically'})

        # Step 5: Aggregate to hourly
        logger.info("Step 5: Aggregating to hourly frequency")
        df, agg_info = aggregate_hourly(df, self.target_col, config.aggregation_freq)
        steps.append({'name': 'aggregate', 'status': 'completed', 'details': f"Frequency: {config.aggregation_freq}, {agg_info}"})

        # Step 6: Split chronologically
        logger.info("Step 6: Chronological split")
        train_df, val_df, test_df, split_info = chronological_split(
            df, config.train_ratio, config.val_ratio, config.test_ratio
        )
        steps.append({'name': 'split', 'status': 'completed', 'details': f"Train/Val/Test: {split_info['train_samples']}/{split_info['val_samples']}/{split_info['test_samples']}"})

        # Step 7: Fit scaler on training data only
        logger.info("Step 7: Fitting scaler on training data")
        feature_cols = [c for c in train_df.columns if c != 'timestamp']
        self.feature_names = feature_cols

        train_values = train_df[feature_cols].values
        self.scaler = fit_scaler(train_values, config.scaler_type)
        steps.append({'name': 'scale', 'status': 'completed', 'details': f"Scaler: {config.scaler_type}, fit on {len(train_values)} training samples"})

        # Step 8: Transform all splits
        logger.info("Step 8: Transforming all splits")
        train_scaled = transform_data(train_df[feature_cols].values, self.scaler)
        val_scaled = transform_data(val_df[feature_cols].values, self.scaler)
        test_scaled = transform_data(test_df[feature_cols].values, self.scaler)

        # Create scaled DataFrames
        train_scaled_df = pd.DataFrame(train_scaled, columns=feature_cols, index=train_df.index)
        train_scaled_df['timestamp'] = train_df['timestamp'].values if 'timestamp' in train_df.columns else train_df.index

        val_scaled_df = pd.DataFrame(val_scaled, columns=feature_cols, index=val_df.index)
        val_scaled_df['timestamp'] = val_df['timestamp'].values if 'timestamp' in val_df.columns else val_df.index

        test_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols, index=test_df.index)
        test_scaled_df['timestamp'] = test_df['timestamp'].values if 'timestamp' in test_df.columns else test_df.index

        # Step 9: Windowing
        logger.info("Step 9: Creating sequences")
        target_idx = feature_cols.index(self.target_col)
        window_size = config.window_size

        X_train, y_train = create_sequences(train_scaled, window_size, target_idx)
        X_val, y_val = create_sequences(val_scaled, window_size, target_idx)
        X_test, y_test = create_sequences(test_scaled, window_size, target_idx)

        # Validate shapes
        validation = validate_sequence_shapes(X_train, y_train, X_val, y_val, X_test, y_test)
        steps.append({'name': 'windowing', 'status': 'completed', 'details': f"Window size: {window_size}, X_train: {X_train.shape}"})

        # Store processed data
        self.processed_data = df
        self.train_data = train_scaled_df
        self.val_data = val_scaled_df
        self.test_data = test_scaled_df
        self.preprocessing_config = config

        # Save artifacts
        self._save_artifacts(df, train_scaled_df, val_scaled_df, test_scaled_df,
                             X_train, y_train, X_val, y_val, X_test, y_test,
                             steps, missing_info, agg_info, split_info, validation)

        # Prepare summary
        summary = {
            'before': {
                'total_rows': len(self.raw_data),
                'date_range': {
                    'start': self.raw_data['Date'].iloc[0] if 'Date' in self.raw_data.columns else None,
                    'end': self.raw_data['Date'].iloc[-1] if 'Date' in self.raw_data.columns else None
                },
                'missing_count': int(self.raw_data.isna().sum().sum()),
                'frequency': 'minute'
            },
            'after': {
                'total_rows': len(df),
                'date_range': split_info['train_range'],
                'missing_count': int(df[self.target_col].isna().sum()),
                'frequency': config.aggregation_freq
            },
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_count': len(feature_cols),
            'window_size': window_size,
            'target_col': self.target_col
        }

        return {
            'success': True,
            'summary': summary,
            'steps': steps,
            'validation': validation
        }

    def _save_artifacts(self, df, train_df, val_df, test_df,
                         X_train, y_train, X_val, y_val, X_test, y_test,
                         steps, missing_info, agg_info, split_info, validation):
        """Save preprocessing artifacts."""
        # Save processed data
        processed_path = self.config.PROCESSED_DATA_DIR / 'processed_data.parquet'
        df.to_parquet(processed_path)

        # Save splits
        train_df.to_parquet(self.config.PROCESSED_DATA_DIR / 'train.parquet')
        val_df.to_parquet(self.config.PROCESSED_DATA_DIR / 'val.parquet')
        test_df.to_parquet(self.config.PROCESSED_DATA_DIR / 'test.parquet')

        # Save sequences
        np.savez(self.config.PROCESSED_DATA_DIR / 'sequences.npz',
                 X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)

        # Save scaler
        save_scaler(self.scaler, self.config.SCALERS_DIR / 'scaler.pkl')

        # Save metadata
        metadata = {
            'config': self.preprocessing_config.to_dict() if self.preprocessing_config else {},
            'feature_names': self.feature_names,
            'target_col': self.target_col,
            'steps': steps,
            'missing_info': missing_info,
            'agg_info': agg_info,
            'split_info': split_info,
            'validation': validation,
            'summary': {
                'before': {
                    'total_rows': len(self.raw_data),
                    'frequency': 'minute'
                },
                'after': {
                    'total_rows': len(df),
                    'frequency': self.preprocessing_config.aggregation_freq if self.preprocessing_config else 'hourly'
                }
            }
        }
        save_preprocessing_metadata(metadata, self.config.METADATA_DIR / 'preprocessing_metadata.json')

    def load_preprocessing_artifacts(self) -> bool:
        """Load existing preprocessing artifacts."""
        try:
            # Load metadata
            metadata_path = self.config.METADATA_DIR / 'preprocessing_metadata.json'
            if not metadata_path.exists():
                return False

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.preprocessing_config = PreprocessingConfig.from_dict(metadata.get('config', {}))
            self.feature_names = metadata.get('feature_names', [])
            self.target_col = metadata.get('target_col', self.target_col)

            # Load processed data
            self.processed_data = pd.read_parquet(self.config.PROCESSED_DATA_DIR / 'processed_data.parquet')
            self.train_data = pd.read_parquet(self.config.PROCESSED_DATA_DIR / 'train.parquet')
            self.val_data = pd.read_parquet(self.config.PROCESSED_DATA_DIR / 'val.parquet')
            self.test_data = pd.read_parquet(self.config.PROCESSED_DATA_DIR / 'test.parquet')

            # Load sequences
            sequences = np.load(self.config.PROCESSED_DATA_DIR / 'sequences.npz')
            self.X_train = sequences['X_train']
            self.y_train = sequences['y_train']
            self.X_val = sequences['X_val']
            self.y_val = sequences['y_val']
            self.X_test = sequences['X_test']
            self.y_test = sequences['y_test']

            # Load scaler
            import joblib
            self.scaler = joblib.load(self.config.SCALERS_DIR / 'scaler.pkl')

            return True
        except Exception as e:
            logger.error(f"Failed to load preprocessing artifacts: {e}")
            return False

    def get_preprocessing_status(self) -> Dict:
        """Get preprocessing status."""
        metadata_path = self.config.METADATA_DIR / 'preprocessing_metadata.json'
        if not metadata_path.exists():
            return {'status': 'not_run', 'message': 'Preprocessing belum dijalankan'}

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return {
                'status': 'completed',
                'config': metadata.get('config', {}),
                'summary': metadata.get('summary', {}),
                'steps': metadata.get('steps', []),
                'validation': metadata.get('validation', {})
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}