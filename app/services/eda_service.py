import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class EDAService:
    def __init__(self, data_service):
        self.data_service = data_service
        self.config = data_service.config

    def get_overview(self) -> Dict:
        """Get descriptive statistics overview."""
        if self.data_service.processed_data is None:
            if not self.data_service.load_preprocessing_artifacts():
                return {'success': False, 'message': 'Preprocessing artifacts not found'}

        df = self.data_service.processed_data
        target_col = self.data_service.target_col

        # Descriptive statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        desc_stats = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                desc_stats[col] = {
                    'count': int(len(series)),
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'q25': float(series.quantile(0.25)),
                    'q50': float(series.quantile(0.50)),
                    'q75': float(series.quantile(0.75)),
                    'max': float(series.max())
                }

        return {
            'success': True,
            'data': {
                'descriptive_stats': desc_stats,
                'target_col': target_col,
                'total_records': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if hasattr(df.index, 'min') else None,
                    'end': df.index.max().isoformat() if hasattr(df.index, 'max') else None
                }
            }
        }

    def get_timeseries(self, start: str = None, end: str = None, range_: str = None) -> Dict:
        """Get time series data for plotting."""
        if self.data_service.processed_data is None:
            if not self.data_service.load_preprocessing_artifacts():
                return {'success': False, 'message': 'Preprocessing artifacts not found'}

        df = self.data_service.processed_data.copy()
        target_col = self.data_service.target_col

        # Filter by date range
        if range_:
            # Calculate from end
            end_date = df.index.max()
            if range_ == '7d':
                start_date = end_date - pd.Timedelta(days=7)
            elif range_ == '30d':
                start_date = end_date - pd.Timedelta(days=30)
            elif range_ == '90d':
                start_date = end_date - pd.Timedelta(days=90)
            else:
                start_date = df.index.min()
            df = df[df.index >= start_date]
        else:
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]

        # Sample if too many points (max 1000 for plotting)
        if len(df) > 1000:
            df = df.resample('h').mean().dropna()  # Already hourly, but ensure
            if len(df) > 1000:
                step = len(df) // 1000 + 1
                df = df.iloc[::step]

        return {
            'success': True,
            'data': {
                'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df[target_col].fillna(0).tolist()
            }
        }

    def get_hourly_profile(self, start: str = None, end: str = None, range_: str = None) -> Dict:
        """Get average hourly profile."""
        if self.data_service.processed_data is None:
            if not self.data_service.load_preprocessing_artifacts():
                return {'success': False, 'message': 'Preprocessing artifacts not found'}

        df = self.data_service.processed_data.copy()
        target_col = self.data_service.target_col

        if range_:
            end_date = df.index.max()
            if range_ == '7d':
                start_date = end_date - pd.Timedelta(days=7)
            elif range_ == '30d':
                start_date = end_date - pd.Timedelta(days=30)
            elif range_ == '90d':
                start_date = end_date - pd.Timedelta(days=90)
            else:
                start_date = df.index.min()
            df = df[df.index >= start_date]
        else:
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]

        # Group by hour
        hourly_avg = df.groupby(df.index.hour)[target_col].mean()

        return {
            'success': True,
            'data': {
                'hours': hourly_avg.index.tolist(),
                'values': hourly_avg.values.tolist()
            }
        }

    def get_weekly_profile(self, start: str = None, end: str = None, range_: str = None) -> Dict:
        """Get average weekly profile."""
        if self.data_service.processed_data is None:
            if not self.data_service.load_preprocessing_artifacts():
                return {'success': False, 'message': 'Preprocessing artifacts not found'}

        df = self.data_service.processed_data.copy()
        target_col = self.data_service.target_col

        if range_:
            end_date = df.index.max()
            if range_ == '7d':
                start_date = end_date - pd.Timedelta(days=7)
            elif range_ == '30d':
                start_date = end_date - pd.Timedelta(days=30)
            elif range_ == '90d':
                start_date = end_date - pd.Timedelta(days=90)
            else:
                start_date = df.index.min()
            df = df[df.index >= start_date]
        else:
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]

        # Group by day of week (0=Monday, 6=Sunday)
        weekly_avg = df.groupby(df.index.dayofweek)[target_col].mean()

        # Reorder to Sunday=0, Monday=1, ..., Saturday=6 for display
        weekly_avg = weekly_avg.reindex([6, 0, 1, 2, 3, 4, 5])

        return {
            'success': True,
            'data': {
                'days': weekly_avg.index.tolist(),
                'values': weekly_avg.values.tolist()
            }
        }

    def get_distribution(self, start: str = None, end: str = None, range_: str = None, bins: int = 30) -> Dict:
        """Get target distribution histogram."""
        if self.data_service.processed_data is None:
            if not self.data_service.load_preprocessing_artifacts():
                return {'success': False, 'message': 'Preprocessing artifacts not found'}

        df = self.data_service.processed_data.copy()
        target_col = self.data_service.target_col

        if range_:
            end_date = df.index.max()
            if range_ == '7d':
                start_date = end_date - pd.Timedelta(days=7)
            elif range_ == '30d':
                start_date = end_date - pd.Timedelta(days=30)
            elif range_ == '90d':
                start_date = end_date - pd.Timedelta(days=90)
            else:
                start_date = df.index.min()
            df = df[df.index >= start_date]
        else:
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]

        values = df[target_col].dropna().values

        if len(values) == 0:
            return {'success': False, 'message': 'No valid data for distribution'}

        hist, bin_edges = np.histogram(values, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'success': True,
            'data': {
                'bins': bin_centers.tolist(),
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        }

    def get_missing_summary(self) -> Dict:
        """Get missing values summary from preprocessing metadata."""
        metadata_path = self.config.METADATA_DIR / 'preprocessing_metadata.json'
        if not metadata_path.exists():
            return {'success': False, 'message': 'Preprocessing metadata not found'}

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Get missing info from raw data if available
        if self.data_service.raw_data is not None:
            from ml.data_loader import get_missing_summary
            missing = get_missing_summary(self.data_service.raw_data)
            return {'success': True, 'data': missing}
        else:
            return {'success': True, 'data': {}}