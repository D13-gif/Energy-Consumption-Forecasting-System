import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ArtifactService:
    def __init__(self, config, artifacts_dir: Path):
        self.config = config
        self.artifacts_dir = artifacts_dir
        self.db_path = config.SQLITE_DB_PATH
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for run metadata."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_seconds REAL,
                    config TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    error TEXT,
                    dataset_hash TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_runs_type_status ON runs(run_type, status)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC)
            ''')
            conn.commit()

    @contextmanager
    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_run(self, run_id: str, run_type: str, status: str,
                 config: Dict = None, metrics: Dict = None,
                 artifacts: Dict = None, error: str = None,
                 dataset_hash: str = None) -> bool:
        """Save or update run metadata."""
        now = datetime.now().isoformat()
        with self._get_db() as conn:
            # Check if exists
            existing = conn.execute('SELECT id FROM runs WHERE id = ?', (run_id,)).fetchone()
            if existing:
                # Update
                updates = []
                params = []
                if status:
                    updates.append('status = ?')
                    params.append(status)
                if run_type:
                    updates.append('run_type = ?')
                    params.append(run_type)
                if config is not None:
                    updates.append('config = ?')
                    params.append(json.dumps(config))
                if metrics is not None:
                    updates.append('metrics = ?')
                    params.append(json.dumps(metrics))
                if artifacts is not None:
                    updates.append('artifacts = ?')
                    params.append(json.dumps(artifacts))
                if error is not None:
                    updates.append('error = ?')
                    params.append(error)
                if dataset_hash is not None:
                    updates.append('dataset_hash = ?')
                    params.append(dataset_hash)

                if status in ['completed', 'failed']:
                    updates.append('completed_at = ?')
                    params.append(now)

                if updates:
                    params.append(run_id)
                    conn.execute(f'UPDATE runs SET {", ".join(updates)} WHERE id = ?', params)
            else:
                # Insert
                conn.execute('''
                    INSERT INTO runs (id, run_type, status, started_at, config, metrics, artifacts, error, dataset_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (run_id, run_type, status, now,
                      json.dumps(config) if config else None,
                      json.dumps(metrics) if metrics else None,
                      json.dumps(artifacts) if artifacts else None,
                      error, dataset_hash))
            conn.commit()
        return True

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get run by ID."""
        with self._get_db() as conn:
            row = conn.execute('SELECT * FROM runs WHERE id = ?', (run_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def get_latest_run(self, run_type: str = None) -> Optional[Dict]:
        """Get latest run, optionally filtered by type."""
        with self._get_db() as conn:
            if run_type:
                row = conn.execute(
                    'SELECT * FROM runs WHERE run_type = ? ORDER BY created_at DESC LIMIT 1',
                    (run_type,)
                ).fetchone()
            else:
                row = conn.execute(
                    'SELECT * FROM runs ORDER BY created_at DESC LIMIT 1'
                ).fetchone()
            if row:
                return dict(row)
            return None

    def list_runs(self, run_type: str = None, status: str = None,
                  limit: int = 50, offset: int = 0) -> List[Dict]:
        """List runs with optional filters."""
        with self._get_db() as conn:
            query = 'SELECT * FROM runs WHERE 1=1'
            params = []

            if run_type:
                query += ' AND run_type = ?'
                params.append(run_type)
            if status:
                query += ' AND status = ?'
                params.append(status)

            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_dashboard_summary(self) -> Dict:
        """Get summary for dashboard."""
        with self._get_db() as conn:
            # Dataset status
            dataset_status = 'not_checked'
            try:
                from ml.data_loader import load_raw_dataset, validate_schema
                df = load_raw_dataset(self.config.DATASET_PATH)
                is_valid, _ = validate_schema(df)
                dataset_status = 'ready' if is_valid else 'error'
            except:
                dataset_status = 'not_found'

            # Preprocessing status
            prep_run = self.get_latest_run('preprocessing')
            preprocessing_status = prep_run['status'] if prep_run else 'not_run'

            # Model status
            model_run = self.get_latest_run('training')
            model_status = model_run['status'] if model_run else 'not_trained'

            # Forecast status
            forecast_run = self.get_latest_run('forecast')
            forecast_status = forecast_run['status'] if forecast_run else 'not_created'

            # Latest metrics
            latest_metrics = None
            if model_run and model_run['metrics']:
                try:
                    latest_metrics = json.loads(model_run['metrics'])
                except:
                    pass

            # Latest forecast
            latest_forecast = None
            if forecast_run and forecast_run['artifacts']:
                try:
                    artifacts = json.loads(forecast_run['artifacts'])
                    if 'forecast' in artifacts:
                        latest_forecast = artifacts['forecast']
                except:
                    pass

            return {
                'dataset_status': dataset_status,
                'preprocessing_status': preprocessing_status,
                'model_status': model_status,
                'forecast_status': forecast_status,
                'latest_metrics': latest_metrics,
                'latest_forecast': latest_forecast,
                'total_records': len(df) if 'df' in locals() and df is not None else 0,
                'date_range': None,
                'frequency': 'hourly',
                'model_version': model_run['id'] if model_run else None
            }