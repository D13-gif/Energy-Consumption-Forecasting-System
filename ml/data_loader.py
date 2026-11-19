import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_raw_dataset(filepath: Path) -> pd.DataFrame:
    """
    Load raw dataset from file.

    Args:
        filepath: Path to the dataset file

    Returns:
        DataFrame with raw data
    """
    logger.info(f"Loading dataset from {filepath}")

    try:
        df = pd.read_csv(
            filepath,
            sep=';',
            decimal=',',
            na_values=['?', 'nan', 'NaN', ''],
            low_memory=False
        )
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate dataset schema.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    expected_columns = [
        'Date', 'Time', 'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity', 'Sub_metering_1',
        'Sub_metering_2', 'Sub_metering_3'
    ]

    errors = []

    # Check columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")

    # Check for empty dataframe
    if df.empty:
        errors.append("Dataset is empty")

    # Check minimum rows
    if len(df) < 100:
        errors.append(f"Dataset too small: {len(df)} rows (minimum 100)")

    return len(errors) == 0, errors


def parse_timestamp(df: pd.DataFrame, date_col: str = 'Date', time_col: str = 'Time',
                    timestamp_format: str = '%d/%m/%Y %H:%M:%S') -> pd.DataFrame:
    """
    Parse Date and Time columns into a single timestamp column.

    Args:
        df: DataFrame with Date and Time columns
        date_col: Name of date column
        time_col: Name of time column
        timestamp_format: Format string for parsing

    Returns:
        DataFrame with added 'timestamp' column
    """
    df = df.copy()

    # Combine date and time
    df['timestamp'] = pd.to_datetime(
        df[date_col].astype(str) + ' ' + df[time_col].astype(str),
        format=timestamp_format,
        errors='coerce'
    )

    # Count parsing failures
    nat_count = df['timestamp'].isna().sum()
    if nat_count > 0:
        logger.warning(f"Failed to parse {nat_count} timestamps")

    return df


def convert_numeric_columns(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Convert columns to numeric, coercing errors to NaN.

    Args:
        df: DataFrame to convert
        exclude_cols: Columns to exclude from conversion

    Returns:
        DataFrame with numeric columns
    """
    if exclude_cols is None:
        exclude_cols = ['Date', 'Time', 'timestamp']

    df = df.copy()
    numeric_cols = [c for c in df.columns if c not in exclude_cols]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def get_missing_summary(df: pd.DataFrame) -> dict:
    """
    Get missing value summary per column.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with missing value info per column
    """
    summary = {}
    for col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / len(df)) * 100 if len(df) > 0 else 0
        summary[col] = {
            'missing_count': int(missing),
            'missing_percentage': round(pct, 2),
            'dtype': str(df[col].dtype)
        }
    return summary