import os
from pathlib import Path


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(os.environ.get('DATA_DIR', BASE_DIR / 'data')).resolve()
    RAW_DATA_DIR = Path(os.environ.get('RAW_DATA_DIR', DATA_DIR / 'raw')).resolve()
    PROCESSED_DATA_DIR = Path(
        os.environ.get('PROCESSED_DATA_DIR', DATA_DIR / 'processed')
    ).resolve()
    ARTIFACTS_DIR = Path(
        os.environ.get('ARTIFACTS_DIR', BASE_DIR / 'artifacts')
    ).resolve()
    MODELS_DIR = Path(
        os.environ.get('MODELS_DIR', ARTIFACTS_DIR / 'models')
    ).resolve()
    SCALERS_DIR = Path(
        os.environ.get('SCALERS_DIR', ARTIFACTS_DIR / 'scalers')
    ).resolve()
    PLOTS_DIR = Path(
        os.environ.get('PLOTS_DIR', ARTIFACTS_DIR / 'plots')
    ).resolve()
    METADATA_DIR = Path(
        os.environ.get('METADATA_DIR', ARTIFACTS_DIR / 'metadata')
    ).resolve()
    INSTANCE_DIR = Path(
        os.environ.get('INSTANCE_DIR', BASE_DIR / 'instance')
    ).resolve()

    SQLITE_DB_PATH = Path(
        os.environ.get('SQLITE_DB_PATH', INSTANCE_DIR / 'energicast.sqlite')
    ).resolve()
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{SQLITE_DB_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DATASET_FILENAME = os.environ.get(
        'DATASET_FILENAME', 'household_power_consumption.txt'
    )
    DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

    TARGET_COLUMN = 'Global_active_power'
    DATE_COLUMN = 'Date'
    TIME_COLUMN = 'Time'
    TIMESTAMP_FORMAT = '%d/%m/%Y %H:%M:%S'

    DEFAULT_AGGREGATION_FREQ = 'h'
    DEFAULT_TRAIN_RATIO = 0.7
    DEFAULT_VAL_RATIO = 0.15
    DEFAULT_TEST_RATIO = 0.15
    DEFAULT_WINDOW_SIZE = 24
    DEFAULT_LSTM_UNITS = 32
    DEFAULT_EPOCHS = 20
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_DROPOUT = 0.1

    FORECAST_HORIZONS = [24, 48, 72]

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024


class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    SQLITE_DB_PATH = Config.BASE_DIR / 'instance' / 'test_energicast.sqlite'
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{SQLITE_DB_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATA_DIR = Config.BASE_DIR / 'tests' / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    ARTIFACTS_DIR = Config.BASE_DIR / 'tests' / 'artifacts'
    MODELS_DIR = ARTIFACTS_DIR / 'models'
    SCALERS_DIR = ARTIFACTS_DIR / 'scalers'
    PLOTS_DIR = ARTIFACTS_DIR / 'plots'
    METADATA_DIR = ARTIFACTS_DIR / 'metadata'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}