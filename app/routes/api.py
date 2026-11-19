from flask import Blueprint, jsonify, request, current_app
from app.services import DataService, EDAService, TrainingService, PredictionService, MetricsService, ArtifactService
from ml.preprocessing import PreprocessingConfig
from app.config import config as app_config

api_bp = Blueprint('api', __name__, url_prefix='/api')


def get_services():
    """Get or create service instances."""
    # Get the config class used for this app
    config_name = current_app.config.get('FLASK_CONFIG', 'default')
    config_class = app_config[config_name]

    if not hasattr(current_app, 'data_service'):
        current_app.data_service = DataService(config_class)
    if not hasattr(current_app, 'eda_service'):
        current_app.eda_service = EDAService(current_app.data_service)
    if not hasattr(current_app, 'training_service'):
        current_app.training_service = TrainingService(current_app.data_service, config_class.ARTIFACTS_DIR)
    if not hasattr(current_app, 'prediction_service'):
        current_app.prediction_service = PredictionService(current_app.data_service, config_class.ARTIFACTS_DIR)
    if not hasattr(current_app, 'metrics_service'):
        current_app.metrics_service = MetricsService(current_app.data_service, config_class.ARTIFACTS_DIR)
    if not hasattr(current_app, 'artifact_service'):
        current_app.artifact_service = ArtifactService(config_class, config_class.ARTIFACTS_DIR)

    return {
        'data': current_app.data_service,
        'eda': current_app.eda_service,
        'training': current_app.training_service,
        'prediction': current_app.prediction_service,
        'metrics': current_app.metrics_service,
        'artifact': current_app.artifact_service
    }


@api_bp.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'EnergiCast'})


# Data endpoints
@api_bp.route('/data/summary')
def data_summary():
    services = get_services()
    result = services['data'].get_data_summary()
    return jsonify({'success': True, 'data': result})


@api_bp.route('/data/preview')
def data_preview():
    services = get_services()
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 25, type=int)
    start = request.args.get('start', type=str)
    end = request.args.get('end', type=str)

    result = services['data'].get_data_preview(page, page_size, start, end)
    return jsonify(result)


@api_bp.route('/data/validation')
def data_validation():
    services = get_services()
    result = services['data'].get_validation_summary()
    return jsonify(result)


@api_bp.route('/data/export')
def data_export():
    services = get_services()
    start = request.args.get('start', type=str)
    end = request.args.get('end', type=str)
    page_size = request.args.get('page_size', 10000, type=int)

    result = services['data'].get_data_preview(1, page_size, start, end)
    # In a real implementation, this would return CSV
    return jsonify({'success': False, 'message': 'Export not implemented yet'})


# Preprocessing endpoints
@api_bp.route('/preprocessing/run', methods=['POST'])
def preprocessing_run():
    services = get_services()
    data = request.get_json() or {}

    config = PreprocessingConfig(
        aggregation_freq=data.get('aggregation_freq', 'h'),
        missing_strategy=data.get('missing_strategy', 'interpolate'),
        train_ratio=data.get('train_ratio', 0.7),
        val_ratio=data.get('val_ratio', 0.15),
        test_ratio=data.get('test_ratio', 0.15),
        window_size=data.get('window_size', 24),
        scaler_type=data.get('scaler_type', 'minmax')
    )

    result = services['data'].run_preprocessing(config)

    # Save run metadata
    if result.get('success'):
        run_id = f"prep_{int(__import__('time').time())}"
        services['artifact'].save_run(
            run_id=run_id,
            run_type='preprocessing',
            status='completed',
            config=config.to_dict(),
            artifacts={'summary': result.get('summary')}
        )

    return jsonify(result)


@api_bp.route('/preprocessing/status')
def preprocessing_status():
    services = get_services()
    result = services['data'].get_preprocessing_status()
    return jsonify({'success': True, 'data': result})


# EDA endpoints
@api_bp.route('/eda/overview')
def eda_overview():
    services = get_services()
    result = services['eda'].get_overview()
    return jsonify(result)


@api_bp.route('/eda/timeseries')
def eda_timeseries():
    services = get_services()
    start = request.args.get('start', type=str)
    end = request.args.get('end', type=str)
    range_ = request.args.get('range', type=str)
    result = services['eda'].get_timeseries(start, end, range_)
    return jsonify(result)


@api_bp.route('/eda/hourly-profile')
def eda_hourly_profile():
    services = get_services()
    start = request.args.get('start', type=str)
    end = request.args.get('end', type=str)
    range_ = request.args.get('range', type=str)
    result = services['eda'].get_hourly_profile(start, end, range_)
    return jsonify(result)


@api_bp.route('/eda/weekly-profile')
def eda_weekly_profile():
    services = get_services()
    start = request.args.get('start', type=str)
    end = request.args.get('end', type=str)
    range_ = request.args.get('range', type=str)
    result = services['eda'].get_weekly_profile(start, end, range_)
    return jsonify(result)


@api_bp.route('/eda/distribution')
def eda_distribution():
    services = get_services()
    start = request.args.get('start', type=str)
    end = request.args.get('end', type=str)
    range_ = request.args.get('range', type=str)
    result = services['eda'].get_distribution(start, end, range_)
    return jsonify(result)


@api_bp.route('/eda/missing-summary')
def eda_missing_summary():
    services = get_services()
    result = services['eda'].get_missing_summary()
    return jsonify(result)


# Training endpoints
@api_bp.route('/training/start', methods=['POST'])
def training_start():
    services = get_services()
    data = request.get_json() or {}

    config = {
        'lstm_units': data.get('lstm_units', 32),
        'dropout': data.get('dropout', 0.1),
        'epochs': data.get('epochs', 20),
        'batch_size': data.get('batch_size', 32),
        'learning_rate': data.get('learning_rate', 0.001),
        'patience': data.get('patience', 5),
        'loss': data.get('loss', 'mse'),
        'optimizer': data.get('optimizer', 'adam')
    }

    result = services['training'].start_training(config)
    return jsonify(result)


@api_bp.route('/training/status/latest')
def training_status_latest():
    services = get_services()
    result = services['training'].get_latest_status()
    return jsonify(result)


@api_bp.route('/training/status/<run_id>')
def training_status(run_id):
    services = get_services()
    result = services['training'].get_training_status(run_id)
    return jsonify(result)


@api_bp.route('/training/stop', methods=['POST'])
def training_stop():
    services = get_services()
    result = services['training'].stop_training()
    return jsonify(result)


# Model endpoints
@api_bp.route('/model/status')
def model_status():
    services = get_services()
    result = services['prediction'].get_model_status()
    return jsonify(result)


@api_bp.route('/model/metrics')
def model_metrics():
    services = get_services()
    result = services['metrics'].get_metrics()
    return jsonify(result)


@api_bp.route('/model/history')
def model_history():
    services = get_services()
    result = services['metrics'].get_metrics()
    if result.get('success') and result.get('data'):
        return jsonify({'success': True, 'data': {'history': result['data'].get('history', {})}})
    return jsonify(result)


@api_bp.route('/model/info')
def model_info():
    services = get_services()
    result = services['metrics'].get_model_info()
    return jsonify(result)


# Forecast endpoints
@api_bp.route('/forecast', methods=['POST'])
def forecast_generate():
    services = get_services()
    data = request.get_json() or {}

    horizon = data.get('horizon', 24)
    model_id = data.get('model_id')

    if not isinstance(horizon, int) or horizon <= 0:
        return jsonify({'success': False, 'message': 'Horizon harus bilangan bulat positif'}), 400

    result = services['prediction'].generate_forecast(horizon, model_id)
    return jsonify(result)


@api_bp.route('/forecast/latest')
def forecast_latest():
    services = get_services()
    result = services['prediction'].get_latest_forecast()
    return jsonify(result)


# Runs endpoint
@api_bp.route('/runs')
def list_runs():
    services = get_services()
    run_type = request.args.get('type', type=str)
    status = request.args.get('status', type=str)
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)

    runs = services['artifact'].list_runs(run_type, status, limit, offset)
    return jsonify({'success': True, 'data': runs})


# Dashboard endpoint
@api_bp.route('/dashboard/summary')
def dashboard_summary():
    services = get_services()
    result = services['artifact'].get_dashboard_summary()
    return jsonify({'success': True, 'data': result})


@api_bp.route('/dashboard/historical')
def dashboard_historical():
    services = get_services()
    range_ = request.args.get('range', '30d', type=str)
    result = services['eda'].get_timeseries(range_=range_)
    return jsonify(result)