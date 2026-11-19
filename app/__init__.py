import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template
from app.config import config
from app.extensions import db


def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'default')

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config[config_name])

    for directory in [
        app.config['DATA_DIR'],
        app.config['RAW_DATA_DIR'],
        app.config['PROCESSED_DATA_DIR'],
        app.config['ARTIFACTS_DIR'],
        app.config['MODELS_DIR'],
        app.config['SCALERS_DIR'],
        app.config['PLOTS_DIR'],
        app.config['METADATA_DIR'],
        app.config['INSTANCE_DIR'],
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    configure_logging(app)

    from app.routes.dashboard import bp as dashboard_bp
    from app.routes.data import bp as data_bp
    from app.routes.preprocessing import bp as preprocessing_bp
    from app.routes.eda import bp as eda_bp
    from app.routes.training import bp as training_bp
    from app.routes.forecast import bp as forecast_bp
    from app.routes.performance import bp as performance_bp
    from app.routes.info import bp as info_bp
    from app.routes.api import api_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(performance_bp)
    app.register_blueprint(info_bp)
    app.register_blueprint(api_bp)

    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500

    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'EnergiCast'}

    return app


def configure_logging(app):
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler(
            'logs/energicast.log', maxBytes=10240, backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('EnergiCast startup')