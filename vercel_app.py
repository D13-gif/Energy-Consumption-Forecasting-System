import os
from pathlib import Path

from flask import Flask, jsonify

from app.routes.dashboard import bp as dashboard_bp
from app.routes.data import bp as data_bp
from app.routes.eda import bp as eda_bp
from app.routes.forecast import bp as forecast_bp
from app.routes.info import bp as info_bp
from app.routes.performance import bp as performance_bp
from app.routes.preprocessing import bp as preprocessing_bp
from app.routes.training import bp as training_bp


ROOT_DIR = Path(__file__).resolve().parent
API_DISABLED_MESSAGE = (
    "Analisis data, preprocessing, training, dan inference tidak aktif pada "
    "deployment Vercel read-only. Gunakan deployment backend persisten."
)

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "app" / "templates"),
    static_folder=str(ROOT_DIR / "public" / "static"),
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

for blueprint in (
    dashboard_bp,
    data_bp,
    preprocessing_bp,
    eda_bp,
    training_bp,
    performance_bp,
    forecast_bp,
    info_bp,
):
    app.register_blueprint(blueprint)


@app.context_processor
def deployment_context():
    return {"serverless_read_only": True}


@app.get("/health")
@app.get("/api/health")
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "service": "EnergiCast",
            "mode": "read-only",
            "dataset_available": False,
            "model_available": False,
            "training_available": False,
            "inference_available": False,
        }
    )


@app.route(
    "/api", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
)
@app.route(
    "/api/<path:path>",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
def disabled_api(path):
    return (
        jsonify(
            {
                "success": False,
                "message": API_DISABLED_MESSAGE,
                "dataset_available": False,
                "model_available": False,
            }
        ),
        503,
    )
