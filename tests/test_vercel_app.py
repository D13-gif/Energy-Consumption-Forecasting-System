from pathlib import Path

import pytest

from vercel_app import app


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_health_reports_read_only_capabilities(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "healthy",
        "service": "EnergiCast",
        "mode": "read-only",
        "dataset_available": False,
        "model_available": False,
        "training_available": False,
        "inference_available": False,
    }


def test_dashboard_and_static_assets_are_served(client):
    page = client.get("/")
    asset = client.get("/static/css/app.css")
    project_root = Path(__file__).resolve().parents[1]
    source_assets = project_root / "app" / "static"
    public_assets = project_root / "public" / "static"

    assert page.status_code == 200
    assert b"Mode demo Vercel (read-only)." in page.data
    assert asset.status_code == 200
    assert b".kpi-card" in asset.data
    for source_asset in source_assets.rglob("*"):
        if source_asset.is_file():
            relative_path = source_asset.relative_to(source_assets)
            public_asset = public_assets / relative_path
            response = client.get(f"/static/{relative_path.as_posix()}")

            assert response.status_code == 200
            assert public_asset.read_bytes() == source_asset.read_bytes()


def test_pages_use_adminlte_four_layout_regions(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'data-bs-theme="light"' in response.data
    assert b'class="app-wrapper"' in response.data
    assert b'class="app-header' in response.data
    assert b'class="app-sidebar' in response.data
    assert b'sidebar-menu flex-column' in response.data
    assert b'class="app-main"' in response.data
    assert b'class="app-footer"' in response.data
    assert b'class="main-sidebar"' not in response.data


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/forecast", "post"),
        ("/api/training/start", "post"),
        ("/api/data/summary", "get"),
    ],
)
def test_unsupported_analytics_endpoints_fail_explicitly(client, path, method):
    response = getattr(client, method)(path)

    assert response.status_code == 503
    assert response.get_json()["success"] is False
    assert response.get_json()["dataset_available"] is False
    assert response.get_json()["model_available"] is False


@pytest.mark.parametrize(
    "path",
    [
        "/data",
        "/preprocessing",
        "/eda",
        "/model-training",
        "/model-performance",
        "/forecast",
        "/model-info",
        "/about",
    ],
)
def test_navigation_pages_render(client, path):
    response = client.get(path)

    assert response.status_code == 200
    assert b"Mode demo Vercel (read-only)." in response.data
