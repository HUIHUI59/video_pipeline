"""M1 smoke tests: module imports cleanly, FastAPI app serves / and /api/health."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.pod_control.api import create_app


def test_create_app_returns_fastapi(tmp_path):
    app = create_app(tmp_path)
    assert app.title == "video_pipeline pod_control"


def test_health_endpoint(tmp_path):
    client = TestClient(create_app(tmp_path, output_root="/tmp/out"))
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["module"] == "pod_control"
    assert data["data_root"] == str(tmp_path)
    assert data["output_root"] == "/tmp/out"


def test_index_served(tmp_path):
    client = TestClient(create_app(tmp_path))
    r = client.get("/")
    assert r.status_code == 200
    assert "Stage 5 Pod Control" in r.text
    assert "/static/app.js" in r.text


def test_static_assets_mounted(tmp_path):
    client = TestClient(create_app(tmp_path))
    for path in ("/static/app.js", "/static/styles.css", "/static/index.html"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} should be served"


def test_main_module_importable():
    # Verifies python -m src.pod_control wiring exists.
    import src.pod_control.__main__ as m  # noqa: F401

    assert hasattr(m, "main")
