"""M4 API tests: pods CRUD + test-connect route.

SSH stays mocked throughout (monkey-patch src.pod_control.api.pcssh.test_connect).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.pod_control.api import create_app
from src.pod_control.ssh import ConnectResult


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    data_root.mkdir()
    output_root.mkdir()
    app = create_app(data_root, output_root=output_root)
    return TestClient(app), data_root


def _pod_body(**overrides):
    base = {
        "name": "h100-01",
        "host": "1.2.3.4",
        "user": "root",
        "ssh_key": "~/.ssh/id_ed25519",
        "port": 22,
        "workspace": "/workspace/video_pipeline",
    }
    base.update(overrides)
    return base


# ── list / get ------------------------------------------------------


def test_list_pods_empty(client):
    c, _ = client
    assert c.get("/api/pods").json() == {"pods": []}


def test_get_pod_404(client):
    c, _ = client
    r = c.get("/api/pods/ghost")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "pod_not_found"


# ── create ----------------------------------------------------------


def test_create_pod_201(client):
    c, _ = client
    r = c.post("/api/pods", json=_pod_body())
    assert r.status_code == 201
    assert r.json()["host"] == "1.2.3.4"
    assert c.get("/api/pods").json()["pods"][0]["name"] == "h100-01"


def test_create_pod_duplicate_409(client):
    c, _ = client
    c.post("/api/pods", json=_pod_body())
    r = c.post("/api/pods", json=_pod_body())
    assert r.status_code == 409
    assert r.json()["detail"]["error"]["code"] == "pod_exists"


def test_create_pod_bad_name_422(client):
    c, _ = client
    r = c.post("/api/pods", json=_pod_body(name="bad name!"))
    assert r.status_code == 422


def test_create_pod_missing_field_422(client):
    c, _ = client
    r = c.post("/api/pods", json={"name": "h100-01"})
    assert r.status_code == 422


# ── update ----------------------------------------------------------


def test_update_pod_changes_host(client):
    c, _ = client
    c.post("/api/pods", json=_pod_body())
    r = c.put("/api/pods/h100-01", json=_pod_body(host="5.6.7.8"))
    assert r.status_code == 200
    assert r.json()["host"] == "5.6.7.8"


def test_update_pod_name_mismatch_400(client):
    c, _ = client
    c.post("/api/pods", json=_pod_body())
    r = c.put("/api/pods/h100-01", json=_pod_body(name="different-name"))
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "name_mismatch"


def test_update_pod_missing_404(client):
    c, _ = client
    r = c.put("/api/pods/ghost", json=_pod_body(name="ghost"))
    assert r.status_code == 404


def test_update_preserves_last_test_fields(client):
    c, _ = client
    c.post("/api/pods", json=_pod_body())
    with patch("src.pod_control.api.pcssh.test_connect") as mock_test:
        mock_test.return_value = ConnectResult(True, 42, "ok")
        c.post("/api/pods/h100-01/test")
    before = c.get("/api/pods/h100-01").json()
    assert before["last_test_ok"] is True
    c.put("/api/pods/h100-01", json=_pod_body(host="9.9.9.9"))
    after = c.get("/api/pods/h100-01").json()
    assert after["host"] == "9.9.9.9"
    assert after["last_test_ok"] is True
    assert after["last_test_at"] == before["last_test_at"]


# ── delete ----------------------------------------------------------


def test_delete_pod_ok(client):
    c, _ = client
    c.post("/api/pods", json=_pod_body())
    r = c.delete("/api/pods/h100-01")
    assert r.status_code == 204
    assert c.get("/api/pods/h100-01").status_code == 404


def test_delete_pod_missing_404(client):
    c, _ = client
    r = c.delete("/api/pods/ghost")
    assert r.status_code == 404


# ── test-connect ---------------------------------------------------


def test_test_connect_missing_pod_404(client):
    c, _ = client
    r = c.post("/api/pods/ghost/test")
    assert r.status_code == 404


def test_test_connect_success_updates_last_test(client):
    c, _ = client
    c.post("/api/pods", json=_pod_body())
    with patch("src.pod_control.api.pcssh.test_connect") as mock_test:
        mock_test.return_value = ConnectResult(True, 123, "ok")
        r = c.post("/api/pods/h100-01/test")
    assert r.status_code == 200
    body = r.json()
    assert body == {"ok": True, "latency_ms": 123, "message": "ok"}
    stored = c.get("/api/pods/h100-01").json()
    assert stored["last_test_ok"] is True
    assert stored["last_test_at"] is not None


def test_test_connect_failure_returns_message(client):
    c, _ = client
    c.post("/api/pods", json=_pod_body())
    with patch("src.pod_control.api.pcssh.test_connect") as mock_test:
        mock_test.return_value = ConnectResult(False, 5, "No route to host")
        r = c.post("/api/pods/h100-01/test")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "No route to host" in body["message"]
    stored = c.get("/api/pods/h100-01").json()
    assert stored["last_test_ok"] is False
