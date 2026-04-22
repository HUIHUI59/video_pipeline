"""M5 API tests: launch / active / kill routes.

Popen and os.setsid patched inside each test — no real subprocess.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.pod_control.api import create_app


def _entry(shot_num: int, movie: str = "M1") -> dict:
    return {
        "shot_id": f"{movie}/shot_{shot_num:04d}",
        "source_movie": movie,
        "path": f"clips/{movie}/shot_{shot_num:04d}.mp4",
        "num_people": 1, "shot_category": "single",
        "duration_sec": 3.0, "width": 1920, "height": 1080, "fps": 24.0,
        "largest_subject_ratio": 0.5, "classifier_confidence": 0.95,
        "classified_at": 1729584000.0, "quality_ok": True,
    }


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    data_root.mkdir()
    output_root.mkdir()
    md = output_root / "manifest"
    md.mkdir()
    (md / "M1.jsonl").write_text(
        "\n".join(json.dumps(_entry(i)) for i in range(5)) + "\n"
    )
    fake_run_all = tmp_path / "run_all.sh"
    fake_run_all.write_text("#!/bin/bash\necho hi\n")
    app = create_app(data_root, output_root=output_root)
    return TestClient(app), data_root, output_root, fake_run_all


def _mk_batch(c, name="b1"):
    r = c.post("/api/batches", json={
        "name": name, "movie": "M1",
        "filter_params": {"categories": ["single"]},
    })
    r.raise_for_status()


def _mk_pod(c, name="h100"):
    r = c.post("/api/pods", json={
        "name": name, "host": "1.2.3.4", "user": "root",
        "ssh_key": "~/.ssh/id_ed25519", "port": 22,
        "workspace": "/workspace/video_pipeline",
    })
    r.raise_for_status()


def test_active_run_null_initially(client):
    c, _, _, _ = client
    assert c.get("/api/runs/active").json()["active_run"] is None


def test_launch_requires_batch_found(client):
    c, _, _, _ = client
    _mk_pod(c)
    r = c.post("/api/runs", json={"batch_name": "ghost", "pod_name": "h100"})
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "batch_not_found"


def test_launch_requires_pod_found(client):
    c, _, _, _ = client
    _mk_batch(c)
    r = c.post("/api/runs", json={"batch_name": "b1", "pod_name": "ghost"})
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "pod_not_found"


def test_kill_nonexistent_run_returns_404(client):
    c, _, _, _ = client
    r = c.post("/api/runs/does-not-exist/kill")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "run_not_found"


def test_history_endpoint_returns_lists(client):
    c, _, _, _ = client
    body = c.get("/api/runs").json()
    assert "active" in body
    assert "history" in body
    assert body["active"] == []
    assert body["history"] == []


def test_quick_launch_requires_movie_found(client):
    c, _, _, _ = client
    _mk_pod(c)
    r = c.post("/api/runs/quick", json={
        "movie": "ghost_film", "pod_name": "h100",
    })
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "movie_not_found"


def test_quick_launch_requires_pod_found(client):
    c, _, _, _ = client
    r = c.post("/api/runs/quick", json={
        "movie": "M1", "pod_name": "ghost",
    })
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "pod_not_found"
