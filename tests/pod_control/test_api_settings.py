"""Settings API tests: GET/POST /api/settings/output-root + favicon."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.pod_control.api import create_app


def _seed_manifest_root(root):
    """Create an <root>/manifest/X.jsonl so _scan finds it."""
    md = root / "manifest"
    md.mkdir(parents=True)
    (md / "X.jsonl").write_text(json.dumps({
        "shot_id": "X/shot_0001", "source_movie": "X",
        "path": "clips/X/shot_0001.mp4", "num_people": 1,
        "shot_category": "single", "duration_sec": 3.0,
        "width": 1920, "height": 1080, "fps": 24.0,
        "largest_subject_ratio": 0.5, "classifier_confidence": 0.9,
        "classified_at": 1729584000.0, "quality_ok": True,
    }) + "\n")


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out_default"
    data_root.mkdir()
    _seed_manifest_root(output_root)
    app = create_app(data_root, output_root=output_root)
    return TestClient(app), data_root, tmp_path


# ── favicon ---------------------------------------------------------


def test_favicon_returns_204(client):
    c, *_ = client
    r = c.get("/favicon.ico")
    assert r.status_code == 204


# ── GET /api/settings/output-root -----------------------------------


def test_get_output_root_defaults_to_cli(client):
    c, _, tmp = client
    body = c.get("/api/settings/output-root").json()
    assert body["current"] == str((tmp / "out_default").resolve())
    assert body["cli_default"] == str(tmp / "out_default")
    assert str((tmp / "out_default").resolve()) in body["candidates"]


def test_get_candidates_includes_siblings_with_manifest(client):
    c, _, tmp = client
    sibling = tmp / "out_122b"
    _seed_manifest_root(sibling)
    body = c.get("/api/settings/output-root").json()
    cands = body["candidates"]
    assert str(sibling.resolve()) in cands
    assert str((tmp / "out_default").resolve()) in cands


def test_get_candidates_excludes_dirs_without_manifest(client):
    c, _, tmp = client
    plain = tmp / "out_nothing"
    plain.mkdir()
    body = c.get("/api/settings/output-root").json()
    assert str(plain.resolve()) not in body["candidates"]


# ── POST /api/settings/output-root ----------------------------------


def test_post_switches_current(client):
    c, _, tmp = client
    other = tmp / "out_122b"
    _seed_manifest_root(other)
    r = c.post("/api/settings/output-root", json={"path": str(other)})
    assert r.status_code == 200
    assert r.json()["current"] == str(other.resolve())
    assert c.get("/api/settings/output-root").json()["current"] == \
        str(other.resolve())


def test_post_persists_across_new_app(client, tmp_path):
    c, data_root, tmp = client
    other = tmp / "out_persisted"
    _seed_manifest_root(other)
    c.post("/api/settings/output-root", json={"path": str(other)})
    app2 = create_app(data_root, output_root=tmp / "out_default")
    c2 = TestClient(app2)
    body = c2.get("/api/settings/output-root").json()
    assert body["current"] == str(other.resolve())


def test_post_missing_dir_rejected(client):
    c, _, tmp = client
    r = c.post("/api/settings/output-root",
               json={"path": str(tmp / "not_a_dir")})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "invalid_path"


def test_post_without_manifest_rejected(client):
    c, _, tmp = client
    plain = tmp / "out_no_manifest"
    plain.mkdir()
    r = c.post("/api/settings/output-root", json={"path": str(plain)})
    assert r.status_code == 400
    assert "manifest" in r.json()["detail"]["error"]["message"]


def test_switching_propagates_to_movies_endpoint(client):
    """/api/movies reflects the currently-selected root, not the CLI one."""
    c, _, tmp = client
    cli_movies = {m["movie"] for m in c.get("/api/movies").json()["movies"]}
    assert cli_movies == {"X"}
    other = tmp / "out_other"
    md = other / "manifest"
    md.mkdir(parents=True)
    (md / "Y.jsonl").write_text(json.dumps({
        "shot_id": "Y/shot_0001", "source_movie": "Y",
        "path": "clips/Y/shot_0001.mp4", "num_people": 1,
        "shot_category": "single", "duration_sec": 3.0,
        "width": 1920, "height": 1080, "fps": 24.0,
        "largest_subject_ratio": 0.5, "classifier_confidence": 0.9,
        "classified_at": 1729584000.0, "quality_ok": True,
    }) + "\n")
    c.post("/api/settings/output-root", json={"path": str(other)})
    new_movies = {m["movie"] for m in c.get("/api/movies").json()["movies"]}
    assert new_movies == {"Y"}


def test_health_reports_current_root(client):
    c, _, tmp = client
    other = tmp / "out_health"
    _seed_manifest_root(other)
    c.post("/api/settings/output-root", json={"path": str(other)})
    health = c.get("/api/health").json()
    assert health["output_root"] == str(other.resolve())
