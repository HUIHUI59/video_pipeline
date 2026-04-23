"""Tests for POST /api/batches/{name}/export — copy clips per movie.

Export is async: POST returns 202 + job_id; client polls
GET /api/exports/{job_id} until status=='done'.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.pod_control.api import create_app


def _wait_export_done(c, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = c.get(f"/api/exports/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"export {job_id} did not finish in {timeout}s")


def _entry(shot_num: int, movie: str = "M1",
           shot_category: str = "single", quality_ok: bool = True,
           duration_sec: float = 3.0) -> dict:
    return {
        "shot_id": f"{movie}/shot_{shot_num:04d}",
        "source_movie": movie,
        "path": f"clips/{movie}/shot_{shot_num:04d}.mp4",
        "num_people": 1, "shot_category": shot_category,
        "duration_sec": duration_sec, "width": 1920, "height": 1080, "fps": 24.0,
        "largest_subject_ratio": 0.5, "classifier_confidence": 0.95,
        "classified_at": 1729584000.0, "quality_ok": quality_ok,
    }


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    data_root.mkdir()
    (output_root / "manifest").mkdir(parents=True)
    for movie in ("M1", "M2"):
        path = output_root / "manifest" / f"{movie}.jsonl"
        path.write_text(
            "\n".join(json.dumps(_entry(i, movie)) for i in range(3)) + "\n"
        )
        clips_dir = output_root / "clips" / movie
        clips_dir.mkdir(parents=True)
        for i in range(3):
            (clips_dir / f"shot_{i:04d}.mp4").write_bytes(b"fake mp4 data")
    app = create_app(data_root, output_root=output_root)
    return TestClient(app), tmp_path


def _mk_batch(c, name="b1", movies=None, max_shots=None):
    fp = {"categories": ["single"]}
    if max_shots is not None:
        fp["max_shots"] = max_shots
    r = c.post("/api/batches", json={
        "name": name,
        "movies": movies or ["M1", "M2"],
        "filter_params": fp,
    })
    r.raise_for_status()


def test_export_404_when_batch_missing(client):
    c, _ = client
    r = c.post("/api/batches/ghost/export", json={"dest_path": "/tmp/x"})
    assert r.status_code == 404


def test_export_rejects_non_absolute_path(client):
    c, _ = client
    _mk_batch(c)
    r = c.post("/api/batches/b1/export", json={"dest_path": "relative/path"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "invalid_dest"


def test_export_rejects_system_root(client):
    c, _ = client
    _mk_batch(c)
    r = c.post("/api/batches/b1/export", json={"dest_path": "/etc"})
    assert r.status_code == 400


def test_export_copies_clips_organized_by_movie(client, tmp_path):
    c, _ = client
    _mk_batch(c)
    dest = tmp_path / "share"
    r = c.post("/api/batches/b1/export", json={"dest_path": str(dest)})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    body = _wait_export_done(c, job_id)
    assert body["status"] == "done"
    assert body["copied"] == 6
    assert body["missing_source"] == 0
    assert set(body["movies"]) == {"M1", "M2"}
    for movie in ("M1", "M2"):
        for i in range(3):
            assert (dest / "clips" / movie / f"shot_{i:04d}.mp4").is_file()
        assert (dest / "manifest" / f"{movie}.jsonl").is_file()
    assert (dest / "batch.json").is_file()
    assert (dest / "README.md").is_file()


def test_export_skips_existing_unless_overwrite(client, tmp_path):
    c, _ = client
    _mk_batch(c)
    dest = tmp_path / "share"
    j1 = c.post("/api/batches/b1/export", json={"dest_path": str(dest)}).json()["job_id"]
    _wait_export_done(c, j1)
    j2 = c.post("/api/batches/b1/export", json={"dest_path": str(dest)}).json()["job_id"]
    body = _wait_export_done(c, j2)
    assert body["copied"] == 0
    assert body["skipped_existing"] == 6
    j3 = c.post("/api/batches/b1/export", json={
        "dest_path": str(dest), "overwrite": True,
    }).json()["job_id"]
    body3 = _wait_export_done(c, j3)
    assert body3["copied"] == 6


def test_export_writes_manifest_subset(client, tmp_path):
    c, _ = client
    _mk_batch(c, max_shots=2)
    dest = tmp_path / "share2"
    job_id = c.post("/api/batches/b1/export",
                    json={"dest_path": str(dest)}).json()["job_id"]
    body = _wait_export_done(c, job_id)
    assert body["copied"] == 2
    total = 0
    for movie in body["movies"]:
        mf = (dest / "manifest" / f"{movie}.jsonl").read_text().strip()
        if mf:
            total += len(mf.splitlines())
    assert total == 2


def test_export_returns_202_with_job_id(client, tmp_path):
    c, _ = client
    _mk_batch(c)
    r = c.post("/api/batches/b1/export",
               json={"dest_path": str(tmp_path / "share3")})
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body and "total" in body
    assert body["total"] == 6


def test_export_status_404_for_unknown_job(client):
    c, _ = client
    r = c.get("/api/exports/notarealjob")
    assert r.status_code == 404


def test_export_rejects_exact_output_root(client, tmp_path):
    c, _ = client
    _mk_batch(c)
    r = c.post("/api/batches/b1/export",
               json={"dest_path": str(tmp_path / "out")})
    assert r.status_code == 400


def test_export_rejects_output_root_clips_subdir(client, tmp_path):
    c, _ = client
    _mk_batch(c)
    r = c.post("/api/batches/b1/export",
               json={"dest_path": str(tmp_path / "out" / "clips")})
    assert r.status_code == 400


def test_export_allows_arbitrary_subdir_of_output_root(client, tmp_path):
    c, _ = client
    _mk_batch(c)
    sibling = tmp_path / "out" / "share_to_friend"
    r = c.post("/api/batches/b1/export", json={"dest_path": str(sibling)})
    assert r.status_code == 202, r.text
    _wait_export_done(c, r.json()["job_id"])
