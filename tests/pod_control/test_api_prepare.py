"""M3 API tests: movies / preview / batches / clip streaming.

Synthetic manifest + fake mp4 bytes under tmp_path per test. AAA structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.pod_control.api import create_app


# ── helpers ----------------------------------------------------------


def _entry(shot_num: int, *, movie: str = "M1",
           category: str = "single", quality_ok: bool = True) -> dict:
    shot_id = f"{movie}/shot_{shot_num:04d}"
    return {
        "shot_id": shot_id,
        "source_movie": movie,
        "path": f"clips/{movie}/shot_{shot_num:04d}.mp4",
        "num_people": 1,
        "shot_category": category,
        "duration_sec": 3.0,
        "width": 1920, "height": 1080, "fps": 24.0,
        "largest_subject_ratio": 0.5,
        "classifier_confidence": 0.95,
        "classified_at": 1729584000.0,
        "quality_ok": quality_ok,
    }


def _write_manifest(output_root: Path, movie: str,
                    entries: list[dict]) -> None:
    md = output_root / "manifest"
    md.mkdir(parents=True, exist_ok=True)
    (md / f"{movie}.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def _write_fake_clip(output_root: Path, movie: str, shot: str,
                     size: int = 8192) -> Path:
    d = output_root / "clips" / movie
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{shot}.mp4"
    p.write_bytes(b"\x00" * size)
    return p


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    data_root.mkdir()
    output_root.mkdir()
    app = create_app(data_root, output_root=output_root)
    return TestClient(app), data_root, output_root


# ── /api/movies ------------------------------------------------------


def test_movies_empty_when_no_manifest(client):
    c, _, _ = client
    r = c.get("/api/movies")
    assert r.status_code == 200
    assert r.json() == {"movies": []}


def test_movies_lists_sorted(client):
    c, _, out = client
    _write_manifest(out, "Zebra",  [_entry(1, movie="Zebra")])
    _write_manifest(out, "Alpha",  [_entry(1, movie="Alpha")])
    data = c.get("/api/movies").json()
    names = [m["movie"] for m in data["movies"]]
    assert names == ["Alpha", "Zebra"]


def test_movies_counts_and_categories(client):
    c, _, out = client
    _write_manifest(out, "M1", [
        _entry(1, category="single"),
        _entry(2, category="dominant"),
        _entry(3, category="landscape", quality_ok=False),
    ])
    m = c.get("/api/movies").json()["movies"][0]
    assert m["total_shots"] == 3
    assert m["quality_ok_count"] == 2
    assert m["by_category"] == {"single": 1, "dominant": 1, "landscape": 1}


# ── /api/movies/{movie}/preview --------------------------------------


def test_preview_404_for_unknown_movie(client):
    c, _, _ = client
    r = c.get("/api/movies/ghost/preview")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "movie_not_found"


def test_preview_default_filter_strips_landscape(client):
    c, _, out = client
    _write_manifest(out, "M1", [
        _entry(1, category="single"),
        _entry(2, category="landscape", quality_ok=False),
    ])
    body = c.get("/api/movies/M1/preview").json()
    assert body["total"] == 1
    assert body["shots"][0]["shot_category"] == "single"


def test_preview_respects_custom_categories(client):
    c, _, out = client
    _write_manifest(out, "M1", [
        _entry(1, category="single"),
        _entry(2, category="wide"),
    ])
    body = c.get("/api/movies/M1/preview?categories=wide").json()
    assert {s["shot_category"] for s in body["shots"]} == {"wide"}


def test_preview_respects_max_shots(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(i) for i in range(1, 21)])
    body = c.get("/api/movies/M1/preview?max_shots=5").json()
    assert body["total"] == 5


def test_preview_pagination(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(i) for i in range(1, 51)])
    p1 = c.get("/api/movies/M1/preview?page=1&page_size=10").json()
    p2 = c.get("/api/movies/M1/preview?page=2&page_size=10").json()
    assert p1["shots"][0]["shot_id"] != p2["shots"][0]["shot_id"]
    assert p1["page"] == 1
    assert p2["page"] == 2
    assert p1["total"] == 50


def test_preview_sample_seed_deterministic(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(i) for i in range(1, 51)])
    a = c.get("/api/movies/M1/preview?sample_seed=42&page_size=10").json()
    b = c.get("/api/movies/M1/preview?sample_seed=42&page_size=10").json()
    assert a["shots"] == b["shots"]
    assert a["sampled"] is True


# ── /api/batches -----------------------------------------------------


def test_create_batch_returns_201_and_shot_count(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(i) for i in range(1, 6)])
    r = c.post("/api/batches", json={
        "name": "b1", "movie": "M1",
        "filter_params": {"categories": ["single"], "skip_bad_quality": True,
                          "skip_landscape": True, "max_shots": None},
    })
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "b1"
    assert data["shot_count"] == 5
    assert data["status"] == "ready"


def test_create_batch_duplicate_returns_409(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(1)])
    payload = {"name": "b1", "movie": "M1",
               "filter_params": {"categories": ["single"]}}
    r1 = c.post("/api/batches", json=payload)
    assert r1.status_code == 201
    r2 = c.post("/api/batches", json=payload)
    assert r2.status_code == 409
    assert r2.json()["detail"]["error"]["code"] == "batch_exists"


def test_create_batch_bad_name_422(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(1)])
    r = c.post("/api/batches", json={
        "name": "has spaces!", "movie": "M1",
        "filter_params": {"categories": ["single"]},
    })
    assert r.status_code == 422


def test_list_batches(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(1)])
    c.post("/api/batches", json={
        "name": "b1", "movie": "M1",
        "filter_params": {"categories": ["single"]},
    })
    data = c.get("/api/batches").json()
    assert [b["name"] for b in data["batches"]] == ["b1"]


def test_get_batch_404(client):
    c, _, _ = client
    r = c.get("/api/batches/nope")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "batch_not_found"


def test_delete_batch_ready_succeeds(client):
    c, _, out = client
    _write_manifest(out, "M1", [_entry(1)])
    c.post("/api/batches", json={
        "name": "b1", "movie": "M1",
        "filter_params": {"categories": ["single"]},
    })
    r = c.delete("/api/batches/b1")
    assert r.status_code == 204
    assert c.get("/api/batches/b1").status_code == 404


def test_delete_batch_missing_404(client):
    c, _, _ = client
    r = c.delete("/api/batches/nope")
    assert r.status_code == 404


# ── /clips/{movie}/{shot}.mp4 ---------------------------------------


def test_clip_404_for_missing_file(client):
    c, _, _ = client
    r = c.get("/clips/Ghost/shot_0001.mp4")
    assert r.status_code == 404


def test_clip_rejects_path_traversal(client):
    c, _, _ = client
    r = c.get("/clips/..%2F..%2Fetc/shot.mp4")
    assert r.status_code in (400, 404)


def test_clip_full_response_when_no_range(client):
    c, _, out = client
    _write_fake_clip(out, "M1", "shot_0001", size=2048)
    r = c.get("/clips/M1/shot_0001.mp4")
    assert r.status_code == 200
    assert len(r.content) == 2048


def test_clip_range_response_returns_206(client):
    c, _, out = client
    _write_fake_clip(out, "M1", "shot_0001", size=4096)
    r = c.get("/clips/M1/shot_0001.mp4",
              headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert len(r.content) == 100
    assert r.headers["content-range"] == "bytes 100-199/4096"


def test_clip_range_open_ended(client):
    c, _, out = client
    _write_fake_clip(out, "M1", "shot_0001", size=4096)
    r = c.get("/clips/M1/shot_0001.mp4",
              headers={"Range": "bytes=2000-"})
    assert r.status_code == 206
    assert len(r.content) == 4096 - 2000


def test_clip_invalid_range_416(client):
    c, _, out = client
    _write_fake_clip(out, "M1", "shot_0001", size=4096)
    r = c.get("/clips/M1/shot_0001.mp4",
              headers={"Range": "bytes=9999-10000"})
    assert r.status_code == 416
