"""M2 unit tests for src/pod_control/filter.py.

Builds synthetic Stage 4 manifests under tmp_path and exercises
list_movies / filter_movie / paginate.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.pod_control.filter import filter_movie, list_movies, paginate
from src.pod_control.store import FilterParams


def _manifest_entry(
    shot_num: int,
    *,
    movie: str = "TestMovie",
    category: str = "single",
    quality_ok: bool = True,
    duration: float = 3.0,
) -> dict:
    """Minimum-valid ManifestEntry dict."""
    shot_id = f"{movie}/shot_{shot_num:04d}"
    return {
        "shot_id": shot_id,
        "source_movie": movie,
        "path": f"clips/{movie}/shot_{shot_num:04d}.mp4",
        "num_people": 1,
        "shot_category": category,
        "duration_sec": duration,
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
        "largest_subject_ratio": 0.5,
        "classifier_confidence": 0.95,
        "classified_at": 1729584000.0,
        "quality_ok": quality_ok,
    }


def _write_manifest(root: Path, movie: str, entries: list[dict]) -> None:
    md = root / "manifest"
    md.mkdir(parents=True, exist_ok=True)
    (md / f"{movie}.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


# ── list_movies ------------------------------------------------------


def test_list_movies_empty_when_no_manifest(tmp_path):
    assert list_movies(tmp_path) == []


def test_list_movies_counts_by_category(tmp_path):
    entries = [
        _manifest_entry(1, category="single"),
        _manifest_entry(2, category="single"),
        _manifest_entry(3, category="dominant"),
        _manifest_entry(4, category="landscape", quality_ok=False),
    ]
    _write_manifest(tmp_path, "M1", entries)
    res = list_movies(tmp_path)
    assert len(res) == 1
    r = res[0]
    assert r["movie"] == "M1"
    assert r["total_shots"] == 4
    assert r["by_category"] == {"single": 2, "dominant": 1, "landscape": 1}
    assert r["quality_ok_count"] == 3


def test_list_movies_multiple_movies_sorted(tmp_path):
    _write_manifest(tmp_path, "ZebraFilm",  [_manifest_entry(1)])
    _write_manifest(tmp_path, "AlphaFilm",  [_manifest_entry(1)])
    _write_manifest(tmp_path, "MiddleFilm", [_manifest_entry(1)])
    names = [r["movie"] for r in list_movies(tmp_path)]
    assert names == ["AlphaFilm", "MiddleFilm", "ZebraFilm"]


# ── filter_movie -----------------------------------------------------


def test_filter_movie_respects_categories(tmp_path):
    _write_manifest(tmp_path, "M1", [
        _manifest_entry(1, category="single"),
        _manifest_entry(2, category="dominant"),
        _manifest_entry(3, category="wide"),
        _manifest_entry(4, category="landscape"),
    ])
    fp = FilterParams(categories=["single", "dominant"])
    out = filter_movie(tmp_path, "M1", fp)
    cats = sorted(e["shot_category"] for e in out)
    assert cats == ["dominant", "single"]


def test_filter_movie_skip_landscape_default(tmp_path):
    _write_manifest(tmp_path, "M1", [
        _manifest_entry(1, category="single"),
        _manifest_entry(2, category="landscape"),
    ])
    fp = FilterParams(categories=[])  # allow any, rely on skip_landscape
    out = filter_movie(tmp_path, "M1", fp)
    assert [e["shot_category"] for e in out] == ["single"]


def test_filter_movie_skip_bad_quality(tmp_path):
    _write_manifest(tmp_path, "M1", [
        _manifest_entry(1, quality_ok=True),
        _manifest_entry(2, quality_ok=False),
    ])
    fp = FilterParams(skip_bad_quality=True)
    out = filter_movie(tmp_path, "M1", fp)
    assert len(out) == 1
    assert out[0]["quality_ok"] is True


def test_filter_movie_include_bad_quality_when_flag_off(tmp_path):
    _write_manifest(tmp_path, "M1", [
        _manifest_entry(1, quality_ok=True),
        _manifest_entry(2, quality_ok=False),
    ])
    fp = FilterParams(skip_bad_quality=False)
    out = filter_movie(tmp_path, "M1", fp)
    assert len(out) == 2


def test_filter_movie_max_shots_caps(tmp_path):
    _write_manifest(tmp_path, "M1",
                    [_manifest_entry(i) for i in range(1, 11)])
    fp = FilterParams(max_shots=3)
    out = filter_movie(tmp_path, "M1", fp)
    assert len(out) == 3


def test_filter_movie_missing_manifest_returns_empty(tmp_path):
    assert filter_movie(tmp_path, "Nope", FilterParams()) == []


# ── paginate ---------------------------------------------------------


def test_paginate_default_slice():
    items = [{"i": i} for i in range(50)]
    r = paginate(items, page=1, page_size=20)
    assert r["total"] == 50
    assert r["page"] == 1
    assert len(r["shots"]) == 20
    assert r["shots"][0]["i"] == 0
    assert r["sampled"] is False


def test_paginate_page_2_offsets_correctly():
    items = [{"i": i} for i in range(50)]
    r = paginate(items, page=2, page_size=20)
    assert r["shots"][0]["i"] == 20
    assert r["shots"][-1]["i"] == 39


def test_paginate_sample_seed_is_deterministic():
    items = [{"i": i} for i in range(50)]
    r1 = paginate(items, page_size=10, sample_seed=42)
    r2 = paginate(items, page_size=10, sample_seed=42)
    assert r1["shots"] == r2["shots"]
    assert r1["sampled"] is True
    assert len(r1["shots"]) == 10


def test_paginate_sample_different_seed_gives_different_result():
    items = [{"i": i} for i in range(50)]
    r1 = paginate(items, page_size=10, sample_seed=1)
    r2 = paginate(items, page_size=10, sample_seed=2)
    assert r1["shots"] != r2["shots"]


def test_paginate_page_clamps_to_1():
    items = [{"i": 0}, {"i": 1}]
    r = paginate(items, page=0, page_size=5)
    assert r["page"] == 1
    assert r["shots"] == [{"i": 0}, {"i": 1}]


def test_paginate_past_end_returns_empty_slice():
    items = [{"i": 0}]
    r = paginate(items, page=5, page_size=10)
    assert r["shots"] == []
    assert r["total"] == 1
