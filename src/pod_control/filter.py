"""Stage 4 manifest scanning + filtering for pod_control.

Delegates the filter semantics to src.runpod.upload._filter_entries so the
UI can never drift from what upload.py actually uploads. This module only
wraps the upload-side helper with a shape convenient for the Prepare page:
a flat list of ManifestEntry dicts (for preview) and summary counts (for
the movie picker).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.runpod.upload import _filter_entries, _iter_manifest_lines

from .store import FilterParams


def manifest_dir(output_root: str | Path) -> Path:
    return Path(output_root) / "manifest"


def list_movies(output_root: str | Path) -> list[dict[str, Any]]:
    """Scan manifest/*.jsonl → [{movie, total_shots, by_category, quality_ok_count}].

    by_category is a dict keyed by shot_category; categories missing from
    a given manifest are omitted rather than zero-filled.
    """
    md = manifest_dir(output_root)
    if not md.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(md.glob("*.jsonl")):
        movie = p.stem
        total = 0
        by_cat: dict[str, int] = {}
        quality_ok = 0
        for _, entry in _iter_manifest_lines(str(md), {movie}):
            total += 1
            cat = entry.shot_category or "unknown"
            by_cat[cat] = by_cat.get(cat, 0) + 1
            if entry.quality_ok:
                quality_ok += 1
        out.append({
            "movie": movie,
            "total_shots": total,
            "by_category": by_cat,
            "quality_ok_count": quality_ok,
        })
    return out


def filter_movie(
    output_root: str | Path,
    movie: str,
    params: FilterParams,
) -> list[dict[str, Any]]:
    """Single-movie wrapper around filter_movies (kept for compat)."""
    return filter_movies(output_root, [movie], params)


def filter_movies(
    output_root: str | Path,
    movies: list[str],
    params: FilterParams,
) -> list[dict[str, Any]]:
    """Apply FilterParams across any number of movies.

    max_shots caps the TOTAL across all movies, matching upload.py
    semantics (the filter loop breaks once the cap is hit).
    """
    md = manifest_dir(output_root)
    if not md.exists() or not movies:
        return []
    raw = list(_iter_manifest_lines(str(md), set(movies)))
    filtered = _filter_entries(
        raw,
        categories=params.categories or None,
        max_shots=params.max_shots,
        skip_bad_quality=params.skip_bad_quality,
        skip_landscape=params.skip_landscape,
    )
    return [e.model_dump() for _movie, e in filtered]


def paginate(
    items: list[dict[str, Any]],
    *,
    page: int = 1,
    page_size: int = 20,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    """Slice + optional random sample. Returns {shots, total, page, page_size}."""
    total = len(items)
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        sampled = items[:]
        rng.shuffle(sampled)
        items = sampled[:page_size]
        return {
            "shots": items,
            "total": total,
            "page": 1,
            "page_size": len(items),
            "sampled": True,
        }
    page = max(1, page)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "shots": items[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "sampled": False,
    }
