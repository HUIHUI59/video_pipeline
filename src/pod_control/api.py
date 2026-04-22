"""FastAPI app for pod_control UI.

M1 scaffold + M3 Prepare page endpoints. Feature endpoints for Pods /
Run / Monitor land in M4–M6 per
docs/superpowers/specs/2026-04-22-pod-control-design.md.
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from . import filter as pcfilter
from .store import Batch, FilterParams, Store, StoreError


_CLIP_CHUNK = 1 << 20  # 1 MiB


class SaveBatchRequest(BaseModel):
    name: str
    movie: str
    filter_params: FilterParams

    @field_validator("name")
    @classmethod
    def _slug_only(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"batch name must be alnum / - / _ only, got {v!r}"
            )
        return v


def _err(code: str, message: str, status: int = 400) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message}},
    )


def create_app(
    data_root: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    data_root: where pods.yaml / batches/ / state.json / runs/ live.
    output_root: pipeline output root (manifest/, clips/, labels/). Used by
        Prepare-page endpoints (M3+) for movie listing and clip streaming.
    """
    app = FastAPI(title="video_pipeline pod_control", version="0.1.0")

    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static"

    data_root_p = Path(data_root)
    output_root_p = Path(output_root) if output_root else None
    store = Store(data_root_p)

    # ── health + index ---------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "module": "pod_control",
            "version": "0.1.0",
            "data_root": str(data_root_p),
            "output_root": str(output_root_p) if output_root_p else None,
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    # ── M3 Prepare ------------------------------------------------------

    def _require_output_root() -> Path:
        if output_root_p is None:
            raise _err(
                "output_root_not_configured",
                "--output-root was not set on the CLI",
                status=500,
            )
        return output_root_p

    @app.get("/api/movies")
    def list_movies() -> dict:
        out = _require_output_root()
        return {"movies": pcfilter.list_movies(out)}

    @app.get("/api/movies/{movie}/preview")
    def preview(
        movie: str,
        categories: str | None = Query(
            None,
            description="comma-separated categories; default single,dominant,multi",
        ),
        skip_bad_quality: bool = True,
        skip_landscape: bool = True,
        max_shots: int | None = None,
        page: int = 1,
        page_size: int = 20,
        sample_seed: int | None = None,
    ) -> dict:
        out = _require_output_root()
        cats = (
            [c.strip() for c in categories.split(",") if c.strip()]
            if categories is not None
            else FilterParams().categories
        )
        params = FilterParams(
            categories=cats,
            skip_bad_quality=skip_bad_quality,
            skip_landscape=skip_landscape,
            max_shots=max_shots,
        )
        try:
            matched = pcfilter.filter_movie(out, movie, params)
        except Exception as ex:
            raise _err("invalid_filter", str(ex))
        if not matched and not pcfilter.manifest_dir(out).joinpath(
            f"{movie}.jsonl"
        ).exists():
            raise _err("movie_not_found", f"no manifest for {movie!r}", status=404)
        return pcfilter.paginate(
            matched,
            page=page,
            page_size=page_size,
            sample_seed=sample_seed,
        )

    @app.post("/api/batches", status_code=201)
    def create_batch(req: SaveBatchRequest) -> dict:
        out = _require_output_root()
        try:
            matched = pcfilter.filter_movie(out, req.movie, req.filter_params)
        except Exception as ex:
            raise _err("invalid_filter", str(ex))
        batch = Batch(
            name=req.name,
            movie=req.movie,
            filter_params=req.filter_params,
            shot_count=len(matched),
        )
        try:
            store.save_batch(batch)
        except StoreError as ex:
            raise _err("batch_exists", str(ex), status=409)
        return batch.model_dump()

    @app.get("/api/batches")
    def list_batches() -> dict:
        return {"batches": [b.model_dump() for b in store.list_batches()]}

    @app.get("/api/batches/{name}")
    def get_batch(name: str) -> dict:
        b = store.get_batch(name)
        if b is None:
            raise _err("batch_not_found", f"batch {name!r} not found", status=404)
        return b.model_dump()

    @app.delete("/api/batches/{name}", status_code=204)
    def delete_batch(name: str) -> Response:
        try:
            store.delete_batch(name)
        except StoreError as ex:
            code = "batch_not_found" if "not found" in str(ex) else "batch_not_ready"
            status = 404 if code == "batch_not_found" else 409
            raise _err(code, str(ex), status=status)
        return Response(status_code=204)

    # ── clip streaming (Range-supporting, for preview) ------------------

    @app.get("/clips/{movie}/{shot}.mp4")
    def clip(movie: str, shot: str, request: Request):
        out = _require_output_root()
        # Reject path components outside the movie dir.
        if "/" in movie or "/" in shot or ".." in movie or ".." in shot:
            raise _err("invalid_path", "illegal segment", status=400)
        path = out / "clips" / movie / f"{shot}.mp4"
        if not path.is_file():
            raise _err("clip_not_found", f"no clip at {path}", status=404)
        file_size = path.stat().st_size
        range_header = request.headers.get("range")
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        if range_header is None:
            return FileResponse(path, media_type=content_type)

        # Range: bytes=start-[end]
        try:
            units, _, rng = range_header.partition("=")
            if units.strip().lower() != "bytes":
                raise ValueError("only bytes ranges supported")
            start_s, _, end_s = rng.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
        except Exception:
            raise _err("invalid_range", range_header, status=416)
        if start >= file_size or end >= file_size or start > end:
            raise _err("invalid_range", range_header, status=416)

        def iter_range(path: Path, start: int, end: int) -> Iterator[bytes]:
            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(_CLIP_CHUNK, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(
            iter_range(path, start, end),
            status_code=206,
            media_type=content_type,
            headers=headers,
        )

    # ── static mount -----------------------------------------------------

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app
