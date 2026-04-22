"""FastAPI app for pod_control UI.

M1 scaffold: only health + static index.html. Feature endpoints land in
M2–M6 per docs/superpowers/specs/2026-04-22-pod-control-design.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def create_app(
    data_root: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    data_root: where pods.yaml / batches/ / state.json / runs/ live.
    output_root: pipeline output root (manifest/, clips/, labels/). Used by
        later milestones for movie listing; M1 accepts it but doesn't read.
    """
    app = FastAPI(title="video_pipeline pod_control", version="0.1.0")

    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static"

    data_root_p = Path(data_root)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "module": "pod_control",
            "version": "0.1.0",
            "data_root": str(data_root_p),
            "output_root": str(output_root) if output_root else None,
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app
