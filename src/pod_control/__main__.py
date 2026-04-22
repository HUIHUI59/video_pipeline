"""CLI entry point: python -m src.pod_control --port 8765 --data-root ...

Design doc: docs/superpowers/specs/2026-04-22-pod-control-design.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

from .api import create_app


# Env var names used to ferry CLI args into the reload-mode factory
# (uvicorn's --reload requires an import-string app, so the factory
# can't take Python kwargs — we round-trip through the environment).
_ENV_DATA_ROOT = "POD_CONTROL_DATA_ROOT"
_ENV_OUTPUT_ROOT = "POD_CONTROL_OUTPUT_ROOT"


def _reload_factory():
    """Factory that uvicorn calls when --reload is on.

    Reads data_root / output_root from env (set by main() before
    uvicorn.run). Returns a fresh FastAPI app each call so reloads
    rebuild the Store / Runner from disk.
    """
    data_root = os.environ.get(_ENV_DATA_ROOT) or "data/pod_control"
    output_root = os.environ.get(_ENV_OUTPUT_ROOT) or None
    Path(data_root).mkdir(parents=True, exist_ok=True)
    return create_app(data_root, output_root=output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 5 pod control UI (launcher + log monitor)"
    )
    parser.add_argument(
        "--data-root",
        default="data/pod_control",
        help="where pods.yaml / batches/ / state.json / runs/ are stored",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/movies/Films/forCloudKorOutput",
        help="pipeline output root (manifest/, clips/, labels/)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="uvicorn autoreload (dev only); rebuilds the app on file change",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[pod_control] data_root={data_root} output_root={args.output_root} "
        f"-> http://{args.host}:{args.port}/"
        + ("  (--reload on)" if args.reload else ""),
        flush=True,
    )

    if args.reload:
        # uvicorn's --reload requires an import string. Pass the factory
        # by name + ferry the runtime config through env vars so the
        # reload-spawned worker process can rebuild the app.
        os.environ[_ENV_DATA_ROOT] = str(data_root)
        os.environ[_ENV_OUTPUT_ROOT] = str(args.output_root or "")
        uvicorn.run(
            "src.pod_control.__main__:_reload_factory",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=True,
            factory=True,
            reload_dirs=[str(Path(__file__).resolve().parent)],
        )
    else:
        app = create_app(data_root, output_root=args.output_root)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
