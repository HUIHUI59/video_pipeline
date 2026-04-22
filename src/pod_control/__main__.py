"""CLI entry point: python -m src.pod_control --port 8765 --data-root ...

Design doc: docs/superpowers/specs/2026-04-22-pod-control-design.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from .api import create_app


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
        help="uvicorn autoreload (dev only)",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    app = create_app(data_root, output_root=args.output_root)

    print(
        f"[pod_control] data_root={data_root} output_root={args.output_root} "
        f"-> http://{args.host}:{args.port}/",
        flush=True,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
