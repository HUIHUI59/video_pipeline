#!/usr/bin/env python3
"""Shim → src/workers/process_videos.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from workers.process_videos import main  # noqa: E402

if __name__ == "__main__":
    main()
