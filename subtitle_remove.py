#!/usr/bin/env python3
"""Shim → src/workers/subtitle_remove.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from workers.subtitle_remove import main  # noqa: E402

if __name__ == "__main__":
    main()
