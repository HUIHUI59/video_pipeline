"""pytest conftest: put src/ and delivery_v1/scripts on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for sub in (
    _ROOT / "src",
    _ROOT / "src" / "runpod",
    _ROOT / "src" / "common",
    _ROOT / "docs" / "labelingStandards" / "external_delivery_v1" / "scripts",
):
    s = str(sub)
    if sub.exists() and s not in sys.path:
        sys.path.insert(0, s)
