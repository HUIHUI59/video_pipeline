"""Repair Stage 4 manifest .jsonl files by dropping malformed lines.

Each line of a manifest .jsonl should be a single valid JSON object that
parses as a ManifestEntry. Concurrent writes (especially over SMB) can
leave behind:

  - empty lines (parse: "Expecting value: line 1 column 1 (char 0)")
  - truncated lines (parse: "Unterminated string ...")
  - two JSONs concatenated on one line (parse: "Extra data ...")

This tool scans every *.jsonl in a manifest dir, validates each line,
and either reports (default = dry-run) or rewrites the file keeping
only valid lines. Original files are backed up to
`<file>.bak.<YYYYMMDDHHMMSS>` so a mistake is recoverable.

Usage:
  # See what would change (no writes):
  python scripts/jsonl_repair.py /mnt/movies/Films/output/manifest

  # Actually fix files:
  python scripts/jsonl_repair.py /mnt/movies/Films/output/manifest --apply

  # Single file:
  python scripts/jsonl_repair.py /path/to/MovieX.jsonl --apply

  # Stricter check (also requires ManifestEntry schema validation, not
  # just JSON parse-ability):
  python scripts/jsonl_repair.py <path> --apply --strict
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _scan_file(path: Path, strict: bool):
    """Return (valid_lines: list[bytes], bad: list[(ln_no, reason)])."""
    valid: list[bytes] = []
    bad: list[tuple[int, str]] = []
    with path.open("rb") as f:
        for ln_no, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped:
                bad.append((ln_no, "empty line"))
                continue
            try:
                obj = json.loads(stripped)
            except Exception as e:
                bad.append((ln_no, f"json: {e}"))
                continue
            if strict:
                try:
                    from src.runpod.schemas import ManifestEntry
                    ManifestEntry.model_validate(obj)
                except Exception as e:
                    bad.append((ln_no, f"schema: {e}"))
                    continue
            valid.append(raw if raw.endswith(b"\n") else raw + b"\n")
    return valid, bad


def _process_one(path: Path, *, apply: bool, strict: bool,
                 min_idle_sec: float, force: bool) -> dict:
    valid, bad = _scan_file(path, strict)
    age_sec = time.time() - path.stat().st_mtime
    in_use = age_sec < min_idle_sec
    stat = {
        "file": str(path),
        "valid": len(valid),
        "bad": len(bad),
        "examples": bad[:3],
        "wrote": False,
        "skipped_in_use": False,
        "age_sec": age_sec,
        "backup": None,
    }
    if not bad:
        return stat
    if apply and in_use and not force:
        # CRITICAL: Stage 4 worker may have an open fd on this inode.
        # rename() doesn't affect that fd — it would keep writing into
        # what becomes our .bak file, and our new file would never see
        # those rows. Refuse unless --force.
        stat["skipped_in_use"] = True
        return stat
    if apply:
        ts = time.strftime("%Y%m%d%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak.{ts}")
        tmp = path.with_suffix(path.suffix + f".tmp.{ts}")
        with tmp.open("wb") as f:
            for line in valid:
                f.write(line)
        path.rename(backup)
        tmp.rename(path)
        stat["wrote"] = True
        stat["backup"] = str(backup)
    return stat


def _iter_targets(target: Path):
    if target.is_file():
        yield target
        return
    if target.is_dir():
        for p in sorted(target.glob("*.jsonl")):
            yield p
        return
    raise SystemExit(f"path is neither a file nor a directory: {target}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", type=Path,
                   help="manifest dir, OR a single .jsonl file")
    p.add_argument("--apply", action="store_true",
                   help="actually rewrite files (default: dry-run report only)")
    p.add_argument("--strict", action="store_true",
                   help="also require ManifestEntry schema validation, "
                        "not just JSON parse-ability")
    p.add_argument("--min-idle-sec", type=float, default=60.0,
                   help="refuse --apply on files modified within the last N "
                        "seconds (Stage 4 worker probably has an open fd). "
                        "Default 60. Use --force to override.")
    p.add_argument("--force", action="store_true",
                   help="ignore --min-idle-sec and rewrite even files that "
                        "look in-use. WARNING: data loss if a writer is "
                        "still appending — Linux rename() doesn't reach the "
                        "writer's open fd; new appends go to the .bak file.")
    args = p.parse_args()

    target = args.path.expanduser().resolve()
    print(f"target: {target}")
    print(f"mode:   {'APPLY (will rewrite)' if args.apply else 'dry-run (no writes)'}")
    print(f"strict: {args.strict}")
    print()

    total_files = 0
    total_bad = 0
    total_valid = 0
    skipped_in_use = 0
    files_with_bad: list[dict] = []
    for path in _iter_targets(target):
        stat = _process_one(path, apply=args.apply, strict=args.strict,
                            min_idle_sec=args.min_idle_sec, force=args.force)
        total_files += 1
        total_valid += stat["valid"]
        total_bad += stat["bad"]
        if stat["bad"] == 0:
            continue
        files_with_bad.append(stat)
        if stat["skipped_in_use"]:
            skipped_in_use += 1
            action = f"SKIPPED (in-use, age={stat['age_sec']:.1f}s)"
        elif stat["wrote"]:
            action = "REWROTE"
        else:
            action = "would drop"
        print(f"  {action} {stat['bad']:>4} bad of "
              f"{stat['valid'] + stat['bad']:>5} lines  in  {Path(stat['file']).name}")
        for ln, reason in stat["examples"]:
            print(f"      L{ln}: {reason}")
        if stat["backup"]:
            print(f"      backup: {stat['backup']}")

    print()
    print(f"summary: {total_files} files scanned, "
          f"{len(files_with_bad)} need(ed) repair, "
          f"{total_bad} bad lines, {total_valid} valid lines")
    if skipped_in_use:
        print(f"WARNING: {skipped_in_use} file(s) skipped because they were "
              f"modified within the last {args.min_idle_sec}s.")
        print("         Stop Stage 4 first, then re-run.")
        print("         (Rewriting an in-use file would lose any rows the "
              "writer appends after the rename.)")
    if not args.apply and total_bad:
        print()
        print("Re-run with --apply to actually fix the files.")
        print("Originals will be saved as <file>.bak.<timestamp>.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
