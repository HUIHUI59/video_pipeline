#!/usr/bin/env python3
"""Standalone filter for Stage-4-processed clips.

Keeps only shots that passed the automatic quality gate and contain between
``--min-faces`` and ``--max-faces`` detected faces (default: 1 to 4).

Zero external dependencies: Python 3.10+ standard library only.

Typical usage:

    # Auto-detect the mounted forCloudKorOutput directory:
    python filter_clips.py

    # Or pass the path explicitly if auto-detect fails:
    python filter_clips.py --output-root /Volumes/movies/Films/forCloudKorOutput

    # Preview stats without writing anything:
    python filter_clips.py --dry-run

Outputs placed under ``--out-dir`` (default: ``<output-root>/filtered_for_prof``):

    selected_shots.jsonl    # one JSON per line
    selected_shots.tsv      # Excel / LibreOffice friendly
    summary.json            # per-movie / category / face-count totals
    <movie>/shot_NNNN.mp4   # symlink (or copy, depending on --mode)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


TARGET_DIR_NAME = "forCloudKorOutput"
CANDIDATE_BASES: tuple[str, ...] = (
    "/mnt/movies/Films",       # Linux / WSL
    "/mnt",
    "/media",
    "/Volumes/movies/Films",   # macOS
    "/Volumes",
    "/mnt/nas/Films",
)


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    source_movie: str
    shot_stem: str
    num_faces: int
    num_people: int
    shot_category: str
    quality_ok: bool
    largest_face_ratio: float
    duration_sec: float
    src_path: Path
    # v3: 可选字段，旧 manifest（无 quality_metrics.camera_motion）为 None
    camera_motion: "float | None" = None


def _iter_manifest(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _shot_stem(shot_id: str) -> str:
    return shot_id.split("/", 1)[1] if "/" in shot_id else shot_id


def _copy_bytes(src: Path, dst: Path) -> None:
    """Copy file contents only.

    ``shutil.copy2`` fails on many SMB / CIFS network mounts because preserving
    mtime requires ``os.utime`` permission that the share usually denies. Since
    this tool's output is a curated mirror, timestamps carry no meaning — just
    copy bytes and let the destination pick a fresh mtime.
    """
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=1 << 20)


def _looks_like_output_root(p: Path) -> bool:
    return (p / "manifest").is_dir() and (p / "clips").is_dir()


def auto_detect_output_root() -> Path | None:
    """Try a handful of common mount points, plus CWD and its ancestors."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if candidate.name == TARGET_DIR_NAME and _looks_like_output_root(candidate):
            return candidate.resolve()
        sub = candidate / TARGET_DIR_NAME
        if sub.is_dir() and _looks_like_output_root(sub):
            return sub.resolve()

    for base_s in CANDIDATE_BASES:
        base = Path(base_s)
        if not base.is_dir():
            continue
        direct = base / TARGET_DIR_NAME
        if direct.is_dir() and _looks_like_output_root(direct):
            return direct.resolve()
        try:
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                nested = child / TARGET_DIR_NAME
                if nested.is_dir() and _looks_like_output_root(nested):
                    return nested.resolve()
        except PermissionError:
            continue
    return None


def collect_records(
    output_root: Path,
    movies: Iterable[str] | None,
    min_faces: int,
    max_faces: int,
    require_quality_ok: bool,
    min_people: int,
    min_face_ratio: float,
    exclude_categories: Iterable[str],
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
    max_camera_motion: "float | None" = None,
) -> list[ShotRecord]:
    manifest_dir = output_root / "manifest"
    clips_root = output_root / "clips"
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"manifest dir not found: {manifest_dir}")
    if not clips_root.is_dir():
        raise FileNotFoundError(f"clips dir not found: {clips_root}")

    wanted = set(movies) if movies else None
    excluded_cats = {c.lower() for c in exclude_categories}

    records: list[ShotRecord] = []
    for mpath in sorted(manifest_dir.glob("*.jsonl")):
        movie = mpath.stem
        if wanted is not None and movie not in wanted:
            continue
        for entry in _iter_manifest(mpath):
            nf = entry.get("num_faces")
            qok = entry.get("quality_ok")
            if nf is None:
                continue
            if not (min_faces <= int(nf) <= max_faces):
                continue
            if require_quality_ok and qok is not True:
                continue
            num_people = int(entry.get("num_people") or 0)
            if num_people < min_people:
                continue
            category = str(entry.get("shot_category") or "").lower()
            if category in excluded_cats:
                continue
            face_ratio_raw = entry.get("largest_face_ratio")
            # If the field is missing (v1 manifests), skip the ratio gate instead
            # of treating missing as 0.0 (which would reject everything).
            if face_ratio_raw is not None and float(face_ratio_raw) < min_face_ratio:
                continue
            duration = float(entry.get("duration_sec") or 0.0)
            if duration and not (min_duration <= duration <= max_duration):
                continue
            # Camera motion: read from quality_metrics.camera_motion; if missing
            # (v2 manifests), skip the gate so old data still passes.
            qm = entry.get("quality_metrics") or {}
            cam_motion = qm.get("camera_motion")
            if (max_camera_motion is not None
                    and cam_motion is not None
                    and float(cam_motion) > max_camera_motion):
                continue
            sid = entry.get("shot_id")
            if not isinstance(sid, str):
                continue
            stem = _shot_stem(sid)
            src = clips_root / movie / f"{stem}.mp4"
            if not src.exists():
                continue
            records.append(
                ShotRecord(
                    shot_id=sid,
                    source_movie=movie,
                    shot_stem=stem,
                    num_faces=int(nf),
                    num_people=num_people,
                    shot_category=str(entry.get("shot_category") or ""),
                    quality_ok=bool(qok) if qok is not None else False,
                    largest_face_ratio=float(face_ratio_raw or 0.0),
                    duration_sec=duration,
                    src_path=src,
                    camera_motion=(float(cam_motion)
                                   if cam_motion is not None else None),
                )
            )
    return records


def materialize(
    records: list[ShotRecord],
    out_dir: Path,
    mode: str,
    overwrite: bool,
) -> list[tuple[ShotRecord, Path | None]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "playlist":
        return [(rec, None) for rec in records]

    symlink_ok = mode == "symlink"
    first_symlink_error: str | None = None
    out: list[tuple[ShotRecord, Path | None]] = []
    for rec in records:
        if mode == "none":
            out.append((rec, None))
            continue
        movie_dir = out_dir / rec.source_movie
        movie_dir.mkdir(parents=True, exist_ok=True)
        dst = movie_dir / f"{rec.shot_stem}.mp4"
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                out.append((rec, dst))
                continue
            try:
                dst.unlink()
            except OSError:
                pass
        if mode == "symlink" and symlink_ok:
            try:
                os.symlink(rec.src_path.resolve(), dst)
            except OSError as e:
                # First failure: announce once and switch the whole batch to
                # copy (avoids spamming one line per file, which happens over
                # SMB on Windows without admin).
                first_symlink_error = str(e)
                symlink_ok = False
                print(
                    "[filter] symlink not permitted on this filesystem "
                    f"({e}).",
                    file=sys.stderr,
                )
                print(
                    "[filter] switching the rest of this run to --mode copy. "
                    "Tip: re-run with --mode playlist to avoid duplicating bytes.",
                    file=sys.stderr,
                )
                _copy_bytes(rec.src_path, dst)
        elif mode == "symlink" and not symlink_ok:
            _copy_bytes(rec.src_path, dst)
        elif mode == "copy":
            _copy_bytes(rec.src_path, dst)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        out.append((rec, dst))
    if first_symlink_error is not None:
        print(
            "[filter] note: this run fell back to copy after the first "
            "symlink failure; use --mode playlist next time on SMB / Windows.",
            file=sys.stderr,
        )
    return out


def write_playlists(
    rows: list[tuple[ShotRecord, Path | None]],
    out_dir: Path,
) -> list[Path]:
    """Write a global and per-movie ``.m3u8`` playlist pointing at source clips."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_movie: dict[str, list[ShotRecord]] = {}
    for rec, _ in rows:
        per_movie.setdefault(rec.source_movie, []).append(rec)

    written: list[Path] = []

    global_m3u = out_dir / "selected_shots.m3u8"
    with global_m3u.open("w", encoding="utf-8-sig", newline="\n") as f:
        f.write("#EXTM3U\n")
        for movie in sorted(per_movie):
            for rec in sorted(per_movie[movie], key=lambda r: r.shot_stem):
                f.write(
                    f"#EXTINF:{int(max(1, round(rec.duration_sec)))},"
                    f"{rec.source_movie} / {rec.shot_stem} "
                    f"(faces={rec.num_faces}, cat={rec.shot_category})\n"
                )
                f.write(str(rec.src_path) + "\n")
    written.append(global_m3u)

    for movie, recs in per_movie.items():
        movie_dir = out_dir / movie
        movie_dir.mkdir(parents=True, exist_ok=True)
        m3u = movie_dir / "playlist.m3u8"
        with m3u.open("w", encoding="utf-8-sig", newline="\n") as f:
            f.write("#EXTM3U\n")
            for rec in sorted(recs, key=lambda r: r.shot_stem):
                f.write(
                    f"#EXTINF:{int(max(1, round(rec.duration_sec)))},"
                    f"{rec.shot_stem} (faces={rec.num_faces}, cat={rec.shot_category})\n"
                )
                f.write(str(rec.src_path) + "\n")
        written.append(m3u)
    return written


def write_manifest(
    rows: list[tuple[ShotRecord, Path | None]],
    out_dir: Path,
    output_root: Path,
    filter_meta: dict,
    total_matched: int,
    sampled: bool,
) -> Path:
    """Write ``shots_manifest.json`` — single top-level JSON consumers can load."""
    path = out_dir / "shots_manifest.json"
    doc = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "output_root": str(output_root),
        "filter": filter_meta,
        "total_matched": total_matched,
        "total_returned": len(rows),
        "sampled": sampled,
        "shots": [
            {
                "shot_id": rec.shot_id,
                "source_movie": rec.source_movie,
                "shot_category": rec.shot_category,
                "num_faces": rec.num_faces,
                "num_people": rec.num_people,
                "largest_face_ratio": rec.largest_face_ratio,
                "duration_sec": rec.duration_sec,
                "camera_motion": rec.camera_motion,
                "src_path": str(rec.src_path),
                "dst_path": str(dst) if dst else None,
            }
            for rec, dst in rows
        ],
    }
    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _sample_records(
    records: list[ShotRecord],
    limit: int | None,
    shuffle: bool,
    seed: int | None,
) -> tuple[list[ShotRecord], bool]:
    """Return (possibly-sampled, sampled_flag)."""
    if not shuffle and (limit is None or limit >= len(records)):
        return records, False
    ordered = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)
    if limit is not None and limit < len(ordered):
        ordered = ordered[:limit]
    return ordered, True


def write_listings(
    rows: list[tuple[ShotRecord, Path | None]],
    out_dir: Path,
) -> tuple[Path, Path]:
    jsonl_path = out_dir / "selected_shots.jsonl"
    tsv_path = out_dir / "selected_shots.tsv"
    with jsonl_path.open("w", encoding="utf-8") as jf, tsv_path.open(
        "w", encoding="utf-8"
    ) as tf:
        tf.write(
            "shot_id\tmovie\tcategory\tnum_faces\tnum_people\t"
            "largest_face_ratio\tduration_sec\tcamera_motion\tquality_ok\t"
            "src_path\tdst_path\n"
        )
        for rec, dst in rows:
            cm_str = f"{rec.camera_motion:.3f}" if rec.camera_motion is not None else ""
            tf.write(
                f"{rec.shot_id}\t{rec.source_movie}\t{rec.shot_category}\t"
                f"{rec.num_faces}\t{rec.num_people}\t"
                f"{rec.largest_face_ratio:.4f}\t{rec.duration_sec:.3f}\t"
                f"{cm_str}\t{rec.quality_ok}\t{rec.src_path}\t{dst or ''}\n"
            )
            jf.write(
                json.dumps(
                    {
                        "shot_id": rec.shot_id,
                        "source_movie": rec.source_movie,
                        "shot_category": rec.shot_category,
                        "num_faces": rec.num_faces,
                        "num_people": rec.num_people,
                        "largest_face_ratio": rec.largest_face_ratio,
                        "duration_sec": rec.duration_sec,
                        "camera_motion": rec.camera_motion,
                        "quality_ok": rec.quality_ok,
                        "src_path": str(rec.src_path),
                        "dst_path": str(dst) if dst else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return jsonl_path, tsv_path


def summarize(rows: list[tuple[ShotRecord, Path | None]]) -> dict:
    per_movie: dict[str, int] = {}
    per_category: dict[str, int] = {}
    per_face_count: dict[int, int] = {}
    total_sec = 0.0
    for rec, _ in rows:
        per_movie[rec.source_movie] = per_movie.get(rec.source_movie, 0) + 1
        per_category[rec.shot_category] = per_category.get(rec.shot_category, 0) + 1
        per_face_count[rec.num_faces] = per_face_count.get(rec.num_faces, 0) + 1
        total_sec += rec.duration_sec
    return {
        "total_shots": len(rows),
        "total_duration_sec": round(total_sec, 1),
        "per_movie": dict(sorted(per_movie.items(), key=lambda kv: -kv[1])),
        "per_category": per_category,
        "per_face_count": dict(sorted(per_face_count.items())),
    }


def _resolve_output_root(raw: str | None) -> Path:
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            sys.exit(f"[filter] --output-root does not exist: {p}")
        if not _looks_like_output_root(p):
            sys.exit(
                f"[filter] {p} does not look like a pipeline output root "
                f"(missing 'manifest/' or 'clips/' subdirectory)"
            )
        return p
    auto = auto_detect_output_root()
    if auto is None:
        sys.exit(
            "[filter] Could not find 'forCloudKorOutput/' automatically. "
            "Pass the path explicitly with --output-root "
            "(it must contain both 'manifest/' and 'clips/' subfolders).\n"
            f"[filter] Locations searched: {', '.join(CANDIDATE_BASES)}"
        )
    print(f"[filter] auto-detected --output-root = {auto}", file=sys.stderr)
    return auto


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter Stage-4-processed clips by quality_ok + face count.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="path to 'forCloudKorOutput' (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write listings + clip links (default: <output-root>/filtered_for_prof)",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=1,
        help="minimum num_faces, inclusive (default: 1)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=3,
        help="maximum num_faces, inclusive (default: 3 — 교수님 기준, 4 이상 crowd 제외)",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy", "playlist", "none"),
        default="symlink",
        help=(
            "how to materialize clips:\n"
            "  symlink  — symlink each mp4 (default; falls back to copy if permission-denied)\n"
            "  copy     — physically copy every mp4 (use only when you need local files)\n"
            "  playlist — write .m3u8 playlists only; zero bytes of video duplicated (recommended on Windows / SMB)\n"
            "  none     — listings only, no playlists and no clip files"
        ),
    )
    parser.add_argument(
        "--allow-any-quality",
        action="store_true",
        help="skip the quality_ok filter (accept shots regardless of quality gate)",
    )
    parser.add_argument(
        "--min-people",
        type=int,
        default=1,
        help=(
            "minimum num_people required (YOLOv8 body detector). Default 1 filters out "
            "Haar face-detector false positives where no body was found."
        ),
    )
    parser.add_argument(
        "--min-face-ratio",
        type=float,
        default=0.08,
        help=(
            "minimum largest_face_ratio (face bbox / frame area). Default 0.08 "
            "(교수님 기준 — 화면의 8% 이상). Set to 0 to disable. Missing field is skipped."
        ),
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=["landscape", "wide"],
        help=(
            "shot_category values to skip (repeatable). Default: landscape + wide "
            "(keeps single / dominant / multi only). Pass --exclude-category '' to "
            "keep all categories."
        ),
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="minimum shot duration in seconds (교수님 기준: 1.5)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="maximum shot duration in seconds (교수님 기준: 6.0)",
    )
    parser.add_argument(
        "--max-camera-motion",
        type=float,
        default=6.0,
        help=(
            "maximum Farneback avg optical-flow magnitude (px/frame on 480-wide "
            "gray); above this the shot is dropped as 'camera_shake'. "
            "Shots whose manifest has no camera_motion field (old v2 manifests) "
            "pass through. Set to a very large number to disable."
        ),
    )
    parser.add_argument(
        "--no-default-guards",
        action="store_true",
        help=(
            "disable the new false-positive guards: reverts to old behaviour "
            "(min_people=0, min_face_ratio=0, no category exclusion)"
        ),
    )
    parser.add_argument(
        "--movie",
        action="append",
        default=None,
        help="restrict to a specific movie stem (may be repeated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing files / symlinks in --out-dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print the summary; do not write any files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "keep only the first N shots after optional shuffling. Useful for "
            "smoke-testing on the H100 before processing the full set."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="randomize shot order before applying --limit (uses --seed for reproducibility)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for --shuffle (default: 42)",
    )
    args = parser.parse_args(argv)

    output_root = _resolve_output_root(args.output_root)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else output_root / "filtered_for_prof"
    )

    if args.min_faces < 0 or args.max_faces < args.min_faces:
        parser.error("--min-faces must be >= 0 and <= --max-faces")

    if args.no_default_guards:
        min_people = 0
        min_face_ratio = 0.0
        exclude_categories: list[str] = []
        max_camera_motion: "float | None" = None
    else:
        min_people = max(0, args.min_people)
        min_face_ratio = max(0.0, args.min_face_ratio)
        exclude_categories = [c for c in args.exclude_category if c]
        max_camera_motion = args.max_camera_motion

    min_duration = max(0.0, args.min_duration)
    max_duration = max(min_duration, args.max_duration)

    print(f"[filter] output_root = {output_root}", file=sys.stderr)
    print(f"[filter] out_dir     = {out_dir}", file=sys.stderr)
    print(
        f"[filter] num_faces in [{args.min_faces}, {args.max_faces}]; "
        f"num_people >= {min_people}; largest_face_ratio >= {min_face_ratio}; "
        f"duration in [{min_duration}, {max_duration}] sec; "
        f"max_camera_motion = {max_camera_motion}; "
        f"exclude_category = {exclude_categories or '(none)'}; "
        f"quality_ok = {'any' if args.allow_any_quality else 'True only'}",
        file=sys.stderr,
    )

    records = collect_records(
        output_root=output_root,
        movies=args.movie,
        min_faces=args.min_faces,
        max_faces=args.max_faces,
        require_quality_ok=not args.allow_any_quality,
        min_people=min_people,
        min_face_ratio=min_face_ratio,
        exclude_categories=exclude_categories,
        min_duration=min_duration,
        max_duration=max_duration,
        max_camera_motion=max_camera_motion,
    )
    total_matched = len(records)
    print(f"[filter] matched shots: {total_matched}", file=sys.stderr)

    records, sampled = _sample_records(
        records, limit=args.limit, shuffle=args.shuffle, seed=args.seed
    )
    if sampled:
        print(
            f"[filter] returning {len(records)} shot(s) after "
            f"{'shuffle+' if args.shuffle else ''}limit",
            file=sys.stderr,
        )

    filter_meta = {
        "quality_ok_required": not args.allow_any_quality,
        "num_faces": [args.min_faces, args.max_faces],
        "min_people": min_people,
        "min_face_ratio": min_face_ratio,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "max_camera_motion": max_camera_motion,
        "exclude_categories": list(exclude_categories),
        "limit": args.limit,
        "shuffle": bool(args.shuffle),
        "seed": args.seed if args.shuffle else None,
        "movies": args.movie or None,
    }

    if args.dry_run:
        rows = [(r, None) for r in records]
        summary = summarize(rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    rows = materialize(records, out_dir, args.mode, overwrite=args.overwrite)
    jsonl_path, tsv_path = write_listings(rows, out_dir)
    manifest_path = write_manifest(
        rows,
        out_dir,
        output_root=output_root,
        filter_meta=filter_meta,
        total_matched=total_matched,
        sampled=sampled,
    )

    summary = summarize(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[filter] wrote {manifest_path}", file=sys.stderr)
    print(f"[filter] wrote {jsonl_path}", file=sys.stderr)
    print(f"[filter] wrote {tsv_path}", file=sys.stderr)
    print(f"[filter] wrote {out_dir / 'summary.json'}", file=sys.stderr)

    if args.mode == "playlist":
        playlists = write_playlists(rows, out_dir)
        print(
            f"[filter] wrote {len(playlists)} playlist(s); "
            f"open {playlists[0]} in VLC to browse everything.",
            file=sys.stderr,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
