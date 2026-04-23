"""FastAPI app for pod_control UI.

M1 scaffold + M3 Prepare page endpoints. Feature endpoints for Pods /
Run / Monitor land in M4–M6 per
docs/superpowers/specs/2026-04-22-pod-control-design.md.
"""
from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Iterator

import yaml

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from . import filter as pcfilter
from . import ssh as pcssh
from .runner import Runner, RunnerError
from .store import Batch, FilterParams, PodProfile, Store, StoreError


_CLIP_CHUNK = 1 << 20  # 1 MiB

# In-memory job registry for async batch exports. Keyed by job_id (uuid).
# Each value is a dict the GET /api/exports/{job_id} endpoint returns
# verbatim. The worker thread mutates it under _EXPORT_JOBS_LOCK.
import threading as _threading
_EXPORT_JOBS: dict[str, dict] = {}
_EXPORT_JOBS_LOCK = _threading.Lock()


def _export_worker(job_id: str, dest: Path, selected: list,
                   batch_obj, overwrite: bool, output_root: Path) -> None:
    """Background thread: copy clips + write manifest subset + batch.json."""
    import shutil
    from collections import defaultdict
    per_movie: dict[str, list] = defaultdict(list)
    for movie, entry in selected:
        per_movie[movie].append(entry)

    clips_root = dest / "clips"
    manifest_out = dest / "manifest"
    clips_root.mkdir(exist_ok=True)
    manifest_out.mkdir(exist_ok=True)

    copied = skipped = missing = 0
    errors: list[str] = []
    current = 0
    total = len(selected)

    for movie, entries in per_movie.items():
        mdir = clips_root / movie
        mdir.mkdir(exist_ok=True)
        with (manifest_out / f"{movie}.jsonl").open("w", encoding="utf-8") as mf:
            for e in entries:
                mf.write(e.model_dump_json() + "\n")
        for e in entries:
            current += 1
            with _EXPORT_JOBS_LOCK:
                _EXPORT_JOBS[job_id]["current"] = current
            src = (output_root / e.path).resolve() if not Path(e.path).is_absolute() \
                  else Path(e.path)
            if not src.is_file():
                missing += 1
                errors.append(f"missing source: {src}")
                continue
            dst = mdir / src.name
            if dst.is_file() and not overwrite:
                skipped += 1
                continue
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as ex:
                errors.append(f"{src.name}: {ex}")
            with _EXPORT_JOBS_LOCK:
                j = _EXPORT_JOBS[job_id]
                j["copied"] = copied
                j["skipped_existing"] = skipped
                j["missing_source"] = missing

    fp = batch_obj.filter_params
    (dest / "batch.json").write_text(
        batch_obj.model_dump_json(indent=2), encoding="utf-8"
    )
    (dest / "README.md").write_text(
        f"# Exported batch: {batch_obj.name}\n\n"
        f"Movies: {', '.join(batch_obj.movies)}\n\n"
        f"Filter:\n```json\n{fp.model_dump_json(indent=2)}\n```\n\n"
        f"Layout:\n"
        f"- `batch.json` — full batch definition\n"
        f"- `manifest/<movie>.jsonl` — only the shots in this batch\n"
        f"- `clips/<movie>/*.mp4` — the actual video clips\n\n"
        f"Stats: copied={copied} skipped_existing={skipped}"
        f" missing_source={missing}\n",
        encoding="utf-8",
    )
    with _EXPORT_JOBS_LOCK:
        j = _EXPORT_JOBS[job_id]
        j["status"] = "done"
        j["ended_at"] = time.time()
        j["copied"] = copied
        j["skipped_existing"] = skipped
        j["missing_source"] = missing
        j["errors"] = errors[:50]
        j["movies"] = list(per_movie.keys())


class ConfigPutBody(BaseModel):
    raw_yaml: str | None = None
    parsed: dict | None = None      # alternative: structured form sends dict


class ExportBatchBody(BaseModel):
    dest_path: str
    overwrite: bool = False


class SaveBatchRequest(BaseModel):
    name: str
    movies: list[str] = []
    movie: str | None = None   # back-compat: accepted, then folded into movies
    filter_params: FilterParams

    @field_validator("name")
    @classmethod
    def _slug_only(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"batch name must be alnum / - / _ only, got {v!r}"
            )
        return v

    def resolved_movies(self) -> list[str]:
        if self.movies:
            return self.movies
        if self.movie:
            return [self.movie]
        return []


class LaunchRunRequest(BaseModel):
    batch_name: str
    pod_name: str
    preset_path: str | None = None


class QuickLaunchRequest(BaseModel):
    movies: list[str] = []
    movie: str | None = None   # back-compat
    pod_name: str
    filter_params: FilterParams = FilterParams()
    preset_path: str | None = None

    def resolved_movies(self) -> list[str]:
        if self.movies:
            return self.movies
        if self.movie:
            return [self.movie]
        return []


class OutputRootRequest(BaseModel):
    path: str


class PodUpsertRequest(BaseModel):
    name: str
    host: str
    user: str
    ssh_key: str
    port: int = 22
    workspace: str

    @field_validator("name")
    @classmethod
    def _slug_only(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"pod name must be alnum / - / _ only, got {v!r}"
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
    cli_output_root_p = Path(output_root) if output_root else None
    store = Store(data_root_p)

    def _current_output_root() -> Path | None:
        # Persisted override beats CLI default.
        state = store.read_state()
        if state.current_output_root:
            return Path(state.current_output_root)
        return cli_output_root_p

    runner = Runner(store, output_root_provider=_current_output_root)

    def _scan_output_root_candidates() -> list[str]:
        """Scan parent of CLI default for siblings containing manifest/."""
        seen: set[str] = set()
        candidates: list[str] = []
        roots_to_scan: list[Path] = []
        if cli_output_root_p is not None:
            roots_to_scan.append(cli_output_root_p)
            roots_to_scan.append(cli_output_root_p.parent)
        # Also include the currently-active root if user already overrode.
        cur = _current_output_root()
        if cur is not None:
            roots_to_scan.append(cur)
            roots_to_scan.append(cur.parent)
        for parent in roots_to_scan:
            if not parent.exists():
                continue
            try:
                children = list(parent.iterdir())
            except PermissionError:
                continue
            # The path itself counts if it has manifest/.
            if (parent / "manifest").is_dir():
                p = str(parent.resolve())
                if p not in seen:
                    seen.add(p)
                    candidates.append(p)
            for child in children:
                if not child.is_dir():
                    continue
                if (child / "manifest").is_dir():
                    p = str(child.resolve())
                    if p not in seen:
                        seen.add(p)
                        candidates.append(p)
        return sorted(candidates)

    # ── health + index ---------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        cur = _current_output_root()
        return {
            "ok": True,
            "module": "pod_control",
            "version": "0.1.0",
            "data_root": str(data_root_p),
            "output_root": str(cur) if cur else None,
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    # ── Settings --------------------------------------------------------

    @app.get("/api/settings/output-root")
    def get_output_root() -> dict:
        cur = _current_output_root()
        return {
            "current": str(cur) if cur else None,
            "cli_default": (
                str(cli_output_root_p) if cli_output_root_p else None
            ),
            "candidates": _scan_output_root_candidates(),
        }

    @app.post("/api/settings/output-root")
    def set_output_root(req: OutputRootRequest) -> dict:
        p = Path(req.path).expanduser().resolve()
        if not p.is_dir():
            raise _err("invalid_path", f"{p} is not a directory", status=400)
        if not (p / "manifest").is_dir():
            raise _err(
                "invalid_path",
                f"{p} has no manifest/ subdir — not a pipeline output root",
                status=400,
            )
        with store.state_lock() as state:
            state.current_output_root = str(p)
        return {"current": str(p)}

    # ── M3 Prepare ------------------------------------------------------

    def _require_output_root() -> Path:
        cur = _current_output_root()
        if cur is None:
            raise _err(
                "output_root_not_configured",
                "output_root is not set (CLI default or runtime override)",
                status=500,
            )
        return cur

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
        movies: str | None = Query(
            None,
            description="comma-separated extra movies to aggregate with {movie}",
        ),
        skip_bad_quality: bool = True,
        skip_landscape: bool = True,
        max_shots: int | None = None,
        min_duration_sec: float | None = None,
        max_duration_sec: float | None = None,
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
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
        )
        # Aggregate path movie + optional extras (client can pass '' to just
        # exercise the path segment as a single-movie preview).
        all_movies = [movie] + (
            [m.strip() for m in movies.split(",") if m.strip()]
            if movies else []
        )
        # De-dupe while preserving order.
        seen: set[str] = set()
        unique_movies = [m for m in all_movies if not (m in seen or seen.add(m))]
        try:
            matched = pcfilter.filter_movies(out, unique_movies, params)
        except Exception as ex:
            raise _err("invalid_filter", str(ex))
        # 404 only when NO manifest exists for any requested movie.
        if not matched:
            any_exists = any(
                pcfilter.manifest_dir(out).joinpath(f"{m}.jsonl").exists()
                for m in unique_movies
            )
            if not any_exists:
                raise _err("movie_not_found",
                           f"no manifest for any of: {unique_movies}",
                           status=404)
        return pcfilter.paginate(
            matched,
            page=page,
            page_size=page_size,
            sample_seed=sample_seed,
        )

    @app.post("/api/batches", status_code=201)
    def create_batch(req: SaveBatchRequest) -> dict:
        out = _require_output_root()
        movies = req.resolved_movies()
        if not movies:
            raise _err("invalid_filter",
                       "movies must be non-empty (use movies: list or movie: str)")
        try:
            matched = pcfilter.filter_movies(out, movies, req.filter_params)
        except Exception as ex:
            raise _err("invalid_filter", str(ex))
        batch = Batch(
            name=req.name,
            movies=movies,
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
        # Poll so any exited run flips its batch out of "running" before
        # the UI reads the list.
        runner.poll_active()
        return {"batches": [b.model_dump() for b in store.list_batches()]}

    @app.get("/api/batches/{name}")
    def get_batch(name: str) -> dict:
        b = store.get_batch(name)
        if b is None:
            raise _err("batch_not_found", f"batch {name!r} not found", status=404)
        return b.model_dump()

    @app.delete("/api/batches/{name}", status_code=204)
    def delete_batch(name: str) -> Response:
        # Refresh state: a still-alive process will keep the lock; a
        # dead one will be finalized here and delete will succeed.
        runner.poll_active()
        try:
            store.delete_batch(name)
        except StoreError as ex:
            code = "batch_not_found" if "not found" in str(ex) else "batch_not_ready"
            status = 404 if code == "batch_not_found" else 409
            raise _err(code, str(ex), status=status)
        return Response(status_code=204)

    @app.post("/api/batches/{name}/reset")
    def reset_batch(name: str) -> dict:
        """Flip a stuck batch (running/failed/done) back to ready so the
        user can re-launch it. Refuses if it's the active run."""
        runner.poll_active()
        b = store.get_batch(name)
        if b is None:
            raise _err("batch_not_found", f"batch {name!r} not found",
                       status=404)
        active = store.read_state().active_run
        if active and active.batch_name == name:
            raise _err("batch_in_use",
                       "batch is currently running — kill the run first",
                       status=409)
        b.status = "ready"
        store.save_batch(b, overwrite=True)
        return b.model_dump()

    @app.post("/api/batches/{name}/export", status_code=202)
    def export_batch(name: str, body: ExportBatchBody) -> dict:
        """Kick off an async export. Returns immediately with job_id;
        client polls GET /api/exports/{job_id} for progress.
        """
        b = store.get_batch(name)
        if b is None:
            raise _err("batch_not_found", f"batch {name!r} not found",
                       status=404)
        out = _current_output_root()
        if out is None:
            raise _err("no_output_root",
                       "no output_root configured", status=400)

        dest = Path(body.dest_path).expanduser()
        if not dest.is_absolute():
            raise _err("invalid_dest",
                       "dest_path must be absolute", status=400)
        try:
            dest_resolved = dest.resolve()
        except Exception as ex:
            raise _err("invalid_dest", f"path resolve failed: {ex}",
                       status=400)
        forbidden_roots = {
            "/", "/etc", "/bin", "/sbin", "/usr", "/var", "/boot",
            "/lib", "/lib64", "/proc", "/sys", "/dev", "/root", "/home",
        }
        if str(dest_resolved) in forbidden_roots:
            raise _err("invalid_dest",
                       f"dest_path may not be a system root: {dest_resolved}",
                       status=400)
        try:
            out_resolved = out.resolve()
            if dest_resolved == out_resolved:
                raise _err("invalid_dest",
                           f"dest_path equals current output_root ({out_resolved})",
                           status=400)
            for sub in ("clips", "labels", "manifest"):
                if dest_resolved == out_resolved / sub:
                    raise _err("invalid_dest",
                               f"dest_path equals output_root/{sub}; "
                               "would overwrite source data",
                               status=400)
        except FileNotFoundError:
            pass

        # Run filtering synchronously (cheap; gives total upfront).
        from src.runpod.upload import _iter_manifest_lines, _filter_entries
        manifest_dir = out / "manifest"
        if not manifest_dir.is_dir():
            raise _err("manifest_missing",
                       f"no manifest dir at {manifest_dir}", status=400)
        all_entries = list(_iter_manifest_lines(
            str(manifest_dir), movies_filter=set(b.movies)
        ))
        fp = b.filter_params
        selected = _filter_entries(
            all_entries,
            categories=fp.categories, max_shots=fp.max_shots,
            skip_bad_quality=fp.skip_bad_quality,
            skip_landscape=fp.skip_landscape,
            min_duration_sec=fp.min_duration_sec,
            max_duration_sec=fp.max_duration_sec,
        )
        dest_resolved.mkdir(parents=True, exist_ok=True)

        import uuid
        job_id = uuid.uuid4().hex
        with _EXPORT_JOBS_LOCK:
            _EXPORT_JOBS[job_id] = {
                "id": job_id, "batch_name": b.name,
                "dest": str(dest_resolved),
                "status": "running", "started_at": time.time(),
                "ended_at": None,
                "total": len(selected), "current": 0,
                "copied": 0, "skipped_existing": 0, "missing_source": 0,
                "movies": [], "errors": [],
            }
        # Cap registry size: drop oldest finished jobs once > 20.
        with _EXPORT_JOBS_LOCK:
            done_ids = sorted(
                (j["id"] for j in _EXPORT_JOBS.values() if j["status"] != "running"),
                key=lambda jid: _EXPORT_JOBS[jid].get("ended_at") or 0,
            )
            for jid in done_ids[:-20]:
                _EXPORT_JOBS.pop(jid, None)

        _threading.Thread(
            target=_export_worker,
            args=(job_id, dest_resolved, selected, b, body.overwrite, out),
            daemon=True,
        ).start()
        return {"job_id": job_id, "total": len(selected),
                "dest": str(dest_resolved)}

    @app.get("/api/exports/{job_id}")
    def get_export(job_id: str) -> dict:
        with _EXPORT_JOBS_LOCK:
            job = _EXPORT_JOBS.get(job_id)
            if job is None:
                raise _err("export_not_found",
                           f"no export job {job_id!r}", status=404)
            return dict(job)   # shallow copy

    # ── M4 Pods ---------------------------------------------------------

    @app.get("/api/pods")
    def list_pods() -> dict:
        return {"pods": [p.model_dump() for p in store.list_pods()]}

    @app.get("/api/pods/{name}")
    def get_pod(name: str) -> dict:
        p = store.get_pod(name)
        if p is None:
            raise _err("pod_not_found", f"pod {name!r} not found", status=404)
        return p.model_dump()

    @app.post("/api/pods", status_code=201)
    def create_pod(req: PodUpsertRequest) -> dict:
        if store.get_pod(req.name) is not None:
            raise _err("pod_exists", f"pod {req.name!r} already exists",
                       status=409)
        pod = PodProfile(**req.model_dump())
        store.upsert_pod(pod)
        return pod.model_dump()

    @app.put("/api/pods/{name}")
    def update_pod(name: str, req: PodUpsertRequest) -> dict:
        if req.name != name:
            raise _err("name_mismatch",
                       f"URL name {name!r} != body name {req.name!r}")
        existing = store.get_pod(name)
        if existing is None:
            raise _err("pod_not_found", f"pod {name!r} not found", status=404)
        pod = PodProfile(
            **req.model_dump(),
            last_test_ok=existing.last_test_ok,
            last_test_at=existing.last_test_at,
        )
        store.upsert_pod(pod)
        return pod.model_dump()

    @app.delete("/api/pods/{name}", status_code=204)
    def delete_pod(name: str) -> Response:
        try:
            store.delete_pod(name)
        except StoreError as ex:
            raise _err("pod_not_found", str(ex), status=404)
        return Response(status_code=204)

    @app.post("/api/pods/{name}/test")
    def test_pod(name: str) -> dict:
        pod = store.get_pod(name)
        if pod is None:
            raise _err("pod_not_found", f"pod {name!r} not found", status=404)
        result = pcssh.test_connect(pod)
        # Persist last-test stamp.
        updated = pod.model_copy(update={
            "last_test_ok": result.ok,
            "last_test_at": time.time(),
        })
        store.upsert_pod(updated)
        return {
            "ok": result.ok,
            "latency_ms": result.latency_ms,
            "message": result.message,
        }

    # ── M5 Runs ---------------------------------------------------------

    @app.post("/api/runs", status_code=201)
    def launch_run(req: LaunchRunRequest) -> dict:
        batch = store.get_batch(req.batch_name)
        if batch is None:
            raise _err("batch_not_found",
                       f"batch {req.batch_name!r} not found", status=404)
        if batch.status != "ready":
            raise _err("batch_not_ready",
                       f"batch is {batch.status}; only ready batches launch",
                       status=409)
        pod = store.get_pod(req.pod_name)
        if pod is None:
            raise _err("pod_not_found",
                       f"pod {req.pod_name!r} not found", status=404)
        try:
            rec = runner.launch(batch, pod, preset_path=req.preset_path)
        except RunnerError as ex:
            code = ("run_already_active"
                    if "run_already_active" in str(ex)
                    else "runner_error")
            raise _err(code, str(ex), status=409 if code == "run_already_active" else 500)
        # Flip batch to running (atomic overwrite).
        batch.status = "running"
        batch.last_run_id = rec.id
        store.save_batch(batch, overwrite=True)
        return rec.model_dump()

    @app.post("/api/runs/quick", status_code=201)
    def quick_launch(req: QuickLaunchRequest) -> dict:
        """Ad-hoc launch: auto-create a throwaway batch, then launch."""
        out = _require_output_root()
        movies = req.resolved_movies()
        if not movies:
            raise _err("invalid_filter",
                       "movies must be non-empty (use movies: list or movie: str)")
        try:
            matched = pcfilter.filter_movies(out, movies, req.filter_params)
        except Exception as ex:
            raise _err("invalid_filter", str(ex))
        # If every movie manifest is missing, bail with a useful error.
        if not matched:
            existing = [
                m for m in movies
                if pcfilter.manifest_dir(out).joinpath(f"{m}.jsonl").exists()
            ]
            if not existing:
                raise _err(
                    "movie_not_found",
                    f"no manifest for any of: {movies}",
                    status=404,
                )

        pod = store.get_pod(req.pod_name)
        if pod is None:
            raise _err("pod_not_found",
                       f"pod {req.pod_name!r} not found", status=404)

        # Auto-name: single movie → slug + timestamp; multi → count + timestamp.
        if len(movies) == 1:
            slug = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in movies[0]
            )[:40] or "batch"
        else:
            slug = f"multi_{len(movies)}movies"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        name = f"{slug}_{stamp}"

        batch = Batch(
            name=name,
            movies=movies,
            filter_params=req.filter_params,
            shot_count=len(matched),
        )
        try:
            store.save_batch(batch)
        except StoreError as ex:
            raise _err("batch_exists", str(ex), status=409)

        try:
            rec = runner.launch(batch, pod, preset_path=req.preset_path)
        except RunnerError as ex:
            # Roll back the auto-created batch so the user isn't left
            # with a phantom "ready" batch pointing at nothing.
            try:
                store.delete_batch(name)
            except Exception:
                pass
            code = ("run_already_active"
                    if "run_already_active" in str(ex)
                    else "runner_error")
            raise _err(code, str(ex),
                       status=409 if code == "run_already_active" else 500)
        batch.status = "running"
        batch.last_run_id = rec.id
        store.save_batch(batch, overwrite=True)
        return rec.model_dump()

    @app.get("/api/runs/active")
    def active_run() -> dict:
        # Poll first — finalize any run whose subprocess has exited.
        runner.poll_active()
        state = store.read_state()
        return {
            "active_run": state.active_run.model_dump()
            if state.active_run else None,
        }

    @app.get("/api/runs")
    def list_runs() -> dict:
        state = store.read_state()
        active = [state.active_run.model_dump()] if state.active_run else []
        return {
            "active": active,
            "history": [r.model_dump() for r in state.history],
        }

    @app.get("/api/runs/{run_id}/tail")
    def tail_run(run_id: str, offset: int = 0) -> dict:
        state = store.read_state()
        active = state.active_run
        if active is None or active.id != run_id:
            # Try history as a fallback (finished runs get a final tail).
            match = next((r for r in state.history if r.id == run_id), None)
            if match is None:
                raise _err("run_not_found",
                           f"no run with id {run_id!r}", status=404)
            # Finished: return empty incremental text; client can still
            # pull full local stdout log from /runs/<id>/stdout.log if
            # they want. For P0 we just say "no new text".
            return {
                "text": "",
                "next_offset": offset,
                "pod_unreachable": False,
                "checkpoint": {"done": 0, "failed": 0, "pending": 0},
                "finished": True,
                "status": match.status,
            }
        pod = store.get_pod(active.pod_name)
        if pod is None:
            raise _err("pod_not_found",
                       f"pod {active.pod_name!r} gone from store",
                       status=404)
        log_path = f"{pod.workspace}/output/pod_runner.log"
        tail = pcssh.tail_remote_log(
            pod, remote_path=log_path, offset=offset,
        )

        # Parse checkpoint (line count per status). We pull a single
        # tail of .checkpoint.jsonl; cheap vs. the log poll cadence.
        checkpoint = _parse_checkpoint(pod) if not tail.pod_unreachable else {
            "done": 0, "failed": 0, "pending": 0,
        }

        return {
            "text": tail.text,
            "next_offset": tail.next_offset,
            "pod_unreachable": tail.pod_unreachable,
            "checkpoint": checkpoint,
            "finished": False,
            "status": "running",
        }

    def _parse_checkpoint(pod: PodProfile) -> dict:
        """Ask the pod for the current checkpoint line count.

        The pod's pod_runner writes one JSON line per completed shot to
        <workspace>/output/.checkpoint.jsonl. We count lines and group by
        the status field if present. Cheap SSH command, no file transfer.
        """
        cmd = (
            f"wc -l {pod.workspace}/output/.checkpoint.jsonl 2>/dev/null "
            f"|| echo 0"
        )
        import shlex
        full = pcssh.build_ssh_args(pod) + [cmd]
        import subprocess as _sp
        try:
            r = _sp.run(full, capture_output=True, timeout=10)
            if r.returncode != 0:
                return {"done": 0, "failed": 0, "pending": 0}
            first = (r.stdout.decode(errors="replace").strip().split() or ["0"])[0]
            done = int(first)
        except Exception:
            done = 0
        return {"done": done, "failed": 0, "pending": 0}

    @app.post("/api/runs/{run_id}/kill")
    def kill_run(run_id: str) -> dict:
        state = store.read_state()
        if state.active_run is None or state.active_run.id != run_id:
            raise _err("run_not_found",
                       f"no active run with id {run_id!r}", status=404)
        try:
            killed = runner.kill_active()
        except RunnerError as ex:
            raise _err("runner_error", str(ex), status=500)
        # Flip batch back to failed so user can re-queue a fresh one.
        batch = store.get_batch(killed.batch_name)
        if batch is not None:
            batch.status = "failed"
            store.save_batch(batch, overwrite=True)
        return killed.model_dump()

    @app.delete("/api/runs")
    def clear_run_history() -> dict:
        with store.state_lock() as state:
            state.history = []
        return {"ok": True}

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str) -> Response:
        """Remove a single history entry. Active run is never touched."""
        with store.state_lock() as state:
            before = len(state.history)
            state.history = [r for r in state.history if r.id != run_id]
            if len(state.history) == before:
                raise _err("run_not_found",
                           f"run {run_id!r} not in history", status=404)
        return Response(status_code=204)

    # ── Config presets (configs/runpod*.yaml) -------------------------

    _CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

    def _safe_config_name(name: str) -> str:
        """Reject path-traversal and `.example`; require runpod*.yaml."""
        if "/" in name or "\\" in name or ".." in name:
            raise _err("invalid_name",
                       "config name must be a bare filename", status=400)
        if name.endswith(".example"):
            raise _err("invalid_name",
                       "cannot modify .example files", status=400)
        if not (name.startswith("runpod") and name.endswith(".yaml")):
            raise _err("invalid_name",
                       "config must match runpod*.yaml", status=400)
        return name

    def _config_meta(p: Path) -> dict:
        """Extract preset metadata for the dropdown."""
        try:
            cfg = yaml.safe_load(p.read_text("utf-8")) or {}
        except Exception as ex:
            return {
                "name": p.name, "path": str(p),
                "model": None, "max_model_len": None, "rounds": None,
                "error": f"yaml parse: {ex}",
            }
        model = (cfg.get("model") or {}).get("name")
        sampling = cfg.get("sampling") or {}
        max_len = (sampling.get("max_model_len")
                   or (cfg.get("model") or {}).get("max_model_len"))
        rounds = ((cfg.get("pipeline") or {}).get("rounds")
                  or cfg.get("rounds"))
        return {
            "name": p.name, "path": str(p),
            "model": model, "max_model_len": max_len, "rounds": rounds,
        }

    @app.get("/api/configs")
    def list_configs() -> dict:
        if not _CONFIGS_DIR.is_dir():
            return {"configs": []}
        items = []
        for p in sorted(_CONFIGS_DIR.glob("runpod*.yaml")):
            if p.name.endswith(".example"):
                continue
            items.append(_config_meta(p))
        return {"configs": items}

    @app.get("/api/configs/{name}")
    def get_config(name: str) -> dict:
        _safe_config_name(name)
        p = _CONFIGS_DIR / name
        if not p.is_file():
            raise _err("config_not_found", f"no config {name!r}", status=404)
        raw = p.read_text("utf-8")
        try:
            parsed = yaml.safe_load(raw) or {}
        except Exception:
            parsed = None
        return {
            "name": name, "raw_yaml": raw,
            "parsed": parsed, "meta": _config_meta(p),
        }

    @app.put("/api/configs/{name}")
    def put_config(name: str, body: ConfigPutBody) -> dict:
        """Accept either raw_yaml (power users) or parsed dict (form UI).
        If both are provided, parsed wins."""
        _safe_config_name(name)
        if body.parsed is not None:
            parsed = body.parsed
            if not isinstance(parsed, dict):
                raise _err("yaml_invalid",
                           "parsed body must be a mapping", status=400)
            raw_to_write = yaml.safe_dump(
                parsed, sort_keys=False, allow_unicode=True, default_flow_style=False,
            )
        elif body.raw_yaml is not None:
            try:
                parsed = yaml.safe_load(body.raw_yaml)
            except yaml.YAMLError as ex:
                raise _err("yaml_invalid", f"YAML parse error: {ex}",
                           status=400)
            if not isinstance(parsed, dict):
                raise _err("yaml_invalid",
                           "top-level must be a mapping", status=400)
            raw_to_write = body.raw_yaml
        else:
            raise _err("yaml_invalid",
                       "request body must include raw_yaml or parsed",
                       status=400)
        for required in ("pod", "paths", "model"):
            if required not in parsed:
                raise _err("yaml_invalid",
                           f"missing required section: {required}",
                           status=400)
        p = _CONFIGS_DIR / name
        from .store import _atomic_write_text
        _atomic_write_text(p, raw_to_write)
        return {"ok": True, "name": name, "meta": _config_meta(p)}

    @app.get("/api/runs/{run_id}/local-tail")
    def local_tail(run_id: str, offset: int = 0) -> dict:
        """Return incremental content from the local stdout.log for this run.

        Only declares `finished=true` when the run is in history AND we've
        caught up to EOF. Frontend can keep polling pod-direct endpoints
        for detached runs even after local is finished.
        """
        run_dir = store.run_dir(run_id)
        log_path = run_dir / "stdout.log"
        if not log_path.exists():
            return {"text": "", "next_offset": 0, "finished": False}
        data = log_path.read_bytes()
        chunk = data[offset:]
        next_offset = offset + len(chunk)
        state = store.read_state()
        run_active = state.active_run is not None and state.active_run.id == run_id
        in_history = any(r.id == run_id for r in state.history)
        caught_up = next_offset == len(data)
        finished = (not run_active) and in_history and caught_up
        return {
            "text": chunk.decode("utf-8", errors="replace"),
            "next_offset": next_offset,
            "finished": finished,
        }

    @app.get("/api/pods/{pod_name}/log-tail")
    def pod_log_tail(pod_name: str, offset: int = 0,
                     path: str = "output/pod_runner.log") -> dict:
        """Direct SSH tail of a log on the pod (works without active_run).

        Lets the Monitor keep tracking a detached run (local bash died,
        but pod_runner is still processing). `path` is relative to
        workspace; leading slash or `..` rejected.
        """
        pod = store.get_pod(pod_name)
        if pod is None:
            raise _err("pod_not_found", f"no pod {pod_name!r}", status=404)
        if path.startswith("/") or ".." in path.split("/"):
            raise _err("invalid_path", "path must be relative to workspace",
                       status=400)
        remote = f"{pod.workspace}/{path}"
        tail = pcssh.tail_remote_log(pod, remote_path=remote, offset=offset)
        return {
            "text": tail.text,
            "next_offset": tail.next_offset,
            "pod_unreachable": tail.pod_unreachable,
        }

    @app.get("/api/pods/{pod_name}/checkpoint")
    def pod_checkpoint(pod_name: str) -> dict:
        """Direct SSH checkpoint count for a pod (independent of active_run)."""
        pod = store.get_pod(pod_name)
        if pod is None:
            raise _err("pod_not_found", f"no pod {pod_name!r}", status=404)
        return _parse_checkpoint(pod)

    @app.post("/api/runs/{run_id}/pull")
    def trigger_pull(run_id: str) -> dict:
        """Fire-and-forget: re-run 03_pull.sh for a finished (or active) run.

        Output is appended to stdout.log so the local-tail endpoint picks it
        up in the next poll cycle.
        """
        run_dir = store.run_dir(run_id)
        config_path = run_dir / "runpod.yaml"
        if not config_path.exists():
            raise _err("config_not_found",
                       f"no runpod.yaml for run {run_id!r}", status=404)
        pull_script = Path(__file__).resolve().parent.parent.parent / \
            "scripts" / "runpod" / "03_pull.sh"
        if not pull_script.exists():
            raise _err("script_not_found",
                       f"03_pull.sh not found at {pull_script}", status=500)
        import subprocess as _sp
        stdout_log = run_dir / "stdout.log"
        fh = stdout_log.open("ab")
        header = f"\n[pod_control:stage=pull]\n══ manual re-pull ══\n".encode()
        fh.write(header)
        fh.flush()
        _sp.Popen(
            ["bash", str(pull_script), str(config_path)],
            stdout=fh,
            stderr=_sp.STDOUT,
            close_fds=True,
        )
        return {"ok": True, "run_id": run_id}

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
