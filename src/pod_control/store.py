"""Filesystem-backed persistence for pod_control.

Only this module is allowed to write to disk (per design doc § 4). Writes
are atomic (*.tmp + os.replace) and state.json is fcntl-locked during
run-lifecycle transitions so the single-run-slot invariant holds even if
someone fires two API requests at once.

Layout under data_root:
    pods.yaml                  list[PodProfile]
    batches/<name>.json        Batch (one file per batch)
    state.json                 ActiveState (active_run + history)
    runs/<run_id>/             per-run subdir (stdout.log, pod_tail.log)

Timestamps: unix epoch float (time.time()).
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Pydantic models (shared with api.py) ──

BatchStatus = Literal["ready", "running", "done", "failed"]
RunStatus = Literal["running", "done", "failed", "killed"]


class FilterParams(BaseModel):
    categories: list[str] = Field(
        default_factory=lambda: ["single", "dominant", "multi"]
    )
    skip_bad_quality: bool = True
    skip_landscape: bool = True
    max_shots: int | None = None
    min_duration_sec: float | None = None
    max_duration_sec: float | None = None


class Batch(BaseModel):
    name: str
    movies: list[str] = Field(default_factory=list)
    filter_params: FilterParams
    shot_count: int = 0
    status: BatchStatus = "ready"
    created_at: float = Field(default_factory=time.time)
    last_run_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_movie_to_movies(cls, data):
        """Backward compat: older batch files stored {"movie": "X"}.

        Convert singular `movie` → `movies=[movie]` on load so existing
        files keep working after the multi-movie migration.
        """
        if not isinstance(data, dict):
            return data
        if "movies" in data:
            return data
        m = data.get("movie")
        if m is not None and not data.get("movies"):
            data = {**data, "movies": [m] if isinstance(m, str) else list(m)}
            data.pop("movie", None)
        return data

    @field_validator("name")
    @classmethod
    def _name_is_slug(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"batch name must be alnum / - / _ only, got {v!r}"
            )
        return v

    @field_validator("movies")
    @classmethod
    def _movies_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("batch.movies must have at least one entry")
        for m in v:
            if not m or not isinstance(m, str):
                raise ValueError(f"invalid movie name {m!r}")
        return v

    @property
    def movie(self) -> str:
        """Back-compat alias — returns first movie or raises."""
        return self.movies[0]


class PodProfile(BaseModel):
    name: str
    host: str
    user: str
    ssh_key: str
    port: int = 22
    workspace: str
    last_test_ok: bool | None = None
    last_test_at: float | None = None

    @field_validator("name")
    @classmethod
    def _name_is_slug(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"pod name must be alnum / - / _ only, got {v!r}"
            )
        return v


class RunRecord(BaseModel):
    id: str
    batch_name: str
    pod_name: str
    preset_path: str | None = None
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    status: RunStatus = "running"
    pid: int | None = None
    pod_log_offset: int = 0
    exit_code: int | None = None


class ActiveState(BaseModel):
    active_run: RunRecord | None = None
    history: list[RunRecord] = Field(default_factory=list)
    current_output_root: str | None = None


# ── Store error + helpers ──


class StoreError(Exception):
    """Persistence failure surfaces here; api.py converts to HTTPException."""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def _locked(path: Path) -> Iterator[int]:
    """Advisory fcntl write-lock on path (created if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ── Store ──


class Store:
    """Persistence facade. Construct once per app instance."""

    def __init__(self, data_root: str | os.PathLike[str]) -> None:
        self.root = Path(data_root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "batches").mkdir(exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)

    # Paths -------------------------------------------------------------

    @property
    def pods_file(self) -> Path:
        return self.root / "pods.yaml"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    def batch_file(self, name: str) -> Path:
        return self.root / "batches" / f"{name}.json"

    def run_dir(self, run_id: str) -> Path:
        d = self.root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Batches -----------------------------------------------------------

    def list_batches(self) -> list[Batch]:
        out: list[Batch] = []
        for p in sorted((self.root / "batches").glob("*.json")):
            try:
                out.append(Batch.model_validate(json.loads(p.read_text("utf-8"))))
            except Exception as ex:
                raise StoreError(f"corrupt batch file {p.name}: {ex}") from ex
        return out

    def get_batch(self, name: str) -> Batch | None:
        p = self.batch_file(name)
        if not p.exists():
            return None
        return Batch.model_validate(json.loads(p.read_text("utf-8")))

    def save_batch(self, batch: Batch, *, overwrite: bool = False) -> Batch:
        p = self.batch_file(batch.name)
        if p.exists() and not overwrite:
            raise StoreError(f"batch {batch.name!r} already exists")
        _atomic_write_text(p, batch.model_dump_json(indent=2))
        return batch

    def delete_batch(self, name: str) -> None:
        p = self.batch_file(name)
        if not p.exists():
            raise StoreError(f"batch {name!r} not found")
        s = self.read_state()
        if s.active_run is not None and s.active_run.batch_name == name:
            raise StoreError(f"batch {name!r} is running; cannot delete")
        p.unlink()

    # Pods --------------------------------------------------------------

    def list_pods(self) -> list[PodProfile]:
        if not self.pods_file.exists():
            return []
        raw = yaml.safe_load(self.pods_file.read_text("utf-8")) or []
        if not isinstance(raw, list):
            raise StoreError(
                f"{self.pods_file} must contain a list, got {type(raw)}"
            )
        return [PodProfile.model_validate(item) for item in raw]

    def get_pod(self, name: str) -> PodProfile | None:
        for p in self.list_pods():
            if p.name == name:
                return p
        return None

    def upsert_pod(self, pod: PodProfile) -> PodProfile:
        pods = {p.name: p for p in self.list_pods()}
        pods[pod.name] = pod
        self._write_pods(sorted(pods.values(), key=lambda p: p.name))
        return pod

    def delete_pod(self, name: str) -> None:
        before = self.list_pods()
        after = [p for p in before if p.name != name]
        if len(after) == len(before):
            raise StoreError(f"pod {name!r} not found")
        self._write_pods(after)

    def _write_pods(self, pods: list[PodProfile]) -> None:
        payload = [p.model_dump() for p in pods]
        _atomic_write_text(
            self.pods_file,
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        )

    # State -------------------------------------------------------------

    def read_state(self) -> ActiveState:
        if not self.state_file.exists():
            return ActiveState()
        return ActiveState.model_validate(
            json.loads(self.state_file.read_text("utf-8"))
        )

    @contextmanager
    def state_lock(self) -> Iterator[ActiveState]:
        """Read → mutate → write, under fcntl lock.

        Usage:
            with store.state_lock() as state:
                state.active_run = RunRecord(...)
        """
        lock_path = self.root / ".state.lock"
        with _locked(lock_path):
            state = self.read_state()
            yield state
            _atomic_write_text(self.state_file, state.model_dump_json(indent=2))
