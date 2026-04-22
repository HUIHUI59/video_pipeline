"""Local subprocess orchestration for pod runs.

Spawns scripts/runpod/run_all.sh in its own process group (setsid) so a
Kill can nuke the whole tree (python + ssh + ffmpeg children). Writes PID
and lifecycle state via store.state_lock() so the single-run-slot
invariant holds across API requests.

The actual upload / inference / download still happens inside the shell
script; this module only owns "one run active" bookkeeping + builds a
per-run YAML config file that bakes in (a) the chosen pod's SSH info,
(b) the user's selected output_root paths, and (c) the batch's filter
params, then passes that config path as the single positional arg to
run_all.sh.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

import yaml

from .store import Batch, PodProfile, RunRecord, Store


class RunnerError(Exception):
    """API layer converts these to HTTPException."""


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RUN_ALL = _REPO_ROOT / "scripts" / "runpod" / "run_all.sh"
_KILL_SH = _REPO_ROOT / "scripts" / "runpod" / "99_kill.sh"


def _new_run_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _build_run_all_cmd(
    *,
    run_all_script: Path,
    config_path: Path,
    batch: Batch,
) -> list[str]:
    """Build the run_all.sh invocation.

    Shape: bash run_all.sh <config_path> [extra CLI flags forwarded to upload.py]

    run_all.sh expects $1 to be the config YAML file path (not a flag!).
    The YAML already encodes pod / paths / filters / model. The only
    upload.py knobs NOT expressible in YAML are the "include" toggles
    and duration bounds — those get forwarded as extra args, which
    run_all.sh passes through "${@:2}" to 01_push.sh.
    """
    cmd: list[str] = ["bash", str(run_all_script), str(config_path)]
    fp = batch.filter_params
    if not fp.skip_bad_quality:
        cmd += ["--include-bad-quality"]
    if not fp.skip_landscape:
        cmd += ["--include-landscape"]
    if fp.min_duration_sec is not None:
        cmd += ["--min-duration", str(fp.min_duration_sec)]
    if fp.max_duration_sec is not None:
        cmd += ["--max-duration", str(fp.max_duration_sec)]
    return cmd


def _write_merged_config(
    *,
    preset_path: Path | None,
    pod: PodProfile,
    batch: Batch,
    output_root: Path,
    target_path: Path,
) -> Path:
    """Build the per-run YAML config and write it to target_path.

    Merge order:
      1) base = preset_path (if given) else configs/runpod.yaml at repo root
      2) override pod: section with this run's PodProfile
      3) override paths: with the selected output_root
      4) override filters: with the batch's movies + FilterParams
    """
    # 1) load base preset
    if preset_path and preset_path.is_file():
        cfg = yaml.safe_load(preset_path.read_text("utf-8")) or {}
    else:
        default = _REPO_ROOT / "configs" / "runpod.yaml"
        if not default.is_file():
            # Fall back to the .example so the user at least gets a
            # skeleton with model / sampling defaults.
            default = _REPO_ROOT / "configs" / "runpod.yaml.example"
        cfg = (
            yaml.safe_load(default.read_text("utf-8")) or {}
            if default.is_file() else {}
        )

    # 2) pod SSH info from PodProfile
    cfg["pod"] = {
        "host": pod.host,
        "port": pod.port,
        "user": pod.user,
        "ssh_key": pod.ssh_key,
    }

    # 3) paths grounded in the web-UI's output_root
    paths = dict(cfg.get("paths") or {})
    paths["local_clips_root"]   = str(output_root / "clips")
    paths["local_labels_root"]  = str(output_root / "labels")
    paths["local_manifest_dir"] = str(output_root / "manifest")
    paths["pod_workspace"]      = pod.workspace
    cfg["paths"] = paths

    # 4) filters from the batch
    fp = batch.filter_params
    cfg["filters"] = {
        "shot_categories": list(fp.categories or []),
        "movies": list(batch.movies),
        "max_shots": fp.max_shots,
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target_path


class Runner:
    """Lifecycle owner for the single active run slot."""

    def __init__(
        self,
        store: Store,
        *,
        run_all_script: Path = _RUN_ALL,
        kill_script: Path = _KILL_SH,
        output_root_provider: Callable[[], Path | None] = lambda: None,
    ) -> None:
        self.store = store
        self.run_all_script = run_all_script
        self.kill_script = kill_script
        self._output_root_provider = output_root_provider
        self._popen = subprocess.Popen
        self._active_popen = None
        self._active_stdout_fh = None

    # ── Launch ──────────────────────────────────────────────────────

    def launch(
        self,
        batch: Batch,
        pod: PodProfile,
        *,
        preset_path: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> RunRecord:
        if not self.run_all_script.is_file():
            raise RunnerError(
                f"run_all script missing: {self.run_all_script}"
            )
        output_root = self._output_root_provider()
        if output_root is None:
            raise RunnerError(
                "output_root not configured — set one in Prepare → Output root"
            )
        run_id = _new_run_id()
        run_dir = self.store.run_dir(run_id)
        stdout_path = run_dir / "stdout.log"

        # Build the per-run config that 01_push.sh / 02_run.sh will read.
        config_path = _write_merged_config(
            preset_path=Path(preset_path) if preset_path else None,
            pod=pod,
            batch=batch,
            output_root=Path(output_root),
            target_path=run_dir / "runpod.yaml",
        )
        cmd = _build_run_all_cmd(
            run_all_script=self.run_all_script,
            config_path=config_path,
            batch=batch,
        )

        env = os.environ.copy()
        # Kept for any future scripts that read SSH info from env. The
        # primary source is now the generated YAML, but env doesn't
        # cost anything.
        env["RUNPOD_SSH_HOST"] = pod.host
        env["RUNPOD_SSH_USER"] = pod.user
        env["RUNPOD_SSH_KEY"] = os.path.expanduser(pod.ssh_key)
        env["RUNPOD_SSH_PORT"] = str(pod.port)
        env["RUNPOD_WORKSPACE"] = pod.workspace
        if extra_env:
            env.update(extra_env)

        with self.store.state_lock() as state:
            if state.active_run is not None:
                raise RunnerError(
                    f"run_already_active: {state.active_run.id}"
                )

            stdout_fh = stdout_path.open("wb")
            try:
                proc = self._popen(
                    cmd,
                    stdout=stdout_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid,
                    close_fds=True,
                )
            except Exception as ex:
                stdout_fh.close()
                raise RunnerError(f"spawn failed: {ex}") from ex

            record = RunRecord(
                id=run_id,
                batch_name=batch.name,
                pod_name=pod.name,
                preset_path=preset_path,
                pid=proc.pid,
            )
            state.active_run = record

        self._active_popen = proc
        self._active_stdout_fh = stdout_fh
        return record

    # ── Poll + finalize ─────────────────────────────────────────────

    def poll_active(self) -> RunRecord | None:
        """Check if the active run has exited and finalize state if so."""
        state = self.store.read_state()
        if state.active_run is None:
            return None
        proc = self._active_popen
        if proc is None:
            return state.active_run
        rc = proc.poll()
        if rc is None:
            return state.active_run
        self._finalize(rc)
        return self.store.read_state().active_run

    def _finalize(self, returncode: int | None) -> None:
        fh = self._active_stdout_fh
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._active_stdout_fh = None
        self._active_popen = None
        with self.store.state_lock() as state:
            if state.active_run is None:
                return
            rec = state.active_run
            rec.ended_at = time.time()
            rec.exit_code = returncode
            if returncode == 0:
                rec.status = "done"
            elif returncode is None:
                rec.status = "killed"
            else:
                rec.status = "failed"
            state.history.insert(0, rec)
            state.history = state.history[:20]
            state.active_run = None

    # ── Kill ────────────────────────────────────────────────────────

    def kill_active(self) -> RunRecord:
        state = self.store.read_state()
        if state.active_run is None:
            raise RunnerError("no active run")
        pid = state.active_run.pid
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
                for _ in range(30):
                    time.sleep(0.1)
                    proc = self._active_popen
                    if proc is None or proc.poll() is not None:
                        break
                else:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except ProcessLookupError:
                pass  # Already dead; continue to finalize.

        self._finalize(returncode=None)
        return self.store.read_state().history[0]
