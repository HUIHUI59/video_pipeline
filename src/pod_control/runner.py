"""Local subprocess orchestration for pod runs.

Spawns scripts/runpod/run_all.sh in its own process group (setsid) so a
Kill can nuke the whole tree (python + ssh + ffmpeg children). Writes PID
and lifecycle state via store.state_lock() so the single-run-slot
invariant holds across API requests.

The actual upload / inference / download still happens inside the shell
script; this module only owns "one run active" bookkeeping.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

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
    pod: PodProfile,
    batch: Batch,
    *,
    preset_path: str | None,
    run_all_script: Path,
) -> list[str]:
    """Translate a Batch + PodProfile to run_all.sh invocation.

    run_all.sh is a thin chain over upload → 02_run → download; the pod
    profile is passed as env (RUNPOD_SSH_*) rather than a file because
    it lives in the pod_control store, not as a servers.yaml row.
    """
    cmd: list[str] = ["bash", str(run_all_script)]
    if preset_path:
        cmd += ["--config", preset_path]
    fp = batch.filter_params
    cmd += ["--movies", batch.movie]
    if fp.categories:
        cmd += ["--categories", ",".join(fp.categories)]
    if not fp.skip_bad_quality:
        cmd += ["--include-bad-quality"]
    if not fp.skip_landscape:
        cmd += ["--include-landscape"]
    if fp.max_shots:
        cmd += ["--max-shots", str(fp.max_shots)]
    return cmd


class Runner:
    """Lifecycle owner for the single active run slot."""

    def __init__(
        self,
        store: Store,
        *,
        run_all_script: Path = _RUN_ALL,
        kill_script: Path = _KILL_SH,
    ) -> None:
        self.store = store
        self.run_all_script = run_all_script
        self.kill_script = kill_script
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
        cmd = _build_run_all_cmd(
            pod, batch,
            preset_path=preset_path,
            run_all_script=self.run_all_script,
        )
        run_id = _new_run_id()
        run_dir = self.store.run_dir(run_id)
        stdout_path = run_dir / "stdout.log"

        env = os.environ.copy()
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
