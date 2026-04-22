"""Thin subprocess wrapper around OpenSSH for pod_control.

Deliberately NOT using paramiko / asyncssh — the pipeline already depends
on the `ssh` binary via scripts/runpod/*.sh, so uniform behaviour + the
exact same known_hosts / key lookup path is important.

Only the test-connect and tail-log primitives live here. upload / run /
kill go through scripts/runpod/*.sh via runner.py (M5).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .store import PodProfile


def _known_hosts_path() -> str:
    # Project-local known_hosts so CI / localdev doesn't fight the user's
    # personal ~/.ssh/known_hosts.
    return str(Path("~/.ssh/pod_control_known_hosts").expanduser())


def build_ssh_args(pod: PodProfile) -> list[str]:
    """Build the leading portion of an ssh command for this pod.

    Return value: ['ssh', '-i', KEY, '-p', PORT, '-o', '...', USER@HOST].
    Caller appends the remote command.
    """
    return [
        "ssh",
        "-i", os.path.expanduser(pod.ssh_key),
        "-p", str(pod.port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={_known_hosts_path()}",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",   # no password prompts; fail fast
        f"{pod.user}@{pod.host}",
    ]


@dataclass
class ConnectResult:
    ok: bool
    latency_ms: int
    message: str


def test_connect(pod: PodProfile, *, timeout_s: float = 15.0) -> ConnectResult:
    """ssh pod 'echo pod_control_ok' — returns ConnectResult.

    No side effects on the remote. Non-zero returncode or non-matching
    stdout → ok=False with the stderr as message.
    """
    cmd = build_ssh_args(pod) + ["echo", "pod_control_ok"]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ConnectResult(False, int((time.time() - t0) * 1000),
                          f"ssh timed out after {timeout_s}s")
    except FileNotFoundError:
        return ConnectResult(False, 0, "ssh binary not found on PATH")

    latency = int((time.time() - t0) * 1000)
    if proc.returncode == 0 and "pod_control_ok" in proc.stdout:
        return ConnectResult(True, latency, "ok")
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return ConnectResult(
        False,
        latency,
        err[-1] if err else f"exit {proc.returncode}",
    )


@dataclass
class TailResult:
    text: str
    next_offset: int
    pod_unreachable: bool = False


def tail_remote_log(
    pod: PodProfile,
    *,
    remote_path: str,
    offset: int,
    timeout_s: float = 10.0,
) -> TailResult:
    """Pull bytes from `remote_path` starting at `offset`.

    Uses `tail -c +OFFSET` (1-based, hence offset+1). If the file doesn't
    exist yet, returns empty text and same offset. If ssh fails, marks
    pod_unreachable=True rather than raising — monitor UI stays alive.
    """
    remote_cmd = (
        f"tail -c +{offset + 1} {shlex.quote(remote_path)} 2>/dev/null || true"
    )
    cmd = build_ssh_args(pod) + [remote_cmd]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return TailResult("", offset, pod_unreachable=True)
    except FileNotFoundError:
        return TailResult("", offset, pod_unreachable=True)

    if proc.returncode != 0:
        return TailResult("", offset, pod_unreachable=True)

    data = proc.stdout or b""
    return TailResult(
        text=data.decode("utf-8", errors="replace"),
        next_offset=offset + len(data),
        pod_unreachable=False,
    )
