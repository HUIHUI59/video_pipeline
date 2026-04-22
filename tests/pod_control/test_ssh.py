"""M4 unit tests: ssh.py wrappers.

All network calls are mocked — tests never actually invoke the ssh binary.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from src.pod_control import ssh as pcssh
from src.pod_control.ssh import ConnectResult, TailResult, build_ssh_args
from src.pod_control.store import PodProfile


def _pod() -> PodProfile:
    return PodProfile(
        name="h100-01",
        host="1.2.3.4",
        user="root",
        ssh_key="~/.ssh/id_ed25519",
        port=22022,
        workspace="/workspace/video_pipeline",
    )


# ── build_ssh_args ---------------------------------------------------


def test_build_ssh_args_includes_identity_port_and_host():
    args = build_ssh_args(_pod())
    assert args[0] == "ssh"
    assert "-i" in args
    assert "-p" in args
    assert args[args.index("-p") + 1] == "22022"
    assert args[-1] == "root@1.2.3.4"


def test_build_ssh_args_enables_batchmode_and_accept_new():
    args = " ".join(build_ssh_args(_pod()))
    assert "BatchMode=yes" in args
    assert "StrictHostKeyChecking=accept-new" in args


# ── test_connect -----------------------------------------------------


def test_test_connect_success():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=0, stdout="pod_control_ok\n", stderr=""
        )
        r = pcssh.test_connect(_pod())
    assert isinstance(r, ConnectResult)
    assert r.ok is True
    assert r.latency_ms >= 0
    assert r.message == "ok"


def test_test_connect_nonzero_returncode_reports_stderr():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="ssh: connect to host 1.2.3.4 port 22022: No route to host",
        )
        r = pcssh.test_connect(_pod())
    assert r.ok is False
    assert "No route to host" in r.message


def test_test_connect_timeout_reports_timeout():
    with patch("src.pod_control.ssh.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd=[], timeout=15)):
        r = pcssh.test_connect(_pod(), timeout_s=15)
    assert r.ok is False
    assert "timed out" in r.message


def test_test_connect_missing_binary_reports_friendly():
    with patch("src.pod_control.ssh.subprocess.run",
               side_effect=FileNotFoundError):
        r = pcssh.test_connect(_pod())
    assert r.ok is False
    assert "ssh binary" in r.message


def test_test_connect_zero_exit_but_wrong_stdout_fails():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=0, stdout="something_else", stderr=""
        )
        r = pcssh.test_connect(_pod())
    assert r.ok is False


# ── tail_remote_log --------------------------------------------------


def test_tail_remote_log_happy_returns_text_and_advances_offset():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=0, stdout=b"hello\n", stderr=b""
        )
        r = pcssh.tail_remote_log(_pod(), remote_path="/w/pod_runner.log",
                            offset=0)
    assert isinstance(r, TailResult)
    assert r.text == "hello\n"
    assert r.next_offset == len("hello\n")
    assert r.pod_unreachable is False


def test_tail_remote_log_advances_from_nonzero_offset():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=0, stdout=b"world", stderr=b""
        )
        r = pcssh.tail_remote_log(_pod(), remote_path="/w/log.txt", offset=100)
    assert r.next_offset == 105


def test_tail_remote_log_ssh_failure_marks_unreachable():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=255, stdout=b"", stderr=b"connection refused"
        )
        r = pcssh.tail_remote_log(_pod(), remote_path="/w/log.txt", offset=42)
    assert r.pod_unreachable is True
    assert r.next_offset == 42
    assert r.text == ""


def test_tail_remote_log_timeout_marks_unreachable():
    with patch("src.pod_control.ssh.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd=[], timeout=10)):
        r = pcssh.tail_remote_log(_pod(), remote_path="/w/log.txt", offset=42)
    assert r.pod_unreachable is True
    assert r.next_offset == 42


def test_tail_remote_log_missing_ssh_binary_marks_unreachable():
    with patch("src.pod_control.ssh.subprocess.run",
               side_effect=FileNotFoundError):
        r = pcssh.tail_remote_log(_pod(), remote_path="/w/log.txt", offset=0)
    assert r.pod_unreachable is True


def test_tail_remote_log_decodes_non_utf8_safely():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=0, stdout=b"\xff\xfebad", stderr=b""
        )
        r = pcssh.tail_remote_log(_pod(), remote_path="/w/log.txt", offset=0)
    assert r.next_offset == 5


def test_tail_remote_log_shlex_quotes_path():
    with patch("src.pod_control.ssh.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=0, stdout=b"", stderr=b""
        )
        pcssh.tail_remote_log(_pod(), remote_path="/weird path/with spaces.log",
                        offset=0)
        cmd = mock_run.call_args.args[0]
        joined = " ".join(str(c) for c in cmd)
        assert "'/weird path/with spaces.log'" in joined
