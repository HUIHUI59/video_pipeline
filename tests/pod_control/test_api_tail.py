"""M6 API tests: GET /api/runs/{id}/tail.

ssh + subprocess are mocked throughout — no real network.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.pod_control.api import create_app
from src.pod_control.ssh import TailResult
from src.pod_control.store import PodProfile, RunRecord, Store


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    data_root.mkdir()
    output_root.mkdir()
    app = create_app(data_root, output_root=output_root)
    store = Store(data_root)
    store.upsert_pod(PodProfile(
        name="h100", host="1.2.3.4", user="root",
        ssh_key="~/.ssh/id_ed25519", port=22,
        workspace="/workspace/video_pipeline",
    ))
    return TestClient(app), store


def _set_active(store: Store, run_id: str = "r1") -> None:
    with store.state_lock() as state:
        state.active_run = RunRecord(
            id=run_id, batch_name="b1", pod_name="h100", pid=123,
        )


def test_tail_404_when_run_not_found(client):
    c, _ = client
    r = c.get("/api/runs/ghost/tail")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "run_not_found"


def test_tail_finished_run_returns_empty_with_status(client):
    c, store = client
    with store.state_lock() as state:
        state.history.append(RunRecord(
            id="done-1", batch_name="b1", pod_name="h100",
            status="done", exit_code=0,
        ))
    r = c.get("/api/runs/done-1/tail")
    assert r.status_code == 200
    body = r.json()
    assert body["finished"] is True
    assert body["status"] == "done"
    assert body["text"] == ""


def test_tail_active_run_returns_log_bytes(client):
    c, store = client
    _set_active(store)
    fake_tail = TailResult(text="hello world\n", next_offset=12,
                           pod_unreachable=False)
    with patch("src.pod_control.api.pcssh.tail_remote_log",
               return_value=fake_tail), \
         patch("src.pod_control.api.pcssh.build_ssh_args",
               return_value=["ssh", "root@1.2.3.4"]), \
         patch("subprocess.run") as mock_sp:
        mock_sp.return_value = SimpleNamespace(
            returncode=0, stdout=b"42 /w/output/.checkpoint.jsonl\n",
            stderr=b"",
        )
        r = c.get("/api/runs/r1/tail?offset=0")
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hello world\n"
    assert body["next_offset"] == 12
    assert body["pod_unreachable"] is False
    assert body["checkpoint"]["done"] == 42
    assert body["finished"] is False


def test_tail_pod_unreachable_sets_flag(client):
    c, store = client
    _set_active(store)
    fake_tail = TailResult(text="", next_offset=0, pod_unreachable=True)
    with patch("src.pod_control.api.pcssh.tail_remote_log",
               return_value=fake_tail):
        r = c.get("/api/runs/r1/tail")
    assert r.status_code == 200
    body = r.json()
    assert body["pod_unreachable"] is True
    assert body["text"] == ""
    assert body["checkpoint"] == {"done": 0, "failed": 0, "pending": 0}


def test_tail_advances_offset(client):
    c, store = client
    _set_active(store)
    fake_tail = TailResult(text="abc", next_offset=203,
                           pod_unreachable=False)
    with patch("src.pod_control.api.pcssh.tail_remote_log",
               return_value=fake_tail), \
         patch("src.pod_control.api.pcssh.build_ssh_args",
               return_value=["ssh", "root@1.2.3.4"]), \
         patch("subprocess.run") as mock_sp:
        mock_sp.return_value = SimpleNamespace(
            returncode=0, stdout=b"0\n", stderr=b"",
        )
        r = c.get("/api/runs/r1/tail?offset=200")
    body = r.json()
    assert body["next_offset"] == 203
