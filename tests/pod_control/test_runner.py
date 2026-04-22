"""M5 unit tests: runner.py lifecycle (launch / poll / kill / finalize).

Popen + os.killpg are mocked — no real subprocess is spawned.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pod_control.runner import Runner, RunnerError, _build_run_all_cmd
from src.pod_control.store import Batch, FilterParams, PodProfile, Store


def _batch(**kw) -> Batch:
    fp = kw.pop("filter_params", FilterParams())
    return Batch(name=kw.pop("name", "b1"),
                 movie=kw.pop("movie", "M1"),
                 filter_params=fp, **kw)


def _pod(**kw) -> PodProfile:
    base = dict(
        name="h100", host="1.2.3.4", user="root",
        ssh_key="~/.ssh/id_ed25519", port=22,
        workspace="/workspace/video_pipeline",
    )
    base.update(kw)
    return PodProfile(**base)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def runner_with_fake_script(tmp_path, store):
    """Runner wired to a dummy run_all.sh + kill.sh so launch() passes the
    existence check. Popen is monkey-patched in each test that launches."""
    fake_run_all = tmp_path / "run_all.sh"
    fake_kill = tmp_path / "99_kill.sh"
    fake_run_all.write_text("#!/bin/bash\necho hi\n")
    fake_kill.write_text("#!/bin/bash\necho kill\n")
    return Runner(store, run_all_script=fake_run_all, kill_script=fake_kill)


# ── _build_run_all_cmd ----------------------------------------------


def test_build_run_all_cmd_basic(tmp_path):
    script = tmp_path / "run_all.sh"
    cmd = _build_run_all_cmd(
        _pod(), _batch(), preset_path=None, run_all_script=script,
    )
    assert cmd[0] == "bash"
    assert cmd[1] == str(script)
    assert "--movies" in cmd
    assert cmd[cmd.index("--movies") + 1] == "M1"


def test_build_run_all_cmd_includes_categories_and_max_shots(tmp_path):
    b = _batch(filter_params=FilterParams(
        categories=["single", "wide"], max_shots=50,
    ))
    cmd = _build_run_all_cmd(
        _pod(), b, preset_path=None, run_all_script=tmp_path / "r.sh",
    )
    assert cmd[cmd.index("--categories") + 1] == "single,wide"
    assert cmd[cmd.index("--max-shots") + 1] == "50"


def test_build_run_all_cmd_preset_and_include_flags(tmp_path):
    b = _batch(filter_params=FilterParams(
        skip_bad_quality=False, skip_landscape=False,
    ))
    cmd = _build_run_all_cmd(
        _pod(), b, preset_path="/tmp/preset.yaml",
        run_all_script=tmp_path / "r.sh",
    )
    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1] == "/tmp/preset.yaml"
    assert "--include-bad-quality" in cmd
    assert "--include-landscape" in cmd


# ── Launch ----------------------------------------------------------


def test_launch_missing_run_all_script_raises(tmp_path, store):
    r = Runner(store, run_all_script=tmp_path / "nope.sh")
    with pytest.raises(RunnerError, match="missing"):
        r.launch(_batch(), _pod())


def test_launch_sets_active_run_and_pid(runner_with_fake_script):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 424242
    with patch("src.pod_control.runner.os.setsid"):
        r._popen = MagicMock(return_value=fake_popen)
        rec = r.launch(_batch(), _pod())
    state = r.store.read_state()
    assert state.active_run is not None
    assert state.active_run.id == rec.id
    assert state.active_run.pid == 424242
    assert state.active_run.status == "running"


def test_launch_writes_stdout_log_file(runner_with_fake_script):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 424242
    with patch("src.pod_control.runner.os.setsid"):
        r._popen = MagicMock(return_value=fake_popen)
        rec = r.launch(_batch(), _pod())
    assert (r.store.run_dir(rec.id) / "stdout.log").exists()


def test_launch_rejects_second_launch_when_active(runner_with_fake_script):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 1
    with patch("src.pod_control.runner.os.setsid"):
        r._popen = MagicMock(return_value=fake_popen)
        r.launch(_batch(), _pod())
        with pytest.raises(RunnerError, match="run_already_active"):
            r.launch(_batch(name="b2"), _pod())


# ── Poll + finalize ------------------------------------------------


def test_poll_active_returns_none_when_no_run(runner_with_fake_script):
    r = runner_with_fake_script
    assert r.poll_active() is None


def test_poll_active_keeps_running_when_process_alive(
    runner_with_fake_script,
):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 1
    fake_popen.poll.return_value = None
    with patch("src.pod_control.runner.os.setsid"):
        r._popen = MagicMock(return_value=fake_popen)
        r.launch(_batch(), _pod())
    result = r.poll_active()
    assert result is not None
    assert result.status == "running"


def test_poll_active_finalizes_done_on_exit_0(runner_with_fake_script):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 1
    fake_popen.poll.return_value = 0
    with patch("src.pod_control.runner.os.setsid"):
        r._popen = MagicMock(return_value=fake_popen)
        r.launch(_batch(), _pod())
    r.poll_active()
    state = r.store.read_state()
    assert state.active_run is None
    assert len(state.history) == 1
    assert state.history[0].status == "done"
    assert state.history[0].exit_code == 0


def test_poll_active_finalizes_failed_on_nonzero(runner_with_fake_script):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 1
    fake_popen.poll.return_value = 2
    with patch("src.pod_control.runner.os.setsid"):
        r._popen = MagicMock(return_value=fake_popen)
        r.launch(_batch(), _pod())
    r.poll_active()
    state = r.store.read_state()
    assert state.history[0].status == "failed"
    assert state.history[0].exit_code == 2


# ── Kill ---------------------------------------------------------


def test_kill_active_raises_when_none(runner_with_fake_script):
    with pytest.raises(RunnerError, match="no active run"):
        runner_with_fake_script.kill_active()


def test_kill_active_sends_sigterm_and_finalizes(runner_with_fake_script):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 99999
    fake_popen.poll.return_value = None
    with patch("src.pod_control.runner.os.setsid"), \
         patch("src.pod_control.runner.os.killpg") as mock_killpg, \
         patch("src.pod_control.runner.time.sleep"):
        r._popen = MagicMock(return_value=fake_popen)
        r.launch(_batch(), _pod())
        fake_popen.poll.side_effect = [None, None, 0]
        killed = r.kill_active()
    assert mock_killpg.called
    assert killed.status == "killed"
    assert r.store.read_state().active_run is None


def test_kill_active_tolerates_process_already_gone(
    runner_with_fake_script,
):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 99999
    fake_popen.poll.return_value = 0
    with patch("src.pod_control.runner.os.setsid"), \
         patch("src.pod_control.runner.os.killpg",
               side_effect=ProcessLookupError), \
         patch("src.pod_control.runner.time.sleep"):
        r._popen = MagicMock(return_value=fake_popen)
        r.launch(_batch(), _pod())
        killed = r.kill_active()
    assert killed.status == "killed"


# ── History cap -------------------------------------------------


def test_history_capped_at_20(runner_with_fake_script):
    r = runner_with_fake_script
    fake_popen = MagicMock()
    fake_popen.pid = 1
    fake_popen.poll.return_value = 0
    with patch("src.pod_control.runner.os.setsid"):
        r._popen = MagicMock(return_value=fake_popen)
        for i in range(25):
            r.launch(_batch(name=f"b{i}"), _pod())
            r.poll_active()
    state = r.store.read_state()
    assert len(state.history) == 20
    assert state.history[0].batch_name == "b24"
