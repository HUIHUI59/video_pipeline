"""M2 unit tests for src/pod_control/store.py.

AAA structure throughout. tmp_path fixture isolates each test.
"""
from __future__ import annotations

import json

import pytest

from src.pod_control.store import (
    Batch,
    FilterParams,
    PodProfile,
    RunRecord,
    Store,
    StoreError,
)


# ── Model validation -------------------------------------------------


def test_filter_params_defaults():
    fp = FilterParams()
    assert fp.categories == ["single", "dominant", "multi"]
    assert fp.skip_bad_quality is True
    assert fp.skip_landscape is True
    assert fp.max_shots is None


def test_batch_name_must_be_slug():
    with pytest.raises(ValueError):
        Batch(name="bad name!", movies=["m"], filter_params=FilterParams())
    Batch(name="ok-name_1", movies=["m"], filter_params=FilterParams())


def test_batch_legacy_movie_field_upgrades_to_movies():
    """Loading a batch JSON written pre-migration should still work."""
    b = Batch.model_validate({
        "name": "legacy1",
        "movie": "The_Dinner_2017",
        "filter_params": {},
    })
    assert b.movies == ["The_Dinner_2017"]
    # Property accessor also works.
    assert b.movie == "The_Dinner_2017"


def test_batch_movies_list_supported():
    b = Batch(name="multi1", movies=["M1", "M2", "M3"],
              filter_params=FilterParams())
    assert b.movies == ["M1", "M2", "M3"]


def test_batch_movies_must_not_be_empty():
    with pytest.raises(ValueError, match="at least one"):
        Batch(name="x", movies=[], filter_params=FilterParams())


def test_pod_name_must_be_slug():
    with pytest.raises(ValueError):
        PodProfile(name="bad name", host="h", user="u",
                   ssh_key="k", workspace="/w")
    PodProfile(name="ok_pod-1", host="h", user="u",
               ssh_key="k", workspace="/w")


# ── Batches ----------------------------------------------------------


def test_save_then_get_batch_round_trip(tmp_path):
    store = Store(tmp_path)
    b = Batch(name="b1", movie="m", filter_params=FilterParams(max_shots=5))
    store.save_batch(b)
    loaded = store.get_batch("b1")
    assert loaded is not None
    assert loaded.name == "b1"
    assert loaded.filter_params.max_shots == 5


def test_save_batch_rejects_duplicate_without_overwrite(tmp_path):
    store = Store(tmp_path)
    b = Batch(name="b1", movie="m", filter_params=FilterParams())
    store.save_batch(b)
    with pytest.raises(StoreError):
        store.save_batch(b)


def test_save_batch_overwrite_replaces(tmp_path):
    store = Store(tmp_path)
    b = Batch(name="b1", movie="m", filter_params=FilterParams())
    store.save_batch(b)
    b2 = Batch(name="b1", movie="m2", filter_params=FilterParams())
    store.save_batch(b2, overwrite=True)
    assert store.get_batch("b1").movie == "m2"


def test_list_batches_sorted(tmp_path):
    store = Store(tmp_path)
    for n in ("b3", "b1", "b2"):
        store.save_batch(Batch(name=n, movie="m", filter_params=FilterParams()))
    names = [b.name for b in store.list_batches()]
    assert names == ["b1", "b2", "b3"]


def test_delete_batch_missing_raises(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(StoreError):
        store.delete_batch("nope")


def test_delete_batch_blocked_when_active_run_matches(tmp_path):
    """Delete is blocked only when this batch IS the current active_run."""
    from src.pod_control.store import RunRecord
    store = Store(tmp_path)
    b = Batch(name="b1", movie="m", filter_params=FilterParams(),
              status="running")
    store.save_batch(b)
    with store.state_lock() as state:
        state.active_run = RunRecord(
            id="r1", batch_name="b1", pod_name="p", pid=9999,
        )
    with pytest.raises(StoreError, match="running"):
        store.delete_batch("b1")


def test_delete_batch_stale_running_status_succeeds(tmp_path):
    """A batch with status='running' but no matching active_run can be deleted."""
    store = Store(tmp_path)
    b = Batch(name="b1", movie="m", filter_params=FilterParams(),
              status="running")
    store.save_batch(b)
    # No active_run in state → stale status, should succeed
    store.delete_batch("b1")
    assert store.get_batch("b1") is None


def test_delete_batch_ready_succeeds(tmp_path):
    store = Store(tmp_path)
    b = Batch(name="b1", movie="m", filter_params=FilterParams())
    store.save_batch(b)
    store.delete_batch("b1")
    assert store.get_batch("b1") is None


def test_corrupt_batch_file_raises(tmp_path):
    store = Store(tmp_path)
    (store.root / "batches" / "broken.json").write_text("not json")
    with pytest.raises(StoreError, match="corrupt"):
        store.list_batches()


# ── Pods -------------------------------------------------------------


def test_list_pods_empty_when_no_file(tmp_path):
    store = Store(tmp_path)
    assert store.list_pods() == []


def test_upsert_pod_creates_then_overwrites(tmp_path):
    store = Store(tmp_path)
    p1 = PodProfile(name="h100", host="1.2.3.4", user="root",
                    ssh_key="~/.ssh/id_ed25519", workspace="/w")
    store.upsert_pod(p1)
    assert len(store.list_pods()) == 1
    p1b = PodProfile(name="h100", host="5.6.7.8", user="root",
                     ssh_key="~/.ssh/id_ed25519", workspace="/w")
    store.upsert_pod(p1b)
    pods = store.list_pods()
    assert len(pods) == 1
    assert pods[0].host == "5.6.7.8"


def test_get_pod_finds_or_returns_none(tmp_path):
    store = Store(tmp_path)
    store.upsert_pod(PodProfile(name="h100", host="h", user="u",
                                ssh_key="k", workspace="/w"))
    assert store.get_pod("h100").name == "h100"
    assert store.get_pod("missing") is None


def test_delete_pod_missing_raises(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(StoreError):
        store.delete_pod("nope")


def test_pod_yaml_persists_across_store_instances(tmp_path):
    s1 = Store(tmp_path)
    s1.upsert_pod(PodProfile(name="h100", host="h", user="u",
                              ssh_key="k", workspace="/w"))
    s2 = Store(tmp_path)
    assert [p.name for p in s2.list_pods()] == ["h100"]


def test_pods_list_sorted_after_upsert(tmp_path):
    store = Store(tmp_path)
    for n in ("z-pod", "a-pod", "m-pod"):
        store.upsert_pod(PodProfile(name=n, host="h", user="u",
                                    ssh_key="k", workspace="/w"))
    assert [p.name for p in store.list_pods()] == ["a-pod", "m-pod", "z-pod"]


def test_invalid_pods_yaml_raises(tmp_path):
    store = Store(tmp_path)
    store.pods_file.write_text("not_a_list: true")
    with pytest.raises(StoreError, match="must contain a list"):
        store.list_pods()


# ── State + lock -----------------------------------------------------


def test_read_state_default_when_missing(tmp_path):
    store = Store(tmp_path)
    s = store.read_state()
    assert s.active_run is None
    assert s.history == []


def test_state_lock_persists_mutation(tmp_path):
    store = Store(tmp_path)
    rec = RunRecord(id="r1", batch_name="b", pod_name="p")
    with store.state_lock() as state:
        state.active_run = rec
    reloaded = store.read_state()
    assert reloaded.active_run is not None
    assert reloaded.active_run.id == "r1"


def test_state_lock_serializes_writers(tmp_path):
    """fcntl LOCK_EX should serialize concurrent writes from threads."""
    import threading

    store = Store(tmp_path)
    seen_inside: list[str] = []

    def writer(tag: str):
        with store.state_lock() as state:
            seen_inside.append(tag + "-enter")
            state.history.append(
                RunRecord(id=tag, batch_name="b", pod_name="p")
            )
            seen_inside.append(tag + "-exit")

    t1 = threading.Thread(target=writer, args=("a",))
    t2 = threading.Thread(target=writer, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Each writer's enter and exit must be adjacent (no interleave).
    for i in range(0, len(seen_inside), 2):
        assert seen_inside[i].split("-")[0] == seen_inside[i + 1].split("-")[0]
    assert len(store.read_state().history) == 2


def test_atomic_state_write_no_partial_file(tmp_path):
    store = Store(tmp_path)
    rec = RunRecord(id="r1", batch_name="b", pod_name="p")
    with store.state_lock() as state:
        state.active_run = rec
    leftover = list(tmp_path.rglob("*.tmp"))
    assert leftover == [], f"unexpected tmp files: {leftover}"


# ── Layout sanity ---------------------------------------------------


def test_store_creates_required_dirs(tmp_path):
    Store(tmp_path)
    assert (tmp_path / "batches").is_dir()
    assert (tmp_path / "runs").is_dir()


def test_run_dir_idempotent(tmp_path):
    store = Store(tmp_path)
    d1 = store.run_dir("run-001")
    d2 = store.run_dir("run-001")
    assert d1 == d2
    assert d1.is_dir()


def test_batch_file_is_valid_json(tmp_path):
    store = Store(tmp_path)
    store.save_batch(Batch(name="b1", movie="m",
                           filter_params=FilterParams(max_shots=3)))
    raw = json.loads(store.batch_file("b1").read_text())
    assert raw["name"] == "b1"
    assert raw["filter_params"]["max_shots"] == 3
