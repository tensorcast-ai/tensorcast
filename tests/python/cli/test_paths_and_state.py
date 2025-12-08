#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from tensorcast.cli_utils import paths
from tensorcast.cli_utils.process import (
    append_process_record,
    clear_runtime_daemon,
    instance_fingerprint,
    read_json_default,
    read_runtime_state,
    update_runtime_daemon,
)


def test_home_dir_and_current_session_are_atomic(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    home = paths.home_dir()
    assert home == tmp_path

    runtime_lock = paths.runtime_lock_path()
    assert runtime_lock.exists()
    assert (runtime_lock.stat().st_mode & 0o777) == 0o600

    paths.set_current_session_id("sess-123")
    assert paths.get_current_session_id() == "sess-123"
    assert (paths.current_session_path().stat().st_mode & 0o777) == 0o600


def test_pids_append_and_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    inst = paths.session_paths("sess-append")

    entry1 = {
        "role": "daemon",
        "pid": 1001,
        "cmd": ["tensorcast_daemon"],
        "stdout": str(inst.logs / "out"),
        "stderr": str(inst.logs / "err"),
        "start_time": 1.0,
    }
    append_process_record(
        pids_path=inst.pids_json, entry=entry1, session_id=inst.id, lock_path=inst.pids_lock
    )
    entry2 = {
        "role": "daemon",
        "pid": 1002,
        "cmd": ["tensorcast_daemon"],
        "stdout": str(inst.logs / "out"),
        "stderr": str(inst.logs / "err"),
        "start_time": 2.0,
    }
    append_process_record(
        pids_path=inst.pids_json, entry=entry2, session_id=inst.id, lock_path=inst.pids_lock
    )

    data = read_json_default(inst.pids_json, {})
    assert data["schema_version"] == 1
    assert data["session_id"] == inst.id
    assert {p["pid"] for p in data["processes"]} == {1001, 1002}


def test_runtime_state_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    inst = paths.session_paths("sess-runtime")
    runtime_state_file = paths.runtime_state_path()

    update_runtime_daemon(
        path=runtime_state_file,
        session_id=inst.id,
        pid=4321,
        address="127.0.0.1:50052",
        p2p_address=None,
        owner=True,
        fingerprint=instance_fingerprint(4321),
    )

    state = read_runtime_state(runtime_state_file)
    assert state["schema_version"] == 1
    assert state["daemon"]["session_id"] == inst.id
    assert state["daemon"]["pid"] == 4321

    clear_runtime_daemon(runtime_state_file)
    cleared = read_runtime_state(runtime_state_file)
    assert "daemon" not in cleared
