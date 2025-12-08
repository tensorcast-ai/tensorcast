#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import contextlib
import subprocess

from tensorcast import runtime
from tensorcast.cli_utils import paths
from tensorcast.cli_utils.paths import set_current_session_id
from tensorcast.cli_utils.process import (
    append_process_record,
    instance_fingerprint,
    read_json_default,
    read_runtime_state,
    update_runtime_daemon,
    write_runtime_state,
)


def test_reconcile_removes_stale_session(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    inst = paths.session_paths("sess-stale")
    set_current_session_id(inst.id)

    append_process_record(
        pids_path=inst.pids_json,
        entry={
            "role": "daemon",
            "pid": 999999,
            "cmd": ["tensorcast_daemon"],
            "stdout": str(inst.logs / "out"),
            "stderr": str(inst.logs / "err"),
            "start_time": 1.0,
        },
        session_id=inst.id,
        lock_path=inst.pids_lock,
    )
    update_runtime_daemon(
        path=paths.runtime_state_path(),
        session_id=inst.id,
        pid=999999,
        address="127.0.0.1:65500",
        p2p_address=None,
        owner=True,
        fingerprint=instance_fingerprint(999999),
    )

    session = runtime.reconcile(inst.id)
    assert session is None
    state = read_runtime_state(paths.runtime_state_path())
    assert "daemon" not in state
    assert paths.get_current_session_id() is None
    data = read_json_default(inst.pids_json, {})
    assert not data.get("processes")


def test_reconcile_keeps_live_pid_without_address(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    inst = paths.session_paths("sess-live")
    proc = subprocess.Popen(["sleep", "2"])
    try:
        append_process_record(
            pids_path=inst.pids_json,
            entry={
                "role": "daemon",
                "pid": proc.pid,
                "cmd": ["sleep", "2"],
                "stdout": str(inst.logs / "out"),
                "stderr": str(inst.logs / "err"),
                "start_time": 1.0,
            },
            session_id=inst.id,
            lock_path=inst.pids_lock,
        )
        update_runtime_daemon(
            path=paths.runtime_state_path(),
            session_id=inst.id,
            pid=proc.pid,
            address=None,
            p2p_address=None,
            owner=True,
            fingerprint=instance_fingerprint(proc.pid),
        )

        session = runtime.reconcile(inst.id)
        assert session is not None
        assert session.session_id == inst.id
        assert session.daemon_pid == proc.pid
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=1.0)


def test_reconcile_fingerprint_mismatch_preserves_global_state(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    inst = paths.session_paths("sess-fp")
    proc = subprocess.Popen(["sleep", "2"])
    try:
        append_process_record(
            pids_path=inst.pids_json,
            entry={
                "role": "daemon",
                "pid": proc.pid,
                "cmd": ["sleep", "2"],
                "stdout": str(inst.logs / "out"),
                "stderr": str(inst.logs / "err"),
                "start_time": 1.0,
            },
            session_id=inst.id,
            lock_path=inst.pids_lock,
        )
        bad_fingerprint = instance_fingerprint(proc.pid)
        bad_fingerprint["boot_id"] = "other-boot"
        update_runtime_daemon(
            path=paths.runtime_state_path(),
            session_id=inst.id,
            pid=proc.pid,
            address="127.0.0.1:65501",
            p2p_address=None,
            owner=True,
            fingerprint=bad_fingerprint,
        )
        state = read_runtime_state(paths.runtime_state_path())
        state["global_store"] = {"cluster_token": "cluster-123", "session_id": "gs-1"}
        write_runtime_state(paths.runtime_state_path(), state)
        set_current_session_id(inst.id)

        session = runtime.reconcile(inst.id)
        assert session is None

        updated_state = read_runtime_state(paths.runtime_state_path())
        assert "daemon" not in updated_state
        assert updated_state.get("global_store", {}).get("cluster_token") == "cluster-123"
        assert paths.get_current_session_id() is None
        data = read_json_default(inst.pids_json, {})
        assert not data.get("processes")
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=1.0)
