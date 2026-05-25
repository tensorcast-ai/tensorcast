#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import io
import json
import sys
from itertools import count

from tensorcast.cli_utils.global_store_manager import (
    start_global_store,
    stop_global_store,
)
from tensorcast.cli_utils.health import GlobalStoreHealth
from tensorcast.cli_utils.paths import global_session_paths, runtime_state_path


class _FakeProc:
    def __init__(self, pid: int):
        self.pid = pid
        self.args = [sys.executable, "-m", "tensorcast.global_store", "--config", "cfg"]
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_start_global_store_records_state(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    proc_calls: list[_FakeProc] = []

    def _fake_popen(args, **kwargs):
        proc = _FakeProc(2000 + len(proc_calls))
        proc_calls.append(proc)
        return proc

    health = GlobalStoreHealth(
        address="127.0.0.1:50051",
        listen_host="127.0.0.1",
        listen_port=50051,
        advertise_host="127.0.0.1",
        advertise_port=50051,
        metrics_port=8000,
        cluster_token="cluster-token",
        version="v1",
        db_file=str(tmp_path / "global_store.duckdb"),
    )

    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.subprocess.Popen", _fake_popen
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.ensure_process_started",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.wait_for_global_store",
        lambda *args, **kwargs: health,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.start_log_threads",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.ping_global_store",
        lambda *args, **kwargs: health,
    )

    instance = start_global_store(
        listen_host="127.0.0.1",
        listen_port=50051,
        metrics_port=8000,
        to_console=False,
    )
    assert instance.address == "127.0.0.1:50051"
    assert len(proc_calls) == 1
    assert proc_calls[0].args[:3] == [sys.executable, "-m", "tensorcast.global_store"]

    state = json.loads(runtime_state_path().read_text(encoding="utf-8"))
    gs_state = state.get("global_store", {})
    assert gs_state["address"] == "127.0.0.1:50051"
    assert gs_state["metrics_port"] == 8000
    assert gs_state["cluster_token"] == "cluster-token"

    inst_paths = global_session_paths(instance.id)
    session_state = json.loads(inst_paths.state_json.read_text(encoding="utf-8"))
    assert session_state["global_store"]["pid"] == proc_calls[0].pid
    pids = json.loads(inst_paths.pids_json.read_text(encoding="utf-8"))
    assert pids["processes"][0]["role"] == "global_store"

    # Second start should reuse existing healthy GS (no new popen)
    second = start_global_store(
        listen_host="127.0.0.1",
        listen_port=50051,
        metrics_port=8000,
        to_console=False,
    )
    assert second.id == instance.id
    assert len(proc_calls) == 1

    stop_global_store(session_id=instance.id, quiet=True, force=True)
    stopped_state = json.loads(runtime_state_path().read_text(encoding="utf-8"))
    assert stopped_state.get("global_store", {}).get("cluster_token") == "cluster-token"


def test_stop_global_store_prunes_reused_live_pid_without_signal(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))

    def _fake_popen(args, **kwargs):
        return _FakeProc(2000)

    health = GlobalStoreHealth(
        address="127.0.0.1:50051",
        listen_host="127.0.0.1",
        listen_port=50051,
        advertise_host="127.0.0.1",
        advertise_port=50051,
        metrics_port=8000,
        cluster_token="cluster-token",
        version="v1",
        db_file=str(tmp_path / "global_store.duckdb"),
    )

    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.subprocess.Popen", _fake_popen
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.ensure_process_started",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.wait_for_global_store",
        lambda *args, **kwargs: health,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.start_log_threads",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.ping_global_store",
        lambda *args, **kwargs: health,
    )

    instance = start_global_store(to_console=False)

    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.os.getpgid",
        lambda pid: 4242,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager._process_group_contains_global_store",
        lambda pgid, *, config_path: False,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager._find_global_store_pgids",
        lambda config_paths: set(),
    )

    def _fail_signal(*args, **kwargs):
        raise AssertionError("stop_global_store signaled a reused pid group")

    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.kill_force", _fail_signal
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.kill_gracefully", _fail_signal
    )

    stop_global_store(session_id=instance.id, quiet=True, force=True)

    pids = json.loads(global_session_paths(instance.id).pids_json.read_text())
    assert pids["processes"] == []


def test_start_global_store_rotates_cluster_token_on_fresh_restart(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    proc_ids = count(3000)
    health_tokens = iter(("cluster-a", "cluster-b"))

    def _fake_popen(args, **kwargs):
        return _FakeProc(next(proc_ids))

    def _fake_wait_for_global_store(*_args, **_kwargs):
        token = next(health_tokens)
        return GlobalStoreHealth(
            address="127.0.0.1:50051",
            listen_host="127.0.0.1",
            listen_port=50051,
            advertise_host="127.0.0.1",
            advertise_port=50051,
            metrics_port=8000,
            cluster_token=token,
            version="v1",
            db_file=str(tmp_path / f"{token}.duckdb"),
        )

    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.subprocess.Popen", _fake_popen
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.ensure_process_started",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.wait_for_global_store",
        _fake_wait_for_global_store,
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.start_log_threads",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "tensorcast.cli_utils.global_store_manager.ping_global_store",
        lambda *args, **kwargs: None,
    )

    first = start_global_store(to_console=False)
    assert first.cluster_token == "cluster-a"

    stop_global_store(session_id=first.id, quiet=True, force=True)

    second = start_global_store(to_console=False)
    assert second.cluster_token == "cluster-b"

    state = json.loads(runtime_state_path().read_text(encoding="utf-8"))
    assert state["global_store"]["cluster_token"] == "cluster-b"
