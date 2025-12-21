#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import io
import json
from pathlib import Path

from tensorcast.cli_utils.global_store_manager import (
    start_global_store,
    stop_global_store,
)
from tensorcast.cli_utils.health import GlobalStoreHealth
from tensorcast.cli_utils.paths import global_session_paths, runtime_state_path


class _FakeProc:
    def __init__(self, pid: int):
        self.pid = pid
        self.args = ["uv", "run", "-m", "tensorcast.global_store", "--config", "cfg"]
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
