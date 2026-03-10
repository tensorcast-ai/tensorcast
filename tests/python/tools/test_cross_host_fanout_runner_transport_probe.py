#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import grpc
import pytest


def _load_runner_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "examples" / "cross_host" / "cross_host_fanout_runner.py"
    spec = importlib.util.spec_from_file_location("cross_host_fanout_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cross_host_fanout_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_query_transport_rows_via_gs_rpc_channel_ready_timeout_sets_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class _FakeChannel:
        def close(self) -> None:
            return None

    class _ReadyFuture:
        def result(self, timeout: float | None = None) -> None:
            _ = timeout
            raise grpc.FutureTimeoutError()

    monkeypatch.setattr(runner.grpc, "insecure_channel", lambda *_: _FakeChannel())
    monkeypatch.setattr(
        runner.grpc, "channel_ready_future", lambda *_: _ReadyFuture()
    )

    now = datetime.now(timezone.utc)
    payload = runner.query_transport_rows_via_gs_rpc(
        gs_addr="127.0.0.1:50051",
        started_at_utc=now - timedelta(minutes=1),
        finished_at_utc=now,
        limit=1024,
    )
    assert payload["error"] is not None
    assert "channel ready timeout" in str(payload["error"]).lower()


def test_classify_timeout_root_reads_daemon_logs_excerpt() -> None:
    runner = _load_runner_module()
    root = runner.classify_timeout_root(
        error_message="Deadline Exceeded",
        failure_probe={
            "daemon_logs_excerpt": "No available replicas for replica xyz",
        },
    )
    assert root == "wait_timeout"


def test_is_transient_daemon_start_error() -> None:
    runner = _load_runner_module()
    transient = RuntimeError(
        "Error: Global Store connect mode requires a reachable address"
    )
    non_transient = RuntimeError("Error: daemon_id must be configured")
    assert runner._is_transient_daemon_start_error(transient) is True
    assert runner._is_transient_daemon_start_error(non_transient) is False


def test_restart_daemon_retries_transient_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    worker = runner.WorkerSpec(
        name="get1",
        process_id="proc-1",
        daemon_addr="127.0.0.1:63611",
        advertise_ip="127.0.0.1",
        grpc_port=63611,
        p2p_port=64611,
        daemon_session="session-1",
        daemon_id="daemon-1",
        home="/tmp/home",
        storage="/tmp/storage",
    )

    call_count = {"value": 0}

    def fake_run_remote(
        process_id: str,
        inner_cmd: str,
        *,
        timeout_sec: float,
    ) -> str:
        _ = (process_id, timeout_sec)
        if "tensorcast-cli daemon start" in inner_cmd:
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise RuntimeError(
                    "Error: Global Store connect mode requires a reachable address"
                )
        return "{}"

    ready_called = {"value": False}

    def fake_wait_daemon_ready(*, worker: Any, timeout_sec: float) -> None:
        _ = timeout_sec
        assert worker.name == "get1"
        ready_called["value"] = True

    monkeypatch.setattr(runner, "run_remote", fake_run_remote)
    monkeypatch.setattr(runner, "wait_daemon_ready", fake_wait_daemon_ready)
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)
    monkeypatch.setattr(runner, "DAEMON_START_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(runner, "DAEMON_START_RETRY_BACKOFF_SEC", 0.0)

    runner.restart_daemon(
        worker=worker,
        daemon_config="examples/config/store_daemon_config_cross_host_bench.yaml",
        gs_addr="127.0.0.1:50051",
        conn=20,
        buffers=16,
        maxw=16,
        expected_gpu_channels=0,
        promotion_max_concurrency=4,
        timeout_sec=30.0,
    )

    assert call_count["value"] == 2
    assert ready_called["value"] is True
