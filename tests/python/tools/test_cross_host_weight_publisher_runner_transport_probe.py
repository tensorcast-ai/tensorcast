#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import grpc
import pytest
from google.protobuf import timestamp_pb2


def _load_runner_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = (
        repo_root / "examples" / "cross_host" / "cross_host_weight_publisher_runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cross_host_weight_publisher_runner",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cross_host_weight_publisher_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transport_group_probe_uses_gs_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner_module()

    class _FakeChannel:
        def close(self) -> None:
            return None

    class _ReadyFuture:
        def result(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

    class _Stub:
        def QueryTransportWindow(self, *_args: object, **_kwargs: object) -> Any:
            created = timestamp_pb2.Timestamp()
            completed = timestamp_pb2.Timestamp()
            now = datetime.now(timezone.utc)
            created.FromDatetime(now - timedelta(seconds=2))
            completed.FromDatetime(now)
            row = runner.global_store_pb2.TransportWindowRow(
                transport_id="transport-1",
                replica_id="replica-1",
                artifact_id="artifact-1",
                status="completed",
                completion_outcome="success",
                request_id="req-1",
                requester_worker_id="receiver-1",
                group_id="",
                group_kind="",
                group_part_id="",
                group_total_parts=0,
                replica_memory_size_bytes=1024,
            )
            row.created_at.CopyFrom(created)
            row.completed_at.CopyFrom(completed)
            return runner.global_store_pb2.QueryTransportWindowResponse(
                status=runner.global_store_pb2.Status.STATUS_OK,
                rows=[row],
            )

    monkeypatch.setattr(runner.grpc, "insecure_channel", lambda *_: _FakeChannel())
    monkeypatch.setattr(
        runner.grpc, "channel_ready_future", lambda *_: _ReadyFuture()
    )
    monkeypatch.setattr(
        runner.global_store_pb2_grpc,
        "ClusterRuntimeServiceStub",
        lambda *_: _Stub(),
    )

    now = datetime.now(timezone.utc)
    probe = runner.query_transport_group_probe(
        gs_addr="127.0.0.1:50051",
        group_mode="none",
        group_kind="tp_version",
        started_at_utc=now - timedelta(minutes=1),
        finished_at_utc=now,
    )
    assert probe["error"] is None
    assert probe["audit_method"] == "gs_rpc"
    assert probe["window_has_transports"] is True
    assert probe["requester_tagged_complete"] is True
    assert probe["group_mode_consistent"] is True
    assert probe["group_contract_consistent"] is True


def test_transport_group_probe_rpc_failure_is_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class _FakeChannel:
        def close(self) -> None:
            return None

    class _ReadyFuture:
        def result(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

    class _RpcError(grpc.RpcError):
        def details(self) -> str:
            return "UNAVAILABLE"

    class _Stub:
        def QueryTransportWindow(self, *_args: object, **_kwargs: object) -> Any:
            raise _RpcError()

    monkeypatch.setattr(runner.grpc, "insecure_channel", lambda *_: _FakeChannel())
    monkeypatch.setattr(
        runner.grpc, "channel_ready_future", lambda *_: _ReadyFuture()
    )
    monkeypatch.setattr(
        runner.global_store_pb2_grpc,
        "ClusterRuntimeServiceStub",
        lambda *_: _Stub(),
    )

    now = datetime.now(timezone.utc)
    probe = runner.query_transport_group_probe(
        gs_addr="127.0.0.1:50051",
        group_mode="tp_version",
        group_kind="tp_version",
        started_at_utc=now - timedelta(minutes=1),
        finished_at_utc=now,
    )
    reasons = runner.evaluate_group_probe_gate_failures(probe=probe, mode="tp_version")
    assert probe["error"] is not None
    assert any(str(reason).startswith("group_probe_error:") for reason in reasons)


def test_validate_daemon_memfd_required_rejects_disabled_config(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    cfg = tmp_path / "daemon.yaml"
    cfg.write_text(
        "engine:\n  cpu_shared_memory:\n    enabled: false\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="daemon memfd preflight failed"):
        runner._validate_daemon_memfd_required(
            process_id="receiver-proc",
            role="receiver",
            daemon_config_path=str(cfg),
        )


def test_validate_daemon_memfd_required_accepts_enabled_config(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    cfg = tmp_path / "daemon.yaml"
    cfg.write_text(
        "engine:\n  cpu_shared_memory:\n    enabled: true\n",
        encoding="utf-8",
    )
    report = runner._validate_daemon_memfd_required(
        process_id="publisher-proc",
        role="publisher",
        daemon_config_path=str(cfg),
    )
    assert report["required"] is True
    assert report["process_id"] == "publisher-proc"
    assert report["role"] == "publisher"
    assert report["checked"] is True
    assert report["safe"] is True
    assert report["cpu_shared_memory_enabled"] is True


def test_compute_transport_metrics_rejects_missing_bytes_on_completed_rows() -> None:
    runner = _load_runner_module()
    payload = {
        "error": None,
        "rows": [
            {
                "transport_id": "transport-missing-bytes",
                "replica_id": "replica-1",
                "created_at_epoch_s": 10.0,
                "completed_at_epoch_s": 11.0,
                "replica_memory_size_bytes": 0,
            }
        ],
    }

    metrics = runner.compute_transport_metrics(
        transport_rows_payload=payload,
        sample_interval_s=1.0,
        max_samples=128,
    )

    assert str(metrics.get("error", "")).startswith(
        "invalid transport metrics input: missing replica_memory_size_bytes"
    )
    assert metrics["invalid_completed_bytes_count"] == 1
    assert metrics["completed_transport_count"] == 0
    assert (
        metrics["per_transport_records"][0]["bytes_source"]
        == "replica_memory_size_bytes"
    )
    assert metrics["per_transport_records"][0]["bytes"] == 0
    assert metrics["per_transport_records"][0]["included_in_sampling"] is False


def test_compute_transport_metrics_keeps_inflight_missing_bytes_out_of_sampling() -> None:
    runner = _load_runner_module()
    payload = {
        "error": None,
        "rows": [
            {
                "transport_id": "transport-inflight",
                "replica_id": "replica-1",
                "created_at_epoch_s": 10.0,
                "completed_at_epoch_s": 0.0,
                "replica_memory_size_bytes": 0,
            },
            {
                "transport_id": "transport-completed",
                "replica_id": "replica-2",
                "created_at_epoch_s": 20.0,
                "completed_at_epoch_s": 22.0,
                "replica_memory_size_bytes": 2 * 1024 * 1024 * 1024,
            },
        ],
    }

    metrics = runner.compute_transport_metrics(
        transport_rows_payload=payload,
        sample_interval_s=1.0,
        max_samples=128,
    )

    assert metrics["error"] is None
    assert metrics["invalid_completed_bytes_count"] == 0
    assert metrics["completed_transport_count"] == 1
    assert metrics["per_transport_records"][0]["included_in_sampling"] is False
    assert metrics["per_transport_records"][1]["included_in_sampling"] is True
    assert metrics["per_transport_records"][1]["throughput_gib_s"] == pytest.approx(
        1.0
    )
