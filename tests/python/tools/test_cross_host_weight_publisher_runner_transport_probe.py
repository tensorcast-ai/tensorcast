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


def test_query_transport_rows_channel_ready_timeout_sets_error(
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
    payload = runner.query_transport_rows(
        gs_addr="127.0.0.1:50051",
        started_at_utc=now - timedelta(minutes=1),
        finished_at_utc=now,
    )
    assert payload["error"] is not None
    assert "channel ready timeout" in str(payload["error"]).lower()


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


def test_estimate_keep_last_floor_for_tp_bind_scaleout() -> None:
    runner = _load_runner_module()
    args = runner.argparse.Namespace(
        keep_last=1,
        payload_mode="tp_ranked",
        tp_total_bytes=40 * 1024**3,
        receiver_apply_mode="tp_bind_into_swap",
        allow_receiver_skips=False,
        receiver_procs="r1,r2,r3",
        num_versions=10,
    )
    assert runner.estimate_keep_last_floor(args) == 2


def test_estimate_keep_last_floor_keeps_single_when_skip_allowed() -> None:
    runner = _load_runner_module()
    args = runner.argparse.Namespace(
        keep_last=1,
        payload_mode="tp_ranked",
        tp_total_bytes=40 * 1024**3,
        receiver_apply_mode="tp_bind_into_swap",
        allow_receiver_skips=True,
        receiver_procs="r1,r2,r3",
        num_versions=10,
    )
    assert runner.estimate_keep_last_floor(args) == 1


def test_estimate_keep_last_stable_cap_from_daemon_config(tmp_path: Path) -> None:
    runner = _load_runner_module()
    cfg = tmp_path / "daemon.yaml"
    cfg.write_text(
        "engine:\n  memory_tiers:\n    stable_bytes: 40gb\n",
        encoding="utf-8",
    )
    args = runner.argparse.Namespace(
        payload_mode="tp_ranked",
        tp_total_bytes=40 * 1024**3,
        publisher_daemon_config=str(cfg),
        daemon_config=str(cfg),
    )
    assert runner.estimate_keep_last_stable_cap(args) == 1


def test_estimate_keep_last_stable_cap_disabled_for_non_tp_payload() -> None:
    runner = _load_runner_module()
    args = runner.argparse.Namespace(
        payload_mode="probe",
        tp_total_bytes=40 * 1024**3,
        publisher_daemon_config="unused",
        daemon_config="unused",
    )
    assert runner.estimate_keep_last_stable_cap(args) is None


def test_estimate_publish_to_apply_floor_for_tp_bind_scaleout() -> None:
    runner = _load_runner_module()
    args = runner.argparse.Namespace(
        max_publish_to_apply_s=30.0,
        receiver_apply_mode="tp_bind_into_swap",
        payload_mode="tp_ranked",
        tp_total_bytes=40 * 1024**3,
        tp_world_size=4,
        receiver_procs="r1,r2,r3,r4,r5,r6,r7",
        publish_interval_s=3.0,
        transport_group_mode="tp_version",
    )
    assert runner.estimate_publish_to_apply_floor_sec(args) >= 50.0


def test_parse_receiver_skip_events_extracts_reason_and_version() -> None:
    runner = _load_runner_module()
    log_text = (
        "[receiver] skipped version=1 key=k1 artifact_id=a1 "
        "newer_version=2 reason=version_deregistered\n"
        "[receiver] skipped version=2 key=k2 artifact_id=a2 "
        "newer_version=n/a reason=group_contract_conflict\n"
    )
    events = runner.parse_receiver_skip_events(log_text)
    assert [event["version"] for event in events] == [1, 2]
    assert events[0]["reason"] == "version_deregistered"
    assert events[0]["newer_version"] == 2
    assert events[1]["reason"] == "group_contract_conflict"
    assert events[1]["newer_version"] is None


def test_assess_receiver_sequence_allows_accounted_missing_when_skip_enabled() -> None:
    runner = _load_runner_module()
    result = runner.assess_receiver_sequence(
        expected_versions=[1, 2, 3],
        actual_versions=[3],
        allow_receiver_skips=True,
        explicit_skipped_versions={1, 2},
    )
    assert result["is_failure"] is False
    assert result["accounted_missing_versions"] == [1, 2]
    assert result["unaccounted_missing_versions"] == []


def test_assess_receiver_sequence_rejects_unaccounted_missing_when_skip_enabled() -> None:
    runner = _load_runner_module()
    result = runner.assess_receiver_sequence(
        expected_versions=[1, 2, 3],
        actual_versions=[3],
        allow_receiver_skips=True,
        explicit_skipped_versions={1},
    )
    assert result["is_failure"] is True
    assert result["accounted_missing_versions"] == [1]
    assert result["unaccounted_missing_versions"] == [2]


def test_summarize_timeout_reasons_splits_wait_and_transport() -> None:
    runner = _load_runner_module()
    timeout_summary = runner.summarize_timeout_reasons(
        receiver_logs={
            "r1": "[receiver][wait] state=key_mapping_absent timed out",
            "r2": "client_rpc_retry_suppressed code=DEADLINE_EXCEEDED",
        },
        receiver_sequence_failures=[
            {"process_id": "r3", "reason": "timed out waiting remote file"},
        ],
    )
    waiting = timeout_summary["waiting_timeout_reason_counts"]
    transport = timeout_summary["transport_timeout_reason_counts"]
    assert waiting.get("queue_or_visibility_wait", 0) >= 2
    assert transport.get("deadline_exceeded", 0) >= 1


def test_evaluate_waiting_lease_renews_when_progress_advances() -> None:
    runner = _load_runner_module()
    lease_eval = runner.evaluate_waiting_lease(
        previous_progress_token=None,
        previous_progress_mono=None,
        now_mono=10.0,
        no_progress_limit_s=20.0,
        probe={
            "total_transports": 2,
            "requester_tagged_transports": 2,
            "grouped_transports": 2,
            "kind_matched_transports": 2,
            "group_contract_transports": 2,
        },
    )
    assert lease_eval["progressed"] is True
    assert lease_eval["waiting_timeout"] is False
    assert lease_eval["no_progress_elapsed_s"] == pytest.approx(0.0)


def test_evaluate_waiting_lease_times_out_only_on_no_progress() -> None:
    runner = _load_runner_module()
    first = runner.evaluate_waiting_lease(
        previous_progress_token=None,
        previous_progress_mono=None,
        now_mono=0.0,
        no_progress_limit_s=2.0,
        probe={"total_transports": 3},
    )
    second = runner.evaluate_waiting_lease(
        previous_progress_token=first["current_progress_token"],
        previous_progress_mono=first["effective_progress_mono"],
        now_mono=2.1,
        no_progress_limit_s=2.0,
        probe={"total_transports": 3},
    )
    assert second["progressed"] is False
    assert second["waiting_timeout"] is True
    assert second["waiting_timeout_reason"] == "waiting_no_progress"
    assert second["no_progress_elapsed_s"] == pytest.approx(2.1)


def test_run_transport_group_p0_guard_triggers_after_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    probes = [
        {
            "error": None,
            "window_has_transports": False,
            "requester_tagged_complete": False,
            "group_mode_consistent": False,
            "group_contract_consistent": False,
            "total_transports": 0,
            "requester_tagged_transports": 0,
            "grouped_transports": 0,
            "kind_matched_transports": 0,
            "group_contract_transports": 0,
        },
        {
            "error": None,
            "window_has_transports": False,
            "requester_tagged_complete": False,
            "group_mode_consistent": False,
            "group_contract_consistent": False,
            "total_transports": 1,
            "requester_tagged_transports": 1,
            "grouped_transports": 1,
            "kind_matched_transports": 1,
            "group_contract_transports": 1,
        },
        {
            "error": None,
            "window_has_transports": False,
            "requester_tagged_complete": False,
            "group_mode_consistent": False,
            "group_contract_consistent": False,
            "total_transports": 1,
            "requester_tagged_transports": 1,
            "grouped_transports": 1,
            "kind_matched_transports": 1,
            "group_contract_transports": 1,
        },
        {
            "error": None,
            "window_has_transports": False,
            "requester_tagged_complete": False,
            "group_mode_consistent": False,
            "group_contract_consistent": False,
            "total_transports": 1,
            "requester_tagged_transports": 1,
            "grouped_transports": 1,
            "kind_matched_transports": 1,
            "group_contract_transports": 1,
        },
    ]
    state = {"idx": 0, "mono": 0.0}

    def _fake_query_transport_group_probe(**_kwargs: Any) -> dict[str, Any]:
        idx = min(int(state["idx"]), len(probes) - 1)
        state["idx"] = idx + 1
        return dict(probes[idx])

    def _fake_monotonic() -> float:
        return float(state["mono"])

    def _fake_sleep(seconds: float) -> None:
        state["mono"] = float(state["mono"]) + float(seconds)

    monkeypatch.setattr(
        runner,
        "query_transport_group_probe",
        _fake_query_transport_group_probe,
    )
    monkeypatch.setattr(runner.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(runner.time, "sleep", _fake_sleep)

    result = runner.run_transport_group_p0_guard(
        enabled=True,
        mode="tp_version",
        group_kind="tp_version",
        gs_addr="127.0.0.1:50051",
        started_at_utc=datetime.now(timezone.utc) - timedelta(minutes=1),
        grace_s=2.0,
        poll_interval_s=1.0,
    )
    assert result["triggered"] is True
    assert result["waiting_timeout_reason"] == "waiting_no_progress"
    assert result["lease_renew_count"] >= 2
    assert result["max_no_progress_elapsed_s"] >= 2.0


def test_merge_timeout_analysis_with_waiting_guard_adds_waiting_reason() -> None:
    runner = _load_runner_module()
    merged = runner.merge_timeout_analysis_with_waiting_guard(
        timeout_analysis={
            "waiting_timeout_reason_counts": {"queue_or_visibility_wait": 2},
            "transport_timeout_reason_counts": {"deadline_exceeded": 1},
            "waiting_timeout_observed": True,
            "transport_timeout_observed": True,
        },
        p0_guard={
            "triggered": True,
            "waiting_timeout_reason": "waiting_no_progress",
            "lease_renew_count": 5,
            "no_progress_limit_s": 20.0,
            "max_no_progress_elapsed_s": 20.2,
            "last_no_progress_elapsed_s": 20.2,
            "last_progress_snapshot": {"total_transports": 16},
        },
    )
    waiting = merged["waiting_timeout_reason_counts"]
    assert waiting["queue_or_visibility_wait"] == 2
    assert waiting["waiting_no_progress"] == 1
    assert merged["waiting_lease"]["renew_count"] == 5
