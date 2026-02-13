#  Copyright (c) 2026, TensorCast Team.

"""Stage-1 execution tests for control-plane conflict and retry observability."""

from __future__ import annotations

import threading
import time
from collections import Counter

import pytest

from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.proto.global_store.v1 import global_store_pb2
from tests.python.global_store.conftest import MockContext


def _sum_counter(counter, **match_labels: str) -> float:
    total = 0.0
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name != metric.name + "_total":
                continue
            labels = sample.labels or {}
            if all(labels.get(key) == value for key, value in match_labels.items()):
                total += float(sample.value)
    return total


def _register_stage1_worker(
    servicer: GlobalStoreServicer,
    *,
    daemon_id: str,
    node_id: str,
    grpc_port: int,
    p2p_port: int,
) -> global_store_pb2.RegisterWorkerResponse:
    response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id=daemon_id,
            node_id=node_id,
            node_address="10.42.0.10",
            grpc_port=grpc_port,
            p2p_port=p2p_port,
            mem_pool_total_size=8 * 1024 * 1024 * 1024,
            mem_pool_available_size=8 * 1024 * 1024 * 1024,
        ),
        MockContext(),
    )
    assert response.status == global_store_pb2.Status.STATUS_OK
    return response


@pytest.fixture
def stage1_servicer(tmp_path) -> GlobalStoreServicer:
    cfg_path = tmp_path / "global_store_stage1.yaml"
    cfg_path.write_text(
        """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
  max_workers: 16
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "1s"
  control_reducer:
    shard_count: 4
    queue_capacity: 2048
    coalesce_window: "0.01s"
observability:
  logging:
    level: LOG_LEVEL_INFO
  otel:
    enabled: false
        """,
        encoding="utf-8",
    )
    set_config(GlobalStoreConfig.from_file(str(cfg_path)))
    return GlobalStoreServicer()


def test_stage1_restart_heartbeat_reconcile_stress_baseline(stage1_servicer):
    daemon_id = "daemon-stage1-worker"
    node_id = "stage1-node"
    register_response = _register_stage1_worker(
        stage1_servicer,
        daemon_id=daemon_id,
        node_id=node_id,
        grpc_port=50052,
        p2p_port=65090,
    )

    shared = {
        "worker_id": register_response.worker_id,
        "generation": int(register_response.reconcile_generation or 1),
        "request_seq": 0,
        "paused": False,
    }
    shared_mu = threading.Lock()
    stop_event = threading.Event()

    heartbeat_statuses: Counter[int] = Counter()
    reconcile_results: Counter[int] = Counter()

    retry_later_result_name = global_store_pb2.ReconcileResultKind.Name(
        global_store_pb2.RECONCILE_RESULT_KIND_RETRY_LATER
    )
    gap_scope = "reconcile_request_gap"

    conflict_before = _sum_counter(
        gs_metrics.CONTROL_PLANE_CONFLICT_COUNTER,
        scope=gap_scope,
    )
    retry_later_before = _sum_counter(
        gs_metrics.RECONCILE_RESULT_COUNTER,
        result_kind=retry_later_result_name,
    )
    retry_later_reason_before = _sum_counter(
        gs_metrics.RECONCILE_RETRY_LATER_COUNTER,
        reason="request_seq_gap",
    )
    reducer_submitted_before = _sum_counter(
        gs_metrics.WORKER_CONTROL_REDUCER_INTENT_COUNTER,
        kind="heartbeat",
        result="submitted",
    )

    def heartbeat_loop() -> None:
        while not stop_event.is_set():
            with shared_mu:
                paused = bool(shared["paused"])
                worker_id = shared["worker_id"]
            if paused:
                time.sleep(0.002)
                continue
            response = stage1_servicer.WorkerHeartbeat(
                global_store_pb2.WorkerHeartbeatRequest(
                    worker_id=worker_id,
                    mem_pool_available_size=7 * 1024 * 1024 * 1024,
                    accepting_new_requests=True,
                    state_version=1,
                    state_checksum="stage1-checksum",
                    daemon_id=daemon_id,
                ),
                MockContext(),
            )
            heartbeat_statuses[int(response.status)] += 1
            time.sleep(0.003)

    def reconcile_loop() -> None:
        tick = 0
        while not stop_event.is_set():
            with shared_mu:
                paused = bool(shared["paused"])
                worker_id = shared["worker_id"]
                generation = int(shared["generation"])
                # Intentionally inject request-seq gaps to force deterministic RETRY_LATER.
                next_seq = int(shared["request_seq"]) + (2 if tick % 5 == 0 else 1)
                shared["request_seq"] = next_seq
            if paused:
                time.sleep(0.002)
                continue
            response = stage1_servicer.ReconcileWorkerState(
                global_store_pb2.ReconcileWorkerStateRequest(
                    worker_id=worker_id,
                    daemon_id=daemon_id,
                    generation=generation,
                    request_seq=next_seq,
                    request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
                ),
                MockContext(),
            )
            reconcile_results[int(response.result_kind)] += 1
            tick += 1
            time.sleep(0.003)

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    reconcile_thread = threading.Thread(target=reconcile_loop, daemon=True)
    heartbeat_thread.start()
    reconcile_thread.start()

    try:
        for cycle in range(12):
            with shared_mu:
                shared["paused"] = True
                worker_id = str(shared["worker_id"])
            stage1_servicer.UnregisterWorker(
                global_store_pb2.UnregisterWorkerRequest(
                    worker_id=worker_id,
                    is_graceful_shutdown=False,
                    client_request_id=f"stage1-unregister-{cycle}",
                ),
                MockContext(),
            )
            reg = _register_stage1_worker(
                stage1_servicer,
                daemon_id=daemon_id,
                node_id=node_id,
                grpc_port=50052,
                p2p_port=65090,
            )
            with shared_mu:
                shared["worker_id"] = reg.worker_id
                shared["generation"] = int(reg.reconcile_generation or 1)
                shared["request_seq"] = 0
                shared["paused"] = False
            time.sleep(0.01)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=3.0)
        reconcile_thread.join(timeout=3.0)

    conflict_after = _sum_counter(
        gs_metrics.CONTROL_PLANE_CONFLICT_COUNTER,
        scope=gap_scope,
    )
    retry_later_after = _sum_counter(
        gs_metrics.RECONCILE_RESULT_COUNTER,
        result_kind=retry_later_result_name,
    )
    retry_later_reason_after = _sum_counter(
        gs_metrics.RECONCILE_RETRY_LATER_COUNTER,
        reason="request_seq_gap",
    )
    reducer_submitted_after = _sum_counter(
        gs_metrics.WORKER_CONTROL_REDUCER_INTENT_COUNTER,
        kind="heartbeat",
        result="submitted",
    )

    conflict_delta = conflict_after - conflict_before
    retry_later_delta = retry_later_after - retry_later_before
    retry_later_reason_delta = retry_later_reason_after - retry_later_reason_before
    reducer_submitted_delta = reducer_submitted_after - reducer_submitted_before

    print(
        "STAGE1_BASELINE "
        f"conflict_gap_delta={int(conflict_delta)} "
        f"retry_later_delta={int(retry_later_delta)} "
        f"retry_later_reason_delta={int(retry_later_reason_delta)} "
        f"heartbeat_submitted_delta={int(reducer_submitted_delta)} "
        f"heartbeat_status_samples={dict(heartbeat_statuses)} "
        f"reconcile_result_samples={dict(reconcile_results)}"
    )

    assert heartbeat_statuses
    assert reconcile_results
    assert conflict_delta > 0
    assert retry_later_delta > 0
    assert retry_later_reason_delta > 0
    assert reducer_submitted_delta > 0
    assert reconcile_results[global_store_pb2.RECONCILE_RESULT_KIND_RETRY_LATER] > 0


def test_stage1_reducer_failure_log_has_worker_context(servicer, monkeypatch):
    captured_messages: list[str] = []

    def _capture_exception(message: str, *args, **kwargs) -> None:
        del kwargs
        rendered = message % args if args else message
        captured_messages.append(rendered)

    monkeypatch.setattr(
        servicer.worker_control_reducer._logger,
        "exception",
        _capture_exception,
    )

    def _fail_operation() -> bool:
        raise RuntimeError("stage1-induced-failure")

    with pytest.raises(RuntimeError, match="stage1-induced-failure"):
        servicer.worker_control_reducer.submit(
            worker_key="worker:stage1-log-check",
            kind="reconcile",
            operation=_fail_operation,
        )

    assert captured_messages
    assert any("worker:stage1-log-check" in msg for msg in captured_messages)
    assert any("shard=" in msg for msg in captured_messages)
