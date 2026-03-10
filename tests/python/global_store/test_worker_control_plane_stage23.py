#  Copyright (c) 2026, TensorCast Team.

"""Stage-2/3 execution tests for worker control-plane serialization semantics."""

from __future__ import annotations

from dataclasses import dataclass

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import get_config, set_config
from tensorcast.global_store.models import Worker
from tensorcast.global_store.services.worker_service import WorkerService
from tensorcast.proto.global_store.v1 import global_store_pb2


def _ensure_config() -> None:
    try:
        get_config()
    except RuntimeError:
        set_config(GlobalStoreConfig())


def test_stage2_worker_ops_share_daemon_lane(servicer, test_context, monkeypatch):
    reducer = servicer.worker_control_reducer
    captured: list[tuple[str, str]] = []
    original_submit = reducer.submit

    def _wrapped(*, worker_key, kind, operation, timeout_s=None):
        captured.append((kind, worker_key))
        return original_submit(
            worker_key=worker_key,
            kind=kind,
            operation=operation,
            timeout_s=timeout_s,
        )

    monkeypatch.setattr(reducer, "submit", _wrapped)

    daemon_id = "daemon-stage23-key"
    register_response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            node_id="stage23-node",
            daemon_id=daemon_id,
            node_address="192.168.4.10",
            grpc_port=9051,
            p2p_port=9052,
            mem_pool_total_size=8 * 1024 * 1024 * 1024,
            mem_pool_available_size=8 * 1024 * 1024 * 1024,
        ),
        test_context,
    )
    assert register_response.status == global_store_pb2.Status.STATUS_OK

    heartbeat_response = servicer.WorkerHeartbeat(
        global_store_pb2.WorkerHeartbeatRequest(
            worker_id=register_response.worker_id,
            mem_pool_available_size=7 * 1024 * 1024 * 1024,
            accepting_new_requests=True,
            state_version=1,
        ),
        test_context,
    )
    assert heartbeat_response.status == global_store_pb2.Status.STATUS_OK

    reconcile_response = servicer.ReconcileWorkerState(
        global_store_pb2.ReconcileWorkerStateRequest(
            worker_id=register_response.worker_id,
            daemon_id=daemon_id,
            generation=int(register_response.reconcile_generation or 1),
            request_seq=1,
            request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
        ),
        test_context,
    )
    assert reconcile_response.result_kind in (
        global_store_pb2.RECONCILE_RESULT_KIND_APPLIED,
        global_store_pb2.RECONCILE_RESULT_KIND_NOOP,
    )

    unregister_response = servicer.UnregisterWorker(
        global_store_pb2.UnregisterWorkerRequest(
            worker_id=register_response.worker_id,
            is_graceful_shutdown=True,
            client_request_id="stage23-unregister",
        ),
        test_context,
    )
    assert unregister_response.status == global_store_pb2.Status.STATUS_OK

    expected_key = f"daemon:{daemon_id}"
    first_key_by_kind: dict[str, str] = {}
    for kind, worker_key in captured:
        first_key_by_kind.setdefault(kind, worker_key)

    assert first_key_by_kind["register"] == expected_key
    assert "heartbeat" not in first_key_by_kind
    assert first_key_by_kind["reconcile"] == expected_key
    assert first_key_by_kind["unregister"] == expected_key


@dataclass
class _CaptureReducer:
    calls: list[tuple[str, str]]

    def submit(self, *, worker_key, kind, operation, timeout_s=None):
        del timeout_s
        self.calls.append((kind, worker_key))
        return operation()


def test_stage2_cleanup_submits_per_worker_intent(repositories):
    _ensure_config()
    reducer = _CaptureReducer(calls=[])
    worker_service = WorkerService(
        repositories["worker"],
        repositories["replica"],
        control_reducer=reducer,
    )

    workers = [
        Worker(
            worker_id="stage23-cleanup-worker-a",
            daemon_id="stage23-cleanup-daemon-a",
            node_id="node-a",
            node_address="10.1.0.1",
            grpc_port=5101,
            p2p_port=6101,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        ),
        Worker(
            worker_id="stage23-cleanup-worker-b",
            daemon_id="stage23-cleanup-daemon-b",
            node_id="node-b",
            node_address="10.1.0.2",
            grpc_port=5102,
            p2p_port=6102,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        ),
    ]
    for worker in workers:
        repositories["worker"].create_or_update(worker)

    cursor = repositories["worker"].get_cursor()
    cursor.execute(
        """
        UPDATE worker_liveness
        SET last_heartbeat = now() - INTERVAL '10 minutes'
        WHERE worker_id IN (?, ?)
        """,
        [workers[0].worker_id, workers[1].worker_id],
    )

    cleaned = worker_service.cleanup_inactive_workers()
    assert set(cleaned) == {workers[0].worker_id, workers[1].worker_id}

    maintenance_keys = [
        worker_key for kind, worker_key in reducer.calls if kind == "maintenance"
    ]
    assert set(maintenance_keys) == {
        "daemon:stage23-cleanup-daemon-a",
        "daemon:stage23-cleanup-daemon-b",
    }
    assert "__maintenance_workers__" not in maintenance_keys


def test_stage2_heartbeat_write_throttle_reduces_duplicate_persists(
    servicer, test_context, monkeypatch
):
    daemon_id = "daemon-stage23-heartbeat-throttle"
    register_response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            node_id="stage23-heartbeat-node",
            daemon_id=daemon_id,
            node_address="192.168.4.20",
            grpc_port=9151,
            p2p_port=9152,
            mem_pool_total_size=8 * 1024 * 1024 * 1024,
            mem_pool_available_size=8 * 1024 * 1024 * 1024,
        ),
        test_context,
    )
    assert register_response.status == global_store_pb2.Status.STATUS_OK

    repository = servicer.worker_service.worker_repository
    original_batch_update_heartbeats = repository.batch_update_heartbeats
    batch_calls = {"count": 0, "rows": 0}

    def _wrapped_batch_update_heartbeats(
        updates: list[tuple[str, int, bool, int | None]],
    ) -> int:
        batch_calls["count"] += 1
        batch_calls["rows"] += len(updates)
        return original_batch_update_heartbeats(updates)

    monkeypatch.setattr(
        repository,
        "batch_update_heartbeats",
        _wrapped_batch_update_heartbeats,
    )

    for _ in range(8):
        heartbeat_response = servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=register_response.worker_id,
                mem_pool_available_size=7 * 1024 * 1024 * 1024,
                accepting_new_requests=True,
                state_version=1,
                daemon_id=daemon_id,
            ),
            test_context,
        )
        assert heartbeat_response.status == global_store_pb2.Status.STATUS_OK

    servicer.worker_service.flush_heartbeats()
    # Multiple identical heartbeats in a short window should collapse into a
    # smaller number of buffered rows flushed to storage.
    assert batch_calls["count"] >= 1
    assert batch_calls["rows"] < 8


def test_stage2_cleanup_flushes_pending_heartbeats_before_timeout(repositories):
    _ensure_config()
    worker_service = WorkerService(
        repositories["worker"],
        repositories["replica"],
    )
    worker = Worker(
        worker_id="stage23-cleanup-flush-worker",
        daemon_id="stage23-cleanup-flush-daemon",
        node_id="stage23-cleanup-flush-node",
        node_address="10.2.0.1",
        grpc_port=5201,
        p2p_port=6201,
        mem_pool_total_size=4096,
        mem_pool_available_size=4096,
    )
    repositories["worker"].create_or_update(worker)

    cursor = repositories["worker"].get_cursor()
    cursor.execute(
        """
        UPDATE worker_liveness
        SET last_heartbeat = now() - INTERVAL '10 minutes'
        WHERE worker_id = ?
        """,
        [worker.worker_id],
    )

    assert (
        worker_service.heartbeat(
            worker_id=worker.worker_id,
            mem_pool_available_size=3072,
            accepting_new_requests=True,
            capability_flags=0,
        )
        is True
    )
    cleaned = worker_service.cleanup_inactive_workers()
    assert cleaned == []
    worker_service.close()
