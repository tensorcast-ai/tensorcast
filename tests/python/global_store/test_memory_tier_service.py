#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

"""Tests for memory tier telemetry and lease coordination."""

import duckdb
import grpc
import pytest

from tensorcast.global_store import metrics
from tensorcast.global_store.db_utils import init_db
from tensorcast.global_store.memory_tier_grpc_service import MemoryTierGrpcServicer
from tensorcast.global_store.models.memory_tier import ChunkRange, MemoryTierSnapshot
from tensorcast.global_store.models.worker import Worker
from tensorcast.global_store.repositories.memory_tier_lease_repository import MemoryTierLeaseRepository
from tensorcast.global_store.repositories.memory_tier_snapshot_repository import MemoryTierSnapshotRepository
from tensorcast.global_store.repositories.worker_repository import WorkerRepository
from tensorcast.global_store.services.memory_tier_service import MemoryTierService
from tensorcast.proto.memory_tier.v1 import memory_tier_pb2
from tests.python.global_store.conftest import MockContext


def _make_service(snapshot_retention_ns: int = 0, snapshot_max_rows: int = 0):
    conn = duckdb.connect()
    init_db(conn)
    worker_repo = WorkerRepository(conn)
    snapshot_repo = MemoryTierSnapshotRepository(conn)
    lease_repo = MemoryTierLeaseRepository(conn)
    service = MemoryTierService(
        snapshot_repository=snapshot_repo,
        lease_repository=lease_repo,
        snapshot_retention_ns=snapshot_retention_ns,
        snapshot_max_rows=snapshot_max_rows,
    )
    return service, worker_repo, snapshot_repo, lease_repo


def test_publish_status_updates_worker_and_metrics() -> None:
    service, worker_repo, snapshot_repo, _ = _make_service(snapshot_retention_ns=50_000, snapshot_max_rows=2)
    worker_repo.create(
        Worker(
            worker_id="w1",
            node_id="n1",
            node_address="127.0.0.1",
            grpc_port=50051,
            p2p_port=7000,
            mem_pool_total_size=1024,
            mem_pool_available_size=512,
        )
    )

    service.publish_status(
        MemoryTierSnapshot(
            node_id="n1",
            epoch_ns=100,
            stable_total_bytes=128,
            stable_used_bytes=64,
            preemptible_total_bytes=256,
            preemptible_marked_bytes=32,
            faults_per_sec=0.5,
            rehydrate_p99_ns=1234,
            enable_preemptible=True,
            memory_tier_config_json='{"cfg":1}',
        )
    )
    service.publish_status(
        MemoryTierSnapshot(
            node_id="n1",
            epoch_ns=200,
            stable_total_bytes=256,
            stable_used_bytes=128,
            preemptible_total_bytes=512,
            preemptible_marked_bytes=64,
            faults_per_sec=1.5,
            rehydrate_p99_ns=4321,
            enable_preemptible=False,
            memory_tier_config_json='{"cfg":2}',
        )
    )
    # Third sample should prune the first one due to max_rows=2.
    service.publish_status(
        MemoryTierSnapshot(
            node_id="n1",
            epoch_ns=300,
            stable_total_bytes=512,
            stable_used_bytes=256,
            preemptible_total_bytes=1024,
            preemptible_marked_bytes=128,
            faults_per_sec=2.5,
            rehydrate_p99_ns=9999,
            enable_preemptible=True,
            memory_tier_config_json='{"cfg":3}',
        )
    )

    worker = worker_repo.find_by_id("w1")
    assert worker is not None
    assert worker.memory_tier_state is not None
    assert worker.memory_tier_state.stable_total_bytes == 512
    assert worker.memory_tier_state.preemptible_total_bytes == 1024
    assert worker.memory_tier_state.memory_tier_config_json == '{"cfg":3}'
    assert worker.memory_tier_state.enable_preemptible is True

    snapshot_count = snapshot_repo.get_cursor().execute(  # pyright: ignore[reportOptionalSubscript]
        "SELECT COUNT(*) FROM memory_tier_snapshots WHERE node_id = 'n1'"
    ).fetchone()[0]
    assert snapshot_count == 2

    # Metrics reflect the latest snapshot
    assert metrics.MEMORY_TIER_STABLE_BYTES.labels(node="n1", state="total")._value.get() == 512
    assert metrics.MEMORY_TIER_STABLE_BYTES.labels(node="n1", state="used")._value.get() == 256
    assert metrics.MEMORY_TIER_PREEMPTIBLE_BYTES.labels(node="n1", state="marked")._value.get() == 128
    assert metrics.MEMORY_TIER_FAULTS_PER_SEC.labels(node="n1")._value.get() == 2.5
    assert metrics.MEMORY_TIER_REHYDRATE_P99_NS.labels(node="n1")._value.get() == 9999
    assert metrics.MEMORY_TIER_ENABLE_PREEMPTIBLE.labels(node="n1", enable_preemptible="true")._value.get() == 1.0
    assert metrics.MEMORY_TIER_ENABLE_PREEMPTIBLE.labels(node="n1", enable_preemptible="false")._value.get() == 0.0


def test_acknowledge_requires_matching_artifact_and_updates_state() -> None:
    service, _, _, _ = _make_service()
    lease = service.request_lease(
        node_id="n1",
        kind="stable",
        artifact_id="artifactA",
        chunk_range=ChunkRange(start=0, count=2),
        chunk_ids=[0, 1],
        ledger_version=1,
        bytes_count=256,
        workload_id="wl-1",
        request_id="req-1",
    )

    acquired = service.acknowledge(
        lease_id=lease.lease_id,
        action="acquired",
        artifact_id="artifactA",
        chunk_ids=[0, 1],
        ledger_version=5,
        chunk_range=ChunkRange(start=0, count=2),
        bytes_count=512,
        request_id="ack-1",
        ack_epoch_ns=123,
    )
    assert acquired is not None
    assert acquired.state == "active"
    assert acquired.chunk_ids == [0, 1]
    assert acquired.ledger_version == 5
    assert acquired.bytes == 512

    released = service.acknowledge(
        lease_id=lease.lease_id,
        action="released",
        artifact_id="artifactA",
        chunk_ids=[0, 1],
        chunk_range=ChunkRange(start=0, count=2),
        ledger_version=6,
        bytes_count=512,
        request_id="ack-2",
        ack_epoch_ns=456,
    )
    assert released is not None
    assert released.state == "expired"
    assert released.chunk_ids == [0, 1]
    assert released.ledger_version == 6

    mismatch = service.acknowledge(
        lease_id=lease.lease_id,
        action="acquired",
        artifact_id="artifactB",
        chunk_ids=[0],
        ledger_version=6,
        chunk_range=ChunkRange(start=0, count=1),
        bytes_count=64,
        request_id="ack-3",
        ack_epoch_ns=None,
    )
    assert mismatch is None


def test_grpc_acknowledge_validates_required_fields() -> None:
    service, _, _, _ = _make_service()
    servicer = MemoryTierGrpcServicer(service)
    ctx = MockContext()

    with pytest.raises(Exception):
        servicer.AcknowledgeMemoryTierLease(
            memory_tier_pb2.AcknowledgeMemoryTierLeaseRequest(
                lease_id="l1",
                node_id="n1",
                action=memory_tier_pb2.LEASE_ACK_ACTION_ACQUIRED,
            ),
            ctx,
        )
    assert ctx._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    ctx2 = MockContext()
    with pytest.raises(Exception):
        servicer.AcknowledgeMemoryTierLease(
            memory_tier_pb2.AcknowledgeMemoryTierLeaseRequest(
                lease_id="l1",
                node_id="n1",
                action=memory_tier_pb2.LEASE_ACK_ACTION_ACQUIRED,
                artifact_id="artifactA",
            ),
            ctx2,
        )
    assert ctx2._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    ctx3 = MockContext()
    with pytest.raises(Exception):
        servicer.AcknowledgeMemoryTierLease(
            memory_tier_pb2.AcknowledgeMemoryTierLeaseRequest(
                lease_id="l1",
                node_id="n1",
                action=memory_tier_pb2.LEASE_ACK_ACTION_RELEASED,
                artifact_id="artifactA",
            ),
            ctx3,
        )
    assert ctx3._abort_code == grpc.StatusCode.INVALID_ARGUMENT


def test_request_lease_idempotent_and_filters_states() -> None:
    service, _, _, _ = _make_service()
    chunk_range = ChunkRange(start=0, count=2)

    lease = service.request_lease(
        node_id="n1",
        kind="stable",
        artifact_id="artifactA",
        chunk_range=chunk_range,
        chunk_ids=[],
        ledger_version=1,
        bytes_count=512,
        workload_id="wl-1",
        request_id="req-1",
    )

    duplicate = service.request_lease(
        node_id="n1",
        kind="stable",
        artifact_id="artifactA",
        chunk_range=ChunkRange(start=5, count=1),
        chunk_ids=[5],
        ledger_version=2,
        bytes_count=128,
        workload_id="wl-1",
        request_id="req-1",
    )

    assert duplicate.lease_id == lease.lease_id
    assert duplicate.state == "pending"

    active = service.acknowledge(
        lease_id=lease.lease_id,
        action="acquired",
        artifact_id="artifactA",
        chunk_ids=[0, 1],
        ledger_version=10,
        chunk_range=chunk_range,
        bytes_count=1024,
        request_id="ack-1",
        ack_epoch_ns=111,
    )
    assert active is not None
    assert active.state == "active"

    revoking = service.revoke(lease.lease_id)
    assert revoking is not None
    assert revoking.state == "revoking"

    outstanding = service.list_outstanding("n1")
    assert any(candidate.lease_id == lease.lease_id for candidate in outstanding)

    only_active = service.list_outstanding("n1", states=["active"])
    assert all(candidate.state == "active" for candidate in only_active)
    assert all(candidate.lease_id != lease.lease_id for candidate in only_active)

    expired = service.acknowledge(
        lease_id=lease.lease_id,
        action="released",
        artifact_id="artifactA",
        chunk_ids=[0, 1],
        chunk_range=chunk_range,
        ledger_version=11,
        bytes_count=1024,
        request_id="ack-2",
        ack_epoch_ns=222,
    )
    assert expired is not None
    assert expired.state == "expired"

    after_expired = service.list_outstanding("n1")
    assert not after_expired


def test_grpc_request_lease_validates_and_reuses_request_id() -> None:
    service, _, _, _ = _make_service()
    servicer = MemoryTierGrpcServicer(service)

    ctx_missing_all = MockContext()
    with pytest.raises(Exception):
        servicer.RequestMemoryTierLease(memory_tier_pb2.RequestMemoryTierLeaseRequest(), ctx_missing_all)
    assert ctx_missing_all._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    ctx_missing_artifact = MockContext()
    with pytest.raises(Exception):
        servicer.RequestMemoryTierLease(
            memory_tier_pb2.RequestMemoryTierLeaseRequest(node_id="n1"), ctx_missing_artifact
        )
    assert ctx_missing_artifact._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    # Proto fields are unsigned, so constructing the request raises ValueError before the RPC layer sees it.
    with pytest.raises(ValueError):
        memory_tier_pb2.RequestMemoryTierLeaseRequest(
            node_id="n1",
            artifact_id="artifactA",
            chunk_range=memory_tier_pb2.ChunkRange(start=-1, count=0),
        )

    ctx_empty_range = MockContext()
    with pytest.raises(Exception):
        servicer.RequestMemoryTierLease(
            memory_tier_pb2.RequestMemoryTierLeaseRequest(
                node_id="n1",
                artifact_id="artifactA",
                chunk_range=memory_tier_pb2.ChunkRange(start=0, count=0),
            ),
            ctx_empty_range,
        )
    assert ctx_empty_range._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    ctx_ok = MockContext()
    resp1 = servicer.RequestMemoryTierLease(
        memory_tier_pb2.RequestMemoryTierLeaseRequest(
            node_id="n1",
            artifact_id="artifactA",
            chunk_range=memory_tier_pb2.ChunkRange(start=0, count=2),
            request_id="req-lease-1",
            bytes=256,
        ),
        ctx_ok,
    )
    lease_id = resp1.lease.lease_id

    ctx_repeat = MockContext()
    resp2 = servicer.RequestMemoryTierLease(
        memory_tier_pb2.RequestMemoryTierLeaseRequest(
            node_id="n1",
            artifact_id="artifactA",
            chunk_range=memory_tier_pb2.ChunkRange(start=5, count=3),
            request_id="req-lease-1",
            bytes=128,
        ),
        ctx_repeat,
    )

    assert resp2.lease.lease_id == lease_id
    assert resp2.lease.state == memory_tier_pb2.LEASE_STATE_PENDING


def test_grpc_memory_tier_lifecycle_filters_states() -> None:
    service, _, _, _ = _make_service()
    servicer = MemoryTierGrpcServicer(service)

    ctx_create = MockContext()
    lease_resp = servicer.RequestMemoryTierLease(
        memory_tier_pb2.RequestMemoryTierLeaseRequest(
            node_id="n1",
            artifact_id="artifactA",
            chunk_range=memory_tier_pb2.ChunkRange(start=0, count=2),
            bytes=512,
            request_id="req-cycle-1",
        ),
        ctx_create,
    )
    lease_id = lease_resp.lease.lease_id

    ctx_acquire = MockContext()
    acquire_resp = servicer.AcknowledgeMemoryTierLease(
        memory_tier_pb2.AcknowledgeMemoryTierLeaseRequest(
            lease_id=lease_id,
            node_id="n1",
            artifact_id="artifactA",
            action=memory_tier_pb2.LEASE_ACK_ACTION_ACQUIRED,
            chunk_ids=[0, 1],
            chunk_range=memory_tier_pb2.ChunkRange(start=0, count=2),
            ledger_version=1,
            bytes=512,
        ),
        ctx_acquire,
    )
    assert acquire_resp.lease.state == memory_tier_pb2.LEASE_STATE_ACTIVE

    ctx_revoke = MockContext()
    revoke_resp = servicer.RevokeMemoryTierLease(
        memory_tier_pb2.RevokeMemoryTierLeaseRequest(lease_id=lease_id), ctx_revoke
    )
    assert revoke_resp.lease.state == memory_tier_pb2.LEASE_STATE_REVOKING

    ctx_filter_pending = MockContext()
    filtered_pending = servicer.ListOutstandingLeases(
        memory_tier_pb2.ListOutstandingLeasesRequest(
            node_id="n1", states=[memory_tier_pb2.LEASE_STATE_PENDING]
        ),
        ctx_filter_pending,
    )
    assert all(lease.state == memory_tier_pb2.LEASE_STATE_PENDING for lease in filtered_pending.leases)
    assert all(lease_resp.lease_id != lease_id for lease_resp in filtered_pending.leases)

    ctx_filter_revoking = MockContext()
    filtered_revoking = servicer.ListOutstandingLeases(
        memory_tier_pb2.ListOutstandingLeasesRequest(
            node_id="n1", states=[memory_tier_pb2.LEASE_STATE_REVOKING]
        ),
        ctx_filter_revoking,
    )
    assert any(lease.lease_id == lease_id for lease in filtered_revoking.leases)

    ctx_release = MockContext()
    released_resp = servicer.AcknowledgeMemoryTierLease(
        memory_tier_pb2.AcknowledgeMemoryTierLeaseRequest(
            lease_id=lease_id,
            node_id="n1",
            artifact_id="artifactA",
            action=memory_tier_pb2.LEASE_ACK_ACTION_RELEASED,
            chunk_ids=[0, 1],
            chunk_range=memory_tier_pb2.ChunkRange(start=0, count=2),
            ledger_version=2,
            bytes=512,
        ),
        ctx_release,
    )
    assert released_resp.lease.state == memory_tier_pb2.LEASE_STATE_EXPIRED

    ctx_after_expired = MockContext()
    after_expired = servicer.ListOutstandingLeases(
        memory_tier_pb2.ListOutstandingLeasesRequest(node_id="n1"), ctx_after_expired
    )
    assert not after_expired.leases
