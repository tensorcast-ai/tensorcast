#  Copyright (c) 2025, StepCast Team.

from __future__ import annotations

"""Daemon crash recovery test – ensures leaked *current_requests* are cleaned.

This corresponds to the design document's recommendation to verify that the
background cleanup (or explicit ``cleanup_expired_transports`` call) correctly
releases counters when a Store-Daemon crashes before completing a transport.
"""

import time

from scstore.proto import global_store_pb2

from tests.python.interaction.utils import FakeContext


def _register_single_replica(gs, model: str, node_id: str, *, max_concurrency: int = 1) -> None:
    """Helper to register a single GPU replica."""
    worker_req = global_store_pb2.RegisterWorkerRequest(
        node_id=node_id,
        node_address="127.0.0.1",
        grpc_port=9000,
        p2p_port=9001,
        mem_pool_total_size=8 * 1024 * 1024,
        mem_pool_available_size=8 * 1024 * 1024,
    )
    worker_resp = gs.RegisterWorker(worker_req, FakeContext())
    assert worker_resp.status == global_store_pb2.Status.OK

    mem_info = global_store_pb2.MemoryInfo(
        node_id=node_id,
        node_address="127.0.0.1",
        node_port=9000,
        remote_memory_keys=[f"key_{node_id}"],
        buffer_sizes=[2 * 1024 * 1024],
        memory_size=2 * 1024 * 1024,
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    reg = global_store_pb2.RegisterModelReplicaRequest(
        model_id=model,
        mem_info=mem_info,
        max_concurrency=max_concurrency,
        worker_id=worker_resp.worker_id,
    )
    rep_resp = gs.RegisterModelReplica(reg, FakeContext())
    assert rep_resp.status == global_store_pb2.Status.OK


def test_daemon_crash_releases_counters(global_store_service):
    """Simulate Daemon crash – verify cleanup_expired_transports frees counters."""

    gs = global_store_service
    model = "crash-model"

    _register_single_replica(gs, model, node_id="CRASH", max_concurrency=1)

    # Acquire transport (simulate load) – but DO NOT complete it.
    req = global_store_pb2.RequestModelReplicaTransportRequest(model_id=model)
    resp = gs.RequestModelReplicaTransport(req, FakeContext())
    assert resp.status == global_store_pb2.Status.OK

    # Verify counter incremented.
    replica = gs.model_replica_repository.find_by_model(model)[0]
    assert replica.current_requests == 1

    # Wait a short moment so created_at is < now (age > 0) then force cleanup.
    time.sleep(0.05)
    cleaned = gs.transport_service.cleanup_expired_transports(expiration_seconds=0)
    assert cleaned == 1  # Our single leaked transport should be cleaned.

    # Replica counter should have been decremented back to zero.
    replica_after = gs.model_replica_repository.find_by_model(model)[0]
    assert replica_after.current_requests == 0