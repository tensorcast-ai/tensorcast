#  Copyright (c) 2025, StepCast Team.

from __future__ import annotations

import time

from scstore.proto import global_store_pb2
from .utils import get_free_port_pair

from tests.python.interaction.utils import FakeContext


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _register_replica(gs, *, model_id: str, node_id: str, max_concurrency: int = 1):
    """Register a worker + GPU replica.  Returns the *worker_id*."""

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
        memory_size=2 * 1024 * 1024,
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    reg_req = global_store_pb2.RegisterModelReplicaRequest(
        model_id=model_id,
        mem_info=mem_info,
        max_concurrency=max_concurrency,
        worker_id=worker_resp.worker_id,
    )
    rep_resp = gs.RegisterModelReplica(reg_req, FakeContext())
    assert rep_resp.status == global_store_pb2.Status.OK

    return worker_resp.worker_id


# -----------------------------------------------------------------------------
# Test case – stale worker heartbeat prevents allocation
# -----------------------------------------------------------------------------


def test_worker_heartbeat_stale(global_store_service):
    """Replicas whose workers have stale heartbeats must not be scheduled (Scenario 8)."""

    gs = global_store_service
    model = "stale-worker-model"

    # Register single replica
    worker_id = _register_replica(gs, model_id=model, node_id="STALE")

    # Mark the worker as stale – simulate heartbeat older than timeout
    # The repository helper sets `last_heartbeat` to 1970-01-01, well beyond any cutoff.
    gs.worker_repository.mark_as_stale(worker_id)

    # Attempt to request transport; should time out quickly because the only replica
    # belongs to the stale worker and thus is filtered out.
    req = global_store_pb2.RequestModelReplicaTransportRequest(
        model_id=model,
        wait_timeout_ms=10,  # small timeout to keep test fast
    )

    start = time.perf_counter()
    resp = gs.RequestModelReplicaTransport(req, FakeContext())
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert resp.status == global_store_pb2.Status.TIMED_OUT
    assert elapsed_ms >= 9  # Must have waited at least ~wait_timeout_ms