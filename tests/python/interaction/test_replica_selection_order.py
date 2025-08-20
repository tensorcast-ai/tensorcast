#  Copyright (c) 2025, StepCast Team.

from __future__ import annotations

import pytest

from scstore.proto import global_store_pb2
from .utils import get_free_port_pair

from tests.python.interaction.utils import FakeContext


# -----------------------------------------------------------------------------
# Helpers – keep self-contained to avoid cross-test dependencies
# -----------------------------------------------------------------------------

def _register_gpu_replica(gs, *, model_id: str, node_id: str, max_concurrency: int) -> None:
    """Register a worker + single-GPU replica with *max_concurrency* for *model_id*."""

    # 1) Worker registration (must precede replica)
    worker_req = global_store_pb2.RegisterWorkerRequest(
        node_id=node_id,
        node_address="127.0.0.1",
        grpc_port=9000,
        p2p_port=9001,
        mem_pool_total_size=32 * 1024 * 1024,
        mem_pool_available_size=32 * 1024 * 1024,
    )
    worker_resp = gs.RegisterWorker(worker_req, FakeContext())
    assert worker_resp.status == global_store_pb2.Status.OK

    mem_info = global_store_pb2.MemoryInfo(
        node_id=node_id,
        node_address="127.0.0.1",
        node_port=9000,
        remote_memory_keys=[f"key_{node_id}"],
        memory_size=1 * 1024 * 1024,  # 1 MiB – arbitrary
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


# -----------------------------------------------------------------------------
# Test case – validate deterministic replica selection order (Scenario 2 in §九)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("capacities", [{"R1": 1, "R2": 4, "R3": 8}])
def test_replica_selection_order(global_store_service, capacities):
    """Ensure load-balancer allocates replicas in *capabilities* ascending order.

    The algorithm should prefer the replica with the **lowest**
    `current_requests / max_concurrency` utilisation.  Given the staged
    capacities (1, 4, 8) the expected assignment sequence for the first
    13 requests is:  R1 (×1) → R2 (×4) → R3 (×8).
    """

    gs = global_store_service
    model_id = "llama-replica-order"

    # Register replicas **largest capacity first** so that the tie-break
    # `updated_at DESC` does not influence the desired deterministic order.
    for node_id, cap in sorted(capacities.items(), key=lambda kv: kv[1], reverse=True):
        _register_gpu_replica(gs, model_id=model_id, node_id=node_id, max_concurrency=cap)

    def _request():
        req = global_store_pb2.RequestModelReplicaTransportRequest(model_id=model_id)
        resp = gs.RequestModelReplicaTransport(req, FakeContext())
        assert resp.status == global_store_pb2.Status.OK
        return resp.remote_memory_info.node_id

    # Collect the node_id selected for each of the first *total_capacity* requests
    total_capacity = sum(capacities.values())
    selections: list[str] = [_request() for _ in range(total_capacity)]

    # Build expected order
    expected: list[str] = [
        "R1",
        *["R2"] * capacities["R2"],
        *["R3"] * capacities["R3"],
    ]

    assert selections == expected, f"Unexpected allocation order: {selections}"

    # House-keeping – complete all transports to keep GS counters balanced
    for transport_id in list(gs.transport_repository.list_with_filters(status="in_progress")):
        complete_req = global_store_pb2.CompleteModelReplicaTransportRequest(
            transport_id=str(transport_id.transport_id),
        )
        gs.CompleteModelReplicaTransport(complete_req, FakeContext())