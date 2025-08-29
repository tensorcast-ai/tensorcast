#  Copyright (c) 2025, TensorCast Team.

import random
import uuid
import os

import pytest

from scstore.proto import global_store_pb2
from tests.python.interaction.utils import FakeContext, get_free_port_pair

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

TOTAL_ITERATIONS = 5000  # Keep small for CI yet exercise the system
FAULT_RATE = 0.2  # 10% transport failures
TIMEOUT_RATE = 0.2  # 5% deliberate timeouts


@pytest.mark.stress
def test_mixed_fault_stress(global_store_service):
    """Scenario 10 & 11 – run mixed operations, ensure counters return to zero."""

    gs = global_store_service
    artifact = "orca-mini"

    # Register two replicas with modest concurrency limits
    capacities = {"S1": 3, "S2": 5}
    for node_id, cap in capacities.items():
        # Get dynamic ports for this worker
        grpc_port, p2p_port = get_free_port_pair()

        # Register worker first
        worker_req = global_store_pb2.RegisterWorkerRequest(
            node_id=node_id,
            node_address="127.0.0.1",
            grpc_port=grpc_port,
            p2p_port=p2p_port,
            mem_pool_total_size=32 * 1024 * 1024,
            mem_pool_available_size=32 * 1024 * 1024,
        )
        worker_resp = gs.RegisterWorker(worker_req, FakeContext())
        assert worker_resp.status == global_store_pb2.Status.OK

        mem_info = global_store_pb2.MemoryInfo(
            node_id=node_id,
            node_address="127.0.0.1",
            node_port=grpc_port,
            remote_memory_keys=[f"key_{node_id}"],
            memory_size=8 * 1024 * 1024,
            memory_type=global_store_pb2.MemoryType.GPU,
            device_id=0,
        )
        reg = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact,
            mem_info=mem_info,
            max_concurrency=capacities[node_id],
            worker_id=worker_resp.worker_id,
        )
        reg_resp = gs.RegisterReplica(reg, FakeContext())
        assert reg_resp.status == global_store_pb2.Status.OK

    active_transport_ids: list[str] = []
    leaked_ids: list[str] = []

    for _ in range(TOTAL_ITERATIONS):
        action_pick = random.random()
        # ~10%: fail one outstanding transport (simulate missing completion)
        if active_transport_ids and action_pick < FAULT_RATE:
            victim = random.choice(active_transport_ids)
            # Simulate lost completion notification by moving to leaked list
            active_transport_ids.remove(victim)
            leaked_ids.append(victim)
            continue

        # ~5%: attempt request that is expected to time-out (simulate saturation)
        if action_pick < FAULT_RATE + TIMEOUT_RATE:
            req_to = global_store_pb2.RequestReplicaTransportRequest(
                artifact_id=artifact,
                wait_timeout_ms=1,
            )
            resp = gs.RequestReplicaTransport(req_to, FakeContext())
            # Either we time-out (preferred) or succeed if capacity available
            assert resp.status in (
                global_store_pb2.Status.TIMED_OUT,
                global_store_pb2.Status.OK,
            )
            if resp.status == global_store_pb2.Status.OK:
                active_transport_ids.append(resp.transport_id)
            continue

        # Otherwise request a normal transport
        req = global_store_pb2.RequestReplicaTransportRequest(artifact_id=artifact)
        resp = gs.RequestReplicaTransport(req, FakeContext())
        if resp.status == global_store_pb2.Status.OK:
            active_transport_ids.append(resp.transport_id)

        # Occasionally complete a random active transport
        if active_transport_ids and random.random() < 0.3:
            tid = active_transport_ids.pop(random.randrange(len(active_transport_ids)))
            comp_req = global_store_pb2.CompleteReplicaTransportRequest(transport_id=tid)
            gs.CompleteReplicaTransport(comp_req, FakeContext())

    # After workload, complete all remaining transports including leaked ones
    # ------------------------------------------------------------------
    for tid in list(active_transport_ids) + leaked_ids:
        comp_req = global_store_pb2.CompleteReplicaTransportRequest(transport_id=tid)
        gs.CompleteReplicaTransport(comp_req, FakeContext())

    # Validate replica counters
    for replica in gs.replica_repository.find_by_artifact(artifact):
        assert replica.current_requests == 0