#  Copyright (c) 2025, StepCast Team.

import time
import uuid

import pytest

from scstore.proto import global_store_pb2

from tests.python.interaction.utils import FakeContext
from tests.python.interaction.fakes.fake_p2p import FakeP2PNetwork

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _register_single_gpu_replica(gs, artifact_id: str, replica_id: str, *, max_concurrency: int = 1):
    """Register a worker and single GPU replica for testing."""

    # 1) Register worker first so replica is considered available
    worker_req = global_store_pb2.RegisterWorkerRequest(
        node_id=replica_id,
        node_address="127.0.0.1",
        grpc_port=9000,
        p2p_port=9001,
        mem_pool_total_size=16 * 1024 * 1024,
        mem_pool_available_size=16 * 1024 * 1024,
    )
    worker_resp = gs.RegisterWorker(worker_req, FakeContext())
    assert worker_resp.status == global_store_pb2.Status.OK

    # 2) Register replica
    mem_info = global_store_pb2.MemoryInfo(
        node_id=replica_id,
        node_address="127.0.0.1",
        node_port=9000,
        remote_memory_keys=[f"key_{replica_id}"],
        memory_size=2 * 1024 * 1024,  # 2 MiB
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    req = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact_id,
        mem_info=mem_info,
        max_concurrency=max_concurrency,
        worker_id=worker_resp.worker_id,
    )
    rep_resp = gs.RegisterReplica(req, FakeContext())
    assert rep_resp.status == global_store_pb2.Status.OK


# -----------------------------------------------------------------------------
# Fault-injection tests (Section C & D of design document)
# -----------------------------------------------------------------------------


def test_transport_failure_keeps_counter(global_store_service):
    """Scenario 6 – transport fails, counter remains incremented and new request times-out."""

    gs = global_store_service
    artifact_id = "phi2-1.7b"

    # 1. Register single replica with capacity 1
    _register_single_gpu_replica(gs, artifact_id, replica_id="R_FAIL", max_concurrency=1)

    # 2. First transport allocation succeeds
    req_ok = global_store_pb2.RequestReplicaTransportRequest(artifact_id=artifact_id)
    resp_ok = gs.RequestReplicaTransport(req_ok, FakeContext())
    assert resp_ok.status == global_store_pb2.Status.OK

    # 3. Immediately request another transport – should time-out because capacity is saturated
    req_to = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id, wait_timeout_ms=5
    )
    start = time.perf_counter()
    resp_to = gs.RequestReplicaTransport(req_to, FakeContext())
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert resp_to.status == global_store_pb2.Status.TIMED_OUT
    # The fake store sleeps exactly wait_timeout_ms on timeout; allow small margin
    assert elapsed_ms >= 4

    # Internal counter should still be 1 (because CompleteTransport not called)
    replica = gs.replica_repository.find_by_artifact(artifact_id)[0]
    assert replica.current_requests == 1

    # Clean-up: complete the lingering transport so that other tests are unaffected
    complete_req = global_store_pb2.CompleteReplicaTransportRequest(
        transport_id=resp_ok.transport_id,
    )
    gs.CompleteReplicaTransport(complete_req, FakeContext())


def test_complete_transport_with_invalid_id_returns_not_found(global_store_service):
    """Scenario 9 – mismatched transport_id causes NOT_FOUND on completion."""

    gs = global_store_service
    artifact = "tiny-gpt"
    _register_single_gpu_replica(gs, artifact, replica_id="R1", max_concurrency=2)

    # Acquire a real transport first (sanity)
    req = global_store_pb2.RequestReplicaTransportRequest(artifact_id=artifact)
    resp = gs.RequestReplicaTransport(req, FakeContext())
    assert resp.status == global_store_pb2.Status.OK

    # Attempt to complete with an unrelated UUID
    bogus_id = str(uuid.uuid4())
    comp_req = global_store_pb2.CompleteReplicaTransportRequest(transport_id=bogus_id)
    comp_resp = gs.CompleteReplicaTransport(comp_req, FakeContext())
    assert comp_resp.status == global_store_pb2.Status.NOT_FOUND

    # Clean up the *valid* transport to keep global counters balanced
    cleanup_req = global_store_pb2.CompleteReplicaTransportRequest(transport_id=resp.transport_id)
    gs.CompleteReplicaTransport(cleanup_req, FakeContext())


@pytest.mark.asyncio
async def test_p2p_network_injected_failure():
    """Validate FakeP2PNetwork.inject_failure triggers IOError on request (part of Scenario 6)."""

    net = FakeP2PNetwork()
    replica_id = "replica-x"
    payload = b"dummy-bytes"

    await net.register_replica(replica_id, payload)
    await net.inject_failure(replica_id)

    with pytest.raises(IOError):
        await net.request(replica_id)