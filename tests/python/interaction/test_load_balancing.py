#  Copyright (c) 2025, StepCast Team.

from scstore.proto import global_store_pb2

from tests.python.interaction.utils import FakeContext

def _register_replica(gs, model_id: str, node_id: str, max_concurrency: int, size_bytes: int = 1_000_000):
    """Helper: register worker then replica so that replica is available for transport."""

    # 1) Register worker
    worker_req = global_store_pb2.RegisterWorkerRequest(
        node_id=node_id,
        node_address="127.0.0.1",
        grpc_port=9000,
        p2p_port=9001,
        mem_pool_total_size=16 * 1024 * 1024,
        mem_pool_available_size=16 * 1024 * 1024,
    )
    worker_resp = gs.RegisterWorker(worker_req, FakeContext())
    assert worker_resp.status == global_store_pb2.Status.OK

    # 2) Register replica linked to this worker
    mem_info = global_store_pb2.MemoryInfo(
        node_id=node_id,
        node_address="127.0.0.1",
        node_port=9000,
        remote_memory_keys=[f"key_{node_id}"],
        memory_size=size_bytes,
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    req = global_store_pb2.RegisterModelReplicaRequest(
        model_id=model_id,
        mem_info=mem_info,
        max_concurrency=max_concurrency,
        worker_id=worker_resp.worker_id,
    )
    rep_resp = gs.RegisterModelReplica(req, FakeContext())
    assert rep_resp.status == global_store_pb2.Status.OK


def test_load_balancing_concurrency(global_store_service):
    """Validate that FakeGlobalStore enforces per-replica concurrency limits.

    Scenario 3-5 from the design doc:
    1. Create three GPU replicas with different `max_concurrency` limits.
    2. Issue 20 consecutive transport requests.
    3. Ensure that no replica exceeds its concurrency limit and that
       once all replicas are saturated, subsequent requests time-out.
    """

    gs = global_store_service  # alias

    model_id = "llama-2-7b"

    # Register three replicas: capacities 1, 4, 8 -> total 13 concurrent slots
    capacities = {"R1": 1, "R2": 4, "R3": 8}
    for node_id, cap in capacities.items():
        _register_replica(gs, model_id, node_id=node_id, max_concurrency=cap)

    # Helper to request a transport and return response
    def _request(wait_ms: int = 0):
        req = global_store_pb2.RequestModelReplicaTransportRequest(
            model_id=model_id,
            wait_timeout_ms=wait_ms,
        )
        return gs.RequestModelReplicaTransport(req, FakeContext())

    # ------------------------------------------------------------------
    # First, consume *all* available capacity (13 slots)
    # ------------------------------------------------------------------
    successes = []
    for _ in range(sum(capacities.values())):
        resp = _request()
        assert resp.status == global_store_pb2.Status.OK
        successes.append(resp)

    # Validate internal replica counters do not exceed capacity
    for replica in gs.model_replica_repository.find_by_model(model_id):
        assert replica.current_requests <= replica.max_concurrency

    # ------------------------------------------------------------------
    # Next request without waiting must immediately time-out because all
    # replicas are saturated (Scenario 4)
    # ------------------------------------------------------------------
    resp_timeout = _request(wait_ms=1)  # minimal wait ensures fast test
    assert resp_timeout.status == global_store_pb2.Status.TIMED_OUT

    # ------------------------------------------------------------------
    # Complete transports to free capacity, then issue additional requests
    # to ensure counters decrement correctly (Scenario 5)
    # ------------------------------------------------------------------
    for ok_resp in successes:
        complete_req = global_store_pb2.CompleteModelReplicaTransportRequest(
            transport_id=ok_resp.transport_id,
        )
        comp = gs.CompleteModelReplicaTransport(complete_req, FakeContext())
        assert comp.status == global_store_pb2.Status.OK

    # After completion, all counters should be back to zero
    for replica in gs.model_replica_repository.find_by_model(model_id):
        assert replica.current_requests == 0

    # Finally, issue one more request and confirm it succeeds now that
    # capacity is available again.
    final_resp = _request()
    assert final_resp.status == global_store_pb2.Status.OK