from __future__ import annotations

from tensorcast.global_store.models import (
    BroadcastEdgeState,
    BroadcastSessionState,
    BroadcastTargetState,
    ExportState,
    MemoryType,
    Replica,
    Worker,
)
from tensorcast.global_store.services import BroadcastService


def _worker(worker_id: str, daemon_id: str, node_id: str) -> Worker:
    return Worker(
        worker_id=worker_id,
        daemon_id=daemon_id,
        node_id=node_id,
        node_address=f"10.0.0.{node_id[-1]}",
        grpc_port=5000 + int(node_id[-1]),
        p2p_port=6000 + int(node_id[-1]),
        mem_pool_total_size=4096,
        mem_pool_available_size=4096,
        accepting_new_requests=True,
    )


def _exportable_replica(artifact_id: str, worker: Worker) -> Replica:
    return Replica(
        artifact_id=artifact_id,
        node_id=worker.node_id,
        node_address=worker.node_address,
        node_port=worker.p2p_port,
        memory_size=1024,
        memory_type=MemoryType.GPU,
        device_id=0,
        max_concurrency=4,
        current_requests=0,
        is_available=True,
        remote_memory_keys=[f"rk-{worker.worker_id}"],
        buffer_sizes=[1024],
        export_state=ExportState.EXPORTABLE,
        worker_id=worker.worker_id,
    )


def test_create_session_plans_first_layer_by_fanout(repositories):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root", "daemon-root", "node1")
    child1 = _worker("worker-child-1", "daemon-child-1", "node2")
    child2 = _worker("worker-child-2", "daemon-child-2", "node3")
    child3 = _worker("worker-child-3", "daemon-child-3", "node4")
    for worker in (root, child1, child2, child3):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-a", root))

    session = service.create_session(
        session_id="session-a",
        artifact_id="mi2:model-a",
        requested_view_id=None,
        epoch=42,
        fanout=2,
        target_daemon_ids=["daemon-child-1", "daemon-child-2", "daemon-child-3"],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )

    assert session.state is BroadcastSessionState.ACTIVE
    targets = broadcast_repo.list_targets("session-a")
    assert len(targets) == 3
    assigned = [t for t in targets if t.state is BroadcastTargetState.ASSIGNED]
    pending = [t for t in targets if t.state is BroadcastTargetState.PENDING]
    assert len(assigned) == 2
    assert len(pending) == 1
    edges = [
        broadcast_repo.find_active_edge_for_child("session-a", t.target_worker_id)
        for t in assigned
    ]
    assert all(edge is not None for edge in edges)
    assert all(edge.state is BroadcastEdgeState.PLANNED for edge in edges if edge)
    assert all(edge.parent_replica_id == root_replica.replica_id for edge in edges if edge)
