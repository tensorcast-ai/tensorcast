from __future__ import annotations

import pytest

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
    assert len(service.list_edges("session-a")) == 2


def test_create_session_duplicate_explicit_root_returns_existing_without_counter_change(
    repositories,
):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-explicit", "daemon-root-explicit", "node1")
    child = _worker("worker-child-explicit", "daemon-child-explicit", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-explicit", root))

    first = service.create_session(
        session_id="session-explicit",
        artifact_id="mi2:model-explicit",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-explicit"],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0

    second = service.create_session(
        session_id="session-explicit",
        artifact_id="mi2:model-explicit",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-explicit"],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )

    assert second.session_id == first.session_id
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0
    assert len(broadcast_repo.list_targets("session-explicit")) == 1
    assert len(service.list_edges("session-explicit")) == 1


def test_create_session_duplicate_auto_root_returns_existing_without_counter_change(
    repositories,
):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-auto", "daemon-root-auto", "node1")
    child = _worker("worker-child-auto", "daemon-child-auto", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-auto", root))

    first = service.create_session(
        session_id="session-auto",
        artifact_id="mi2:model-auto",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-auto"],
        root_replica_id="",
        strict_parent=True,
        max_attempts=3,
    )
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0

    second = service.create_session(
        session_id="session-auto",
        artifact_id="mi2:model-auto",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-auto"],
        root_replica_id="",
        strict_parent=True,
        max_attempts=3,
    )

    assert second.session_id == first.session_id
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0
    assert len(broadcast_repo.list_targets("session-auto")) == 1
    assert len(service.list_edges("session-auto")) == 1


def test_create_session_auto_root_failure_releases_counter_and_rolls_back(
    repositories,
    monkeypatch,
):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-failure", "daemon-root-failure", "node1")
    child = _worker("worker-child-failure", "daemon-child-failure", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-failure", root))

    def fail_planning(*args, **kwargs):
        raise RuntimeError("forced planning failure")

    monkeypatch.setattr(service, "_plan_more_edges", fail_planning)

    with pytest.raises(RuntimeError, match="forced planning failure"):
        service.create_session(
            session_id="session-failure",
            artifact_id="mi2:model-failure",
            requested_view_id=None,
            epoch=1,
            fanout=1,
            target_daemon_ids=["daemon-child-failure"],
            root_replica_id=None,
            strict_parent=True,
            max_attempts=3,
        )

    assert replica_repo.get_current_requests(root_replica.replica_id) == 0
    assert broadcast_repo.find_session("session-failure") is None
    assert broadcast_repo.list_targets("session-failure") == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"session_id": ""}, "session_id is required"),
        ({"artifact_id": ""}, "artifact_id is required"),
        ({"epoch": -1}, "epoch must be >= 0"),
    ],
)
def test_create_session_validates_required_inputs(repositories, overrides, message):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-validation", "daemon-root-validation", "node1")
    child = _worker("worker-child-validation", "daemon-child-validation", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(
        _exportable_replica("mi2:model-validation", root)
    )

    kwargs = {
        "session_id": "session-validation",
        "artifact_id": "mi2:model-validation",
        "requested_view_id": None,
        "epoch": 1,
        "fanout": 1,
        "target_daemon_ids": ["daemon-child-validation"],
        "root_replica_id": str(root_replica.replica_id),
        "strict_parent": True,
        "max_attempts": 3,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        service.create_session(**kwargs)
    assert broadcast_repo.find_session("session-validation") is None
