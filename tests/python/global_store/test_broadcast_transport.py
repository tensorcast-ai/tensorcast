from __future__ import annotations

from uuid import UUID

from tensorcast.global_store.models import (
    BroadcastEdgeState,
    BroadcastSessionState,
    BroadcastTargetState,
    Transport,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _register_worker(servicer, context, *, worker_id: str, node_id: str, port: int) -> str:
    response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id=f"daemon-{worker_id}",
            node_id=node_id,
            node_address=f"10.30.0.{port % 100}",
            grpc_port=port,
            p2p_port=port + 1,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        context,
    )
    assert response.status == global_store_pb2.STATUS_OK
    return response.worker_id


def _register_exportable_replica(
    servicer,
    context,
    *,
    artifact_id: str,
    worker_id: str,
    node_id: str,
    node_address: str,
    node_port: int,
    remote_key: str,
) -> str:
    mem_info = common_pb2.MemoryInfo(
        node_id=node_id,
        node_address=node_address,
        node_port=node_port,
        memory_size=1024,
        memory_type=common_pb2.MEMORY_TYPE_GPU,
        device_id=0,
        byte_space=common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
        ),
    )
    transport = mem_info.transport
    transport.export_state = common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    transport.export_generation = 1
    transport.remote_memory_keys.append(remote_key)
    transport.buffer_sizes.append(1024)

    response = servicer.RegisterReplica(
        global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            worker_id=worker_id,
            mem_info=mem_info,
            max_concurrency=4,
        ),
        context,
    )
    assert response.status == global_store_pb2.STATUS_OK
    return response.replica_id


def _create_broadcast_session(
    servicer,
    context,
    *,
    session_id: str,
    artifact_id: str,
    root_replica_id: str,
    child_worker_id: str,
    max_attempts: int = 3,
) -> None:
    response = servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id=session_id,
            artifact_id=artifact_id,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            epoch=1,
            fanout=1,
            strict_parent=True,
            max_attempts=max_attempts,
            root_replica_id=root_replica_id,
            targets=[
                global_store_pb2.BroadcastTargetIdentity(
                    worker_id=child_worker_id,
                )
            ],
        ),
        context,
    )
    assert response.status == global_store_pb2.STATUS_OK
    assert len(response.edges) == 1


def test_broadcast_transport_uses_edge_parent(servicer, test_context):
    artifact_id = "mi2:broadcast-transport-parent"
    root_worker = _register_worker(
        servicer, test_context, worker_id="root", node_id="node-1", port=53100
    )
    alternate_worker = _register_worker(
        servicer, test_context, worker_id="alt", node_id="node-2", port=53200
    )
    child_worker = _register_worker(
        servicer, test_context, worker_id="child", node_id="node-3", port=53300
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.30.0.1",
        node_port=53101,
        remote_key="rk-root",
    )
    _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=alternate_worker,
        node_id="node-2",
        node_address="10.30.0.2",
        node_port=53201,
        remote_key="rk-alt",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-transport-parent",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )

    response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.30.9.9",
            source_port=59000,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-broadcast-parent",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-transport-parent",
                strict_parent=True,
            ),
        ),
        test_context,
    )

    assert response.status == global_store_pb2.STATUS_OK
    assert response.remote_memory_info.node_id == "node-1"
    assert list(response.remote_memory_info.transport.remote_memory_keys) == ["rk-root"]


def test_broadcast_failed_transport_requeues_target(servicer, test_context):
    artifact_id = "mi2:broadcast-transport-failed"
    root_worker = _register_worker(
        servicer, test_context, worker_id="root-fail", node_id="node-1", port=54100
    )
    child_worker = _register_worker(
        servicer, test_context, worker_id="child-fail", node_id="node-2", port=54200
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.40.0.1",
        node_port=54101,
        remote_key="rk-root-fail",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-transport-failed",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )

    transport_response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.40.9.9",
            source_port=59001,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-broadcast-failed",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-transport-failed",
                strict_parent=True,
            ),
        ),
        test_context,
    )
    assert transport_response.status == global_store_pb2.STATUS_OK

    complete_response = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=transport_response.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_FAILED,
            outcome_detail="forced test failure",
        ),
        test_context,
    )
    assert complete_response.status == global_store_pb2.STATUS_OK

    edges = servicer.broadcast_service.list_edges("session-transport-failed")
    assert any(edge.state is BroadcastEdgeState.FAILED for edge in edges)


def test_duplicate_broadcast_request_reuses_existing_edge(servicer, test_context):
    artifact_id = "mi2:broadcast-transport-duplicate"
    root_worker = _register_worker(
        servicer, test_context, worker_id="root-dup", node_id="node-1", port=55100
    )
    child_worker = _register_worker(
        servicer, test_context, worker_id="child-dup", node_id="node-2", port=55200
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.50.0.1",
        node_port=55101,
        remote_key="rk-root-dup",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-transport-duplicate",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )
    request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        source_node_id="requester-node",
        source_address="10.50.9.9",
        source_port=59002,
        requested_byte_space=common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
        ),
        requester_worker_id=child_worker,
        request_id="request-broadcast-duplicate",
        broadcast=global_store_pb2.BroadcastTransportHint(
            session_id="session-transport-duplicate",
            strict_parent=True,
        ),
    )

    first = servicer.RequestReplicaTransport(request, test_context)
    second = servicer.RequestReplicaTransport(request, test_context)

    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    assert second.transport_id == first.transport_id
    edges = servicer.broadcast_service.list_edges("session-transport-duplicate")
    assert len(edges) == 1
    assert edges[0].state is BroadcastEdgeState.MATERIALIZING
    targets = servicer.broadcast_service.list_targets("session-transport-duplicate")
    assert len(targets) == 1
    assert targets[0].state is BroadcastTargetState.MATERIALIZING
    assert targets[0].assigned_edge_id == edges[0].edge_id


def test_failed_broadcast_transport_replay_claims_retry_edge(servicer, test_context):
    artifact_id = "mi2:broadcast-replay-after-failure"
    root_worker = _register_worker(
        servicer,
        test_context,
        worker_id="root-replay-failure",
        node_id="node-1",
        port=55300,
    )
    child_worker = _register_worker(
        servicer,
        test_context,
        worker_id="child-replay-failure",
        node_id="node-2",
        port=55400,
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.53.0.1",
        node_port=55301,
        remote_key="rk-root-replay-failure",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-replay-failure",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )
    request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        source_node_id="requester-node",
        source_address="10.53.9.9",
        source_port=59010,
        requested_byte_space=common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
        ),
        requester_worker_id=child_worker,
        request_id="request-replay-failure",
        broadcast=global_store_pb2.BroadcastTransportHint(
            session_id="session-replay-failure",
            strict_parent=True,
        ),
    )
    first = servicer.RequestReplicaTransport(request, test_context)
    assert first.status == global_store_pb2.STATUS_OK
    complete = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=first.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_FAILED,
        ),
        test_context,
    )
    assert complete.status == global_store_pb2.STATUS_OK
    retry_edge = [
        edge
        for edge in servicer.broadcast_service.list_edges("session-replay-failure")
        if edge.state is BroadcastEdgeState.PLANNED
    ][0]

    replay = servicer.RequestReplicaTransport(request, test_context)

    assert replay.status == global_store_pb2.STATUS_OK
    assert replay.transport_id != first.transport_id
    edges = servicer.broadcast_service.list_edges("session-replay-failure")
    retry = [edge for edge in edges if edge.edge_id == retry_edge.edge_id][0]
    assert retry.state is BroadcastEdgeState.MATERIALIZING


def test_success_without_child_replay_claims_retry_edge(servicer, test_context):
    artifact_id = "mi2:broadcast-replay-after-no-child"
    root_worker = _register_worker(
        servicer,
        test_context,
        worker_id="root-replay-no-child",
        node_id="node-1",
        port=56300,
    )
    child_worker = _register_worker(
        servicer,
        test_context,
        worker_id="child-replay-no-child",
        node_id="node-2",
        port=56400,
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.63.0.1",
        node_port=56301,
        remote_key="rk-root-replay-no-child",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-replay-no-child",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )
    request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        source_node_id="requester-node",
        source_address="10.63.9.9",
        source_port=59011,
        requested_byte_space=common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
        ),
        requester_worker_id=child_worker,
        request_id="request-replay-no-child",
        broadcast=global_store_pb2.BroadcastTransportHint(
            session_id="session-replay-no-child",
            strict_parent=True,
        ),
    )
    first = servicer.RequestReplicaTransport(request, test_context)
    assert first.status == global_store_pb2.STATUS_OK
    complete = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=first.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
        ),
        test_context,
    )
    assert complete.status == global_store_pb2.STATUS_OK
    retry_edge = [
        edge
        for edge in servicer.broadcast_service.list_edges("session-replay-no-child")
        if edge.state is BroadcastEdgeState.PLANNED
    ][0]

    replay = servicer.RequestReplicaTransport(request, test_context)

    assert replay.status == global_store_pb2.STATUS_OK
    assert replay.transport_id != first.transport_id
    edges = servicer.broadcast_service.list_edges("session-replay-no-child")
    retry = [edge for edge in edges if edge.edge_id == retry_edge.edge_id][0]
    assert retry.state is BroadcastEdgeState.MATERIALIZING


def test_broadcast_request_existing_transport_missing_replica_does_not_claim_edge(
    servicer,
    test_context,
):
    artifact_id = "mi2:broadcast-transport-missing-replica"
    root_worker = _register_worker(
        servicer,
        test_context,
        worker_id="root-missing-replica",
        node_id="node-1",
        port=55500,
    )
    child_worker = _register_worker(
        servicer,
        test_context,
        worker_id="child-missing-replica",
        node_id="node-2",
        port=55600,
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.55.0.1",
        node_port=55501,
        remote_key="rk-root-missing-replica",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-missing-replica",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )
    servicer.transport_repository.create(
        Transport(
            replica_id=UUID("00000000-0000-0000-0000-00000000dead"),
            artifact_id=artifact_id,
            source_node_id="stale-node",
            source_address="10.55.9.9",
            source_port=59006,
            requester_worker_id=child_worker,
            request_id="request-missing-replica",
        )
    )

    response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.55.9.10",
            source_port=59007,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-missing-replica",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-missing-replica",
                strict_parent=True,
            ),
        ),
        test_context,
    )

    assert response.status == global_store_pb2.STATUS_NOT_FOUND
    edges = servicer.broadcast_service.list_edges("session-missing-replica")
    assert len(edges) == 1
    assert edges[0].state is BroadcastEdgeState.PLANNED
    target = servicer.broadcast_service.list_targets("session-missing-replica")[0]
    assert target.state is BroadcastTargetState.ASSIGNED
    assert target.assigned_edge_id == edges[0].edge_id


def test_broadcast_parent_ineligible_exhausts_attempt_and_fails_session(
    servicer,
    test_context,
):
    artifact_id = "mi2:broadcast-parent-ineligible-max"
    root_worker = _register_worker(
        servicer,
        test_context,
        worker_id="root-ineligible-max",
        node_id="node-1",
        port=55700,
    )
    child_worker = _register_worker(
        servicer,
        test_context,
        worker_id="child-ineligible-max",
        node_id="node-2",
        port=55800,
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.57.0.1",
        node_port=55701,
        remote_key="rk-root-ineligible-max",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-parent-ineligible-max",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
        max_attempts=1,
    )
    servicer.replica_repository.mark_unavailable(UUID(root_replica_id))

    response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.57.9.9",
            source_port=59008,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-parent-ineligible-max",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-parent-ineligible-max",
                strict_parent=True,
            ),
        ),
        test_context,
    )

    assert response.status == global_store_pb2.STATUS_NOT_FOUND
    edges = servicer.broadcast_service.list_edges("session-parent-ineligible-max")
    target = servicer.broadcast_service.list_targets("session-parent-ineligible-max")[0]
    session = servicer.broadcast_service.get_session("session-parent-ineligible-max")
    assert session is not None
    assert len(edges) == 1
    assert edges[0].state is BroadcastEdgeState.FAILED
    assert edges[0].failure_reason == "parent_replica_not_transport_eligible"
    assert target.state is BroadcastTargetState.FAILED
    assert session.state is BroadcastSessionState.FAILED


def test_broadcast_parent_ineligible_retries_without_stuck_active_edge(
    servicer,
    test_context,
):
    artifact_id = "mi2:broadcast-parent-ineligible-retry"
    root_worker = _register_worker(
        servicer,
        test_context,
        worker_id="root-ineligible-retry",
        node_id="node-1",
        port=55900,
    )
    child_worker = _register_worker(
        servicer,
        test_context,
        worker_id="child-ineligible-retry",
        node_id="node-2",
        port=56000,
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.59.0.1",
        node_port=55901,
        remote_key="rk-root-ineligible-retry",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-parent-ineligible-retry",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
        max_attempts=3,
    )
    original_edge = servicer.broadcast_service.list_edges(
        "session-parent-ineligible-retry"
    )[0]
    servicer.replica_repository.mark_unavailable(UUID(root_replica_id))

    response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.59.9.9",
            source_port=59009,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-parent-ineligible-retry",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-parent-ineligible-retry",
                strict_parent=True,
            ),
        ),
        test_context,
    )

    assert response.status == global_store_pb2.STATUS_NOT_FOUND
    edges = servicer.broadcast_service.list_edges("session-parent-ineligible-retry")
    target = servicer.broadcast_service.list_targets(
        "session-parent-ineligible-retry"
    )[0]
    original = [edge for edge in edges if edge.edge_id == original_edge.edge_id][0]
    retry_edges = [edge for edge in edges if edge.edge_id != original_edge.edge_id]
    assert original.state is BroadcastEdgeState.FAILED
    assert original.failure_reason == "parent_replica_not_transport_eligible"
    assert len(retry_edges) == 1
    assert retry_edges[0].state is BroadcastEdgeState.PLANNED
    assert retry_edges[0].attempt == 2
    assert target.state is BroadcastTargetState.ASSIGNED
    assert target.assigned_edge_id == retry_edges[0].edge_id


def test_broadcast_success_without_child_replica_requeues(servicer, test_context):
    artifact_id = "mi2:broadcast-transport-success-no-child"
    root_worker = _register_worker(
        servicer,
        test_context,
        worker_id="root-no-child",
        node_id="node-1",
        port=56100,
    )
    child_worker = _register_worker(
        servicer,
        test_context,
        worker_id="child-no-child",
        node_id="node-2",
        port=56200,
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.60.0.1",
        node_port=56101,
        remote_key="rk-root-no-child",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-success-no-child",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )
    transport_response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.60.9.9",
            source_port=59003,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-success-no-child",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-success-no-child",
                strict_parent=True,
            ),
        ),
        test_context,
    )
    assert transport_response.status == global_store_pb2.STATUS_OK

    complete_response = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=transport_response.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
        ),
        test_context,
    )

    assert complete_response.status == global_store_pb2.STATUS_OK
    edges = servicer.broadcast_service.list_edges("session-success-no-child")
    assert [edge.state for edge in edges].count(BroadcastEdgeState.FAILED) == 1
    assert [edge.state for edge in edges].count(BroadcastEdgeState.PLANNED) == 1
    target = servicer.broadcast_service.list_targets("session-success-no-child")[0]
    assert target.state is BroadcastTargetState.ASSIGNED
    assert target.attempt == 2


def test_broadcast_max_attempt_exhaustion_marks_session_failed(servicer, test_context):
    artifact_id = "mi2:broadcast-transport-max-attempts"
    root_worker = _register_worker(
        servicer, test_context, worker_id="root-max", node_id="node-1", port=57100
    )
    child_worker = _register_worker(
        servicer, test_context, worker_id="child-max", node_id="node-2", port=57200
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.70.0.1",
        node_port=57101,
        remote_key="rk-root-max",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-max-attempts",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
        max_attempts=1,
    )
    transport_response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.70.9.9",
            source_port=59004,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-max-attempts",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-max-attempts",
                strict_parent=True,
            ),
        ),
        test_context,
    )
    assert transport_response.status == global_store_pb2.STATUS_OK

    complete_response = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=transport_response.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_FAILED,
        ),
        test_context,
    )

    assert complete_response.status == global_store_pb2.STATUS_OK
    session = servicer.broadcast_service.get_session("session-max-attempts")
    target = servicer.broadcast_service.list_targets("session-max-attempts")[0]
    assert session is not None
    assert session.state is BroadcastSessionState.FAILED
    assert target.state is BroadcastTargetState.FAILED


def test_duplicate_broadcast_completion_is_noop(servicer, test_context):
    artifact_id = "mi2:broadcast-transport-complete-twice"
    root_worker = _register_worker(
        servicer, test_context, worker_id="root-twice", node_id="node-1", port=58100
    )
    child_worker = _register_worker(
        servicer, test_context, worker_id="child-twice", node_id="node-2", port=58200
    )
    root_replica_id = _register_exportable_replica(
        servicer,
        test_context,
        artifact_id=artifact_id,
        worker_id=root_worker,
        node_id="node-1",
        node_address="10.80.0.1",
        node_port=58101,
        remote_key="rk-root-twice",
    )
    _create_broadcast_session(
        servicer,
        test_context,
        session_id="session-complete-twice",
        artifact_id=artifact_id,
        root_replica_id=root_replica_id,
        child_worker_id=child_worker,
    )
    transport_response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id="requester-node",
            source_address="10.80.9.9",
            source_port=59005,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
            ),
            requester_worker_id=child_worker,
            request_id="request-complete-twice",
            broadcast=global_store_pb2.BroadcastTransportHint(
                session_id="session-complete-twice",
                strict_parent=True,
            ),
        ),
        test_context,
    )
    assert transport_response.status == global_store_pb2.STATUS_OK
    complete_request = global_store_pb2.CompleteReplicaTransportRequest(
        transport_id=transport_response.transport_id,
        outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_FAILED,
    )

    first = servicer.CompleteReplicaTransport(complete_request, test_context)
    edges_after_first = servicer.broadcast_service.list_edges("session-complete-twice")
    second = servicer.CompleteReplicaTransport(complete_request, test_context)
    edges_after_second = servicer.broadcast_service.list_edges("session-complete-twice")

    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    assert [(e.edge_id, e.state) for e in edges_after_second] == [
        (e.edge_id, e.state) for e in edges_after_first
    ]
