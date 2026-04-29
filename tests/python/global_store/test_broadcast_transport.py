from __future__ import annotations

from tensorcast.global_store.models import BroadcastEdgeState
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
            max_attempts=3,
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
