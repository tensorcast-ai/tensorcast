#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2

ARTIFACT_ID = "mi2:model-e2e"


def _register_worker(servicer, context, idx: int) -> str:  # noqa: ANN001
    response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id=f"daemon-e2e-{idx}",
            node_id=f"node-e2e-{idx}",
            node_address=f"10.30.0.{idx}",
            grpc_port=53000 + idx,
            p2p_port=54000 + idx,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        context,
    )
    assert response.status == global_store_pb2.STATUS_OK
    return response.worker_id


def _register_exportable_replica(
    servicer,  # noqa: ANN001
    context,  # noqa: ANN001
    *,
    worker_id: str,
    idx: int,
) -> str:
    mem_info = common_pb2.MemoryInfo(
        node_id=f"node-e2e-{idx}",
        node_address=f"10.30.0.{idx}",
        node_port=54000 + idx,
        memory_size=1024,
        memory_type=common_pb2.MEMORY_TYPE_GPU,
        device_id=0,
        byte_space=common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
        ),
    )
    mem_info.transport.export_state = (
        common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    )
    mem_info.transport.export_generation = 1
    mem_info.transport.remote_memory_keys.append(f"rk-e2e-{idx}")
    mem_info.transport.buffer_sizes.append(1024)
    request = global_store_pb2.RegisterReplicaRequest(
        artifact_id=ARTIFACT_ID,
        worker_id=worker_id,
        mem_info=mem_info,
        max_concurrency=4,
    )
    response = servicer.RegisterReplica(request, context)
    assert response.status == global_store_pb2.STATUS_OK
    return response.replica_id


def _request_broadcast_transport(
    servicer,  # noqa: ANN001
    context,  # noqa: ANN001
    *,
    session_id: str,
    worker_id: str,
    idx: int,
    request_id: str,
) -> global_store_pb2.RequestReplicaTransportResponse:
    request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=ARTIFACT_ID,
        source_node_id=f"node-e2e-{idx}",
        source_address=f"10.30.0.{idx}",
        source_port=54000 + idx,
        requester_worker_id=worker_id,
        request_id=request_id,
        requested_byte_space=common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
        ),
    )
    request.local_memory_info.memory_type = common_pb2.MEMORY_TYPE_GPU
    request.local_memory_info.device_id = 0
    request.broadcast.session_id = session_id
    request.broadcast.strict_parent = True
    response = servicer.RequestReplicaTransport(request, context)
    assert response.status == global_store_pb2.STATUS_OK
    return response


def test_tree_broadcast_promotes_first_child_to_second_layer_parent(
    servicer,  # noqa: ANN001
    test_context,  # noqa: ANN001
) -> None:
    root = _register_worker(servicer, test_context, 1)
    child1 = _register_worker(servicer, test_context, 2)
    child2 = _register_worker(servicer, test_context, 3)
    root_replica = _register_exportable_replica(
        servicer,
        test_context,
        worker_id=root,
        idx=1,
    )

    create = servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id="session-e2e",
            artifact_id=ARTIFACT_ID,
            epoch=1,
            fanout=1,
            root_replica_id=root_replica,
            strict_parent=True,
            max_attempts=3,
            targets=[
                global_store_pb2.BroadcastTargetIdentity(worker_id=child1),
                global_store_pb2.BroadcastTargetIdentity(worker_id=child2),
            ],
        ),
        test_context,
    )
    assert create.status == global_store_pb2.STATUS_OK

    first = _request_broadcast_transport(
        servicer,
        test_context,
        session_id="session-e2e",
        worker_id=child1,
        idx=2,
        request_id="request-child-1",
    )
    assert first.remote_memory_info.node_id == "node-e2e-1"

    _register_exportable_replica(servicer, test_context, worker_id=child1, idx=2)
    complete = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=first.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
        ),
        test_context,
    )
    assert complete.status == global_store_pb2.STATUS_OK

    second = _request_broadcast_transport(
        servicer,
        test_context,
        session_id="session-e2e",
        worker_id=child2,
        idx=3,
        request_id="request-child-2",
    )

    assert second.remote_memory_info.node_id == "node-e2e-2"
