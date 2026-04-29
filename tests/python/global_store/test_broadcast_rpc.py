from __future__ import annotations

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def test_broadcast_proto_contract_uses_byte_space_and_targets():
    session_fields = global_store_pb2.BroadcastSessionInfo.DESCRIPTOR.fields_by_name
    create_fields = (
        global_store_pb2.CreateBroadcastSessionRequest.DESCRIPTOR.fields_by_name
    )
    create_response_fields = (
        global_store_pb2.CreateBroadcastSessionResponse.DESCRIPTOR.fields_by_name
    )
    get_response_fields = (
        global_store_pb2.GetBroadcastSessionResponse.DESCRIPTOR.fields_by_name
    )
    cancel_fields = global_store_pb2.CancelBroadcastSessionRequest.DESCRIPTOR.fields_by_name

    assert "requested_view_id" not in session_fields
    assert session_fields["requested_byte_space"].number == 3
    assert (
        session_fields["requested_byte_space"].message_type.full_name
        == "tensorcast.common.v1.ByteSpaceRef"
    )
    assert "requested_view_id" not in create_fields
    assert create_fields["requested_byte_space"].number == 3
    assert (
        create_fields["requested_byte_space"].message_type.full_name
        == "tensorcast.common.v1.ByteSpaceRef"
    )
    assert create_response_fields["targets"].number == 3
    assert create_response_fields["edges"].number == 4
    assert get_response_fields["targets"].number == 3
    assert cancel_fields["reason"].number == 2


def test_create_broadcast_session_rpc_returns_edges(servicer, test_context, memory_info):
    root_worker = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id="daemon-root",
            node_id="node-root",
            node_address="10.10.0.1",
            grpc_port=50101,
            p2p_port=50102,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        test_context,
    ).worker_id
    child_worker = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id="daemon-child",
            node_id="node-child",
            node_address="10.10.0.2",
            grpc_port=50201,
            p2p_port=50202,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        test_context,
    ).worker_id
    memory_info.node_id = "node-root"
    memory_info.node_address = "10.10.0.1"
    memory_info.node_port = 50102
    register_resp = servicer.RegisterReplica(
        global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:model-rpc",
            worker_id=root_worker,
            mem_info=memory_info,
            max_concurrency=4,
        ),
        test_context,
    )
    assert register_resp.status == global_store_pb2.STATUS_OK

    response = servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id="session-rpc",
            artifact_id="mi2:model-rpc",
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_VIEW,
                id="view-rpc",
            ),
            epoch=7,
            fanout=1,
            strict_parent=True,
            max_attempts=3,
            root_replica_id=register_resp.replica_id,
            targets=[
                global_store_pb2.BroadcastTargetIdentity(
                    worker_id=child_worker,
                    daemon_id="daemon-child",
                )
            ],
        ),
        test_context,
    )

    assert response.status == global_store_pb2.STATUS_OK
    assert response.session.session_id == "session-rpc"
    assert response.session.requested_byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
    assert response.session.requested_byte_space.id == "view-rpc"
    assert response.session.state == global_store_pb2.BROADCAST_SESSION_STATE_ACTIVE
    assert len(response.targets) == 1
    assert response.targets[0].target_worker_id == child_worker
    get_resp = servicer.GetBroadcastSession(
        global_store_pb2.GetBroadcastSessionRequest(session_id="session-rpc"),
        test_context,
    )
    assert get_resp.status == global_store_pb2.STATUS_OK
    assert len(get_resp.targets) == 1
    assert get_resp.targets[0].target_worker_id == child_worker
    edge_resp = servicer.ListBroadcastEdges(
        global_store_pb2.ListBroadcastEdgesRequest(session_id="session-rpc"),
        test_context,
    )
    assert edge_resp.status == global_store_pb2.STATUS_OK
    assert len(edge_resp.edges) == 1
    assert edge_resp.edges[0].child_worker_id == child_worker


def test_create_broadcast_session_accepts_worker_only_and_daemon_only_targets(
    servicer,
    test_context,
    memory_info,
):
    root_worker = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id="daemon-root-mixed",
            node_id="node-root-mixed",
            node_address="10.20.0.1",
            grpc_port=51101,
            p2p_port=51102,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        test_context,
    ).worker_id
    worker_only_target = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id="daemon-worker-only",
            node_id="node-worker-only",
            node_address="10.20.0.2",
            grpc_port=51201,
            p2p_port=51202,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        test_context,
    ).worker_id
    daemon_only_target = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id="daemon-daemon-only",
            node_id="node-daemon-only",
            node_address="10.20.0.3",
            grpc_port=51301,
            p2p_port=51302,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        test_context,
    ).worker_id
    memory_info.node_id = "node-root-mixed"
    memory_info.node_address = "10.20.0.1"
    memory_info.node_port = 51102
    register_resp = servicer.RegisterReplica(
        global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:model-mixed-targets",
            worker_id=root_worker,
            mem_info=memory_info,
            max_concurrency=4,
        ),
        test_context,
    )
    assert register_resp.status == global_store_pb2.STATUS_OK

    response = servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id="session-mixed-targets",
            artifact_id="mi2:model-mixed-targets",
            epoch=9,
            fanout=2,
            strict_parent=True,
            max_attempts=3,
            root_replica_id=register_resp.replica_id,
            targets=[
                global_store_pb2.BroadcastTargetIdentity(
                    worker_id=worker_only_target,
                ),
                global_store_pb2.BroadcastTargetIdentity(
                    daemon_id="daemon-daemon-only",
                ),
            ],
        ),
        test_context,
    )

    assert response.status == global_store_pb2.STATUS_OK
    assert {target.target_worker_id for target in response.targets} == {
        worker_only_target,
        daemon_only_target,
    }
