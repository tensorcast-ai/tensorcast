from __future__ import annotations

from tensorcast.proto.global_store.v1 import global_store_pb2


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
    assert response.session.state == global_store_pb2.BROADCAST_SESSION_STATE_ACTIVE
    edge_resp = servicer.ListBroadcastEdges(
        global_store_pb2.ListBroadcastEdgesRequest(session_id="session-rpc"),
        test_context,
    )
    assert edge_resp.status == global_store_pb2.STATUS_OK
    assert len(edge_resp.edges) == 1
    assert edge_resp.edges[0].child_worker_id == child_worker
