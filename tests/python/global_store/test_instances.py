#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from tensorcast.proto.global_store.v1 import global_store_pb2


def test_instance_lifecycle(servicer, test_context) -> None:
    register = global_store_pb2.RegisterInstanceRequest(
        instance_id="inst-1",
        daemon_id="daemon-1",
        engine="engine-x",
        signals_endpoint="http://localhost:9000",
        execution_endpoint="127.0.0.1:7001",
        execution_host_kind="node_agent_grpc",
        labels={"role": "inference"},
    )
    resp = servicer.RegisterInstance(register, test_context)
    assert resp.status == global_store_pb2.Status.STATUS_OK
    assert resp.instance_id == "inst-1"

    hb = global_store_pb2.InstanceHeartbeatRequest(instance_id="inst-1")
    hb_resp = servicer.InstanceHeartbeat(hb, test_context)
    assert hb_resp.status == global_store_pb2.Status.STATUS_OK

    listed = servicer.ListActiveInstances(
        global_store_pb2.ListActiveInstancesRequest(), test_context
    )
    instance = next(
        entry for entry in listed.instances if entry.instance_id == "inst-1"
    )
    assert instance.execution_endpoint == "127.0.0.1:7001"
    assert instance.execution_host_kind == "node_agent_grpc"

    unreg = servicer.UnregisterInstance(
        global_store_pb2.UnregisterInstanceRequest(
            instance_id="inst-1", is_graceful_shutdown=True
        ),
        test_context,
    )
    assert unreg.status == global_store_pb2.Status.STATUS_OK

    listed_after = servicer.ListActiveInstances(
        global_store_pb2.ListActiveInstancesRequest(), test_context
    )
    assert all(entry.instance_id != "inst-1" for entry in listed_after.instances)


def test_node_agent_instance_requires_execution_endpoint(
    servicer, test_context
) -> None:
    request = global_store_pb2.RegisterInstanceRequest(
        instance_id="inst-node-agent",
        daemon_id="daemon-1",
        engine="engine-x",
        capability_flags=(
            1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_NODE_AGENT_ENABLED
        ),
    )

    response = servicer.RegisterInstance(request, test_context)

    assert response.status == global_store_pb2.Status.STATUS_ERROR
