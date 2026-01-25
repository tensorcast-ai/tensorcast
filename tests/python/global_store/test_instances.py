#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from tensorcast.proto.global_store.v1 import global_store_pb2


def test_instance_lifecycle(servicer, test_context) -> None:
    register = global_store_pb2.RegisterInstanceRequest(
        instance_id="inst-1",
        daemon_id="daemon-1",
        engine="engine-x",
        signals_endpoint="http://localhost:9000",
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
    assert any(entry.instance_id == "inst-1" for entry in listed.instances)

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
