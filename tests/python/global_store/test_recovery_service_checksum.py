#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from tensorcast.global_store.models import MemoryType, Replica, Worker
from tensorcast.global_store.services.recovery_service import RecoveryService
from tensorcast.proto.global_store.v1 import global_store_pb2


def test_checksum_is_stable_and_availability_sensitive(repositories):
    service = RecoveryService(repositories["worker"], repositories["replica"])

    replicas = [
        Replica(
            artifact_id="artifact-a",
            node_id="node-1",
            node_address="10.0.0.1",
            node_port=50051,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id="worker-1",
            is_available=True,
        ),
        Replica(
            artifact_id="artifact-b",
            node_id="node-1",
            node_address="10.0.0.1",
            node_port=50051,
            memory_size=2048,
            memory_type=MemoryType.RAM,
            device_id=0,
            worker_id="worker-1",
            is_available=True,
        ),
    ]

    checksum1 = service._compute_state_checksum(replicas)
    # Order should not matter once sorted
    checksum2 = service._compute_state_checksum(list(reversed(replicas)))
    assert checksum1 == checksum2

    replicas[0].is_available = False
    checksum3 = service._compute_state_checksum(replicas)
    assert checksum3 != checksum1


def test_force_full_sync_allows_empty_inventory_removal(repositories):
    recovery = RecoveryService(repositories["worker"], repositories["replica"])
    worker_id = "worker-force"
    repositories["worker"].create_or_update(
        Worker(
            worker_id=worker_id,
            node_id="node-force",
            node_address="10.0.0.2",
            grpc_port=50051,
            p2p_port=65090,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        )
    )

    replicas = [
        Replica(
            artifact_id="artifact-x",
            node_id="node-force",
            node_address="10.0.0.2",
            node_port=50051,
            memory_size=512,
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id=worker_id,
        ),
        Replica(
            artifact_id="artifact-y",
            node_id="node-force",
            node_address="10.0.0.2",
            node_port=50051,
            memory_size=256,
            memory_type=MemoryType.RAM,
            device_id=0,
            worker_id=worker_id,
        ),
    ]
    for replica in replicas:
        repositories["replica"].create_or_update(replica)

    local_state = global_store_pb2.WorkerLocalState(
        worker_id=worker_id,
        state_version=0,
        state_checksum="",
    )

    success, changes, new_version, checksum = recovery.synchronize_worker_state(
        worker_id, local_state, True
    )

    assert success is True
    assert any(
        change.type == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA
        for change in changes
    )
    assert sum(
        1 for change in changes if change.type == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA
    ) == len(replicas)
    assert new_version >= 1
