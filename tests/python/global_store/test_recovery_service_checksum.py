#  Copyright (c) 2025-2026, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from tensorcast.global_store.models import MemoryType, Replica, Worker
from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import get_config, set_config
from tensorcast.global_store.services.recovery_service import RecoveryService
from tensorcast.global_store.services.worker_service import WorkerService
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _fnv1a_64(value: str) -> str:
    hash_val = 0xCBF29CE484222325
    fnv_prime = 0x100000001B3
    for b in value.encode():
        hash_val ^= b
        hash_val = (hash_val * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_val:016x}"


def _make_recovery_service(repositories) -> RecoveryService:
    try:
        get_config()
    except RuntimeError:
        set_config(GlobalStoreConfig())
    worker_service = WorkerService(repositories["worker"], repositories["replica"])
    return RecoveryService(
        repositories["worker"],
        repositories["replica"],
        worker_service,
    )


def test_checksum_format_matches_daemon(repositories):
    service = _make_recovery_service(repositories)

    replicas = [
        Replica(
            artifact_id="artifact-format",
            node_id="node-format",
            node_address="10.0.0.1",
            node_port=50051,
            memory_size=256,
            memory_type=MemoryType.RAM,
            device_id=0,
            worker_id="worker-format",
            is_available=True,
        )
    ]

    expected_state = "artifact-format::node-format:10.0.0.1:50051:0:RAM:1;"
    wrong_state = "artifact-format:RAM:0:1:node-format:10.0.0.1:50051;"

    assert service._compute_state_checksum(replicas) == _fnv1a_64(expected_state)
    assert service._compute_state_checksum(replicas) != _fnv1a_64(wrong_state)


def test_checksum_includes_view_identity(repositories):
    service = _make_recovery_service(repositories)

    canonical = Replica(
        artifact_id="artifact-view",
        node_id="node-view",
        node_address="10.0.0.1",
        node_port=50051,
        memory_size=256,
        memory_type=MemoryType.RAM,
        device_id=0,
        worker_id="worker-view",
        is_available=True,
    )
    view = Replica(
        artifact_id="artifact-view",
        node_id="node-view",
        node_address="10.0.0.1",
        node_port=50051,
        memory_size=256,
        memory_type=MemoryType.RAM,
        device_id=0,
        worker_id="worker-view",
        is_available=True,
    )
    view.byte_space = view.byte_space.view("view-1")

    checksum_canonical = service._compute_state_checksum([canonical])
    checksum_view = service._compute_state_checksum([view])
    assert checksum_canonical != checksum_view


def test_checksum_is_stable_and_availability_sensitive(repositories):
    service = _make_recovery_service(repositories)

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
    recovery = _make_recovery_service(repositories)
    worker_id = "worker-force"
    repositories["worker"].create_or_update(
        Worker(
            worker_id=worker_id,
            daemon_id="daemon-force",
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
        state_version=1,
        state_checksum="",
    )

    success, changes, new_version, checksum, ignored = (
        recovery.synchronize_worker_state(worker_id, local_state, 1, 1, True)
    )

    assert success is True
    assert ignored is False
    assert any(
        change.type == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA
        for change in changes
    )
    assert sum(
        1 for change in changes if change.type == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA
    ) == len(replicas)
    assert new_version >= 1


def test_availability_drift_triggers_update(repositories):
    recovery = _make_recovery_service(repositories)
    worker_id = "worker-availability"

    repositories["worker"].create_or_update(
        Worker(
            worker_id=worker_id,
            daemon_id="daemon-availability",
            node_id="node-availability",
            node_address="10.0.0.10",
            grpc_port=50051,
            p2p_port=65090,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        )
    )

    replica = Replica(
        artifact_id="artifact-availability",
        node_id="node-availability",
        node_address="10.0.0.10",
        node_port=50051,
        memory_size=512,
        memory_type=MemoryType.GPU,
        device_id=1,
        worker_id=worker_id,
        is_available=True,
    )
    repositories["replica"].create(replica)

    local_state = global_store_pb2.WorkerLocalState(
        worker_id=worker_id,
        state_version=1,
        state_checksum="",
    )
    local_replica = local_state.local_replicas.add()
    local_replica.ref.artifact_id = replica.artifact_id
    local_replica.memory_info.node_id = replica.node_id
    local_replica.memory_info.memory_type = common_pb2.MEMORY_TYPE_GPU
    local_replica.memory_info.device_id = replica.device_id
    local_replica.stats.is_available = False

    original_checksum = recovery._compute_state_checksum([replica])

    success, changes, new_version, new_checksum, ignored = (
        recovery.synchronize_worker_state(worker_id, local_state, 1, 1, False)
    )

    assert success is True
    assert ignored is False
    assert new_version == 2
    assert any(
        change.type == global_store_pb2.StateChange.CHANGE_TYPE_UPDATE_REPLICA
        for change in changes
    )

    updated_replicas = repositories["replica"].get_replicas_by_worker(worker_id)
    assert updated_replicas[0].is_available is False

    expected_checksum = recovery._compute_state_checksum(
        [
            Replica(
                artifact_id=replica.artifact_id,
                node_id=replica.node_id,
                node_address=replica.node_address,
                node_port=replica.node_port,
                memory_size=replica.memory_size,
                memory_type=replica.memory_type,
                device_id=replica.device_id,
                worker_id=worker_id,
                is_available=False,
            )
        ]
    )
    assert new_checksum == expected_checksum


def test_stale_sync_token_is_ignored(repositories):
    recovery = _make_recovery_service(repositories)
    worker_id = "worker-stale"
    repositories["worker"].create_or_update(
        Worker(
            worker_id=worker_id,
            daemon_id="daemon-stale",
            node_id="node-stale",
            node_address="10.0.0.30",
            grpc_port=50051,
            p2p_port=65090,
            mem_pool_total_size=2048,
            mem_pool_available_size=2048,
        )
    )

    local_state = global_store_pb2.WorkerLocalState(
        worker_id=worker_id,
        state_version=1,
        state_checksum="",
    )

    success, changes, new_version, checksum, ignored = (
        recovery.synchronize_worker_state(worker_id, local_state, 1, 1, False)
    )
    assert success is True
    assert ignored is False
    first_checksum = checksum

    success, changes, new_version, checksum, ignored = (
        recovery.synchronize_worker_state(worker_id, local_state, 1, 1, False)
    )
    assert success is True
    assert ignored is True
    assert checksum == first_checksum


def test_endpoint_drift_triggers_update(repositories):
    recovery = _make_recovery_service(repositories)
    worker_id = "worker-endpoint"

    repositories["worker"].create_or_update(
        Worker(
            worker_id=worker_id,
            daemon_id="daemon-endpoint",
            node_id="node-endpoint",
            node_address="10.0.0.10",
            grpc_port=50051,
            p2p_port=65090,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        )
    )

    replica = Replica(
        artifact_id="artifact-endpoint",
        node_id="node-endpoint",
        node_address="10.0.0.10",
        node_port=50051,
        memory_size=512,
        memory_type=MemoryType.GPU,
        device_id=0,
        worker_id=worker_id,
        is_available=True,
    )
    repositories["replica"].create(replica)

    local_state = global_store_pb2.WorkerLocalState(
        worker_id=worker_id,
        state_version=1,
        state_checksum="",
    )
    local_replica = local_state.local_replicas.add()
    local_replica.ref.artifact_id = replica.artifact_id
    local_replica.memory_info.node_id = replica.node_id
    local_replica.memory_info.node_address = "10.0.0.20"
    local_replica.memory_info.node_port = 55000
    local_replica.memory_info.memory_type = common_pb2.MEMORY_TYPE_GPU
    local_replica.memory_info.device_id = replica.device_id
    local_replica.memory_info.memory_size = replica.memory_size
    local_replica.stats.is_available = replica.is_available

    success, changes, new_version, new_checksum, ignored = (
        recovery.synchronize_worker_state(worker_id, local_state, 1, 1, False)
    )

    assert success is True
    assert ignored is False
    assert new_version == 2
    assert any(
        change.type == global_store_pb2.StateChange.CHANGE_TYPE_UPDATE_REPLICA
        for change in changes
    )

    updated_replicas = repositories["replica"].get_replicas_by_worker(worker_id)
    assert updated_replicas[0].node_address == "10.0.0.20"
    assert updated_replicas[0].node_port == 55000

    expected_checksum = recovery._compute_state_checksum(
        [
            Replica(
                artifact_id=replica.artifact_id,
                node_id=replica.node_id,
                node_address="10.0.0.20",
                node_port=55000,
                memory_size=replica.memory_size,
                memory_type=replica.memory_type,
                device_id=replica.device_id,
                worker_id=worker_id,
                is_available=True,
            )
        ]
    )
    assert new_checksum == expected_checksum
