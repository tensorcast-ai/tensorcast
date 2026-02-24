#  Copyright (c) 2025-2026, TensorCast Team.

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import get_config, set_config
from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.global_store.models import MemoryType, Replica, Worker
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


def _make_inventory_replica(
    *,
    artifact_id: str,
    node_id: str,
    memory_type: common_pb2.MemoryType.ValueType,
    device_id: int,
    node_address: str = "",
    node_port: int = 0,
    memory_size: int = 0,
    is_available: bool = True,
) -> common_pb2.ReplicaInfo:
    replica = common_pb2.ReplicaInfo()
    replica.ref.artifact_id = artifact_id
    replica.memory_info.node_id = node_id
    replica.memory_info.memory_type = memory_type
    replica.memory_info.device_id = device_id
    if node_address:
        replica.memory_info.node_address = node_address
    if node_port > 0:
        replica.memory_info.node_port = node_port
    if memory_size > 0:
        replica.memory_info.memory_size = memory_size
    replica.stats.is_available = is_available
    return replica


def _reconcile(
    *,
    recovery: RecoveryService,
    worker_id: str,
    generation: int,
    request_seq: int,
    inventory: list[common_pb2.ReplicaInfo],
    request_kind: global_store_pb2.ReconcileRequestKind.ValueType,
    daemon_id: str = "",
):
    return recovery.reconcile_worker_state(
        worker_id=worker_id,
        daemon_id=daemon_id,
        generation=generation,
        request_seq=request_seq,
        inventory=inventory,
        request_kind=request_kind,
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
    checksum2 = service._compute_state_checksum(list(reversed(replicas)))
    assert checksum1 == checksum2

    replicas[0].is_available = False
    checksum3 = service._compute_state_checksum(replicas)
    assert checksum3 != checksum1


def test_snapshot_request_allows_empty_inventory_removal(repositories):
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

    result_kind, new_version, _, state_changes, _, _ = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=1,
        inventory=[],
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
    )

    assert result_kind == global_store_pb2.RECONCILE_RESULT_KIND_APPLIED
    assert sum(
        1
        for change in state_changes
        if change.type == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA
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

    inventory = [
        _make_inventory_replica(
            artifact_id=replica.artifact_id,
            node_id=replica.node_id,
            memory_type=common_pb2.MEMORY_TYPE_GPU,
            device_id=replica.device_id,
            is_available=False,
        )
    ]
    result_kind, new_version, new_checksum, state_changes, _, _ = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=1,
        inventory=inventory,
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_DELTA,
    )

    assert result_kind == global_store_pb2.RECONCILE_RESULT_KIND_APPLIED
    assert new_version == 2
    assert any(
        change.type == global_store_pb2.StateChange.CHANGE_TYPE_UPDATE_REPLICA
        for change in state_changes
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


def test_reconcile_replay_is_idempotent_and_stale_lower_seq_is_ignored(repositories):
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

    first = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=1,
        inventory=[],
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_DELTA,
    )
    assert first[0] in (
        global_store_pb2.RECONCILE_RESULT_KIND_APPLIED,
        global_store_pb2.RECONCILE_RESULT_KIND_NOOP,
    )

    second = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=2,
        inventory=[],
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_DELTA,
    )
    assert second[0] in (
        global_store_pb2.RECONCILE_RESULT_KIND_APPLIED,
        global_store_pb2.RECONCILE_RESULT_KIND_NOOP,
    )

    replay = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=2,
        inventory=[],
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_DELTA,
    )
    assert replay[0] == global_store_pb2.RECONCILE_RESULT_KIND_NOOP

    stale = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=1,
        inventory=[],
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_DELTA,
    )
    assert stale[0] == global_store_pb2.RECONCILE_RESULT_KIND_IGNORED_STALE


def test_snapshot_noop_fast_path_uses_cache_and_short_circuits_replay(
    repositories, monkeypatch
):
    recovery = _make_recovery_service(repositories)
    worker_id = "worker-noop-fast-path"
    repositories["worker"].create_or_update(
        Worker(
            worker_id=worker_id,
            daemon_id="daemon-noop-fast-path",
            node_id="node-noop-fast-path",
            node_address="10.0.0.31",
            grpc_port=50051,
            p2p_port=65090,
            mem_pool_total_size=2048,
            mem_pool_available_size=2048,
        )
    )
    inventory = [
        _make_inventory_replica(
            artifact_id="artifact-noop-fast-path",
            node_id="node-noop-fast-path",
            node_address="10.0.0.31",
            node_port=50051,
            memory_type=common_pb2.MEMORY_TYPE_GPU,
            device_id=0,
            memory_size=4096,
            is_available=True,
        )
    ]

    first = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=1,
        inventory=inventory,
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
        daemon_id="daemon-noop-fast-path",
    )
    assert first[0] == global_store_pb2.RECONCILE_RESULT_KIND_APPLIED

    def _fail_global_replica_scan(*args, **kwargs):
        del args, kwargs
        raise AssertionError("fast NOOP should not scan replicas")

    monkeypatch.setattr(
        recovery.replica_repository,
        "get_replicas_by_worker_atomic",
        _fail_global_replica_scan,
    )

    second = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=2,
        inventory=inventory,
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
        daemon_id="daemon-noop-fast-path",
    )
    assert second[0] == global_store_pb2.RECONCILE_RESULT_KIND_NOOP

    def _fail_transaction(*args, **kwargs):
        del args, kwargs
        raise AssertionError("replay should be served from cache without transaction")

    monkeypatch.setattr(recovery.worker_repository, "transaction", _fail_transaction)
    replay = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=2,
        inventory=inventory,
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
        daemon_id="daemon-noop-fast-path",
    )
    assert replay[0] == global_store_pb2.RECONCILE_RESULT_KIND_NOOP


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

    inventory = [
        _make_inventory_replica(
            artifact_id=replica.artifact_id,
            node_id=replica.node_id,
            node_address="10.0.0.20",
            node_port=55000,
            memory_type=common_pb2.MEMORY_TYPE_GPU,
            device_id=replica.device_id,
            memory_size=replica.memory_size,
            is_available=True,
        )
    ]
    result_kind, new_version, new_checksum, state_changes, _, _ = _reconcile(
        recovery=recovery,
        worker_id=worker_id,
        generation=1,
        request_seq=1,
        inventory=inventory,
        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_DELTA,
    )

    assert result_kind == global_store_pb2.RECONCILE_RESULT_KIND_APPLIED
    assert new_version == 2
    assert any(
        change.type == global_store_pb2.StateChange.CHANGE_TYPE_UPDATE_REPLICA
        for change in state_changes
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


def test_transient_reconcile_conflict_exhausted_returns_retry_later(
    repositories,
    monkeypatch,
):
    recovery = _make_recovery_service(repositories)
    worker_id = "worker-conflict-retry-later"

    repositories["worker"].create_or_update(
        Worker(
            worker_id=worker_id,
            daemon_id="daemon-conflict-retry-later",
            node_id="node-conflict-retry-later",
            node_address="10.0.0.40",
            grpc_port=50051,
            p2p_port=65090,
            mem_pool_total_size=2048,
            mem_pool_available_size=2048,
        )
    )

    def _always_conflict(**kwargs):
        del kwargs
        raise DatabaseError(
            "Transaction failed: TransactionContext Error: Conflict on tuple deletion!"
        )

    monkeypatch.setattr(
        recovery,
        "_reconcile_worker_state_internal",
        _always_conflict,
    )

    result_kind, version, checksum, state_changes, expected_replicas, retry_after_ms = (
        _reconcile(
            recovery=recovery,
            worker_id=worker_id,
            generation=1,
            request_seq=1,
            inventory=[],
            request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
            daemon_id="daemon-conflict-retry-later",
        )
    )

    assert result_kind == global_store_pb2.RECONCILE_RESULT_KIND_RETRY_LATER
    assert version >= 1
    assert isinstance(checksum, str)
    assert state_changes == []
    assert expected_replicas == []
    assert retry_after_ms == 500
