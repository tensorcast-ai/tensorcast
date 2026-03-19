#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for Global Store service layer."""

import base64
import hashlib
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from tensorcast.global_store import metrics
from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import set_config
from tensorcast.global_store.exceptions import (
    DatabaseError,
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from tensorcast.global_store.models import (
    ByteSpaceRef,
    ExportState,
    MemoryType,
    PendingTransportRequest,
    PendingTransportState,
    Replica,
    Transport,
    TransportCompletionOutcome,
    TransportSchedulingGroup,
    Worker,
)
from tensorcast.global_store.repositories.transport_repository import (
    TransportGroupProgress,
)
from tensorcast.global_store.services import (
    ArtifactService,
    TransportService,
    WorkerService,
)
from tensorcast.global_store.services.view_state_service import (
    PROOF_SCHEMA_V1,
    LeafWritePayload,
    PieceProofDigestPayload,
    ViewUpsertPayload,
)
from tensorcast.proto.layout.v1 import layout_pb2


class TestServices:
    """Test service layer."""

    def test_worker_service_registration(self, services):
        """Test worker registration logic."""
        worker_service = services["worker"]

        # Register new worker
        worker = Worker(
            daemon_id="daemon_node1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        )

        registered = worker_service.register_worker(worker)
        assert registered.worker_id.startswith("worker_node1_")

        # Register again (should update)
        worker2 = Worker(
            daemon_id="daemon_node1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50053,  # Different comm port
            mem_pool_total_size=2048,
            mem_pool_available_size=2048,
        )

        updated = worker_service.register_worker(worker2)
        assert updated.worker_id == registered.worker_id
        assert updated.p2p_port == 50053

    def test_worker_service_registration_daemon_id_identity(self, services):
        """Test daemon_id-based identity (address/port may change)."""
        worker_service = services["worker"]

        registered = worker_service.register_worker(
            Worker(
                daemon_id="daemon_a",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        assert registered.daemon_id == "daemon_a"

        updated = worker_service.register_worker(
            Worker(
                daemon_id="daemon_a",
                node_id="node1",
                node_address="192.168.1.2",
                grpc_port=50055,
                p2p_port=50056,
                mem_pool_total_size=2048,
                mem_pool_available_size=2048,
            )
        )
        assert updated.worker_id == registered.worker_id
        assert updated.daemon_id == "daemon_a"
        assert updated.node_address == "192.168.1.2"
        assert updated.grpc_port == 50055

    def test_worker_service_registration_reclaims_inactive_endpoint_conflict(
        self, services
    ):
        """Rebind daemon_id when endpoint is blocked by another inactive row."""
        worker_service = services["worker"]
        worker_repo = worker_service.worker_repository

        stable = worker_service.register_worker(
            Worker(
                daemon_id="daemon_rebind",
                node_id="node_a",
                node_address="192.168.1.10",
                grpc_port=50071,
                p2p_port=50072,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        assert worker_repo.mark_inactive(stable.worker_id)

        stale = worker_service.register_worker(
            Worker(
                daemon_id="daemon_stale",
                node_id="node_b",
                node_address="192.168.1.20",
                grpc_port=50081,
                p2p_port=50082,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        assert worker_repo.mark_inactive(stale.worker_id)

        rebound = worker_service.register_worker(
            Worker(
                daemon_id="daemon_rebind",
                node_id="node_c",
                node_address="192.168.1.20",
                grpc_port=50081,
                p2p_port=50083,
                mem_pool_total_size=2048,
                mem_pool_available_size=2048,
            )
        )

        assert rebound.worker_id == stable.worker_id
        assert rebound.daemon_id == "daemon_rebind"
        assert rebound.node_address == "192.168.1.20"
        assert rebound.grpc_port == 50081
        assert rebound.p2p_port == 50083
        assert worker_repo.find_by_id(stale.worker_id, include_inactive=True) is None

    def test_worker_service_registration_reclaims_active_endpoint_takeover(
        self, services
    ):
        """A new daemon can take over an active endpoint row after unclean exit."""
        worker_service = services["worker"]
        worker_repo = worker_service.worker_repository

        previous = worker_service.register_worker(
            Worker(
                daemon_id="daemon_prev",
                node_id="node_prev",
                node_address="192.168.1.30",
                grpc_port=50091,
                p2p_port=50092,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        successor = worker_service.register_worker(
            Worker(
                daemon_id="daemon_next",
                node_id="node_next",
                node_address="192.168.1.30",
                grpc_port=50091,
                p2p_port=50093,
                mem_pool_total_size=2048,
                mem_pool_available_size=2048,
            )
        )

        assert successor.worker_id != previous.worker_id
        assert successor.daemon_id == "daemon_next"
        assert successor.node_address == "192.168.1.30"
        assert successor.grpc_port == 50091
        assert successor.p2p_port == 50093
        assert worker_repo.find_by_id(previous.worker_id, include_inactive=True) is None

    def test_worker_service_registration_daemon_rebind_reclaims_active_endpoint(
        self, services
    ):
        """Stable daemon_id rebind can reclaim an active endpoint row."""
        worker_service = services["worker"]
        worker_repo = worker_service.worker_repository

        stable = worker_service.register_worker(
            Worker(
                daemon_id="daemon_stable_rebind",
                node_id="node_a",
                node_address="192.168.1.40",
                grpc_port=50101,
                p2p_port=50102,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        blocker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_blocker",
                node_id="node_b",
                node_address="192.168.1.41",
                grpc_port=50111,
                p2p_port=50112,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        rebound = worker_service.register_worker(
            Worker(
                daemon_id="daemon_stable_rebind",
                node_id="node_c",
                node_address="192.168.1.41",
                grpc_port=50111,
                p2p_port=50113,
                mem_pool_total_size=2048,
                mem_pool_available_size=2048,
            )
        )

        assert rebound.worker_id == stable.worker_id
        assert rebound.daemon_id == "daemon_stable_rebind"
        assert rebound.node_address == "192.168.1.41"
        assert rebound.grpc_port == 50111
        assert rebound.p2p_port == 50113
        assert worker_repo.find_by_id(blocker.worker_id, include_inactive=True) is None

    def test_worker_service_validation(self, services):
        """Test worker validation."""
        worker_service = services["worker"]

        # Missing node_id
        with pytest.raises(ValidationError, match="Node ID is required"):
            worker_service.register_worker(Worker())

        # Missing daemon_id
        with pytest.raises(ValidationError, match="daemon_id is required"):
            worker_service.register_worker(
                Worker(
                    node_id="node1",
                    node_address="192.168.1.1",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )

        # Invalid port
        with pytest.raises(ValidationError, match="gRPC port must be between"):
            worker_service.register_worker(
                Worker(node_id="node1", node_address="192.168.1.1", grpc_port=0)
            )

        # Loopback address should be rejected
        with pytest.raises(ValidationError, match="Invalid node_address"):
            worker_service.register_worker(
                Worker(
                    node_id="node1",
                    node_address="127.0.0.1",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )

        # Loopback hostname should also be rejected
        with pytest.raises(ValidationError, match="Invalid node_address"):
            worker_service.register_worker(
                Worker(
                    node_id="node1",
                    node_address="localhost",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )

    def test_worker_service_heartbeat(self, services):
        """Test worker heartbeat processing."""
        worker_service = services["worker"]

        # Register worker first
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        # Send heartbeat
        success = worker_service.heartbeat(worker.worker_id, 512, True)
        assert success is True

        # Heartbeat for non-existent worker
        success = worker_service.heartbeat("nonexistent", 512, True)
        assert success is False

    def test_worker_service_unregistration(self, services):
        """Test worker unregistration."""
        worker_service = services["worker"]
        artifact_service = services["artifact"]

        # Register worker
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        # Register a replica for this worker
        artifact_service.register_replica(
            Replica(
                artifact_id="test_artifact",
                node_id="node1",
                node_address="192.168.1.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                worker_id=worker.worker_id,
            )
        )

        # Unregister worker
        success = worker_service.unregister_worker(worker.worker_id)
        assert success is True

        # Verify replica is marked unavailable
        found_replicas = artifact_service.list_replicas(artifact_id="test_artifact")
        assert len(found_replicas) == 1
        assert found_replicas[0].is_available is False

    def test_worker_service_list_active(self, services):
        """Test listing active workers."""
        worker_service = services["worker"]

        # Register multiple workers
        workers = []
        for i in range(3):
            worker = worker_service.register_worker(
                Worker(
                    daemon_id=f"daemon_{i}",
                    node_id=f"node_{i}",
                    node_address=f"192.168.1.{i + 1}",
                    grpc_port=50051 + i,
                    p2p_port=50052 + i,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                    accepting_new_requests=(i % 2 == 0),
                )
            )
            workers.append(worker)

        # List all workers (including those not accepting new requests)
        all_workers = worker_service.list_active_workers(include_unavailable=True)
        assert len(all_workers) >= 3

        # List only accepting workers
        accepting_workers = worker_service.list_active_workers(
            include_unavailable=False
        )
        accepting_count = sum(1 for w in accepting_workers if w.accepting_new_requests)
        assert accepting_count >= 2

    def test_artifact_service_registration(self, services, repositories):
        """Test artifact replica registration."""
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Register worker first
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        # Register replica
        replica = Replica(
            artifact_id="test_artifact",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id=worker.worker_id,
        )

        registered = artifact_service.register_replica(replica)
        assert registered.replica_id is not None

        # Register again (should update)
        replica2 = Replica(
            artifact_id="test_artifact",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=2048,  # Different size
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id=worker.worker_id,
            max_concurrency=20,
        )

        updated = artifact_service.register_replica(replica2)
        assert updated.replica_id == registered.replica_id
        assert updated.memory_size == 2048
        assert updated.max_concurrency == 20

    def test_artifact_service_rejects_loopback_address(self, services):
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        with pytest.raises(ValidationError, match="Invalid node_address"):
            artifact_service.register_replica(
                Replica(
                    artifact_id="test_artifact",
                    node_id="node1",
                    node_address="127.0.0.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    worker_id=worker.worker_id,
                )
            )

    def test_artifact_service_unregistration(self, services):
        """Test artifact replica unregistration."""
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Register worker and replica
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        replica = artifact_service.register_replica(
            Replica(
                artifact_id="test_artifact",
                node_id="node1",
                node_address="192.168.1.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                worker_id=worker.worker_id,
            )
        )

        # Unregister replica
        success = artifact_service.unregister_replica(
            replica.replica_id, "test_artifact"
        )
        assert success is True

        # Verify replica is removed
        found_replicas = artifact_service.list_replicas(artifact_id="test_artifact")
        assert len(found_replicas) == 0

    def test_artifact_service_list_replicas(self, services):
        """Test listing artifact replicas."""
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Register worker
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        # Register multiple replicas
        for i in range(3):
            artifact_service.register_replica(
                Replica(
                    artifact_id=f"model_{i}",
                    node_id="node1",
                    node_address="192.168.1.1",
                    node_port=8080 + i,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=i,
                    worker_id=worker.worker_id,
                )
            )

        # List all replicas
        all_replicas = artifact_service.list_replicas()
        assert len(all_replicas) >= 3

        # List replicas for specific artifact
        replicas = artifact_service.list_replicas(artifact_id="model_0")
        assert len(replicas) == 1
        assert replicas[0].artifact_id == "model_0"

    def test_transport_service_request(self, services, repositories):
        """Test transport request logic."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Setup worker and replica
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        replica = artifact_service.register_replica(
            Replica(
                artifact_id="test_artifact",
                node_id="node1",
                node_address="192.168.1.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,
            )
        )

        # Request transport
        selected, transport_id = transport_service.request_transport(
            artifact_id="test_artifact",
            view_id=None,
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="transport-basic-1",
        )

        assert selected.replica_id == replica.replica_id
        assert selected.current_requests == 1

        transport_row = repositories["transport"].find_by_id(transport_id)
        assert transport_row is not None
        assert transport_row.replica_memory_size_bytes == replica.memory_size

        # Complete transport
        current, max_conc = transport_service.complete_transport(
            transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )
        assert current == 0
        assert max_conc == 2

    def test_transport_service_complete_is_idempotent(self, services):
        """Repeated completion must not decrement current_requests twice."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node_idempotent",
                node_id="node_idempotent",
                node_address="192.168.9.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        replica = artifact_service.register_replica(
            Replica(
                artifact_id="idempotent_artifact",
                node_id="node_idempotent",
                node_address="192.168.9.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        selected, transport_id = transport_service.request_transport(
            artifact_id="idempotent_artifact",
            view_id=None,
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="transport-idempotent-1",
        )
        assert selected.replica_id == replica.replica_id

        current, max_conc = transport_service.complete_transport(
            transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )
        assert current == 0
        assert max_conc == 1

        current2, max_conc2 = transport_service.complete_transport(
            transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )
        assert current2 == 0
        assert max_conc2 == 1

    def test_transport_service_request_idempotency_reuses_existing_transport(
        self, services
    ):
        """Duplicate request_id should reuse the same transport row and lease."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node_idempotency_request",
                node_id="node_idempotency_request",
                node_address="192.168.9.10",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        artifact_service.register_replica(
            Replica(
                artifact_id="idempotency_request_artifact",
                node_id="node_idempotency_request",
                node_address="192.168.9.10",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        first_replica, first_transport_id = transport_service.request_transport(
            artifact_id="idempotency_request_artifact",
            view_id=None,
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="transport-request-id-1",
        )
        second_replica, second_transport_id = transport_service.request_transport(
            artifact_id="idempotency_request_artifact",
            view_id=None,
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="transport-request-id-1",
        )

        assert first_transport_id == second_transport_id
        assert first_replica.replica_id == second_replica.replica_id
        assert second_replica.current_requests == 1

        current, max_conc = transport_service.complete_transport(
            first_transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )
        assert current == 0
        assert max_conc == 1

    def test_transport_service_request_id_rejects_payload_mismatch(self, services):
        """Same request_id with a different payload must fail deterministically."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node_request_id_conflict",
                node_id="node_request_id_conflict",
                node_address="192.168.9.30",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        artifact_service.register_replica(
            Replica(
                artifact_id="request_id_conflict_artifact",
                node_id="node_request_id_conflict",
                node_address="192.168.9.30",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,
            )
        )

        _, transport_id = transport_service.request_transport(
            artifact_id="request_id_conflict_artifact",
            view_id=None,
            source_node_id="source_node_a",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="shared-request-id",
        )
        with pytest.raises(
            ValidationError, match="already used with different payload"
        ):
            transport_service.request_transport(
                artifact_id="request_id_conflict_artifact",
                view_id=None,
                source_node_id="source_node_b",
                source_address="192.168.2.1",
                source_port=9090,
                request_id="shared-request-id",
            )

        transport_service.complete_transport(
            transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )

    def test_transport_service_request_id_recycles_after_terminal_failure(
        self, services, repositories
    ):
        """Completed failed/expired request_id should be recyclable for retry."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        transport_repo = repositories["transport"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node_request_id_recycle",
                node_id="node_request_id_recycle",
                node_address="192.168.9.31",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        artifact_service.register_replica(
            Replica(
                artifact_id="request_id_recycle_artifact",
                node_id="node_request_id_recycle",
                node_address="192.168.9.31",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,
            )
        )

        request_id = "shared-request-id-recycle"
        _, first_transport_id = transport_service.request_transport(
            artifact_id="request_id_recycle_artifact",
            view_id=None,
            source_node_id="source_node_a",
            source_address="192.168.2.1",
            source_port=9090,
            request_id=request_id,
        )
        transport_service.complete_transport(
            first_transport_id,
            outcome=TransportCompletionOutcome.EXPIRED,
            outcome_detail="simulated_expired",
        )

        _, second_transport_id = transport_service.request_transport(
            artifact_id="request_id_recycle_artifact",
            view_id=None,
            source_node_id="source_node_b",
            source_address="192.168.2.2",
            source_port=9091,
            request_id=request_id,
        )

        assert second_transport_id != first_transport_id
        first_transport = transport_repo.find_by_id(first_transport_id)
        assert first_transport is not None
        assert first_transport.request_id is None
        assert first_transport.completion_outcome == TransportCompletionOutcome.EXPIRED
        second_transport = transport_repo.find_by_id(second_transport_id)
        assert second_transport is not None
        assert second_transport.request_id == request_id

        transport_service.complete_transport(
            second_transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )

    def test_transport_service_recovers_stale_dispatched_pending_without_transport(
        self, services, repositories
    ):
        """Stale dispatched pending row must not block request_id retries."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        pending_repo = repositories["pending_transport_request"]
        replica_repo = repositories["replica"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node_stale_dispatched",
                node_id="node_stale_dispatched",
                node_address="192.168.9.32",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        artifact_service.register_replica(
            Replica(
                artifact_id="stale_dispatched_artifact",
                node_id="node_stale_dispatched",
                node_address="192.168.9.32",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        request_id = "stale-dispatched-request-1"
        stale_fingerprint = transport_service._build_request_fingerprint(
            artifact_id="stale_dispatched_artifact",
            view_id=None,
            source_node_id="stale_source",
            source_address="192.168.2.10",
            source_port=9100,
            requester_worker_id=None,
            scheduling_group=None,
        )
        with replica_repo.transaction() as tx:
            pending_repo.create_if_absent_with_cursor(
                PendingTransportRequest(
                    request_id=request_id,
                    request_fingerprint=stale_fingerprint,
                    artifact_id="stale_dispatched_artifact",
                    requested_view_id=None,
                    source_node_id="stale_source",
                    source_address="192.168.2.10",
                    source_port=9100,
                ),
                tx,
            )
            assert pending_repo.mark_dispatched(request_id, tx)

        _, transport_id = transport_service.request_transport(
            artifact_id="stale_dispatched_artifact",
            view_id=None,
            source_node_id="fresh_source",
            source_address="192.168.2.11",
            source_port=9101,
            request_id=request_id,
            wait_timeout_ms=200,
        )
        assert transport_id is not None
        pending = pending_repo.find_by_request_id(request_id)
        assert pending is not None
        assert pending.state == PendingTransportState.DISPATCHED
        transport_service.complete_transport(
            transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )

    def test_transport_service_group_progress_counts_success_only(self, services):
        """Group progress uses completion outcome SUCCESS only."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node_group_outcome",
                node_id="node_group_outcome",
                node_address="192.168.9.20",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        artifact_service.register_replica(
            Replica(
                artifact_id="group_outcome_artifact",
                node_id="node_group_outcome",
                node_address="192.168.9.20",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,
            )
        )

        group_failed = TransportSchedulingGroup(
            group_id="group-outcome",
            group_kind="tp_rank",
            total_parts=2,
            part_id="part-0",
            priority=0,
            epoch=1,
        )
        group_success = TransportSchedulingGroup(
            group_id="group-outcome",
            group_kind="tp_rank",
            total_parts=2,
            part_id="part-1",
            priority=0,
            epoch=1,
        )

        _, transport_failed_id = transport_service.request_transport(
            artifact_id="group_outcome_artifact",
            view_id=None,
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="group-outcome-failed",
            scheduling_group=group_failed,
        )
        transport_service.complete_transport(
            transport_failed_id,
            outcome=TransportCompletionOutcome.FAILED,
            outcome_detail="simulated_failure",
        )

        _, transport_success_id = transport_service.request_transport(
            artifact_id="group_outcome_artifact",
            view_id=None,
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="group-outcome-success",
            scheduling_group=group_success,
        )
        transport_service.complete_transport(
            transport_success_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )

        progress = transport_service.transport_repository.get_group_progress(
            group_kind="tp_rank",
            group_id="group-outcome",
            group_epoch=1,
            total_parts_hint=2,
        )
        assert progress.completed_parts == 1
        assert progress.total_parts == 2
        assert progress.completion_ratio == pytest.approx(0.5)

    def test_transport_service_no_replicas(self, services):
        """Test transport when no replicas exist."""
        transport_service = services["transport"]

        # No replicas available
        with pytest.raises(NotFoundError):
            transport_service.request_transport(
                artifact_id="nonexistent_artifact",
                view_id=None,
                source_node_id="source",
                source_address="192.168.1.1",
                source_port=8080,
                wait_timeout_ms=100,
                request_id="transport-no-replica-1",
            )

    def test_transport_service_timeout(self, services):
        """Test transport timeout when replicas exist but are all busy."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Setup worker and replica with low concurrency
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node_timeout_test",
                node_id="node_timeout_test",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        artifact_service.register_replica(
            Replica(
                artifact_id="test_timeout_artifact",
                node_id="node_timeout_test",
                node_address="192.168.1.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,  # Only 1 concurrent request
            )
        )

        # Request first transport (should succeed)
        _, transport_id = transport_service.request_transport(
            artifact_id="test_timeout_artifact",
            view_id=None,
            source_node_id="source_1",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="transport-timeout-first",
        )

        # Request second transport with short timeout (should fail with TimeoutError)
        with pytest.raises(TimeoutError):
            transport_service.request_transport(
                artifact_id="test_timeout_artifact",
                view_id=None,
                source_node_id="source_2",
                source_address="192.168.2.2",
                source_port=9091,
                wait_timeout_ms=100,  # Short timeout
                request_id="transport-timeout-second",
            )

        # Complete the first transport
        transport_service.complete_transport(
            transport_id,
            outcome=TransportCompletionOutcome.SUCCESS,
        )

    def test_transport_service_concurrency_limit(self, services):
        """Test concurrency limiting."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Setup worker and replica with low concurrency
        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_node1",
                node_id="node1",
                node_address="192.168.1.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        artifact_service.register_replica(
            Replica(
                artifact_id="test_artifact",
                node_id="node1",
                node_address="192.168.1.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,  # Only 2 concurrent requests
            )
        )

        # Request 2 transports (should succeed)
        transport_ids = []
        for i in range(2):
            _, transport_id = transport_service.request_transport(
                artifact_id="test_artifact",
                view_id=None,
                source_node_id=f"source_{i}",
                source_address="192.168.2.1",
                source_port=9090 + i,
                request_id=f"transport-concurrency-{i}",
            )
            transport_ids.append(transport_id)

        # Third request should timeout (no capacity)
        with pytest.raises(TimeoutError):
            transport_service.request_transport(
                artifact_id="test_artifact",
                view_id=None,
                source_node_id="source_3",
                source_address="192.168.2.1",
                source_port=9093,
                wait_timeout_ms=100,
                request_id="transport-concurrency-overflow",
            )

        # Complete one transport
        transport_service.complete_transport(
            transport_ids[0],
            outcome=TransportCompletionOutcome.SUCCESS,
        )

        # Now request should succeed
        _, transport_id = transport_service.request_transport(
            artifact_id="test_artifact",
            view_id=None,
            source_node_id="source_3",
            source_address="192.168.2.1",
            source_port=9093,
            request_id="transport-concurrency-retry",
        )
        assert transport_id is not None

    def test_transport_service_nonexistent_transport_completion(self, services):
        """Test completing a non-existent transport."""
        transport_service = services["transport"]

        # Try to complete non-existent transport
        with pytest.raises(NotFoundError):
            transport_service.complete_transport(
                "nonexistent_transport_id",
                outcome=TransportCompletionOutcome.SUCCESS,
            )

    def test_dispatch_rolls_back_claim_when_transport_already_exists(
        self, services, repositories
    ):
        """Dispatch must rollback replica claim when request_id transport already exists."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        pending_repo = repositories["pending_transport_request"]
        transport_repo = repositories["transport"]
        replica_repo = repositories["replica"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_dispatch_reuse",
                node_id="node_dispatch_reuse",
                node_address="192.168.11.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        replica = artifact_service.register_replica(
            Replica(
                artifact_id="dispatch_reuse_artifact",
                node_id="node_dispatch_reuse",
                node_address="192.168.11.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        request_id = "dispatch-reuse-request-1"
        request_fingerprint = transport_service._build_request_fingerprint(
            artifact_id="dispatch_reuse_artifact",
            view_id=None,
            source_node_id="source-node",
            source_address="192.168.12.1",
            source_port=9090,
            requester_worker_id=None,
            scheduling_group=None,
        )
        transport_repo.create(
            Transport(
                replica_id=replica.replica_id,
                artifact_id="dispatch_reuse_artifact",
                source_node_id="source-node",
                source_address="192.168.12.1",
                source_port=9090,
                request_id=request_id,
                request_fingerprint=request_fingerprint,
            )
        )

        with replica_repo.transaction() as tx:
            pending_repo.create_if_absent_with_cursor(
                PendingTransportRequest(
                    request_id=request_id,
                    request_fingerprint=request_fingerprint,
                    artifact_id="dispatch_reuse_artifact",
                    requested_view_id=None,
                    source_node_id="source-node",
                    source_address="192.168.12.1",
                    source_port=9090,
                ),
                tx,
            )
            dispatched = transport_service._dispatch_pending_requests(tx=tx)

        assert dispatched == 1
        assert replica_repo.get_current_requests(replica.replica_id) == 0
        pending = pending_repo.find_by_request_id(request_id)
        assert pending is not None
        assert pending.state == PendingTransportState.DISPATCHED

    def test_dispatch_rolls_back_claim_on_mark_dispatched_race(
        self, services, repositories, monkeypatch
    ):
        """Dispatch race rollback should delete newly-created transport and release claim."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        pending_repo = repositories["pending_transport_request"]
        transport_repo = repositories["transport"]
        replica_repo = repositories["replica"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_dispatch_race",
                node_id="node_dispatch_race",
                node_address="192.168.21.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        replica = artifact_service.register_replica(
            Replica(
                artifact_id="dispatch_race_artifact",
                node_id="node_dispatch_race",
                node_address="192.168.21.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        request_id = "dispatch-race-request-1"
        request_fingerprint = transport_service._build_request_fingerprint(
            artifact_id="dispatch_race_artifact",
            view_id=None,
            source_node_id="source-node",
            source_address="192.168.22.1",
            source_port=9090,
            requester_worker_id=None,
            scheduling_group=None,
        )

        with replica_repo.transaction() as tx:
            pending_repo.create_if_absent_with_cursor(
                PendingTransportRequest(
                    request_id=request_id,
                    request_fingerprint=request_fingerprint,
                    artifact_id="dispatch_race_artifact",
                    requested_view_id=None,
                    source_node_id="source-node",
                    source_address="192.168.22.1",
                    source_port=9090,
                ),
                tx,
            )

        monkeypatch.setattr(
            pending_repo, "mark_dispatched", lambda _request_id, _tx: False
        )

        with replica_repo.transaction() as tx:
            dispatched = transport_service._dispatch_pending_requests(tx=tx)

        assert dispatched == 0
        assert replica_repo.get_current_requests(replica.replica_id) == 0
        pending = pending_repo.find_by_request_id(request_id)
        assert pending is not None
        assert pending.state == PendingTransportState.ENQUEUED
        assert transport_repo.find_by_request_id(request_id) is None

    def test_cleanup_expired_transports_forces_completion_and_releases_capacity(
        self, services, repositories
    ):
        """cleanup_expired_transports must force-complete stale in-flight rows."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        replica_repo = repositories["replica"]
        transport_repo = repositories["transport"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_cleanup_expired",
                node_id="node_cleanup_expired",
                node_address="192.168.31.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        replica = artifact_service.register_replica(
            Replica(
                artifact_id="cleanup_expired_artifact",
                node_id="node_cleanup_expired",
                node_address="192.168.31.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        _, transport_id = transport_service.request_transport(
            artifact_id="cleanup_expired_artifact",
            view_id=None,
            source_node_id="source-node",
            source_address="192.168.32.1",
            source_port=9090,
            request_id="cleanup-expired-request-1",
        )
        assert replica_repo.get_current_requests(replica.replica_id) == 1

        stale_created_at = datetime.now(timezone.utc) - timedelta(seconds=30)
        transport_repo.connection.execute(
            "UPDATE artifact_transports SET created_at = ? WHERE transport_id = ?",
            [stale_created_at, str(transport_id)],
        )

        cleaned = transport_service.cleanup_expired_transports(expiration_seconds=1)
        assert cleaned == 1

        transport_row = transport_repo.find_by_id(transport_id)
        assert transport_row is not None
        assert transport_row.status == "completed"
        assert transport_row.completion_outcome == TransportCompletionOutcome.EXPIRED
        assert transport_row.completion_detail is not None
        assert "cleanup_expired_transports" in transport_row.completion_detail
        assert replica_repo.get_current_requests(replica.replica_id) == 0

    def test_cleanup_expired_transports_reclaims_malformed_in_progress_status(
        self, services, repositories
    ):
        """Malformed in-progress rows should be reclaimed even before stale timeout."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        replica_repo = repositories["replica"]
        transport_repo = repositories["transport"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_cleanup_malformed",
                node_id="node_cleanup_malformed",
                node_address="192.168.33.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        replica = artifact_service.register_replica(
            Replica(
                artifact_id="cleanup_malformed_artifact",
                node_id="node_cleanup_malformed",
                node_address="192.168.33.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        _, transport_id = transport_service.request_transport(
            artifact_id="cleanup_malformed_artifact",
            view_id=None,
            source_node_id="source-node",
            source_address="192.168.34.1",
            source_port=9090,
            request_id="cleanup-malformed-request-1",
        )
        assert replica_repo.get_current_requests(replica.replica_id) == 1

        created_at = datetime.now(timezone.utc)
        transport_repo.connection.execute(
            """
            UPDATE artifact_transports
            SET status = ?, request_id = ?, created_at = ?, completed_at = NULL
            WHERE transport_id = ?
            """,
            [
                "in_progress" * 2,
                (
                    "transport:canonical:cleanup-malformed-request-1"
                    "transport:canonical:cleanup-malformed-request-1"
                ),
                created_at,
                str(transport_id),
            ],
        )

        cleaned = transport_service.cleanup_expired_transports(expiration_seconds=3600)
        assert cleaned == 1

        transport_row = transport_repo.find_by_id(transport_id)
        assert transport_row is not None
        assert transport_row.status == "completed"
        assert transport_row.completion_outcome == TransportCompletionOutcome.EXPIRED
        assert transport_row.completion_detail is not None
        assert "cleanup_malformed_inflight" in transport_row.completion_detail
        assert replica_repo.get_current_requests(replica.replica_id) == 0

    def test_cleanup_expired_transports_keeps_dispatched_request_after_queue_deadline(
        self, services, repositories
    ):
        """Queue wait deadline only applies before dispatch and must not expire in-flight transport."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        pending_repo = repositories["pending_transport_request"]
        replica_repo = repositories["replica"]
        transport_repo = repositories["transport"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_cleanup_dispatched_deadline",
                node_id="node_cleanup_dispatched_deadline",
                node_address="192.168.35.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        replica = artifact_service.register_replica(
            Replica(
                artifact_id="cleanup_dispatched_deadline_artifact",
                node_id="node_cleanup_dispatched_deadline",
                node_address="192.168.35.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        request_id = "cleanup-dispatched-deadline-request-1"
        _, transport_id = transport_service.request_transport(
            artifact_id="cleanup_dispatched_deadline_artifact",
            view_id=None,
            source_node_id="source-node",
            source_address="192.168.36.1",
            source_port=9090,
            wait_timeout_ms=5_000,
            request_id=request_id,
        )
        assert replica_repo.get_current_requests(replica.replica_id) == 1

        past_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
        pending_repo.connection.execute(
            """
            UPDATE pending_transport_requests
            SET state = 'dispatched', deadline_at = ?, updated_at = now()
            WHERE request_id = ?
            """,
            [past_deadline, request_id],
        )

        cleaned = transport_service.cleanup_expired_transports(expiration_seconds=3600)
        assert cleaned == 0

        pending_row = pending_repo.find_by_request_id(request_id)
        assert pending_row is not None
        assert pending_row.state == PendingTransportState.DISPATCHED

        transport_row = transport_repo.find_by_id(transport_id)
        assert transport_row is not None
        assert transport_row.status == "in_progress"
        assert transport_row.completed_at is None
        assert (
            transport_row.completion_outcome == TransportCompletionOutcome.UNSPECIFIED
        )
        assert replica_repo.get_current_requests(replica.replica_id) == 1

    def test_query_transport_window_normalizes_exact_doubled_tokens(
        self, services, repositories
    ):
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]
        transport_repo = repositories["transport"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_transport_window_normalize",
                node_id="node_transport_window_normalize",
                node_address="192.168.38.1",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        _ = artifact_service.register_replica(
            Replica(
                artifact_id="transport_window_normalize_artifact",
                node_id="node_transport_window_normalize",
                node_address="192.168.38.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,
            )
        )

        _, transport_id = transport_service.request_transport(
            artifact_id="transport_window_normalize_artifact",
            view_id=None,
            source_node_id="source-node",
            source_address="192.168.39.1",
            source_port=9090,
            request_id="suite:v3:rx4:r1:a0",
        )

        created_at = datetime.now(timezone.utc)
        transport_repo.connection.execute(
            """
            UPDATE artifact_transports
            SET artifact_id = ?,
                status = ?,
                request_id = ?,
                requester_worker_id = ?,
                group_id = ?,
                group_kind = ?,
                group_part_id = ?,
                group_total_parts = ?,
                created_at = ?,
                completed_at = NULL
            WHERE transport_id = ?
            """,
            [
                ("cgid:weights-suite-v3cgid:weights-suite-v3"),
                "in_progress" * 2,
                "suite:v3:rx4:r1:a0suite:v3:rx4:r1:a0",
                "worker-aworker-a",
                "suite:v3suite:v3",
                "tp_versiontp_version",
                "rx4:r1rx4:r1",
                28,
                created_at,
                str(transport_id),
            ],
        )

        rows = transport_service.query_transport_window(
            started_at=created_at - timedelta(seconds=1),
            finished_at=created_at + timedelta(seconds=1),
            limit=200,
        )
        target_rows = [row for row in rows if row.transport_id == str(transport_id)]
        assert len(target_rows) == 1
        target = target_rows[0]
        assert target.artifact_id == "cgid:weights-suite-v3"
        assert target.status == "in_progress"
        assert target.request_id == "suite:v3:rx4:r1:a0"
        assert target.requester_worker_id == "worker-a"
        assert target.group_id == "suite:v3"
        assert target.group_kind == "tp_version"
        assert target.group_part_id == "rx4:r1"
        assert target.group_total_parts == 28

    def test_group_dispatch_honors_dispatch_batch_limit(self, tmp_path, repositories):
        """A single dispatch cycle should respect dispatch_batch_limit."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_batch_limit.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
    group_dispatch:
      queue_scan_limit: 16
      dispatch_batch_limit: 1
            """,
            encoding="utf-8",
        )
        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_batch_limit",
                    node_id="node_group_batch_limit",
                    node_address="192.168.41.1",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_batch_limit_artifact",
                    node_id="node_group_batch_limit",
                    node_address="192.168.41.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk0"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )

            pending_repo = repositories["pending_transport_request"]
            for i in range(3):
                with repositories["replica"].transaction() as tx:
                    pending_repo.create_if_absent_with_cursor(
                        PendingTransportRequest(
                            request_id=f"group-batch-limit-{i}",
                            request_fingerprint=f"fp-group-batch-limit-{i}",
                            artifact_id="group_batch_limit_artifact",
                            requested_view_id=None,
                            source_node_id=f"source-{i}",
                            source_address="192.168.42.1",
                            source_port=9090 + i,
                        ),
                        tx,
                    )

            with repositories["replica"].transaction() as tx:
                dispatched = transport_service._dispatch_pending_requests(tx=tx)
            assert dispatched == 1

            states = [
                pending_repo.find_by_request_id(f"group-batch-limit-{i}").state
                for i in range(3)
            ]
            assert states.count(PendingTransportState.DISPATCHED) == 1
            assert states.count(PendingTransportState.ENQUEUED) == 2
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_honors_queue_scan_limit(self, tmp_path, repositories):
        """queue_scan_limit should cap how many enqueued requests are considered per cycle."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_scan_limit.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
    group_dispatch:
      queue_scan_limit: 1
      dispatch_batch_limit: 8
            """,
            encoding="utf-8",
        )
        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_scan_limit",
                    node_id="node_group_scan_limit",
                    node_address="192.168.51.1",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_scan_limit_artifact",
                    node_id="node_group_scan_limit",
                    node_address="192.168.51.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk0"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )

            pending_repo = repositories["pending_transport_request"]
            for i in range(3):
                with repositories["replica"].transaction() as tx:
                    pending_repo.create_if_absent_with_cursor(
                        PendingTransportRequest(
                            request_id=f"group-scan-limit-{i}",
                            request_fingerprint=f"fp-group-scan-limit-{i}",
                            artifact_id="group_scan_limit_artifact",
                            requested_view_id=None,
                            source_node_id=f"source-{i}",
                            source_address="192.168.52.1",
                            source_port=9190 + i,
                        ),
                        tx,
                    )

            with repositories["replica"].transaction() as tx:
                dispatched = transport_service._dispatch_pending_requests(tx=tx)
            assert dispatched == 1

            states = [
                pending_repo.find_by_request_id(f"group-scan-limit-{i}").state
                for i in range(3)
            ]
            assert states.count(PendingTransportState.DISPATCHED) == 1
            assert states.count(PendingTransportState.ENQUEUED) == 2
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_soft_cap_skips_group_hotspot_replica(
        self, tmp_path, repositories
    ):
        """Soft cap should skip an over-concentrated source when alternatives exist."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_source_soft_cap.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
    group_dispatch:
      queue_scan_limit: 32
      dispatch_batch_limit: 8
      group_source_spread_weight: 0.0
      group_source_soft_cap_ratio: 1.3
      group_source_min_candidates_for_enforce: 3
            """,
            encoding="utf-8",
        )
        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_source_soft_cap",
                    node_id="node_group_source_soft_cap",
                    node_address="192.168.71.1",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )

            hotspot_replica = artifact_service.register_replica(
                Replica(
                    replica_id=UUID("00000000-0000-0000-0000-000000000001"),
                    artifact_id="group_source_soft_cap_artifact",
                    node_id="node_group_source_soft_cap",
                    node_address="192.168.71.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk-hot"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )
            warm_replica = artifact_service.register_replica(
                Replica(
                    replica_id=UUID("00000000-0000-0000-0000-000000000002"),
                    artifact_id="group_source_soft_cap_artifact",
                    node_id="node_group_source_soft_cap",
                    node_address="192.168.71.1",
                    node_port=8081,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=1,
                    remote_memory_keys=["rk-warm"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )
            cold_replica = artifact_service.register_replica(
                Replica(
                    replica_id=UUID("00000000-0000-0000-0000-000000000003"),
                    artifact_id="group_source_soft_cap_artifact",
                    node_id="node_group_source_soft_cap",
                    node_address="192.168.71.1",
                    node_port=8082,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=2,
                    remote_memory_keys=["rk-cold"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )

            transport_repo = repositories["transport"]
            history_group = TransportSchedulingGroup(
                group_id="group-source-soft-cap",
                group_kind="tp_rank",
                total_parts=128,
                part_id="part-history-bootstrap",
                priority=0,
                epoch=9,
            )

            def _insert_history(replica: Replica, index: int, prefix: str) -> None:
                row = Transport(
                    replica_id=replica.replica_id,
                    artifact_id="group_source_soft_cap_artifact",
                    source_node_id=f"source-{prefix}",
                    source_address="192.168.72.1",
                    source_port=9200 + index,
                    request_id=f"group-source-history-{prefix}-{index}",
                    request_fingerprint=f"fp-group-source-history-{prefix}-{index}",
                )
                row.set_scheduling_group(
                    TransportSchedulingGroup(
                        group_id=history_group.group_id,
                        group_kind=history_group.group_kind,
                        total_parts=history_group.total_parts,
                        part_id=f"{prefix}-history-part-{index}",
                        priority=0,
                        epoch=history_group.epoch,
                    )
                )
                transport_repo.create(row)

            for i in range(6):
                _insert_history(hotspot_replica, i, "hot")
            _insert_history(warm_replica, 0, "warm")
            _insert_history(cold_replica, 0, "cold")

            selected, transport_id = transport_service.request_transport(
                artifact_id="group_source_soft_cap_artifact",
                view_id=None,
                source_node_id="source-live",
                source_address="192.168.72.2",
                source_port=9300,
                request_id="group-source-soft-cap-live",
                scheduling_group=TransportSchedulingGroup(
                    group_id=history_group.group_id,
                    group_kind=history_group.group_kind,
                    total_parts=history_group.total_parts,
                    part_id="part-live",
                    priority=0,
                    epoch=history_group.epoch,
                ),
            )

            assert selected.replica_id != hotspot_replica.replica_id
            assert selected.replica_id in {
                warm_replica.replica_id,
                cold_replica.replica_id,
            }
            transport_service.complete_transport(
                transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_soft_cap_not_forced_when_candidates_too_few(
        self, tmp_path, repositories
    ):
        """Soft cap enforcement should be disabled below candidate threshold."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_source_soft_cap_min_candidates.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
    group_dispatch:
      queue_scan_limit: 32
      dispatch_batch_limit: 8
      group_source_spread_weight: 0.0
      group_source_soft_cap_ratio: 1.3
      group_source_min_candidates_for_enforce: 3
            """,
            encoding="utf-8",
        )
        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_source_soft_cap_min",
                    node_id="node_group_source_soft_cap_min",
                    node_address="192.168.73.1",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )

            hotspot_replica = artifact_service.register_replica(
                Replica(
                    replica_id=UUID("00000000-0000-0000-0000-000000000010"),
                    artifact_id="group_source_soft_cap_min_artifact",
                    node_id="node_group_source_soft_cap_min",
                    node_address="192.168.73.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk-hot-min"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )
            warm_replica = artifact_service.register_replica(
                Replica(
                    replica_id=UUID("00000000-0000-0000-0000-000000000020"),
                    artifact_id="group_source_soft_cap_min_artifact",
                    node_id="node_group_source_soft_cap_min",
                    node_address="192.168.73.1",
                    node_port=8081,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=1,
                    remote_memory_keys=["rk-warm-min"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )

            transport_repo = repositories["transport"]
            history_group = TransportSchedulingGroup(
                group_id="group-source-soft-cap-min",
                group_kind="tp_rank",
                total_parts=64,
                part_id="part-history-bootstrap",
                priority=0,
                epoch=5,
            )

            def _insert_history(replica: Replica, index: int, prefix: str) -> None:
                row = Transport(
                    replica_id=replica.replica_id,
                    artifact_id="group_source_soft_cap_min_artifact",
                    source_node_id=f"source-{prefix}",
                    source_address="192.168.74.1",
                    source_port=9400 + index,
                    request_id=f"group-source-min-history-{prefix}-{index}",
                    request_fingerprint=f"fp-group-source-min-history-{prefix}-{index}",
                )
                row.set_scheduling_group(
                    TransportSchedulingGroup(
                        group_id=history_group.group_id,
                        group_kind=history_group.group_kind,
                        total_parts=history_group.total_parts,
                        part_id=f"{prefix}-history-part-{index}",
                        priority=0,
                        epoch=history_group.epoch,
                    )
                )
                transport_repo.create(row)

            for i in range(6):
                _insert_history(hotspot_replica, i, "hot")
            _insert_history(warm_replica, 0, "warm")

            selected, transport_id = transport_service.request_transport(
                artifact_id="group_source_soft_cap_min_artifact",
                view_id=None,
                source_node_id="source-live",
                source_address="192.168.74.2",
                source_port=9500,
                request_id="group-source-soft-cap-min-live",
                scheduling_group=TransportSchedulingGroup(
                    group_id=history_group.group_id,
                    group_kind=history_group.group_kind,
                    total_parts=history_group.total_parts,
                    part_id="part-live",
                    priority=0,
                    epoch=history_group.epoch,
                ),
            )

            assert selected.replica_id == hotspot_replica.replica_id
            assert selected.replica_id != warm_replica.replica_id
            transport_service.complete_transport(
                transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_refreshes_group_source_counts_per_pending(
        self, tmp_path, repositories
    ):
        """Each pending request in one cycle should observe updated group source counts."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_source_counts_refresh.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
    group_dispatch:
      queue_scan_limit: 16
      dispatch_batch_limit: 2
      group_source_spread_weight: 0.0
      group_source_soft_cap_ratio: 1.3
      group_source_min_candidates_for_enforce: 2
            """,
            encoding="utf-8",
        )
        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )
            pending_repo = repositories["pending_transport_request"]
            transport_repo = repositories["transport"]

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_source_refresh",
                    node_id="node_group_source_refresh",
                    node_address="192.168.75.1",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            first_replica = artifact_service.register_replica(
                Replica(
                    replica_id=UUID("00000000-0000-0000-0000-000000000100"),
                    artifact_id="group_source_refresh_artifact",
                    node_id="node_group_source_refresh",
                    node_address="192.168.75.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk-refresh-0"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )
            second_replica = artifact_service.register_replica(
                Replica(
                    replica_id=UUID("00000000-0000-0000-0000-000000000200"),
                    artifact_id="group_source_refresh_artifact",
                    node_id="node_group_source_refresh",
                    node_address="192.168.75.1",
                    node_port=8081,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=1,
                    remote_memory_keys=["rk-refresh-1"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=8,
                )
            )

            group = TransportSchedulingGroup(
                group_id="group-source-refresh",
                group_kind="tp_rank",
                total_parts=16,
                part_id="part-bootstrap",
                priority=0,
                epoch=2,
            )

            for index in range(2):
                pending = PendingTransportRequest(
                    request_id=f"group-source-refresh-pending-{index}",
                    request_fingerprint=f"fp-group-source-refresh-pending-{index}",
                    artifact_id="group_source_refresh_artifact",
                    requested_view_id=None,
                    source_node_id=f"source-refresh-{index}",
                    source_address="192.168.76.1",
                    source_port=9600 + index,
                )
                pending.set_scheduling_group(
                    TransportSchedulingGroup(
                        group_id=group.group_id,
                        group_kind=group.group_kind,
                        total_parts=group.total_parts,
                        part_id=f"part-{index}",
                        priority=0,
                        epoch=group.epoch,
                    )
                )
                with repositories["replica"].transaction() as tx:
                    pending_repo.create_if_absent_with_cursor(pending, tx)

            with repositories["replica"].transaction() as tx:
                dispatched = transport_service._dispatch_pending_requests(tx=tx)
            assert dispatched == 2

            first_transport = transport_repo.find_by_request_id(
                "group-source-refresh-pending-0"
            )
            second_transport = transport_repo.find_by_request_id(
                "group-source-refresh-pending-1"
            )
            assert first_transport is not None
            assert second_transport is not None
            assert first_transport.replica_id == first_replica.replica_id
            assert second_transport.replica_id == second_replica.replica_id
            assert first_transport.replica_id != second_transport.replica_id
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_sort_key_prefers_starved_low_progress_group(
        self, services, monkeypatch
    ):
        """Sort key should prefer stale groups under fairness floor."""
        transport_service = services["transport"]
        now_utc = datetime.now(timezone.utc)

        lagging = PendingTransportRequest(
            request_id="group-sort-lagging",
            request_fingerprint="fp-lagging",
            artifact_id="group-sort-artifact",
            requested_view_id=None,
            source_node_id="source-lagging",
            source_address="192.168.61.1",
            source_port=9090,
            created_at=now_utc,
        )
        lagging.set_scheduling_group(
            TransportSchedulingGroup(
                group_id="group-lagging",
                group_kind="tp_rank",
                total_parts=4,
                part_id="part-0",
                priority=0,
                epoch=1,
            )
        )

        leading = PendingTransportRequest(
            request_id="group-sort-leading",
            request_fingerprint="fp-leading",
            artifact_id="group-sort-artifact",
            requested_view_id=None,
            source_node_id="source-leading",
            source_address="192.168.61.2",
            source_port=9091,
            created_at=now_utc,
        )
        leading.set_scheduling_group(
            TransportSchedulingGroup(
                group_id="group-leading",
                group_kind="tp_rank",
                total_parts=4,
                part_id="part-1",
                priority=0,
                epoch=1,
            )
        )

        def _fake_progress(
            *, group_kind, group_id, group_epoch, total_parts_hint, cursor
        ):
            if group_id == "group-lagging":
                return TransportGroupProgress(
                    completed_parts=0,
                    total_parts=4,
                    last_success_at=now_utc - timedelta(seconds=30),
                )
            return TransportGroupProgress(
                completed_parts=3,
                total_parts=4,
                last_success_at=now_utc,
            )

        monkeypatch.setattr(
            transport_service.transport_repository,
            "get_group_progress",
            _fake_progress,
        )

        lagging_key = transport_service._group_dispatch_sort_key(
            pending_request=lagging,
            min_completion_ratio=0.0,
            now_utc=now_utc,
            tx=None,
        )
        leading_key = transport_service._group_dispatch_sort_key(
            pending_request=leading,
            min_completion_ratio=0.0,
            now_utc=now_utc,
            tx=None,
        )

        assert lagging_key < leading_key

    def test_group_dispatch_rejects_duplicate_group_part_id(
        self, tmp_path, repositories
    ):
        """GROUP_DISPATCH enforces unique part_id within (group_kind, group_id, epoch)."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_config.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
            """,
            encoding="utf-8",
        )

        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_dispatch_contract",
                    node_id="node_group_dispatch_contract",
                    node_address="192.168.10.10",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_dispatch_contract_artifact",
                    node_id="node_group_dispatch_contract",
                    node_address="192.168.10.10",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk0"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )

            scheduling_group = TransportSchedulingGroup(
                group_id="group-dispatch-contract",
                group_kind="tp_rank",
                total_parts=2,
                part_id="part-0",
                priority=0,
                epoch=7,
            )
            _, first_transport_id = transport_service.request_transport(
                artifact_id="group_dispatch_contract_artifact",
                view_id=None,
                source_node_id="source_node",
                source_address="192.168.2.1",
                source_port=9090,
                request_id="group-dispatch-req-1",
                scheduling_group=scheduling_group,
            )
            transport_service.complete_transport(
                first_transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )

            with pytest.raises(
                ValidationError, match="duplicate part_id in transport history"
            ):
                transport_service.request_transport(
                    artifact_id="group_dispatch_contract_artifact",
                    view_id=None,
                    source_node_id="source_node",
                    source_address="192.168.2.1",
                    source_port=9090,
                    request_id="group-dispatch-req-2",
                    scheduling_group=scheduling_group,
                )
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_rejects_cross_view_group_epoch(
        self, tmp_path, repositories
    ):
        """GROUP_DISPATCH enforces one requested view per group epoch."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_view_contract.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
            """,
            encoding="utf-8",
        )

        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_dispatch_view_contract",
                    node_id="node_group_dispatch_view_contract",
                    node_address="192.168.10.20",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_dispatch_view_contract_artifact",
                    byte_space=ByteSpaceRef.view("view-a"),
                    node_id="node_group_dispatch_view_contract",
                    node_address="192.168.10.20",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk0"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_dispatch_view_contract_artifact",
                    byte_space=ByteSpaceRef.view("view-b"),
                    node_id="node_group_dispatch_view_contract",
                    node_address="192.168.10.20",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=1,
                    remote_memory_keys=["rk1"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )

            _, first_transport_id = transport_service.request_transport(
                artifact_id="group_dispatch_view_contract_artifact",
                view_id="view-a",
                source_node_id="source_node",
                source_address="192.168.2.1",
                source_port=9090,
                request_id="group-view-req-1",
                scheduling_group=TransportSchedulingGroup(
                    group_id="group-view-contract",
                    group_kind="tp_rank",
                    total_parts=2,
                    part_id="part-0",
                    priority=0,
                    epoch=3,
                ),
            )
            transport_service.complete_transport(
                first_transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )

            with pytest.raises(
                ValidationError, match="artifact/view/total_parts mismatch"
            ):
                transport_service.request_transport(
                    artifact_id="group_dispatch_view_contract_artifact",
                    view_id="view-b",
                    source_node_id="source_node",
                    source_address="192.168.2.1",
                    source_port=9090,
                    request_id="group-view-req-2",
                    scheduling_group=TransportSchedulingGroup(
                        group_id="group-view-contract",
                        group_kind="tp_rank",
                        total_parts=2,
                        part_id="part-1",
                        priority=0,
                        epoch=3,
                    ),
                )
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_tp_version_allows_cross_view_group_epoch(
        self, tmp_path, repositories
    ):
        """tp_version groups allow per-part view variance in the same group epoch."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_tp_version_view_contract.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
            """,
            encoding="utf-8",
        )

        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_dispatch_tp_version_view_contract",
                    node_id="node_group_dispatch_tp_version_view_contract",
                    node_address="192.168.10.21",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_dispatch_tp_version_view_contract_artifact",
                    byte_space=ByteSpaceRef.view("view-a"),
                    node_id="node_group_dispatch_tp_version_view_contract",
                    node_address="192.168.10.21",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk0"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_dispatch_tp_version_view_contract_artifact",
                    byte_space=ByteSpaceRef.view("view-b"),
                    node_id="node_group_dispatch_tp_version_view_contract",
                    node_address="192.168.10.21",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=1,
                    remote_memory_keys=["rk1"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )

            _, first_transport_id = transport_service.request_transport(
                artifact_id="group_dispatch_tp_version_view_contract_artifact",
                view_id="view-a",
                source_node_id="source_node",
                source_address="192.168.2.1",
                source_port=9090,
                request_id="group-tp-version-view-req-1",
                scheduling_group=TransportSchedulingGroup(
                    group_id="group-tp-version-view-contract",
                    group_kind="tp_version",
                    total_parts=2,
                    part_id="part-0",
                    priority=0,
                    epoch=3,
                ),
            )
            transport_service.complete_transport(
                first_transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )

            _, second_transport_id = transport_service.request_transport(
                artifact_id="group_dispatch_tp_version_view_contract_artifact",
                view_id="view-b",
                source_node_id="source_node",
                source_address="192.168.2.1",
                source_port=9090,
                request_id="group-tp-version-view-req-2",
                scheduling_group=TransportSchedulingGroup(
                    group_id="group-tp-version-view-contract",
                    group_kind="tp_version",
                    total_parts=2,
                    part_id="part-1",
                    priority=0,
                    epoch=3,
                ),
            )
            assert second_transport_id is not None
            transport_service.complete_transport(
                second_transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )
        finally:
            set_config(legacy_cfg)

    def test_group_dispatch_tp_version_allows_cross_artifact_group_epoch(
        self, tmp_path, repositories
    ):
        """tp_version groups allow per-part artifact variance in the same group epoch."""
        legacy_cfg = GlobalStoreConfig()
        cfg_path = tmp_path / "group_dispatch_tp_version_artifact_contract.yaml"
        cfg_path.write_text(
            """
database:
  db_file: ""
server:
  listen:
    host: "127.0.0.1"
    port: 50051
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
            """,
            encoding="utf-8",
        )

        try:
            set_config(GlobalStoreConfig.from_file(str(cfg_path)))
            artifact_service = ArtifactService(repositories["replica"])
            worker_service = WorkerService(
                repositories["worker"], repositories["replica"]
            )
            transport_service = TransportService(
                repositories["replica"],
                repositories["transport"],
                repositories["pending_transport_request"],
            )

            worker = worker_service.register_worker(
                Worker(
                    daemon_id="daemon_group_dispatch_tp_version_artifact_contract",
                    node_id="node_group_dispatch_tp_version_artifact_contract",
                    node_address="192.168.10.23",
                    grpc_port=50051,
                    p2p_port=50052,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_dispatch_tp_version_artifact_contract_artifact_a",
                    byte_space=ByteSpaceRef.view("view-a"),
                    node_id="node_group_dispatch_tp_version_artifact_contract",
                    node_address="192.168.10.23",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    remote_memory_keys=["rk0"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )
            artifact_service.register_replica(
                Replica(
                    artifact_id="group_dispatch_tp_version_artifact_contract_artifact_b",
                    byte_space=ByteSpaceRef.view("view-a"),
                    node_id="node_group_dispatch_tp_version_artifact_contract",
                    node_address="192.168.10.23",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=1,
                    remote_memory_keys=["rk1"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )

            _, first_transport_id = transport_service.request_transport(
                artifact_id="group_dispatch_tp_version_artifact_contract_artifact_a",
                view_id="view-a",
                source_node_id="source_node",
                source_address="192.168.2.1",
                source_port=9090,
                request_id="group-tp-version-artifact-req-1",
                scheduling_group=TransportSchedulingGroup(
                    group_id="group-tp-version-artifact-contract",
                    group_kind="tp_version",
                    total_parts=2,
                    part_id="part-0",
                    priority=0,
                    epoch=3,
                ),
            )
            transport_service.complete_transport(
                first_transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )

            _, second_transport_id = transport_service.request_transport(
                artifact_id="group_dispatch_tp_version_artifact_contract_artifact_b",
                view_id="view-a",
                source_node_id="source_node",
                source_address="192.168.2.1",
                source_port=9090,
                request_id="group-tp-version-artifact-req-2",
                scheduling_group=TransportSchedulingGroup(
                    group_id="group-tp-version-artifact-contract",
                    group_kind="tp_version",
                    total_parts=2,
                    part_id="part-1",
                    priority=0,
                    epoch=3,
                ),
            )
            assert second_transport_id is not None
            transport_service.complete_transport(
                second_transport_id,
                outcome=TransportCompletionOutcome.SUCCESS,
            )
        finally:
            set_config(legacy_cfg)

    def test_transport_service_tp_version_request_id_replay_allows_view_change(
        self, services
    ):
        """tp_version request replay should not fail on view_id variance."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        worker = worker_service.register_worker(
            Worker(
                daemon_id="daemon_tp_version_replay_view_change",
                node_id="node_tp_version_replay_view_change",
                node_address="192.168.10.22",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )
        artifact_service.register_replica(
            Replica(
                artifact_id="tp_version_replay_view_change_artifact",
                byte_space=ByteSpaceRef.view("view-a"),
                node_id="node_tp_version_replay_view_change",
                node_address="192.168.10.22",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,
            )
        )
        artifact_service.register_replica(
            Replica(
                artifact_id="tp_version_replay_view_change_artifact",
                byte_space=ByteSpaceRef.view("view-b"),
                node_id="node_tp_version_replay_view_change",
                node_address="192.168.10.22",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=1,
                remote_memory_keys=["rk1"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                worker_id=worker.worker_id,
                max_concurrency=2,
            )
        )

        group = TransportSchedulingGroup(
            group_id="group-tp-version-request-replay-view-change",
            group_kind="tp_version",
            total_parts=2,
            part_id="part-0",
            priority=0,
            epoch=7,
        )
        replica_first, transport_id_first = transport_service.request_transport(
            artifact_id="tp_version_replay_view_change_artifact",
            view_id="view-a",
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="group-tp-version-request-replay-view-change",
            scheduling_group=group,
        )
        replica_second, transport_id_second = transport_service.request_transport(
            artifact_id="tp_version_replay_view_change_artifact",
            view_id="view-b",
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
            request_id="group-tp-version-request-replay-view-change",
            scheduling_group=group,
        )

        assert transport_id_first == transport_id_second
        assert replica_first.replica_id == replica_second.replica_id
        transport_service.complete_transport(
            transport_id_first,
            outcome=TransportCompletionOutcome.SUCCESS,
        )

    def test_artifact_service_validation(self, services):
        """Test artifact service validation."""
        artifact_service = services["artifact"]

        # Missing artifact id
        with pytest.raises(ValidationError):
            artifact_service.register_replica(
                Replica(
                    artifact_id="",  # Empty artifact id
                    node_id="node1",
                    node_address="192.168.1.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=0,
                    worker_id="worker1",
                )
            )

    def test_transport_service_with_load_balancing(self, services):
        """Test transport service with multiple replicas and load balancing."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Register multiple workers
        workers = []
        for i in range(3):
            worker = worker_service.register_worker(
                Worker(
                    daemon_id=f"daemon_{i}",
                    node_id=f"node_{i}",
                    node_address=f"192.168.1.{i + 1}",
                    grpc_port=50051 + i,
                    p2p_port=50052 + i,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            workers.append(worker)

        # Register replicas with different loads
        replicas = []
        loads = [5, 2, 8]  # Different current request loads
        memory_types = [MemoryType.GPU, MemoryType.GPU, MemoryType.RAM]

        for i, (load, mem_type) in enumerate(zip(loads, memory_types, strict=False)):
            replica = artifact_service.register_replica(
                Replica(
                    artifact_id="balanced_artifact",
                    node_id=f"node_{i}",
                    node_address=f"192.168.1.{i + 1}",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=mem_type,
                    device_id=0,
                    remote_memory_keys=[f"rk{i}"],
                    buffer_sizes=[1024],
                    export_state=ExportState.EXPORTABLE,
                    worker_id=workers[i].worker_id,
                    max_concurrency=10,
                    current_requests=load,
                )
            )
            replicas.append(replica)

        # Request transport - should select GPU replica with lower load (node_1)
        selected, transport_id = transport_service.request_transport(
            artifact_id="balanced_artifact",
            view_id=None,
            source_node_id="client",
            source_address="192.168.2.1",
            source_port=9000,
            request_id="transport-balanced-1",
        )

        # Should select the GPU replica with lower load (node_1, load=2)
        assert selected.node_id == "node_1"
        assert selected.memory_type == MemoryType.GPU
        assert selected.current_requests == 3  # Incremented from 2 to 3

    def test_view_state_service_update_and_fetch(self, services):
        """Persist view metadata and leaves atomically."""
        view_state_service = services["view_state"]
        now = datetime.now(timezone.utc)
        view_payload = ViewUpsertPayload(
            artifact_id="mi2:index:data",
            view_id="view-1",
            view_spec_json="{}",
            view_size=128,
            view_data_hash="mhash",
            verified_at=now,
            canonical_size_bytes=128,
            canonical_bytes_covered=128,
        )
        leaf_payloads = [
            LeafWritePayload(
                artifact_id="mi2:index:data",
                space_kind="V",
                space_id="view-1",
                leaf_idx=0,
                digest=b"\x01" * 32,
            ),
            LeafWritePayload(
                artifact_id="mi2:index:data",
                space_kind="V",
                space_id="view-1",
                leaf_idx=1,
                digest=b"\x02" * 32,
            ),
        ]

        view_state_service.update_view_state(
            view=view_payload,
            leaf_writes=leaf_payloads,
            proof_digests=(),
            tensor_intervals=None,
        )

        stored_view = view_state_service.get_view(
            artifact_id="mi2:index:data", view_id="view-1"
        )
        assert stored_view is not None
        assert stored_view["view_size"] == 128
        assert stored_view["view_spec_json"] == "{}"

        leaves = list(
            view_state_service.get_leaves(
                artifact_id="mi2:index:data",
                space_kind="V",
                space_id="view-1",
            )
        )
        assert len(leaves) == 2
        assert leaves[0][0] == 0
        assert leaves[0][1] == b"\x01" * 32

    def test_view_state_service_record_view_registration(self, services):
        """record_view_registration delegates to update_view_state."""
        view_state_service = services["view_state"]
        now = datetime.now(timezone.utc)
        leaf_payloads = [
            LeafWritePayload(
                artifact_id="mi2:index:data",
                space_kind="V",
                space_id="view-2",
                leaf_idx=0,
                digest=b"\x03" * 32,
            )
        ]

        view_state_service.record_view_registration(
            artifact_id="mi2:index:data",
            view_id="view-2",
            view_spec_json='{"tensor":"weights"}',
            view_size=512,
            view_data_hash="mhash-2",
            verified_at=now,
            canonical_size_bytes=512,
            canonical_bytes_covered=512,
            leaf_writes=leaf_payloads,
        )

        stored_view = view_state_service.get_view(
            artifact_id="mi2:index:data",
            view_id="view-2",
        )
        assert stored_view is not None
        assert stored_view["view_size"] == 512
        assert stored_view["view_data_hash"] == "mhash-2"

        leaves = list(
            view_state_service.get_leaves(
                artifact_id="mi2:index:data",
                space_kind="V",
                space_id="view-2",
            )
        )
        assert len(leaves) == 1
        assert leaves[0][0] == 0
        assert leaves[0][1] == b"\x03" * 32

    def test_view_state_service_metrics_track_backlog(self, services):
        view_state_service = services["view_state"]
        metrics.VIEW_PARTIAL_BACKLOG_GAUGE.clear()
        complete_counter = metrics.VIEW_REGISTRATION_COUNTER.labels(result="complete")
        partial_counter = metrics.VIEW_REGISTRATION_COUNTER.labels(result="partial")
        baseline_complete = complete_counter._value.get()
        baseline_partial = partial_counter._value.get()

        view_state_service.record_view_registration(
            artifact_id="mi2:index:metrics",
            view_id="view-metrics",
            view_spec_json="{}",
            view_size=256,
            view_data_hash=None,
            verified_at=None,
            canonical_size_bytes=1024,
            canonical_bytes_covered=512,
            canonical_ranges=[(0, 512)],
        )

        backlog_value = metrics.VIEW_PARTIAL_BACKLOG_GAUGE.labels(
            artifact_id="mi2:index:metrics", view_id="view-metrics"
        )._value.get()
        assert backlog_value == 512
        assert partial_counter._value.get() == baseline_partial + 1

        view_state_service.record_view_registration(
            artifact_id="mi2:index:metrics",
            view_id="view-metrics",
            view_spec_json="{}",
            view_size=256,
            view_data_hash=None,
            verified_at=None,
            canonical_size_bytes=1024,
            canonical_bytes_covered=1024,
        )

        backlog_after = metrics.VIEW_PARTIAL_BACKLOG_GAUGE.labels(
            artifact_id="mi2:index:metrics", view_id="view-metrics"
        )._value.get()
        assert backlog_after == 0
        assert complete_counter._value.get() == baseline_complete + 1

    def test_view_state_service_rejects_mismatched_artifact(self, services):
        """Leaf writes must share artifact_id."""
        view_state_service = services["view_state"]
        with pytest.raises(ValueError, match="artifact_id mismatch"):
            view_state_service.update_view_state(
                view=None,
                leaf_writes=[
                    LeafWritePayload(
                        artifact_id="mi2:index:data",
                        space_kind="C",
                        space_id="index",
                        leaf_idx=0,
                        digest=b"\x00" * 32,
                    ),
                    LeafWritePayload(
                        artifact_id="mi2:other:data",
                        space_kind="C",
                        space_id="index",
                        leaf_idx=1,
                        digest=b"\x01" * 32,
                    ),
                ],
                proof_digests=(),
                tensor_intervals=None,
            )

    def test_view_state_service_requires_coverage_metadata_for_partial(self, services):
        view_state_service = services["view_state"]
        with pytest.raises(DatabaseError, match="coverage metadata missing"):
            view_state_service.record_view_registration(
                artifact_id="mi2:index:partial",
                view_id="view-partial",
                view_spec_json="{}",
                view_size=64,
                view_data_hash="vh",
                verified_at=None,
                canonical_size_bytes=256,
                canonical_bytes_covered=128,
                canonical_ranges=None,
            )

    def test_view_state_service_rejects_overlapping_ranges(self, services):
        view_state_service = services["view_state"]
        tensor_intervals = {"weights": (0, 256)}
        view_state_service.record_view_registration(
            artifact_id="cgid:overlap",
            view_id="view-a",
            view_spec_json="{}",
            view_size=64,
            view_data_hash="vh-a",
            verified_at=None,
            canonical_size_bytes=256,
            canonical_bytes_covered=64,
            canonical_ranges=[(0, 64)],
            tensor_intervals=tensor_intervals,
        )

        with pytest.raises(DatabaseError, match="overlapping canonical coverage"):
            view_state_service.record_view_registration(
                artifact_id="cgid:overlap",
                view_id="view-b",
                view_spec_json="{}",
                view_size=64,
                view_data_hash="vh-b",
                verified_at=None,
                canonical_size_bytes=256,
                canonical_bytes_covered=64,
                canonical_ranges=[(32, 64)],
                tensor_intervals=tensor_intervals,
            )

    def test_view_state_service_allows_replicate_equal_overlaps_with_proofs(
        self, services, repositories
    ):
        view_state_service = services["view_state"]
        assembly_id = "cgid:replicate_equal"

        def _multibase_multihash_sha256(digest: bytes) -> str:
            if len(digest) != 32:
                raise ValueError("SHA256 digest must be 32 bytes")
            mh = b"\x12\x20" + digest
            encoded = base64.b32encode(mh).decode("ascii").lower().rstrip("=")
            return f"b{encoded}"

        canonical_index_bytes = b'{"weights":[0,32,[8],[1],"torch.float32",0],"bias":[32,32,[8],[1],"torch.float32",0]}'
        index_multihash = _multibase_multihash_sha256(
            hashlib.sha256(canonical_index_bytes).digest()
        )

        layout = layout_pb2.LayoutSpec(
            layout_schema_version=1,
            index_multihash=index_multihash,
            proof_schema_version=PROOF_SCHEMA_V1,
        )
        layout.tensors["weights"].overlap_mode = layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
        payload = layout.SerializeToString(deterministic=True)
        layout_id = _multibase_multihash_sha256(hashlib.sha256(payload).digest())
        repositories["layout_spec"].put(
            layout_id=layout_id,
            index_multihash=index_multihash,
            layout_proto=payload,
            layout_json=None,
        )
        repositories["assembly_layout_binding"].update(
            assembly_id=assembly_id,
            layout_id=layout_id,
            expected_binding_version=0,
        )

        tensor_intervals = {"weights": (0, 32), "bias": (32, 32)}
        weights_digest = hashlib.sha256(b"\x01" * 32).digest()
        bias_digest = hashlib.sha256(b"\x02" * 32).digest()

        view_state_service.record_view_registration(
            artifact_id=assembly_id,
            view_id="view-a",
            view_spec_json="{}",
            view_size=32,
            view_data_hash="vh-a",
            verified_at=None,
            canonical_size_bytes=64,
            canonical_bytes_covered=32,
            canonical_ranges=[(0, 32)],
            proof_digests=[
                PieceProofDigestPayload(
                    artifact_id=assembly_id,
                    view_id="view-a",
                    tensor_name="weights",
                    proof_schema_version=PROOF_SCHEMA_V1,
                    proof_chunk_idx=0,
                    digest=weights_digest,
                ),
                # Non-participating tensor digests are allowed (ignored by v2 policy).
                PieceProofDigestPayload(
                    artifact_id=assembly_id,
                    view_id="view-a",
                    tensor_name="bias",
                    proof_schema_version=PROOF_SCHEMA_V1,
                    proof_chunk_idx=0,
                    digest=bias_digest,
                ),
            ],
            tensor_intervals=tensor_intervals,
        )

        view_state_service.record_view_registration(
            artifact_id=assembly_id,
            view_id="view-b",
            view_spec_json="{}",
            view_size=32,
            view_data_hash="vh-b",
            verified_at=None,
            canonical_size_bytes=64,
            canonical_bytes_covered=32,
            canonical_ranges=[(0, 32)],
            proof_digests=[
                PieceProofDigestPayload(
                    artifact_id=assembly_id,
                    view_id="view-b",
                    tensor_name="weights",
                    proof_schema_version=PROOF_SCHEMA_V1,
                    proof_chunk_idx=0,
                    digest=weights_digest,
                )
            ],
            tensor_intervals=tensor_intervals,
        )

        with pytest.raises(DatabaseError, match="assembly_proof_commitments conflict"):
            view_state_service.record_view_registration(
                artifact_id=assembly_id,
                view_id="view-c",
                view_spec_json="{}",
                view_size=32,
                view_data_hash="vh-c",
                verified_at=None,
                canonical_size_bytes=64,
                canonical_bytes_covered=32,
                canonical_ranges=[(0, 32)],
                proof_digests=[
                    PieceProofDigestPayload(
                        artifact_id=assembly_id,
                        view_id="view-c",
                        tensor_name="weights",
                        proof_schema_version=PROOF_SCHEMA_V1,
                        proof_chunk_idx=0,
                        digest=hashlib.sha256(b"\x03" * 32).digest(),
                    )
                ],
                tensor_intervals=tensor_intervals,
            )

    def test_view_state_service_rejects_view_data_hash_conflict(self, services):
        view_state_service = services["view_state"]
        view_state_service.record_view_registration(
            artifact_id="mi2:index:hash",
            view_id="view-hash",
            view_spec_json="{}",
            view_size=128,
            view_data_hash="hash-a",
            verified_at=None,
            canonical_size_bytes=128,
            canonical_bytes_covered=128,
            canonical_ranges=[(0, 128)],
        )

        with pytest.raises(DatabaseError, match="view_data_hash conflict"):
            view_state_service.record_view_registration(
                artifact_id="mi2:index:hash",
                view_id="view-hash",
                view_spec_json="{}",
                view_size=128,
                view_data_hash="hash-b",
                verified_at=None,
                canonical_size_bytes=128,
                canonical_bytes_covered=128,
                canonical_ranges=[(0, 128)],
            )

    def test_view_state_service_replaces_assembly_leaf_state_for_same_view(
        self,
        services,
    ):
        view_state_service = services["view_state"]
        assembly_id = "cgid:assembly-replace"

        view_state_service.record_view_registration(
            artifact_id=assembly_id,
            view_id="view-a",
            view_spec_json="{}",
            view_size=64,
            view_data_hash="vh-a",
            verified_at=None,
            canonical_size_bytes=64,
            canonical_bytes_covered=64,
            canonical_ranges=[(0, 64)],
            leaf_writes=[
                LeafWritePayload(
                    artifact_id=assembly_id,
                    space_kind="V",
                    space_id="view-a",
                    leaf_idx=0,
                    digest=b"\x01" * 32,
                ),
                LeafWritePayload(
                    artifact_id=assembly_id,
                    space_kind="C",
                    space_id="indexmh",
                    leaf_idx=0,
                    digest=b"\x02" * 32,
                ),
            ],
            tensor_intervals={"weights": (0, 64)},
        )

        view_state_service.record_view_registration(
            artifact_id=assembly_id,
            view_id="view-a",
            view_spec_json="{}",
            view_size=64,
            view_data_hash="vh-b",
            verified_at=None,
            canonical_size_bytes=64,
            canonical_bytes_covered=64,
            canonical_ranges=[(0, 64)],
            leaf_writes=[
                LeafWritePayload(
                    artifact_id=assembly_id,
                    space_kind="V",
                    space_id="view-a",
                    leaf_idx=0,
                    digest=b"\x03" * 32,
                ),
                LeafWritePayload(
                    artifact_id=assembly_id,
                    space_kind="C",
                    space_id="indexmh",
                    leaf_idx=0,
                    digest=b"\x04" * 32,
                ),
            ],
            tensor_intervals={"weights": (0, 64)},
        )

        stored_view = view_state_service.get_view(
            artifact_id=assembly_id,
            view_id="view-a",
        )
        assert stored_view is not None
        assert stored_view["view_data_hash"] == "vh-b"

        assert list(
            view_state_service.get_leaves(
                artifact_id=assembly_id,
                space_kind="V",
                space_id="view-a",
            )
        ) == [(0, b"\x03" * 32)]

        assert list(
            view_state_service.get_leaves(
                artifact_id=assembly_id,
                space_kind="C",
                space_id="indexmh",
            )
        ) == [(0, b"\x04" * 32)]
