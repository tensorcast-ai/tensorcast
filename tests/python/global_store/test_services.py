#  Copyright (c) 2025, TensorCast Team.

"""Tests for Global Store service layer."""

from datetime import datetime, timezone

import pytest

from tensorcast.global_store.exceptions import (
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from tensorcast.global_store.models import MemoryType, Replica, Worker
from tensorcast.global_store.services.view_state_service import (
    LeafWritePayload,
    VariantUpsertPayload,
)


class TestServices:
    """Test service layer."""

    def test_worker_service_registration(self, services):
        """Test worker registration logic."""
        worker_service = services["worker"]

        # Register new worker
        worker = Worker(
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

    def test_worker_service_validation(self, services):
        """Test worker validation."""
        worker_service = services["worker"]

        # Missing node_id
        with pytest.raises(ValidationError, match="Node ID is required"):
            worker_service.register_worker(Worker())

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
                    node_id=f"node_{i}",
                    node_address=f"192.168.1.{i+1}",
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
        accepting_workers = worker_service.list_active_workers(include_unavailable=False)
        accepting_count = sum(1 for w in accepting_workers if w.accepting_new_requests)
        assert accepting_count >= 2

    def test_artifact_service_registration(self, services, repositories):
        """Test artifact replica registration."""
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Register worker first
        worker = worker_service.register_worker(
            Worker(
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
        success = artifact_service.unregister_replica(replica.replica_id, "test_artifact")
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
                max_concurrency=2,
            )
        )

        # Request transport
        selected, transport_id = transport_service.request_transport(
            artifact_id="test_artifact",
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9090,
        )

        assert selected.replica_id == replica.replica_id
        assert selected.current_requests == 1

        # Complete transport
        current, max_conc = transport_service.complete_transport(transport_id)
        assert current == 0
        assert max_conc == 2

    def test_transport_service_no_replicas(self, services):
        """Test transport when no replicas exist."""
        transport_service = services["transport"]

        # No replicas available
        with pytest.raises(NotFoundError):
            transport_service.request_transport(
                artifact_id="nonexistent_artifact",
                source_node_id="source",
                source_address="192.168.1.1",
                source_port=8080,
                wait_timeout_ms=100,
            )

    def test_transport_service_timeout(self, services):
        """Test transport timeout when replicas exist but are all busy."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Setup worker and replica with low concurrency
        worker = worker_service.register_worker(
            Worker(
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
                worker_id=worker.worker_id,
                max_concurrency=1,  # Only 1 concurrent request
            )
        )

        # Request first transport (should succeed)
        _, transport_id = transport_service.request_transport(
            artifact_id="test_timeout_artifact",
            source_node_id="source_1",
            source_address="192.168.2.1",
            source_port=9090,
        )

        # Request second transport with short timeout (should fail with TimeoutError)
        with pytest.raises(TimeoutError):
            transport_service.request_transport(
                artifact_id="test_timeout_artifact",
                source_node_id="source_2",
                source_address="192.168.2.2",
                source_port=9091,
                wait_timeout_ms=100,  # Short timeout
            )

        # Complete the first transport
        transport_service.complete_transport(transport_id)

    def test_transport_service_concurrency_limit(self, services):
        """Test concurrency limiting."""
        transport_service = services["transport"]
        artifact_service = services["artifact"]
        worker_service = services["worker"]

        # Setup worker and replica with low concurrency
        worker = worker_service.register_worker(
            Worker(
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
                worker_id=worker.worker_id,
                max_concurrency=2,  # Only 2 concurrent requests
            )
        )

        # Request 2 transports (should succeed)
        transport_ids = []
        for i in range(2):
            _, transport_id = transport_service.request_transport(
                artifact_id="test_artifact",
                source_node_id=f"source_{i}",
                source_address="192.168.2.1",
                source_port=9090 + i,
            )
            transport_ids.append(transport_id)

        # Third request should timeout (no capacity)
        with pytest.raises(TimeoutError):
            transport_service.request_transport(
                artifact_id="test_artifact",
                source_node_id="source_3",
                source_address="192.168.2.1",
                source_port=9093,
                wait_timeout_ms=100,
            )

        # Complete one transport
        transport_service.complete_transport(transport_ids[0])

        # Now request should succeed
        _, transport_id = transport_service.request_transport(
            artifact_id="test_artifact",
            source_node_id="source_3",
            source_address="192.168.2.1",
            source_port=9093,
        )
        assert transport_id is not None

    def test_transport_service_nonexistent_transport_completion(self, services):
        """Test completing a non-existent transport."""
        transport_service = services["transport"]

        # Try to complete non-existent transport
        with pytest.raises(NotFoundError):
            transport_service.complete_transport("nonexistent_transport_id")

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
                    node_id=f"node_{i}",
                    node_address=f"192.168.1.{i+1}",
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
                    node_address=f"192.168.1.{i+1}",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=mem_type,
                    device_id=0,
                    worker_id=workers[i].worker_id,
                    max_concurrency=10,
                    current_requests=load,
                )
            )
            replicas.append(replica)

        # Request transport - should select GPU replica with lower load (node_1)
        selected, transport_id = transport_service.request_transport(
            artifact_id="balanced_artifact",
            source_node_id="client",
            source_address="192.168.2.1",
            source_port=9000,
        )

        # Should select the GPU replica with lower load (node_1, load=2)
        assert selected.node_id == "node_1"
        assert selected.memory_type == MemoryType.GPU
        assert selected.current_requests == 3  # Incremented from 2 to 3

    def test_view_state_service_update_and_fetch(self, services):
        """Persist variant metadata and leaves atomically."""
        view_state_service = services["view_state"]
        now = datetime.now(timezone.utc)
        variant_payload = VariantUpsertPayload(
            artifact_id="mi2:index:data",
            view_id="view-1",
            view_spec_json="{}",
            view_size=128,
            view_data_hash="mhash",
            verified_at=now,
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
            variant=variant_payload,
            leaf_writes=leaf_payloads,
        )

        stored_variant = view_state_service.get_variant(
            artifact_id="mi2:index:data", view_id="view-1"
        )
        assert stored_variant is not None
        assert stored_variant["view_size"] == 128
        assert stored_variant["view_spec_json"] == "{}"

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

    def test_view_state_service_rejects_mismatched_artifact(self, services):
        """Leaf writes must share artifact_id."""
        view_state_service = services["view_state"]
        with pytest.raises(ValueError, match="artifact_id mismatch"):
            view_state_service.update_view_state(
                variant=None,
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
            )
