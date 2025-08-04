#  Copyright (c) 2025, StepCast Team.

"""Tests for Global Store service layer."""

import pytest

from scstore.global_store.models import ModelReplica, Worker, MemoryType
from scstore.global_store.exceptions import ValidationError, TimeoutError, NotFoundError


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
        model_service = services["model"]

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
        replica = model_service.register_replica(
            ModelReplica(
                model_name="test_model",
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
        found_replicas = model_service.list_replicas(model_name="test_model")
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

    def test_model_service_registration(self, services, repositories):
        """Test model replica registration."""
        model_service = services["model"]
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
        replica = ModelReplica(
            model_name="test_model",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id=worker.worker_id,
        )

        registered = model_service.register_replica(replica)
        assert registered.replica_id is not None

        # Register again (should update)
        replica2 = ModelReplica(
            model_name="test_model",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=2048,  # Different size
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id=worker.worker_id,
            max_concurrency=20,
        )

        updated = model_service.register_replica(replica2)
        assert updated.replica_id == registered.replica_id
        assert updated.memory_size == 2048
        assert updated.max_concurrency == 20

    def test_model_service_unregistration(self, services):
        """Test model replica unregistration."""
        model_service = services["model"]
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

        replica = model_service.register_replica(
            ModelReplica(
                model_name="test_model",
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
        success = model_service.unregister_replica(replica.replica_id, "test_model")
        assert success is True

        # Verify replica is removed
        found_replicas = model_service.list_replicas(model_name="test_model")
        assert len(found_replicas) == 0

    def test_model_service_list_replicas(self, services):
        """Test listing model replicas."""
        model_service = services["model"]
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
            model_service.register_replica(
                ModelReplica(
                    model_name=f"model_{i}",
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
        all_replicas = model_service.list_replicas()
        assert len(all_replicas) >= 3

        # List replicas for specific model
        model_0_replicas = model_service.list_replicas(model_name="model_0")
        assert len(model_0_replicas) == 1
        assert model_0_replicas[0].model_name == "model_0"

    def test_transport_service_request(self, services, repositories):
        """Test transport request logic."""
        transport_service = services["transport"]
        model_service = services["model"]
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

        replica = model_service.register_replica(
            ModelReplica(
                model_name="test_model",
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
            model_name="test_model",
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

    def test_transport_service_timeout(self, services):
        """Test transport timeout."""
        transport_service = services["transport"]

        # No replicas available
        with pytest.raises(TimeoutError):
            transport_service.request_transport(
                model_name="nonexistent_model",
                source_node_id="source",
                source_address="192.168.1.1",
                source_port=8080,
                wait_timeout_ms=100,
            )

    def test_transport_service_concurrency_limit(self, services):
        """Test concurrency limiting."""
        transport_service = services["transport"]
        model_service = services["model"]
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

        replica = model_service.register_replica(
            ModelReplica(
                model_name="test_model",
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
                model_name="test_model",
                source_node_id=f"source_{i}",
                source_address="192.168.2.1",
                source_port=9090 + i,
            )
            transport_ids.append(transport_id)

        # Third request should timeout (no capacity)
        with pytest.raises(TimeoutError):
            transport_service.request_transport(
                model_name="test_model",
                source_node_id="source_3",
                source_address="192.168.2.1",
                source_port=9093,
                wait_timeout_ms=100,
            )

        # Complete one transport
        transport_service.complete_transport(transport_ids[0])

        # Now request should succeed
        _, transport_id = transport_service.request_transport(
            model_name="test_model",
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

    def test_model_service_validation(self, services):
        """Test model service validation."""
        model_service = services["model"]

        # Missing model name
        with pytest.raises(ValidationError):
            model_service.register_replica(
                ModelReplica(
                    model_name="",  # Empty model name
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
        model_service = services["model"]
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

        for i, (load, mem_type) in enumerate(zip(loads, memory_types)):
            replica = model_service.register_replica(
                ModelReplica(
                    model_name="balanced_model",
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
            model_name="balanced_model",
            source_node_id="client",
            source_address="192.168.2.1",
            source_port=9000,
        )

        # Should select the GPU replica with lower load (node_1, load=2)
        assert selected.node_id == "node_1"
        assert selected.memory_type == MemoryType.GPU
        assert selected.current_requests == 3  # Incremented from 2 to 3