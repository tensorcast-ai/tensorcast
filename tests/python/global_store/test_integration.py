#  Copyright (c) 2025, StepCast Team.

"""Integration tests for Global Store full stack."""


from scstore.global_store.models import MemoryType, ModelReplica, Worker


class TestIntegration:
    """Integration tests for the full stack."""

    def test_worker_lifecycle(self, services):
        """Test complete worker lifecycle."""
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

        # Register replicas
        for i in range(3):
            model_service.register_replica(
                ModelReplica(
                    model_id=f"model_{i}",
                    node_id="node1",
                    node_address="192.168.1.1",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=i,
                    worker_id=worker.worker_id,
                )
            )

        # List active workers
        workers = worker_service.list_active_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == worker.worker_id

        # Unregister worker
        success = worker_service.unregister_worker(worker.worker_id)
        assert success is True

        # Verify replicas are marked unavailable
        replicas = model_service.list_replicas(node_id="node1")
        for replica in replicas:
            assert replica.is_available is False

    def test_multi_node_load_balancing(self, services):
        """Test load balancing across multiple nodes."""
        worker_service = services["worker"]
        model_service = services["model"]
        transport_service = services["transport"]

        # Register 3 workers on different nodes
        workers = []
        for i in range(3):
            worker = worker_service.register_worker(
                Worker(
                    node_id=f"node{i}",
                    node_address=f"192.168.1.{i+1}",
                    grpc_port=50051 + i,
                    p2p_port=50061 + i,
                    mem_pool_total_size=1024,
                    mem_pool_available_size=1024,
                )
            )
            workers.append(worker)

        # Register replicas with different characteristics
        replicas_data = [
            (workers[0].worker_id, MemoryType.GPU, 2),  # GPU, low load
            (workers[1].worker_id, MemoryType.GPU, 8),  # GPU, high load
            (workers[2].worker_id, MemoryType.RAM, 5),  # RAM, medium load
        ]

        for i, (worker_id, mem_type, current_requests) in enumerate(replicas_data):
            model_service.register_replica(
                ModelReplica(
                    model_id="distributed_model",
                    node_id=f"node{i}",
                    node_address=f"192.168.1.{i+1}",
                    node_port=8080,
                    memory_size=1024,
                    memory_type=mem_type,
                    device_id=0,
                    worker_id=worker_id,
                    max_concurrency=10,
                    current_requests=current_requests,  # Set initial load directly
                )
            )

        # Request should go to GPU node with lowest load (node0)
        selected, _ = transport_service.request_transport(
            model_id="distributed_model",
            source_node_id="client",
            source_address="10.0.0.100",
            source_port=9999,
        )

        assert selected.node_id == "node0"
        assert selected.memory_type == MemoryType.GPU

    def test_concurrent_transport_requests(self, services):
        """Test handling concurrent transport requests."""
        worker_service = services["worker"]
        model_service = services["model"]
        transport_service = services["transport"]

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
                model_id="concurrent_model",
                node_id="node1",
                node_address="192.168.1.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                worker_id=worker.worker_id,
                max_concurrency=3,
            )
        )

        # Make concurrent requests up to the limit
        transport_ids = []
        for i in range(3):
            selected, transport_id = transport_service.request_transport(
                model_id="concurrent_model",
                source_node_id=f"client_{i}",
                source_address="192.168.2.1",
                source_port=9000 + i,
            )
            transport_ids.append(transport_id)
            assert selected.replica_id == replica.replica_id

        # Verify current load
        updated_replicas = model_service.list_replicas(model_id="concurrent_model")
        assert len(updated_replicas) == 1
        assert updated_replicas[0].current_requests == 3

        # Complete all transports
        for transport_id in transport_ids:
            result = transport_service.complete_transport(transport_id)
            assert result is not None

        # Verify load is back to zero
        final_replicas = model_service.list_replicas(model_id="concurrent_model")
        assert len(final_replicas) == 1
        assert final_replicas[0].current_requests == 0

    def test_worker_heartbeat_and_cleanup(self, services):
        """Test worker heartbeat and cleanup functionality."""
        worker_service = services["worker"]
        model_service = services["model"]

        # Register worker
        worker = worker_service.register_worker(
            Worker(
                node_id="heartbeat_node",
                node_address="192.168.1.100",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        # Register replica for this worker
        model_service.register_replica(
            ModelReplica(
                model_id="heartbeat_model",
                node_id="heartbeat_node",
                node_address="192.168.1.100",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                worker_id=worker.worker_id,
                max_concurrency=10,
            )
        )

        # Send heartbeat
        success = worker_service.heartbeat(worker.worker_id, 512, True)
        assert success is True

        # Verify worker is still active
        active_workers = worker_service.list_active_workers()
        worker_ids = [w.worker_id for w in active_workers]
        assert worker.worker_id in worker_ids

        # Simulate worker going offline by not sending heartbeats
        # In a real scenario, cleanup would happen automatically via scheduled task
        # For testing, we manually trigger worker removal
        success = worker_service.unregister_worker(worker.worker_id)
        assert success is True

        # Verify replica is marked unavailable
        updated_replicas = model_service.list_replicas(model_id="heartbeat_model")
        assert len(updated_replicas) == 1
        assert updated_replicas[0].is_available is False

    def test_model_replica_update_flow(self, services):
        """Test model replica registration and update flow."""
        worker_service = services["worker"]
        model_service = services["model"]

        # Register worker
        worker = worker_service.register_worker(
            Worker(
                node_id="update_node",
                node_address="192.168.1.200",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        # Initial replica registration
        initial_replica = model_service.register_replica(
            ModelReplica(
                model_id="update_model",
                node_id="update_node",
                node_address="192.168.1.200",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                worker_id=worker.worker_id,
                max_concurrency=5,
            )
        )

        # Verify initial state
        replicas = model_service.list_replicas(model_id="update_model")
        assert len(replicas) == 1
        assert replicas[0].max_concurrency == 5
        assert replicas[0].memory_size == 1024

        # Update replica with different parameters
        updated_replica = model_service.register_replica(
            ModelReplica(
                model_id="update_model",
                node_id="update_node",
                node_address="192.168.1.200",
                node_port=8080,
                memory_size=2048,  # Different memory size
                memory_type=MemoryType.GPU,
                device_id=0,
                worker_id=worker.worker_id,
                max_concurrency=10,  # Different concurrency
            )
        )

        # Verify update
        assert updated_replica.replica_id == initial_replica.replica_id
        assert updated_replica.memory_size == 2048
        assert updated_replica.max_concurrency == 10

        # Verify in database
        updated_replicas = model_service.list_replicas(model_id="update_model")
        assert len(updated_replicas) == 1
        assert updated_replicas[0].memory_size == 2048
        assert updated_replicas[0].max_concurrency == 10

    def test_cross_service_data_consistency(self, services):
        """Test data consistency across service boundaries."""
        worker_service = services["worker"]
        model_service = services["model"]
        transport_service = services["transport"]

        # Register worker
        worker = worker_service.register_worker(
            Worker(
                node_id="consistency_node",
                node_address="192.168.1.150",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
            )
        )

        # Register replica
        model_service.register_replica(
            ModelReplica(
                model_id="consistency_model",
                node_id="consistency_node",
                node_address="192.168.1.150",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                worker_id=worker.worker_id,
                max_concurrency=1,
            )
        )

        # Request transport
        selected, transport_id = transport_service.request_transport(
            model_id="consistency_model",
            source_node_id="consistency_client",
            source_address="192.168.2.100",
            source_port=9000,
        )

        # Verify consistency across services
        # Model service should show updated current_requests
        model_replicas = model_service.list_replicas(model_id="consistency_model")
        assert len(model_replicas) == 1
        assert model_replicas[0].current_requests == 1

        # Worker should still be available
        active_workers = worker_service.list_active_workers()
        worker_ids = [w.worker_id for w in active_workers]
        assert worker.worker_id in worker_ids

        # Complete transport
        result = transport_service.complete_transport(transport_id)
        assert result is not None

        # Verify consistency after completion
        final_replicas = model_service.list_replicas(model_id="consistency_model")
        assert len(final_replicas) == 1
        assert final_replicas[0].current_requests == 0

    def test_multiple_models_same_worker(self, services):
        """Test multiple models hosted on the same worker."""
        worker_service = services["worker"]
        model_service = services["model"]
        transport_service = services["transport"]

        # Register single worker
        worker = worker_service.register_worker(
            Worker(
                node_id="multi_model_node",
                node_address="192.168.1.250",
                grpc_port=50051,
                p2p_port=50052,
                mem_pool_total_size=4096,
                mem_pool_available_size=4096,
            )
        )

        # Register multiple models on same worker
        model_ids = ["model_a", "model_b", "model_c"]
        replicas = []

        for i, model_id in enumerate(model_ids):
            replica = model_service.register_replica(
                ModelReplica(
                    model_id=model_id,
                    node_id="multi_model_node",
                    node_address="192.168.1.250",
                    node_port=8080 + i,
                    memory_size=1024,
                    memory_type=MemoryType.GPU,
                    device_id=i,
                    worker_id=worker.worker_id,
                    max_concurrency=2,
                )
            )
            replicas.append(replica)

        # Request transports for different models
        transport_ids = {}
        for model_id in model_ids:
            selected, transport_id = transport_service.request_transport(
                model_id=model_id,
                source_node_id="multi_client",
                source_address="192.168.3.1",
                source_port=9000,
            )
            transport_ids[model_id] = transport_id
            assert selected.worker_id == worker.worker_id

        # Verify each model has one active request
        for model_id in model_ids:
            model_replicas = model_service.list_replicas(model_id=model_id)
            assert len(model_replicas) == 1
            assert model_replicas[0].current_requests == 1

        # Unregister worker - should affect all models
        success = worker_service.unregister_worker(worker.worker_id)
        assert success is True

        # Verify all replicas are marked unavailable
        for model_id in model_ids:
            model_replicas = model_service.list_replicas(model_id=model_id)
            assert len(model_replicas) == 1
            assert model_replicas[0].is_available is False
