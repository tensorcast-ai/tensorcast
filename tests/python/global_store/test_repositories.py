#  Copyright (c) 2025, StepCast Team.

"""Tests for Global Store repository layer."""

import pytest

from scstore.global_store.models import ModelReplica, Worker, MemoryType


class TestRepositories:
    """Test repository layer."""

    def test_worker_repository_crud(self, repositories):
        """Test Worker CRUD operations."""
        worker_repo = repositories["worker"]

        # Create
        worker = Worker(
            worker_id="test_worker",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        )
        created = worker_repo.create(worker)
        assert created.worker_id == "test_worker"

        # Read
        found = worker_repo.find_by_id("test_worker")
        assert found is not None
        assert found.node_id == "node1"

        # Update heartbeat
        success = worker_repo.update_heartbeat("test_worker", 512, True)
        assert success is True

        # Delete
        deleted = worker_repo.delete("test_worker")
        assert deleted is True
        assert worker_repo.find_by_id("test_worker") is None

    def test_worker_repository_find_by_node(self, repositories):
        """Test finding worker by node_id."""
        worker_repo = repositories["worker"]

        # Create worker
        worker = Worker(
            worker_id="test_worker",
            node_id="unique_node",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        )
        worker_repo.create(worker)

        # Find by node
        found = worker_repo.find_by_node_id("unique_node")
        assert found is not None
        assert found.worker_id == "test_worker"

    def test_worker_repository_list_active(self, repositories):
        """Test listing active workers."""
        worker_repo = repositories["worker"]

        # Create multiple workers
        for i in range(3):
            worker = Worker(
                worker_id=f"worker_{i}",
                node_id=f"node_{i}",
                node_address=f"192.168.1.{i+1}",
                grpc_port=50051 + i,
                p2p_port=50052 + i,
                mem_pool_total_size=1024,
                mem_pool_available_size=1024,
                accepting_new_requests=(i % 2 == 0),  # Alternate accepting status
            )
            worker_repo.create(worker)

        # List all workers
        all_workers = worker_repo.list_active()
        assert len(all_workers) >= 3

        # List only accepting workers
        accepting_workers = worker_repo.list_active(accepting_only=True)
        accepting_count = sum(1 for w in accepting_workers if w.accepting_new_requests)
        assert accepting_count >= 2  # At least workers 0 and 2

    def test_model_replica_repository_crud(self, repositories):
        """Test ModelReplica CRUD operations."""
        replica_repo = repositories["replica"]

        # Create
        replica = ModelReplica(
            model_name="test_model",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id="worker1",
        )
        created = replica_repo.create(replica)
        assert created.model_name == "test_model"

        # Find by ID
        found = replica_repo.find_by_id(created.replica_id, "test_model")
        assert found is not None
        assert found.memory_type == MemoryType.GPU

        # Update
        found.max_concurrency = 20
        updated = replica_repo.update(found)
        assert updated.max_concurrency == 20

        # Delete
        deleted = replica_repo.delete(created.replica_id, "test_model")
        assert deleted is True

    def test_model_replica_repository_find_by_model(self, repositories):
        """Test finding replicas by model name."""
        replica_repo = repositories["replica"]

        # Create replicas for different models
        for i in range(3):
            replica = ModelReplica(
                model_name=f"model_{i % 2}",  # Two different models
                node_id=f"node_{i}",
                node_address=f"192.168.1.{i+1}",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=i,
                worker_id="worker1",
            )
            replica_repo.create(replica)

        # Find replicas for model_0
        model_0_replicas = replica_repo.find_by_model("model_0")
        assert len(model_0_replicas) >= 2  # Created for indices 0 and 2

        # Find replicas for model_1
        model_1_replicas = replica_repo.find_by_model("model_1")
        assert len(model_1_replicas) >= 1  # Created for index 1

    def test_model_replica_load_balancing(self, repositories):
        """Test load balancing query."""
        replica_repo = repositories["replica"]
        worker_repo = repositories["worker"]

        # Create worker first
        worker = Worker(
            worker_id="worker1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
        worker_repo.create(worker)

        # Create replicas with different priorities
        replicas = [
            ModelReplica(
                model_name="test_model",
                node_id="node1",
                node_address="192.168.1.1",
                node_port=8080,
                memory_type=MemoryType.DISK,
                device_id=0,
                max_concurrency=10,
                current_requests=5,
                worker_id="worker1",
            ),
            ModelReplica(
                model_name="test_model",
                node_id="node2",
                node_address="192.168.1.2",
                node_port=8080,
                memory_type=MemoryType.GPU,
                device_id=0,
                max_concurrency=10,
                current_requests=2,
                worker_id="worker1",
            ),
            ModelReplica(
                model_name="test_model",
                node_id="node3",
                node_address="192.168.1.3",
                node_port=8080,
                memory_type=MemoryType.RAM,
                device_id=0,
                max_concurrency=10,
                current_requests=8,
                worker_id="worker1",
            ),
        ]

        for replica in replicas:
            replica_repo.create(replica)

        # Test load balancing selection
        selected = replica_repo.find_available_for_transport(
            "test_model", heartbeat_timeout_seconds=60
        )

        # Should select GPU replica (lowest load among GPU replicas)
        assert selected is not None
        assert selected.memory_type == MemoryType.GPU
        assert selected.current_requests == 3  # Incremented by query

    def test_model_replica_no_available_for_transport(self, repositories):
        """Test when no replicas are available for transport."""
        replica_repo = repositories["replica"]

        # No replicas created
        selected = replica_repo.find_available_for_transport(
            "nonexistent_model", heartbeat_timeout_seconds=60
        )
        assert selected is None

    def test_model_replica_full_capacity(self, repositories):
        """Test replicas at full capacity."""
        replica_repo = repositories["replica"]
        worker_repo = repositories["worker"]

        # Create worker first
        worker = Worker(
            worker_id="worker1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
        worker_repo.create(worker)

        # Create replica at full capacity
        replica = ModelReplica(
            model_name="test_model",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_type=MemoryType.GPU,
            device_id=0,
            max_concurrency=2,
            current_requests=2,  # Full capacity
            worker_id="worker1",
        )
        replica_repo.create(replica)

        # Should not be selected for transport
        selected = replica_repo.find_available_for_transport(
            "test_model", heartbeat_timeout_seconds=60
        )
        assert selected is None

    def test_transport_repository_crud(self, repositories):
        """Test Transport CRUD operations."""
        transport_repo = repositories["transport"]
        replica_repo = repositories["replica"]

        # Create a replica first for foreign key constraint
        replica = ModelReplica(
            model_name="test_model",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id="worker1",
        )
        created_replica = replica_repo.create(replica)

        # Create transport
        from scstore.global_store.models import Transport
        transport = Transport(
            replica_id=created_replica.replica_id,
            model_name="test_model",
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9000,
        )
        created_transport = transport_repo.create(transport)
        assert created_transport.model_name == "test_model"

        # Find by ID
        found = transport_repo.find_by_id(created_transport.transport_id)
        assert found is not None
        assert found.source_node_id == "source_node"

        # Delete
        deleted = transport_repo.delete(created_transport.transport_id)
        assert deleted is True
        assert transport_repo.find_by_id(created_transport.transport_id) is None

    def test_replica_by_worker_cleanup(self, repositories):
        """Test marking replicas unavailable when worker is removed."""
        replica_repo = repositories["replica"]
        worker_repo = repositories["worker"]

        # Create worker
        worker = Worker(
            worker_id="temp_worker",
            node_id="temp_node",
            node_address="192.168.1.100",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        )
        worker_repo.create(worker)

        # Create replicas for this worker
        for i in range(3):
            replica = ModelReplica(
                model_name=f"worker_model_{i}",
                node_id="temp_node",
                node_address="192.168.1.100",
                node_port=8080 + i,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=i,
                worker_id="temp_worker",
            )
            replica_repo.create(replica)

        # Mark replicas unavailable by worker
        updated_count = replica_repo.mark_unavailable_by_worker("temp_worker")
        assert updated_count == 3

        # Verify replicas are marked unavailable
        for i in range(3):
            replicas = replica_repo.find_by_model(f"worker_model_{i}")
            assert len(replicas) == 1
            assert replicas[0].is_available is False