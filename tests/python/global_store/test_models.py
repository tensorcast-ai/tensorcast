#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for Global Store domain models."""

from uuid import uuid4

import pytest

from tensorcast.global_store.models import MemoryType, Replica, Transport, Worker


class TestModels:
    """Test domain models."""

    def test_artifact_replica_creation(self):
        """Test Replica creation and properties."""
        replica = Replica(
            artifact_id="test_artifact",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            max_concurrency=10,
            current_requests=5,
        )

        assert replica.artifact_id == "test_artifact"
        assert replica.memory_type == MemoryType.GPU
        assert replica.load_ratio == 0.5
        assert replica.has_capacity is True

    def test_artifact_replica_increment_requests(self):
        """Test request counting."""
        replica = Replica(max_concurrency=2, current_requests=1)

        assert replica.increment_requests() is True
        assert replica.current_requests == 2
        assert replica.has_capacity is False
        assert replica.increment_requests() is False

    def test_artifact_replica_no_capacity(self):
        """Test replica at max capacity."""
        replica = Replica(
            artifact_id="test_artifact",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            max_concurrency=2,
            current_requests=2,
        )

        assert replica.has_capacity is False
        assert replica.load_ratio == 1.0

    def test_artifact_replica_zero_concurrency(self):
        """Test replica with zero max concurrency."""
        replica = Replica(
            artifact_id="test_artifact",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            max_concurrency=0,
            current_requests=0,
        )

        assert replica.has_capacity is False
        assert replica.load_ratio == 0.0

    def test_worker_creation(self):
        """Test Worker creation and properties."""
        worker = Worker(
            worker_id="worker_1",
            daemon_id="daemon_1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=512,
        )

        assert worker.memory_utilization == 50.0
        assert worker.is_healthy is False  # No heartbeat yet

    def test_worker_full_memory(self):
        """Test worker with full memory utilization."""
        worker = Worker(
            worker_id="worker_1",
            daemon_id="daemon_1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=0,
        )

        assert worker.memory_utilization == 100.0

    def test_worker_zero_total_memory(self):
        """Test worker with zero total memory."""
        worker = Worker(
            worker_id="worker_1",
            daemon_id="daemon_1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=0,
            mem_pool_available_size=0,
        )

        assert worker.memory_utilization == 0.0

    def test_worker_accepting_requests(self):
        """Test worker accepting requests status."""
        worker = Worker(
            worker_id="worker_1",
            daemon_id="daemon_1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=512,
            accepting_new_requests=True,
        )

        assert worker.accepting_new_requests is True

    def test_memory_type_priority(self):
        """Test memory type priority ordering."""
        # GPU should have highest priority (lowest value)
        assert MemoryType.GPU.priority < MemoryType.RAM.priority
        assert MemoryType.RAM.priority < MemoryType.DISK.priority

    def test_transport_creation(self):
        """Test Transport creation."""
        transport = Transport(
            transport_id=uuid4(),
            replica_id=uuid4(),
            artifact_id="test_artifact",
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
        )

        assert transport.artifact_id == "test_artifact"
        assert transport.source_node_id == "source_node"
        assert transport.source_address == "192.168.1.2"
        assert transport.source_port == 9000
