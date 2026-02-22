#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for Global Store repository layer."""

from datetime import datetime, timezone

import pytest

from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.global_store.models import ExportState, MemoryType, Replica, Worker
from tensorcast.global_store.repositories import ArtifactDiskLocationRepository
from tensorcast.global_store.repositories.base import BaseRepository


class TestRepositories:
    """Test repository layer."""

    def test_base_repository_transaction_keeps_primary_error_on_rollback_noop(self):
        class _FailingCursor:
            def execute(self, sql: str):
                if sql == "COMMIT":
                    raise RuntimeError("commit_conflict")
                if sql == "ROLLBACK":
                    raise RuntimeError("cannot rollback - no transaction is active")
                return self

            def close(self):
                return None

        class _FailingConnection:
            def cursor(self):
                return _FailingCursor()

        repo = BaseRepository(_FailingConnection())  # type: ignore[arg-type]
        with pytest.raises(DatabaseError) as exc, repo.transaction():
            pass
        assert "commit_conflict" in str(exc.value)
        assert "UnboundLocalError" not in str(exc.value)

    def test_worker_repository_crud(self, repositories):
        """Test Worker CRUD operations."""
        worker_repo = repositories["worker"]

        # Create
        worker = Worker(
            worker_id="test_worker",
            daemon_id="daemon_1",
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
        assert found.daemon_id == "daemon_1"
        assert found.node_id == "node1"

        # Update heartbeat
        success = worker_repo.update_heartbeat("test_worker", 512, True)
        assert success is True

        # Delete
        deleted = worker_repo.delete("test_worker")
        assert deleted is True
        assert worker_repo.find_by_id("test_worker") is None

    def test_worker_repository_find_by_daemon_id(self, repositories):
        """Test finding worker by stable daemon_id."""
        worker_repo = repositories["worker"]

        worker = Worker(
            worker_id="test_worker",
            daemon_id="daemon_abc",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        )
        worker_repo.create(worker)

        found = worker_repo.find_by_daemon_id("daemon_abc")
        assert found is not None
        assert found.worker_id == "test_worker"

        assert worker_repo.find_by_daemon_id("missing") is None

        # By default inactive workers are excluded.
        assert worker_repo.mark_inactive("test_worker") is True
        assert worker_repo.find_by_daemon_id("daemon_abc") is None
        assert (
            worker_repo.find_by_daemon_id("daemon_abc", include_inactive=True)
            is not None
        )

    def test_worker_repository_find_by_node(self, repositories):
        """Test finding worker by node_id."""
        worker_repo = repositories["worker"]

        # Create worker
        worker = Worker(
            worker_id="test_worker",
            daemon_id="daemon_unique_node",
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
                daemon_id=f"daemon_{i}",
                node_id=f"node_{i}",
                node_address=f"192.168.1.{i + 1}",
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

    def test_artifact_disk_locations_soft_delete_is_sticky(self, db_connection):
        repo = ArtifactDiskLocationRepository(db_connection)
        artifact_id = "mi2:idx:dat"
        cluster_id = "cluster-1"
        relative_path = "clusters/cluster-1/objects/mi2_idx_dat"

        repo.upsert(
            artifact_id=artifact_id,
            cluster_id=cluster_id,
            relative_path=relative_path,
            kind="MANAGED",
            is_deleted=False,
        )
        rows = repo.list_by_artifact(artifact_id)
        assert len(rows) == 1
        assert rows[0]["is_deleted"] is False
        assert rows[0]["deleted_at"] is None

        repo.upsert(
            artifact_id=artifact_id,
            cluster_id=cluster_id,
            relative_path=relative_path,
            kind="MANAGED",
            is_deleted=True,
        )
        assert repo.list_by_artifact(artifact_id) == []
        rows = repo.list_by_artifact(artifact_id, include_deleted=True)
        assert len(rows) == 1
        assert rows[0]["is_deleted"] is True
        assert rows[0]["deleted_at"] is not None

        # Deletion is sticky: an upsert must not revive a deleted location.
        repo.upsert(
            artifact_id=artifact_id,
            cluster_id=cluster_id,
            relative_path=relative_path,
            kind="MANAGED",
            is_deleted=False,
        )
        rows = repo.list_by_artifact(artifact_id, include_deleted=True)
        assert len(rows) == 1
        assert rows[0]["is_deleted"] is True

    def test_artifact_replica_repository_crud(self, repositories):
        """Test Replica CRUD operations."""
        replica_repo = repositories["replica"]

        # Create
        replica = Replica(
            artifact_id="test_artifact",
            node_id="node1",
            node_address="192.168.1.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            worker_id="worker1",
        )
        created = replica_repo.create(replica)
        assert created.artifact_id == "test_artifact"

        # Find by ID
        found = replica_repo.find_by_id(created.replica_id, "test_artifact")
        assert found is not None
        assert found.memory_type == MemoryType.GPU

        # Update
        found.max_concurrency = 20
        updated = replica_repo.update(found)
        assert updated.max_concurrency == 20

        # Delete
        deleted = replica_repo.delete(created.replica_id, "test_artifact")
        assert deleted is True

    def test_artifact_replica_repository_find_by_model(self, repositories):
        """Test finding replicas by artifact name."""
        replica_repo = repositories["replica"]

        # Create replicas for different models
        for i in range(3):
            replica = Replica(
                artifact_id=f"model_{i % 2}",  # Two different models
                node_id=f"node_{i}",
                node_address=f"192.168.1.{i + 1}",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=i,
                worker_id="worker1",
            )
            replica_repo.create(replica)

        # Find replicas for artifact name 'model_0'
        replicas = replica_repo.find_by_artifact("model_0")
        assert len(replicas) >= 2  # Created for indices 0 and 2

        # Find replicas for artifact name 'model_1'
        replicas = replica_repo.find_by_artifact("model_1")
        assert len(replicas) >= 1  # Created for index 1

    def test_artifact_replica_load_balancing(self, repositories):
        """Test load balancing query."""
        replica_repo = repositories["replica"]
        worker_repo = repositories["worker"]

        # Create worker first
        worker = Worker(
            worker_id="worker1",
            daemon_id="daemon_worker1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat("worker1", 1024, True) is True

        # Create replicas with different priorities
        replicas = [
            Replica(
                artifact_id="test_artifact",
                node_id="node1",
                node_address="192.168.1.1",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.DISK,
                device_id=0,
                max_concurrency=10,
                current_requests=5,
                worker_id="worker1",
            ),
            Replica(
                artifact_id="test_artifact",
                node_id="node2",
                node_address="192.168.1.2",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.GPU,
                device_id=0,
                remote_memory_keys=["rk0"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                max_concurrency=10,
                current_requests=2,
                worker_id="worker1",
            ),
            Replica(
                artifact_id="test_artifact",
                node_id="node3",
                node_address="192.168.1.3",
                node_port=8080,
                memory_size=1024,
                memory_type=MemoryType.RAM,
                device_id=0,
                remote_memory_keys=["rk1"],
                buffer_sizes=[1024],
                export_state=ExportState.EXPORTABLE,
                max_concurrency=10,
                current_requests=8,
                worker_id="worker1",
            ),
        ]

        for replica in replicas:
            replica_repo.create(replica)

        # Test load balancing selection
        selection = replica_repo.find_available_for_transport(
            "test_artifact", heartbeat_timeout_seconds=60
        )
        assert selection.replica is not None
        selected = selection.replica

        # Should select GPU replica (lowest load among GPU replicas)
        assert selected.memory_type == MemoryType.GPU
        assert selected.current_requests == 3  # Incremented by query

    def test_artifact_replica_load_balancing_avoids_small_capacity_bias(
        self, repositories
    ):
        """Prefer lower utilization instead of preferring smaller max_concurrency."""
        replica_repo = repositories["replica"]
        worker_repo = repositories["worker"]

        worker_low = Worker(
            worker_id="worker_low_cap",
            daemon_id="daemon_worker_low_cap",
            node_id="node_low_cap",
            node_address="192.168.10.1",
            grpc_port=50101,
            p2p_port=50102,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
        worker_high = Worker(
            worker_id="worker_high_cap",
            daemon_id="daemon_worker_high_cap",
            node_id="node_high_cap",
            node_address="192.168.10.2",
            grpc_port=50111,
            p2p_port=50112,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
        worker_repo.create(worker_low)
        worker_repo.create(worker_high)
        assert worker_repo.update_heartbeat("worker_low_cap", 1024, True) is True
        assert worker_repo.update_heartbeat("worker_high_cap", 1024, True) is True

        low_capacity = Replica(
            artifact_id="capacity_bias_artifact",
            node_id="node_low_cap",
            node_address="192.168.10.1",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            remote_memory_keys=["rk_low"],
            buffer_sizes=[1024],
            export_state=ExportState.EXPORTABLE,
            max_concurrency=2,
            current_requests=1,
            worker_id="worker_low_cap",
        )
        high_capacity = Replica(
            artifact_id="capacity_bias_artifact",
            node_id="node_high_cap",
            node_address="192.168.10.2",
            node_port=8080,
            memory_size=1024,
            memory_type=MemoryType.GPU,
            device_id=0,
            remote_memory_keys=["rk_high"],
            buffer_sizes=[1024],
            export_state=ExportState.EXPORTABLE,
            max_concurrency=20,
            current_requests=1,
            worker_id="worker_high_cap",
        )
        replica_repo.create(low_capacity)
        replica_repo.create(high_capacity)

        selection = replica_repo.find_available_for_transport(
            "capacity_bias_artifact", heartbeat_timeout_seconds=60
        )
        assert selection.replica is not None
        selected = selection.replica
        assert selected.node_id == "node_high_cap"
        assert selected.current_requests == 2

    def test_artifact_replica_no_available_for_transport(self, repositories):
        """Test when no replicas are available for transport."""
        replica_repo = repositories["replica"]

        # No replicas created
        selection = replica_repo.find_available_for_transport(
            "nonexistent_artifact", heartbeat_timeout_seconds=60
        )
        assert selection.replica is None
        assert selection.exportable_replicas == 0

    def test_artifact_replica_full_capacity(self, repositories):
        """Test replicas at full capacity."""
        replica_repo = repositories["replica"]
        worker_repo = repositories["worker"]

        # Create worker first
        worker = Worker(
            worker_id="worker1",
            daemon_id="daemon_worker1",
            node_id="node1",
            node_address="192.168.1.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat("worker1", 1024, True) is True

        # Create replica at full capacity
        replica = Replica(
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
            max_concurrency=2,
            current_requests=2,  # Full capacity
            worker_id="worker1",
        )
        replica_repo.create(replica)

        # Should not be selected for transport
        selection = replica_repo.find_available_for_transport(
            "test_artifact", heartbeat_timeout_seconds=60
        )
        assert selection.replica is None

    def test_transport_repository_crud(self, repositories):
        """Test Transport CRUD operations."""
        transport_repo = repositories["transport"]
        replica_repo = repositories["replica"]

        # Create a replica first for foreign key constraint
        replica = Replica(
            artifact_id="test_artifact",
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
        from tensorcast.global_store.models import Transport

        transport = Transport(
            replica_id=created_replica.replica_id,
            artifact_id="test_artifact",
            source_node_id="source_node",
            source_address="192.168.2.1",
            source_port=9000,
        )
        created_transport = transport_repo.create(transport)
        assert created_transport.artifact_id == "test_artifact"

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
            daemon_id="daemon_temp_worker",
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
            replica = Replica(
                artifact_id=f"worker_model_{i}",
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
            replicas = replica_repo.find_by_artifact(f"worker_model_{i}")
            assert len(replicas) == 1
            assert replicas[0].is_available is False

    def test_view_repository_upsert_and_get(self, repositories):
        """Views are upserted and fetched by composite key."""
        view_repo = repositories["view"]
        now = datetime.now(timezone.utc)

        view_repo.upsert(
            artifact_id="mi2:index:data",
            view_id="view-123",
            view_spec_json='{"ops":[]}',
            view_size=1024,
            view_data_hash="mhash123",
            verified_at=now,
        )

        row = view_repo.get(artifact_id="mi2:index:data", view_id="view-123")
        assert row is not None
        assert row["view_spec_json"] == '{"ops":[]}'
        assert row["view_size"] == 1024
        assert row["view_data_hash"] == "mhash123"
        assert row["verified_at"] is not None

        view_repo.upsert(
            artifact_id="mi2:index:data",
            view_id="view-123",
            view_spec_json='{"ops":[{"type":"narrow"}]}',
            view_size=2048,
            view_data_hash=None,
            verified_at=None,
        )

        updated = view_repo.get(artifact_id="mi2:index:data", view_id="view-123")
        assert updated is not None
        assert updated["view_spec_json"] == '{"ops":[{"type":"narrow"}]}'
        assert updated["view_size"] == 2048
        assert updated["view_data_hash"] is None
        assert updated["verified_at"] is None

    def test_leaf_repository_upsert_and_fetch(self, repositories):
        """Leaf digests persist values per space and index."""
        leaf_repo = repositories["leaf"]
        artifact_id = "mi2:index:data"
        space_kind = "C"
        space_id = "indexmh"

        leaf_repo.upsert_many(
            artifact_id=artifact_id,
            space_kind=space_kind,
            space_id=space_id,
            entries=[(0, b"\x00" * 32), (1, b"\x01" * 32)],
        )

        rows = leaf_repo.fetch(
            artifact_id=artifact_id,
            space_kind=space_kind,
            space_id=space_id,
        )
        assert [row.leaf_idx for row in rows] == [0, 1]
        assert rows[0].digest == b"\x00" * 32

        with pytest.raises(ValueError, match="leaves conflict"):
            leaf_repo.upsert_many(
                artifact_id=artifact_id,
                space_kind=space_kind,
                space_id=space_id,
                entries=[(1, b"\xff" * 32), (2, b"\x02" * 32)],
            )

        filtered = leaf_repo.fetch(
            artifact_id=artifact_id,
            space_kind=space_kind,
            space_id=space_id,
            leaf_idxs=[1, 2],
        )
        assert [row.leaf_idx for row in filtered] == [1]
        assert filtered[0].digest == b"\x01" * 32

        leaf_repo.upsert_many(
            artifact_id=artifact_id,
            space_kind=space_kind,
            space_id=space_id,
            entries=[(2, b"\x02" * 32)],
        )
        filtered = leaf_repo.fetch(
            artifact_id=artifact_id,
            space_kind=space_kind,
            space_id=space_id,
            leaf_idxs=[1, 2],
        )
        assert [row.leaf_idx for row in filtered] == [1, 2]
        assert filtered[0].digest == b"\x01" * 32
        assert filtered[1].digest == b"\x02" * 32
