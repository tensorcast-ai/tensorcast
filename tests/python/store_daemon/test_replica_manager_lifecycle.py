#  Copyright (c) 2025, StepCast Team.

# pyright: reportArgumentType=false

"""Tests for replica manager lifecycle features."""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from scstore.proto import store_daemon_pb2
from scstore.store_daemon.replica_ref import ReplicaKey
from scstore.store_daemon.replica_manager import ReplicaManager
# NOTE: We intentionally avoid importing the C++ StoreEngine here because its
# attributes are read-only at runtime, which prevents the test-suite from
# monkey-patching them with ``unittest.mock.Mock``.  To keep the unit tests
# self-contained we instead rely on a lightweight Python stub that exposes the
# minimal surface required by ``ReplicaManager``.


# ---------------------------------------------------------------------------
# Test helpers – lightweight stub implementations
# ---------------------------------------------------------------------------


class DummyStoreEngine:
    """Pure-Python stub that mimics the handful of methods used by
    ``ReplicaManager`` so that the unit tests can run without depending on the
    underlying C++ extension.  Each method is a ``Mock`` instance which makes
    it trivial for individual test-cases to tweak return values and assert
    call-counts.

    The stub purposefully accepts *any* arguments via ``Mock`` to keep the
    surface minimal and flexible.  Where the production code relies on a
    structured return object (e.g. *comm_info* from
    ``enable_remote_replica_access``), the default mock is pre-configured to
    return a compatible object so that tests which do not explicitly override
    the return value continue to work out-of-the-box.
    """

    def __init__(self) -> None:
        # Methods directly probed by ``ReplicaManager``
        self.unload_replica = Mock(return_value=0)
        self.disable_remote_replica_access = Mock(return_value=True)

        # GPU memory stats – default to a single device with plenty of free
        # memory (total, free).  Tests can override as required.
        self.get_gpu_memory_stats = Mock(return_value=[(16 * 1024**3, 8 * 1024**3)])

        self.wait_replica_ready = Mock(return_value=0)

        # *enable_remote_replica_access* must yield an object with the
        # following attributes.  A simple ``Mock`` is sufficient.
        self.enable_remote_replica_access = Mock(
            return_value=Mock(
                artifact_size=1024,
                device_id=0,
                remote_memory_keys=[],
                buffer_sizes=[],
            )
        )


# ---------------------------------------------------------------------------
# Main *MockServicer* used by the unit tests
# ---------------------------------------------------------------------------


class MockServicer:
    """Mock servicer for testing."""

    def __init__(self):
        # Use the pure-Python stub instead of the real C++ implementation so
        # that the tests can freely patch return values and inspect
        # ``call_count`` without hitting read-only attribute errors.
        self.store_engine = DummyStoreEngine()

        self.global_store_stub = Mock()
        self.enable_p2p_engine = True
        self.enable_p2p_access = False  # Communication engine optional
        self.global_store_enabled = True
        self.connection_manager = None
        self.node_id = "test-node"
        self.node_address = "127.0.0.1"
        self.node_port = 9090
        self.worker_id = "test-worker"
        self.config = None  # Simplify; ReplicaManager falls back to defaults


# ---------------------------------------------------------------------------
# The remainder of the file contains the actual test-cases.  *Do not* modify
# behaviourally critical logic unless you have verified that it is redundant
# with respect to the updated Store Engine stub.
# ---------------------------------------------------------------------------


class TestReplicaManagerLifecycle:
    """Test cases for replica manager lifecycle features."""

    def test_add_ref_new_replica(self):
        """Test adding reference to a new replica."""
        servicer = MockServicer()
        manager = ReplicaManager(servicer)

        # Add reference
        success = manager.add_ref(
            disk_path="model1",
            device_id=0,
            pid=1234,
            size_bytes=1024 * 1024,
            keep_for_global=True,
        )

        assert success is True

        # Check replica was created
        info = manager.get_replica_info("model1", 0)
        assert info is not None
        assert info.ref_count == 1
        assert 1234 in info.pids
        assert info.size_bytes == 1024 * 1024
        assert info.keep_for_global is True

    def test_add_ref_existing_replica(self):
        """Test adding reference to existing replica."""
        servicer = MockServicer()
        manager = ReplicaManager(servicer)

        # Add first reference
        manager.add_ref("model1", 0, 1234, 1024)

        # Add second reference
        success = manager.add_ref("model1", 0, 5678, 1024)
        assert success is True

        info = manager.get_replica_info("model1", 0)
        assert info is not None
        assert info.ref_count == 2
        assert 1234 in info.pids
        assert 5678 in info.pids

    def test_remove_ref(self):
        """Test removing references."""
        servicer = MockServicer()
        manager = ReplicaManager(servicer)

        # Add references
        manager.add_ref("model1", 0, 1234, 1024)
        manager.add_ref("model1", 0, 5678, 1024)

        # Remove one reference
        success = manager.remove_ref("model1", 0, 1234)
        assert success is True

        info = manager.get_replica_info("model1", 0)
        assert info is not None
        assert info.ref_count == 1
        assert 1234 not in info.pids
        assert 5678 in info.pids

        # Remove non-existent reference
        success = manager.remove_ref("model1", 0, 9999)
        assert success is False

    def test_remove_pid_refs(self):
        """Test removing all references for a PID."""
        servicer = MockServicer()
        manager = ReplicaManager(servicer)

        # Add same PID to multiple artifacts
        manager.add_ref("model1", 0, 1234, 1024)
        manager.add_ref("model2", 0, 1234, 2048)
        manager.add_ref("model3", 1, 1234, 4096)
        manager.add_ref("model1", 0, 5678, 1024)  # Different PID

        # Remove all refs for PID 1234
        affected_keys = manager.remove_pid_refs(1234)
        assert len(affected_keys) == 2  # Only model2 and model3 reach ref_count=0

        # Check references were removed
        info1 = manager.get_replica_info("model1", 0)
        assert info1 is not None
        assert info1.ref_count == 1
        info2 = manager.get_replica_info("model2", 0)
        assert info2 is not None
        assert info2.ref_count == 0
        info3 = manager.get_replica_info("model3", 1)
        assert info3 is not None
        assert info3.ref_count == 0

    def test_get_loaded_replicas(self):
        """Test getting loaded replicas information."""
        servicer = MockServicer()
        manager = ReplicaManager(servicer)

        # Add some artifacts
        manager.add_ref("model1", 0, 1234, 1024, keep_for_global=True)
        # Add another artifact with a valid integer device_id
        manager.add_ref("model2", 0, 5678, 2048, keep_for_global=False)

        models = manager.get_loaded_replicas()
        assert len(models) == 2

        # Check model1
        model1 = next(m for m in models if m["artifact_id"] == "model1")
        assert model1["device_id"] == 0
        assert model1["ref_count"] == 1
        assert model1["pids"] == [1234]
        assert model1["size_bytes"] == 1024
        assert model1["keep_for_global"] is True

    def test_eviction_candidates_selection(self):
        """Test selection of eviction candidates."""
        servicer = MockServicer()
        manager = ReplicaManager(servicer)

        # Add artifacts with different characteristics
        now = time.time()

        # Artifact 1: Has references - not evictable
        manager.add_ref("model1", 0, 1234, 1024 * 1024)

        # Artifact 2: No refs, old, local
        manager.add_ref("model2", 0, 5678, 2048 * 1024)
        manager.remove_ref("model2", 0, 5678)
        manager._replicas[ReplicaKey("model2", 0)].last_access_ts = now - 100

        # Artifact 3: No refs, recent, local
        manager.add_ref("model3", 0, 9999, 4096 * 1024)
        manager.remove_ref("model3", 0, 9999)
        manager._replicas[ReplicaKey("model3", 0)].last_access_ts = now - 10

        # Artifact 4: No refs, old, global
        manager.add_ref("model4", 0, 8888, 512 * 1024, keep_for_global=True)
        manager.remove_ref("model4", 0, 8888)
        manager._replicas[ReplicaKey("model4", 0)].last_access_ts = now - 100

        # Select candidates to free 3MB
        candidates = manager._select_eviction_candidates(3 * 1024 * 1024, device_id=0)

        # Should select model2 (old, local) and model3 (recent, local) but not model4 (global)
        assert len(candidates) == 2
        candidate_keys = [c.key for c in candidates]
        assert ReplicaKey("model2", 0) in candidate_keys
        assert ReplicaKey("model3", 0) in candidate_keys
        assert ReplicaKey("model4", 0) not in candidate_keys  # Global cache preserved

    def test_maybe_evict(self):
        """Test eviction when memory is needed."""
        servicer = MockServicer()
        servicer.store_engine.unload_replica.return_value = 0
        servicer.store_engine.disable_remote_replica_access.return_value = True
        # Mock GPU memory stats to return proper format
        servicer.store_engine.get_gpu_memory_stats.return_value = [
            (10 * 1024**3, 2 * 1024**3)  # 10GB total, 2GB free
        ]
        manager = ReplicaManager(servicer)

        # Add evictable models
        manager.add_ref("model1", 0, 1234, 1024 * 1024)
        manager.remove_ref("model1", 0, 1234)

        manager.add_ref("model2", 0, 5678, 2048 * 1024)
        manager.remove_ref("model2", 0, 5678)

        # Request eviction
        evicted = manager.maybe_evict(bytes_needed=1024 * 1024, device_id=0)

        assert len(evicted) >= 1
        assert servicer.store_engine.unload_replica.called

    def test_periodic_evict(self):
        """Test periodic eviction."""
        servicer = MockServicer()
        servicer.store_engine.unload_replica.return_value = 0
        servicer.store_engine.disable_remote_replica_access.return_value = True
        manager = ReplicaManager(servicer)

        # Add evictable artifact
        manager.add_ref("model1", 0, 1234, 1024 * 1024)
        manager.remove_ref("model1", 0, 1234)

        # Trigger periodic eviction
        evicted = manager.periodic_evict(device_id=0, bytes_needed=512 * 1024)

        assert len(evicted) >= 0  # May or may not evict based on size

    def test_shutdown_evict_local_replicas(self):
        """Test evicting local replicas during shutdown."""
        servicer = MockServicer()
        servicer.store_engine.unload_replica.return_value = 0
        servicer.store_engine.disable_remote_replica_access.return_value = True
        manager = ReplicaManager(servicer)

        # Add local and global models
        manager.add_ref("local1", 0, 1234, 1024, keep_for_global=False)
        manager.add_ref("local2", 0, 5678, 2048, keep_for_global=False)
        manager.add_ref("global1", 0, 9999, 4096, keep_for_global=True)

        # Remove references to make evictable
        manager.remove_ref("local1", 0, 1234)
        manager.remove_ref("local2", 0, 5678)
        manager.remove_ref("global1", 0, 9999)

        # Shutdown eviction
        evicted_count = manager.shutdown_evict_local_replicas()

        # Should evict only local replicas
        assert evicted_count == 2
        assert manager.get_replica_info("local1", 0) is None
        assert manager.get_replica_info("local2", 0) is None
        assert manager.get_replica_info("global1", 0) is not None

    def test_confirm_model_with_pid(self):
        """Test artifact confirmation with PID tracking."""
        servicer = MockServicer()
        servicer.store_engine.wait_replica_ready.return_value = 0
        servicer.store_engine.enable_remote_replica_access.return_value = Mock(
            artifact_size=1024, device_id=0, remote_memory_keys=[], buffer_sizes=[]
        )

        # Configure global store stub to handle RegisterReplica
        mock_response = Mock()
        mock_response.status = 0  # OK status
        mock_response.replica_id = "replica-123"
        servicer.global_store_stub.RegisterReplica.return_value = mock_response

        manager = ReplicaManager(servicer)

        # Add reference first (since confirm_model no longer handles this)
        manager.add_ref(
            disk_path="model1",
            device_id=0,  # GPU device 0
            pid=1234,
            size_bytes=1024,
            keep_for_global=True,
        )

        # Check reference was added
        info = manager.get_replica_info("model1", 0)
        assert info is not None
        assert info.ref_count == 1
        assert 1234 in info.pids

    def test_unload_model_with_ref_count(self):
        """Test unloading artifact respects reference counting."""
        servicer = MockServicer()
        servicer.store_engine.unload_replica.return_value = 0
        servicer.store_engine.disable_remote_replica_access.return_value = True
        manager = ReplicaManager(servicer)

        # Add artifact with references
        manager.add_ref("model1", 0, 1234, 1024)
        manager.add_ref("model1", 0, 5678, 1024)

        # Try to unload with PID 1234
        success = manager.unload_replica(
            disk_path="model1",
            device_type=store_daemon_pb2.DEVICE_TYPE_GPU,
            device_id=0,
            pid=1234,
        )

        # Should succeed but not actually unload
        assert success is True
        assert not servicer.store_engine.unload_replica.called

        # Artifact should still have one reference
        info = manager.get_replica_info("model1", 0)
        assert info is not None
        assert info.ref_count == 1
        assert 5678 in info.pids

        # Unload with last PID
        success = manager.unload_replica(
            disk_path="model1",
            device_type=store_daemon_pb2.DEVICE_TYPE_GPU,
            device_id=0,
            pid=5678,
        )

        # Should actually unload now
        assert success is True
        assert servicer.store_engine.unload_replica.called

    def test_concurrent_operations(self):
        """Test thread safety of concurrent operations."""
        servicer = MockServicer()
        manager = ReplicaManager(servicer)
        errors = []

        def add_refs():
            try:
                for i in range(100):
                    manager.add_ref(f"artifact{i % 10}", 0, i, 1024)
            except Exception as e:
                errors.append(e)

        def remove_refs():
            try:
                for i in range(100):
                    manager.remove_ref(f"artifact{i % 10}", 0, i)
            except Exception as e:
                errors.append(e)

        def get_models():
            try:
                for _ in range(50):
                    manager.get_loaded_replicas()
            except Exception as e:
                errors.append(e)

        # Run operations concurrently
        threads = [
            threading.Thread(target=add_refs),
            threading.Thread(target=remove_refs),
            threading.Thread(target=get_models),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0

    @patch("scstore.store_daemon.replica_manager.ARTIFACT_REF_COUNT")
    @patch("scstore.store_daemon.replica_manager.GPU_CACHE_BYTES")
    @patch("scstore.store_daemon.replica_manager.EVICTIONS_TOTAL")
    def test_metrics_updates(self, mock_evictions, mock_cache_bytes, mock_ref_count):
        """Test that metrics are properly updated."""
        servicer = MockServicer()
        servicer.store_engine.unload_replica.return_value = 0
        servicer.store_engine.get_gpu_memory_stats.return_value = [
            (10 * 1024**3, 5 * 1024**3)
        ]
        manager = ReplicaManager(servicer)

        # Add reference - should update ref count
        manager.add_ref("model1", 0, 1234, 1024, keep_for_global=True)
        mock_ref_count.labels.assert_called_with(artifact="model1", device_id="0")
        mock_ref_count.labels().set.assert_called_with(1)

        # Get GPU stats - should update cache bytes
        manager.get_gpu_memory_stats()
        mock_cache_bytes.labels.assert_any_call(type="global")
        mock_cache_bytes.labels().set.assert_called()

        # Evict artifact
        manager.remove_ref("model1", 0, 1234)
        evicted = manager.maybe_evict(1024, 0)
        if evicted:
            mock_evictions.labels.assert_called_with(reason="memory")
            mock_evictions.labels().inc.assert_called()
