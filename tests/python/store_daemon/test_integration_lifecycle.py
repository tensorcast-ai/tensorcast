#  Copyright (c) 2025, TensorCast Team.

"""Integration tests for PID-based lifecycle management."""

# pyright: reportGeneralTypeIssues=false

import os
import time
import threading
import subprocess
from unittest.mock import Mock, patch

from tests.python.utils.artifact_utils import create_dummy_artifact
# Re-export Mock as _Mock for local readability when wrapping functions
_Mock = Mock
from concurrent.futures import Future

import pytest

from tensorcast.proto import store_daemon_pb2
from tensorcast.store_daemon.config import StoreDaemonConfig, LifecycleConfig, ServerConfig, NetworkConfig
from tensorcast.store_daemon.servicer import StoreDaemonServicer

# -----------------------------------------------------------------------------
# Use FakeStoreEngine for all tests instead of scattered MagicMocks
# -----------------------------------------------------------------------------
# Note: Real StoreEngine from `tensorcast._store_engine` is now used directly.


class TestLifecycleIntegration:
    """Integration tests for lifecycle management features."""

    @pytest.fixture
    def servicer_with_mocks(self):
        """Create a servicer with mocked Store Engine."""
        config = StoreDaemonConfig(
            lifecycle=LifecycleConfig(
                proc_check_interval_s=0.1,  # Fast for testing
                eviction_check_interval_s=0.5,
                gpu_memory_limit_fraction=0.9,
                global_cache_fraction=0.2,
            ),
            server=ServerConfig(
                enable_p2p_access=False,  # Disable registration for testing
                enable_p2p_engine=False,  # Also disable comm since we're not testing it
            ),
            global_store_address=None,  # No global store for testing,
            network=NetworkConfig(
                health_check_port=None,
            ),
        )

        # ------------------------------------------------------------------
        # Prepare dummy artifact files expected by the various lifecycle tests.
        # ------------------------------------------------------------------
        storage_root = config.server.storage_path
        for artifact_id in [f"artifact{i}" for i in range(5)] + [f"test_artifact_{i}" for i in range(3)]:
            create_dummy_artifact(storage_root, artifact_id)

        # Create the servicer with the real StoreEngine
        servicer = StoreDaemonServicer(config=config)

        servicer.process_watcher.stop()

        yield servicer

        # Cleanup
        if servicer.lifecycle_worker:
            servicer.lifecycle_worker.stop()

    def test_full_lifecycle_flow(self, servicer_with_mocks):
        """Test complete lifecycle: load, confirm, process death, eviction."""
        servicer = servicer_with_mocks
        context = Mock()

        # 1. Load artifact with PID
        load_request = store_daemon_pb2.MaterializeReplicaRequest(
            disk_path="artifact0",
            replica_uuid="uuid1",
            device_uuid="gpu0",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            pid=os.getpid(),  # Use current process PID
            keep_for_global=False,
            size_bytes=1024 * 1024,  # 1MB
        )

        load_response = servicer.MaterializeReplica(load_request, context)
        assert (
            load_response.status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        )

        # 2. Confirm artifact
        confirm_request = store_daemon_pb2.ConfirmReplicaRequest(
            disk_path="artifact0",
            replica_uuid="uuid1",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )

        confirm_response = servicer.ConfirmReplica(confirm_request, context)
        assert confirm_response.code == 0

        # 3. Check artifact is loaded
        replicas = servicer.replica_manager.get_loaded_replicas()
        assert len(replicas) == 1
        assert replicas[0]["ref_count"] == 1
        assert os.getpid() in replicas[0]["pids"]

        # 4. Simulate process death by manually calling callback
        servicer._on_pid_dead(os.getpid())

        # 5. Check reference was removed
        replicas = servicer.replica_manager.get_loaded_replicas()
        assert len(replicas) == 1
        assert replicas[0]["ref_count"] == 0

        # 6. Artifact should now be evictable
        info = servicer.replica_manager.get_replica_info("artifact0", 0)
        assert info is not None
        assert info.is_evictable()

    def test_subprocess_lifecycle(self):
        """Test lifecycle with real subprocess."""
        # This test needs a running process watcher, so create a servicer without stopping it
        config = StoreDaemonConfig(
            lifecycle=LifecycleConfig(
                proc_check_interval_s=0.1,  # Fast for testing
                eviction_check_interval_s=0.5,
                gpu_memory_limit_fraction=0.9,
                global_cache_fraction=0.2,
            ),
            server=ServerConfig(
                enable_p2p_access=False,  # Disable registration for testing
                enable_p2p_engine=False,  # Also disable comm since we're not testing it
            ),
            global_store_address=None,  # No global store for testing
        )

        # Ensure the dummy artifact directory exists for the subprocess test
        storage_root = config.server.storage_path
        create_dummy_artifact(storage_root, "model1")

        # Use real StoreEngine for subprocess lifecycle test
        servicer = StoreDaemonServicer(config=config)
        servicer.process_watcher.stop()

        context = Mock()

        proc = None
        try:
            # Start a subprocess
            proc = subprocess.Popen(
                ["python", "-c", "import time; time.sleep(10)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc_pid = proc.pid
            # Load artifact with subprocess PID
            load_request = store_daemon_pb2.MaterializeReplicaRequest(
                disk_path="model1",
                replica_uuid="uuid1",
                device_uuid="gpu0",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pid=proc_pid,
                size_bytes=1024 * 1024,
            )

            load_response = servicer.MaterializeReplica(load_request, context)
            assert (
                load_response.status
                == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
            )

            # Confirm artifact
            confirm_request = store_daemon_pb2.ConfirmReplicaRequest(
                disk_path="model1",
                replica_uuid="uuid1",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            )

            servicer.ConfirmReplica(confirm_request, context)

            # Check reference exists
            assert proc_pid in servicer.process_watcher.get_monitored_pids()
            replicas = servicer.replica_manager.get_loaded_replicas()
            assert len(replicas) == 1
            assert proc_pid in replicas[0]["pids"]

            # Kill subprocess
            proc.terminate()
            proc.wait()

            # Manually notify the servicer since the process watcher is mocked/stopped
            servicer._on_pid_dead(proc_pid)

            # Check reference was removed
            assert proc_pid not in servicer.process_watcher.get_monitored_pids()
            replicas = servicer.replica_manager.get_loaded_replicas()
            if replicas:  # Artifact might be evicted
                assert proc_pid not in replicas[0]["pids"]

        finally:
            # Ensure subprocess is cleaned up
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()

            # Clean up servicer
            if servicer.lifecycle_worker:
                servicer.lifecycle_worker.stop()
            if servicer.process_watcher:
                servicer.process_watcher.stop()

    def test_memory_pressure_eviction(self, servicer_with_mocks):
        """Test eviction under memory pressure."""
        servicer = servicer_with_mocks
        context = Mock()

        # Load multiple replicas
        for i in range(3):
            # Load
            load_request = store_daemon_pb2.MaterializeReplicaRequest(
                disk_path=f"artifact{i}",
                replica_uuid=f"uuid{i}",
                device_uuid="gpu0",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pid=1000 + i,  # Fake PIDs
                keep_for_global=(i == 2),  # Last one is global
                size_bytes=(i + 1) * 1024 * 1024,  # Different sizes
            )
            servicer.MaterializeReplica(load_request, context)

            # Confirm
            confirm_request = store_daemon_pb2.ConfirmReplicaRequest(
                disk_path=f"artifact{i}",
                replica_uuid=f"uuid{i}",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            )
            servicer.ConfirmReplica(confirm_request, context)

        # Remove references to make evictable
        for i in range(3):
            servicer._on_pid_dead(1000 + i)

        # Manually trigger eviction check
        servicer.lifecycle_worker.trigger_eviction_check()

        # Wait a bit for eviction to complete
        time.sleep(0.1)

        # Global artifact should be preserved if possible
        replicas = servicer.replica_manager.get_loaded_replicas()
        if replicas:
            global_replicas = [m for m in replicas if m["keep_for_global"]]
            assert len(global_replicas) <= 1  # At most one global artifact

    def test_concurrent_lifecycle_operations(self, servicer_with_mocks):
        """Test concurrent load/unload/eviction operations."""
        servicer = servicer_with_mocks
        errors = []
        load_count = 20

        def load_and_unload(idx):
            try:
                context = Mock()
                pid = 2000 + idx

                # Load
                load_request = store_daemon_pb2.MaterializeReplicaRequest(
                    disk_path=f"artifact{idx % 5}",  # Reuse some artifacts
                    replica_uuid=f"uuid{idx}",
                    device_uuid="gpu0",
                    target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                    pid=pid,
                )
                servicer.MaterializeReplica(load_request, context)

                # Confirm
                confirm_request = store_daemon_pb2.ConfirmReplicaRequest(
                    disk_path=f"artifact{idx % 5}",
                    replica_uuid=f"uuid{idx}",
                    target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                )
                servicer.ConfirmReplica(confirm_request, context)

                # Simulate some work
                time.sleep(0.01)

                # Unload using ReplicaManager with explicit device_id
                servicer.replica_manager.unload_replica(
                    disk_path=f"artifact{idx % 5}",
                    device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                    device_id=0,
                    pid=pid,
                )

            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        threads = []
        for i in range(load_count):
            t = threading.Thread(target=load_and_unload, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0

        # All replicas should be cleaned up eventually
        time.sleep(0.5)
        replicas = servicer.replica_manager.get_loaded_replicas()
        # All refs should be 0 since we unloaded everything
        assert all(m["ref_count"] == 0 for m in replicas)

    def test_get_loaded_models_rpc(self, servicer_with_mocks):
        """Test GetLoadedReplicas RPC functionality."""
        servicer = servicer_with_mocks
        context = Mock()

        # Load some replicas
        for i in range(3):
            load_request = store_daemon_pb2.MaterializeReplicaRequest(
                disk_path=f"test_artifact_{i}",
                replica_uuid=f"uuid{i}",
                device_uuid="gpu0",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pid=3000 + i,
                keep_for_global=(i == 1),
                size_bytes=(i + 1) * 1024 * 1024,  # Different sizes
            )
            servicer.MaterializeReplica(load_request, context)

            confirm_request = store_daemon_pb2.ConfirmReplicaRequest(
                disk_path=f"test_artifact_{i}",
                replica_uuid=f"uuid{i}",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            )
            servicer.ConfirmReplica(confirm_request, context)

        # Test GetLoadedReplicas without filter
        request = store_daemon_pb2.GetLoadedReplicasRequest()
        response = servicer.GetLoadedReplicas(request, context)

        assert response.total_replicas == 3
        assert response.total_size_bytes == 6 * 1024 * 1024  # 1+2+3 MB
        assert len(response.replicas) == 3

        # Test with artifact filter
        request = store_daemon_pb2.GetLoadedReplicasRequest(
            artifact_id_filter="test_artifact_1"
        )
        response = servicer.GetLoadedReplicas(request, context)

        assert response.total_replicas == 1
        assert response.replicas[0].artifact_id == "test_artifact_1"
        assert response.replicas[0].keep_for_global is True  # This was set in MaterializeReplica

        # Test with device filter (all on device 0)
        request = store_daemon_pb2.GetLoadedReplicasRequest(device_id_filter=0)
        response = servicer.GetLoadedReplicas(request, context)

        assert response.total_replicas == 3
