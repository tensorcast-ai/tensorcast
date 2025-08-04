#  Copyright (c) 2025, StepCast Team.

"""Integration tests for PID-based lifecycle management."""

# pyright: reportGeneralTypeIssues=false

import os
import time
import threading
import subprocess
from unittest.mock import Mock, patch

from tests.python.utils.model_utils import create_dummy_model
# Re-export Mock as _Mock for local readability when wrapping functions
_Mock = Mock
from concurrent.futures import Future

import pytest

from scstore.proto import store_daemon_pb2
from scstore.store_daemon.config import StoreDaemonConfig, LifecycleConfig, ServerConfig, NetworkConfig
from scstore.store_daemon.servicer import StoreDaemonServicer

# -----------------------------------------------------------------------------
# Use FakeCheckpointStore for all tests instead of scattered MagicMocks
# -----------------------------------------------------------------------------
# Note: Real CheckpointStore from `scstore._checkpoint_store` is now used directly.


class TestLifecycleIntegration:
    """Integration tests for lifecycle management features."""

    @pytest.fixture
    def servicer_with_mocks(self):
        """Create a servicer with mocked checkpoint store."""
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
        # Prepare dummy model files expected by the various lifecycle tests.
        # ------------------------------------------------------------------
        storage_root = config.server.storage_path
        for model_name in [f"model{i}" for i in range(5)] + [f"test_model_{i}" for i in range(3)]:
            create_dummy_model(storage_root, model_name)

        # Create the servicer with the real CheckpointStore
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

        # 1. Load model with PID
        load_request = store_daemon_pb2.LoadModelRequest(
            model_path="model1",
            replica_uuid="uuid1",
            device_uuid="gpu0",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            pid=os.getpid(),  # Use current process PID
            keep_for_global=False,
            size_bytes=1024 * 1024,  # 1MB
        )

        load_response = servicer.LoadModel(load_request, context)
        assert (
            load_response.status
            == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
        )

        # 2. Confirm model
        confirm_request = store_daemon_pb2.ConfirmModelRequest(
            model_path="model1",
            replica_uuid="uuid1",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )

        confirm_response = servicer.ConfirmModel(confirm_request, context)
        assert confirm_response.code == 0

        # 3. Check model is loaded
        models = servicer.replica_manager.get_loaded_models()
        assert len(models) == 1
        assert models[0]["ref_count"] == 1
        assert os.getpid() in models[0]["pids"]

        # 4. Simulate process death by manually calling callback
        servicer._on_pid_dead(os.getpid())

        # 5. Check reference was removed
        models = servicer.replica_manager.get_loaded_models()
        assert len(models) == 1
        assert models[0]["ref_count"] == 0

        # 6. Model should now be evictable
        info = servicer.replica_manager.get_replica_info("model1", 0)
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

        # Ensure the dummy model directory exists for the subprocess test
        storage_root = config.server.storage_path
        create_dummy_model(storage_root, "model1")

        # Use real CheckpointStore for subprocess lifecycle test
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
            # Load model with subprocess PID
            load_request = store_daemon_pb2.LoadModelRequest(
                model_path="model1",
                replica_uuid="uuid1",
                device_uuid="gpu0",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pid=proc_pid,
                size_bytes=1024 * 1024,
            )

            load_response = servicer.LoadModel(load_request, context)
            assert (
                load_response.status
                == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
            )

            # Confirm model
            confirm_request = store_daemon_pb2.ConfirmModelRequest(
                model_path="model1",
                replica_uuid="uuid1",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            )

            servicer.ConfirmModel(confirm_request, context)

            # Check reference exists
            assert proc_pid in servicer.process_watcher.get_monitored_pids()
            models = servicer.replica_manager.get_loaded_models()
            assert len(models) == 1
            assert proc_pid in models[0]["pids"]

            # Kill subprocess
            proc.terminate()
            proc.wait()

            # Manually notify the servicer since the process watcher is mocked/stopped
            servicer._on_pid_dead(proc_pid)

            # Check reference was removed
            assert proc_pid not in servicer.process_watcher.get_monitored_pids()
            models = servicer.replica_manager.get_loaded_models()
            if models:  # Model might be evicted
                assert proc_pid not in models[0]["pids"]

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

        # Load multiple models
        for i in range(3):
            # Load
            load_request = store_daemon_pb2.LoadModelRequest(
                model_path=f"model{i}",
                replica_uuid=f"uuid{i}",
                device_uuid="gpu0",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pid=1000 + i,  # Fake PIDs
                keep_for_global=(i == 2),  # Last one is global
                size_bytes=(i + 1) * 1024 * 1024,  # Different sizes
            )
            servicer.LoadModel(load_request, context)

            # Confirm
            confirm_request = store_daemon_pb2.ConfirmModelRequest(
                model_path=f"model{i}",
                replica_uuid=f"uuid{i}",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            )
            servicer.ConfirmModel(confirm_request, context)

        # Remove references to make evictable
        for i in range(3):
            servicer._on_pid_dead(1000 + i)

        # Manually trigger eviction check
        servicer.lifecycle_worker.trigger_eviction_check()

        # Wait a bit for eviction to complete
        time.sleep(0.1)

        # Global model should be preserved if possible
        models = servicer.replica_manager.get_loaded_models()
        if models:
            global_models = [m for m in models if m["keep_for_global"]]
            assert len(global_models) <= 1  # At most one global model

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
                load_request = store_daemon_pb2.LoadModelRequest(
                    model_path=f"model{idx % 5}",  # Reuse some models
                    replica_uuid=f"uuid{idx}",
                    device_uuid="gpu0",
                    target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                    pid=pid,
                )
                servicer.LoadModel(load_request, context)

                # Confirm
                confirm_request = store_daemon_pb2.ConfirmModelRequest(
                    model_path=f"model{idx % 5}",
                    replica_uuid=f"uuid{idx}",
                    target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                )
                servicer.ConfirmModel(confirm_request, context)

                # Simulate some work
                time.sleep(0.01)

                # Unload using ReplicaManager with explicit device_id
                servicer.replica_manager.unload_model(
                    model_path=f"model{idx % 5}",
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

        # All models should be cleaned up eventually
        time.sleep(0.5)
        models = servicer.replica_manager.get_loaded_models()
        # All refs should be 0 since we unloaded everything
        assert all(m["ref_count"] == 0 for m in models)

    def test_get_loaded_models_rpc(self, servicer_with_mocks):
        """Test GetLoadedModels RPC functionality."""
        servicer = servicer_with_mocks
        context = Mock()

        # Load some models
        for i in range(3):
            load_request = store_daemon_pb2.LoadModelRequest(
                model_path=f"test_model_{i}",
                replica_uuid=f"uuid{i}",
                device_uuid="gpu0",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pid=3000 + i,
                keep_for_global=(i == 1),
                size_bytes=(i + 1) * 1024 * 1024,  # Different sizes
            )
            servicer.LoadModel(load_request, context)

            confirm_request = store_daemon_pb2.ConfirmModelRequest(
                model_path=f"test_model_{i}",
                replica_uuid=f"uuid{i}",
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            )
            servicer.ConfirmModel(confirm_request, context)

        # Test GetLoadedModels without filter
        request = store_daemon_pb2.GetLoadedModelsRequest()
        response = servicer.GetLoadedModels(request, context)

        assert response.total_models == 3
        assert response.total_size_bytes == 6 * 1024 * 1024  # 1+2+3 MB
        assert len(response.models) == 3

        # Test with model filter
        request = store_daemon_pb2.GetLoadedModelsRequest(
            model_id_filter="test_model_1"
        )
        response = servicer.GetLoadedModels(request, context)

        assert response.total_models == 1
        assert response.models[0].model_id == "test_model_1"
        assert response.models[0].keep_for_global is True  # This was set in LoadModel

        # Test with device filter (all on device 0)
        request = store_daemon_pb2.GetLoadedModelsRequest(device_id_filter=0)
        response = servicer.GetLoadedModels(request, context)

        assert response.total_models == 3
