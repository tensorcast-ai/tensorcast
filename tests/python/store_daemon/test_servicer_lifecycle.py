#  Copyright (c) 2025, StepCast Team.

"""Tests for servicer lifecycle features."""

import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

import grpc
from pydantic import ByteSize

from scstore.proto import store_daemon_pb2
from scstore.store_daemon.config import LifecycleConfig, ServerConfig, StoreDaemonConfig
from scstore.store_daemon.servicer import StoreDaemonServicer


class MockContext:
    """Mock gRPC context for testing."""

    def __init__(self):
        self.code = None
        self.details = None
        self.metadata = {}

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class TestServicerLifecycle:
    """Test cases for servicer lifecycle features."""

    def test_servicer_initialization_with_lifecycle(self):
        """Test servicer initialization with lifecycle components."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            ),
            lifecycle=LifecycleConfig(
                proc_check_interval_s=5.0,
                eviction_check_interval_s=30.0,
                gpu_memory_limit_fraction=0.9,
                global_cache_fraction=0.2,
            ),
        )

        servicer = StoreDaemonServicer(config)

        # Check components were created
        assert servicer.replica_manager is not None
        assert servicer.process_watcher is not None
        assert servicer.lifecycle_worker is not None

        # Check process watcher configuration
        assert servicer.process_watcher.check_interval == 5.0
        assert servicer.process_watcher.on_pid_dead == servicer._on_pid_dead

        # Check lifecycle worker configuration
        assert servicer.lifecycle_worker.eviction_interval == 30.0
        assert servicer.lifecycle_worker.gpu_memory_threshold == 0.9

        # Clean up
        servicer.lifecycle_worker.stop()

    def test_on_pid_dead_callback(self):
        """Test PID death callback."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Mock replica manager
        servicer.replica_manager = Mock()
        servicer.replica_manager.remove_pid_refs.return_value = []

        # Mock process watcher
        servicer.process_watcher = Mock()

        # Call callback
        servicer._on_pid_dead(1234)

        # Check actions
        servicer.replica_manager.remove_pid_refs.assert_called_once_with(1234)
        servicer.process_watcher.remove_pid.assert_called_once_with(1234)

    def test_load_model_with_pid(self):
        """Test LoadModel RPC with PID tracking."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Mock components
        servicer.process_watcher = Mock()
        servicer.model_loader = Mock()

        # Mock async load result
        future = Future()
        future.set_result((True, None))
        servicer.model_loader.start_async_load.return_value = (True, b"handle", future)

        # Create request with PID
        request = store_daemon_pb2.LoadModelRequest(
            model_path="model1",
            replica_uuid="uuid1",
            device_uuid="gpu0",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            pid=1234,
            keep_for_global=True,
        )
        context = MockContext()

        # Call LoadModel
        response = servicer.LoadModel(request, context)  # pyright: ignore[reportArgumentType]

        # Check PID was tracked
        servicer.process_watcher.add_pid.assert_called_once_with(1234)

        # Check response
        assert (
            response.status
            == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
        )
        assert response.model_path == "model1"

    def test_confirm_model_simple(self):
        """Test ConfirmModel RPC now only waits for async loading."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Create request
        request = store_daemon_pb2.ConfirmModelRequest(
            model_path="model1",
            replica_uuid="uuid1",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        context = MockContext()

        # Test case 1: No pending load - should succeed immediately
        response = servicer.ConfirmModel(request, context)  # pyright: ignore[reportArgumentType]
        assert response.code == 0
        assert response.model_path == "model1"

        # Test case 2: With pending load that succeeds
        future = Future()
        future.set_result((True, None))
        with servicer._pending_loads_lock:
            servicer._pending_loads[("model1", "uuid1")] = future

        response = servicer.ConfirmModel(request, context)  # pyright: ignore[reportArgumentType]
        assert response.code == 0
        assert response.model_path == "model1"

        # Test case 3: With pending load that fails
        future_fail = Future()
        future_fail.set_result((False, "Load failed"))
        with servicer._pending_loads_lock:
            servicer._pending_loads[("model2", "uuid2")] = future_fail

        request.model_path = "model2"
        request.replica_uuid = "uuid2"
        response = servicer.ConfirmModel(request, context)  # pyright: ignore[reportArgumentType]
        assert response.code == 1
        assert response.model_path == "model2"
        assert context.code == grpc.StatusCode.INTERNAL

    def test_unload_model_with_pid(self):
        """Test UnloadModel RPC with PID tracking."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Mock components
        servicer.process_watcher = Mock()
        servicer.replica_manager = Mock()
        servicer.replica_manager.unload_model.return_value = True

        # Create request with PID
        request = store_daemon_pb2.UnloadModelRequest(
            model_path="model1",
            replica_uuid="uuid1",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            pid=1234,
        )
        context = MockContext()

        # Call UnloadModel
        response = servicer.UnloadModel(request, context)  # pyright: ignore[reportArgumentType]

        # Check PID was NOT removed from tracking (intentional design - see comment in UnloadModel)
        servicer.process_watcher.remove_pid.assert_not_called()

        # Check replica manager was called
        servicer.replica_manager.unload_model.assert_called_once_with(
            model_path="model1",
            device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            pid=1234,
        )

        # Check response
        assert response.code == 0
        assert response.model_path == "model1"

    def test_get_loaded_models(self):
        """Test GetLoadedModels RPC."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Mock replica manager
        servicer.replica_manager = Mock()
        servicer.replica_manager.get_loaded_models.return_value = [
            {
                "model_id": "model1",
                "device_id": 0,
                "ref_count": 2,
                "pids": [1234, 5678],
                "size_bytes": 1024 * 1024,
                "keep_for_global": True,
                "last_access_ts": time.time(),
            },
            {
                "model_id": "model2",
                "device_id": 1,
                "ref_count": 1,
                "pids": [9999],
                "size_bytes": 2048 * 1024,
                "keep_for_global": False,
                "last_access_ts": time.time(),
            },
        ]

        # Test without filters
        request = store_daemon_pb2.GetLoadedModelsRequest()
        context = MockContext()

        response = servicer.GetLoadedModels(request, context)  # pyright: ignore[reportArgumentType]

        assert response.total_models == 2
        assert response.total_size_bytes == 3 * 1024 * 1024
        assert len(response.models) == 2

        # Check first model
        model1 = response.models[0]
        assert model1.model_id == "model1"
        assert model1.device_id == 0
        assert model1.ref_count == 2
        assert list(model1.pids) == [1234, 5678]
        assert model1.keep_for_global is True

        # Clean up
        if servicer.lifecycle_worker:
            servicer.lifecycle_worker.stop()

    def test_get_loaded_models_with_filters(self):
        """Test GetLoadedModels RPC with filters."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Mock replica manager
        servicer.replica_manager = Mock()
        servicer.replica_manager.get_loaded_models.return_value = [
            {
                "model_id": "model1",
                "device_id": 0,
                "ref_count": 1,
                "pids": [1234],
                "size_bytes": 1024,
                "keep_for_global": False,
                "last_access_ts": time.time(),
            },
            {
                "model_id": "model2",
                "device_id": 1,
                "ref_count": 1,
                "pids": [5678],
                "size_bytes": 2048,
                "keep_for_global": False,
                "last_access_ts": time.time(),
            },
            {
                "model_id": "other_model",
                "device_id": 0,
                "ref_count": 1,
                "pids": [9999],
                "size_bytes": 4096,
                "keep_for_global": False,
                "last_access_ts": time.time(),
            },
        ]

        # Test with model_id filter - use a more specific filter
        request = store_daemon_pb2.GetLoadedModelsRequest(model_id_filter="model1")
        context = MockContext()

        response = servicer.GetLoadedModels(request, context)  # pyright: ignore[reportArgumentType]

        # Should match only model1
        assert response.total_models == 1
        assert response.models[0].model_id == "model1"

        # Test with device_id filter
        request = store_daemon_pb2.GetLoadedModelsRequest(device_id_filter=0)

        response = servicer.GetLoadedModels(request, context)  # pyright: ignore[reportArgumentType]

        # Should match models on device 0
        assert response.total_models == 2
        assert all(m.device_id == 0 for m in response.models)

        # Clean up
        if servicer.lifecycle_worker:
            servicer.lifecycle_worker.stop()

    def test_graceful_shutdown_with_lifecycle(self):
        """Test graceful shutdown with lifecycle components."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Mock components
        servicer.lifecycle_worker = Mock()
        servicer.process_watcher = Mock()
        servicer.replica_manager = Mock()
        servicer.replica_manager.shutdown_evict_local_replicas.return_value = 3
        servicer.worker_manager = None
        servicer.health_check_server = Mock()
        servicer.model_loader = Mock()

        # Call graceful shutdown
        servicer.graceful_shutdown()

        # Check shutdown sequence
        assert servicer.shutting_down is True
        servicer.lifecycle_worker.stop.assert_called_once()
        servicer.replica_manager.shutdown_evict_local_replicas.assert_called_once()
        servicer.health_check_server.stop.assert_called_once()
        servicer.model_loader.shutdown.assert_called_once()

    def test_load_model_during_shutdown(self):
        """Test LoadModel behavior during shutdown."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path("/tmp/models"),
                mem_pool_size=ByteSize(1024),
                num_threads=10,
                chunk_size=ByteSize(1024),
                enable_p2p_access=False,
            )
        )

        servicer = StoreDaemonServicer(config)

        # Set shutting down
        servicer.shutting_down = True

        # Create request
        request = store_daemon_pb2.LoadModelRequest(
            model_path="model1",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        context = MockContext()

        # Call LoadModel
        response = servicer.LoadModel(request, context)  # pyright: ignore[reportArgumentType]

        # Should fail
        assert response.status == store_daemon_pb2.LOAD_MODEL_STATUS_FAILED
        assert context.code == grpc.StatusCode.UNAVAILABLE
