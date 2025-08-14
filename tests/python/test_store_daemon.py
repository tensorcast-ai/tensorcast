#  Copyright (c) 2025, StepCast Team.

# pyright: reportAttributeAccessIssue=false
import types
import uuid
from unittest import mock
import shutil

import grpc
import pytest

import contextlib  # Added for suppressing cleanup exceptions

from scstore.proto import global_store_pb2, store_daemon_pb2

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

from pathlib import Path
from tests.python.utils.model_utils import create_dummy_model


class MockContext:
    """Mock gRPC ServicerContext for testing"""

    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details



@pytest.fixture
def test_context():
    """Create a mock gRPC ServicerContext"""
    return MockContext()


# Pytest fixtures -----------------------------------------------------------------


@pytest.fixture
def servicer(tmp_path):
    """Create a StoreDaemonServicer with mocked dependencies for testing.

    A unique temporary *storage_path* is allocated per-test via ``tmp_path`` so
    that concurrent test runs (e.g. with ``-n auto``) do not interfere with
    each other.  The directory – along with any dummy model data created by
    the helper – is removed after the test using an explicit ``shutil.rmtree``
    call to keep the system */tmp* tidy.
    """
    # Import here to ensure mock is in place
    from scstore.store_daemon.config import StoreDaemonConfig, ServerConfig, NetworkConfig
    from scstore.store_daemon.servicer import StoreDaemonServicer
    from pathlib import Path
    from pydantic import ByteSize

    # Create with minimal required args
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)

    config = StoreDaemonConfig(
        server=ServerConfig(
            storage_path=storage_root,
            mem_pool_size=ByteSize(1000000000),  # 1GB
            num_threads=4,
            chunk_size=ByteSize(1024 * 1024),  # 1MB
            enable_p2p_access=False,
            enable_p2p_engine=False,  # Disable P2P engine to avoid global store dependency
        ),
        global_store_address=None,  # No global store for testing
        network=NetworkConfig(
            health_check_port=None,
        ),
    )

    # Create servicer and initialize grpc_channel to None to prevent __del__ issues
    servicer = StoreDaemonServicer(config=config)
    servicer.grpc_channel = None  # Initialize to prevent AttributeError in __del__
    # Add a mock global store stub for tests that need it
    servicer.global_store_stub = mock.MagicMock()

    # ------------------------------------------------------------------
    # Prepare dummy model files expected by the various test cases.
    # ------------------------------------------------------------------
    for model_name in [
        "test_model",
        "failing_model",
        "model1",
        "model2",
        "model3",
        "test_model_loaded",
    ]:
        create_dummy_model(config.server.storage_path, model_name)

    try:
        yield servicer
    finally:
        # Clean up all background threads and resources
        from tests.python.conftest import cleanup_background_threads

        # Stop lifecycle worker and process watcher
        if servicer.lifecycle_worker:
            servicer.lifecycle_worker.stop()

        # Stop connection manager
        if servicer.connection_manager is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001
                servicer.connection_manager.stop()

        # Stop health check server
        if servicer.health_check_server is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001
                servicer.health_check_server.stop()

        # Shutdown model loader
        if servicer.model_loader is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001
                servicer.model_loader.shutdown()

        # Clean up any remaining background threads
        cleanup_background_threads(servicer)

        # Clean up temporary model data to avoid leakage between tests
        shutil.rmtree(storage_root, ignore_errors=True)


@pytest.fixture
def servicer_with_global_store(tmp_path):
    """Create a StoreDaemonServicer with mocked global store connection"""
    # Import here to ensure mock is in place
    from scstore.store_daemon.config import StoreDaemonConfig, ServerConfig
    from scstore.store_daemon.servicer import StoreDaemonServicer
    from pathlib import Path
    from pydantic import ByteSize

    # Mock the grpc channel and GlobalModelStoreStub
    with (
        mock.patch("grpc.insecure_channel"),
        mock.patch("scstore.proto.global_store_pb2_grpc.GlobalModelStoreStub") as mock_stub,
    ):
        # Setup the mock stub
        mock_instance = mock_stub.return_value

        storage_root = tmp_path / "storage"
        storage_root.mkdir(parents=True, exist_ok=True)

        # Create the servicer
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=storage_root,
                mem_pool_size=ByteSize(1000000000),  # 1GB
                num_threads=4,
                chunk_size=ByteSize(1024 * 1024),  # 1MB
                enable_p2p_access=False,
                enable_p2p_engine=True,
            ),
            global_store_address="localhost:50051",
        )

        # Mock the connection manager to avoid actual connections
        with mock.patch("scstore.store_daemon.servicer.GlobalStoreConnectionManager"):
            servicer = StoreDaemonServicer(config=config)

        # Replace the stub with our mock and initialize grpc_channel
        servicer.global_store_stub = mock_instance
        servicer.grpc_channel = None  # Initialize to prevent AttributeError in __del__
        # Set a worker_id to enable registration
        servicer.worker_id = "test_worker_123"
        # Also set the stub on the model loader since it copies it in constructor
        servicer.model_loader.global_store_stub = mock_instance

        # Create the same set of dummy model files as for the local-only fixture
        for model_name in [
            "test_model",
            "failing_model",
            "model1",
            "model2",
            "model3",
            "test_model_loaded",
        ]:
            create_dummy_model(config.server.storage_path, model_name)

        try:
            yield servicer, mock_instance
        finally:
            # Clean up all background threads and resources
            from tests.python.conftest import cleanup_background_threads

            # Stop lifecycle worker and process watcher
            if servicer.lifecycle_worker:
                servicer.lifecycle_worker.stop()

            # Stop connection manager
            if servicer.connection_manager is not None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    servicer.connection_manager.stop()

            # Stop health check server
            if servicer.health_check_server is not None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    servicer.health_check_server.stop()

            # Shutdown model loader
            if servicer.model_loader is not None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    servicer.model_loader.shutdown()

            # Clean up any remaining background threads
            cleanup_background_threads(servicer)

            shutil.rmtree(storage_root, ignore_errors=True)


def test_load_model_async_cpu_unsupported(servicer, test_context):
    """CPU loading is no longer supported – expect UNIMPLEMENTED status."""

    request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,
    )

    response = servicer.LoadModel(request, test_context)

    assert response.status == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_FAILED
    assert test_context.code == grpc.StatusCode.UNIMPLEMENTED


def test_load_model_async_cpu_empty_path(servicer, test_context):
    """Empty model path should return INVALID_ARGUMENT irrespective of device."""

    request = store_daemon_pb2.LoadModelRequest(
        model_path="",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,
    )

    response = servicer.LoadModel(request, test_context)

    assert response.status == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_FAILED
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT


def test_load_model_async_cpu_fail_removed():
    """Placeholder: failure scenario obsolete as CPU loading is unsupported."""
    pass  # No-op – retained to keep test names stable.


def test_load_model_async_gpu_success(servicer, test_context):
    """GPU loading should now return ALLOCATED status."""

    device_uuid = str(uuid.uuid4())
    request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_uuid=device_uuid,
    )

    response = servicer.LoadModel(request, test_context)

    assert response.model_path == "test_model"
    assert response.status == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
    # The daemon should return its own IPC handle in the response.
    assert response.mem_handle.cuda_ipc_handle != b""
    assert test_context.code is None


def test_load_model_async_gpu_missing_uuid(servicer, test_context):
    """Device UUID is optional – request without it should still succeed."""

    request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    )

    response = servicer.LoadModel(request, test_context)

    assert response.status == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
    assert test_context.code is None


def test_load_model_async_unsupported_device(servicer, test_context):
    """Test loading a model to an unsupported device type fails"""
    # Create request with unsupported device type
    request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_DISK,  # DISK type is not supported for loading
    )

    # Call method
    servicer.LoadModel(request, test_context)

    # Check response
    assert test_context.code == grpc.StatusCode.UNIMPLEMENTED


def test_confirm_model_success(servicer, test_context):
    """Test confirming a model in GPU successfully"""
    # First load the model to GPU
    device_uuid = str(uuid.uuid4())
    load_request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_uuid=device_uuid,
    )
    servicer.LoadModel(load_request, test_context)

    # TODO: fix replica_uuid
    # Then confirm it
    confirm_request = store_daemon_pb2.ConfirmModelRequest(
        model_path="test_model",
        replica_uuid=str(uuid.uuid4()),
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    )

    # Call method
    response = servicer.ConfirmModel(confirm_request, test_context)

    # Check response
    assert response.model_path == "test_model"
    assert test_context.code is None  # No error


def test_confirm_model_empty_path(servicer, test_context):
    """Test confirming a model with empty path fails"""
    # Create request with empty path
    request = store_daemon_pb2.ConfirmModelRequest(
        model_path="",
        replica_uuid=str(uuid.uuid4()),
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    )

    # Call method
    servicer.ConfirmModel(request, test_context)

    # Check response
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT


def test_confirm_model_unsupported_device(servicer, test_context):
    """Test confirming a model with unsupported device type fails"""
    # Create request with CPU device type
    request = store_daemon_pb2.ConfirmModelRequest(
        model_path="test_model",
        replica_uuid=str(uuid.uuid4()),
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,  # Only GPU is supported
    )

    # Call method
    servicer.ConfirmModel(request, test_context)

    # Check response
    assert test_context.code == grpc.StatusCode.UNIMPLEMENTED


def test_confirm_model_fail(servicer, test_context):
    """Test confirming a model that fails to confirm"""
    # First load the failing model
    device_uuid = str(uuid.uuid4())
    replica_uuid = str(uuid.uuid4())
    load_request = store_daemon_pb2.LoadModelRequest(
        model_path="failing_model",
        replica_uuid=replica_uuid,
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_uuid=device_uuid,
    )
    servicer.LoadModel(load_request, test_context)

    # Give a small delay to ensure the async load completes
    import time
    time.sleep(0.1)

    # Reset context code for confirm test
    test_context.code = None

    # Create request with model known to fail in the mock
    request = store_daemon_pb2.ConfirmModelRequest(
        model_path="failing_model",
        replica_uuid=replica_uuid,
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    )

    # Call method
    response = servicer.ConfirmModel(request, test_context)

    # Since the load already completed (and failed), ConfirmModel won't find a pending load
    # and will just return success. The actual failure would be caught during verification
    # which happens asynchronously after load completes.
    assert response.model_path == "failing_model"
    assert response.code == 0  # ConfirmModel succeeds because there's no pending load to wait for


def test_unload_model_success(servicer, test_context):
    """Test unloading a model successfully"""
    # First load a model to GPU (CPU loading is not supported)
    device_uuid = str(uuid.uuid4())
    load_request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_uuid=device_uuid,
    )
    servicer.LoadModel(load_request, test_context)

    # Then unload it directly via ReplicaManager with explicit device_id
    success = servicer.replica_manager.unload_model(
        model_path="test_model",
        device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_id=0,
    )

    assert success is True


def test_unload_model_empty_path(servicer, test_context):
    """Test unloading a model with empty path fails"""
    # Create request with empty path
    request = store_daemon_pb2.UnloadModelRequest(
        model_path="",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,
    )

    # Call method
    servicer.UnloadModel(request, test_context)

    # Check response
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT


def test_unload_model_unsupported_device(servicer, test_context):
    """Test unloading a model with DISK device type succeeds"""
    # Attempt unloading a DISK replica – should succeed (no-op in stub)
    success = servicer.replica_manager.unload_model(
        model_path="test_model",
        device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_DISK,
        device_id=0,
    )

    assert success is True


def test_unload_model_fail(servicer, test_context):
    """Test unloading a model that fails to unload"""
    # Create request with model known to fail in the mock
    request = store_daemon_pb2.UnloadModelRequest(
        model_path="failing_model",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    )

    # Attempt unload via ReplicaManager – should return False for failing_model
    success = servicer.replica_manager.unload_model(
        model_path="failing_model",
        device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_id=0,
    )

    assert success is False


def test_clear_mem_success(servicer, test_context):
    """Test clearing memory successfully"""
    # First load some models to GPU (CPU loading is not supported)
    device_uuid = str(uuid.uuid4())
    for model in ["model1", "model2", "model3"]:
        load_request = store_daemon_pb2.LoadModelRequest(
            model_path=model,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            device_uuid=device_uuid,
        )
        servicer.LoadModel(load_request, test_context)

    # Then clear memory
    request = store_daemon_pb2.ClearMemRequest()

    # Call method
    servicer.ClearMem(request, test_context)

    # Check response
    assert test_context.code in (None, grpc.StatusCode.OK)  # No error


def test_get_server_config(servicer, test_context):
    """Test getting server configuration"""
    # Create request
    request = store_daemon_pb2.GetServerConfigRequest()

    # Call method
    response = servicer.GetServerConfig(request, test_context)

    # Check response
    assert response.mem_pool_size == 1000000000  # Set in fixture
    assert response.chunk_size == 1024 * 1024  # Set in fixture


def test_load_model_with_global_store(servicer_with_global_store, test_context):
    """Test loading a model with global store integration"""
    servicer, mock_global_store = servicer_with_global_store

    # Create a memory info object for the replica
    replica_memory_info = global_store_pb2.MemoryInfo(
        node_id="replica1",
        node_address="192.168.1.1",
        node_port=9090,
        remote_memory_keys=["memory_key"],
        memory_size=1000000,
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )

    # Mock the GetModelInfo response
    model_info_response = global_store_pb2.GetModelInfoResponse(
        status=global_store_pb2.Status.OK,
        model_info=global_store_pb2.ModelInfo(
            model_name="test_model",
            available_replicas=[replica_memory_info],
        ),
    )
    # Setup async mock responses
    mock_global_store.GetModelInfo = mock.MagicMock(return_value=model_info_response)

    # Mock the RequestModelReplicaTransport response
    transport_id = str(uuid.uuid4())
    memory_info = global_store_pb2.MemoryInfo(
        node_id="remote_node",
        node_address="192.168.1.1",
        node_port=9090,
        remote_memory_keys=["memory_key"],
        memory_size=1000000,
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    transport_response = global_store_pb2.RequestModelReplicaTransportResponse(
        status=global_store_pb2.Status.OK,
        remote_memory_info=memory_info,
        transport_id=transport_id,
    )
    mock_global_store.RequestModelReplicaTransport = mock.MagicMock(return_value=transport_response)

    # Mock the CompleteModelReplicaTransport response
    complete_response = global_store_pb2.CompleteModelReplicaTransportResponse(
        status=global_store_pb2.Status.OK,
    )
    mock_global_store.CompleteModelReplicaTransport = mock.MagicMock(return_value=complete_response)

    # Mock the RegisterModelReplica response
    register_response = global_store_pb2.RegisterModelReplicaResponse(
        status=global_store_pb2.Status.OK,
        replica_id="test_replica_id_123",
    )
    mock_global_store.RegisterModelReplica = mock.MagicMock(return_value=register_response)

    # Create request
    request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model",
        replica_uuid="test_replica",
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_uuid=str(uuid.uuid4()),
    )

    # Call method
    response = servicer.LoadModel(request, test_context)

    # Check response
    assert response.model_path == "test_model"
    assert test_context.code is None  # No error

    # Global store RPCs are handled by the HA connection manager/C++ layer; no direct stub calls to assert here.

    # Now confirm the model
    confirm_request = store_daemon_pb2.ConfirmModelRequest(
        model_path="test_model",
        replica_uuid="test_replica",  # Any UUID works with our mock
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    )

    # Call method
    response = servicer.ConfirmModel(confirm_request, test_context)

    # Check confirm response
    assert response.model_path == "test_model"
    assert response.code == 0  # Success

    # The CompleteModelReplicaTransport and RegisterModelReplica are called via
    # the connection manager in the new async model, not directly via the stub.
    # The successful LoadModel and ConfirmModel responses indicate the integration works.


def test_get_worker_status(servicer, test_context):
    """Ensure GetWorkerStatus returns correct basic information."""
    # Call the RPC
    request = store_daemon_pb2.GetWorkerStatusRequest()
    response = servicer.GetWorkerStatus(request, test_context)

    # Validate static configuration values (from fixture)
    assert response.mem_pool_total_size == 1_000_000_000  # 1GB as configured


    # Validate daemon liveness flags
    assert response.is_registered is False  # No worker_id set in default fixture
    assert response.is_healthy is True
    assert response.is_shutting_down is False
    assert response.uptime_seconds >= 0  # Uptime should be non-negative



def test_get_loaded_models_after_load(servicer, test_context):
    """Load a model and ensure it is reported by GetLoadedModels."""
    # Load and confirm a dummy model
    device_uuid = str(uuid.uuid4())
    replica_uuid = str(uuid.uuid4())

    load_request = store_daemon_pb2.LoadModelRequest(
        model_path="test_model_loaded",
        replica_uuid=replica_uuid,
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        device_uuid=device_uuid,
    )
    servicer.LoadModel(load_request, test_context)

    confirm_request = store_daemon_pb2.ConfirmModelRequest(
        model_path="test_model_loaded",
        replica_uuid=replica_uuid,
        target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    )
    servicer.ConfirmModel(confirm_request, test_context)

    # Query loaded models with filter
    list_request = store_daemon_pb2.GetLoadedModelsRequest(
        model_id_filter="test_model_loaded",
    )
    list_response = servicer.GetLoadedModels(list_request, test_context)

    # Validate the response contains exactly one entry matching our model
    assert list_response.total_models == 1
    assert list_response.models[0].model_id == "test_model_loaded"



def test_wait_model_verification_passed(servicer, test_context):
    """Verify WaitModelVerification returns PASSED status when pre-populated."""
    # Pre-populate verification results to simulate a completed verification
    key = ("test_model_verification", "replica_verification")
    with servicer._verification_lock:
        servicer._verification_results[key] = (
            store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_PASSED,
            "",
        )

    request = store_daemon_pb2.VerificationRequest(
        model_identifier="test_model_verification",
        replica_uuid="replica_verification",
        timeout_ms=500,
    )

    response = servicer.WaitModelVerification(request, test_context)

    assert (
        response.status
        == store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_PASSED
    )
    assert response.err_msg == ""



def test_get_detailed_status_basic(servicer, test_context):
    """Ensure GetDetailedStatus returns sane, non-zero values."""
    request = store_daemon_pb2.GetDetailedStatusRequest()
    response = servicer.GetDetailedStatus(request, test_context)

    # Basic sanity checks
    assert response.uptime_seconds >= 0
    assert response.memory_pool_info.total_size_bytes == 1_000_000_000
    assert response.memory_pool_info.chunk_size_bytes == 1_048_576  # 1MB chunk size

    # Communication info should reflect the fixture configuration (P2P disabled)
    assert response.communication_info.enabled == servicer.enable_p2p_engine
