#  Copyright (c) 2025, StepCast Team.

import uuid

import pytest

import grpc

import contextlib  # Added for suppressing cleanup exceptions

from pathlib import Path
from pydantic import ByteSize

from scstore.proto import store_daemon_pb2, global_store_pb2
from scstore.store_daemon.config import (
    StoreDaemonConfig,
    ServerConfig,
    NetworkConfig,
)
from scstore.store_daemon.servicer import StoreDaemonServicer

from .utils import get_free_port_pair, FakeContext as _MockContext
from tests.python.utils.model_utils import create_dummy_model


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


# Helper accepts gs_addr param.
def _make_servicer(
    *,
    enable_global: bool = False,
    gs_addr: str | None = None,
    grpc_port: int | None = None,
    p2p_port: int | None = None,
) -> StoreDaemonServicer:
    """Create a StoreDaemonServicer for tests.

    Parameters
    ----------
    enable_global:
        Whether to enable the high-availability connection to the Global Store.
    gs_addr:
        If ``enable_global`` is *True*, this must provide the bind address of
        the Global Store gRPC server (e.g. "127.0.0.1:50051").
    grpc_port:
        Optional gRPC port override.
    p2p_port:
        Optional P2P port override.
    """

    if enable_global and not gs_addr:
        raise ValueError("gs_addr must be provided when enable_global=True")

    # Use dynamic ports if not provided
    if grpc_port is None or p2p_port is None:
        grpc_port, p2p_port = get_free_port_pair()

    cfg = StoreDaemonConfig(
        server=ServerConfig(
            storage_path=Path("/tmp/fake"),
            mem_pool_size=ByteSize(1_000_000_000),  # 1GB
            num_threads=4,
            chunk_size=ByteSize(1 << 20),  # 1MB
            enable_p2p_access=enable_global,
            enable_p2p_engine=enable_global,
            port=grpc_port,
        ),
        network=NetworkConfig(
            p2p_port=p2p_port,
        ),
        global_store_address=gs_addr if enable_global else None,
    )

    srv = StoreDaemonServicer(config=cfg)
    srv._grpc_port = grpc_port  # Store for test use
    srv._p2p_port = p2p_port

    # ------------------------------------------------------------------
    # In tests, StoreDaemonServicer spawns the HA connection manager thread
    # asynchronously.  We block briefly here to ensure the stub is ready
    # before returning, avoiding race conditions in ConfirmModel.
    # ------------------------------------------------------------------

    if enable_global:
        import time as _time

        timeout_s = 1.0
        start = _time.time()
        while srv.global_store_stub is None and _time.time() - start < timeout_s:
            _time.sleep(0.01)

        if srv.global_store_stub is None:
            raise RuntimeError(
                "Global Store stub not initialised within timeout during test setup"
            )

    return srv


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------


def test_single_daemon_disk_load():
    """Scenario 1: Disk load inside a single Store Daemon succeeds."""
    servicer = _make_servicer(enable_global=False)
    try:
        # --------------------
        # Ensure on-disk dummy model exists for the requested path
        create_dummy_model(servicer.config.server.storage_path, "alpaca-7b.ckpt")

        ctx = _MockContext()

        request = store_daemon_pb2.LoadModelRequest(
            model_path="alpaca-7b.ckpt",
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            device_uuid=str(uuid.uuid4()),
        )
        response = servicer.LoadModel(request, ctx)

        # Allocation succeeded (asynchronous load).  The status should be ALLOCATED
        assert (
            response.status
            == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
        )
        assert response.mem_handle.cuda_ipc_handle != b""
        assert ctx.code is None

        # Confirm the model to simulate client Acknowledgement
        confirm_req = store_daemon_pb2.ConfirmModelRequest(
            model_path="alpaca-7b.ckpt",
            replica_uuid=request.replica_uuid or str(uuid.uuid4()),
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        conf_resp = servicer.ConfirmModel(confirm_req, ctx)
        assert conf_resp.code == 0
    finally:
        # Clean up daemon
        from tests.python.conftest import cleanup_background_threads

        if servicer.lifecycle_worker:
            servicer.lifecycle_worker.stop()
        if servicer.connection_manager is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001
                servicer.connection_manager.stop()
        cleanup_background_threads(servicer)


def test_remote_load_between_two_daemons(global_store_service):
    """Scenario 2: Daemon A hosts a replica; Daemon B pulls it remotely via GS."""

    daemons_to_cleanup = []
    try:
        # ------------------------------------------------------------------
        # Step 1.  Spin up Daemon A and load a model to GPU (disk source)
        # ------------------------------------------------------------------
        daemon_a = _make_servicer(
            enable_global=True, gs_addr=global_store_service._address
        )
        daemons_to_cleanup.append(daemon_a)
        ctx = _MockContext()

        content_model_id = "mi2:test-index:test-data"  # Simulated content-addressed ID
        disk_path = "mistral-7b.ckpt"

        # Create dummy model file on daemon A's local storage so that disk load succeeds
        # Use 4 MiB to match the registered buffer size for P2P fallback consistency
        create_dummy_model(daemon_a.config.server.storage_path, disk_path, size_bytes=4 * 1024 * 1024)

        device_uuid_a = str(uuid.uuid4())
        replica_uuid_a = str(uuid.uuid4())

        load_req_a = store_daemon_pb2.LoadModelRequest(
            model_path=disk_path,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            device_uuid=device_uuid_a,
            replica_uuid=replica_uuid_a,
            keep_for_global=True,
            size_bytes=4 * 1024 * 1024,  # 4 MB (arbitrary)
        )
        resp_a = daemon_a.LoadModel(load_req_a, ctx)
        assert (
            resp_a.status
            == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
        )

        # Confirm to finish lifecycle and trigger registration path in connection manager
        confirm_req_a = store_daemon_pb2.ConfirmModelRequest(
            model_path=disk_path,
            replica_uuid=replica_uuid_a,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        daemon_a.ConfirmModel(confirm_req_a, ctx)

        # Register a worker and then the replica with the Global Store, reflecting what
        # the connection-manager would usually do.

        worker_req_a = global_store_pb2.RegisterWorkerRequest(
            node_id="daemon_a",
            node_address="127.0.0.1",
            grpc_port=daemon_a._grpc_port,
            p2p_port=daemon_a._p2p_port,
            mem_pool_total_size=16 * 1024 * 1024,
            mem_pool_available_size=16 * 1024 * 1024,
        )
        worker_resp_a = global_store_service.RegisterWorker(worker_req_a, ctx)
        assert worker_resp_a.status == global_store_pb2.Status.OK

        # Export remote memory keys from daemon A's checkpoint store for P2P
        from scstore import _checkpoint_store as _cs

        dev = _cs.DeviceKey()
        dev.type = _cs.DeviceType.GPU
        dev.ordinal = 0
        dev.uuid = ""

        inst_key = _cs.InstanceKey()
        inst_key.model_id = disk_path  # Instance is keyed by disk path for local load
        inst_key.device = dev
        inst_key.replica = 0

        comm_info = daemon_a.checkpoint_store.enable_remote_instance_access(
            inst_key, _cs.ModelLocation.GPU
        )

        mem_info = global_store_pb2.MemoryInfo(
            node_id="daemon_a",
            node_address="127.0.0.1",
            node_port=daemon_a._p2p_port,
            remote_memory_keys=list(comm_info.remote_memory_keys),
            buffer_sizes=list(comm_info.buffer_sizes),
            memory_size=comm_info.model_size,
            memory_type=global_store_pb2.MemoryType.GPU,
            device_id=dev.ordinal,
        )
        reg_req = global_store_pb2.RegisterModelReplicaRequest(
            model_id=content_model_id,
            mem_info=mem_info,
            max_concurrency=2,
            worker_id=worker_resp_a.worker_id,
        )
        global_store_service.RegisterModelReplica(reg_req, ctx)

        # ------------------------------------------------------------------
        # Step 2.  Daemon B attempts remote load of the same model
        # ------------------------------------------------------------------
        daemon_b = _make_servicer(
            enable_global=True, gs_addr=global_store_service._address
        )
        daemons_to_cleanup.append(daemon_b)
        device_uuid_b = str(uuid.uuid4())
        replica_uuid_b = str(uuid.uuid4())

        load_req_b = store_daemon_pb2.LoadModelRequest(
            model_id=content_model_id,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            device_uuid=device_uuid_b,
            replica_uuid=replica_uuid_b,
            size_bytes=4 * 1024 * 1024,
        )
        resp_b = daemon_b.LoadModel(load_req_b, ctx)

        # Should allocate quickly using remote source
        assert (
            resp_b.status
            == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
        )
        assert resp_b.mem_handle.cuda_ipc_handle != b""

        # Ensure there are no lingering in-progress records at this point
        assert global_store_service.transport_repository.count_with_filters("in_progress") == 0

        # Confirming should call CompleteModelReplicaTransport underneath.  We don't
        # test that directly but ensure confirmation passes.
        confirm_req_b = store_daemon_pb2.ConfirmModelRequest(
            model_path=disk_path,
            replica_uuid=replica_uuid_b,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        conf_resp_b = daemon_b.ConfirmModel(confirm_req_b, ctx)
        assert conf_resp_b.code == 0

        # ------------------------------------------------------------------
        # Finally, ensure Global Store transport counter is decremented (<= max)
        # ------------------------------------------------------------------
        model_info = global_store_service.GetModelInfoById(
            global_store_pb2.GetModelInfoByIdRequest(model_id=content_model_id), ctx
        )
        # Replica count may include duplicates due to automatic registration from
        # both daemons.  Ensure *at least one* replica reports the expected size
        # (4 MiB) that we manually registered for Daemon-A.
        expected_size = mem_info.memory_size
        assert any(
            rep.memory_size == expected_size
            for rep in model_info.replicas
        ), f"Expected replica with {expected_size} bytes memory_size not found"

        # All in-flight transports should be completed after confirmation – the
        # FakeGlobalStore keeps an internal dict, which must now be empty.
        # Ensure there are no transports in progress
        assert (
            global_store_service.transport_repository.count_with_filters(
                status="in_progress"
            )
            == 0
        )
        # Completed transports should be recorded by GS after confirmation
        assert global_store_service.transport_repository.count_with_filters("completed") >= 1

    finally:
        # Clean up daemons
        from tests.python.conftest import cleanup_background_threads

        for daemon in daemons_to_cleanup:
            if daemon.lifecycle_worker:
                daemon.lifecycle_worker.stop()
            if daemon.connection_manager is not None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    daemon.connection_manager.stop()
            cleanup_background_threads(daemon)


def test_remote_load_p2p_failure_fallback_to_disk(global_store_service):
    """Scenario: P2P load fails, daemon falls back to disk and succeeds.

    We register a remote replica in the Global Store with a non-existent
    remote-memory key so that the P2P transfer fails.  Daemon-B has the
    model on its local disk, so it must fall back to disk-loading and
    still return ALLOCATED.
    """

    daemons_to_cleanup = []
    try:
        # ------------------------------------------------------------------
        # Step 1.  Spin up Daemon A (source) but do NOT export remote memory
        # ------------------------------------------------------------------
        daemon_a = _make_servicer(enable_global=True, gs_addr=global_store_service._address)
        daemons_to_cleanup.append(daemon_a)
        ctx = _MockContext()

        model_id = "falcon-7b.ckpt"
        disk_path = model_id

        # Register worker A
        worker_req_a = global_store_pb2.RegisterWorkerRequest(
            node_id="daemon_a_bad",
            node_address="127.0.0.1",
            grpc_port=daemon_a._grpc_port,
            p2p_port=daemon_a._p2p_port,
            mem_pool_total_size=16 * 1024 * 1024,
            mem_pool_available_size=16 * 1024 * 1024,
        )
        worker_resp_a = global_store_service.RegisterWorker(worker_req_a, ctx)
        assert worker_resp_a.status == global_store_pb2.Status.OK

        # Register a replica with a non-existent remote key to force P2P failure
        mem_info_bad = global_store_pb2.MemoryInfo(
            node_id="daemon_a_bad",
            node_address="127.0.0.1",
            node_port=daemon_a._p2p_port,
            remote_memory_keys=["missing_key"],
            buffer_sizes=[4 * 1024 * 1024],
            memory_size=4 * 1024 * 1024,
            memory_type=global_store_pb2.MemoryType.GPU,
            device_id=0,
        )
        reg_req = global_store_pb2.RegisterModelReplicaRequest(
            model_id=model_id,
            mem_info=mem_info_bad,
            max_concurrency=1,
            worker_id=worker_resp_a.worker_id,
        )
        rep_resp = global_store_service.RegisterModelReplica(reg_req, ctx)
        assert rep_resp.status == global_store_pb2.Status.OK

        # Force the Global Store to time out transport requests so the daemon
        # immediately falls back to disk without attempting a P2P allocation.
        from scstore.global_store.exceptions import TimeoutError as _GSTimeout
        _orig_request_transport = global_store_service.service.transport_service.request_transport
        def _always_timeout(*args, **kwargs):
            raise _GSTimeout("forced timeout for test")
        global_store_service.service.transport_service.request_transport = _always_timeout

        try:
            # ------------------------------------------------------------------
            # Step 2.  Daemon B attempts remote load of the same model
            #          P2P will fail fast (timeout); ensure disk fallback succeeds
            # ------------------------------------------------------------------
            daemon_b = _make_servicer(enable_global=True, gs_addr=global_store_service._address)
            daemons_to_cleanup.append(daemon_b)
            ctx_b = _MockContext()

            # Prepare local disk so fallback path can succeed
            create_dummy_model(daemon_b.config.server.storage_path, disk_path)

            load_req_b = store_daemon_pb2.LoadModelRequest(
                model_path=disk_path,
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                device_uuid=str(uuid.uuid4()),
                replica_uuid=str(uuid.uuid4()),
                size_bytes=1 * 1024 * 1024,
            )
            resp_b = daemon_b.LoadModel(load_req_b, ctx_b)

            # Allocation should succeed via disk fallback, returning ALLOCATED
            assert (
                resp_b.status
                == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
            )
            assert resp_b.mem_handle.cuda_ipc_handle != b""

            # Ensure no P2P transport record was created due to forced timeout
            assert global_store_service.transport_repository.count_with_filters("in_progress") == 0
            assert global_store_service.transport_repository.count_with_filters("completed") == 0

            # Confirm the model to complete the lifecycle
            confirm_req_b = store_daemon_pb2.ConfirmModelRequest(
                model_path=disk_path,
                replica_uuid=load_req_b.replica_uuid,
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            )
            conf_resp_b = daemon_b.ConfirmModel(confirm_req_b, ctx_b)
            assert conf_resp_b.code == 0
        finally:
            # Restore transport service behaviour
            global_store_service.service.transport_service.request_transport = _orig_request_transport

        # Confirm the model to complete the lifecycle
        confirm_req_b = store_daemon_pb2.ConfirmModelRequest(
            model_path=disk_path,
            replica_uuid=load_req_b.replica_uuid,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        conf_resp_b = daemon_b.ConfirmModel(confirm_req_b, ctx_b)
        assert conf_resp_b.code == 0

    finally:
        # Clean up daemons
        from tests.python.conftest import cleanup_background_threads

        for daemon in daemons_to_cleanup:
            if daemon.lifecycle_worker:
                daemon.lifecycle_worker.stop()
            if daemon.connection_manager is not None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    daemon.connection_manager.stop()
            cleanup_background_threads(daemon)
