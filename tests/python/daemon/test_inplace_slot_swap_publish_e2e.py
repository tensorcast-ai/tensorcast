#  Copyright (c) 2026, TensorCast Team.

"""End-to-end swap+publish coverage for binding-based materialization."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

import grpc
import pytest
import torch
import yaml

from tensorcast.api import GetArtifactOptions
from tensorcast.api.store import Store
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.view_composer import compute_index_multihash
from tensorcast.cli_utils.proc import build_daemon_process_env, ensure_cpp_daemon_binary
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2, store_daemon_pb2_grpc
from tensorcast.proto.global_store.v1 import global_store_pb2
from tests.python.utils.hardware import synchronize_cuda

pytestmark = [pytest.mark.requires_cuda_or_fake, pytest.mark.integration]

_CAPABILITY_SECRET = "c2VjcmV0"  # base64("secret")


def _skip_if_no_cuda() -> None:
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        pytest.skip("fake CUDA backend does not support IPC swap/publish")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - swap/publish requires real GPU")


def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(addr: str, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            response = stub.GetServerConfig(
                store_daemon_pb2.GetServerConfigRequest(), timeout=1.0
            )
            channel.close()
            if response.startup_phase == store_daemon_pb2.DAEMON_STARTUP_PHASE_READY:
                return
            if (
                response.startup_phase
                == store_daemon_pb2.DAEMON_STARTUP_PHASE_UNSPECIFIED
            ):
                return
            if response.startup_phase == store_daemon_pb2.DAEMON_STARTUP_PHASE_FAILED:
                raise RuntimeError("daemon startup failed")
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("daemon failed to start")


def _wait_artifact_index_ready(
    addr: str,
    artifact_id: str,
    timeout_s: float = 180.0,
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        channel = grpc.insecure_channel(addr)
        stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
        try:
            response = stub.GetArtifactIndexById(
                store_daemon_pb2.GetArtifactIndexByIdRequest(artifact_id=artifact_id),
                timeout=1.0,
            )
            if response.tensor_index_data:
                channel.close()
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        finally:
            channel.close()
        time.sleep(0.2)
    raise RuntimeError(
        f"daemon data-plane failed to become ready for artifact_id={artifact_id}: {last_error}"
    )


def _wait_for_worker(gs_port: int, listen_port: int, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    stub = GlobalStoreCompositeStub(channel)
    try:
        while time.time() < deadline:
            resp = stub.ListActiveWorkers(
                global_store_pb2.ListActiveWorkersRequest(include_unavailable=True)
            )
            if any(worker.grpc_port == listen_port for worker in resp.workers):
                return
            time.sleep(0.25)
    finally:
        channel.close()
    raise RuntimeError("daemon did not register with Global Store")


def _build_daemon_config(
    *,
    listen_port: int,
    p2p_port: int,
    storage_dir: Path,
    gs_port: int,
    daemon_id: str,
) -> dict:
    return {
        "server": {
            "listen": {"host": "0.0.0.0", "port": listen_port},
            "p2p_listen": {"host": "0.0.0.0", "port": p2p_port},
            "storage_path": str(storage_dir),
            "num_threads": 2,
            "grpc": {"tcp_nodelay": True, "so_reuseport": False},
        },
        "daemon_id": daemon_id,
        "engine": {
            "artifact_chunk_bytes": 1 * 1024 * 1024,
            "streaming_buffer_chunks": 4,
        },
        "pinned_memory": {
            "allocation_timeout": "30s",
            "classes": [
                {
                    "name": "engine",
                    "slice_bytes": 1 * 1024 * 1024,
                    "pool_bytes": 64 * 1024 * 1024,
                },
                {
                    "name": "comm_gpu",
                    "slice_bytes": 1 * 1024 * 1024,
                    "pool_bytes": 16 * 1024 * 1024,
                },
                {
                    "name": "comm_cpu",
                    "slice_bytes": 1 * 1024 * 1024,
                    "pool_bytes": 8 * 1024 * 1024,
                },
            ],
        },
        "high_availability": {
            "enabled": True,
            "global_store_endpoints": [{"host": "127.0.0.1", "port": gs_port}],
        },
        "communicator": {
            "enable_rdma": False,
            "stager": {"buffers_per_flow": 1},
            "transport": {"tcp_conn_count": 2},
        },
        "observability": {
            "otel": {"enabled": False},
            "logging": {"level": "INFO"},
            "tracing": {"chrome_trace_dir": ""},
        },
        "debug": {"cuda": {"enable_same_process_ipc_fallback": True}},
        "capability_tokens": {
            "active": {"version": 1, "secret": _CAPABILITY_SECRET},
            "previous": [],
        },
        "capability_directory": {"enabled": True},
    }


def _make_index_bytes() -> bytes:
    index = {
        "alpha": [0, 16, [4], [1], "torch.float32", 0],
        "beta": [16, 16, [4], [1], "torch.float32", 4],
    }
    return json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _pack_floats(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _write_artifact_dir(
    artifact_dir: Path, index_bytes: bytes, data_bytes: bytes
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    index_path = artifact_dir / "tensor_index.json"
    index_path.write_bytes(index_bytes)
    data_path = artifact_dir / "tensor.data_0"
    data_path.write_bytes(data_bytes)


def _seed_global_store(
    servicer: GlobalStoreServicer, *, artifact_id: str, index_bytes: bytes
) -> None:
    index_mh = compute_index_multihash(index_bytes)
    _ = servicer.artifact_indices.upsert_index(
        index_data=index_bytes,
        encoding="json",
        schema_version="v3",
    )
    servicer.artifacts_repo.upsert_artifact(
        artifact_id=artifact_id,
        index_multihash=index_mh,
        data_multihash=None,
        schema_version="v3",
        encoding="json",
        id_kind="CGID",
    )
    servicer.connection.commit()


def _wait_replica_state(
    servicer: GlobalStoreServicer,
    replica_id: str,
    *,
    expected_available: bool,
    timeout_s: float = 15.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        replica = servicer.replica_repository.find_by_replica_id(UUID(replica_id))
        if replica is not None and replica.is_available == expected_available:
            return
        time.sleep(0.2)
    raise AssertionError(
        f"replica {replica_id} availability did not become {expected_available}"
    )


def _wait_replica_retired(
    servicer: GlobalStoreServicer,
    replica_id: str,
    *,
    timeout_s: float = 15.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        replica = servicer.replica_repository.find_by_replica_id(UUID(replica_id))
        if replica is None:
            return
        if not replica.is_available:
            return
        time.sleep(0.2)
    raise AssertionError(f"replica {replica_id} was not retired")


def _request_replica_transport(
    *,
    gs_port: int,
    artifact_id: str,
    requester_daemon_id: str,
    requester_p2p_port: int,
) -> global_store_pb2.RequestReplicaTransportResponse:
    channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    stub = GlobalStoreCompositeStub(channel)
    try:
        request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            source_node_id=requester_daemon_id,
            source_address="127.0.0.1",
            source_port=requester_p2p_port,
            request_id=f"pytest-{uuid4().hex}",
        )
        request.requested_byte_space.kind = common_pb2.BYTE_SPACE_KIND_CANONICAL
        response = stub.RequestReplicaTransport(request)
        if response.status == global_store_pb2.Status.STATUS_OK:
            complete = stub.CompleteReplicaTransport(
                global_store_pb2.CompleteReplicaTransportRequest(
                    transport_id=response.transport_id,
                    outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
                    outcome_detail="pytest probe",
                )
            )
            assert complete.status == global_store_pb2.Status.STATUS_OK
        return response
    finally:
        channel.close()


def _make_store(daemon_addr: str) -> Store:
    return Store(
        daemon_addr,
        runtime=StoreRuntimeContext(daemon_addr, client_factory=DaemonCtl),
    )


def _wait_json_file(path: Path, *, timeout_s: float = 30.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {path}: {last_error}")


def _start_published_binding_child(
    *,
    daemon_addr: str,
    artifact_id: str,
    ready_path: Path,
    log_fd: object,
    env: dict[str, str],
) -> subprocess.Popen:
    child_script = textwrap.dedent(
        """
        import json
        import os
        import sys
        import time

        from tensorcast.api import GetArtifactOptions
        from tensorcast.api.store import Store
        from tensorcast.api.store.runtime import StoreRuntimeContext
        from tensorcast.daemon_ctl import DaemonCtl
        from tests.python.utils.hardware import synchronize_cuda

        daemon_addr, artifact_id, ready_path = sys.argv[1:4]
        store = Store(
            daemon_addr,
            runtime=StoreRuntimeContext(daemon_addr, client_factory=DaemonCtl),
        )
        binding = None
        try:
            artifact = store.artifact(artifact_id=artifact_id)
            binding = artifact.bind(
                device="cuda:0",
                packing="byte_space",
                options=GetArtifactOptions(
                    source="disk_only",
                    verify_checksums=False,
                ),
            )
            synchronize_cuda()
            binding.publish_replica()
            payload = {
                "pid": os.getpid(),
                "lease_id": binding._slot.published_lease_id,
                "replica_id": binding._slot.published_replica_id,
            }
            tmp_path = ready_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as out:
                json.dump(payload, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_path, ready_path)
            while True:
                time.sleep(1.0)
        finally:
            if binding is not None:
                binding.close()
            store.close()
        """
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_script,
            daemon_addr,
            artifact_id,
            str(ready_path),
        ],
        stdout=log_fd,
        stderr=log_fd,
        env=env,
    )


@pytest.fixture(scope="module")
def gs_server() -> Iterator[tuple[grpc.Server, int, GlobalStoreServicer]]:
    set_config(GlobalStoreConfig())
    servicer = GlobalStoreServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    register_global_store_servicers(server, servicer)
    port = server.add_insecure_port("127.0.0.1:0")
    if port <= 0:
        raise RuntimeError("failed to bind Global Store server port")
    server.start()
    try:
        yield (server, port, servicer)
    finally:
        server.stop(grace=None)


@pytest.mark.integration
def test_inplace_slot_swap_publish_e2e(
    gs_server: tuple[grpc.Server, int, GlobalStoreServicer],
    tmp_path: Path,
) -> None:
    if os.environ.get("TENSORCAST_RUN_CPP_DAEMON_IT", "1") != "1":
        pytest.skip("Set TENSORCAST_RUN_CPP_DAEMON_IT=1 to run daemon integration test")
    _skip_if_no_cuda()

    bin_path: Path = ensure_cpp_daemon_binary()

    _, gs_port, servicer = gs_server
    index_bytes = _make_index_bytes()
    artifact_a_id = "cgid:artifact_a"
    artifact_b_id = "cgid:artifact_b"
    _seed_global_store(servicer, artifact_id=artifact_a_id, index_bytes=index_bytes)
    _seed_global_store(servicer, artifact_id=artifact_b_id, index_bytes=index_bytes)
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    artifact_a_dir = storage_dir / artifact_a_id
    artifact_b_dir = storage_dir / artifact_b_id

    data_a = _pack_floats([1.0, 2.0, 3.0, 4.0]) + _pack_floats([5.0, 6.0, 7.0, 8.0])
    data_b = _pack_floats([9.0, 10.0, 11.0, 12.0]) + _pack_floats(
        [13.0, 14.0, 15.0, 16.0]
    )
    _write_artifact_dir(artifact_a_dir, index_bytes, data_a)
    _write_artifact_dir(artifact_b_dir, index_bytes, data_b)
    channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    stub = GlobalStoreCompositeStub(channel)
    try:
        info = stub.GetServerInfo(global_store_pb2.GetServerInfoRequest())
        cluster_id = info.cluster_id
        for artifact_id in (artifact_a_id, artifact_b_id):
            resp = stub.UpsertArtifactDiskLocation(
                global_store_pb2.UpsertArtifactDiskLocationRequest(
                    artifact_id=artifact_id,
                    cluster_id=cluster_id,
                    relative_path=artifact_id,
                    kind=global_store_pb2.DISK_LOCATION_KIND_MANAGED,
                )
            )
            if resp.status != global_store_pb2.Status.STATUS_OK:
                raise RuntimeError("failed to upsert artifact disk location")
    finally:
        channel.close()

    producer_listen_port = _get_free_port()
    producer_p2p_port = _get_free_port()
    producer_daemon_id = f"daemon_slot_{producer_listen_port}"
    producer_cfg = _build_daemon_config(
        listen_port=producer_listen_port,
        p2p_port=producer_p2p_port,
        storage_dir=storage_dir,
        gs_port=gs_port,
        daemon_id=producer_daemon_id,
    )
    consumer_listen_port = _get_free_port()
    consumer_p2p_port = _get_free_port()
    consumer_daemon_id = f"daemon_slot_consumer_{consumer_listen_port}"
    consumer_cfg = _build_daemon_config(
        listen_port=consumer_listen_port,
        p2p_port=consumer_p2p_port,
        storage_dir=storage_dir,
        gs_port=gs_port,
        daemon_id=consumer_daemon_id,
    )

    with tempfile.NamedTemporaryFile(
        prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False
    ) as producer_cfg_file:
        yaml.safe_dump(producer_cfg, producer_cfg_file, sort_keys=False)
        producer_cfg_path = Path(producer_cfg_file.name)
    with tempfile.NamedTemporaryFile(
        prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False
    ) as consumer_cfg_file:
        yaml.safe_dump(consumer_cfg, consumer_cfg_file, sort_keys=False)
        consumer_cfg_path = Path(consumer_cfg_file.name)

    proc_log = storage_dir / "daemon_proc.log"
    env = build_daemon_process_env(os.environ)
    binding = None
    store = None
    consumer_store = None
    kill_owner_proc = None
    with proc_log.open("a") as log_fd:
        producer_proc = subprocess.Popen(
            [str(bin_path), f"--config={producer_cfg_path}"],
            stdout=log_fd,
            stderr=log_fd,
            env=env,
        )
        consumer_proc = subprocess.Popen(
            [str(bin_path), f"--config={consumer_cfg_path}"],
            stdout=log_fd,
            stderr=log_fd,
            env=env,
        )
        try:
            producer_addr = f"127.0.0.1:{producer_listen_port}"
            consumer_addr = f"127.0.0.1:{consumer_listen_port}"
            _wait_ready(producer_addr)
            _wait_ready(consumer_addr)
            _wait_for_worker(gs_port, producer_listen_port)
            _wait_for_worker(gs_port, consumer_listen_port)
            _wait_artifact_index_ready(producer_addr, artifact_a_id)
            _wait_artifact_index_ready(producer_addr, artifact_b_id)
            _wait_artifact_index_ready(consumer_addr, artifact_a_id)
            _wait_artifact_index_ready(consumer_addr, artifact_b_id)

            store = _make_store(producer_addr)
            consumer_store = _make_store(consumer_addr)
            disk_only = GetArtifactOptions(
                source="disk_only",
                verify_checksums=False,
            )
            p2p_only = GetArtifactOptions(
                source={
                    "preference": "prefer_p2p",
                    "allow_p2p": True,
                    "allow_disk": False,
                },
                verify_checksums=False,
                enable_verification=False,
            )
            artifact_a = store.artifact(artifact_id=artifact_a_id)
            artifact_b = store.artifact(artifact_id=artifact_b_id)

            binding = artifact_a.bind(
                device="cuda:0",
                packing="byte_space",
                options=disk_only,
            )

            synchronize_cuda()
            torch.testing.assert_close(
                binding.tensors["alpha"].cpu(),
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            )
            torch.testing.assert_close(
                binding.tensors["beta"].cpu(),
                torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32),
            )

            binding.publish_replica()
            old_replica_id = binding._slot.published_replica_id
            assert old_replica_id is not None
            _wait_replica_state(
                servicer, old_replica_id, expected_available=True, timeout_s=15.0
            )

            transport_probe = _request_replica_transport(
                gs_port=gs_port,
                artifact_id=artifact_a_id,
                requester_daemon_id=consumer_daemon_id,
                requester_p2p_port=consumer_p2p_port,
            )
            assert transport_probe.status == global_store_pb2.Status.STATUS_OK
            assert transport_probe.remote_memory_info.node_port == producer_p2p_port

            mismatch_probe = _request_replica_transport(
                gs_port=gs_port,
                artifact_id=artifact_b_id,
                requester_daemon_id=consumer_daemon_id,
                requester_p2p_port=consumer_p2p_port,
            )
            assert (
                mismatch_probe.status != global_store_pb2.Status.STATUS_OK
                or mismatch_probe.remote_memory_info.node_port != producer_p2p_port
            )

            consumer_payload, _ = consumer_store._materialization.materialize_subset(
                artifact_id=artifact_a_id,
                key=None,
                device="cuda:0",
                tensor_names=None,
                options=p2p_only,
            )
            try:
                assert consumer_payload.source == (
                    store_daemon_pb2.MATERIALIZATION_SOURCE_P2P
                )
                assert {desc.name for desc in consumer_payload.descriptors} == {
                    "alpha",
                    "beta",
                }
            finally:
                consumer_store._materialization._release_materialized(
                    consumer_payload,
                    consumer_store._runtime.ensure_client(),
                )

            binding.swap(artifact_b, options=disk_only, publish=True)
            synchronize_cuda()
            new_replica_id = binding._slot.published_replica_id
            assert new_replica_id is not None
            assert new_replica_id != old_replica_id

            torch.testing.assert_close(
                binding.tensors["alpha"].cpu(),
                torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float32),
            )
            torch.testing.assert_close(
                binding.tensors["beta"].cpu(),
                torch.tensor([13.0, 14.0, 15.0, 16.0], dtype=torch.float32),
            )

            _wait_replica_state(
                servicer, new_replica_id, expected_available=True, timeout_s=15.0
            )
            _wait_replica_retired(servicer, old_replica_id, timeout_s=15.0)

            retired_transport_probe = _request_replica_transport(
                gs_port=gs_port,
                artifact_id=artifact_a_id,
                requester_daemon_id=consumer_daemon_id,
                requester_p2p_port=consumer_p2p_port,
            )
            assert (
                retired_transport_probe.status != global_store_pb2.Status.STATUS_OK
                or retired_transport_probe.remote_memory_info.node_port
                != producer_p2p_port
            )

            binding.close()
            binding = None
            _wait_replica_retired(servicer, new_replica_id, timeout_s=15.0)
            closed_transport_probe = _request_replica_transport(
                gs_port=gs_port,
                artifact_id=artifact_b_id,
                requester_daemon_id=consumer_daemon_id,
                requester_p2p_port=consumer_p2p_port,
            )
            assert (
                closed_transport_probe.status != global_store_pb2.Status.STATUS_OK
                or closed_transport_probe.remote_memory_info.node_port
                != producer_p2p_port
            )

            kill_owner_ready = storage_dir / "kill_owner_publish.json"
            kill_owner_proc = _start_published_binding_child(
                daemon_addr=producer_addr,
                artifact_id=artifact_a_id,
                ready_path=kill_owner_ready,
                log_fd=log_fd,
                env=env,
            )
            kill_owner_payload = _wait_json_file(
                kill_owner_ready,
                timeout_s=30.0,
            )
            kill_owner_replica_id = str(kill_owner_payload["replica_id"])
            assert kill_owner_replica_id
            assert int(kill_owner_payload["pid"]) == kill_owner_proc.pid
            _wait_replica_state(
                servicer,
                kill_owner_replica_id,
                expected_available=True,
                timeout_s=15.0,
            )
            kill_owner_probe = _request_replica_transport(
                gs_port=gs_port,
                artifact_id=artifact_a_id,
                requester_daemon_id=consumer_daemon_id,
                requester_p2p_port=consumer_p2p_port,
            )
            assert kill_owner_probe.status == global_store_pb2.Status.STATUS_OK
            assert kill_owner_probe.remote_memory_info.node_port == producer_p2p_port

            kill_owner_proc.kill()
            kill_owner_proc.wait(timeout=10)
            kill_owner_proc = None
            _wait_replica_retired(
                servicer,
                kill_owner_replica_id,
                timeout_s=30.0,
            )
            killed_owner_probe = _request_replica_transport(
                gs_port=gs_port,
                artifact_id=artifact_a_id,
                requester_daemon_id=consumer_daemon_id,
                requester_p2p_port=consumer_p2p_port,
            )
            assert (
                killed_owner_probe.status != global_store_pb2.Status.STATUS_OK
                or killed_owner_probe.remote_memory_info.node_port
                != producer_p2p_port
            )
        finally:
            if kill_owner_proc is not None:
                kill_owner_proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    kill_owner_proc.wait(timeout=5)
            with contextlib.suppress(Exception):
                if binding is not None:
                    binding.close()
            with contextlib.suppress(Exception):
                if store is not None:
                    store.close()
            with contextlib.suppress(Exception):
                if consumer_store is not None:
                    consumer_store.close()
            producer_proc.terminate()
            consumer_proc.terminate()
            try:
                producer_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                producer_proc.kill()
            try:
                consumer_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                consumer_proc.kill()
