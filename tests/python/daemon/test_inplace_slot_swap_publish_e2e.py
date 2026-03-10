#  Copyright (c) 2026, TensorCast Team.

"""End-to-end swap+publish coverage for slot-based materialization."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import grpc
import pytest
import torch
import yaml

from tensorcast.api.store import FallbackOptions, Store
from tensorcast.api.store.view_composer import compute_index_multihash
from tensorcast.cli_utils.proc import build_daemon_process_env, ensure_cpp_daemon_binary
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2, store_daemon_pb2_grpc
from tensorcast.proto.global_store.v1 import global_store_pb2

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


def _wait_ready(addr: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=1.0)
            channel.close()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("daemon failed to start")


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
    artifact_a_id = "artifact_a"
    artifact_b_id = "artifact_b"
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

    listen_port = _get_free_port()
    p2p_port = _get_free_port()
    daemon_id = f"daemon_slot_{listen_port}"
    cfg = _build_daemon_config(
        listen_port=listen_port,
        p2p_port=p2p_port,
        storage_dir=storage_dir,
        gs_port=gs_port,
        daemon_id=daemon_id,
    )

    with tempfile.NamedTemporaryFile(
        prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False
    ) as cfg_file:
        yaml.safe_dump(cfg, cfg_file, sort_keys=False)
        cfg_path = Path(cfg_file.name)

    proc_log = storage_dir / "daemon_proc.log"
    env = build_daemon_process_env(os.environ)
    with proc_log.open("a") as log_fd:
        proc = subprocess.Popen(
            [str(bin_path), f"--config={cfg_path}"],
            stdout=log_fd,
            stderr=log_fd,
            env=env,
        )
        try:
            _wait_ready(f"127.0.0.1:{listen_port}")
            _wait_for_worker(gs_port, listen_port)

            store = Store(f"127.0.0.1:{listen_port}")
            fallback_a = FallbackOptions(
                prefer="disk",
                allow_p2p=False,
                verify_checksums=False,
            )
            fallback_b = FallbackOptions(
                prefer="disk",
                allow_p2p=False,
                verify_checksums=False,
            )
            artifact_a = store.artifact(artifact_id=artifact_a_id, fallback=fallback_a)
            artifact_b = store.artifact(artifact_id=artifact_b_id, fallback=fallback_b)

            with artifact_a.deferred_loader(
                device="cuda:0", packing="byte_space"
            ) as loader:
                _ = loader.tensor("alpha")
                _ = loader.tensor("beta")
                slot = loader.commit()

            torch.cuda.synchronize()
            torch.testing.assert_close(
                slot.tensors["alpha"].cpu(),
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            )
            torch.testing.assert_close(
                slot.tensors["beta"].cpu(),
                torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32),
            )

            slot.publish_replica()
            old_replica_id = slot.published_replica_id
            assert old_replica_id is not None
            _wait_replica_state(
                servicer, old_replica_id, expected_available=True, timeout_s=15.0
            )

            slot.swap(artifact_b, publish=True)
            torch.cuda.synchronize()
            new_replica_id = slot.published_replica_id
            assert new_replica_id is not None
            assert new_replica_id != old_replica_id

            torch.testing.assert_close(
                slot.tensors["alpha"].cpu(),
                torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float32),
            )
            torch.testing.assert_close(
                slot.tensors["beta"].cpu(),
                torch.tensor([13.0, 14.0, 15.0, 16.0], dtype=torch.float32),
            )

            _wait_replica_state(
                servicer, new_replica_id, expected_available=True, timeout_s=15.0
            )
            _wait_replica_retired(servicer, old_replica_id, timeout_s=15.0)
        finally:
            with contextlib.suppress(Exception):
                slot.close()
            with contextlib.suppress(Exception):
                store.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
