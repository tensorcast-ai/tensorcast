#  Copyright (c) 2025-2026, TensorCast Team.

"""Acceptance tests for dense piece assembly + sealing."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest
import yaml

from tensorcast.api.store.view_composer import compute_view_id
from tensorcast.cli_utils.proc import build_daemon_process_env, ensure_cpp_daemon_binary
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from google.protobuf import wrappers_pb2
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2_grpc
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2_grpc

pytestmark = [pytest.mark.requires_cuda_or_fake, pytest.mark.integration]


def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(addr: str, proc: subprocess.Popen, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=1.0)
            channel.close()
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("daemon failed to start")


@pytest.fixture(scope="module")
def gs_server():
    set_config(GlobalStoreConfig())
    servicer = GlobalStoreServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    global_store_pb2_grpc.add_GlobalStoreServiceServicer_to_server(servicer, server)
    port = _get_free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    try:
        yield (server, port)
    finally:
        server.stop(grace=None)


@pytest.fixture
def daemon_process(gs_server, tmp_path: Path):
    _, gs_port = gs_server
    bin_path = ensure_cpp_daemon_binary()
    listen_port = _get_free_port()
    storage_dir = tmp_path / "daemon"
    storage_dir.mkdir(parents=True, exist_ok=True)
    daemon_id = f"daemon_piece_{listen_port}"

    cfg = {
        "server": {
            "listen": {"host": "0.0.0.0", "port": listen_port},
            "p2p_listen": {"host": "0.0.0.0", "port": listen_port + 1000},
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
                {"name": "engine", "slice_bytes": 1 * 1024 * 1024, "pool_bytes": 64 * 1024 * 1024},
                {"name": "comm_gpu", "slice_bytes": 1 * 1024 * 1024, "pool_bytes": 16 * 1024 * 1024},
                {"name": "comm_cpu", "slice_bytes": 1 * 1024 * 1024, "pool_bytes": 8 * 1024 * 1024},
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
    }

    with tempfile.NamedTemporaryFile(
        prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = Path(f.name)

    env = build_daemon_process_env(os.environ)
    proc = subprocess.Popen(
        [str(bin_path), f"--config={cfg_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        _wait_ready(f"127.0.0.1:{listen_port}", proc)
        yield (f"127.0.0.1:{listen_port}", gs_port, proc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _make_index_bytes() -> bytes:
    index = {
        "weights": [
            0,
            32,
            [8],
            [1],
            "torch.float32",
            0,
        ]
    }
    return json.dumps(index, separators=(",", ":")).encode("utf-8")


def _make_view_spec(start: int, length: int) -> store_daemon_pb2.ViewSpec:
    spec = store_daemon_pb2.ViewSpec()
    ops = store_daemon_pb2.TensorViewOps()
    narrow = ops.ops.add().narrow
    narrow.dim = 0
    narrow.start = start
    narrow.length = length
    spec.tensors["weights"].CopyFrom(ops)
    return spec


def _register_piece(
    stub: store_daemon_pb2_grpc.StoreDaemonServiceStub,
    *,
    assembly_id: str,
    canonical_index_bytes: bytes,
    view_spec: store_daemon_pb2.ViewSpec,
    view_bytes: bytes,
) -> store_daemon_pb2.CommitRegisteredArtifactResponse:
    req = store_daemon_pb2.BeginRegisterArtifactRequest(
        device_id=0,
        total_size=len(view_bytes),
        owner_pid=os.getpid(),
        client_artifact_id=assembly_id,
    )
    req.tensor_index_data.data = canonical_index_bytes
    req.tensor_index_data.schema_version = "v3"
    req.tensor_index_data.encoding = "json"
    req.coalesced.max_inflight_bytes = max(1, len(view_bytes))
    req.view.spec.CopyFrom(view_spec)
    req.view.placement = store_daemon_pb2.TRANSFORM_PLACEMENT_SERVER
    req.view.canonical_size_bytes = 32
    req.view.registration_kind = store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE

    begin_resp = stub.BeginRegisterArtifact(req)

    def _iter():
        yield store_daemon_pb2.FeedRegisterArtifactStreamRequest(
            registration_id=begin_resp.registration_id,
            view_chunk=store_daemon_pb2.ViewUploadChunk(
                view_offset=0,
                data=view_bytes,
            ),
        )

    stub.FeedRegisterArtifactStream(_iter())
    return stub.CommitRegisteredArtifact(
        store_daemon_pb2.CommitRegisteredArtifactRequest(
            registration_id=begin_resp.registration_id
        )
    )


def _pack_floats(values: list[float]) -> bytes:
    fmt = "<" + "f" * len(values)
    return struct.pack(fmt, *values)


def test_piece_bootstrap_and_seal(daemon_process, gs_server):
    listen_addr, gs_port, _ = daemon_process
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)

    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = global_store_pb2_grpc.GlobalStoreServiceStub(gs_channel)

    assembly_id = "cgid:assembly-test"
    canonical_index_bytes = _make_index_bytes()

    view_spec_a = _make_view_spec(start=0, length=4)
    view_id_a = compute_view_id(view_spec_a, canonical_index_bytes)
    piece_a = _pack_floats([1.0, 2.0, 3.0, 4.0])

    commit_a = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        view_spec=view_spec_a,
        view_bytes=piece_a,
    )

    assert commit_a.view_id == view_id_a
    assert commit_a.view_data_hash
    assert commit_a.allow_partial is True

    index_resp = gs_stub.GetArtifactIndexById(
        global_store_pb2.GetArtifactIndexByIdRequest(artifact_id=assembly_id)
    )
    assert index_resp.status == global_store_pb2.Status.STATUS_OK
    assert index_resp.tensor_index_data == canonical_index_bytes

    info_req = global_store_pb2.GetArtifactInfoByIdRequest(artifact_id=assembly_id)
    info_req.requested_byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
    info_req.requested_byte_space.id = commit_a.view_id
    info_req.include_view_meta = True
    info_req.include_replicas.CopyFrom(wrappers_pb2.BoolValue(value=True))
    info_resp = gs_stub.GetArtifactInfoById(info_req)
    assert info_resp.status == global_store_pb2.Status.STATUS_OK
    assert info_resp.view_meta.view_size == len(piece_a)
    assert info_resp.view_meta.view_data_hash == commit_a.view_data_hash
    assert info_resp.replicas
    replica = info_resp.replicas[0]
    assert replica.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
    assert replica.byte_space.id == commit_a.view_id
    assert replica.memory_size == len(piece_a)

    # Idempotent retry should succeed with same hash.
    commit_retry = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        view_spec=view_spec_a,
        view_bytes=piece_a,
    )
    assert commit_retry.view_data_hash == commit_a.view_data_hash

    # Conflict should fail with different bytes.
    with pytest.raises(grpc.RpcError) as exc_info:
        _register_piece(
            stub,
            assembly_id=assembly_id,
            canonical_index_bytes=canonical_index_bytes,
            view_spec=view_spec_a,
            view_bytes=_pack_floats([9.0, 9.0, 9.0, 9.0]),
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    view_spec_b = _make_view_spec(start=4, length=4)
    piece_b = _pack_floats([5.0, 6.0, 7.0, 8.0])
    _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        view_spec=view_spec_b,
        view_bytes=piece_b,
    )

    seal_resp = stub.SealAssembly(
        store_daemon_pb2.SealAssemblyRequest(
            assembly_id=assembly_id,
            publish_canonical=True,
        )
    )
    assert seal_resp.sealed_artifact_id.startswith("mi2:")
    assert seal_resp.descriptor.artifact_id == seal_resp.sealed_artifact_id
    assert seal_resp.already_sealed is False

    seal_resp_2 = stub.SealAssembly(
        store_daemon_pb2.SealAssemblyRequest(
            assembly_id=assembly_id,
            publish_canonical=True,
        )
    )
    assert seal_resp_2.sealed_artifact_id == seal_resp.sealed_artifact_id
    assert seal_resp_2.already_sealed is True

    binding = gs_stub.GetArtifactBinding(
        global_store_pb2.GetArtifactBindingRequest(artifact_id=assembly_id)
    )
    assert binding.status == global_store_pb2.Status.STATUS_OK
    assert binding.binding.to_artifact_id == seal_resp.sealed_artifact_id

    replicas_resp = gs_stub.ListReplicasV2(
        global_store_pb2.ListReplicasV2Request(artifact_id=seal_resp.sealed_artifact_id)
    )
    assert len(replicas_resp.replicas) >= 1

    with pytest.raises(grpc.RpcError) as exc_info:
        _register_piece(
            stub,
            assembly_id=assembly_id,
            canonical_index_bytes=canonical_index_bytes,
            view_spec=_make_view_spec(start=0, length=4),
            view_bytes=piece_a,
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    channel.close()
    gs_channel.close()
