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
import torch
import yaml
from google.protobuf import wrappers_pb2

from tensorcast.api.store import ArtifactError, Store
from tensorcast.api.store.view_composer import compute_index_multihash, compute_view_id
from tensorcast.cli_utils.proc import build_daemon_process_env, ensure_cpp_daemon_binary
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2, store_daemon_pb2_grpc
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.layout.v1 import layout_pb2
from tensorcast.proto.operation.v1 import operation_pb2

pytestmark = [pytest.mark.requires_cuda_or_fake, pytest.mark.integration]


def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(addr: str, proc: subprocess.Popen, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("daemon exited before becoming ready")
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            resp = stub.GetServerConfig(
                store_daemon_pb2.GetServerConfigRequest(), timeout=1.0
            )
            channel.close()
            if resp.startup_phase == store_daemon_pb2.DAEMON_STARTUP_PHASE_READY:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("daemon failed to start")


def _wait_for_artifact_view_info(
    gs_stub: GlobalStoreCompositeStub,
    req: global_store_pb2.GetArtifactInfoByIdRequest,
    *,
    expected_view_data_hash: str | None = None,
    timeout_s: float = 5.0,
) -> global_store_pb2.GetArtifactInfoByIdResponse:
    deadline = time.time() + timeout_s
    last_resp = gs_stub.GetArtifactInfoById(req)
    while time.time() < deadline:
        if (
            last_resp.status == global_store_pb2.Status.STATUS_OK
            and last_resp.HasField("view_meta")
            and (
                expected_view_data_hash is None
                or last_resp.view_meta.view_data_hash == expected_view_data_hash
            )
        ):
            return last_resp
        time.sleep(0.1)
        last_resp = gs_stub.GetArtifactInfoById(req)
    return last_resp


def _wait_for_view_ready_for_seal(
    gs_stub: GlobalStoreCompositeStub,
    *,
    artifact_id: str,
    view_id: str,
    expected_view_data_hash: str,
    expected_view_size: int,
    min_replicas: int = 1,
    timeout_s: float = 15.0,
) -> global_store_pb2.GetArtifactInfoByIdResponse:
    deadline = time.time() + timeout_s
    req = global_store_pb2.GetArtifactInfoByIdRequest(artifact_id=artifact_id)
    req.requested_byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
    req.requested_byte_space.id = view_id
    req.include_view_meta = True
    req.include_replicas.CopyFrom(wrappers_pb2.BoolValue(value=True))
    last_resp = gs_stub.GetArtifactInfoById(req)
    while time.time() < deadline:
        if (
            last_resp.status == global_store_pb2.Status.STATUS_OK
            and last_resp.HasField("view_meta")
            and last_resp.view_meta.view_data_hash == expected_view_data_hash
            and last_resp.view_meta.view_size == expected_view_size
            and len(last_resp.replicas) >= min_replicas
        ):
            return last_resp
        time.sleep(0.1)
        last_resp = gs_stub.GetArtifactInfoById(req)
    return last_resp


def _artifact_index_multihash(
    gs_stub: GlobalStoreCompositeStub,
    *,
    artifact_id: str,
) -> str:
    resp = gs_stub.GetArtifactInfoById(
        global_store_pb2.GetArtifactInfoByIdRequest(artifact_id=artifact_id)
    )
    assert resp.status == global_store_pb2.Status.STATUS_OK
    assert resp.descriptor.index_multihash
    return str(resp.descriptor.index_multihash)


def _put_layout_for_source_artifact(
    gs_stub: GlobalStoreCompositeStub,
    *,
    artifact_id: str,
    expected_view_ids: list[str],
    replicated_tensors: list[str] | None = None,
) -> str:
    layout = layout_pb2.LayoutSpec(
        layout_schema_version=1,
        index_multihash=_artifact_index_multihash(gs_stub, artifact_id=artifact_id),
        expected_view_ids=sorted(expected_view_ids),
    )
    if replicated_tensors:
        layout.proof_schema_version = "v1"
        for tensor_name in replicated_tensors:
            layout.tensors[
                tensor_name
            ].overlap_mode = layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
    put_layout = gs_stub.PutLayoutSpec(
        global_store_pb2.PutLayoutSpecRequest(layout=layout)
    )
    assert put_layout.status == global_store_pb2.Status.STATUS_OK
    assert put_layout.layout_id
    return str(put_layout.layout_id)


def _artifact_tensor_dict(
    store: Store,
    *,
    artifact_id: str,
) -> dict[str, torch.Tensor]:
    tensors = store.artifact(artifact_id=artifact_id).tensor_dict(device="cpu")
    return {name: tensor.cpu() for name, tensor in tensors.items()}


def _start_and_wait_seal_success(
    stub: store_daemon_pb2_grpc.StoreDaemonServiceStub,
    *,
    assembly_id: str,
    timeout_s: float = 60.0,
) -> store_daemon_pb2.WaitOperationResponse:
    deadline = time.time() + timeout_s
    last_error = "seal did not start"
    while time.time() < deadline:
        start_resp = stub.StartSealAssembly(
            store_daemon_pb2.StartSealAssemblyRequest(assembly_id=assembly_id)
        )
        assert start_resp.operation.operation_id
        operation_id = start_resp.operation.operation_id
        while time.time() < deadline:
            remaining_ms = max(1, int((deadline - time.time()) * 1000))
            wait_resp = stub.WaitOperation(
                store_daemon_pb2.WaitOperationRequest(
                    operation_id=operation_id,
                    timeout_ms=min(2_000, remaining_ms),
                )
            )
            status = wait_resp.operation.status
            if status.state == operation_pb2.OPERATION_STATE_SUCCESS:
                return wait_resp
            if status.state in (
                operation_pb2.OPERATION_STATE_PENDING,
                operation_pb2.OPERATION_STATE_RUNNING,
            ):
                continue

            error_code = ""
            error_msg = status.message
            retryable = False
            if status.HasField("error"):
                error_code = status.error.status_code
                error_msg = status.error.message or status.message
                retryable = status.error.retryable
            last_error = f"state={status.state} code={error_code} msg={error_msg}"

            if (
                retryable
                and error_code == "UNAVAILABLE"
                and "missing canonical ranges" in error_msg
            ):
                time.sleep(0.2)
                break
            if "RequestReplicaTransport(view) failed: STATUS_TIMED_OUT" in error_msg:
                time.sleep(0.2)
                break
            if "RequestReplicaTransport(view) failed: STATUS_NOT_FOUND" in error_msg:
                time.sleep(0.2)
                break

            pytest.fail(f"seal operation failed: {last_error}")

    pytest.fail(f"seal operation timed out after {timeout_s:.1f}s: {last_error}")


def _build_daemon_config(
    *,
    listen_port: int,
    storage_dir: Path,
    gs_port: int,
    daemon_id: str,
    post_seal: dict[str, bool] | None = None,
) -> dict:
    cfg: dict = {
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
            # Keep heartbeats below Global Store heartbeat_timeout so replicas
            # remain eligible during longer seal retries.
            "heartbeat_interval": "5s",
            "periodic_sync_interval": "0s",
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
    if post_seal is not None:
        cfg["post_seal"] = post_seal
    return cfg


def _spawn_daemon(
    *,
    gs_port: int,
    tmp_path: Path,
    post_seal: dict[str, bool] | None = None,
) -> tuple[str, int, subprocess.Popen]:
    bin_path = ensure_cpp_daemon_binary()
    listen_port = _get_free_port()
    storage_dir = tmp_path / f"daemon_{listen_port}"
    storage_dir.mkdir(parents=True, exist_ok=True)
    daemon_id = f"daemon_piece_{listen_port}"

    cfg = _build_daemon_config(
        listen_port=listen_port,
        storage_dir=storage_dir,
        gs_port=gs_port,
        daemon_id=daemon_id,
        post_seal=post_seal,
    )

    with tempfile.NamedTemporaryFile(
        prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = Path(f.name)

    env = build_daemon_process_env(os.environ)
    proc_log = storage_dir / "daemon_proc.log"
    with proc_log.open("a") as log_fd:
        proc = subprocess.Popen(
            [str(bin_path), f"--config={cfg_path}"],
            stdout=log_fd,
            stderr=log_fd,
            env=env,
        )
        try:
            _wait_ready(f"127.0.0.1:{listen_port}", proc)
            return (f"127.0.0.1:{listen_port}", gs_port, proc)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise


@pytest.fixture(scope="module")
def gs_server():
    set_config(GlobalStoreConfig())
    servicer = GlobalStoreServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    register_global_store_servicers(server, servicer)
    port = server.add_insecure_port("127.0.0.1:0")
    if port <= 0:
        raise RuntimeError("failed to bind Global Store server port")
    server.start()
    try:
        yield (server, port)
    finally:
        server.stop(grace=None)


@pytest.fixture
def daemon_process(gs_server, tmp_path: Path):
    _, gs_port = gs_server
    listen_addr, gs_port, proc = _spawn_daemon(gs_port=gs_port, tmp_path=tmp_path)
    try:
        yield (listen_addr, gs_port, proc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def daemon_process_reuse(gs_server, tmp_path: Path):
    _, gs_port = gs_server
    listen_addr, gs_port, proc = _spawn_daemon(
        gs_port=gs_port,
        tmp_path=tmp_path,
        post_seal={"reuse_views_if_safe": True},
    )
    try:
        yield (listen_addr, gs_port, proc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def daemon_process_migrate(gs_server, tmp_path: Path):
    _, gs_port = gs_server
    listen_addr, gs_port, proc = _spawn_daemon(
        gs_port=gs_port,
        tmp_path=tmp_path,
        post_seal={"migrate_views": True},
    )
    try:
        yield (listen_addr, gs_port, proc)
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
        ],
        "bias": [
            32,
            32,
            [8],
            [1],
            "torch.float32",
            0,
        ],
    }
    return json.dumps(index, separators=(",", ":")).encode("utf-8")


def _make_view_spec(*, bias_start: int, bias_length: int) -> common_pb2.ViewSpec:
    spec = common_pb2.ViewSpec()
    weights_ops = common_pb2.TensorViewOps()
    weights_narrow = weights_ops.ops.add().narrow
    weights_narrow.dim = 0
    weights_narrow.start = 0
    weights_narrow.length = 8
    spec.tensors["weights"].CopyFrom(weights_ops)

    bias_ops = common_pb2.TensorViewOps()
    bias_narrow = bias_ops.ops.add().narrow
    bias_narrow.dim = 0
    bias_narrow.start = bias_start
    bias_narrow.length = bias_length
    spec.tensors["bias"].CopyFrom(bias_ops)
    return spec


def _register_piece(
    stub: store_daemon_pb2_grpc.StoreDaemonServiceStub,
    *,
    assembly_id: str,
    canonical_index_bytes: bytes,
    canonical_size_bytes: int,
    view_spec: common_pb2.ViewSpec,
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
    req.view.canonical_size_bytes = int(canonical_size_bytes)
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


def _seal_two_piece_assembly(
    stub: store_daemon_pb2_grpc.StoreDaemonServiceStub,
    gs_stub: GlobalStoreCompositeStub,
    *,
    assembly_id: str,
    canonical_index_bytes: bytes,
    canonical_size_bytes: int,
) -> tuple[str, str, str]:
    seed_port = _get_free_port()
    worker_resp = gs_stub.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            node_id=f"seed_node_{assembly_id}",
            node_address="192.168.1.1",
            grpc_port=seed_port,
            p2p_port=seed_port + 1,
            mem_pool_total_size=1,
            mem_pool_available_size=1,
            daemon_id=f"daemon_seed_{assembly_id}",
        )
    )
    assert worker_resp.status == global_store_pb2.Status.STATUS_OK
    _ = gs_stub.RegisterReplica(
        global_store_pb2.RegisterReplicaRequest(
            artifact_id=f"{assembly_id}-seed",
            mem_info=common_pb2.MemoryInfo(
                node_id=f"seed_node_{assembly_id}",
                node_address="192.168.1.1",
                node_port=seed_port,
                memory_size=1,
                memory_type=common_pb2.MemoryType.MEMORY_TYPE_RAM,
                device_id=0,
            ),
            max_concurrency=1,
            worker_id=worker_resp.worker_id,
            tensor_index_data=canonical_index_bytes,
            encoding="json",
            schema_version="v3",
        )
    )

    view_spec_a = _make_view_spec(bias_start=0, bias_length=4)
    view_id_a = compute_view_id(view_spec_a, canonical_index_bytes)
    view_spec_b = _make_view_spec(bias_start=4, bias_length=4)
    view_id_b = compute_view_id(view_spec_b, canonical_index_bytes)

    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    bias = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    piece_a = _pack_floats(bias[:4]) + _pack_floats(weights)
    piece_b = _pack_floats(bias[4:]) + _pack_floats(weights)

    index_mh = compute_index_multihash(canonical_index_bytes)
    layout = layout_pb2.LayoutSpec(
        layout_schema_version=1,
        index_multihash=index_mh,
        expected_view_ids=sorted([view_id_a, view_id_b]),
        proof_schema_version="v1",
    )
    layout.tensors["weights"].overlap_mode = layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
    put_layout = gs_stub.PutLayoutSpec(
        global_store_pb2.PutLayoutSpecRequest(layout=layout)
    )
    assert put_layout.status == global_store_pb2.Status.STATUS_OK

    bind_layout = gs_stub.UpdateAssemblyLayoutBinding(
        global_store_pb2.UpdateAssemblyLayoutBindingRequest(
            assembly_id=assembly_id,
            layout_id=put_layout.layout_id,
            expected_binding_version=0,
        )
    )
    assert bind_layout.status == global_store_pb2.Status.STATUS_OK

    commit_a = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_a,
        view_bytes=piece_a,
    )
    commit_b = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_b,
        view_bytes=piece_b,
    )
    assert commit_a.view_id == view_id_a
    assert commit_b.view_id == view_id_b
    info_resp_a = _wait_for_view_ready_for_seal(
        gs_stub,
        artifact_id=assembly_id,
        view_id=view_id_a,
        expected_view_data_hash=commit_a.view_data_hash,
        expected_view_size=len(piece_a),
        min_replicas=1,
    )
    assert info_resp_a.status == global_store_pb2.Status.STATUS_OK
    assert info_resp_a.view_meta.view_size == len(piece_a)
    assert info_resp_a.view_meta.view_data_hash == commit_a.view_data_hash
    info_resp_b = _wait_for_view_ready_for_seal(
        gs_stub,
        artifact_id=assembly_id,
        view_id=view_id_b,
        expected_view_data_hash=commit_b.view_data_hash,
        expected_view_size=len(piece_b),
        min_replicas=1,
    )
    assert info_resp_b.status == global_store_pb2.Status.STATUS_OK
    assert info_resp_b.view_meta.view_size == len(piece_b)
    assert info_resp_b.view_meta.view_data_hash == commit_b.view_data_hash

    wait_resp = _start_and_wait_seal_success(stub, assembly_id=assembly_id)
    assert wait_resp.operation.status.state == operation_pb2.OPERATION_STATE_SUCCESS
    op_result = store_daemon_pb2.SealAssemblyResult()
    assert wait_resp.operation.status.result.Unpack(op_result) is True
    return op_result.artifact.artifact_id, view_id_a, view_id_b


def test_piece_bootstrap_and_seal(daemon_process, gs_server):
    listen_addr, gs_port, _ = daemon_process
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)

    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = GlobalStoreCompositeStub(gs_channel)

    assembly_id = "cgid:assembly-test"
    canonical_index_bytes = _make_index_bytes()
    canonical_size_bytes = 64

    # PutLayoutSpec validates tensor keys against the canonical index. Seed the index bytes
    # into `artifact_indices` via the RegisterReplica UPSERT path.
    worker_resp = gs_stub.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            node_id="seed_node",
            node_address="192.168.1.1",
            grpc_port=12345,
            p2p_port=12346,
            mem_pool_total_size=1,
            mem_pool_available_size=1,
            daemon_id="daemon_seed_layout",
        )
    )
    assert worker_resp.status == global_store_pb2.Status.STATUS_OK
    _ = gs_stub.RegisterReplica(
        global_store_pb2.RegisterReplicaRequest(
            artifact_id="cgid:layout-seed",
            mem_info=common_pb2.MemoryInfo(
                node_id="seed_node",
                node_address="192.168.1.1",
                node_port=12345,
                memory_size=1,
                memory_type=common_pb2.MemoryType.MEMORY_TYPE_RAM,
                device_id=0,
            ),
            max_concurrency=1,
            worker_id=worker_resp.worker_id,
            tensor_index_data=canonical_index_bytes,
            encoding="json",
            schema_version="v3",
        )
    )

    view_spec_a = _make_view_spec(bias_start=0, bias_length=4)
    view_id_a = compute_view_id(view_spec_a, canonical_index_bytes)
    view_spec_b = _make_view_spec(bias_start=4, bias_length=4)
    view_id_b = compute_view_id(view_spec_b, canonical_index_bytes)
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    bias = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    # View byte stream order follows the ViewPlanner ordering (canonical index tensor name order).
    piece_a = _pack_floats(bias[:4]) + _pack_floats(weights)

    index_mh = compute_index_multihash(canonical_index_bytes)
    layout = layout_pb2.LayoutSpec(
        layout_schema_version=1,
        index_multihash=index_mh,
        expected_view_ids=sorted([view_id_a, view_id_b]),
        proof_schema_version="v1",
    )
    layout.tensors["weights"].overlap_mode = layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
    put_layout = gs_stub.PutLayoutSpec(
        global_store_pb2.PutLayoutSpecRequest(layout=layout)
    )
    assert put_layout.status == global_store_pb2.Status.STATUS_OK
    assert put_layout.layout_id
    bind_layout = gs_stub.UpdateAssemblyLayoutBinding(
        global_store_pb2.UpdateAssemblyLayoutBindingRequest(
            assembly_id=assembly_id,
            layout_id=put_layout.layout_id,
            expected_binding_version=0,
        )
    )
    assert bind_layout.status == global_store_pb2.Status.STATUS_OK

    commit_a = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_a,
        view_bytes=piece_a,
    )

    assert commit_a.view_id == view_id_a
    assert commit_a.view_data_hash
    assert commit_a.registration_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE

    index_resp = gs_stub.GetArtifactIndexById(
        global_store_pb2.GetArtifactIndexByIdRequest(artifact_id=assembly_id)
    )
    assert index_resp.status == global_store_pb2.Status.STATUS_OK
    assert index_resp.tensor_index_data == canonical_index_bytes

    info_req = global_store_pb2.GetArtifactInfoByIdRequest(artifact_id=assembly_id)
    info_req.requested_byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
    info_req.requested_byte_space.id = commit_a.view_id
    info_req.include_view_meta = True
    info_req.include_replicas.CopyFrom(wrappers_pb2.BoolValue(value=False))
    info_resp = _wait_for_artifact_view_info(
        gs_stub, info_req, expected_view_data_hash=commit_a.view_data_hash
    )
    assert info_resp.status == global_store_pb2.Status.STATUS_OK
    assert info_resp.view_meta.view_size == len(piece_a)
    assert info_resp.view_meta.view_data_hash == commit_a.view_data_hash

    # Idempotent retry should succeed with same hash.
    commit_retry = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
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
            canonical_size_bytes=canonical_size_bytes,
            view_spec=view_spec_a,
            view_bytes=_pack_floats([9.0] * 12),
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    piece_b = _pack_floats(bias[4:]) + _pack_floats(weights)
    commit_b = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_b,
        view_bytes=piece_b,
    )
    assert commit_b.view_id == view_id_b
    assert commit_b.view_data_hash
    commit_b_retry = _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_b,
        view_bytes=piece_b,
    )
    assert commit_b_retry.view_data_hash == commit_b.view_data_hash
    info_resp_a_ready = _wait_for_view_ready_for_seal(
        gs_stub,
        artifact_id=assembly_id,
        view_id=commit_a.view_id,
        expected_view_data_hash=commit_a.view_data_hash,
        expected_view_size=len(piece_a),
        min_replicas=1,
    )
    assert info_resp_a_ready.status == global_store_pb2.Status.STATUS_OK
    assert info_resp_a_ready.view_meta.view_size == len(piece_a)
    assert info_resp_a_ready.view_meta.view_data_hash == commit_a.view_data_hash
    _maybe_debug_dump_view_replicas("piece_a", info_resp_a_ready)
    info_resp_b = _wait_for_view_ready_for_seal(
        gs_stub,
        artifact_id=assembly_id,
        view_id=commit_b.view_id,
        expected_view_data_hash=commit_b.view_data_hash,
        expected_view_size=len(piece_b),
        min_replicas=1,
    )
    assert info_resp_b.status == global_store_pb2.Status.STATUS_OK
    assert info_resp_b.view_meta.view_size == len(piece_b)
    assert info_resp_b.view_meta.view_data_hash == commit_b.view_data_hash
    _maybe_debug_dump_view_replicas("piece_b", info_resp_b)

    wait_resp = _start_and_wait_seal_success(stub, assembly_id=assembly_id)
    assert wait_resp.operation.status.state == operation_pb2.OPERATION_STATE_SUCCESS
    op_result = store_daemon_pb2.SealAssemblyResult()
    assert wait_resp.operation.status.result.Unpack(op_result) is True
    assert op_result.artifact.artifact_id.startswith("mi2:")
    snapshot = store_daemon_pb2.SealAssemblySnapshot()
    assert wait_resp.operation.snapshot.Unpack(snapshot) is True
    assert snapshot.layout_id == put_layout.layout_id
    assert (
        snapshot.assembly_layout_binding_version == bind_layout.binding.binding_version
    )

    seal_resp_2 = stub.SealAssembly(
        store_daemon_pb2.SealAssemblyRequest(
            assembly_id=assembly_id,
            # Verify idempotent seal lookup through the legacy sync path.
            # Re-publishing canonical bytes is handled by StartSealAssembly.
            publish_canonical=False,
        )
    )
    assert seal_resp_2.sealed_artifact_id == op_result.artifact.artifact_id
    assert seal_resp_2.already_sealed is True

    binding = gs_stub.GetArtifactBinding(
        global_store_pb2.GetArtifactBindingRequest(artifact_id=assembly_id)
    )
    assert binding.status == global_store_pb2.Status.STATUS_OK
    assert binding.binding.to_artifact_id == op_result.artifact.artifact_id
    attached = gs_stub.ListArtifactLayouts(
        global_store_pb2.ListArtifactLayoutsRequest(
            mi2_id=op_result.artifact.artifact_id
        )
    )
    assert attached.status == global_store_pb2.Status.STATUS_OK
    assert put_layout.layout_id in attached.layout_ids

    replicas_resp = gs_stub.ListReplicasV2(
        global_store_pb2.ListReplicasV2Request(
            artifact_id=op_result.artifact.artifact_id
        )
    )
    assert len(replicas_resp.replicas) >= 1

    with pytest.raises(grpc.RpcError) as exc_info:
        _register_piece(
            stub,
            assembly_id=assembly_id,
            canonical_index_bytes=canonical_index_bytes,
            canonical_size_bytes=canonical_size_bytes,
            view_spec=_make_view_spec(bias_start=0, bias_length=4),
            view_bytes=piece_a,
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    channel.close()
    gs_channel.close()


@pytest.mark.skip(
    reason=(
        "Blocked by binding-backed acceptance path on the current branch; "
        "see 0085 plan status for remaining canonical_full/PP/EP blockers"
    )
)
def test_binding_canonical_full_attempt_publishes_lineage(daemon_process, gs_server):
    listen_addr, gs_port, _ = daemon_process
    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = GlobalStoreCompositeStub(gs_channel)
    store = Store(listen_addr)
    binding = None
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
    try:
        source_artifact_id, _, _ = _seal_two_piece_assembly(
            stub,
            gs_stub,
            assembly_id="cgid:binding-source-canonical",
            canonical_index_bytes=_make_index_bytes(),
            canonical_size_bytes=64,
        )
        layout_id = _put_layout_for_source_artifact(
            gs_stub,
            artifact_id=source_artifact_id,
            expected_view_ids=[],
        )

        source_artifact = store.artifact(artifact_id=source_artifact_id)
        binding = source_artifact.bind(device="cuda:0", packing="byte_space")
        sealed = binding.seal_current(update_epoch=binding.begin_update())
        attempt = store.start_assembly_attempt(layout_id=layout_id)

        source_version_key = "models/demo/source/v1"
        serving_version_key = "models/demo/serving/v1"
        policy_resp = gs_stub.UpdateAssemblyRuntimePolicy(
            global_store_pb2.UpdateAssemblyRuntimePolicyRequest(
                assembly_id=attempt.assembly_id,
                policy_json=json.dumps(
                    {
                        "source_version_key": source_version_key,
                        "serving_version_key": serving_version_key,
                        "serving_manifest_ref": "manifest://demo/v1",
                    }
                ),
                expected_policy_version=0,
            )
        )
        assert policy_resp.status == global_store_pb2.Status.STATUS_OK

        partial = sealed.contribute_to_assembly(attempt=attempt)
        assert partial.contribution_kind == "canonical_full"

        result = store.wait_assembly_attempt(attempt, timeout_s=60.0)
        assert result.source_version_key == source_version_key
        assert result.serving_version_key == serving_version_key
        assert result.serving_manifest_ref == "manifest://demo/v1"
        assert result.serving_artifact_id == result.source_artifact_id

        source_mapping = gs_stub.ResolveKeyMapping(
            global_store_pb2.ResolveKeyMappingRequest(key=source_version_key)
        )
        assert source_mapping.status == global_store_pb2.Status.STATUS_OK
        assert source_mapping.artifact_id == result.source_artifact_id

        serving_mapping = gs_stub.ResolveKeyMapping(
            global_store_pb2.ResolveKeyMappingRequest(key=serving_version_key)
        )
        assert serving_mapping.status == global_store_pb2.Status.STATUS_OK
        assert serving_mapping.artifact_id == result.source_artifact_id

        tensors = _artifact_tensor_dict(store, artifact_id=result.source_artifact_id)
        torch.testing.assert_close(
            tensors["weights"],
            torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            tensors["bias"],
            torch.tensor(
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                dtype=torch.float32,
            ),
        )
    finally:
        with contextlib.suppress(Exception):
            channel.close()
        with contextlib.suppress(Exception):
            if binding is not None:
                binding.close()
        with contextlib.suppress(Exception):
            store.close()
        gs_channel.close()


def test_binding_piece_partial_replacement_and_pp_attempt(daemon_process, gs_server):
    listen_addr, gs_port, _ = daemon_process
    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = GlobalStoreCompositeStub(gs_channel)
    store = Store(listen_addr)
    binding_a_old = None
    binding_a_new = None
    binding_b = None
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
    try:
        source_artifact_id, _, _ = _seal_two_piece_assembly(
            stub,
            gs_stub,
            assembly_id="cgid:binding-source-pp",
            canonical_index_bytes=_make_index_bytes(),
            canonical_size_bytes=64,
        )
        source_artifact = store.artifact(artifact_id=source_artifact_id)

        binding_a_old = source_artifact.view(slices={"bias": (slice(0, 4),)}).bind(
            device="cuda:0", packing="byte_space"
        )
        binding_a_new = source_artifact.view(slices={"bias": (slice(0, 4),)}).bind(
            device="cuda:0", packing="byte_space"
        )
        binding_b = source_artifact.view(slices={"bias": (slice(4, 8),)}).bind(
            device="cuda:0", packing="byte_space"
        )

        view_id_a = str(binding_a_old.selection.view_id)
        view_id_b = str(binding_b.selection.view_id)
        layout_id = _put_layout_for_source_artifact(
            gs_stub,
            artifact_id=source_artifact_id,
            expected_view_ids=[view_id_a, view_id_b],
            replicated_tensors=["weights"],
        )
        attempt = store.start_assembly_attempt(layout_id=layout_id)

        sealed_a_old = binding_a_old.seal_current(
            update_epoch=binding_a_old.begin_update()
        )

        update_epoch_a_new = binding_a_new.begin_update()
        binding_a_new.tensors["bias"].copy_(
            torch.tensor(
                [101.0, 102.0, 103.0, 104.0],
                dtype=torch.float32,
                device=binding_a_new.tensors["bias"].device,
            )
        )
        sealed_a_new = binding_a_new.seal_current(update_epoch=update_epoch_a_new)

        sealed_b = binding_b.seal_current(update_epoch=binding_b.begin_update())

        first = sealed_a_old.contribute_to_assembly(attempt=attempt)
        assert first.contribution_kind == "piece_partial"
        assert first.view_id == view_id_a

        replacement = sealed_a_new.contribute_to_assembly(attempt=attempt)
        assert replacement.contribution_kind == "piece_partial"
        assert replacement.view_id == view_id_a

        reopened_epoch = binding_a_old.begin_update()
        assert reopened_epoch

        second = sealed_b.contribute_to_assembly(attempt=attempt)
        assert second.contribution_kind == "piece_partial"
        assert second.view_id == view_id_b

        result = store.wait_assembly_attempt(attempt, timeout_s=60.0)
        tensors = _artifact_tensor_dict(store, artifact_id=result.source_artifact_id)
        torch.testing.assert_close(
            tensors["weights"],
            torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            tensors["bias"],
            torch.tensor(
                [101.0, 102.0, 103.0, 104.0, 13.0, 14.0, 15.0, 16.0],
                dtype=torch.float32,
            ),
        )
    finally:
        with contextlib.suppress(Exception):
            channel.close()
        with contextlib.suppress(Exception):
            if binding_a_old is not None:
                binding_a_old.close()
        with contextlib.suppress(Exception):
            if binding_a_new is not None:
                binding_a_new.close()
        with contextlib.suppress(Exception):
            if binding_b is not None:
                binding_b.close()
        with contextlib.suppress(Exception):
            store.close()
        gs_channel.close()


def test_binding_ep_subset_attempt(daemon_process, gs_server):
    listen_addr, gs_port, _ = daemon_process
    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = GlobalStoreCompositeStub(gs_channel)
    store = Store(listen_addr)
    binding_e0 = None
    binding_e1 = None
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
    try:
        source_artifact_id, _, _ = _seal_two_piece_assembly(
            stub,
            gs_stub,
            assembly_id="cgid:binding-source-ep",
            canonical_index_bytes=_make_index_bytes(),
            canonical_size_bytes=64,
        )
        source_artifact = store.artifact(artifact_id=source_artifact_id)

        binding_e0 = source_artifact.subset(["weights"]).bind(
            device="cuda:0",
            packing="byte_space",
        )
        binding_e1 = source_artifact.subset(["bias"]).bind(
            device="cuda:0",
            packing="byte_space",
        )

        view_id_e0 = str(binding_e0.selection.view_id)
        view_id_e1 = str(binding_e1.selection.view_id)
        layout_id = _put_layout_for_source_artifact(
            gs_stub,
            artifact_id=source_artifact_id,
            expected_view_ids=[view_id_e0, view_id_e1],
        )
        attempt = store.start_assembly_attempt(layout_id=layout_id)

        sealed_e0 = binding_e0.seal_current(update_epoch=binding_e0.begin_update())
        sealed_e1 = binding_e1.seal_current(update_epoch=binding_e1.begin_update())
        sealed_e0.contribute_to_assembly(attempt=attempt)
        sealed_e1.contribute_to_assembly(attempt=attempt)

        result = store.wait_assembly_attempt(attempt, timeout_s=60.0)
        tensors = _artifact_tensor_dict(store, artifact_id=result.source_artifact_id)
        torch.testing.assert_close(
            tensors["weights"],
            torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            tensors["bias"],
            torch.tensor(
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                dtype=torch.float32,
            ),
        )
    finally:
        with contextlib.suppress(Exception):
            channel.close()
        with contextlib.suppress(Exception):
            if binding_e0 is not None:
                binding_e0.close()
        with contextlib.suppress(Exception):
            if binding_e1 is not None:
                binding_e1.close()
        with contextlib.suppress(Exception):
            store.close()
        gs_channel.close()


def test_binding_attempt_requires_all_expected_views(daemon_process, gs_server):
    listen_addr, gs_port, _ = daemon_process
    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = GlobalStoreCompositeStub(gs_channel)
    store = Store(listen_addr)
    binding_a = None
    binding_b = None
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
    try:
        source_artifact_id, _, _ = _seal_two_piece_assembly(
            stub,
            gs_stub,
            assembly_id="cgid:binding-source-incomplete",
            canonical_index_bytes=_make_index_bytes(),
            canonical_size_bytes=64,
        )
        source_artifact = store.artifact(artifact_id=source_artifact_id)
        binding_a = source_artifact.view(slices={"bias": (slice(0, 4),)}).bind(
            device="cuda:0", packing="byte_space"
        )
        binding_b = source_artifact.view(slices={"bias": (slice(4, 8),)}).bind(
            device="cuda:0", packing="byte_space"
        )

        layout_id = _put_layout_for_source_artifact(
            gs_stub,
            artifact_id=source_artifact_id,
            expected_view_ids=[
                str(binding_a.selection.view_id),
                str(binding_b.selection.view_id),
            ],
            replicated_tensors=["weights"],
        )
        attempt = store.start_assembly_attempt(layout_id=layout_id)
        sealed_a = binding_a.seal_current(update_epoch=binding_a.begin_update())
        sealed_a.contribute_to_assembly(attempt=attempt)

        with pytest.raises(ArtifactError) as exc_info:
            store.wait_assembly_attempt(attempt, timeout_s=20.0)
        assert "required expected_view_id missing" in str(exc_info.value)
    finally:
        with contextlib.suppress(Exception):
            channel.close()
        with contextlib.suppress(Exception):
            if binding_a is not None:
                binding_a.close()
        with contextlib.suppress(Exception):
            if binding_b is not None:
                binding_b.close()
        with contextlib.suppress(Exception):
            store.close()
        gs_channel.close()


def test_post_seal_reuse_views_if_safe(daemon_process_reuse, gs_server):
    listen_addr, gs_port, _ = daemon_process_reuse
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)

    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = GlobalStoreCompositeStub(gs_channel)

    assembly_id = "cgid:assembly-reuse"
    canonical_index_bytes = _make_index_bytes()
    canonical_size_bytes = 64

    sealed_id, view_id_a, _ = _seal_two_piece_assembly(
        stub,
        gs_stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
    )

    mat_resp = stub.MaterializeReplica(
        store_daemon_pb2.MaterializeReplicaRequest(
            selection=common_pb2.ArtifactSelection(
                artifact_id=assembly_id,
                view_id=view_id_a,
            ),
            target_device_type=store_daemon_pb2.DEVICE_TYPE_GPU,
            wait_for_completion=True,
        )
    )
    assert mat_resp.status == store_daemon_pb2.MATERIALIZE_REPLICA_STATUS_ALLOCATED
    assert mat_resp.artifact_id == sealed_id

    channel.close()
    gs_channel.close()


def test_post_seal_migrate_views(daemon_process_migrate, gs_server):
    listen_addr, gs_port, _ = daemon_process_migrate
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)

    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = GlobalStoreCompositeStub(gs_channel)

    assembly_id = "cgid:assembly-migrate"
    canonical_index_bytes = _make_index_bytes()
    canonical_size_bytes = 64

    sealed_id, view_id_a, view_id_b = _seal_two_piece_assembly(
        stub,
        gs_stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
    )

    list_resp = gs_stub.ListViews(
        global_store_pb2.ListViewsRequest(artifact_id=sealed_id)
    )
    assert list_resp.status == global_store_pb2.Status.STATUS_OK
    view_ids = {entry.view_id for entry in list_resp.views}
    assert view_id_a in view_ids
    assert view_id_b in view_ids

    channel.close()
    gs_channel.close()
