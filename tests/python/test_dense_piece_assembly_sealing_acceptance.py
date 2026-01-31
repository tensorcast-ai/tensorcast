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

from tensorcast.api.store.view_composer import compute_index_multihash, compute_view_id
from tensorcast.cli_utils.proc import build_daemon_process_env, ensure_cpp_daemon_binary
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from google.protobuf import wrappers_pb2
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2_grpc
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2_grpc
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
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=1.0)
            channel.close()
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("daemon failed to start")


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
    global_store_pb2_grpc.add_GlobalStoreServiceServicer_to_server(servicer, server)
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
        ]
    }
    return json.dumps(index, separators=(",", ":")).encode("utf-8")


def _make_view_spec(*, bias_start: int, bias_length: int) -> store_daemon_pb2.ViewSpec:
    spec = store_daemon_pb2.ViewSpec()
    weights_ops = store_daemon_pb2.TensorViewOps()
    weights_narrow = weights_ops.ops.add().narrow
    weights_narrow.dim = 0
    weights_narrow.start = 0
    weights_narrow.length = 8
    spec.tensors["weights"].CopyFrom(weights_ops)

    bias_ops = store_daemon_pb2.TensorViewOps()
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
    gs_stub: global_store_pb2_grpc.GlobalStoreServiceStub,
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

    _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_a,
        view_bytes=piece_a,
    )
    _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_b,
        view_bytes=piece_b,
    )

    start_resp = stub.StartSealAssembly(
        store_daemon_pb2.StartSealAssemblyRequest(assembly_id=assembly_id)
    )
    wait_resp = stub.WaitOperation(
        store_daemon_pb2.WaitOperationRequest(
            operation_id=start_resp.operation.operation_id,
            timeout_ms=60_000,
        )
    )
    assert wait_resp.operation.status.state == operation_pb2.OPERATION_STATE_SUCCESS
    op_result = store_daemon_pb2.SealAssemblyResult()
    assert wait_resp.operation.status.result.Unpack(op_result) is True
    return op_result.artifact.artifact_id, view_id_a, view_id_b


def test_piece_bootstrap_and_seal(daemon_process, gs_server):
    listen_addr, gs_port, _ = daemon_process
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)

    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = global_store_pb2_grpc.GlobalStoreServiceStub(gs_channel)

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
    _register_piece(
        stub,
        assembly_id=assembly_id,
        canonical_index_bytes=canonical_index_bytes,
        canonical_size_bytes=canonical_size_bytes,
        view_spec=view_spec_b,
        view_bytes=piece_b,
    )

    start_resp = stub.StartSealAssembly(
        store_daemon_pb2.StartSealAssemblyRequest(assembly_id=assembly_id)
    )
    assert start_resp.operation.operation_id

    wait_resp = stub.WaitOperation(
        store_daemon_pb2.WaitOperationRequest(
            operation_id=start_resp.operation.operation_id,
            timeout_ms=60_000,
        )
    )
    assert wait_resp.operation.status.state == operation_pb2.OPERATION_STATE_SUCCESS
    op_result = store_daemon_pb2.SealAssemblyResult()
    assert wait_resp.operation.status.result.Unpack(op_result) is True
    assert op_result.artifact.artifact_id.startswith("mi2:")
    snapshot = store_daemon_pb2.SealAssemblySnapshot()
    assert wait_resp.operation.snapshot.Unpack(snapshot) is True
    assert snapshot.layout_id == put_layout.layout_id
    assert snapshot.assembly_layout_binding_version == bind_layout.binding.binding_version

    seal_resp_2 = stub.SealAssembly(
        store_daemon_pb2.SealAssemblyRequest(
            assembly_id=assembly_id,
            publish_canonical=True,
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
        global_store_pb2.ListArtifactLayoutsRequest(mi2_id=op_result.artifact.artifact_id)
    )
    assert attached.status == global_store_pb2.Status.STATUS_OK
    assert put_layout.layout_id in attached.layout_ids

    replicas_resp = gs_stub.ListReplicasV2(
        global_store_pb2.ListReplicasV2Request(artifact_id=op_result.artifact.artifact_id)
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


def test_post_seal_reuse_views_if_safe(daemon_process_reuse, gs_server):
    listen_addr, gs_port, _ = daemon_process_reuse
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)

    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = global_store_pb2_grpc.GlobalStoreServiceStub(gs_channel)

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
            artifact_id=assembly_id,
            view_id=view_id_a,
            target_device_type=store_daemon_pb2.DEVICE_TYPE_GPU,
            wait_for_completion=True,
        )
    )
    assert (
        mat_resp.status
        == store_daemon_pb2.MATERIALIZE_REPLICA_STATUS_ALLOCATED
    )
    assert mat_resp.artifact_id == sealed_id

    channel.close()
    gs_channel.close()


def test_post_seal_migrate_views(daemon_process_migrate, gs_server):
    listen_addr, gs_port, _ = daemon_process_migrate
    channel = grpc.insecure_channel(listen_addr)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)

    gs_channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
    gs_stub = global_store_pb2_grpc.GlobalStoreServiceStub(gs_channel)

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
