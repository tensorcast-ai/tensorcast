#  Copyright (c) 2025, TensorCast Team.

import contextlib
import time
import uuid

import grpc
from pathlib import Path
import pytest

from tensorcast.proto import store_daemon_pb2


def _ensure_minimal_model_files(storage_root: Path, artifact_id: str, size_bytes: int) -> None:
    artifact_dir = storage_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Remove any existing partition files to avoid size mismatches
    for p in artifact_dir.glob("tensor.data*"):
        try:
            p.unlink()
        except Exception:
            pass
    # Create a single-file artifact matching the expected size
    data_file = artifact_dir / "tensor.data"
    with data_file.open("wb") as f:
        f.truncate(size_bytes)


@pytest.fixture
def servicer(tmp_path):
    """Create a StoreDaemonServicer with minimal config for memory registration tests."""
    from tensorcast.store_daemon.config import StoreDaemonConfig, ServerConfig, NetworkConfig
    from tensorcast.store_daemon.servicer import StoreDaemonServicer
    from pydantic import ByteSize

    cfg = StoreDaemonConfig(
        server=ServerConfig(
            storage_path=tmp_path / "storage",
            mem_pool_size=ByteSize(1_000_000_000),
            num_threads=2,
            chunk_size=ByteSize(1 << 20),
            enable_p2p_access=False,
            enable_p2p_engine=False,
        ),
        network=NetworkConfig(),
        global_store_address=None,
    )

    s = StoreDaemonServicer(config=cfg)
    s.grpc_channel = None
    try:
        yield s
    finally:
        # Graceful shutdown
        from tests.python.conftest import cleanup_background_threads
        if s.lifecycle_worker:
            s.lifecycle_worker.stop()
        if s.connection_manager is not None:
            with contextlib.suppress(Exception):
                s.connection_manager.stop()
        cleanup_background_threads(s)


class _Ctx:
    def __init__(self):
        self.code = None
        self.details = None
    def set_code(self, code):
        self.code = code
    def set_details(self, details):
        self.details = details


def test_begin_commit_abort_memory_registration(servicer):
    ctx = _Ctx()

    # Ensure minimal artifact directory/files exist to satisfy Artifact::create checks
    _ensure_minimal_model_files(Path(servicer.storage_path), "py_mem_artifact", 1 * 1024 * 1024)

    # Begin with index data path to ensure deterministic index hashing
    idx = store_daemon_pb2.TensorIndexData(
        data=b"{}",
        schema_version="v2",
        encoding="json",
    )
    req = store_daemon_pb2.BeginRegisterArtifactRequest(
        artifact_id="py_mem_artifact",
        device_id=0,
        total_size=1 * 1024 * 1024,
        enable_p2p=False,
        tensor_index_data=idx,
    )
    resp = servicer.BeginRegisterArtifact(req, ctx)
    assert ctx.code is None
    assert resp.registration_id
    assert resp.daemon_ipc_handle != b""

    # Commit
    c_resp = servicer.CommitRegisteredArtifact(
        store_daemon_pb2.CommitRegisteredArtifactRequest(
            registration_id=resp.registration_id
        ),
        ctx,
    )
    assert ctx.code is None
    assert c_resp.registration_id == resp.registration_id
    # Enforce content addressing (mi2) per RFC-0007
    assert c_resp.artifact_id.startswith("mi2:")
    assert c_resp.descriptor.artifact_id == c_resp.artifact_id
    # Multihashes must be populated under content-addressing
    assert c_resp.descriptor.index_multihash != ""
    assert c_resp.descriptor.data_multihash != ""
    assert c_resp.descriptor.total_size == c_resp.size
    assert c_resp.device_id == 0
    assert c_resp.size == 1 * 1024 * 1024

    # Abort after commit should return ok=False or INTERNAL
    a_resp = servicer.AbortRegisteredArtifact(
        store_daemon_pb2.AbortRegisteredArtifactRequest(
            registration_id=resp.registration_id
        ),
        ctx,
    )
    # Either INTERNAL set_code or ok=False is acceptable depending on impl path
    assert (ctx.code is not None) or (a_resp.ok is False)

    # Double commit should surface NOT_FOUND/INTERNAL
    ctx2 = _Ctx()
    servicer.CommitRegisteredArtifact(
        store_daemon_pb2.CommitRegisteredArtifactRequest(
            registration_id=resp.registration_id
        ),
        ctx2,
    )
    assert ctx2.code in (grpc.StatusCode.NOT_FOUND, grpc.StatusCode.INTERNAL)


def test_begin_with_index_data_and_ttl(servicer):
    ctx = _Ctx()

    # Ensure minimal artifact directory/files exist to satisfy Artifact::create checks
    _ensure_minimal_model_files(Path(servicer.storage_path), "py_mem_ttl", 1024)

    # Provide index data oneof
    idx = store_daemon_pb2.TensorIndexData(
        data=b"{}",
        schema_version="v2",
        encoding="json",
    )
    req = store_daemon_pb2.BeginRegisterArtifactRequest(
        artifact_id="py_mem_ttl",
        device_id=0,
        total_size=1024,
        enable_p2p=False,
        ttl_ms=5,
        tensor_index_data=idx,
    )
    resp = servicer.BeginRegisterArtifact(req, ctx)
    assert ctx.code is None

    # Wait past TTL and attempt commit
    time.sleep(0.02)
    c_resp = servicer.CommitRegisteredArtifact(
        store_daemon_pb2.CommitRegisteredArtifactRequest(
            registration_id=resp.registration_id
        ),
        ctx,
    )
    assert ctx.code in (grpc.StatusCode.INTERNAL, grpc.StatusCode.DEADLINE_EXCEEDED)

    # Abort unknown id should return NOT_FOUND/INTERNAL
    ctx2 = _Ctx()
    a_resp = servicer.AbortRegisteredArtifact(
        store_daemon_pb2.AbortRegisteredArtifactRequest(
            registration_id=str(uuid.uuid4())
        ),
        ctx2,
    )
    assert (ctx2.code in (grpc.StatusCode.NOT_FOUND, grpc.StatusCode.INTERNAL)) or (a_resp.ok is False)


def test_begin_invalid_args(servicer):
    ctx = _Ctx()

    # Missing key/data => handled by service path; here pass empty key with size 0 to trigger errors downstream
    req = store_daemon_pb2.BeginRegisterArtifactRequest(
        artifact_id="",
        device_id=-1,
        total_size=0,
        enable_p2p=False,
        tensor_index_key="",
    )
    resp = servicer.BeginRegisterArtifact(req, ctx)
    assert ctx.code in (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.INTERNAL)


