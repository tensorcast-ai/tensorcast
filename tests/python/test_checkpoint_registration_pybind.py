#  Copyright (c) 2025, TensorCast Team.

import os
from pathlib import Path
from typing import TypedDict, Optional

import pytest


import scstore._store_engine as _cs
import scstore._C as _C


def _ensure_minimal_model_files(storage_root: Path, artifact_id: str, size_bytes: int) -> None:
    artifact_dir = storage_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_file = artifact_dir / "tensor.data_0"
    with data_file.open("wb") as f:
        f.truncate(size_bytes)


def _skip_if_no_cuda() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available – skipping memory registration tests.")
    except Exception:  # noqa: BLE001
        pytest.skip("torch not available – skipping CUDA-dependent tests.")


class _Reg(TypedDict, total=False):
    artifact_id: str
    tensor_index_key: str
    tensor_index_data: Optional[str]
    schema_version: str
    encoding: str
    device_id: int
    total_size_bytes: int
    enable_p2p: bool
    ttl_ms: int


def test_pybind_begin_commit_and_ipc_map(tmp_path: Path):
    _skip_if_no_cuda()

    storage_root = tmp_path / "models"
    # Enable same-process CUDA IPC fallback for unit tests. Real CUDA IPC
    # requires separate processes, but our tests run export+open in one.
    os.environ.setdefault("SCSTORE_ENABLE_IPC_SAME_PROCESS_FALLBACK", "1")
    cs = _cs.create_store_engine({
        "storage_path": str(storage_root),
        "memory_pool_size": 4 * 1024 * 1024,
        "num_thread": 1,
        "chunk_size": 1 * 1024 * 1024,
        "enable_p2p_engine": False,
        "enable_rdma": False,
        "pinned_memory_timeout_ms": 0,
    })

    reg: _Reg = {
        "artifact_id": "pybind_mem_artifact",
        # RFC-0007: tensor_index_key must be a 32-byte sha256 hex (64 hex chars)
        "tensor_index_key": "0" * 64,
        "device_id": 0,
        "total_size_bytes": 1 * 1024 * 1024,
        "enable_p2p": False,
    }

    _ensure_minimal_model_files(storage_root, reg["artifact_id"], reg["total_size_bytes"])  # satisfy Artifact::create

    out = cs.begin_register_artifact(reg)

    assert out["registration_id"]
    assert out["device_id"] == 0
    assert out["size_bytes"] == 1 * 1024 * 1024
    assert out["daemon_ipc_handle"] != b""

    # Map and immediately unmap the daemon IPC handle to validate plumbing
    ptr = _C.get_cuda_memory_ptr(out["device_id"], out["daemon_ipc_handle"])  # returns int
    assert isinstance(ptr, int) and ptr != 0
    assert _C.close_cuda_memory_handle(out["device_id"], ptr) is True

    res = cs.commit_registered_artifact(out["registration_id"])
    assert res["registration_id"] == out["registration_id"]
    # RFC-0007: commit returns content-addressed artifact_id (mi2:...)
    assert isinstance(res["artifact_id"], str) and res["artifact_id"].startswith("mi2:")
    assert res["device_id"] == 0

    # Query GPU pointer via ReplicaKey
    dev = _cs.DeviceKey()
    dev.type = _cs.DeviceType.GPU
    dev.ordinal = 0
    dev.uuid = ""
    key = _cs.ReplicaKey()
    key.artifact_id = "pybind_mem_artifact"
    key.device = dev
    key.replica = 0

    # Query using the content-addressed artifact_id returned by commit
    key.artifact_id = res["artifact_id"]
    gpu_ptr = cs.get_replica_gpu_ptr(key)
    assert isinstance(gpu_ptr, int) and gpu_ptr != 0


def test_pybind_begin_abort_then_commit_fails(tmp_path: Path):
    _skip_if_no_cuda()

    storage_root = tmp_path / "models"
    cs = _cs.create_store_engine({
        "storage_path": str(storage_root),
        "memory_pool_size": 4 * 1024 * 1024,
        "num_thread": 1,
        "chunk_size": 1 * 1024 * 1024,
        "enable_p2p_engine": False,
        "enable_rdma": False,
        "pinned_memory_timeout_ms": 0,
    })

    reg: _Reg = {
        "artifact_id": "pybind_abort_artifact",
        "tensor_index_key": "deadbeef",
        "device_id": 0,
        "total_size_bytes": 2 * 1024 * 1024,
        "enable_p2p": False,
    }

    _ensure_minimal_model_files(storage_root, reg["artifact_id"], reg["total_size_bytes"])

    out = cs.begin_register_artifact(reg)
    assert out["registration_id"]

    assert cs.abort_registered_artifact(out["registration_id"]) is True

    with pytest.raises(RuntimeError):
        cs.commit_registered_artifact(out["registration_id"])  # should raise NotFound


def test_pybind_ttl_expiry(tmp_path: Path):
    _skip_if_no_cuda()

    storage_root = tmp_path / "models"
    cs = _cs.create_store_engine({
        "storage_path": str(storage_root),
        "memory_pool_size": 4 * 1024 * 1024,
        "num_thread": 1,
        "chunk_size": 1 * 1024 * 1024,
        "enable_p2p_engine": False,
        "enable_rdma": False,
        "pinned_memory_timeout_ms": 0,
    })

    reg: _Reg = {
        "artifact_id": "pybind_ttl_artifact",
        "tensor_index_key": "0123",
        "device_id": 0,
        "total_size_bytes": 1 * 1024 * 1024,
        "enable_p2p": False,
        "ttl_ms": 5,
    }

    _ensure_minimal_model_files(storage_root, reg["artifact_id"], reg["total_size_bytes"])

    out = cs.begin_register_artifact(reg)
    assert out["registration_id"]

    import time
    time.sleep(0.02)

    with pytest.raises(RuntimeError):
        cs.commit_registered_artifact(out["registration_id"])  # should raise DeadlineExceeded


def test_pybind_invalid_args(tmp_path: Path):
    _skip_if_no_cuda()

    storage_root = tmp_path / "models"
    cs = _cs.create_store_engine({
        "storage_path": str(storage_root),
        "memory_pool_size": 4 * 1024 * 1024,
        "num_thread": 1,
        "chunk_size": 1 * 1024 * 1024,
        "enable_p2p_engine": False,
        "enable_rdma": False,
        "pinned_memory_timeout_ms": 0,
    })

    # total_size_bytes == 0
    with pytest.raises(RuntimeError):
        cs.begin_register_artifact({
            "artifact_id": "m1",
            "tensor_index_key": "a",
            "device_id": 0,
            "total_size_bytes": 0,
        })

    # missing tensor_index_key
    with pytest.raises(RuntimeError):
        cs.begin_register_artifact({
            "artifact_id": "m2",
            "tensor_index_key": "",
            "device_id": 0,
            "total_size_bytes": 1024,
        })

    # negative device_id
    with pytest.raises(RuntimeError):
        cs.begin_register_artifact({
            "artifact_id": "m3",
            "tensor_index_key": "a",
            "device_id": -1,
            "total_size_bytes": 1024,
        })


