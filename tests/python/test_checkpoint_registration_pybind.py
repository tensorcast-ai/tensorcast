#  Copyright (c) 2025, TensorCast Team.

import os
import subprocess
import time
from pathlib import Path

import grpc
import pytest

import tensorcast._C as _C
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.daemon.v1 import store_daemon_pb2 as _pb2
from tensorcast.proto.daemon.v1 import store_daemon_pb2_grpc as _pb2_grpc
from tensorcast.types import CoalescedPlan, CoalescedHandshake


def _skip_if_no_cuda() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available – skipping memory registration tests.")
    except Exception:  # noqa: BLE001
        pytest.skip("torch not available – skipping CUDA-dependent tests.")


def _start_daemon_binary(listen_addr: str, storage_path: Path) -> subprocess.Popen:
    bin_path = Path(__file__).resolve().parents[2] / "tensorcast" / "bin" / "tensorcast_daemon"
    assert bin_path.exists() and os.access(bin_path, os.X_OK)
    args = [
        str(bin_path),
        f"--listen_addr={listen_addr}",
        f"--storage_path={str(storage_path)}",
        "--p2p_port=9090",
        "--mem_pool_size=268435456",  # 256MB
        "--chunk_size=8388608",       # 8MB
        "--io_threads=2",
        "--enable_p2p_access=true",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for readiness
    deadline = time.time() + 10
    ok = False
    while time.time() < deadline:
        try:
            chan = grpc.insecure_channel(listen_addr)
            stub = _pb2_grpc.StoreDaemonServiceStub(chan)
            stub.GetServerConfig(_pb2.GetServerConfigRequest(), timeout=1.0)
            chan.close()
            ok = True
            break
        except Exception:
            time.sleep(0.2)
    if not ok:
        try:
            out, err = proc.communicate(timeout=1)
        except Exception:
            out = b""; err = b""
        raise RuntimeError(f"daemon failed to start: out={out.decode()} err={err.decode()}")
    return proc


@pytest.mark.timeout(60)
def test_sdk_begin_commit_and_ipc_map(tmp_path: Path):
    _skip_if_no_cuda()
    # Same-process IPC fallback for tests
    os.environ.setdefault("TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK", "1")
    listen = "127.0.0.1:50740"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.skip(f"daemon not startable in this environment: {e}")
    try:
        ctl = DaemonCtl(listen)
        size = 1 * 1024 * 1024
        index_bytes = b"{}"  # minimal canonical index JSON
        begin = ctl.begin_register_artifact(
            device_id=0,
            total_size_bytes=size,
            ttl_ms=0,
            tensor_index_data=index_bytes,
            encoding="json",
            schema_version="v2",
            plan=CoalescedPlan(kind="coalesced", max_inflight_bytes=1 << 20),
        )
        assert begin.registration_id
        assert begin.device_id == 0
        assert begin.total_size == size
        assert isinstance(begin.handshake, CoalescedHandshake)

        # Map and unmap IPC handle
        ptr = _C.get_cuda_memory_ptr(0, begin.handshake.daemon_ipc_handle)
        assert isinstance(ptr, int) and ptr != 0
        assert _C.close_cuda_memory_handle(0, ptr) is True

        desc = ctl.commit_registered_artifact(begin.registration_id)
        assert desc.artifact_id.startswith("mi2:")
        assert desc.total_size == size
    finally:
        try:
            proc.terminate(); proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_sdk_begin_abort_then_commit_fails(tmp_path: Path):
    listen = "127.0.0.1:50741"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.skip(f"daemon not startable in this environment: {e}")
    try:
        ctl = DaemonCtl(listen)
        begin = ctl.begin_register_artifact(
            device_id=0,
            total_size_bytes=2 * 1024 * 1024,
            ttl_ms=0,
            tensor_index_data=b"{}",
            encoding="json",
            schema_version="v2",
            plan=CoalescedPlan(kind="coalesced"),
        )
        reg = begin.registration_id
        assert ctl.abort_registered_artifact(reg)
        with pytest.raises((KeyError, RuntimeError)):
            _ = ctl.commit_registered_artifact(reg)
    finally:
        try:
            proc.terminate(); proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_sdk_ttl_expiry(tmp_path: Path):
    listen = "127.0.0.1:50742"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.skip(f"daemon not startable in this environment: {e}")
    try:
        ctl = DaemonCtl(listen)
        ttl_ms = 5
        begin = ctl.begin_register_artifact(
            device_id=0,
            total_size_bytes=1 * 1024 * 1024,
            ttl_ms=ttl_ms,
            tensor_index_data=b"{}",
            encoding="json",
            schema_version="v2",
            plan=CoalescedPlan(kind="coalesced"),
        )
        time.sleep(0.02)
        with pytest.raises(TimeoutError):
            _ = ctl.commit_registered_artifact(begin.registration_id)  # TTL expired
    finally:
        try:
            proc.terminate(); proc.wait(timeout=3)
        except Exception:
            pass


def test_sdk_invalid_args_client_side(monkeypatch):
    # Avoid strict OTel requirement for this client-side validation test
    monkeypatch.setattr("tensorcast.daemon_ctl.ensure_client_otel", lambda *a, **k: None)
    ctl = DaemonCtl("127.0.0.1:9")  # address not used for client-side validation
    # total_size_bytes == 0
    with pytest.raises(ValueError):
        ctl.begin_register_artifact(
            device_id=0,
            total_size_bytes=0,
            tensor_index_key="a",
            plan=CoalescedPlan(kind="coalesced"),
        )
    # missing index
    with pytest.raises(ValueError):
        ctl.begin_register_artifact(
            device_id=0,
            total_size_bytes=1024,
            plan=CoalescedPlan(kind="coalesced"),
        )
    # negative device_id
    with pytest.raises(ValueError):
        ctl.begin_register_artifact(
            device_id=-1,
            total_size_bytes=1024,
            tensor_index_key="a",
            plan=CoalescedPlan(kind="coalesced"),
        )
