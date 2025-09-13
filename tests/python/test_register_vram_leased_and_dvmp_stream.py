#  Copyright (c) 2025, TensorCast Team.

import os
import sys
from pathlib import Path

import pytest

import torch

from tensorcast.api._config import RegisterArtifactOptions
from tensorcast.api import register_artifact
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.api._register import begin_register_artifact_sdk
from tensorcast._C import get_cuda_memory_handle
from tensorcast.types import CpuPlan, LeasePlan, LeaseSegment, CoalescedHandshake
from tensorcast.proto.daemon.v1 import store_daemon_pb2 as _pb2
from tensorcast.proto.daemon.v1 import store_daemon_pb2_grpc as _pb2_grpc
import subprocess
import time
import grpc

def _start_daemon_binary(listen_addr: str, storage_path: Path) -> subprocess.Popen:
    repo_root = Path(__file__).resolve().parents[2]
    # Allow overriding daemon binary via env for local builds.
    override = os.environ.get("TENSORCAST_DAEMON_BIN")
    if override:
        bin_path = Path(override)
    else:
        # Prefer a locally built Bazel daemon if present (supports FakeCuda build).
        bazel_bin = repo_root / "bazel-bin" / "daemon" / "tensorcast_daemon"
        if bazel_bin.exists() and os.access(bazel_bin, os.X_OK):
            bin_path = bazel_bin
        else:
            # Fallback to packaged binary from the Python wheel
            bin_path = repo_root / "tensorcast" / "bin" / "tensorcast_daemon"
    assert bin_path.exists() and os.access(bin_path, os.X_OK)
    # Ensure libtorch shared libraries are discoverable by the daemon.
    # This helps environments where system loader paths do not include
    # the PyTorch wheel's bundled libraries.
    torch_libdir = Path(torch.__file__).resolve().parent / "lib"
    env = os.environ.copy()
    if torch_libdir.exists():
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{torch_libdir}:{ld_path}" if ld_path else str(torch_libdir)
    # Build unified DaemonConfig YAML and pass via --config
    import tempfile, yaml
    host, port_s = listen_addr.split(":", 1)
    port = int(port_s)
    storage_path.mkdir(parents=True, exist_ok=True)
    log_path = storage_path.parent / "daemon.log"
    cfg = {
        "server": {
            "listen": {"host": host, "port": port},
            "p2p_listen": {"host": host, "port": 9090},
            "storage_path": str(storage_path),
            "num_threads": 2,
            "grpc": {"max_message_size_mb": 128, "tcp_nodelay": True, "so_reuseport": False},
        },
        "engine": {
            "mem_pool_size_bytes": 268435456,
            "chunk_bytes": 8388608,
            "cpu_chunk_size_bytes": 8388608,
            "streaming_buffer_max_concurrent_sessions": 1,
        },
        "communicator": {"enable_rdma": False},
        "observability": {
            "otel": {"enabled": False},
            "logging": {"level": "INFO", "otel_context_enabled": False, "file": str(log_path)},
            "tracing": {"chrome_trace_dir": ""},
        },
        "debug": {"cuda": {"enable_same_process_ipc_fallback": True}},
    }
    with tempfile.NamedTemporaryFile(prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = Path(f.name)
    proc = subprocess.Popen([str(bin_path), f"--config={cfg_path}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
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


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available – skipping VRAM Lease test")


@pytest.mark.timeout(60)
def test_register_cpu_stream_commit(tmp_path: Path):
    listen = "127.0.0.1:50730"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        # Build small CPU tensors, enforce CPU plan by passing device_id
        t1 = torch.zeros((4, 4), dtype=torch.float32)
        t2 = torch.ones((2, 8), dtype=torch.float32)
        state = {"a": t1, "b": t2}

        opts = RegisterArtifactOptions(plan="cpu")
        # device_id enforces CPU input mode for cpu path in SDK
        res = register_artifact(state, options=opts, device_id=0, daemon_address=listen)
        desc = res.descriptor
        assert desc.artifact_id.startswith("mi2:")
        assert desc.total_size > 0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_register_vram_leased_commit(tmp_path: Path):
    _skip_if_no_cuda()
    # Enable same-process IPC fallback for tests (export/open in one process)
    os.environ.setdefault("TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK", "1")

    listen = "127.0.0.1:50731"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))
    try:
        device = torch.device("cuda", 0)
        # Two tensors that share no storage (unique blocks)
        t1 = torch.arange(0, 64, dtype=torch.float32, device=device).reshape(16, 1)
        t2 = torch.zeros((8, 8), dtype=torch.float32, device=device)
        state = {"t1": t1, "t2": t2}

        opts = RegisterArtifactOptions(plan="vram_leased")
        # For lease: do not pass device_id so SDK infers and uses CUDA path
        res = register_artifact(state, options=opts, daemon_address=listen)
        desc = res.descriptor
        assert desc.artifact_id.startswith("mi2:")
        assert desc.total_size > 0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_register_vram_lease_shuffled_segments(tmp_path: Path):
    _skip_if_no_cuda()
    os.environ.setdefault("TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK", "1")

    listen = "127.0.0.1:50736"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        import json, random
        device = torch.device("cuda", 0)
        # Three tensors with disjoint storages
        a = torch.arange(0, 16, dtype=torch.uint8, device=device)
        b = torch.arange(16, 64, dtype=torch.uint8, device=device)
        c = torch.full((32,), 0xAA, dtype=torch.uint8, device=device)
        state = {"a": a, "b": b, "c": c}

        # Build meta and source index
        tensor_meta_index: dict[str, tuple[list[int], list[int], str, int]] = {}
        tensor_source_index: dict[str, tuple[int, int]] = {}
        for name, t in state.items():
            storage = t.untyped_storage()
            tensor_source_index[name] = (int(storage.data_ptr()), int(storage.size()))
            tensor_meta_index[name] = (
                list(map(int, t.shape)),
                list(map(int, t.stride())),
                str(t.dtype),
                int(t.storage_offset()),
            )

        # Compute coalesced layout and total size
        from tensorcast.api._indices import calculate_tensor_device_offsets
        device_offsets, copy_chunks = calculate_tensor_device_offsets(
            tensor_source_index, device
        )
        chunks = list(copy_chunks[device])  # (src_ptr, size, dst_off, stream)
        total_size = max(dst + sz for _, sz, dst, _ in chunks)

        # Build canonical index JSON using destination offsets
        tensor_index_v2: dict[str, tuple[int, int, list[int], list[int], str, int]] = {}
        for name in sorted(tensor_meta_index.keys()):
            shape, stride, dtype, storage_offset = tensor_meta_index[name]
            _, storage_size = tensor_source_index[name]
            dst_off = int(device_offsets[device][name])
            tensor_index_v2[name] = (
                dst_off,
                int(storage_size),
                list(shape),
                list(stride),
                dtype,
                int(storage_offset),
            )
        index_bytes = json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode("utf-8")

        # Begin lease registration via SDK helper (no device_id to force lease path)
        from tensorcast.api._register import begin_register_artifact_sdk
        handle, _hs = begin_register_artifact_sdk(
            device_id=device.index,
            total_size_bytes=total_size,
            ttl_ms=500,
            tensor_index_data=index_bytes,
            plan=LeasePlan(kind="lease", min_tensor_bytes=0, max_tensor_count=16, lease_bytes_limit=0),
            daemon_address=listen,
        )

        # Export CUDA IPC handles for each unique storage and build segments, then shuffle
        def export_ipc(ptr: int) -> bytes:
            return get_cuda_memory_handle(device.index, int(ptr))

        segments: list[LeaseSegment] = []
        for src_ptr, size, dst_off, _s in chunks:
            segments.append(
                LeaseSegment(
                    device_id=device.index,
                    cuda_ipc_handle=export_ipc(int(src_ptr)),
                    base_addr=0,
                    length=int(size),
                    dst_offset=int(dst_off),
                )
            )
        random.shuffle(segments)

        ctl = DaemonCtl(listen)
        ok = ctl.feed_register_artifact_lease_segments(handle.registration_id, segments)
        assert ok
        commit = handle.commit()
        assert commit and commit.descriptor.artifact_id.startswith("mi2:")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_dvmp_ttl_expiry(tmp_path: Path):
    """CPU commit should fail if TTL expires before commit."""
    listen = "127.0.0.1:50732"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        # Build minimal v2 index JSON for a single tensor
        import json
        import numpy as np
        ttl = 50  # ms
        shape = [2, 2]
        stride = [2, 1]
        size_bytes = 4 * 4  # float32 * 4
        tensor_index_v2 = {"x": [0, size_bytes, shape, stride, "torch.float32", 0]}
        index_bytes = json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode("utf-8")

        # Begin with cpu plan and TTL using SDK handle
        handle, _hs = begin_register_artifact_sdk(
            device_id=0,
            total_size_bytes=size_bytes,
            ttl_ms=ttl,
            tensor_index_data=index_bytes,
            plan=CpuPlan(kind="cpu", preferred_channel=2, ring_bytes=0),
            daemon_address=listen,
        )

        # Feed cpu chunk
        buf = (np.zeros((2, 2), dtype=np.float32)).tobytes()
        ctl = DaemonCtl(listen)
        ok = ctl.feed_register_artifact_cpu_stream_data(handle.registration_id, buf, offset=0)
        assert ok

        # Sleep beyond TTL to force expiry
        time.sleep((ttl * 3) / 1000.0)

        # Commit should fail due to expiry (DEADLINE_EXCEEDED)
        with pytest.raises(Exception):
            _ = handle.commit()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_cpu_ttl_keepalive_success(tmp_path: Path):
    """CPU commit should succeed even after TTL, if keepalive is sent periodically."""
    listen = "127.0.0.1:50733"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        # Build minimal v2 index JSON for a single tensor
        import json
        shape = [2, 2]
        stride = [2, 1]
        size_bytes = 4 * 4  # float32 * 4
        tensor_index_v2 = {"x": [0, size_bytes, shape, stride, "torch.float32", 0]}
        index_bytes = json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode("utf-8")

        # Begin with cpu plan and TTL using SDK handle (auto-keepalive)
        ttl = 100  # ms
        handle, _hs = begin_register_artifact_sdk(
            device_id=0,
            total_size_bytes=size_bytes,
            ttl_ms=ttl,
            tensor_index_data=index_bytes,
            plan=CpuPlan(kind="cpu", preferred_channel=2, ring_bytes=0),
            daemon_address=listen,
        )
        reg_id = handle.registration_id

        # Start keepalive thread
        import threading, time
        stop = threading.Event()

        def _ka():
            epoch = 0
            while not stop.wait(ttl / 2000.0):  # ttl/2 seconds
                ctl.keep_alive_registered_artifact(reg_id, ttl, epoch)
                epoch += 1

        th = threading.Thread(target=_ka, daemon=True)
        th.start()

        # Feed cpu chunk
        import numpy as np
        buf = (np.zeros((2, 2), dtype=np.float32)).tobytes()
        ctl = DaemonCtl(listen)
        ok = ctl.feed_register_artifact_cpu_stream_data(handle.registration_id, buf, offset=0)
        assert ok

        # Sleep beyond initial TTL so that commit would have failed without keepalive
        time.sleep((ttl * 3) / 1000.0)

        # Commit should succeed due to keepalive thread
        commit = handle.commit()
        assert commit and commit.descriptor.artifact_id.startswith("mi2:")
        stop.set()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_ttl_expiry_on_feed_paths(tmp_path: Path):
    """TTL expiry should fail fast in Feed for CPU and Lease."""
    # Start daemon
    listen = "127.0.0.1:50737"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        # CPU: begin with very short TTL, then attempt to feed after expiry
        import json
        size_bytes = 64
        tensor_index_v2 = {"x": [0, size_bytes, [2, 32], [32, 1], "torch.uint8", 0]}
        index_bytes = json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode("utf-8")
        from tensorcast.api._register import begin_register_artifact_sdk
        handle, _ = begin_register_artifact_sdk(
            device_id=0,
            total_size_bytes=size_bytes,
            ttl_ms=50,
            tensor_index_data=index_bytes,
            plan=CpuPlan(kind="cpu", preferred_channel=2, ring_bytes=0),
            daemon_address=listen,
        )
        time.sleep(0.08)
        ctl = DaemonCtl(listen)
        ok = ctl.feed_register_artifact_cpu_stream_data(handle.registration_id, bytes([0] * size_bytes), offset=0)
        assert not ok, "CPU feed should fail after TTL expiry"

        # Lease path: if CUDA available
        if torch.cuda.is_available():
            os.environ.setdefault("TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK", "1")
            device = torch.device("cuda", 0)
            t = torch.full((64,), 0x5A, dtype=torch.uint8, device=device)
            storage = t.untyped_storage()
            ptr = int(storage.data_ptr()); sz = int(storage.size())
            tensor_index_v2 = {"x": [0, sz, [64], [1], "torch.uint8", 0]}
            index_bytes = json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode("utf-8")
            handle2, _ = begin_register_artifact_sdk(
                device_id=device.index,
                total_size_bytes=sz,
                ttl_ms=50,
                tensor_index_data=index_bytes,
                plan=LeasePlan(kind="lease", min_tensor_bytes=0, max_tensor_count=8, lease_bytes_limit=0),
                daemon_address=listen,
            )
            time.sleep(0.08)
            # Export IPC and attempt feed
            def export_ipc(p: int) -> bytes:
                return get_cuda_memory_handle(device.index, int(p))
            seg = LeaseSegment(device_id=device.index, cuda_ipc_handle=export_ipc(ptr), base_addr=0, length=sz, dst_offset=0)
            ok = ctl.feed_register_artifact_lease_segments(handle2.registration_id, [seg])
            assert not ok, "Lease feed should fail after TTL expiry"
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
