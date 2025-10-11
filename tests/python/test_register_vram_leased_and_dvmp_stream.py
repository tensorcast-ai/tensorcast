#  Copyright (c) 2025, TensorCast Team.

import json
import random
import subprocess
import time
from pathlib import Path

import pytest
import torch

from tensorcast import startup
from tensorcast._C import get_cuda_memory_handle
from tensorcast.api import PlanType, RegisterArtifactOptions, Store
from tensorcast.api._register import begin_register_artifact_sdk
from tensorcast.api.store import RegisteredArtifact as StoreRegisteredArtifact
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.types import LeasePlan, LeaseSegment
from tests.python.utils.daemon import start_daemon_binary


def _start_daemon_binary(listen_addr: str, storage_path: Path) -> subprocess.Popen:
    return start_daemon_binary(listen_addr, storage_path, config_mode="yaml")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available – skipping VRAM Lease test")


@pytest.mark.timeout(60)
def test_register_vram_leased_commit(tmp_path: Path):
    _skip_if_no_cuda()

    listen = "127.0.0.1:50731"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))
    try:
        startup.init(address=listen)
        store = Store(listen)
        try:
            device = torch.device("cuda", 0)
            # Two tensors that share no storage (unique blocks)
            # Ensure shapes and element counts are consistent
            t1 = torch.arange(0, 64, dtype=torch.float32, device=device).reshape(8, 8)
            t2 = torch.zeros((8, 8), dtype=torch.float32, device=device)
            state = {"t1": t1, "t2": t2}

            opts = RegisterArtifactOptions(plan=PlanType.VRAM_LEASED, lease_in_place=True)
            # For lease: do not pass device_id so SDK infers and uses CUDA path
            res = store.register(state, options=opts)
            assert isinstance(res, StoreRegisteredArtifact)
            assert res.registration_result is not None
            desc = res.registration_result.descriptor
            assert desc.artifact_id.startswith("mi2:")
            assert desc.total_size > 0
        finally:
            store.close()
            startup.shutdown()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_register_vram_lease_shuffled_segments(tmp_path: Path):
    _skip_if_no_cuda()

    listen = "127.0.0.1:50736"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        startup.init(address=listen)
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
            plan=LeasePlan(kind="lease", min_tensor_bytes=0, max_tensor_count=16, lease_bytes_limit=0, in_place=True),
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
        startup.shutdown()
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_ttl_expiry_on_lease_feed_path(tmp_path: Path):
    """TTL expiry should fail fast in Lease feed."""
    _skip_if_no_cuda()
    # Start daemon
    listen = "127.0.0.1:50737"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        startup.init(address=listen)
        # Lease path only
        import json
        device = torch.device("cuda", 0)
        t = torch.full((64,), 0x5A, dtype=torch.uint8, device=device)
        storage = t.untyped_storage()
        ptr = int(storage.data_ptr())
        sz = int(storage.size())
        tensor_index_v2 = {"x": [0, sz, [64], [1], "torch.uint8", 0]}
        index_bytes = json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode("utf-8")
        handle2, _ = begin_register_artifact_sdk(
            device_id=device.index,
            total_size_bytes=sz,
            ttl_ms=50,
            tensor_index_data=index_bytes,
            plan=LeasePlan(kind="lease", min_tensor_bytes=0, max_tensor_count=8, lease_bytes_limit=0, in_place=True),
        )
        time.sleep(0.08)
        ctl = DaemonCtl(listen)
        # Export IPC and attempt feed
        def export_ipc(p: int) -> bytes:
            return get_cuda_memory_handle(device.index, int(p))
        seg = LeaseSegment(device_id=device.index, cuda_ipc_handle=export_ipc(ptr), base_addr=0, length=sz, dst_offset=0)
        ok = ctl.feed_register_artifact_lease_segments(handle2.registration_id, [seg])
        assert not ok, "Lease feed should fail after TTL expiry"
    finally:
        startup.shutdown()
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
