#  Copyright (c) 2025, TensorCast Team.

import random
import subprocess
import time
from pathlib import Path

import pytest
import torch

from concurrent.futures import CancelledError

from tensorcast import ArtifactError, startup
from tensorcast.api import PlanType, RegisterArtifactOptions, Store
from tensorcast.api.store import RegisteredArtifact as StoreRegisteredArtifact
from tensorcast.types import LeaseSegment
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
def test_register_vram_lease_shuffled_segments(tmp_path: Path, monkeypatch):
    _skip_if_no_cuda()

    listen = "127.0.0.1:50736"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))

    try:
        startup.init(address=listen)
        store = Store(listen)
        try:
            device = torch.device("cuda", 0)
            state = {
                "a": torch.arange(0, 16, dtype=torch.uint8, device=device),
                "b": torch.arange(16, 64, dtype=torch.uint8, device=device),
                "c": torch.full((32,), 0xAA, dtype=torch.uint8, device=device),
            }

            from tensorcast.api import _register as register_mod

            original_upload = register_mod._LeaseUploader.upload

            def shuffled_upload(
                self,
                *,
                artifact,
                ctx,
                layout,
                handle,
                handshake,
                cancel_event=None,
            ):
                def _export_cuda_ipc_handle(ptr: int) -> bytes:
                    return register_mod.get_cuda_memory_handle(ctx.device_id, int(ptr))

                segments: list[LeaseSegment] = []
                chunks = list(layout.unique_chunks)
                random.shuffle(chunks)
                for src_ptr, size, dst_off, _ in chunks:
                    if cancel_event and cancel_event.is_set():
                        raise CancelledError
                    handle_bytes = _export_cuda_ipc_handle(int(src_ptr))
                    segments.append(
                        register_mod.LeaseSegment(
                            device_id=int(ctx.device_id),
                            cuda_ipc_handle=handle_bytes,
                            base_addr=0,
                            length=int(size),
                            dst_offset=int(dst_off),
                        )
                    )
                if cancel_event and cancel_event.is_set():
                    raise CancelledError
                ok = handle.client.feed_register_artifact_lease_segments(
                    handle.registration_id, segments
                )
                if not ok:
                    raise register_mod.FeedFailed(
                        "Lease segments feed failed (shuffled)"
                    )
                return artifact

            monkeypatch.setattr(
                register_mod._LeaseUploader,
                "upload",
                shuffled_upload,
                raising=False,
            )

            opts = RegisterArtifactOptions(
                plan=PlanType.VRAM_LEASED,
                lease_in_place=True,
                min_tensor_bytes=0,
                max_tensor_count=16,
                lease_bytes_limit=0,
            )
            res = store.register(state, options=opts)
            assert isinstance(res, StoreRegisteredArtifact)
            assert res.registration_result is not None
            assert res.registration_result.descriptor.artifact_id.startswith("mi2:")
        finally:
            store.close()
    finally:
        startup.shutdown()
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.timeout(60)
def test_ttl_expiry_on_lease_feed_path(tmp_path: Path, monkeypatch):
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
        store = Store(listen)
        try:
            device = torch.device("cuda", 0)
            state = {
                "x": torch.full((64,), 0x5A, dtype=torch.uint8, device=device),
            }

            from tensorcast.api import _register as register_mod

            original_upload = register_mod._LeaseUploader.upload

            def slow_upload(
                self,
                *,
                artifact,
                ctx,
                layout,
                handle,
                handshake,
                cancel_event=None,
            ):
                time.sleep(0.08)
                return original_upload(
                    self,
                    artifact=artifact,
                    ctx=ctx,
                    layout=layout,
                    handle=handle,
                    handshake=handshake,
                    cancel_event=cancel_event,
                )

            monkeypatch.setattr(
                register_mod._LeaseUploader,
                "upload",
                slow_upload,
                raising=False,
            )

            opts = RegisterArtifactOptions(
                plan=PlanType.VRAM_LEASED,
                lease_in_place=True,
                min_tensor_bytes=0,
                max_tensor_count=8,
                lease_bytes_limit=0,
            )
            with pytest.raises(ArtifactError) as exc_info:
                store.register(state, options=opts, ttl_ms=50)
            assert exc_info.value.status_code == "FAILED_PRECONDITION"
        finally:
            store.close()
    finally:
        startup.shutdown()
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
