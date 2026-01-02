#  Copyright (c) 2025-2026, TensorCast Team.

import time
import os
from pathlib import Path

import pytest
import torch

from tensorcast import startup
from tensorcast.api import PlanType, RegisterArtifactOptions, Store
from tensorcast.api.store import RegisteredArtifact as StoreRegisteredArtifact
from tests.python.utils.daemon import start_daemon_binary


def _start_daemon_binary(listen_addr: str, storage_path: Path):
    return start_daemon_binary(listen_addr, storage_path, config_mode="yaml", enable_same_process_ipc_fallback=True)


def _skip_if_no_cuda() -> None:
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        pytest.skip("Fake CUDA backend enabled; skipping real-CUDA LIP helper test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available – skipping LIP helper test")


@pytest.mark.timeout(60)
def test_register_artifact_lease_in_place_helper(tmp_path: Path):
    _skip_if_no_cuda()
    listen = "127.0.0.1:50741"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))
    try:
        startup.init(mode="connect", address=listen)
        store = Store(listen)
        try:
            dev = torch.device("cuda", 0)
            a = torch.arange(0, 32, dtype=torch.uint8, device=dev)
            b = torch.full((64,), 0x77, dtype=torch.uint8, device=dev)
            state = {"a": a, "b": b}
            opts = RegisterArtifactOptions(plan=PlanType.VRAM_LEASED, lease_in_place=True)
            res = store.register(state, options=opts, ttl_ms=2000)
            assert isinstance(res, StoreRegisteredArtifact)
            assert res.registration_result is not None
            desc = res.registration_result.descriptor
            lease = res.registration_result.lease
            assert desc.artifact_id.startswith("mi2:")
            # Keepalive thread should be running; sleep to allow a keepalive tick
            time.sleep(0.5)

            assert lease is not None
            # Context revoke
            with lease:
                pass
        finally:
            store.close()
            startup.shutdown()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
