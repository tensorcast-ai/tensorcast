#  Copyright (c) 2025, TensorCast Team.

import types
import sys

import pytest

# Provide a stub for tensorcast.cuda before importing Store to avoid hard dependency
sys.modules.setdefault(
    "tensorcast.cuda",
    types.SimpleNamespace(get_cuda_memory_handle=lambda device_id, base_ptr: b"fake-handle"),
)

from tensorcast.api.store import Store
from tensorcast.types import VramRegionHandle
from tensorcast.api import _region_cache as region_cache


@pytest.fixture
def store(monkeypatch) -> Store:
    s = Store("dummy:0")

    # Fake CUDA handle exporter (redundant with sys.modules stub but keeps isolation per test)
    monkeypatch.setattr("tensorcast.api.store.get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle")

    # Fake daemon client with register/unregister methods
    client = types.SimpleNamespace()

    def _register_vram_region(*, device_id: int, size_bytes: int, ttl_ms: int, cuda_ipc_handle: bytes, region_name: str | None = None):
        assert device_id >= 0
        assert size_bytes > 0
        assert ttl_ms > 0
        assert cuda_ipc_handle == b"fake-handle"
        return VramRegionHandle(region_id="region:test123", ttl_ms=ttl_ms)

    def _unregister_vram_region(region_id: str, *, force: bool | None = None) -> bool:
        assert region_id == "region:test123"
        return True

    client.register_vram_region = _register_vram_region
    client.unregister_vram_region = _unregister_vram_region

    monkeypatch.setattr(s, "_ensure_client", lambda: client)
    return s


def test_register_and_unregister_vram_region_updates_cache(store: Store):
    device_id = 0
    base_ptr = 0x1000
    size_bytes = 4096
    ttl_ms = 1000

    # Register a region
    handle = store.register_vram_region(
        device_id=device_id, base_ptr=base_ptr, size_bytes=size_bytes, ttl_ms=ttl_ms, name="test"
    )
    assert handle.region_id == "region:test123"

    # Cache should contain the region
    dev_regions = region_cache.get_regions_for_device(device_id)
    assert any(r.region_id == "region:test123" and r.base_ptr == base_ptr and r.size_bytes == size_bytes for r in dev_regions)

    # Unregister should drop from cache
    ok = store.unregister_vram_region("region:test123")
    assert ok is True
    dev_regions2 = region_cache.get_regions_for_device(device_id)
    assert all(r.region_id != "region:test123" for r in dev_regions2)


