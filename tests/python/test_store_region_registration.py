#  Copyright (c) 2025-2026, TensorCast Team.

import sys
import types

import pytest

# Provide a stub for tensorcast.cuda before importing Store to avoid hard dependency
sys.modules.setdefault(
    "tensorcast.cuda",
    types.SimpleNamespace(
        get_cuda_memory_handle=lambda device_id, base_ptr: b"fake-handle",
        get_cuda_memory_handle_with_offset=lambda device_id, base_ptr: (b"fake-handle", 0),
    ),
)

from tensorcast.api.store import Store, StoreOptions
from tensorcast.types import (
    HostSharedRegionAttachment,
    HostSharedRegionClass,
    LocalRegionHandle,
    RegionMemoryKind,
    VramRegionHandle,
)
from tensorcast.api import _region_cache as region_cache


@pytest.fixture
def store(monkeypatch) -> Store:
    client = types.SimpleNamespace()
    calls: dict[str, object] = {}
    runtime = types.SimpleNamespace(
        daemon_endpoint="dummy:0",
        opts=StoreOptions(),
        retry_policies={},
        ensure_client=lambda: client,
        operation_span=lambda *args, **kwargs: types.SimpleNamespace(
            __enter__=lambda self: self,
            __exit__=lambda *exc: False,
            add_event=lambda *a, **k: None,
            set_attribute=lambda *a, **k: None,
            set_status=lambda *a, **k: None,
            record_exception=lambda *a, **k: None,
        ),
    )
    s = Store("dummy:0", runtime=runtime)

    # Fake CUDA handle exporter (redundant with sys.modules stub but keeps isolation per test)
    monkeypatch.setattr("tensorcast.api.store.get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle")
    monkeypatch.setattr(
        "tensorcast.api.store.get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )

    # Fake daemon client with register/unregister methods
    def _register_vram_region(*, device_id: int, size_bytes: int, ttl_ms: int, cuda_ipc_handle: bytes, region_name: str | None = None):
        assert device_id >= 0
        assert size_bytes > 0
        assert ttl_ms >= 0
        assert cuda_ipc_handle == b"fake-handle"
        return VramRegionHandle(region_id="region:test123", ttl_ms=ttl_ms)

    def _unregister_vram_region(region_id: str, *, force: bool | None = None) -> bool:
        assert region_id == "region:test123"
        return True

    def _register_region(
        *,
        memory_kind: RegionMemoryKind,
        size_bytes: int,
        ttl_ms: int,
        device_id: int | None = None,
        cuda_ipc_handle: bytes | None = None,
        host_shared_attach_token: bytes | None = None,
        daemon_managed: bool = False,
        host_shared_region_class: HostSharedRegionClass | None = None,
        session_id: str | None = None,
        region_name: str | None = None,
        timeout_s: float = 10.0,
    ) -> LocalRegionHandle:
        calls["register_region"] = {
            "memory_kind": memory_kind,
            "size_bytes": size_bytes,
            "ttl_ms": ttl_ms,
            "device_id": device_id,
            "cuda_ipc_handle": cuda_ipc_handle,
            "host_shared_attach_token": host_shared_attach_token,
            "daemon_managed": daemon_managed,
            "host_shared_region_class": host_shared_region_class,
            "session_id": session_id,
            "region_name": region_name,
            "timeout_s": timeout_s,
        }
        return LocalRegionHandle(
            region_id="region:host123",
            memory_kind=memory_kind,
            ttl_ms=ttl_ms,
            size_bytes=size_bytes,
            device_id=device_id,
            attach_token=b"attach-token",
            daemon_managed=daemon_managed,
            host_shared_region_class=host_shared_region_class,
        )

    def _unregister_region(
        region_id: str,
        *,
        session_id: str | None = None,
        force: bool | None = None,
        timeout_s: float = 10.0,
    ) -> bool:
        calls["unregister_region"] = {
            "region_id": region_id,
            "session_id": session_id,
            "force": force,
            "timeout_s": timeout_s,
        }
        return True

    def _attach_host_shared_region(
        handle: LocalRegionHandle,
        *,
        timeout_s: float = 5.0,
    ) -> HostSharedRegionAttachment:
        calls["attach_host_shared_region"] = {
            "handle": handle,
            "timeout_s": timeout_s,
        }
        return HostSharedRegionAttachment(
            region_id=handle.region_id,
            size_bytes=handle.size_bytes,
            attach_token=handle.attach_token,
            fd=123,
        )

    def _release_host_shared_region(
        handle: LocalRegionHandle,
        *,
        timeout_s: float = 5.0,
    ) -> bool:
        calls["release_host_shared_region"] = {
            "handle": handle,
            "timeout_s": timeout_s,
        }
        return True

    client.register_vram_region = _register_vram_region
    client.unregister_vram_region = _unregister_vram_region
    client.register_region = _register_region
    client.unregister_region = _unregister_region
    client.attach_host_shared_region = _attach_host_shared_region
    client.release_host_shared_region = _release_host_shared_region

    s._test_calls = calls

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


def test_register_vram_region_ttl_0_is_allowed(store: Store):
    device_id = 0
    base_ptr = 0x1000
    size_bytes = 4096
    ttl_ms = 0

    handle = store.register_vram_region(
        device_id=device_id, base_ptr=base_ptr, size_bytes=size_bytes, ttl_ms=ttl_ms
    )
    assert handle.region_id == "region:test123"
    assert handle.ttl_ms == 0


def test_register_region_forwards_host_shared_args(store: Store):
    handle = store.register_region(
        memory_kind=RegionMemoryKind.HOST_SHARED,
        size_bytes=8192,
        ttl_ms=3000,
        daemon_managed=True,
        host_shared_region_class=HostSharedRegionClass.ALLOCATOR,
        session_id="sess-1",
        name="allocator-region",
        timeout_s=12.5,
    )
    calls = store._test_calls
    assert handle.region_id == "region:host123"
    assert calls["register_region"] == {
        "memory_kind": RegionMemoryKind.HOST_SHARED,
        "size_bytes": 8192,
        "ttl_ms": 3000,
        "device_id": None,
        "cuda_ipc_handle": None,
        "host_shared_attach_token": None,
        "daemon_managed": True,
        "host_shared_region_class": HostSharedRegionClass.ALLOCATOR,
        "session_id": "sess-1",
        "region_name": "allocator-region",
        "timeout_s": 12.5,
    }


def test_host_shared_region_attach_release_and_unregister_forward(store: Store):
    handle = LocalRegionHandle(
        region_id="region:host123",
        memory_kind=RegionMemoryKind.HOST_SHARED,
        ttl_ms=1000,
        size_bytes=4096,
        attach_token=b"attach-token",
        daemon_managed=True,
        host_shared_region_class=HostSharedRegionClass.SCRATCH,
    )
    attachment = store.attach_host_shared_region(handle, timeout_s=7.0)
    released = store.release_host_shared_region(handle, timeout_s=8.0)
    removed = store.unregister_region(
        handle.region_id,
        session_id="sess-2",
        force=True,
        timeout_s=9.0,
    )

    calls = store._test_calls
    assert attachment.fd == 123
    assert released is True
    assert removed is True
    assert calls["attach_host_shared_region"] == {
        "handle": handle,
        "timeout_s": 7.0,
    }
    assert calls["release_host_shared_region"] == {
        "handle": handle,
        "timeout_s": 8.0,
    }
    assert calls["unregister_region"] == {
        "region_id": "region:host123",
        "session_id": "sess-2",
        "force": True,
        "timeout_s": 9.0,
    }
