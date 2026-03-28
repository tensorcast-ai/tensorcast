#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorcast import startup as startup_mod
from tensorcast.api.store import runtime as store_runtime
from tensorcast.types import ServerConfig


class _FakeClient:
    def get_server_config(self):
        return ServerConfig(
            mem_pool_size=1,
            tx_slice_bytes=2,
            artifact_chunk_bytes=3,
            local_handle_socket_path="/tmp/local-handle.sock",
            cpu_shared_memory_enabled=True,
        )

    def close(self) -> None:
        return None


class _RetryingConfigClient:
    def __init__(self) -> None:
        self.calls = 0

    def get_server_config(self):
        self.calls += 1
        if self.calls < 3:
            return ServerConfig(
                mem_pool_size=1,
                tx_slice_bytes=2,
                artifact_chunk_bytes=3,
                local_handle_socket_path="",
                cpu_shared_memory_enabled=True,
            )
        return ServerConfig(
            mem_pool_size=1,
            tx_slice_bytes=2,
            artifact_chunk_bytes=3,
            local_handle_socket_path="/tmp/retry-ready.sock",
            cpu_shared_memory_enabled=True,
        )

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))


def test_get_context_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    with contextlib.suppress(Exception):
        store_runtime.shutdown_context()

    startup_calls: list[str] = []

    def _fake_init(*, mode, address=None, **_kwargs):  # noqa: ANN001
        del address
        startup_calls.append(mode)
        if mode == "connect":
            raise RuntimeError("no running daemon")

    monkeypatch.setattr(startup_mod, "init", _fake_init)
    monkeypatch.setattr(startup_mod, "is_initialized", lambda: False)

    attempts = {"count": 0}

    def _runtime_provider() -> object:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("runtime missing")
        return SimpleNamespace(address="127.0.0.1:61010")

    ctx = store_runtime.get_context(
        runtime_provider=_runtime_provider,
        client_factory=lambda _addr: _FakeClient(),
    )
    try:
        assert ctx.daemon_endpoint == "127.0.0.1:61010"
        assert startup_calls == ["connect", "auto"]
    finally:
        store_runtime.shutdown_context()


def test_store_runtime_retries_server_config_until_local_handle_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RetryingConfigClient()
    monkeypatch.setattr(store_runtime.StoreRuntimeContext,
                        "_SERVER_CONFIG_RETRY_INTERVAL_S", 0.0)
    monkeypatch.setattr(store_runtime.StoreRuntimeContext,
                        "_SERVER_CONFIG_READY_TIMEOUT_S", 1.0)

    ctx = store_runtime.StoreRuntimeContext(
        "127.0.0.1:61010",
        client_factory=lambda _addr: client,
    )
    try:
        assert client.calls == 3
        assert ctx.capabilities.server_config is not None
        assert ctx.capabilities.server_config.local_handle_socket_path == \
            "/tmp/retry-ready.sock"
    finally:
        ctx.close()
