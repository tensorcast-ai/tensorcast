#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorcast import startup as startup_mod
from tensorcast.api.store import runtime as store_runtime


class _FakeClient:
    def get_server_config(self):
        return SimpleNamespace(
            mem_pool_size=1,
            tx_slice_bytes=2,
            artifact_chunk_bytes=3,
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
