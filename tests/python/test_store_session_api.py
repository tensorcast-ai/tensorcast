#  Copyright (c) 2025, TensorCast Team.

"""Store session API tests covering sync/async verbs and helper surfaces."""

from __future__ import annotations

import concurrent.futures
import contextlib
import importlib
import socket
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tensorcast import startup
from tensorcast.api import ArtifactFuture, Store
from tensorcast.api._config import GetArtifactOptions, RegisterArtifactOptions
from tests.python.utils.daemon import start_daemon_binary


def _free_port() -> int:
    """Return an available TCP port on localhost."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def store_session(tmp_path: Path) -> Iterator[Store]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Store session integration test")

    listen = f"127.0.0.1:{_free_port()}"
    storage_path = tmp_path / "store-data"
    proc = start_daemon_binary(listen, storage_path)
    try:
        startup.init(address=listen)
        store = Store(listen)
        try:
            yield store
        finally:
            store.close()
            startup.shutdown()
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=3)


def test_store_put_and_get_round_trip(store_session: Store) -> None:
    device = torch.device("cuda", 0)
    key = f"store:test:{uuid.uuid4()}"
    state = {
        "linear.weight": torch.arange(16, dtype=torch.float32, device=device).view(4, 4),
        "linear.bias": torch.arange(4, dtype=torch.float32, device=device),
    }

    registered = store_session.put(
        state,
        key=key,
        options=RegisterArtifactOptions(plan="vram_coalesced"),
        device=device,
    )
    assert registered.registration_result is not None

    loaded = store_session.get(key=key, device=device)
    for name, tensor in state.items():
        assert torch.equal(loaded[name], tensor)

    target = {name: torch.empty_like(tensor) for name, tensor in state.items()}
    store_session.get_into(target, key=key, device=device)
    for name, tensor in state.items():
        assert torch.equal(target[name], tensor)

    future = store_session.get_async(key=key, device=device)
    result = future.result(timeout=5.0)
    for name, tensor in state.items():
        assert torch.equal(result[name], tensor)


def test_artifact_future_callbacks_and_cancel() -> None:
    invoked: dict[str, bool] = {"confirm": False, "cancel": False}
    base_future: "concurrent.futures.Future[str]" = concurrent.futures.Future()

    def _confirm() -> None:
        invoked["confirm"] = True
        base_future.set_result("done")

    def _cancel() -> bool:
        invoked["cancel"] = True
        return True

    future = ArtifactFuture(base_future, confirm=_confirm, cancel_callback=_cancel)
    assert future.result(timeout=1.0) == "done"
    assert invoked["confirm"] is True

    base_future_cancel: "concurrent.futures.Future[str]" = concurrent.futures.Future()
    cancel_future = ArtifactFuture(base_future_cancel, cancel_callback=_cancel)
    assert cancel_future.cancel() is True
    assert invoked["cancel"] is True


def _reload_api_with_env(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("TENSORCAST_STORE_SESSION_REQUIRED", raising=False)
    else:
        monkeypatch.setenv("TENSORCAST_STORE_SESSION_REQUIRED", value)
    sys.modules.pop("tensorcast.api", None)
    import tensorcast.api as api_module

    return importlib.reload(api_module)


def test_legacy_helpers_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _reload_api_with_env(monkeypatch, "1")
    with pytest.raises(RuntimeError, match="Legacy helper usage is disabled"):
        api.get_artifact_sync(key="demo", options=GetArtifactOptions())

    _reload_api_with_env(monkeypatch, None)
