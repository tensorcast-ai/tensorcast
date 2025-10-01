#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from tensorcast.api import store as store_mod
from tensorcast.api._config import GetArtifactOptions, PlanType, RegisterArtifactOptions
from tensorcast.api.store import (
    ArtifactError,
    ArtifactFuture,
    FallbackOptions,
    Store,
)
from tensorcast.api._loader import MaterializedArtifact
from tensorcast.api._register import RegistrationResult
from tensorcast.types import ArtifactDescriptor, ServerConfig


class FakeHandle:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self, timeout_s: float = 5.0) -> bool:
        self.aborted = True
        return True


@dataclass
class FakeDaemonCtl:
    resolves: dict[str, tuple[str | None, str | None]]

    def __post_init__(self) -> None:
        self.unload_calls: list[tuple[str, str]] = []
        self.closed = False
        self.keepalive_calls: list[tuple[str, int, int]] = []
        self.server_address = "fake://daemon"

    def get_server_config(self) -> ServerConfig:
        return ServerConfig(tx_slice_bytes=4096, mem_pool_size=1 << 20, artifact_chunk_bytes=1 << 18)

    def resolve_key_mapping(self, key: str) -> tuple[str | None, str | None]:
        return self.resolves.get(key, (None, None))

    def keep_alive_registered_artifact(self, registration_id: str, ttl_ms: int, epoch: int) -> bool:  # noqa: D401
        self.keepalive_calls.append((registration_id, ttl_ms, epoch))
        return True

    def unload_replica(self, replica_uuid: str, *, disk_path: str = "") -> bool:
        self.unload_calls.append((replica_uuid, disk_path))
        return True

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeEnvironment:
    client: FakeDaemonCtl
    futures: list[FakeHandle]
    block_registration: bool
    register_started: threading.Event
    materialized_by_id: dict[str, MaterializedArtifact]

    def add_materialized(
        self,
        artifact_id: str,
        tensors: dict[str, torch.Tensor],
        *,
        replica_uuid: str | None = None,
    ) -> None:
        if replica_uuid is None:
            replica_uuid = f"replica-{artifact_id}"
        index: dict[str, tuple[int, int, list[int], list[int], str, int]] = {}
        offset = 0
        for name, tensor in tensors.items():
            size_bytes = int(tensor.element_size() * tensor.nelement())
            index[name] = (
                offset,
                size_bytes,
                list(map(int, tensor.shape)),
                list(map(int, tensor.stride())),
                str(tensor.dtype),
                int(tensor.storage_offset()),
            )
            offset += size_bytes
        canonical = json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.materialized_by_id[artifact_id] = MaterializedArtifact(
            artifact_id=artifact_id,
            state_dict=tensors,
            canonical_index_bytes=canonical,
            replica_uuid=replica_uuid,
            disk_path=None,
        )

    def make_registration_result(
        self,
        plan: PlanType,
        artifact: dict[str, torch.Tensor],
        *,
        device_id: int | None,
    ) -> RegistrationResult:
        index: dict[str, tuple[int, int, list[int], list[int], str, int]] = {}
        offset = 0
        for name, tensor in artifact.items():
            size_bytes = int(tensor.element_size() * tensor.nelement())
            index[name] = (
                offset,
                size_bytes,
                list(map(int, tensor.shape)),
                list(map(int, tensor.stride())),
                str(tensor.dtype),
                int(tensor.storage_offset()),
            )
            offset += size_bytes
        artifact_id = f"mi2:test:{plan.value}:{offset}"
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            index_multihash="hash-index",
            data_multihash="hash-data",
            schema_version="v2",
            encoding="raw",
            total_size=offset,
        )
        index_bytes = json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")
        device = 0 if device_id is None else int(device_id)
        build = SimpleNamespace(device_id=device)
        layout = SimpleNamespace(total_size=offset)
        return RegistrationResult(
            state_dict=dict(artifact),
            descriptor=descriptor,
            lease=None,
            build=build,
            layout=layout,
            index_bytes=index_bytes,
            plan=plan,
        )

    def fake_register(
        self,
        *,
        artifact: dict[str, torch.Tensor],
        options: RegisterArtifactOptions,
        device_id: int | None,
        ttl_ms: int | None,
        force_lease_in_place: bool,
        prevalidate_disk: bool,
        client: FakeDaemonCtl,
        daemon_address: str,
        cancel_event: threading.Event | None,
        on_begin: Callable[[FakeHandle], None] | None,
    ) -> RegistrationResult:
        handle = FakeHandle()
        if on_begin is not None:
            on_begin(handle)
        self.futures.append(handle)
        self.register_started.set()
        while self.block_registration and (cancel_event is None or not cancel_event.is_set()):
            time.sleep(0.01)
        if cancel_event is not None and cancel_event.is_set():
            raise concurrent.futures.CancelledError
        plan = PlanType.VRAM_LEASED if force_lease_in_place else options.plan
        return self.make_registration_result(plan, artifact, device_id=device_id)

    def fake_materialize(
        self,
        *,
        client: FakeDaemonCtl,
        daemon_address: str,
        device_id: int,
        artifact_id: str | None,
        key: str | None,
        options: GetArtifactOptions | None = None,
    ) -> MaterializedArtifact:
        if artifact_id is not None and artifact_id in self.materialized_by_id:
            return self.materialized_by_id[artifact_id]
        if key is not None:
            resolved_id, _ = client.resolves.get(key, (None, None))
            if resolved_id and resolved_id in self.materialized_by_id:
                return self.materialized_by_id[resolved_id]
        raise RuntimeError("artifact not found")


@pytest.fixture
def store_env(monkeypatch: pytest.MonkeyPatch) -> tuple[Store, FakeEnvironment]:
    client = FakeDaemonCtl(resolves={})
    env = FakeEnvironment(
        client=client,
        futures=[],
        block_registration=False,
        register_started=threading.Event(),
        materialized_by_id={},
    )

    monkeypatch.setattr(store_mod, "get_daemon_client", lambda endpoint: client)
    monkeypatch.setattr(store_mod, "_register_artifact_core", env.fake_register)
    monkeypatch.setattr(store_mod, "materialize_artifact", env.fake_materialize)

    store = store_mod.Store("fake://daemon")
    return store, env


def test_artifact_future_confirm_sets_result() -> None:
    fut: concurrent.futures.Future[str] = concurrent.futures.Future()
    called = []

    def confirm() -> None:
        called.append(True)
        fut.set_result("done")

    wrapped = ArtifactFuture(fut, confirm=confirm)
    assert wrapped.result() == "done"
    assert called == [True]


def test_artifact_future_cancel_invokes_callback() -> None:
    fut: concurrent.futures.Future[None] = concurrent.futures.Future()
    called = []

    def cancel_cb() -> bool:
        called.append(True)
        return True

    wrapped = ArtifactFuture(fut, cancel_callback=cancel_cb)
    assert wrapped.cancel() is True
    assert called == [True]


def test_store_put_and_register_sync(store_env: tuple[Store, FakeEnvironment]) -> None:
    store, env = store_env
    tensor = torch.arange(4, dtype=torch.float32)
    put_result = store.put({"weights": tensor})
    assert put_result.replica.plan is PlanType.VRAM_COALESCED
    assert put_result.canonical_index.total_size_bytes == tensor.element_size() * tensor.nelement()

    reg_result = store.register({"weights": tensor})
    assert reg_result.replica.plan is PlanType.VRAM_LEASED
    assert reg_result.canonical_index.entries[0].name == "weights"


def test_store_put_async_cancel_triggers_abort(store_env: tuple[Store, FakeEnvironment]) -> None:
    store, env = store_env
    env.block_registration = True
    env.register_started.clear()

    future = store.put_async({"v": torch.ones(2, dtype=torch.float32)})
    assert env.register_started.wait(timeout=1.0)
    assert future.cancel() is True
    assert env.futures[-1].aborted is True
    with pytest.raises(ArtifactError):
        future.result(timeout=1.0)


def test_store_get_into_copies_tensors(
    store_env: tuple[Store, FakeEnvironment], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, env = store_env
    state = {"bias": torch.arange(3, dtype=torch.float32)}
    env.add_materialized("artifact-1", state)

    target = {"bias": torch.zeros(3, dtype=torch.float32)}

    def fake_validate(
        self: Store,
        *,
        canonical_index: Any,
        target: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        device_id: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(target["bias"], source["bias"])]

    monkeypatch.setattr(store, "_validate_targets", fake_validate.__get__(store, Store))

    store.get_into(target, artifact_id="artifact-1", device=torch.device("cuda", 0))
    assert torch.allclose(target["bias"], state["bias"])


def test_store_get_into_enforces_device(store_env: tuple[Store, FakeEnvironment]) -> None:
    store, env = store_env
    state = {"bias": torch.arange(3, dtype=torch.float32)}
    env.add_materialized("artifact-err", state)
    target = {"bias": torch.zeros(3, dtype=torch.float32)}
    with pytest.raises(ArtifactError):
        store.get_into(target, artifact_id="artifact-err", device=torch.device("cuda", 0))


def test_store_get_async_cancel_unloads_replica(
    store_env: tuple[Store, FakeEnvironment], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, env = store_env
    artifact_id = "artifact-cancel"
    env.add_materialized(artifact_id, {"w": torch.ones(2, dtype=torch.float32)}, replica_uuid="rep-100")

    cancel_ready = threading.Event()

    def slow_materialize(
        self: Store,
        *,
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        options: GetArtifactOptions,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None,
        span: Any = None,
    ) -> MaterializedArtifact:
        cancel_ready.set()
        assert cancel_event is not None
        while not cancel_event.is_set():
            time.sleep(0.01)
        assert artifact_id is not None
        return env.materialized_by_id[artifact_id]

    monkeypatch.setattr(store, "_materialize", slow_materialize.__get__(store, Store))

    future = store.get_async(artifact_id=artifact_id, device=torch.device("cuda", 0))
    assert cancel_ready.wait(timeout=1.0)
    assert future.cancel() is True
    with pytest.raises(ArtifactError):
        future.result(timeout=1.0)
    assert ("rep-100", "") in env.client.unload_calls


def test_store_get_prefers_disk_when_available(
    store_env: tuple[Store, FakeEnvironment], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, env = store_env
    disk_called = {}

    def fake_materialize_from_disk(
        self: Store,
        *,
        disk_path: str,
        artifact_id: str | None,
        device_id: int,
        verify_checksums: bool,
    ) -> MaterializedArtifact:
        disk_called["path"] = disk_path
        return MaterializedArtifact(
            artifact_id=artifact_id or "disk-artifact",
            state_dict={"t": torch.zeros(1, dtype=torch.float32)},
            canonical_index_bytes=json.dumps(
                {
                    "t": [
                        0,
                        4,
                        [1],
                        [1],
                        "torch.float32",
                        0,
                    ]
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            replica_uuid="",
            disk_path=disk_path,
        )

    monkeypatch.setattr(store, "_materialize_from_disk", fake_materialize_from_disk.__get__(store, Store))

    result = store.get(
        key="does-not-matter",
        fallback=FallbackOptions(disk_path="/tmp/artifact", prefer_disk=True, allow_p2p=False),
        device=torch.device("cuda", 0),
    )
    assert disk_called["path"] == "/tmp/artifact"
    assert "t" in result
