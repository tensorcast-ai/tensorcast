#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest
import torch

import tensorcast.api.store as store_mod
from tensorcast.api._config import GetArtifactOptions, PlanType, RegisterArtifactOptions
from tensorcast.api._materialize import MaterializedArtifact
from tensorcast.api._register import (
    BuildContext,
    CoalescedLayout,
    RegistrationResult,
    ViewRegistrationContext,
)
from tensorcast.api.store import (
    ArtifactError,
    ArtifactFuture,
    FallbackOptions,
    Store,
    StoreOptions,
)
from tensorcast.common.identity import ArtifactIdKind
from tensorcast.types import ArtifactDescriptor, ServerConfig
from tensorcast.proto.daemon.v1 import store_daemon_pb2


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
        self.resolve_calls: list[str] = []

    def get_server_config(self) -> ServerConfig:
        return ServerConfig(tx_slice_bytes=4096, mem_pool_size=1 << 20, artifact_chunk_bytes=1 << 18)

    def resolve_key_mapping(self, key: str) -> tuple[str | None, str | None]:
        self.resolve_calls.append(key)
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
        client_artifact_id: str | None = None,
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
        normalized_client_id = (
            client_artifact_id.strip() if client_artifact_id and client_artifact_id.strip() else None
        )
        artifact_id = normalized_client_id or f"mi2:test:{plan.value}:{offset}"
        id_kind = ArtifactIdKind.CGID if artifact_id.startswith("cgid:") else ArtifactIdKind.MI2
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            index_multihash="hash-index",
            data_multihash="hash-data",
            schema_version="v3",
            encoding="raw",
            total_size=offset,
            id_kind=id_kind,
        )
        index_bytes = json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")
        device = 0 if device_id is None else int(device_id)
        build_ns = SimpleNamespace(
            device_id=device,
            input_mode="cuda" if device_id is None else "cpu",
            tensor_meta_index={},
            tensor_source_index={},
        )
        build = cast(BuildContext, build_ns)
        layout_ns = SimpleNamespace(
            total_size=offset,
            device_id=device,
            offsets={},
            unique_chunks=[],
        )
        layout = cast(CoalescedLayout, layout_ns)
        plan_enum = PlanType.parse(plan)
        return RegistrationResult(
            state_dict=dict(artifact),
            descriptor=descriptor,
            lease=None,
            build=build,
            layout=layout,
            index_bytes=index_bytes,
            plan=plan_enum,
        )

    def fake_register(
        self,
        *,
        artifact: dict[str, torch.Tensor],
        options: RegisterArtifactOptions,
        device_id: int | None,
        ttl_ms: int | None,
        client_artifact_id: str | None = None,
        force_lease_in_place: bool,
        prevalidate_disk: bool,
        client: FakeDaemonCtl,
        daemon_address: str,
        cancel_event: threading.Event | None,
        on_begin: Callable[[FakeHandle], None] | None,
        view: ViewRegistrationContext | None = None,
    ) -> RegistrationResult:
        del view
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
        return self.make_registration_result(
            plan,
            artifact,
            device_id=device_id,
            client_artifact_id=client_artifact_id,
        )

    def fake_materialize(
        self,
        *,
        client: FakeDaemonCtl,
        daemon_address: str,
        device_id: int,
        artifact_id: str | None,
        key: str | None,
        options: GetArtifactOptions | None = None,
        view: store_daemon_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: int | None = None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
    ) -> MaterializedArtifact:
        del daemon_address, device_id, options, view, view_id, placement, canonical_index_hint
        disk_hint = disk_path_hint
        if artifact_id is not None and artifact_id in self.materialized_by_id:
            resolved = self.materialized_by_id[artifact_id]
        elif key is not None:
            resolved_id, mapped_disk = client.resolves.get(key, (None, None))
            if resolved_id and resolved_id in self.materialized_by_id:
                resolved = self.materialized_by_id[resolved_id]
                if disk_hint is None:
                    disk_hint = mapped_disk
            else:
                resolved = None
        else:
            resolved = None
        if resolved is None:
            raise RuntimeError("artifact not found")
        if disk_hint is not None and resolved.disk_path != disk_hint:
            resolved = replace(resolved, disk_path=disk_hint)
            self.materialized_by_id[resolved.artifact_id] = resolved
        return resolved


def _make_registered_artifact(
    artifact_id: str = "registered",
) -> store_mod.RegisteredArtifact:
    replica = store_mod.ReplicaInfo(
        replica_id="replica",
        replica_type="VRAM_LEASED",
        device=torch.device("cpu"),
        plan=PlanType.VRAM_LEASED,
        size_bytes=0,
    )
    canonical_index = store_mod.CanonicalIndex(
        entries=(),
        total_size_bytes=0,
        avbs_hash="",
    )
    return store_mod.RegisteredArtifact(
        artifact_id=artifact_id,
        replica=replica,
        canonical_index=canonical_index,
        lease=None,
    )


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


def test_store_get_into_unloads_replica(
    store_env: tuple[Store, FakeEnvironment], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, env = store_env
    state = {"bias": torch.arange(3, dtype=torch.float32)}
    env.add_materialized("artifact-release", state, replica_uuid="rep-unload-1")
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

    store.get_into(target, artifact_id="artifact-release", device=torch.device("cuda", 0))

    assert torch.allclose(target["bias"], state["bias"])
    assert env.client.unload_calls == [("rep-unload-1", "")]


def test_store_get_into_unloads_on_validation_error(
    store_env: tuple[Store, FakeEnvironment], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, env = store_env
    state = {"bias": torch.arange(3, dtype=torch.float32)}
    env.add_materialized("artifact-invalid", state, replica_uuid="rep-unload-2")
    target = {"bias": torch.zeros(3, dtype=torch.float32)}

    def failing_validate(
        self: Store,
        *,
        canonical_index: Any,
        target: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        device_id: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        raise ArtifactError("bad layout", status_code="FAILED_PRECONDITION", retryable=False)

    monkeypatch.setattr(store, "_validate_targets", failing_validate.__get__(store, Store))

    with pytest.raises(ArtifactError):
        store.get_into(target, artifact_id="artifact-invalid", device=torch.device("cuda", 0))

    assert env.client.unload_calls == [("rep-unload-2", "")]


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
        **kwargs: Any,
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


def test_store_get_into_async_unloads_replica(
    store_env: tuple[Store, FakeEnvironment], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, env = store_env
    state = {"bias": torch.arange(3, dtype=torch.float32)}
    env.add_materialized("artifact-async", state, replica_uuid="rep-unload-3")
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

    future = store.get_into_async(
        target,
        artifact_id="artifact-async",
        device=torch.device("cuda", 0),
    )

    assert future.result(timeout=1.0) is None
    assert torch.allclose(target["bias"], state["bias"])
    assert env.client.unload_calls == [("rep-unload-3", "")]


def test_store_get_into_async_unloads_on_validation_error(
    store_env: tuple[Store, FakeEnvironment], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, env = store_env
    state = {"bias": torch.arange(3, dtype=torch.float32)}
    env.add_materialized("artifact-async-invalid", state, replica_uuid="rep-unload-4")
    target = {"bias": torch.zeros(3, dtype=torch.float32)}

    def failing_validate(
        self: Store,
        *,
        canonical_index: Any,
        target: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        device_id: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        raise ArtifactError("bad layout", status_code="FAILED_PRECONDITION", retryable=False)

    monkeypatch.setattr(store, "_validate_targets", failing_validate.__get__(store, Store))

    future = store.get_into_async(
        target,
        artifact_id="artifact-async-invalid",
        device=torch.device("cuda", 0),
    )

    with pytest.raises(ArtifactError):
        future.result(timeout=1.0)

    assert env.client.unload_calls == [("rep-unload-4", "")]


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


def test_store_key_resolution_cache_reuses_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORCAST_STORE_KEY_CACHE_TTL_SECONDS", "3600")

    client = FakeDaemonCtl(resolves={})
    env = FakeEnvironment(
        client=client,
        futures=[],
        block_registration=False,
        register_started=threading.Event(),
        materialized_by_id={},
    )

    key = "artifact-key"
    artifact_id_value = "mi2:test:cached"
    disk_path_value = "/tmp/cached-artifact"
    client.resolves[key] = (artifact_id_value, disk_path_value)

    tensor = torch.ones(2, dtype=torch.float32)
    env.add_materialized(artifact_id_value, {"weight": tensor}, replica_uuid="rep-cache")

    monkeypatch.setattr(store_mod, "get_daemon_client", lambda endpoint: client)
    monkeypatch.setattr(store_mod, "materialize_artifact", env.fake_materialize)

    def fake_materialize_from_disk(
        self: Store,
        *,
        disk_path: str,
        artifact_id: str | None,
        device_id: int,
        verify_checksums: bool,
    ) -> MaterializedArtifact:
        assert disk_path == disk_path_value
        target_id = artifact_id or artifact_id_value
        return env.materialized_by_id[target_id]

    monkeypatch.setattr(
        store_mod.Store,
        "_materialize_from_disk",
        fake_materialize_from_disk,
    )

    store = store_mod.Store("fake://daemon")
    fallback = FallbackOptions(prefer_disk=True, allow_p2p=False, verify_checksums=False)

    first = store.get(key=key, fallback=fallback, device=torch.device("cuda", 0))
    second = store.get(key=key, fallback=fallback, device=torch.device("cuda", 0))

    assert torch.allclose(first["weight"], tensor)
    assert torch.allclose(second["weight"], tensor)
    assert client.resolve_calls.count(key) == 1

    store.close()


def test_register_function_delegates_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.zeros(1)

    class DummyStore:
        def __init__(self) -> None:
            self.calls: list[tuple[store_mod.TensorDict, dict[str, object]]] = []
            self.result = _make_registered_artifact()

        def register(
            self,
            tensors: store_mod.TensorDict,
            *,
            artifact_id: str | None = None,
            key: str | None = None,
            options: RegisterArtifactOptions | None = None,
            ttl_ms: int | None = None,
        ) -> store_mod.RegisteredArtifact:
            self.calls.append(
                (
                    tensors,
                    {
                        "artifact_id": artifact_id,
                        "key": key,
                        "options": options,
                        "ttl_ms": ttl_ms,
                    },
                )
            )
            return self.result

    session = DummyStore()
    monkeypatch.setattr(store_mod, "store", lambda: session)
    payload = {"w": tensor}
    result = store_mod.register(payload, key="demo", ttl_ms=1234)

    assert result is session.result
    assert session.calls[0][0] is payload
    assert session.calls[0][1]["key"] == "demo"
    assert session.calls[0][1]["ttl_ms"] == 1234


def test_put_async_function_delegates_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.zeros(1)

    class DummyStore:
        def __init__(self) -> None:
            self.calls: list[tuple[store_mod.TensorDict, dict[str, object]]] = []
            self.future = object()

        def put_async(
            self,
            tensors: store_mod.TensorDict,
            *,
            artifact_id: str | None = None,
            key: str | None = None,
            options: RegisterArtifactOptions | None = None,
            device: int | torch.device | None = None,
        ) -> object:
            self.calls.append(
                (
                    tensors,
                    {
                        "artifact_id": artifact_id,
                        "key": key,
                        "options": options,
                        "device": device,
                    },
                )
            )
            return self.future

    session = DummyStore()
    monkeypatch.setattr(store_mod, "store", lambda: session)
    payload = {"w": tensor}
    result = store_mod.put_async(
        payload, key="demo", device=torch.device("cuda", 0)
    )

    assert result is session.future
    assert session.calls[0][0] is payload
    assert session.calls[0][1]["key"] == "demo"
    assert session.calls[0][1]["device"] == torch.device("cuda", 0)


def test_get_function_delegates_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyStore:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.result = {"value": torch.ones(1)}

        def get(
            self,
            *,
            artifact_id: str | None = None,
            key: str | None = None,
            device: torch.device | str | None = None,
            fallback: FallbackOptions | None = None,
            options: GetArtifactOptions | None = None,
        ) -> dict[str, torch.Tensor]:
            self.kwargs = {
                "artifact_id": artifact_id,
                "key": key,
                "device": device,
                "fallback": fallback,
                "options": options,
            }
            return self.result

    session = DummyStore()
    monkeypatch.setattr(store_mod, "store", lambda: session)
    outcome = store_mod.get(
        key="demo",
        device="cuda:0",
        fallback=FallbackOptions(prefer_disk=True),
    )

    assert outcome is session.result
    assert session.kwargs is not None
    assert session.kwargs["key"] == "demo"
    assert session.kwargs["device"] == "cuda:0"


def test_get_into_async_function_delegates_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.zeros(1)

    class DummyStore:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.target: dict[str, torch.Tensor] | None = None
            self.future = object()

        def get_into_async(
            self,
            target: dict[str, torch.Tensor],
            *,
            artifact_id: str | None = None,
            key: str | None = None,
            device: torch.device | str | None = None,
            fallback: FallbackOptions | None = None,
            options: GetArtifactOptions | None = None,
        ) -> object:
            self.target = target
            self.kwargs = {
                "artifact_id": artifact_id,
                "key": key,
                "device": device,
                "fallback": fallback,
                "options": options,
            }
            return self.future

    session = DummyStore()
    monkeypatch.setattr(store_mod, "store", lambda: session)
    buffers = {"w": tensor}
    result = store_mod.get_into_async(buffers, key="demo", device="cuda:0")

    assert result is session.future
    assert session.target is buffers
    assert session.kwargs is not None
    assert session.kwargs["key"] == "demo"
    assert session.kwargs["device"] == "cuda:0"


def test_store_singleton_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyStore:
        def __init__(
            self, daemon_endpoint: str, *, opts: store_mod.StoreOptions | None = None
        ) -> None:
            self.daemon_endpoint = daemon_endpoint
            self.opts = opts
            self.closed = False

        def close(self) -> None:
            self.closed = True

    runtime_handle = SimpleNamespace(address="fake://daemon")
    monkeypatch.setattr(store_mod, "require_runtime", lambda: runtime_handle)
    monkeypatch.setattr(store_mod, "Store", DummyStore)

    store_mod.shutdown_process_store()

    first = store_mod.store()
    second = store_mod.store()

    assert isinstance(first, DummyStore)
    assert first is second
    assert first.daemon_endpoint == "fake://daemon"

    store_mod.shutdown_process_store()
    assert first.closed


def test_store_force_recreate_and_option_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyStore:
        def __init__(self, daemon_endpoint: str, *, opts: StoreOptions | None = None) -> None:
            self.daemon_endpoint = daemon_endpoint
            self.opts = opts
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    created: list[DummyStore] = []

    runtime_handle = SimpleNamespace(address="fake://daemon")
    monkeypatch.setattr(store_mod, "require_runtime", lambda: runtime_handle)
    monkeypatch.setattr(store_mod, "Store", DummyStore)

    store_mod.shutdown_process_store()

    initial_opts = store_mod.StoreOptions(fallback=FallbackOptions(prefer_disk=False))
    first = cast(DummyStore, store_mod.store(opts=initial_opts))

    mismatch_opts = store_mod.StoreOptions(fallback=FallbackOptions(prefer_disk=True))
    with pytest.raises(RuntimeError):
        store_mod.store(opts=mismatch_opts)

    refreshed = cast(DummyStore, store_mod.store(opts=mismatch_opts, force_recreate=True))

    assert created == [first, refreshed]
    assert first.closed
    assert refreshed.opts == mismatch_opts

    store_mod.shutdown_process_store()
