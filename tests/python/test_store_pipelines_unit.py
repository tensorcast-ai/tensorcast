#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorcast.api._config import PlanType
from tensorcast.api._materialize import MaterializedArtifact
from tensorcast.api._register import RegistrationResult
from tensorcast.api.store.async_ops import TrackedExecutor
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.registration import RegistrationPipeline
from tensorcast.api.store.retry import build_retry_policies
from tensorcast.api.store.types import (
    ArtifactError,
    FallbackOptions,
    RetryPolicy,
    StoreOptions,
)
from tensorcast.api.store.views import ViewOrchestrator


class _DummySpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.exceptions: list[Exception] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:  # pragma: no cover - noop
        self.attributes["status"] = status

    def add_event(self, name: str, attrs: dict[str, object]) -> None:
        self.events.append((name, attrs))

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)


class _FakeClient:
    def __init__(self) -> None:
        self.unloaded: list[str] = []

    def resolve_key_mapping(self, key: str) -> tuple[str | None, str | None]:
        return None, None

    def unload_replica(self, replica_uuid: str, *, disk_path: str = "") -> bool:
        self.unloaded.append(f"{replica_uuid}:{disk_path}")
        return True


class _DummyRuntime:
    def __init__(self) -> None:
        self.daemon_endpoint = "daemon"
        self.session_id = "sess"
        self.opts = StoreOptions()
        self.retry_policies = build_retry_policies(
            {
                "register": RetryPolicy(
                    deadline_seconds=2.0,
                    max_attempts=2,
                    base_backoff_seconds=0.0,
                    backoff_multiplier=1.0,
                    jitter=0.0,
                )
            }
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.leases: list[object] = []
        self.futures: list[object] = []
        self.client = _FakeClient()

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        self.futures.append(future)

    def track_lease(self, lease: object | None) -> None:
        if lease is not None:
            self.leases.append(lease)

    def ensure_client(self) -> _FakeClient:
        return self.client

    def cache_key_mapping(self, *_, **__) -> None:  # pragma: no cover - noop
        return None

    def resolve_key_mapping_cached(
        self, *, key: str
    ) -> tuple[str | None, str | None]:  # pragma: no cover - noop
        return self.client.resolve_key_mapping(key)

    @contextmanager
    def operation_span(self, *_args, **_kwargs):
        yield _DummySpan()

    @property
    def capabilities(self):
        class _Caps:
            mem_pool_bytes = 0
            tx_slice_bytes = 0
            artifact_chunk_bytes = 0
            supports_coalesced = False
            supports_lease = False
            server_config = None

        return _Caps()

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def _registration_result(artifact_id: str = "aid") -> RegistrationResult:
    index_bytes = json.dumps({"x": [0, 4, [1], [1], "float32", 0]}).encode("utf-8")
    descriptor = SimpleNamespace(artifact_id=artifact_id, data_multihash="hash")
    layout = SimpleNamespace(total_size=4)
    build = SimpleNamespace(device_id=0)
    return RegistrationResult(
        state_dict={"x": torch.zeros(1)},
        descriptor=descriptor,
        lease=None,
        build=build,
        layout=layout,
        index_bytes=index_bytes,
        plan=PlanType.VRAM_COALESCED,
    )


def _materialized_artifact(replica_uuid: str = "r1") -> MaterializedArtifact:
    index_bytes = json.dumps({"x": [0, 4, [1], [1], "float32", 0]}).encode("utf-8")
    return MaterializedArtifact(
        artifact_id="aid",
        state_dict={"x": torch.zeros(1, device="cpu")},
        canonical_index_bytes=index_bytes,
        replica_uuid=replica_uuid,
        disk_path=None,
    )


def test_registration_retries_then_succeeds():
    runtime = _DummyRuntime()
    views = ViewOrchestrator(runtime)

    attempts: list[int] = []

    def fake_register(**_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise ArtifactError("unavailable", status_code="UNAVAILABLE", retryable=True)
        return _registration_result()

    pipeline = RegistrationPipeline(runtime, views, register_fn=fake_register)

    result = pipeline.register({"x": torch.zeros(1)}, key="k1")
    runtime.close()

    assert result.artifact_id == "aid"
    assert len(attempts) == 2


def test_materialization_rejects_empty_disk_fallback():
    runtime = _DummyRuntime()
    pipeline = MaterializationPipeline(runtime, ViewOrchestrator(runtime))

    with pytest.raises(ArtifactError) as excinfo:
        pipeline.get(fallback=FallbackOptions(disk_path=""))
    runtime.close()

    assert excinfo.value.status_code == "INVALID_ARGUMENT"


def test_get_into_validates_targets(monkeypatch):
    runtime = _DummyRuntime()
    pipeline = MaterializationPipeline(
        runtime, ViewOrchestrator(runtime), materialize_fn=lambda **_: _materialized_artifact(replica_uuid="r2")
    )

    called_release: list[str] = []

    def fake_release(mat: MaterializedArtifact, client: _FakeClient) -> None:
        called_release.append(mat.replica_uuid)

    monkeypatch.setattr(pipeline, "_release_materialized", fake_release)

    target: dict[str, torch.Tensor] = {}
    with pytest.raises(ArtifactError) as excinfo:
        pipeline.get_into(target, artifact_id="aid")

    runtime.close()
    assert excinfo.value.status_code == "INVALID_ARGUMENT"
    assert called_release == ["r2"]


def test_tracked_executor_cancel_invokes_callback():
    runtime = _DummyRuntime()
    executor = TrackedExecutor(runtime)
    cancel_called = threading.Event()

    def work() -> int:
        time.sleep(0.2)
        return 1

    def cancel() -> bool:
        cancel_called.set()
        return True

    fut = executor.submit(work, cancel_callback=cancel)
    cancelled = fut.cancel()

    runtime.close()
    assert cancelled
    assert cancel_called.is_set()


def test_store_import_dag_smoke():
    import importlib

    types_mod = importlib.import_module("tensorcast.api.store.types")
    handles_mod = importlib.import_module("tensorcast.api.store.handles")
    runtime_mod = importlib.import_module("tensorcast.api.store.runtime")
    registration_mod = importlib.import_module("tensorcast.api.store.registration")
    materialization_mod = importlib.import_module("tensorcast.api.store.materialization")

    assert hasattr(types_mod, "StoreOptions")
    assert hasattr(handles_mod, "RegisteredArtifact")
    assert hasattr(runtime_mod, "StoreRuntimeContext")
    assert hasattr(registration_mod, "RegistrationPipeline")
    assert hasattr(materialization_mod, "MaterializationPipeline")
    # Layering sanity: types remain free of runtime symbols.
    assert not hasattr(types_mod, "StoreRuntimeContext")
