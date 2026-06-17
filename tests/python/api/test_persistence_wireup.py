#  Copyright (c) 2025-2026, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import tensorcast
import tensorcast.api.store as store_module
from tensorcast.api._config import PlanType, RegisterArtifactOptions
from tensorcast.api.store import Store
from tensorcast.api.store.registration import RegistrationPipeline
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.types import ArtifactError, CanonicalIndex, ReplicaInfo
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _DummyExecutor:
    def submit(self, func):
        future: concurrent.futures.Future[object] = concurrent.futures.Future()
        try:
            future.set_result(func())
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class _RuntimeStub:
    def __init__(self, client) -> None:
        self.client = client
        self._daemon_endpoint = "daemon-test"
        self.executor = _DummyExecutor()
        self.tracked_futures: list[object] = []
        self.tracked_leases: list[object] = []

    def ensure_client(self):
        return self.client

    @property
    def daemon_endpoint(self) -> str:
        return self._daemon_endpoint

    def track_future(self, future) -> None:  # pragma: no cover - not used
        self.tracked_futures.append(future)

    def track_lease(self, lease) -> None:  # pragma: no cover - not used
        self.tracked_leases.append(lease)


class _ClientStub:
    def __init__(
        self, *, response=None, query_response=None, exc: Exception | None = None
    ):
        self.response = response
        self.query_response = query_response
        self.exc = exc
        self.start_calls: list[tuple[str, str | None, object | None]] = []
        self.query_calls: list[tuple[str | None, str | None]] = []

    def start_persistence(
        self,
        *,
        artifact_id: str,
        key_hint: str | None = None,
        policy=None,
        timeout_s: float | None = None,
    ):
        del timeout_s
        self.start_calls.append((artifact_id, key_hint, policy))
        if self.exc:
            raise self.exc
        if self.response is not None:
            return self.response
        return store_daemon_pb2.StartPersistenceResponse(
            task_id="task-123",
            plan_id="plan-1",
            state=store_daemon_pb2.PERSISTENCE_STATE_PENDING,
            progress=0.0,
        )

    def query_persistence_status(
        self,
        *,
        task_id: str | None = None,
        artifact_id: str | None = None,
        timeout_s: float | None = None,
    ):
        del timeout_s
        self.query_calls.append((task_id, artifact_id))
        return self.query_response


class _ResultStub:
    def __init__(self, artifact_id: str | None):
        self.descriptor = SimpleNamespace(artifact_id=artifact_id)
        self.lease = None


def _pipeline(client: _ClientStub) -> RegistrationPipeline:
    runtime = cast(StoreRuntimeContext, _RuntimeStub(client))
    return RegistrationPipeline(runtime, views=SimpleNamespace())


def _store_with_client(client: _ClientStub) -> Store:
    store = Store.__new__(Store)
    store._runtime = cast(StoreRuntimeContext, _RuntimeStub(client))
    return store


def _registered_artifact(
    *, task_id: str | None, daemon_endpoint: str | None
) -> store_module.RegisteredArtifact:
    return store_module.RegisteredArtifact(
        artifact_id="artifact-persist",
        replica=ReplicaInfo(
            replica_id="replica-persist",
            replica_type="VRAM_LEASED",
            device=torch.device("cpu"),
            plan=PlanType.VRAM_LEASED,
            size_bytes=0,
        ),
        canonical_index=CanonicalIndex(entries=(), total_size_bytes=0, avbs_hash=""),
        lease=None,
        persistence_task_id=task_id,
        _daemon_endpoint=daemon_endpoint,
    )


def test_maybe_start_persistence_skips_when_disabled() -> None:
    client = _ClientStub()
    pipeline = _pipeline(client)
    options = RegisterArtifactOptions(policy="cache")

    task_id = pipeline._maybe_start_persistence(options, _ResultStub("artifact-1"))

    assert task_id is None
    assert client.start_calls == []


def test_maybe_start_persistence_requests_task() -> None:
    response = store_daemon_pb2.StartPersistenceResponse(
        task_id="task-42",
        plan_id="plan-9",
        state=store_daemon_pb2.PERSISTENCE_STATE_RUNNING,
        progress=0.5,
    )
    client = _ClientStub(response=response)
    pipeline = _pipeline(client)
    options = RegisterArtifactOptions(policy="durable")

    task_id = pipeline._maybe_start_persistence(options, _ResultStub("artifact-7"))

    assert task_id == "task-42"
    assert client.start_calls == [("artifact-7", None, options.policy)]


def test_maybe_start_persistence_raises_for_missing_artifact() -> None:
    client = _ClientStub()
    pipeline = _pipeline(client)
    options = RegisterArtifactOptions(policy="durable")

    with pytest.raises(ArtifactError):
        pipeline._maybe_start_persistence(options, _ResultStub(None))


def test_maybe_start_persistence_propagates_artifact_error() -> None:
    client = _ClientStub(
        exc=ArtifactError(
            "boom", status_code="FAILED_PRECONDITION", retryable=False
        )
    )
    pipeline = _pipeline(client)
    options = RegisterArtifactOptions(policy="durable")

    with pytest.raises(ArtifactError):
        pipeline._maybe_start_persistence(options, _ResultStub("artifact-8"))


def test_maybe_start_persistence_swallows_unexpected_errors() -> None:
    client = _ClientStub(exc=RuntimeError("transient failure"))
    pipeline = _pipeline(client)
    options = RegisterArtifactOptions(policy="durable")

    task_id = pipeline._maybe_start_persistence(options, _ResultStub("artifact-9"))

    assert task_id is None
    assert client.start_calls == [("artifact-9", None, options.policy)]


def test_query_persistence_status_requires_identifier() -> None:
    store = _store_with_client(_ClientStub())
    with pytest.raises(ArtifactError):
        store.query_persistence_status()


def test_query_persistence_status_maps_proto_fields() -> None:
    resp = store_daemon_pb2.QueryPersistenceStatusResponse(
        task_id="task-11",
        artifact_id="artifact-11",
        plan_id="plan-11",
        state=store_daemon_pb2.PERSISTENCE_STATE_DEGRADED,
        progress=0.6,
        degraded_reason="remote_failed",
        last_error="copy failed",
    )
    shard_proto = resp.shards.add()
    shard_proto.shard_id = "artifact-11:0"
    shard_proto.shard_idx = 0
    shard_proto.state = store_daemon_pb2.PERSISTENCE_STATE_RUNNING
    shard_proto.progress = 0.5
    shard_proto.degraded_reason = "pending_lease"
    shard_proto.target_nodes.extend(["node-a", "node-b"])
    shard_proto.lease_ids.extend(["lease-1", ""])
    client = _ClientStub(query_response=resp)
    store = _store_with_client(client)

    result = store.query_persistence_status(task_id="task-11")

    assert result.task_id == "task-11"
    assert result.artifact_id == "artifact-11"
    assert result.state == "degraded"
    assert result.degraded_reason == "remote_failed"
    assert result.last_error == "copy failed"
    assert len(result.shards) == 1
    shard = result.shards[0]
    assert shard.shard_id == "artifact-11:0"
    assert shard.state == "running"
    assert shard.degraded_reason == "pending_lease"
    assert shard.target_nodes == ("node-a", "node-b")
    assert shard.lease_ids == ("lease-1", "")
    assert client.query_calls == [("task-11", None)]


def test_persist_artifact_starts_persistence_with_key_hint() -> None:
    response = store_daemon_pb2.StartPersistenceResponse(
        task_id="task-55",
        plan_id="plan-55",
        state=store_daemon_pb2.PERSISTENCE_STATE_RUNNING,
        progress=0.2,
    )
    query_response = store_daemon_pb2.QueryPersistenceStatusResponse(
        task_id="task-55",
        artifact_id="artifact-55",
        plan_id="plan-55",
        state=store_daemon_pb2.PERSISTENCE_STATE_SUCCESS,
        progress=1.0,
    )
    client = _ClientStub(response=response, query_response=query_response)
    store = _store_with_client(client)

    operation = store.persist_artifact(
        artifact_id="artifact-55",
        key_hint="models/demo/serving/v1",
        policy="durable",
    )
    result = operation.wait(timeout_s=1.0)

    assert result.task_id == "task-55"
    assert result.artifact_id == "artifact-55"
    assert client.start_calls == [
        ("artifact-55", "models/demo/serving/v1", "durable")
    ]


def test_persistence_operation_status_carries_operation_context() -> None:
    resp = store_daemon_pb2.QueryPersistenceStatusResponse(
        task_id="task-21",
        artifact_id="artifact-21",
        plan_id="plan-21",
        state=store_daemon_pb2.PERSISTENCE_STATE_DEGRADED,
        progress=0.3,
        degraded_reason="pending_remote",
        last_error="copy still running elsewhere",
    )
    client = _ClientStub(query_response=resp)
    store = _store_with_client(client)

    status = store.persistence_operation(task_id="task-21").status()

    assert status.state == "degraded"
    assert status.error is not None
    assert status.error.context == {
        "operation_kind": "persistence_task",
        "task_id": "task-21",
        "artifact_id": "artifact-21",
        "target_artifact_id": "artifact-21",
    }


def test_module_persistence_operation_delegates_to_store(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _ClientStub(
        query_response=store_daemon_pb2.QueryPersistenceStatusResponse(
            task_id="task-31",
            artifact_id="artifact-31",
            plan_id="plan-31",
            state=store_daemon_pb2.PERSISTENCE_STATE_SUCCESS,
            progress=1.0,
        )
    )
    store = _store_with_client(client)
    monkeypatch.setattr(store_module, "_coerce_store", lambda: store)

    result = store_module.persistence_operation(task_id="task-31").result()

    assert result.task_id == "task-31"
    assert result.artifact_id == "artifact-31"
    assert client.query_calls == [("task-31", None), ("task-31", None)]


def test_top_level_persistence_operation_delegates_to_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ClientStub(
        query_response=store_daemon_pb2.QueryPersistenceStatusResponse(
            task_id="task-32",
            artifact_id="artifact-32",
            plan_id="plan-32",
            state=store_daemon_pb2.PERSISTENCE_STATE_SUCCESS,
            progress=1.0,
        )
    )
    store = _store_with_client(client)
    monkeypatch.setattr(store_module, "_coerce_store", lambda: store)

    result = tensorcast.persistence_operation(task_id="task-32").result()

    assert result.task_id == "task-32"
    assert result.artifact_id == "artifact-32"
    assert client.query_calls == [("task-32", None), ("task-32", None)]


def test_registered_artifact_persistence_operation_reattaches_via_origin_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ClientStub(
        query_response=store_daemon_pb2.QueryPersistenceStatusResponse(
            task_id="task-41",
            artifact_id="artifact-41",
            plan_id="plan-41",
            state=store_daemon_pb2.PERSISTENCE_STATE_SUCCESS,
            progress=1.0,
        )
    )
    stores: list[Store] = []

    def _store_factory(daemon_endpoint: str) -> Store:
        store = _store_with_client(client)
        store._runtime._daemon_endpoint = daemon_endpoint  # type: ignore[attr-defined]
        stores.append(store)
        return store

    monkeypatch.setattr(store_module, "Store", _store_factory)
    artifact = _registered_artifact(task_id="task-41", daemon_endpoint="daemon-origin")

    result = artifact.persistence_operation().result()

    assert result.task_id == "task-41"
    assert result.artifact_id == "artifact-41"
    assert len(stores) == 1
    assert stores[0]._runtime.daemon_endpoint == "daemon-origin"
    assert client.query_calls == [("task-41", None), ("task-41", None)]


def test_registered_artifact_persistence_operation_fails_closed_without_task_id() -> None:
    artifact = _registered_artifact(task_id=None, daemon_endpoint="daemon-origin")

    with pytest.raises(ArtifactError) as excinfo:
        artifact.persistence_operation()

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_registered_artifact_persistence_operation_fails_closed_without_daemon_endpoint() -> None:
    artifact = _registered_artifact(task_id="task-42", daemon_endpoint=None)

    with pytest.raises(ArtifactError) as excinfo:
        artifact.persistence_operation()

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_attempt_registration_rejects_conflicting_policy_inputs() -> None:
    client = _ClientStub()
    pipeline = _pipeline(client)

    with pytest.raises(ArtifactError) as excinfo:
        pipeline._attempt_registration(
            {"weights": torch.zeros(1)},
            artifact_id=None,
            key=None,
            plan=PlanType.VRAM_LEASED,
            policy_override="pinned",
            options_override=RegisterArtifactOptions(policy="durable"),
        )

    assert excinfo.value.status_code == "INVALID_ARGUMENT"
