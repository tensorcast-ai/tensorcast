#  Copyright (c) 2025-2026, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tensorcast.api._config import PlanType, RegisterArtifactOptions
from tensorcast.api.store import Store
from tensorcast.api.store.registration import RegistrationPipeline
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.types import ArtifactError
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
        self.executor = _DummyExecutor()
        self.tracked_futures: list[object] = []
        self.tracked_leases: list[object] = []

    def ensure_client(self):
        return self.client

    def track_future(self, future) -> None:  # pragma: no cover - not used
        self.tracked_futures.append(future)

    def track_lease(self, lease) -> None:  # pragma: no cover - not used
        self.tracked_leases.append(lease)


class _ClientStub:
    def __init__(self, *, response=None, query_response=None, exc: Exception | None = None):
        self.response = response
        self.query_response = query_response
        self.exc = exc
        self.start_calls: list[tuple[str, object | None]] = []
        self.query_calls: list[tuple[str | None, str | None]] = []

    def start_persistence(self, *, artifact_id: str, policy=None):
        self.start_calls.append((artifact_id, policy))
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
        self, *, task_id: str | None = None, artifact_id: str | None = None
    ):
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
    assert client.start_calls == [("artifact-7", options.policy)]


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
    assert client.start_calls == [("artifact-9", options.policy)]


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
