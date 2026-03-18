#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tensorcast.api.operation import OperationStatus
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig


class _FakeRegistered(SimpleNamespace):
    artifact_id: str
    persistence_task_id: str | None


class _FakeOperation:
    def __init__(self, states: list[str]) -> None:
        self._states = list(states)
        self.calls = 0

    def status(self) -> OperationStatus:
        index = min(self.calls, len(self._states) - 1)
        self.calls += 1
        return OperationStatus(state=self._states[index])  # type: ignore[arg-type]


class _FakeOutcome(SimpleNamespace):
    drained: bool
    removed: bool
    released_region_ids: list[str]
    message: str


def _write_history(path, items: list[tuple[int, str]]) -> None:
    payload = [{"version": int(v), "artifact_id": aid} for v, aid in items]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_history(path) -> list[tuple[int, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [(int(item["version"]), str(item["artifact_id"])) for item in raw]


def _build_publisher(
    *,
    history_path: str,
    wait_persistence: bool = False,
) -> WeightPublisher:
    cfg = WeightPublisherConfig(
        model_name="test-model",
        keep_last=2,
        history_path=history_path,
        policy=None,
        use_cgid=False,
        trigger_reload=False,
        verify_key_mapping=False,
        wait_persistence=wait_persistence,
    )
    return WeightPublisher(cfg)


def test_pre_publish_trim_runs_before_put(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    history = tmp_path / "history.json"
    _write_history(history, [(1, "artifact-1"), (2, "artifact-2")])
    publisher = _build_publisher(
        history_path=str(history),
    )

    calls: list[str] = []

    def fake_put(*args, **kwargs):
        calls.append("put")
        return _FakeRegistered(artifact_id="artifact-3", persistence_task_id=None)

    def fake_deregister(artifact_id: str, **kwargs):
        calls.append(f"deregister:{artifact_id}")
        return _FakeOutcome(
            drained=True,
            removed=True,
            released_region_ids=[],
            message="ok",
        )

    monkeypatch.setattr("tensorcast.tools.weight_publisher.tensorcast.put", fake_put)
    monkeypatch.setattr(
        "tensorcast.tools.weight_publisher.tensorcast.deregister_artifact",
        fake_deregister,
    )

    artifact_id = publisher.publish(tensors={}, version=3)

    assert artifact_id == "artifact-3"
    assert calls == ["deregister:artifact-1", "put"]
    assert _read_history(history) == [(3, "artifact-3"), (2, "artifact-2")]


def test_publish_waits_for_persistence_via_operation_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    history = tmp_path / "history.json"
    publisher = _build_publisher(
        history_path=str(history),
        wait_persistence=True,
    )

    operation = _FakeOperation(["running", "success"])

    class _Registered(SimpleNamespace):
        artifact_id: str
        persistence_task_id: str | None

        def persistence_operation(self):
            return operation

    def fake_put(*args, **kwargs):
        return _Registered(artifact_id="artifact-4", persistence_task_id="task-4")

    def fail_query(*args, **kwargs):
        raise AssertionError("raw query_persistence_status should not be used")

    monkeypatch.setattr("tensorcast.tools.weight_publisher.tensorcast.put", fake_put)
    monkeypatch.setattr(
        "tensorcast.tools.weight_publisher.tensorcast.query_persistence_status",
        fail_query,
    )

    artifact_id = publisher.publish(tensors={}, version=4)

    assert artifact_id == "artifact-4"
    assert operation.calls >= 2


def test_publish_waits_for_persistence_via_top_level_fallback_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    history = tmp_path / "history.json"
    publisher = _build_publisher(
        history_path=str(history),
        wait_persistence=True,
    )

    operation = _FakeOperation(["running", "success"])

    def fake_put(*args, **kwargs):
        return _FakeRegistered(artifact_id="artifact-5", persistence_task_id="task-5")

    monkeypatch.setattr("tensorcast.tools.weight_publisher.tensorcast.put", fake_put)
    monkeypatch.setattr(
        "tensorcast.tools.weight_publisher.tensorcast.persistence_operation",
        lambda **kwargs: operation,
    )

    artifact_id = publisher.publish(tensors={}, version=5)

    assert artifact_id == "artifact-5"
    assert operation.calls >= 2
