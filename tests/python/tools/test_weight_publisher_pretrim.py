#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig


class _FakeRegistered(SimpleNamespace):
    artifact_id: str
    persistence_task_id: str | None


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
) -> WeightPublisher:
    cfg = WeightPublisherConfig(
        model_name="test-model",
        keep_last=2,
        history_path=history_path,
        policy=None,
        use_cgid=False,
        trigger_reload=False,
        verify_key_mapping=False,
        wait_persistence=False,
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
