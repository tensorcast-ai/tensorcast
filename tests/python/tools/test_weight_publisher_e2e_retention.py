#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

from tensorcast.tools.weight_publisher_e2e import (
    TP_FULL_VALIDATION_MAX_BYTES,
    PublishEvent,
    WeightUpdatePublisher,
)


def _build_event(version: int) -> PublishEvent:
    return PublishEvent(
        version=version,
        key=f"model:test:v{version}",
        artifact_id=f"artifact-{version}",
        export_dir="in_memory://cpu",
        publish_device="cpu",
        published_at_s=0.0,
        publish_latency_s=0.0,
    )


def _make_publisher(*, keep_last: int, strict: bool, tp_total_bytes: int):
    publisher = WeightUpdatePublisher.__new__(WeightUpdatePublisher)
    publisher._config = SimpleNamespace(keep_last=keep_last)
    publisher._strict_drop_check = strict
    publisher._tp_total_bytes = tp_total_bytes
    publisher._payload_mode = "tp_ranked"
    return publisher


def test_verify_retention_window_skips_heavy_materialize_for_large_tp_payload() -> None:
    publisher = _make_publisher(
        keep_last=2,
        strict=False,
        tp_total_bytes=64 * 1024**3,
    )

    key_mapping_calls: list[tuple[str, str]] = []
    materialization_calls: list[tuple[str, int, bool]] = []
    registry_calls: list[tuple[str, bool]] = []

    publisher._wait_key_mapping_state = (
        lambda *, key, expected_artifact_id: key_mapping_calls.append(
            (key, expected_artifact_id)
        )
    )
    publisher._wait_materialization_state = (
        lambda *, key, version, expected_materializable: materialization_calls.append(
            (key, version, expected_materializable)
        )
    )
    publisher._wait_artifact_registry_state = (
        lambda *, artifact_id, expected_exists: registry_calls.append(
            (artifact_id, expected_exists)
        )
    )

    events = [_build_event(1), _build_event(2), _build_event(3)]
    publisher._verify_retention_window(events=events)

    assert len(key_mapping_calls) == 3
    assert materialization_calls == []
    assert registry_calls == [
        ("artifact-2", True),
        ("artifact-3", True),
    ]


def test_verify_retention_window_keeps_strict_probe_for_large_tp_payload() -> None:
    publisher = _make_publisher(
        keep_last=2,
        strict=True,
        tp_total_bytes=64 * 1024**3,
    )

    key_mapping_calls: list[tuple[str, str]] = []
    materialization_calls: list[tuple[str, int, bool]] = []
    registry_calls: list[tuple[str, bool]] = []

    publisher._wait_key_mapping_state = (
        lambda *, key, expected_artifact_id: key_mapping_calls.append(
            (key, expected_artifact_id)
        )
    )
    publisher._wait_materialization_state = (
        lambda *, key, version, expected_materializable: materialization_calls.append(
            (key, version, expected_materializable)
        )
    )
    publisher._wait_artifact_registry_state = (
        lambda *, artifact_id, expected_exists: registry_calls.append(
            (artifact_id, expected_exists)
        )
    )

    events = [_build_event(1), _build_event(2), _build_event(3)]
    publisher._verify_retention_window(events=events)

    assert len(key_mapping_calls) == 3
    assert materialization_calls == [
        ("model:test:v1", 1, False),
        ("model:test:v2", 2, True),
        ("model:test:v3", 3, True),
    ]
    assert registry_calls == [
        ("artifact-1", False),
        ("artifact-2", True),
        ("artifact-3", True),
    ]


def test_verify_retention_window_small_payload_still_materializes() -> None:
    publisher = _make_publisher(
        keep_last=1,
        strict=False,
        tp_total_bytes=TP_FULL_VALIDATION_MAX_BYTES,
    )

    materialization_calls: list[tuple[str, int, bool]] = []
    registry_calls: list[tuple[str, bool]] = []
    publisher._wait_key_mapping_state = lambda **_: None
    publisher._wait_materialization_state = (
        lambda *, key, version, expected_materializable: materialization_calls.append(
            (key, version, expected_materializable)
        )
    )
    publisher._wait_artifact_registry_state = (
        lambda *, artifact_id, expected_exists: registry_calls.append(
            (artifact_id, expected_exists)
        )
    )

    events = [_build_event(1)]
    publisher._verify_retention_window(events=events)

    assert materialization_calls == [("model:test:v1", 1, True)]
    assert registry_calls == [("artifact-1", True)]


def test_publish_one_version_releases_source_before_retention(
    monkeypatch,
) -> None:
    publisher = WeightUpdatePublisher.__new__(WeightUpdatePublisher)
    publisher._model_name = "test"
    publisher._key_template = "model:{model_name}:v{weight_version}"
    publisher._publish_device = "cpu"
    publisher._payload_mode = "tp_ranked"
    publisher._tp_world_size = 4
    publisher._tp_total_bytes = 8 * 1024**3

    class _FakeInnerPublisher:
        def publish(self, tensors, *, version: int) -> str:  # noqa: ANN001
            _ = (tensors, version)
            return "artifact-1"

    publisher._publisher = _FakeInnerPublisher()

    released = {"value": False}

    class _TrackedTensorDict(dict):
        def __del__(self) -> None:
            released["value"] = True

    monkeypatch.setattr(
        "tensorcast.tools.weight_publisher_e2e._build_publish_tensors",
        lambda **_: _TrackedTensorDict(),
    )
    monkeypatch.setattr(
        "tensorcast.tools.weight_publisher_e2e._publish_memory_log",
        lambda **_: None,
    )

    retention_checked = {"value": False}

    def _verify_retention_window(*, events):
        _ = events
        retention_checked["value"] = True
        assert released["value"] is True

    publisher._verify_retention_window = _verify_retention_window

    events: list[PublishEvent] = []
    event = publisher.publish_one_version(version=1, events=events)

    assert retention_checked["value"] is True
    assert released["value"] is True
    assert event.artifact_id == "artifact-1"
    assert [e.version for e in events] == [1]
