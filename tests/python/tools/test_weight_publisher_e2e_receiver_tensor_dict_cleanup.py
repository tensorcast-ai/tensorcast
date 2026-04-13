#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tensorcast.tools import weight_publisher_e2e as e2e


class _FakeClient:
    def __init__(self, responses: list[bool]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def unload_replica(self, replica_uuid: str, *, disk_path: str = "") -> bool:
        self.calls.append((str(replica_uuid), str(disk_path)))
        if self._responses:
            return bool(self._responses.pop(0))
        return False


class _FakeStoreContext:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def ensure_client(self) -> _FakeClient:
        return self._client


class _FakeArtifact:
    artifact_id = "artifact-1"

    def __init__(
        self,
        *,
        tensors: dict[str, Any],
        replica_uuid: str,
        disk_path: str,
    ) -> None:
        self._result = SimpleNamespace(
            tensors=tensors,
            diagnostics=SimpleNamespace(
                replica_uuid=str(replica_uuid),
                disk_path=str(disk_path),
            ),
        )

    def tensor_dict_with_diagnostics(
        self,
        *,
        device: str,
        options: Any | None = None,
    ) -> Any:
        _ = device
        _ = options
        return self._result


def _build_receiver_for_tensor_dict(
    artifact: _FakeArtifact,
) -> e2e.WeightUpdateReceiver:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._per_version_timeout_s = 1.0
    receiver._poll_interval_s = 0.0
    receiver._materialize_device = "cpu"
    receiver._payload_mode = "probe"
    receiver._tp_world_size = 1
    receiver._tp_total_bytes = 0
    receiver._resolve_artifact_for_key = lambda *, key: artifact
    receiver._maybe_log_wait_progress = lambda **kwargs: kwargs["next_log_at"]
    receiver._resolve_artifact_id_for_key = lambda *, key: artifact.artifact_id
    receiver._find_newer_materializable_version = lambda **kwargs: None
    return receiver


def test_wait_one_tensor_dict_releases_replica_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensors = {"weight": object()}
    artifact = _FakeArtifact(
        tensors=tensors,
        replica_uuid="replica-v1",
        disk_path="",
    )
    receiver = _build_receiver_for_tensor_dict(artifact)
    client = _FakeClient([True])
    monkeypatch.setattr(e2e, "_validate_payload", lambda **_: None)
    monkeypatch.setattr(e2e, "get_store_context", lambda: _FakeStoreContext(client))

    event = receiver._wait_one_tensor_dict(
        version=1,
        key="model:test:v1",
        max_version=1,
    )

    assert event.version == 1
    assert event.artifact_id == "artifact-1"
    assert client.calls == [("replica-v1", "")]
    assert tensors == {}


def test_release_tensor_dict_replica_raises_when_unload_never_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    client = _FakeClient([False, False, False])
    monkeypatch.setattr(e2e, "get_store_context", lambda: _FakeStoreContext(client))
    monkeypatch.setattr(e2e.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="failed to unload tensor_dict replica"):
        receiver._release_tensor_dict_replica_after_apply(
            replica_uuid="replica-v2",
            disk_path="",
            tensors={"weight": object()},
        )
    assert len(client.calls) == e2e.RECEIVER_TENSOR_DICT_UNLOAD_RETRIES
