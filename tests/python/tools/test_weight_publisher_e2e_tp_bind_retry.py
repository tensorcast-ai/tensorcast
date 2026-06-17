#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest

from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.tools import weight_publisher_e2e as e2e


class _FakeTensor:
    def __init__(self, ptr: int) -> None:
        self._ptr = int(ptr)

    def data_ptr(self) -> int:
        return self._ptr


class _FakeBinding:
    def __init__(self, tensors: dict[str, _FakeTensor]) -> None:
        self.tensors = tensors
        self.artifact_id = "artifact-1"
        self.staged_value = None

    def swap(self, *_: object, **__: object) -> None:
        return None

    def publish_replica(self, *_: object, **__: object) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeStagedBinding(_FakeBinding):
    staged_value = object()

    def __init__(self, tensors: dict[str, _FakeTensor]) -> None:
        super().__init__(tensors)
        self.staged_value = object()
        self.acquire_calls: list[dict[str, object]] = []

    def acquire_staged_value(self, **kwargs: object) -> object:
        self.acquire_calls.append(dict(kwargs))
        return self.staged_value


class _SuccessfulGroupArtifact:
    artifact_id = "artifact-group"

    def bind(self, *_: object, **__: object) -> _FakeStagedBinding:
        return _FakeStagedBinding(
            {
                "rank_col_weight": _FakeTensor(111),
                "rank_row_weight": _FakeTensor(222),
            }
        )


def test_tp_materialize_ctx_uses_group_realization_version_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_world_size = 2
    receiver._tp_materialize_deadline_s = 600.0
    receiver._transport_group_kind = "group_realization_transport"
    receiver._transport_group_namespace = "run-1:receiver"
    receiver._transport_group_total_parts = 4
    receiver._transport_group_receiver_index = 1
    receiver._transport_group_epoch = 9

    def _fake_context(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    monkeypatch.setattr(e2e.tc, "context", _fake_context)
    version_set = e2e.tc.GroupVersionSetRef(
        version_set_id="gvs-1",
        manifest_hash=b"hash",
        manifest_generation=3,
    )

    ctx = receiver._make_tp_materialize_ctx(
        version=7,
        rank=1,
        remaining_s=10.0,
        attempt=2,
        version_set=version_set,
    )

    assert isinstance(ctx, dict)
    group = ctx["group_realization"]
    assert isinstance(group, e2e.tc.GroupRealization)
    assert group.group_kind == "group_realization_transport"
    assert group.group_id == "run-1:receiver:v7"
    assert group.part_id == "rx1:r1"
    assert group.required_part_ids == ("rx0:r0", "rx0:r1", "rx1:r0", "rx1:r1")
    assert group.version_set == version_set
    assert group.require_staged_publish is True
    tags = ctx["tags"]
    assert isinstance(tags, dict)
    assert tags["tc.group_realization.mode"] == "per_part_selection"


def test_tp_materialize_ctx_requires_positive_deadline() -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_materialize_deadline_s = 0.0

    with pytest.raises(ValueError, match="tp_materialize_deadline_s must be > 0"):
        receiver._make_tp_materialize_ctx(
            version=1,
            rank=0,
            remaining_s=1.0,
        )


def test_tp_group_version_set_registration_is_daemon_mediated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._transport_group_namespace = "run-1:receiver"
    receiver._transport_group_total_parts = 4
    receiver._tp_world_size = 2
    receiver._tp_materialize_deadline_s = 10.0
    receiver._group_version_set_refs = {}

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def register_group_version_set(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            response = store_daemon_pb2.RegisterGroupVersionSetResponse()
            response.version_set.version_set_id = "gvs-registered"
            response.version_set.manifest_hash = b"manifest"
            response.version_set.manifest_generation = 1
            response.realization_kind = (
                store_daemon_pb2.GROUP_REALIZATION_KIND_PER_PART_SELECTION
            )
            return response

    class _FakeRuntime:
        def __init__(self, client: _FakeClient) -> None:
            self._client = client

        def ensure_client(self) -> _FakeClient:
            return self._client

    client = _FakeClient()
    monkeypatch.setattr(e2e, "get_store_context", lambda: _FakeRuntime(client))

    ref = receiver._ensure_tp_group_version_set(
        version=7,
        key="model:m:v7",
        artifact_id="mi2:artifact",
    )
    cached = receiver._ensure_tp_group_version_set(
        version=7,
        key="model:m:v7",
        artifact_id="mi2:artifact",
    )

    assert ref == cached
    assert ref.version_set_id == "gvs-registered"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["realization_kind"] == "per_part_selection"
    assert call["namespace"] == "run-1:receiver"
    assert call["key"] == "model:m:v7"
    parts = call["parts"]
    assert isinstance(parts, list)
    assert [part["part_id"] for part in parts] == [
        "rx0:r0",
        "rx0:r1",
        "rx1:r0",
        "rx1:r1",
    ]


def test_group_realization_tp_rank_uses_owned_staged_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_bindings = {}
    receiver._tp_binding_ptrs = {}
    receiver._tp_world_size = 2
    receiver._tp_total_bytes = 0
    receiver._tp_device_base_index = 0
    receiver._materialize_options = object()

    monkeypatch.setattr(e2e, "_build_tp_rank_copy_plan", lambda **_: ("plan",))
    monkeypatch.setattr(e2e, "_validate_tp_rank_targets", lambda **_: None)

    op, pointer_stable = receiver._apply_tp_rank(
        version=1,
        rank=0,
        artifact=_SuccessfulGroupArtifact(),
        ctx=None,
    )

    assert op == "stage"
    assert pointer_stable is True
    assert 0 in receiver._tp_bindings
    binding = receiver._tp_bindings[0]
    assert isinstance(binding, _FakeStagedBinding)


def test_group_realization_tp_acquires_all_staged_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_world_size = 2
    receiver._tp_total_bytes = 40 * 1024**3
    receiver._tp_bindings = {
        0: _FakeStagedBinding({"weight": _FakeTensor(101)}),
        1: _FakeStagedBinding({"weight": _FakeTensor(202)}),
    }

    validate_calls: list[int] = []

    def _validate(**kwargs: object) -> None:
        rank = kwargs.get("rank")
        assert isinstance(rank, int)
        validate_calls.append(rank)

    monkeypatch.setattr(e2e, "_validate_tp_rank_targets", _validate)

    receiver._acquire_tp_group_realization_bindings(
        version=3,
        deadline=e2e.time.monotonic() + 1.0,
    )

    assert validate_calls == [0, 1]
    for binding in receiver._tp_bindings.values():
        assert isinstance(binding, _FakeStagedBinding)
        assert binding.acquire_calls
        assert binding.acquire_calls[0]["wait_for_publish"] is True


def test_resolve_tp_rank_devices_auto_modulo_fallback() -> None:
    mapping, mode = e2e._resolve_tp_rank_devices(
        tp_world_size=4,
        tp_device_base_index=0,
        visible_device_count=1,
        map_policy="auto",
    )
    assert mode == "modulo"
    assert mapping == {
        0: "cuda:0",
        1: "cuda:0",
        2: "cuda:0",
        3: "cuda:0",
    }


def test_resolve_tp_rank_devices_strict_requires_contiguous() -> None:
    with pytest.raises(ValueError, match="strict device mapping requires enough"):
        e2e._resolve_tp_rank_devices(
            tp_world_size=4,
            tp_device_base_index=0,
            visible_device_count=1,
            map_policy="strict",
        )


def test_tp_bind_poisoned_error_raises_version_dropped() -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._per_version_timeout_s = 2.0
    receiver._poll_interval_s = 0.01
    receiver._progress_log_interval_s = 10.0
    receiver._tp_world_size = 1
    receiver._tp_total_bytes = 40 * 1024**3
    receiver._tp_bindings = {0: _FakeBinding({"weight": _FakeTensor(101)})}
    receiver._tp_binding_ptrs = {0: {"weight": 101}}
    receiver._tp_rank_device = lambda _rank: "cuda:0"
    receiver._make_tp_materialize_ctx = lambda **_: None
    receiver._resolve_artifact_for_key = (
        lambda *, key: _SuccessfulGroupArtifact()  # noqa: ARG005
    )
    receiver._ensure_tp_group_version_set = lambda **_: object()
    receiver._resolve_artifact_id_for_key = (
        lambda *, key: "artifact-v1"  # noqa: ARG005
    )
    receiver._find_newer_materializable_version = (
        lambda *, version, max_version: 2  # noqa: ARG005
    )
    receiver._find_newer_resolved_version = (
        lambda *, version, max_version: 2  # noqa: ARG005
    )
    receiver._apply_tp_rank = lambda **_: (_ for _ in ()).throw(
        RuntimeError("Artifact id 'artifact-v1' was not found; region is poisoned")
    )

    with pytest.raises(e2e.VersionDroppedError) as raised:
        receiver._wait_one_tp_binding(version=1, key="k1", max_version=3)

    assert raised.value.version == 1
    assert raised.value.artifact_id == "artifact-v1"
    assert raised.value.newer_version == 2
    assert receiver._tp_bindings == {}
    assert receiver._tp_binding_ptrs == {}


def test_tp_bind_failure_reset_closes_staged_ranks() -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_bindings = {0: _FakeBinding({"weight": _FakeTensor(777)})}
    receiver._tp_binding_ptrs = {0: {"weight": 777}}

    receiver._reset_tp_bindings_after_apply_failure(version=1)

    assert receiver._tp_bindings == {}
    assert receiver._tp_binding_ptrs == {}


def test_tp_bind_rank_reset_closes_staged_binding() -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_bindings = {0: _FakeBinding({"weight": _FakeTensor(808)})}
    receiver._tp_binding_ptrs = {0: {"weight": 808}}

    receiver._reset_tp_rank_after_apply_failure(
        version=1,
        rank=0,
        clear_pending_target=True,
    )

    assert receiver._tp_bindings == {}
    assert receiver._tp_binding_ptrs == {}


def test_tp_rank_attempt_timeout_retry_boosts_tail_rank() -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_world_size = 4
    receiver._tp_total_bytes = 40 * 1024**3

    first_attempt = receiver._tp_rank_attempt_timeout_s(
        remaining_s=500.0,
        attempt=0,
        completed_ranks_count=3,
    )
    retry_tail = receiver._tp_rank_attempt_timeout_s(
        remaining_s=500.0,
        attempt=1,
        completed_ranks_count=3,
    )
    retry_capped = receiver._tp_rank_attempt_timeout_s(
        remaining_s=150.0,
        attempt=1,
        completed_ranks_count=3,
    )

    assert first_attempt == pytest.approx(128.0)
    assert retry_tail > first_attempt
    assert retry_capped == pytest.approx(150.0)
