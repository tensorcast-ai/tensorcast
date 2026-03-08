#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest

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

    def swap(self, *_: object, **__: object) -> None:
        return None

    def publish_replica(self, *_: object, **__: object) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeBindingPublishFail(_FakeBinding):
    def publish_replica(self, *_: object, **__: object) -> None:
        raise RuntimeError("target_publication_token missing; daemon publish not available")


class _FailingArtifact:
    def bind_into(self, **_: object) -> _FakeBinding:
        raise RuntimeError("artifact id not found")


class _SuccessfulArtifact:
    def bind_into(
        self,
        *,
        target_tensors: dict[str, _FakeTensor],
        **_: object,
    ) -> _FakeBinding:
        return _FakeBinding(target_tensors)


class _SuccessfulArtifactPublishFail:
    def bind_into(
        self,
        *,
        target_tensors: dict[str, _FakeTensor],
        **_: object,
    ) -> _FakeBinding:
        return _FakeBindingPublishFail(target_tensors)


class _BindingCtxCapture:
    def __init__(self) -> None:
        self.tensors: dict[str, _FakeTensor] = {"weight": _FakeTensor(111)}
        self.artifact_id = "artifact-v1"
        self.bind_ctxs: list[object] = []
        self.swap_ctxs: list[object] = []

    def swap(self, artifact: object, **kwargs: object) -> None:
        self.swap_ctxs.append(kwargs.get("ctx"))
        self.artifact_id = str(getattr(artifact, "artifact_id", self.artifact_id))


class _BindingCtxArtifact:
    def __init__(self, *, binding: _BindingCtxCapture, artifact_id: str) -> None:
        self._binding = binding
        self.artifact_id = artifact_id

    def bind(self, **kwargs: object) -> _BindingCtxCapture:
        self._binding.bind_ctxs.append(kwargs.get("ctx"))
        self._binding.artifact_id = self.artifact_id
        return self._binding


def test_tp_bind_retry_reuses_preallocated_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_bindings = {}
    receiver._tp_binding_ptrs = {}
    receiver._tp_pending_targets = {}
    receiver._tp_world_size = 8
    receiver._tp_total_bytes = 320 * 1024**3
    receiver._tp_device_base_index = 0

    alloc_calls = {"count": 0}

    def _alloc(**_: object) -> dict[str, _FakeTensor]:
        alloc_calls["count"] += 1
        return {
            "rank_col_weight_0": _FakeTensor(101),
            "rank_row_weight_0": _FakeTensor(202),
        }

    monkeypatch.setattr(e2e, "_allocate_tp4_rank_targets", _alloc)
    monkeypatch.setattr(e2e, "_build_tp4_rank_copy_plan", lambda **_: ())
    monkeypatch.setattr(e2e, "_validate_tp4_rank_targets", lambda **_: None)

    with pytest.raises(RuntimeError, match="artifact id not found"):
        receiver._apply_tp4_rank(
            version=1,
            rank=0,
            artifact=_FailingArtifact(),
            ctx=None,
        )

    assert alloc_calls["count"] == 1
    assert 0 in receiver._tp_pending_targets

    op, pointer_stable = receiver._apply_tp4_rank(
        version=1,
        rank=0,
        artifact=_SuccessfulArtifact(),
        ctx=None,
    )

    assert op == "bind_into"
    assert pointer_stable is True
    assert alloc_calls["count"] == 1
    assert 0 not in receiver._tp_pending_targets
    assert 0 in receiver._tp_bindings


def test_tp_bind_publish_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_bindings = {}
    receiver._tp_binding_ptrs = {}
    receiver._tp_pending_targets = {}
    receiver._tp_world_size = 4
    receiver._tp_total_bytes = 40 * 1024**3
    receiver._tp_device_base_index = 0

    monkeypatch.setattr(
        e2e,
        "_allocate_tp4_rank_targets",
        lambda **_: {
            "rank_col_weight_0": _FakeTensor(101),
            "rank_row_weight_0": _FakeTensor(202),
        },
    )
    monkeypatch.setattr(e2e, "_build_tp4_rank_copy_plan", lambda **_: ())
    monkeypatch.setattr(e2e, "_validate_tp4_rank_targets", lambda **_: None)

    op, pointer_stable = receiver._apply_tp4_rank(
        version=1,
        rank=0,
        artifact=_SuccessfulArtifactPublishFail(),
        ctx=None,
    )

    assert op == "bind_into"
    assert pointer_stable is True
    captured = capsys.readouterr().out
    assert "publish=skipped" in captured
    assert "target_publication_token missing" in captured


def test_binding_swap_uses_group_ctx_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._per_version_timeout_s = 2.0
    receiver._poll_interval_s = 0.01
    receiver._progress_log_interval_s = 10.0
    receiver._binding = None
    receiver._binding_ptrs = None
    receiver._materialize_device = "cuda:0"
    receiver._payload_mode = "tp_ranked"
    receiver._tp_world_size = 8
    receiver._tp_total_bytes = 40 * 1024**3
    receiver._tp_device_base_index = 0
    receiver._tp_materialize_deadline_s = 600.0
    receiver._transport_group_mode = "tp_version"
    receiver._transport_group_kind = "tp_version"
    receiver._transport_group_namespace = "a1-group:receiver"
    receiver._transport_group_total_parts = 16
    receiver._transport_group_receiver_index = 1
    receiver._transport_group_priority = 0
    receiver._transport_group_epoch = 0

    binding = _BindingCtxCapture()
    artifacts = [
        _BindingCtxArtifact(binding=binding, artifact_id="artifact-v1"),
        _BindingCtxArtifact(binding=binding, artifact_id="artifact-v2"),
    ]

    def _fake_context(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    monkeypatch.setattr(e2e.tc, "context", _fake_context)
    monkeypatch.setattr(e2e, "_validate_payload", lambda **_: None)
    monkeypatch.setattr(
        receiver,
        "_resolve_artifact_for_key",
        lambda *, key: artifacts.pop(0),  # noqa: ARG005
    )

    first = receiver._wait_one_binding(version=1, key="k1", max_version=2)
    second = receiver._wait_one_binding(version=2, key="k2", max_version=2)

    assert first.apply_operation == "bind"
    assert second.apply_operation == "swap"
    assert len(binding.bind_ctxs) == 1
    assert len(binding.swap_ctxs) == 1

    bind_ctx = binding.bind_ctxs[0]
    swap_ctx = binding.swap_ctxs[0]
    assert isinstance(bind_ctx, dict)
    assert isinstance(swap_ctx, dict)

    bind_tags = bind_ctx.get("tags")
    swap_tags = swap_ctx.get("tags")
    assert isinstance(bind_tags, dict)
    assert isinstance(swap_tags, dict)
    assert bind_tags.get("tc.transport.group.kind") == "tp_version"
    assert bind_tags.get("tc.transport.group.id") == "a1-group:receiver:v1"
    assert bind_tags.get("tc.transport.group.part_id") == "rx1:r0"
    assert bind_tags.get("tc.transport.group.total_parts") == 16
    assert swap_tags.get("tc.transport.group.kind") == "tp_version"
    assert swap_tags.get("tc.transport.group.id") == "a1-group:receiver:v2"
    assert swap_tags.get("tc.transport.group.part_id") == "rx1:r0"


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
    receiver._apply_mode = "tp_bind_into_swap"
    receiver._tp_world_size = 1
    receiver._tp_total_bytes = 40 * 1024**3
    receiver._tp_bindings = {}
    receiver._tp_binding_ptrs = {}
    receiver._tp_pending_targets = {}
    receiver._tp_rank_device = lambda _rank: "cuda:0"
    receiver._make_tp_materialize_ctx = lambda **_: None
    receiver._resolve_artifact_for_key = lambda *, key: object()  # noqa: ARG005
    receiver._resolve_artifact_id_for_key = (
        lambda *, key: "artifact-v1"  # noqa: ARG005
    )
    receiver._find_newer_materializable_version = (
        lambda *, version, max_version: 2  # noqa: ARG005
    )
    receiver._find_newer_resolved_version = (
        lambda *, version, max_version: 2  # noqa: ARG005
    )
    receiver._apply_tp4_rank = (
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("Artifact id 'artifact-v1' was not found; region is poisoned")
        )
    )

    with pytest.raises(e2e.VersionDroppedError) as raised:
        receiver._wait_one_tp4_binding(version=1, key="k1", max_version=3)

    assert raised.value.version == 1
    assert raised.value.artifact_id == "artifact-v1"
    assert raised.value.newer_version == 2


def test_tp_bind_group_contract_error_raises_non_retryable_drop() -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._per_version_timeout_s = 2.0
    receiver._poll_interval_s = 0.01
    receiver._progress_log_interval_s = 10.0
    receiver._apply_mode = "tp_bind_into_swap"
    receiver._transport_group_mode = "tp_version"
    receiver._tp_world_size = 1
    receiver._tp_total_bytes = 40 * 1024**3
    receiver._tp_bindings = {}
    receiver._tp_binding_ptrs = {}
    receiver._tp_pending_targets = {}
    receiver._tp_rank_device = lambda _rank: "cuda:0"
    receiver._make_tp_materialize_ctx = lambda **_: None
    receiver._resolve_artifact_for_key = lambda *, key: object()  # noqa: ARG005
    receiver._resolve_artifact_id_for_key = (
        lambda *, key: "artifact-v3"  # noqa: ARG005
    )
    receiver._find_newer_materializable_version = (
        lambda *, version, max_version: None  # noqa: ARG005
    )
    receiver._find_newer_resolved_version = (
        lambda *, version, max_version: None  # noqa: ARG005
    )
    receiver._apply_tp4_rank = (
        lambda **_: (_ for _ in ()).throw(
            RuntimeError(
                "group contract violation: duplicate part_id in transport history"
            )
        )
    )

    with pytest.raises(e2e.VersionDroppedError) as raised:
        receiver._wait_one_tp4_binding(version=3, key="k3", max_version=3)

    assert raised.value.version == 3
    assert raised.value.artifact_id == "artifact-v3"
    assert raised.value.newer_version is None


def test_tp_bind_failure_reset_recycles_rank_targets() -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._tp_bindings = {0: _FakeBinding({"weight": _FakeTensor(777)})}
    receiver._tp_binding_ptrs = {0: {"weight": 777}}
    receiver._tp_pending_targets = {}

    receiver._reset_tp_bindings_after_apply_failure(version=1)

    assert receiver._tp_bindings == {}
    assert receiver._tp_binding_ptrs == {}
    assert 0 in receiver._tp_pending_targets
    recycled = receiver._tp_pending_targets[0]
    assert isinstance(recycled, dict)
    assert isinstance(recycled["weight"], _FakeTensor)
    assert recycled["weight"].data_ptr() == 777


def test_tp_bind_retry_uses_unique_request_id_per_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = e2e.WeightUpdateReceiver.__new__(e2e.WeightUpdateReceiver)
    receiver._per_version_timeout_s = 2.0
    receiver._poll_interval_s = 0.01
    receiver._progress_log_interval_s = 10.0
    receiver._apply_mode = "tp_bind_into_swap"
    receiver._tp_world_size = 1
    receiver._tp_total_bytes = 40 * 1024**3
    receiver._tp_materialize_deadline_s = 600.0
    receiver._tp_bindings = {}
    receiver._tp_binding_ptrs = {}
    receiver._tp_pending_targets = {}
    receiver._tp_rank_device = lambda _rank: "cuda:0"
    receiver._transport_group_mode = "tp_version"
    receiver._transport_group_kind = "tp_version"
    receiver._transport_group_namespace = "suite-group"
    receiver._transport_group_total_parts = 8
    receiver._transport_group_receiver_index = 2
    receiver._transport_group_priority = 0
    receiver._transport_group_epoch = 0
    receiver._resolve_artifact_for_key = lambda *, key: object()  # noqa: ARG005
    receiver._resolve_artifact_id_for_key = (
        lambda *, key: "artifact-v1"  # noqa: ARG005
    )
    receiver._find_newer_materializable_version = (
        lambda *, version, max_version: None  # noqa: ARG005
    )
    receiver._find_newer_resolved_version = (
        lambda *, version, max_version: None  # noqa: ARG005
    )

    request_ids: list[str] = []
    request_attempts: list[int] = []

    def _fake_context(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    monkeypatch.setattr(e2e.tc, "context", _fake_context)

    call_count = {"n": 0}

    def _fake_apply(**kwargs: object) -> tuple[str, bool]:
        ctx = kwargs.get("ctx")
        assert isinstance(ctx, dict)
        tags = ctx.get("tags")
        assert isinstance(tags, dict)
        request_id = tags.get("tc.transport.request_id")
        assert isinstance(request_id, str)
        request_ids.append(request_id)
        request_attempt = tags.get("tc.transport.request_attempt")
        assert isinstance(request_attempt, int)
        request_attempts.append(request_attempt)
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient deadline exceeded")
        receiver._tp_bindings[0] = _FakeBinding({"weight": _FakeTensor(123)})
        receiver._tp_binding_ptrs[0] = {"weight": 123}
        return ("bind_into", True)

    receiver._apply_tp4_rank = _fake_apply

    event = receiver._wait_one_tp4_binding(version=1, key="k1", max_version=3)

    assert event.version == 1
    assert len(request_ids) == 2
    assert request_ids[0] != request_ids[1]
    assert request_attempts == [0, 1]
