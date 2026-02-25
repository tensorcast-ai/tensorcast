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
        raise RuntimeError("target_write_token missing; daemon publish not available")


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
    assert "target_write_token missing" in captured
