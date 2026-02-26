#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import weakref

from tensorcast.api.context import CallContext
from tensorcast.api.plan import Plan, Worker
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.common import canonical_index_from_bytes


def _canonical_index_bytes() -> bytes:
    return b'{"w":[0,4,[1],[1],"torch.float32",0]}'


class _StoreStub:
    closed = False
    _runtime = None


def test_plan_to_spec_is_deterministic() -> None:
    store = _StoreStub()
    canonical_bytes = _canonical_index_bytes()
    artifact = Artifact(
        store_ref=weakref.ref(store),
        artifact_id="mi2:test",
        canonical_index_bytes=canonical_bytes,
        canonical_index=canonical_index_from_bytes(canonical_bytes),
    )
    ctx = CallContext(request_id="req-1", idempotency_key="idem-1")
    plan = Plan(ctx)
    worker = Worker(
        worker_id="worker-1",
        daemon_address="127.0.0.1:50051",
        daemon_id="daemon-1",
    )
    plan.on_worker(worker).prefetch(artifact, device=0)
    spec_a = plan.to_spec()
    spec_b = plan.to_spec()
    assert spec_a.SerializeToString(deterministic=True) == spec_b.SerializeToString(
        deterministic=True
    )
    step = spec_a.steps[0]
    selection = step.action.prefetch.selection
    assert selection.artifact_id == "mi2:test"
    assert selection.logical_layout_hash
    assert selection.selection_hash
    assert selection.view_id == ""


def test_plan_view_selection_hash_populated() -> None:
    store = _StoreStub()
    canonical_bytes = _canonical_index_bytes()
    base = Artifact(
        store_ref=weakref.ref(store),
        artifact_id="mi2:view-test",
        canonical_index_bytes=canonical_bytes,
        canonical_index=canonical_index_from_bytes(canonical_bytes),
    )
    view = base.subset(["w"])
    ctx = CallContext(request_id="req-2")
    plan = Plan(ctx)
    worker = Worker(
        worker_id="worker-2",
        daemon_address="127.0.0.1:50051",
        daemon_id="daemon-2",
    )
    plan.on_worker(worker).prefetch(view, device=0)
    spec = plan.to_spec()
    selection = spec.steps[0].action.prefetch.selection
    assert selection.view_id == ""
    assert selection.view_subset_hash
    assert list(selection.tensor_names) == ["w"]
