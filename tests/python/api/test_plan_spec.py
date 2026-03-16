#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import weakref

from tensorcast.api.context import CallContext
from tensorcast.api.plan import Instance, Plan, PlanResult, Worker
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.engine_adapter.kvcache_adapter import (
    BatchResult,
    HydrateResult,
    ManifestResult,
    PublishResult,
)
from tensorcast.proto.node_agent.v1 import node_agent_pb2


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


def test_plan_publish_serializes_canonical_action() -> None:
    ctx = CallContext(request_id="req-cache", idempotency_key="idem-cache")
    plan = Plan(ctx)
    inst = Instance(instance_id="inst-a", worker_id="worker-a", engine="sglang")

    step_first = plan.on_instance(inst).publish(
        engine_request_id="rid-123",
        ttl_ms=60_000,
    )
    step_second = plan.on_instance(inst).publish(
        engine_request_id="rid-123",
        ttl_ms=60_000,
        depends_on=[step_first],
    )

    spec = plan.to_spec()
    assert len(spec.steps) == 2
    first_step = spec.steps[0]
    second_step = spec.steps[1]
    assert first_step.step_id == step_first.step_id
    assert second_step.step_id == step_second.step_id
    assert first_step.action.WhichOneof("kind") == "publish"
    assert second_step.action.WhichOneof("kind") == "publish"
    assert first_step.action.publish.engine_request_id == "rid-123"
    assert int(first_step.action.publish.ttl_ms) == 60_000


def test_plan_result_decodes_node_agent_artifact_results() -> None:
    response = node_agent_pb2.ExecutePlanResponse(
        request_id="req-node-agent",
        ok=True,
    )

    manifest_step = response.steps.add(
        step_id="s1",
        target_id="inst-a",
        action="manifest",
    )
    manifest_step.status.state = node_agent_pb2.OPERATION_STATE_SUCCESS
    manifest_step.status.message = "manifest completed"
    manifest_step.artifact_result.manifest.engine_request_id = "rid-123"
    manifest_step.artifact_result.manifest.layout_id = "layout-v1"
    manifest_step.artifact_result.manifest.artifact_ids.append(
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~layout-v1~b64u.azE"
    )
    manifest_step.artifact_result.manifest.key_set_digest_alg = "sha256"
    manifest_step.artifact_result.manifest.key_set_digest_hex = "abc123"

    publish_step = response.steps.add(
        step_id="s2",
        target_id="inst-a",
        action="publish",
    )
    publish_step.status.state = node_agent_pb2.OPERATION_STATE_SUCCESS
    publish_step.artifact_result.publish.manifest.CopyFrom(
        manifest_step.artifact_result.manifest
    )
    publish_outcome = publish_step.artifact_result.publish.put_outcomes.add()
    publish_outcome.artifact_id = (
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~layout-v1~b64u.azE"
    )
    publish_outcome.status_code = "OK"
    publish_outcome.message = "created"

    hydrate_step = response.steps.add(
        step_id="s3",
        target_id="inst-a",
        action="hydrate",
    )
    hydrate_step.status.state = node_agent_pb2.OPERATION_STATE_DEGRADED
    hydrate_step.status.message = "partial hydrate"
    hydrate_step.artifact_result.hydrate.manifest.CopyFrom(
        manifest_step.artifact_result.manifest
    )
    hydrate_outcome = hydrate_step.artifact_result.hydrate.get_outcomes.add()
    hydrate_outcome.artifact_id = (
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~layout-v1~b64u.azE"
    )
    hydrate_outcome.status_code = "MISS"
    hydrate_step.artifact_result.hydrate.missing_artifact_ids.append(
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~layout-v1~b64u.azI"
    )

    evict_step = response.steps.add(
        step_id="s4",
        target_id="inst-a",
        action="evict_local",
    )
    evict_step.status.state = node_agent_pb2.OPERATION_STATE_SUCCESS
    evict_step.artifact_result.evict_local.engine_request_id = "rid-123"
    evict_outcome = evict_step.artifact_result.evict_local.outcomes.add()
    evict_outcome.artifact_id = (
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~layout-v1~b64u.azE"
    )
    evict_outcome.status_code = "OK"

    result = PlanResult.from_node_agent_response(response)

    assert result.ok is True
    manifest_result = result.steps["s1"].artifact_result
    assert isinstance(manifest_result, ManifestResult)
    assert manifest_result.engine_request_id == "rid-123"
    publish_result = result.steps["s2"].artifact_result
    assert isinstance(publish_result, PublishResult)
    assert publish_result.put_outcomes[0].message == "created"
    hydrate_result = result.steps["s3"].artifact_result
    assert isinstance(hydrate_result, HydrateResult)
    assert hydrate_result.missing_artifact_ids == (
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~layout-v1~b64u.azI",
    )
    evict_result = result.steps["s4"].artifact_result
    assert isinstance(evict_result, BatchResult)
    assert evict_result.engine_request_id == "rid-123"
