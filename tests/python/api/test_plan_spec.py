#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import weakref

from tensorcast.api.context import CallContext, GovernanceContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.plan import (
    ARTIFACT_SET_CARRIER_INLINE,
    ARTIFACT_SET_CARRIER_MANIFEST_BACKED,
    ArtifactSetRef,
    Instance,
    Plan,
    PlanResult,
    Worker,
)
from tensorcast.api.plan.artifact_set import resolve_artifact_set_ref
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.engine_adapter.artifact_api import (
    BatchResult,
    HydrateResult,
    ManifestArtifactSetBridge,
    ManifestResult,
    PublishResult,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.node_agent.v1 import node_agent_pb2
from tensorcast.proto.plan.v1 import plan_pb2


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
    assert selection.view_id
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


def test_prefetch_many_lowers_to_inline_artifact_set_ref() -> None:
    store = _StoreStub()
    canonical_bytes = _canonical_index_bytes()
    artifact_a = Artifact(
        store_ref=weakref.ref(store),
        artifact_id="mi2:a",
        canonical_index_bytes=canonical_bytes,
        canonical_index=canonical_index_from_bytes(canonical_bytes),
    )
    artifact_b = Artifact(
        store_ref=weakref.ref(store),
        artifact_id="mi2:b",
        canonical_index_bytes=canonical_bytes,
        canonical_index=canonical_index_from_bytes(canonical_bytes),
    )
    ctx = CallContext(request_id="req-set", idempotency_key="idem-set")
    worker = Worker(
        worker_id="worker-set",
        daemon_address="127.0.0.1:50051",
        daemon_id="daemon-set",
    )

    lowered_plan = Plan(ctx)
    lowered_plan.on_worker(worker).prefetch_many(
        [artifact_b, artifact_a, artifact_b], device=0
    )
    lowered_spec = lowered_plan.to_spec()

    explicit_set = ArtifactSetRef.inline(
        (
            artifact_a._build_artifact_selection(),
            artifact_b._build_artifact_selection(),
        )
    )
    explicit_plan = Plan(ctx)
    explicit_plan.on_worker(worker).prefetch_set(explicit_set, device=0)
    explicit_spec = explicit_plan.to_spec()

    lowered_action = lowered_spec.steps[0].action.prefetch_set
    explicit_action = explicit_spec.steps[0].action.prefetch_set
    assert lowered_spec.steps[0].action.WhichOneof("kind") == "prefetch_set"
    assert explicit_spec.steps[0].action.WhichOneof("kind") == "prefetch_set"
    assert lowered_action.artifact_set.carrier_form == ARTIFACT_SET_CARRIER_INLINE
    assert (
        lowered_action.artifact_set.set_digest_hex
        == explicit_action.artifact_set.set_digest_hex
    )
    assert int(lowered_action.artifact_set.item_count) == 2
    assert lowered_action.artifact_set.SerializeToString(
        deterministic=True
    ) == explicit_action.artifact_set.SerializeToString(deterministic=True)


def test_prefetch_manifest_result_preserves_explicit_bridge() -> None:
    manifest_result = ManifestResult.from_artifact_selections(
        engine_request_id="rid-bridge",
        layout_id="layout-v1",
        manifest_selection=common_pb2.ArtifactSelection(
            artifact_id="engine-manifest:rid-bridge",
            logical_layout_hash=b"manifest-logical",
            selection_hash=b"manifest-selection",
        ),
        artifact_selections=(
            common_pb2.ArtifactSelection(
                artifact_id="mi2:a",
                logical_layout_hash=b"logical-a",
                selection_hash=b"selection-a",
            ),
            common_pb2.ArtifactSelection(
                artifact_id="mi2:b",
                logical_layout_hash=b"logical-b",
                selection_hash=b"selection-b",
            ),
        ),
    )
    ctx = CallContext(request_id="req-manifest-bridge", idempotency_key="idem-bridge")
    worker = Worker(
        worker_id="worker-bridge",
        daemon_address="127.0.0.1:50051",
        daemon_id="daemon-bridge",
    )
    plan = Plan(ctx)

    plan.on_worker(worker).prefetch_manifest_result(manifest_result, device=0)
    spec = plan.to_spec()

    action = spec.steps[0].action.prefetch_set
    assert action.artifact_set.carrier_form == ARTIFACT_SET_CARRIER_MANIFEST_BACKED
    assert action.HasField("manifest_bridge")
    assert action.manifest_bridge.artifact_set_ref.SerializeToString(
        deterministic=True
    ) == action.artifact_set.SerializeToString(deterministic=True)


def test_plan_spec_serializes_typed_governance_context() -> None:
    ctx = CallContext(
        request_id="req-governance",
        governance=GovernanceContext(
            lane="lane-a",
            policy_version=7,
            staleness_budget_ms=250,
        ),
    )
    plan = Plan(ctx)
    spec = plan.to_spec()

    assert spec.governance.lane == "lane-a"
    assert int(spec.governance.policy_version) == 7
    assert int(spec.governance.staleness_budget_ms) == 250


def test_plan_proto_reserves_cluster_transport_slot() -> None:
    assert int(Plan(CallContext(request_id="req")).to_spec().steps.__len__()) == 0
    assert plan_pb2.TARGET_TYPE_CLUSTER > plan_pb2.TARGET_TYPE_INSTANCE
    action = plan_pb2.PlanAction(
        cluster_action=plan_pb2.ClusterActionRef(action_ref="wf:activate")
    )
    assert action.WhichOneof("kind") == "cluster_action"


def test_local_plan_run_fails_closed_on_instance_steps_without_node_agent_bridge() -> None:
    ctx = CallContext(request_id="req-instance-local")
    plan = Plan(ctx)
    inst = Instance(instance_id="inst-a", worker_id="worker-a", engine="sglang")
    step = plan.on_instance(inst).manifest(engine_request_id="rid-123")

    result = plan.run(raise_on_error=False)

    assert result.ok is False
    step_result = result.step(step)
    assert step_result.action == "manifest"
    assert step_result.status.state == "failed"
    assert "Node Agent" in step_result.status.message


def test_artifact_set_ref_resolution_fails_closed_on_mismatch() -> None:
    selection = common_pb2.ArtifactSelection(
        artifact_id="mi2:test",
        logical_layout_hash=b"logical",
        selection_hash=b"selection",
    )
    artifact_set = ArtifactSetRef(
        set_digest_hex="0" * 64,
        item_count=1,
        carrier_form=ARTIFACT_SET_CARRIER_INLINE,
        inline_items=(selection,),
    )

    try:
        resolve_artifact_set_ref(artifact_set)
    except ArtifactError as exc:
        assert exc.status_code == "FAILED_PRECONDITION"
    else:
        raise AssertionError("expected ArtifactSetRef mismatch to fail closed")


def test_manifest_backed_artifact_set_ref_fails_closed_without_owner_resolver() -> None:
    selection = common_pb2.ArtifactSelection(
        artifact_id="mi2:manifest",
        logical_layout_hash=b"logical",
        selection_hash=b"selection",
    )
    artifact_set = ArtifactSetRef.manifest_backed(
        set_digest_hex="1" * 64,
        item_count=7,
        manifest_selection=selection,
    )

    try:
        resolve_artifact_set_ref(artifact_set)
    except ArtifactError as exc:
        assert exc.status_code == "FAILED_PRECONDITION"
    else:
        raise AssertionError("expected manifest_backed ArtifactSetRef to fail closed")


def test_manifest_backed_artifact_set_ref_fails_closed_on_bridge_digest_mismatch() -> (
    None
):
    manifest_result = ManifestResult.from_artifact_selections(
        engine_request_id="rid-bridge",
        layout_id="layout-v1",
        manifest_selection=common_pb2.ArtifactSelection(
            artifact_id="engine-manifest:rid-bridge",
            logical_layout_hash=b"manifest-logical",
            selection_hash=b"manifest-selection",
        ),
        artifact_selections=(
            common_pb2.ArtifactSelection(
                artifact_id="mi2:a",
                logical_layout_hash=b"logical-a",
                selection_hash=b"selection-a",
            ),
        ),
    )
    bridge = manifest_result.require_artifact_set_bridge()
    mismatched_bridge = ManifestArtifactSetBridge(
        bridge_schema=bridge.bridge_schema,
        bridge_version=bridge.bridge_version,
        artifact_set_ref=bridge.artifact_set_ref,
        resolved_items=bridge.resolved_items
        + (
            common_pb2.ArtifactSelection(
                artifact_id="mi2:b",
                logical_layout_hash=b"logical-b",
                selection_hash=b"selection-b",
            ),
        ),
    )

    try:
        resolve_artifact_set_ref(
            bridge.artifact_set_ref,
            manifest_resolver=mismatched_bridge.resolve_artifact_set,
        )
    except ArtifactError as exc:
        assert exc.status_code == "FAILED_PRECONDITION"
    else:
        raise AssertionError(
            "expected ManifestArtifactSetBridge mismatch to fail closed"
        )


def test_plan_result_decodes_node_agent_artifact_results() -> None:
    manifest_result = ManifestResult.from_artifact_selections(
        engine_request_id="rid-123",
        layout_id="layout-v1",
        manifest_selection=common_pb2.ArtifactSelection(
            artifact_id="engine-manifest:rid-123",
            logical_layout_hash=b"manifest-logical",
            selection_hash=b"manifest-selection",
        ),
        artifact_selections=(
            common_pb2.ArtifactSelection(
                artifact_id="cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",
                logical_layout_hash=b"logical-a",
                selection_hash=b"selection-a",
            ),
        ),
    )
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
    manifest_step.artifact_result.manifest.engine_request_id = (
        manifest_result.engine_request_id
    )
    manifest_step.artifact_result.manifest.layout_id = manifest_result.layout_id
    manifest_step.artifact_result.manifest.artifact_ids.extend(
        manifest_result.artifact_ids
    )
    manifest_step.artifact_result.manifest.key_set_digest_alg = (
        manifest_result.key_set_digest_alg
    )
    manifest_step.artifact_result.manifest.key_set_digest_hex = (
        manifest_result.key_set_digest_hex
    )
    manifest_step.artifact_result.manifest.manifest_bridge.CopyFrom(
        manifest_result.require_artifact_set_bridge().to_proto()
    )

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
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE"
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
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE"
    )
    hydrate_outcome.status_code = "MISS"
    hydrate_step.artifact_result.hydrate.missing_artifact_ids.append(
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azI"
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
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE"
    )
    evict_outcome.status_code = "OK"

    result = PlanResult.from_node_agent_response(response)

    assert result.ok is True
    manifest_result = result.steps["s1"].artifact_result
    assert isinstance(manifest_result, ManifestResult)
    assert manifest_result.engine_request_id == "rid-123"
    assert manifest_result.artifact_set_bridge is not None
    assert (
        manifest_result.artifact_set_bridge.artifact_set_ref.carrier_form
        == ARTIFACT_SET_CARRIER_MANIFEST_BACKED
    )
    publish_result = result.steps["s2"].artifact_result
    assert isinstance(publish_result, PublishResult)
    assert publish_result.put_outcomes[0].message == "created"
    hydrate_result = result.steps["s3"].artifact_result
    assert isinstance(hydrate_result, HydrateResult)
    assert hydrate_result.missing_artifact_ids == (
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azI",
    )
    evict_result = result.steps["s4"].artifact_result
    assert isinstance(evict_result, BatchResult)
    assert evict_result.engine_request_id == "rid-123"


def test_plan_result_decodes_node_agent_artifact_set_results() -> None:
    selection_a = common_pb2.ArtifactSelection(
        artifact_id="mi2:a",
        logical_layout_hash=b"logical-a",
        selection_hash=b"selection-a",
    )
    selection_b = common_pb2.ArtifactSelection(
        artifact_id="mi2:b",
        logical_layout_hash=b"logical-b",
        selection_hash=b"selection-b",
    )
    artifact_set = ArtifactSetRef.inline((selection_b, selection_a))

    response = node_agent_pb2.ExecutePlanResponse(
        request_id="req-node-agent-set",
        ok=True,
    )
    step = response.steps.add(
        step_id="s-set",
        target_id="daemon-a",
        action="prefetch_set",
    )
    step.status.state = node_agent_pb2.OPERATION_STATE_SUCCESS
    step.status.message = "prefetch_set completed"
    step.artifact_set_result.set_digest_hex = artifact_set.set_digest_hex
    for selection in artifact_set.inline_items or ():
        outcome = step.artifact_set_result.outcomes.add()
        outcome.item_identity.artifact_id = selection.artifact_id
        outcome.item_identity.logical_layout_hash = selection.logical_layout_hash
        outcome.item_identity.selection_hash = selection.selection_hash
        outcome.artifact_id = selection.artifact_id
        outcome.status.state = node_agent_pb2.OPERATION_STATE_SUCCESS
        outcome.status.message = "local_replica_ready"

    result = PlanResult.from_node_agent_response(response)

    decoded = result.steps["s-set"].artifact_set_result
    assert decoded is not None
    assert decoded.set_digest_hex == artifact_set.set_digest_hex
    assert len(decoded.outcomes) == 2
    assert decoded.outcomes[0].status is not None
    assert decoded.outcomes[0].status.state == "success"
    assert result.steps["s-set"].value == decoded
