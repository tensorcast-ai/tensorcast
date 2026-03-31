#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import weakref

import pytest
import torch

import tensorcast.node_agent.executor as executor_mod
from tensorcast.api._errors import DeviceMismatch
from tensorcast.api.context import CallContext
from tensorcast.api.plan import ArtifactSetRef, Plan, Worker
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.engine_adapter import (
    BatchOutcome,
    BatchResult,
    EngineAdapter,
    HydrateResult,
    ManifestArtifactSetBridge,
    ManifestResult,
    PublishResult,
)
from tensorcast.node_agent.executor import NodeAgentExecutor
from tensorcast.node_agent.server import NodeAgentServicer
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.node_agent.v1 import node_agent_pb2
from tensorcast.proto.plan.v1 import plan_pb2


class _DaemonStub:
    def __init__(self) -> None:
        self.materialize_timeout_s: float | None = None
        self.placement_timeout_s: float | None = None
        self.release_timeout_s: float | None = None
        self.materialized_artifact_ids: list[str] = []
        self.wait_for_completion_values: list[bool] = []

    def materialize_by_artifact_id_v2(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.materialize_timeout_s = kwargs.get("timeout_s")
        selection = kwargs.get("selection")
        if selection is not None:
            self.materialized_artifact_ids.append(str(selection.artifact_id))
        self.wait_for_completion_values.append(
            bool(kwargs.get("wait_for_completion", True))
        )
        return None

    def create_placement_lease(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.placement_timeout_s = kwargs.get("timeout_s")
        return None

    def release_placement_lease(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.release_timeout_s = kwargs.get("timeout_s")

        class _Resp:
            released = True

        return _Resp()


def _selection() -> common_pb2.ArtifactSelection:
    return common_pb2.ArtifactSelection(
        artifact_id="mi2:test",
        logical_layout_hash=b"logical",
        selection_hash=b"selection",
    )


def _canonical_index_bytes() -> bytes:
    return b'{"w":[0,4,[1],[1],"torch.float32",0]}'


def test_node_agent_executes_instance_transform_into() -> None:
    adapter = EngineAdapter(
        instance_id="inst-1", engine="test", register_identity_transform=False
    )
    called = {"into": False}

    def _into(ctx):  # noqa: ANN001
        assert ctx.targets is not None
        assert "w" in ctx.targets
        called["into"] = True

    adapter.register_transform_fn("noop.v1", into=_into)
    target = adapter.mint_target("target", {"w": torch.zeros(1)})

    spec = plan_pb2.PlanSpec(plan_id="plan-1")
    spec.context.request_id = "req-1"
    step = spec.steps.add()
    step.step_id = "s1"
    step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    step.target.target_id = "inst-1"
    action = step.action.transform_into
    action.selection.CopyFrom(_selection())
    action.spec.name = "noop.v1"
    action.target.CopyFrom(target.to_proto())

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=adapter,
        client_factory=lambda _addr: _DaemonStub(),
    )
    result = executor.execute_plan(spec)
    assert result.ok
    assert result.steps["s1"].status.state == "success"
    assert called["into"] is True


def test_node_agent_marks_dependents_cancelled_on_failure() -> None:
    adapter = EngineAdapter(
        instance_id="inst-1", engine="test", register_identity_transform=False
    )
    adapter.register_transform_fn("noop.v1", into=lambda _ctx: None)
    target = adapter.mint_target("target", {"w": torch.zeros(1)})

    spec = plan_pb2.PlanSpec(plan_id="plan-2")
    spec.context.request_id = "req-2"
    step1 = spec.steps.add()
    step1.step_id = "s1"
    step1.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    step1.target.target_id = "inst-1"
    action1 = step1.action.transform_into
    action1.selection.CopyFrom(_selection())
    action1.spec.name = "missing.v1"
    action1.target.CopyFrom(target.to_proto())

    step2 = spec.steps.add()
    step2.step_id = "s2"
    step2.depends_on.append("s1")
    step2.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    step2.target.target_id = "inst-1"
    action2 = step2.action.transform_into
    action2.selection.CopyFrom(_selection())
    action2.spec.name = "noop.v1"
    action2.target.CopyFrom(target.to_proto())

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=adapter,
        client_factory=lambda _addr: _DaemonStub(),
    )
    result = executor.execute_plan(spec)
    assert result.ok is False
    assert result.steps["s1"].status.state == "failed"
    assert result.steps["s2"].status.state == "cancelled"


def test_node_agent_propagates_deadline_to_worker_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _DaemonStub()
    monkeypatch.setattr(executor_mod, "device_uuid_for", lambda _device_id: "gpu-0")
    spec = plan_pb2.PlanSpec(plan_id="plan-3")
    spec.context.request_id = "req-3"
    spec.context.deadline_ms = 1500

    step1 = spec.steps.add()
    step1.step_id = "s1"
    step1.target.target_type = plan_pb2.TARGET_TYPE_WORKER
    step1.target.target_id = "daemon-1"
    action1 = step1.action.prefetch
    action1.selection.CopyFrom(_selection())
    action1.device_id = 0

    step2 = spec.steps.add()
    step2.step_id = "s2"
    step2.target.target_type = plan_pb2.TARGET_TYPE_WORKER
    step2.target.target_id = "daemon-1"
    action2 = step2.action.pin_device_residency
    action2.selection.CopyFrom(_selection())
    action2.device_id = 0

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=None,
        client_factory=lambda _addr: daemon,
    )
    result = executor.execute_plan(spec)
    assert result.ok
    assert daemon.materialize_timeout_s is not None
    assert abs(float(daemon.materialize_timeout_s) - 1.5) < 0.1
    assert daemon.placement_timeout_s is not None
    assert abs(float(daemon.placement_timeout_s) - 1.5) < 0.1


def test_node_agent_prefetch_reports_device_mismatch_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _DaemonStub()

    def _raise_device_mismatch(_device_id: int) -> str:
        raise DeviceMismatch("device map unavailable")

    monkeypatch.setattr(executor_mod, "device_uuid_for", _raise_device_mismatch)
    spec = plan_pb2.PlanSpec(plan_id="plan-4")
    spec.context.request_id = "req-4"

    step = spec.steps.add()
    step.step_id = "s1"
    step.target.target_type = plan_pb2.TARGET_TYPE_WORKER
    step.target.target_id = "daemon-1"
    action = step.action.prefetch
    action.selection.CopyFrom(_selection())
    action.device_id = 0

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=None,
        client_factory=lambda _addr: daemon,
    )
    result = executor.execute_plan(spec)
    assert result.ok is False
    step_result = result.steps["s1"]
    assert step_result.status.state == "failed"
    assert step_result.status.error is not None
    assert step_result.status.error.status_code == "FAILED_PRECONDITION"
    assert "device map unavailable" in step_result.status.message


def test_node_agent_executes_prefetch_set_from_plan_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _DaemonStub()
    monkeypatch.setattr(executor_mod, "device_uuid_for", lambda _device_id: "gpu-0")

    class _StoreStub:
        closed = False
        _runtime = None

    canonical_bytes = _canonical_index_bytes()
    store = _StoreStub()
    artifacts = [
        Artifact(
            store_ref=weakref.ref(store),
            artifact_id=f"mi2:set:{index:04d}",
            canonical_index_bytes=canonical_bytes,
            canonical_index=canonical_index_from_bytes(canonical_bytes),
        )
        for index in range(300)
    ]
    plan = Plan(CallContext(request_id="req-prefetch-set", idempotency_key="idem-set"))
    worker = Worker(
        worker_id="worker-1",
        daemon_address="127.0.0.1:50051",
        daemon_id="daemon-1",
    )
    plan.on_worker(worker).prefetch_many(artifacts, device=0)
    spec = plan.to_spec()

    assert spec.steps[0].action.WhichOneof("kind") == "prefetch_set"

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=None,
        client_factory=lambda _addr: daemon,
    )
    result = executor.execute_plan(spec)

    assert result.ok is True
    step_result = result.steps["step-0001"]
    assert step_result.status.state == "success"
    assert step_result.artifact_set_result is not None
    assert len(step_result.artifact_set_result.outcomes) == 300
    assert all(
        outcome.status is not None and outcome.status.state == "success"
        for outcome in step_result.artifact_set_result.outcomes
    )
    assert daemon.materialized_artifact_ids == [
        f"mi2:set:{index:04d}" for index in range(300)
    ]
    assert daemon.wait_for_completion_values
    assert all(daemon.wait_for_completion_values)


def test_node_agent_executes_manifest_backed_prefetch_set_with_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _DaemonStub()
    monkeypatch.setattr(executor_mod, "device_uuid_for", lambda _device_id: "gpu-0")
    manifest_result = ManifestResult.from_artifact_selections(
        engine_request_id="rid-bridge",
        layout_id="layout-v1",
        manifest_selection=common_pb2.ArtifactSelection(
            artifact_id="engine-manifest:rid-bridge",
            logical_layout_hash=b"manifest-logical",
            selection_hash=b"manifest-selection",
        ),
        artifact_selections=tuple(
            common_pb2.ArtifactSelection(
                artifact_id=f"mi2:bridge:{index:02d}",
                logical_layout_hash=f"logical-{index:02d}".encode("utf-8"),
                selection_hash=f"selection-{index:02d}".encode("utf-8"),
            )
            for index in range(8)
        ),
    )
    plan = Plan(CallContext(request_id="req-manifest-prefetch", idempotency_key="idem"))
    worker = Worker(
        worker_id="worker-1",
        daemon_address="127.0.0.1:50051",
        daemon_id="daemon-1",
    )
    plan.on_worker(worker).prefetch_manifest_result(manifest_result, device=0)

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=None,
        client_factory=lambda _addr: daemon,
    )
    result = executor.execute_plan(plan.to_spec())

    assert result.ok is True
    step_result = result.steps["step-0001"]
    assert step_result.status.state == "success"
    assert step_result.artifact_set_result is not None
    assert len(step_result.artifact_set_result.outcomes) == 8
    assert daemon.materialized_artifact_ids == [
        f"mi2:bridge:{index:02d}" for index in range(8)
    ]


def test_node_agent_prefetch_set_fails_closed_on_unsupported_manifest_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _DaemonStub()
    monkeypatch.setattr(executor_mod, "device_uuid_for", lambda _device_id: "gpu-0")
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
                artifact_id="mi2:bridge:a",
                logical_layout_hash=b"logical-a",
                selection_hash=b"selection-a",
            ),
        ),
    )
    bridge = manifest_result.require_artifact_set_bridge()
    unsupported_bridge = ManifestArtifactSetBridge(
        bridge_schema="tensorcast.unsupported_bridge",
        bridge_version=bridge.bridge_version,
        artifact_set_ref=bridge.artifact_set_ref,
        resolved_items=bridge.resolved_items,
    )
    plan = Plan(CallContext(request_id="req-manifest-prefetch", idempotency_key="idem"))
    worker = Worker(
        worker_id="worker-1",
        daemon_address="127.0.0.1:50051",
        daemon_id="daemon-1",
    )
    plan.on_worker(worker).prefetch_set(
        bridge.artifact_set_ref,
        device=0,
        manifest_bridge=unsupported_bridge,
    )

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=None,
        client_factory=lambda _addr: daemon,
    )
    result = executor.execute_plan(plan.to_spec())

    assert result.ok is False
    assert result.steps["step-0001"].status.state == "failed"
    assert "unsupported" in str(result.steps["step-0001"].status.message)


def test_node_agent_executes_artifact_actions() -> None:
    observed: dict[str, list[str]] = {
        "manifest": [],
        "publish": [],
        "hydrate": [],
        "evict": [],
    }

    adapter = EngineAdapter(
        instance_id="inst-1", engine="test", register_identity_transform=False
    )

    def _manifest(rid: str, _ctx):  # noqa: ANN001
        observed["manifest"].append(rid)
        return ManifestResult.from_artifact_ids(
            engine_request_id=rid,
            layout_id="layout-v1",
            artifact_ids=("cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",),
        )

    def _publish(rid: str, ttl_ms: int | None, _sealed, _ctx):  # noqa: ANN001
        observed["publish"].append(f"{rid}:{ttl_ms}")
        manifest = ManifestResult.from_artifact_ids(
            engine_request_id=rid,
            layout_id="layout-v1",
            artifact_ids=("cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",),
        )
        return PublishResult(
            manifest=manifest,
            put_outcomes=(
                BatchOutcome(
                    artifact_id=manifest.artifact_ids[0],
                    status_code="OK",
                ),
            ),
        )

    def _hydrate(rid: str, _ctx):  # noqa: ANN001
        observed["hydrate"].append(rid)
        manifest = ManifestResult.from_artifact_ids(
            engine_request_id=rid,
            layout_id="layout-v1",
            artifact_ids=(),
        )
        return HydrateResult(
            manifest=manifest,
            get_outcomes=(),
            missing_artifact_ids=(),
        )

    def _evict(rid: str | None, _ctx):  # noqa: ANN001
        observed["evict"].append("" if rid is None else rid)
        return BatchResult(engine_request_id=rid, outcomes=())

    adapter.register_artifact_fns(
        manifest=_manifest,
        publish=_publish,
        hydrate=_hydrate,
        evict_local=_evict,
    )

    spec = plan_pb2.PlanSpec(plan_id="plan-cache-1")
    spec.context.request_id = "req-cache-1"
    step1 = spec.steps.add()
    step1.step_id = "s1"
    step1.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    step1.target.target_id = "inst-1"
    step1.action.manifest.engine_request_id = "rid-123"

    step2 = spec.steps.add()
    step2.step_id = "s2"
    step2.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    step2.target.target_id = "inst-1"
    step2.action.publish.engine_request_id = "rid-123"
    step2.action.publish.ttl_ms = 60000

    step3 = spec.steps.add()
    step3.step_id = "s3"
    step3.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    step3.target.target_id = "inst-1"
    step3.action.hydrate.engine_request_id = "rid-123"

    step4 = spec.steps.add()
    step4.step_id = "s4"
    step4.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    step4.target.target_id = "inst-1"
    step4.action.evict_local.engine_request_id = "rid-123"

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=adapter,
        client_factory=lambda _addr: _DaemonStub(),
    )
    result = executor.execute_plan(spec)
    assert result.ok is True
    assert result.steps["s1"].status.state == "success"
    assert result.steps["s2"].status.state == "success"
    assert result.steps["s3"].status.state == "success"
    assert result.steps["s4"].status.state == "success"
    manifest_result = result.steps["s1"].artifact_result
    assert isinstance(manifest_result, ManifestResult)
    assert manifest_result.engine_request_id == "rid-123"
    publish_result = result.steps["s2"].artifact_result
    assert isinstance(publish_result, PublishResult)
    assert publish_result.put_outcomes[0].status_code == "OK"
    hydrate_result = result.steps["s3"].artifact_result
    assert isinstance(hydrate_result, HydrateResult)
    assert hydrate_result.manifest is not None
    evict_result = result.steps["s4"].artifact_result
    assert isinstance(evict_result, BatchResult)
    assert evict_result.engine_request_id == "rid-123"
    assert observed["manifest"] == ["rid-123"]
    assert observed["publish"] == ["rid-123:60000"]
    assert observed["hydrate"] == ["rid-123"]
    assert observed["evict"] == ["rid-123"]


def test_node_agent_servicer_serializes_artifact_results() -> None:
    adapter = EngineAdapter(
        instance_id="inst-1", engine="test", register_identity_transform=False
    )

    def _manifest(rid: str, _ctx):  # noqa: ANN001
        return ManifestResult.from_artifact_selections(
            engine_request_id=rid,
            layout_id="layout-v1",
            manifest_selection=common_pb2.ArtifactSelection(
                artifact_id=f"engine-manifest:{rid}",
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

    def _publish(rid: str, ttl_ms: int | None, _sealed, _ctx):  # noqa: ANN001
        _ = ttl_ms
        manifest = ManifestResult.from_artifact_selections(
            engine_request_id=rid,
            layout_id="layout-v1",
            manifest_selection=common_pb2.ArtifactSelection(
                artifact_id=f"engine-manifest:{rid}",
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
        return PublishResult(
            manifest=manifest,
            put_outcomes=(
                BatchOutcome(
                    artifact_id=manifest.artifact_ids[0],
                    status_code="OK",
                    message="created",
                ),
            ),
        )

    adapter.register_artifact_fns(manifest=_manifest, publish=_publish)

    spec = plan_pb2.PlanSpec(plan_id="plan-node-agent-proto")
    spec.context.request_id = "req-node-agent-proto"

    manifest_step = spec.steps.add()
    manifest_step.step_id = "s1"
    manifest_step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    manifest_step.target.target_id = "inst-1"
    manifest_step.action.manifest.engine_request_id = "rid-123"

    publish_step = spec.steps.add()
    publish_step.step_id = "s2"
    publish_step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    publish_step.target.target_id = "inst-1"
    publish_step.action.publish.engine_request_id = "rid-123"

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=adapter,
        client_factory=lambda _addr: _DaemonStub(),
    )
    servicer = NodeAgentServicer(executor)
    response = servicer.ExecutePlan(
        node_agent_pb2.ExecutePlanRequest(plan=spec),
        None,  # type: ignore[arg-type]
    )

    assert response.ok is True
    assert response.steps[0].HasField("artifact_result")
    assert response.steps[0].artifact_result.manifest.engine_request_id == "rid-123"
    assert response.steps[0].artifact_result.manifest.HasField("manifest_bridge")
    assert response.steps[1].HasField("artifact_result")
    assert response.steps[1].artifact_result.publish.manifest.layout_id == "layout-v1"
    assert response.steps[1].artifact_result.publish.manifest.HasField(
        "manifest_bridge"
    )
    assert (
        response.steps[1].artifact_result.publish.put_outcomes[0].message == "created"
    )


def test_node_agent_servicer_serializes_artifact_set_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _DaemonStub()
    monkeypatch.setattr(executor_mod, "device_uuid_for", lambda _device_id: "gpu-0")
    spec = plan_pb2.PlanSpec(plan_id="plan-node-agent-set-proto")
    spec.context.request_id = "req-node-agent-set-proto"
    step = spec.steps.add()
    step.step_id = "s1"
    step.target.target_type = plan_pb2.TARGET_TYPE_WORKER
    step.target.target_id = "daemon-1"
    action = step.action.prefetch_set
    action.device_id = 0
    artifact_set = ArtifactSetRef.inline(
        (
            common_pb2.ArtifactSelection(
                artifact_id="mi2:set:a",
                logical_layout_hash=b"logical-a",
                selection_hash=b"selection-a",
            ),
            common_pb2.ArtifactSelection(
                artifact_id="mi2:set:b",
                logical_layout_hash=b"logical-b",
                selection_hash=b"selection-b",
            ),
        )
    )
    action.artifact_set.CopyFrom(artifact_set.to_proto())

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=None,
        client_factory=lambda _addr: daemon,
    )
    servicer = NodeAgentServicer(executor)
    response = servicer.ExecutePlan(
        node_agent_pb2.ExecutePlanRequest(plan=spec),
        None,  # type: ignore[arg-type]
    )

    assert response.ok is True
    assert response.steps[0].HasField("artifact_set_result")
    assert (
        response.steps[0].artifact_set_result.set_digest_hex
        == artifact_set.set_digest_hex
    )
    assert len(response.steps[0].artifact_set_result.outcomes) == 2
    assert (
        response.steps[0].artifact_set_result.outcomes[0].status.state
        == node_agent_pb2.OPERATION_STATE_SUCCESS
    )


def test_node_agent_publish_actions_share_idempotency_key_without_request_identity() -> (
    None
):
    adapter = EngineAdapter(
        instance_id="inst-1", engine="test", register_identity_transform=False
    )
    observed_keys: list[str] = []

    def _publish(rid: str, ttl_ms: int | None, _sealed, ctx):  # noqa: ANN001
        _ = rid, ttl_ms
        observed_keys.append("" if ctx is None else str(ctx.idempotency_key or ""))
        manifest = ManifestResult.from_artifact_ids(
            engine_request_id="rid-123",
            layout_id="layout-v1",
            artifact_ids=(),
        )
        return PublishResult(manifest=manifest, put_outcomes=())

    adapter.register_artifact_fns(publish=_publish)

    first_spec = plan_pb2.PlanSpec(plan_id="plan-cache-first")
    first_spec.context.request_id = "req-cache-first"
    first_spec.context.idempotency_key = "idem-123"
    first_step = first_spec.steps.add()
    first_step.step_id = "s1"
    first_step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    first_step.target.target_id = "inst-1"
    first_step.action.publish.engine_request_id = "rid-123"
    first_step.action.publish.ttl_ms = 123

    second_spec = plan_pb2.PlanSpec(plan_id="plan-cache-second")
    second_spec.context.request_id = "req-cache-second"
    second_spec.context.idempotency_key = "idem-123"
    second_step = second_spec.steps.add()
    second_step.step_id = "s1"
    second_step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
    second_step.target.target_id = "inst-1"
    second_step.action.publish.engine_request_id = "rid-456"
    second_step.action.publish.ttl_ms = 123

    executor = NodeAgentExecutor(
        daemon_id="daemon-1",
        daemon_address="127.0.0.1:50051",
        instance_id="inst-1",
        engine_adapter=adapter,
        client_factory=lambda _addr: _DaemonStub(),
    )
    first_result = executor.execute_plan(first_spec)
    second_result = executor.execute_plan(second_spec)
    assert first_result.ok is True
    assert second_result.ok is True
    assert len(observed_keys) == 2
    assert observed_keys[0]
    assert observed_keys[0] == observed_keys[1]
