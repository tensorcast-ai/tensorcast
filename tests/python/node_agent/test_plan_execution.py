#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import torch

from tensorcast.engine_adapter import EngineAdapter
from tensorcast.node_agent.executor import NodeAgentExecutor
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.plan.v1 import plan_pb2


class _DaemonStub:
    def __init__(self) -> None:
        self.materialize_timeout_s: float | None = None
        self.placement_timeout_s: float | None = None
        self.release_timeout_s: float | None = None

    def materialize_by_artifact_id_v2(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.materialize_timeout_s = kwargs.get("timeout_s")
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


def test_node_agent_executes_instance_transform_into() -> None:
    adapter = EngineAdapter(instance_id="inst-1", engine="test", register_identity_transform=False)
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
    adapter = EngineAdapter(instance_id="inst-1", engine="test", register_identity_transform=False)
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


def test_node_agent_propagates_deadline_to_worker_actions() -> None:
    daemon = _DaemonStub()
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
