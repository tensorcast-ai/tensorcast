#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Callable, Mapping

from tensorcast.api._config import StorePolicy
from tensorcast.api._device import CPU_DEVICE_ID, device_uuid_for
from tensorcast.api._errors import DeviceMismatch
from tensorcast.api._view_ops import NarrowOp, TransposeOp, ViewSpecBuildResult
from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.operation import OperationError, OperationStatus
from tensorcast.api.plan.artifact_set import (
    ArtifactSetItemResult,
    ArtifactSetRef,
    ArtifactSetResult,
    resolve_artifact_set_ref,
    summarize_artifact_set_outcomes,
)
from tensorcast.api.plan.targets import TargetSpec
from tensorcast.api.plan.transforms import TransformSpec
from tensorcast.api.store import Artifact, Store
from tensorcast.daemon_ctl import DaemonCtl, get_daemon_client
from tensorcast.engine_adapter import (
    BatchResult,
    EngineAdapter,
    HydrateResult,
    ManifestArtifactSetBridge,
    ManifestResult,
    PublishResult,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.plan.v1 import plan_pb2

ArtifactActionResult = ManifestResult | PublishResult | HydrateResult | BatchResult


@dataclass(frozen=True, slots=True)
class NodeAgentStepResult:
    step_id: str
    target_id: str
    action: str
    status: OperationStatus
    artifact_result: ArtifactActionResult | None = None
    artifact_set_result: ArtifactSetResult | None = None


@dataclass(frozen=True, slots=True)
class NodeAgentExecutionResult:
    ok: bool
    request_id: str
    steps: Mapping[str, NodeAgentStepResult]


def _status_failed(
    message: str, *, status_code: str, retryable: bool
) -> OperationStatus:
    return OperationStatus(
        state="failed",
        message=message,
        as_of_ms=int(time.time() * 1000),
        error=OperationError(
            status_code=status_code,
            message=message,
            retryable=retryable,
        ),
    )


def _status_success(message: str) -> OperationStatus:
    return OperationStatus(
        state="success",
        message=message,
        as_of_ms=int(time.time() * 1000),
    )


def _derive_action_idempotency_key(
    *,
    base_key: str,
    action: str,
    target_id: str,
    selection: common_pb2.ArtifactSelection,
    device_id: int | None,
    ttl_ms: int | None,
    extra: str | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(base_key.encode("utf-8"))
    digest.update(b"|")
    digest.update(action.encode("utf-8"))
    digest.update(b"|target=")
    digest.update(target_id.encode("utf-8"))
    digest.update(b"|artifact=")
    digest.update(str(selection.artifact_id).encode("utf-8"))
    if device_id is not None:
        digest.update(f"|device={int(device_id)}".encode("utf-8"))
    if ttl_ms is not None:
        digest.update(f"|ttl={int(ttl_ms)}".encode("utf-8"))
    digest.update(b"|logical=")
    digest.update(selection.logical_layout_hash)
    digest.update(b"|selection=")
    digest.update(selection.selection_hash)
    if extra:
        digest.update(b"|extra=")
        digest.update(extra.encode("utf-8"))
    return f"tc.plan.action.v1:{digest.hexdigest()}"


def _derive_instance_action_idempotency_key(
    *,
    base_key: str,
    action: str,
    target_id: str,
    step_id: str,
    ttl_ms: int | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(base_key.encode("utf-8"))
    digest.update(b"|")
    digest.update(action.encode("utf-8"))
    digest.update(b"|target=")
    digest.update(target_id.encode("utf-8"))
    digest.update(b"|step=")
    digest.update(step_id.encode("utf-8"))
    if ttl_ms is not None:
        digest.update(f"|ttl={int(ttl_ms)}".encode("utf-8"))
    return f"tc.plan.action.v1:{digest.hexdigest()}"


def _call_context_from_proto(ctx: plan_pb2.CallContext) -> CallContext:
    qos = "interactive"
    if ctx.qos == plan_pb2.QOS_CLASS_REALTIME:
        qos = "realtime"
    elif ctx.qos == plan_pb2.QOS_CLASS_BACKGROUND:
        qos = "background"
    return CallContext(
        request_id=str(ctx.request_id),
        qos=qos,
        deadline_ms=int(ctx.deadline_ms) if ctx.HasField("deadline_ms") else None,
        idempotency_key=str(ctx.idempotency_key) if ctx.idempotency_key else None,
        tags=dict(ctx.tags) if ctx.tags else None,
    )


def _ctx_timeout_s(ctx: CallContext) -> float | None:
    if ctx.deadline_ms is None:
        return None
    timeout_s = float(ctx.deadline_ms) / 1000.0
    return max(0.0, timeout_s)


def _view_spec_from_proto(
    view_spec: common_pb2.ViewSpec,
) -> ViewSpecBuildResult | None:
    if view_spec is None or not view_spec.tensors:
        return None
    tensor_ops: dict[str, tuple[NarrowOp | TransposeOp, ...]] = {}
    for name, ops in view_spec.tensors.items():
        converted = []
        for op in ops.ops:
            kind = op.WhichOneof("kind")
            if kind == "narrow":
                converted.append(
                    NarrowOp(
                        dim=int(op.narrow.dim),
                        start=int(op.narrow.start),
                        length=int(op.narrow.length),
                    )
                )
            elif kind == "transpose":
                converted.append(
                    TransposeOp(
                        dim0=int(op.transpose.dim0),
                        dim1=int(op.transpose.dim1),
                    )
                )
        tensor_ops[str(name)] = tuple(converted)
    return ViewSpecBuildResult(proto=view_spec, tensor_ops=tensor_ops)


def _store_policy_from_proto(
    policy: plan_pb2.StorePolicy | None,
) -> StorePolicy | None:
    if policy is None:
        return None
    if (
        policy.profile == plan_pb2.POLICY_PROFILE_UNSPECIFIED
        and not policy.must
        and not policy.should
        and not policy.may
        and policy.overflow_policy == plan_pb2.OVERFLOW_POLICY_UNSPECIFIED
        and policy.layout == plan_pb2.POLICY_LAYOUT_UNSPECIFIED
    ):
        return None
    from tensorcast.api._config import (
        OverflowPolicy,
        PolicyLayout,
        PolicyScope,
        PolicyTier,
        RetentionPolicy,
        StorePolicy,
        StorePolicyProfile,
        TierSpec,
    )

    profile = None
    if policy.profile != plan_pb2.POLICY_PROFILE_UNSPECIFIED:
        name = plan_pb2.PolicyProfile.Name(policy.profile).replace(
            "POLICY_PROFILE_", ""
        )
        profile = StorePolicyProfile.parse(name.lower())

    def _tier_from_proto(spec: plan_pb2.TierSpec) -> TierSpec:
        tier_name = plan_pb2.PolicyTier.Name(spec.tier).replace("POLICY_TIER_", "")
        scope_name = plan_pb2.PolicyScope.Name(spec.scope).replace("POLICY_SCOPE_", "")
        retention_name = plan_pb2.RetentionPolicy.Name(spec.retention_policy).replace(
            "RETENTION_POLICY_", ""
        )
        retention = (
            RetentionPolicy.parse(retention_name.lower())
            if spec.retention_policy != plan_pb2.RETENTION_POLICY_UNSPECIFIED
            else RetentionPolicy.BEST_EFFORT
        )
        ttl = int(spec.retention_ttl_ms) if spec.HasField("retention_ttl_ms") else None
        return TierSpec(
            tier=PolicyTier.parse(tier_name.lower()),
            scope=PolicyScope.parse(scope_name.lower()),
            min_replicas=int(spec.min_replicas) if spec.min_replicas else 1,
            retention_policy=retention,
            retention_ttl_ms=ttl,
        )

    overflow_name = plan_pb2.OverflowPolicy.Name(policy.overflow_policy).replace(
        "OVERFLOW_POLICY_", ""
    )
    overflow = (
        OverflowPolicy.parse(overflow_name.lower())
        if policy.overflow_policy != plan_pb2.OVERFLOW_POLICY_UNSPECIFIED
        else OverflowPolicy.EVICT
    )
    layout_name = plan_pb2.PolicyLayout.Name(policy.layout).replace(
        "POLICY_LAYOUT_", ""
    )
    layout = (
        PolicyLayout.parse(layout_name.lower())
        if policy.layout != plan_pb2.POLICY_LAYOUT_UNSPECIFIED
        else PolicyLayout.AUTO
    )

    return StorePolicy(
        profile=profile,
        must=tuple(_tier_from_proto(tier) for tier in policy.must),
        should=tuple(_tier_from_proto(tier) for tier in policy.should),
        may=tuple(_tier_from_proto(tier) for tier in policy.may),
        overflow_policy=overflow,
        layout=layout,
    )


class NodeAgentExecutor:
    def __init__(
        self,
        *,
        daemon_id: str,
        daemon_address: str,
        instance_id: str | None = None,
        agent_id: str | None = None,
        version: str | None = None,
        engine_adapter: EngineAdapter | None = None,
        client_factory: Callable[[str], DaemonCtl] = get_daemon_client,
    ) -> None:
        self._daemon_id = daemon_id
        self._daemon_address = daemon_address
        self._instance_id = instance_id
        self._agent_id = agent_id or uuid.uuid4().hex
        self._version = version or "unknown"
        self._engine_adapter = engine_adapter
        self._client = client_factory(daemon_address)
        self._store: Store | None = None

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def daemon_id(self) -> str:
        return self._daemon_id

    @property
    def instance_id(self) -> str | None:
        return self._instance_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def engine_adapter(self) -> EngineAdapter | None:
        return self._engine_adapter

    def _store_for_daemon(self) -> Store:
        if self._store is None:
            self._store = Store(self._daemon_address)
        return self._store

    def execute_plan(
        self, plan: plan_pb2.PlanSpec, *, dry_run: bool = False
    ) -> NodeAgentExecutionResult:
        steps_by_id = {step.step_id: step for step in plan.steps}
        dependencies: dict[str, set[str]] = {
            step_id: set(step.depends_on) for step_id, step in steps_by_id.items()
        }
        dependents: dict[str, set[str]] = {step_id: set() for step_id in steps_by_id}
        for step_id, deps in dependencies.items():
            for dep in deps:
                if dep not in steps_by_id:
                    raise ArtifactError(
                        f"Unknown dependency step id: {dep}",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                dependents[dep].add(step_id)

        ready = [step_id for step_id, deps in dependencies.items() if not deps]
        if steps_by_id and not ready:
            raise ArtifactError(
                "Plan has no runnable steps (cycle detected)",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        results: dict[str, NodeAgentStepResult] = {}
        call_ctx = _call_context_from_proto(plan.context)
        failed = False
        while ready:
            step_id = ready.pop(0)
            if failed:
                results[step_id] = self._cancelled_result(
                    steps_by_id[step_id],
                    "skipped due to prior failure",
                )
                continue
            result = self._execute_step(
                plan, steps_by_id[step_id], call_ctx=call_ctx, dry_run=dry_run
            )
            results[step_id] = result
            if result.status.state != "success":
                failed = True
            for child in dependents[step_id]:
                child_deps = dependencies.get(child)
                if child_deps is None:
                    continue
                child_deps.discard(step_id)
                if not child_deps and child not in results and child not in ready:
                    ready.append(child)

        for step_id, step in steps_by_id.items():
            if step_id not in results:
                results[step_id] = self._cancelled_result(
                    step, "skipped due to prior failure"
                )

        request_id = plan.context.request_id or uuid.uuid4().hex
        ok = all(result.status.state == "success" for result in results.values())
        return NodeAgentExecutionResult(
            ok=ok,
            request_id=request_id,
            steps=results,
        )

    def _execute_step(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        *,
        call_ctx: CallContext,
        dry_run: bool,
    ) -> NodeAgentStepResult:
        target = step.target
        action_name = step.action.WhichOneof("kind") or "unknown"
        target_id = target.target_id
        if target.target_type == plan_pb2.TARGET_TYPE_WORKER:
            if target_id != self._daemon_id:
                return NodeAgentStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action=action_name,
                    status=_status_failed(
                        "worker target does not match this agent",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    ),
                )
            if dry_run:
                return NodeAgentStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action=action_name,
                    status=_status_success("dry-run"),
                )
            return self._execute_worker_action(plan, step, call_ctx=call_ctx)
        if target.target_type == plan_pb2.TARGET_TYPE_INSTANCE:
            if self._instance_id is None or target_id != self._instance_id:
                return NodeAgentStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action=action_name,
                    status=_status_failed(
                        "instance target does not match this agent",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    ),
                )
            if dry_run:
                return NodeAgentStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action=action_name,
                    status=_status_success("dry-run"),
                )
            return self._execute_instance_action(plan, step, call_ctx=call_ctx)
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action=action_name,
            status=_status_failed(
                "unknown target type",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ),
        )

    def _execute_worker_action(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        *,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        action = step.action
        action_kind = action.WhichOneof("kind")
        if action_kind == "prefetch":
            return self._prefetch(plan, step, action.prefetch, call_ctx=call_ctx)
        if action_kind == "prefetch_set":
            return self._prefetch_set(
                plan, step, action.prefetch_set, call_ctx=call_ctx
            )
        if action_kind == "pin_device_residency":
            return self._pin(plan, step, action.pin_device_residency, call_ctx=call_ctx)
        if action_kind == "unpin_device_residency":
            return self._unpin(step, action.unpin_device_residency, call_ctx=call_ctx)
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=step.target.target_id,
            action=action_kind or "unknown",
            status=_status_failed(
                "unknown action",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ),
        )

    def _execute_instance_action(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        *,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        if self._engine_adapter is None:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=step.target.target_id,
                action=step.action.WhichOneof("kind") or "unknown",
                status=_status_failed(
                    "engine adapter is not configured",
                    status_code="UNIMPLEMENTED",
                    retryable=False,
                ),
            )
        action = step.action
        action_kind = action.WhichOneof("kind")
        if action_kind == "transform_into":
            return self._transform_into(plan, step, action.transform_into, call_ctx)
        if action_kind == "transform_register":
            return self._transform_register(
                plan, step, action.transform_register, call_ctx
            )
        if action_kind == "manifest":
            return self._manifest(plan, step, action.manifest, call_ctx)
        if action_kind == "publish":
            return self._publish(plan, step, action.publish, call_ctx)
        if action_kind == "hydrate":
            return self._hydrate(plan, step, action.hydrate, call_ctx)
        if action_kind == "evict_local":
            return self._evict_local(plan, step, action.evict_local, call_ctx)
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=step.target.target_id,
            action=action_kind or "unknown",
            status=_status_failed(
                "unknown instance action",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ),
        )

    def _action_context(
        self,
        *,
        plan: plan_pb2.PlanSpec,
        action: str,
        target_id: str,
        selection: common_pb2.ArtifactSelection,
        extra: str | None,
        call_ctx: CallContext,
    ) -> CallContext:
        if not plan.context.idempotency_key:
            return call_ctx
        derived_key = _derive_action_idempotency_key(
            base_key=plan.context.idempotency_key,
            action=action,
            target_id=target_id,
            selection=selection,
            device_id=None,
            ttl_ms=None,
            extra=extra,
        )
        return CallContext(
            request_id=call_ctx.request_id,
            qos=call_ctx.qos,
            deadline_ms=call_ctx.deadline_ms,
            idempotency_key=derived_key,
            tags=call_ctx.tags,
        )

    def _instance_action_context(
        self,
        *,
        plan: plan_pb2.PlanSpec,
        action: str,
        target_id: str,
        step_id: str,
        ttl_ms: int | None,
        call_ctx: CallContext,
    ) -> CallContext:
        if not plan.context.idempotency_key:
            return call_ctx
        derived_key = _derive_instance_action_idempotency_key(
            base_key=plan.context.idempotency_key,
            action=action,
            target_id=target_id,
            step_id=step_id,
            ttl_ms=ttl_ms,
        )
        return CallContext(
            request_id=call_ctx.request_id,
            qos=call_ctx.qos,
            deadline_ms=call_ctx.deadline_ms,
            idempotency_key=derived_key,
            tags=call_ctx.tags,
        )

    def _artifact_from_selection(
        self, selection: common_pb2.ArtifactSelection
    ) -> tuple[Artifact, tuple[str, ...] | None]:
        view_spec = None
        if selection.HasField("view_spec"):
            view_spec = _view_spec_from_proto(selection.view_spec)
        store = self._store_for_daemon()
        artifact = Artifact(
            store_ref=weakref.ref(store),
            artifact_id=selection.artifact_id,
            view_spec=view_spec,
        )
        tensor_names = tuple(selection.tensor_names) if selection.tensor_names else None
        if tensor_names:
            artifact = artifact.subset(list(tensor_names))
        return artifact, tensor_names

    def _transform_into(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.TransformIntoAction,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        try:
            spec = TransformSpec.from_proto(action.spec)
            targets = TargetSpec.from_proto(action.target)
            if targets.instance_id != target_id:
                raise ArtifactError(
                    "TargetSpec instance_id does not match step target",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            engine_adapter = self._engine_adapter
            if engine_adapter is None:
                raise ArtifactError(
                    "EngineAdapter is not configured on node agent",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            artifact, tensor_names = self._artifact_from_selection(action.selection)
            ctx = self._action_context(
                plan=plan,
                action=f"transform_into:{spec.name}",
                target_id=target_id,
                selection=action.selection,
                extra=spec.fingerprint(),
                call_ctx=call_ctx,
            )
            engine_adapter.execute_transform_into(
                spec=spec,
                source=artifact,
                targets=targets,
                store=self._store_for_daemon(),
                ctx=ctx,
                tensor_names=tensor_names,
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="transform_into",
                status=_status_failed(
                    f"transform_into failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="transform_into",
            status=_status_success("transform_into completed"),
        )

    def _transform_register(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.TransformRegisterAction,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        try:
            spec = TransformSpec.from_proto(action.spec)
            engine_adapter = self._engine_adapter
            if engine_adapter is None:
                raise ArtifactError(
                    "EngineAdapter is not configured on node agent",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            artifact, tensor_names = self._artifact_from_selection(action.selection)
            policy = (
                _store_policy_from_proto(action.policy)
                if action.HasField("policy")
                else None
            )
            ctx = self._action_context(
                plan=plan,
                action=f"transform_register:{spec.name}",
                target_id=target_id,
                selection=action.selection,
                extra=spec.fingerprint(),
                call_ctx=call_ctx,
            )
            engine_adapter.execute_transform_register(
                spec=spec,
                source=artifact,
                out_key=action.out_key,
                store=self._store_for_daemon(),
                policy=policy,
                ctx=ctx,
                tensor_names=tensor_names,
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="transform_register",
                status=_status_failed(
                    f"transform_register failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="transform_register",
            status=_status_success("transform_register completed"),
        )

    def _manifest(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.ManifestAction,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        try:
            engine_adapter = self._engine_adapter
            if engine_adapter is None:
                raise ArtifactError(
                    "EngineAdapter is not configured on node agent",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            scoped_ctx = self._instance_action_context(
                plan=plan,
                action="manifest",
                target_id=target_id,
                step_id=step.step_id,
                ttl_ms=None,
                call_ctx=call_ctx,
            )
            manifest_result = engine_adapter.execute_manifest(
                engine_request_id=action.engine_request_id,
                ctx=scoped_ctx,
            )
        except ArtifactError as exc:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="manifest",
                status=_status_failed(
                    f"manifest failed: {exc}",
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="manifest",
                status=_status_failed(
                    f"manifest failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="manifest",
            status=_status_success("manifest completed"),
            artifact_result=manifest_result,
        )

    def _publish(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.PublishAction,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        ttl_ms = int(action.ttl_ms) if action.HasField("ttl_ms") else None
        try:
            engine_adapter = self._engine_adapter
            if engine_adapter is None:
                raise ArtifactError(
                    "EngineAdapter is not configured on node agent",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            scoped_ctx = self._instance_action_context(
                plan=plan,
                action="publish",
                target_id=target_id,
                step_id=step.step_id,
                ttl_ms=ttl_ms,
                call_ctx=call_ctx,
            )
            publish_result = engine_adapter.execute_publish(
                engine_request_id=action.engine_request_id,
                ttl_ms=ttl_ms,
                ctx=scoped_ctx,
            )
        except ArtifactError as exc:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="publish",
                status=_status_failed(
                    f"publish failed: {exc}",
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="publish",
                status=_status_failed(
                    f"publish failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="publish",
            status=_status_success("publish completed"),
            artifact_result=publish_result,
        )

    def _hydrate(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.HydrateAction,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        try:
            engine_adapter = self._engine_adapter
            if engine_adapter is None:
                raise ArtifactError(
                    "EngineAdapter is not configured on node agent",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            scoped_ctx = self._instance_action_context(
                plan=plan,
                action="hydrate",
                target_id=target_id,
                step_id=step.step_id,
                ttl_ms=None,
                call_ctx=call_ctx,
            )
            hydrate_result = engine_adapter.execute_hydrate(
                engine_request_id=action.engine_request_id,
                ctx=scoped_ctx,
            )
        except ArtifactError as exc:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="hydrate",
                status=_status_failed(
                    f"hydrate failed: {exc}",
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="hydrate",
                status=_status_failed(
                    f"hydrate failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="hydrate",
            status=_status_success("hydrate completed"),
            artifact_result=hydrate_result,
        )

    def _evict_local(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.EvictLocalAction,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        engine_request_id = (
            action.engine_request_id if action.HasField("engine_request_id") else None
        )
        try:
            engine_adapter = self._engine_adapter
            if engine_adapter is None:
                raise ArtifactError(
                    "EngineAdapter is not configured on node agent",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            scoped_ctx = self._instance_action_context(
                plan=plan,
                action="evict_local",
                target_id=target_id,
                step_id=step.step_id,
                ttl_ms=None,
                call_ctx=call_ctx,
            )
            evict_result = engine_adapter.execute_evict_local(
                engine_request_id=engine_request_id,
                ctx=scoped_ctx,
            )
        except ArtifactError as exc:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="evict_local",
                status=_status_failed(
                    f"evict_local failed: {exc}",
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="evict_local",
                status=_status_failed(
                    f"evict_local failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="evict_local",
            status=_status_success("evict_local completed"),
            artifact_result=evict_result,
        )

    def _prefetch(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.PrefetchAction,
        *,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        selection = action.selection
        target_id = step.target.target_id
        base_key = plan.context.idempotency_key
        if base_key:
            action_key = _derive_action_idempotency_key(
                base_key=base_key,
                action="prefetch",
                target_id=target_id,
                selection=selection,
                device_id=action.device_id,
                ttl_ms=None,
            )
            ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
            replica_uuid = str(uuid.uuid5(ns, action_key))
        else:
            replica_uuid = uuid.uuid4().hex
        timeout_s = _ctx_timeout_s(call_ctx)
        if timeout_s is not None and timeout_s <= 0:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="prefetch",
                status=_status_failed(
                    "CallContext deadline exceeded",
                    status_code="DEADLINE_EXCEEDED",
                    retryable=True,
                ),
            )
        try:
            self._materialize_selection(
                selection=selection,
                device_id=int(action.device_id),
                replica_uuid=replica_uuid,
                timeout_s=timeout_s,
                wait_for_completion=False,
            )
        except DeviceMismatch as exc:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="prefetch",
                status=_status_failed(
                    f"prefetch failed: {exc}",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="prefetch",
                status=_status_failed(
                    f"prefetch failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="prefetch",
            status=_status_success("prefetch issued"),
        )

    def _prefetch_set(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.PrefetchSetAction,
        *,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        timeout_s = _ctx_timeout_s(call_ctx)
        if timeout_s is not None and timeout_s <= 0:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="prefetch_set",
                status=_status_failed(
                    "CallContext deadline exceeded",
                    status_code="DEADLINE_EXCEEDED",
                    retryable=True,
                ),
            )
        try:
            artifact_set = ArtifactSetRef.from_proto(action.artifact_set)
            manifest_bridge = (
                ManifestArtifactSetBridge.from_proto(action.manifest_bridge)
                if action.HasField("manifest_bridge")
                else None
            )
        except ArtifactError as exc:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="prefetch_set",
                status=_status_failed(
                    f"prefetch_set failed: {exc}",
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                ),
            )
        try:
            resolved_items = resolve_artifact_set_ref(
                artifact_set,
                manifest_resolver=(
                    manifest_bridge.resolve_artifact_set
                    if manifest_bridge is not None
                    else None
                ),
            )
        except ArtifactError as exc:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="prefetch_set",
                status=_status_failed(
                    f"prefetch_set failed: {exc}",
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                ),
            )
        outcomes: list[ArtifactSetItemResult] = []
        for item in resolved_items:
            try:
                scoped_ctx = self._action_context(
                    plan=plan,
                    action="prefetch_set",
                    target_id=target_id,
                    selection=item.selection,
                    extra=artifact_set.set_digest_hex,
                    call_ctx=call_ctx,
                )
                base_key = scoped_ctx.idempotency_key
                if base_key:
                    ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
                    replica_uuid = str(uuid.uuid5(ns, base_key))
                else:
                    replica_uuid = uuid.uuid4().hex
                self._materialize_selection(
                    selection=item.selection,
                    device_id=int(action.device_id),
                    replica_uuid=replica_uuid,
                    timeout_s=timeout_s,
                    wait_for_completion=True,
                )
                item_status = OperationStatus(
                    state="success",
                    message="local_replica_ready",
                    as_of_ms=int(time.time() * 1000),
                )
            except DeviceMismatch as exc:
                item_status = _status_failed(
                    f"prefetch_set failed: {exc}",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            except Exception as exc:  # noqa: BLE001
                item_status = _status_failed(
                    f"prefetch_set failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                )
            outcomes.append(
                ArtifactSetItemResult(
                    item_identity=item.item_identity,
                    artifact_id=item.item_identity.artifact_id,
                    status=item_status,
                )
            )

        artifact_set_result = ArtifactSetResult(
            set_digest_hex=artifact_set.set_digest_hex,
            outcomes=tuple(outcomes),
        )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="prefetch_set",
            status=summarize_artifact_set_outcomes(
                action_name="prefetch_set",
                outcomes=artifact_set_result.outcomes,
            ),
            artifact_set_result=artifact_set_result,
        )

    def _materialize_selection(
        self,
        *,
        selection: common_pb2.ArtifactSelection,
        device_id: int,
        replica_uuid: str,
        timeout_s: float | None,
        wait_for_completion: bool,
    ) -> None:
        if device_id == CPU_DEVICE_ID:
            target_device_type = store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU
            device_uuid = ""
        else:
            target_device_type = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU
            device_uuid = device_uuid_for(device_id)
        self._client.materialize_by_artifact_id_v2(
            selection=selection,
            replica_uuid=replica_uuid,
            device_uuid=device_uuid,
            wait_for_completion=wait_for_completion,
            return_response=True,
            target_device_type=target_device_type,
            lease_mode=store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE,
            timeout_s=timeout_s,
        )

    def _pin(
        self,
        plan: plan_pb2.PlanSpec,
        step: plan_pb2.PlanStep,
        action: plan_pb2.PinDeviceResidencyAction,
        *,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        view_id = action.selection.view_id
        ttl_ms = action.ttl_ms if action.HasField("ttl_ms") else None
        timeout_s = _ctx_timeout_s(call_ctx)
        if timeout_s is not None and timeout_s <= 0:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="pin_device_residency",
                status=_status_failed(
                    "CallContext deadline exceeded",
                    status_code="DEADLINE_EXCEEDED",
                    retryable=True,
                ),
            )
        try:
            self._client.create_placement_lease(
                artifact_id=action.selection.artifact_id,
                view_id=view_id,
                device_id=int(action.device_id),
                ttl_ms=ttl_ms,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="pin_device_residency",
                status=_status_failed(
                    f"pin failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="pin_device_residency",
            status=_status_success("placement pin created"),
        )

    def _unpin(
        self,
        step: plan_pb2.PlanStep,
        action: plan_pb2.UnpinDeviceResidencyAction,
        *,
        call_ctx: CallContext,
    ) -> NodeAgentStepResult:
        target_id = step.target.target_id
        timeout_s = _ctx_timeout_s(call_ctx)
        if timeout_s is not None and timeout_s <= 0:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="unpin_device_residency",
                status=_status_failed(
                    "CallContext deadline exceeded",
                    status_code="DEADLINE_EXCEEDED",
                    retryable=True,
                ),
            )
        try:
            resp = self._client.release_placement_lease(
                lease_token=bytes(action.capability_token),
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="unpin_device_residency",
                status=_status_failed(
                    f"unpin failed: {exc}",
                    status_code="INTERNAL",
                    retryable=True,
                ),
            )
        if not resp.released:
            return NodeAgentStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action="unpin_device_residency",
                status=_status_failed(
                    "placement pin release failed",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                ),
            )
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action="unpin_device_residency",
            status=_status_success("placement pin released"),
        )

    def _cancelled_result(
        self, step: plan_pb2.PlanStep, message: str
    ) -> NodeAgentStepResult:
        return NodeAgentStepResult(
            step_id=step.step_id,
            target_id=step.target.target_id,
            action=step.action.WhichOneof("kind") or "unknown",
            status=OperationStatus(
                state="cancelled",
                message=message,
                as_of_ms=int(time.time() * 1000),
                error=OperationError(
                    status_code="CANCELLED",
                    message=message,
                    retryable=True,
                ),
            ),
        )


__all__ = [
    "NodeAgentExecutionResult",
    "NodeAgentExecutor",
    "NodeAgentStepResult",
]
