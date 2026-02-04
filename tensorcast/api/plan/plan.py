#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Mapping, Sequence, TypeVar

from tensorcast.api._config import StorePolicy
from tensorcast.api._device import CPU_DEVICE_ID
from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.operation import (
    Operation,
    OperationError,
    OperationStatus,
    OperationTimeoutError,
)
from tensorcast.api.plan.targets import TargetSpec
from tensorcast.api.plan.transforms import TransformSpec
from tensorcast.api.store.artifact import (
    Artifact,
    PlacementPin,
    PrefetchedReplica,
    _decode_capability_token,
)
from tensorcast.api.store.view_composer import compute_view_id
from tensorcast.common.selection_identity import (
    compute_logical_layout_hash,
    compute_selection_hash,
    compute_view_subset_hash,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.plan.v1 import plan_pb2

if TYPE_CHECKING:
    from tensorcast.api.store import Store

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Worker:
    worker_id: str
    daemon_address: str
    daemon_id: str
    p2p_port: int | None = None
    labels: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class Instance:
    instance_id: str
    worker_id: str
    engine: str
    signals_endpoint: str | None = None
    labels: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class PlanStepRef(Generic[T]):
    """Typed reference to a planned step (IR), not an executing operation."""

    step_id: str


@dataclass(frozen=True, slots=True)
class PlanStepResult:
    step_id: str
    target_id: str
    action: str
    status: OperationStatus
    value: Any | None = None


@dataclass(frozen=True, slots=True)
class PlanResult:
    ok: bool
    request_id: str
    steps: Mapping[str, PlanStepResult]

    def step(self, ref: PlanStepRef[Any]) -> PlanStepResult:
        return self.steps[ref.step_id]


class PlanFailedError(ArtifactError):
    """Raised when Plan.run observes any step failure."""

    def __init__(self, result: PlanResult) -> None:
        super().__init__(
            "Plan execution failed",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
        self.result = result


@dataclass(frozen=True, slots=True)
class _ArtifactSelection:
    artifact_id: str
    view_id: str
    logical_layout_hash: bytes
    selection_hash: bytes
    view_subset_hash: bytes | None
    view_spec: common_pb2.ViewSpec | None
    tensor_names: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _PrefetchAction:
    artifact: Artifact
    device: str | int
    device_id: int


@dataclass(frozen=True, slots=True)
class _PinAction:
    artifact: Artifact
    device_id: int
    ttl_ms: int | None


@dataclass(frozen=True, slots=True)
class _UnpinAction:
    pin: PlacementPin


@dataclass(frozen=True, slots=True)
class _TransformIntoAction:
    artifact: Artifact
    spec: TransformSpec
    targets: TargetSpec


@dataclass(frozen=True, slots=True)
class _TransformRegisterAction:
    artifact: Artifact
    spec: TransformSpec
    out_key: str
    policy: StorePolicy | None


_PlanAction = (
    _PrefetchAction
    | _PinAction
    | _UnpinAction
    | _TransformIntoAction
    | _TransformRegisterAction
)


@dataclass(frozen=True, slots=True)
class _PlanStep:
    step_id: str
    target: Worker | Instance
    action: _PlanAction
    depends_on: tuple[str, ...]


def _derive_action_idempotency_key(
    *,
    base_key: str,
    action: str,
    target_id: str,
    selection: _ArtifactSelection,
    device_id: int | None,
    ttl_ms: int | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(base_key.encode("utf-8"))
    digest.update(b"|")
    digest.update(action.encode("utf-8"))
    digest.update(b"|target=")
    digest.update(target_id.encode("utf-8"))
    digest.update(b"|artifact=")
    digest.update(selection.artifact_id.encode("utf-8"))
    if device_id is not None:
        digest.update(f"|device={int(device_id)}".encode("utf-8"))
    if ttl_ms is not None:
        digest.update(f"|ttl={int(ttl_ms)}".encode("utf-8"))
    digest.update(b"|logical=")
    digest.update(selection.logical_layout_hash)
    digest.update(b"|selection=")
    digest.update(selection.selection_hash)
    return f"tc.plan.action.v1:{digest.hexdigest()}"


def _resolve_device_id(*, device: str | int, allow_cpu: bool) -> int:
    if isinstance(device, int):
        if device < 0:
            if not allow_cpu:
                raise ArtifactError(
                    "CPU device is not supported for this operation",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return CPU_DEVICE_ID
        return int(device)
    import torch

    if device.strip().lower() == "dram":
        if not allow_cpu:
            raise ArtifactError(
                "CPU device is not supported for this operation",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return CPU_DEVICE_ID
    device_obj = torch.device(device)
    if device_obj.type == "cpu":
        if not allow_cpu:
            raise ArtifactError(
                "CPU device is not supported for this operation",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return CPU_DEVICE_ID
    return int(device_obj.index if device_obj.index is not None else 0)


def _ensure_index_bytes(artifact: Artifact) -> bytes:
    if artifact._canonical_index_bytes is not None:
        return artifact._canonical_index_bytes
    artifact._ensure_metadata()
    if artifact._canonical_index_bytes is None:
        raise ArtifactError(
            "Canonical index bytes unavailable for plan selection",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return artifact._canonical_index_bytes


def _resolve_view_id(
    *, artifact: Artifact, canonical_index_bytes: bytes
) -> tuple[str, common_pb2.ViewSpec | None]:
    if artifact._view_metadata is not None:
        return str(artifact._view_metadata.view_id), None
    if artifact._view_spec is None or artifact._view_spec.is_identity:
        return "", None
    view_proto = artifact._view_spec.proto
    if view_proto is None:
        raise ArtifactError(
            "View spec missing while resolving plan selection",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    try:
        return compute_view_id(view_proto, canonical_index_bytes), view_proto
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            "Failed to compute view_id for plan selection",
            status_code="INTERNAL",
            retryable=False,
        ) from exc


def _resolve_view_index_bytes(
    *,
    artifact: Artifact,
    view_proto: common_pb2.ViewSpec | None,
    canonical_index_bytes: bytes,
) -> bytes:
    if artifact._view_metadata is not None and artifact._view_metadata.view_index_bytes:
        return artifact._view_metadata.view_index_bytes
    if view_proto is None:
        raise ArtifactError(
            "View spec required for view index computation",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    from tensorcast._c_ext import compute_view_index_bytes

    normalized_ops: dict[str, list[dict[str, int | str]]] = {}
    for name, ops in view_proto.tensors.items():
        op_list: list[dict[str, int | str]] = []
        for op in ops.ops:
            if op.HasField("narrow"):
                op_list.append(
                    {
                        "type": "narrow",
                        "dim": int(op.narrow.dim),
                        "start": int(op.narrow.start),
                        "length": int(op.narrow.length),
                    }
                )
            elif op.HasField("transpose"):
                op_list.append(
                    {
                        "type": "transpose",
                        "dim0": int(op.transpose.dim0),
                        "dim1": int(op.transpose.dim1),
                    }
                )
        if op_list:
            normalized_ops[str(name)] = op_list

    payload = compute_view_index_bytes(canonical_index_bytes, normalized_ops)
    return bytes(payload["view_index_bytes"])


def _resolve_artifact_selection(artifact: Artifact) -> _ArtifactSelection:
    artifact_id = artifact._artifact_id or artifact._ensure_identified()
    canonical_index_bytes = _ensure_index_bytes(artifact)
    view_id, view_proto = _resolve_view_id(
        artifact=artifact, canonical_index_bytes=canonical_index_bytes
    )
    needs_view_index = bool(view_id)
    if needs_view_index:
        view_index_bytes = _resolve_view_index_bytes(
            artifact=artifact,
            view_proto=view_proto,
            canonical_index_bytes=canonical_index_bytes,
        )
        logical_layout_hash = compute_logical_layout_hash(
            index_bytes=view_index_bytes, needs_view_index=True
        )
    else:
        logical_layout_hash = compute_logical_layout_hash(
            index_bytes=canonical_index_bytes, needs_view_index=False
        )

    view_subset_hash: bytes | None = None
    tensor_names: tuple[str, ...] | None = None
    if artifact._view_metadata is not None and artifact._view_metadata.tensor_names:
        tensor_names = tuple(artifact._view_metadata.tensor_names)
        view_subset_hash = compute_view_subset_hash(tensor_names)

    selection_hash = compute_selection_hash(
        view_id=view_id, view_subset_hash=view_subset_hash
    )

    return _ArtifactSelection(
        artifact_id=artifact_id,
        view_id=view_id,
        logical_layout_hash=logical_layout_hash,
        selection_hash=selection_hash,
        view_subset_hash=view_subset_hash,
        view_spec=view_proto,
        tensor_names=tensor_names,
    )


def _clone_artifact_for_store(artifact: Artifact, store: "Store") -> Artifact:
    clone = Artifact(
        store_ref=weakref.ref(store),
        artifact_id=artifact._artifact_id,
        key=artifact._key_hint,
        disk_path=artifact._disk_path_hint,
        fallback=artifact._fallback,
        canonical_index_bytes=artifact._canonical_index_bytes,
        canonical_index=artifact._canonical_index,
        generation=artifact._generation,
        view_spec=artifact._view_spec,
        view_metadata=artifact._view_metadata,
        view_depth=artifact._view_depth,
    )
    clone._tensor_metas = dict(artifact._tensor_metas or {})
    clone._released = artifact._released
    return clone


def _status_from_exception(exc: Exception) -> OperationStatus:
    now_ms = int(time.time() * 1000)
    if isinstance(exc, OperationTimeoutError):
        error = OperationError(
            status_code="DEADLINE_EXCEEDED",
            message=str(exc),
            retryable=True,
        )
        return OperationStatus(
            state="degraded",
            message=str(exc),
            as_of_ms=now_ms,
            error=error,
        )
    if isinstance(exc, ArtifactError):
        error = OperationError(
            status_code=exc.status_code,
            message=str(exc),
            retryable=exc.retryable,
        )
        return OperationStatus(
            state="failed",
            message=str(exc),
            as_of_ms=now_ms,
            error=error,
        )
    error = OperationError(
        status_code="UNKNOWN",
        message=str(exc),
        retryable=False,
    )
    return OperationStatus(
        state="failed",
        message=str(exc),
        as_of_ms=now_ms,
        error=error,
    )


class WorkerStepBuilder:
    def __init__(self, plan: "Plan", worker: Worker) -> None:
        self._plan = plan
        self._worker = worker

    def prefetch(
        self,
        art: Artifact,
        *,
        device: str | int,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[PrefetchedReplica]:
        device_id = _resolve_device_id(device=device, allow_cpu=True)
        device_input: str | int = device
        return self._plan._add_step(
            target=self._worker,
            action=_PrefetchAction(
                artifact=art, device=device_input, device_id=device_id
            ),
            depends_on=depends_on,
        )

    def pin_device_residency(
        self,
        art: Artifact,
        *,
        device: str | int,
        ttl_ms: int | None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[PlacementPin]:
        if ttl_ms is not None and int(ttl_ms) <= 0:
            raise ArtifactError(
                "ttl_ms must be positive when provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        device_id = _resolve_device_id(device=device, allow_cpu=False)
        return self._plan._add_step(
            target=self._worker,
            action=_PinAction(artifact=art, device_id=device_id, ttl_ms=ttl_ms),
            depends_on=depends_on,
        )

    def unpin_device_residency(
        self,
        pin: PlacementPin,
        *,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[None]:
        if pin.daemon_id and pin.daemon_id != self._worker.daemon_id:
            raise ArtifactError(
                "PlacementPin daemon_id does not match worker target",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return self._plan._add_step(
            target=self._worker,
            action=_UnpinAction(pin=pin),
            depends_on=depends_on,
        )


class InstanceStepBuilder:
    def __init__(self, plan: "Plan", inst: Instance) -> None:
        self._plan = plan
        self._inst = inst

    def transform_into(
        self,
        art: Artifact,
        *,
        spec: TransformSpec,
        target: TargetSpec | None = None,
        targets: TargetSpec | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[None]:
        if not spec.name:
            raise ArtifactError(
                "TransformSpec.name is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if target is not None and targets is not None:
            raise ArtifactError(
                "Specify only one of target or targets",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resolved_target = target if target is not None else targets
        if resolved_target is None:
            raise ArtifactError(
                "target is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if resolved_target.instance_id != self._inst.instance_id:
            raise ArtifactError(
                "TargetSpec instance_id does not match instance target",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return self._plan._add_step(
            target=self._inst,
            action=_TransformIntoAction(
                artifact=art,
                spec=spec,
                targets=resolved_target,
            ),
            depends_on=depends_on,
        )

    def transform_register(
        self,
        art: Artifact,
        *,
        spec: TransformSpec,
        out_key: str,
        policy: StorePolicy | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[Artifact]:
        if not spec.name:
            raise ArtifactError(
                "TransformSpec.name is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not out_key:
            raise ArtifactError(
                "out_key is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return self._plan._add_step(
            target=self._inst,
            action=_TransformRegisterAction(
                artifact=art,
                spec=spec,
                out_key=out_key,
                policy=policy,
            ),
            depends_on=depends_on,
        )


class Plan:
    def __init__(self, ctx: CallContext) -> None:
        self._ctx = ctx
        self._request_id = ctx.request_id or uuid.uuid4().hex
        if ctx.idempotency_key:
            ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.plan.v1")
            self._plan_id = str(uuid.uuid5(ns, ctx.idempotency_key))
        else:
            self._plan_id = uuid.uuid4().hex
        self._steps: list[_PlanStep] = []
        self._step_ids: set[str] = set()
        self._selection_cache: dict[int, _ArtifactSelection] = {}
        self._stores: dict[str, "Store"] = {}
        self._lock = threading.RLock()
        self._step_counter = 0

    def on_worker(self, worker: Worker) -> WorkerStepBuilder:
        return WorkerStepBuilder(self, worker)

    def on_instance(self, inst: Instance) -> InstanceStepBuilder:
        return InstanceStepBuilder(self, inst)

    def _next_step_id(self) -> str:
        with self._lock:
            self._step_counter += 1
            return f"step-{self._step_counter:04d}"

    def _add_step(
        self,
        *,
        target: Worker | Instance,
        action: _PlanAction,
        depends_on: Sequence[PlanStepRef[Any]] | None,
    ) -> PlanStepRef[Any]:
        step_id = self._next_step_id()
        dep_ids = tuple(ref.step_id for ref in depends_on or ())
        missing = [dep for dep in dep_ids if dep not in self._step_ids]
        if missing:
            raise ArtifactError(
                f"Unknown dependency step ids: {', '.join(sorted(missing))}",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        step = _PlanStep(
            step_id=step_id,
            target=target,
            action=action,
            depends_on=dep_ids,
        )
        self._steps.append(step)
        self._step_ids.add(step_id)
        return PlanStepRef(step_id)

    def _selection_for_artifact(self, artifact: Artifact) -> _ArtifactSelection:
        key = id(artifact)
        cached = self._selection_cache.get(key)
        if cached is not None:
            return cached
        selection = _resolve_artifact_selection(artifact)
        self._selection_cache[key] = selection
        return selection

    def _call_context_proto(self) -> plan_pb2.CallContext:
        qos = self._ctx.qos
        qos_value = plan_pb2.QOS_CLASS_INTERACTIVE
        if qos == "realtime":
            qos_value = plan_pb2.QOS_CLASS_REALTIME
        elif qos == "background":
            qos_value = plan_pb2.QOS_CLASS_BACKGROUND
        proto = plan_pb2.CallContext(
            request_id=self._request_id,
            qos=qos_value,
            idempotency_key=self._ctx.idempotency_key or "",
        )
        if self._ctx.deadline_ms is not None:
            proto.deadline_ms = int(self._ctx.deadline_ms)
        if self._ctx.tags:
            proto.tags.update({str(k): str(v) for k, v in self._ctx.tags.items()})
        return proto

    def to_spec(self) -> plan_pb2.PlanSpec:
        spec = plan_pb2.PlanSpec(plan_id=self._plan_id)
        spec.context.CopyFrom(self._call_context_proto())
        for step in self._steps:
            step_msg = spec.steps.add()
            step_msg.step_id = step.step_id
            step_msg.depends_on.extend(step.depends_on)
            if isinstance(step.target, Worker):
                step_msg.target.target_type = plan_pb2.TARGET_TYPE_WORKER
                step_msg.target.target_id = step.target.daemon_id
            else:
                step_msg.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
                step_msg.target.target_id = step.target.instance_id
            if isinstance(step.action, _PrefetchAction):
                selection = self._selection_for_artifact(step.action.artifact)
                prefetch_action = step_msg.action.prefetch
                _fill_selection_proto(selection, prefetch_action.selection)
                prefetch_action.device_id = int(step.action.device_id)
            elif isinstance(step.action, _PinAction):
                selection = self._selection_for_artifact(step.action.artifact)
                pin_action = step_msg.action.pin_device_residency
                _fill_selection_proto(selection, pin_action.selection)
                pin_action.device_id = int(step.action.device_id)
                if step.action.ttl_ms is not None:
                    pin_action.ttl_ms = int(step.action.ttl_ms)
            elif isinstance(step.action, _UnpinAction):
                unpin_action = step_msg.action.unpin_device_residency
                unpin_action.pin_id = str(step.action.pin.pin_id)
                unpin_action.capability_token = _decode_capability_token(
                    step.action.pin.capability_token
                )
            elif isinstance(step.action, _TransformIntoAction):
                selection = self._selection_for_artifact(step.action.artifact)
                transform_into_action = step_msg.action.transform_into
                _fill_selection_proto(selection, transform_into_action.selection)
                transform_into_action.spec.CopyFrom(step.action.spec.to_proto())
                transform_into_action.target.CopyFrom(step.action.targets.to_proto())
            elif isinstance(step.action, _TransformRegisterAction):
                selection = self._selection_for_artifact(step.action.artifact)
                transform_register_action = step_msg.action.transform_register
                _fill_selection_proto(selection, transform_register_action.selection)
                transform_register_action.spec.CopyFrom(step.action.spec.to_proto())
                transform_register_action.out_key = str(step.action.out_key)
                if step.action.policy is not None:
                    transform_register_action.policy.CopyFrom(
                        step.action.policy.to_proto()
                    )
            else:
                raise ArtifactError(
                    "Unknown plan action",
                    status_code="INTERNAL",
                    retryable=False,
                )
        return spec

    def _store_for_worker(self, worker: Worker) -> "Store":
        store = self._stores.get(worker.daemon_id)
        if store is None:
            from tensorcast.api.store import Store

            store = Store(worker.daemon_address)
            self._stores[worker.daemon_id] = store
        return store

    def _action_context(
        self,
        *,
        action: str,
        target_id: str,
        selection: _ArtifactSelection,
        device_id: int | None,
        ttl_ms: int | None,
    ) -> CallContext:
        if not self._ctx.idempotency_key:
            return self._ctx
        derived_key = _derive_action_idempotency_key(
            base_key=self._ctx.idempotency_key,
            action=action,
            target_id=target_id,
            selection=selection,
            device_id=device_id,
            ttl_ms=ttl_ms,
        )
        return CallContext(
            request_id=self._request_id,
            qos=self._ctx.qos,
            deadline_ms=self._ctx.deadline_ms,
            idempotency_key=derived_key,
            tags=self._ctx.tags,
        )

    def _execute_step(
        self,
        step: _PlanStep,
        *,
        cancel_event: threading.Event,
        op_registry: dict[str, Operation[Any]],
        op_lock: threading.Lock,
    ) -> PlanStepResult:
        if cancel_event.is_set():
            return _cancelled_result(step, "skipped due to prior failure")
        if isinstance(step.target, Instance):
            return PlanStepResult(
                step_id=step.step_id,
                target_id=step.target.instance_id,
                action=_action_name(step.action),
                status=OperationStatus(
                    state="failed",
                    message="Instance steps require Node Agent execution",
                    as_of_ms=int(time.time() * 1000),
                    error=OperationError(
                        status_code="UNIMPLEMENTED",
                        message="Instance steps require Node Agent execution",
                        retryable=False,
                    ),
                ),
            )
        worker = step.target
        store = self._store_for_worker(worker)
        target_id = worker.daemon_id
        try:
            if isinstance(step.action, _PrefetchAction):
                selection = self._selection_for_artifact(step.action.artifact)
                ctx = self._action_context(
                    action="prefetch",
                    target_id=target_id,
                    selection=selection,
                    device_id=step.action.device_id,
                    ttl_ms=None,
                )
                bound = _clone_artifact_for_store(step.action.artifact, store)
                prefetch_op = bound.prefetch(device=step.action.device, ctx=ctx)
                with op_lock:
                    op_registry[step.step_id] = prefetch_op
                prefetch_value = prefetch_op.wait()
                status = prefetch_op.status()
                return PlanStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action="prefetch",
                    status=status,
                    value=prefetch_value,
                )
            if isinstance(step.action, _PinAction):
                selection = self._selection_for_artifact(step.action.artifact)
                ctx = self._action_context(
                    action="pin_device_residency",
                    target_id=target_id,
                    selection=selection,
                    device_id=step.action.device_id,
                    ttl_ms=step.action.ttl_ms,
                )
                bound = _clone_artifact_for_store(step.action.artifact, store)
                pin_op = bound.pin_device_residency(
                    device=step.action.device_id,
                    ttl_ms=step.action.ttl_ms,
                    ctx=ctx,
                )
                with op_lock:
                    op_registry[step.step_id] = pin_op
                pin_value = pin_op.wait()
                status = pin_op.status()
                return PlanStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action="pin_device_residency",
                    status=status,
                    value=pin_value,
                )
            if isinstance(step.action, _UnpinAction):
                pin = step.action.pin
                token = _decode_capability_token(pin.capability_token)
                released = store._runtime.ensure_client().release_placement_lease(
                    lease_token=token
                )
                status = OperationStatus(
                    state="success" if released.released else "failed",
                    message="placement pin released"
                    if released.released
                    else "placement pin release failed",
                    as_of_ms=int(time.time() * 1000),
                    error=None
                    if released.released
                    else OperationError(
                        status_code="FAILED_PRECONDITION",
                        message="placement pin release failed",
                        retryable=False,
                    ),
                )
                return PlanStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action="unpin_device_residency",
                    status=status,
                    value=None,
                )
        except Exception as exc:  # noqa: BLE001
            return PlanStepResult(
                step_id=step.step_id,
                target_id=target_id,
                action=_action_name(step.action),
                status=_status_from_exception(exc),
                value=None,
            )
        return PlanStepResult(
            step_id=step.step_id,
            target_id=target_id,
            action=_action_name(step.action),
            status=_status_from_exception(
                ArtifactError(
                    "Unknown plan action",
                    status_code="INTERNAL",
                    retryable=False,
                )
            ),
            value=None,
        )

    def run(
        self,
        *,
        concurrency: int = 16,
        raise_on_error: bool = True,
    ) -> PlanResult:
        if concurrency <= 0:
            raise ArtifactError(
                "concurrency must be positive",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        step_map = {step.step_id: step for step in self._steps}
        if not step_map:
            empty_result = PlanResult(ok=True, request_id=self._request_id, steps={})
            return empty_result
        dependencies: dict[str, set[str]] = {
            step_id: set(step.depends_on) for step_id, step in step_map.items()
        }
        dependents: dict[str, set[str]] = {step_id: set() for step_id in step_map}
        for step_id, deps in dependencies.items():
            for dep in deps:
                if dep not in step_map:
                    raise ArtifactError(
                        f"Unknown dependency step id: {dep}",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                dependents[dep].add(step_id)

        ready = [step_id for step_id, deps in dependencies.items() if not deps]
        if not ready:
            raise ArtifactError(
                "Plan has no runnable steps (cycle detected)",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        results: dict[str, PlanStepResult] = {}
        in_flight: dict[concurrent.futures.Future[PlanStepResult], str] = {}
        cancel_event = threading.Event()
        op_registry: dict[str, Operation[Any]] = {}
        op_lock = threading.Lock()

        def submit(step_id: str) -> None:
            future = executor.submit(
                self._execute_step,
                step_map[step_id],
                cancel_event=cancel_event,
                op_registry=op_registry,
                op_lock=op_lock,
            )
            in_flight[future] = step_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            while len(results) < len(step_map):
                while (
                    ready and len(in_flight) < concurrency and not cancel_event.is_set()
                ):
                    submit(ready.pop(0))
                if not in_flight:
                    break
                done, _ = concurrent.futures.wait(
                    in_flight.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    step_id = in_flight.pop(future)
                    try:
                        step_result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        step_result = PlanStepResult(
                            step_id=step_id,
                            target_id="",
                            action="unknown",
                            status=_status_from_exception(exc),
                            value=None,
                        )
                    results[step_id] = step_result
                    if step_result.status.state != "success":
                        cancel_event.set()
                    for child in dependents[step_id]:
                        child_deps = dependencies.get(child)
                        if child_deps is None:
                            continue
                        child_deps.discard(step_id)
                        if (
                            not child_deps
                            and child not in results
                            and child not in ready
                        ):
                            ready.append(child)

                if cancel_event.is_set():
                    with op_lock:
                        for op in op_registry.values():
                            op.cancel()
                    for future in list(in_flight.keys()):
                        future.cancel()

        if cancel_event.is_set():
            for step_id in step_map:
                if step_id in results:
                    continue
                results[step_id] = _cancelled_result(
                    step_map[step_id], "skipped due to prior failure"
                )

        ok = all(result.status.state == "success" for result in results.values())
        plan_result = PlanResult(
            ok=ok,
            request_id=self._request_id,
            steps=dict(results),
        )
        if raise_on_error and not ok:
            raise PlanFailedError(plan_result)
        return plan_result


def _fill_selection_proto(
    selection: _ArtifactSelection, target: common_pb2.ArtifactSelection
) -> None:
    target.artifact_id = selection.artifact_id
    target.view_id = selection.view_id
    target.logical_layout_hash = selection.logical_layout_hash
    target.selection_hash = selection.selection_hash
    if selection.view_subset_hash is not None:
        target.view_subset_hash = selection.view_subset_hash
    if selection.view_spec is not None:
        target.view_spec.CopyFrom(selection.view_spec)
    if selection.tensor_names:
        target.tensor_names.extend(selection.tensor_names)


def _action_name(action: _PlanAction) -> str:
    if isinstance(action, _PrefetchAction):
        return "prefetch"
    if isinstance(action, _PinAction):
        return "pin_device_residency"
    if isinstance(action, _UnpinAction):
        return "unpin_device_residency"
    if isinstance(action, _TransformIntoAction):
        return "transform_into"
    if isinstance(action, _TransformRegisterAction):
        return "transform_register"
    return "unknown"


def _cancelled_result(step: _PlanStep, message: str) -> PlanStepResult:
    target_id = ""
    if isinstance(step.target, Worker):
        target_id = step.target.daemon_id
    elif isinstance(step.target, Instance):
        target_id = step.target.instance_id
    status = OperationStatus(
        state="cancelled",
        message=message,
        as_of_ms=int(time.time() * 1000),
        error=OperationError(
            status_code="CANCELLED",
            message=message,
            retryable=True,
        ),
    )
    return PlanStepResult(
        step_id=step.step_id,
        target_id=target_id,
        action=_action_name(step.action),
        status=status,
        value=None,
    )


def plan(ctx: CallContext) -> Plan:
    return Plan(ctx)


__all__ = [
    "Instance",
    "Plan",
    "PlanFailedError",
    "PlanResult",
    "PlanStepRef",
    "PlanStepResult",
    "Worker",
    "plan",
]
