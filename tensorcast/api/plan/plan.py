#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Mapping, Sequence, TypeVar, cast

from tensorcast.api._config import StorePolicy
from tensorcast.api._device import CPU_DEVICE_ID
from tensorcast.api._view_ops import NarrowOp, TransposeOp, ViewSpecBuildResult
from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.operation import (
    Operation,
    OperationError,
    OperationState,
    OperationStatus,
    OperationTimeoutError,
)
from tensorcast.api.plan.artifact_set import (
    ArtifactSetItemResult,
    ArtifactSetRef,
    ArtifactSetResult,
    resolve_artifact_set_ref,
    selection_identity_from_proto,
    summarize_artifact_set_outcomes,
)
from tensorcast.api.plan.targets import TargetSpec
from tensorcast.api.plan.transforms import TransformSpec
from tensorcast.api.store.artifact import (
    Artifact,
    PlacementPin,
    PrefetchedReplica,
    _decode_capability_token,
)
from tensorcast.api.store.serving_builder import build_pure_transform_transform_spec
from tensorcast.api.store.view_composer import compute_view_id
from tensorcast.engine_adapter.artifact_api import (
    BatchOutcome,
    BatchResult,
    HydrateResult,
    ManifestArtifactSetBridge,
    ManifestResult,
    PublishManifest,
    PublishResult,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.node_agent.v1 import node_agent_pb2
from tensorcast.proto.plan.v1 import plan_pb2
from tensorcast.types import (
    _SERVING_READINESS_TO_PROTO,
    AssemblyCloseoutContract,
    AssemblyContractFamily,
    AssemblyReadinessPolicy,
    AssemblyRequirementSetRef,
    PrefetchedServingBinding,
    PrefetchedServingBindingSet,
    PrefetchRetentionPolicy,
    RepresentationPublishContract,
    RepresentationPublishSpec,
    ServingArtifactManifest,
    ServingBindingReadiness,
    ServingBindingSetTarget,
    ServingBindingTarget,
)

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.types import ServingBuildIntent

T = TypeVar("T")
ArtifactActionResult = (
    ManifestResult
    | PublishResult
    | HydrateResult
    | BatchResult
    | RepresentationPublishSpec
)
PlanExecutionClass = str

_TERMINAL_ONLY_EXECUTION_CLASS = "terminal_only"
_PUBLIC_CONTINUATION_REQUIRED_EXECUTION_CLASS = "public_continuation_required"
_SUPPORTED_EXECUTION_CLASSES = {
    _TERMINAL_ONLY_EXECUTION_CLASS,
    _PUBLIC_CONTINUATION_REQUIRED_EXECUTION_CLASS,
}


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
    worker_id: str | None = None
    engine: str = ""
    daemon_id: str | None = None
    execution_endpoint: str | None = None
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
    artifact_result: ArtifactActionResult | None = None
    artifact_set_result: ArtifactSetResult | None = None


@dataclass(frozen=True, slots=True)
class PlanResult:
    ok: bool
    request_id: str
    steps: Mapping[str, PlanStepResult]

    def step(self, ref: PlanStepRef[Any]) -> PlanStepResult:
        return self.steps[ref.step_id]

    def require_representation_publish_spec(
        self, ref: PlanStepRef[Any]
    ) -> RepresentationPublishSpec:
        step = self.steps.get(ref.step_id)
        if step is None:
            raise ArtifactError(
                f"Unknown plan step id: {ref.step_id}",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if step.status.state != "success":
            raise ArtifactError(
                f"Plan step {ref.step_id} did not succeed; pure-transform publication is unavailable",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not isinstance(step.artifact_result, RepresentationPublishSpec):
            raise ArtifactError(
                f"Plan step {ref.step_id} does not carry a RepresentationPublishSpec",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return step.artifact_result

    @classmethod
    def from_node_agent_response(
        cls, response: node_agent_pb2.ExecutePlanResponse
    ) -> "PlanResult":
        steps: dict[str, PlanStepResult] = {}
        for step in response.steps:
            artifact_result = (
                _artifact_result_from_proto(step.artifact_result)
                if step.HasField("artifact_result")
                else None
            )
            artifact_set_result = (
                _artifact_set_result_from_proto(step.artifact_set_result)
                if step.HasField("artifact_set_result")
                else None
            )
            steps[str(step.step_id)] = PlanStepResult(
                step_id=str(step.step_id),
                target_id=str(step.target_id),
                action=str(step.action),
                status=_operation_status_from_node_agent_proto(step.status),
                value=artifact_set_result,
                artifact_result=artifact_result,
                artifact_set_result=artifact_set_result,
            )
        return cls(
            ok=bool(response.ok),
            request_id=str(response.request_id),
            steps=steps,
        )


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
    device: str | int | None
    device_id: int
    target: ServingBindingTarget | ServingBindingSetTarget | None = None
    readiness: ServingBindingReadiness = "serving_local_ready"
    retention: PrefetchRetentionPolicy | None = None


@dataclass(frozen=True, slots=True)
class _PrefetchSetAction:
    artifact_set: ArtifactSetRef
    manifest_bridge: ManifestArtifactSetBridge | None
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


@dataclass(frozen=True, slots=True)
class _ManifestAction:
    engine_request_id: str


@dataclass(frozen=True, slots=True)
class _PublishAction:
    engine_request_id: str
    ttl_ms: int | None


@dataclass(frozen=True, slots=True)
class _HydrateAction:
    engine_request_id: str | None
    publish_manifest: PublishManifest | None


@dataclass(frozen=True, slots=True)
class _EvictLocalAction:
    engine_request_id: str | None


_PlanAction = (
    _PrefetchAction
    | _PrefetchSetAction
    | _PinAction
    | _UnpinAction
    | _TransformIntoAction
    | _TransformRegisterAction
    | _ManifestAction
    | _PublishAction
    | _HydrateAction
    | _EvictLocalAction
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
    extra: str | None = None,
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
    if extra:
        digest.update(b"|extra=")
        digest.update(extra.encode("utf-8"))
    return f"tc.plan.action.v1:{digest.hexdigest()}"


_NODE_AGENT_STATE_TO_OP_STATE: dict[int, OperationState] = {
    node_agent_pb2.OPERATION_STATE_PENDING: "pending",
    node_agent_pb2.OPERATION_STATE_RUNNING: "running",
    node_agent_pb2.OPERATION_STATE_SUCCESS: "success",
    node_agent_pb2.OPERATION_STATE_FAILED: "failed",
    node_agent_pb2.OPERATION_STATE_CANCELLED: "cancelled",
    node_agent_pb2.OPERATION_STATE_DEGRADED: "degraded",
}


def _timestamp_to_epoch_ms(ts: object) -> int | None:
    seconds = getattr(ts, "seconds", None)
    nanos = getattr(ts, "nanos", None)
    if seconds is None or nanos is None:
        return None
    return int(int(seconds) * 1000 + int(nanos) // 1_000_000)


def _batch_outcome_from_proto(
    outcome: node_agent_pb2.ArtifactBatchOutcome,
) -> BatchOutcome:
    return BatchOutcome(
        artifact_id=str(outcome.artifact_id),
        status_code=str(outcome.status_code),
        message=str(outcome.message) if outcome.message else None,
    )


def _manifest_result_from_proto(
    result: object,
) -> ManifestResult:
    return ManifestResult.from_proto(result)


def _artifact_result_from_proto(
    result: node_agent_pb2.ArtifactActionResult,
) -> ArtifactActionResult | None:
    kind = result.WhichOneof("result")
    if kind == "manifest":
        return _manifest_result_from_proto(result.manifest)
    if kind == "representation_publish":
        return RepresentationPublishSpec.from_proto(result.representation_publish)
    if kind == "publish":
        return PublishResult(
            manifest=_manifest_result_from_proto(result.publish.manifest),
            put_outcomes=tuple(
                _batch_outcome_from_proto(item) for item in result.publish.put_outcomes
            ),
            publish_manifest=(
                PublishManifest.from_proto(result.publish.publish_manifest)
                if result.publish.HasField("publish_manifest")
                else None
            ),
        )
    if kind == "hydrate":
        return HydrateResult(
            manifest=(
                _manifest_result_from_proto(result.hydrate.manifest)
                if result.hydrate.HasField("manifest")
                else None
            ),
            get_outcomes=tuple(
                _batch_outcome_from_proto(item) for item in result.hydrate.get_outcomes
            ),
            missing_artifact_ids=tuple(
                str(item) for item in result.hydrate.missing_artifact_ids
            ),
        )
    if kind == "evict_local":
        return BatchResult(
            engine_request_id=(
                str(result.evict_local.engine_request_id)
                if result.evict_local.HasField("engine_request_id")
                else None
            ),
            outcomes=tuple(
                _batch_outcome_from_proto(item) for item in result.evict_local.outcomes
            ),
        )
    if kind == "pure_transform_publication":
        return RepresentationPublishSpec(
            serving_artifact_id=(
                str(result.pure_transform_publication.serving_artifact_id) or None
            ),
            serving_manifest_ref=str(
                result.pure_transform_publication.serving_manifest_ref
            ),
            serving_manifest=ServingArtifactManifest.from_bytes(
                bytes(result.pure_transform_publication.serving_manifest_bytes)
            ),
            serving_manifest_bytes=bytes(
                result.pure_transform_publication.serving_manifest_bytes
            ),
            source_artifact_ref=(
                str(result.pure_transform_publication.source_artifact_ref)
                if result.pure_transform_publication.HasField("source_artifact_ref")
                else None
            ),
            contract_family=cast(
                AssemblyContractFamily | None,
                str(result.pure_transform_publication.contract_family)
                if result.pure_transform_publication.HasField("contract_family")
                else None,
            ),
            structural_view_ids=tuple(
                str(item)
                for item in result.pure_transform_publication.structural_view_ids
            ),
            representation_publish_contract=RepresentationPublishContract.from_proto(
                result.pure_transform_publication.representation_publish_contract
            ),
            closeout_contract=AssemblyCloseoutContract.from_proto(
                result.pure_transform_publication.closeout_contract
            ),
        )
    return None


def _artifact_set_result_from_proto(
    result: node_agent_pb2.ArtifactSetResult,
) -> ArtifactSetResult:
    return ArtifactSetResult(
        set_digest_hex=str(result.set_digest_hex),
        outcomes=tuple(
            ArtifactSetItemResult(
                item_identity=selection_identity_from_proto(item.item_identity),
                artifact_id=(
                    str(item.artifact_id) if item.HasField("artifact_id") else None
                ),
                status=_operation_status_from_node_agent_proto(item.status),
            )
            for item in result.outcomes
        ),
    )


def _operation_status_from_node_agent_proto(
    status: node_agent_pb2.OperationStatus,
) -> OperationStatus:
    error: OperationError | None = None
    if status.HasField("error"):
        error = OperationError(
            status_code=str(status.error.status_code or "UNKNOWN"),
            message=str(status.error.message or ""),
            retryable=bool(status.error.retryable),
        )
    return OperationStatus(
        state=_NODE_AGENT_STATE_TO_OP_STATE.get(int(status.state), "running"),
        message=str(status.message) if status.message else None,
        progress=float(status.progress) if status.progress else None,
        as_of_ms=_timestamp_to_epoch_ms(status.as_of)
        if status.HasField("as_of")
        else None,
        error=error,
    )


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
    selection = artifact._build_artifact_selection()
    view_subset_hash = (
        bytes(selection.view_subset_hash) if selection.view_subset_hash else None
    )
    view_proto: common_pb2.ViewSpec | None = (
        selection.view_spec if selection.HasField("view_spec") else None
    )
    tensor_names = tuple(selection.tensor_names) if selection.tensor_names else None

    return _ArtifactSelection(
        artifact_id=str(selection.artifact_id),
        view_id=str(selection.view_id),
        logical_layout_hash=bytes(selection.logical_layout_hash),
        selection_hash=bytes(selection.selection_hash),
        view_subset_hash=view_subset_hash,
        view_spec=view_proto,
        tensor_names=tensor_names,
    )


def _clone_artifact_for_store(artifact: Artifact, store: "Store") -> Artifact:
    clone = Artifact(
        store_ref=weakref.ref(store),
        artifact_id=artifact._artifact_id,
        key=artifact._key_hint,
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


def _artifact_from_selection_proto(
    *,
    store: "Store",
    selection: common_pb2.ArtifactSelection,
) -> Artifact:
    view_spec = (
        _view_spec_from_proto(selection.view_spec)
        if selection.HasField("view_spec")
        else None
    )
    artifact = Artifact(
        store_ref=weakref.ref(store),
        artifact_id=str(selection.artifact_id),
        view_spec=view_spec,
    )
    if selection.tensor_names:
        artifact = artifact.subset(list(selection.tensor_names))
    return artifact


def _view_spec_from_proto(
    view_spec: common_pb2.ViewSpec,
) -> ViewSpecBuildResult | None:
    if view_spec is None or not view_spec.tensors:
        return None
    tensor_ops: dict[str, tuple[NarrowOp | TransposeOp, ...]] = {}
    for name, ops in view_spec.tensors.items():
        converted = []
        for op in ops.ops:
            if op.HasField("narrow"):
                converted.append(
                    NarrowOp(
                        dim=int(op.narrow.dim),
                        start=int(op.narrow.start),
                        length=int(op.narrow.length),
                    )
                )
            elif op.HasField("transpose"):
                converted.append(
                    TransposeOp(
                        dim0=int(op.transpose.dim0),
                        dim1=int(op.transpose.dim1),
                    )
                )
        tensor_ops[str(name)] = tuple(converted)
    return ViewSpecBuildResult(proto=view_spec, tensor_ops=tensor_ops)


def _selection_from_proto(
    selection: common_pb2.ArtifactSelection,
) -> _ArtifactSelection:
    return _ArtifactSelection(
        artifact_id=str(selection.artifact_id),
        view_id=str(selection.view_id),
        logical_layout_hash=bytes(selection.logical_layout_hash),
        selection_hash=bytes(selection.selection_hash),
        view_subset_hash=(
            bytes(selection.view_subset_hash) if selection.view_subset_hash else None
        ),
        view_spec=selection.view_spec if selection.HasField("view_spec") else None,
        tensor_names=tuple(selection.tensor_names) if selection.tensor_names else None,
    )


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
        device: str | int | None = None,
        target: ServingBindingTarget | ServingBindingSetTarget | None = None,
        readiness: ServingBindingReadiness = "serving_local_ready",
        retention: PrefetchRetentionPolicy | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[
        PrefetchedReplica | PrefetchedServingBinding | PrefetchedServingBindingSet
    ]:
        if target is not None and device is not None:
            raise ArtifactError(
                "prefetch target and device are mutually exclusive",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if target is None:
            if device is None:
                raise ArtifactError(
                    "prefetch requires device or target",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            device_id = _resolve_device_id(device=device, allow_cpu=True)
        else:
            device_id = -1
        return self._plan._add_step(
            target=self._worker,
            action=_PrefetchAction(
                artifact=art,
                device=device,
                device_id=device_id,
                target=target,
                readiness=readiness,
                retention=retention,
            ),
            depends_on=depends_on,
        )

    def prefetch_set(
        self,
        art_set: ArtifactSetRef,
        *,
        device: str | int,
        manifest_bridge: ManifestArtifactSetBridge | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[ArtifactSetResult]:
        device_id = _resolve_device_id(device=device, allow_cpu=True)
        if (
            manifest_bridge is not None
            and manifest_bridge.artifact_set_ref.to_proto().SerializeToString(
                deterministic=True
            )
            != art_set.to_proto().SerializeToString(deterministic=True)
        ):
            raise ArtifactError(
                "ManifestArtifactSetBridge.artifact_set_ref must match prefetch_set ArtifactSetRef",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return self._plan._add_step(
            target=self._worker,
            action=_PrefetchSetAction(
                artifact_set=art_set,
                manifest_bridge=manifest_bridge,
                device=device,
                device_id=device_id,
            ),
            depends_on=depends_on,
        )

    def prefetch_many(
        self,
        arts: Sequence[Artifact],
        *,
        device: str | int,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[ArtifactSetResult]:
        artifact_set = ArtifactSetRef.inline(
            tuple(
                self._plan._selection_proto_for_artifact(artifact) for artifact in arts
            )
        )
        return self.prefetch_set(
            artifact_set,
            device=device,
            depends_on=depends_on,
        )

    def prefetch_manifest_result(
        self,
        manifest_result: ManifestResult,
        *,
        device: str | int,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[ArtifactSetResult]:
        manifest_bridge = manifest_result.require_artifact_set_bridge()
        return self.prefetch_set(
            manifest_bridge.artifact_set_ref,
            device=device,
            manifest_bridge=manifest_bridge,
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

    @staticmethod
    def _validated_engine_request_id(engine_request_id: str) -> str:
        # `engine_request_id` remains adapter-local context. Cross-step set
        # identity belongs to manifest/projection bridge contracts instead.
        value = str(engine_request_id).strip()
        if not value:
            raise ArtifactError(
                "engine_request_id is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return value

    @staticmethod
    def _validated_publish_manifest(
        publish_manifest: PublishManifest,
    ) -> PublishManifest:
        if not isinstance(publish_manifest, PublishManifest):
            raise ArtifactError(
                "publish_manifest must be a PublishManifest",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return publish_manifest

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

    def transform_register_pure_transform(
        self,
        art: Artifact,
        *,
        build_intent: "ServingBuildIntent",
        contract_family: str | None = None,
        out_key: str,
        transform_name: str = "identity.v1",
        source_version_key: str | None = None,
        serving_version_key: str | None = None,
        logical_topology_json: str | None = None,
        serving_manifest_ref: str | None = None,
        layout_id: str | None = None,
        requirements: AssemblyRequirementSetRef | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        structural_view_ids: Sequence[str] | None = None,
        transform_args: dict[str, str | int] | None = None,
        layout_hash: str | None = None,
        policy: StorePolicy | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[RepresentationPublishSpec]:
        spec = build_pure_transform_transform_spec(
            transform_name=transform_name,
            build_intent=build_intent,
            contract_family=contract_family,
            source_version_key=source_version_key,
            serving_version_key=serving_version_key,
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=serving_manifest_ref,
            layout_id=layout_id,
            requirements=requirements,
            readiness_policy=readiness_policy,
            structural_view_ids=tuple(
                str(view_id).strip()
                for view_id in (structural_view_ids or ())
                if str(view_id).strip()
            ),
            transform_args=transform_args,
            layout_hash=layout_hash,
        )
        return cast(
            PlanStepRef[RepresentationPublishSpec],
            self.transform_register(
                art,
                spec=spec,
                out_key=out_key,
                policy=policy,
                depends_on=depends_on,
            ),
        )

    def manifest(
        self,
        *,
        engine_request_id: str,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[Any]:
        return self._manifest(
            engine_request_id=engine_request_id, depends_on=depends_on
        )

    def publish(
        self,
        *,
        engine_request_id: str,
        ttl_ms: int | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[Any]:
        return self._publish(
            engine_request_id=engine_request_id,
            ttl_ms=ttl_ms,
            depends_on=depends_on,
        )

    def hydrate(
        self,
        *,
        engine_request_id: str | None = None,
        publish_manifest: PublishManifest | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[Any]:
        return self._hydrate(
            engine_request_id=engine_request_id,
            publish_manifest=publish_manifest,
            depends_on=depends_on,
        )

    def evict_local(
        self,
        *,
        engine_request_id: str | None = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[Any]:
        return self._evict_local(
            engine_request_id=engine_request_id, depends_on=depends_on
        )

    def _manifest(
        self,
        *,
        engine_request_id: str,
        depends_on: Sequence[PlanStepRef[Any]] | None,
    ) -> PlanStepRef[Any]:
        return self._plan._add_step(
            target=self._inst,
            action=_ManifestAction(
                engine_request_id=self._validated_engine_request_id(engine_request_id),
            ),
            depends_on=depends_on,
        )

    def _publish(
        self,
        *,
        engine_request_id: str,
        ttl_ms: int | None,
        depends_on: Sequence[PlanStepRef[Any]] | None,
    ) -> PlanStepRef[Any]:
        if ttl_ms is not None and int(ttl_ms) <= 0:
            raise ArtifactError(
                "ttl_ms must be positive when provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return self._plan._add_step(
            target=self._inst,
            action=_PublishAction(
                engine_request_id=self._validated_engine_request_id(engine_request_id),
                ttl_ms=int(ttl_ms) if ttl_ms is not None else None,
            ),
            depends_on=depends_on,
        )

    def _hydrate(
        self,
        *,
        engine_request_id: str | None,
        publish_manifest: PublishManifest | None,
        depends_on: Sequence[PlanStepRef[Any]] | None,
    ) -> PlanStepRef[Any]:
        resolved_engine_request_id = (
            self._validated_engine_request_id(engine_request_id)
            if engine_request_id is not None
            else None
        )
        resolved_publish_manifest = (
            self._validated_publish_manifest(publish_manifest)
            if publish_manifest is not None
            else None
        )
        if (resolved_engine_request_id is None) == (resolved_publish_manifest is None):
            raise ArtifactError(
                "exactly one of engine_request_id or publish_manifest is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return self._plan._add_step(
            target=self._inst,
            action=_HydrateAction(
                engine_request_id=resolved_engine_request_id,
                publish_manifest=resolved_publish_manifest,
            ),
            depends_on=depends_on,
        )

    def _evict_local(
        self,
        *,
        engine_request_id: str | None,
        depends_on: Sequence[PlanStepRef[Any]] | None,
    ) -> PlanStepRef[Any]:
        resolved_engine_request_id: str | None = None
        if engine_request_id is not None:
            resolved_engine_request_id = self._validated_engine_request_id(
                engine_request_id
            )
        return self._plan._add_step(
            target=self._inst,
            action=_EvictLocalAction(
                engine_request_id=resolved_engine_request_id,
            ),
            depends_on=depends_on,
        )


class Plan:
    def __init__(
        self,
        ctx: CallContext,
        *,
        runtime: Any | None = None,
        execution_class: PlanExecutionClass = _TERMINAL_ONLY_EXECUTION_CLASS,
    ) -> None:
        if execution_class not in _SUPPORTED_EXECUTION_CLASSES:
            raise ArtifactError(
                "execution_class must be 'terminal_only' or 'public_continuation_required'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._ctx = ctx
        self._runtime = runtime
        self._execution_class = execution_class
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

    def _selection_proto_for_artifact(
        self, artifact: Artifact
    ) -> common_pb2.ArtifactSelection:
        proto = common_pb2.ArtifactSelection()
        _fill_selection_proto(self._selection_for_artifact(artifact), proto)
        return proto

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

    def _governance_proto(self) -> plan_pb2.GovernanceContext | None:
        governance = self._ctx.governance
        if governance is None:
            return None
        proto = plan_pb2.GovernanceContext()
        if governance.lane:
            proto.lane = str(governance.lane)
        if governance.policy_version is not None:
            proto.policy_version = int(governance.policy_version)
        if governance.staleness_budget_ms is not None:
            proto.staleness_budget_ms = int(governance.staleness_budget_ms)
        return proto

    def to_spec(self) -> plan_pb2.PlanSpec:
        spec = plan_pb2.PlanSpec(plan_id=self._plan_id)
        spec.context.CopyFrom(self._call_context_proto())
        governance = self._governance_proto()
        if governance is not None:
            spec.governance.CopyFrom(governance)
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
                if step.action.target is not None:
                    if isinstance(step.action.target, ServingBindingTarget):
                        prefetch_action.serving_binding_target.CopyFrom(
                            step.action.target.to_proto()
                        )
                    else:
                        prefetch_action.serving_binding_set_target.CopyFrom(
                            step.action.target.to_proto()
                        )
                    prefetch_action.requested_readiness = _SERVING_READINESS_TO_PROTO[
                        step.action.readiness
                    ]
                    prefetch_action.retention_policy.CopyFrom(
                        (
                            step.action.retention
                            if step.action.retention is not None
                            else PrefetchRetentionPolicy()
                        ).to_proto()
                    )
            elif isinstance(step.action, _PrefetchSetAction):
                prefetch_set_action = step_msg.action.prefetch_set
                prefetch_set_action.artifact_set.CopyFrom(
                    step.action.artifact_set.to_proto()
                )
                prefetch_set_action.device_id = int(step.action.device_id)
                if step.action.manifest_bridge is not None:
                    prefetch_set_action.manifest_bridge.CopyFrom(
                        step.action.manifest_bridge.to_proto()
                    )
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
                        _store_policy_to_plan_proto(step.action.policy)
                    )
            elif isinstance(step.action, _ManifestAction):
                manifest_action = step_msg.action.manifest
                manifest_action.engine_request_id = str(step.action.engine_request_id)
            elif isinstance(step.action, _PublishAction):
                publish_action = step_msg.action.publish
                publish_action.engine_request_id = str(step.action.engine_request_id)
                if step.action.ttl_ms is not None:
                    publish_action.ttl_ms = int(step.action.ttl_ms)
            elif isinstance(step.action, _HydrateAction):
                hydrate_action = step_msg.action.hydrate
                if step.action.engine_request_id is not None:
                    hydrate_action.engine_request_id = str(
                        step.action.engine_request_id
                    )
                elif step.action.publish_manifest is not None:
                    hydrate_action.publish_manifest.CopyFrom(
                        step.action.publish_manifest.to_proto()
                    )
            elif isinstance(step.action, _EvictLocalAction):
                evict_action = step_msg.action.evict_local
                if step.action.engine_request_id:
                    evict_action.engine_request_id = str(step.action.engine_request_id)
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
        extra: str | None = None,
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
            extra=extra,
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
            # Local Plan.run still does not own instance-host routing. Instance
            # execution remains on the Node Agent / Instance Agent boundary.
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
                    action=(
                        "prefetch_serving_binding"
                        if step.action.target is not None
                        else "prefetch"
                    ),
                    target_id=target_id,
                    selection=selection,
                    device_id=step.action.device_id,
                    ttl_ms=None,
                    extra=(
                        hashlib.sha256(
                            step.action.target.to_proto().SerializeToString(
                                deterministic=True
                            )
                        ).hexdigest()
                        if step.action.target is not None
                        else None
                    ),
                )
                bound = _clone_artifact_for_store(step.action.artifact, store)
                prefetch_op = bound.prefetch(
                    device=step.action.device,
                    target=step.action.target,
                    readiness=step.action.readiness,
                    retention=step.action.retention,
                    ctx=ctx,
                )
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
            if isinstance(step.action, _PrefetchSetAction):
                resolved_items = resolve_artifact_set_ref(
                    step.action.artifact_set,
                    manifest_resolver=(
                        step.action.manifest_bridge.resolve_artifact_set
                        if step.action.manifest_bridge is not None
                        else None
                    ),
                )
                outcomes: list[ArtifactSetItemResult] = []
                for index, item in enumerate(resolved_items):
                    selection = _selection_from_proto(item.selection)
                    ctx = self._action_context(
                        action="prefetch_set",
                        target_id=target_id,
                        selection=selection,
                        device_id=step.action.device_id,
                        ttl_ms=None,
                        extra=step.action.artifact_set.set_digest_hex,
                    )
                    bound = _artifact_from_selection_proto(
                        store=store,
                        selection=item.selection,
                    )
                    prefetch_op = bound.prefetch(device=step.action.device, ctx=ctx)
                    with op_lock:
                        op_registry[f"{step.step_id}:{index}"] = prefetch_op
                    try:
                        prefetch_op.wait()
                        item_status = prefetch_op.status()
                    except Exception as exc:  # noqa: BLE001
                        item_status = _status_from_exception(exc)
                    outcomes.append(
                        ArtifactSetItemResult(
                            item_identity=item.item_identity,
                            artifact_id=item.item_identity.artifact_id,
                            status=item_status,
                        )
                    )
                artifact_set_result = ArtifactSetResult(
                    set_digest_hex=step.action.artifact_set.set_digest_hex,
                    outcomes=tuple(outcomes),
                )
                return PlanStepResult(
                    step_id=step.step_id,
                    target_id=target_id,
                    action="prefetch_set",
                    status=summarize_artifact_set_outcomes(
                        action_name="prefetch_set",
                        outcomes=artifact_set_result.outcomes,
                    ),
                    value=artifact_set_result,
                    artifact_set_result=artifact_set_result,
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
        if self._runtime is not None:
            result = self._runtime.execute_plan(
                self.to_spec(),
                execution_class=self._execution_class,
            )
            if raise_on_error and not result.ok:
                raise PlanFailedError(result)
            return result
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


def _store_policy_to_plan_proto(policy: StorePolicy) -> plan_pb2.StorePolicy:
    from tensorcast.api._config import (
        OverflowPolicy,
        PolicyLayout,
        PolicyScope,
        PolicyTier,
        RetentionPolicy,
        StorePolicyProfile,
    )

    profile_map = {
        StorePolicyProfile.CACHE: plan_pb2.POLICY_PROFILE_CACHE,
        StorePolicyProfile.DURABLE: plan_pb2.POLICY_PROFILE_DURABLE,
        StorePolicyProfile.HA: plan_pb2.POLICY_PROFILE_HA,
        StorePolicyProfile.COLD: plan_pb2.POLICY_PROFILE_COLD,
        StorePolicyProfile.PINNED: plan_pb2.POLICY_PROFILE_PINNED,
        StorePolicyProfile.WARM: plan_pb2.POLICY_PROFILE_WARM,
    }
    tier_map = {
        PolicyTier.STABLE_DRAM: plan_pb2.POLICY_TIER_STABLE_DRAM,
        PolicyTier.SHARED_DISK: plan_pb2.POLICY_TIER_SHARED_DISK,
    }
    scope_map = {
        PolicyScope.LOCAL: plan_pb2.POLICY_SCOPE_LOCAL,
        PolicyScope.REMOTE: plan_pb2.POLICY_SCOPE_REMOTE,
        PolicyScope.ANY: plan_pb2.POLICY_SCOPE_ANY,
    }
    retention_map = {
        RetentionPolicy.BEST_EFFORT: plan_pb2.RETENTION_POLICY_BEST_EFFORT,
        RetentionPolicy.TTL: plan_pb2.RETENTION_POLICY_TTL,
        RetentionPolicy.PINNED: plan_pb2.RETENTION_POLICY_PINNED,
    }
    overflow_map = {
        OverflowPolicy.EVICT: plan_pb2.OVERFLOW_POLICY_EVICT,
        OverflowPolicy.SPILL: plan_pb2.OVERFLOW_POLICY_SPILL,
        OverflowPolicy.REJECT: plan_pb2.OVERFLOW_POLICY_REJECT,
    }
    layout_map = {
        PolicyLayout.AUTO: plan_pb2.POLICY_LAYOUT_AUTO,
        PolicyLayout.UNSHARDED: plan_pb2.POLICY_LAYOUT_UNSHARDED,
        PolicyLayout.SHARDED: plan_pb2.POLICY_LAYOUT_SHARDED,
    }

    def _tier_proto(spec) -> plan_pb2.TierSpec:  # noqa: ANN001
        message = plan_pb2.TierSpec(
            tier=tier_map[spec.tier],
            scope=scope_map[spec.scope],
            min_replicas=int(spec.min_replicas),
            retention_policy=retention_map[spec.retention_policy],
        )
        if spec.retention_ttl_ms is not None:
            message.retention_ttl_ms = int(spec.retention_ttl_ms)
        return message

    message = plan_pb2.StorePolicy(
        overflow_policy=overflow_map[policy.overflow_policy],
        layout=layout_map[policy.layout],
    )
    if policy.profile is not None:
        message.profile = profile_map[policy.profile]
    message.must.extend(_tier_proto(spec) for spec in policy.must)
    message.should.extend(_tier_proto(spec) for spec in policy.should)
    message.may.extend(_tier_proto(spec) for spec in policy.may)
    return message


def _action_name(action: _PlanAction) -> str:
    if isinstance(action, _PrefetchAction):
        return "prefetch"
    if isinstance(action, _PrefetchSetAction):
        return "prefetch_set"
    if isinstance(action, _PinAction):
        return "pin_device_residency"
    if isinstance(action, _UnpinAction):
        return "unpin_device_residency"
    if isinstance(action, _TransformIntoAction):
        return "transform_into"
    if isinstance(action, _TransformRegisterAction):
        return "transform_register"
    if isinstance(action, _ManifestAction):
        return "manifest"
    if isinstance(action, _PublishAction):
        return "publish"
    if isinstance(action, _HydrateAction):
        return "hydrate"
    if isinstance(action, _EvictLocalAction):
        return "evict_local"
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
    from tensorcast.api.runtime import get_active_runtime

    active_runtime = get_active_runtime()
    if active_runtime is not None:
        return active_runtime.plan(ctx)
    return Plan(ctx)


__all__ = [
    "ArtifactActionResult",
    "Instance",
    "Plan",
    "PlanFailedError",
    "PlanResult",
    "PlanStepRef",
    "PlanStepResult",
    "PlanExecutionClass",
    "Worker",
    "plan",
]
