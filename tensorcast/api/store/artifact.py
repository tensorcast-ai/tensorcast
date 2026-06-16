#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Sequence,
    TypedDict,
    cast,
)

import torch

from tensorcast.api._device import CPU_DEVICE_ID, device_uuid_for, resolve_device
from tensorcast.api._view_ops import (
    SliceSpec,
    ViewSpecBuildResult,
    _coerce_slice_spec,
    build_view_spec,
)
from tensorcast.api.context import CallContext
from tensorcast.api.operation import (
    DaemonGlobalStoreOperation,
    DaemonReplicaOperation,
    Operation,
    OperationState,
    OperationStatus,
    PollingOperation,
)
from tensorcast.api.store.binding import Binding
from tensorcast.api.store.binding_state import (
    BindingValueMetadata,
    parse_binding_value_or_raise,
)
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
)
from tensorcast.api.store.inplace_slot import InplaceSlot
from tensorcast.api.store.mapped_binding import (
    CopyPlan,
    compute_mapped_view_id,
    compute_mapped_view_id_from_specs,
    infer_mapped_target_entries,
    normalize_copy_plan,
    validate_copy_plan,
    view_narrow_ranges,
)
from tensorcast.api.store.materialization import (
    GetIntoResult,
    MaterializationPipeline,
    _resolve_source_policy_from_options,
)
from tensorcast.api.store.owned_binding_layout import (
    BindingLayout,
    build_binding_layout,
    build_mapped_tensor_spec,
    build_owned_layout,
)
from tensorcast.api.store.owned_binding_slot import (
    OwnedBindingSlot,
    restore_owned_binding_tensors,
)
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationHandle,
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
    RealizationTargetKind,
    RealizationTargetPlan,
    ResolvedArtifactSelection,
    binding_materialization_diagnostics_from_response,
    emit_artifact_realization_profile_event,
    envelope_for_binding,
    envelope_for_caller_tensors,
    envelope_for_mounted_source,
    envelope_for_retained_binding,
    envelope_for_retained_replica,
    envelope_for_target_set,
    envelope_for_tensor_dict,
    lifecycle_plan_for_envelope,
    materialization_source_label,
    mounted_source_target_digest,
    publishability_report_for,
    report_for_binding_realization,
    report_for_mounted_source,
    report_for_target_set,
    representation_admission_for_target,
    resolve_artifact_selection,
    retained_binding_lifecycle_plan_for,
    retained_binding_reports_for,
    risk_labels_for_target,
    selection_report_fields,
    strategy_plan_for_execution,
)
from tensorcast.api.store.retry import (
    map_materialization_error,
    raise_mapped_materialization_error,
)
from tensorcast.api.store.target_region_lifecycle import (
    register_store_target_regions_for_realization as _register_target_regions_for_realization,
)
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
)
from tensorcast.api.store.view_composer import (
    ViewBuilder,
    ViewMetadataCache,
    ViewSpecComposer,
    compute_view_id,
)
from tensorcast.common.identity import is_byte_artifact_id, validate_byte_artifact_cgid
from tensorcast.common.selection_contract import compute_selected_index_bytes
from tensorcast.common.selection_identity import (
    compute_view_subset_hash,
)
from tensorcast.profile_utils import tensorcast_profile_stage
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    GroupRealizationAcquireRef,
    PrefetchHandoff,
    PrefetchHandoffSet,
    PrefetchRetentionPolicy,
    RealizationTarget,
    RealizationTargetSet,
    RuntimeArtifactPolicy,
    RuntimeArtifactPolicyInput,
    RuntimeBindingReadiness,
    RuntimeBindingSourceReuseDecision,
    coerce_runtime_artifact_policy,
)

logger = logging.getLogger(__name__)


def _resolve_runtime_artifact_policy(
    runtime_artifact_policy: RuntimeArtifactPolicyInput | None,
) -> RuntimeArtifactPolicy | None:
    return coerce_runtime_artifact_policy(runtime_artifact_policy)


def _has_validated_byte_artifact_profile(artifact_id: str) -> bool:
    if not is_byte_artifact_id(artifact_id):
        return False
    try:
        validate_byte_artifact_cgid(artifact_id)
    except ValueError as exc:
        raise ArtifactError(
            f"artifact_id must be a valid byte artifact cgid: {exc}",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        ) from exc
    return True


if TYPE_CHECKING:
    from tensorcast.api._config import GetArtifactOptions
    from tensorcast.api._materialize import MaterializationPayload
    from tensorcast.api.store import Store
    from tensorcast.api.store.batch_context import BatchContext
    from tensorcast.api.store.runtime import StoreRuntimeContext


@dataclass(frozen=True, slots=True)
class TensorMeta:
    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    artifact_id: str
    tensor_names: tuple[str, ...]
    tensor_metas: Mapping[str, TensorMeta]
    total_bytes: int
    generation: int | None


def _ordered_subset_entry_names(
    canonical_index: CanonicalIndex,
    tensor_names: Sequence[str],
) -> tuple[str, ...]:
    requested = tuple(str(name) for name in tensor_names)
    if not requested:
        return ()
    requested_set = set(requested)
    ordered = tuple(
        str(entry.name)
        for entry in canonical_index.entries
        if str(entry.name) in requested_set
    )
    if len(ordered) != len(requested_set):
        known = {str(entry.name) for entry in canonical_index.entries}
        unknown = sorted(requested_set - known)
        raise ArtifactError(
            f"View references unknown tensor(s): {', '.join(unknown)}",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return ordered


def _build_subset_view_spec_proto(
    *,
    canonical_index: CanonicalIndex,
    tensor_names: Sequence[str],
) -> common_pb2.ViewSpec | None:
    ordered_names = _ordered_subset_entry_names(canonical_index, tensor_names)
    if not ordered_names:
        return None
    entry_by_name = {str(entry.name): entry for entry in canonical_index.entries}
    proto = common_pb2.ViewSpec()
    for name in ordered_names:
        entry = entry_by_name[name]
        if not entry.shape:
            return None
        dim0 = int(entry.shape[0])
        if dim0 <= 0:
            return None
        op = proto.tensors[name].ops.add()
        op.narrow.dim = 0
        op.narrow.start = 0
        op.narrow.length = dim0
    return proto


@dataclass(frozen=True, slots=True)
class PrefetchedReplica:
    artifact_id: str
    view_id: str
    operation_id: str
    device_id: int
    daemon_id: str
    source: str | None
    report: ArtifactRealizationReport | None = None


RuntimePrefetchResult = PrefetchHandoff | PrefetchHandoffSet


def _parse_serving_prefetch_result_any(
    result: Any,
) -> RuntimePrefetchResult:
    binding_result = operation_pb2.PrefetchServingBindingResult()
    if result.Is(binding_result.DESCRIPTOR):
        result.Unpack(binding_result)
        return PrefetchHandoff.from_proto(binding_result)
    set_result = operation_pb2.PrefetchServingBindingSetResult()
    if result.Is(set_result.DESCRIPTOR):
        result.Unpack(set_result)
        return PrefetchHandoffSet.from_proto(set_result)
    raise ArtifactError(
        "Runtime prefetch operation did not return a typed prefetch handoff result",
        status_code="DATA_LOSS",
        retryable=False,
    )


def _serving_prefetch_result_from_operation_response(
    response: operation_pb2.GetOperationResponse,
) -> RuntimePrefetchResult:
    if response.status.HasField("result"):
        return _parse_serving_prefetch_result_any(response.status.result)
    if response.HasField("snapshot"):
        return _parse_serving_prefetch_result_any(response.snapshot)
    raise ArtifactError(
        "Runtime prefetch operation completed without result metadata",
        status_code="DATA_LOSS",
        retryable=False,
    )


def _digest_hex(label: str, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _serving_target_layout_digest(
    target: RealizationTarget | RealizationTargetSet,
    *,
    target_bytes: bytes,
) -> str:
    if isinstance(target, RealizationTarget):
        return str(target.resolved_layout.target_layout_hash or "") or _digest_hex(
            "serving-target-layout",
            target_bytes,
        )
    return _digest_hex("serving-target-set-layout", target_bytes)


def _serving_target_copy_plan_digest(
    target: RealizationTarget | RealizationTargetSet,
    *,
    target_bytes: bytes,
) -> str:
    if isinstance(target, RealizationTarget):
        digest = str(target.resolved_layout.spec_digest or "")
        if digest:
            return digest
        copy_plan_bytes = bytes(target.resolved_layout.copy_plan_bytes or b"")
        if copy_plan_bytes:
            return _digest_hex("serving-target-copy-plan", copy_plan_bytes)
    return _digest_hex("serving-target-copy-plan", target_bytes)


def _binding_layout_target_digest(binding_layout_id: str) -> str | None:
    normalized = str(binding_layout_id or "").strip()
    if not normalized:
        return None
    return f"binding-layout:{normalized}"


def _mapped_target_specs_from_layout(
    layout: object | None,
) -> tuple[dict[str, object], ...]:
    if layout is None:
        return ()
    specs = getattr(layout, "dst_specs", None)
    if specs is None:
        return ()
    return tuple(
        {
            "name": str(getattr(spec, "name", "")),
            "dtype": str(getattr(spec, "dtype", "")),
            "shape": tuple(int(v) for v in getattr(spec, "shape", ())),
            "stride": tuple(int(v) for v in getattr(spec, "stride", ())),
            "logical_length": int(getattr(spec, "logical_length", 0) or 0),
        }
        for spec in specs
    )


def _with_retained_binding_report(
    result: RuntimePrefetchResult,
    *,
    selection: ResolvedArtifactSelection,
    target: RealizationTarget | RealizationTargetSet,
    target_bytes: bytes,
    operation_id: str,
) -> RuntimePrefetchResult:
    retained_bindings = retained_binding_reports_for(result)
    is_target_set = isinstance(result, PrefetchHandoffSet)
    target_plan = RealizationTargetPlan(
        kind="target_set" if is_target_set else "retained_binding",
        target_layout_digest=_serving_target_layout_digest(
            target,
            target_bytes=target_bytes,
        ),
        copy_plan_digest=_serving_target_copy_plan_digest(
            target,
            target_bytes=target_bytes,
        ),
        member_count=len(retained_bindings),
    )
    envelope = (
        envelope_for_target_set(retained_bindings)
        if is_target_set
        else envelope_for_retained_binding(retained_bindings)
    )
    envelope.validate_for_target(target_plan)
    if is_target_set:
        report = report_for_target_set(
            selection=selection,
            target_plan=target_plan,
            target=target,
            result=result,
            envelope=envelope,
            operation_id=operation_id,
        )
        emit_artifact_realization_profile_event(report)
        members = tuple(
            member.model_copy(update={"report": report}) for member in result.members
        )
        return result.model_copy(update={"report": report, "members": members})

    report = ArtifactRealizationReport(
        target_kind="retained_binding",
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=target_plan.target_layout_digest,
        copy_plan_digest=target_plan.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        **selection_report_fields(selection),
        envelope=envelope,
        target_plan=target_plan,
        strategy_plan=strategy_plan_for_execution(envelope=envelope),
        representation_admission=representation_admission_for_target(target_plan),
        lifecycle_plan=retained_binding_lifecycle_plan_for(
            retained_bindings,
            envelope=envelope,
        ),
        operation_id=operation_id,
        operation_backend="daemon_prefetch_serving_binding",
        risk_labels=risk_labels_for_target(
            target_plan,
            envelope,
            source_selection_digest=selection.source_selection_digest,
        ),
        retained_bindings=retained_bindings,
        publishability=publishability_report_for(),
    )
    emit_artifact_realization_profile_event(report)
    return result.model_copy(update={"report": report})


def _operation_status_from_proto(
    status: operation_pb2.OperationStatus,
) -> OperationStatus:
    state_by_proto: dict[int, OperationState] = {
        int(operation_pb2.OPERATION_STATE_PENDING): "pending",
        int(operation_pb2.OPERATION_STATE_RUNNING): "running",
        int(operation_pb2.OPERATION_STATE_SUCCESS): "success",
        int(operation_pb2.OPERATION_STATE_FAILED): "failed",
        int(operation_pb2.OPERATION_STATE_CANCELLED): "cancelled",
        int(operation_pb2.OPERATION_STATE_DEGRADED): "degraded",
    }
    state = state_by_proto.get(int(status.state), "running")
    error = None
    if status.HasField("error"):
        from tensorcast.api.operation import OperationError

        error = OperationError(
            status_code=str(status.error.status_code or "UNKNOWN"),
            message=str(status.error.message or ""),
            retryable=bool(status.error.retryable),
        )
    return OperationStatus(
        state=state,
        message=str(status.message) if status.message else None,
        progress=float(status.progress) if status.progress else None,
        error=error,
    )


def _serving_target_source_reuse(
    target: RealizationTarget | RealizationTargetSet,
) -> RuntimeBindingSourceReuseDecision:
    if isinstance(target, RealizationTarget):
        return target.resolved_layout.source_reuse
    if not target.members:
        raise ArtifactError(
            "Realization target set members must use one source reuse decision",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    reuse_decision = target.members[0].resolved_layout.source_reuse
    if any(
        member.resolved_layout.source_reuse != reuse_decision
        for member in target.members[1:]
    ):
        raise ArtifactError(
            "Realization target set members must use one source reuse decision",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return reuse_decision


@dataclass(frozen=True, slots=True)
class MaterializationDiagnostics:
    source: str | None
    source_code: int | None
    replica_uuid: str
    ticket_replica_uuid: str | None
    ticket_status: str | None
    disk_path: str | None
    tensor_count: int
    total_bytes: int
    generation: int | None
    view_data_hash: str | None
    view_index_bytes_len: int
    materialize_sec: float
    tensor_bind_sec: float
    total_sec: float
    retry_attempts: int
    retry_reason_buckets: Mapping[str, int]
    budget_deadline_sec: float | None
    budget_elapsed_sec: float | None
    budget_remaining_sec: float | None
    budget_exit_reason: str | None
    breakdown: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class TensorDictMaterializationResult:
    tensors: dict[str, torch.Tensor]
    diagnostics: MaterializationDiagnostics
    release_fn: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def release(self) -> None:
        if self.release_fn is not None:
            self.release_fn()


def _release_materialized_once(
    *,
    pipeline: object,
    runtime: object,
    payload: MaterializationPayload,
) -> Callable[[], None]:
    released = False
    lock = threading.Lock()

    def release_materialized() -> None:
        nonlocal released
        with lock:
            if released:
                return
            release = getattr(pipeline, "_release_materialized", None)
            ensure_client = getattr(runtime, "ensure_client", None)
            if callable(release) and callable(ensure_client):
                release(payload, ensure_client())
            released = True

    return release_materialized


@dataclass(frozen=True, slots=True)
class _SelectionMaterializationInputs:
    requested_names: tuple[str, ...] | None
    view_spec_proto: common_pb2.ViewSpec | None
    view_data_hash: str | None
    view_index_hint: bytes | None
    view_id_hint: str | None
    view_subset_hash: bytes | None


def _materialization_ticket_status_label(
    status: store_daemon_pb2.MaterializeReplicaStatus | None,
) -> str | None:
    if status is None:
        return None
    try:
        raw = store_daemon_pb2.MaterializeReplicaStatus.Name(int(status))
    except Exception:  # noqa: BLE001
        return str(int(status))
    prefix = "MATERIALIZE_REPLICA_STATUS_"
    if raw.startswith(prefix):
        return raw[len(prefix) :].lower()
    return raw.lower()


def _payload_total_bytes(payload: "MaterializationPayload") -> int:
    total = 0
    for desc in payload.descriptors:
        total += int(desc.byte_length)
    return total


def _encode_capability_token(token: bytes) -> str:
    raw = base64.urlsafe_b64encode(token).decode("ascii")
    return raw.rstrip("=")


def _decode_capability_token(token: str) -> bytes:
    raw = token.strip()
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _ctx_timeout_s(ctx: CallContext | None) -> float | None:
    if ctx is None or ctx.deadline_ms is None:
        return None
    timeout_s = float(ctx.deadline_ms) / 1000.0
    if timeout_s <= 0:
        raise ArtifactError(
            "CallContext deadline exceeded",
            status_code="DEADLINE_EXCEEDED",
            retryable=False,
        )
    return max(0.001, timeout_s)


def _build_transport_operation_id(
    *,
    base_operation_id: str,
    ctx: CallContext | None,
) -> str:
    _ = ctx
    return base_operation_id


def _close_client_binding_best_effort(
    runtime: "StoreRuntimeContext",
    binding_id: str,
    *,
    context: str,
) -> None:
    try:
        runtime.ensure_client().close_owned_binding(binding_id=binding_id)
    except Exception:
        logger.exception(
            "%s failed to close client binding during rollback: binding_id=%s",
            context,
            binding_id,
        )


def _register_client_binding(
    *,
    runtime: "StoreRuntimeContext",
    device_id: int,
    region_layout,
    canonical_index_bytes: bytes,
    selection: common_pb2.ArtifactSelection | None,
    source_artifact_id: str | None,
    ctx: CallContext | None,
) -> tuple[str, BindingLayout, BindingValueMetadata | None, bytes | None]:
    binding_layout = build_binding_layout(
        target_layout=region_layout.layout,
        target_index_bytes=bytes(
            region_layout.view_index_bytes or canonical_index_bytes
        ),
    )
    timeout_s = _ctx_timeout_s(ctx)
    response = runtime.ensure_client().create_binding(
        ownership=store_daemon_pb2.BindingOwnership.BINDING_OWNERSHIP_CLIENT,
        target_layout=binding_layout.target_layout,
        target_index_bytes=binding_layout.target_index_bytes,
        device_uuid=device_uuid_for(device_id),
        binding_layout_id=binding_layout.binding_layout_id,
        initial_selection=selection,
        source_artifact_id=source_artifact_id,
        timeout_s=timeout_s if timeout_s is not None else 60.0,
    )
    binding_id = str(response.binding_id)
    try:
        metadata = parse_binding_value_or_raise(
            response.current_value,
            rpc_name="CreateBinding",
            expected_binding_id=binding_id,
            expected_binding_layout_id=binding_layout.binding_layout_id,
        )
    except ArtifactError:
        _close_client_binding_best_effort(
            runtime,
            binding_id,
            context="CreateBinding current_value parse",
        )
        raise
    if metadata is None:
        _close_client_binding_best_effort(
            runtime,
            binding_id,
            context="CreateBinding missing current_value",
        )
        raise ArtifactError(
            "CreateBinding returned empty current_value for artifact-backed binding",
            status_code="DATA_LOSS",
            retryable=False,
        )
    response_publication_token = bytes(response.binding_current_value_publication_token)
    return (
        binding_id,
        binding_layout,
        metadata,
        response_publication_token or None,
    )


@dataclass(frozen=True, slots=True)
class PlacementPin:
    pin_id: int
    capability_token: str
    daemon_id: str
    artifact_id: str
    view_id: str
    device_id: int
    expires_at_ms: int | None
    runtime_ref: "weakref.ReferenceType[StoreRuntimeContext]" = field(
        repr=False, compare=False
    )

    def _runtime(self) -> "StoreRuntimeContext":
        runtime = self.runtime_ref()
        if runtime is None or runtime.closed:
            raise ArtifactError(
                "Store runtime is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return runtime

    def renew(self, *, ttl_ms: int, ctx: CallContext | None = None) -> "PlacementPin":
        timeout_s = _ctx_timeout_s(ctx)
        resp = (
            self._runtime()
            .ensure_client()
            .renew_placement_lease(
                lease_token=_decode_capability_token(self.capability_token),
                ttl_ms=int(ttl_ms),
                timeout_s=timeout_s,
            )
        )
        new_token = self.capability_token
        if resp.lease_token:
            new_token = _encode_capability_token(bytes(resp.lease_token))
        expires_at_ms: int | None = None
        if resp.HasField("expires_at"):
            try:
                expires_at_ms = int(
                    resp.expires_at.ToDatetime(tzinfo=timezone.utc).timestamp() * 1000
                )
            except Exception:  # noqa: BLE001
                expires_at_ms = None
        return PlacementPin(
            pin_id=int(resp.lease_id),
            capability_token=new_token,
            daemon_id=self.daemon_id,
            artifact_id=self.artifact_id,
            view_id=self.view_id,
            device_id=self.device_id,
            expires_at_ms=expires_at_ms,
            runtime_ref=self.runtime_ref,
        )

    def release(self, *, ctx: CallContext | None = None) -> None:
        timeout_s = _ctx_timeout_s(ctx)
        _ = (
            self._runtime()
            .ensure_client()
            .release_placement_lease(
                lease_token=_decode_capability_token(self.capability_token),
                timeout_s=timeout_s,
            )
        )


class ArtifactSerialized(TypedDict):
    artifact_id: str | None
    key: str | None
    canonical_index: str | None
    generation: int | None


def _meta_from_entry(entry: CanonicalIndexEntry) -> TensorMeta:
    return TensorMeta(
        name=entry.name,
        shape=tuple(entry.shape),
        stride=tuple(entry.stride),
        dtype=entry.dtype,
        storage_offset=int(entry.storage_offset),
        size_bytes=int(entry.size_bytes),
    )


class Artifact:
    """Lazy handle to a TensorCast artifact.

    Construction is non-blocking and does not trigger RPCs; identity and
    metadata are resolved lazily on first materialization (`tensor*`,
    `tensor_dict*`, or `exists()`), and view composition is purely local until
    then.
    """

    def __init__(
        self,
        *,
        store_ref: weakref.ReferenceType["Store"],
        artifact_id: str | None = None,
        key: str | None = None,
        canonical_index_bytes: bytes | None = None,
        canonical_index: CanonicalIndex | None = None,
        generation: int | None = None,
        key_generation: int | None = None,
        view_spec: ViewSpecBuildResult | None = None,
        view_metadata: ViewMetadataCache | None = None,
        view_depth: int = 0,
        source_subject: Any | None = None,
    ) -> None:
        identifiers = [bool(artifact_id), bool(key)]
        if sum(identifiers) == 0:
            raise ArtifactError(
                "At least one of artifact_id or key is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._artifact_id = artifact_id
        self._key_hint = key
        self._canonical_index_bytes = canonical_index_bytes
        self._canonical_index = canonical_index
        self._tensor_metas: dict[str, TensorMeta] | None = None
        self._view_spec = view_spec
        self._view_metadata = view_metadata
        self._view_depth = max(0, int(view_depth))
        self._source_subject = source_subject
        effective_index = (
            view_metadata.selected_index
            if view_metadata is not None
            else canonical_index
        )
        if effective_index is not None:
            self._tensor_metas = {
                entry.name: _meta_from_entry(entry) for entry in effective_index.entries
            }
        self._generation = generation
        self._key_generation = key_generation
        self._store_ref = store_ref
        self._lock = threading.RLock()
        self._released = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def artifact_id(self) -> str:
        return self._ensure_identified()

    @property
    def key(self) -> str | None:
        return self._key_hint

    @property
    def tensor_names(self) -> tuple[str, ...]:
        canonical_index = self._effective_index()
        return tuple(entry.name for entry in canonical_index.entries)

    def tensor_meta(self, name: str) -> TensorMeta:
        canonical_index = self._effective_index()
        metas = self._tensor_metas or {}
        if name in metas:
            return metas[name]
        for entry in canonical_index.entries:
            if entry.name == name:
                meta = _meta_from_entry(entry)
                metas[name] = meta
                self._tensor_metas = metas
                return meta
        raise ArtifactError(
            f"Tensor '{name}' not found in artifact",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    def describe(self) -> ArtifactDescriptor:
        canonical_index = self._effective_index()
        metas = self._tensor_metas or {
            entry.name: _meta_from_entry(entry) for entry in canonical_index.entries
        }
        self._tensor_metas = metas
        return ArtifactDescriptor(
            artifact_id=self._ensure_identified(),
            tensor_names=tuple(entry.name for entry in canonical_index.entries),
            tensor_metas=dict(metas),
            total_bytes=int(canonical_index.total_size_bytes),
            generation=self._generation,
        )

    def _realization_handle(
        self,
        *,
        target_kind: RealizationTargetKind,
        report: ArtifactRealizationReport,
        tensor_dict_value: Mapping[str, Any] | None = None,
        binding_value: Any | None = None,
        prefetch_value: Any | None = None,
        promote_fn: Callable[..., Any] | None = None,
        attach_fn: Callable[..., Any] | None = None,
        close_fn: Callable[[], None] | None = None,
    ) -> ArtifactRealizationHandle:
        handle = ArtifactRealizationHandle(
            target_kind=target_kind,
            report=report,
            tensor_dict_value=tensor_dict_value,
            binding_value=binding_value,
            prefetch_value=prefetch_value,
            promote_fn=promote_fn,
            attach_fn=attach_fn,
            close_fn=close_fn,
        )
        emit_artifact_realization_profile_event(handle.report)
        return handle

    def _model_runtime_request_facts(
        self,
        spec: ArtifactRealizationSpec,
        runtime_context: Any | None,
    ) -> tuple[ArtifactRealizationSpec, Any]:
        from tensorcast.artifact_runtime.request_facts import (
            ModelRuntimeRequestFactsError,
            resolve_model_runtime_request_facts,
        )

        try:
            facts = resolve_model_runtime_request_facts(
                spec=spec,
                runtime_context=runtime_context,
            )
        except ModelRuntimeRequestFactsError as exc:
            raise ArtifactError(
                str(exc),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ) from exc
        return cast(ArtifactRealizationSpec, facts.spec), facts.context

    def _store_bound_runtime_artifact_resolver(self) -> Any:
        from tensorcast.artifact_runtime.artifact.resolver import (
            RuntimeArtifactResolver,
        )
        from tensorcast.types import (
            SERVING_MANIFEST_TENSOR_NAME,
            RuntimeArtifactManifest,
        )

        store, _runtime, _pipeline = self._require_components()
        return RuntimeArtifactResolver(
            manifest_tensor_name=SERVING_MANIFEST_TENSOR_NAME,
            schema_version=int(
                RuntimeArtifactManifest.model_fields["schema_version"].default
            ),
            open_artifact_fn=lambda ref: store.artifact(ref=ref),
        )

    def _execute_model_runtime_realization(
        self,
        spec: ArtifactRealizationSpec,
        *,
        runtime_host: Any | None,
        runtime_context: Any | None,
        runtime_resolver: Any | None,
        profile_sink: Any | None,
        runtime_prepared_local_ready: Any | None,
    ) -> ArtifactRealizationHandle:
        if runtime_host is None:
            raise ArtifactError(
                "model_runtime realization requires runtime_host",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        from tensorcast.artifact_runtime.lifecycle import ArtifactRuntimeIntegration

        artifact_id = self._ensure_identified()
        resolved_spec, context = self._model_runtime_request_facts(
            spec,
            runtime_context,
        )
        resolved_runtime_resolver = runtime_resolver
        if resolved_runtime_resolver is None and not artifact_id.startswith("msa1:"):
            resolved_runtime_resolver = self._store_bound_runtime_artifact_resolver()
        integration = ArtifactRuntimeIntegration(
            resolver=resolved_runtime_resolver,
            profile_sink=profile_sink,
            host=runtime_host,
        )
        from tensorcast.artifact_runtime.request_facts import (
            ModelRuntimeRequestFactsError,
        )

        try:
            if artifact_id.startswith("msa1:"):
                if self._source_subject is None:
                    raise ArtifactError(
                        "mounted-source model_runtime realization requires a "
                        "daemon-attested source handle; create the artifact with "
                        "tensorcast.from_disk(...)",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
                source_selection = self._resolve_model_runtime_source_selection(
                    artifact_id
                )
                attachment = integration.realize_mounted_source_model_runtime(
                    artifact_ref=artifact_id,
                    source_subject=self._source_subject,
                    spec=resolved_spec,
                    context=context,
                    source_selection=source_selection,
                    materialization=resolved_spec.options,
                    prepared_local_ready=runtime_prepared_local_ready,
                )
            else:
                if runtime_prepared_local_ready is not None:
                    raise ArtifactError(
                        "runtime_prepared_local_ready is only valid for "
                        "mounted-source model_runtime realization",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                source_selection = self._resolve_model_runtime_source_selection(
                    artifact_id
                )
                attachment = integration.realize_model_runtime(
                    artifact_ref=artifact_id,
                    spec=resolved_spec,
                    context=context,
                    source_selection=source_selection,
                    runtime_artifact_policy=resolved_spec.runtime_artifact_policy,
                    materialization=resolved_spec.options,
                )
        except ModelRuntimeRequestFactsError as exc:
            raise ArtifactError(
                str(exc),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ) from exc
        handle = getattr(attachment.state, "model_runtime_handle", None)
        if not isinstance(handle, ArtifactRealizationHandle):
            raise ArtifactError(
                "model_runtime realization completed without a realization handle",
                status_code="INTERNAL",
                retryable=False,
            )
        return handle

    def _resolve_model_runtime_source_selection(
        self,
        artifact_id: str,
    ) -> ResolvedArtifactSelection | None:
        if not artifact_id.startswith("msa1:"):
            if (
                self._canonical_index_bytes is None
                and not self._model_runtime_can_resolve_artifact_index()
                and self._view_spec is None
                and self._view_metadata is None
            ):
                return None
            return self._resolve_realization_selection()
        canonical_index_bytes = self._canonical_index_bytes
        if canonical_index_bytes is None and self._source_subject is not None:
            canonical_index_bytes = bytes(
                getattr(self._source_subject, "canonical_index_bytes", None) or b""
            )
        return resolve_artifact_selection(
            artifact_id=artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            generation_hint=(
                self._key_generation
                if self._key_generation is not None
                else self._generation
            ),
        )

    def _model_runtime_can_resolve_artifact_index(self) -> bool:
        store = self._store_ref() if self._store_ref is not None else None
        if store is None or bool(getattr(store, "closed", False)):
            return False
        runtime = getattr(store, "_runtime", None)
        return callable(getattr(runtime, "ensure_client", None))

    def realize(
        self,
        spec: ArtifactRealizationSpec,
        *,
        ctx: CallContext | None = None,
        runtime_host: Any | None = None,
        runtime_context: Any | None = None,
        runtime_resolver: Any | None = None,
        profile_sink: Any | None = None,
        runtime_prepared_local_ready: Any | None = None,
    ) -> ArtifactRealizationHandle:
        if spec.target_kind == "tensor_dict":
            if spec.device is None:
                raise ArtifactError(
                    "tensor_dict realization requires device",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            selection = self._resolve_realization_selection()
            result = self._execute_tensor_dict_with_diagnostics(
                device=cast(torch.device | str, spec.device),
                options=cast("GetArtifactOptions | None", spec.options),
                ctx=ctx,
            )
            envelope = envelope_for_tensor_dict(
                result.tensors,
                source=result.diagnostics.source,
                retry_reason_buckets=result.diagnostics.retry_reason_buckets,
            )
            target_plan = RealizationTargetPlan(
                kind="tensor_dict",
                device=cast(torch.device | str, spec.device),
            )
            envelope.validate_for_target(target_plan)
            report = ArtifactRealizationReport(
                target_kind="tensor_dict",
                source_selection_digest=selection.source_selection_digest,
                target_layout_digest=target_plan.target_layout_digest,
                copy_plan_digest=target_plan.copy_plan_digest,
                artifact_id=selection.artifact_id,
                view_id=selection.view_id,
                artifact_profile=selection.artifact_profile,
                authority_scope=selection.authority_scope,
                generation_hint=selection.generation_hint,
                **selection_report_fields(selection),
                envelope=envelope,
                target_plan=target_plan,
                strategy_plan=strategy_plan_for_execution(
                    envelope=envelope,
                    options=spec.options,
                    ctx=ctx,
                ),
                representation_admission=representation_admission_for_target(
                    target_plan
                ),
                lifecycle_plan=lifecycle_plan_for_envelope(target_plan, envelope),
                materialize_sec=result.diagnostics.materialize_sec,
                tensor_bind_sec=result.diagnostics.tensor_bind_sec,
                total_sec=result.diagnostics.total_sec,
                source=result.diagnostics.source,
                operation_backend="daemon_materialization",
                risk_labels=risk_labels_for_target(
                    target_plan,
                    envelope,
                    source_selection_digest=selection.source_selection_digest,
                ),
                publishability=publishability_report_for(),
                materialization_diagnostics=result.diagnostics,
            )
            return self._realization_handle(
                target_kind="tensor_dict",
                report=report,
                tensor_dict_value=result.tensors,
                close_fn=result.release,
            )
        if spec.target_kind == "caller_tensors":
            if spec.target is None:
                raise ArtifactError(
                    "caller_tensors realization requires target tensors",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            selection = self._resolve_realization_selection()
            target_digest = self._target_tensors_digest(spec.target)
            target_plan = RealizationTargetPlan(
                kind="caller_tensors",
                device=spec.device,
                target_layout_digest=target_digest,
            )
            into_result = self._execute_tensor_dict_into(
                cast(dict[str, torch.Tensor], spec.target),
                device=cast(torch.device | str | None, spec.device),
                options=cast("GetArtifactOptions | None", spec.options),
                ctx=ctx,
            )
            envelope = envelope_for_caller_tensors(
                cast(Mapping[str, object], spec.target),
                used_region_backed=(
                    into_result.used_region_backed if into_result is not None else None
                ),
                actual_total_bytes=(
                    into_result.total_bytes if into_result is not None else None
                ),
                fallback_reason_buckets=(
                    into_result.fallback_reason_buckets
                    if into_result is not None
                    else None
                ),
            )
            envelope.validate_for_target(target_plan)
            report = ArtifactRealizationReport(
                target_kind="caller_tensors",
                source_selection_digest=selection.source_selection_digest,
                target_layout_digest=target_digest,
                copy_plan_digest=None,
                artifact_id=selection.artifact_id,
                view_id=selection.view_id,
                artifact_profile=selection.artifact_profile,
                authority_scope=selection.authority_scope,
                generation_hint=selection.generation_hint,
                **selection_report_fields(selection),
                envelope=envelope,
                target_plan=target_plan,
                strategy_plan=strategy_plan_for_execution(
                    envelope=envelope,
                    options=spec.options,
                    ctx=ctx,
                ),
                representation_admission=representation_admission_for_target(
                    target_plan
                ),
                lifecycle_plan=lifecycle_plan_for_envelope(target_plan, envelope),
                operation_backend="daemon_materialize_into_target",
                source=into_result.source if into_result is not None else None,
                risk_labels=risk_labels_for_target(
                    target_plan,
                    envelope,
                    source_selection_digest=selection.source_selection_digest,
                ),
                publishability=publishability_report_for(),
            )
            return self._realization_handle(
                target_kind="caller_tensors",
                report=report,
            )
        if spec.target_kind == "binding_owned":
            if spec.device is None:
                raise ArtifactError(
                    "binding realization requires device",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            selection = self._resolve_realization_selection()
            binding = self._execute_bind_owned(
                cast(torch.device | str, spec.device),
                mapping=cast(CopyPlan | None, spec.mapping),
                packing=spec.packing,
                options=cast("GetArtifactOptions | None", spec.options),
                capacity_bytes=spec.capacity_bytes,
                publish=spec.publish,
                runtime_artifact_policy=cast(
                    RuntimeArtifactPolicyInput | None,
                    spec.runtime_artifact_policy,
                ),
                ctx=ctx,
            )
            binding_layout = getattr(binding, "layout", None)
            layout_binding_id = str(
                getattr(binding_layout, "binding_layout_id", "") or ""
            )
            binding_layout_id = str(
                getattr(binding, "binding_layout_id", "") or layout_binding_id
            )
            mapped_view_id = None
            copy_plan_digest = None
            if spec.mapping is not None:
                target_layout = getattr(binding_layout, "target_layout", None)
                mapped_view_id = str(getattr(target_layout, "view_id", "") or "")
                if not mapped_view_id:
                    target_specs = _mapped_target_specs_from_layout(binding_layout)
                    if target_specs:
                        mapped_view_id = compute_mapped_view_id_from_specs(
                            canonical_index_bytes=selection.canonical_index_bytes,
                            source_view_id=selection.view_id,
                            plan=normalize_copy_plan(cast(CopyPlan, spec.mapping)),
                            target_specs=target_specs,
                        )
                if not mapped_view_id:
                    raise ArtifactError(
                        "mapped owned binding realization requires mapped target "
                        "layout identity",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
                copy_plan_digest = mapped_view_id
            target_plan = RealizationTargetPlan(
                kind="binding_owned",
                device=spec.device,
                target_layout_digest=_binding_layout_target_digest(binding_layout_id),
                binding_layout_id=binding_layout_id,
                mapped_view_id=mapped_view_id,
                copy_plan_digest=copy_plan_digest,
            )
            envelope = envelope_for_binding(
                binding,
                target_kind="binding_owned",
                publish_requested=spec.publish,
            )
            envelope.validate_for_target(target_plan)
            report = report_for_binding_realization(
                target_kind="binding_owned",
                selection=selection,
                target_plan=target_plan,
                binding=binding,
                envelope=envelope,
                publish_requested=spec.publish,
                options=spec.options,
                ctx=ctx,
            )
            return self._realization_handle(
                target_kind="binding_owned",
                report=report,
                binding_value=binding,
                close_fn=(
                    binding.close if callable(getattr(binding, "close", None)) else None
                ),
            )
        if spec.target_kind == "binding_adopted":
            if spec.target is None:
                raise ArtifactError(
                    "adopted binding realization requires target tensors",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            selection = self._resolve_realization_selection()
            target_digest = self._target_tensors_digest(spec.target)
            binding = self._execute_bind_into(
                cast(Mapping[str, torch.Tensor], spec.target),
                mapping=cast(CopyPlan | None, spec.mapping),
                packing=spec.packing,
                options=cast("GetArtifactOptions | None", spec.options),
                publish=spec.publish,
                runtime_artifact_policy=cast(
                    RuntimeArtifactPolicyInput | None,
                    spec.runtime_artifact_policy,
                ),
                ctx=ctx,
            )
            binding_layout_id = str(getattr(binding, "binding_layout_id", "") or "")
            copy_plan_digest = (
                compute_mapped_view_id(
                    canonical_index_bytes=selection.canonical_index_bytes,
                    source_view_id=selection.view_id,
                    plan=normalize_copy_plan(cast(CopyPlan, spec.mapping)),
                    target_tensors=cast(Mapping[str, torch.Tensor], spec.target),
                )
                if spec.mapping is not None
                else None
            )
            target_plan = RealizationTargetPlan(
                kind="binding_adopted",
                target_layout_digest=target_digest,
                copy_plan_digest=copy_plan_digest,
                binding_layout_id=binding_layout_id,
            )
            envelope = envelope_for_binding(
                binding,
                target_kind="binding_adopted",
                target_tensors=cast(Mapping[str, object], spec.target),
                publish_requested=spec.publish,
            )
            envelope.validate_for_target(target_plan)
            report = report_for_binding_realization(
                target_kind="binding_adopted",
                selection=selection,
                target_plan=target_plan,
                binding=binding,
                envelope=envelope,
                publish_requested=spec.publish,
                options=spec.options,
                ctx=ctx,
            )
            return self._realization_handle(
                target_kind="binding_adopted",
                report=report,
                binding_value=binding,
                close_fn=(
                    binding.close if callable(getattr(binding, "close", None)) else None
                ),
            )
        if spec.target_kind == "mounted_source":
            if self._key_hint:
                raise ArtifactError(
                    "mounted_source realization requires an explicit msa1 artifact id, not a key",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            source_artifact_id = self._ensure_identified()
            if not source_artifact_id.startswith("msa1:"):
                raise ArtifactError(
                    "mounted_source realization requires an msa1 mounted-source artifact",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            store, _, _ = self._require_components()
            promote = getattr(store, "_promote_mounted_source_direct", None)
            if not callable(promote):
                raise ArtifactError(
                    "mounted_source realization requires Store promotion support",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            promoted = promote(
                source_artifact_id,
                verify_checksums=spec.verify_checksums,
                timeout_s=spec.timeout_s,
            )
            promoted_artifact_id = str(getattr(promoted, "artifact_id", "") or "")
            canonical_index_bytes = bytes(
                getattr(promoted, "_canonical_index_bytes", None) or b""
            )
            promoted_generation_raw = getattr(promoted, "_generation", None)
            promoted_generation = (
                int(promoted_generation_raw)
                if promoted_generation_raw is not None
                else None
            )
            selection = resolve_artifact_selection(
                artifact_id=source_artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                generation_hint=(
                    self._key_generation
                    if self._key_generation is not None
                    else self._generation
                ),
            )
            target_plan = RealizationTargetPlan(
                kind="mounted_source",
                target_layout_digest=mounted_source_target_digest(
                    source_artifact_id=source_artifact_id,
                    promoted_artifact_id=promoted_artifact_id,
                    canonical_index_bytes=canonical_index_bytes,
                ),
            )
            envelope = envelope_for_mounted_source(
                canonical_index_bytes=canonical_index_bytes,
            )
            envelope.validate_for_target(target_plan)
            report = report_for_mounted_source(
                selection=selection,
                promoted_artifact_id=promoted_artifact_id,
                generation=promoted_generation,
                canonical_index_bytes=canonical_index_bytes,
                verify_checksums=spec.verify_checksums,
                target_plan=target_plan,
                envelope=envelope,
            )
            return self._realization_handle(
                target_kind="mounted_source",
                report=report,
                promote_fn=lambda: promoted,
            )
        if spec.target_kind == "model_runtime":
            return self._execute_model_runtime_realization(
                spec,
                runtime_host=runtime_host,
                runtime_context=runtime_context,
                runtime_resolver=runtime_resolver,
                profile_sink=profile_sink,
                runtime_prepared_local_ready=runtime_prepared_local_ready,
            )
        raise ArtifactError(
            f"Unsupported realization target kind: {spec.target_kind}",
            status_code="UNIMPLEMENTED",
            retryable=False,
        )

    def realize_async(
        self,
        spec: ArtifactRealizationSpec,
        *,
        ctx: CallContext | None = None,
    ) -> Operation[PrefetchedReplica] | Operation[RuntimePrefetchResult]:
        if spec.target_kind == "retained_replica":
            if spec.device is None:
                raise ArtifactError(
                    "retained_replica realization requires device",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return self._execute_prefetch(
                device=cast(torch.device | str | int, spec.device),
                options=cast("GetArtifactOptions | None", spec.options),
                ctx=ctx,
            )
        if spec.target_kind == "retained_binding":
            if spec.target is None:
                raise ArtifactError(
                    "retained_binding realization requires target",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if isinstance(spec.target, RealizationTargetSet):
                raise ArtifactError(
                    "RealizationTargetSet requires target_set realization",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            readiness = (
                cast(RuntimeBindingReadiness, spec.readiness)
                if spec.readiness is not None
                else "runtime_local_ready"
            )
            return self._execute_prefetch(
                target=cast(RealizationTarget | RealizationTargetSet, spec.target),
                readiness=readiness,
                retention=cast(PrefetchRetentionPolicy | None, spec.retention),
                ctx=ctx,
            )
        if spec.target_kind == "target_set":
            if spec.target is None:
                raise ArtifactError(
                    "target_set realization requires target",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not isinstance(spec.target, RealizationTargetSet):
                raise ArtifactError(
                    "target_set realization requires RealizationTargetSet",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            readiness = (
                cast(RuntimeBindingReadiness, spec.readiness)
                if spec.readiness is not None
                else "runtime_local_ready"
            )
            return self._execute_prefetch(
                target=spec.target,
                readiness=readiness,
                retention=cast(PrefetchRetentionPolicy | None, spec.retention),
                ctx=ctx,
            )
        raise ArtifactError(
            f"Unsupported async realization target kind: {spec.target_kind}",
            status_code="UNIMPLEMENTED",
            retryable=False,
        )

    def tensor_dict(
        self,
        *,
        device: torch.device | str,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> dict[str, torch.Tensor]:
        handle = self.realize(
            ArtifactRealizationSpec.tensor_dict(device=device, options=options),
            ctx=ctx,
        )
        return handle.tensor_dict()

    def tensor_dict_with_diagnostics(
        self,
        *,
        device: torch.device | str,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> TensorDictMaterializationResult:
        handle = self.realize(
            ArtifactRealizationSpec.tensor_dict(device=device, options=options),
            ctx=ctx,
        )
        diagnostics = handle.report.materialization_diagnostics
        if not isinstance(diagnostics, MaterializationDiagnostics):
            raise ArtifactError(
                "tensor_dict realization did not produce materialization diagnostics",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return TensorDictMaterializationResult(
            tensors=handle.tensor_dict(),
            diagnostics=diagnostics,
            release_fn=handle.close,
        )

    def _execute_tensor_dict_with_diagnostics(
        self,
        *,
        device: torch.device | str,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> TensorDictMaterializationResult:
        artifact_id = self._ensure_identified()
        selection_breakdown: dict[str, float] = {}
        selection_prepare_start = time.perf_counter()
        self._ensure_view_metadata_cache(
            require_index_bytes=True,
            timing_out=selection_breakdown,
        )
        selection_inputs = self._resolve_selection_materialization_inputs()
        selection_prepare_sec = time.perf_counter() - selection_prepare_start
        requested_names = selection_inputs.requested_names
        _, runtime, pipeline = self._require_components()
        view_spec_proto = selection_inputs.view_spec_proto
        view_data_hash = selection_inputs.view_data_hash
        view_index_hint = selection_inputs.view_index_hint
        view_id_hint = selection_inputs.view_id_hint
        replica_uuid = options.replica_uuid if options is not None else None
        materialize_start = time.perf_counter()
        payload, _ = pipeline.materialize_subset(
            artifact_id=artifact_id,
            key=None,
            device=device,
            tensor_names=requested_names,
            view_id=view_id_hint,
            canonical_index_hint=self._canonical_index_bytes,
            view_spec=view_spec_proto,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
            options=options,
            ctx=ctx,
        )
        materialize_end = time.perf_counter()
        state: dict[str, torch.Tensor] | None = None
        try:
            self._update_metadata_from_payload(payload, runtime)
            state = pipeline._payload_state_dict(payload)
            bind_end = time.perf_counter()
            if requested_names is None:
                output = state
            else:
                output = {name: state[name] for name in requested_names}
            return_end = time.perf_counter()
            breakdown: dict[str, float] = {}
            if selection_prepare_sec > 0:
                breakdown["selection_prepare_sec"] = selection_prepare_sec
            for name, value in selection_breakdown.items():
                breakdown[f"selection_{name}"] = float(value)
            payload_bind_timing = getattr(payload, "bind_timing", None) or {}
            payload_materialize_timing = (
                getattr(payload, "materialize_timing", None) or {}
            )
            for name, value in payload_materialize_timing.items():
                breakdown[f"materialize_{name}"] = float(value)
            for name, value in payload_bind_timing.items():
                breakdown[f"bind_{name}"] = float(value)
            diagnostics = MaterializationDiagnostics(
                source=materialization_source_label(payload.source),
                source_code=(
                    int(payload.source) if payload.source is not None else None
                ),
                replica_uuid=str(payload.replica_uuid),
                ticket_replica_uuid=(
                    str(payload.ticket_replica_uuid)
                    if payload.ticket_replica_uuid
                    else None
                ),
                ticket_status=_materialization_ticket_status_label(
                    payload.ticket_status
                ),
                disk_path=str(payload.disk_path) if payload.disk_path else None,
                tensor_count=int(len(payload.descriptors)),
                total_bytes=int(_payload_total_bytes(payload)),
                generation=(
                    int(payload.generation) if payload.generation is not None else None
                ),
                view_data_hash=(
                    str(payload.view_data_hash) if payload.view_data_hash else None
                ),
                view_index_bytes_len=int(len(payload.view_index_bytes or b"")),
                materialize_sec=(materialize_end - materialize_start),
                tensor_bind_sec=(bind_end - materialize_end),
                total_sec=(return_end - materialize_start),
                retry_attempts=max(1, int(payload.retry_attempts)),
                retry_reason_buckets=dict(payload.retry_reason_buckets or {}),
                budget_deadline_sec=payload.budget_deadline_sec,
                budget_elapsed_sec=payload.budget_elapsed_sec,
                budget_remaining_sec=payload.budget_remaining_sec,
                budget_exit_reason=payload.budget_exit_reason,
                breakdown=breakdown or None,
            )
            logger.debug(
                "store.tensor_dict.materialized",
                extra={
                    "tc.artifact.id": artifact_id,
                    "tc.store.source": diagnostics.source or "",
                    "tc.tensor.count": diagnostics.tensor_count,
                    "tc.tensor.bytes": diagnostics.total_bytes,
                    "tc.store.replica_uuid": diagnostics.replica_uuid,
                    "tc.store.ticket_status": diagnostics.ticket_status or "",
                    "tc.store.materialize_sec": diagnostics.materialize_sec,
                    "tc.store.total_sec": diagnostics.total_sec,
                },
            )
            release_materialized = _release_materialized_once(
                pipeline=pipeline,
                runtime=runtime,
                payload=payload,
            )
            return TensorDictMaterializationResult(
                tensors=output,
                diagnostics=diagnostics,
                release_fn=release_materialized,
            )
        finally:
            if state is None:
                pipeline._release_materialized(payload, runtime.ensure_client())

    def tensor(
        self,
        name: str,
        *,
        device: torch.device | str,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> torch.Tensor:
        result = self.subset([name]).tensor_dict(
            device=device, options=options, ctx=ctx
        )
        if name not in result:
            raise ArtifactError(
                f"Tensor '{name}' missing from materialized payload",
                status_code="NOT_FOUND",
                retryable=False,
            )
        return result[name]

    def tensor_dict_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self.realize(
            ArtifactRealizationSpec.caller_tensors(
                target=target,
                device=device,
                options=options,
            ),
            ctx=ctx,
        ).complete()

    def _execute_tensor_dict_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> GetIntoResult | None:
        artifact_id = self._ensure_identified()
        effective_index = self._effective_index()
        requested_names = tuple(str(name) for name in target)
        _ = self._validate_tensor_names(effective_index, requested_names)
        _, _, pipeline = self._require_components()
        view_metadata = self._ensure_view_metadata_cache(require_index_bytes=True)
        view_spec_proto = self._view_spec.proto if self._view_spec else None
        view_data_hash = view_metadata.view_data_hash or None if view_metadata else None
        view_index_hint = (
            view_metadata.view_index_bytes or None if view_metadata else None
        )
        replica_uuid = options.replica_uuid if options is not None else None
        return pipeline.get_into(
            target,
            artifact_id=artifact_id,
            key=None,
            device=device,
            options=options,
            view_spec=view_spec_proto,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
            tensor_names=requested_names,
            ctx=ctx,
        )

    def tensor_into(
        self,
        name: str,
        target_tensor: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        if not isinstance(target_tensor, torch.Tensor):
            raise ArtifactError(
                "tensor_into target must be a torch.Tensor",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resolved_device = device if device is not None else target_tensor.device
        self.subset([name]).realize(
            ArtifactRealizationSpec.caller_tensors(
                target={str(name): target_tensor},
                device=resolved_device,
                options=options,
            ),
            ctx=ctx,
        ).complete()

    def view(
        self,
        *,
        slices: Mapping[str, Sequence[object]] | None = None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    ) -> Artifact:
        """Return a derived lazy view; no RPCs occur until materialization or exists()."""
        return self._derive_view(
            slices=slices,
            transpose=transpose,
            subset=None,
            composer=ViewSpecComposer(),
        )

    def subset(self, names: Sequence[str]) -> Artifact:
        return self._derive_view(
            slices=None,
            transpose=None,
            subset=list(names),
            composer=ViewSpecComposer(),
        )

    def slice(self, slices: Mapping[str, Sequence[object]]) -> Artifact:
        return self.view(slices=slices)

    def view_builder(self) -> ViewBuilder:
        return ViewBuilder(artifact_ref=weakref.ref(self), composer=ViewSpecComposer())

    def bind(
        self,
        device: torch.device | str,
        *,
        mapping: CopyPlan | None = None,
        packing: str = "byte_space",
        options: GetArtifactOptions | None = None,
        capacity_bytes: int | None = None,
        publish: bool = False,
        runtime_artifact_policy: RuntimeArtifactPolicyInput | None = None,
        ctx: CallContext | None = None,
    ) -> Binding:
        handle = self.realize(
            ArtifactRealizationSpec.binding(
                device=device,
                mapping=mapping,
                packing=packing,
                options=options,
                capacity_bytes=capacity_bytes,
                publish=publish,
                runtime_artifact_policy=runtime_artifact_policy,
            ),
            ctx=ctx,
        )
        return handle.binding()

    def _execute_bind_owned(
        self,
        device: torch.device | str,
        *,
        mapping: CopyPlan | None = None,
        packing: str = "byte_space",
        options: GetArtifactOptions | None = None,
        capacity_bytes: int | None = None,
        publish: bool = False,
        runtime_artifact_policy: RuntimeArtifactPolicyInput | None = None,
        ctx: CallContext | None = None,
    ) -> Binding:
        """Allocate daemon-owned target tensors, fill from this artifact, and return a Binding."""
        self._require_components()
        effective_index = self._effective_index()
        if not effective_index.entries:
            raise ArtifactError(
                "Artifact index is empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise ArtifactError(
                "bind() requires a CUDA device",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        required_capacity = int(effective_index.total_size_bytes)
        if mapping is not None:
            self._ensure_metadata()
            canonical_index = self._canonical_index
            if canonical_index is None:
                raise ArtifactError(
                    "Missing canonical index for bind()",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            inferred_entries = infer_mapped_target_entries(
                plan=normalize_copy_plan(mapping),
                canonical_index=canonical_index,
                view_narrows=view_narrow_ranges(self._view_spec),
            )
            required_capacity = sum(int(entry.size_bytes) for entry in inferred_entries)
        if capacity_bytes is not None:
            requested_capacity = int(capacity_bytes)
            if requested_capacity <= 0:
                raise ArtifactError(
                    "capacity_bytes must be positive",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if requested_capacity < required_capacity:
                raise ArtifactError(
                    "capacity_bytes is smaller than the selected artifact layout",
                    status_code="RESOURCE_EXHAUSTED",
                    retryable=False,
                )
        return self._bind_owned(
            device=device_obj,
            mapping=mapping,
            packing=packing,
            options=options,
            publish=publish,
            runtime_artifact_policy=_resolve_runtime_artifact_policy(
                runtime_artifact_policy
            ),
            ctx=ctx,
        )

    def bind_into(
        self,
        target_tensors: Mapping[str, torch.Tensor],
        *,
        mapping: CopyPlan | None = None,
        packing: str = "byte_space",
        options: GetArtifactOptions | None = None,
        publish: bool = False,
        runtime_artifact_policy: RuntimeArtifactPolicyInput | None = None,
        ctx: CallContext | None = None,
    ) -> Binding:
        handle = self.realize(
            ArtifactRealizationSpec.adopted_binding(
                target=target_tensors,
                mapping=mapping,
                packing=packing,
                options=options,
                publish=publish,
                runtime_artifact_policy=runtime_artifact_policy,
            ),
            ctx=ctx,
        )
        return handle.binding()

    def _execute_bind_into(
        self,
        target_tensors: Mapping[str, torch.Tensor],
        *,
        mapping: CopyPlan | None = None,
        packing: str = "byte_space",
        options: GetArtifactOptions | None = None,
        publish: bool = False,
        runtime_artifact_policy: RuntimeArtifactPolicyInput | None = None,
        ctx: CallContext | None = None,
    ) -> Binding:
        """Adopt user-owned CUDA tensors, fill once, and return a Binding."""
        resolved_runtime_artifact_policy = _resolve_runtime_artifact_policy(
            runtime_artifact_policy
        )
        if mapping is not None:
            return self._bind_into_mapped(
                target_tensors=target_tensors,
                mapping=mapping,
                packing=packing,
                options=options,
                publish=publish,
                runtime_artifact_policy=resolved_runtime_artifact_policy,
                ctx=ctx,
            )
        store, runtime, pipeline = self._require_components()
        if not isinstance(target_tensors, Mapping) or not target_tensors:
            raise ArtifactError(
                "bind_into target_tensors must be a non-empty mapping",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        mode = str(packing).strip().lower()
        if mode not in {"append", "plan", "byte_space"}:
            raise ArtifactError(
                "packing must be 'append', 'plan', or 'byte_space'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        self._ensure_metadata()
        canonical_index = self._canonical_index
        canonical_index_bytes = self._canonical_index_bytes
        if canonical_index is None or canonical_index_bytes is None:
            raise ArtifactError(
                "Missing canonical index for bind_into",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        effective_index = self._effective_index()
        expected_names = [entry.name for entry in effective_index.entries]
        target_names = [str(name) for name in target_tensors]
        if set(target_names) != set(expected_names):
            raise ArtifactError(
                "bind_into target tensors must cover the artifact selection",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        first_tensor = next(iter(target_tensors.values()))
        if not isinstance(first_tensor, torch.Tensor):
            raise ArtifactError(
                "bind_into targets must be torch.Tensor instances",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not first_tensor.is_cuda:
            raise ArtifactError(
                "bind_into requires CUDA tensors",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        device_id = resolve_device(first_tensor.device, allow_cpu=False)
        for name, tensor in target_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise ArtifactError(
                    f"bind_into target '{name}' must be a torch.Tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_cuda:
                raise ArtifactError(
                    f"bind_into target '{name}' must be CUDA",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if resolve_device(tensor.device, allow_cpu=False) != device_id:
                raise ArtifactError(
                    "bind_into targets must share the same CUDA device",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )

        ttl_ms = 0
        target_regions = _register_target_regions_for_realization(
            store=store,
            target_tensors=target_tensors,
            device_id=device_id,
            ttl_ms=ttl_ms,
            context="bind_into.register_regions",
        )

        selection_order: tuple[str, ...] | None = None
        if mode == "byte_space":
            canonical_names = {entry.name for entry in canonical_index.entries}
            if set(expected_names) != canonical_names:
                selection_order = tuple(expected_names)
        else:
            selection_order = tuple(target_names)

        view_spec_proto = None
        if self._view_spec is not None and not self._view_spec.is_identity:
            view_spec_proto = self._view_spec.proto
        view_index_hint = (
            self._view_metadata.view_index_bytes if self._view_metadata else None
        )

        source_policy = _resolve_source_policy_from_options(options)

        client = runtime.ensure_client()
        operation_id = _build_transport_operation_id(
            base_operation_id=uuid.uuid4().hex,
            ctx=ctx,
        )
        rpc_timeout_s = _ctx_timeout_s(ctx)
        try:
            response = None
            region_layout = None
            attempt = 0
            while attempt < 2:
                region_layout = pipeline._build_region_backed_layout(
                    canonical_index=canonical_index,
                    canonical_index_bytes=canonical_index_bytes,
                    target=target_tensors,
                    device_id=device_id,
                    tensor_names=selection_order,
                    view_spec=view_spec_proto,
                    view_id=None,
                    view_index_hint=view_index_hint,
                    selection_order=selection_order,
                )
                try:
                    selection = self._build_region_layout_selection(
                        region_layout=region_layout,
                        view_spec_proto=view_spec_proto,
                    )
                    response = client.materialize_into_target(
                        selection=selection,
                        target_layout=region_layout.layout,
                        device_uuid=device_uuid_for(device_id),
                        source_policy=source_policy,
                        runtime_artifact_policy=resolved_runtime_artifact_policy,
                        operation_id=operation_id,
                        group_realization=ctx.group_realization
                        if ctx is not None
                        else None,
                        timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = map_materialization_error(exc)
                    if (
                        error.status_code
                        in {
                            "DATA_LOSS",
                            "FAILED_PRECONDITION",
                            "NOT_FOUND",
                        }
                        and attempt == 0
                    ):
                        target_regions.release(context="bind_into.retry_rebind")
                        target_regions = _register_target_regions_for_realization(
                            store=store,
                            target_tensors=target_tensors,
                            device_id=device_id,
                            ttl_ms=ttl_ms,
                            context="bind_into.retry_rebind.register_regions",
                        )
                        attempt += 1
                        continue
                    target_regions.release(context="bind_into.materialize_error")
                    raise ArtifactError(
                        str(error),
                        status_code=error.status_code,
                        retryable=False,
                    ) from exc

                if (
                    response.status
                    != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
                ):
                    target_regions.release(context="bind_into.materialize_status")
                    raise ArtifactError(
                        "MaterializeIntoTarget returned non-success status",
                        status_code="DATA_LOSS",
                        retryable=False,
                    )
                break

            if response is None or region_layout is None:
                raise ArtifactError(
                    "MaterializeIntoTarget retry failed to produce a response",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
        except Exception:
            target_regions.release(context="bind_into.exception")
            raise

        try:
            (
                binding_id,
                binding_layout,
                current_value_metadata,
                binding_current_value_publication_token,
            ) = _register_client_binding(
                runtime=runtime,
                device_id=device_id,
                region_layout=region_layout,
                canonical_index_bytes=canonical_index_bytes,
                selection=response.resolved_selection,
                source_artifact_id=self._ensure_identified(),
                ctx=ctx,
            )
        except Exception:
            target_regions.release(context="bind_into.create_binding_error")
            raise

        slot = InplaceSlot(
            store=store,
            runtime=runtime,
            pipeline=pipeline,
            tensors=target_tensors,
            device=first_tensor.device,
            device_id=device_id,
            layout=binding_layout,
            binding_id=binding_id,
            region_ids=region_layout.region_ids,
            selection_names=region_layout.selection_names,
            view_id=region_layout.view_id,
            view_subset_hash=region_layout.view_subset_hash,
            view_spec=view_spec_proto,
            current_value_metadata=current_value_metadata,
            binding_current_value_publication_token=(
                binding_current_value_publication_token
            ),
        )
        return Binding(slot, publish=publish, ctx=ctx)

    def _bind_into_mapped(
        self,
        *,
        target_tensors: Mapping[str, torch.Tensor],
        mapping: CopyPlan,
        packing: str,
        options: GetArtifactOptions | None,
        publish: bool,
        runtime_artifact_policy: RuntimeArtifactPolicy | None,
        ctx: CallContext | None,
    ) -> Binding:
        store, runtime, pipeline = self._require_components()
        if not isinstance(target_tensors, Mapping) or not target_tensors:
            raise ArtifactError(
                "bind_into target_tensors must be a non-empty mapping",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        total_start = time.perf_counter()
        mode = str(packing).strip().lower()
        if mode != "byte_space":
            raise ArtifactError(
                "mapped binding requires packing='byte_space'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        self._ensure_metadata()
        canonical_index = self._canonical_index
        canonical_index_bytes = self._canonical_index_bytes
        if canonical_index is None or canonical_index_bytes is None:
            raise ArtifactError(
                "Missing canonical index for bind_into",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        copy_plan = normalize_copy_plan(mapping)
        view_narrows = view_narrow_ranges(self._view_spec)
        validate_copy_plan(
            plan=copy_plan,
            canonical_index=canonical_index,
            target_tensors=target_tensors,
            view_narrows=view_narrows,
            require_full_coverage=True,
        )

        first_tensor = next(iter(target_tensors.values()))
        if not isinstance(first_tensor, torch.Tensor):
            raise ArtifactError(
                "bind_into targets must be torch.Tensor instances",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not first_tensor.is_cuda:
            raise ArtifactError(
                "bind_into requires CUDA tensors",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        device_id = resolve_device(first_tensor.device, allow_cpu=False)
        for name, tensor in target_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise ArtifactError(
                    f"bind_into target '{name}' must be a torch.Tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_cuda:
                raise ArtifactError(
                    f"bind_into target '{name}' must be CUDA",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if resolve_device(tensor.device, allow_cpu=False) != device_id:
                raise ArtifactError(
                    "bind_into targets must share the same CUDA device",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )

        ttl_ms = 0

        stage_start = time.perf_counter()
        target_regions = _register_target_regions_for_realization(
            store=store,
            target_tensors=target_tensors,
            device_id=device_id,
            ttl_ms=ttl_ms,
            context="bind_into_mapped.register_regions",
        )
        region_ids = target_regions.region_ids
        region_register_sec = time.perf_counter() - stage_start
        selection_order = tuple(sorted(str(name) for name in target_tensors))

        view_spec_proto = None
        if self._view_spec is not None and not self._view_spec.is_identity:
            view_spec_proto = self._view_spec.proto
        selection_index_bytes = bytes(canonical_index_bytes)
        if view_spec_proto is not None and view_spec_proto.tensors:
            selection_index_bytes = compute_selected_index_bytes(
                canonical_index_bytes=canonical_index_bytes,
                view_spec=view_spec_proto,
                tensor_names=None,
            )
        elif (
            self._view_metadata is not None
            and self._view_metadata.view_index_bytes
            and not str(self._view_metadata.view_id).startswith("mapped:v1:")
        ):
            selection_index_bytes = bytes(self._view_metadata.view_index_bytes)

        source_policy = _resolve_source_policy_from_options(options)

        client = runtime.ensure_client()
        operation_id = _build_transport_operation_id(
            base_operation_id=uuid.uuid4().hex,
            ctx=ctx,
        )
        rpc_timeout_s = _ctx_timeout_s(ctx)
        source_view_id = self._mapped_source_view_id(runtime)
        mapped_view_id = compute_mapped_view_id(
            canonical_index_bytes=canonical_index_bytes,
            source_view_id=source_view_id,
            plan=copy_plan,
            target_tensors=target_tensors,
        )
        region_layout_build_sec = 0.0
        materialize_rpc_sec = 0.0
        try:
            response = None
            region_layout = None
            attempt = 0
            while attempt < 2:
                stage_start = time.perf_counter()
                region_layout = pipeline._build_mapped_region_backed_layout(
                    target=target_tensors,
                    device_id=device_id,
                    selection_order=selection_order,
                    mapped_view_id=mapped_view_id,
                    selection_index_bytes=selection_index_bytes,
                )
                region_layout_build_sec += time.perf_counter() - stage_start
                try:
                    selection = self._build_region_layout_selection(
                        region_layout=region_layout,
                        view_spec_proto=view_spec_proto,
                    )
                    stage_start = time.perf_counter()
                    response = client.materialize_into_mapped_target(
                        selection=selection,
                        target_layout=region_layout.layout,
                        device_uuid=device_uuid_for(device_id),
                        source_policy=source_policy,
                        runtime_artifact_policy=runtime_artifact_policy,
                        copy_plan=copy_plan,
                        dst_tensors=target_tensors,
                        operation_id=operation_id,
                        group_realization=ctx.group_realization
                        if ctx is not None
                        else None,
                        timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
                    )
                    materialize_rpc_sec += time.perf_counter() - stage_start
                except Exception as exc:  # noqa: BLE001
                    message = str(exc)
                    if (
                        "MaterializeIntoMappedTarget" in message
                        and "not supported" in message.lower()
                    ):
                        raise ArtifactError(
                            "Mapped binding is not supported by the connected StoreDaemon",
                            status_code="FAILED_PRECONDITION",
                            retryable=False,
                        ) from exc
                    error = map_materialization_error(exc)
                    if (
                        error.status_code
                        in {
                            "DATA_LOSS",
                            "FAILED_PRECONDITION",
                            "NOT_FOUND",
                        }
                        and attempt == 0
                    ):
                        target_regions.release(context="bind_into_mapped.retry_rebind")
                        target_regions = _register_target_regions_for_realization(
                            store=store,
                            target_tensors=target_tensors,
                            device_id=device_id,
                            ttl_ms=ttl_ms,
                            context="bind_into_mapped.retry_rebind.register_regions",
                        )
                        region_ids = target_regions.region_ids
                        attempt += 1
                        continue
                    target_regions.release(context="bind_into_mapped.materialize_error")
                    raise ArtifactError(
                        str(error),
                        status_code=error.status_code,
                        retryable=False,
                    ) from exc

                if (
                    response.status
                    != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
                ):
                    target_regions.release(
                        context="bind_into_mapped.materialize_status"
                    )
                    raise ArtifactError(
                        "MaterializeIntoMappedTarget returned non-success status",
                        status_code="DATA_LOSS",
                        retryable=False,
                    )
                break

            if response is None or region_layout is None:
                raise ArtifactError(
                    "MaterializeIntoMappedTarget retry failed to produce a response",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
        except Exception:
            target_regions.release(context="bind_into_mapped.exception")
            raise

        stage_start = time.perf_counter()
        try:
            (
                binding_id,
                binding_layout,
                current_value_metadata,
                binding_current_value_publication_token,
            ) = _register_client_binding(
                runtime=runtime,
                device_id=device_id,
                region_layout=region_layout,
                canonical_index_bytes=canonical_index_bytes,
                selection=response.resolved_selection,
                source_artifact_id=self._ensure_identified(),
                ctx=ctx,
            )
        except Exception:
            target_regions.release(context="bind_into_mapped.create_binding_error")
            raise
        binding_register_sec = time.perf_counter() - stage_start

        slot = InplaceSlot(
            store=store,
            runtime=runtime,
            pipeline=pipeline,
            tensors=target_tensors,
            device=first_tensor.device,
            device_id=device_id,
            layout=binding_layout,
            binding_id=binding_id,
            region_ids=region_layout.region_ids,
            selection_names=region_layout.selection_names,
            view_id=region_layout.view_id,
            view_subset_hash=region_layout.view_subset_hash,
            view_spec=view_spec_proto,
            current_value_metadata=current_value_metadata,
            binding_current_value_publication_token=(
                binding_current_value_publication_token
            ),
            copy_plan=copy_plan,
        )
        logger.info(
            "TensorCast bind_into_mapped timings: artifact_id=%s, targets=%d, "
            "copy_entries=%d, regions=%d, collective=%s, region_register=%.3fs, "
            "layout_build=%.3fs, materialize_rpc=%.3fs, binding_register=%.3fs, "
            "total=%.3fs",
            self._ensure_identified(),
            len(target_tensors),
            len(copy_plan),
            len(region_ids),
            bool(ctx is not None and ctx.collective is not None),
            region_register_sec,
            region_layout_build_sec,
            materialize_rpc_sec,
            binding_register_sec,
            time.perf_counter() - total_start,
        )
        return Binding(slot, publish=publish, ctx=ctx)

    def _bind_owned(
        self,
        *,
        device: torch.device,
        mapping: CopyPlan | None,
        packing: str,
        options: GetArtifactOptions | None,
        publish: bool,
        runtime_artifact_policy: RuntimeArtifactPolicy | None,
        ctx: CallContext | None,
    ) -> Binding:
        store, runtime, _ = self._require_components()
        self._ensure_metadata()
        canonical_index_bytes = self._canonical_index_bytes
        canonical_index = self._canonical_index
        if canonical_index_bytes is None or canonical_index is None:
            raise ArtifactError(
                "Missing canonical index for bind()",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        mode = str(packing).strip().lower()
        if mode not in {"append", "plan", "byte_space"}:
            raise ArtifactError(
                "packing must be 'append', 'plan', or 'byte_space'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        device_id = resolve_device(device, allow_cpu=False)

        with tensorcast_profile_stage(
            "tensorcast",
            "artifact.bind_owned.prepare",
            logger=logger,
            extra={
                "artifact_id": self._artifact_id,
                "device": str(device),
                "mapping_provided": mapping is not None,
                "packing": mode,
            },
        ) as profile:
            selection_names_override = (
                tuple(entry.name for entry in self._effective_index().entries)
                if mapping is None and mode != "byte_space"
                else None
            )
            source_selection = self._resolve_realization_selection(
                tensor_names_override=selection_names_override
            ).proto
            index_kind = (
                store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
                if source_selection.view_id or source_selection.tensor_names
                else store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED
            )
            view_id = str(source_selection.view_id or "")
            logical_layout_hash = bytes(source_selection.logical_layout_hash)

            copy_plan_proto: store_daemon_pb2.CopyPlan | None = None
            dst_specs: tuple[store_daemon_pb2.MappedTensorSpec, ...] = ()
            target_tensor_count = 0
            if mapping is None:
                target_tensor_count = len(self._effective_index().entries)
                owner_layout = build_owned_layout(
                    entries=self._effective_index().entries,
                    device_id=device_id,
                    index_kind=index_kind,
                    logical_layout_hash=logical_layout_hash,
                    view_id=view_id or None,
                    ordered_names=tuple(source_selection.tensor_names)
                    if source_selection.tensor_names
                    else None,
                )
            else:
                if mode != "byte_space":
                    raise ArtifactError(
                        "mapped binding requires packing='byte_space'",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                normalized_plan = normalize_copy_plan(mapping)
                inferred_entries = infer_mapped_target_entries(
                    plan=normalized_plan,
                    canonical_index=canonical_index,
                    view_narrows=view_narrow_ranges(self._view_spec),
                )
                target_tensor_count = len(inferred_entries)
                target_spec_payloads = [
                    {
                        "name": entry.name,
                        "dtype": str(entry.dtype),
                        "shape": tuple(int(v) for v in entry.shape),
                        "stride": tuple(int(v) for v in entry.stride),
                        "logical_length": int(entry.size_bytes),
                    }
                    for entry in inferred_entries
                ]
                mapped_view_id = compute_mapped_view_id_from_specs(
                    canonical_index_bytes=canonical_index_bytes,
                    source_view_id=self._mapped_source_view_id(runtime),
                    plan=normalized_plan,
                    target_specs=target_spec_payloads,
                )
                dst_specs = tuple(
                    build_mapped_tensor_spec(
                        name=entry.name,
                        shape=entry.shape,
                        stride=entry.stride,
                        dtype=str(entry.dtype),
                        logical_length=int(entry.size_bytes),
                    )
                    for entry in inferred_entries
                )
                owner_layout = build_owned_layout(
                    entries=inferred_entries,
                    device_id=device_id,
                    index_kind=store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW,
                    logical_layout_hash=None,
                    view_id=mapped_view_id,
                    dst_specs=dst_specs,
                    separate_storages=True,
                )
                copy_plan_proto = store_daemon_pb2.CopyPlan(version=1)
                for entry in normalized_plan:
                    entry_proto = copy_plan_proto.entries.add()
                    entry_proto.ckpt_name = str(entry.ckpt_name)
                    entry_proto.dst_name = str(entry.dst_name)
                    if entry.ckpt_range is not None:
                        entry_proto.ckpt_range.dim = int(entry.ckpt_range.dim)
                        entry_proto.ckpt_range.start = int(entry.ckpt_range.start)
                        entry_proto.ckpt_range.end = int(entry.ckpt_range.end)
                    if entry.dst_range is not None:
                        entry_proto.dst_range.dim = int(entry.dst_range.dim)
                        entry_proto.dst_range.start = int(entry.dst_range.start)
                        entry_proto.dst_range.end = int(entry.dst_range.end)
            if profile is not None:
                profile["source_tensor_count"] = len(source_selection.tensor_names)
                profile["binding_layout_id"] = owner_layout.binding_layout_id
                profile["target_index_bytes_len"] = len(owner_layout.target_index_bytes)
                profile["target_tensor_count"] = target_tensor_count

        source_policy = _resolve_source_policy_from_options(options)
        rpc_timeout_s = _ctx_timeout_s(ctx)
        operation_id = _build_transport_operation_id(
            base_operation_id=uuid.uuid4().hex,
            ctx=ctx,
        )
        try:
            with tensorcast_profile_stage(
                "tensorcast",
                "artifact.bind_owned.create_owned_binding_rpc",
                logger=logger,
                extra={
                    "artifact_id": self._artifact_id,
                    "device": str(device),
                    "binding_layout_id": owner_layout.binding_layout_id,
                    "target_tensor_count": target_tensor_count,
                    "target_index_bytes_len": len(owner_layout.target_index_bytes),
                },
            ) as profile:
                response = runtime.ensure_client().create_owned_binding(
                    source_selection=source_selection,
                    target_layout=owner_layout.target_layout,
                    target_index_bytes=owner_layout.target_index_bytes,
                    device_uuid=device_uuid_for(device_id),
                    binding_layout_id=owner_layout.binding_layout_id,
                    source_policy=source_policy,
                    runtime_artifact_policy=runtime_artifact_policy,
                    copy_plan=copy_plan_proto,
                    dst_specs=dst_specs,
                    operation_id=operation_id,
                    group_realization=ctx.group_realization
                    if ctx is not None
                    else None,
                    timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
                )
                if profile is not None:
                    profile["binding_id"] = getattr(response, "binding_id", None)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if any(
                marker in message
                for marker in (
                    "CreateOwnedBinding",
                    "createownedbinding",
                    "Method not found",
                    "UNIMPLEMENTED",
                )
            ):
                raise ArtifactError(
                    "Owned binding is not supported by the connected StoreDaemon",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                ) from exc
            error = map_materialization_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=False,
            ) from exc

        try:
            with tensorcast_profile_stage(
                "tensorcast",
                "artifact.bind_owned.restore_binding_tensors",
                logger=logger,
                extra={
                    "artifact_id": self._artifact_id,
                    "device": str(device),
                    "binding_id": str(response.binding_id),
                },
            ) as profile:
                tensors = restore_owned_binding_tensors(
                    response=response,
                    runtime=runtime,
                    device_id=device_id,
                )
                if profile is not None:
                    profile["restored_tensor_count"] = len(tensors)
                    profile["restored_tensor_bytes"] = sum(
                        int(tensor.numel() * tensor.element_size())
                        for tensor in tensors.values()
                    )
        except Exception:
            _close_client_binding_best_effort(
                runtime,
                str(response.binding_id),
                context="CreateOwnedBinding tensor restore",
            )
            raise

        current_value_metadata = None
        staged_value_metadata = None
        group_realization_acquire = None
        try:
            with tensorcast_profile_stage(
                "tensorcast",
                "artifact.bind_owned.parse_binding_value",
                logger=logger,
                extra={
                    "artifact_id": self._artifact_id,
                    "binding_id": str(response.binding_id),
                    "created_staged_value": bool(response.created_staged_value),
                },
            ) as profile:
                if bool(response.created_staged_value):
                    staged_value_metadata = parse_binding_value_or_raise(
                        response.staged_value,
                        rpc_name="CreateOwnedBinding staged_value",
                        expected_binding_id=str(response.binding_id),
                        expected_binding_layout_id=owner_layout.binding_layout_id,
                    )
                    group_realization_acquire = GroupRealizationAcquireRef.from_proto(
                        response.group_realization_acquire
                    )
                    if profile is not None and staged_value_metadata is not None:
                        profile["staged_artifact_id"] = getattr(
                            staged_value_metadata,
                            "artifact_id",
                            None,
                        )
                else:
                    current_value_metadata = parse_binding_value_or_raise(
                        response.current_value,
                        rpc_name="CreateOwnedBinding",
                        expected_binding_id=str(response.binding_id),
                        expected_binding_layout_id=owner_layout.binding_layout_id,
                    )
                    if profile is not None and current_value_metadata is not None:
                        profile["sealed_artifact_id"] = getattr(
                            current_value_metadata, "artifact_id", None
                        )
        except ArtifactError:
            _close_client_binding_best_effort(
                runtime,
                str(response.binding_id),
                context="CreateOwnedBinding value parse",
            )
            raise
        except ValueError as exc:
            _close_client_binding_best_effort(
                runtime,
                str(response.binding_id),
                context="CreateOwnedBinding malformed staged metadata",
            )
            raise ArtifactError(
                f"CreateOwnedBinding returned malformed staged metadata: {exc}",
                status_code="DATA_LOSS",
                retryable=False,
            ) from exc

        slot = OwnedBindingSlot(
            store=store,
            runtime=runtime,
            tensors=tensors,
            layout=owner_layout,
            binding_id=str(response.binding_id),
            current_value_metadata=current_value_metadata,
            device=device,
            device_id=device_id,
            binding_current_value_publication_token=bytes(
                response.binding_current_value_publication_token
            )
            or None,
            staged_value_metadata=staged_value_metadata,
            group_realization_acquire=group_realization_acquire,
            binding_current_value_publication_operation_id=operation_id,
            materialization_diagnostics=binding_materialization_diagnostics_from_response(
                response,
                layout=owner_layout,
            ),
        )
        if slot.current_value_metadata is None and slot.staged_value_metadata is None:
            _close_client_binding_best_effort(
                runtime,
                str(response.binding_id),
                context="CreateOwnedBinding missing value metadata",
            )
            raise ArtifactError(
                "CreateOwnedBinding returned neither current_value nor staged_value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return Binding(slot, publish=publish, ctx=ctx)

    def batch(self, *, device: torch.device | str) -> "BatchContext":
        from tensorcast.api.store.batch_context import BatchContext

        return BatchContext(self, device=device)

    async def tensor_async(
        self, name: str, *, device: torch.device | str
    ) -> torch.Tensor:
        store = self._store_ref() if self._store_ref is not None else None
        if store is None or store.closed or self._released:
            raise ArtifactError(
                "Artifact handle is released or store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        self._ensure_identified()
        loop = asyncio.get_running_loop()
        view_hash = None
        if self._view_metadata is not None:
            view_hash = self._view_metadata.view_data_hash
        batcher = getattr(store, "_batcher", None)
        if batcher is None:
            return await loop.run_in_executor(
                None, lambda: self.tensor(name, device=device)
            )
        future = batcher.submit(
            self,
            name,
            device=device,
            view_hash=view_hash,
            target_loop=loop,
        )
        return await future

    async def tensor_dict_async(
        self,
        *,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.tensor_dict(device=device))

    def _prefetch_serving_binding(
        self,
        *,
        target: RealizationTarget | RealizationTargetSet,
        readiness: RuntimeBindingReadiness,
        retention: PrefetchRetentionPolicy | None,
        ctx: CallContext | None,
    ) -> Operation[RuntimePrefetchResult]:
        artifact_id = self._ensure_identified()
        _, runtime, _ = self._require_components()
        resolved_selection = self._resolve_realization_selection()
        selection = resolved_selection.proto
        source_reuse = _serving_target_source_reuse(target)
        if source_reuse.mode in {"runtime_transform_required", "unsupported"}:
            reason = source_reuse.reason or (
                "source-to-target runtime transform requires a topology-scoped executor"
                if source_reuse.mode == "runtime_transform_required"
                else "runtime binding source is unsupported"
            )
            raise ArtifactError(
                f"runtime binding prefetch rejected before allocation: {reason}",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        target_proto = target.to_proto()
        target_bytes = target_proto.SerializeToString(deterministic=True)
        daemon_id = (
            getattr(runtime, "daemon_id", None) or None
        ) or runtime.daemon_endpoint
        operation_id = uuid.uuid4().hex
        if ctx is not None and ctx.idempotency_key:
            ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
            idempotency_key_hex = hashlib.sha256(
                ctx.idempotency_key.encode("utf-8")
            ).hexdigest()
            action_fingerprint = hashlib.sha256(
                b"prefetch_serving_binding|"
                + selection.SerializeToString(deterministic=True)
                + b"|"
                + target_bytes
                + f"|readiness={readiness}|daemon={daemon_id}".encode("utf-8")
            ).hexdigest()
            operation_id = str(
                uuid.uuid5(ns, f"{idempotency_key_hex}|{action_fingerprint}")
            )

        client = runtime.ensure_client()
        try:
            response = client.prefetch_serving_binding(
                source_selection=selection,
                target=target,
                requested_readiness=readiness,
                retention_policy=retention,
                operation_id=operation_id,
                group_realization=ctx.group_realization if ctx is not None else None,
            )
        except RuntimeError as exc:
            raise ArtifactError(
                str(exc),
                status_code="UNIMPLEMENTED"
                if "not supported" in str(exc).lower()
                or "unimplemented" in str(exc).lower()
                else "UNKNOWN",
                retryable=False,
            ) from exc

        operation_ref = (
            response.operation_ref if response.HasField("operation_ref") else None
        )
        if operation_ref is not None and operation_ref.operation_id:
            operation_id = str(operation_ref.operation_id)
        initial_status = _operation_status_from_proto(response.status)
        if initial_status.state in {"success", "failed", "cancelled"}:
            return PollingOperation(
                operation_id=operation_id,
                status_fn=lambda: initial_status,
                result_fn=lambda: _with_retained_binding_report(
                    _parse_serving_prefetch_result_any(response.status.result),
                    selection=resolved_selection,
                    target=target,
                    target_bytes=target_bytes,
                    operation_id=operation_id,
                ),
                cancel_fn=lambda: False,
                ctx=ctx,
            )

        def _result_factory(
            operation_response: operation_pb2.GetOperationResponse,
        ) -> RuntimePrefetchResult:
            return _with_retained_binding_report(
                _serving_prefetch_result_from_operation_response(operation_response),
                selection=resolved_selection,
                target=target,
                target_bytes=target_bytes,
                operation_id=operation_id,
            )

        return DaemonGlobalStoreOperation(
            operation_id=operation_id,
            runtime_ref=weakref.ref(runtime),
            ctx=ctx,
            context={
                "operation_kind": "prefetch_serving_binding",
                "target_artifact_id": artifact_id,
                "daemon_endpoint": str(getattr(runtime, "daemon_endpoint", "")),
            },
            result_factory=_result_factory,
            operation_ref=operation_ref,
        )

    def prefetch(
        self,
        *,
        device: torch.device | str | int | None = None,
        target: RealizationTarget | RealizationTargetSet | None = None,
        readiness: RuntimeBindingReadiness = "runtime_local_ready",
        retention: PrefetchRetentionPolicy | None = None,
        ctx: CallContext | None = None,
        options: GetArtifactOptions | None = None,
    ) -> Operation[PrefetchedReplica] | Operation[RuntimePrefetchResult]:
        if target is not None:
            if device is not None:
                raise ArtifactError(
                    "prefetch target and device are mutually exclusive",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            spec = (
                ArtifactRealizationSpec.target_set(
                    target=target,
                    readiness=readiness,
                    retention=retention,
                )
                if isinstance(target, RealizationTargetSet)
                else ArtifactRealizationSpec.retained_binding(
                    target=target,
                    readiness=readiness,
                    retention=retention,
                )
            )
            return self.realize_async(spec, ctx=ctx)
        if device is None:
            raise ArtifactError(
                "prefetch requires device or target",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return self.realize_async(
            ArtifactRealizationSpec.retained_replica(
                device=device,
                options=options,
                retention=retention,
            ),
            ctx=ctx,
        )

    def _execute_prefetch(
        self,
        *,
        device: torch.device | str | int | None = None,
        target: RealizationTarget | RealizationTargetSet | None = None,
        readiness: RuntimeBindingReadiness = "runtime_local_ready",
        retention: PrefetchRetentionPolicy | None = None,
        ctx: CallContext | None = None,
        options: GetArtifactOptions | None = None,
    ) -> Operation[PrefetchedReplica] | Operation[RuntimePrefetchResult]:
        from tensorcast.api._config import GetArtifactOptions

        artifact_id = self._ensure_identified()
        store, runtime, pipeline = self._require_components()
        if getattr(store, "_enable_prefetch", True) is False:
            raise ArtifactError(
                "Prefetch is disabled",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if target is not None:
            if device is not None:
                raise ArtifactError(
                    "prefetch target and device are mutually exclusive",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return self._prefetch_serving_binding(
                target=target,
                readiness=readiness,
                retention=retention,
                ctx=ctx,
            )
        if device is None:
            raise ArtifactError(
                "prefetch requires device or target",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        selection_inputs = self._resolve_selection_materialization_inputs()
        requested_names = selection_inputs.requested_names
        view_spec_proto = selection_inputs.view_spec_proto
        view_data_hash = selection_inputs.view_data_hash
        view_index_hint = selection_inputs.view_index_hint
        view_id_hint = selection_inputs.view_id_hint

        if ctx is not None and ctx.idempotency_key:
            # Deterministic operation ids require stable index bytes for logical layout hashing.
            self._ensure_metadata()

        if isinstance(device, int):
            if device < 0:
                device_obj = torch.device("cpu")
                device_id = CPU_DEVICE_ID
            else:
                device_obj = torch.device(f"cuda:{int(device)}")
                device_id = int(device)
        else:
            if isinstance(device, str) and device.strip().lower() == "dram":
                device_obj = torch.device("cpu")
                device_id = CPU_DEVICE_ID
            else:
                device_obj = (
                    device if isinstance(device, torch.device) else torch.device(device)
                )
                if device_obj.type == "cpu":
                    device_id = CPU_DEVICE_ID
                elif device_obj.type == "cuda":
                    device_id = int(
                        device_obj.index if device_obj.index is not None else 0
                    )
                else:
                    raise ArtifactError(
                        f"prefetch does not support device type {device_obj.type!r}",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
        daemon_id = (
            getattr(runtime, "daemon_id", None) or None
        ) or runtime.daemon_endpoint
        resolved_selection = self._resolve_realization_selection()
        selection = resolved_selection.proto
        view_id = view_id_hint or selection.view_id
        selection_hash = bytes(selection.selection_hash).hex()

        deterministic_replica_uuid: str | None = None
        if ctx is not None and ctx.idempotency_key:
            logical_layout_hash = bytes(selection.logical_layout_hash).hex()
            device_uuid = device_uuid_for(device_id)
            ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
            idempotency_key_hex = hashlib.sha256(
                ctx.idempotency_key.encode("utf-8")
            ).hexdigest()
            action_fingerprint = (
                f"prefetch|daemon={daemon_id}|artifact={artifact_id}|layout={logical_layout_hash}"
                f"|selection={selection_hash}|device={device_id}|device_uuid={device_uuid}|lease=NO_LEASE|v2"
            )
            deterministic_replica_uuid = str(
                uuid.uuid5(ns, f"{idempotency_key_hex}|{action_fingerprint}")
            )

        opts = options or GetArtifactOptions()
        opts = opts.model_copy(
            update={
                "wait_for_completion": False,
                "enable_verification": False,
            }
        )
        replica_uuid = deterministic_replica_uuid or opts.replica_uuid
        if not replica_uuid:
            replica_uuid = uuid.uuid4().hex

        payload, _ = pipeline.materialize_subset(
            artifact_id=artifact_id,
            key=None,
            device=device_obj,
            tensor_names=requested_names,
            view_id=view_id_hint,
            canonical_index_hint=self._canonical_index_bytes,
            view_spec=view_spec_proto,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
            options=opts,
            ctx=ctx,
            lease_mode=store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE,
        )
        self._update_metadata_from_payload(payload, runtime)
        operation_id = payload.ticket_replica_uuid or payload.replica_uuid or ""
        if not operation_id:
            raise ArtifactError(
                "Daemon returned empty operation_id for prefetch",
                status_code="DATA_LOSS",
                retryable=False,
            )

        source = materialization_source_label(payload.source)

        target_plan = RealizationTargetPlan(
            kind="retained_replica",
            device=device_obj,
        )
        envelope = envelope_for_retained_replica(
            total_bytes=_payload_total_bytes(payload),
            device_kind="cpu" if device_id == CPU_DEVICE_ID else "cuda",
            retry_reason_buckets=payload.retry_reason_buckets,
        )
        envelope.validate_for_target(target_plan)
        report = ArtifactRealizationReport(
            target_kind="retained_replica",
            source_selection_digest=resolved_selection.source_selection_digest,
            target_layout_digest=target_plan.target_layout_digest,
            copy_plan_digest=target_plan.copy_plan_digest,
            artifact_id=resolved_selection.artifact_id,
            view_id=view_id,
            artifact_profile=resolved_selection.artifact_profile,
            authority_scope=resolved_selection.authority_scope,
            generation_hint=resolved_selection.generation_hint,
            **selection_report_fields(resolved_selection),
            envelope=envelope,
            target_plan=target_plan,
            strategy_plan=strategy_plan_for_execution(
                envelope=envelope,
                options=opts,
                ctx=ctx,
                lease_mode=store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE,
            ),
            representation_admission=representation_admission_for_target(target_plan),
            lifecycle_plan=lifecycle_plan_for_envelope(target_plan, envelope),
            source=source,
            operation_id=operation_id,
            operation_backend="daemon_materialization",
            risk_labels=risk_labels_for_target(
                target_plan,
                envelope,
                source_selection_digest=resolved_selection.source_selection_digest,
            ),
            publishability=publishability_report_for(),
            materialization_diagnostics={
                "replica_uuid": str(payload.replica_uuid or ""),
                "ticket_replica_uuid": str(payload.ticket_replica_uuid or ""),
                "ticket_status": _materialization_ticket_status_label(
                    payload.ticket_status
                ),
                "retry_attempts": int(payload.retry_attempts),
                "retry_reason_buckets": dict(payload.retry_reason_buckets or {}),
            },
        )
        replica = PrefetchedReplica(
            artifact_id=artifact_id,
            view_id=view_id,
            operation_id=operation_id,
            device_id=device_id,
            daemon_id=daemon_id,
            source=source,
            report=report,
        )

        try:
            from tensorcast.api import _metrics as store_metrics

            store_metrics.record_prefetch_event(
                runtime.daemon_endpoint, status="issued"
            )
        except Exception:  # noqa: BLE001
            pass

        def _replica_result_factory() -> PrefetchedReplica:
            emit_artifact_realization_profile_event(report)
            return replica

        return DaemonReplicaOperation(
            operation_id=operation_id,
            runtime_ref=weakref.ref(runtime),
            ctx=ctx,
            result_factory=_replica_result_factory,
        )

    def pin_device_residency(
        self,
        *,
        device: str | int,
        ttl_ms: int | None,
        ctx: CallContext | None = None,
    ) -> Operation[PlacementPin]:
        if ttl_ms is not None and int(ttl_ms) <= 0:
            raise ArtifactError(
                "ttl_ms must be positive when provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        artifact_id = self._ensure_identified()
        _, runtime, _ = self._require_components()
        if isinstance(device, int):
            device_id = int(device)
        else:
            device_obj = torch.device(device)
            if device_obj.type == "cpu":
                raise ArtifactError(
                    "pin_device_residency does not support CPU devices",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            device_id = int(device_obj.index if device_obj.index is not None else 0)
        view_id = self._control_plane_view_id(runtime)
        timeout_s = _ctx_timeout_s(ctx)
        resp = runtime.ensure_client().create_placement_lease(
            artifact_id=artifact_id,
            view_id=view_id,
            device_id=device_id,
            ttl_ms=ttl_ms,
            timeout_s=timeout_s,
        )
        token = bytes(resp.lease_token)
        if not token:
            raise ArtifactError(
                "CreatePlacementLease returned empty lease_token",
                status_code="DATA_LOSS",
                retryable=False,
            )
        expires_at_ms: int | None = None
        if resp.HasField("expires_at"):
            try:
                expires_at_ms = int(
                    resp.expires_at.ToDatetime(tzinfo=timezone.utc).timestamp() * 1000
                )
            except Exception:  # noqa: BLE001
                expires_at_ms = None

        pin = PlacementPin(
            pin_id=int(resp.lease_id),
            capability_token=_encode_capability_token(token),
            daemon_id=(getattr(runtime, "daemon_id", None) or None)
            or runtime.daemon_endpoint,
            artifact_id=artifact_id,
            view_id=view_id,
            device_id=device_id,
            expires_at_ms=expires_at_ms,
            runtime_ref=weakref.ref(runtime),
        )

        status = OperationStatus(
            state="success",
            message="placement pin active",
            as_of_ms=int(time.time() * 1000),
        )

        def _cancel() -> bool:
            try:
                return bool(
                    runtime.ensure_client()
                    .release_placement_lease(
                        lease_token=_decode_capability_token(pin.capability_token),
                    )
                    .released
                )
            except Exception:
                return False

        return PollingOperation(
            operation_id=f"pin:{pin.pin_id}",
            status_fn=lambda: status,
            result_fn=lambda: pin,
            cancel_fn=_cancel,
            ctx=ctx,
        )

    def exists(self) -> bool:
        """Check existence lazily, surfacing ArtifactError on failures."""
        if self._canonical_index is not None:
            return True
        runtime = self._runtime_if_available()
        if runtime is None:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        try:
            artifact_id = self._ensure_identified()
        except ArtifactError as exc:
            if exc.status_code == "NOT_FOUND":
                return False
            raise
        cached = runtime.get_artifact_index_cached(artifact_id)
        if cached:
            with self._lock:
                self._hydrate_from_cache_entry(cached)
            return True
        try:
            canonical_index_bytes = runtime.ensure_client().get_artifact_index_by_id(
                artifact_id
            )
        except Exception as exc:  # noqa: BLE001
            error = map_materialization_error(exc)
            if error.status_code == "NOT_FOUND":
                runtime.invalidate_artifact(
                    artifact_id, key=self._key_hint, reason="exists_not_found"
                )
                return False
            raise error from exc
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        with self._lock:
            if self._canonical_index is None or self._canonical_index_bytes is None:
                self._set_metadata(
                    canonical_index_bytes,
                    canonical_index,
                    generation=self._generation,
                )
        runtime.cache_artifact_index(
            ArtifactCacheEntry(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                parsed_index=canonical_index,
                generation=self._generation,
                expires_at=time.monotonic(),
            )
        )
        return True

    @property
    def is_valid(self) -> bool:
        store = self._store_ref() if self._store_ref is not None else None
        return not self._released and store is not None and not store.closed

    def release(self) -> None:
        with self._lock:
            self._released = True

    def to_dict(self) -> ArtifactSerialized:
        encoded_index: str | None = None
        if self._canonical_index_bytes:
            encoded_index = base64.b64encode(self._canonical_index_bytes).decode(
                "utf-8"
            )
        return {
            "artifact_id": self._artifact_id,
            "key": self._key_hint,
            "canonical_index": encoded_index,
            "generation": self._generation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object], store: "Store") -> Artifact:
        artifact_id = data.get("artifact_id")
        key_hint = data.get("key")
        canonical_blob = data.get("canonical_index")
        generation = data.get("generation")
        canonical_index_bytes: bytes | None = None
        canonical_index: CanonicalIndex | None = None
        if canonical_blob:
            try:
                canonical_index_bytes = base64.b64decode(
                    str(canonical_blob).encode("utf-8")
                )
                canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            except Exception as exc:  # noqa: BLE001
                raise ArtifactError(
                    "Failed to decode serialized artifact metadata",
                    status_code="DATA_LOSS",
                    retryable=False,
                ) from exc
        generation_value = (
            int(generation) if isinstance(generation, (int, float)) else None
        )
        return cls(
            store_ref=weakref.ref(store),
            artifact_id=str(artifact_id) if artifact_id else None,
            key=str(key_hint) if key_hint else None,
            canonical_index_bytes=canonical_index_bytes,
            canonical_index=canonical_index,
            generation=generation_value,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_realization_selection(
        self,
        *,
        tensor_names_override: Sequence[str] | None = None,
    ) -> ResolvedArtifactSelection:
        artifact_id = self._ensure_identified()
        runtime = self._runtime_if_available()
        inputs = self._resolve_selection_materialization_inputs()
        canonical_index_bytes = self._canonical_index_bytes
        if canonical_index_bytes is None and runtime is not None:
            canonical_index_bytes = runtime.ensure_client().get_artifact_index_by_id(
                artifact_id
            )
        requested_names = inputs.requested_names
        view_spec_proto = inputs.view_spec_proto
        view_index_hint = inputs.view_index_hint
        if tensor_names_override is not None:
            requested_names = tuple(str(name) for name in tensor_names_override)
            if view_spec_proto is None and requested_names:
                view_spec_proto = _build_subset_view_spec_proto(
                    canonical_index=self._ensure_metadata(),
                    tensor_names=requested_names,
                )
            if canonical_index_bytes is None:
                raise ArtifactError(
                    "Canonical index bytes missing while resolving source selection",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if view_spec_proto is not None or requested_names:
                view_index_hint = compute_selected_index_bytes(
                    canonical_index_bytes=canonical_index_bytes,
                    view_spec=view_spec_proto,
                    tensor_names=requested_names,
                )
        try:
            return resolve_artifact_selection(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                canonical_index_resolver=(
                    runtime.ensure_client().get_artifact_index_by_id
                    if runtime is not None
                    else None
                ),
                view_spec=view_spec_proto,
                view_id=inputs.view_id_hint,
                tensor_names=requested_names,
                view_subset_hash=inputs.view_subset_hash,
                view_index_hint=view_index_hint,
                generation_hint=(
                    self._key_generation
                    if self._key_generation is not None
                    else self._generation
                ),
                allow_view_id_without_spec=bool(
                    inputs.view_id_hint
                    and not (view_spec_proto is not None and view_spec_proto.tensors)
                ),
            )
        except ValueError as exc:
            raise ArtifactError(
                str(exc),
                status_code="FAILED_PRECONDITION",
                retryable=False,
            ) from exc

    def _target_tensors_digest(self, target: object) -> str:
        if not isinstance(target, Mapping) or not target:
            raise ArtifactError(
                "target tensors must be a non-empty mapping",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        digest = hashlib.sha256()
        for raw_name, tensor in sorted(target.items(), key=lambda item: str(item[0])):
            name = str(raw_name)
            if not isinstance(tensor, torch.Tensor):
                raise ArtifactError(
                    f"target '{name}' must be a torch.Tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(int(v) for v in tensor.shape)).encode("utf-8"))
            digest.update(str(tuple(int(v) for v in tensor.stride())).encode("utf-8"))
            digest.update(str(int(tensor.storage_offset())).encode("utf-8"))
            digest.update(str(tensor.device).encode("utf-8"))
        return digest.hexdigest()

    def _require_components(
        self,
    ) -> tuple["Store", StoreRuntimeContext, MaterializationPipeline]:
        store = self._store_ref() if self._store_ref is not None else None
        if store is None or store.closed or self._released:
            raise ArtifactError(
                "Artifact handle is released or store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return store, store._runtime, store._materialization

    def _control_plane_view_id(self, runtime: "StoreRuntimeContext") -> str:
        if self._view_spec is None or self._view_spec.is_identity:
            return ""
        view_proto = self._view_spec.proto
        if view_proto is None:
            raise ArtifactError(
                "View spec proto missing while resolving view_id",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        canonical_index_bytes = self._canonical_index_bytes
        if canonical_index_bytes is None:
            try:
                canonical_index_bytes = (
                    runtime.ensure_client().get_artifact_index_by_id(
                        self._ensure_identified()
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise_mapped_materialization_error(exc)
        if canonical_index_bytes is None:
            raise ArtifactError(
                "Canonical index bytes missing while resolving view_id",
                status_code="INTERNAL",
                retryable=False,
            )
        try:
            return compute_view_id(view_proto, canonical_index_bytes)
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                "Failed to compute view_id for control-plane action",
                status_code="INTERNAL",
                retryable=False,
            ) from exc

    def _mapped_source_view_id(self, runtime: "StoreRuntimeContext") -> str:
        view_metadata = self._ensure_view_metadata_cache(require_view_id=True)
        if view_metadata is not None and view_metadata.view_id:
            return str(view_metadata.view_id)
        return self._control_plane_view_id(runtime)

    def _resolve_selection_materialization_inputs(
        self,
    ) -> _SelectionMaterializationInputs:
        view_metadata = self._ensure_view_metadata_cache(require_index_bytes=True)
        requested_names = (
            tuple(view_metadata.tensor_names)
            if view_metadata is not None and view_metadata.tensor_names
            else None
        )
        view_spec_proto = self._view_spec.proto if self._view_spec else None
        if view_spec_proto is None and requested_names:
            canonical_index = self._ensure_metadata()
            subset_view_proto = _build_subset_view_spec_proto(
                canonical_index=canonical_index,
                tensor_names=requested_names,
            )
            if subset_view_proto is not None:
                view_spec_proto = subset_view_proto
        view_data_hash = view_metadata.view_data_hash or None if view_metadata else None
        view_index_hint = (
            view_metadata.view_index_bytes or None if view_metadata else None
        )
        view_id_hint = (
            str(view_metadata.view_id)
            if view_metadata is not None
            and view_metadata.view_id
            and view_spec_proto is None
            else None
        )
        view_subset_hash = (
            compute_view_subset_hash(requested_names) if requested_names else None
        )
        return _SelectionMaterializationInputs(
            requested_names=requested_names,
            view_spec_proto=view_spec_proto,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            view_id_hint=view_id_hint,
            view_subset_hash=view_subset_hash,
        )

    def _build_region_layout_selection(
        self,
        *,
        region_layout,
        view_spec_proto: common_pb2.ViewSpec | None,
    ) -> common_pb2.ArtifactSelection:
        artifact_id = self._ensure_identified()
        canonical_index_bytes = self._canonical_index_bytes
        runtime = self._runtime_if_available()
        if canonical_index_bytes is None and runtime is not None:
            canonical_index_bytes = runtime.ensure_client().get_artifact_index_by_id(
                artifact_id
            )
        if canonical_index_bytes is None:
            raise ArtifactError(
                "Canonical index bytes missing while building region selection",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        subset_hash = bytes(region_layout.view_subset_hash or b"")
        selection_names: tuple[str, ...] = (
            tuple(region_layout.selection_names) if subset_hash else ()
        )
        layout_index_bytes: bytes | None = None
        if (
            region_layout.layout.index_kind
            == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
        ):
            if region_layout.view_index_bytes:
                layout_index_bytes = bytes(region_layout.view_index_bytes)
            else:
                layout_index_bytes = compute_selected_index_bytes(
                    canonical_index_bytes=canonical_index_bytes,
                    view_spec=view_spec_proto,
                    tensor_names=selection_names,
                )

        selection_view_id = str(region_layout.view_id or "")
        if selection_view_id.startswith("mapped:v1:") and view_spec_proto is not None:
            # For mapped materialization the selection still refers to the source
            # artifact view. The mapped target view id belongs to the target
            # layout/publication contract, not the source selection identity.
            selection_view_id = ""

        try:
            return resolve_artifact_selection(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                view_spec=view_spec_proto,
                tensor_names=selection_names,
                view_subset_hash=subset_hash if subset_hash else None,
                view_id=selection_view_id,
                view_index_hint=layout_index_bytes,
                allow_view_id_without_spec=bool(
                    selection_view_id and view_spec_proto is None
                ),
            ).proto
        except ValueError as exc:
            raise ArtifactError(
                str(exc),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ) from exc

    def _runtime_if_available(self) -> StoreRuntimeContext | None:
        store = self._store_ref() if self._store_ref is not None else None
        if store is None or store.closed:
            return None
        return store._runtime

    def _ensure_identified(self) -> str:
        if self._artifact_id:
            return self._artifact_id
        runtime = self._runtime_if_available()
        if runtime is None:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        with self._lock:
            if self._artifact_id:
                return self._artifact_id
            if self._key_hint:
                resolved_mapping = runtime.resolve_key_mapping_cached(
                    key=self._key_hint
                )
                if isinstance(resolved_mapping, tuple):
                    artifact_id = resolved_mapping[0]
                    generation = (
                        int(resolved_mapping[2])
                        if len(resolved_mapping) > 2 and resolved_mapping[2] is not None
                        else None
                    )
                else:
                    artifact_id = getattr(resolved_mapping, "artifact_id", None)
                    generation = getattr(resolved_mapping, "generation", None)
                if not artifact_id:
                    raise ArtifactError(
                        f"Artifact key '{self._key_hint}' is not mapped",
                        status_code="NOT_FOUND",
                        retryable=False,
                    )
                self._artifact_id = artifact_id
                if self._key_generation is None and generation is not None:
                    self._key_generation = int(generation)
                return artifact_id
            raise ArtifactError(
                "Artifact handle missing identity",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

    def _ensure_metadata(self) -> CanonicalIndex:
        if (
            self._canonical_index is not None
            and self._canonical_index_bytes is not None
        ):
            return self._canonical_index
        runtime = self._runtime_if_available()
        if runtime is None:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        artifact_id = self._ensure_identified()
        cache_generation: int | None = None
        generation_hint: int | None = None
        force_remote_fetch = False
        with self._lock:
            if (
                self._canonical_index is not None
                and self._canonical_index_bytes is not None
            ):
                return self._canonical_index
            cached = runtime.get_artifact_index_cached(artifact_id)
            if cached:
                has_generation_mismatch = bool(
                    self._generation is not None
                    and cached.generation is not None
                    and cached.generation != self._generation
                )
                force_remote_fetch = bool(has_generation_mismatch)
                if force_remote_fetch:
                    runtime.invalidate_artifact(
                        artifact_id,
                        reason="generation_mismatch",
                    )
                else:
                    self._hydrate_from_cache_entry(cached)
                    assert self._canonical_index is not None
                    return self._canonical_index
            generation_hint = self._generation
        try:
            canonical_index_bytes = runtime.ensure_client().get_artifact_index_by_id(
                artifact_id
            )
        except Exception as exc:  # noqa: BLE001
            raise_mapped_materialization_error(exc)
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        with self._lock:
            if (
                self._canonical_index is not None
                and self._canonical_index_bytes is not None
            ):
                result_index = self._canonical_index
                cache_bytes = self._canonical_index_bytes or canonical_index_bytes
                cache_generation = self._generation
            else:
                generation_value = self._generation
                if generation_value is None:
                    generation_value = generation_hint
                self._set_metadata(
                    canonical_index_bytes,
                    canonical_index,
                    generation=generation_value,
                )
                result_index = canonical_index
                cache_bytes = canonical_index_bytes
                cache_generation = self._generation
        runtime.cache_artifact_index(
            ArtifactCacheEntry(
                artifact_id=artifact_id,
                canonical_index_bytes=cache_bytes,
                parsed_index=result_index,
                generation=cache_generation,
                expires_at=time.monotonic(),
            )
        )
        return result_index

    def _effective_index(self) -> CanonicalIndex:
        base_index = self._ensure_metadata()
        view_metadata = self._ensure_view_metadata_cache(require_selected_index=True)
        if view_metadata is not None and view_metadata.selected_index is not None:
            return view_metadata.selected_index
        return base_index

    def _ensure_view_metadata_cache(
        self,
        *,
        require_index_bytes: bool = False,
        require_view_hash: bool = False,
        require_view_id: bool = False,
        require_selected_index: bool = False,
        timing_out: dict[str, float] | None = None,
    ) -> ViewMetadataCache | None:
        cache = self._view_metadata
        if cache is None:
            return None
        needs_index_bytes = require_index_bytes and not cache.view_index_bytes
        needs_view_hash = require_view_hash and not cache.view_data_hash
        needs_view_id = (
            require_view_id
            and not cache.view_id
            and self._view_spec is not None
            and not self._view_spec.is_identity
        )
        needs_selected_index = require_selected_index and cache.selected_index is None
        if not (
            needs_index_bytes
            or needs_view_hash
            or needs_view_id
            or needs_selected_index
        ):
            return cache

        canonical_index = self._ensure_metadata()
        canonical_index_bytes = self._canonical_index_bytes
        if canonical_index_bytes is None:
            canonical_index_bytes = canonical_index_to_bytes(canonical_index)

        with self._lock:
            cache = self._view_metadata
            if cache is None:
                return None
            view_index_bytes = cache.view_index_bytes
            view_data_hash = cache.view_data_hash
            view_id = cache.view_id
            selected_index = cache.selected_index

            if require_index_bytes and not view_index_bytes:
                view_spec_proto = self._view_spec.proto if self._view_spec else None
                compute_index_start = time.perf_counter()
                view_index_bytes = compute_selected_index_bytes(
                    canonical_index_bytes=canonical_index_bytes,
                    view_spec=view_spec_proto,
                    tensor_names=cache.tensor_names,
                )
                if timing_out is not None:
                    timing_out["index_bytes_sec"] = (
                        time.perf_counter() - compute_index_start
                    )
            if require_selected_index and selected_index is None:
                if not view_index_bytes:
                    view_spec_proto = self._view_spec.proto if self._view_spec else None
                    compute_index_start = time.perf_counter()
                    view_index_bytes = compute_selected_index_bytes(
                        canonical_index_bytes=canonical_index_bytes,
                        view_spec=view_spec_proto,
                        tensor_names=cache.tensor_names,
                    )
                    if timing_out is not None:
                        timing_out["index_bytes_sec"] = timing_out.get(
                            "index_bytes_sec", 0.0
                        ) + (time.perf_counter() - compute_index_start)
                parse_index_start = time.perf_counter()
                selected_index = canonical_index_from_bytes(view_index_bytes)
                if timing_out is not None:
                    timing_out["index_parse_sec"] = (
                        time.perf_counter() - parse_index_start
                    )
            if require_view_hash and not view_data_hash:
                hash_start = time.perf_counter()
                view_data_hash = ViewSpecComposer.hash_view_spec(
                    self._view_spec,
                    subset=cache.tensor_names,
                )
                if timing_out is not None:
                    timing_out["view_hash_sec"] = time.perf_counter() - hash_start
            if (
                require_view_id
                and not view_id
                and (
                    (self._view_spec is not None and not self._view_spec.is_identity)
                    or cache.tensor_names
                )
            ):
                view_proto = (
                    self._view_spec.proto
                    if self._view_spec is not None and not self._view_spec.is_identity
                    else _build_subset_view_spec_proto(
                        canonical_index=canonical_index,
                        tensor_names=cache.tensor_names,
                    )
                )
                if view_proto is None:
                    raise ArtifactError(
                        "View spec proto missing while computing view_id",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
                view_id_start = time.perf_counter()
                view_id = compute_view_id(view_proto, canonical_index_bytes)
                if timing_out is not None:
                    timing_out["view_id_sec"] = time.perf_counter() - view_id_start

            if (
                view_index_bytes == cache.view_index_bytes
                and view_data_hash == cache.view_data_hash
                and view_id == cache.view_id
                and selected_index == cache.selected_index
            ):
                return cache

            cache = ViewMetadataCache(
                view_id=view_id,
                view_index_bytes=view_index_bytes,
                view_data_hash=view_data_hash,
                tensor_names=cache.tensor_names,
                nbytes=(
                    int(selected_index.total_size_bytes)
                    if selected_index is not None
                    else cache.nbytes
                ),
                selected_index=selected_index,
            )
            self._view_metadata = cache
            if selected_index is not None:
                self._tensor_metas = {
                    entry.name: _meta_from_entry(entry)
                    for entry in selected_index.entries
                }
            return cache

    @staticmethod
    def _normalize_view_inputs(
        *,
        slices: Mapping[str, Sequence[object]] | None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None,
    ) -> tuple[
        Mapping[str, SliceSpec] | None, Mapping[str, Sequence[tuple[int, int]]] | None
    ]:
        if slices is not None and not isinstance(slices, Mapping):
            raise ArtifactError(
                "Slice spec must be a mapping of tensor name to slice",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if transpose is not None and not isinstance(transpose, Mapping):
            raise ArtifactError(
                "Transpose spec must be a mapping of tensor name to dim pairs",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        typed_slices: dict[str, SliceSpec] | None = None
        if slices:
            typed_slices = {}
            for name, spec_seq in slices.items():
                if not isinstance(spec_seq, Sequence) or not spec_seq:
                    raise ArtifactError(
                        f"Slice spec for '{name}' must be a non-empty sequence",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                typed_slices[name] = _coerce_slice_spec(spec_seq)

        if transpose:
            for name, ops in transpose.items():
                if not isinstance(ops, Sequence) or not ops:
                    raise ArtifactError(
                        f"Transpose spec for '{name}' must be a non-empty sequence",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
        return typed_slices, transpose

    def _derive_view(
        self,
        *,
        slices: Mapping[str, Sequence[object]] | None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None,
        subset: Sequence[str] | None,
        composer: ViewSpecComposer,
    ) -> Artifact:
        if self._released:
            raise ArtifactError(
                "Artifact handle is released",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        self._ensure_metadata()
        typed_slices, normalized_transpose = self._normalize_view_inputs(
            slices=slices,
            transpose=transpose,
        )
        if (
            subset is None
            and not typed_slices
            and not normalized_transpose
            and self._view_metadata is not None
        ):
            return Artifact(
                store_ref=self._store_ref,
                artifact_id=self._artifact_id,
                key=self._key_hint,
                canonical_index_bytes=self._canonical_index_bytes,
                canonical_index=self._canonical_index,
                generation=self._generation,
                view_spec=self._view_spec,
                view_metadata=self._view_metadata,
                view_depth=self._view_depth,
                source_subject=self._source_subject,
            )
        base_index = self._effective_index()
        entry_shapes = {entry.name: tuple(entry.shape) for entry in base_index.entries}
        child_spec: ViewSpecBuildResult | None = None
        if typed_slices or normalized_transpose:
            child_spec = build_view_spec(
                entry_shapes=entry_shapes,
                slices=typed_slices,
                transpose=normalized_transpose,
            )
        composed_spec, view_cache, depth = composer.compose(
            canonical_index=base_index,
            identity_index_bytes=self._canonical_index_bytes,
            parent_spec=self._view_spec,
            child_spec=child_spec,
            parent_depth=self._view_depth,
            subset_names=subset,
        )
        return Artifact(
            store_ref=self._store_ref,
            artifact_id=self._artifact_id,
            key=self._key_hint,
            canonical_index_bytes=self._canonical_index_bytes,
            canonical_index=self._canonical_index,
            generation=self._generation,
            key_generation=self._key_generation,
            view_spec=composed_spec,
            view_metadata=view_cache,
            view_depth=depth,
            source_subject=self._source_subject,
        )

    def _hydrate_from_cache_entry(self, entry: ArtifactCacheEntry) -> None:
        self._set_metadata(
            entry.canonical_index_bytes,
            entry.parsed_index,
            generation=entry.generation,
        )

    def _set_metadata(
        self,
        canonical_index_bytes: bytes,
        canonical_index: CanonicalIndex,
        *,
        generation: int | None,
    ) -> None:
        self._canonical_index_bytes = canonical_index_bytes
        self._canonical_index = canonical_index
        self._generation = generation
        effective_index = (
            self._view_metadata.selected_index
            if self._view_metadata is not None
            else canonical_index
        )
        if effective_index is not None:
            self._tensor_metas = {
                entry.name: _meta_from_entry(entry) for entry in effective_index.entries
            }

    def _update_metadata_from_payload(
        self, payload, runtime: StoreRuntimeContext
    ) -> None:
        artifact_id = self._artifact_id
        if not artifact_id:
            return
        canonical_index_bytes = getattr(payload, "canonical_index_bytes", b"") or b""
        view_index_bytes = getattr(payload, "view_index_bytes", b"") or b""
        generation = getattr(payload, "generation", None)

        if canonical_index_bytes:
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            with self._lock:
                if self._canonical_index is None:
                    self._set_metadata(
                        canonical_index_bytes,
                        canonical_index,
                        generation=generation,
                    )
            runtime.cache_artifact_index(
                ArtifactCacheEntry(
                    artifact_id=artifact_id,
                    canonical_index_bytes=canonical_index_bytes,
                    parsed_index=canonical_index,
                    generation=generation,
                    expires_at=time.monotonic(),
                )
            )

        if view_index_bytes:
            view_index = canonical_index_from_bytes(view_index_bytes)
            view_hash = getattr(payload, "view_data_hash", None)
            subset_names = tuple(entry.name for entry in view_index.entries)
            resolved_view_id: str | None = None
            if self._view_spec is not None and not self._view_spec.is_identity:
                view_proto = self._view_spec.proto
                if view_proto is None:
                    raise ArtifactError(
                        "View spec proto missing while resolving view_id",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
                resolved_view_id = compute_view_id(view_proto, canonical_index_bytes)
            view_cache = ViewMetadataCache(
                view_id=str(resolved_view_id or ""),
                view_index_bytes=view_index_bytes,
                view_data_hash=(str(view_hash) if view_hash else None),
                tensor_names=subset_names,
                nbytes=sum(entry.size_bytes for entry in view_index.entries),
                selected_index=view_index,
            )
            with self._lock:
                self._view_metadata = view_cache
                self._view_depth = max(self._view_depth, 1)
                self._tensor_metas = {
                    entry.name: _meta_from_entry(entry) for entry in view_index.entries
                }

    @staticmethod
    def _validate_tensor_names(
        canonical_index: CanonicalIndex, names: Sequence[str] | None
    ) -> tuple[str, ...] | None:
        if names is None:
            return None
        available = {entry.name for entry in canonical_index.entries}
        ordered: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            if name not in available:
                raise ArtifactError(
                    f"Tensor '{name}' not found in artifact",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            seen.add(name)
            ordered.append(name)
        return tuple(ordered)


__all__ = [
    "Artifact",
    "ArtifactDescriptor",
    "MaterializationDiagnostics",
    "PlacementPin",
    "PrefetchedReplica",
    "TensorMeta",
    "TensorDictMaterializationResult",
]
