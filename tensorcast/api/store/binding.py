#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import json
import threading
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from tensorcast.api.context import CallContext
from tensorcast.api.operation import Operation, OperationStatus, PollingOperation
from tensorcast.api.store.binding_state import clone_selection
from tensorcast.api.store.inplace_slot import InplaceSlot, _ctx_timeout_s
from tensorcast.api.store.owned_binding_slot import OwnedBindingSlot
from tensorcast.api.store.retry import raise_mapped_registration_error
from tensorcast.api.store.types import ArtifactError
from tensorcast.common.identity import ArtifactIdKind, infer_artifact_id_kind
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    ArtifactDescriptor as TypedArtifactDescriptor,
)
from tensorcast.types import (
    AssemblyAttemptRef,
    BindingValueRef,
    ExecutionDiagnostics,
    PartialSealResult,
    PublicDiskSourceHandle,
    ServingRuntimePolicyInput,
    SourceBoundPlanDiagnostics,
    coerce_serving_runtime_policy,
)

if TYPE_CHECKING:
    from tensorcast.api._config import GetArtifactOptions
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.owned_binding_layout import BindingLayout
    from tensorcast.api.store.runtime import StoreRuntimeContext
    from tensorcast.daemon_ctl import DaemonCtl
    from tensorcast.proto.common.v1 import common_pb2


_TRANSPORT_GROUP_OPID_MARKER = "#tcg:"
_TRANSPORT_GROUP_KIND_TAG = "tc.transport.group.kind"
_TRANSPORT_GROUP_ID_TAG = "tc.transport.group.id"
_TRANSPORT_GROUP_TOTAL_PARTS_TAG = "tc.transport.group.total_parts"
_TRANSPORT_GROUP_PART_ID_TAG = "tc.transport.group.part_id"
_TRANSPORT_GROUP_PRIORITY_TAG = "tc.transport.group.priority"
_TRANSPORT_GROUP_EPOCH_TAG = "tc.transport.group.epoch"
_TRANSPORT_REQUEST_ID_TAG = "tc.transport.request_id"
_COLLECTIVE_GROUP_ID_OPID_KEY = "clid"
_COLLECTIVE_GROUP_WORLD_SIZE_OPID_KEY = "clws"
_COLLECTIVE_GROUP_RANK_OPID_KEY = "clrk"
_CANONICAL_FULL_CONTRIBUTION_VIEW_ID = "__canonical_full__"
_ALLOWED_OPERATION_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
)


def _artifact_id_kind_from_proto(kind: int, artifact_id: str) -> ArtifactIdKind:
    if kind == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2:
        return ArtifactIdKind.MI2
    if kind == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID:
        return ArtifactIdKind.CGID
    inferred = infer_artifact_id_kind(artifact_id)
    return inferred or ArtifactIdKind.MI2


def _typed_descriptor_from_proto(
    descriptor: common_pb2.ArtifactDescriptor | None,
) -> TypedArtifactDescriptor:
    artifact_id = (
        "" if descriptor is None else str(getattr(descriptor, "artifact_id", "") or "")
    )
    if not artifact_id:
        raise ArtifactError(
            "PromoteBindingCurrentValue returned empty artifact_descriptor",
            status_code="DATA_LOSS",
            retryable=False,
        )
    return TypedArtifactDescriptor(
        artifact_id=artifact_id,
        index_multihash=(
            None
            if descriptor is None
            else str(getattr(descriptor, "index_multihash", "") or "") or None
        ),
        data_multihash=(
            None
            if descriptor is None
            else str(getattr(descriptor, "data_multihash", "") or "") or None
        ),
        schema_version=(
            None
            if descriptor is None
            else str(getattr(descriptor, "schema_version", "") or "") or None
        ),
        encoding=None
        if descriptor is None
        else str(getattr(descriptor, "encoding", "") or "") or None,
        total_size=0
        if descriptor is None
        else int(getattr(descriptor, "total_size", 0) or 0),
        id_kind=_artifact_id_kind_from_proto(
            int(common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_UNSPECIFIED)
            if descriptor is None
            else int(getattr(descriptor, "id_kind", 0) or 0),
            artifact_id,
        ),
    )


def _sanitize_operation_token(value: str | None) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ""
    return "".join(ch if ch in _ALLOWED_OPERATION_TOKEN_CHARS else "_" for ch in raw)


def _read_context_tag_str(
    tags: Mapping[str, object] | None,
    key: str,
) -> str:
    if not tags:
        return ""
    value = tags.get(key)
    if value is None:
        return ""
    return _sanitize_operation_token(str(value))


def _read_context_tag_int(
    tags: Mapping[str, object] | None,
    key: str,
    *,
    default: int,
) -> int:
    if not tags:
        return default
    value = tags.get(key)
    if value is None:
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _build_transport_operation_id(
    *,
    base_operation_id: str,
    ctx: CallContext | None,
    include_collective: bool = True,
) -> str:
    if ctx is None:
        return base_operation_id

    tags = ctx.tags
    group_kind = _read_context_tag_str(tags, _TRANSPORT_GROUP_KIND_TAG)
    group_id = _read_context_tag_str(tags, _TRANSPORT_GROUP_ID_TAG)
    part_id = _read_context_tag_str(tags, _TRANSPORT_GROUP_PART_ID_TAG)
    total_parts = _read_context_tag_int(
        tags,
        _TRANSPORT_GROUP_TOTAL_PARTS_TAG,
        default=0,
    )
    priority = _read_context_tag_int(
        tags,
        _TRANSPORT_GROUP_PRIORITY_TAG,
        default=0,
    )
    epoch = _read_context_tag_int(
        tags,
        _TRANSPORT_GROUP_EPOCH_TAG,
        default=0,
    )
    request_id = _read_context_tag_str(tags, _TRANSPORT_REQUEST_ID_TAG)
    collective_group_id = ""
    collective_world_size = 0
    collective_rank = 0
    if include_collective:
        collective = ctx.collective
        if collective is not None:
            collective_group_id = _sanitize_operation_token(collective.group_id)
            try:
                collective_world_size = int(collective.world_size)
                collective_rank = int(collective.rank)
            except (TypeError, ValueError):
                collective_group_id = ""
                collective_world_size = 0
                collective_rank = 0

    metadata_parts: list[str] = []
    if group_kind and group_id and part_id and total_parts > 0:
        if not request_id:
            request_id = _sanitize_operation_token(f"{group_id}:{part_id}")
        metadata_parts.extend(
            [
                f"kind={group_kind}",
                f"gid={group_id}",
                f"tot={int(total_parts)}",
                f"part={part_id}",
                f"pri={int(priority)}",
                f"ep={int(epoch)}",
            ]
        )
    if (
        collective_group_id
        and collective_world_size > 1
        and 0 <= collective_rank < collective_world_size
    ):
        metadata_parts.extend(
            [
                f"{_COLLECTIVE_GROUP_ID_OPID_KEY}={collective_group_id}",
                f"{_COLLECTIVE_GROUP_WORLD_SIZE_OPID_KEY}={int(collective_world_size)}",
                f"{_COLLECTIVE_GROUP_RANK_OPID_KEY}={int(collective_rank)}",
            ]
        )

    if request_id:
        metadata_parts.append(f"rid={request_id}")

    if metadata_parts:
        return (
            f"{base_operation_id}{_TRANSPORT_GROUP_OPID_MARKER}"
            f"{';'.join(metadata_parts)}"
        )

    return base_operation_id


def _clone_view_spec(
    view_spec: common_pb2.ViewSpec | None,
) -> common_pb2.ViewSpec | None:
    if view_spec is None or not view_spec.tensors:
        return None
    cloned = common_pb2.ViewSpec()
    cloned.CopyFrom(view_spec)
    return cloned


def _slot_contribution_view_spec(
    slot: InplaceSlot | OwnedBindingSlot,
) -> common_pb2.ViewSpec | None:
    view_spec = getattr(slot, "_view_spec", None)
    cloned = _clone_view_spec(view_spec)
    if cloned is not None:
        return cloned
    selection = getattr(slot, "contribution_selection", None)
    if selection is not None and selection.HasField("view_spec"):
        return _clone_view_spec(selection.view_spec)
    live_selection = getattr(slot, "selection", None)
    if live_selection is not None and live_selection.HasField("view_spec"):
        return _clone_view_spec(live_selection.view_spec)
    return None


def _slot_contribution_selection(
    slot: InplaceSlot | OwnedBindingSlot,
) -> common_pb2.ArtifactSelection | None:
    selection = getattr(slot, "contribution_selection", None)
    if selection is not None:
        return clone_selection(selection)
    live_selection = getattr(slot, "selection", None)
    if live_selection is not None:
        return clone_selection(live_selection)
    return None


def _slot_contribution_view_id_hint(
    slot: InplaceSlot | OwnedBindingSlot,
) -> str | None:
    selection = _slot_contribution_selection(slot)
    if selection is not None and selection.view_id:
        return str(selection.view_id)
    slot_view_id = getattr(slot, "_view_id", None)
    if slot_view_id:
        return str(slot_view_id)
    return None


def _compute_coverage_plan_hash(
    *,
    contribution_kind: str,
    binding: "Binding",
    view_id: str | None,
    selection: common_pb2.ArtifactSelection | None,
) -> str:
    payload = {
        "binding_layout_id": binding.binding_layout_id,
        "contribution_kind": contribution_kind,
        "view_id": view_id or "",
        "source_artifact_id": (
            str(selection.artifact_id)
            if selection is not None and selection.artifact_id
            else ""
        ),
        "logical_layout_hash": (
            selection.logical_layout_hash.hex()
            if selection is not None and selection.logical_layout_hash
            else ""
        ),
        "selection_hash": (
            selection.selection_hash.hex()
            if selection is not None and selection.selection_hash
            else ""
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"bcp1:{digest}"


def _wait_for_events(wait_events: tuple[object, ...] | list[object] | None) -> None:
    if not wait_events:
        return
    for idx, event in enumerate(wait_events):
        if event is None:
            raise ArtifactError(
                f"wait_events[{idx}] must not be None",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        synchronize = getattr(event, "synchronize", None)
        if not callable(synchronize):
            raise ArtifactError(
                f"wait_events[{idx}] must provide synchronize()",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        try:
            synchronize()
        except ArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                f"wait_events[{idx}] synchronize() failed: {exc}",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            ) from exc


class _PublishedLeaseKeepalive:
    def __init__(self, client: "DaemonCtl", ttl_ms: int) -> None:
        self._client = client
        self._ttl_ms = int(ttl_ms)
        self._lease_id: str | None = None
        self._epoch = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self, lease_id: str) -> None:
        if not lease_id or self._ttl_ms <= 0:
            return
        with self._lock:
            if self._lease_id == lease_id and self._thread and self._thread.is_alive():
                return
            self._stop_locked()
            self._epoch = 0
            self._lease_id = lease_id
            self._stop.clear()

            interval = max(1.0, self._ttl_ms / 2000.0)

            def _run() -> None:
                import random

                while not self._stop.wait(interval * (0.9 + 0.2 * random.random())):
                    try:
                        self._client.keep_alive_registered_artifact(
                            lease_id, self._ttl_ms, self._epoch
                        )
                        self._epoch += 1
                    except Exception:  # noqa: BLE001
                        continue

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._thread = t

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        thread = self._thread
        self._thread = None
        self._lease_id = None
        self._epoch = 0
        self._stop.set()
        if thread and thread.is_alive():
            thread.join(timeout=1.0)


@dataclass(frozen=True, slots=True)
class BindingUpdateEpoch:
    binding_id: str
    update_epoch: str

    def __str__(self) -> str:
        return self.update_epoch


@dataclass(frozen=True, slots=True)
class SealedBindingValue:
    binding_id: str
    binding_layout_id: str
    binding_value_id: str
    seal_generation: int
    source_artifact_id: str | None
    selection: "common_pb2.ArtifactSelection | None"
    is_artifact_backed: bool
    _binding_ref: weakref.ReferenceType["Binding"] = field(
        repr=False,
        compare=False,
    )
    verification_state: int = (
        store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_UNSPECIFIED
    )
    verification_job_id: str | None = None
    source_artifact_ref: str | None = None
    local_serving_ref: str | None = None
    serving_artifact_id: str | None = None
    verification_failure_reason: str | None = None

    @property
    def artifact_id(self) -> str | None:
        return self.source_artifact_id

    @property
    def is_verified(self) -> bool:
        return (
            self.verification_state
            == store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_VERIFIED
        )

    @property
    def is_verification_pending(self) -> bool:
        return (
            self.verification_state
            == store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_PENDING
        )

    @property
    def is_current(self) -> bool:
        binding = self._binding_ref()
        current_value = None if binding is None else binding.current_value
        return (
            current_value is not None
            and current_value.binding_value_id == self.binding_value_id
            and current_value.seal_generation == self.seal_generation
        )

    @property
    def is_published(self) -> bool:
        binding = self._binding_ref()
        current_value = None if binding is None else binding.current_value
        return (
            self.is_current
            and current_value is not None
            and binding is not None
            and binding._slot.published_lease_id is not None
        )

    def publish_replica(self, *, ctx: CallContext | None = None) -> None:
        binding = self._require_current_binding()
        binding.publish_replica(ctx=ctx)

    def publish_replica_operation(
        self, *, ctx: CallContext | None = None
    ) -> "Operation[SealedBindingValue]":
        binding = self._require_current_binding()
        return binding.publish_replica_operation(ctx=ctx)

    def promote_serving_artifact(
        self,
        *,
        ctx: CallContext | None = None,
    ) -> TypedArtifactDescriptor:
        binding = self._require_current_binding()
        return binding.promote_current_value(
            binding_value_id=self.binding_value_id,
            ctx=ctx,
        )

    def to_binding_value_ref(self) -> BindingValueRef:
        return BindingValueRef(
            binding_id=str(self.binding_id),
            binding_layout_id=str(self.binding_layout_id),
            binding_value_id=str(self.binding_value_id),
            seal_generation=int(self.seal_generation),
        )

    def activate_key(
        self,
        key: str,
        *,
        expected_active_artifact_id: str | None = None,
        expected_active_generation: int | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        binding = self._require_current_binding()
        binding._activate_key(
            key,
            expected_active_artifact_id=expected_active_artifact_id,
            expected_active_generation=expected_active_generation,
            operation_id=_build_transport_operation_id(
                base_operation_id=uuid.uuid4().hex,
                ctx=ctx,
            ),
            ctx=ctx,
        )

    def contribute_to_assembly(
        self,
        *,
        attempt: AssemblyAttemptRef,
        ctx: CallContext | None = None,
    ) -> PartialSealResult:
        binding = self._require_current_binding()
        slot = binding._slot
        selection = _slot_contribution_selection(slot)
        view_spec = _slot_contribution_view_spec(slot)
        view_id_hint = _slot_contribution_view_id_hint(slot)
        piece_partial = (
            view_spec is not None
            or view_id_hint is not None
            or (selection is not None and bool(selection.tensor_names))
        )
        if piece_partial:
            contribution_kind = "piece_partial"
            coverage_plan_hash = _compute_coverage_plan_hash(
                contribution_kind=contribution_kind,
                binding=binding,
                view_id=view_id_hint,
                selection=selection,
            )
            submit_kind = store_daemon_pb2.BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL
        else:
            contribution_kind = "canonical_full"
            coverage_plan_hash = _compute_coverage_plan_hash(
                contribution_kind=contribution_kind,
                binding=binding,
                view_id=_CANONICAL_FULL_CONTRIBUTION_VIEW_ID,
                selection=selection,
            )
            submit_kind = store_daemon_pb2.BINDING_CONTRIBUTION_KIND_CANONICAL_FULL

        timeout_s = _ctx_timeout_s(ctx)
        try:
            response = binding._runtime.ensure_client().submit_binding_contribution(
                attempt_id=attempt.attempt_id,
                workspace_assembly_id=attempt.workspace_assembly_id,
                binding_id=self.binding_id,
                binding_value_id=self.binding_value_id,
                coverage_plan_hash=coverage_plan_hash,
                contribution_kind=submit_kind,
                coordinator_operation_id=attempt.coordinator_operation_id,
                coordinator_generation=attempt.coordinator_generation,
                attempt_intent_digest=attempt.attempt_intent_digest,
                view_id=view_id_hint,
                timeout_s=timeout_s if timeout_s is not None else 30.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise_mapped_registration_error(exc)
        returned_slot_id = str(getattr(response, "slot_id", "") or "") or None
        returned_view_id = (returned_slot_id if piece_partial else None) or view_id_hint
        return PartialSealResult(
            attempt_id=attempt.attempt_id,
            workspace_assembly_id=attempt.workspace_assembly_id,
            slot_id=returned_slot_id,
            binding_id=self.binding_id,
            binding_value_id=self.binding_value_id,
            contribution_kind=contribution_kind,
            view_id=returned_view_id,
            coverage_plan_hash=coverage_plan_hash,
            accepted=bool(getattr(response, "accepted", False)),
            already_exists=bool(getattr(response, "already_exists", False)),
        )

    def _require_current_binding(self) -> "Binding":
        binding = self._binding_ref()
        if binding is None:
            raise ArtifactError(
                "Binding is no longer available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self.is_current:
            raise ArtifactError(
                "SealedBindingValue is no longer current for this binding",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return binding


class Binding:
    """Stable, client-owned CUDA layout that can be refilled in-place."""

    def __init__(
        self,
        slot: InplaceSlot | OwnedBindingSlot,
        *,
        publish: bool = False,
        ctx: CallContext | None = None,
    ) -> None:
        self._slot = slot
        runtime: StoreRuntimeContext = slot._runtime
        self._runtime = runtime
        self._publish_ttl_ms = 0
        self._keepalive = _PublishedLeaseKeepalive(
            runtime.ensure_client(), self._publish_ttl_ms
        )
        if publish:
            self.publish_replica(ctx=ctx)

    def __enter__(self) -> "Binding":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._slot.tensors

    @property
    def binding_id(self) -> str:
        return self._slot.binding_id

    @property
    def binding_layout_id(self) -> str:
        return self._slot.binding_layout_id

    @property
    def layout(self) -> "BindingLayout":
        return self._slot.layout

    @property
    def current_value(self) -> SealedBindingValue | None:
        metadata = self._slot.current_value_metadata
        if metadata is None:
            return None
        return SealedBindingValue(
            binding_id=metadata.binding_id,
            binding_layout_id=metadata.binding_layout_id,
            binding_value_id=metadata.binding_value_id,
            seal_generation=metadata.seal_generation,
            source_artifact_id=metadata.source_artifact_id,
            selection=clone_selection(metadata.selection),
            is_artifact_backed=metadata.is_artifact_backed,
            verification_state=metadata.verification_state,
            verification_job_id=metadata.verification_job_id,
            source_artifact_ref=metadata.source_artifact_ref,
            local_serving_ref=metadata.local_serving_ref,
            serving_artifact_id=metadata.serving_artifact_id,
            verification_failure_reason=metadata.verification_failure_reason,
            _binding_ref=weakref.ref(self),
        )

    @property
    def artifact_id(self) -> str | None:
        current_value = self.current_value
        if current_value is None or not current_value.is_artifact_backed:
            return None
        return current_value.source_artifact_id

    @property
    def selection(self) -> "common_pb2.ArtifactSelection | None":
        current_value = self.current_value
        if current_value is None or not current_value.is_artifact_backed:
            return None
        return current_value.selection

    @property
    def last_execution_diagnostics(self) -> ExecutionDiagnostics | None:
        diagnostics = getattr(self._slot, "last_execution_diagnostics", None)
        return diagnostics if isinstance(diagnostics, ExecutionDiagnostics) else None

    @property
    def last_source_bound_plan_diagnostics(self) -> SourceBoundPlanDiagnostics | None:
        diagnostics = getattr(self._slot, "last_source_bound_plan_diagnostics", None)
        return (
            diagnostics if isinstance(diagnostics, SourceBoundPlanDiagnostics) else None
        )

    def swap(
        self,
        artifact: "Artifact | str",
        *,
        options: "GetArtifactOptions | None" = None,
        publish: bool = False,
        serving_runtime_policy: ServingRuntimePolicyInput | None = None,
        activate_key: str | None = None,
        expected_active_artifact_id: str | None = None,
        expected_active_generation: int | None = None,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> SealedBindingValue:
        include_collective = not isinstance(self._slot, OwnedBindingSlot)
        operation_id = _build_transport_operation_id(
            base_operation_id=uuid.uuid4().hex,
            ctx=ctx,
            include_collective=include_collective,
        )
        self._stop_keepalive()
        try:
            self._slot.swap(
                artifact,
                options=options,
                publish=publish,
                serving_runtime_policy=coerce_serving_runtime_policy(
                    serving_runtime_policy
                ),
                wait=wait,
                drain_timeout_s=drain_timeout_s,
                ctx=ctx,
                operation_id=operation_id,
                publish_ttl_ms=self._publish_ttl_ms if publish else None,
            )
        except Exception:
            if self._slot.published_lease_id:
                self._start_keepalive()
            raise
        if publish:
            self._start_keepalive()
        if activate_key:
            self._activate_key(
                activate_key,
                expected_active_artifact_id=expected_active_artifact_id,
                expected_active_generation=expected_active_generation,
                operation_id=operation_id,
                ctx=ctx,
            )
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "swap() completed without a current sealed value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return current_value

    def realize_from(
        self,
        artifact: "Artifact | PublicDiskSourceHandle | str",
        *,
        realization_plan: object,
        options: "GetArtifactOptions | None" = None,
        ctx: CallContext | None = None,
    ) -> BindingUpdateEpoch:
        operation_id = _build_transport_operation_id(
            base_operation_id=uuid.uuid4().hex,
            ctx=ctx,
            include_collective=False,
        )
        self._stop_keepalive()
        realize = getattr(self._slot, "realize_from", None)
        if not callable(realize):
            raise ArtifactError(
                "realize_from() requires a daemon-owned binding",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        update_epoch = realize(
            artifact,
            realization_plan=realization_plan,
            options=options,
            ctx=ctx,
            operation_id=operation_id,
        )
        if not isinstance(update_epoch, BindingUpdateEpoch):
            raise ArtifactError(
                "realize_from() completed without a binding update epoch",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return update_epoch

    def begin_update(
        self,
        *,
        wait_events: tuple[object, ...] | list[object] | None = None,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> BindingUpdateEpoch:
        _wait_for_events(wait_events)
        self._stop_keepalive()
        return self._slot.begin_update(
            drain_timeout_s=drain_timeout_s,
            ctx=ctx,
        )

    def seal_current(
        self,
        *,
        update_epoch: BindingUpdateEpoch | str | int,
        wait_events: tuple[object, ...] | list[object] | None = None,
        ctx: CallContext | None = None,
    ) -> SealedBindingValue:
        _wait_for_events(wait_events)
        self._stop_keepalive()
        self._slot.seal_current(update_epoch=update_epoch, ctx=ctx)
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "seal_current() completed without a current sealed value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return current_value

    def freeze_current(
        self,
        *,
        update_epoch: BindingUpdateEpoch | str | int,
        source_artifact_ref: str | None = None,
        wait_events: tuple[object, ...] | list[object] | None = None,
        ctx: CallContext | None = None,
    ) -> SealedBindingValue:
        _wait_for_events(wait_events)
        self._stop_keepalive()
        freeze = getattr(self._slot, "freeze_current", None)
        if not callable(freeze):
            raise ArtifactError(
                "freeze_current() requires a daemon-owned binding",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        freeze(
            update_epoch=update_epoch,
            source_artifact_ref=source_artifact_ref,
            ctx=ctx,
        )
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "freeze_current() completed without a current local-ready value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return current_value

    def promote_current_value(
        self,
        *,
        binding_value_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> TypedArtifactDescriptor:
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "promote_current_value() requires a current sealed value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        resolved_binding_value_id = str(
            binding_value_id or current_value.binding_value_id
        )
        promote = getattr(self._slot, "promote_current_value", None)
        if not callable(promote):
            raise ArtifactError(
                "promote_current_value() requires a daemon-owned binding",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        response = promote(
            binding_value_id=resolved_binding_value_id,
            ctx=ctx,
        )
        return _typed_descriptor_from_proto(
            response.artifact_descriptor
            if hasattr(response, "artifact_descriptor")
            else None
        )

    def start_promote_current_value(
        self,
        *,
        binding_value_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> store_daemon_pb2.BindingPromotionStatus:
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "start_promote_current_value() requires a current value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        resolved_binding_value_id = str(
            binding_value_id or current_value.binding_value_id
        )
        start_promote = getattr(self._slot, "start_promote_current_value", None)
        if not callable(start_promote):
            raise ArtifactError(
                "start_promote_current_value() requires a daemon-owned binding",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return cast(
            store_daemon_pb2.BindingPromotionStatus,
            start_promote(
                binding_value_id=resolved_binding_value_id,
                ctx=ctx,
            ),
        )

    def get_promotion_status(
        self,
        *,
        verification_job_id: str | None = None,
        binding_value_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> store_daemon_pb2.BindingPromotionStatus:
        current_value = self.current_value
        resolved_binding_value_id = str(
            binding_value_id
            or (current_value.binding_value_id if current_value is not None else "")
        )
        get_status = getattr(self._slot, "get_promotion_status", None)
        if not callable(get_status):
            raise ArtifactError(
                "get_promotion_status() requires a daemon-owned binding",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return cast(
            store_daemon_pb2.BindingPromotionStatus,
            get_status(
                verification_job_id=verification_job_id,
                binding_value_id=resolved_binding_value_id,
                ctx=ctx,
            ),
        )

    def close(self) -> None:
        self._stop_keepalive()
        self._slot.close()

    def retire(
        self,
        *,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self._stop_keepalive()
        self._slot.retire(
            wait=True,
            drain_timeout_s=drain_timeout_s,
            ctx=ctx,
        )

    def publish_replica(
        self,
        *,
        ctx: CallContext | None = None,
    ) -> SealedBindingValue:
        self._slot.publish_replica(ttl_ms=self._publish_ttl_ms, ctx=ctx)
        self._start_keepalive()
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "publish_replica() requires a current sealed value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return current_value

    def publish_replica_operation(
        self,
        *,
        ctx: CallContext | None = None,
    ) -> "Operation[SealedBindingValue]":
        slot_op = self._slot.publish_replica_operation(
            ttl_ms=self._publish_ttl_ms,
            ctx=ctx,
        )

        def _status() -> OperationStatus:
            return slot_op.status()

        def _result() -> SealedBindingValue:
            _ = slot_op.wait()
            self._start_keepalive()
            current_value = self.current_value
            if current_value is None:
                raise ArtifactError(
                    "publish_replica_operation() requires a current sealed value",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            return current_value

        return PollingOperation(
            operation_id=slot_op.operation_id,
            status_fn=_status,
            result_fn=_result,
            cancel_fn=slot_op.cancel,
            ctx=ctx,
        )

    def _start_keepalive(self) -> None:
        lease_id = self._slot.published_lease_id
        if lease_id:
            self._keepalive.start(lease_id)

    def _stop_keepalive(self) -> None:
        self._keepalive.stop()

    def _activate_key(
        self,
        key: str,
        *,
        expected_active_artifact_id: str | None,
        expected_active_generation: int | None,
        operation_id: str,
        ctx: CallContext | None,
    ) -> None:
        if not key:
            raise ArtifactError(
                "activate_key must be non-empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        current_artifact_id = self._slot.artifact_id
        if not current_artifact_id:
            raise ArtifactError(
                "activate_key requires an artifact-backed current value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        result = self._runtime.ensure_client().swap_key_mapping(
            key=key,
            new_artifact_id=current_artifact_id,
            expected_artifact_id=expected_active_artifact_id,
            expected_generation=expected_active_generation,
            operation_id=operation_id,
            timeout_s=timeout_s if timeout_s is not None else 10.0,
        )
        if not result.ok:
            message = (
                f"Activation key '{key}' conflict: current artifact_id="
                f"{result.artifact_id or 'unknown'} generation={result.generation}"
            )
            raise ArtifactError(
                message,
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        self._runtime.invalidate_artifact(None, key=key, reason="activate_key")


__all__ = [
    "Binding",
    "BindingUpdateEpoch",
    "SealedBindingValue",
]
