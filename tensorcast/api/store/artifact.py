#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Mapping, Sequence, TypedDict, cast

import torch

from tensorcast.api._device import device_uuid_for, resolve_device
from tensorcast.api._view_ops import (
    SliceSpec,
    ViewSpecBuildResult,
    _coerce_slice_spec,
    build_view_spec,
)
from tensorcast.api.context import CallContext
from tensorcast.api.operation import (
    DaemonReplicaOperation,
    Operation,
    OperationStatus,
    PollingOperation,
)
from tensorcast.api.store.binding import Binding
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.inplace_slot import InplaceSlot
from tensorcast.api.store.mapped_binding import (
    CopyPlan,
    compute_mapped_view_id,
    normalize_copy_plan,
    validate_copy_plan,
    view_narrow_ranges,
)
from tensorcast.api.store.materialization import (
    MaterializationPipeline,
    _build_source_policy,
)
from tensorcast.api.store.region_utils import collect_storage_bases
from tensorcast.api.store.retry import (
    map_materialization_error,
    raise_mapped_materialization_error,
)
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
    FallbackOptions,
    FallbackPreference,
)
from tensorcast.api.store.view_composer import (
    ViewBuilder,
    ViewMetadataCache,
    ViewSpecComposer,
    compute_view_id,
)
from tensorcast.common.identity import is_byte_artifact_id, validate_byte_artifact_cgid
from tensorcast.common.selection_contract import (
    build_artifact_selection,
    compute_selected_index_bytes,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class PrefetchedReplica:
    artifact_id: str
    view_id: str
    operation_id: str
    device_id: int
    daemon_id: str
    source: str | None


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


@dataclass(frozen=True, slots=True)
class TensorDictMaterializationResult:
    tensors: dict[str, torch.Tensor]
    diagnostics: MaterializationDiagnostics


def _materialization_source_label(
    source: store_daemon_pb2.MaterializationSource | None,
) -> str | None:
    if source is None:
        return None
    if source == store_daemon_pb2.MATERIALIZATION_SOURCE_P2P:
        return "p2p"
    if source == store_daemon_pb2.MATERIALIZATION_SOURCE_DISK:
        return "disk"
    if source == store_daemon_pb2.MATERIALIZATION_SOURCE_LOCAL_REPLICA:
        return "local_replica"
    return None


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


def _is_active_reference_cleanup_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return "region has active references" in message or "active reference" in message


def _cleanup_region_ids_best_effort(
    *,
    store: "Store",
    region_ids: Sequence[str],
    context: str,
) -> None:
    for region_id in region_ids:
        try:
            released = store.unregister_vram_region(region_id)
            if not released:
                logger.warning(
                    "%s: unregister_vram_region returned False (region_id=%s)",
                    context,
                    region_id,
                )
            continue
        except Exception as exc:  # noqa: BLE001
            if not _is_active_reference_cleanup_error(exc):
                logger.warning(
                    "%s: unregister_vram_region failed (region_id=%s): %s",
                    context,
                    region_id,
                    exc,
                )
                continue
            try:
                forced = store.unregister_vram_region(region_id, force=True)
            except Exception as force_exc:  # noqa: BLE001
                logger.warning(
                    "%s: active-reference cleanup failed for region_id=%s "
                    "(normal=%s, force=%s)",
                    context,
                    region_id,
                    exc,
                    force_exc,
                )
                continue
            if not forced:
                logger.warning(
                    "%s: region cleanup deferred due to active references "
                    "(region_id=%s)",
                    context,
                    region_id,
                )


_TRANSPORT_GROUP_OPID_MARKER = "#tcg:"
_TRANSPORT_GROUP_KIND_TAG = "tc.transport.group.kind"
_TRANSPORT_GROUP_ID_TAG = "tc.transport.group.id"
_TRANSPORT_GROUP_TOTAL_PARTS_TAG = "tc.transport.group.total_parts"
_TRANSPORT_GROUP_PART_ID_TAG = "tc.transport.group.part_id"
_TRANSPORT_GROUP_PRIORITY_TAG = "tc.transport.group.priority"
_TRANSPORT_GROUP_EPOCH_TAG = "tc.transport.group.epoch"
_TRANSPORT_REQUEST_ID_TAG = "tc.transport.request_id"
_ALLOWED_OPERATION_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
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
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_transport_operation_id(
    *,
    base_operation_id: str,
    ctx: CallContext | None,
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

    if group_kind and group_id and part_id and total_parts > 0:
        if not request_id:
            request_id = _sanitize_operation_token(f"{group_id}:{part_id}")
        metadata = (
            f"kind={group_kind};gid={group_id};tot={int(total_parts)};"
            f"part={part_id};pri={int(priority)};ep={int(epoch)};rid={request_id}"
        )
        return f"{base_operation_id}{_TRANSPORT_GROUP_OPID_MARKER}{metadata}"

    if request_id:
        return f"{base_operation_id}{_TRANSPORT_GROUP_OPID_MARKER}rid={request_id}"

    return base_operation_id


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
        if hasattr(resp, "lease_token") and resp.lease_token:
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


class ArtifactSerializedFallback(TypedDict):
    prefer: FallbackPreference
    prefer_disk: bool | None
    allow_p2p: bool
    allow_disk: bool
    verify_checksums: bool
    replica_uuid: str | None


class ArtifactSerialized(TypedDict):
    artifact_id: str | None
    key: str | None
    fallback: ArtifactSerializedFallback | None
    canonical_index: str | None
    generation: int | None


def _fallback_to_dict(
    fallback: FallbackOptions | None,
) -> ArtifactSerializedFallback | None:
    if fallback is None:
        return None
    return {
        "prefer": fallback.prefer,
        "prefer_disk": (
            bool(fallback.prefer_disk) if fallback.prefer_disk is not None else None
        ),
        "allow_p2p": bool(fallback.allow_p2p),
        "allow_disk": bool(fallback.allow_disk),
        "verify_checksums": bool(fallback.verify_checksums),
        "replica_uuid": fallback.replica_uuid,
    }


def _fallback_from_dict(
    data: ArtifactSerializedFallback | Mapping[str, object] | None,
) -> FallbackOptions | None:
    if data is None:
        return None
    prefer_value = data.get("prefer")
    prefer = prefer_value if isinstance(prefer_value, str) else "auto"
    prefer_disk_raw = data.get("prefer_disk")
    prefer_disk = prefer_disk_raw if isinstance(prefer_disk_raw, bool) else None
    replica_uuid_value = data.get("replica_uuid")
    replica_uuid = replica_uuid_value if isinstance(replica_uuid_value, str) else None
    return FallbackOptions(
        prefer=prefer,  # pyright: ignore[reportArgumentType]
        prefer_disk=prefer_disk,
        allow_p2p=bool(data.get("allow_p2p", True)),
        allow_disk=bool(data.get("allow_disk", True)),
        verify_checksums=bool(data.get("verify_checksums", True)),
        replica_uuid=replica_uuid,
    )


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
        fallback: FallbackOptions | str | None = None,
        canonical_index_bytes: bytes | None = None,
        canonical_index: CanonicalIndex | None = None,
        generation: int | None = None,
        view_spec: ViewSpecBuildResult | None = None,
        view_metadata: ViewMetadataCache | None = None,
        view_depth: int = 0,
    ) -> None:
        identifiers = [bool(artifact_id), bool(key)]
        if sum(identifiers) == 0:
            raise ArtifactError(
                "At least one of artifact_id or key is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._fallback = FallbackOptions.parse(fallback)
        self._artifact_id = artifact_id
        self._key_hint = key
        self._canonical_index_bytes = canonical_index_bytes
        self._canonical_index = canonical_index
        self._tensor_metas: dict[str, TensorMeta] | None = None
        self._view_spec = view_spec
        self._view_metadata = view_metadata
        self._view_depth = max(0, int(view_depth))
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

    def tensor_dict(
        self,
        *,
        device: torch.device | str,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> dict[str, torch.Tensor]:
        result = self.tensor_dict_with_diagnostics(
            device=device,
            options=options,
            ctx=ctx,
        )
        return result.tensors

    def tensor_dict_with_diagnostics(
        self,
        *,
        device: torch.device | str,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> TensorDictMaterializationResult:
        artifact_id = self._ensure_identified()
        requested_names = (
            tuple(self._view_metadata.tensor_names)
            if self._view_metadata is not None and self._view_metadata.tensor_names
            else None
        )
        _, runtime, pipeline = self._require_components()
        view_spec_proto = self._view_spec.proto if self._view_spec else None
        view_data_hash = (
            self._view_metadata.view_data_hash if self._view_metadata else None
        )
        view_index_hint = (
            self._view_metadata.view_index_bytes if self._view_metadata else None
        )
        replica_uuid = self._fallback.replica_uuid if self._fallback else None
        materialize_start = time.perf_counter()
        payload, _ = pipeline.materialize_subset(
            artifact_id=artifact_id,
            key=None,
            device=device,
            fallback=self._fallback,
            tensor_names=requested_names,
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
            diagnostics = MaterializationDiagnostics(
                source=_materialization_source_label(payload.source),
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
            return TensorDictMaterializationResult(
                tensors=output,
                diagnostics=diagnostics,
            )
        finally:
            if state is None:
                pipeline._release_materialized(payload, runtime.ensure_client())

    def tensor(
        self,
        name: str,
        *,
        device: torch.device | str,
        cache: bool = True,  # cache retained for compatibility, no-op in v2
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
    ) -> torch.Tensor:
        _ = cache  # cache parameter is reserved for future use
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
        artifact_id = self._ensure_identified()
        effective_index = self._effective_index()
        _ = self._validate_tensor_names(effective_index, None)
        _, _, pipeline = self._require_components()
        view_spec_proto = self._view_spec.proto if self._view_spec else None
        view_data_hash = (
            self._view_metadata.view_data_hash if self._view_metadata else None
        )
        view_index_hint = (
            self._view_metadata.view_index_bytes if self._view_metadata else None
        )
        replica_uuid = self._fallback.replica_uuid if self._fallback else None
        pipeline.get_into(
            target,
            artifact_id=artifact_id,
            key=None,
            device=device,
            fallback=self._fallback,
            options=options,
            view_spec=view_spec_proto,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
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
        artifact_id = self._ensure_identified()
        _, _, pipeline = self._require_components()
        requested_names = (name,)
        view_spec_proto = self._view_spec.proto if self._view_spec else None
        view_data_hash = (
            self._view_metadata.view_data_hash if self._view_metadata else None
        )
        view_index_hint = (
            self._view_metadata.view_index_bytes if self._view_metadata else None
        )
        replica_uuid = self._fallback.replica_uuid if self._fallback else None
        resolved_device = device if device is not None else target_tensor.device
        pipeline.get_into(
            {requested_names[0]: target_tensor},
            artifact_id=artifact_id,
            key=None,
            device=resolved_device,
            fallback=self._fallback,
            options=options,
            view_spec=view_spec_proto,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
            tensor_names=requested_names,
            ctx=ctx,
        )

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
        packing: str = "byte_space",
        capacity_bytes: int | None = None,
        publish: bool = False,
        ctx: CallContext | None = None,
    ) -> Binding:
        """Allocate target tensors, fill from this artifact, and return a Binding."""
        self._require_components()
        base_index = self._effective_index()
        if not base_index.entries:
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
        if capacity_bytes is not None:
            requested_capacity = int(capacity_bytes)
            if requested_capacity <= 0:
                raise ArtifactError(
                    "capacity_bytes must be positive",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            required_capacity = int(base_index.total_size_bytes)
            if requested_capacity < required_capacity:
                raise ArtifactError(
                    "capacity_bytes is smaller than the selected artifact layout",
                    status_code="RESOURCE_EXHAUSTED",
                    retryable=False,
                )
        target_tensors: dict[str, torch.Tensor] = {}
        for entry in base_index.entries:
            target_tensors[entry.name] = torch.empty_strided(
                size=tuple(int(v) for v in entry.shape),
                stride=tuple(int(v) for v in entry.stride),
                dtype=entry.dtype,
                device=device_obj,
            )
        return self.bind_into(
            target_tensors,
            packing=packing,
            publish=publish,
            ctx=ctx,
        )

    def bind_into(
        self,
        target_tensors: Mapping[str, torch.Tensor],
        *,
        mapping: CopyPlan | None = None,
        packing: str = "byte_space",
        publish: bool = False,
        ctx: CallContext | None = None,
    ) -> Binding:
        """Adopt user-owned CUDA tensors, fill once, and return a Binding."""
        if mapping is not None:
            return self._bind_into_mapped(
                target_tensors=target_tensors,
                mapping=mapping,
                packing=packing,
                publish=publish,
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

        def _register_regions() -> tuple[str, ...]:
            region_ids: list[str] = []
            bases = collect_storage_bases(target_tensors)
            try:
                for base_ptr, nbytes in sorted(bases.items()):
                    handle = store.register_vram_region(
                        device_id=device_id,
                        base_ptr=base_ptr,
                        size_bytes=nbytes,
                        ttl_ms=int(ttl_ms),
                    )
                    region_ids.append(handle.region_id)
            except Exception as exc:  # noqa: BLE001
                _cleanup_region_ids_best_effort(
                    store=store,
                    region_ids=tuple(region_ids),
                    context="bind_into.register_regions",
                )
                raise ArtifactError(
                    "bind_into requires user-owned CUDA memory (daemon-owned tensors cannot be used)",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                ) from exc
            return tuple(region_ids)

        region_ids = _register_regions()

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

        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        effective_prefer = (
            self._fallback.prefer if self._fallback is not None else "auto"
        )
        if self._fallback is not None:
            if self._fallback.prefer == "p2p":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_P2P
                )
            elif self._fallback.prefer == "disk":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
                )

        allow_p2p = True if self._fallback is None else bool(self._fallback.allow_p2p)
        if effective_prefer == "local":
            allow_p2p = False
        allow_disk = True if self._fallback is None else bool(self._fallback.allow_disk)
        if effective_prefer == "local":
            allow_disk = False
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )

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
                    response = client.materialize_into_target_v2(
                        selection=selection,
                        target_layout=region_layout.layout,
                        device_uuid=device_uuid_for(device_id),
                        preference=preference,
                        source_policy=source_policy,
                        operation_id=operation_id,
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
                        _cleanup_region_ids_best_effort(
                            store=store,
                            region_ids=tuple(region_ids),
                            context="bind_into.retry_rebind",
                        )
                        region_ids = _register_regions()
                        attempt += 1
                        continue
                    _cleanup_region_ids_best_effort(
                        store=store,
                        region_ids=tuple(region_ids),
                        context="bind_into.materialize_error",
                    )
                    raise ArtifactError(
                        str(error),
                        status_code=error.status_code,
                        retryable=False,
                    ) from exc

                if (
                    response.status
                    != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
                ):
                    _cleanup_region_ids_best_effort(
                        store=store,
                        region_ids=tuple(region_ids),
                        context="bind_into.materialize_status",
                    )
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
            _cleanup_region_ids_best_effort(
                store=store,
                region_ids=tuple(region_ids),
                context="bind_into.exception",
            )
            raise

        slot = InplaceSlot(
            store=store,
            runtime=runtime,
            pipeline=pipeline,
            tensors=target_tensors,
            device=first_tensor.device,
            device_id=device_id,
            region_id=None,
            region_layout=region_layout,
            view_spec=view_spec_proto,
            fallback=self._fallback,
            artifact_id=self._ensure_identified(),
            canonical_index_bytes=canonical_index_bytes,
            target_publication_token=getattr(
                response, "target_publication_token", None
            ),
        )
        return Binding(slot, publish=publish, ctx=ctx)

    def _bind_into_mapped(
        self,
        *,
        target_tensors: Mapping[str, torch.Tensor],
        mapping: CopyPlan,
        packing: str,
        publish: bool,
        ctx: CallContext | None,
    ) -> Binding:
        store, runtime, pipeline = self._require_components()
        if not isinstance(target_tensors, Mapping) or not target_tensors:
            raise ArtifactError(
                "bind_into target_tensors must be a non-empty mapping",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
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

        def _register_regions() -> tuple[str, ...]:
            region_ids: list[str] = []
            bases = collect_storage_bases(target_tensors)
            try:
                for base_ptr, nbytes in sorted(bases.items()):
                    handle = store.register_vram_region(
                        device_id=device_id,
                        base_ptr=base_ptr,
                        size_bytes=nbytes,
                        ttl_ms=int(ttl_ms),
                    )
                    region_ids.append(handle.region_id)
            except Exception as exc:  # noqa: BLE001
                _cleanup_region_ids_best_effort(
                    store=store,
                    region_ids=tuple(region_ids),
                    context="bind_into_mapped.register_regions",
                )
                raise ArtifactError(
                    "bind_into requires user-owned CUDA memory (daemon-owned tensors cannot be used)",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                ) from exc
            return tuple(region_ids)

        region_ids = _register_regions()
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

        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        effective_prefer = (
            self._fallback.prefer if self._fallback is not None else "auto"
        )
        if self._fallback is not None:
            if self._fallback.prefer == "p2p":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_P2P
                )
            elif self._fallback.prefer == "disk":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
                )

        allow_p2p = True if self._fallback is None else bool(self._fallback.allow_p2p)
        if effective_prefer == "local":
            allow_p2p = False
        allow_disk = True if self._fallback is None else bool(self._fallback.allow_disk)
        if effective_prefer == "local":
            allow_disk = False
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )

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
        try:
            response = None
            region_layout = None
            attempt = 0
            while attempt < 2:
                region_layout = pipeline._build_mapped_region_backed_layout(
                    target=target_tensors,
                    device_id=device_id,
                    selection_order=selection_order,
                    mapped_view_id=mapped_view_id,
                    selection_index_bytes=selection_index_bytes,
                )
                try:
                    selection = self._build_region_layout_selection(
                        region_layout=region_layout,
                        view_spec_proto=view_spec_proto,
                    )
                    response = client.materialize_into_mapped_target(
                        selection=selection,
                        target_layout=region_layout.layout,
                        device_uuid=device_uuid_for(device_id),
                        preference=preference,
                        source_policy=source_policy,
                        copy_plan=copy_plan,
                        dst_tensors=target_tensors,
                        operation_id=operation_id,
                        timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
                    )
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
                        _cleanup_region_ids_best_effort(
                            store=store,
                            region_ids=tuple(region_ids),
                            context="bind_into_mapped.retry_rebind",
                        )
                        region_ids = _register_regions()
                        attempt += 1
                        continue
                    _cleanup_region_ids_best_effort(
                        store=store,
                        region_ids=tuple(region_ids),
                        context="bind_into_mapped.materialize_error",
                    )
                    raise ArtifactError(
                        str(error),
                        status_code=error.status_code,
                        retryable=False,
                    ) from exc

                if (
                    response.status
                    != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
                ):
                    _cleanup_region_ids_best_effort(
                        store=store,
                        region_ids=tuple(region_ids),
                        context="bind_into_mapped.materialize_status",
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
            _cleanup_region_ids_best_effort(
                store=store,
                region_ids=tuple(region_ids),
                context="bind_into_mapped.exception",
            )
            raise

        slot = InplaceSlot(
            store=store,
            runtime=runtime,
            pipeline=pipeline,
            tensors=target_tensors,
            device=first_tensor.device,
            device_id=device_id,
            region_id=None,
            region_layout=region_layout,
            view_spec=view_spec_proto,
            fallback=self._fallback,
            artifact_id=self._ensure_identified(),
            canonical_index_bytes=canonical_index_bytes,
            target_publication_token=getattr(
                response, "target_publication_token", None
            ),
            copy_plan=copy_plan,
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
        elif self._view_spec is not None:
            view_hash = ViewSpecComposer.hash_view_spec(self._view_spec)
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

    def prefetch(
        self,
        *,
        device: torch.device | str | int,
        ctx: CallContext | None = None,
        options: GetArtifactOptions | None = None,
    ) -> Operation[PrefetchedReplica]:
        from tensorcast.api._config import GetArtifactOptions

        artifact_id = self._ensure_identified()
        store, runtime, pipeline = self._require_components()
        if getattr(store, "_enable_prefetch", True) is False:
            raise ArtifactError(
                "Prefetch is disabled",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        view_spec_proto = self._view_spec.proto if self._view_spec else None
        view_data_hash = (
            self._view_metadata.view_data_hash if self._view_metadata else None
        )
        view_index_hint = (
            self._view_metadata.view_index_bytes if self._view_metadata else None
        )

        device_obj = (
            torch.device(f"cuda:{int(device)}")
            if isinstance(device, int)
            else torch.device(device)
        )
        if device_obj.type == "cpu":
            raise ArtifactError(
                "prefetch does not support CPU devices",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        device_id = int(device_obj.index if device_obj.index is not None else 0)
        daemon_id = (
            getattr(runtime, "daemon_id", None) or None
        ) or runtime.daemon_endpoint
        selection = self._build_artifact_selection()
        view_id = selection.view_id
        selection_hash = bytes(selection.selection_hash).hex()

        deterministic_replica_uuid: str | None = None
        if ctx is not None and ctx.idempotency_key:
            ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
            action_fingerprint = f"prefetch|daemon={daemon_id}|selection={selection_hash}|device={device_id}|lease=NO_LEASE|v1"
            deterministic_replica_uuid = str(
                uuid.uuid5(ns, f"{ctx.idempotency_key}|{action_fingerprint}")
            )

        replica_uuid = deterministic_replica_uuid or (
            self._fallback.replica_uuid if self._fallback else None
        )
        if not replica_uuid:
            replica_uuid = uuid.uuid4().hex

        opts = options or GetArtifactOptions()
        opts = opts.model_copy(
            update={
                "wait_for_completion": False,
                "enable_verification": False,
            }
        )

        payload, _ = pipeline.materialize_subset(
            artifact_id=artifact_id,
            key=None,
            device=device_obj,
            fallback=self._fallback,
            tensor_names=None,
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

        source: str | None = None
        if payload.source == store_daemon_pb2.MATERIALIZATION_SOURCE_P2P:
            source = "p2p"
        elif payload.source == store_daemon_pb2.MATERIALIZATION_SOURCE_DISK:
            source = "disk"
        elif payload.source == store_daemon_pb2.MATERIALIZATION_SOURCE_LOCAL_REPLICA:
            source = "local"

        replica = PrefetchedReplica(
            artifact_id=artifact_id,
            view_id=view_id,
            operation_id=operation_id,
            device_id=device_id,
            daemon_id=daemon_id,
            source=source,
        )

        try:
            from tensorcast.api import _metrics as store_metrics

            store_metrics.record_prefetch_event(
                runtime.daemon_endpoint, status="issued"
            )
        except Exception:  # noqa: BLE001
            pass

        return DaemonReplicaOperation(
            operation_id=operation_id,
            runtime_ref=weakref.ref(runtime),
            ctx=ctx,
            result_factory=lambda: replica,
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
        token = bytes(resp.lease_token) if hasattr(resp, "lease_token") else b""
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

    def with_fallback(self, fallback: FallbackOptions | str) -> Artifact:
        parsed = FallbackOptions.parse(fallback)
        clone = Artifact(
            store_ref=self._store_ref,
            artifact_id=self._artifact_id,
            key=self._key_hint,
            fallback=parsed,
            canonical_index_bytes=self._canonical_index_bytes,
            canonical_index=self._canonical_index,
            generation=self._generation,
            view_spec=self._view_spec,
            view_metadata=self._view_metadata,
            view_depth=self._view_depth,
        )
        clone._released = self._released
        clone._tensor_metas = dict(self._tensor_metas or {})
        return clone

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
            "fallback": _fallback_to_dict(self._fallback),
            "canonical_index": encoded_index,
            "generation": self._generation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object], store: "Store") -> Artifact:
        artifact_id = data.get("artifact_id")
        key_hint = data.get("key")
        fallback_dict = data.get("fallback")
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
        fallback = None
        if isinstance(fallback_dict, Mapping):
            fallback = _fallback_from_dict(cast(Mapping[str, object], fallback_dict))
        generation_value = (
            int(generation) if isinstance(generation, (int, float)) else None
        )
        return cls(
            store_ref=weakref.ref(store),
            artifact_id=str(artifact_id) if artifact_id else None,
            key=str(key_hint) if key_hint else None,
            fallback=fallback,
            canonical_index_bytes=canonical_index_bytes,
            canonical_index=canonical_index,
            generation=generation_value,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
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
        if self._view_metadata is not None and self._view_metadata.view_id:
            return str(self._view_metadata.view_id)
        return self._control_plane_view_id(runtime)

    def _build_artifact_selection(self) -> common_pb2.ArtifactSelection:
        artifact_id = self._ensure_identified()
        runtime = self._runtime_if_available()
        byte_artifact_profile = _has_validated_byte_artifact_profile(artifact_id)

        view_spec_proto: common_pb2.ViewSpec | None = None
        if self._view_spec is not None and not self._view_spec.is_identity:
            view_spec_proto = self._view_spec.proto
            if view_spec_proto is None:
                raise ArtifactError(
                    "View spec proto missing while building selection",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )

        selection_names: tuple[str, ...] = ()
        if self._view_metadata is not None and self._view_metadata.tensor_names:
            selection_names = tuple(self._view_metadata.tensor_names)

        canonical_index_bytes = self._canonical_index_bytes
        if (
            canonical_index_bytes is None
            and runtime is not None
            and not byte_artifact_profile
        ):
            canonical_index_bytes = runtime.ensure_client().get_artifact_index_by_id(
                artifact_id
            )
        if canonical_index_bytes is None and not byte_artifact_profile:
            raise ArtifactError(
                "Canonical index bytes missing while building selection",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        has_transform = bool(view_spec_proto is not None and view_spec_proto.tensors)
        has_subset = bool(selection_names)
        layout_index_bytes: bytes | None = None
        if self._view_metadata is not None and self._view_metadata.view_index_bytes:
            layout_index_bytes = bytes(self._view_metadata.view_index_bytes)
        elif has_transform or has_subset:
            if byte_artifact_profile:
                raise ArtifactError(
                    "byte artifact selection does not support view/subset selection",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            assert canonical_index_bytes is not None
            layout_index_bytes = compute_selected_index_bytes(
                canonical_index_bytes=canonical_index_bytes,
                view_spec=view_spec_proto,
                tensor_names=selection_names,
            )

        try:
            return build_artifact_selection(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes or b"",
                layout_index_bytes=layout_index_bytes,
                view_spec=view_spec_proto,
                tensor_names=selection_names,
            )
        except ValueError as exc:
            raise ArtifactError(
                str(exc),
                status_code="FAILED_PRECONDITION",
                retryable=False,
            ) from exc

    def _build_region_layout_selection(
        self,
        *,
        region_layout,
        view_spec_proto: common_pb2.ViewSpec | None,
    ) -> common_pb2.ArtifactSelection:
        artifact_id = self._ensure_identified()
        byte_artifact_profile = _has_validated_byte_artifact_profile(artifact_id)
        canonical_index_bytes = self._canonical_index_bytes
        runtime = self._runtime_if_available()
        if (
            canonical_index_bytes is None
            and runtime is not None
            and not byte_artifact_profile
        ):
            canonical_index_bytes = runtime.ensure_client().get_artifact_index_by_id(
                artifact_id
            )
        if canonical_index_bytes is None and not byte_artifact_profile:
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
            if byte_artifact_profile:
                raise ArtifactError(
                    "byte artifact selection does not support view-index region layouts",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if region_layout.view_index_bytes:
                layout_index_bytes = bytes(region_layout.view_index_bytes)
            else:
                assert canonical_index_bytes is not None
                layout_index_bytes = compute_selected_index_bytes(
                    canonical_index_bytes=canonical_index_bytes,
                    view_spec=view_spec_proto,
                    tensor_names=selection_names,
                )

        try:
            return build_artifact_selection(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes or b"",
                layout_index_bytes=layout_index_bytes,
                view_spec=view_spec_proto,
                tensor_names=selection_names,
                view_subset_hash=subset_hash if subset_hash else None,
                view_id=str(region_layout.view_id or ""),
                allow_view_id_without_spec=bool(
                    region_layout.view_id and view_spec_proto is None
                ),
            )
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
                resolved = runtime.resolve_key_mapping_cached(key=self._key_hint)
                artifact_id = resolved[0] if isinstance(resolved, tuple) else resolved
                if not artifact_id:
                    raise ArtifactError(
                        f"Artifact key '{self._key_hint}' is not mapped",
                        status_code="NOT_FOUND",
                        retryable=False,
                    )
                self._artifact_id = artifact_id
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
        if self._view_metadata is not None:
            return self._view_metadata.selected_index
        return base_index

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
        artifact_id = self._ensure_identified()
        if _has_validated_byte_artifact_profile(artifact_id) and (
            slices or transpose or subset
        ):
            raise ArtifactError(
                "byte artifacts do not support view/subset derivation",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._ensure_metadata()
        typed_slices, normalized_transpose = self._normalize_view_inputs(
            slices=slices,
            transpose=transpose,
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
            fallback=self._fallback,
            canonical_index_bytes=self._canonical_index_bytes,
            canonical_index=self._canonical_index,
            generation=self._generation,
            view_spec=composed_spec,
            view_metadata=view_cache,
            view_depth=depth,
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
            if view_hash is None:
                view_hash = ViewSpecComposer.hash_view_spec(
                    self._view_spec, subset=subset_names
                )
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
                view_data_hash=str(view_hash),
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
