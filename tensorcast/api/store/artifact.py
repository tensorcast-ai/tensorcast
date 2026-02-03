#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import asyncio
import base64
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
from tensorcast.api.store.deferred_loader import DeferredCommitResult, DeferredLoader
from tensorcast.api.store.inplace_slot import InplaceSlot
from tensorcast.api.store.mapped_binding import (
    CopyPlan,
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
from tensorcast.common.selection_identity import (
    compute_selection_hash,
    compute_view_subset_hash,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2

if TYPE_CHECKING:
    from tensorcast.api._config import GetArtifactOptions
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
    disk_path: str | None
    prefer_disk: bool | None
    allow_p2p: bool
    verify_checksums: bool
    replica_uuid: str | None


class ArtifactSerialized(TypedDict):
    artifact_id: str | None
    key: str | None
    disk_path: str | None
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
        "disk_path": fallback.disk_path,
        "prefer_disk": (
            bool(fallback.prefer_disk) if fallback.prefer_disk is not None else None
        ),
        "allow_p2p": bool(fallback.allow_p2p),
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
    disk_path_value = data.get("disk_path")
    disk_path = disk_path_value if isinstance(disk_path_value, str) else None
    prefer_disk_raw = data.get("prefer_disk")
    prefer_disk = prefer_disk_raw if isinstance(prefer_disk_raw, bool) else None
    replica_uuid_value = data.get("replica_uuid")
    replica_uuid = replica_uuid_value if isinstance(replica_uuid_value, str) else None
    return FallbackOptions(
        prefer=prefer,  # pyright: ignore[reportArgumentType]
        disk_path=disk_path,
        prefer_disk=prefer_disk,
        allow_p2p=bool(data.get("allow_p2p", True)),
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
        disk_path: str | None = None,
        fallback: FallbackOptions | str | None = None,
        canonical_index_bytes: bytes | None = None,
        canonical_index: CanonicalIndex | None = None,
        generation: int | None = None,
        view_spec: ViewSpecBuildResult | None = None,
        view_metadata: ViewMetadataCache | None = None,
        view_depth: int = 0,
    ) -> None:
        identifiers = [bool(artifact_id), bool(key), bool(disk_path)]
        if sum(identifiers) == 0:
            raise ArtifactError(
                "At least one of artifact_id, key, or disk_path is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._fallback = FallbackOptions.parse(fallback)
        self._artifact_id = artifact_id
        self._key_hint = key
        self._disk_path_hint = disk_path or (
            self._fallback.disk_path if self._fallback else None
        )
        self._canonical_index_bytes = canonical_index_bytes
        self._canonical_index = canonical_index
        self._tensor_metas: dict[str, TensorMeta] | None = None
        self._view_spec = view_spec
        self._view_metadata = view_metadata
        self._view_depth = max(0, int(view_depth))
        effective_index = (
            view_metadata.canonical_index
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
        names: Sequence[str] | None = None,
        ctx: CallContext | None = None,
    ) -> dict[str, torch.Tensor]:
        artifact_id = self._ensure_identified()
        effective_index = self._effective_index()
        requested_names = self._validate_tensor_names(effective_index, names)
        store, runtime, pipeline = self._require_components()
        view_spec_proto = self._view_spec.proto if self._view_spec else None
        view_data_hash = (
            self._view_metadata.view_data_hash if self._view_metadata else None
        )
        view_index_hint = (
            self._view_metadata.view_index_bytes if self._view_metadata else None
        )
        replica_uuid = self._fallback.replica_uuid if self._fallback else None
        payload, _ = pipeline.materialize_subset(
            artifact_id=artifact_id,
            key=None,
            device=device,
            fallback=self._fallback,
            tensor_names=requested_names,
            canonical_index_hint=self._canonical_index_bytes,
            disk_path_hint=self._disk_path_hint,
            view_spec=view_spec_proto,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
            ctx=ctx,
        )
        state: dict[str, torch.Tensor] | None = None
        try:
            self._update_metadata_from_payload(payload, runtime)
            state = pipeline._payload_state_dict(payload)
            if requested_names is None:
                return state
            return {name: state[name] for name in requested_names}
        finally:
            if state is None:
                pipeline._release_materialized(payload, runtime.ensure_client())

    def tensor(
        self,
        name: str,
        *,
        device: torch.device | str,
        cache: bool = True,  # cache retained for compatibility, no-op in v2
        ctx: CallContext | None = None,
    ) -> torch.Tensor:
        _ = cache  # cache parameter is reserved for future use
        result = self.tensor_dict(device=device, names=[name], ctx=ctx)
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
        names: Sequence[str] | None = None,
    ) -> Artifact:
        """Return a derived lazy view; no RPCs occur until materialization or exists()."""
        return self._derive_view(
            slices=slices,
            transpose=transpose,
            subset=list(names) if names is not None else None,
            composer=ViewSpecComposer(),
        )

    def subset(self, names: Sequence[str]) -> Artifact:
        return self.view(names=names)

    def slice(self, slices: Mapping[str, Sequence[object]]) -> Artifact:
        return self.view(slices=slices)

    def view_builder(self) -> ViewBuilder:
        return ViewBuilder(artifact_ref=weakref.ref(self), composer=ViewSpecComposer())

    def deferred_loader(
        self,
        *,
        device: torch.device | str,
        packing: str = "append",
        capacity_bytes: int | None = None,
    ) -> DeferredLoader:
        """Return a deferred loader for vLLM-style placeholder binding."""
        self._require_components()
        return DeferredLoader(
            artifact=self,
            device=device,
            packing=packing,
            capacity_bytes=capacity_bytes,
        )

    def bind(
        self,
        device: torch.device | str,
        *,
        packing: str = "byte_space",
        capacity_bytes: int | None = None,
        publish: bool = False,
        ctx: CallContext | None = None,
    ) -> Binding:
        """Allocate placeholders, fill from this artifact, and return a Binding."""
        self._require_components()
        base_index = self._effective_index()
        if not base_index.entries:
            raise ArtifactError(
                "Artifact index is empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        with self.deferred_loader(
            device=device,
            packing=packing,
            capacity_bytes=capacity_bytes,
        ) as loader:
            for entry in base_index.entries:
                loader.tensor(entry.name)
            slot = loader.commit()
        return Binding(slot, publish=publish, ctx=ctx)

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
                for region_id in region_ids:
                    store.unregister_vram_region(region_id)
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
        disk_path: str | None = None
        verify_checksums = True
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
            disk_path = self._fallback.disk_path
            verify_checksums = bool(self._fallback.verify_checksums)
        if (
            disk_path
            and preference == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        ):
            preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK

        allow_p2p = True if self._fallback is None else bool(self._fallback.allow_p2p)
        if effective_prefer == "local":
            allow_p2p = False
        allow_disk = effective_prefer != "local" or bool(disk_path)
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )

        client = runtime.ensure_client()
        operation_id = uuid.uuid4().hex
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
                    response = client.materialize_into_target_v2(
                        artifact_id=self._ensure_identified(),
                        target_layout=region_layout.layout,
                        device_uuid=device_uuid_for(device_id),
                        preference=preference,
                        source_policy=source_policy,
                        disk_path=disk_path,
                        verify_checksums=verify_checksums,
                        tensor_names=region_layout.selection_names,
                        view=view_spec_proto,
                        view_id=region_layout.view_id
                        if view_spec_proto is None
                        else None,
                        view_subset_hash=region_layout.view_subset_hash,
                        operation_id=operation_id,
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
                        for region_id in region_ids:
                            store.unregister_vram_region(region_id)
                        region_ids = _register_regions()
                        attempt += 1
                        continue
                    for region_id in region_ids:
                        store.unregister_vram_region(region_id)
                    raise ArtifactError(
                        str(error),
                        status_code=error.status_code,
                        retryable=False,
                    ) from exc

                if (
                    response.status
                    != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
                ):
                    for region_id in region_ids:
                        store.unregister_vram_region(region_id)
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
            for region_id in region_ids:
                store.unregister_vram_region(region_id)
            raise

        storage_ids = tuple(
            storage.storage_id for storage in region_layout.layout.storages
        )
        commit_result = DeferredCommitResult(
            tensor_names=region_layout.selection_names,
            view_id=region_layout.view_id,
            view_subset_hash=region_layout.view_subset_hash,
            storage_ids=storage_ids,
            logical_size_bytes=region_layout.logical_total_size,
            published_artifact=None,
        )
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
            commit_result=commit_result,
            artifact_id=self._ensure_identified(),
            canonical_index_bytes=canonical_index_bytes,
            target_write_token=getattr(response, "target_write_token", None),
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
        if publish:
            raise ArtifactError(
                "publish is not supported for mapped binding in v1",
                status_code="FAILED_PRECONDITION",
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

        if self._view_metadata is not None and self._view_spec is None:
            raise ArtifactError(
                "mapped binding requires a view spec (view_id-only handles not supported)",
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
                for region_id in region_ids:
                    store.unregister_vram_region(region_id)
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

        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        disk_path: str | None = None
        verify_checksums = True
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
            disk_path = self._fallback.disk_path
            verify_checksums = bool(self._fallback.verify_checksums)
        if (
            disk_path
            and preference == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        ):
            preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK

        allow_p2p = True if self._fallback is None else bool(self._fallback.allow_p2p)
        if effective_prefer == "local":
            allow_p2p = False
        allow_disk = effective_prefer != "local" or bool(disk_path)
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )

        client = runtime.ensure_client()
        operation_id = uuid.uuid4().hex
        try:
            response = None
            region_layout = None
            attempt = 0
            while attempt < 2:
                region_layout = pipeline._build_mapped_region_backed_layout(
                    target=target_tensors,
                    device_id=device_id,
                    selection_order=selection_order,
                )
                try:
                    response = client.materialize_into_mapped_target(
                        artifact_id=self._ensure_identified(),
                        target_layout=region_layout.layout,
                        device_uuid=device_uuid_for(device_id),
                        preference=preference,
                        source_policy=source_policy,
                        disk_path=disk_path,
                        verify_checksums=verify_checksums,
                        copy_plan=copy_plan,
                        dst_tensors=target_tensors,
                        view=view_spec_proto,
                        operation_id=operation_id,
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
                        for region_id in region_ids:
                            store.unregister_vram_region(region_id)
                        region_ids = _register_regions()
                        attempt += 1
                        continue
                    for region_id in region_ids:
                        store.unregister_vram_region(region_id)
                    raise ArtifactError(
                        str(error),
                        status_code=error.status_code,
                        retryable=False,
                    ) from exc

                if (
                    response.status
                    != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
                ):
                    for region_id in region_ids:
                        store.unregister_vram_region(region_id)
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
            for region_id in region_ids:
                store.unregister_vram_region(region_id)
            raise

        storage_ids = tuple(
            storage.storage_id for storage in region_layout.layout.storages
        )
        commit_result = DeferredCommitResult(
            tensor_names=region_layout.selection_names,
            view_id=region_layout.view_id,
            view_subset_hash=region_layout.view_subset_hash,
            storage_ids=storage_ids,
            logical_size_bytes=region_layout.logical_total_size,
            published_artifact=None,
        )
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
            commit_result=commit_result,
            artifact_id=self._ensure_identified(),
            canonical_index_bytes=canonical_index_bytes,
            target_write_token=getattr(response, "target_write_token", None),
            copy_plan=copy_plan,
        )
        return Binding(slot, publish=False, ctx=ctx)

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
        names: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.tensor_dict(device=device, names=names)
        )

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
        view_id = self._control_plane_view_id(runtime)
        view_subset_hash: bytes | None = None
        if self._view_metadata is not None and self._view_metadata.tensor_names:
            view_subset_hash = compute_view_subset_hash(
                self._view_metadata.tensor_names
            )
        selection_hash = compute_selection_hash(
            view_id=view_id, view_subset_hash=view_subset_hash
        ).hex()

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
            disk_path_hint=self._disk_path_hint,
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
            disk_path=self._disk_path_hint,
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
        disk_path_hint = self._disk_path_hint or (
            self._fallback.disk_path if self._fallback else None
        )
        if disk_path_hint:
            try:
                disk_index = self._resolve_metadata_from_disk(runtime, disk_path_hint)
            except ArtifactError as disk_error:
                if disk_error.status_code != "NOT_FOUND":
                    raise
            else:
                if disk_index is not None:
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
                    disk_path=self._disk_path_hint,
                )
        runtime.cache_artifact_index(
            ArtifactCacheEntry(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                parsed_index=canonical_index,
                generation=self._generation,
                disk_path=self._disk_path_hint,
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
            "disk_path": self._disk_path_hint,
            "fallback": _fallback_to_dict(self._fallback),
            "canonical_index": encoded_index,
            "generation": self._generation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object], store: "Store") -> Artifact:
        artifact_id = data.get("artifact_id")
        key_hint = data.get("key")
        disk_path = data.get("disk_path")
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
            disk_path=str(disk_path) if disk_path else None,
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
        if self._view_metadata is not None:
            return str(self._view_metadata.view_id)
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
                artifact_id, disk_path = runtime.resolve_key_mapping_cached(
                    key=self._key_hint
                )
                if disk_path and not self._disk_path_hint:
                    self._disk_path_hint = disk_path
                if not artifact_id:
                    raise ArtifactError(
                        f"Artifact key '{self._key_hint}' is not mapped",
                        status_code="NOT_FOUND",
                        retryable=False,
                    )
                self._artifact_id = artifact_id
                return artifact_id
            if self._disk_path_hint:
                cached = runtime.get_artifact_index_by_disk_path(self._disk_path_hint)
                if cached and cached.artifact_id:
                    self._artifact_id = cached.artifact_id
                    self._hydrate_from_cache_entry(cached)
                    return cached.artifact_id
                verify_checksums = True
                if self._fallback is not None:
                    verify_checksums = bool(self._fallback.verify_checksums)
                try:
                    resolved = runtime.ensure_client().resolve_artifact_from_disk_v2(
                        disk_path=self._disk_path_hint,
                        verify_checksums=verify_checksums,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise_mapped_materialization_error(exc)
                artifact_id = getattr(resolved, "artifact_id", "") or None
                if not artifact_id:
                    raise ArtifactError(
                        f"Failed to resolve artifact from disk path '{self._disk_path_hint}'",
                        status_code="NOT_FOUND",
                        retryable=False,
                    )
                disk_path = getattr(resolved, "disk_path", "") or self._disk_path_hint
                canonical_index_bytes = bytes(
                    getattr(resolved, "canonical_index_bytes", b"") or b""
                )
                generation_raw = getattr(resolved, "generation", 0)
                generation = int(generation_raw) if generation_raw else None
                canonical_index = None
                if canonical_index_bytes:
                    canonical_index = canonical_index_from_bytes(canonical_index_bytes)
                    self._set_metadata(
                        canonical_index_bytes,
                        canonical_index,
                        generation=generation,
                        disk_path=disk_path,
                    )
                    runtime.cache_artifact_index(
                        ArtifactCacheEntry(
                            artifact_id=artifact_id,
                            canonical_index_bytes=canonical_index_bytes,
                            parsed_index=canonical_index,
                            generation=generation,
                            disk_path=disk_path,
                            expires_at=time.monotonic(),
                        )
                    )
                elif generation is not None and self._generation is None:
                    self._generation = generation
                self._artifact_id = artifact_id
                if disk_path and not self._disk_path_hint:
                    self._disk_path_hint = disk_path
                return artifact_id
            raise ArtifactError(
                "Artifact handle missing identity",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

    def _resolve_metadata_from_disk(
        self, runtime: "StoreRuntimeContext", disk_path: str
    ) -> CanonicalIndex | None:
        cached = runtime.get_artifact_index_by_disk_path(disk_path)
        if cached:
            with self._lock:
                self._hydrate_from_cache_entry(cached)
                if (
                    self._canonical_index is not None
                    and self._canonical_index_bytes is not None
                ):
                    return self._canonical_index

        verify_checksums = True
        if self._fallback is not None:
            verify_checksums = bool(self._fallback.verify_checksums)
        try:
            resolved = runtime.ensure_client().resolve_artifact_from_disk_v2(
                disk_path=disk_path,
                verify_checksums=verify_checksums,
            )
        except Exception as exc:  # noqa: BLE001
            raise_mapped_materialization_error(exc)

        resolved_artifact_id = (
            getattr(resolved, "artifact_id", "") or self._artifact_id or None
        )
        resolved_disk_path = getattr(resolved, "disk_path", "") or disk_path
        canonical_index_bytes = bytes(
            getattr(resolved, "canonical_index_bytes", b"") or b""
        )
        generation_raw = getattr(resolved, "generation", 0)
        generation = int(generation_raw) if generation_raw else None

        canonical_index = (
            canonical_index_from_bytes(canonical_index_bytes)
            if canonical_index_bytes
            else None
        )
        with self._lock:
            if resolved_artifact_id and not self._artifact_id:
                self._artifact_id = resolved_artifact_id
            if resolved_disk_path:
                self._disk_path_hint = resolved_disk_path
            if canonical_index is not None:
                self._set_metadata(
                    canonical_index_bytes,
                    canonical_index,
                    generation=self._generation or generation,
                    disk_path=resolved_disk_path,
                )
            elif generation is not None and self._generation is None:
                self._generation = generation

        if canonical_index is not None and resolved_artifact_id:
            runtime.cache_artifact_index(
                ArtifactCacheEntry(
                    artifact_id=resolved_artifact_id,
                    canonical_index_bytes=canonical_index_bytes,
                    parsed_index=canonical_index,
                    generation=generation,
                    disk_path=resolved_disk_path,
                    expires_at=time.monotonic(),
                )
            )
            return canonical_index
        return None

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
        cache_disk_path: str | None = None
        generation_hint: int | None = None
        disk_path_hint: str | None = None
        force_remote_fetch = False
        with self._lock:
            if (
                self._canonical_index is not None
                and self._canonical_index_bytes is not None
            ):
                return self._canonical_index
            cached = runtime.get_artifact_index_cached(artifact_id)
            if cached:
                has_disk_mismatch = bool(
                    self._disk_path_hint
                    and cached.disk_path
                    and cached.disk_path != self._disk_path_hint
                )
                has_generation_mismatch = bool(
                    self._generation is not None
                    and cached.generation is not None
                    and cached.generation != self._generation
                )
                force_remote_fetch = bool(has_disk_mismatch or has_generation_mismatch)
                if force_remote_fetch:
                    runtime.invalidate_artifact(
                        artifact_id,
                        reason="disk_path_mismatch"
                        if has_disk_mismatch
                        else "generation_mismatch",
                    )
                else:
                    self._hydrate_from_cache_entry(cached)
                    assert self._canonical_index is not None
                    return self._canonical_index
            generation_hint = self._generation
            disk_path_hint = self._disk_path_hint or (
                self._fallback.disk_path if self._fallback else None
            )
        if disk_path_hint and not force_remote_fetch:
            try:
                disk_index = self._resolve_metadata_from_disk(runtime, disk_path_hint)
            except ArtifactError as disk_error:
                if disk_error.status_code != "NOT_FOUND":
                    raise
            else:
                if disk_index is not None:
                    return disk_index
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
                cache_disk_path = self._disk_path_hint
            else:
                generation_value = self._generation
                if generation_value is None:
                    generation_value = generation_hint
                disk_path_value = self._disk_path_hint or disk_path_hint
                self._set_metadata(
                    canonical_index_bytes,
                    canonical_index,
                    generation=generation_value,
                    disk_path=disk_path_value,
                )
                result_index = canonical_index
                cache_bytes = canonical_index_bytes
                cache_generation = self._generation
                cache_disk_path = self._disk_path_hint
        runtime.cache_artifact_index(
            ArtifactCacheEntry(
                artifact_id=artifact_id,
                canonical_index_bytes=cache_bytes,
                parsed_index=result_index,
                generation=cache_generation,
                disk_path=cache_disk_path,
                expires_at=time.monotonic(),
            )
        )
        return result_index

    def _effective_index(self) -> CanonicalIndex:
        base_index = self._ensure_metadata()
        if self._view_metadata is not None:
            return self._view_metadata.canonical_index
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
            parent_spec=self._view_spec,
            child_spec=child_spec,
            parent_depth=self._view_depth,
            subset_names=subset,
        )
        return Artifact(
            store_ref=self._store_ref,
            artifact_id=self._artifact_id,
            key=self._key_hint,
            disk_path=self._disk_path_hint,
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
            disk_path=entry.disk_path,
        )

    def _set_metadata(
        self,
        canonical_index_bytes: bytes,
        canonical_index: CanonicalIndex,
        *,
        generation: int | None,
        disk_path: str | None,
    ) -> None:
        self._canonical_index_bytes = canonical_index_bytes
        self._canonical_index = canonical_index
        self._generation = generation
        if disk_path and not self._disk_path_hint:
            self._disk_path_hint = disk_path
        effective_index = (
            self._view_metadata.canonical_index
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
        disk_path = getattr(payload, "disk_path", None)

        if canonical_index_bytes:
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            with self._lock:
                if self._canonical_index is None:
                    self._set_metadata(
                        canonical_index_bytes,
                        canonical_index,
                        generation=generation,
                        disk_path=disk_path,
                    )
            runtime.cache_artifact_index(
                ArtifactCacheEntry(
                    artifact_id=artifact_id,
                    canonical_index_bytes=canonical_index_bytes,
                    parsed_index=canonical_index,
                    generation=generation,
                    disk_path=disk_path,
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
                view_id=str(resolved_view_id or view_hash),
                view_index_bytes=view_index_bytes,
                view_data_hash=str(view_hash),
                tensor_names=subset_names,
                nbytes=sum(entry.size_bytes for entry in view_index.entries),
                canonical_index=view_index,
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
    "PlacementPin",
    "PrefetchedReplica",
    "TensorMeta",
]
