#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

from tensorcast._c_ext import compute_view_index_bytes
from tensorcast.api import _metrics as store_metrics
from tensorcast.api._device import device_uuid_for, resolve_device
from tensorcast.api._view_ops import (
    SliceSpec,
    ViewSpecBuildResult,
    build_view_spec,
    validate_narrow,
)
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.inplace_slot import InplaceSlot
from tensorcast.api.store.materialization import _build_source_policy
from tensorcast.api.store.retry import map_materialization_error
from tensorcast.api.store.types import ArtifactError, FallbackOptions
from tensorcast.api.store.view_composer import ViewSpecComposer
from tensorcast.proto.daemon.v2 import store_daemon_pb2

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.handles import RegisteredArtifact
    from tensorcast.api.store.runtime import StoreRuntimeContext
    from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry


_ALIGNMENT_BYTES = 8


@dataclass(frozen=True, slots=True)
class DeferredCommitResult:
    tensor_names: tuple[str, ...]
    view_id: str | None
    view_subset_hash: bytes | None
    storage_ids: tuple[str, ...]
    logical_size_bytes: int
    published_artifact: "RegisteredArtifact | None" = None


@dataclass(frozen=True, slots=True)
class _TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int
    offset_bytes: int


def _align_bytes(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


def _compact_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride: list[int] = []
    acc = 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= max(1, int(dim))
    return tuple(reversed(stride))


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        dim_value = int(dim)
        if dim_value <= 0:
            return 0
        total *= dim_value
    return total


class DeferredLoader:
    """Allocate CUDA placeholders for slices and materialize once at commit()."""

    def __init__(
        self,
        *,
        artifact: "Artifact",
        device: torch.device | str,
        packing: str = "append",
        capacity_bytes: int | None = None,
    ) -> None:
        store, runtime, pipeline = artifact._require_components()
        artifact._ensure_metadata()

        self._store: Store = store
        self._runtime: StoreRuntimeContext = runtime
        self._pipeline = pipeline
        self._artifact = artifact
        self._fallback: FallbackOptions | None = artifact._fallback

        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise ArtifactError(
                "DeferredLoader requires a CUDA device",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._device = device_obj
        self._device_id = resolve_device(device_obj, allow_cpu=False)

        mode = str(packing).strip().lower()
        if mode not in {"append", "plan", "byte_space"}:
            raise ArtifactError(
                "packing must be 'append', 'plan', or 'byte_space'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._packing = mode

        base_index = artifact._effective_index()
        self._base_index: CanonicalIndex = base_index
        canonical_index = artifact._canonical_index
        self._canonical_index_bytes = artifact._canonical_index_bytes
        if canonical_index is None:
            raise ArtifactError(
                "Missing canonical index for deferred materialization",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        self._canonical_index: CanonicalIndex = canonical_index
        self._entries_by_name: dict[str, CanonicalIndexEntry] = {
            entry.name: entry for entry in base_index.entries
        }
        self._canonical_entries_by_name: dict[str, CanonicalIndexEntry] = {
            entry.name: entry for entry in canonical_index.entries
        }
        if self._canonical_index_bytes is None:
            raise ArtifactError(
                "Missing canonical index bytes for deferred materialization",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self._entries_by_name:
            raise ArtifactError(
                "Artifact canonical index is empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        base_spec = artifact._view_spec
        if base_spec is not None and base_spec.has_transpose:
            raise ArtifactError(
                "DeferredLoader only supports narrow view operations",
                status_code="UNIMPLEMENTED",
                retryable=False,
            )
        self._base_view_spec = base_spec
        self._base_view_depth = artifact._view_depth
        self._slice_specs: dict[str, SliceSpec] = {}

        self._byte_space_entries: dict[str, CanonicalIndexEntry] = {}
        self._byte_space_logical_size: int | None = None
        self._byte_space_order: list[str] = []
        self._byte_space_full_selection: bool | None = None
        view_metadata = artifact._view_metadata
        self._view_index_hint: bytes | None = (
            bytes(view_metadata.view_index_bytes)
            if view_metadata is not None and view_metadata.view_index_bytes
            else None
        )

        if capacity_bytes is None:
            if self._packing == "byte_space":
                self._resolve_byte_space_index()
                capacity = int(
                    self._byte_space_logical_size
                    if self._byte_space_logical_size is not None
                    else base_index.total_size_bytes
                )
            else:
                padding = max(0, len(base_index.entries) - 1) * (_ALIGNMENT_BYTES - 1)
                capacity = int(base_index.total_size_bytes) + int(padding)
        else:
            capacity = int(capacity_bytes)
        if capacity <= 0:
            raise ArtifactError(
                "capacity_bytes must be positive",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._capacity_bytes = capacity

        self._cursor_bytes = 0
        self._order: list[str] = []
        self._offsets: dict[str, int] = {}
        self._specs: dict[str, _TensorSpec] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        self._element_size_cache: dict[torch.dtype, int] = {}

        self._arena: torch.Tensor | None = None
        self._arena_storage: torch.UntypedStorage | None = None
        self._region_id: str | None = None

        self._planned = False
        self._committed = False
        self._closed = False

    def __enter__(self) -> "DeferredLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def device(self) -> torch.device:
        return self._device

    def plan(
        self,
        tensor_names: Sequence[str],
        *,
        slices: Mapping[str, SliceSpec] | None = None,
    ) -> None:
        self._ensure_open()
        if self._packing != "plan":
            raise ArtifactError(
                "plan() is only valid when packing='plan'",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._planned:
            raise ArtifactError(
                "DeferredLoader plan already finalized",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._tensors:
            raise ArtifactError(
                "plan() must be called before tensor()",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        ordered = [str(name) for name in tensor_names]
        if not ordered:
            raise ArtifactError(
                "plan() requires at least one tensor name",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if len(set(ordered)) != len(ordered):
            raise ArtifactError(
                "plan() tensor_names must be unique",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        for name in ordered:
            if name not in self._entries_by_name:
                raise ArtifactError(
                    f"Unknown tensor '{name}' in plan()",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )

        ordered_set = set(ordered)
        if slices:
            for name, slice_spec in slices.items():
                if name not in self._entries_by_name:
                    raise ArtifactError(
                        f"Unknown tensor '{name}' in plan() slices",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                if name not in ordered_set:
                    raise ArtifactError(
                        f"Tensor '{name}' in plan() slices is not in tensor_names",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                self._validate_slice(name, slice_spec)
                self._slice_specs[name] = slice_spec

        cursor = 0
        offsets: dict[str, int] = {}
        specs: dict[str, _TensorSpec] = {}
        for name in ordered:
            spec = self._build_tensor_spec(name, self._slice_specs.get(name))
            offset = _align_bytes(cursor, _ALIGNMENT_BYTES)
            if offset % self._element_size(spec.dtype) != 0:
                raise ArtifactError(
                    f"Aligned offset for '{name}' is not divisible by dtype size",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            cursor = offset + spec.size_bytes
            offsets[name] = offset
            specs[name] = _TensorSpec(
                shape=spec.shape,
                stride=spec.stride,
                dtype=spec.dtype,
                size_bytes=spec.size_bytes,
                offset_bytes=offset,
            )

        if cursor > self._capacity_bytes:
            raise ArtifactError(
                "DeferredLoader plan exceeds capacity_bytes",
                status_code="RESOURCE_EXHAUSTED",
                retryable=False,
            )

        self._order = list(ordered)
        self._offsets = offsets
        self._specs = specs
        self._cursor_bytes = cursor
        self._planned = True
        self._ensure_arena()

    def tensor(self, name: str, *, slice: SliceSpec | None = None) -> torch.Tensor:
        self._ensure_open()
        if self._committed:
            raise ArtifactError(
                "DeferredLoader already committed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        tensor_name = str(name)
        existing = self._tensors.get(tensor_name)
        if existing is not None:
            if slice is not None:
                prior = self._slice_specs.get(tensor_name)
                if prior is None or prior != slice:
                    raise ArtifactError(
                        f"Tensor '{tensor_name}' already requested with a different slice",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
            return existing

        if tensor_name not in self._entries_by_name:
            raise ArtifactError(
                f"Unknown tensor '{tensor_name}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        if self._packing == "plan" and not self._planned:
            raise ArtifactError(
                "packing='plan' requires calling plan() before tensor()",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        if slice is not None:
            if self._packing == "byte_space":
                raise ArtifactError(
                    "packing='byte_space' does not support per-tensor slices; use Artifact.view instead",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if self._packing == "plan" and self._planned:
                planned = self._slice_specs.get(tensor_name)
                if planned is None or planned != slice:
                    raise ArtifactError(
                        f"Tensor '{tensor_name}' slice does not match the plan",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
            else:
                self._validate_slice(tensor_name, slice)
                self._slice_specs[tensor_name] = slice

        spec = self._specs.get(tensor_name)
        if spec is None:
            spec = self._build_tensor_spec(
                tensor_name, self._slice_specs.get(tensor_name)
            )

        offset = 0
        end = 0
        if self._packing == "byte_space":
            self._resolve_byte_space_index()
            entry = self._byte_space_entries.get(tensor_name)
            if entry is None:
                raise ArtifactError(
                    f"Tensor '{tensor_name}' is not part of the byte_space layout",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            offset = int(entry.segment_offset)
            end = offset + spec.size_bytes
            if end > self._capacity_bytes:
                raise ArtifactError(
                    "DeferredLoader arena exhausted; increase capacity_bytes",
                    status_code="RESOURCE_EXHAUSTED",
                    retryable=False,
                )
            spec = _TensorSpec(
                shape=spec.shape,
                stride=spec.stride,
                dtype=spec.dtype,
                size_bytes=spec.size_bytes,
                offset_bytes=offset,
            )
        elif self._packing == "append":
            offset = _align_bytes(self._cursor_bytes, _ALIGNMENT_BYTES)
            if offset % self._element_size(spec.dtype) != 0:
                raise ArtifactError(
                    f"Aligned offset for '{tensor_name}' is not divisible by dtype size",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            end = offset + spec.size_bytes
            if end > self._capacity_bytes:
                raise ArtifactError(
                    "DeferredLoader arena exhausted; increase capacity_bytes",
                    status_code="RESOURCE_EXHAUSTED",
                    retryable=False,
                )
            spec = _TensorSpec(
                shape=spec.shape,
                stride=spec.stride,
                dtype=spec.dtype,
                size_bytes=spec.size_bytes,
                offset_bytes=offset,
            )
        else:
            if tensor_name not in self._offsets:
                raise ArtifactError(
                    f"Tensor '{tensor_name}' is not part of the plan",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            spec = self._specs[tensor_name]

        self._ensure_arena()
        tensor = self._materialize_tensor(spec)
        if self._packing == "append":
            self._cursor_bytes = end
            self._order.append(tensor_name)
            self._offsets[tensor_name] = offset
            self._specs[tensor_name] = spec
        elif self._packing == "byte_space":
            self._offsets[tensor_name] = offset
            self._specs[tensor_name] = spec
        self._tensors[tensor_name] = tensor
        return tensor

    def commit(self) -> InplaceSlot:
        self._ensure_open()
        if self._committed:
            raise ArtifactError(
                "DeferredLoader already committed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self._tensors:
            raise ArtifactError(
                "DeferredLoader has no tensors to materialize",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if self._packing == "plan":
            planned = set(self._order)
            if planned and planned != set(self._tensors):
                raise ArtifactError(
                    "DeferredLoader plan requires requesting every planned tensor",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        if self._packing == "byte_space":
            self._resolve_byte_space_index()
            expected = set(self._byte_space_entries)
            if expected and expected != set(self._tensors):
                raise ArtifactError(
                    "packing='byte_space' requires requesting every tensor in the layout",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )

        composed_spec = self._compose_view_spec()
        view_spec_proto = composed_spec.proto if composed_spec else None

        target = dict(self._tensors)
        artifact_id = self._artifact._ensure_identified()
        canonical_index = self._artifact._canonical_index
        canonical_index_bytes = self._artifact._canonical_index_bytes
        if canonical_index is None or canonical_index_bytes is None:
            raise ArtifactError(
                "Missing canonical index for deferred materialization",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        selection_order: tuple[str, ...] | None
        if self._packing == "byte_space":
            self._resolve_byte_space_index()
            if self._byte_space_full_selection:
                selection_order = None
            else:
                selection_order = tuple(self._byte_space_order)
        else:
            selection_order = tuple(self._order)
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
        allow_disk = True if self._fallback is None else bool(self._fallback.allow_disk)
        if effective_prefer == "local":
            allow_disk = False
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )

        client = self._runtime.ensure_client()
        operation_id = uuid.uuid4().hex
        response = None
        region_layout = None
        attempt = 0
        while attempt < 2:
            region_layout = self._pipeline._build_region_backed_layout(
                canonical_index=canonical_index,
                canonical_index_bytes=canonical_index_bytes,
                target=target,
                device_id=self._device_id,
                tensor_names=selection_order,
                view_spec=view_spec_proto,
                view_id=None,
                view_index_hint=None,
                selection_order=selection_order,
            )
            try:
                response = client.materialize_into_target_v2(
                    artifact_id=artifact_id,
                    target_layout=region_layout.layout,
                    device_uuid=device_uuid_for(self._device_id),
                    preference=preference,
                    source_policy=source_policy,
                    disk_path=disk_path,
                    verify_checksums=verify_checksums,
                    tensor_names=region_layout.selection_names,
                    view=view_spec_proto,
                    view_id=region_layout.view_id if view_spec_proto is None else None,
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
                    self._unregister_region()
                    self._ensure_arena()
                    attempt += 1
                    continue
                if error.status_code in {"DATA_LOSS", "FAILED_PRECONDITION"}:
                    self._unregister_region()
                raise ArtifactError(
                    str(error),
                    status_code=error.status_code,
                    retryable=False,
                ) from exc

            if (
                response.status
                != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
            ):
                self._unregister_region()
                raise ArtifactError(
                    "MaterializeIntoTarget returned non-success status",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            break
        if response is None or region_layout is None:
            self._unregister_region()
            raise ArtifactError(
                "MaterializeIntoTarget retry failed to produce a response",
                status_code="DATA_LOSS",
                retryable=False,
            )
        store_metrics.record_region_backed_verification_skipped(
            self._runtime.daemon_endpoint
        )

        self._committed = True
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
        target_write_token = getattr(response, "target_write_token", None)
        slot = InplaceSlot(
            store=self._store,
            runtime=self._runtime,
            pipeline=self._pipeline,
            tensors=target,
            device=self._device,
            device_id=self._device_id,
            region_id=self._region_id,
            region_layout=region_layout,
            view_spec=view_spec_proto,
            fallback=self._fallback,
            commit_result=commit_result,
            artifact_id=artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            target_write_token=target_write_token or None,
        )
        self._region_id = None
        return slot

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unregister_region()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactError(
                "DeferredLoader is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._runtime.closed:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

    def _ensure_arena(self) -> None:
        if self._arena is not None:
            if self._region_id is None:
                self._register_region(self._arena)
            return
        arena = torch.empty(
            (self._capacity_bytes,),
            dtype=torch.uint8,
            device=self._device,
        )
        self._arena = arena
        self._arena_storage = arena.untyped_storage()
        self._register_region(arena)

    def _register_region(self, arena: torch.Tensor) -> None:
        if self._region_id is not None:
            return
        ttl_ms = 0
        handle = self._store.register_vram_region(
            device_id=self._device_id,
            base_ptr=int(arena.data_ptr()),
            size_bytes=int(self._capacity_bytes),
            ttl_ms=int(ttl_ms),
        )
        self._region_id = handle.region_id

    def _unregister_region(self) -> None:
        if self._region_id is None:
            return
        region_id = self._region_id
        self._region_id = None
        with contextlib.suppress(Exception):
            self._store.unregister_vram_region(region_id)

    def _resolve_byte_space_index(self) -> None:
        if self._packing != "byte_space":
            return
        if self._byte_space_entries:
            return
        canonical_index_bytes = self._canonical_index_bytes
        if canonical_index_bytes is None:
            raise ArtifactError(
                "Missing canonical index bytes for byte_space layout",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        composed_spec = self._compose_view_spec()
        view_spec_proto = composed_spec.proto if composed_spec else None
        selected_names = [entry.name for entry in self._base_index.entries]
        canonical_names = set(self._canonical_entries_by_name)
        subset_payload: list[str] | None = None
        if canonical_names and set(selected_names) != canonical_names:
            subset_payload = selected_names

        normalized_ops: dict[str, list[dict[str, int | str]]] = {}
        if view_spec_proto is not None and view_spec_proto.tensors:
            for name, ops in view_spec_proto.tensors.items():
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

        index_bytes: bytes
        if not normalized_ops and subset_payload is None:
            index_bytes = canonical_index_bytes
        else:
            view_payload = compute_view_index_bytes(
                canonical_index_bytes, normalized_ops, subset_payload
            )
            index_bytes = bytes(view_payload["view_index_bytes"])
        view_index = canonical_index_from_bytes(index_bytes)
        self._byte_space_entries = {entry.name: entry for entry in view_index.entries}
        self._byte_space_order = [entry.name for entry in view_index.entries]
        logical_size = 0
        for entry in view_index.entries:
            logical_size = max(
                logical_size, int(entry.segment_offset) + int(entry.size_bytes)
            )
        self._byte_space_logical_size = logical_size
        canonical_names = set(self._canonical_entries_by_name)
        self._byte_space_full_selection = (
            set(self._byte_space_entries) == canonical_names
            if canonical_names
            else False
        )

    def _element_size(self, dtype: torch.dtype) -> int:
        cached = self._element_size_cache.get(dtype)
        if cached is not None:
            return cached
        size = int(torch.empty((), dtype=dtype).element_size())
        self._element_size_cache[dtype] = size
        return size

    def _validate_slice(self, name: str, spec: SliceSpec) -> None:
        entry = self._entries_by_name[name]
        try:
            validate_narrow(name, entry.shape, spec)
        except ValueError as exc:
            raise ArtifactError(
                str(exc),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ) from exc

    def _build_tensor_spec(self, name: str, spec: SliceSpec | None) -> _TensorSpec:
        entry = self._entries_by_name[name]
        shape = entry.shape
        stride = entry.stride
        narrow = None
        if spec is not None:
            try:
                narrow = validate_narrow(name, entry.shape, spec)
            except ValueError as exc:
                raise ArtifactError(
                    str(exc),
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                ) from exc
            if narrow is not None:
                shape = tuple(
                    narrow.length if idx == narrow.dim else int(dim)
                    for idx, dim in enumerate(entry.shape)
                )
                stride = _compact_stride(shape)
        if not stride:
            stride = _compact_stride(shape)
        if tuple(stride) != _compact_stride(shape):
            raise ArtifactError(
                f"Tensor '{name}' is not contiguous; deferred loader requires contiguous layouts",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        numel = _numel(shape)
        if numel <= 0:
            raise ArtifactError(
                f"Tensor '{name}' has invalid shape for deferred materialization",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if spec is None or narrow is None:
            size_bytes = int(entry.size_bytes)
        else:
            size_bytes = numel * self._element_size(entry.dtype)
        return _TensorSpec(
            shape=tuple(int(dim) for dim in shape),
            stride=tuple(int(dim) for dim in stride),
            dtype=entry.dtype,
            size_bytes=int(size_bytes),
            offset_bytes=0,
        )

    def _materialize_tensor(self, spec: _TensorSpec) -> torch.Tensor:
        if self._arena_storage is None:
            raise ArtifactError(
                "DeferredLoader arena is not initialized",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        offset_elems = spec.offset_bytes // self._element_size(spec.dtype)
        tensor = torch.empty((), device=self._device, dtype=spec.dtype)
        tensor.set_(
            self._arena_storage,
            int(offset_elems),
            spec.shape,
            spec.stride,
        )
        return tensor

    def _compose_view_spec(self) -> ViewSpecBuildResult | None:
        if not self._slice_specs and (
            self._base_view_spec is None or self._base_view_spec.is_identity
        ):
            return None
        entry_shapes = {
            entry.name: tuple(entry.shape) for entry in self._base_index.entries
        }
        if self._slice_specs:
            build_spec = build_view_spec(
                entry_shapes=entry_shapes,
                slices=self._slice_specs,
                transpose=None,
            )
            if build_spec.has_transpose:
                raise ArtifactError(
                    "DeferredLoader only supports narrow view operations",
                    status_code="UNIMPLEMENTED",
                    retryable=False,
                )
        else:
            build_spec = None
        composer = ViewSpecComposer()
        composed_spec, _, _ = composer.compose(
            canonical_index=self._base_index,
            parent_spec=self._base_view_spec,
            child_spec=build_spec,
            parent_depth=self._base_view_depth,
            subset_names=None,
        )
        if composed_spec is None or composed_spec.is_identity:
            return None
        return composed_spec


__all__ = ["DeferredCommitResult", "DeferredLoader"]
