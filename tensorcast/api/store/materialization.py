#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from typing import TypedDict

import torch
from opentelemetry.trace import Status, StatusCode

from tensorcast._C import restore_tensors
from tensorcast.api import _metrics as store_metrics
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._materialize import (
    MaterializationPayload,
    TensorPayloadDescriptor,
    materialize_artifact_v2,
)
from tensorcast.api.store.async_ops import ArtifactFuture, TrackedExecutor
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
)
from tensorcast.api.store.retry import (
    compute_retry_delay,
    map_materialization_error,
    remaining_budget,
    should_retry,
)
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.types import (
    ArtifactError,
    FallbackOptions,
    SpanAttributeValue,
)
from tensorcast.api.store.views import (
    ResolvedViewInputs,
    TransformPlacement,
    ViewOrchestrator,
)
from tensorcast.proto.daemon.v1 import store_daemon_pb2

logger = logging.getLogger(__name__)


class _MaterializationSummary(TypedDict):
    count: int | None
    bytes: int | None
    selection: str | None


def _source_label(
    source: store_daemon_pb2.MaterializationSource | None,
) -> str | None:
    mapping = {
        store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_DISK: "disk",
        store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_P2P: "p2p",
        store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_LOCAL_REPLICA: "local_replica",
    }
    if source is None:
        return None
    return mapping.get(source)


class FallbackResolver:
    """Disk/P2P fallback utilities shared by materialization flows."""

    def __init__(self, runtime: StoreRuntimeContext) -> None:
        self._runtime = runtime

    def ensure_supported(self, fallback: FallbackOptions | None) -> None:
        if fallback and fallback.disk_path and fallback.disk_path.strip() == "":
            raise ArtifactError(
                "Fallback disk_path must not be empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if fallback and fallback.prefer not in {"auto", "local", "p2p", "disk"}:
            raise ArtifactError(
                f"Unknown fallback preference '{fallback.prefer}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

    def resolve_disk_path(
        self,
        *,
        fallback: FallbackOptions,
        client,
        key: str | None,
        artifact_id: str | None,
    ) -> tuple[str | None, str | None, Exception | None]:
        disk_path = fallback.disk_path
        resolved_artifact_id = artifact_id
        if disk_path:
            return disk_path, resolved_artifact_id, None
        if key is None:
            return None, resolved_artifact_id, None
        try:
            resolved_id, mapped_path = self._runtime.resolve_key_mapping_cached(key=key)
        except Exception as exc:  # noqa: BLE001
            return None, resolved_artifact_id, exc
        if mapped_path:
            if not resolved_artifact_id:
                resolved_artifact_id = resolved_id or resolved_artifact_id
            return mapped_path, resolved_artifact_id, None
        return None, resolved_artifact_id, None

    def record_event(
        self,
        *,
        mode: str,
        artifact_id: str | None,
        key: str | None,
        detail: Mapping[str, object],
    ) -> None:
        logger.info(
            "store.fallback",
            extra={
                "tc.store.daemon": self._runtime.daemon_endpoint,
                "tc.store.mode": mode,
                "tc.store.artifact_id": artifact_id or "",
                "tc.store.key": key or "",
                "tc.store.detail": dict(detail),
            },
        )


class MaterializationPipeline:
    """Retrieval orchestration for get/get_into/get_view flows."""

    def __init__(
        self,
        runtime: StoreRuntimeContext,
        views: ViewOrchestrator,
        *,
        materialize_fn: Callable[..., MaterializationPayload] = materialize_artifact_v2,
    ) -> None:
        self._runtime = runtime
        self._views = views
        self._fallback = FallbackResolver(runtime)
        self._executor = TrackedExecutor(runtime)
        self._materialize_fn = materialize_fn

    def set_materialize_fn(
        self, materialize_fn: Callable[..., MaterializationPayload]
    ) -> None:
        self._materialize_fn = materialize_fn

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def get(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
        options: GetArtifactOptions | None = None,
        tensor_names: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        payload, device_id = self._perform_get_with_retry(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            method="get",
            cancel_event=None,
            options_override=options,
            tensor_names=tensor_names,
        )
        try:
            state = self._payload_state_dict(payload, device_id=device_id)
            if tensor_names:
                missing = [name for name in tensor_names if name not in state]
                if missing:
                    raise ArtifactError(
                        f"Materialization payload missing tensors: {', '.join(missing)}",
                        status_code="NOT_FOUND",
                        retryable=False,
                    )
                state = {name: state[name] for name in tensor_names}
            return state
        finally:
            self._release_materialized(payload, self._runtime.ensure_client())

    def materialize_subset(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device: torch.device | str | None,
        fallback: FallbackOptions | None,
        tensor_names: Sequence[str] | None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
        view_spec: store_daemon_pb2.ViewSpec | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        options: GetArtifactOptions | None = None,
    ) -> tuple[MaterializationPayload, int]:
        return self._perform_get_with_retry(
            method="get",
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            cancel_event=None,
            options_override=options,
            canonical_index_hint=canonical_index_hint,
            disk_path_hint=disk_path_hint,
            tensor_names=tensor_names,
            view=view_spec,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
        )

    def get_view(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        view_id: str | None = None,
        placement: str | None = None,
        device: torch.device | str | None = None,
        options: GetArtifactOptions | None = None,
        resolver: Callable[..., ResolvedViewInputs] | None = None,
    ) -> dict[str, torch.Tensor]:
        view_resolver = resolver or self._views.resolve_view_inputs
        resolved = self._call_view_resolver(
            view_resolver,
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
        )
        placement_enum: TransformPlacement | None = None
        if resolved.view_spec is not None or placement is not None:
            placement_enum = self._views.resolve_transform_placement(
                placement, has_transpose=resolved.has_transpose
            )

        payload, device_id = self._perform_get_with_retry(
            method="get_view",
            artifact_id=resolved.artifact_id,
            key=None,
            device=device,
            fallback=None,
            cancel_event=None,
            options_override=options,
            view=resolved.view_spec,
            view_id=resolved.view_id,
            placement=placement_enum,
            canonical_index_hint=resolved.canonical_index_bytes,
            disk_path_hint=resolved.disk_path_hint,
        )
        try:
            return self._payload_state_dict(payload, device_id=device_id)
        finally:
            self._release_materialized(payload, self._runtime.ensure_client())

    def get_async(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
        options: GetArtifactOptions | None = None,
        tensor_names: Sequence[str] | None = None,
    ) -> ArtifactFuture[dict[str, torch.Tensor]]:
        cancel_event = threading.Event()
        mat_lock = threading.Lock()
        mat_ref: dict[str, MaterializationPayload | None] = {"value": None}

        def _task() -> dict[str, torch.Tensor]:
            materialized: MaterializationPayload | None = None
            try:
                materialized, device_id = self._perform_get_with_retry(
                    artifact_id=artifact_id,
                    key=key,
                    device=device,
                    fallback=fallback,
                    method="get",
                    cancel_event=cancel_event,
                    options_override=options,
                    tensor_names=tensor_names,
                )
                with mat_lock:
                    mat_ref["value"] = materialized
                if cancel_event.is_set():
                    with mat_lock:
                        mat_ref["value"] = None
                    self._release_materialized(
                        materialized, self._runtime.ensure_client()
                    )
                    materialized = None
                    raise CancelledError
                return self._payload_state_dict(materialized, device_id=device_id)
            except CancelledError as exc:
                raise ArtifactError(
                    "Retrieval cancelled",
                    status_code="CANCELLED",
                    retryable=False,
                ) from exc
            except ArtifactError as exc:
                with mat_lock:
                    mat_ref["value"] = None
                raise exc
            finally:
                with mat_lock:
                    mat_ref["value"] = None
                if materialized is not None:
                    self._release_materialized(
                        materialized, self._runtime.ensure_client()
                    )
                    materialized = None

        def _cancel() -> bool:
            if cancel_event.is_set():
                return False
            cancel_event.set()
            with mat_lock:
                materialized = mat_ref["value"]
                mat_ref["value"] = None
            if materialized is not None:
                self._release_materialized(materialized, self._runtime.ensure_client())
            return True

        return self._executor.submit(_task, cancel_callback=_cancel)

    def get_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
        options: GetArtifactOptions | None = None,
        tensor_names: Sequence[str] | None = None,
        view_spec: store_daemon_pb2.ViewSpec | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
    ) -> None:
        payload, device_id = self._perform_get_with_retry(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            method="get_into",
            cancel_event=None,
            options_override=options,
            tensor_names=tensor_names,
            view=view_spec,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
        )
        try:
            layout_bytes = payload.view_index_bytes or payload.canonical_index_bytes
            self._stream_into_targets(
                payload=payload,
                target=target,
                device_id=device_id,
                layout_bytes=layout_bytes,
            )
        finally:
            self._release_materialized(payload, self._runtime.ensure_client())

    def get_into_async(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
        options: GetArtifactOptions | None = None,
        tensor_names: Sequence[str] | None = None,
    ) -> ArtifactFuture[None]:
        cancel_event = threading.Event()
        mat_lock = threading.Lock()
        mat_ref: dict[str, MaterializationPayload | None] = {"value": None}

        def _task() -> None:
            materialized: MaterializationPayload | None = None
            try:
                materialized, device_id = self._perform_get_with_retry(
                    artifact_id=artifact_id,
                    key=key,
                    device=device,
                    fallback=fallback,
                    method="get_into",
                    cancel_event=cancel_event,
                    options_override=options,
                    tensor_names=tensor_names,
                )
                with mat_lock:
                    mat_ref["value"] = materialized
                if cancel_event.is_set():
                    with mat_lock:
                        mat_ref["value"] = None
                    self._release_materialized(
                        materialized, self._runtime.ensure_client()
                    )
                    materialized = None
                    raise CancelledError
                try:
                    layout_bytes = (
                        materialized.view_index_bytes
                        or materialized.canonical_index_bytes
                    )
                    self._stream_into_targets(
                        payload=materialized,
                        target=target,
                        device_id=device_id,
                        layout_bytes=layout_bytes,
                        cancel_event=cancel_event,
                    )
                finally:
                    if materialized is not None:
                        self._release_materialized(
                            materialized, self._runtime.ensure_client()
                        )
                        materialized = None
            except CancelledError as exc:
                raise ArtifactError(
                    "Retrieval cancelled",
                    status_code="CANCELLED",
                    retryable=False,
                ) from exc
            except ArtifactError as exc:
                with mat_lock:
                    mat_ref["value"] = None
                raise exc
            finally:
                with mat_lock:
                    mat_ref["value"] = None

        def _cancel() -> bool:
            if cancel_event.is_set():
                return False
            cancel_event.set()
            with mat_lock:
                materialized = mat_ref["value"]
                mat_ref["value"] = None
            if materialized is not None:
                self._release_materialized(materialized, self._runtime.ensure_client())
            return True

        return self._executor.submit(_task, cancel_callback=_cancel)

    def get_view_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        view_id: str | None = None,
        placement: str | None = None,
        device: torch.device | str | None = None,
        options: GetArtifactOptions | None = None,
        resolver: Callable[..., ResolvedViewInputs] | None = None,
    ) -> None:
        view_resolver = resolver or self._views.resolve_view_inputs
        resolved = self._call_view_resolver(
            view_resolver,
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
        )
        placement_enum: TransformPlacement | None = None
        if resolved.view_spec is not None or placement is not None:
            placement_enum = self._views.resolve_transform_placement(
                placement, has_transpose=resolved.has_transpose
            )

        payload, device_id = self._perform_get_with_retry(
            method="get_view_into",
            artifact_id=resolved.artifact_id,
            key=None,
            device=device,
            fallback=None,
            cancel_event=None,
            options_override=options,
            view=resolved.view_spec,
            view_id=resolved.view_id,
            placement=placement_enum,
            canonical_index_hint=resolved.canonical_index_bytes,
            disk_path_hint=resolved.disk_path_hint,
        )
        try:
            layout_bytes = payload.view_index_bytes or payload.canonical_index_bytes
            self._stream_into_targets(
                payload=payload,
                target=target,
                device_id=device_id,
                layout_bytes=layout_bytes,
            )
        finally:
            self._release_materialized(payload, self._runtime.ensure_client())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_identifiers(
        artifact_id: str | None,
        key: str | None,
        fallback: FallbackOptions | None,
        *,
        disk_path_hint: str | None = None,
    ) -> tuple[str | None, str | None]:
        if artifact_id and key:
            raise ArtifactError(
                "Specify either artifact_id or key, not both",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not artifact_id and not key:
            disk_path: str | None = None
            if fallback:
                if isinstance(fallback.disk_path, str):
                    disk_path = fallback.disk_path.strip()
                else:
                    disk_path = fallback.disk_path
            if disk_path or disk_path_hint:
                return None, None
            raise ArtifactError(
                "Either artifact_id or key is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return artifact_id, key

    def _resolve_device_selector(
        self,
        selector: torch.device | str | None,
        fallback: FallbackOptions | None,
    ) -> int:
        def _has_disk_fallback() -> bool:
            if not fallback:
                return False
            disk = fallback.disk_path
            if isinstance(disk, str):
                return disk.strip() != ""
            return disk is not None

        if selector is None:
            if not torch.cuda.is_available():
                if _has_disk_fallback():
                    # Disk fallback provides the bytes for CPU materialization.
                    return 0
                raise ArtifactError(
                    "CUDA device required for retrieval",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            return int(torch.cuda.current_device())
        if isinstance(selector, torch.device):
            if selector.type != "cuda":
                if selector.type == "cpu" and _has_disk_fallback():
                    return 0
                raise ArtifactError(
                    f"Unsupported device selector {selector}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return int(selector.index or 0)
        if isinstance(selector, str):
            device = torch.device(selector)
            if device.type != "cuda":
                if device.type == "cpu" and _has_disk_fallback():
                    return 0
                raise ArtifactError(
                    f"Unsupported device selector {selector}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return int(device.index or 0)
        return int(selector)

    def _payload_state_dict(
        self, payload: MaterializationPayload, *, device_id: int | None = None
    ) -> dict[str, torch.Tensor]:
        if payload.descriptors and payload.state_dict is not None:
            missing = [
                d.name for d in payload.descriptors if d.name not in payload.state_dict
            ]
            if missing:
                raise ArtifactError(
                    f"Materialization payload missing tensors: {', '.join(missing)}",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            return {d.name: payload.state_dict[d.name] for d in payload.descriptors}
        state: dict[str, torch.Tensor] = {}
        for desc, tensor_view in payload.payload_iter():
            state[desc.name] = self._tensor_from_segment(
                desc, tensor_view, device_id=device_id
            )
        return state

    def _stream_into_targets(
        self,
        *,
        payload: MaterializationPayload,
        target: Mapping[str, torch.Tensor],
        device_id: int,
        layout_bytes: bytes,
        cancel_event: threading.Event | None = None,
    ) -> None:
        canonical_index = canonical_index_from_bytes(layout_bytes)
        expected: dict[str, torch.Tensor] = {}
        target_device = (
            next(iter(target.values())).device
            if target
            else torch.device("cuda", device_id)
        )
        expected_device = target_device
        if target_device.type == "cuda":
            expected_device = torch.device("cuda", device_id)
        for entry in canonical_index.entries:
            tensor_name = entry.name
            if tensor_name not in target:
                raise ArtifactError(
                    f"Target tensor '{tensor_name}' missing",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            tgt = target[tensor_name]
            if not isinstance(tgt, torch.Tensor):
                raise ArtifactError(
                    f"Target '{tensor_name}' must be a tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tgt.device != expected_device:
                raise ArtifactError(
                    f"Target tensor '{tensor_name}' on {tgt.device}, expected {expected_device}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tuple(tgt.shape) != entry.shape:
                raise ArtifactError(
                    f"Target tensor '{tensor_name}' shape {tuple(tgt.shape)} != {entry.shape}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tgt.dtype != entry.dtype:
                raise ArtifactError(
                    f"Target tensor '{tensor_name}' dtype {tgt.dtype} != {entry.dtype}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            expected[tensor_name] = tgt

        seen: set[str] = set()
        for desc, tensor in payload.payload_iter():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError
            target_tensor = expected.get(desc.name)
            if target_tensor is None:
                raise ArtifactError(
                    f"Unexpected tensor '{desc.name}' in materialization payload",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            source_tensor = self._tensor_from_segment(desc, tensor, device_id=device_id)
            target_tensor.copy_(source_tensor)
            seen.add(desc.name)
        missing = [name for name in expected if name not in seen]
        if missing:
            raise ArtifactError(
                f"Materialization payload missing tensors: {', '.join(missing)}",
                status_code="DATA_LOSS",
                retryable=False,
            )

    def _tensor_from_segment(
        self,
        desc: TensorPayloadDescriptor,
        segment: memoryview | torch.Tensor,
        *,
        device_id: int | None,
    ) -> torch.Tensor:
        if isinstance(segment, torch.Tensor):
            return segment
        if device_id is None:
            raise ArtifactError(
                "Materialization payload missing device context",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        try:
            start_ptr = ctypes.addressof(ctypes.c_char.from_buffer(segment))
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                "Materialization payload is not backed by a writable buffer",
                status_code="DATA_LOSS",
                retryable=False,
            ) from exc
        base_ptr = int(start_ptr - int(desc.buffer_offset))
        meta_state_dict = {
            desc.name: (
                list(desc.shape),
                list(desc.stride),
                desc.dtype,
                int(desc.storage_offset),
            )
        }
        tensor_offsets = {device_id: {desc.name: int(desc.buffer_offset)}}
        memory_ptrs = {device_id: base_ptr}
        tensors = restore_tensors(meta_state_dict, memory_ptrs, tensor_offsets, True)
        tensor = tensors.get(desc.name)
        if tensor is None:
            raise ArtifactError(
                f"Tensor '{desc.name}' missing from materialized payload",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return tensor

    def _summarize_materialized(
        self,
        materialized: MaterializationPayload,
        requested_names: Sequence[str] | None,
    ) -> _MaterializationSummary:
        summary: _MaterializationSummary = {
            "count": None,
            "bytes": None,
            "selection": None,
        }
        descriptor_names: list[str] = []
        bytes_total = 0
        if materialized.descriptors:
            descriptor_names = [desc.name for desc in materialized.descriptors]
            bytes_total = sum(
                int(desc.byte_length) for desc in materialized.descriptors
            )
        elif materialized.state_dict is not None:
            descriptor_names = list(materialized.state_dict.keys())
        if descriptor_names:
            summary["count"] = len(descriptor_names)

        canonical_index_bytes = (
            getattr(materialized, "canonical_index_bytes", b"") or b""
        )
        canonical_count: int | None = None
        if canonical_index_bytes:
            try:
                index_obj = canonical_index_from_bytes(canonical_index_bytes)
                canonical_count = len(index_obj.entries)
                if bytes_total == 0:
                    bytes_total = int(index_obj.total_size_bytes)
            except Exception:  # noqa: BLE001
                canonical_count = None

        if bytes_total > 0:
            summary["bytes"] = bytes_total

        selection: str | None = None
        subset = bool(requested_names)
        count_value = summary["count"]
        if (
            canonical_count is not None
            and count_value is not None
            and count_value < canonical_count
        ):
            subset = True
        if subset:
            selection = "subset"
        elif canonical_count is not None:
            selection = "full"
        summary["selection"] = selection
        return summary

    def _build_get_options(
        self,
        fallback: FallbackOptions | None,
        options_override: GetArtifactOptions | None,
    ) -> GetArtifactOptions:
        return options_override or GetArtifactOptions()

    def _materialize(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        options: GetArtifactOptions,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None = None,
        span=None,
        view: store_daemon_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
    ) -> MaterializationPayload:
        return self._materialize_payload(
            artifact_id=artifact_id,
            key=key,
            device_id=device_id,
            options=options,
            fallback=fallback,
            cancel_event=cancel_event,
            span=span,
            view=view,
            view_id=view_id,
            placement=placement,
            canonical_index_hint=canonical_index_hint,
            disk_path_hint=disk_path_hint,
            tensor_names=tensor_names,
            view_data_hash=view_data_hash,
            view_index_hint=view_index_hint,
            replica_uuid=replica_uuid,
        )

    def _materialize_payload(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        options: GetArtifactOptions,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None = None,
        span=None,
        view: store_daemon_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
    ) -> MaterializationPayload:
        client = self._runtime.ensure_client()
        disk_error: Exception | None = None
        disk_path: str | None = disk_path_hint
        resolved_artifact_id = artifact_id
        canonical_hint: bytes | None = canonical_index_hint
        generation_hint: int | None = None
        cached_entry: ArtifactCacheEntry | None = None
        fallback_opts = fallback
        verify_checksums = (
            bool(fallback_opts.verify_checksums) if fallback_opts else True
        )
        effective_prefer = fallback_opts.prefer if fallback_opts else "auto"
        if (
            fallback_opts
            and fallback_opts.prefer_disk is not None
            and effective_prefer == "auto"
            and fallback_opts.prefer_disk
        ):
            effective_prefer = "disk"
        allow_p2p = True if fallback_opts is None else bool(fallback_opts.allow_p2p)
        if effective_prefer == "local":
            allow_p2p = False
        requested_disk = (
            effective_prefer == "disk"
            or bool(fallback_opts and fallback_opts.disk_path)
            or bool(disk_path)
        )
        # Local-only preference disables P2P but should not force a disk fallback.
        disk_required = requested_disk or (
            not allow_p2p and effective_prefer != "local"
        )

        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        if effective_prefer == "p2p":
            preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_P2P
        elif effective_prefer == "disk":
            preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
        if (
            disk_path
            and preference == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        ):
            preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
        if disk_path and fallback_opts is None:
            allow_p2p = False

        if disk_path and canonical_hint is None:
            cached_entry = self._runtime.get_artifact_index_by_disk_path(disk_path)
            if cached_entry is not None:
                canonical_hint = cached_entry.canonical_index_bytes
                generation_hint = cached_entry.generation
                if cached_entry.artifact_id:
                    resolved_artifact_id = cached_entry.artifact_id

        if fallback_opts and (requested_disk or not allow_p2p):
            disk_path, resolved_artifact_id, disk_error = (
                self._fallback.resolve_disk_path(
                    fallback=fallback_opts,
                    client=client,
                    key=key,
                    artifact_id=artifact_id,
                )
            )
            if disk_path:
                requested_disk = True
                disk_required = True
                if (
                    effective_prefer == "disk"
                    or preference
                    == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
                ):
                    preference = (
                        store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
                    )
                self._fallback.record_event(
                    mode="disk",
                    artifact_id=resolved_artifact_id,
                    key=key,
                    detail={
                        "disk_path": disk_path,
                        "verify": bool(fallback_opts.verify_checksums),
                    },
                )
                if span is not None:
                    span.add_event(
                        "store.fallback.disk",
                        {
                            "tc.artifact.id": resolved_artifact_id or "",
                            "tc.store.disk_path_present": True,
                            "tc.store.verify_checksums": True,
                        },
                    )
                resolved_artifact_id = resolved_artifact_id or artifact_id
            if disk_required and not disk_path:
                if disk_error is not None:
                    raise ArtifactError(
                        f"Disk fallback lookup failed: {disk_error}",
                        status_code="UNAVAILABLE",
                        retryable=True,
                    ) from disk_error
                raise ArtifactError(
                    "Disk fallback required but disk_path unavailable",
                    status_code="NOT_FOUND",
                    retryable=False,
                )
        if requested_disk and disk_path and canonical_hint is None:
            try:
                disk_meta = client.resolve_artifact_from_disk_v2(
                    disk_path=disk_path,
                    verify_checksums=verify_checksums,
                )
                canonical_hint = bytes(disk_meta.canonical_index_bytes)
                if disk_meta.artifact_id:
                    resolved_artifact_id = disk_meta.artifact_id
                if getattr(disk_meta, "generation", 0):
                    generation_hint = int(disk_meta.generation)
                if canonical_hint and resolved_artifact_id:
                    parsed_index = canonical_index_from_bytes(canonical_hint)
                    cache_entry = ArtifactCacheEntry(
                        artifact_id=resolved_artifact_id,
                        canonical_index_bytes=canonical_hint,
                        parsed_index=parsed_index,
                        generation=generation_hint,
                        disk_path=disk_path,
                        expires_at=time.monotonic(),
                    )
                    self._runtime.cache_artifact_index(cache_entry)
            except Exception as exc:  # noqa: BLE001
                if not allow_p2p:
                    raise map_materialization_error(exc) from exc
                logger.info(
                    "store.materialize.disk.resolve_failed",
                    extra={
                        "tc.store.daemon": self._runtime.daemon_endpoint,
                        "tc.store.disk_path": disk_path,
                    },
                    exc_info=exc,
                )

        request_artifact_id = resolved_artifact_id or artifact_id
        view_subset_hash = view_data_hash.encode("utf-8") if view_data_hash else None
        index_hint = view_index_hint or canonical_hint
        result = self._materialize_fn(
            client=client,
            daemon_address=self._runtime.daemon_endpoint,
            device_id=device_id,
            artifact_id=request_artifact_id,
            key=None if requested_disk and disk_path else key,
            options=options,
            view=view,
            view_id=view_id,
            placement=placement,
            canonical_index_hint=index_hint,
            disk_path_hint=disk_path,
            preference=preference,
            tensor_names=tensor_names,
            verify_checksums=verify_checksums,
            view_subset_hash=view_subset_hash,
            replica_uuid=replica_uuid,
            view_index_hint=view_index_hint,
            generation_hint=generation_hint,
        )
        if fallback_opts and not allow_p2p:
            disallowed_sources = {
                store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_P2P
            }
            if requested_disk:
                disallowed_sources.add(
                    store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_LOCAL_REPLICA
                )
            if result.source in disallowed_sources:
                source_label = _source_label(result.source) or "non-disk"
                self._release_materialized(result, client)
                raise ArtifactError(
                    f"Materialization source {source_label} not allowed by fallback policy",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        source_label = _source_label(result.source) or ""
        self._fallback.record_event(
            mode="disk" if requested_disk else "p2p",
            artifact_id=result.artifact_id or request_artifact_id,
            key=key,
            detail={
                "disk_requested": bool(requested_disk),
                "allow_p2p": bool(allow_p2p),
                "source": source_label,
            },
        )
        if key:
            resolved_id = result.artifact_id or resolved_artifact_id or artifact_id
            self._runtime.cache_key_mapping(
                key,
                artifact_id=resolved_id,
                disk_path=result.disk_path,
            )
        if span is not None:
            span.add_event(
                "store.materialize.p2p",
                {
                    "tc.artifact.id": result.artifact_id or artifact_id or "",
                    "tc.store.disk_requested": bool(requested_disk),
                    "tc.store.allow_p2p": bool(allow_p2p),
                    "tc.store.preference": effective_prefer,
                    "tc.store.source": source_label,
                },
            )
        if cancel_event and cancel_event.is_set():
            self._release_materialized(result, client)
            raise CancelledError
        if tensor_names and result.descriptors:
            present = {desc.name for desc in result.descriptors}
            missing = [name for name in tensor_names if name not in present]
            if missing:
                self._release_materialized(result, client)
                raise ArtifactError(
                    f"Requested tensors missing from materialized payload: {', '.join(missing)}",
                    status_code="NOT_FOUND",
                    retryable=False,
                )
            requested_names = tuple(tensor_names)
            requested = set(requested_names)
            base_payload = result
            filtered_descriptors = tuple(
                desc for desc in result.descriptors if desc.name in requested
            )
            try:
                canonical_index = canonical_index_from_bytes(
                    base_payload.canonical_index_bytes
                )
                canonical_bytes = canonical_index_to_bytes(
                    canonical_index, requested_names
                )
            except Exception:  # noqa: BLE001
                canonical_bytes = base_payload.canonical_index_bytes

            def _filtered_iter():
                for desc, tensor in base_payload.payload_iter():
                    if desc.name in requested:
                        yield desc, tensor

            state_dict = None
            if base_payload.state_dict is not None:
                state_dict = {
                    name: tensor
                    for name, tensor in base_payload.state_dict.items()
                    if name in requested
                }

            result = MaterializationPayload(
                artifact_id=base_payload.artifact_id,
                canonical_index_bytes=canonical_bytes,
                descriptors=filtered_descriptors,
                payload_iter=_filtered_iter,
                generation=base_payload.generation,
                state_dict=state_dict,
                replica_uuid=base_payload.replica_uuid,
                disk_path=base_payload.disk_path,
                view_index_bytes=base_payload.view_index_bytes,
                view_data_hash=base_payload.view_data_hash,
                source=base_payload.source,
                device_uuid=base_payload.device_uuid,
                ticket_replica_uuid=base_payload.ticket_replica_uuid,
                ticket_status=base_payload.ticket_status,
                ticket_created_at_ts=base_payload.ticket_created_at_ts,
                ticket_expires_at_ts=base_payload.ticket_expires_at_ts,
            )
        return result

    def _call_view_resolver(
        self,
        resolver: Callable[..., ResolvedViewInputs],
        *,
        artifact_id: str | None,
        key: str | None,
        slices: Mapping[str, Sequence[object]] | None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None,
        view_id: str | None,
    ) -> ResolvedViewInputs:
        return resolver(
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
        )

    def _perform_get_with_retry(
        self,
        *,
        method: str,
        artifact_id: str | None,
        key: str | None,
        device: torch.device | str | None,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None,
        options_override: GetArtifactOptions | None,
        view: store_daemon_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
    ) -> tuple[MaterializationPayload, int]:
        policy = self._runtime.retry_policies.get(
            method, self._runtime.retry_policies.get("get")
        )
        attempt = 1
        start_time = time.monotonic()
        span_name = "Store/GetInto" if method == "get_into" else "Store/Get"
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.daemon": self._runtime.daemon_endpoint,
            "tc.store.session_id": self._runtime.session_id,
            "tc.store.method": method,
            "tc.store.lookup.by_key": bool(key),
            "tc.store.lookup.by_id": bool(artifact_id),
            "tc.store.fallback.prefer": (fallback.prefer if fallback else "auto"),
            "tc.store.fallback.allow_p2p": bool(fallback is None or fallback.allow_p2p),
        }
        source_label: str | None = None
        selection_label: str | None = None

        with self._runtime.operation_span(span_name, attributes) as span:

            def record_outcome(status: str) -> None:
                duration = time.monotonic() - start_time
                store_metrics.observe_latency(
                    method,
                    self._runtime.daemon_endpoint,
                    status,
                    duration,
                    source=source_label,
                    selection=selection_label,
                )
                span.set_attribute("tc.store.status", status)
                if status not in {"OK", "CANCELLED"}:
                    store_metrics.increment_error(
                        method,
                        self._runtime.daemon_endpoint,
                        status,
                        source=source_label,
                        selection=selection_label,
                    )

            while True:
                selection_label = None
                if cancel_event and cancel_event.is_set():
                    record_outcome("CANCELLED")
                    span.set_status(Status(StatusCode.ERROR, "CANCELLED"))
                    raise ArtifactError(
                        "Retrieval cancelled",
                        status_code="CANCELLED",
                        retryable=False,
                    )
                try:
                    materialized, device_id = self._attempt_get(
                        artifact_id=artifact_id,
                        key=key,
                        device=device,
                        fallback=fallback,
                        cancel_event=cancel_event,
                        options_override=options_override,
                        span=span,
                        view=view,
                        view_id=view_id,
                        placement=placement,
                        canonical_index_hint=canonical_index_hint,
                        disk_path_hint=disk_path_hint,
                        tensor_names=tensor_names,
                        view_data_hash=view_data_hash,
                        view_index_hint=view_index_hint,
                        replica_uuid=replica_uuid,
                    )
                    summary = self._summarize_materialized(materialized, tensor_names)
                    selection_label = summary["selection"]
                    source_label = _source_label(materialized.source)
                    count_value = summary["count"]
                    bytes_value = summary["bytes"]
                    if count_value is not None:
                        span.set_attribute("tc.tensor.count", int(count_value))
                    if bytes_value is not None:
                        span.set_attribute("tc.tensor.bytes", int(bytes_value))
                    if selection_label:
                        span.set_attribute("tc.tensor.selection", selection_label)
                    record_outcome("OK")
                    span.set_attribute("tc.store.retry.count", attempt)
                    span.set_attribute("tc.device.id", int(device_id))
                    span.set_attribute("tc.artifact.id", materialized.artifact_id)
                    span.set_attribute(
                        "tc.store.fallback.used_disk", bool(materialized.disk_path)
                    )
                    span.set_attribute("tc.store.source", source_label or "")
                    span.set_status(Status(StatusCode.OK))
                    return materialized, device_id
                except ArtifactError as error:
                    span.record_exception(error)
                    if error.status_code in {"NOT_FOUND", "FAILED_PRECONDITION"}:
                        self._runtime.invalidate_artifact(
                            artifact_id, key=key, reason="materialize_error"
                        )
                    should_retry_op = should_retry(
                        error=error,
                        attempt=attempt,
                        policy=policy,
                        start_time=start_time,
                        cancel_event=cancel_event,
                    )
                    if not should_retry_op:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise
                    assert policy is not None
                    delay = compute_retry_delay(policy, attempt)
                    remaining = remaining_budget(policy, start_time)
                    if remaining is not None and remaining <= 0:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining))
                    store_metrics.increment_retry(
                        method,
                        self._runtime.daemon_endpoint,
                        error.status_code,
                        source=source_label,
                        selection=selection_label,
                    )
                    span.add_event(
                        "store.retry",
                        {
                            "tc.store.retry_attempt": attempt + 1,
                            "tc.store.status": error.status_code,
                        },
                    )
                    if delay > 0:
                        logger.info(
                            "store.get_retry",
                            extra={
                                "tc.store.daemon": self._runtime.daemon_endpoint,
                                "tc.store.method": method,
                                "tc.store.attempt": attempt + 1,
                                "tc.store.delay_sec": delay,
                                "tc.store.status_code": error.status_code,
                            },
                        )
                        time.sleep(delay)
                    attempt += 1
                except Exception as exc:  # noqa: BLE001
                    error = map_materialization_error(exc)
                    span.record_exception(error)
                    if error.status_code in {"NOT_FOUND", "FAILED_PRECONDITION"}:
                        self._runtime.invalidate_artifact(
                            artifact_id, key=key, reason="materialize_error"
                        )
                    should_retry_op = should_retry(
                        error=error,
                        attempt=attempt,
                        policy=policy,
                        start_time=start_time,
                        cancel_event=cancel_event,
                    )
                    if not should_retry_op:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise error from None
                    assert policy is not None
                    delay = compute_retry_delay(policy, attempt)
                    remaining = remaining_budget(policy, start_time)
                    if remaining is not None and remaining <= 0:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise error from None
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining))
                    store_metrics.increment_retry(
                        method,
                        self._runtime.daemon_endpoint,
                        error.status_code,
                        source=source_label,
                        selection=selection_label,
                    )
                    span.add_event(
                        "store.retry",
                        {
                            "tc.store.retry_attempt": attempt + 1,
                            "tc.store.status": error.status_code,
                        },
                    )
                    if delay > 0:
                        logger.info(
                            "store.get_retry",
                            extra={
                                "tc.store.daemon": self._runtime.daemon_endpoint,
                                "tc.store.method": method,
                                "tc.store.attempt": attempt + 1,
                                "tc.store.delay_sec": delay,
                                "tc.store.status_code": error.status_code,
                            },
                        )
                        time.sleep(delay)
                    attempt += 1

    def _attempt_get(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device: torch.device | str | None,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None,
        options_override: GetArtifactOptions | None,
        span=None,
        view: store_daemon_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
    ) -> tuple[MaterializationPayload, int]:
        resolved_fallback = fallback or self._runtime.opts.fallback
        artifact_id, key = self._resolve_identifiers(
            artifact_id, key, resolved_fallback, disk_path_hint=disk_path_hint
        )
        self._fallback.ensure_supported(resolved_fallback)
        device_id = self._resolve_device_selector(device, resolved_fallback)
        options = self._build_get_options(resolved_fallback, options_override)
        try:
            materialized = self._materialize(
                artifact_id=artifact_id,
                key=key,
                device_id=device_id,
                options=options,
                fallback=resolved_fallback,
                cancel_event=cancel_event,
                span=span,
                view=view,
                view_id=view_id,
                placement=placement,
                canonical_index_hint=canonical_index_hint,
                disk_path_hint=disk_path_hint,
                tensor_names=tensor_names,
                view_data_hash=view_data_hash,
                view_index_hint=view_index_hint,
                replica_uuid=replica_uuid,
            )
        except Exception as exc:  # noqa: BLE001
            raise map_materialization_error(exc) from exc
        return materialized, device_id

    def _release_materialized(
        self, materialized: MaterializationPayload, client
    ) -> None:
        replica_uuid = getattr(materialized, "replica_uuid", "") or ""
        disk_path = getattr(materialized, "disk_path", "") or ""
        if not replica_uuid:
            return
        if not client.unload_replica(replica_uuid, disk_path=disk_path):
            logger.warning(
                "store.cancel_unload_failed",
                extra={
                    "tc.store.daemon": self._runtime.daemon_endpoint,
                    "tc.store.replica_uuid": replica_uuid,
                    "tc.store.disk_path": disk_path,
                },
            )


__all__ = ["MaterializationPipeline", "FallbackResolver"]
