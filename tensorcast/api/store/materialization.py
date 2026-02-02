#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import TypedDict

import torch
from opentelemetry.trace import Status, StatusCode

from tensorcast._c_ext import compute_view_index_bytes
from tensorcast.api import _metrics as store_metrics
from tensorcast.api import _region_cache as region_cache
from tensorcast.api._config import GetArtifactOptions, RegionBackedMode
from tensorcast.api._device import device_uuid_for
from tensorcast.api._materialize import (
    MaterializationPayload,
    materialize_artifact_v2,
)
from tensorcast.api.context import CallContext
from tensorcast.api.store.async_ops import ArtifactFuture, TrackedExecutor
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
    validate_targets,
)
from tensorcast.api.store.retry import (
    compute_retry_delay,
    map_materialization_error,
    raise_mapped_materialization_error,
    remaining_budget,
    should_retry,
)
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    FallbackOptions,
    RetryPolicy,
    SpanAttributeValue,
)
from tensorcast.api.store.view_composer import compute_view_id
from tensorcast.api.store.views import (
    ResolvedViewInputs,
    TransformPlacement,
    ViewOrchestrator,
)
from tensorcast.common.selection_identity import (
    compute_logical_layout_hash,
    compute_view_subset_hash,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2

logger = logging.getLogger(__name__)


class _MaterializationSummary(TypedDict):
    count: int | None
    bytes: int | None
    selection: str | None


@dataclass(frozen=True, slots=True)
class _RegionBackedLayout:
    layout: store_daemon_pb2.TargetLayout
    region_ids: tuple[str, ...]
    logical_total_size: int
    view_index_bytes: bytes | None
    view_id: str | None
    selection_names: tuple[str, ...]
    view_subset_hash: bytes | None


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


def _build_source_policy(
    *,
    preference: store_daemon_pb2.SourcePreference,
    allow_p2p: bool,
    allow_disk: bool,
) -> store_daemon_pb2.SourcePolicy:
    policy = store_daemon_pb2.SourcePolicy(preference=preference)
    policy.allow_p2p = bool(allow_p2p)
    policy.allow_disk = bool(allow_disk)
    return policy


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
            "store.materialize.source",
            extra={
                "tc.store.daemon": self._runtime.daemon_endpoint,
                "tc.store.mode": mode,
                "tc.store.artifact_id": artifact_id or "",
                "tc.store.key": key or "",
                "tc.store.detail": dict(detail),
            },
        )


class MaterializationPipeline:
    """Retrieval orchestration for artifact materialization and view flows."""

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
        ctx: CallContext | None = None,
    ) -> dict[str, torch.Tensor]:
        payload, _ = self._perform_get_with_retry(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            method="get",
            cancel_event=None,
            options_override=options,
            tensor_names=tensor_names,
            allow_cpu=True,
            ctx=ctx,
        )
        state = self._payload_state_dict(payload)
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
        view_spec: common_pb2.ViewSpec | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
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
            allow_cpu=True,
            ctx=ctx,
            lease_mode=lease_mode,
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
        ctx: CallContext | None = None,
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

        payload, _ = self._perform_get_with_retry(
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
            allow_cpu=True,
            ctx=ctx,
        )
        return self._payload_state_dict(payload)

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
            state: dict[str, torch.Tensor] | None = None
            try:
                materialized, _ = self._perform_get_with_retry(
                    artifact_id=artifact_id,
                    key=key,
                    device=device,
                    fallback=fallback,
                    method="get",
                    cancel_event=cancel_event,
                    options_override=options,
                    tensor_names=tensor_names,
                    allow_cpu=True,
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
                state = self._payload_state_dict(materialized)
                return state
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
                if materialized is not None and state is None:
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
        view_spec: common_pb2.ViewSpec | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        options = self._apply_client_defaults(
            self._build_get_options(fallback, options)
        )
        start_time = time.monotonic()
        try:
            if self._maybe_region_backed_into(
                target=target,
                artifact_id=artifact_id,
                key=key,
                device_id=self._resolve_device_selector(device, fallback),
                fallback=fallback,
                options=options,
                tensor_names=tensor_names,
                view=view_spec,
                view_id=None,
                view_index_hint=view_index_hint,
            ):
                store_metrics.observe_latency(
                    "get_into",
                    self._runtime.daemon_endpoint,
                    "OK",
                    time.monotonic() - start_time,
                    selection="region_backed",
                )
                return
        except ArtifactError as exc:
            store_metrics.observe_latency(
                "get_into",
                self._runtime.daemon_endpoint,
                exc.status_code,
                time.monotonic() - start_time,
                selection="region_backed",
            )
            store_metrics.increment_error(
                "get_into",
                self._runtime.daemon_endpoint,
                exc.status_code,
                selection="region_backed",
            )
            raise
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
            ctx=ctx,
        )
        try:
            layout_bytes = payload.view_index_bytes or payload.canonical_index_bytes
            canonical_index = canonical_index_from_bytes(layout_bytes)
            requested = tuple(tensor_names) if tensor_names else None
            pairs = validate_targets(
                canonical_index=canonical_index,
                target=target,
                source=self._payload_state_dict(payload),
                device_id=device_id,
                required_names=requested,
            )
            for tgt, src in pairs:
                tgt.copy_(src)
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
        view_spec: common_pb2.ViewSpec | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        ctx: CallContext | None = None,
    ) -> ArtifactFuture[None]:
        options = self._apply_client_defaults(
            self._build_get_options(fallback, options)
        )
        cancel_event = threading.Event()
        mat_lock = threading.Lock()
        mat_ref: dict[str, MaterializationPayload | None] = {"value": None}
        region_backed_lock = threading.Lock()
        region_backed_state: dict[str, bool] = {"started": False}

        def _mark_region_backed_started() -> None:
            with region_backed_lock:
                region_backed_state["started"] = True

        def _task() -> None:
            materialized: MaterializationPayload | None = None
            try:
                if not cancel_event.is_set():
                    start_time = time.monotonic()
                    try:
                        if self._maybe_region_backed_into(
                            target=target,
                            artifact_id=artifact_id,
                            key=key,
                            device_id=self._resolve_device_selector(device, fallback),
                            fallback=fallback,
                            options=options,
                            tensor_names=tensor_names,
                            view=view_spec,
                            view_id=None,
                            view_index_hint=view_index_hint,
                            mark_started=_mark_region_backed_started,
                        ):
                            store_metrics.observe_latency(
                                "get_into",
                                self._runtime.daemon_endpoint,
                                "OK",
                                time.monotonic() - start_time,
                                selection="region_backed",
                            )
                            return
                    except ArtifactError as exc:
                        store_metrics.observe_latency(
                            "get_into",
                            self._runtime.daemon_endpoint,
                            exc.status_code,
                            time.monotonic() - start_time,
                            selection="region_backed",
                        )
                        store_metrics.increment_error(
                            "get_into",
                            self._runtime.daemon_endpoint,
                            exc.status_code,
                            selection="region_backed",
                        )
                        raise
                materialized, device_id = self._perform_get_with_retry(
                    artifact_id=artifact_id,
                    key=key,
                    device=device,
                    fallback=fallback,
                    method="get_into",
                    cancel_event=cancel_event,
                    options_override=options,
                    tensor_names=tensor_names,
                    view=view_spec,
                    view_data_hash=view_data_hash,
                    view_index_hint=view_index_hint,
                    replica_uuid=replica_uuid,
                    ctx=ctx,
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
                    canonical_index = canonical_index_from_bytes(layout_bytes)
                    source_map = self._payload_state_dict(materialized)
                    requested = tuple(tensor_names) if tensor_names else None
                    pairs = validate_targets(
                        canonical_index=canonical_index,
                        target=target,
                        source=source_map,
                        device_id=device_id,
                        required_names=requested,
                    )
                    for tgt, src in pairs:
                        if cancel_event.is_set():
                            raise CancelledError
                        tgt.copy_(src)
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
            with region_backed_lock:
                if region_backed_state["started"]:
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
            layout_index = canonical_index_from_bytes(layout_bytes)
            pairs = validate_targets(
                canonical_index=layout_index,
                target=target,
                source=self._payload_state_dict(payload),
                device_id=device_id,
            )
            for tgt, src in pairs:
                tgt.copy_(src)
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
        *,
        allow_cpu: bool = False,
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
                if allow_cpu:
                    return -1
                if _has_disk_fallback():
                    # Legacy fallback: treat CPU as cuda:0 for disk-based retrieval paths.
                    return 0
                raise ArtifactError(
                    "CUDA device required for retrieval",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            return int(torch.cuda.current_device())
        if isinstance(selector, torch.device):
            if selector.type != "cuda":
                if selector.type == "cpu" and allow_cpu:
                    return -1
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
                if device.type == "cpu" and allow_cpu:
                    return -1
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
        self, payload: MaterializationPayload
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
        if payload.state_dict is not None:
            return dict(payload.state_dict)
        state: dict[str, torch.Tensor] = {}
        for desc, tensor in payload.payload_iter():
            state[desc.name] = tensor
        return state

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

    def _apply_client_defaults(self, options: GetArtifactOptions) -> GetArtifactOptions:
        from tensorcast.api._runtime import apply_client_load_defaults_if_present

        (
            pinned_ms,
            enable_ver,
            wait_for_completion,
            region_backed_mode,
        ) = apply_client_load_defaults_if_present(
            options.pinned_allocation_timeout_ms,
            options.enable_verification,
            options.wait_for_completion,
            options.region_backed_mode,
            runtime_address=self._runtime.daemon_endpoint,
        )
        return options.model_copy(
            update={
                "pinned_allocation_timeout_ms": pinned_ms,
                "enable_verification": enable_ver,
                "wait_for_completion": wait_for_completion,
                "region_backed_mode": region_backed_mode,
            }
        )

    def _build_region_backed_layout(
        self,
        *,
        canonical_index: CanonicalIndex,
        canonical_index_bytes: bytes,
        target: Mapping[str, torch.Tensor],
        device_id: int,
        tensor_names: Sequence[str] | None,
        view_spec: common_pb2.ViewSpec | None,
        view_id: str | None,
        view_index_hint: bytes | None,
        selection_order: Sequence[str] | None = None,
    ) -> _RegionBackedLayout:
        entries_by_name = {entry.name: entry for entry in canonical_index.entries}
        if not entries_by_name:
            raise ArtifactError(
                "Canonical index is empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        target_names = {str(name) for name in target}
        if tensor_names is not None and set(tensor_names) != target_names:
            raise ArtifactError(
                "Target tensors must match tensor_names for region-backed get_into",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not target_names:
            raise ArtifactError(
                "Target tensors are empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        canonical_names = set(entries_by_name.keys())
        if not target_names.issubset(canonical_names):
            raise ArtifactError(
                "Target tensors reference unknown canonical entries",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        full_selection = target_names == canonical_names

        selection_names: tuple[str, ...] = ()
        if selection_order is not None:
            selection_names = tuple(str(name) for name in selection_order)
        elif tensor_names is not None:
            selection_names = tuple(str(name) for name in tensor_names)
        elif not full_selection:
            selection_names = tuple(
                entry.name
                for entry in canonical_index.entries
                if entry.name in target_names
            )

        if (
            selection_order is not None
            or tensor_names is not None
            or not full_selection
        ):
            if not selection_names:
                raise ArtifactError(
                    "selection_order must not be empty when provided",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if len(set(selection_names)) != len(selection_names):
                raise ArtifactError(
                    "selection_order must not contain duplicates",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if set(selection_names) != target_names:
                raise ArtifactError(
                    "selection_order must match target tensor names",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )

        normalized_ops: dict[str, list[dict[str, int | str]]] = {}
        if view_spec is not None and view_spec.tensors:
            for name, ops in view_spec.tensors.items():
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
        resolved_view_id: str | None = None
        if view_spec is not None and view_spec.tensors:
            resolved_view_id = compute_view_id(view_spec, canonical_index_bytes)
        if view_id is not None:
            if resolved_view_id is not None and view_id != resolved_view_id:
                raise ArtifactError(
                    "view_id does not match view_spec",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            resolved_view_id = view_id

        needs_view_index = (
            bool(normalized_ops) or not full_selection or bool(selection_names)
        )
        view_index_bytes: bytes | None = None
        if needs_view_index:
            if view_spec is None and view_index_hint is not None:
                view_index_bytes = view_index_hint
            else:
                subset_payload = None
                if selection_names:
                    subset_payload = list(selection_names)
                view_payload = compute_view_index_bytes(
                    canonical_index_bytes, normalized_ops, subset_payload
                )
                view_index_bytes = bytes(view_payload["view_index_bytes"])
        elif view_id is not None:
            raise ArtifactError(
                "view_id requires view_spec or view_index_hint for region-backed get_into",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        index_kind = (
            store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
            if needs_view_index
            else store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED
        )
        index_bytes = view_index_bytes or canonical_index_bytes
        layout_index = canonical_index_from_bytes(index_bytes)
        layout_entries_by_name = {entry.name: entry for entry in layout_index.entries}
        if set(layout_entries_by_name.keys()) != target_names:
            raise ArtifactError(
                "Target tensors do not match selected index entries",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        def _storage_nbytes(tensor: torch.Tensor) -> int:
            if hasattr(tensor, "untyped_storage"):
                return int(tensor.untyped_storage().nbytes())
            return int(tensor.storage().nbytes())

        logical_total_size = 0

        @dataclass
        class _StorageGroup:
            base_ptr: int
            base_offset: int
            nbytes: int
            max_used: int
            tensors: list[str]

        storage_groups: dict[int, _StorageGroup] = {}

        for entry in layout_index.entries:
            tensor = target.get(entry.name)
            if tensor is None:
                raise ArtifactError(
                    f"Target tensor '{entry.name}' missing",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_cuda:
                raise ArtifactError(
                    f"Target tensor '{entry.name}' must be CUDA",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if (tensor.device.index or 0) != device_id:
                raise ArtifactError(
                    f"Target tensor '{entry.name}' on cuda:{tensor.device.index or 0}, expected cuda:{device_id}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_contiguous():
                raise ArtifactError(
                    f"Target tensor '{entry.name}' is not contiguous",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if tensor.dtype != entry.dtype:
                raise ArtifactError(
                    f"Target tensor '{entry.name}' dtype {tensor.dtype} != {entry.dtype}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tuple(tensor.shape) != tuple(entry.shape):
                raise ArtifactError(
                    f"Target tensor '{entry.name}' shape {tuple(tensor.shape)} != {entry.shape}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tuple(tensor.stride()) != tuple(entry.stride):
                raise ArtifactError(
                    f"Target tensor '{entry.name}' stride {tuple(tensor.stride())} != {entry.stride}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if (
                index_kind
                == store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED
            ):
                elem_bytes = int(tensor.element_size())
                if entry.segment_offset != entry.storage_offset * elem_bytes:
                    raise ArtifactError(
                        "Canonical layout is not COALESCED; region-backed get_into requires coalesced offsets",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )

            logical_offset = int(entry.segment_offset)
            logical_total_size = max(
                logical_total_size, logical_offset + int(entry.size_bytes)
            )
            storage_offset_bytes = int(tensor.storage_offset()) * int(
                tensor.element_size()
            )
            base_ptr = int(tensor.data_ptr()) - storage_offset_bytes
            base_offset = logical_offset - storage_offset_bytes
            if base_offset < 0:
                raise ArtifactError(
                    "Target tensor storage offsets exceed logical offsets",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            group = storage_groups.get(base_ptr)
            if group is None:
                storage_groups[base_ptr] = _StorageGroup(
                    base_ptr=base_ptr,
                    base_offset=base_offset,
                    nbytes=_storage_nbytes(tensor),
                    max_used=storage_offset_bytes + int(entry.size_bytes),
                    tensors=[entry.name],
                )
            else:
                if group.base_offset != base_offset:
                    raise ArtifactError(
                        "Target tensors do not share a consistent storage base offset",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
                group.max_used = max(
                    group.max_used, storage_offset_bytes + int(entry.size_bytes)
                )
                group.tensors.append(entry.name)

        storage_list = sorted(
            storage_groups.values(), key=lambda group: group.base_offset
        )
        if not storage_list or storage_list[0].base_offset != 0:
            raise ArtifactError(
                "Target storages must cover logical offset 0",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        storage_specs: list[tuple[_StorageGroup, int]] = []
        for idx, group in enumerate(storage_list):
            next_offset = (
                storage_list[idx + 1].base_offset
                if idx + 1 < len(storage_list)
                else logical_total_size
            )
            length = next_offset - group.base_offset
            if length <= 0:
                raise ArtifactError(
                    "Target storages must be ordered by logical offset",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if group.max_used > length:
                raise ArtifactError(
                    "Target storage does not cover tensor ranges",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if length > group.nbytes:
                raise ArtifactError(
                    "Target storage exceeds backing allocation",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            storage_specs.append((group, length))

        storage_id_by_ptr: dict[int, str] = {}
        region_ids: list[str] = []
        layout = store_daemon_pb2.TargetLayout(
            layout_kind=store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=index_kind,
            tensor_spec_kind=store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )
        if (
            index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
            and resolved_view_id
        ):
            layout.view_id = str(resolved_view_id)

        for group, length in storage_specs:
            region = region_cache.find_region_for(device_id, group.base_ptr, length)
            if region is None:
                raise ArtifactError(
                    "No registered region covers the target tensors",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            region_base_offset = int(group.base_ptr) - int(region.base_ptr)
            storage_id = f"storage:{device_id}:{group.base_ptr:x}:{length}"
            layout.storages.add(
                storage_id=storage_id,
                device_id=int(device_id),
                storage_length=int(length),
                vram_region_id=region.region_id,
                mapping_base_offset=int(region_base_offset),
            )
            storage_id_by_ptr[group.base_ptr] = storage_id
            region_ids.append(region.region_id)

        offsets: list[store_daemon_pb2.TargetTensorOffset] = []
        for entry in layout_index.entries:
            tensor = target[entry.name]
            storage_offset_bytes = int(tensor.storage_offset()) * int(
                tensor.element_size()
            )
            base_ptr = int(tensor.data_ptr()) - storage_offset_bytes
            storage_id = storage_id_by_ptr[base_ptr]
            offsets.append(
                store_daemon_pb2.TargetTensorOffset(
                    name=entry.name,
                    storage_id=storage_id,
                    storage_offset=int(storage_offset_bytes),
                    logical_length=int(entry.size_bytes),
                )
            )
        layout.offsets.extend(offsets)

        layout.logical_layout_hash = compute_logical_layout_hash(
            index_bytes=index_bytes,
            needs_view_index=needs_view_index,
        )
        view_subset_hash = (
            b"" if full_selection else compute_view_subset_hash(selection_names)
        )
        return _RegionBackedLayout(
            layout=layout,
            region_ids=tuple(region_ids),
            logical_total_size=logical_total_size,
            view_index_bytes=view_index_bytes,
            view_id=resolved_view_id,
            selection_names=selection_names,
            view_subset_hash=view_subset_hash,
        )

    def _materialize_into_target(
        self,
        *,
        target: dict[str, torch.Tensor],
        artifact_id: str,
        device_id: int,
        fallback: FallbackOptions | None,
        options: GetArtifactOptions,
        tensor_names: Sequence[str] | None,
        view: common_pb2.ViewSpec | None,
        view_id: str | None,
        view_index_hint: bytes | None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        mark_started: Callable[[], None] | None = None,
        span=None,
    ) -> None:
        if not artifact_id:
            raise ArtifactError(
                "artifact_id is required for region-backed get_into",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        cached_entry = self._runtime.get_artifact_index_cached(artifact_id)
        if cached_entry is not None:
            canonical_bytes = cached_entry.canonical_index_bytes
            canonical_index = cached_entry.parsed_index
        else:
            client = self._runtime.ensure_client()
            canonical_bytes = client.get_artifact_index_by_id(artifact_id)
            canonical_index = canonical_index_from_bytes(canonical_bytes)
            cache_entry = ArtifactCacheEntry(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_bytes,
                parsed_index=canonical_index,
                generation=None,
                disk_path=None,
                expires_at=time.monotonic(),
            )
            self._runtime.cache_artifact_index(cache_entry)

        region_layout = self._build_region_backed_layout(
            canonical_index=canonical_index,
            canonical_index_bytes=canonical_bytes,
            target=target,
            device_id=device_id,
            tensor_names=tensor_names,
            view_spec=view,
            view_id=view_id,
            view_index_hint=view_index_hint,
        )

        effective_prefer = fallback.prefer if fallback is not None else "auto"
        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        disk_path: str | None = None
        if fallback is not None:
            if fallback.prefer == "p2p":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_P2P
                )
            elif fallback.prefer == "disk":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
                )
            disk_path = fallback.disk_path
        if (
            disk_path
            and preference == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        ):
            preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK

        allow_p2p = True if fallback is None else bool(fallback.allow_p2p)
        if effective_prefer == "local":
            allow_p2p = False
        allow_disk = effective_prefer != "local" or bool(disk_path)
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )

        client = self._runtime.ensure_client()
        if mark_started is not None:
            mark_started()
        try:
            response = client.materialize_into_target_v2(
                artifact_id=artifact_id,
                target_layout=region_layout.layout,
                device_uuid=device_uuid_for(device_id),
                preference=preference,
                source_policy=source_policy,
                disk_path=disk_path,
                tensor_names=region_layout.selection_names,
                view=view,
                view_id=view_id if view is None else None,
                view_subset_hash=region_layout.view_subset_hash,
                placement=placement,
            )
        except Exception as exc:  # noqa: BLE001
            error = map_materialization_error(exc)
            if error.status_code in {"DATA_LOSS", "FAILED_PRECONDITION"}:
                for region_id in region_layout.region_ids:
                    region_cache.unregister_region(region_id)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=False,
            ) from exc
        if (
            response.status
            != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            for region_id in region_layout.region_ids:
                region_cache.unregister_region(region_id)
            raise ArtifactError(
                "MaterializeIntoTarget returned non-success status",
                status_code="DATA_LOSS",
                retryable=False,
            )
        store_metrics.record_region_backed_verification_skipped(
            self._runtime.daemon_endpoint
        )
        return None

    def _maybe_region_backed_into(
        self,
        *,
        target: dict[str, torch.Tensor],
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        fallback: FallbackOptions | None,
        options: GetArtifactOptions,
        tensor_names: Sequence[str] | None,
        view: common_pb2.ViewSpec | None,
        view_id: str | None,
        view_index_hint: bytes | None,
        span=None,
        mark_started: Callable[[], None] | None = None,
    ) -> bool:
        mode = options.region_backed_mode
        if mode is RegionBackedMode.DISABLE:
            return False
        if key is not None:
            if mode is RegionBackedMode.REQUIRE:
                raise ArtifactError(
                    "region_backed_mode requires artifact_id (key not supported)",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            store_metrics.record_region_backed_fallback(
                self._runtime.daemon_endpoint, "key_not_supported"
            )
            return False
        if not artifact_id:
            if mode is RegionBackedMode.REQUIRE:
                raise ArtifactError(
                    "region_backed_mode requires artifact_id",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            store_metrics.record_region_backed_fallback(
                self._runtime.daemon_endpoint, "missing_artifact_id"
            )
            return False
        try:
            self._materialize_into_target(
                target=target,
                artifact_id=artifact_id,
                device_id=device_id,
                fallback=fallback,
                options=options,
                tensor_names=tensor_names,
                view=view,
                view_id=view_id,
                view_index_hint=view_index_hint,
                mark_started=mark_started,
                span=span,
            )
        except ArtifactError:
            if mode is RegionBackedMode.REQUIRE:
                raise
            store_metrics.record_region_backed_fallback(
                self._runtime.daemon_endpoint, "layout_mismatch"
            )
            if span is not None:
                span.add_event(
                    "store.region_backed.fallback",
                    {"tc.store.reason": "layout_mismatch"},
                )
            return False
        return True

    def _materialize(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        options: GetArtifactOptions,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None = None,
        ctx: CallContext | None,
        timeout_s: float | None,
        lease_mode: store_daemon_pb2.LeaseMode,
        span=None,
        view: common_pb2.ViewSpec | None = None,
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
            ctx=ctx,
            timeout_s=timeout_s,
            lease_mode=lease_mode,
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
        ctx: CallContext | None,
        timeout_s: float | None,
        lease_mode: store_daemon_pb2.LeaseMode,
        span=None,
        view: common_pb2.ViewSpec | None = None,
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
            resolved_path, resolved_artifact_id, disk_error = (
                self._fallback.resolve_disk_path(
                    fallback=fallback_opts,
                    client=client,
                    key=key,
                    artifact_id=artifact_id,
                )
            )
            if resolved_path:
                disk_path = resolved_path
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
                        "store.materialize.disk_fallback",
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
                    raise_mapped_materialization_error(exc)
                logger.info(
                    "store.materialize.disk.resolve_failed",
                    extra={
                        "tc.store.daemon": self._runtime.daemon_endpoint,
                        "tc.store.disk_path": disk_path,
                    },
                    exc_info=exc,
                )

        allow_disk = effective_prefer != "local" or requested_disk
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )

        request_artifact_id = resolved_artifact_id or artifact_id
        view_subset_hash = (
            compute_view_subset_hash(tensor_names) if tensor_names else None
        )
        index_hint = view_index_hint or canonical_hint
        result = self._materialize_fn(
            client=client,
            daemon_address=self._runtime.daemon_endpoint,
            server_config=self._runtime.capabilities.server_config,
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
            source_policy=source_policy,
            tensor_names=tensor_names,
            verify_checksums=verify_checksums,
            view_subset_hash=view_subset_hash,
            replica_uuid=replica_uuid,
            view_index_hint=view_index_hint,
            generation_hint=generation_hint,
            ctx=ctx,
            timeout_s=timeout_s,
            lease_mode=lease_mode,
        )
        if fallback_opts:
            disallowed_sources: set[store_daemon_pb2.MaterializationSource] = set()
            if not allow_p2p:
                disallowed_sources.add(
                    store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_P2P
                )
            if not allow_disk:
                disallowed_sources.add(
                    store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_DISK
                )
            if requested_disk:
                disallowed_sources.add(
                    store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_LOCAL_REPLICA
                )
            if disallowed_sources and result.source in disallowed_sources:
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
                "allow_disk": bool(allow_disk),
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
                    "tc.store.allow_disk": bool(allow_disk),
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
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        allow_cpu: bool = False,
        ctx: CallContext | None = None,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
    ) -> tuple[MaterializationPayload, int]:
        policy = self._runtime.retry_policies.get(
            method, self._runtime.retry_policies.get("get")
        )
        attempt = 1
        start_time = time.monotonic()
        if ctx is not None and ctx.deadline_ms is not None and policy is not None:
            ctx_budget_s = float(ctx.deadline_ms) / 1000.0
            if ctx_budget_s <= 0:
                raise ArtifactError(
                    "CallContext deadline exceeded",
                    status_code="DEADLINE_EXCEEDED",
                    retryable=False,
                )
            if policy.deadline_seconds <= 0:
                effective_deadline = ctx_budget_s
            else:
                effective_deadline = min(policy.deadline_seconds, ctx_budget_s)
            if effective_deadline != policy.deadline_seconds:
                policy = RetryPolicy(
                    deadline_seconds=effective_deadline,
                    max_attempts=policy.max_attempts,
                    base_backoff_seconds=policy.base_backoff_seconds,
                    backoff_multiplier=policy.backoff_multiplier,
                    jitter=policy.jitter,
                )
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

        if ctx is not None:
            if ctx.request_id:
                attributes["tc.request_id"] = str(ctx.request_id)
            if ctx.qos:
                attributes["tc.qos"] = str(ctx.qos)
            if ctx.tags:
                for k, v in ctx.tags.items():
                    attributes[f"tc.tags.{k}"] = v

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
                    rpc_timeout_s: float | None = None
                    if ctx is not None and ctx.deadline_ms is not None:
                        ctx_remaining_s = (float(ctx.deadline_ms) / 1000.0) - (
                            time.monotonic() - start_time
                        )
                        if ctx_remaining_s <= 0:
                            raise ArtifactError(
                                "CallContext deadline exceeded",
                                status_code="DEADLINE_EXCEEDED",
                                retryable=False,
                            )
                        rpc_timeout_s = ctx_remaining_s
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
                        allow_cpu=allow_cpu,
                        lease_mode=lease_mode,
                        ctx=ctx,
                        timeout_s=rpc_timeout_s,
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
        ctx: CallContext | None,
        timeout_s: float | None,
        lease_mode: store_daemon_pb2.LeaseMode,
        span=None,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        disk_path_hint: str | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        allow_cpu: bool = False,
    ) -> tuple[MaterializationPayload, int]:
        resolved_fallback = fallback or self._runtime.opts.fallback
        artifact_id, key = self._resolve_identifiers(
            artifact_id, key, resolved_fallback, disk_path_hint=disk_path_hint
        )
        self._fallback.ensure_supported(resolved_fallback)
        device_id = self._resolve_device_selector(
            device, resolved_fallback, allow_cpu=allow_cpu
        )
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
                ctx=ctx,
                timeout_s=timeout_s,
                lease_mode=lease_mode,
            )
        except Exception as exc:  # noqa: BLE001
            raise_mapped_materialization_error(exc)
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
