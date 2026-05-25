#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from typing import TypedDict

import torch
from opentelemetry.trace import Status, StatusCode

from tensorcast._c_ext import (
    compute_view_index_bytes,
    get_cuda_memory_handle_with_offset,
)
from tensorcast.api import _metrics as store_metrics
from tensorcast.api import _region_cache as region_cache
from tensorcast.api._config import (
    GetArtifactOptions,
    RegionBackedMode,
    RetrievalPolicy,
    RetrievalPreference,
)
from tensorcast.api._device import device_uuid_for
from tensorcast.api._materialize import (
    MaterializationPayload,
    materialize_artifact,
)
from tensorcast.api.context import CallContext
from tensorcast.api.store.async_ops import ArtifactFuture, TrackedExecutor
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
    validate_targets,
)
from tensorcast.api.store.realization_kernel import resolve_artifact_selection
from tensorcast.api.store.retry import (
    compute_retry_delay,
    map_materialization_error,
    raise_mapped_materialization_error,
    remaining_budget,
    retry_reason_bucket,
    should_retry,
)
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.target_region_lifecycle import (
    TargetRegionRegistration,
    register_target_regions_for_realization,
    release_target_region_ids_for_realization,
)
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
    RetryPolicy,
    SpanAttributeValue,
)
from tensorcast.api.store.view_composer import compute_view_id
from tensorcast.api.store.views import (
    ResolvedViewInputs,
    TransformPlacement,
    ViewOrchestrator,
)
from tensorcast.common.selection_contract import compute_selected_index_bytes
from tensorcast.common.selection_identity import (
    compute_logical_layout_hash,
    compute_view_subset_hash,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2

logger = logging.getLogger(__name__)


def _debug_error(exc: Exception) -> str:
    detail = str(exc).strip()
    if not detail:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {detail}"


def _drop_cached_region(region_id: str, *, context: str) -> None:
    try:
        region_cache.unregister_region(region_id)
    except Exception:  # noqa: BLE001
        logger.exception(
            "%s: region-cache cleanup failed for region_id=%s",
            context,
            region_id,
        )


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


@dataclass(frozen=True, slots=True)
class _RegionBackedIntoResult:
    used: bool
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GetIntoResult:
    used_region_backed: bool
    source: str | None = None
    source_code: int | None = None
    replica_uuid: str | None = None
    total_bytes: int = 0
    fallback_reason_buckets: Mapping[str, int] = field(default_factory=dict)


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


def _resolve_source_policy_from_options(
    options: GetArtifactOptions | None,
) -> store_daemon_pb2.SourcePolicy:
    retrieval = (
        options.source
        if options is not None and options.source is not None
        else RetrievalPolicy()
    )
    preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
    if retrieval.preference is RetrievalPreference.PREFER_P2P:
        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_P2P
    elif retrieval.preference is RetrievalPreference.PREFER_DISK:
        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
    return _build_source_policy(
        preference=preference,
        allow_p2p=bool(retrieval.allow_p2p),
        allow_disk=bool(retrieval.allow_disk),
    )


def _selection_debug_fields(
    *,
    view: common_pb2.ViewSpec | None,
    tensor_names: Sequence[str] | None,
    canonical_index_hint: bytes | None,
    view_index_hint: bytes | None,
) -> dict[str, object]:
    view_ops_count = 0
    view_tensor_count = 0
    if view is not None and view.tensors:
        view_tensor_count = len(view.tensors)
        view_ops_count = sum(len(ops.ops) for ops in view.tensors.values())

    ordered_names = tuple(str(name) for name in (tensor_names or ()))
    debug: dict[str, object] = {
        "view_tensor_count": int(view_tensor_count),
        "view_ops_count": int(view_ops_count),
        "tensor_names_count": int(len(ordered_names)),
        "tensor_names_sample": list(ordered_names[:5]),
        "canonical_index_hint_len": int(len(canonical_index_hint or b"")),
        "view_index_hint_len": int(len(view_index_hint or b"")),
    }

    resolved_view_index_hint = bytes(view_index_hint or b"")
    if resolved_view_index_hint:
        try:
            debug["view_index_hint_hash"] = compute_logical_layout_hash(
                index_bytes=resolved_view_index_hint,
                needs_view_index=True,
            ).hex()
        except Exception as exc:  # noqa: BLE001
            debug["view_index_hint_hash_error"] = _debug_error(exc)

    resolved_canonical_index_hint = bytes(canonical_index_hint or b"")
    has_selection = bool(ordered_names or (view is not None and view.tensors))
    if not resolved_canonical_index_hint or not has_selection:
        return debug

    try:
        recomputed_view_index = compute_selected_index_bytes(
            canonical_index_bytes=resolved_canonical_index_hint,
            view_spec=view if view is not None and view.tensors else None,
            tensor_names=ordered_names if ordered_names else None,
        )
        debug["recomputed_view_index_len"] = int(len(recomputed_view_index))
        debug["recomputed_view_index_hash"] = compute_logical_layout_hash(
            index_bytes=recomputed_view_index,
            needs_view_index=True,
        ).hex()
        if resolved_view_index_hint:
            debug["view_index_hint_matches_recomputed"] = bool(
                resolved_view_index_hint == recomputed_view_index
            )
    except Exception as exc:  # noqa: BLE001
        debug["recomputed_view_index_error"] = _debug_error(exc)

    return debug


def _payload_total_bytes(payload: MaterializationPayload) -> int:
    return sum(int(desc.byte_length) for desc in payload.descriptors)


def _get_into_result_from_payload(
    payload: MaterializationPayload,
    *,
    fallback_reason_buckets: Mapping[str, int] | None = None,
) -> GetIntoResult:
    buckets: dict[str, int] = {}
    for key, count in (fallback_reason_buckets or {}).items():
        buckets[str(key)] = int(count)
    for key, count in (payload.retry_reason_buckets or {}).items():
        normalized_key = str(key)
        buckets[normalized_key] = buckets.get(normalized_key, 0) + int(count)
    return GetIntoResult(
        used_region_backed=False,
        source=_source_label(payload.source),
        source_code=int(payload.source) if payload.source is not None else None,
        replica_uuid=str(payload.replica_uuid) if payload.replica_uuid else None,
        total_bytes=_payload_total_bytes(payload),
        fallback_reason_buckets=buckets,
    )


def _register_target_regions(
    runtime: StoreRuntimeContext,
    *,
    target: Mapping[str, torch.Tensor],
    device_id: int,
) -> TargetRegionRegistration:
    ttl_ms = 0
    client = runtime.ensure_client()

    def _register_region(
        *,
        device_id: int,
        base_ptr: int,
        size_bytes: int,
        ttl_ms: int,
    ) -> object:
        base_ptr_value = int(base_ptr)
        size_value = int(size_bytes)
        handle_bytes, base_offset = get_cuda_memory_handle_with_offset(
            int(device_id), base_ptr_value
        )
        if base_offset:
            base_ptr_value -= int(base_offset)
            size_value += int(base_offset)
        handle = client.register_vram_region(
            device_id=int(device_id),
            size_bytes=int(size_value),
            ttl_ms=int(ttl_ms),
            cuda_ipc_handle=handle_bytes,
            region_name=None,
        )
        region_cache.register_region(
            region_id=handle.region_id,
            device_id=int(device_id),
            base_ptr=int(base_ptr_value),
            size_bytes=int(size_value),
            ttl_ms=int(ttl_ms),
        )
        return handle

    def _unregister_region(region_id: str, *, force: bool | None = None) -> bool:
        try:
            return bool(client.unregister_vram_region(region_id, force=force))
        finally:
            _drop_cached_region(
                region_id,
                context="get_into.unregister_region",
            )

    return register_target_regions_for_realization(
        register_region=_register_region,
        unregister_region=_unregister_region,
        target_tensors=target,
        device_id=device_id,
        ttl_ms=ttl_ms,
        context="get_into.register_regions",
        operation_name="get_into",
    )


def _release_target_regions(
    runtime: StoreRuntimeContext,
    *,
    region_ids: Sequence[str],
    context: str,
) -> None:
    client = runtime.ensure_client()

    def _unregister_region(region_id: str, *, force: bool | None = None) -> bool:
        try:
            return bool(client.unregister_vram_region(region_id, force=force))
        finally:
            _drop_cached_region(region_id, context=context)

    release_target_region_ids_for_realization(
        unregister_region=_unregister_region,
        region_ids=region_ids,
        context=context,
    )


def _build_region_layout_selection(
    *,
    artifact_id: str,
    canonical_index_bytes: bytes,
    region_layout: _RegionBackedLayout,
    view_spec: common_pb2.ViewSpec | None,
) -> common_pb2.ArtifactSelection:
    subset_hash = bytes(region_layout.view_subset_hash or b"")
    selection_names: tuple[str, ...] = (
        tuple(region_layout.selection_names) if subset_hash else ()
    )
    view_spec_for_selection = (
        view_spec if view_spec is not None and view_spec.tensors else None
    )

    layout_index_bytes: bytes | None = None
    if region_layout.layout.index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW:
        if region_layout.view_index_bytes:
            layout_index_bytes = bytes(region_layout.view_index_bytes)
        else:
            layout_index_bytes = compute_selected_index_bytes(
                canonical_index_bytes=canonical_index_bytes,
                view_spec=view_spec_for_selection,
                tensor_names=selection_names,
            )

    return resolve_artifact_selection(
        artifact_id=artifact_id,
        canonical_index_bytes=canonical_index_bytes,
        view_spec=view_spec_for_selection,
        tensor_names=selection_names,
        view_subset_hash=subset_hash if subset_hash else None,
        view_id=str(region_layout.view_id or ""),
        view_index_hint=layout_index_bytes,
        allow_view_id_without_spec=bool(
            region_layout.view_id and view_spec_for_selection is None
        ),
    ).proto


def _record_retrieval_event(
    runtime: StoreRuntimeContext,
    *,
    mode: str,
    artifact_id: str | None,
    key: str | None,
    detail: Mapping[str, object],
) -> None:
    logger.info(
        "store.materialize.source",
        extra={
            "tc.store.daemon": runtime.daemon_endpoint,
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
        materialize_fn: Callable[..., MaterializationPayload] = materialize_artifact,
    ) -> None:
        self._runtime = runtime
        self._views = views
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
        options: GetArtifactOptions | None = None,
        tensor_names: Sequence[str] | None = None,
        ctx: CallContext | None = None,
    ) -> dict[str, torch.Tensor]:
        payload, _ = self._perform_get_with_retry(
            artifact_id=artifact_id,
            key=key,
            device=device,
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
        tensor_names: Sequence[str] | None,
        view_id: str | None = None,
        canonical_index_hint: bytes | None = None,
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
            cancel_event=None,
            options_override=options,
            canonical_index_hint=canonical_index_hint,
            tensor_names=tensor_names,
            view_id=view_id,
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
            cancel_event=None,
            options_override=options,
            view=resolved.view_spec,
            view_id=resolved.view_id,
            placement=placement_enum,
            canonical_index_hint=resolved.canonical_index_bytes,
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
        options: GetArtifactOptions | None = None,
        tensor_names: Sequence[str] | None = None,
        view_spec: common_pb2.ViewSpec | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        ctx: CallContext | None = None,
    ) -> GetIntoResult:
        options = self._apply_client_defaults(self._build_get_options(options))
        start_time = time.monotonic()
        region_backed_result = _RegionBackedIntoResult(used=False)
        try:
            region_backed_result = self._maybe_region_backed_into(
                target=target,
                artifact_id=artifact_id,
                key=key,
                device_id=self._resolve_device_selector(device),
                options=options,
                tensor_names=tensor_names,
                view=view_spec,
                view_id=None,
                view_index_hint=view_index_hint,
            )
            if region_backed_result.used:
                store_metrics.observe_latency(
                    "get_into",
                    self._runtime.daemon_endpoint,
                    "OK",
                    time.monotonic() - start_time,
                    selection="region_backed",
                )
                return GetIntoResult(used_region_backed=True)
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
        fallback_reason_buckets: dict[str, int] = {}
        if region_backed_result.fallback_reason:
            fallback_reason_buckets[region_backed_result.fallback_reason] = 1
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
        return _get_into_result_from_payload(
            payload,
            fallback_reason_buckets=fallback_reason_buckets,
        )

    def get_into_async(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        options: GetArtifactOptions | None = None,
        tensor_names: Sequence[str] | None = None,
        view_spec: common_pb2.ViewSpec | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        ctx: CallContext | None = None,
    ) -> ArtifactFuture[None]:
        options = self._apply_client_defaults(self._build_get_options(options))
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
                        region_backed_result = self._maybe_region_backed_into(
                            target=target,
                            artifact_id=artifact_id,
                            key=key,
                            device_id=self._resolve_device_selector(device),
                            options=options,
                            tensor_names=tensor_names,
                            view=view_spec,
                            view_id=None,
                            view_index_hint=view_index_hint,
                            mark_started=_mark_region_backed_started,
                        )
                        if region_backed_result.used:
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
            cancel_event=None,
            options_override=options,
            view=resolved.view_spec,
            view_id=resolved.view_id,
            placement=placement_enum,
            canonical_index_hint=resolved.canonical_index_bytes,
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
    ) -> tuple[str | None, str | None]:
        if artifact_id and key:
            raise ArtifactError(
                "Specify either artifact_id or key, not both",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not artifact_id and not key:
            raise ArtifactError(
                "Either artifact_id or key is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return artifact_id, key

    def _resolve_device_selector(
        self,
        selector: torch.device | str | None,
        *,
        allow_cpu: bool = False,
    ) -> int:
        if selector is None:
            if not torch.cuda.is_available():
                if allow_cpu:
                    return -1
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
        bind_timing = payload.bind_timing
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
        if payload.state_dict_loader is not None:
            loaded = payload.state_dict_loader()
            if payload.descriptors:
                missing = [d.name for d in payload.descriptors if d.name not in loaded]
                if missing:
                    raise ArtifactError(
                        f"Materialization payload missing tensors: {', '.join(missing)}",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
                descriptor_names = tuple(d.name for d in payload.descriptors)
                if tuple(loaded.keys()) == descriptor_names:
                    return loaded
                projection_start = time.perf_counter()
                projected = {name: loaded[name] for name in descriptor_names}
                if bind_timing is not None:
                    bind_timing["dict_projection_sec"] = (
                        time.perf_counter() - projection_start
                    )
                return projected
            return dict(loaded)
        if payload.state_dict is not None:
            return dict(payload.state_dict)
        dict_assembly_start = time.perf_counter()
        state: dict[str, torch.Tensor] = {}
        for desc, tensor in payload.payload_iter():
            state[desc.name] = tensor
        if bind_timing is not None:
            bind_timing["dict_assembly_sec"] = time.perf_counter() - dict_assembly_start
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
        options_override: GetArtifactOptions | None,
    ) -> GetArtifactOptions:
        return options_override or self._runtime.opts.get or GetArtifactOptions()

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
                if entry.segment_offset % elem_bytes != 0:
                    raise ArtifactError(
                        "Canonical layout segment_offset must be element-size aligned",
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

    def _build_mapped_region_backed_layout(
        self,
        *,
        target: Mapping[str, torch.Tensor],
        device_id: int,
        selection_order: Sequence[str] | None = None,
        mapped_view_id: str | None = None,
        selection_index_bytes: bytes | None = None,
    ) -> _RegionBackedLayout:
        if not target:
            raise ArtifactError(
                "Target tensors are empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        target_names = {str(name) for name in target}
        if selection_order is not None:
            selection_names = tuple(str(name) for name in selection_order)
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
        else:
            selection_names = tuple(sorted(target_names))

        @dataclass
        class _StorageGroup:
            base_ptr: int
            nbytes: int
            max_used: int
            name: str

        def _storage_nbytes(tensor: torch.Tensor) -> int:
            if hasattr(tensor, "untyped_storage"):
                return int(tensor.untyped_storage().nbytes())
            return int(tensor.storage().nbytes())

        storage_groups: dict[int, _StorageGroup] = {}
        for name in selection_names:
            tensor = target.get(name)
            if tensor is None:
                raise ArtifactError(
                    f"Target tensor '{name}' missing",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not isinstance(tensor, torch.Tensor):
                raise ArtifactError(
                    f"Target tensor '{name}' must be a torch.Tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_cuda:
                raise ArtifactError(
                    f"Target tensor '{name}' must be CUDA",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if (tensor.device.index or 0) != device_id:
                raise ArtifactError(
                    f"Target tensor '{name}' on cuda:{tensor.device.index or 0}, expected cuda:{device_id}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_contiguous():
                raise ArtifactError(
                    f"Target tensor '{name}' is not contiguous",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if int(tensor.storage_offset()) != 0:
                raise ArtifactError(
                    f"Target tensor '{name}' must have storage_offset=0",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
            if tensor_bytes <= 0:
                raise ArtifactError(
                    f"Target tensor '{name}' has empty storage",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            base_ptr = int(tensor.data_ptr())
            group = storage_groups.get(base_ptr)
            if group is not None:
                raise ArtifactError(
                    "Mapped binding does not support shared storages",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            storage_groups[base_ptr] = _StorageGroup(
                base_ptr=base_ptr,
                nbytes=_storage_nbytes(tensor),
                max_used=tensor_bytes,
                name=name,
            )

        storage_list = sorted(storage_groups.values(), key=lambda group: group.base_ptr)
        if not storage_list:
            raise ArtifactError(
                "Target storages are empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        storage_specs: list[tuple[_StorageGroup, int]] = []
        for group in storage_list:
            length = int(group.max_used)
            if length <= 0:
                raise ArtifactError(
                    "Target storage must be non-empty",
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

        resolved_mapped_view_id = str(mapped_view_id or "").strip()
        needs_view_index = bool(resolved_mapped_view_id)

        layout = store_daemon_pb2.TargetLayout(
            layout_kind=store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=(
                store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
                if needs_view_index
                else store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED
            ),
            tensor_spec_kind=store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )

        region_ids: list[str] = []
        storage_id_by_ptr: dict[int, str] = {}
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
        for name in selection_names:
            tensor = target[name]
            storage_id = storage_id_by_ptr[int(tensor.data_ptr())]
            tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
            offsets.append(
                store_daemon_pb2.TargetTensorOffset(
                    name=name,
                    storage_id=storage_id,
                    storage_offset=0,
                    logical_length=tensor_bytes,
                )
            )
        layout.offsets.extend(offsets)

        storage_base_offsets: dict[str, int] = {}
        cursor = 0
        for storage in layout.storages:
            storage_base_offsets[storage.storage_id] = cursor
            cursor += int(storage.storage_length)
        logical_total_size = cursor

        index_entries: list[CanonicalIndexEntry] = []
        for name in sorted(selection_names):
            tensor = target[name]
            storage_id = storage_id_by_ptr[int(tensor.data_ptr())]
            logical_offset = storage_base_offsets[storage_id]
            tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
            index_entries.append(
                CanonicalIndexEntry(
                    name=name,
                    dtype=tensor.dtype,
                    shape=tuple(int(v) for v in tensor.shape),
                    stride=tuple(int(v) for v in tensor.stride()),
                    storage_offset=int(tensor.storage_offset()),
                    segment_offset=logical_offset,
                    size_bytes=tensor_bytes,
                )
            )
        index = CanonicalIndex(
            entries=tuple(index_entries),
            total_size_bytes=logical_total_size,
            avbs_hash="",
        )
        index_bytes = canonical_index_to_bytes(index)
        resolved_selection_index_bytes = (
            bytes(selection_index_bytes)
            if selection_index_bytes is not None
            else index_bytes
        )
        layout.logical_layout_hash = compute_logical_layout_hash(
            index_bytes=index_bytes,
            needs_view_index=needs_view_index,
        )
        if resolved_mapped_view_id:
            layout.view_id = resolved_mapped_view_id

        return _RegionBackedLayout(
            layout=layout,
            region_ids=tuple(region_ids),
            logical_total_size=logical_total_size,
            view_index_bytes=(
                resolved_selection_index_bytes if needs_view_index else None
            ),
            view_id=resolved_mapped_view_id or None,
            selection_names=selection_names,
            view_subset_hash=None,
        )

    def _materialize_into_target(
        self,
        *,
        target: dict[str, torch.Tensor],
        artifact_id: str,
        device_id: int,
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
                expires_at=time.monotonic(),
            )
            self._runtime.cache_artifact_index(cache_entry)

        source_policy = _resolve_source_policy_from_options(options)

        client = self._runtime.ensure_client()
        started = False
        attempt = 0
        while attempt < 2:
            try:
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
            except ArtifactError as exc:
                message = str(exc)
                if (
                    attempt == 0
                    and exc.status_code == "FAILED_PRECONDITION"
                    and "No registered region covers the target tensors" in message
                ):
                    _register_target_regions(
                        self._runtime, target=target, device_id=device_id
                    )
                    attempt += 1
                    continue
                raise

            if mark_started is not None and not started:
                mark_started()
                started = True
            try:
                selection = _build_region_layout_selection(
                    artifact_id=artifact_id,
                    canonical_index_bytes=canonical_bytes,
                    region_layout=region_layout,
                    view_spec=view,
                )
                response = client.materialize_into_target(
                    selection=selection,
                    target_layout=region_layout.layout,
                    device_uuid=device_uuid_for(device_id),
                    source_policy=source_policy,
                    placement=placement,
                )
            except Exception as exc:  # noqa: BLE001
                error = map_materialization_error(exc)
                if attempt == 0 and error.status_code in {
                    "DATA_LOSS",
                    "FAILED_PRECONDITION",
                    "NOT_FOUND",
                }:
                    _release_target_regions(
                        self._runtime,
                        region_ids=region_layout.region_ids,
                        context="get_into.retry_region_invalidation",
                    )
                    _register_target_regions(
                        self._runtime, target=target, device_id=device_id
                    )
                    attempt += 1
                    continue
                if error.status_code in {"DATA_LOSS", "FAILED_PRECONDITION"}:
                    _release_target_regions(
                        self._runtime,
                        region_ids=region_layout.region_ids,
                        context="get_into.failed_region_invalidation",
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
                _release_target_regions(
                    self._runtime,
                    region_ids=region_layout.region_ids,
                    context="get_into.non_success_region_invalidation",
                )
                raise ArtifactError(
                    "MaterializeIntoTarget returned non-success status",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            store_metrics.record_region_backed_verification_skipped(
                self._runtime.daemon_endpoint
            )
            return None

        raise ArtifactError(
            "MaterializeIntoTarget retry failed to produce a response",
            status_code="DATA_LOSS",
            retryable=False,
        )

    def _maybe_region_backed_into(
        self,
        *,
        target: dict[str, torch.Tensor],
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        options: GetArtifactOptions,
        tensor_names: Sequence[str] | None,
        view: common_pb2.ViewSpec | None,
        view_id: str | None,
        view_index_hint: bytes | None,
        span=None,
        mark_started: Callable[[], None] | None = None,
    ) -> _RegionBackedIntoResult:
        mode = options.region_backed_mode
        if mode is RegionBackedMode.DISABLE:
            return _RegionBackedIntoResult(used=False)
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
            return _RegionBackedIntoResult(
                used=False,
                fallback_reason="key_not_supported",
            )
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
            return _RegionBackedIntoResult(
                used=False,
                fallback_reason="missing_artifact_id",
            )
        try:
            self._materialize_into_target(
                target=target,
                artifact_id=artifact_id,
                device_id=device_id,
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
            return _RegionBackedIntoResult(
                used=False,
                fallback_reason="layout_mismatch",
            )
        return _RegionBackedIntoResult(used=True)

    def _materialize(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        options: GetArtifactOptions,
        cancel_event: threading.Event | None = None,
        ctx: CallContext | None,
        timeout_s: float | None,
        lease_mode: store_daemon_pb2.LeaseMode,
        span=None,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
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
            cancel_event=cancel_event,
            ctx=ctx,
            timeout_s=timeout_s,
            lease_mode=lease_mode,
            span=span,
            view=view,
            view_id=view_id,
            placement=placement,
            canonical_index_hint=canonical_index_hint,
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
        cancel_event: threading.Event | None = None,
        ctx: CallContext | None,
        timeout_s: float | None,
        lease_mode: store_daemon_pb2.LeaseMode,
        span=None,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
    ) -> MaterializationPayload:
        client = self._runtime.ensure_client()
        resolved_artifact_id = artifact_id
        canonical_hint: bytes | None = canonical_index_hint
        generation_hint: int | None = None
        cached_entry: ArtifactCacheEntry | None = None
        verify_checksums = bool(options.verify_checksums)
        retrieval_policy = options.source or RetrievalPolicy()
        allow_p2p = bool(retrieval_policy.allow_p2p)
        allow_disk = bool(retrieval_policy.allow_disk)
        requested_disk = retrieval_policy.preference is RetrievalPreference.PREFER_DISK
        source_policy = _resolve_source_policy_from_options(options)
        if resolved_artifact_id is None and key:
            try:
                resolved_mapping = self._runtime.resolve_key_mapping_cached(key=key)
                if isinstance(resolved_mapping, tuple):
                    resolved_artifact_id = resolved_mapping[0]
                else:
                    resolved_artifact_id = resolved_mapping
            except Exception:  # noqa: BLE001
                logger.exception(
                    "store.materialize.key_mapping_prefetch_failed",
                    extra={
                        "tc.store.daemon": self._runtime.daemon_endpoint,
                        "tc.store.key": key,
                    },
                )
        if resolved_artifact_id and canonical_hint is None:
            cached_entry = self._runtime.get_artifact_index_cached(resolved_artifact_id)
            if cached_entry is not None:
                canonical_hint = cached_entry.canonical_index_bytes
                generation_hint = cached_entry.generation

        request_artifact_id = resolved_artifact_id or artifact_id
        view_subset_hash = (
            compute_view_subset_hash(tensor_names) if tensor_names else None
        )
        result = self._materialize_fn(
            client=client,
            daemon_address=self._runtime.daemon_endpoint,
            server_config=self._runtime.capabilities.server_config,
            device_id=device_id,
            artifact_id=request_artifact_id,
            key=key,
            options=options,
            view=view,
            view_id=view_id,
            placement=placement,
            canonical_index_hint=canonical_hint,
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
        disallowed_sources: set[store_daemon_pb2.MaterializationSource] = set()
        if not allow_p2p:
            disallowed_sources.add(
                store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_P2P
            )
        if not allow_disk:
            disallowed_sources.add(
                store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_DISK
            )
        if disallowed_sources and result.source in disallowed_sources:
            source_label = _source_label(result.source) or "non-disk"
            self._release_materialized(result, client)
            raise ArtifactError(
                f"Materialization source {source_label} not allowed by retrieval policy",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        source_label = _source_label(result.source) or ""
        _record_retrieval_event(
            self._runtime,
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
            )
        if span is not None:
            span.add_event(
                "store.materialize.p2p",
                {
                    "tc.artifact.id": result.artifact_id or artifact_id or "",
                    "tc.store.disk_requested": bool(requested_disk),
                    "tc.store.allow_p2p": bool(allow_p2p),
                    "tc.store.allow_disk": bool(allow_disk),
                    "tc.store.preference": str(retrieval_policy.preference.value),
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
            state_dict_loader = None
            if base_payload.state_dict_loader is not None:
                base_state_dict_loader = base_payload.state_dict_loader
                requested_name_set = set(requested_names)

                def _filtered_state_dict_loader() -> dict[str, torch.Tensor]:
                    loaded = base_state_dict_loader()
                    return {
                        name: tensor
                        for name, tensor in loaded.items()
                        if name in requested_name_set
                    }

                state_dict_loader = _filtered_state_dict_loader

            result = MaterializationPayload(
                artifact_id=base_payload.artifact_id,
                canonical_index_bytes=canonical_bytes,
                descriptors=filtered_descriptors,
                payload_iter=_filtered_iter,
                state_dict_loader=state_dict_loader,
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
                materialize_timing=base_payload.materialize_timing,
                bind_timing=base_payload.bind_timing,
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
        cancel_event: threading.Event | None,
        options_override: GetArtifactOptions | None,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: TransformPlacement | None = None,
        canonical_index_hint: bytes | None = None,
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        allow_cpu: bool = False,
        ctx: CallContext | None = None,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
    ) -> tuple[MaterializationPayload, int]:
        options_snapshot = self._build_get_options(options_override)
        retrieval_policy = options_snapshot.source or RetrievalPolicy()
        wait_for_shared_disk_ms = int(options_snapshot.wait_for_shared_disk_ms)
        options_override_no_wait = options_snapshot
        if wait_for_shared_disk_ms > 0:
            options_override_no_wait = options_snapshot.model_copy(
                update={"wait_for_shared_disk_ms": 0}
            )
        wait_attempted = False

        policy = self._runtime.retry_policies.get(
            method, self._runtime.retry_policies.get("get")
        )
        attempt = 1
        start_time = time.monotonic()
        if ctx is not None and ctx.deadline_ms is not None:
            ctx_budget_s = float(ctx.deadline_ms) / 1000.0
            if ctx_budget_s <= 0:
                raise ArtifactError(
                    "CallContext deadline exceeded",
                    status_code="DEADLINE_EXCEEDED",
                    retryable=False,
                )
            if policy is None:
                policy = RetryPolicy(
                    deadline_seconds=ctx_budget_s,
                    max_attempts=1,
                    base_backoff_seconds=0.0,
                    backoff_multiplier=1.0,
                    jitter=0.0,
                )
            elif ctx_budget_s != policy.deadline_seconds:
                policy = RetryPolicy(
                    deadline_seconds=ctx_budget_s,
                    max_attempts=policy.max_attempts,
                    base_backoff_seconds=policy.base_backoff_seconds,
                    backoff_multiplier=policy.backoff_multiplier,
                    jitter=policy.jitter,
                )
        retry_reason_buckets: dict[str, int] = {}

        def record_retry_reason(error: ArtifactError) -> None:
            bucket = retry_reason_bucket(error)
            retry_reason_buckets[bucket] = retry_reason_buckets.get(bucket, 0) + 1

        def attach_retry_metadata(
            payload: MaterializationPayload,
            *,
            attempts: int,
            exit_reason: str,
        ) -> MaterializationPayload:
            elapsed = time.monotonic() - start_time
            deadline: float | None = None
            remaining: float | None = None
            if policy is not None and policy.deadline_seconds > 0:
                deadline = float(policy.deadline_seconds)
                remaining = max(0.0, deadline - elapsed)
            return replace(
                payload,
                retry_attempts=max(1, int(attempts)),
                retry_reason_buckets=dict(retry_reason_buckets),
                budget_deadline_sec=deadline,
                budget_elapsed_sec=float(elapsed),
                budget_remaining_sec=remaining,
                budget_exit_reason=exit_reason,
            )

        span_name = "Store/GetInto" if method == "get_into" else "Store/Get"
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.daemon": self._runtime.daemon_endpoint,
            "tc.store.session_id": self._runtime.session_id,
            "tc.store.method": method,
            "tc.store.lookup.by_key": bool(key),
            "tc.store.lookup.by_id": bool(artifact_id),
            "tc.store.preference": str(retrieval_policy.preference.value),
            "tc.store.allow_p2p": bool(retrieval_policy.allow_p2p),
            "tc.store.allow_disk": bool(retrieval_policy.allow_disk),
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
                        cancel_event=cancel_event,
                        options_override=options_override_no_wait,
                        span=span,
                        view=view,
                        view_id=view_id,
                        placement=placement,
                        canonical_index_hint=canonical_index_hint,
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
                    materialized = attach_retry_metadata(
                        materialized,
                        attempts=attempt,
                        exit_reason="success",
                    )
                    record_outcome("OK")
                    span.set_attribute("tc.store.retry.count", attempt)
                    span.set_attribute(
                        "tc.store.retry.reason_buckets", str(retry_reason_buckets)
                    )
                    span.set_attribute("tc.store.budget.exit_reason", "success")
                    span.set_attribute("tc.device.id", int(device_id))
                    span.set_attribute("tc.artifact.id", materialized.artifact_id)
                    span.set_attribute(
                        "tc.store.used_disk", bool(materialized.disk_path)
                    )
                    span.set_attribute("tc.store.source", source_label or "")
                    span.set_status(Status(StatusCode.OK))
                    return materialized, device_id
                except Exception as exc:  # noqa: BLE001
                    exc_is_artifact_error = isinstance(exc, ArtifactError)
                    error = map_materialization_error(exc)
                    record_retry_reason(error)
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
                        if (
                            wait_for_shared_disk_ms > 0
                            and not wait_attempted
                            and error.status_code == "NOT_FOUND"
                        ):
                            wait_attempted = True
                            span.add_event(
                                "store.wait_for_shared_disk",
                                {
                                    "tc.store.wait_for_shared_disk_ms": wait_for_shared_disk_ms
                                },
                            )

                            wait_timeout_s: float | None = None
                            if ctx is not None and ctx.deadline_ms is not None:
                                ctx_remaining_s = (float(ctx.deadline_ms) / 1000.0) - (
                                    time.monotonic() - start_time
                                )
                                if ctx_remaining_s <= 0:
                                    raise ArtifactError(
                                        "CallContext deadline exceeded",
                                        status_code="DEADLINE_EXCEEDED",
                                        retryable=False,
                                    ) from None
                                wait_timeout_s = ctx_remaining_s
                            else:
                                required_s = (
                                    float(wait_for_shared_disk_ms) / 1000.0
                                ) + 5.0
                                if required_s > 60.0:
                                    wait_timeout_s = required_s

                            try:
                                materialized, device_id = self._attempt_get(
                                    artifact_id=artifact_id,
                                    key=key,
                                    device=device,
                                    cancel_event=cancel_event,
                                    options_override=options_override,
                                    span=span,
                                    view=view,
                                    view_id=view_id,
                                    placement=placement,
                                    canonical_index_hint=canonical_index_hint,
                                    tensor_names=tensor_names,
                                    view_data_hash=view_data_hash,
                                    view_index_hint=view_index_hint,
                                    replica_uuid=replica_uuid,
                                    allow_cpu=allow_cpu,
                                    lease_mode=lease_mode,
                                    ctx=ctx,
                                    timeout_s=wait_timeout_s,
                                )
                            except Exception as exc:  # noqa: BLE001
                                error = map_materialization_error(exc)
                                record_retry_reason(error)
                            else:
                                summary = self._summarize_materialized(
                                    materialized, tensor_names
                                )
                                selection_label = summary["selection"]
                                source_label = _source_label(materialized.source)
                                count_value = summary["count"]
                                bytes_value = summary["bytes"]
                                if count_value is not None:
                                    span.set_attribute(
                                        "tc.tensor.count", int(count_value)
                                    )
                                if bytes_value is not None:
                                    span.set_attribute(
                                        "tc.tensor.bytes", int(bytes_value)
                                    )
                                if selection_label:
                                    span.set_attribute(
                                        "tc.tensor.selection", selection_label
                                    )
                                materialized = attach_retry_metadata(
                                    materialized,
                                    attempts=attempt + 1,
                                    exit_reason="wait_for_shared_disk",
                                )
                                record_outcome("OK")
                                span.set_attribute("tc.store.retry.count", attempt + 1)
                                span.set_attribute(
                                    "tc.store.retry.reason_buckets",
                                    str(retry_reason_buckets),
                                )
                                span.set_attribute(
                                    "tc.store.budget.exit_reason",
                                    "wait_for_shared_disk",
                                )
                                span.set_attribute("tc.device.id", int(device_id))
                                span.set_attribute(
                                    "tc.artifact.id", materialized.artifact_id
                                )
                                span.set_attribute(
                                    "tc.store.used_disk",
                                    bool(materialized.disk_path),
                                )
                                span.set_attribute(
                                    "tc.store.source", source_label or ""
                                )
                                span.set_status(Status(StatusCode.OK))
                                return materialized, device_id

                        record_outcome(error.status_code)
                        span.set_attribute(
                            "tc.store.retry.reason_buckets", str(retry_reason_buckets)
                        )
                        span.set_attribute(
                            "tc.store.budget.exit_reason",
                            "non_retryable_or_max_attempts",
                        )
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        if exc_is_artifact_error:
                            raise
                        raise error from None
                    assert policy is not None
                    delay = compute_retry_delay(policy, attempt)
                    remaining = remaining_budget(policy, start_time)
                    if remaining is not None and remaining <= 0:
                        record_outcome(error.status_code)
                        span.set_attribute(
                            "tc.store.retry.reason_buckets", str(retry_reason_buckets)
                        )
                        span.set_attribute(
                            "tc.store.budget.exit_reason", "retry_deadline_exhausted"
                        )
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        if exc_is_artifact_error:
                            raise
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
        tensor_names: Sequence[str] | None = None,
        view_data_hash: str | None = None,
        view_index_hint: bytes | None = None,
        replica_uuid: str | None = None,
        allow_cpu: bool = False,
    ) -> tuple[MaterializationPayload, int]:
        artifact_id, key = self._resolve_identifiers(artifact_id, key)
        options = self._build_get_options(options_override)
        device_id = self._resolve_device_selector(device, allow_cpu=allow_cpu)
        try:
            materialized = self._materialize(
                artifact_id=artifact_id,
                key=key,
                device_id=device_id,
                options=options,
                cancel_event=cancel_event,
                span=span,
                view=view,
                view_id=view_id,
                placement=placement,
                canonical_index_hint=canonical_index_hint,
                tensor_names=tensor_names,
                view_data_hash=view_data_hash,
                view_index_hint=view_index_hint,
                replica_uuid=replica_uuid,
                ctx=ctx,
                timeout_s=timeout_s,
                lease_mode=lease_mode,
            )
        except Exception as exc:  # noqa: BLE001
            if "selection.logical_layout_hash does not match resolved selection" in str(
                exc
            ):
                logger.error(
                    "store.materialize.selection_layout_mismatch",
                    extra={
                        "tc.store.daemon": self._runtime.daemon_endpoint,
                        "tc.artifact.id": artifact_id or "",
                        "tc.store.key": key or "",
                        "tc.store.selection_debug": _selection_debug_fields(
                            view=view,
                            tensor_names=tensor_names,
                            canonical_index_hint=canonical_index_hint,
                            view_index_hint=view_index_hint,
                        ),
                    },
                )
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


__all__ = ["GetIntoResult", "MaterializationPipeline"]
