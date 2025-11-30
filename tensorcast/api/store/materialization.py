#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError

import torch
from opentelemetry.trace import Status, StatusCode

from tensorcast.api import _metrics as store_metrics
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._materialize import MaterializedArtifact, materialize_artifact
from tensorcast.api.store.async_ops import ArtifactFuture, TrackedExecutor
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    validate_targets,
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
        materialize_fn: Callable[..., MaterializedArtifact] = materialize_artifact,
    ) -> None:
        self._runtime = runtime
        self._views = views
        self._fallback = FallbackResolver(runtime)
        self._executor = TrackedExecutor(runtime)
        self._materialize_fn = materialize_fn

    def set_materialize_fn(
        self, materialize_fn: Callable[..., MaterializedArtifact]
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
    ) -> dict[str, torch.Tensor]:
        materialized, _ = self._perform_get_with_retry(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            method="get",
            cancel_event=None,
            options_override=options,
        )
        return materialized.state_dict

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

        materialized, _ = self._perform_get_with_retry(
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
        return materialized.state_dict

    def get_async(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
        options: GetArtifactOptions | None = None,
    ) -> ArtifactFuture[dict[str, torch.Tensor]]:
        cancel_event = threading.Event()
        mat_lock = threading.Lock()
        mat_ref: dict[str, MaterializedArtifact | None] = {"value": None}

        def _task() -> dict[str, torch.Tensor]:
            materialized: MaterializedArtifact | None = None
            try:
                materialized, _ = self._perform_get_with_retry(
                    artifact_id=artifact_id,
                    key=key,
                    device=device,
                    fallback=fallback,
                    method="get",
                    cancel_event=cancel_event,
                    options_override=options,
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
                return materialized.state_dict
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
    ) -> None:
        materialized, device_id = self._perform_get_with_retry(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            method="get_into",
            cancel_event=None,
            options_override=options,
        )
        try:
            canonical_index = canonical_index_from_bytes(
                materialized.canonical_index_bytes
            )
            pairs = validate_targets(
                canonical_index=canonical_index,
                target=target,
                source=materialized.state_dict,
                device_id=device_id,
            )
            for tgt, src in pairs:
                tgt.copy_(src)
        finally:
            self._release_materialized(materialized, self._runtime.ensure_client())

    def get_into_async(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
        options: GetArtifactOptions | None = None,
    ) -> ArtifactFuture[None]:
        cancel_event = threading.Event()
        mat_lock = threading.Lock()
        mat_ref: dict[str, MaterializedArtifact | None] = {"value": None}

        def _task() -> None:
            materialized: MaterializedArtifact | None = None
            try:
                materialized, device_id = self._perform_get_with_retry(
                    artifact_id=artifact_id,
                    key=key,
                    device=device,
                    fallback=fallback,
                    method="get_into",
                    cancel_event=cancel_event,
                    options_override=options,
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
                    canonical_index = canonical_index_from_bytes(
                        materialized.canonical_index_bytes
                    )
                    pairs = validate_targets(
                        canonical_index=canonical_index,
                        target=target,
                        source=materialized.state_dict,
                        device_id=device_id,
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

        materialized, device_id = self._perform_get_with_retry(
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
            layout_bytes = (
                materialized.view_index_bytes or materialized.canonical_index_bytes
            )
            layout_index = canonical_index_from_bytes(layout_bytes)
            pairs = validate_targets(
                canonical_index=layout_index,
                target=target,
                source=materialized.state_dict,
                device_id=device_id,
            )
            for tgt, src in pairs:
                tgt.copy_(src)
        finally:
            self._release_materialized(materialized, self._runtime.ensure_client())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_identifiers(
        artifact_id: str | None,
        key: str | None,
        fallback: FallbackOptions | None,
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
            if disk_path:
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
    ) -> MaterializedArtifact:
        client = self._runtime.ensure_client()
        disk_error: Exception | None = None
        disk_path: str | None = disk_path_hint
        resolved_artifact_id = artifact_id
        fallback_opts = fallback
        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        requested_disk = False
        if (view is not None or view_id is not None) and fallback_opts:
            raise ArtifactError(
                "Disk fallback is not supported for view materialization",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        if fallback_opts and (
            fallback_opts.prefer_disk
            or fallback_opts.disk_path
            or not fallback_opts.allow_p2p
        ):
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
            if fallback_opts and not fallback_opts.allow_p2p:
                if disk_error is not None:
                    raise ArtifactError(
                        f"Disk fallback lookup failed: {disk_error}",
                        status_code="UNAVAILABLE",
                        retryable=True,
                    ) from disk_error
                if not disk_path:
                    raise ArtifactError(
                        "Disk fallback required but disk_path unavailable",
                        status_code="NOT_FOUND",
                        retryable=False,
                    )

        request_artifact_id = resolved_artifact_id or artifact_id
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
            canonical_index_hint=canonical_index_hint,
            disk_path_hint=disk_path,
            preference=preference,
        )
        if fallback_opts and not fallback_opts.allow_p2p:
            disallowed_sources = {
                store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_P2P,
                store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_LOCAL_REPLICA,
            }
            if result.source in disallowed_sources:
                source_label = _source_label(result.source) or "non-disk"
                self._release_materialized(result, client)
                raise ArtifactError(
                    f"Disk-only fallback requested but daemon served from {source_label}",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        source_label = _source_label(result.source)
        self._fallback.record_event(
            mode="disk" if requested_disk else "p2p",
            artifact_id=result.artifact_id or request_artifact_id,
            key=key,
            detail={
                "disk_requested": bool(fallback and fallback.prefer_disk),
                "allow_p2p": bool(fallback is None or fallback.allow_p2p),
                "source": source_label or "",
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
                    "tc.store.disk_requested": bool(fallback and fallback.prefer_disk),
                    "tc.store.allow_p2p": bool(fallback is None or fallback.allow_p2p),
                    "tc.store.source": source_label or "",
                },
            )
        if cancel_event and cancel_event.is_set():
            self._release_materialized(result, client)
            raise CancelledError
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
    ) -> tuple[MaterializedArtifact, int]:
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
            "tc.store.fallback.prefer_disk": bool(fallback and fallback.prefer_disk),
            "tc.store.fallback.allow_p2p": bool(fallback is None or fallback.allow_p2p),
        }
        source_label: str | None = None

        with self._runtime.operation_span(span_name, attributes) as span:

            def record_outcome(status: str) -> None:
                duration = time.monotonic() - start_time
                store_metrics.observe_latency(
                    method,
                    self._runtime.daemon_endpoint,
                    status,
                    duration,
                    source=source_label,
                )
                span.set_attribute("tc.store.status", status)
                if status not in {"OK", "CANCELLED"}:
                    store_metrics.increment_error(
                        method,
                        self._runtime.daemon_endpoint,
                        status,
                        source=source_label,
                    )

            while True:
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
                    )
                    source_label = _source_label(materialized.source)
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
    ) -> tuple[MaterializedArtifact, int]:
        resolved_fallback = fallback or self._runtime.opts.fallback
        artifact_id, key = self._resolve_identifiers(
            artifact_id, key, resolved_fallback
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
            )
        except Exception as exc:  # noqa: BLE001
            raise map_materialization_error(exc) from exc
        return materialized, device_id

    def _release_materialized(self, materialized: MaterializedArtifact, client) -> None:
        replica_uuid = materialized.replica_uuid
        if not replica_uuid:
            return
        disk_path = materialized.disk_path or ""
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
