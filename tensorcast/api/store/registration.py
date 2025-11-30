#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Callable, Sequence

import torch
from opentelemetry.trace import Status, StatusCode

from tensorcast.api import _metrics as store_metrics
from tensorcast.api._config import PlanType, RegisterArtifactOptions
from tensorcast.api._device import resolve_device
from tensorcast.api._register import (
    RegisteredArtifact as _RegisterHandle,
)
from tensorcast.api._register import (
    RegistrationResult,
    ViewRegistrationContext,
    _compute_view_plan_metadata,
    _materialize_canonical_tensors,
    _merge_canonical_ranges,
    _register_artifact_core,
)
from tensorcast.api.store.async_ops import ArtifactFuture, TrackedExecutor
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_from_result,
    lease_handle_from_result,
    replica_info_from_result,
    select_device_for_put,
)
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.retry import (
    compute_retry_delay,
    map_registration_error,
    remaining_budget,
    should_retry,
)
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.types import ArtifactError, SpanAttributeValue
from tensorcast.api.store.views import (
    ResolvedViewInputs,
    TransformPlacement,
    ViewOrchestrator,
)
from tensorcast.proto.daemon.v1 import store_daemon_pb2

logger = logging.getLogger(__name__)


class RegistrationPipeline:
    """Registration orchestration shared by sync and async verbs."""

    def __init__(
        self,
        runtime: StoreRuntimeContext,
        views: ViewOrchestrator,
        *,
        register_fn: Callable[..., RegistrationResult] = _register_artifact_core,
    ) -> None:
        self._runtime = runtime
        self._views = views
        self._executor = TrackedExecutor(runtime)
        self._default_lease_ttl_ms = StoreRuntimeContext._DEFAULT_LEASE_TTL_MS
        self._register_fn = register_fn

    def register(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> RegisteredArtifact:
        return self._perform_registration(
            tensors,
            artifact_id=artifact_id,
            key=key,
            plan=PlanType.VRAM_LEASED,
            options_override=options,
            ttl_override=ttl_ms,
        )

    def register_async(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        cancel_event = threading.Event()
        handle_lock = threading.Lock()
        handle_ref: dict[str, _RegisterHandle | None] = {"handle": None}

        def _on_begin(handle: _RegisterHandle) -> None:
            with handle_lock:
                handle_ref["handle"] = handle

        def _task() -> RegisteredArtifact:
            try:
                return self._perform_registration(
                    tensors,
                    artifact_id=artifact_id,
                    key=key,
                    plan=PlanType.VRAM_LEASED,
                    cancel_event=cancel_event,
                    on_begin=_on_begin,
                    options_override=options,
                    ttl_override=ttl_ms,
                )
            finally:
                with handle_lock:
                    handle_ref["handle"] = None

        def _cancel() -> bool:
            if cancel_event.is_set():
                return False
            cancel_event.set()
            with handle_lock:
                handle = handle_ref.get("handle")
            if handle is not None:
                with contextlib.suppress(Exception):
                    return bool(handle.abort(timeout_s=5.0))
            return True

        return self._executor.submit(_task, cancel_callback=_cancel)

    def register_view(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        view_id: str | None = None,
        placement: str | None = None,
        ttl_ms: int | None = None,
        allow_partial: bool = False,
        options: RegisterArtifactOptions | None = None,
        resolver: Callable[..., ResolvedViewInputs] | None = None,
    ) -> RegisteredArtifact:
        view_resolver = resolver or self._views.resolve_view_inputs
        resolved = view_resolver(
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
        )
        if resolved.variant == "id":
            raise ArtifactError(
                "View registration via pre-existing view_id is not supported yet",
                status_code="UNIMPLEMENTED",
                retryable=False,
            )
        assert resolved.build_result is not None
        if resolved.build_result.is_identity:
            raise ArtifactError(
                "View registration requires explicit view operations",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        placement_enum = self._views.resolve_transform_placement(
            placement,
            has_transpose=resolved.has_transpose,
            for_registration=True,
        )
        assert resolved.canonical_index_bytes is not None
        canonical_index_bytes = resolved.canonical_index_bytes
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        plan_metadata = _compute_view_plan_metadata(
            canonical_index_bytes, resolved.build_result
        )
        canonical_ranges = _merge_canonical_ranges(plan_metadata.write_chunks)
        view_options = store_daemon_pb2.ViewRegistrationOptions()
        if resolved.view_id:
            view_options.view_id = resolved.view_id
        assert resolved.build_result.proto is not None
        view_options.spec.CopyFrom(resolved.build_result.proto)
        view_options.placement = placement_enum
        view_options.canonical_size_bytes = canonical_index.total_size_bytes
        for rng in canonical_ranges:
            range_proto = view_options.ranges.add()
            range_proto.offset = rng.offset
            range_proto.length = rng.length
        view_options.allow_partial = bool(allow_partial)
        upload_tensors: dict[str, torch.Tensor]
        if placement_enum == TransformPlacement.TRANSFORM_PLACEMENT_SERVER:
            upload_tensors = dict(tensors)
        elif placement_enum == TransformPlacement.TRANSFORM_PLACEMENT_CLIENT:
            upload_tensors = _materialize_canonical_tensors(
                canonical_index_bytes, resolved.build_result, tensors
            )
        else:
            raise ArtifactError(
                "Unsupported placement for register_view",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        view_ctx = ViewRegistrationContext(
            canonical_index_bytes=canonical_index_bytes,
            view_options=view_options,
            placement=placement_enum,
            plan=plan_metadata,
            tensors=dict(tensors),
            canonical_ranges=canonical_ranges,
            allow_partial=allow_partial,
        )
        return self._perform_registration(
            upload_tensors,
            key=key,
            plan=PlanType.VRAM_COALESCED,
            options_override=options,
            ttl_override=ttl_ms,
            view_registration=view_ctx,
        )

    def put(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        return self._perform_registration(
            tensors,
            artifact_id=artifact_id,
            key=key,
            plan=PlanType.VRAM_COALESCED,
            options_override=options,
            device_override=device,
        )

    def put_async(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        cancel_event = threading.Event()
        handle_lock = threading.Lock()
        handle_ref: dict[str, _RegisterHandle | None] = {"handle": None}

        def _on_begin(handle: _RegisterHandle) -> None:
            with handle_lock:
                handle_ref["handle"] = handle

        def _task() -> RegisteredArtifact:
            try:
                return self._perform_registration(
                    tensors,
                    artifact_id=artifact_id,
                    key=key,
                    plan=PlanType.VRAM_COALESCED,
                    cancel_event=cancel_event,
                    on_begin=_on_begin,
                    options_override=options,
                    device_override=device,
                )
            finally:
                with handle_lock:
                    handle_ref["handle"] = None

        def _cancel() -> bool:
            if cancel_event.is_set():
                return False
            cancel_event.set()
            with handle_lock:
                handle = handle_ref.get("handle")
            if handle is not None:
                with contextlib.suppress(Exception):
                    return bool(handle.abort(timeout_s=5.0))
            return True

        return self._executor.submit(_task, cancel_callback=_cancel)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _attempt_registration(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None,
        key: str | None,
        plan: PlanType,
        cancel_event: threading.Event | None = None,
        on_begin: Callable[[_RegisterHandle], None] | None = None,
        options_override: RegisterArtifactOptions | None = None,
        ttl_override: int | None = None,
        device_override: int | torch.device | None = None,
        view_registration: ViewRegistrationContext | None = None,
    ) -> RegisteredArtifact:
        material = dict(tensors)
        if not material:
            raise ArtifactError(
                "Artifact tensors must not be empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if view_registration is not None and plan is not PlanType.VRAM_COALESCED:
            raise ArtifactError(
                "View registration requires vram_coalesced plan",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resolved_key = key
        options = options_override
        if options is not None:
            if options.plan is not plan:
                options = replace(options, plan=plan)
            if plan is PlanType.VRAM_LEASED and not options.lease_in_place:
                options = replace(options, lease_in_place=True)
            if resolved_key is None:
                resolved_key = options.key
            elif options.key != resolved_key:
                options = replace(options, key=resolved_key)
        else:
            options = RegisterArtifactOptions(
                plan=plan,
                lease_in_place=plan is PlanType.VRAM_LEASED,
                key=resolved_key,
            )
        normalized_artifact_id = artifact_id.strip() if artifact_id else None
        if normalized_artifact_id == "":
            normalized_artifact_id = None
        ttl_ms = (
            ttl_override
            if ttl_override is not None
            else self._default_lease_ttl_ms
            if plan is PlanType.VRAM_LEASED
            else None
        )
        if plan is PlanType.VRAM_LEASED:
            device_id = None
        elif device_override is not None:
            device_id = resolve_device(device_override)
        else:
            device_id = select_device_for_put(material)
        try:
            registration_result = self._register_fn(
                artifact=material,
                options=options,
                device_id=device_id,
                ttl_ms=ttl_ms,
                client_artifact_id=normalized_artifact_id,
                force_lease_in_place=plan is PlanType.VRAM_LEASED,
                prevalidate_disk=self._should_prevalidate_disk(options),
                client=self._runtime.ensure_client(),
                daemon_address=self._runtime.daemon_endpoint,
                cancel_event=cancel_event,
                on_begin=on_begin,
                view=view_registration,
            )
        except Exception as exc:  # noqa: BLE001
            raise map_registration_error(exc) from exc
        return self._registration_to_artifact(registration_result)

    def _should_prevalidate_disk(self, options: RegisterArtifactOptions) -> bool:
        disk_path = options.disk_path
        if disk_path is None:
            return False
        if not isinstance(disk_path, str):
            return False
        return disk_path.strip() != ""

    def _registration_to_artifact(
        self, result: RegistrationResult
    ) -> RegisteredArtifact:
        canonical_index = canonical_index_from_result(result)
        replica = replica_info_from_result(result)
        lease_handle = lease_handle_from_result(result.lease)
        self._runtime.track_lease(result.lease)
        return RegisteredArtifact(
            artifact_id=result.descriptor.artifact_id,
            replica=replica,
            canonical_index=canonical_index,
            lease=lease_handle,
            state_dict=result.state_dict,
            registration_result=result,
        )

    def _perform_registration(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None,
        plan: PlanType,
        cancel_event: threading.Event | None = None,
        on_begin: Callable[[_RegisterHandle], None] | None = None,
        options_override: RegisterArtifactOptions | None = None,
        ttl_override: int | None = None,
        device_override: int | torch.device | None = None,
        view_registration: ViewRegistrationContext | None = None,
    ) -> RegisteredArtifact:
        method = "register" if plan is PlanType.VRAM_LEASED else "put"
        span_name = "Store/Register" if plan is PlanType.VRAM_LEASED else "Store/Put"
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.daemon": self._runtime.daemon_endpoint,
            "tc.store.session_id": self._runtime.session_id,
            "tc.store.plan": plan.value,
            "tc.store.key_present": bool(key),
            "tc.store.client_artifact_id": artifact_id or "",
        }
        policy = self._runtime.retry_policies.get(method)
        attempt = 1
        start_time = time.monotonic()

        with self._runtime.operation_span(span_name, attributes) as span:

            def record_outcome(status: str) -> None:
                duration = time.monotonic() - start_time
                store_metrics.observe_latency(
                    method, self._runtime.daemon_endpoint, status, duration
                )
                span.set_attribute("tc.store.status", status)
                if status not in {"OK", "CANCELLED"}:
                    store_metrics.increment_error(
                        method, self._runtime.daemon_endpoint, status
                    )

            while True:
                if cancel_event and cancel_event.is_set():
                    record_outcome("CANCELLED")
                    span.set_status(Status(StatusCode.ERROR, "CANCELLED"))
                    raise ArtifactError(
                        "Registration cancelled",
                        status_code="CANCELLED",
                        retryable=False,
                    )
                try:
                    result = self._attempt_registration(
                        tensors,
                        artifact_id=artifact_id,
                        key=key,
                        plan=plan,
                        cancel_event=cancel_event,
                        on_begin=on_begin,
                        options_override=options_override,
                        ttl_override=ttl_override,
                        device_override=device_override,
                        view_registration=view_registration,
                    )
                    record_outcome("OK")
                    span.set_attribute("tc.store.retry.count", attempt)
                    span.set_attribute("tc.replica.type", result.replica.replica_type)
                    span.set_attribute(
                        "tc.device.id", int(result.replica.device.index or 0)
                    )
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("tc.artifact.id", result.artifact_id)
                    return result
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
                        method, self._runtime.daemon_endpoint, error.status_code
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
                            "store.registration_retry",
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


__all__ = ["RegistrationPipeline"]
