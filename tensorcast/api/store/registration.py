#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
import time
from collections.abc import Mapping
from typing import Callable, Sequence

import grpc
import torch
from opentelemetry.trace import Status, StatusCode

from tensorcast.api import _metrics as store_metrics
from tensorcast.api._config import (
    PlanType,
    RegisterArtifactOptions,
    StorePolicy,
    policy_requires_persistence,
)
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
from tensorcast.api._view_ops import ViewSpecBuildResult
from tensorcast.api.store.async_ops import ArtifactFuture, TrackedExecutor
from tensorcast.api.store.cache import ArtifactCacheEntry
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
    raise_mapped_registration_error,
    remaining_budget,
    should_retry,
)
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.types import ArtifactError, CanonicalIndex, SpanAttributeValue
from tensorcast.api.store.views import (
    ResolvedViewInputs,
    TransformPlacement,
    ViewOrchestrator,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2

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

    def set_register_fn(self, register_fn: Callable[..., RegistrationResult]) -> None:
        self._register_fn = register_fn

    def register(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> RegisteredArtifact:
        return self._perform_registration(
            tensors,
            artifact_id=artifact_id,
            key=key,
            plan=PlanType.VRAM_LEASED,
            policy_override=policy,
            options_override=options,
            ttl_override=ttl_ms,
        )

    def register_async(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
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
                    policy_override=policy,
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
        canonical_index_bytes: bytes | None = None,
        registration_kind: str | int | None = None,
        resolver: Callable[..., ResolvedViewInputs] | None = None,
    ) -> RegisteredArtifact:
        resolved_kind = self._normalize_registration_kind(
            registration_kind, allow_partial=allow_partial
        )
        if (
            resolved_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
            and not artifact_id
            and not key
        ):
            raise ArtifactError(
                "Piece registration requires an assembly_id (artifact_id) or key",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        view_resolver = resolver or self._views.resolve_view_inputs
        resolved_artifact_id: str | None = artifact_id
        if canonical_index_bytes is not None:
            if artifact_id is None:
                raise ArtifactError(
                    "canonical_index_bytes requires artifact_id",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if key is not None:
                raise ArtifactError(
                    "Specify either artifact_id or key, not both",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if view_id is not None:
                raise ArtifactError(
                    "View registration via pre-existing view_id is not supported yet",
                    status_code="UNIMPLEMENTED",
                    retryable=False,
                )
            if not slices and not transpose:
                raise ArtifactError(
                    "View registration requires slices/transpose or view_id",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            build_result = self._views._build_view_spec(
                canonical_index=canonical_index,
                slices=slices,
                transpose=transpose,
            )
        else:
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
            canonical_index_bytes = resolved.canonical_index_bytes
            assert canonical_index_bytes is not None
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            assert resolved.build_result is not None
            build_result = resolved.build_result
            resolved_artifact_id = resolved.artifact_id

        placement_enum = self._views.resolve_transform_placement(
            placement,
            has_transpose=build_result.has_transpose,
            for_registration=True,
        )
        view_ctx, upload_tensors = self._build_view_registration(
            canonical_index_bytes=canonical_index_bytes,
            canonical_index=canonical_index,
            build_result=build_result,
            tensors=tensors,
            placement_enum=placement_enum,
            registration_kind=resolved_kind,
        )
        return self._perform_registration(
            upload_tensors,
            artifact_id=resolved_artifact_id
            if resolved_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
            else None,
            key=key,
            plan=PlanType.VRAM_COALESCED,
            options_override=options,
            ttl_override=ttl_ms,
            view_registration=view_ctx,
        )

    def register_piece(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        assembly_id: str,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        canonical_index_bytes: bytes | None = None,
        placement: str | None = None,
        ttl_ms: int | None = None,
        options: RegisterArtifactOptions | None = None,
        resolver: Callable[..., ResolvedViewInputs] | None = None,
    ) -> RegisteredArtifact:
        if not assembly_id:
            raise ArtifactError(
                "assembly_id is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not slices:
            raise ArtifactError(
                "Piece registration requires slices",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if canonical_index_bytes is not None:
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            build_result = self._views._build_view_spec(
                canonical_index=canonical_index,
                slices=slices,
                transpose=None,
            )
        else:
            view_resolver = resolver or self._views.resolve_view_inputs
            resolved = view_resolver(
                artifact_id=assembly_id,
                key=None,
                slices=slices,
                transpose=None,
                view_id=None,
            )
            if resolved.variant == "id":
                raise ArtifactError(
                    "Piece registration via pre-existing view_id is not supported yet",
                    status_code="UNIMPLEMENTED",
                    retryable=False,
                )
            canonical_index_bytes = resolved.canonical_index_bytes
            assert canonical_index_bytes is not None
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            assert resolved.build_result is not None
            build_result = resolved.build_result

        placement_enum = self._views.resolve_transform_placement(
            placement,
            has_transpose=False,
            for_registration=True,
        )
        view_ctx, upload_tensors = self._build_view_registration(
            canonical_index_bytes=canonical_index_bytes,
            canonical_index=canonical_index,
            build_result=build_result,
            tensors=tensors,
            placement_enum=placement_enum,
            registration_kind=store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE,
        )
        return self._perform_registration(
            upload_tensors,
            artifact_id=assembly_id,
            key=key,
            plan=PlanType.VRAM_COALESCED,
            options_override=options,
            ttl_override=ttl_ms,
            view_registration=view_ctx,
        )

    @staticmethod
    def _normalize_registration_kind(
        registration_kind: str | int | None,
        *,
        allow_partial: bool,
    ) -> store_daemon_pb2.ViewRegistrationKind:
        resolved: store_daemon_pb2.ViewRegistrationKind | None = None
        if registration_kind is None:
            resolved = (
                store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
                if allow_partial
                else store_daemon_pb2.VIEW_REGISTRATION_KIND_CANONICAL
            )
        elif isinstance(registration_kind, str):
            normalized = registration_kind.strip().lower()
            if normalized in {"piece", "partial"}:
                resolved = store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
            elif normalized in {"canonical", "canon"}:
                resolved = store_daemon_pb2.VIEW_REGISTRATION_KIND_CANONICAL
        elif isinstance(registration_kind, int):
            if registration_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_CANONICAL:
                resolved = store_daemon_pb2.VIEW_REGISTRATION_KIND_CANONICAL
            elif registration_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE:
                resolved = store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
        if resolved is None:
            raise ArtifactError(
                "registration_kind must be 'canonical' or 'piece'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if allow_partial:
            if resolved != store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE:
                raise ArtifactError(
                    "allow_partial is deprecated; use registration_kind='piece'",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            logger.warning("allow_partial is deprecated; use registration_kind='piece'")
        return resolved

    def _build_view_registration(
        self,
        *,
        canonical_index_bytes: bytes,
        canonical_index: CanonicalIndex,
        build_result: ViewSpecBuildResult,
        tensors: Mapping[str, torch.Tensor],
        placement_enum: TransformPlacement,
        registration_kind: store_daemon_pb2.ViewRegistrationKind,
    ) -> tuple[ViewRegistrationContext, dict[str, torch.Tensor]]:
        if build_result.is_identity:
            raise ArtifactError(
                "View registration requires explicit view operations",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if registration_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE:
            if build_result.has_transpose:
                raise ArtifactError(
                    "Piece registration does not allow transpose",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if placement_enum != TransformPlacement.TRANSFORM_PLACEMENT_SERVER:
                raise ArtifactError(
                    "Piece registration requires placement='SERVER'",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
        plan_metadata = _compute_view_plan_metadata(canonical_index_bytes, build_result)
        canonical_ranges = _merge_canonical_ranges(plan_metadata.write_chunks)
        if (
            registration_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
            and not canonical_ranges
        ):
            raise ArtifactError(
                "Piece registration produced empty canonical coverage",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        view_options = store_daemon_pb2.ViewRegistrationOptions()
        if build_result.proto is None:
            raise ArtifactError(
                "View spec missing for registration",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        view_options.spec.CopyFrom(build_result.proto)
        view_options.placement = placement_enum
        view_options.canonical_size_bytes = canonical_index.total_size_bytes
        view_options.registration_kind = registration_kind
        for rng in canonical_ranges:
            range_proto = view_options.ranges.add()
            range_proto.offset = rng.offset
            range_proto.length = rng.length
        upload_tensors: dict[str, torch.Tensor]
        if placement_enum == TransformPlacement.TRANSFORM_PLACEMENT_SERVER:
            upload_tensors = dict(tensors)
        elif placement_enum == TransformPlacement.TRANSFORM_PLACEMENT_CLIENT:
            upload_tensors = _materialize_canonical_tensors(
                canonical_index_bytes, build_result, tensors
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
            registration_kind=registration_kind,
        )
        return view_ctx, upload_tensors

    @staticmethod
    def _require_cuda_tensors(tensors: Mapping[str, torch.Tensor]) -> None:
        if any(not tensor.is_cuda for tensor in tensors.values()):
            raise ArtifactError(
                "put requires CUDA tensors; CPU tensors are not supported yet",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

    def put(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        self._require_cuda_tensors(tensors)
        return self._perform_registration(
            tensors,
            artifact_id=artifact_id,
            key=key,
            plan=PlanType.DRAM_STABLE,
            policy_override=policy,
            options_override=options,
            device_override=device,
            cache_on_success=True,
            force_plan=False,
        )

    def put_async(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        self._require_cuda_tensors(tensors)
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
                    plan=PlanType.DRAM_STABLE,
                    policy_override=policy,
                    cancel_event=cancel_event,
                    on_begin=_on_begin,
                    options_override=options,
                    device_override=device,
                    cache_on_success=True,
                    force_plan=False,
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
        policy_override: StorePolicy | str | None = None,
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
        options: RegisterArtifactOptions
        if options_override is not None:
            options = options_override
            if options.plan is not plan:
                options = options.model_copy(update={"plan": plan})
            if plan is PlanType.VRAM_LEASED and not options.lease_in_place:
                options = options.model_copy(update={"lease_in_place": True})
            if resolved_key is None:
                resolved_key = options.key
            elif options.key != resolved_key:
                options = options.model_copy(update={"key": resolved_key})
        else:
            options = RegisterArtifactOptions(
                plan=plan,
                lease_in_place=plan is PlanType.VRAM_LEASED,
                key=resolved_key,
            )
        if policy_override is not None:
            try:
                normalized_override = StorePolicy.parse(policy_override)
            except Exception as exc:  # noqa: BLE001
                raise ArtifactError(
                    f"Invalid policy: {exc}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                ) from exc
            if normalized_override is not None:
                normalized_options = StorePolicy.parse(options.policy)
                if (
                    normalized_options is not None
                    and normalized_options.expanded() != normalized_override.expanded()
                ):
                    raise ArtifactError(
                        "Conflicting policy arguments: "
                        "`policy` does not match `options.policy` after normalization",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
                options = options.model_copy(update={"policy": normalized_override})
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
            raise_mapped_registration_error(exc)
        task_id = self._maybe_start_persistence(options, registration_result)
        return self._registration_to_artifact(
            registration_result, persistence_task_id=task_id
        )

    def _precheck_key_mapping(
        self,
        *,
        key: str,
        artifact_id: str | None,
    ) -> None:
        try:
            mapped_id, _ = self._runtime.ensure_client().resolve_key_mapping(key)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                return
            raise_mapped_registration_error(exc)
        except Exception as exc:  # noqa: BLE001
            raise_mapped_registration_error(exc)
        if not mapped_id:
            return
        if artifact_id and mapped_id == artifact_id:
            return
        raise ArtifactError(
            f"Failed to publish key '{key}': key already mapped to {mapped_id}",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    def _should_prevalidate_disk(self, options: RegisterArtifactOptions) -> bool:
        disk_path = options.disk_path
        if disk_path is None:
            return False
        if not isinstance(disk_path, str):
            return False
        return disk_path.strip() != ""

    def _maybe_start_persistence(
        self, options: RegisterArtifactOptions, result: RegistrationResult
    ) -> str | None:
        policy = StorePolicy.parse(options.policy)
        if not policy_requires_persistence(policy):
            return None
        artifact_id = result.descriptor.artifact_id
        if not artifact_id:
            raise ArtifactError(
                "Persistence requested but artifact_id is missing",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        try:
            response = self._runtime.ensure_client().start_persistence(
                artifact_id=artifact_id,
                policy=policy,
            )
        except ArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "store.persistence_start.failed",
                extra={
                    "tc.artifact.id": artifact_id,
                    "tc.persist.policy": (
                        policy.profile.value if policy and policy.profile else "custom"
                    ),
                    "tc.persist.error": str(exc),
                },
            )
            return None
        task_id = getattr(response, "task_id", None)
        if isinstance(task_id, str) and task_id.strip():
            return task_id
        return None

    def _registration_to_artifact(
        self, result: RegistrationResult, *, persistence_task_id: str | None = None
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
            persistence_task_id=persistence_task_id,
            local_stable_tier=result.local_stable_tier,
        )

    @staticmethod
    def _compute_generation(canonical_index_bytes: bytes) -> int | None:
        if not canonical_index_bytes:
            return None
        digest = hashlib.sha256(canonical_index_bytes).digest()
        return int.from_bytes(digest[:8], "big")

    def _cache_local_metadata(
        self,
        result: RegisteredArtifact,
        *,
        key: str | None,
        options: RegisterArtifactOptions | None,
    ) -> None:
        artifact_id = result.artifact_id
        if not artifact_id:
            return
        resolved_key = key or (options.key if options is not None else None)
        disk_path = None
        if options is not None and options.disk_path:
            disk_path = options.disk_path
        if resolved_key:
            self._runtime.cache_key_mapping(
                resolved_key, artifact_id=artifact_id, disk_path=disk_path
            )
        registration = result.registration_result
        if registration is None or not registration.index_bytes:
            return
        entry = ArtifactCacheEntry(
            artifact_id=artifact_id,
            canonical_index_bytes=registration.index_bytes,
            parsed_index=result.canonical_index,
            generation=self._compute_generation(registration.index_bytes),
            disk_path=disk_path,
            expires_at=time.monotonic(),
        )
        self._runtime.cache_artifact_index(entry)

    def _perform_registration(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None,
        plan: PlanType,
        policy_override: StorePolicy | str | None = None,
        cancel_event: threading.Event | None = None,
        on_begin: Callable[[_RegisterHandle], None] | None = None,
        options_override: RegisterArtifactOptions | None = None,
        ttl_override: int | None = None,
        device_override: int | torch.device | None = None,
        view_registration: ViewRegistrationContext | None = None,
        cache_on_success: bool = False,
        force_plan: bool = True,
    ) -> RegisteredArtifact:
        resolved_plan = plan
        resolved_options = options_override
        resolved_key = key
        if options_override is not None:
            if force_plan:
                if options_override.plan is not plan:
                    resolved_options = options_override.model_copy(
                        update={"plan": plan}
                    )
            else:
                resolved_plan = options_override.plan
            if resolved_key is None:
                resolved_key = options_override.key
        normalized_artifact_id = artifact_id.strip() if artifact_id else None
        if normalized_artifact_id == "":
            normalized_artifact_id = None
        method = "register" if resolved_plan is PlanType.VRAM_LEASED else "put"
        span_name = (
            "Store/Register" if resolved_plan is PlanType.VRAM_LEASED else "Store/Put"
        )
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.daemon": self._runtime.daemon_endpoint,
            "tc.store.session_id": self._runtime.session_id,
            "tc.store.plan": resolved_plan.value,
            "tc.store.key_present": bool(resolved_key),
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

            if resolved_key:
                try:
                    self._precheck_key_mapping(
                        key=resolved_key,
                        artifact_id=normalized_artifact_id,
                    )
                except ArtifactError as error:
                    span.record_exception(error)
                    record_outcome(error.status_code)
                    span.set_status(Status(StatusCode.ERROR, error.status_code))
                    raise

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
                        key=resolved_key,
                        plan=resolved_plan,
                        policy_override=policy_override,
                        cancel_event=cancel_event,
                        on_begin=on_begin,
                        options_override=resolved_options,
                        ttl_override=ttl_override,
                        device_override=device_override,
                        view_registration=view_registration,
                    )
                    self._runtime.invalidate_artifact(
                        result.artifact_id, key=resolved_key, reason="registration"
                    )
                    if cache_on_success:
                        self._cache_local_metadata(
                            result, key=resolved_key, options=resolved_options
                        )
                    record_outcome("OK")
                    span.set_attribute("tc.store.retry.count", attempt)
                    span.set_attribute("tc.replica.type", result.replica.replica_type)
                    device_index = result.replica.device.index
                    span.set_attribute(
                        "tc.device.id",
                        int(device_index) if device_index is not None else -1,
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
