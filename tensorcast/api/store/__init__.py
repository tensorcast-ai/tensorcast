#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Mapping, Sequence

import torch

from tensorcast._C import get_cuda_memory_handle
from tensorcast.api._config import GetArtifactOptions, PlanType, RegisterArtifactOptions
from tensorcast.api._materialize import MaterializedArtifact, materialize_artifact
from tensorcast.api._region_cache import (
    register_region as _cache_register_region,
)
from tensorcast.api._region_cache import (
    unregister_region as _cache_unregister_region,
)
from tensorcast.api._register import RegistrationResult, _register_artifact_core
from tensorcast.api._runtime import require_runtime
from tensorcast.api.store.async_ops import ArtifactFuture
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.registration import RegistrationPipeline
from tensorcast.api.store.runtime import (
    StoreRuntimeContext,
    shutdown_context,
)
from tensorcast.api.store.runtime import (
    get_context as get_runtime_context,
)
from tensorcast.api.store.types import (
    ArtifactError,
    ArtifactStatusCode,
    CanonicalIndex,
    CanonicalIndexEntry,
    FallbackOptions,
    LeaseHandle,
    ReplicaInfo,
    RetryPolicy,
    StoreCapabilities,
    StoreOptions,
    TensorDict,
)
from tensorcast.api.store.views import TransformPlacement, ViewOrchestrator
from tensorcast.daemon_ctl import get_daemon_client
from tensorcast.types import DeregisterArtifactOutcome, VramRegionHandle


class Store:
    """Store façade delegating to runtime, registration, and materialization pipelines."""

    def __init__(
        self,
        daemon_endpoint: str,
        *,
        opts: StoreOptions | None = None,
        runtime: StoreRuntimeContext | None = None,
        register_fn: Callable[..., RegistrationResult] | None = None,
        materialize_fn: Callable[..., MaterializedArtifact] | None = None,
    ) -> None:
        self._runtime = runtime or StoreRuntimeContext(
            daemon_endpoint, opts=opts, client_factory=get_daemon_client
        )
        self._views = ViewOrchestrator(self._runtime)
        self._registration = RegistrationPipeline(
            self._runtime,
            self._views,
            register_fn=register_fn or _register_artifact_core,
        )
        self._materialization = MaterializationPipeline(
            self._runtime,
            self._views,
            materialize_fn=materialize_fn or materialize_artifact,
        )

    def set_register_fn(self, register_fn: Callable[..., RegistrationResult]) -> None:
        self._registration.set_register_fn(register_fn)

    def set_materialize_fn(
        self, materialize_fn: Callable[..., MaterializedArtifact]
    ) -> None:
        self._materialization.set_materialize_fn(materialize_fn)

    # ------------------------------------------------------------------
    # Registration APIs
    # ------------------------------------------------------------------
    def register(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register(
            tensors,
            artifact_id=artifact_id,
            key=key,
            options=options,
            ttl_ms=ttl_ms,
        )

    def register_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.register_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            options=options,
            ttl_ms=ttl_ms,
        )

    def register_view(
        self,
        tensors: TensorDict,
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
    ) -> RegisteredArtifact:
        return self._registration.register_view(
            tensors,
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
            placement=placement,
            ttl_ms=ttl_ms,
            allow_partial=allow_partial,
            options=options,
            resolver=self._views.resolve_view_inputs,
        )

    def put(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        return self._registration.put(
            tensors,
            artifact_id=artifact_id,
            key=key,
            options=options,
            device=device,
        )

    def put_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.put_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            options=options,
            device=device,
        )

    # ------------------------------------------------------------------
    # Retrieval APIs
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
        return self._materialization.get(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            options=options,
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
    ) -> dict[str, torch.Tensor]:
        return self._materialization.get_view(
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
            placement=placement,
            device=device,
            options=options,
            resolver=self._views.resolve_view_inputs,
        )

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
    ) -> None:
        self._materialization.get_view_into(
            target,
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
            placement=placement,
            device=device,
            options=options,
            resolver=self._views.resolve_view_inputs,
        )

    def get_async(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
        options: GetArtifactOptions | None = None,
    ) -> ArtifactFuture[dict[str, torch.Tensor]]:
        return self._materialization.get_async(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            options=options,
        )

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
        self._materialization.get_into(
            target,
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            options=options,
        )

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
        return self._materialization.get_into_async(
            target,
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            options=options,
        )

    # ------------------------------------------------------------------
    # Region-backed registration
    # ------------------------------------------------------------------
    def register_vram_region(
        self,
        *,
        device_id: int,
        base_ptr: int,
        size_bytes: int,
        ttl_ms: int,
        name: str | None = None,
    ) -> VramRegionHandle:
        client = self._runtime.ensure_client()
        handle_bytes = get_cuda_memory_handle(int(device_id), int(base_ptr))
        handle = client.register_vram_region(
            device_id=int(device_id),
            size_bytes=int(size_bytes),
            ttl_ms=int(ttl_ms),
            cuda_ipc_handle=handle_bytes,
            region_name=name,
        )
        with contextlib.suppress(Exception):
            _cache_register_region(
                region_id=handle.region_id,
                device_id=int(device_id),
                base_ptr=int(base_ptr),
                size_bytes=int(size_bytes),
                ttl_ms=int(ttl_ms),
            )
        return handle

    def unregister_vram_region(
        self, region_id: str, *, force: bool | None = None
    ) -> bool:
        client = self._runtime.ensure_client()
        released = client.unregister_vram_region(region_id, force=force)
        if released:
            with contextlib.suppress(Exception):
                _cache_unregister_region(region_id)
        return released

    def deregister_artifact(
        self,
        artifact_id: str,
        *,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        extend_ttl_ms: int | None = None,
        device_id: int | None = None,
    ) -> DeregisterArtifactOutcome:
        client = self._runtime.ensure_client()
        drain_ms = int(drain_timeout_s * 1000) if drain_timeout_s is not None else None
        return client.deregister_artifact(
            artifact_id,
            wait_for_drain=bool(wait),
            drain_timeout_ms=drain_ms,
            extend_ttl_ms=extend_ttl_ms,
            device_id=device_id,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        return self._runtime.capabilities

    @property
    def daemon_endpoint(self) -> str:
        return self._runtime.daemon_endpoint

    @property
    def closed(self) -> bool:
        return self._runtime.closed

    def close(self) -> None:
        self._runtime.close()


_PROCESS_STORE_LOCK = threading.RLock()
_PROCESS_STORE: Store | None = None
_PROCESS_STORE_OPTS: StoreOptions | None = None


def _ensure_process_store(
    *,
    opts: StoreOptions | None = None,
    force_recreate: bool = False,
) -> Store:
    global _PROCESS_STORE, _PROCESS_STORE_OPTS

    context = get_runtime_context(
        opts=opts,
        force_recreate=force_recreate,
        client_factory=get_daemon_client,
        runtime_provider=require_runtime,
    )

    with _PROCESS_STORE_LOCK:
        current = _PROCESS_STORE
        current_closed = current.closed if current is not None else False
        if current is None or force_recreate or current_closed:
            prior = _PROCESS_STORE
            opts_marker: StoreOptions | None
            if opts is not None:
                opts_marker = opts
            elif current_closed and not force_recreate:
                opts_marker = _PROCESS_STORE_OPTS
            else:
                opts_marker = None
            effective_opts: StoreOptions = opts_marker or context.opts
            _PROCESS_STORE = Store(
                context.daemon_endpoint, opts=effective_opts, runtime=context
            )
            _PROCESS_STORE_OPTS = opts_marker
            if prior is not None:
                with contextlib.suppress(Exception):
                    prior.close()
        elif (
            opts is not None
            and _PROCESS_STORE_OPTS is not None
            and opts != _PROCESS_STORE_OPTS
        ):
            raise RuntimeError(
                "Store already initialized with different options. Pass force_recreate=True to replace."
            )
        result = _PROCESS_STORE
        if result is None:
            raise RuntimeError("Failed to initialize process Store")
        return result


def store(
    *,
    opts: StoreOptions | None = None,
    force_recreate: bool = False,
) -> Store:
    """Return the process-wide Store session, creating it lazily as needed."""

    return _ensure_process_store(opts=opts, force_recreate=force_recreate)


def shutdown_process_store() -> None:
    """Close and clear the process-wide Store if it exists."""

    global _PROCESS_STORE, _PROCESS_STORE_OPTS

    with _PROCESS_STORE_LOCK:
        current = _PROCESS_STORE
        _PROCESS_STORE = None
        _PROCESS_STORE_OPTS = None
    if current is not None:
        with contextlib.suppress(Exception):
            current.close()
    shutdown_context()


def _coerce_store() -> Store:
    current = _PROCESS_STORE
    if current is not None and not current.closed:
        return current
    return store()


def register(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register(
        tensors,
        artifact_id=artifact_id,
        key=key,
        options=options,
        ttl_ms=ttl_ms,
    )


def register_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().register_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        options=options,
        ttl_ms=ttl_ms,
    )


def register_kv_block(
    tensor: torch.Tensor,
    block_hash: str,
    *,
    ttl_ms: int | None = None,
) -> RegisteredArtifact:
    artifact_id = f"cgid:kv:{block_hash}"
    opts = RegisterArtifactOptions(
        plan=PlanType.VRAM_LEASED,
        lease_in_place=True,
    )
    return _coerce_store().register(
        {"kv": tensor}, artifact_id=artifact_id, options=opts, ttl_ms=ttl_ms
    )


def register_view(
    tensors: TensorDict,
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
) -> RegisteredArtifact:
    return _coerce_store().register_view(
        tensors,
        artifact_id=artifact_id,
        key=key,
        slices=slices,
        transpose=transpose,
        view_id=view_id,
        placement=placement,
        ttl_ms=ttl_ms,
        allow_partial=allow_partial,
        options=options,
    )


def register_vram_region(
    *,
    device_id: int,
    base_ptr: int,
    size_bytes: int,
    ttl_ms: int,
    name: str | None = None,
) -> VramRegionHandle:
    return _coerce_store().register_vram_region(
        device_id=device_id,
        base_ptr=base_ptr,
        size_bytes=size_bytes,
        ttl_ms=ttl_ms,
        name=name,
    )


def unregister_vram_region(region_id: str, *, force: bool | None = None) -> bool:
    return _coerce_store().unregister_vram_region(region_id, force=force)


def deregister_artifact(
    artifact_id: str,
    *,
    wait: bool = True,
    drain_timeout_s: float | None = None,
    extend_ttl_ms: int | None = None,
    device_id: int | None = None,
) -> DeregisterArtifactOutcome:
    return _coerce_store().deregister_artifact(
        artifact_id,
        wait=wait,
        drain_timeout_s=drain_timeout_s,
        extend_ttl_ms=extend_ttl_ms,
        device_id=device_id,
    )


def put(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> RegisteredArtifact:
    return _coerce_store().put(
        tensors,
        artifact_id=artifact_id,
        key=key,
        options=options,
        device=device,
    )


def put_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().put_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        options=options,
        device=device,
    )


def get(
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    device: torch.device | str | None = None,
    fallback: FallbackOptions | None = None,
    options: GetArtifactOptions | None = None,
) -> dict[str, torch.Tensor]:
    return _coerce_store().get(
        artifact_id=artifact_id,
        key=key,
        device=device,
        fallback=fallback,
        options=options,
    )


def get_view(
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    slices: Mapping[str, Sequence[object]] | None = None,
    transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    view_id: str | None = None,
    placement: str | None = None,
    device: torch.device | str | None = None,
    options: GetArtifactOptions | None = None,
) -> dict[str, torch.Tensor]:
    return _coerce_store().get_view(
        artifact_id=artifact_id,
        key=key,
        slices=slices,
        transpose=transpose,
        view_id=view_id,
        placement=placement,
        device=device,
        options=options,
    )


def get_view_into(
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
) -> None:
    _coerce_store().get_view_into(
        target,
        artifact_id=artifact_id,
        key=key,
        slices=slices,
        transpose=transpose,
        view_id=view_id,
        placement=placement,
        device=device,
        options=options,
    )


def get_async(
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    device: torch.device | str | None = None,
    fallback: FallbackOptions | None = None,
    options: GetArtifactOptions | None = None,
) -> ArtifactFuture[dict[str, torch.Tensor]]:
    return _coerce_store().get_async(
        artifact_id=artifact_id,
        key=key,
        device=device,
        fallback=fallback,
        options=options,
    )


def get_into(
    target: dict[str, torch.Tensor],
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    device: torch.device | str | None = None,
    fallback: FallbackOptions | None = None,
    options: GetArtifactOptions | None = None,
) -> None:
    _coerce_store().get_into(
        target,
        artifact_id=artifact_id,
        key=key,
        device=device,
        fallback=fallback,
        options=options,
    )


def get_into_async(
    target: dict[str, torch.Tensor],
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    device: torch.device | str | None = None,
    fallback: FallbackOptions | None = None,
    options: GetArtifactOptions | None = None,
) -> ArtifactFuture[None]:
    return _coerce_store().get_into_async(
        target,
        artifact_id=artifact_id,
        key=key,
        device=device,
        fallback=fallback,
        options=options,
    )


__all__ = [
    "ArtifactError",
    "ArtifactFuture",
    "ArtifactStatusCode",
    "CanonicalIndex",
    "CanonicalIndexEntry",
    "FallbackOptions",
    "LeaseHandle",
    "MaterializedArtifact",
    "RegisteredArtifact",
    "ReplicaInfo",
    "RetryPolicy",
    "StoreCapabilities",
    "Store",
    "StoreOptions",
    "TensorDict",
    "TransformPlacement",
    "store",
    "shutdown_process_store",
    "register",
    "register_async",
    "put",
    "put_async",
    "get",
    "get_async",
    "get_view",
    "get_view_into",
    "get_into",
    "get_into_async",
    "register_view",
    "register_kv_block",
    "register_vram_region",
    "unregister_vram_region",
    "deregister_artifact",
    "get_daemon_client",
    "require_runtime",
    "_register_artifact_core",
    "materialize_artifact",
]
