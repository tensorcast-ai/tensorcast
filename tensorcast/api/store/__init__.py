#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
import os
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence

import torch

from tensorcast._c_ext import get_cuda_memory_handle
from tensorcast.api._config import RegisterArtifactOptions, StorePolicy
from tensorcast.api._materialize import (
    MaterializationPayload,
    materialize_artifact_v2,
)
from tensorcast.api._region_cache import (
    register_region as _cache_register_region,
)
from tensorcast.api._region_cache import (
    unregister_region as _cache_unregister_region,
)
from tensorcast.api._register import RegistrationResult, _register_artifact_core
from tensorcast.api._runtime import require_runtime
from tensorcast.api.store.artifact import Artifact, ArtifactDescriptor, TensorMeta
from tensorcast.api.store.async_ops import ArtifactFuture
from tensorcast.api.store.batch_context import (
    BatchContext,
    MaterializationBatcher,
    PrefetchTicket,
)
from tensorcast.api.store.deferred_loader import DeferredCommitResult, DeferredLoader
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
    PersistenceShardStatus,
    PersistenceStatusResult,
    ReplicaInfo,
    RetryPolicy,
    StoreCapabilities,
    StoreOptions,
    TensorDict,
)
from tensorcast.api.store.views import TransformPlacement, ViewOrchestrator
from tensorcast.daemon_ctl import get_daemon_client
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import DeregisterArtifactOutcome, VramRegionHandle


class Store:
    """Store façade delegating to runtime, registration, and materialization pipelines."""

    _PERSISTENCE_STATE_FROM_PROTO = {
        store_daemon_pb2.PERSISTENCE_STATE_PENDING: "pending",
        store_daemon_pb2.PERSISTENCE_STATE_RUNNING: "running",
        store_daemon_pb2.PERSISTENCE_STATE_DEGRADED: "degraded",
        store_daemon_pb2.PERSISTENCE_STATE_SUCCESS: "success",
        store_daemon_pb2.PERSISTENCE_STATE_FAILED: "failed",
    }

    def __init__(
        self,
        daemon_endpoint: str,
        *,
        opts: StoreOptions | None = None,
        runtime: StoreRuntimeContext | None = None,
        register_fn: Callable[..., RegistrationResult] | None = None,
        materialize_fn: Callable[..., MaterializationPayload] | None = None,
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
            materialize_fn=materialize_fn or materialize_artifact_v2,
        )
        self._enable_batcher = os.getenv(
            "TENSORCAST_STORE_ENABLE_BATCHER", "1"
        ).lower() not in ("0", "false", "no")
        self._enable_prefetch = os.getenv(
            "TENSORCAST_STORE_ENABLE_PREFETCH", "1"
        ).lower() not in ("0", "false", "no")
        self._batcher: MaterializationBatcher | None = (
            MaterializationBatcher(
                self._runtime,
                self._materialization,
            )
            if self._enable_batcher
            else None
        )

    def set_register_fn(self, register_fn: Callable[..., RegistrationResult]) -> None:
        self._registration.set_register_fn(register_fn)

    def set_materialize_fn(
        self, materialize_fn: Callable[..., MaterializationPayload]
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
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            ttl_ms=ttl_ms,
        )

    def register_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.register_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
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
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        return self._registration.put(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            device=device,
        )

    def put_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.put_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            device=device,
        )

    def query_persistence_status(
        self, *, task_id: str | None = None, artifact_id: str | None = None
    ) -> PersistenceStatusResult:
        """Query persistence task state via the local daemon."""
        if not task_id and not artifact_id:
            raise ArtifactError(
                "task_id or artifact_id must be provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resp = self._runtime.ensure_client().query_persistence_status(
            task_id=task_id, artifact_id=artifact_id
        )
        return self._persistence_status_from_proto(resp)

    def _persistence_status_from_proto(
        self, resp: store_daemon_pb2.QueryPersistenceStatusResponse
    ) -> PersistenceStatusResult:
        shards: list[PersistenceShardStatus] = []
        for shard in resp.shards:
            state = self._PERSISTENCE_STATE_FROM_PROTO.get(shard.state, "unknown")
            shards.append(
                PersistenceShardStatus(
                    shard_id=shard.shard_id,
                    shard_idx=int(shard.shard_idx),
                    state=state,
                    progress=float(shard.progress),
                    degraded_reason=shard.degraded_reason or None,
                    last_error=shard.last_error or None,
                    target_nodes=tuple(shard.target_nodes),
                    lease_ids=tuple(shard.lease_ids),
                )
            )
        state = self._PERSISTENCE_STATE_FROM_PROTO.get(resp.state, "unknown")
        return PersistenceStatusResult(
            task_id=resp.task_id,
            artifact_id=resp.artifact_id,
            plan_id=resp.plan_id,
            state=state,
            progress=float(resp.progress),
            degraded_reason=resp.degraded_reason or None,
            last_error=resp.last_error or None,
            shards=tuple(shards),
        )

    # ------------------------------------------------------------------
    # Retrieval APIs
    # ------------------------------------------------------------------
    def artifact(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        disk_path: str | None = None,
        fallback: FallbackOptions | str | None = None,
    ) -> Artifact:
        effective_fallback = FallbackOptions.parse(fallback)
        if disk_path and effective_fallback is None:
            effective_fallback = FallbackOptions.for_disk(str(disk_path))
        return Artifact(
            store_ref=weakref.ref(self),
            artifact_id=artifact_id,
            key=key,
            disk_path=disk_path,
            fallback=effective_fallback,
        )

    async def artifact_async(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        disk_path: str | None = None,
        fallback: FallbackOptions | str | None = None,
    ) -> Artifact:
        return self.artifact(
            artifact_id=artifact_id,
            key=key,
            disk_path=disk_path,
            fallback=fallback,
        )

    def from_disk(
        self, path: str, *, fallback: FallbackOptions | str | None = None
    ) -> Artifact:
        disk_path = os.fspath(path)
        effective_fallback = FallbackOptions.parse(fallback)
        if effective_fallback is None:
            effective_fallback = FallbackOptions.for_disk(str(disk_path))
        return self.artifact(disk_path=str(disk_path), fallback=effective_fallback)

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
        outcome = client.deregister_artifact(
            artifact_id,
            wait_for_drain=bool(wait),
            drain_timeout_ms=drain_ms,
            extend_ttl_ms=extend_ttl_ms,
            device_id=device_id,
        )
        self._runtime.invalidate_artifact(artifact_id, key=None, reason="deregister")
        return outcome

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

    @property
    def batcher(self) -> MaterializationBatcher:
        if self._batcher is None:
            raise RuntimeError("Materialization batcher is disabled")
        return self._batcher

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._batcher is not None:
                self._batcher.close()
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
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        ttl_ms=ttl_ms,
    )


def register_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().register_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        ttl_ms=ttl_ms,
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
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> RegisteredArtifact:
    return _coerce_store().put(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        device=device,
    )


def put_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().put_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        device=device,
    )


def query_persistence_status(
    *, task_id: str | None = None, artifact_id: str | None = None
) -> PersistenceStatusResult:
    return _coerce_store().query_persistence_status(
        task_id=task_id, artifact_id=artifact_id
    )


def artifact(
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    disk_path: str | None = None,
    fallback: FallbackOptions | str | None = None,
) -> Artifact:
    return _coerce_store().artifact(
        artifact_id=artifact_id,
        key=key,
        disk_path=disk_path,
        fallback=fallback,
    )


async def artifact_async(
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    disk_path: str | None = None,
    fallback: FallbackOptions | str | None = None,
) -> Artifact:
    return await _coerce_store().artifact_async(
        artifact_id=artifact_id,
        key=key,
        disk_path=disk_path,
        fallback=fallback,
    )


def from_disk(path: str, *, fallback: FallbackOptions | str | None = None) -> Artifact:
    return _coerce_store().from_disk(path, fallback=fallback)


__all__ = [
    "Artifact",
    "ArtifactDescriptor",
    "ArtifactError",
    "ArtifactFuture",
    "ArtifactStatusCode",
    "CanonicalIndex",
    "CanonicalIndexEntry",
    "DeferredCommitResult",
    "DeferredLoader",
    "FallbackOptions",
    "LeaseHandle",
    "MaterializationPayload",
    "MaterializationBatcher",
    "PrefetchTicket",
    "RegisteredArtifact",
    "ReplicaInfo",
    "RetryPolicy",
    "StoreCapabilities",
    "Store",
    "StoreOptions",
    "TensorMeta",
    "TensorDict",
    "TransformPlacement",
    "PersistenceStatusResult",
    "PersistenceShardStatus",
    "artifact",
    "artifact_async",
    "from_disk",
    "store",
    "shutdown_process_store",
    "BatchContext",
    "register",
    "register_async",
    "put",
    "put_async",
    "query_persistence_status",
    "register_view",
    "register_vram_region",
    "unregister_vram_region",
    "deregister_artifact",
    "get_daemon_client",
    "require_runtime",
    "_register_artifact_core",
    "materialize_artifact_v2",
]
