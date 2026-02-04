#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator, Mapping

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind

from tensorcast.api._register import RegisteredLease
from tensorcast.api._runtime import require_runtime
from tensorcast.api.store.cache import ArtifactCache, ArtifactCacheEntry
from tensorcast.api.store.retry import build_retry_policies
from tensorcast.api.store.types import (
    RetryPolicy,
    SpanAttributeValue,
    StoreCapabilities,
    StoreOptions,
)
from tensorcast.daemon_ctl import DaemonCtl, get_daemon_client
from tensorcast.observability.otel import set_span_attributes
from tensorcast.store_session_registry import StoreSessionRecord, write_record
from tensorcast.types import ServerConfig

logger = logging.getLogger(__name__)


class _ForkAwareHandle:
    """Track per-context resources that must be refreshed across fork."""

    def __init__(self, cleanup: Callable[[], None]) -> None:
        self._cleanup = cleanup

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._cleanup()


class _KeyCacheEntry:
    def __init__(
        self,
        *,
        artifact_id: str | None,
        expires_at: float,
    ) -> None:
        self.artifact_id = artifact_id
        self.expires_at = expires_at


class StoreRuntimeContext:
    """Process-wide runtime and lifecycle manager for Store operations."""

    _AT_FORK_REGISTRY: "weakref.WeakSet[StoreRuntimeContext]" = weakref.WeakSet()
    _DEFAULT_LEASE_TTL_MS = 600_000

    def __init__(
        self,
        daemon_endpoint: str,
        *,
        opts: StoreOptions | None = None,
        client_factory: Callable[[str], DaemonCtl] = get_daemon_client,
    ) -> None:
        self._daemon_endpoint = daemon_endpoint
        self._opts = opts or StoreOptions()
        self._client_factory = client_factory
        self._session_id = uuid.uuid4().hex
        self._tracer = trace.get_tracer(__name__)
        self._session_labels: dict[str, str] = {
            "daemon_endpoint": daemon_endpoint,
            "session_id": self._session_id,
        }
        self._daemon_id_lock = threading.RLock()
        self._daemon_id: str | None = None
        self._capabilities: StoreCapabilities | None = None
        self._retry_policies = build_retry_policies(self._opts.retry_overrides)
        self._client_lock = threading.RLock()
        self._client: DaemonCtl | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tensorcast-store"
        )
        self._executor_handle = _ForkAwareHandle(self._executor.shutdown)
        self._pending_futures: set[concurrent.futures.Future[object]] = set()
        self._leases_lock = threading.RLock()
        self._active_leases: set[RegisteredLease] = set()
        self._metadata_lock = threading.RLock()
        self._session_record: StoreSessionRecord | None = None
        self._key_cache_lock = threading.RLock()
        self._key_cache: dict[str, _KeyCacheEntry] = {}
        self._key_cache_ttl_seconds = self._configure_key_cache_ttl()
        self._artifact_cache_ttl_seconds = self._configure_index_cache_ttl()
        self._artifact_cache_max_entries = self._configure_index_cache_size()
        self._artifact_cache = ArtifactCache(
            daemon_endpoint=self._daemon_endpoint,
            ttl_seconds=self._artifact_cache_ttl_seconds,
            max_entries=self._artifact_cache_max_entries,
        )
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="tensorcast-store-loop",
        )
        self._loop_thread.start()
        self._closed = False

        self._init_session_record()
        self._install_at_fork()
        self._client = self._create_client()
        self._initialize_session_metadata(self._client)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _create_client(self) -> DaemonCtl:
        return self._client_factory(self._daemon_endpoint)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @staticmethod
    def _default_capabilities() -> StoreCapabilities:
        return StoreCapabilities(
            mem_pool_bytes=0,
            tx_slice_bytes=0,
            artifact_chunk_bytes=0,
            server_config=None,
        )

    def _capabilities_from_config(self, config: ServerConfig) -> StoreCapabilities:
        mem_pool = int(getattr(config, "mem_pool_size", 0))
        tx_slice = int(getattr(config, "tx_slice_bytes", 0))
        artifact_chunk = int(getattr(config, "artifact_chunk_bytes", 0))
        return StoreCapabilities(
            mem_pool_bytes=mem_pool,
            tx_slice_bytes=tx_slice,
            artifact_chunk_bytes=artifact_chunk,
            server_config=config if isinstance(config, ServerConfig) else None,
        )

    def _configure_key_cache_ttl(self) -> float:
        raw = os.getenv("TENSORCAST_STORE_KEY_CACHE_TTL_SECONDS")
        if not raw:
            return 30.0
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "store.key_cache_ttl.invalid",
                extra={"tc.store.daemon": self._daemon_endpoint, "ttl": raw},
            )
            return 30.0
        if value <= 0:
            return 0.0
        return value

    def _configure_index_cache_ttl(self) -> float:
        raw = os.getenv("TENSORCAST_STORE_INDEX_CACHE_TTL_SECONDS")
        if raw is None or raw == "":
            return 600.0
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "store.index_cache_ttl.invalid",
                extra={"tc.store.daemon": self._daemon_endpoint, "ttl": raw},
            )
            return 600.0
        if value <= 0:
            return 0.0
        return value

    def _configure_index_cache_size(self) -> int:
        raw = os.getenv("TENSORCAST_STORE_CACHE_MAX_ENTRIES")
        if raw is None or raw == "":
            return 1000
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "store.index_cache_size.invalid",
                extra={"tc.store.daemon": self._daemon_endpoint, "size": raw},
            )
            return 1000
        return max(0, value)

    def _active_lease_count(self) -> int:
        with self._leases_lock:
            return len(self._active_leases)

    @staticmethod
    def _capabilities_to_dict(capabilities: StoreCapabilities) -> dict[str, object]:
        data: dict[str, object] = {
            "mem_pool_bytes": int(capabilities.mem_pool_bytes),
            "tx_slice_bytes": int(capabilities.tx_slice_bytes),
            "artifact_chunk_bytes": int(capabilities.artifact_chunk_bytes),
        }
        if capabilities.server_config is not None:
            try:
                data["server_config"] = capabilities.server_config.model_dump()
            except Exception:  # noqa: BLE001
                data["server_config"] = None
        return data

    def _init_session_record(self) -> None:
        now = time.time()
        record = StoreSessionRecord(
            session_id=self._session_id,
            pid=os.getpid(),
            daemon_endpoint=self._daemon_endpoint,
            created_at=now,
            last_activity_at=now,
        )
        with self._metadata_lock:
            self._session_record = record
            self._persist_session_record_locked(record=record)

    def _persist_session_record_locked(
        self,
        *,
        record: StoreSessionRecord | None = None,
        activity: bool = False,
        closed: bool = False,
        capabilities: StoreCapabilities | None = None,
    ) -> None:
        target = record if record is not None else self._session_record
        if target is None:
            return
        if capabilities is not None:
            target.capabilities = self._capabilities_to_dict(capabilities)
        if activity:
            target.mark_activity()
        if closed:
            target.mark_closed()
        target.active_leases = self._active_lease_count()
        target.pending_futures = len(self._pending_futures)
        try:
            write_record(target)
        except Exception:  # noqa: BLE001
            logger.debug("store.session_metadata_write_failed", exc_info=True)

    def _update_session_record(
        self,
        *,
        activity: bool = False,
        closed: bool = False,
        capabilities: StoreCapabilities | None = None,
    ) -> None:
        with self._metadata_lock:
            self._persist_session_record_locked(
                activity=activity,
                closed=closed,
                capabilities=capabilities,
            )

    def cache_key_mapping(
        self,
        key: str,
        *,
        artifact_id: str | None,
        ttl_override: float | None = None,
    ) -> None:
        if not key:
            return
        ttl = self._key_cache_ttl_seconds if ttl_override is None else ttl_override
        if ttl <= 0:
            return
        expires_at = time.monotonic() + ttl
        with self._key_cache_lock:
            self._key_cache[key] = _KeyCacheEntry(
                artifact_id=artifact_id,
                expires_at=expires_at,
            )

    def resolve_key_mapping_cached(self, *, key: str) -> str | None:
        now = time.monotonic()
        with self._key_cache_lock:
            cached = self._key_cache.get(key)
            if cached and cached.expires_at > now:
                return cached.artifact_id
            if cached is not None:
                del self._key_cache[key]
        mapping = self.ensure_client().resolve_key_mapping(key)
        resolved_id = mapping.artifact_id or None
        ttl_override = float(mapping.cache_ttl_seconds)
        self.cache_key_mapping(
            key,
            artifact_id=resolved_id,
            ttl_override=ttl_override,
        )
        return resolved_id

    def get_artifact_index_cached(self, artifact_id: str) -> ArtifactCacheEntry | None:
        return self._artifact_cache.get_artifact_index_cached(artifact_id)

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        self._artifact_cache.cache_artifact_index(entry)

    def invalidate_artifact(
        self,
        artifact_id: str | None,
        *,
        key: str | None = None,
        reason: str | None = None,
    ) -> None:
        if artifact_id:
            self._artifact_cache.invalidate_artifact(artifact_id, reason=reason)
        with self._key_cache_lock:
            keys_to_remove = []
            for cached_key, cached_entry in self._key_cache.items():
                matches_artifact = bool(
                    artifact_id and cached_entry.artifact_id == artifact_id
                )
                matches_key = bool(key is not None and cached_key == key)
                if matches_artifact or matches_key:
                    keys_to_remove.append(cached_key)
            for cached_key in keys_to_remove:
                del self._key_cache[cached_key]

    def _record_session_start(self, capabilities: StoreCapabilities) -> None:
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.session_id": self._session_id,
            "tc.store.daemon": self._daemon_endpoint,
            "tc.store.mem_pool_bytes": int(capabilities.mem_pool_bytes),
            "tc.store.tx_slice_bytes": int(capabilities.tx_slice_bytes),
            "tc.store.artifact_chunk_bytes": int(capabilities.artifact_chunk_bytes),
        }
        with self._tracer.start_as_current_span("Store/Init", kind=SpanKind.INTERNAL):
            set_span_attributes(attributes)
        logger.info(
            "store.session_initialized",
            extra={
                "tc.store.session_id": self._session_id,
                "tc.store.daemon": self._daemon_endpoint,
                "tc.store.capabilities": {
                    "mem_pool_bytes": capabilities.mem_pool_bytes,
                    "tx_slice_bytes": capabilities.tx_slice_bytes,
                    "artifact_chunk_bytes": capabilities.artifact_chunk_bytes,
                },
            },
        )

    def _initialize_session_metadata(self, client: DaemonCtl) -> None:
        capabilities = self._default_capabilities()
        try:
            config = client.get_server_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "store.capabilities_fetch_failed",
                extra={
                    "tc.store.daemon": self._daemon_endpoint,
                    "tc.store.session_id": self._session_id,
                },
                exc_info=exc,
            )
        else:
            capabilities = self._capabilities_from_config(config)
        self._capabilities = capabilities
        self._update_session_record(capabilities=capabilities, activity=True)
        self._record_session_start(capabilities)
        self._session_labels.update(
            {
                "mem_pool_bytes": str(capabilities.mem_pool_bytes),
                "tx_slice_bytes": str(capabilities.tx_slice_bytes),
            }
        )

    def _install_at_fork(self) -> None:
        StoreRuntimeContext._AT_FORK_REGISTRY.add(self)

        os.register_at_fork(
            before=self._before_fork,
            after_in_parent=self._after_fork_parent,
            after_in_child=self._after_fork_child,
        )

    def _before_fork(self) -> None:
        if self._closed:
            return
        self._client_lock.acquire()

    def _after_fork_parent(self) -> None:
        if self._closed:
            return
        self._client_lock.release()

    def _after_fork_child(self) -> None:
        if self._closed:
            return
        # Child processes may not reuse the parent's gRPC channel; rebuild lazily.
        self._client_lock = threading.RLock()
        self._client = None
        # Executors are not safe to share across fork; rebuild lazily on access.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tensorcast-store"
        )
        self._executor_handle = _ForkAwareHandle(self._executor.shutdown)
        self._pending_futures = set()
        self._leases_lock = threading.RLock()
        self._active_leases = set()
        self._metadata_lock = threading.RLock()
        self._session_record = None
        self._session_id = uuid.uuid4().hex
        self._session_labels = {
            "daemon_endpoint": self._daemon_endpoint,
            "session_id": self._session_id,
        }
        self._daemon_id_lock = threading.RLock()
        self._daemon_id = None
        self._capabilities = None
        self._retry_policies = build_retry_policies(self._opts.retry_overrides)
        self._key_cache_lock = threading.RLock()
        self._key_cache = {}
        self._key_cache_ttl_seconds = self._configure_key_cache_ttl()
        self._artifact_cache_ttl_seconds = self._configure_index_cache_ttl()
        self._artifact_cache_max_entries = self._configure_index_cache_size()
        self._artifact_cache = ArtifactCache(
            daemon_endpoint=self._daemon_endpoint,
            ttl_seconds=self._artifact_cache_ttl_seconds,
            max_entries=self._artifact_cache_max_entries,
        )
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="tensorcast-store-loop",
        )
        self._loop_thread.start()
        self._init_session_record()

    def ensure_client(self) -> DaemonCtl:
        if self._closed:
            raise RuntimeError("Store runtime is closed")
        with self._client_lock:
            if self._client is None:
                self._client = self._create_client()
                self._initialize_session_metadata(self._client)
            elif self._capabilities is None:
                self._initialize_session_metadata(self._client)
            return self._client

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._client_lock:
            if self._client is not None:
                with contextlib.suppress(Exception):
                    self._client.close()
                self._client = None
        self.release_all_leases()
        self._executor_handle.close()
        with contextlib.suppress(Exception):
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread") and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            self._loop.close()
        with self._key_cache_lock:
            self._key_cache.clear()
        self._artifact_cache.clear()
        self._update_session_record(activity=True, closed=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        with self._metadata_lock:
            self._pending_futures.add(future)
            self._persist_session_record_locked(activity=False)

        def _cleanup(_future: concurrent.futures.Future[object]) -> None:
            with self._metadata_lock:
                self._pending_futures.discard(_future)
                self._persist_session_record_locked(activity=False)

        future.add_done_callback(_cleanup)

    def track_lease(self, lease: RegisteredLease | None) -> None:
        if lease is None:
            return
        lease.__enter__()
        with self._leases_lock:
            self._active_leases.add(lease)
        self._update_session_record(activity=False)

    def release_all_leases(self) -> None:
        with self._leases_lock:
            leases = list(self._active_leases)
            self._active_leases.clear()
        for lease in leases:
            with contextlib.suppress(Exception):
                lease.__exit__(None, None, None)
        self._update_session_record(activity=False)

    @contextmanager
    def operation_span(
        self,
        name: str,
        attributes: Mapping[str, SpanAttributeValue] | None = None,
    ) -> Iterator[Span]:
        self._update_session_record(activity=True)
        with self._tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(str(key), value)
            yield span

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        if self._capabilities is None:
            return self._default_capabilities()
        return self._capabilities

    @property
    def daemon_endpoint(self) -> str:
        return self._daemon_endpoint

    @property
    def daemon_id(self) -> str | None:
        if self._closed:
            return None
        with self._daemon_id_lock:
            cached = self._daemon_id
        if cached is not None:
            return cached or None
        try:
            status = self.ensure_client().get_worker_status()
        except Exception:
            return None
        daemon_id = getattr(status, "daemon_id", "") or ""
        with self._daemon_id_lock:
            self._daemon_id = daemon_id
        return daemon_id or None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_labels(self) -> Mapping[str, str]:
        return self._session_labels

    @property
    def retry_policies(self) -> Mapping[str, RetryPolicy]:
        return self._retry_policies

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self._executor

    @property
    def opts(self) -> StoreOptions:
        return self._opts

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def closed(self) -> bool:
        return self._closed


_GLOBAL_CONTEXT_LOCK = threading.RLock()
_GLOBAL_CONTEXT: StoreRuntimeContext | None = None
_GLOBAL_CONTEXT_ADDRESS: str | None = None
_GLOBAL_CONTEXT_OPTIONS: StoreOptions | None = None


def get_context(
    *,
    daemon_endpoint: str | None = None,
    opts: StoreOptions | None = None,
    force_recreate: bool = False,
    client_factory: Callable[[str], DaemonCtl] = get_daemon_client,
    runtime_provider: Callable[[], object] = require_runtime,
) -> StoreRuntimeContext:
    """Return the process-wide runtime context, creating or refreshing as needed."""

    try:
        runtime = runtime_provider()
    except RuntimeError:
        from tensorcast import startup  # Local import to avoid cycles

        if not startup.is_initialized():
            if daemon_endpoint:
                startup.init(mode="connect", address=daemon_endpoint)
            else:
                try:
                    startup.init(mode="connect")
                except RuntimeError:
                    startup.init(mode="create")
            runtime = runtime_provider()
        else:
            raise
    address = daemon_endpoint or runtime.address
    prior: StoreRuntimeContext | None = None

    with _GLOBAL_CONTEXT_LOCK:
        global _GLOBAL_CONTEXT, _GLOBAL_CONTEXT_ADDRESS, _GLOBAL_CONTEXT_OPTIONS

        current = _GLOBAL_CONTEXT
        current_closed = current.closed if current else False
        same_address = (_GLOBAL_CONTEXT_ADDRESS == address) if current else False

        if current is not None and not force_recreate:
            if current_closed or not same_address:
                prior = current
                _GLOBAL_CONTEXT = None
                _GLOBAL_CONTEXT_ADDRESS = None
                _GLOBAL_CONTEXT_OPTIONS = None
            else:
                if (
                    opts is not None
                    and _GLOBAL_CONTEXT_OPTIONS is not None
                    and opts != _GLOBAL_CONTEXT_OPTIONS
                ):
                    raise RuntimeError(
                        "Store runtime already initialized with different options. "
                        "Pass force_recreate=True to replace the process runtime."
                    )
                return current
        elif current is not None and force_recreate:
            prior = current
            _GLOBAL_CONTEXT = None
            _GLOBAL_CONTEXT_ADDRESS = None
            _GLOBAL_CONTEXT_OPTIONS = None

        effective_opts = opts or _GLOBAL_CONTEXT_OPTIONS or StoreOptions()
        new_context = StoreRuntimeContext(
            address, opts=effective_opts, client_factory=client_factory
        )
        _GLOBAL_CONTEXT = new_context
        _GLOBAL_CONTEXT_ADDRESS = address
        _GLOBAL_CONTEXT_OPTIONS = effective_opts

    if prior is not None:
        with contextlib.suppress(Exception):
            prior.close()

    return new_context


def shutdown_context() -> None:
    """Close and clear the process-wide runtime context."""

    with _GLOBAL_CONTEXT_LOCK:
        global _GLOBAL_CONTEXT, _GLOBAL_CONTEXT_ADDRESS, _GLOBAL_CONTEXT_OPTIONS
        current = _GLOBAL_CONTEXT
        _GLOBAL_CONTEXT = None
        _GLOBAL_CONTEXT_ADDRESS = None
        _GLOBAL_CONTEXT_OPTIONS = None

    if current is not None:
        with contextlib.suppress(Exception):
            current.close()


__all__ = ["StoreRuntimeContext", "get_context", "shutdown_context"]
