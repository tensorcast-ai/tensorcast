#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

# NOTE: The Store session API is under active development.  This module
# scaffolds the type surface and lifecycle primitives described in
# design 0014 so registration/loading flows can be incrementally
# migrated without rewriting daemon plumbing in one step.  Follow-up
# changes in subsequent milestones will plug in the concrete implementations.
import concurrent.futures
import contextlib
import json
import logging
import os
import random
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, Iterator, Literal, TypeVar, cast

import torch
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from tensorcast.api import _metrics as store_metrics
from tensorcast.api._config import GetArtifactOptions, PlanType, RegisterArtifactOptions
from tensorcast.api._device import resolve_device
from tensorcast.api._errors import DeviceMismatch, TensorCastError
from tensorcast.api._indices import (
    build_v2_index_bytes,
    calculate_tensor_device_offsets,
    load_tensor_indices_from_dir,
)
from tensorcast.api._io_disk import load_dict_from_disk
from tensorcast.api._loader import MaterializedArtifact, materialize_artifact
from tensorcast.api._register import (
    RegisteredArtifact as _RegisterHandle,
)
from tensorcast.api._register import (
    RegisteredLease,
    RegistrationResult,
    _register_artifact_core,
)
from tensorcast.api._utils import validate_disk_index_matches
from tensorcast.daemon_ctl import DaemonCtl, get_daemon_client
from tensorcast.observability.otel import set_span_attributes
from tensorcast.store_session_registry import StoreSessionRecord, write_record
from tensorcast.types import ServerConfig

T = TypeVar("T")


logger = logging.getLogger(__name__)

TensorDict = Mapping[str, torch.Tensor]

SpanAttributeValue = bool | int | float | str

ArtifactStatusCode = Literal[
    "OK",
    "INVALID_ARGUMENT",
    "FAILED_PRECONDITION",
    "NOT_FOUND",
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
    "UNAVAILABLE",
    "ABORTED",
    "DATA_LOSS",
    "CANCELLED",
    "UNKNOWN",
]


class ArtifactError(RuntimeError):
    """Structured exception surfaced by Store verbs.

    Mirrors gRPC canonical codes while providing a retryability hint so
    callers can implement tailored backoff policies.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: ArtifactStatusCode,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.status_code: ArtifactStatusCode = status_code
        self.retryable = bool(retryable)


@dataclass(frozen=True)
class RetryPolicy:
    deadline_seconds: float
    max_attempts: int
    base_backoff_seconds: float
    backoff_multiplier: float
    jitter: float


@dataclass(frozen=True)
class FallbackOptions:
    disk_path: str | None = None
    prefer_disk: bool = False
    allow_p2p: bool = True
    verify_checksums: bool = True


@dataclass(frozen=True)
class StoreOptions:
    fallback: FallbackOptions | None = None
    retry_overrides: Mapping[str, RetryPolicy] | None = None


@dataclass(frozen=True)
class CanonicalIndexEntry:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    segment_offset: int
    size_bytes: int


@dataclass(frozen=True)
class CanonicalIndex:
    entries: tuple[CanonicalIndexEntry, ...]
    total_size_bytes: int
    avbs_hash: str


ReplicaType = Literal["COALESCED_VRAM", "VRAM_LEASE_IN_PLACE", "VRAM_LEASED"]


@dataclass(frozen=True)
class ReplicaInfo:
    replica_id: str
    replica_type: ReplicaType
    device: torch.device
    plan: PlanType
    size_bytes: int


@dataclass(frozen=True)
class LeaseHandle:
    lease_id: str
    ttl_ms: int
    expires_at_monotonic: float
    owner_pid: int


@dataclass(frozen=True)
class RegisteredArtifact:
    artifact_id: str
    replica: ReplicaInfo
    canonical_index: CanonicalIndex
    lease: LeaseHandle | None
    state_dict: dict[str, torch.Tensor] | None = None
    registration_result: RegistrationResult | None = None


@dataclass(frozen=True)
class StoreCapabilities:
    mem_pool_bytes: int
    tx_slice_bytes: int
    artifact_chunk_bytes: int
    supports_coalesced: bool
    supports_lease: bool
    server_config: ServerConfig | None = None


class ArtifactFuture(Generic[T]):
    """Wrapper around :class:`concurrent.futures.Future` with confirm support."""

    def __init__(
        self,
        future: concurrent.futures.Future[T],
        *,
        confirm: Callable[[], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> None:
        self._future = future
        self._confirm = confirm
        self._cancel = cancel_callback

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> T:
        if not self._future.done() and self._confirm is not None:
            # Ensure daemon-side confirmation happens before materializing.
            self._confirm()
        return self._future.result(timeout=timeout)

    def confirm(self, timeout: float | None = None) -> T:
        """Idempotently block until the operation finishes."""

        return self.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if not self._future.done() and self._confirm is not None:
            self._confirm()
        return self._future.exception(timeout=timeout)

    def add_done_callback(
        self, callback: Callable[["ArtifactFuture[T]"], None]
    ) -> None:
        def _wrap(_: concurrent.futures.Future[T]) -> None:
            callback(self)

        self._future.add_done_callback(_wrap)

    def cancel(self) -> bool:
        if self._future.done():
            return False
        cancelled = False
        if self._cancel is not None:
            with contextlib.suppress(Exception):
                cancelled = bool(self._cancel())
        if not cancelled:
            cancelled = self._future.cancel()
        return cancelled


class _ForkAwareHandle:
    """Track per-Store resources that must be refreshed across fork."""

    def __init__(self, cleanup: Callable[[], None]) -> None:
        self._cleanup = cleanup

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._cleanup()


class Store:
    """Store-centric session orchestrating daemon interactions."""

    _AT_FORK_REGISTRY: "weakref.WeakSet[Store]" = weakref.WeakSet()
    _DEFAULT_LEASE_TTL_MS = 600_000

    def __init__(
        self, daemon_endpoint: str, *, opts: StoreOptions | None = None
    ) -> None:
        self._daemon_endpoint = daemon_endpoint
        self._opts = opts or StoreOptions()
        self._session_id = uuid.uuid4().hex
        self._tracer = trace.get_tracer(__name__)
        self._session_labels: dict[str, str] = {
            "daemon_endpoint": daemon_endpoint,
            "session_id": self._session_id,
        }
        self._capabilities: StoreCapabilities | None = None
        self._retry_policies = self._build_retry_policies()
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
        self._closed = False

        self._init_session_record()

        self._install_at_fork()
        self._client = self._create_client()
        self._initialize_session_metadata(self._client)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _create_client(self) -> DaemonCtl:
        return get_daemon_client(self._daemon_endpoint)

    @staticmethod
    def _default_capabilities() -> StoreCapabilities:
        return StoreCapabilities(
            mem_pool_bytes=0,
            tx_slice_bytes=0,
            artifact_chunk_bytes=0,
            supports_coalesced=False,
            supports_lease=False,
            server_config=None,
        )

    def _capabilities_from_config(self, config: ServerConfig) -> StoreCapabilities:
        mem_pool = int(getattr(config, "mem_pool_size", 0))
        tx_slice = int(getattr(config, "tx_slice_bytes", 0))
        artifact_chunk = int(getattr(config, "artifact_chunk_bytes", 0))
        supports_coalesced = mem_pool > 0
        supports_lease = True
        return StoreCapabilities(
            mem_pool_bytes=mem_pool,
            tx_slice_bytes=tx_slice,
            artifact_chunk_bytes=artifact_chunk,
            supports_coalesced=supports_coalesced,
            supports_lease=supports_lease,
            server_config=config,
        )

    def _active_lease_count(self) -> int:
        with self._leases_lock:
            return len(self._active_leases)

    @staticmethod
    def _capabilities_to_dict(capabilities: StoreCapabilities) -> dict[str, object]:
        data: dict[str, object] = {
            "mem_pool_bytes": int(capabilities.mem_pool_bytes),
            "tx_slice_bytes": int(capabilities.tx_slice_bytes),
            "artifact_chunk_bytes": int(capabilities.artifact_chunk_bytes),
            "supports_coalesced": bool(capabilities.supports_coalesced),
            "supports_lease": bool(capabilities.supports_lease),
        }
        if capabilities.server_config is not None:
            data["server_config"] = capabilities.server_config.model_dump()
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

    def _build_retry_policies(self) -> dict[str, RetryPolicy]:
        defaults: dict[str, RetryPolicy] = {
            "register": RetryPolicy(
                deadline_seconds=30.0,
                max_attempts=2,
                base_backoff_seconds=0.2,
                backoff_multiplier=2.0,
                jitter=0.5,
            ),
            "put": RetryPolicy(
                deadline_seconds=45.0,
                max_attempts=2,
                base_backoff_seconds=0.2,
                backoff_multiplier=2.0,
                jitter=0.5,
            ),
            "get": RetryPolicy(
                deadline_seconds=40.0,
                max_attempts=3,
                base_backoff_seconds=0.1,
                backoff_multiplier=2.0,
                jitter=0.5,
            ),
            "get_into": RetryPolicy(
                deadline_seconds=40.0,
                max_attempts=3,
                base_backoff_seconds=0.1,
                backoff_multiplier=2.0,
                jitter=0.5,
            ),
        }
        overrides = dict(self._opts.retry_overrides or {})
        defaults.update(overrides)
        return defaults

    def _record_session_start(self, capabilities: StoreCapabilities) -> None:
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.session_id": self._session_id,
            "tc.store.daemon": self._daemon_endpoint,
            "tc.store.mem_pool_bytes": int(capabilities.mem_pool_bytes),
            "tc.store.tx_slice_bytes": int(capabilities.tx_slice_bytes),
            "tc.store.artifact_chunk_bytes": int(capabilities.artifact_chunk_bytes),
            "tc.store.supports_coalesced": bool(capabilities.supports_coalesced),
            "tc.store.supports_lease": bool(capabilities.supports_lease),
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
                    "supports_coalesced": capabilities.supports_coalesced,
                    "supports_lease": capabilities.supports_lease,
                },
            },
        )
        self._session_labels.update(
            {
                "supports_coalesced": str(capabilities.supports_coalesced),
                "supports_lease": str(capabilities.supports_lease),
            }
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
                "supports_coalesced": str(capabilities.supports_coalesced),
                "supports_lease": str(capabilities.supports_lease),
            }
        )

    def _install_at_fork(self) -> None:
        Store._AT_FORK_REGISTRY.add(self)

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
        self._capabilities = None
        self._retry_policies = self._build_retry_policies()
        self._init_session_record()

    def _ensure_client(self) -> DaemonCtl:
        if self._closed:
            raise RuntimeError("Store is closed")
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
        self._release_all_leases()
        self._executor_handle.close()
        self._update_session_record(activity=True, closed=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dtype_from_string(value: str) -> torch.dtype:
        canonical = value
        if "." in value:
            _, canonical = value.rsplit(".", 1)
        dtype_obj = getattr(torch, canonical, None)
        if isinstance(dtype_obj, torch.dtype):
            return dtype_obj
        raise ArtifactError(
            f"Unsupported dtype '{value}' in canonical index",
            status_code="DATA_LOSS",
            retryable=False,
        )

    def _canonical_index_from_bytes(
        self, index_bytes: bytes, *, avbs_hash: str = ""
    ) -> CanonicalIndex:
        try:
            raw = json.loads(index_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                "Failed to parse canonical index JSON",
                status_code="DATA_LOSS",
                retryable=False,
            ) from exc

        entries: list[CanonicalIndexEntry] = []
        total = 0
        for name, meta in raw.items():
            if not isinstance(meta, (list, tuple)) or len(meta) != 6:
                raise ArtifactError(
                    f"Invalid canonical index entry for '{name}'",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            offset, size_bytes, shape, stride, dtype_str, storage_offset = meta
            dtype = self._dtype_from_string(str(dtype_str))
            entry = CanonicalIndexEntry(
                name=name,
                dtype=dtype,
                shape=tuple(int(x) for x in shape),
                stride=tuple(int(x) for x in stride),
                storage_offset=int(storage_offset),
                segment_offset=int(offset),
                size_bytes=int(size_bytes),
            )
            entries.append(entry)
            total += entry.size_bytes

        return CanonicalIndex(
            entries=tuple(entries),
            total_size_bytes=total,
            avbs_hash=avbs_hash,
        )

    def _canonical_index_from_result(
        self, result: RegistrationResult
    ) -> CanonicalIndex:
        return self._canonical_index_from_bytes(
            result.index_bytes, avbs_hash=result.descriptor.data_multihash
        )

    @staticmethod
    def _replica_type_for_plan(plan: PlanType) -> tuple[ReplicaType, PlanType]:
        if plan is PlanType.VRAM_COALESCED:
            return "COALESCED_VRAM", PlanType.VRAM_COALESCED
        return "VRAM_LEASE_IN_PLACE", PlanType.VRAM_LEASED

    def _replica_info_from_result(self, result: RegistrationResult) -> ReplicaInfo:
        replica_type, plan = self._replica_type_for_plan(result.plan)
        device = torch.device("cuda", int(result.build.device_id))
        size_bytes = int(result.layout.total_size)
        replica_id = result.descriptor.artifact_id
        return ReplicaInfo(
            replica_id=replica_id,
            replica_type=replica_type,
            device=device,
            plan=plan,
            size_bytes=size_bytes,
        )

    def _lease_handle_from_result(
        self, lease: RegisteredLease | None
    ) -> LeaseHandle | None:
        if lease is None:
            return None
        ttl = max(0, int(lease.ttl_ms))
        return LeaseHandle(
            lease_id=lease.registration_id,
            ttl_ms=ttl,
            expires_at_monotonic=time.monotonic() + ttl / 1000.0 if ttl else 0.0,
            owner_pid=int(lease.owner_pid),
        )

    def _track_lease(self, lease: RegisteredLease | None) -> None:
        if lease is None:
            return
        lease.__enter__()
        with self._leases_lock:
            self._active_leases.add(lease)
        self._update_session_record(activity=False)

    def _release_all_leases(self) -> None:
        with self._leases_lock:
            leases = list(self._active_leases)
            self._active_leases.clear()
        for lease in leases:
            with contextlib.suppress(Exception):
                lease.__exit__(None, None, None)
        self._update_session_record(activity=False)

    @staticmethod
    def _select_device_for_put(tensors: TensorDict) -> int:
        device_id: int | None = None
        for tensor in tensors.values():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.is_cuda:
                idx = tensor.device.index or 0
                if device_id is None:
                    device_id = idx
                elif device_id != idx:
                    raise ArtifactError(
                        "All CUDA tensors must reside on the same device",
                        status_code="INVALID_ARGUMENT",
                        retryable=False,
                    )
        if device_id is not None:
            return device_id
        if torch.cuda.is_available():
            try:
                return int(torch.cuda.current_device())
            except Exception:  # noqa: BLE001
                return 0
        return 0

    def _map_registration_error(self, exc: Exception) -> ArtifactError:
        if isinstance(exc, ArtifactError):
            return exc
        if isinstance(exc, CancelledError):
            return ArtifactError(
                "Registration cancelled",
                status_code="CANCELLED",
                retryable=False,
            )
        message = str(exc) or "registration failed"
        if isinstance(exc, DeviceMismatch):
            return ArtifactError(
                message, status_code="INVALID_ARGUMENT", retryable=False
            )
        if isinstance(exc, MemoryError):
            return ArtifactError(
                message, status_code="RESOURCE_EXHAUSTED", retryable=True
            )
        if isinstance(exc, TimeoutError):
            return ArtifactError(
                message, status_code="DEADLINE_EXCEEDED", retryable=True
            )
        if isinstance(exc, TensorCastError):
            return ArtifactError(
                message, status_code="FAILED_PRECONDITION", retryable=False
            )
        if isinstance(exc, RuntimeError) and "not available" in message.lower():
            return ArtifactError(message, status_code="UNAVAILABLE", retryable=True)
        return ArtifactError(message, status_code="UNKNOWN", retryable=False)

    def _attempt_registration(
        self,
        tensors: TensorDict,
        *,
        key: str | None,
        plan: PlanType,
        cancel_event: threading.Event | None = None,
        on_begin: Callable[[_RegisterHandle], None] | None = None,
        options_override: RegisterArtifactOptions | None = None,
        ttl_override: int | None = None,
        device_override: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        material = dict(tensors)
        if not material:
            raise ArtifactError(
                "Artifact tensors must not be empty",
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
        ttl_ms = (
            ttl_override
            if ttl_override is not None
            else self._DEFAULT_LEASE_TTL_MS
            if plan is PlanType.VRAM_LEASED
            else None
        )
        if plan is PlanType.VRAM_LEASED:
            device_id = None
        elif device_override is not None:
            device_id = resolve_device(device_override)
        else:
            device_id = self._select_device_for_put(material)
        try:
            registration_result = _register_artifact_core(
                artifact=material,
                options=options,
                device_id=device_id,
                ttl_ms=ttl_ms,
                force_lease_in_place=plan is PlanType.VRAM_LEASED,
                prevalidate_disk=self._should_prevalidate_disk(options),
                client=self._ensure_client(),
                daemon_address=self._daemon_endpoint,
                cancel_event=cancel_event,
                on_begin=on_begin,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._map_registration_error(exc) from exc
        return self._registration_to_artifact(registration_result)

    def _should_retry_registration(
        self,
        method: str,
        error: ArtifactError,
        *,
        attempt: int,
        policy: RetryPolicy | None,
        start_time: float,
        cancel_event: threading.Event | None,
    ) -> bool:
        if cancel_event and cancel_event.is_set():
            return False
        if policy is None:
            return False
        if attempt >= policy.max_attempts:
            return False
        if not error.retryable:
            return False
        if error.status_code not in {"UNAVAILABLE", "DEADLINE_EXCEEDED", "ABORTED"}:
            return False
        if policy.deadline_seconds > 0:
            elapsed = time.monotonic() - start_time
            if elapsed >= policy.deadline_seconds:
                return False
        return True

    @staticmethod
    def _compute_retry_delay(policy: RetryPolicy, attempt: int) -> float:
        delay = policy.base_backoff_seconds * (
            policy.backoff_multiplier ** max(0, attempt - 1)
        )
        if policy.jitter > 0:
            factor = 1.0 + random.uniform(-policy.jitter, policy.jitter)
            delay = max(0.0, delay * factor)
        return delay

    @staticmethod
    def _remaining_budget(policy: RetryPolicy, start_time: float) -> float | None:
        if policy.deadline_seconds <= 0:
            return None
        remaining = policy.deadline_seconds - (time.monotonic() - start_time)
        return remaining

    @staticmethod
    def _should_prevalidate_disk(options: RegisterArtifactOptions) -> bool:
        disk_path = options.disk_path
        if disk_path is None:
            return False
        if not isinstance(disk_path, str):
            return False
        return disk_path.strip() != ""

    def _perform_registration(
        self,
        tensors: TensorDict,
        *,
        key: str | None,
        plan: PlanType,
        cancel_event: threading.Event | None = None,
        on_begin: Callable[[_RegisterHandle], None] | None = None,
        options_override: RegisterArtifactOptions | None = None,
        ttl_override: int | None = None,
        device_override: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        method = "register" if plan is PlanType.VRAM_LEASED else "put"
        span_name = "Store/Register" if plan is PlanType.VRAM_LEASED else "Store/Put"
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.daemon": self._daemon_endpoint,
            "tc.store.session_id": self._session_id,
            "tc.store.plan": plan.value,
            "tc.store.key_present": bool(key),
        }
        policy = self._retry_policies.get(method)
        attempt = 1
        start_time = time.monotonic()

        with self._operation_span(span_name, attributes) as span:

            def record_outcome(status: str) -> None:
                duration = time.monotonic() - start_time
                store_metrics.observe_latency(
                    method, self._daemon_endpoint, status, duration
                )
                span.set_attribute("tc.store.status", status)
                if status not in {"OK", "CANCELLED"}:
                    store_metrics.increment_error(method, self._daemon_endpoint, status)

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
                        key=key,
                        plan=plan,
                        cancel_event=cancel_event,
                        on_begin=on_begin,
                        options_override=options_override,
                        ttl_override=ttl_override,
                        device_override=device_override,
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
                    should_retry = self._should_retry_registration(
                        method,
                        error,
                        attempt=attempt,
                        policy=policy,
                        start_time=start_time,
                        cancel_event=cancel_event,
                    )
                    if not should_retry:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise
                    assert policy is not None
                    delay = self._compute_retry_delay(policy, attempt)
                    remaining = self._remaining_budget(policy, start_time)
                    if remaining is not None and remaining <= 0:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining))
                    store_metrics.increment_retry(
                        method, self._daemon_endpoint, error.status_code
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
                                "tc.store.daemon": self._daemon_endpoint,
                                "tc.store.method": method,
                                "tc.store.attempt": attempt + 1,
                                "tc.store.delay_sec": delay,
                                "tc.store.status_code": error.status_code,
                            },
                        )
                        time.sleep(delay)
                    attempt += 1

    def _registration_to_artifact(
        self, result: RegistrationResult
    ) -> RegisteredArtifact:
        canonical_index = self._canonical_index_from_result(result)
        replica = self._replica_info_from_result(result)
        lease_handle = self._lease_handle_from_result(result.lease)
        self._track_lease(result.lease)
        return RegisteredArtifact(
            artifact_id=result.descriptor.artifact_id,
            replica=replica,
            canonical_index=canonical_index,
            lease=lease_handle,
            state_dict=result.state_dict,
            registration_result=result,
        )

    def _track_future(self, future: concurrent.futures.Future[object]) -> None:
        with self._metadata_lock:
            self._pending_futures.add(future)
            self._persist_session_record_locked(activity=False)

        def _cleanup(_future: concurrent.futures.Future[object]) -> None:
            with self._metadata_lock:
                self._pending_futures.discard(_future)
                self._persist_session_record_locked(activity=False)

        future.add_done_callback(_cleanup)

    @contextlib.contextmanager
    def _operation_span(
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

    @staticmethod
    def _resolve_identifiers(
        artifact_id: str | None, key: str | None
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

    @staticmethod
    def _resolve_device_selector(selector: torch.device | str | None) -> int:
        if selector is None:
            if not torch.cuda.is_available():
                raise ArtifactError(
                    "CUDA device required for retrieval",
                    status_code="FAILED_PRECONDITION",
                    retryable=True,
                )
            return int(torch.cuda.current_device())
        if isinstance(selector, torch.device):
            if selector.type != "cuda":
                raise ArtifactError(
                    f"Unsupported device selector {selector}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return int(selector.index or 0)
        if isinstance(selector, str):
            device = torch.device(selector)
            if device.type != "cuda":
                raise ArtifactError(
                    f"Unsupported device selector {selector}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return int(device.index or 0)
        return resolve_device(selector)

    def _build_get_options(
        self,
        fallback: FallbackOptions | None,
        options_override: GetArtifactOptions | None,
    ) -> GetArtifactOptions:
        options = options_override or GetArtifactOptions()
        if (
            fallback is not None
            and options.enable_verification != fallback.verify_checksums
        ):
            options = replace(options, enable_verification=fallback.verify_checksums)
        return options

    def _ensure_fallback_supported(self, fallback: FallbackOptions | None) -> None:
        if fallback and fallback.disk_path and fallback.disk_path.strip() == "":
            raise ArtifactError(
                "Fallback disk_path must not be empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

    def _materialize(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device_id: int,
        options: GetArtifactOptions,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None = None,
        span: Span | None = None,
    ) -> MaterializedArtifact:
        client = self._ensure_client()
        disk_error: Exception | None = None
        disk_path: str | None = None
        resolved_artifact_id = artifact_id
        fallback_opts = fallback

        result: MaterializedArtifact | None = None

        if fallback_opts and (
            fallback_opts.prefer_disk
            or fallback_opts.disk_path
            or not fallback_opts.allow_p2p
        ):
            disk_path, resolved_artifact_id, disk_error = self._resolve_disk_path(
                fallback=fallback_opts,
                client=client,
                key=key,
                artifact_id=artifact_id,
            )
            if disk_path:
                self._record_fallback_event(
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
                            "tc.store.verify_checksums": bool(
                                fallback_opts.verify_checksums
                            ),
                        },
                    )
                result = self._materialize_from_disk(
                    disk_path=disk_path,
                    artifact_id=resolved_artifact_id,
                    device_id=device_id,
                    verify_checksums=fallback_opts.verify_checksums,
                )
                if cancel_event and cancel_event.is_set():
                    raise CancelledError
                return result
            if fallback_opts and not fallback_opts.allow_p2p:
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

        if result is None:
            result = materialize_artifact(
                client=client,
                daemon_address=self._daemon_endpoint,
                device_id=device_id,
                artifact_id=artifact_id,
                key=key,
                options=options,
            )
        assert result is not None
        self._record_fallback_event(
            mode="p2p",
            artifact_id=result.artifact_id or artifact_id,
            key=key,
            detail={
                "disk_requested": bool(fallback and fallback.prefer_disk),
                "allow_p2p": bool(fallback is None or fallback.allow_p2p),
            },
        )
        if span is not None:
            span.add_event(
                "store.materialize.p2p",
                {
                    "tc.artifact.id": result.artifact_id or artifact_id or "",
                    "tc.store.disk_requested": bool(fallback and fallback.prefer_disk),
                    "tc.store.allow_p2p": bool(fallback is None or fallback.allow_p2p),
                },
            )
        if cancel_event and cancel_event.is_set():
            self._release_materialized(result, client)
            raise CancelledError
        return result

    def _resolve_disk_path(
        self,
        *,
        fallback: FallbackOptions,
        client: DaemonCtl,
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
            resolved_id, mapped_path = client.resolve_key_mapping(key)
        except Exception as exc:  # noqa: BLE001
            return None, resolved_artifact_id, exc
        if mapped_path:
            if not resolved_artifact_id:
                resolved_artifact_id = resolved_id or resolved_artifact_id
            return mapped_path, resolved_artifact_id, None
        return None, resolved_artifact_id, None

    def _materialize_from_disk(
        self,
        *,
        disk_path: str,
        artifact_id: str | None,
        device_id: int,
        verify_checksums: bool,
    ) -> MaterializedArtifact:
        raw_path = Path(str(disk_path))
        if not raw_path.exists():
            raise ArtifactError(
                f"Disk path '{disk_path}' not found",
                status_code="NOT_FOUND",
                retryable=False,
            )
        tensor_meta_index, tensor_data_index = load_tensor_indices_from_dir(raw_path)
        tensor_device_offsets, _ = calculate_tensor_device_offsets(
            tensor_data_index, device_id
        )
        canonical_index = build_v2_index_bytes(
            tensor_meta_index,
            tensor_data_index,
            tensor_device_offsets,
            device_id,
        )
        if verify_checksums:
            validate_disk_index_matches(canonical_index, str(raw_path))
        state_dict = load_dict_from_disk(
            raw_path,
            device_id=device_id,
        )
        return MaterializedArtifact(
            artifact_id=artifact_id or "",
            state_dict=state_dict,
            canonical_index_bytes=canonical_index,
            replica_uuid="",
            disk_path=str(raw_path),
        )

    def _record_fallback_event(
        self,
        *,
        mode: str,
        artifact_id: str | None,
        key: str | None,
        detail: Mapping[str, object],
    ) -> None:
        logger.info(
            "store.fallback",  # structured log tag
            extra={
                "tc.store.daemon": self._daemon_endpoint,
                "tc.store.mode": mode,
                "tc.store.artifact_id": artifact_id or "",
                "tc.store.key": key or "",
                "tc.store.detail": dict(detail),
            },
        )

    def _validate_targets(
        self,
        *,
        canonical_index: CanonicalIndex,
        target: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        device_id: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        validated: list[tuple[torch.Tensor, torch.Tensor]] = []
        device = torch.device("cuda", device_id)
        for entry in canonical_index.entries:
            if entry.name not in target:
                raise ArtifactError(
                    f"Target tensor '{entry.name}' missing",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if entry.name not in source:
                raise ArtifactError(
                    f"Source tensor '{entry.name}' missing",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            tgt = target[entry.name]
            src = source[entry.name]
            if not isinstance(tgt, torch.Tensor):
                raise ArtifactError(
                    f"Target '{entry.name}' must be a tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tgt.device != device:
                raise ArtifactError(
                    (
                        f"Target tensor '{entry.name}' on {tgt.device}, expected cuda:{device.index}"
                    ),
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tgt.dtype != src.dtype:
                raise ArtifactError(
                    f"Target tensor '{entry.name}' dtype {tgt.dtype} != {src.dtype}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if tuple(tgt.shape) != entry.shape:
                raise ArtifactError(
                    f"Target tensor '{entry.name}' shape {tuple(tgt.shape)} != {entry.shape}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            validated.append((tgt, src))
        return validated

    def _perform_get(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device: torch.device | str | None,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None,
        options_override: GetArtifactOptions | None,
    ) -> tuple[MaterializedArtifact, int]:
        return self._attempt_get(
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
            cancel_event=cancel_event,
            options_override=options_override,
        )

    def _attempt_get(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        device: torch.device | str | None,
        fallback: FallbackOptions | None,
        cancel_event: threading.Event | None,
        options_override: GetArtifactOptions | None,
        span: Span | None = None,
    ) -> tuple[MaterializedArtifact, int]:
        artifact_id, key = self._resolve_identifiers(artifact_id, key)
        resolved_fallback = fallback or self._opts.fallback
        self._ensure_fallback_supported(resolved_fallback)
        device_id = self._resolve_device_selector(device)
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
            )
        except Exception as exc:  # noqa: BLE001
            raise self._map_materialization_error(exc) from exc
        return materialized, device_id

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
    ) -> tuple[MaterializedArtifact, int]:
        policy = self._retry_policies.get(method, self._retry_policies.get("get"))
        attempt = 1
        start_time = time.monotonic()
        span_name = "Store/GetInto" if method == "get_into" else "Store/Get"
        attributes: dict[str, SpanAttributeValue] = {
            "tc.store.daemon": self._daemon_endpoint,
            "tc.store.session_id": self._session_id,
            "tc.store.method": method,
            "tc.store.lookup.by_key": bool(key),
            "tc.store.lookup.by_id": bool(artifact_id),
            "tc.store.fallback.prefer_disk": bool(fallback and fallback.prefer_disk),
            "tc.store.fallback.allow_p2p": bool(fallback is None or fallback.allow_p2p),
        }

        with self._operation_span(span_name, attributes) as span:

            def record_outcome(status: str) -> None:
                duration = time.monotonic() - start_time
                store_metrics.observe_latency(
                    method, self._daemon_endpoint, status, duration
                )
                span.set_attribute("tc.store.status", status)
                if status not in {"OK", "CANCELLED"}:
                    store_metrics.increment_error(method, self._daemon_endpoint, status)

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
                    )
                    record_outcome("OK")
                    span.set_attribute("tc.store.retry.count", attempt)
                    span.set_attribute("tc.device.id", int(device_id))
                    span.set_attribute("tc.artifact.id", materialized.artifact_id)
                    span.set_attribute(
                        "tc.store.fallback.used_disk", bool(materialized.disk_path)
                    )
                    span.set_status(Status(StatusCode.OK))
                    return materialized, device_id
                except ArtifactError as error:
                    span.record_exception(error)
                    should_retry = self._should_retry_registration(
                        method,
                        error,
                        attempt=attempt,
                        policy=policy,
                        start_time=start_time,
                        cancel_event=cancel_event,
                    )
                    if not should_retry:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise
                    assert policy is not None
                    delay = self._compute_retry_delay(policy, attempt)
                    remaining = self._remaining_budget(policy, start_time)
                    if remaining is not None and remaining <= 0:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining))
                    store_metrics.increment_retry(
                        method, self._daemon_endpoint, error.status_code
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
                                "tc.store.daemon": self._daemon_endpoint,
                                "tc.store.method": method,
                                "tc.store.attempt": attempt + 1,
                                "tc.store.delay_sec": delay,
                                "tc.store.status_code": error.status_code,
                            },
                        )
                        time.sleep(delay)
                    attempt += 1
                except Exception as exc:  # noqa: BLE001
                    error = self._map_materialization_error(exc)
                    span.record_exception(error)
                    should_retry = self._should_retry_registration(
                        method,
                        error,
                        attempt=attempt,
                        policy=policy,
                        start_time=start_time,
                        cancel_event=cancel_event,
                    )
                    if not should_retry:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise error from None
                    assert policy is not None
                    delay = self._compute_retry_delay(policy, attempt)
                    remaining = self._remaining_budget(policy, start_time)
                    if remaining is not None and remaining <= 0:
                        record_outcome(error.status_code)
                        span.set_status(Status(StatusCode.ERROR, error.status_code))
                        raise error from None
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining))
                    store_metrics.increment_retry(
                        method, self._daemon_endpoint, error.status_code
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
                                "tc.store.daemon": self._daemon_endpoint,
                                "tc.store.method": method,
                                "tc.store.attempt": attempt + 1,
                                "tc.store.delay_sec": delay,
                                "tc.store.status_code": error.status_code,
                            },
                        )
                        time.sleep(delay)
                    attempt += 1

    def _release_materialized(
        self, materialized: MaterializedArtifact, client: DaemonCtl
    ) -> None:
        replica_uuid = materialized.replica_uuid
        if not replica_uuid:
            return
        disk_path = materialized.disk_path or ""
        if not client.unload_replica(replica_uuid, disk_path=disk_path):
            logger.warning(
                "store.cancel_unload_failed",
                extra={
                    "tc.store.daemon": self._daemon_endpoint,
                    "tc.store.replica_uuid": replica_uuid,
                    "tc.store.disk_path": disk_path,
                },
            )

    def _map_materialization_error(self, exc: Exception) -> ArtifactError:
        if isinstance(exc, ArtifactError):
            return exc
        if isinstance(exc, CancelledError):
            return ArtifactError(
                "Retrieval cancelled",
                status_code="CANCELLED",
                retryable=False,
            )
        message = str(exc) or "retrieval failed"
        if isinstance(exc, TimeoutError):
            return ArtifactError(
                message, status_code="DEADLINE_EXCEEDED", retryable=True
            )
        if isinstance(exc, TensorCastError):
            return ArtifactError(
                message, status_code="FAILED_PRECONDITION", retryable=False
            )
        if isinstance(exc, RuntimeError):
            lowered = message.lower()
            if "not found" in lowered:
                return ArtifactError(message, status_code="NOT_FOUND", retryable=False)
            if "unavailable" in lowered or "not available" in lowered:
                return ArtifactError(message, status_code="UNAVAILABLE", retryable=True)
        return ArtifactError(message, status_code="UNKNOWN", retryable=False)

    @property
    def capabilities(self) -> StoreCapabilities:
        if self._capabilities is None:
            return self._default_capabilities()
        return self._capabilities

    # ------------------------------------------------------------------
    # Verb placeholders – wired up in later milestones.
    # ------------------------------------------------------------------
    def register(
        self,
        tensors: TensorDict,
        *,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> RegisteredArtifact:
        return self._perform_registration(
            tensors,
            key=key,
            plan=PlanType.VRAM_LEASED,
            options_override=options,
            ttl_override=ttl_ms,
        )

    def register_async(
        self,
        tensors: TensorDict,
        *,
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

        future = self._executor.submit(_task)
        self._track_future(cast(concurrent.futures.Future[object], future))

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

        return ArtifactFuture(future, cancel_callback=_cancel)

    def put(
        self,
        tensors: TensorDict,
        *,
        key: str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        return self._perform_registration(
            tensors,
            key=key,
            plan=PlanType.VRAM_COALESCED,
            options_override=options,
            device_override=device,
        )

    def put_async(
        self,
        tensors: TensorDict,
        *,
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

        future = self._executor.submit(_task)
        self._track_future(cast(concurrent.futures.Future[object], future))

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

        return ArtifactFuture(future, cancel_callback=_cancel)

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
                    self._release_materialized(materialized, self._ensure_client())
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
                    self._release_materialized(materialized, self._ensure_client())

        future = self._executor.submit(_task)
        self._track_future(cast(concurrent.futures.Future[object], future))

        def _cancel() -> bool:
            if cancel_event.is_set():
                return False
            cancel_event.set()
            with mat_lock:
                materialized = mat_ref["value"]
                mat_ref["value"] = None
            if materialized is not None:
                self._release_materialized(materialized, self._ensure_client())
            return True

        return ArtifactFuture(future, cancel_callback=_cancel)

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
            canonical_index = self._canonical_index_from_bytes(
                materialized.canonical_index_bytes
            )
            pairs = self._validate_targets(
                canonical_index=canonical_index,
                target=target,
                source=materialized.state_dict,
                device_id=device_id,
            )
            for tgt, src in pairs:
                tgt.copy_(src)
        finally:
            self._release_materialized(materialized, self._ensure_client())

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
                    self._release_materialized(materialized, self._ensure_client())
                    materialized = None
                    raise CancelledError
                try:
                    canonical_index = self._canonical_index_from_bytes(
                        materialized.canonical_index_bytes
                    )
                    pairs = self._validate_targets(
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
                        self._release_materialized(materialized, self._ensure_client())
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

        future = self._executor.submit(_task)
        self._track_future(cast(concurrent.futures.Future[object], future))

        def _cancel() -> bool:
            if cancel_event.is_set():
                return False
            cancel_event.set()
            with mat_lock:
                materialized = mat_ref["value"]
                mat_ref["value"] = None
            if materialized is not None:
                self._release_materialized(materialized, self._ensure_client())
            return True

        return ArtifactFuture(future, cancel_callback=_cancel)


__all__ = [
    "ArtifactError",
    "ArtifactFuture",
    "ArtifactStatusCode",
    "CanonicalIndex",
    "CanonicalIndexEntry",
    "FallbackOptions",
    "LeaseHandle",
    "RegisteredArtifact",
    "ReplicaInfo",
    "StoreCapabilities",
    "RetryPolicy",
    "Store",
    "StoreOptions",
    "TensorDict",
]
