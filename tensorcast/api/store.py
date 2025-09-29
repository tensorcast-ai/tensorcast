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
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

import torch

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
    RegisteredLease,
    RegistrationResult,
    _register_artifact_core,
)
from tensorcast.api._utils import compose_artifact_dir, validate_disk_index_matches
from tensorcast.daemon_ctl import DaemonCtl, get_daemon_client

T = TypeVar("T")


logger = logging.getLogger(__name__)

TensorDict = Mapping[str, torch.Tensor]

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
    plan: Literal["vram_coalesced", "vram_leased"]
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
        self._client_lock = threading.RLock()
        self._client: DaemonCtl | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tensorcast-store"
        )
        self._executor_handle = _ForkAwareHandle(self._executor.shutdown)
        self._pending_futures: set[concurrent.futures.Future[object]] = set()
        self._leases_lock = threading.RLock()
        self._active_leases: set[RegisteredLease] = set()
        self._closed = False

        self._install_at_fork()
        self._client = self._create_client()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _create_client(self) -> DaemonCtl:
        return get_daemon_client(self._daemon_endpoint)

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

    def _ensure_client(self) -> DaemonCtl:
        if self._closed:
            raise RuntimeError("Store is closed")
        with self._client_lock:
            if self._client is None:
                self._client = self._create_client()
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
    def _replica_type_for_plan(plan: PlanType) -> tuple[ReplicaType, str]:
        if plan is PlanType.VRAM_COALESCED:
            return "COALESCED_VRAM", "vram_coalesced"
        return "VRAM_LEASE_IN_PLACE", "vram_leased"

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

    def _release_all_leases(self) -> None:
        with self._leases_lock:
            leases = list(self._active_leases)
            self._active_leases.clear()
        for lease in leases:
            with contextlib.suppress(Exception):
                lease.__exit__(None, None, None)

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

    def _raise_registration_error(self, exc: Exception) -> None:
        if isinstance(exc, ArtifactError):
            raise exc
        message = str(exc) or "registration failed"
        if isinstance(exc, DeviceMismatch):
            raise ArtifactError(
                message, status_code="INVALID_ARGUMENT", retryable=False
            ) from exc
        if isinstance(exc, MemoryError):
            raise ArtifactError(
                message, status_code="RESOURCE_EXHAUSTED", retryable=True
            ) from exc
        if isinstance(exc, TimeoutError):
            raise ArtifactError(
                message, status_code="DEADLINE_EXCEEDED", retryable=True
            ) from exc
        if isinstance(exc, TensorCastError):
            raise ArtifactError(
                message, status_code="FAILED_PRECONDITION", retryable=False
            ) from exc
        if isinstance(exc, RuntimeError) and "not available" in message.lower():
            raise ArtifactError(
                message, status_code="UNAVAILABLE", retryable=True
            ) from exc
        raise ArtifactError(message, status_code="UNKNOWN", retryable=False) from exc

    def _perform_registration(
        self,
        tensors: TensorDict,
        *,
        key: str | None,
        plan: PlanType,
    ) -> RegisteredArtifact:
        material = dict(tensors)
        if not material:
            raise ArtifactError(
                "Artifact tensors must not be empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        options = RegisterArtifactOptions(
            plan=plan,
            lease_in_place=plan is PlanType.VRAM_LEASED,
            key=key,
        )
        ttl_ms = self._DEFAULT_LEASE_TTL_MS if plan is PlanType.VRAM_LEASED else None
        device_id = (
            None
            if plan is PlanType.VRAM_LEASED
            else self._select_device_for_put(material)
        )
        try:
            result = _register_artifact_core(
                artifact=material,
                options=options,
                device_id=device_id,
                ttl_ms=ttl_ms,
                force_lease_in_place=plan is PlanType.VRAM_LEASED,
                prevalidate_disk=False,
                client=self._ensure_client(),
                daemon_address=self._daemon_endpoint,
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_registration_error(exc)
        return self._registration_to_artifact(result)

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
        )

    def _track_future(self, future: concurrent.futures.Future[object]) -> None:
        self._pending_futures.add(future)

        def _cleanup(_future: concurrent.futures.Future[object]) -> None:
            self._pending_futures.discard(_future)

        future.add_done_callback(_cleanup)

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
        return resolve_device(selector)

    def _build_get_options(
        self, fallback: FallbackOptions | None
    ) -> GetArtifactOptions:
        enable_verification = True if fallback is None else fallback.verify_checksums
        return GetArtifactOptions(
            prefer="p2p",
            enable_verification=enable_verification,
            wait_for_completion=True,
        )

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
    ) -> MaterializedArtifact:
        client = self._ensure_client()
        disk_error: Exception | None = None
        disk_path: str | None = None
        resolved_artifact_id = artifact_id
        fallback_opts = fallback

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
                return self._materialize_from_disk(
                    disk_path=disk_path,
                    artifact_id=resolved_artifact_id,
                    device_id=device_id,
                    verify_checksums=fallback_opts.verify_checksums,
                )
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

        result = materialize_artifact(
            client=client,
            daemon_address=self._daemon_endpoint,
            device_id=device_id,
            artifact_id=artifact_id,
            key=key,
            options=options,
        )
        self._record_fallback_event(
            mode="p2p",
            artifact_id=result.artifact_id or artifact_id,
            key=key,
            detail={
                "disk_requested": bool(fallback and fallback.prefer_disk),
                "allow_p2p": bool(fallback is None or fallback.allow_p2p),
            },
        )
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
        raw_path = compose_artifact_dir(Path(str(disk_path)), None)
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
            disk_path,
            device_id=device_id,
            storage_path=None,
        )
        return MaterializedArtifact(
            artifact_id=artifact_id or "",
            state_dict=state_dict,
            canonical_index_bytes=canonical_index,
            replica_uuid="",
            disk_path=disk_path,
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

    # ------------------------------------------------------------------
    # Verb placeholders – wired up in later milestones.
    # ------------------------------------------------------------------
    def register(
        self, tensors: TensorDict, *, key: str | None = None
    ) -> RegisteredArtifact:
        return self._perform_registration(
            tensors,
            key=key,
            plan=PlanType.VRAM_LEASED,
        )

    def register_async(
        self, tensors: TensorDict, *, key: str | None = None
    ) -> ArtifactFuture[RegisteredArtifact]:
        future = self._executor.submit(
            self._perform_registration,
            tensors,
            key=key,
            plan=PlanType.VRAM_LEASED,
        )
        self._track_future(future)
        return ArtifactFuture(future)

    def put(self, tensors: TensorDict, *, key: str | None = None) -> RegisteredArtifact:
        return self._perform_registration(
            tensors,
            key=key,
            plan=PlanType.VRAM_COALESCED,
        )

    def put_async(
        self, tensors: TensorDict, *, key: str | None = None
    ) -> ArtifactFuture[RegisteredArtifact]:
        future = self._executor.submit(
            self._perform_registration,
            tensors,
            key=key,
            plan=PlanType.VRAM_COALESCED,
        )
        self._track_future(future)
        return ArtifactFuture(future)

    def get(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
    ) -> dict[str, torch.Tensor]:
        artifact_id, key = self._resolve_identifiers(artifact_id, key)
        resolved_fallback = fallback or self._opts.fallback
        self._ensure_fallback_supported(resolved_fallback)
        device_id = self._resolve_device_selector(device)
        options = self._build_get_options(resolved_fallback)
        materialized = self._materialize(
            artifact_id=artifact_id,
            key=key,
            device_id=device_id,
            options=options,
            fallback=resolved_fallback,
        )
        return materialized.state_dict

    def get_async(
        self,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
    ) -> ArtifactFuture[dict[str, torch.Tensor]]:
        future = self._executor.submit(
            self.get,
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
        )
        self._track_future(future)
        return ArtifactFuture(future)

    def get_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
    ) -> None:
        artifact_id, key = self._resolve_identifiers(artifact_id, key)
        resolved_fallback = fallback or self._opts.fallback
        self._ensure_fallback_supported(resolved_fallback)
        device_id = self._resolve_device_selector(device)
        options = self._build_get_options(resolved_fallback)
        materialized = self._materialize(
            artifact_id=artifact_id,
            key=key,
            device_id=device_id,
            options=options,
            fallback=resolved_fallback,
        )
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

    def get_into_async(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        device: torch.device | str | None = None,
        fallback: FallbackOptions | None = None,
    ) -> ArtifactFuture[None]:
        future = self._executor.submit(
            self.get_into,
            target,
            artifact_id=artifact_id,
            key=key,
            device=device,
            fallback=fallback,
        )
        self._track_future(future)
        return ArtifactFuture(future)


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
    "RetryPolicy",
    "Store",
    "StoreOptions",
    "TensorDict",
]
