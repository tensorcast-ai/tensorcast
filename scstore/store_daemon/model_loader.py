#  Copyright (c) 2025, StepCast Team.

"""Model loading and management functionality for StoreDaemon."""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps

# NOTE: Forward references for type hints to avoid circular imports
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    TypeVar,
)

import grpc

from scstore import _store_engine as _cs
from scstore.logger import init_logger
from scstore.proto import store_daemon_pb2
from scstore.store_daemon.utils import resolve_device_id

# Note: ModelLoader now solely interacts with GPUs.  All loading paths
# go through the unified `store_engine.load_model()` API.  Legacy
# helpers that targeted CPU have been removed for clarity.
from .metrics import (
    MODEL_LOAD_DURATION,
    MODELS_LOAD_FAILURES_TOTAL,
    MODELS_LOADED_TOTAL,
    get_device_type_label,
)

logger = init_logger(__name__)


class LoadResult(NamedTuple):
    """Result of a model loading operation."""

    success: bool
    error: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "LoadResult":
        """Convert various return types to LoadResult.

        Args:
            value: Can be:
                - LoadResult instance (returned as-is)
                - bool (converted to LoadResult)
                - tuple[bool, ...] (first element used as success)
                - Any other value (treated as success=True)

        Returns:
            LoadResult instance
        """
        if hasattr(value, "success") and hasattr(value, "error"):
            # Already a LoadResult
            return value
        elif value is False:
            return cls(success=False, error="Load operation failed")
        elif hasattr(value, "__getitem__"):
            # Tuple-like object, use first element
            try:
                return cls(success=bool(value[0]), error=None)
            except (IndexError, TypeError):
                return cls(success=True, error=None)
        else:
            # Any other truthy value
            return cls(success=bool(value), error=None)


# ---------------------------------------------------------------------------
# Forward declarations (TYPE_CHECKING) to prevent heavy import at runtime
# ---------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from .servicer import StoreDaemonServicer


@dataclass
class MemoryHandle:
    """Container holding CUDA IPC handle bytes together with the base GPU pointer.

    This replaces the loosely-typed (bytes, int) tuple that was previously used
    across the loader implementation, providing clearer semantics and static
    type-checking support.
    """

    handle_bytes: bytes  # The CUDA IPC handle owned by the daemon
    gpu_ptr: int  # Base GPU pointer returned by StoreEngine


TFunc = TypeVar("TFunc", bound=Callable[..., int])


def with_retry(max_retries: int, delay: float) -> Callable[[TFunc], TFunc]:
    """Decorator for retrying operations with exponential backoff.

    The decorated callable must return an ``int`` status code where ``0``
    indicates success.
    """

    def decorator(func: TFunc) -> TFunc:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = -1  # Default error code
            for attempt in range(1, max_retries + 1):
                result = func(self, *args, **kwargs)
                if result == 0:  # Assume 0 is success
                    return 0

                # Last attempt failed, don't sleep
                if attempt == max_retries:
                    break

                # Exponential backoff with some randomization
                sleep_time = min(delay * (2 ** (attempt - 1)), 1.0) * (
                    0.9 + 0.2 * (attempt / max_retries)
                )
                time.sleep(sleep_time)

            return result  # Return the last error code

        return wrapper  # type: ignore[return-value]

    return decorator


class ModelLoader:
    """Handles model loading, unloading, and registration."""

    # Class constants
    # Lifecycle constants moved to ReplicaManager
    COMM_TIMEOUT_MS: int = 5000
    DEFAULT_GPU_DEVICE_ID: int = 0

    # ---------------------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------------------

    def __init__(self, servicer: "StoreDaemonServicer") -> None:
        """Initialize :class:`ModelLoader`.

        Parameters
        ----------
        servicer:
            The owning :class:`StoreDaemonServicer` instance that provides
            access to shared resources such as the ``store_engine`` and
            gRPC stubs.
        """

        # Keep a typed reference to the parent servicer
        self.servicer: "StoreDaemonServicer" = servicer
        self.store_engine = servicer.store_engine
        self.global_store_stub = servicer.global_store_stub
        # Thread pool for async loading operations
        self._async_executor = ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="ModelLoader"
        )

        self._replica_manager = servicer.replica_manager

    def start_async_load(
        self,
        request: store_daemon_pb2.LoadModelRequest,
        context: grpc.ServicerContext,
    ) -> tuple[bool, bytes | None, Future]:
        """Start asynchronous model loading, returning immediately after memory allocation.

        This method initiates the model loading process and returns a Future that will
        complete when the actual data loading finishes. The method returns immediately
        after memory allocation succeeds (ALLOCATED state).

        Args:
            request: LoadModelRequest containing model path, device uuid, and memory handle
            context: gRPC service context

        Returns:
            Tuple of (allocation_success, cuda_ipc_handle_bytes, loading_future)
            - allocation_success: True if memory allocation succeeded
            - cuda_ipc_handle_bytes: CUDA IPC handle for GPU loads
            - loading_future: Future that resolves when loading completes
        """
        # Inputs separation: prefer explicit model_id (mi2:...) when present; otherwise use disk_path
        # (backward-compat: fall back to legacy model_path as disk path).
        model_id_input: str | None = None
        disk_path_input: str | None = None
        try:
            # New fields may not exist on older stubs; guard via getattr
            model_id_input = request.model_id if hasattr(request, "model_id") else None
        except Exception:
            model_id_input = None
        try:
            disk_path_input = (
                request.disk_path if hasattr(request, "disk_path") else None
            )
        except Exception:
            disk_path_input = None
        # Backward compatibility: use legacy model_path as disk path when explicit fields absent
        legacy_model_path = request.model_path
        device_type = request.target_device_type
        device_type_label = get_device_type_label(device_type)

        # Determine allocation timeout for pinned memory operations
        pinned_allocation_timeout_ms = request.pinned_allocation_timeout_ms

        # Check if service is shutting down
        if self.servicer.shutting_down:
            MODELS_LOAD_FAILURES_TOTAL.labels(
                device_type=device_type_label,
                error_type="service_shutting_down",
            ).inc()
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Service is shutting down")
            # Return a failed future
            future = Future()
            future.set_result(False)
            return False, None, future

        # Validate request
        effective_disk_path = disk_path_input or legacy_model_path
        # Validate: at least one of (model_id mi2) or disk_path must be provided
        if not model_id_input and not self._validate_load_request(
            effective_disk_path, context
        ):
            MODELS_LOAD_FAILURES_TOTAL.labels(
                device_type=device_type_label,
                error_type="invalid_request",
            ).inc()
            future = Future()
            future.set_result(False)
            return False, None, future

        # Only GPU loading is supported
        if device_type != store_daemon_pb2.DEVICE_TYPE_GPU:
            logger.error("Only GPU loading is currently supported by ModelLoader")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            future = Future()
            future.set_result(False)
            return False, None, future

        # Call the C++ StoreEngine.prepare() API directly – Python no longer
        # needs the `_load_with_fallback` wrapper now that all source-selection
        # logic has moved into C++.

        mem_handle: MemoryHandle | None
        try:
            device_id = resolve_device_id(request.device_uuid)
            target_device_spec = f"gpu:{device_id}"

            # Attempt a best-effort eviction proportional to the expected model size
            expected_size_bytes: int = int(0.5 * request.size_bytes)
            try:
                self._replica_manager.maybe_evict(expected_size_bytes, device_id)
            except Exception:  # noqa: BLE001 – eviction is opportunistic
                logger.debug(
                    "Memory eviction attempt failed – continuing with prepare()"
                )

            # Prefer Global Store orchestrated AUTO routing only when the caller supplies
            # a content-addressed model_id (mi2:...). Never treat a disk path as model_id.
            if (
                self.servicer.global_store_enabled
                and isinstance(model_id_input, str)
                and model_id_input.startswith("mi2:")
            ):
                model_handle = self.store_engine.prepare(
                    target_device_spec,
                    _cs.PrepareMode.AUTO,
                    pinned_timeout_ms=(
                        pinned_allocation_timeout_ms
                        if pinned_allocation_timeout_ms > 0
                        else None
                    ),
                    model_id=model_id_input,
                )
            else:
                # Local-only path: load from disk explicitly.
                model_handle = self.store_engine.prepare(
                    target_device_spec,
                    _cs.PrepareMode.LOAD_ONLY,
                    pinned_timeout_ms=(
                        pinned_allocation_timeout_ms
                        if pinned_allocation_timeout_ms > 0
                        else None
                    ),
                    disk_path=effective_disk_path,
                )

            mem_handle = MemoryHandle(
                handle_bytes=model_handle.ipc_handle_bytes,
                gpu_ptr=model_handle.gpu_ptr,
            )

            def _wait_wrapper():
                try:
                    timeout_ms = (
                        pinned_allocation_timeout_ms
                        if pinned_allocation_timeout_ms > 0
                        else 0
                    )
                    model_handle.wait_ready(timeout_ms)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("wait_ready failed: %s", exc)
                    return False

            success, wait_fn, source_type = True, _wait_wrapper, "auto"
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "StoreEngine.prepare failed for %s: %s", request.model_path, exc
            )
            success, mem_handle, wait_fn, source_type = False, None, None, "error"

        if success and wait_fn and mem_handle:
            # --------------------------------------------------------------
            # Asynchronous path — schedule verification *after* the load
            # completes successfully.  This prevents premature verification
            # attempts on an un-initialised buffer that would inevitably
            # fail.
            # --------------------------------------------------------------

            start_time = time.time()

            # ------------------------------------------------------------------
            # Submit a **wrapped** function to the executor so that all
            # post-processing (metrics, remote-transport completion, optional
            # verification scheduling) runs *inside* the worker thread **before**
            # the Future transitions to FINISHED.  This guarantees callers that
            # block on `future.result()` (e.g. ConfirmModel) observe a fully
            # completed state, eliminating the need for explicit cleanup in
            # higher-level code.
            # ------------------------------------------------------------------

            def _load_and_finalize():  # executed in background thread
                # 1) Wait for the underlying load to finish
                try:
                    res = wait_fn()
                    result = LoadResult.from_value(res)
                    load_ok = result.success
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Load future raised exception: %s", exc)
                    result = LoadResult(success=False, error=str(exc))
                    load_ok = False
                    res = False

                # 2) Metrics & logging
                try:
                    load_duration = time.time() - start_time
                    MODEL_LOAD_DURATION.labels(
                        device_type=device_type_label,
                        source_type=source_type,
                    ).observe(load_duration)

                    MODELS_LOADED_TOTAL.labels(
                        device_type=device_type_label,
                        source_type=source_type,
                    ).inc()

                    logger.info(
                        "Async LoadModel: completed %s with target %s, duration=%.2fs",
                        request.model_path,
                        device_type_label,
                        load_duration,
                    )
                except Exception:  # pragma: no cover – best-effort metrics
                    logger.exception("Error recording metrics on load completion")

                # 3) Always attempt to complete the remote transport
                # The C++ orchestrator now handles completing remote transports.  The
                # Python layer no longer has visibility into transport IDs, so this
                # cleanup is obsolete and has been removed.

                # 4) Verification scheduling on success
                if load_ok:
                    try:
                        self._maybe_schedule_verification(
                            model_path=request.model_path,
                            request=request,
                            returned_mem_handle=mem_handle,
                            device_type=device_type,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to schedule verification post-load for %s: %s",
                            request.model_path,
                            exc,
                        )

                return res

            loading_future = self._async_executor.submit(_load_and_finalize)

            return True, mem_handle.handle_bytes, loading_future

        failed_future = Future()
        failed_future.set_result(False)
        return False, None, failed_future

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # The `_load_with_fallback` helper is no longer required – all logic has been
    # inlined into `start_async_load` above.

    def _maybe_schedule_verification(
        self,
        *,
        model_path: str,
        request: store_daemon_pb2.LoadModelRequest,
        returned_mem_handle: MemoryHandle | None,
        device_type: store_daemon_pb2.DeviceType,
    ) -> None:
        """Kick off integrity verification (GPU only).

        All heavy-lifting (device-ID resolution, reference tracking, JSON read,
        etc.) is performed by *StoreDaemonServicer.schedule_verification* to
        avoid duplicated logic in the loader layer.
        """

        # Only GPU loads with a valid memory handle are eligible.
        if device_type != store_daemon_pb2.DEVICE_TYPE_GPU or not returned_mem_handle:
            return

        # Delegate to servicer – this handles reference counting *and* (if
        # possible) schedules the background verification job.
        try:
            self.servicer.schedule_verification(
                model_path=model_path,
                replica_uuid=request.replica_uuid,
                device_uuid=request.device_uuid,
                cuda_ptr=returned_mem_handle.gpu_ptr,
                pid=request.pid,
                keep_for_global=request.keep_for_global,
                size_bytes=request.size_bytes,
            )
        except Exception as exc:  # noqa: BLE001 – best-effort pathway
            logger.warning(
                "Failed to schedule verification for %s (replica=%s): %s",
                model_path,
                request.replica_uuid,
                exc,
            )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_load_request(model_path: str, context: grpc.ServicerContext) -> bool:
        """Validate LoadModel request parameters."""
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return False
        return True

    def confirm_model(
        self,
        model_path: str,
        replica_uuid: str,
        device_type: store_daemon_pb2.DeviceType,
    ) -> bool:
        """Proxy to ReplicaManager.confirm_model."""
        return self._replica_manager.confirm_model(
            model_path=model_path,
            replica_uuid=replica_uuid,
            device_type=device_type,
        )

    def unload_model(
        self, model_path: str, device_type: store_daemon_pb2.DeviceType
    ) -> bool:
        """Proxy to ReplicaManager.unload_model."""
        return self._replica_manager.unload_model(model_path, device_type)

    def shutdown(self):
        """Clean up resources when shutting down."""
        # _async_executor is created during __init__, so it is always available here.
        self._async_executor.shutdown(wait=True)
