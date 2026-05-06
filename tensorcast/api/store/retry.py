#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import random
import re
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import CancelledError
from typing import NoReturn, cast

import grpc

from tensorcast.api._errors import DeviceMismatch, TensorCastError
from tensorcast.api.errors import CollectiveFailureClassName
from tensorcast.api.store.types import ArtifactError, RetryPolicy
from tensorcast.error_reporting import debug_errors_enabled, debug_errors_hint

RetryPolicyMap = Mapping[str, RetryPolicy]

_TRANSIENT_STATUS_CODES = frozenset(
    {
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "ABORTED",
        "RESOURCE_EXHAUSTED",
        "INTERNAL",
        "UNKNOWN",
    }
)
_TRANSIENT_MESSAGE_TOKENS = (
    "failed to confirm artifact loading",
    "stale",
    "no available replica",
    "within timeout",
    "transport",
    "write-write conflict",
    "conflict",
    "serialization",
    "try again",
    "temporary",
    "busy",
    "capacity",
    "resource exhausted",
    "too many requests",
    "connection reset",
    "connection refused",
    "broken pipe",
)
_COLLECTIVE_FAILURE_CLASS_METADATA_KEY = "tc.collective_failure_class"
_COLLECTIVE_FAILURE_CLASS_PATTERN = re.compile(
    r"\s*\[tc\.collective_failure_class=(not_eligible|execution_failed)\]\s*",
    re.IGNORECASE,
)


def _looks_transient_message(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in _TRANSIENT_MESSAGE_TOKENS)


def _append_debug_hint(message: str) -> str:
    if debug_errors_enabled():
        return message
    return f"{message}\nDebug: {debug_errors_hint()}"


def _append_hint(message: str, hint: str) -> str:
    return _append_debug_hint(f"{message}\nHint: {hint}")


def _normalize_collective_failure_class(
    value: object,
) -> CollectiveFailureClassName | None:
    normalized = str(value).strip().lower()
    if normalized in {"not_eligible", "execution_failed"}:
        return cast(CollectiveFailureClassName, normalized)
    return None


def _extract_collective_failure_class_metadata(
    exc: grpc.RpcError,
) -> CollectiveFailureClassName | None:
    try:
        metadata = exc.trailing_metadata()
    except Exception:  # noqa: BLE001
        return None
    raw_metadata = metadata
    if raw_metadata is None:
        return None
    metadata = cast(Iterable[tuple[str, object]], raw_metadata)
    for key, value in metadata:
        if str(key).lower() != _COLLECTIVE_FAILURE_CLASS_METADATA_KEY:
            continue
        normalized = _normalize_collective_failure_class(value)
        if normalized is not None:
            return normalized
    return None


def _extract_collective_failure_class(
    message: str,
) -> tuple[str, CollectiveFailureClassName | None]:
    match = _COLLECTIVE_FAILURE_CLASS_PATTERN.search(message)
    if match is None:
        return message, None
    cleaned = _COLLECTIVE_FAILURE_CLASS_PATTERN.sub(" ", message, count=1)
    return cleaned.strip(), _normalize_collective_failure_class(match.group(1))


def _global_store_not_connected_hint() -> str:
    return (
        "Materialization requires the daemon to be connected to Global Store. "
        "Set this when creating/starting daemon, for example "
        "`tc.init(mode='create', global_store_mode='connect', "
        "global_store_address='127.0.0.1:50051')`, or start the daemon with "
        "`uv run tensorcast-cli daemon start --global-store-mode connect "
        "--global-store-address 127.0.0.1:50051`. "
        "In `tc.init(mode='connect', ...)`, `global_store_*` parameters do not "
        "reconfigure an existing daemon. "
        "If you want TensorCast to launch a local Global Store, use "
        "`tc.init(mode='create', global_store_mode='start')`."
    )


def _selection_layout_mismatch_hint() -> str:
    return (
        "Client and daemon computed different selection layout hashes. "
        "This usually means view_index bytes are inconsistent with "
        "view_spec/tensor_names for the same request. "
        "Regenerate selection bytes from canonical index + view inputs, "
        "or retry without passing a custom view_index hint."
    )


def _grpc_details(exc: grpc.RpcError) -> str:
    try:
        details = exc.details()
    except Exception:  # noqa: BLE001
        return ""
    return str(details) if details else ""


def build_retry_policies(
    overrides: Mapping[str, RetryPolicy] | None = None,
) -> dict[str, RetryPolicy]:
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
    if overrides:
        defaults.update(dict(overrides))
    return defaults


def map_registration_error(exc: Exception) -> ArtifactError:
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
        return ArtifactError(message, status_code="INVALID_ARGUMENT", retryable=False)
    if isinstance(exc, MemoryError):
        return ArtifactError(message, status_code="RESOURCE_EXHAUSTED", retryable=True)
    if isinstance(exc, TimeoutError):
        return ArtifactError(message, status_code="DEADLINE_EXCEEDED", retryable=True)
    if isinstance(exc, TensorCastError):
        return ArtifactError(
            message, status_code="FAILED_PRECONDITION", retryable=False
        )
    if isinstance(exc, grpc.RpcError):
        status_code = exc.code()
        details = _grpc_details(exc)
        status_name = status_code.name if status_code is not None else "UNKNOWN"
        if (
            status_code == grpc.StatusCode.FAILED_PRECONDITION
            and "retry with placement=CLIENT" in details
        ):
            guidance = (
                "Daemon rejected SERVER placement for view registration; "
                "retry with placement='CLIENT'."
            )
            return ArtifactError(
                guidance,
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        retryable = status_code in {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.ABORTED,
        }
        mapped = details or "registration failed"
        if not details:
            mapped = _append_debug_hint(mapped)
        return ArtifactError(mapped, status_code=status_name, retryable=retryable)
    if isinstance(exc, RuntimeError) and "not available" in message.lower():
        return ArtifactError(message, status_code="UNAVAILABLE", retryable=True)
    return ArtifactError(message, status_code="UNKNOWN", retryable=False)


def map_materialization_error(exc: Exception) -> ArtifactError:
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
        return ArtifactError(message, status_code="DEADLINE_EXCEEDED", retryable=True)
    if isinstance(exc, TensorCastError):
        return ArtifactError(
            message, status_code="FAILED_PRECONDITION", retryable=False
        )
    if isinstance(exc, grpc.RpcError):
        status_code = exc.code()
        details = _grpc_details(exc)
        status_name = status_code.name if status_code is not None else "UNKNOWN"
        mapped = details or "retrieval failed"
        collective_failure_class = _extract_collective_failure_class_metadata(exc)
        mapped, message_failure_class = _extract_collective_failure_class(mapped)
        if collective_failure_class is None:
            collective_failure_class = message_failure_class
        lowered = mapped.lower()
        if "selection.logical_layout_hash does not match resolved selection" in mapped:
            return ArtifactError(
                _append_hint(mapped, _selection_layout_mismatch_hint()),
                status_code="INVALID_ARGUMENT",
                retryable=False,
                collective_failure_class=collective_failure_class,
            )
        if "globalstoreclient not connected" in lowered:
            return ArtifactError(
                _append_hint(mapped, _global_store_not_connected_hint()),
                status_code="FAILED_PRECONDITION",
                retryable=False,
                collective_failure_class=collective_failure_class,
            )
        if "tensor index not found" in lowered:
            hint = (
                "Ensure the directory contains tensor_index.json, tensor_index.cbor, or *.safetensors files. "
                "If this is a raw model folder, run a TensorCast save path "
                "(e.g., tensorcast.save_* examples) to generate the index."
            )
            return ArtifactError(
                _append_hint(mapped, hint),
                status_code="NOT_FOUND",
                retryable=False,
                collective_failure_class=collective_failure_class,
            )
        if "artifact_descriptor.json required when verify_checksums=true" in lowered:
            hint = (
                "Add artifact_descriptor.json alongside the tensors. For explicit imports via "
                "Store.from_disk(...), you can pass verify_checksums=False to skip descriptor validation."
            )
            return ArtifactError(
                _append_hint(mapped, hint),
                status_code="FAILED_PRECONDITION",
                retryable=False,
                collective_failure_class=collective_failure_class,
            )
        retryable = status_name in _TRANSIENT_STATUS_CODES or _looks_transient_message(
            mapped
        )
        if not details:
            mapped = _append_debug_hint(mapped)
        return ArtifactError(
            mapped,
            status_code=status_name,
            retryable=retryable,
            collective_failure_class=collective_failure_class,
        )
    message, collective_failure_class = _extract_collective_failure_class(message)
    lowered = message.lower()
    if "selection.logical_layout_hash does not match resolved selection" in message:
        return ArtifactError(
            _append_hint(message, _selection_layout_mismatch_hint()),
            status_code="INVALID_ARGUMENT",
            retryable=False,
            collective_failure_class=collective_failure_class,
        )
    if "globalstoreclient not connected" in lowered:
        return ArtifactError(
            _append_hint(message, _global_store_not_connected_hint()),
            status_code="FAILED_PRECONDITION",
            retryable=False,
            collective_failure_class=collective_failure_class,
        )
    if "tensor index not found" in lowered:
        hint = (
            "Ensure the directory contains tensor_index.json, tensor_index.cbor, or *.safetensors files. "
            "If this is a raw model folder, run a TensorCast save path "
            "(e.g., tensorcast.save_* examples) to generate the index."
        )
        message = _append_hint(message, hint)
        return ArtifactError(
            message,
            status_code="NOT_FOUND",
            retryable=False,
            collective_failure_class=collective_failure_class,
        )
    if "artifact_descriptor.json required when verify_checksums=true" in lowered:
        hint = (
            "Add artifact_descriptor.json alongside the tensors. For explicit imports via "
            "Store.from_disk(...), you can pass verify_checksums=False to skip descriptor validation."
        )
        message = _append_hint(message, hint)
        return ArtifactError(
            message,
            status_code="FAILED_PRECONDITION",
            retryable=False,
            collective_failure_class=collective_failure_class,
        )
    if isinstance(exc, RuntimeError):
        if "failed to confirm artifact loading" in lowered:
            hint = (
                "Transport source may be stale/unavailable (for example peer daemon just exited). "
                "Retrying get should reselect a healthy source."
            )
            return ArtifactError(
                _append_hint(message, hint),
                status_code="UNAVAILABLE",
                retryable=True,
                collective_failure_class=collective_failure_class,
            )
        if "not found" in lowered:
            return ArtifactError(
                message,
                status_code="NOT_FOUND",
                retryable=False,
                collective_failure_class=collective_failure_class,
            )
        if "unavailable" in lowered or "not available" in lowered:
            return ArtifactError(
                message,
                status_code="UNAVAILABLE",
                retryable=True,
                collective_failure_class=collective_failure_class,
            )
        if _looks_transient_message(message):
            return ArtifactError(
                message,
                status_code="UNAVAILABLE",
                retryable=True,
                collective_failure_class=collective_failure_class,
            )
    return ArtifactError(
        message,
        status_code="UNKNOWN",
        retryable=False,
        collective_failure_class=collective_failure_class,
    )


def raise_mapped_materialization_error(exc: Exception) -> NoReturn:
    error = map_materialization_error(exc)
    if debug_errors_enabled():
        raise error from exc
    raise error from None


def raise_mapped_registration_error(exc: Exception) -> NoReturn:
    error = map_registration_error(exc)
    if debug_errors_enabled():
        raise error from exc
    raise error from None


def should_retry(
    *,
    error: ArtifactError,
    attempt: int,
    policy: RetryPolicy | None,
    start_time: float,
    cancel_event: threading.Event | None = None,
) -> bool:
    if cancel_event and cancel_event.is_set():
        return False
    if policy is None:
        return False
    if attempt >= policy.max_attempts:
        return False
    if not error.retryable:
        return False
    if error.status_code not in _TRANSIENT_STATUS_CODES:
        return False
    if policy.deadline_seconds > 0:
        elapsed = time.monotonic() - start_time
        if elapsed >= policy.deadline_seconds:
            return False
    return True


def retry_reason_bucket(error: ArtifactError) -> str:
    status = str(error.status_code or "UNKNOWN").upper()
    message = str(error).lower()
    if status == "DEADLINE_EXCEEDED":
        return "deadline_exceeded"
    if status == "UNAVAILABLE":
        if "stale" in message or "confirm artifact loading" in message:
            return "stale_source"
        if "transport" in message:
            return "transport_unavailable"
        return "unavailable"
    if status == "ABORTED":
        return "transaction_conflict"
    if status == "RESOURCE_EXHAUSTED":
        return "resource_exhausted"
    if status in {"INTERNAL", "UNKNOWN"}:
        if _looks_transient_message(message):
            return "transient_internal"
        return "internal_or_unknown"
    if status == "CANCELLED":
        return "cancelled"
    if status == "NOT_FOUND":
        return "not_found"
    if status == "FAILED_PRECONDITION":
        return "failed_precondition"
    if status == "INVALID_ARGUMENT":
        return "invalid_argument"
    return status.lower()


def compute_retry_delay(policy: RetryPolicy, attempt: int) -> float:
    delay = policy.base_backoff_seconds * (
        policy.backoff_multiplier ** max(0, attempt - 1)
    )
    if policy.jitter > 0:
        factor = 1.0 + random.uniform(-policy.jitter, policy.jitter)
        delay = max(0.0, delay * factor)
    return delay


def remaining_budget(policy: RetryPolicy, start_time: float) -> float | None:
    if policy.deadline_seconds <= 0:
        return None
    remaining = policy.deadline_seconds - (time.monotonic() - start_time)
    return remaining


__all__ = [
    "RetryPolicyMap",
    "build_retry_policies",
    "compute_retry_delay",
    "map_materialization_error",
    "map_registration_error",
    "raise_mapped_materialization_error",
    "raise_mapped_registration_error",
    "remaining_budget",
    "retry_reason_bucket",
    "should_retry",
]
