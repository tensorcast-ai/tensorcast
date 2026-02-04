#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import CancelledError
from typing import Mapping, NoReturn

import grpc

from tensorcast.api._errors import DeviceMismatch, TensorCastError
from tensorcast.api.store.types import ArtifactError, RetryPolicy
from tensorcast.error_reporting import debug_errors_enabled, debug_errors_hint

RetryPolicyMap = Mapping[str, RetryPolicy]


def _append_debug_hint(message: str) -> str:
    if debug_errors_enabled():
        return message
    return f"{message}\nDebug: {debug_errors_hint()}"


def _append_hint(message: str, hint: str) -> str:
    return _append_debug_hint(f"{message}\nHint: {hint}")


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
        lowered = mapped.lower()
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
            )
        retryable = status_code in {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.ABORTED,
        }
        if not details:
            mapped = _append_debug_hint(mapped)
        return ArtifactError(mapped, status_code=status_name, retryable=retryable)
    lowered = message.lower()
    if "tensor index not found" in lowered:
        hint = (
            "Ensure the directory contains tensor_index.json, tensor_index.cbor, or *.safetensors files. "
            "If this is a raw model folder, run a TensorCast save path "
            "(e.g., tensorcast.save_* examples) to generate the index."
        )
        message = _append_hint(message, hint)
        return ArtifactError(message, status_code="NOT_FOUND", retryable=False)
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
        )
    if isinstance(exc, RuntimeError):
        if "not found" in lowered:
            return ArtifactError(message, status_code="NOT_FOUND", retryable=False)
        if "unavailable" in lowered or "not available" in lowered:
            return ArtifactError(message, status_code="UNAVAILABLE", retryable=True)
    return ArtifactError(message, status_code="UNKNOWN", retryable=False)


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
    if error.status_code not in {"UNAVAILABLE", "DEADLINE_EXCEEDED", "ABORTED"}:
        return False
    if policy.deadline_seconds > 0:
        elapsed = time.monotonic() - start_time
        if elapsed >= policy.deadline_seconds:
            return False
    return True


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
    "should_retry",
]
