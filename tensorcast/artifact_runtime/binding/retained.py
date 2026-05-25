#  Copyright (c) 2026, TensorCast Team.
"""Retained runtime binding authority and acquire helpers."""

from __future__ import annotations

import inspect
import logging
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterator

import torch

import tensorcast as tc
from tensorcast.api.store.realization_kernel import (
    RealizationReleaseContract,
    envelope_for_runtime_attachment,
    release_contract_for,
)
from tensorcast.artifact_runtime.config import (
    RetainedBindingAcquireSettings as _RetainedBindingAcquireSettings,
)
from tensorcast.retained_realization import (
    retained_realization_claim_mode,
)
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
    RetainedRealizationExpectedDigests,
)

RetainedBindingAcquireSettings = _RetainedBindingAcquireSettings
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BindingPromotionResult:
    verification_state: str
    verification_job_id: str | None


class _RetainedBindingLifecycleState:
    def __init__(
        self,
        *,
        client: Any,
        response: Any,
        runtime: Any,
        authority: ParsedRetainedRealizationAuthority,
        binding_value_ref: tc.BindingValueRef,
        binding_layout_id: str,
        member_ref: tc.RuntimeBindingMemberRef,
        reservation_bytes: int,
        lease_token: bytes,
    ) -> None:
        self.client = client
        self.response = response
        self.runtime = runtime
        self.authority = authority
        self.binding_value_ref = binding_value_ref
        self.binding_layout_id = binding_layout_id
        self.member_ref = member_ref
        self.reservation_bytes = reservation_bytes
        self.lease_token = lease_token
        self.tensors: Mapping[str, torch.Tensor] | None = None
        self.state = "acquired"
        self._closed = False
        self._release_timeout_s: float | None = None
        self.release_contract = release_contract_for(
            envelope_for_runtime_attachment(
                {},
                retained=True,
                reservation_bytes=reservation_bytes,
            ),
            self._release_placement_lease,
        )

    def _release_placement_lease(self) -> None:
        if not self.lease_token:
            return
        _release_lease_token(
            self.client,
            lease_token=self.lease_token,
            timeout_s=self._release_timeout_s,
        )

    def release(self, *, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self.state = "closed"
        prior_timeout_s = self._release_timeout_s
        self._release_timeout_s = timeout_s
        try:
            self.release_contract.release()
        finally:
            self._release_timeout_s = prior_timeout_s


@dataclass(frozen=True)
class RuntimeRetainedBindingAttachmentHandle:
    """Runtime-owned close handle for a restored retained binding."""

    tensors: Mapping[str, torch.Tensor]
    binding_layout_id: str
    binding_value_ref: tc.BindingValueRef
    member_ref: tc.RuntimeBindingMemberRef
    reservation_bytes: int
    _state: _RetainedBindingLifecycleState

    @property
    def authority(self) -> ParsedRetainedRealizationAuthority:
        return self._state.authority

    @property
    def release_contract(self) -> RealizationReleaseContract:
        return self._state.release_contract

    def status(self) -> str:
        return self._state.state

    def debug_status(self) -> dict[str, Any]:
        return _retained_binding_debug_status(self._state)

    def close(self) -> None:
        if self._state.state not in {"runtime_owned", "closed"}:
            raise RuntimeError(
                "RuntimeRetainedBindingAttachmentHandle.close() called before "
                "ownership was transferred to runtime"
            )
        self._state.release()

    def __enter__(self) -> RuntimeRetainedBindingAttachmentHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class AttachedRetainedBinding:
    """Restored tensors that have not yet been transferred to runtime."""

    tensors: Mapping[str, torch.Tensor]
    binding_layout_id: str
    binding_value_ref: tc.BindingValueRef
    member_ref: tc.RuntimeBindingMemberRef
    reservation_bytes: int
    _state: _RetainedBindingLifecycleState

    @property
    def authority(self) -> ParsedRetainedRealizationAuthority:
        return self._state.authority

    @property
    def release_contract(self) -> RealizationReleaseContract:
        return self._state.release_contract

    def status(self) -> str:
        return self._state.state

    def debug_status(self) -> dict[str, Any]:
        return _retained_binding_debug_status(self._state)

    def transfer_to_runtime(self) -> RuntimeRetainedBindingAttachmentHandle:
        if self._state.state != "restored":
            raise RuntimeError(
                "AttachedRetainedBinding.transfer_to_runtime() requires a "
                "restored attachment owner"
            )
        self._state.state = "runtime_owned"
        return RuntimeRetainedBindingAttachmentHandle(
            tensors=self.tensors,
            binding_layout_id=self.binding_layout_id,
            binding_value_ref=self.binding_value_ref,
            member_ref=self.member_ref,
            reservation_bytes=self.reservation_bytes,
            _state=self._state,
        )

    def close(self) -> None:
        if self._state.state in {"runtime_owned", "closed"}:
            return
        if self._state.state != "restored":
            raise RuntimeError(
                "AttachedRetainedBinding.close() requires a restored attachment owner"
            )
        self._state.release()

    def __enter__(self) -> AttachedRetainedBinding:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class BorrowedRetainedBindingLease:
    """Single-owner acquire lease for a retained binding value."""

    authority: ParsedRetainedRealizationAuthority
    binding_value_ref: tc.BindingValueRef
    member_ref: tc.RuntimeBindingMemberRef
    reservation_bytes: int
    _state: _RetainedBindingLifecycleState

    @property
    def release_contract(self) -> RealizationReleaseContract:
        return self._state.release_contract

    def status(self) -> str:
        return self._state.state

    def debug_status(self) -> dict[str, Any]:
        return _retained_binding_debug_status(self._state)

    def restore(
        self,
        *,
        target_device: torch.device,
        restore_fn: Callable[..., Mapping[str, torch.Tensor]] | None = None,
    ) -> AttachedRetainedBinding:
        if self._state.state != "acquired":
            raise RuntimeError(
                "BorrowedRetainedBindingLease.restore() requires an acquired lease"
            )
        device_index = torch.device(target_device).index
        if device_index is None:
            raise RuntimeError(
                "TensorCast retained binding restore requires an explicit "
                "CUDA device index"
            )
        if restore_fn is None:
            from tensorcast.api.store import restore_owned_binding_tensors

            restore_fn = restore_owned_binding_tensors
        try:
            tensors = dict(
                restore_fn(
                    response=self._state.response,
                    runtime=self._state.runtime,
                    device_id=int(device_index),
                )
            )
        except Exception:
            self._state.release()
            raise
        self._state.tensors = tensors
        self._state.state = "restored"
        return AttachedRetainedBinding(
            tensors=tensors,
            binding_layout_id=self._state.binding_layout_id,
            binding_value_ref=self._state.binding_value_ref,
            member_ref=self._state.member_ref,
            reservation_bytes=self._state.reservation_bytes,
            _state=self._state,
        )

    def close(self) -> None:
        if self._state.state in {"runtime_owned", "closed"}:
            return
        if self._state.state not in {"acquired", "restored"}:
            raise RuntimeError(
                "BorrowedRetainedBindingLease.close() requires an acquired or "
                "restored lease"
            )
        self._state.release()


def _validate_authority_is_attachable(
    authority: ParsedRetainedRealizationAuthority,
) -> None:
    if authority.readiness == "runtime_reserved":
        raise ValueError(
            "retained_binding_acquire.authority.readiness='runtime_reserved' "
            "is not attachable"
        )
    group_acquire = authority.group_realization_acquire
    if group_acquire is not None and not group_acquire.wait_for_publish:
        raise ValueError(
            "retained_binding_acquire.authority.group_realization_acquire must "
            "wait for group publish before attach"
        )
    expires_at_ms = authority.reservation_capability.expires_at_ms
    if expires_at_ms is not None and int(expires_at_ms) <= _current_epoch_ms():
        raise ValueError(
            "retained_binding_acquire.authority.reservation_capability has expired"
        )


def _current_epoch_ms() -> int:
    return int(time.time() * 1000)


def _retained_binding_debug_status(
    state: _RetainedBindingLifecycleState,
) -> dict[str, Any]:
    authority = state.authority
    capability = authority.reservation_capability
    payload: dict[str, Any] = {
        "state": state.state,
        "binding_value_ref": state.binding_value_ref.model_dump(mode="python"),
        "binding_layout_id": state.binding_layout_id,
        "member": state.member_ref.model_dump(mode="python"),
        "group_id": authority.group_id,
        "local_serving_ref": authority.local_serving_ref,
        "daemon_id": authority.daemon_id,
        "daemon_session_id": authority.daemon_session_id,
        "device_uuid": authority.device_uuid,
        "readiness": authority.readiness,
        "verification_state": authority.verification_state,
        "serving_artifact_id": authority.serving_artifact_id,
        "reservation_bytes": state.reservation_bytes,
        "reservation_capability_id": capability.capability_id,
        "reservation_scope_digest": capability.scope_digest,
        "reservation_expires_at_ms": capability.expires_at_ms,
        "lease_token_present": bool(state.lease_token),
        "release_policy": state.release_contract.release_policy,
        "release_strictness": state.release_contract.release_strictness,
        "released": state.release_contract.released,
    }
    if authority.group_realization_acquire is not None:
        payload["group_realization_acquire"] = (
            authority.group_realization_acquire.model_dump(mode="python")
        )
    return payload


def _binding_value_ref_from_response(
    response: Any,
    *,
    default: tc.BindingValueRef,
) -> tc.BindingValueRef:
    has_field = getattr(response, "HasField", None)
    for field_name in ("current_value", "acquired_value"):
        if callable(has_field) and not has_field(field_name):
            continue
        value = getattr(response, field_name, None)
        if value is None:
            continue
        binding_id = str(getattr(value, "binding_id", "") or "")
        binding_layout_id = str(getattr(value, "binding_layout_id", "") or "")
        binding_value_id = str(getattr(value, "binding_value_id", "") or "")
        if binding_id and binding_layout_id and binding_value_id:
            return tc.BindingValueRef(
                binding_id=binding_id,
                binding_layout_id=binding_layout_id,
                binding_value_id=binding_value_id,
                seal_generation=int(getattr(value, "seal_generation", 0) or 0),
            )
    return default


def _lease_token_from_response(response: Any) -> bytes:
    mem_handle = getattr(response, "mem_handle", None)
    if mem_handle is not None:
        lease_token = bytes(getattr(mem_handle, "lease_token", b"") or b"")
        if lease_token:
            return lease_token
    return bytes(getattr(response, "lease_token", b"") or b"")


def _validate_acquire_response(
    response: Any,
    authority: ParsedRetainedRealizationAuthority,
) -> tc.BindingValueRef:
    acquired_ref = _binding_value_ref_from_response(
        response,
        default=authority.binding_value_ref,
    )
    if acquired_ref != authority.binding_value_ref:
        raise RuntimeError(
            "TensorCast retained binding acquire returned a different binding value "
            "than requested"
        )
    response_reservation = int(getattr(response, "reservation_bytes", 0) or 0)
    if response_reservation and response_reservation != authority.reservation_bytes:
        raise RuntimeError(
            "TensorCast retained binding acquire reservation byte mismatch: "
            f"expected={authority.reservation_bytes}, "
            f"actual={response_reservation}"
        )
    return acquired_ref


def _acquire_retained_binding_response(
    client: Any,
    authority: ParsedRetainedRealizationAuthority,
    *,
    caller_pid: int,
    timeout_s: float | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "binding_value_ref": authority.binding_value_ref,
        "reservation_capability": authority.reservation_capability,
        "expected_device_uuid": authority.device_uuid,
        "expected_target_layout_hash": authority.expected.target_layout_hash,
        "expected_tensor_schema_hash": authority.expected.tensor_schema_hash,
        "expected_serving_build_digest": authority.expected.runtime_build_digest,
        "expected_daemon_id": authority.daemon_id,
        "expected_daemon_session_id": authority.daemon_session_id,
        "expected_member": authority.member,
        "local_serving_ref": authority.local_serving_ref,
        "caller_pid": caller_pid,
    }
    if authority.group_realization_acquire is not None:
        kwargs["group_realization_acquire"] = authority.group_realization_acquire
    if timeout_s is not None:
        kwargs["timeout_s"] = timeout_s
    return client.acquire_binding_value(**kwargs)


def _release_lease_token(
    client: Any,
    *,
    lease_token: bytes,
    timeout_s: float | None = None,
) -> None:
    if not lease_token:
        return
    release = getattr(client, "release_placement_lease", None)
    if not callable(release):
        return
    if timeout_s is None:
        release(lease_token=lease_token)
    else:
        release(lease_token=lease_token, timeout_s=timeout_s)


def _release_lease_token_after_acquire_failure(
    client: Any,
    *,
    lease_token: bytes,
) -> None:
    try:
        _release_lease_token(client, lease_token=lease_token)
    except Exception:
        _LOGGER.exception(
            "Failed to release retained runtime binding lease after acquire failure",
        )


@contextmanager
def acquire_local_ready_retained_binding_lease(
    *,
    local_serving_ref: str,
    expected_device_uuid: str,
    expected_member: tc.RuntimeBindingMemberRef,
    expected_tensor_schema_hash: str,
    expected_serving_build_digest: str,
    expected_target_layout_hash: str | None = None,
    expected_daemon_id: str | None = None,
    expected_daemon_session_id: str | None = None,
    serving_artifact_id: str | None = None,
    caller_pid: int | None = None,
    runtime: Any | None = None,
    client: Any | None = None,
    timeout_s: float | None = None,
) -> Iterator[BorrowedRetainedBindingLease]:
    """Acquire an already-retained local-ready runtime binding by local ref."""

    if runtime is None:
        from tensorcast.api.store import get_runtime_context

        runtime = get_runtime_context()
    if client is None:
        client = runtime.ensure_client()
    if client is None:
        raise RuntimeError("TensorCast runtime did not provide a store daemon client")
    response = client.acquire_binding_value_by_local_ref(
        local_serving_ref=local_serving_ref,
        expected_device_uuid=expected_device_uuid,
        expected_member=expected_member,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        expected_serving_build_digest=expected_serving_build_digest,
        expected_target_layout_hash=expected_target_layout_hash,
        expected_daemon_id=expected_daemon_id,
        expected_daemon_session_id=expected_daemon_session_id,
        caller_pid=caller_pid or os.getpid(),
        timeout_s=30.0 if timeout_s is None else float(timeout_s),
    )
    lease_token = _lease_token_from_response(response)
    missing_ref = tc.BindingValueRef(
        binding_id="missing",
        binding_layout_id="missing",
        binding_value_id="missing",
        seal_generation=0,
    )
    binding_value_ref = _binding_value_ref_from_response(
        response,
        default=missing_ref,
    )
    if binding_value_ref == missing_ref:
        _release_lease_token_after_acquire_failure(client, lease_token=lease_token)
        raise RuntimeError(
            "TensorCast local-ready acquire did not return a binding value"
        )
    reservation_bytes = int(getattr(response, "reservation_bytes", 0) or 0)
    daemon_id = (
        expected_daemon_id
        or getattr(runtime, "daemon_id", None)
        or getattr(runtime, "daemon_endpoint", "")
        or "local-daemon"
    )
    daemon_session_id = (
        expected_daemon_session_id
        or getattr(runtime, "session_id", "")
        or "local-session"
    )
    expected = RetainedRealizationExpectedDigests(
        target_layout_hash=expected_target_layout_hash or "local-ready-direct",
        tensor_schema_hash=expected_tensor_schema_hash,
        runtime_build_digest=expected_serving_build_digest,
        resolved_spec_digest="local-ready-direct",
    )
    reservation_capability = tc.BindingReservationCapability(
        capability_id=(
            f"local-ready-direct:{binding_value_ref.binding_id}:"
            f"{binding_value_ref.binding_value_id}"
        ),
        binding_value_ref=binding_value_ref,
        daemon_id=str(daemon_id),
        daemon_session_id=str(daemon_session_id),
        device_uuid=str(expected_device_uuid),
        member=expected_member,
        reservation_bytes=reservation_bytes,
        scope_digest=(
            f"{expected.target_layout_hash}:"
            f"{expected.tensor_schema_hash}:"
            f"{expected.runtime_build_digest}"
        ),
    )
    authority = ParsedRetainedRealizationAuthority(
        group_id=expected_member.group_id or "",
        local_serving_ref=local_serving_ref,
        binding_value_ref=binding_value_ref,
        reservation_capability=reservation_capability,
        daemon_id=str(daemon_id),
        daemon_session_id=str(daemon_session_id),
        device_uuid=str(expected_device_uuid),
        member=expected_member,
        reservation_bytes=reservation_bytes,
        expected=expected,
        readiness="runtime_local_ready",
        verification_state="local_only",
        serving_artifact_id=serving_artifact_id,
    )
    state = _RetainedBindingLifecycleState(
        client=client,
        response=response,
        runtime=runtime,
        authority=authority,
        binding_value_ref=binding_value_ref,
        binding_layout_id=binding_value_ref.binding_layout_id,
        member_ref=expected_member,
        reservation_bytes=reservation_bytes,
        lease_token=lease_token,
    )
    lease = BorrowedRetainedBindingLease(
        authority=authority,
        binding_value_ref=binding_value_ref,
        member_ref=expected_member,
        reservation_bytes=reservation_bytes,
        _state=state,
    )
    try:
        yield lease
    finally:
        lease.close()


@contextmanager
def acquire_retained_binding_lease(
    authority: ParsedRetainedRealizationAuthority,
    *,
    caller_pid: int | None = None,
    runtime: Any | None = None,
    client: Any | None = None,
    timeout_s: float | None = None,
) -> Iterator[BorrowedRetainedBindingLease]:
    """Acquire a retained binding and yield the sole lease owner.

    The yielded lease owns the raw placement lease token internally. If restore
    or a later framework attach step fails, closing the current owner releases
    the token without exposing it to the framework integration.
    """

    if runtime is None:
        from tensorcast.api.store import get_runtime_context

        runtime = get_runtime_context()
    if client is None:
        client = runtime.ensure_client()
    _validate_authority_is_attachable(authority)
    response = _acquire_retained_binding_response(
        client,
        authority,
        caller_pid=caller_pid or os.getpid(),
        timeout_s=timeout_s,
    )
    lease_token = _lease_token_from_response(response)
    try:
        acquired_ref = _validate_acquire_response(response, authority)
    except Exception:
        _release_lease_token_after_acquire_failure(client, lease_token=lease_token)
        raise
    state = _RetainedBindingLifecycleState(
        client=client,
        response=response,
        runtime=runtime,
        authority=authority,
        binding_value_ref=acquired_ref,
        binding_layout_id=authority.binding_value_ref.binding_layout_id,
        member_ref=authority.member,
        reservation_bytes=authority.reservation_bytes,
        lease_token=lease_token,
    )
    lease = BorrowedRetainedBindingLease(
        authority=authority,
        binding_value_ref=acquired_ref,
        member_ref=authority.member,
        reservation_bytes=authority.reservation_bytes,
        _state=state,
    )
    try:
        yield lease
    finally:
        lease.close()


def retained_binding_acquire_mode(extra: Mapping[str, Any] | None) -> str:
    return retained_realization_claim_mode(extra)


@contextmanager
def acquire_retained_binding(
    *,
    authority: ParsedRetainedRealizationAuthority | None = None,
    local_serving_ref: str | None = None,
    target_device: torch.device | str | None = None,
    expected_member: tc.RuntimeBindingMemberRef | None = None,
    expected_tensor_schema_hash: str | None = None,
    expected_serving_build_digest: str | None = None,
    expected_target_layout_hash: str | None = None,
    expected_daemon_id: str | None = None,
    expected_daemon_session_id: str | None = None,
    serving_artifact_id: str | None = None,
    caller_pid: int | None = None,
    runtime: Any | None = None,
    client: Any | None = None,
    timeout_s: float | None = None,
) -> Iterator[BorrowedRetainedBindingLease]:
    if authority is not None:
        if local_serving_ref is not None:
            raise ValueError(
                "acquire_retained_binding accepts either authority "
                "or local_serving_ref, not both"
            )
        if expected_member is not None and authority.member != expected_member:
            raise RuntimeError(
                "TensorCast retained binding authority does not match "
                "the expected runtime placement: "
                f"authority={authority.member}, expected={expected_member}"
            )
        with acquire_retained_binding_lease(
            authority,
            caller_pid=caller_pid,
            runtime=runtime,
            client=client,
            timeout_s=timeout_s,
        ) as lease:
            yield lease
        return

    if local_serving_ref is None:
        raise ValueError(
            "acquire_retained_binding requires authority or local_serving_ref"
        )
    if target_device is None or expected_member is None:
        raise ValueError(
            "local-ready retained acquire requires target_device and expected_member"
        )
    if not expected_tensor_schema_hash or not expected_serving_build_digest:
        raise ValueError(
            "local-ready retained acquire requires expected tensor schema "
            "hash and serving build digest"
        )

    device = torch.device(target_device)
    device_index = device.index
    if device_index is None:
        raise RuntimeError(
            "TensorCast local-ready retained acquire requires an explicit "
            "CUDA device index"
        )
    if runtime is None:
        from tensorcast.api.store import get_runtime_context

        runtime = get_runtime_context()
    from tensorcast.api.store import device_uuid_for

    with acquire_local_ready_retained_binding_lease(
        local_serving_ref=local_serving_ref,
        expected_device_uuid=device_uuid_for(int(device_index)),
        expected_member=expected_member,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        expected_serving_build_digest=expected_serving_build_digest,
        expected_target_layout_hash=expected_target_layout_hash,
        expected_daemon_id=expected_daemon_id,
        expected_daemon_session_id=expected_daemon_session_id,
        serving_artifact_id=serving_artifact_id,
        caller_pid=caller_pid,
        runtime=runtime,
        client=client,
        timeout_s=timeout_s,
    ) as lease:
        yield lease


def promote_current_value_and_wait(
    *,
    binding: Any,
    current_value: Any,
    timeout_s: float = 600.0,
    poll_interval_s: float = 0.25,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    state_name_fn: Callable[[Any], str] | None = None,
    start_scope: Callable[[], ContextManager[Any]] | None = None,
    poll_scope: Callable[[], ContextManager[Any]] | None = None,
    on_start: Callable[..., None] | None = None,
    on_poll: Callable[..., None] | None = None,
) -> BindingPromotionResult:
    binding_value_id = str(getattr(current_value, "binding_value_id", "") or "")
    if not binding_value_id:
        raise RuntimeError(
            "TensorCast promotion requires a binding_value_id before "
            "serving publication"
        )
    with _promotion_scope(start_scope) as scope_payload:
        status = binding.start_promote_current_value(binding_value_id=binding_value_id)
        _notify_promotion_observer(on_start, status, scope_payload)
    verification_job_id = str(getattr(status, "verification_job_id", "") or "") or None
    deadline = clock() + float(timeout_s)
    while True:
        state_name = _promotion_state_name(status, state_name_fn=state_name_fn)
        if state_name in {"succeeded", "failed", "canceled"}:
            break
        if clock() >= deadline:
            raise TimeoutError("TensorCast promotion timed out")
        sleep_fn(float(poll_interval_s))
        with _promotion_scope(poll_scope) as scope_payload:
            status = binding.get_promotion_status(
                verification_job_id=verification_job_id,
                binding_value_id=binding_value_id,
            )
            _notify_promotion_observer(on_poll, status, scope_payload)
    state_name = _promotion_state_name(status, state_name_fn=state_name_fn)
    if state_name != "succeeded":
        failure = str(getattr(status, "failure_reason", "") or state_name)
        raise RuntimeError(f"TensorCast promotion failed: {failure}")
    return BindingPromotionResult(
        verification_state="verified",
        verification_job_id=verification_job_id,
    )


def _promotion_scope(
    factory: Callable[[], ContextManager[Any]] | None,
) -> ContextManager[Any]:
    if factory is None:
        return nullcontext(None)
    return factory()


def _notify_promotion_observer(
    callback: Callable[..., None] | None,
    status: Any,
    scope_payload: Any,
) -> None:
    if callback is None:
        return
    if _promotion_observer_accepts_scope(callback):
        callback(status, scope_payload)
    else:
        callback(status)


def _promotion_observer_accepts_scope(callback: Callable[..., None]) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    parameters = tuple(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return True
    positional = tuple(
        param
        for param in parameters
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return len(positional) >= 2


def _promotion_state_name(
    status: Any,
    *,
    state_name_fn: Callable[[Any], str] | None,
) -> str:
    if state_name_fn is not None:
        return str(state_name_fn(status))
    state = getattr(status, "state", status)
    value = getattr(state, "value", state)
    name = getattr(state, "name", None)
    if isinstance(value, str):
        return value.strip().lower()
    if name is not None:
        return str(name).strip().lower()
    if isinstance(value, int):
        mapped = tc.BindingPromotionStatusState.from_proto(int(value))
        if mapped is not None:
            return str(mapped.value).strip().lower()
    return str(value).strip().lower()
