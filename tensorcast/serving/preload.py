#  Copyright (c) 2026, TensorCast Team.

"""External preloaded serving binding authority schema."""

from __future__ import annotations

import inspect
import json
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterator

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import tensorcast as tc

_PRELOAD_MODES = {"disabled", "external"}
_READINESS_STATES = {
    "serving_reserved",
    "serving_local_ready",
    "serving_published_ready",
}


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_enum(value: Any, *, allowed: set[str], field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"{field_name} must be one of {sorted(allowed)}, got: {value!r}"
        )
    return normalized


class ExternalPreloadExpectedDigests(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_layout_hash: str
    tensor_schema_hash: str
    serving_build_digest: str
    resolved_spec_digest: str

    @field_validator(
        "target_layout_hash",
        "tensor_schema_hash",
        "serving_build_digest",
        "resolved_spec_digest",
        mode="before",
    )
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("expected digest fields must be non-empty")
        return normalized


class ExternalPreloadAuthority(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    member_ref: dict[str, Any]
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    binding_value_ref: dict[str, Any]
    reservation_capability: dict[str, Any]
    group_realization_acquire: dict[str, Any] | None = None
    local_serving_ref: str | None = None
    readiness: str
    verification_state: str = "local_only"
    serving_artifact_id: str | None = None
    trusted_reservation_bytes: int = Field(ge=0)
    expected: ExternalPreloadExpectedDigests

    @field_validator(
        "group_id",
        "daemon_id",
        "daemon_session_id",
        "device_uuid",
        mode="before",
    )
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("external preload authority text fields required")
        return normalized

    @field_validator(
        "local_serving_ref",
        "verification_state",
        "serving_artifact_id",
        mode="before",
    )
    @classmethod
    def _normalize_optional_fields(cls, value: Any) -> Any:
        return _normalize_optional_text(value)

    @field_validator("readiness", mode="before")
    @classmethod
    def _normalize_readiness(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            allowed=_READINESS_STATES,
            field_name="preload.authority.readiness",
        )

    @model_validator(mode="after")
    def _validate_published_ready(self) -> ExternalPreloadAuthority:
        if self.readiness == "serving_published_ready" and not self.serving_artifact_id:
            raise ValueError(
                "preload.authority.serving_artifact_id is required when "
                "readiness='serving_published_ready'"
            )
        return self


@dataclass(frozen=True)
class ParsedExternalPreloadAuthority:
    group_id: str
    local_serving_ref: str | None
    binding_value_ref: tc.BindingValueRef
    reservation_capability: tc.BindingReservationCapability
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    member: tc.ServingBindingMemberRef
    reservation_bytes: int
    expected: ExternalPreloadExpectedDigests
    readiness: str
    verification_state: str
    serving_artifact_id: str | None = None
    group_realization_acquire: tc.GroupRealizationAcquireRef | None = None


@dataclass(frozen=True)
class BindingPromotionResult:
    verification_state: str
    verification_job_id: str | None


class _PreloadLifecycleState:
    def __init__(
        self,
        *,
        client: Any,
        response: Any,
        runtime: Any,
        binding_value_ref: tc.BindingValueRef,
        binding_layout_id: str,
        member_ref: tc.ServingBindingMemberRef,
        reservation_bytes: int,
        lease_token: bytes,
    ) -> None:
        self.client = client
        self.response = response
        self.runtime = runtime
        self.binding_value_ref = binding_value_ref
        self.binding_layout_id = binding_layout_id
        self.member_ref = member_ref
        self.reservation_bytes = reservation_bytes
        self.lease_token = lease_token
        self.tensors: Mapping[str, torch.Tensor] | None = None
        self.state = "acquired"
        self._closed = False

    def release(self, *, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self.state = "closed"
        if not self.lease_token:
            return
        _release_lease_token(
            self.client,
            lease_token=self.lease_token,
            timeout_s=timeout_s,
        )


@dataclass(frozen=True)
class RuntimePreloadAttachmentHandle:
    """Runtime-owned close handle for a restored external preload binding."""

    tensors: Mapping[str, torch.Tensor]
    binding_layout_id: str
    binding_value_ref: tc.BindingValueRef
    member_ref: tc.ServingBindingMemberRef
    reservation_bytes: int
    _state: _PreloadLifecycleState

    def close(self) -> None:
        if self._state.state not in {"runtime_owned", "closed"}:
            raise RuntimeError(
                "RuntimePreloadAttachmentHandle.close() called before "
                "ownership was transferred to runtime"
            )
        self._state.release()

    def __enter__(self) -> RuntimePreloadAttachmentHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class AttachedPreloadBinding:
    """Restored tensors that have not yet been transferred to runtime."""

    tensors: Mapping[str, torch.Tensor]
    binding_layout_id: str
    binding_value_ref: tc.BindingValueRef
    member_ref: tc.ServingBindingMemberRef
    reservation_bytes: int
    _state: _PreloadLifecycleState

    def transfer_to_runtime(self) -> RuntimePreloadAttachmentHandle:
        if self._state.state != "restored":
            raise RuntimeError(
                "AttachedPreloadBinding.transfer_to_runtime() requires a "
                "restored attachment owner"
            )
        self._state.state = "runtime_owned"
        return RuntimePreloadAttachmentHandle(
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
                "AttachedPreloadBinding.close() requires a restored attachment owner"
            )
        self._state.release()

    def __enter__(self) -> AttachedPreloadBinding:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class BorrowedPreloadLease:
    """Single-owner acquire lease for an external preloaded binding value."""

    authority: ParsedExternalPreloadAuthority
    binding_value_ref: tc.BindingValueRef
    member_ref: tc.ServingBindingMemberRef
    reservation_bytes: int
    _state: _PreloadLifecycleState

    def restore(
        self,
        *,
        target_device: torch.device,
        restore_fn: Callable[..., Mapping[str, torch.Tensor]] | None = None,
    ) -> AttachedPreloadBinding:
        if self._state.state != "acquired":
            raise RuntimeError(
                "BorrowedPreloadLease.restore() requires an acquired lease"
            )
        device_index = torch.device(target_device).index
        if device_index is None:
            raise RuntimeError(
                "TensorCast external preload restore requires an explicit "
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
        return AttachedPreloadBinding(
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
                "BorrowedPreloadLease.close() requires an acquired or restored lease"
            )
        self._state.release()


def _payload_to_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be a JSON object") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(f"{field_name} must be a JSON object")
        return dict(parsed)
    raise ValueError(f"{field_name} must be a dict or JSON object")


def _model_validate(model_type: Any, value: Any, *, field_name: str) -> Any:
    payload = _payload_to_dict(value, field_name=field_name)
    try:
        return model_type.model_validate(payload)
    except Exception as exc:
        raise ValueError(
            f"{field_name} is invalid for TensorCast external preload: {exc}"
        ) from exc


def _validate_authority_consistency(
    authority: ParsedExternalPreloadAuthority,
) -> None:
    capability = authority.reservation_capability
    if capability.binding_value_ref != authority.binding_value_ref:
        raise ValueError(
            "preload.authority.reservation_capability.binding_value_ref "
            "must match preload.authority.binding_value_ref"
        )
    if capability.daemon_id != authority.daemon_id:
        raise ValueError("preload.authority.reservation_capability.daemon_id mismatch")
    if capability.daemon_session_id != authority.daemon_session_id:
        raise ValueError(
            "preload.authority.reservation_capability.daemon_session_id mismatch"
        )
    if capability.device_uuid != authority.device_uuid:
        raise ValueError(
            "preload.authority.reservation_capability.device_uuid mismatch"
        )
    if capability.member != authority.member:
        raise ValueError("preload.authority.reservation_capability.member mismatch")
    if capability.reservation_bytes != authority.reservation_bytes:
        raise ValueError(
            "preload.authority.reservation_capability.reservation_bytes "
            "must match preload.authority.trusted_reservation_bytes"
        )
    if authority.member.group_id is not None and authority.member.group_id != (
        authority.group_id
    ):
        raise ValueError(
            "preload.authority.member_ref.group_id must match "
            "preload.authority.group_id"
        )
    if (
        authority.readiness == "serving_published_ready"
        and not authority.serving_artifact_id
    ):
        raise ValueError(
            "preload.authority.serving_artifact_id is required when "
            "preload.authority.readiness='serving_published_ready'"
        )


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
    authority: ParsedExternalPreloadAuthority,
) -> tc.BindingValueRef:
    acquired_ref = _binding_value_ref_from_response(
        response,
        default=authority.binding_value_ref,
    )
    if acquired_ref != authority.binding_value_ref:
        raise RuntimeError(
            "TensorCast external preload acquired a different binding value "
            "than requested"
        )
    response_reservation = int(getattr(response, "reservation_bytes", 0) or 0)
    if response_reservation and response_reservation != authority.reservation_bytes:
        raise RuntimeError(
            "TensorCast external preload reservation byte mismatch: "
            f"expected={authority.reservation_bytes}, "
            f"actual={response_reservation}"
        )
    return acquired_ref


def _acquire_preload_response(
    client: Any,
    authority: ParsedExternalPreloadAuthority,
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
        "expected_serving_build_digest": authority.expected.serving_build_digest,
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


@contextmanager
def acquire_local_ready_preload_lease(
    *,
    local_serving_ref: str,
    expected_device_uuid: str,
    expected_member: tc.ServingBindingMemberRef,
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
) -> Iterator[BorrowedPreloadLease]:
    """Acquire an already-retained local-ready serving binding by local ref."""

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
        with suppress(Exception):
            _release_lease_token(client, lease_token=lease_token)
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
    expected = ExternalPreloadExpectedDigests(
        target_layout_hash=expected_target_layout_hash or "local-ready-direct",
        tensor_schema_hash=expected_tensor_schema_hash,
        serving_build_digest=expected_serving_build_digest,
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
            f"{expected.serving_build_digest}"
        ),
    )
    authority = ParsedExternalPreloadAuthority(
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
        readiness="serving_local_ready",
        verification_state="local_only",
        serving_artifact_id=serving_artifact_id,
    )
    state = _PreloadLifecycleState(
        client=client,
        response=response,
        runtime=runtime,
        binding_value_ref=binding_value_ref,
        binding_layout_id=binding_value_ref.binding_layout_id,
        member_ref=expected_member,
        reservation_bytes=reservation_bytes,
        lease_token=lease_token,
    )
    lease = BorrowedPreloadLease(
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
def acquire_preload_lease(
    authority: ParsedExternalPreloadAuthority,
    *,
    caller_pid: int | None = None,
    runtime: Any | None = None,
    client: Any | None = None,
    timeout_s: float | None = None,
) -> Iterator[BorrowedPreloadLease]:
    """Acquire a preloaded binding and yield the sole lease owner.

    The yielded lease owns the raw placement lease token internally. If restore
    or a later framework attach step fails, closing the current owner releases
    the token without exposing it to the framework integration.
    """

    if runtime is None:
        from tensorcast.api.store import get_runtime_context

        runtime = get_runtime_context()
    if client is None:
        client = runtime.ensure_client()
    response = _acquire_preload_response(
        client,
        authority,
        caller_pid=caller_pid or os.getpid(),
        timeout_s=timeout_s,
    )
    lease_token = _lease_token_from_response(response)
    try:
        acquired_ref = _validate_acquire_response(response, authority)
    except Exception:
        with suppress(Exception):
            _release_lease_token(client, lease_token=lease_token)
        raise
    state = _PreloadLifecycleState(
        client=client,
        response=response,
        runtime=runtime,
        binding_value_ref=acquired_ref,
        binding_layout_id=authority.binding_value_ref.binding_layout_id,
        member_ref=authority.member,
        reservation_bytes=authority.reservation_bytes,
        lease_token=lease_token,
    )
    lease = BorrowedPreloadLease(
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


def parse_external_preload_authority(
    extra: Mapping[str, Any] | Any,
) -> ParsedExternalPreloadAuthority:
    from tensorcast.serving.config import ServingConfig

    config = (
        extra if isinstance(extra, ServingConfig) else ServingConfig.from_mapping(extra)
    )
    authority_config = config.preload.authority
    if config.preload.mode != "external" or authority_config is None:
        raise ValueError(
            "TensorCast external preload authority requires "
            "preload.mode='external' and preload.authority"
        )

    binding_value_ref = _model_validate(
        tc.BindingValueRef,
        authority_config.binding_value_ref,
        field_name="preload.authority.binding_value_ref",
    )
    member = _model_validate(
        tc.ServingBindingMemberRef,
        authority_config.member_ref,
        field_name="preload.authority.member_ref",
    )
    capability_payload = _payload_to_dict(
        authority_config.reservation_capability,
        field_name="preload.authority.reservation_capability",
    )
    capability_payload.setdefault(
        "binding_value_ref", binding_value_ref.model_dump(mode="python")
    )
    capability_payload.setdefault("member", member.model_dump(mode="python"))
    reservation_capability = _model_validate(
        tc.BindingReservationCapability,
        capability_payload,
        field_name="preload.authority.reservation_capability",
    )
    group_realization_acquire = None
    if authority_config.group_realization_acquire is not None:
        group_realization_acquire = _model_validate(
            tc.GroupRealizationAcquireRef,
            authority_config.group_realization_acquire,
            field_name="preload.authority.group_realization_acquire",
        )

    authority = ParsedExternalPreloadAuthority(
        group_id=authority_config.group_id,
        local_serving_ref=authority_config.local_serving_ref,
        binding_value_ref=binding_value_ref,
        reservation_capability=reservation_capability,
        daemon_id=authority_config.daemon_id,
        daemon_session_id=authority_config.daemon_session_id,
        device_uuid=authority_config.device_uuid,
        member=member,
        reservation_bytes=int(authority_config.trusted_reservation_bytes),
        expected=authority_config.expected,
        readiness=authority_config.readiness,
        verification_state=authority_config.verification_state or "local_only",
        serving_artifact_id=authority_config.serving_artifact_id,
        group_realization_acquire=group_realization_acquire,
    )
    _validate_authority_consistency(authority)
    return authority


def external_preload_mode(extra: Mapping[str, Any] | None) -> str:
    if extra is None or not isinstance(extra, Mapping):
        return "disabled"
    from tensorcast.serving.config import ServingConfig

    return ServingConfig.from_mapping(extra).preload.mode


@contextmanager
def acquire_retained_serving_binding(
    *,
    authority: ParsedExternalPreloadAuthority | None = None,
    local_serving_ref: str | None = None,
    target_device: torch.device | str | None = None,
    expected_member: tc.ServingBindingMemberRef | None = None,
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
) -> Iterator[BorrowedPreloadLease]:
    if authority is not None:
        if local_serving_ref is not None:
            raise ValueError(
                "acquire_retained_serving_binding accepts either authority "
                "or local_serving_ref, not both"
            )
        with acquire_preload_lease(
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
            "acquire_retained_serving_binding requires authority or local_serving_ref"
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

    with acquire_local_ready_preload_lease(
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


def external_preload_trusted_reservation_bytes(load_config_or_extra: Any) -> int:
    extra = getattr(
        load_config_or_extra, "model_loader_extra_config", load_config_or_extra
    )
    if extra is None or not isinstance(extra, Mapping):
        return 0
    if external_preload_mode(extra) != "external":
        return 0
    return parse_external_preload_authority(extra).reservation_bytes


def external_preload_extra_from_prefetched_binding(
    *,
    prefetched: tc.PrefetchedServingBinding,
    target: tc.ServingBindingTarget,
    expected_member: tc.ServingBindingMemberRef | None = None,
) -> dict[str, Any]:
    member = prefetched.member
    if expected_member is not None and member != expected_member:
        raise ValueError(
            "Prefetched serving binding member does not match expected "
            f"placement: prefetched={member}, expected={expected_member}"
        )
    authority: dict[str, Any] = {
        "group_id": member.group_id or "",
        "member_ref": _model_dump(member),
        "daemon_id": prefetched.daemon_id,
        "daemon_session_id": prefetched.daemon_session_id,
        "device_uuid": prefetched.device_uuid,
        "binding_value_ref": _model_dump(prefetched.binding_value_ref),
        "reservation_capability": _model_dump(prefetched.reservation_capability),
        "local_serving_ref": prefetched.local_serving_ref,
        "readiness": str(getattr(prefetched.readiness, "value", prefetched.readiness)),
        "verification_state": str(
            getattr(
                prefetched.verification_state,
                "value",
                prefetched.verification_state,
            )
        ),
        "serving_artifact_id": prefetched.serving_artifact_id,
        "trusted_reservation_bytes": prefetched.reservation_bytes,
        "expected": {
            "target_layout_hash": target.resolved_layout.target_layout_hash,
            "tensor_schema_hash": target.resolved_layout.tensor_schema_hash,
            "serving_build_digest": target.serving_build_digest,
            "resolved_spec_digest": target.resolved_layout.spec_digest,
        },
    }
    if prefetched.group_realization_acquire is not None:
        authority["group_realization_acquire"] = _model_dump(
            prefetched.group_realization_acquire
        )
    return {
        "preload": {
            "mode": "external",
            "authority": authority,
        },
    }


def external_preload_extra_json(
    *,
    prefetched: tc.PrefetchedServingBinding,
    target: tc.ServingBindingTarget,
    expected_member: tc.ServingBindingMemberRef | None = None,
) -> str:
    return json.dumps(
        external_preload_extra_from_prefetched_binding(
            prefetched=prefetched,
            target=target,
            expected_member=expected_member,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


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


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


class PreloadSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "disabled"
    authority: ExternalPreloadAuthority | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "disabled"
        return _normalize_enum(
            value,
            allowed=_PRELOAD_MODES,
            field_name="preload.mode",
        )

    @model_validator(mode="after")
    def _validate_authority(self) -> PreloadSettings:
        if self.mode == "external" and self.authority is None:
            raise ValueError(
                "preload.authority is required when preload.mode='external'"
            )
        if self.mode != "external" and self.authority is not None:
            raise ValueError(
                "preload.authority is only valid when preload.mode='external'"
            )
        return self
