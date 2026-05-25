#  Copyright (c) 2026, TensorCast Team.

"""Runtime-owned artifact replica publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationHandle,
    ArtifactRealizationSpec,
    RealizationReleaseContract,
    RealizationResourceEnvelope,
    RealizationTargetPlan,
    artifact_realization_report_to_dict,
    emit_artifact_realization_profile_event,
    envelope_for_publication,
    release_contract_for,
    report_for_publication,
)
from tensorcast.serving.errors import ReplicaPublicationError
from tensorcast.serving.runtime_attachment import (
    RuntimeAttachment,
    RuntimeBindingState,
)
from tensorcast.serving.runtime_view import (
    BindingValueRefProjection,
    PublishedReplicaProjection,
)

_ACTIVE_PUBLICATION_STATES = {"publishing", "published", "retiring"}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nonempty_binding_value_ref(
    value: object | None,
) -> BindingValueRefProjection | None:
    projection = BindingValueRefProjection.from_value(value)
    if projection is None:
        return None
    if not (
        projection.binding_id
        and projection.binding_layout_id
        and projection.binding_value_id
    ):
        return None
    return projection


def _binding_value_refs_match(
    expected: BindingValueRefProjection | None,
    actual: BindingValueRefProjection | None,
) -> bool:
    if expected is None or actual is None:
        return True
    return expected.to_dict() == actual.to_dict()


def publication_generation(attachment: RuntimeAttachment) -> str:
    weight_version = attachment.view.endpoint.weight_version
    binding_value_ref = weight_version.binding_value_ref
    if binding_value_ref is not None:
        payload: object = binding_value_ref.to_dict()
    else:
        payload = {
            "serving_artifact_ref": weight_version.serving_artifact_ref,
            "local_serving_ref": weight_version.local_serving_ref,
            "representation_contract_hash": weight_version.representation_contract_hash,
            "tensor_schema_hash": weight_version.tensor_schema_hash,
            "attachment_id": id(attachment),
        }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def state_publication_binding(state: RuntimeBindingState) -> object | None:
    return state.binding or state.ownership_handle


def _nested_attr(value: object | None, *names: str) -> object | None:
    current = value
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def _first_attr(value: object | None, *names: str) -> object | None:
    if value is None:
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _binding_published_lease_id(binding: object | None) -> str | None:
    return _optional_text(_first_attr(binding, "published_lease_id")) or _optional_text(
        _nested_attr(binding, "_slot", "published_lease_id")
    )


def _binding_published_replica_id(binding: object | None) -> str | None:
    return _optional_text(
        _first_attr(binding, "published_replica_id")
    ) or _optional_text(_nested_attr(binding, "_slot", "published_replica_id"))


def binding_has_active_published_replica(binding: object | None) -> bool:
    return (
        _binding_published_lease_id(binding) is not None
        or _binding_published_replica_id(binding) is not None
    )


def _published_projection_matches_binding(
    *,
    projection: PublishedReplicaProjection,
    attachment: RuntimeAttachment,
    binding: object,
) -> bool:
    lease_id = _binding_published_lease_id(binding)
    if projection.lease_id is not None and projection.lease_id != lease_id:
        return False
    replica_id = _binding_published_replica_id(binding)
    if (
        projection.replica_id is not None
        and replica_id is not None
        and projection.replica_id != replica_id
    ):
        return False
    artifact_ref = _optional_text(getattr(binding, "artifact_id", None))
    if artifact_ref is not None and projection.artifact_ref != artifact_ref:
        return False
    actual_ref = _nonempty_binding_value_ref(getattr(binding, "current_value", None))
    expected_ref = projection.binding_value_ref or (
        attachment.view.endpoint.weight_version.binding_value_ref
    )
    if actual_ref is not None and not _binding_value_refs_match(
        expected_ref, actual_ref
    ):
        return False
    return lease_id is not None or replica_id is not None


def _binding_byte_space_fields(binding: object | None) -> dict[str, str | None]:
    byte_space = getattr(binding, "byte_space", None)
    return {
        "byte_space_kind": (
            _optional_text(getattr(byte_space, "kind", None))
            or _optional_text(getattr(byte_space, "type", None))
        ),
        "byte_space_id": (
            _optional_text(getattr(byte_space, "id", None))
            or _optional_text(getattr(byte_space, "device_id", None))
            or _optional_text(getattr(byte_space, "name", None))
        ),
    }


def _call_operation_wait(
    operation: object,
    *,
    timeout_s: float,
) -> object | None:
    wait = getattr(operation, "wait", None)
    if callable(wait):
        return wait(timeout_s=timeout_s)
    result = getattr(operation, "result", None)
    if callable(result):
        return result(timeout_s=timeout_s)
    return operation


def _published_replica_projection_from_result(
    *,
    attachment: RuntimeAttachment,
    binding: object,
    operation: object | None,
    result: object | None,
    state: str,
    reason: str | None = None,
) -> PublishedReplicaProjection:
    current_value = getattr(binding, "current_value", None)
    binding_value_ref = (
        _nonempty_binding_value_ref(result)
        or _nonempty_binding_value_ref(current_value)
        or attachment.view.endpoint.weight_version.binding_value_ref
    )
    byte_space_fields = _binding_byte_space_fields(binding)
    return PublishedReplicaProjection(
        state=state,
        operation_id=_optional_text(getattr(operation, "operation_id", None)),
        replica_id=(
            _optional_text(_first_attr(result, "replica_id", "published_replica_id"))
            or _binding_published_replica_id(binding)
        ),
        lease_id=(
            _optional_text(_first_attr(result, "lease_id", "published_lease_id"))
            or _binding_published_lease_id(binding)
        ),
        artifact_ref=(
            _optional_text(_first_attr(result, "serving_artifact_id", "artifact_id"))
            or attachment.view.endpoint.weight_version.serving_artifact_ref
        ),
        device_uuid=(
            _optional_text(_first_attr(result, "device_uuid", "device_id"))
            or _optional_text(getattr(binding, "device_uuid", None))
        ),
        owner_pid=_optional_int(_first_attr(result, "owner_pid")) or os.getpid(),
        binding_layout_id=(
            _optional_text(_first_attr(result, "binding_layout_id"))
            or _optional_text(getattr(binding, "binding_layout_id", None))
            or attachment.view.endpoint.weight_version.binding_layout_id
        ),
        binding_value_ref=binding_value_ref,
        generation=publication_generation(attachment),
        reason=reason,
        byte_space_kind=byte_space_fields["byte_space_kind"],
        byte_space_id=byte_space_fields["byte_space_id"],
    )


def _published_replica_projection_for_state(
    *,
    attachment: RuntimeAttachment,
    state: str,
    reason: str | None = None,
    binding: object | None = None,
    operation: object | None = None,
    result: object | None = None,
) -> PublishedReplicaProjection:
    if binding is not None:
        return _published_replica_projection_from_result(
            attachment=attachment,
            binding=binding,
            operation=operation,
            result=result,
            state=state,
            reason=reason,
        )
    weight_version = attachment.view.endpoint.weight_version
    return PublishedReplicaProjection(
        state=state,
        operation_id=_optional_text(getattr(operation, "operation_id", None)),
        artifact_ref=weight_version.serving_artifact_ref,
        owner_pid=os.getpid(),
        binding_layout_id=weight_version.binding_layout_id,
        binding_value_ref=weight_version.binding_value_ref,
        generation=publication_generation(attachment),
        reason=reason,
    )


def _publication_release_contract(
    *,
    attachment: RuntimeAttachment,
    projection: PublishedReplicaProjection,
    envelope: RealizationResourceEnvelope,
) -> RealizationReleaseContract:
    binding = state_publication_binding(attachment.state)
    release_actions: list[Callable[[], None]] = []
    if (
        projection.state in _ACTIVE_PUBLICATION_STATES
        or binding_has_active_published_replica(binding)
    ):
        release_actions.append(attachment.state.retire_active_publication)
    existing_release = getattr(attachment.state.release_contract, "release", None)
    if callable(existing_release):

        def release_existing_contract() -> None:
            existing_release()

        release_actions.append(release_existing_contract)
    else:
        release_actions.append(attachment.state.close_binding_handle)
    return release_contract_for(envelope, *release_actions)


def _attachment_with_published_replica(
    attachment: RuntimeAttachment,
    projection: PublishedReplicaProjection,
) -> RuntimeAttachment:
    binding = state_publication_binding(attachment.state)
    spec = ArtifactRealizationSpec.publication(target=projection)
    target_layout_digest = (
        projection.binding_layout_id
        or attachment.view.endpoint.weight_version.binding_layout_id
        or publication_generation(attachment)
    )
    target_plan = RealizationTargetPlan(
        kind=spec.target_kind,
        target_layout_digest=target_layout_digest,
        binding_layout_id=projection.binding_layout_id,
    )
    envelope = envelope_for_publication(projection=projection, binding=binding)
    envelope.validate_for_target(target_plan)
    report = report_for_publication(
        artifact_id=(
            projection.artifact_ref
            or attachment.view.endpoint.weight_version.serving_artifact_ref
            or ""
        ),
        source_selection_digest=publication_generation(attachment),
        target_plan=target_plan,
        envelope=envelope,
        projection=projection,
        binding_handle=binding,
        risk_labels=(projection.state,),
    )
    weight_version = replace(
        attachment.view.endpoint.weight_version,
        published_replica=projection,
    )
    endpoint = replace(attachment.view.endpoint, weight_version=weight_version)
    diagnostics = dict(attachment.view.diagnostics)
    diagnostics["artifact_publication_report"] = artifact_realization_report_to_dict(
        report
    )
    view = replace(attachment.view, endpoint=endpoint, diagnostics=diagnostics)
    release_contract = _publication_release_contract(
        attachment=attachment,
        projection=projection,
        envelope=envelope,
    )
    publication_handle = ArtifactRealizationHandle(
        target_kind=spec.target_kind,
        report=report,
        binding_value=binding,
        release_contract=release_contract,
    )
    emit_artifact_realization_profile_event(report)
    state = replace(
        attachment.state,
        release_contract=release_contract,
        publication_handle=publication_handle,
    )
    return replace(attachment, state=state, view=view)


def _publication_error(
    message: str,
    *,
    attachment: RuntimeAttachment | None = None,
    state: str = "failed",
    reason: str | None = None,
    binding: object | None = None,
    operation: object | None = None,
    result: object | None = None,
    operation_name: str | None = None,
    retryable: bool | None = None,
    worker_suspect: bool | None = None,
    details: Mapping[str, object] | None = None,
) -> ReplicaPublicationError:
    projected_attachment: RuntimeAttachment | None = None
    if attachment is not None:
        projection = _published_replica_projection_for_state(
            attachment=attachment,
            state=state,
            reason=reason,
            binding=binding,
            operation=operation,
            result=result,
        )
        projected_attachment = _attachment_with_published_replica(
            attachment, projection
        )
    return ReplicaPublicationError(
        message,
        attachment=projected_attachment,
        operation=operation_name,
        retryable=retryable,
        worker_suspect=worker_suspect,
        details=details,
    )


def _best_effort_cancel_operation(operation: object | None) -> str | None:
    cancel = getattr(operation, "cancel", None)
    if not callable(cancel):
        return None
    try:
        cancel()
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


def _best_effort_retire_publication_binding(
    binding: object | None,
    *,
    drain_timeout_s: float | None,
) -> str | None:
    retire = getattr(binding, "retire", None)
    if not callable(retire):
        return None
    try:
        retire(drain_timeout_s=drain_timeout_s)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


def _emit_publication_profile(
    profile_sink: Callable[[Mapping[str, object]], object] | None,
    event: str,
    *,
    attachment: RuntimeAttachment,
    duration_s: float,
    published_replica: PublishedReplicaProjection | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> None:
    if not callable(profile_sink):
        return
    payload: dict[str, object] = {
        "event": event,
        "duration_s": duration_s,
        "serving_artifact_ref": attachment.view.endpoint.weight_version.serving_artifact_ref,
        "generation": publication_generation(attachment),
    }
    if published_replica is not None:
        payload["published_replica_state"] = published_replica.state
        if published_replica.replica_id is not None:
            payload["replica_id"] = published_replica.replica_id
        if published_replica.lease_id is not None:
            payload["lease_id"] = published_replica.lease_id
    if reason is not None:
        payload["reason"] = reason
    if error is not None:
        payload["error"] = error
    profile_sink(payload)


def publish_current_replica(
    *,
    current_attachment: RuntimeAttachment,
    policy: object,
    ensure_runtime_initialized: Callable[[], None],
    profile_sink: Callable[[Mapping[str, object]], object] | None = None,
) -> RuntimeAttachment:
    """Publish the current artifact-backed runtime attachment as a replica."""

    if not isinstance(current_attachment, RuntimeAttachment):
        raise ReplicaPublicationError(
            "publish_current_replica requires a RuntimeAttachment"
        )
    if getattr(policy, "mode", None) == "disabled":
        return current_attachment
    profile_start = time.perf_counter()
    ensure_runtime_initialized()
    weight_version = current_attachment.view.endpoint.weight_version
    if not weight_version.serving_artifact_ref:
        raise _publication_error(
            "Replica publication requires an artifact-backed serving attachment",
            attachment=current_attachment,
            reason="not_artifact_backed",
        )
    binding = state_publication_binding(current_attachment.state)
    if binding is None:
        raise _publication_error(
            "Runtime attachment has no publication-capable binding",
            attachment=current_attachment,
            reason="missing_publication_binding",
        )
    actual_artifact_ref = _optional_text(getattr(binding, "artifact_id", None))
    if (
        actual_artifact_ref is not None
        and actual_artifact_ref != weight_version.serving_artifact_ref
    ):
        raise _publication_error(
            "Runtime attachment publication artifact does not match "
            "the current weight version",
            attachment=current_attachment,
            binding=binding,
            reason="artifact_scope_mismatch",
            details={
                "expected_artifact_ref": weight_version.serving_artifact_ref,
                "actual_artifact_ref": actual_artifact_ref,
            },
        )
    published = weight_version.published_replica
    if published is not None and published.state in _ACTIVE_PUBLICATION_STATES:
        if published.state == "publishing":
            current_generation = publication_generation(current_attachment)
            if published.generation not in (None, current_generation) or (
                (published.replica_id is not None or published.lease_id is not None)
                and not _published_projection_matches_binding(
                    projection=published,
                    attachment=current_attachment,
                    binding=binding,
                )
            ):
                raise _publication_error(
                    "Runtime attachment publishing projection does not "
                    "match the current publication binding",
                    attachment=current_attachment,
                    binding=binding,
                    reason="active_projection_mismatch",
                    details={
                        "published_replica_state": published.state,
                        "replica_id": published.replica_id,
                        "lease_id": published.lease_id,
                    },
                )
        if published.state == "published" and (
            _published_projection_matches_binding(
                projection=published,
                attachment=current_attachment,
                binding=binding,
            )
        ):
            _emit_publication_profile(
                profile_sink,
                "runtime_publication.publish.replay",
                attachment=current_attachment,
                duration_s=time.perf_counter() - profile_start,
                published_replica=published,
            )
            return current_attachment
        if published.state != "publishing":
            raise _publication_error(
                "Runtime attachment active published replica does not "
                "match the current publication binding",
                attachment=current_attachment,
                binding=binding,
                reason="active_projection_mismatch",
                details={
                    "published_replica_state": published.state,
                    "replica_id": published.replica_id,
                    "lease_id": published.lease_id,
                },
            )
    operation = None
    result = None
    try:
        publish_operation = getattr(binding, "publish_replica_operation", None)
        if callable(publish_operation):
            operation = publish_operation()
            result = _call_operation_wait(
                operation,
                timeout_s=float(getattr(policy, "timeout_s", 0.0)),
            )
        else:
            publish = getattr(binding, "publish_replica", None)
            if not callable(publish):
                raise _publication_error(
                    "Runtime binding does not expose publish_replica_operation",
                    attachment=current_attachment,
                    binding=binding,
                    reason="missing_publication_capability",
                )
            result = publish()
    except ReplicaPublicationError:
        _emit_publication_profile(
            profile_sink,
            "runtime_publication.publish.failed",
            attachment=current_attachment,
            duration_s=time.perf_counter() - profile_start,
            error="replica_publication",
        )
        raise
    except Exception as exc:
        cancel_error = _best_effort_cancel_operation(operation)
        retire_error = _best_effort_retire_publication_binding(
            binding,
            drain_timeout_s=getattr(policy, "drain_timeout_s", None),
        )
        details: dict[str, object] = {"reason": str(exc)}
        if cancel_error is not None:
            details["cancel_error"] = cancel_error
        if retire_error is not None:
            details["retire_error"] = retire_error
        wrapped = _publication_error(
            "Runtime replica publication failed",
            attachment=current_attachment,
            binding=binding,
            operation=operation,
            reason="publish_error",
            details=details,
        )
        _emit_publication_profile(
            profile_sink,
            "runtime_publication.publish.failed",
            attachment=current_attachment,
            duration_s=time.perf_counter() - profile_start,
            error=str(exc),
        )
        raise wrapped from exc
    actual_ref = _nonempty_binding_value_ref(result) or _nonempty_binding_value_ref(
        getattr(binding, "current_value", None)
    )
    if not _binding_value_refs_match(weight_version.binding_value_ref, actual_ref):
        retire_error = _best_effort_retire_publication_binding(
            binding,
            drain_timeout_s=getattr(policy, "drain_timeout_s", None),
        )
        stale_details: dict[str, object] = {
            "expected_binding_value_ref": None
            if weight_version.binding_value_ref is None
            else weight_version.binding_value_ref.to_dict(),
            "actual_binding_value_ref": None
            if actual_ref is None
            else actual_ref.to_dict(),
        }
        if retire_error is not None:
            stale_details["retire_error"] = retire_error
        _emit_publication_profile(
            profile_sink,
            "runtime_publication.publish.stale",
            attachment=current_attachment,
            duration_s=time.perf_counter() - profile_start,
            error="stale_publish_result",
        )
        raise _publication_error(
            "Runtime attachment publication result is stale",
            attachment=current_attachment,
            state="stale",
            reason="stale_publish_result",
            binding=binding,
            operation=operation,
            result=result,
            details=stale_details,
        )
    projection = _published_replica_projection_from_result(
        attachment=current_attachment,
        binding=binding,
        operation=operation,
        result=result,
        state="published",
    )
    _emit_publication_profile(
        profile_sink,
        "runtime_publication.publish.done",
        attachment=current_attachment,
        duration_s=time.perf_counter() - profile_start,
        published_replica=projection,
    )
    return _attachment_with_published_replica(current_attachment, projection)


def project_current_replica_publication_state(
    *,
    current_attachment: RuntimeAttachment,
    state: str,
    reason: str | None = None,
    operation_id: str | None = None,
) -> RuntimeAttachment:
    """Return an attachment with a non-authoritative publication projection."""

    if not isinstance(current_attachment, RuntimeAttachment):
        raise ReplicaPublicationError(
            "project_current_replica_publication_state requires a RuntimeAttachment"
        )
    normalized_state = str(state).strip().lower()
    if normalized_state not in {"publishing", "failed", "stale"}:
        raise ReplicaPublicationError(
            "project_current_replica_publication_state only supports "
            "publishing, failed, or stale"
        )
    binding = state_publication_binding(current_attachment.state)
    operation = (
        SimpleNamespace(operation_id=str(operation_id)) if operation_id else None
    )
    projection = _published_replica_projection_for_state(
        attachment=current_attachment,
        state=normalized_state,
        reason=reason,
        binding=binding,
        operation=operation,
        result=getattr(binding, "current_value", None),
    )
    return _attachment_with_published_replica(current_attachment, projection)


def retire_current_replica(
    *,
    current_attachment: RuntimeAttachment,
    reason: str = "retire",
    drain_timeout_s: float | None = None,
    default_drain_timeout_s: float | None = None,
    ensure_runtime_initialized: Callable[[], None],
    profile_sink: Callable[[Mapping[str, object]], object] | None = None,
) -> RuntimeAttachment:
    """Retire the published replica tied to a runtime attachment."""

    if not isinstance(current_attachment, RuntimeAttachment):
        raise ReplicaPublicationError(
            "retire_current_replica requires a RuntimeAttachment"
        )
    weight_version = current_attachment.view.endpoint.weight_version
    published = weight_version.published_replica
    binding = state_publication_binding(current_attachment.state)
    binding_has_active_publication = binding_has_active_published_replica(binding)
    active_projection = (
        published is not None and published.state in _ACTIVE_PUBLICATION_STATES
    )
    if not active_projection and not binding_has_active_publication:
        return current_attachment
    profile_start = time.perf_counter()
    ensure_runtime_initialized()
    if binding is None:
        raise ReplicaPublicationError(
            "Runtime attachment has no publication-capable binding"
        )
    retire = getattr(binding, "retire", None)
    if not callable(retire):
        raise ReplicaPublicationError("Runtime binding does not expose retire")
    drain_timeout = (
        drain_timeout_s if drain_timeout_s is not None else default_drain_timeout_s
    )
    binding_projection = None
    if not active_projection:
        binding_projection = _published_replica_projection_from_result(
            attachment=current_attachment,
            binding=binding,
            operation=None,
            result=getattr(binding, "current_value", None),
            state="retiring",
            reason=reason,
        )
    try:
        retire(drain_timeout_s=drain_timeout)
    except Exception as exc:
        wrapped = ReplicaPublicationError(
            "Runtime replica retirement failed",
            details={"reason": str(exc)},
        )
        _emit_publication_profile(
            profile_sink,
            "runtime_publication.retire.failed",
            attachment=current_attachment,
            duration_s=time.perf_counter() - profile_start,
            published_replica=published,
            reason=reason,
            error=str(exc),
        )
        raise wrapped from exc
    projection = published if active_projection else binding_projection
    if projection is None:
        raise ReplicaPublicationError(
            "Runtime replica retirement lost publication identity",
            details={"reason": reason},
        )
    projection = replace(projection, state="retired", reason=reason)
    _emit_publication_profile(
        profile_sink,
        "runtime_publication.retire.done",
        attachment=current_attachment,
        duration_s=time.perf_counter() - profile_start,
        published_replica=projection,
        reason=reason,
    )
    return _attachment_with_published_replica(current_attachment, projection)


def reject_reload_with_active_publication(
    current_attachment: RuntimeAttachment,
) -> None:
    published = current_attachment.view.endpoint.weight_version.published_replica
    if published is not None and published.state in _ACTIVE_PUBLICATION_STATES:
        raise ReplicaPublicationError(
            "TensorCast serving reload requires retiring the active published "
            "replica before swap",
            operation="reload",
            details={
                "published_replica_state": published.state,
                "replica_id": published.replica_id,
                "lease_id": published.lease_id,
            },
        )

    binding = state_publication_binding(current_attachment.state)
    if not binding_has_active_published_replica(binding):
        return
    raise ReplicaPublicationError(
        "TensorCast serving reload found an active published replica on the "
        "runtime binding but no active attachment projection; retire the "
        "current replica before swap",
        operation="reload",
        details={
            "published_replica_state": None if published is None else published.state,
            "replica_id": _binding_published_replica_id(binding),
            "lease_id": _binding_published_lease_id(binding),
        },
    )


__all__ = [
    "binding_has_active_published_replica",
    "project_current_replica_publication_state",
    "publication_generation",
    "publish_current_replica",
    "reject_reload_with_active_publication",
    "retire_current_replica",
    "state_publication_binding",
]
