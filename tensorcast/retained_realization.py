#  Copyright (c) 2026, TensorCast Team.
"""Neutral retained realization claim helpers.

Retained realization claims are serialized handoffs produced by artifact
prefetch. They expose the trusted reservation credit needed before framework
admission while keeping the existing retained binding authority validation as
the source of truth during the migration away from serving-rooted public names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tensorcast.api.errors import ArtifactError
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationHandle,
    ArtifactRealizationSpec,
)
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
)
from tensorcast.retained_realization_authority import (
    RetainedRealizationAuthority as RetainedRealizationAuthorityConfig,
)
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    GroupRealizationAcquireRef,
    PrefetchHandoff,
    RealizationTarget,
    RuntimeBindingMemberRef,
)


@dataclass(frozen=True)
class RetainedRealizationExpectedDigests:
    """Expected identity digests embedded in a retained realization claim."""

    target_layout_hash: str
    tensor_schema_hash: str
    runtime_build_digest: str
    resolved_spec_digest: str


@dataclass(frozen=True)
class RetainedRealizationClaim:
    """Validated retained realization handoff for admission and acquire."""

    _authority: ParsedRetainedRealizationAuthority

    @property
    def group_id(self) -> str:
        return self._authority.group_id

    @property
    def local_ref(self) -> str | None:
        return self._authority.local_serving_ref

    @property
    def binding_value_ref(self) -> BindingValueRef:
        return self._authority.binding_value_ref

    @property
    def reservation_capability(self) -> BindingReservationCapability:
        return self._authority.reservation_capability

    @property
    def daemon_id(self) -> str:
        return self._authority.daemon_id

    @property
    def daemon_session_id(self) -> str:
        return self._authority.daemon_session_id

    @property
    def device_uuid(self) -> str:
        return self._authority.device_uuid

    @property
    def member(self) -> RuntimeBindingMemberRef:
        return self._authority.member

    @property
    def reservation_bytes(self) -> int:
        return self._authority.reservation_bytes

    @property
    def expected(self) -> RetainedRealizationExpectedDigests:
        expected = self._authority.expected
        return RetainedRealizationExpectedDigests(
            target_layout_hash=expected.target_layout_hash,
            tensor_schema_hash=expected.tensor_schema_hash,
            runtime_build_digest=expected.runtime_build_digest,
            resolved_spec_digest=expected.resolved_spec_digest,
        )

    @property
    def readiness(self) -> str:
        return self._authority.readiness

    @property
    def verification_state(self) -> str:
        return self._authority.verification_state

    @property
    def serving_artifact_id(self) -> str | None:
        return self._authority.serving_artifact_id

    @property
    def group_realization_acquire(self) -> GroupRealizationAcquireRef | None:
        return self._authority.group_realization_acquire

    @property
    def authority(self) -> ParsedRetainedRealizationAuthority:
        return self._authority

    def as_authority(self) -> ParsedRetainedRealizationAuthority:
        return self._authority

    @staticmethod
    def _request_facts(
        spec: ArtifactRealizationSpec,
        runtime_context: Any | None,
    ) -> tuple[ArtifactRealizationSpec, Any]:
        from tensorcast.artifact_runtime.request_facts import (
            ModelRuntimeRequestFactsError,
            resolve_model_runtime_request_facts,
        )

        try:
            facts = resolve_model_runtime_request_facts(
                spec=spec,
                runtime_context=runtime_context,
            )
        except ModelRuntimeRequestFactsError as exc:
            raise ArtifactError(
                str(exc),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ) from exc
        return facts.spec, facts.context

    def realize_model_runtime(
        self,
        spec: ArtifactRealizationSpec,
        *,
        runtime_host: Any,
        runtime_context: Any | None = None,
        profile_sink: Any | None = None,
    ) -> ArtifactRealizationHandle:
        """Realize this retained claim as a model runtime attachment."""

        if runtime_host is None:
            raise ArtifactError(
                "retained model_runtime realization requires runtime_host",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if spec.target_kind != "model_runtime":
            raise ArtifactError(
                "retained realization claim requires a model_runtime spec",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        from tensorcast.artifact_runtime.lifecycle import ArtifactRuntimeIntegration

        resolved_spec, context = self._request_facts(spec, runtime_context)
        attachment = ArtifactRuntimeIntegration(
            profile_sink=profile_sink,
            host=runtime_host,
        ).realize_retained_model_runtime(
            authority=self._authority,
            spec=resolved_spec,
            context=context,
        )
        handle = getattr(attachment.state, "model_runtime_handle", None)
        if not isinstance(handle, ArtifactRealizationHandle):
            raise ArtifactError(
                "retained model_runtime realization completed without a "
                "realization handle",
                status_code="INTERNAL",
                retryable=False,
            )
        return handle


def parse_retained_realization_claim(
    extra: Mapping[str, Any] | Any,
    *,
    expected_member: RuntimeBindingMemberRef | None = None,
) -> RetainedRealizationClaim:
    """Parse and validate a retained realization claim from loader config."""

    return RetainedRealizationClaim(
        parse_retained_realization_authority(
            extra,
            expected_member=expected_member,
        )
    )


def parse_retained_realization_authority(
    extra: Mapping[str, Any] | Any,
    *,
    expected_member: RuntimeBindingMemberRef | None = None,
) -> ParsedRetainedRealizationAuthority:
    """Parse and validate retained realization authority from runtime config."""

    from tensorcast.artifact_runtime.config import TensorCastRuntimeConfig

    config = (
        extra
        if isinstance(extra, TensorCastRuntimeConfig)
        else TensorCastRuntimeConfig.from_mapping(extra)
    )
    if config.retained_binding_acquire.mode != "external":
        raise ValueError(
            "TensorCast retained realization authority requires "
            "retained_binding_acquire.mode='external' and "
            "retained_binding_acquire.authority"
        )
    authority_config = _select_retained_realization_authority_config(
        config,
        expected_member=expected_member,
    )

    binding_value_ref = _model_validate(
        BindingValueRef,
        authority_config.binding_value_ref,
        field_name="retained_binding_acquire.authority.binding_value_ref",
    )
    member = _model_validate(
        RuntimeBindingMemberRef,
        authority_config.member_ref,
        field_name="retained_binding_acquire.authority.member_ref",
    )
    capability_payload = _payload_to_dict(
        authority_config.reservation_capability,
        field_name="retained_binding_acquire.authority.reservation_capability",
    )
    capability_payload.setdefault(
        "binding_value_ref", binding_value_ref.model_dump(mode="python")
    )
    capability_payload.setdefault("member", member.model_dump(mode="python"))
    reservation_capability = _model_validate(
        BindingReservationCapability,
        capability_payload,
        field_name="retained_binding_acquire.authority.reservation_capability",
    )
    group_realization_acquire = None
    if authority_config.group_realization_acquire is not None:
        group_realization_acquire = _model_validate(
            GroupRealizationAcquireRef,
            authority_config.group_realization_acquire,
            field_name="retained_binding_acquire.authority.group_realization_acquire",
        )

    authority = ParsedRetainedRealizationAuthority(
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
    _validate_retained_realization_authority_consistency(authority)
    if expected_member is not None and authority.member != expected_member:
        raise ValueError(
            "TensorCast retained realization authority member does not match "
            f"expected member: authority={authority.member!r}, "
            f"expected={expected_member!r}"
        )
    return authority


def retained_realization_claim_mode(extra: Mapping[str, Any] | None) -> str:
    """Return the retained claim acquire mode encoded in extra config."""

    if extra is None or not isinstance(extra, Mapping):
        return "disabled"
    from tensorcast.artifact_runtime.config import TensorCastRuntimeConfig

    return TensorCastRuntimeConfig.from_mapping(extra).retained_binding_acquire.mode


def retained_realization_trusted_reservation_bytes(
    load_config_or_extra: Any,
    *,
    expected_member: RuntimeBindingMemberRef | None = None,
) -> int:
    """Return trusted retained reservation bytes after full claim validation."""

    extra = getattr(
        load_config_or_extra,
        "model_loader_extra_config",
        load_config_or_extra,
    )
    if extra is None or not isinstance(extra, Mapping):
        return 0
    if retained_realization_claim_mode(extra) != "external":
        return 0
    return parse_retained_realization_claim(
        extra,
        expected_member=expected_member,
    ).reservation_bytes


def retained_realization_claim_extra_from_handoff(
    *,
    handoff: PrefetchHandoff,
    target: RealizationTarget,
    expected_member: RuntimeBindingMemberRef | None = None,
) -> dict[str, Any]:
    """Build serialized retained claim config from a prefetch handoff."""

    return _retained_realization_claim_extra(
        authority=_retained_realization_authority_from_handoff(
            handoff=handoff,
            target=target,
            expected_member=expected_member,
        ),
        config_key="retained_binding_acquire",
    )


def retained_realization_claim_extra_json_from_handoff(
    *,
    handoff: PrefetchHandoff,
    target: RealizationTarget,
    expected_member: RuntimeBindingMemberRef | None = None,
) -> str:
    """Serialize retained claim config using stable JSON ordering."""

    return json.dumps(
        retained_realization_claim_extra_from_handoff(
            handoff=handoff,
            target=target,
            expected_member=expected_member,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


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
            f"{field_name} is invalid for TensorCast retained realization "
            f"acquire: {exc}"
        ) from exc


def _select_retained_realization_authority_config(
    config: Any,
    *,
    expected_member: RuntimeBindingMemberRef | None = None,
) -> RetainedRealizationAuthorityConfig:
    acquire_config = config.retained_binding_acquire
    authority_config = acquire_config.authority
    if authority_config is not None:
        return authority_config

    authority_configs = tuple(acquire_config.authorities)
    if not authority_configs:
        raise ValueError(
            "TensorCast retained realization authority requires "
            "retained_binding_acquire.mode='external' and "
            "retained_binding_acquire.authority or "
            "retained_binding_acquire.authorities"
        )
    if expected_member is None:
        if len(authority_configs) == 1:
            return authority_configs[0]
        raise ValueError(
            "TensorCast retained realization authority set requires an expected "
            "serving member to select the worker authority"
        )

    for index, candidate in enumerate(authority_configs):
        member = _model_validate(
            RuntimeBindingMemberRef,
            candidate.member_ref,
            field_name=(f"retained_binding_acquire.authorities[{index}].member_ref"),
        )
        if member == expected_member:
            return candidate
    raise ValueError(
        "TensorCast retained realization authority set has no authority for "
        f"expected member {expected_member!r}"
    )


def _validate_retained_realization_authority_consistency(
    authority: ParsedRetainedRealizationAuthority,
) -> None:
    capability = authority.reservation_capability
    if capability.binding_value_ref != authority.binding_value_ref:
        raise ValueError(
            "retained_binding_acquire.authority.reservation_capability."
            "binding_value_ref must match retained_binding_acquire.authority."
            "binding_value_ref"
        )
    if capability.daemon_id != authority.daemon_id:
        raise ValueError(
            "retained_binding_acquire.authority.reservation_capability."
            "daemon_id mismatch"
        )
    if capability.daemon_session_id != authority.daemon_session_id:
        raise ValueError(
            "retained_binding_acquire.authority.reservation_capability."
            "daemon_session_id mismatch"
        )
    if capability.device_uuid != authority.device_uuid:
        raise ValueError(
            "retained_binding_acquire.authority.reservation_capability."
            "device_uuid mismatch"
        )
    if capability.member != authority.member:
        raise ValueError(
            "retained_binding_acquire.authority.reservation_capability.member mismatch"
        )
    if capability.reservation_bytes != authority.reservation_bytes:
        raise ValueError(
            "retained_binding_acquire.authority.reservation_capability."
            "reservation_bytes must match retained_binding_acquire.authority."
            "trusted_reservation_bytes"
        )
    if authority.member.group_id is not None and authority.member.group_id != (
        authority.group_id
    ):
        raise ValueError(
            "retained_binding_acquire.authority.member_ref.group_id must match "
            "retained_binding_acquire.authority.group_id"
        )
    if (
        authority.readiness == "runtime_published_ready"
        and not authority.serving_artifact_id
    ):
        raise ValueError(
            "retained_binding_acquire.authority.serving_artifact_id is required "
            "when retained_binding_acquire.authority.readiness="
            "'runtime_published_ready'"
        )


def _retained_realization_authority_from_handoff(
    *,
    handoff: PrefetchHandoff,
    target: RealizationTarget,
    expected_member: RuntimeBindingMemberRef | None = None,
) -> dict[str, Any]:
    member = handoff.member
    if expected_member is not None and member != expected_member:
        raise ValueError(
            "Prefetched retained realization member does not match expected "
            f"placement: prefetched={member}, expected={expected_member}"
        )
    authority: dict[str, Any] = {
        "group_id": member.group_id or "",
        "member_ref": _model_dump(member),
        "daemon_id": handoff.daemon_id,
        "daemon_session_id": handoff.daemon_session_id,
        "device_uuid": handoff.device_uuid,
        "binding_value_ref": _model_dump(handoff.binding_value_ref),
        "reservation_capability": _model_dump(handoff.reservation_capability),
        "local_serving_ref": handoff.local_serving_ref,
        "readiness": str(getattr(handoff.readiness, "value", handoff.readiness)),
        "verification_state": str(
            getattr(
                handoff.verification_state,
                "value",
                handoff.verification_state,
            )
        ),
        "serving_artifact_id": handoff.serving_artifact_id,
        "trusted_reservation_bytes": handoff.reservation_bytes,
        "expected": {
            "target_layout_hash": target.resolved_layout.target_layout_hash,
            "tensor_schema_hash": target.resolved_layout.tensor_schema_hash,
            "runtime_build_digest": target.runtime_build_digest,
            "resolved_spec_digest": target.resolved_layout.spec_digest,
        },
    }
    if handoff.group_realization_acquire is not None:
        authority["group_realization_acquire"] = _model_dump(
            handoff.group_realization_acquire
        )
    return authority


def _retained_realization_claim_extra(
    *,
    authority: dict[str, Any],
    config_key: str,
) -> dict[str, Any]:
    return {
        config_key: {
            "mode": "external",
            "authority": authority,
        },
    }


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


__all__ = [
    "RetainedRealizationClaim",
    "RetainedRealizationExpectedDigests",
    "parse_retained_realization_authority",
    "parse_retained_realization_claim",
    "retained_realization_claim_mode",
    "retained_realization_trusted_reservation_bytes",
    "retained_realization_claim_extra_from_handoff",
    "retained_realization_claim_extra_json_from_handoff",
]
