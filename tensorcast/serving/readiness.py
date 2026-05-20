#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral serving readiness and admission helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tensorcast.types import BuilderMode, FinalizeClass, ServingSupportLevel

_SUPPORT_LEVEL_ORDER: dict[ServingSupportLevel, int] = {
    ServingSupportLevel.BLOCKED: -1,
    ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY: 0,
    ServingSupportLevel.BUILDER_PUBLICATION_READY: 1,
    ServingSupportLevel.RUNTIME_BIND_SWAP_READY: 2,
}

_SUPPORT_LEVEL_DISPLAY_NAMES: dict[ServingSupportLevel, str] = {
    ServingSupportLevel.BLOCKED: "blocked",
    ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY: "publication_not_ready",
    ServingSupportLevel.BUILDER_PUBLICATION_READY: "serving_artifact_publication_ready",
    ServingSupportLevel.RUNTIME_BIND_SWAP_READY: "serving_bind_swap_ready",
}


def coerce_finalize_class(
    value: Any,
    *,
    default: FinalizeClass,
) -> FinalizeClass:
    if value is None:
        return default
    if isinstance(value, FinalizeClass):
        return value
    return FinalizeClass(str(value).strip())


def coerce_serving_support_level(
    value: Any,
    *,
    default: ServingSupportLevel = ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY,
) -> ServingSupportLevel:
    if value is None:
        return default
    if isinstance(value, ServingSupportLevel):
        return value
    normalized = str(value).strip().lower()
    aliases = {
        "blocked": ServingSupportLevel.BLOCKED,
        "publication_not_ready": ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY,
        "serving_artifact_publication_ready": ServingSupportLevel.BUILDER_PUBLICATION_READY,
        "serving_bind_swap_ready": ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
    }
    if normalized in aliases:
        return aliases[normalized]
    return ServingSupportLevel(normalized)


def serving_support_level_at_least(
    value: ServingSupportLevel | str,
    minimum: ServingSupportLevel | str,
) -> bool:
    resolved_value = coerce_serving_support_level(value)
    resolved_minimum = coerce_serving_support_level(minimum)
    return (
        _SUPPORT_LEVEL_ORDER[resolved_value] >= _SUPPORT_LEVEL_ORDER[resolved_minimum]
    )


def serving_support_level_display_name(value: ServingSupportLevel | str) -> str:
    return _SUPPORT_LEVEL_DISPLAY_NAMES[coerce_serving_support_level(value)]


def readiness_family(row: Any) -> str:
    return str(getattr(row, "family", "") or "")


def readiness_process_after_load_class(row: Any) -> FinalizeClass:
    return coerce_finalize_class(
        getattr(row, "process_after_load_class", None),
        default=FinalizeClass.UNKNOWN_BLOCKED,
    )


def readiness_post_bind_finalize_class(row: Any) -> FinalizeClass:
    return coerce_finalize_class(
        getattr(row, "post_bind_finalize_class", None),
        default=FinalizeClass.RUNTIME_ONLY,
    )


def readiness_support_level(row: Any) -> ServingSupportLevel:
    return coerce_serving_support_level(getattr(row, "support_level", None))


def readiness_publication_modes(row: Any) -> tuple[str, ...]:
    modes = getattr(row, "publication_modes", ()) or ()
    return tuple(str(mode).strip() for mode in modes if str(mode).strip())


def is_pure_transform_publication_allowlisted(row: Any) -> bool:
    modes = set(readiness_publication_modes(row))
    pure_transform_candidate = bool(
        getattr(row, "pure_transform_candidate", False)
        or BuilderMode.PURE_TRANSFORM.value in modes
    )
    return (
        pure_transform_candidate
        and readiness_process_after_load_class(row) == FinalizeClass.RUNTIME_ONLY
        and readiness_post_bind_finalize_class(row) == FinalizeClass.RUNTIME_ONLY
        and serving_support_level_at_least(
            readiness_support_level(row),
            ServingSupportLevel.BUILDER_PUBLICATION_READY,
        )
    )


def is_binding_finalize_publication_allowlisted(row: Any) -> bool:
    modes = set(readiness_publication_modes(row))
    binding_finalize_candidate = (
        BuilderMode.BINDING_FINALIZE.value in modes
        or readiness_process_after_load_class(row)
        == FinalizeClass.REPRESENTATION_CHANGING
    )
    return (
        binding_finalize_candidate
        and serving_support_level_at_least(
            readiness_support_level(row),
            ServingSupportLevel.BUILDER_PUBLICATION_READY,
        )
        and readiness_process_after_load_class(row)
        == FinalizeClass.REPRESENTATION_CHANGING
        and readiness_post_bind_finalize_class(row) == FinalizeClass.RUNTIME_ONLY
    )


def is_runtime_bind_swap_allowlisted(row: Any) -> bool:
    allowed = bool(
        getattr(row, "runtime_bind_swap_allowed", False)
        or getattr(row, "serving_only_runtime_allowed", False)
    )
    return (
        allowed
        and readiness_post_bind_finalize_class(row) == FinalizeClass.RUNTIME_ONLY
        and serving_support_level_at_least(
            readiness_support_level(row),
            ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        )
    )


class ReadinessInventoryAdmissionPolicy:
    """AdmissionPolicy implementation backed by a framework readiness resolver."""

    def __init__(
        self,
        resolve_readiness: Callable[[Any], Any],
        *,
        endpoint_fields: Callable[[Any], dict[str, object]] | None = None,
    ) -> None:
        self._resolve_readiness = resolve_readiness
        self._endpoint_fields = endpoint_fields

    def admit(self, request: Any) -> Any:
        from tensorcast.serving.integration import AdmissionDecision

        row = self._resolve_readiness(request.model_config)
        missing_semantic_proofs = (
            request.placement_admission.missing_framework_semantic_proofs()
        )
        allowed = is_runtime_bind_swap_allowlisted(row) and not missing_semantic_proofs
        support_level = serving_support_level_display_name(readiness_support_level(row))
        if missing_semantic_proofs:
            support_level = (
                f"{support_level}:placement_missing_semantic_proof:"
                f"{','.join(missing_semantic_proofs)}"
            )
        elif (
            request.placement_admission.requires_framework_semantic_proof()
            and not allowed
        ):
            support_level = f"{support_level}:placement_fail_closed"
        family = readiness_family(row)
        endpoint_fields: dict[str, object] = {"family": family}
        if self._endpoint_fields is not None:
            endpoint_fields.update(self._endpoint_fields(row))
        return AdmissionDecision(
            family=family,
            support_level=support_level,
            startup_allowed=allowed,
            reload_allowed=allowed,
            local_bootstrap_allowed=allowed,
            endpoint_fields=endpoint_fields,
        )


__all__ = [
    "ReadinessInventoryAdmissionPolicy",
    "coerce_finalize_class",
    "coerce_serving_support_level",
    "is_binding_finalize_publication_allowlisted",
    "is_pure_transform_publication_allowlisted",
    "is_runtime_bind_swap_allowlisted",
    "readiness_family",
    "readiness_post_bind_finalize_class",
    "readiness_process_after_load_class",
    "readiness_publication_modes",
    "readiness_support_level",
    "serving_support_level_at_least",
    "serving_support_level_display_name",
]
