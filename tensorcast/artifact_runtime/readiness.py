#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime readiness and admission helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tensorcast.types import BuilderMode, FinalizeClass, RuntimeSupportLevel

_SUPPORT_LEVEL_ORDER: dict[RuntimeSupportLevel, int] = {
    RuntimeSupportLevel.BLOCKED: -1,
    RuntimeSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY: 0,
    RuntimeSupportLevel.BUILDER_PUBLICATION_READY: 1,
    RuntimeSupportLevel.RUNTIME_BIND_SWAP_READY: 2,
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


def coerce_runtime_support_level(
    value: Any,
    *,
    default: RuntimeSupportLevel = RuntimeSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY,
) -> RuntimeSupportLevel:
    if value is None:
        return default
    if isinstance(value, RuntimeSupportLevel):
        return value
    normalized = str(value).strip().lower()
    return RuntimeSupportLevel(normalized)


def runtime_support_level_at_least(
    value: RuntimeSupportLevel | str,
    minimum: RuntimeSupportLevel | str,
) -> bool:
    resolved_value = coerce_runtime_support_level(value)
    resolved_minimum = coerce_runtime_support_level(minimum)
    return (
        _SUPPORT_LEVEL_ORDER[resolved_value] >= _SUPPORT_LEVEL_ORDER[resolved_minimum]
    )


def runtime_support_level_display_name(value: RuntimeSupportLevel | str) -> str:
    return coerce_runtime_support_level(value).value


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


def readiness_support_level(row: Any) -> RuntimeSupportLevel:
    return coerce_runtime_support_level(getattr(row, "support_level", None))


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
        and runtime_support_level_at_least(
            readiness_support_level(row),
            RuntimeSupportLevel.BUILDER_PUBLICATION_READY,
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
        and runtime_support_level_at_least(
            readiness_support_level(row),
            RuntimeSupportLevel.BUILDER_PUBLICATION_READY,
        )
        and readiness_process_after_load_class(row)
        == FinalizeClass.REPRESENTATION_CHANGING
        and readiness_post_bind_finalize_class(row) == FinalizeClass.RUNTIME_ONLY
    )


def is_runtime_bind_swap_allowlisted(row: Any) -> bool:
    allowed = bool(getattr(row, "runtime_bind_swap_allowed", False))
    return (
        allowed
        and readiness_post_bind_finalize_class(row) == FinalizeClass.RUNTIME_ONLY
        and runtime_support_level_at_least(
            readiness_support_level(row),
            RuntimeSupportLevel.RUNTIME_BIND_SWAP_READY,
        )
    )


class ReadinessInventoryAdmissionPolicy:
    """Admission policy backed by a framework readiness resolver."""

    def __init__(
        self,
        resolve_readiness: Callable[[Any], Any],
        *,
        endpoint_fields: Callable[[Any], dict[str, object]] | None = None,
    ) -> None:
        self._resolve_readiness = resolve_readiness
        self._endpoint_fields = endpoint_fields

    def admit(self, request: Any) -> Any:
        from tensorcast.artifact_runtime.host import RuntimeAdmissionDecision

        row = self._resolve_readiness(request.model_config)
        missing_semantic_proofs = (
            request.placement_admission.missing_framework_semantic_proofs()
        )
        allowed = is_runtime_bind_swap_allowlisted(row) and not missing_semantic_proofs
        support_level = runtime_support_level_display_name(readiness_support_level(row))
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
        return RuntimeAdmissionDecision(
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
    "coerce_runtime_support_level",
    "is_binding_finalize_publication_allowlisted",
    "is_pure_transform_publication_allowlisted",
    "is_runtime_bind_swap_allowlisted",
    "readiness_family",
    "readiness_post_bind_finalize_class",
    "readiness_process_after_load_class",
    "readiness_publication_modes",
    "readiness_support_level",
    "runtime_support_level_at_least",
    "runtime_support_level_display_name",
]
