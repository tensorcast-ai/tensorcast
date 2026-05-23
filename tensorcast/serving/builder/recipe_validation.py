#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral serving recipe fact validation."""

from __future__ import annotations

from typing import Any

from tensorcast.types import BuilderMode, FinalizeClass, ServingSupportLevel

_SUPPORT_LEVEL_ORDER = {
    ServingSupportLevel.BLOCKED: 0,
    ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY: 1,
    ServingSupportLevel.BUILDER_PUBLICATION_READY: 2,
    ServingSupportLevel.RUNTIME_BIND_SWAP_READY: 3,
}


def serving_support_level_at_least(
    value: ServingSupportLevel | str,
    minimum: ServingSupportLevel | str,
) -> bool:
    resolved_value = _coerce_support_level(value)
    resolved_minimum = _coerce_support_level(minimum)
    return (
        _SUPPORT_LEVEL_ORDER[resolved_value] >= _SUPPORT_LEVEL_ORDER[resolved_minimum]
    )


def serving_support_level_display_name(value: ServingSupportLevel | str) -> str:
    return str(_coerce_support_level(value).value)


def validate_recipe_for_builder_mode(recipe: Any, mode: BuilderMode | str) -> Any:
    facts = recipe.serving_facts
    builder_mode = _coerce_builder_mode(mode)
    failures: list[str] = []
    if not serving_support_level_at_least(
        facts.support_level, ServingSupportLevel.BUILDER_PUBLICATION_READY
    ):
        failures.append(
            "support_level="
            f"{serving_support_level_display_name(facts.support_level)} "
            "is below builder_publication_ready"
        )
    if builder_mode == BuilderMode.PURE_TRANSFORM:
        if facts.process_after_load_class != FinalizeClass.RUNTIME_ONLY:
            failures.append(
                "process_after_load_class must be runtime_only for "
                f"PURE_TRANSFORM, got {facts.process_after_load_class.value}"
            )
        if facts.post_bind_finalize_class != FinalizeClass.RUNTIME_ONLY:
            failures.append(
                "post_bind_finalize_class must be runtime_only for "
                f"PURE_TRANSFORM, got {facts.post_bind_finalize_class.value}"
            )
    elif builder_mode == BuilderMode.BINDING_FINALIZE:
        if facts.process_after_load_class != FinalizeClass.REPRESENTATION_CHANGING:
            failures.append(
                "process_after_load_class must be representation_changing for "
                f"BINDING_FINALIZE, got "
                f"{facts.process_after_load_class.value}"
            )
        if facts.post_bind_finalize_class != FinalizeClass.RUNTIME_ONLY:
            failures.append(
                "post_bind_finalize_class must remain runtime_only for the "
                f"BINDING_FINALIZE runner, got "
                f"{facts.post_bind_finalize_class.value}"
            )
    else:
        raise ValueError(f"Unsupported TensorCast builder mode: {mode!r}")
    if failures:
        raise ValueError(
            f"TensorCast {builder_mode.value.upper()} recipe is not "
            "publication-ready: " + "; ".join(failures)
        )
    return recipe


def _coerce_support_level(value: ServingSupportLevel | str) -> ServingSupportLevel:
    if isinstance(value, ServingSupportLevel):
        return value
    return ServingSupportLevel(str(value).strip())


def _coerce_builder_mode(value: BuilderMode | str) -> BuilderMode:
    if isinstance(value, BuilderMode):
        return value
    return BuilderMode(str(value).strip())


__all__ = [
    "serving_support_level_at_least",
    "serving_support_level_display_name",
    "validate_recipe_for_builder_mode",
]
