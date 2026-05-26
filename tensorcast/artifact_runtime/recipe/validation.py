#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral runtime recipe fact validation."""

from __future__ import annotations

from typing import Any

from tensorcast.artifact_runtime.readiness import (
    runtime_support_level_at_least,
    runtime_support_level_display_name,
)
from tensorcast.types import BuilderMode, FinalizeClass, RuntimeSupportLevel


def validate_recipe_for_builder_mode(recipe: Any, mode: BuilderMode | str) -> Any:
    facts = recipe.runtime_facts
    builder_mode = _coerce_builder_mode(mode)
    failures: list[str] = []
    if not runtime_support_level_at_least(
        facts.support_level, RuntimeSupportLevel.BUILDER_PUBLICATION_READY
    ):
        failures.append(
            "support_level="
            f"{runtime_support_level_display_name(facts.support_level)} "
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


def _coerce_builder_mode(value: BuilderMode | str) -> BuilderMode:
    if isinstance(value, BuilderMode):
        return value
    return BuilderMode(str(value).strip())


__all__ = [
    "runtime_support_level_at_least",
    "runtime_support_level_display_name",
    "validate_recipe_for_builder_mode",
]
