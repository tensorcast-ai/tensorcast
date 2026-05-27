#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime publication context helpers for recipe-backed artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tensorcast.artifact_runtime.contract import logical_topology_json
from tensorcast.types import BuilderMode, RuntimeArtifactBuildIntent, RuntimeTopologyRef


@dataclass(frozen=True)
class RecipePublicationContext:
    source_artifact_ref: str
    framework_name: str
    adapter_version: str
    serving_abi_version: str
    logical_topology_json: str | None = None


def logical_topology_json_from_recipe(
    recipe: Any,
    *,
    topology: RuntimeTopologyRef | None = None,
    framework_payload: dict[str, Any] | None = None,
) -> str | None:
    if topology is None:
        if (
            getattr(recipe, "topology_ref", None) is None
            and getattr(recipe, "member_ref", None) is None
        ):
            return None
        raise ValueError(
            "TensorCast publication manifest requires RuntimeTopologyRef for "
            "a topology-sensitive recipe"
        )
    return logical_topology_json(
        topology,
        framework_payload=framework_payload or {},
    )


def publication_context_from_recipe(
    recipe: Any,
    *,
    logical_topology_json_payload: str | None = None,
) -> RecipePublicationContext:
    return RecipePublicationContext(
        source_artifact_ref=recipe.source_artifact_ref,
        framework_name=recipe.runtime_facts.framework_name,
        adapter_version=recipe.runtime_facts.adapter_version,
        serving_abi_version=recipe.runtime_facts.serving_abi_version,
        logical_topology_json=logical_topology_json_payload,
    )


def build_recipe_runtime_build_intent(
    context: RecipePublicationContext,
    *,
    builder_mode: BuilderMode,
    build_pipeline_version: str,
    representation_contract_hash: str | None = None,
) -> RuntimeArtifactBuildIntent:
    return RuntimeArtifactBuildIntent(
        representation_contract_hash=representation_contract_hash,
        builder_mode=builder_mode,
        framework_name=context.framework_name,
        adapter_version=context.adapter_version,
        serving_abi_version=context.serving_abi_version,
        build_pipeline_version=str(build_pipeline_version),
        source_artifact_ref=context.source_artifact_ref,
    )


def build_pure_transform_build_intent(
    context: RecipePublicationContext,
    *,
    build_pipeline_version: str,
    representation_contract_hash: str | None = None,
) -> RuntimeArtifactBuildIntent:
    return build_recipe_runtime_build_intent(
        context,
        builder_mode=BuilderMode.PURE_TRANSFORM,
        build_pipeline_version=build_pipeline_version,
        representation_contract_hash=representation_contract_hash,
    )


def build_binding_finalize_build_intent(
    context: RecipePublicationContext,
    *,
    build_pipeline_version: str,
    representation_contract_hash: str,
) -> RuntimeArtifactBuildIntent:
    return build_recipe_runtime_build_intent(
        context,
        builder_mode=BuilderMode.BINDING_FINALIZE,
        build_pipeline_version=build_pipeline_version,
        representation_contract_hash=str(representation_contract_hash),
    )


__all__ = [
    "RecipePublicationContext",
    "build_binding_finalize_build_intent",
    "build_pure_transform_build_intent",
    "build_recipe_runtime_build_intent",
    "logical_topology_json",
    "logical_topology_json_from_recipe",
    "publication_context_from_recipe",
]
