#  Copyright (c) 2026, TensorCast Team.

"""Recipe-oriented serving publication helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

import tensorcast as tc
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.serving_builder import (
    build_binding_finalize_admission_facts,
    build_binding_finalize_publication_bundle,
    build_pure_transform_publication_bundle_from_registered_artifact,
    build_pure_transform_publication_spec,
    prepare_binding_finalize_serving_registration,
    prepare_pure_transform_serving_registration,
)
from tensorcast.api.store.types import CanonicalIndex
from tensorcast.serving.builder.materialization import (
    load_source_tensors_for_recipe,
    materialize_pure_transform_serving_tensors,
)
from tensorcast.types import (
    AssemblyReadinessPolicy,
    AssemblyRequirementSetRef,
    BindingValueRef,
    BuilderMode,
    PureTransformPublicationSpec,
    RepresentationPublishSpec,
    ServingAdmissionFacts,
    ServingBuildIntent,
    ServingPublicationSubject,
    ServingSupportLevel,
)


@dataclass(frozen=True)
class RecipePublicationContext:
    source_artifact_ref: str
    framework_name: str
    adapter_version: str
    serving_abi_version: str
    logical_topology_json: str | None = None


def build_recipe_serving_build_intent(
    context: RecipePublicationContext,
    *,
    builder_mode: BuilderMode,
    build_pipeline_version: str,
    representation_contract_hash: str | None = None,
) -> ServingBuildIntent:
    return ServingBuildIntent(
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
) -> ServingBuildIntent:
    return build_recipe_serving_build_intent(
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
) -> ServingBuildIntent:
    return build_recipe_serving_build_intent(
        context,
        builder_mode=BuilderMode.BINDING_FINALIZE,
        build_pipeline_version=build_pipeline_version,
        representation_contract_hash=str(representation_contract_hash),
    )


def prepare_pure_transform_serving_registration_from_context(
    context: RecipePublicationContext,
    *,
    tensors: Mapping[str, torch.Tensor],
    build_pipeline_version: str,
    source_artifact: object | None = None,
    representation_contract_hash: str | None = None,
    serving_manifest_ref: str | None = None,
    topology_admission_digest: str | None = None,
) -> Any:
    return prepare_pure_transform_serving_registration(
        build_intent=build_pure_transform_build_intent(
            context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        ),
        source_artifact=source_artifact,
        tensors=dict(tensors),
        logical_topology_json=context.logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        topology_admission_digest=topology_admission_digest,
    )


def build_pure_transform_publication_spec_from_context(
    context: RecipePublicationContext,
    *,
    build_pipeline_version: str,
    representation_contract_hash: str | None = None,
    contract_family: str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: Any = None,
    readiness_policy: Any = None,
    structural_view_ids: tuple[str, ...] = (),
) -> PureTransformPublicationSpec:
    return build_pure_transform_publication_spec(
        build_intent=build_pure_transform_build_intent(
            context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        ),
        contract_family=contract_family,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=context.logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        requirements=requirements,
        readiness_policy=readiness_policy,
        structural_view_ids=structural_view_ids,
    )


def build_pure_transform_publication_bundle_from_context(
    context: RecipePublicationContext,
    *,
    serving_artifact: RegisteredArtifact,
    build_pipeline_version: str,
    source_artifact: object | None = None,
    representation_contract_hash: str | None = None,
    contract_family: str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: Any = None,
    readiness_policy: Any = None,
    structural_view_ids: tuple[str, ...] = (),
) -> RepresentationPublishSpec:
    return build_pure_transform_publication_bundle_from_registered_artifact(
        build_intent=build_pure_transform_build_intent(
            context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        ),
        source_artifact=source_artifact,
        contract_family=contract_family,
        serving_artifact=serving_artifact,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=context.logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        requirements=requirements,
        readiness_policy=readiness_policy,
        structural_view_ids=structural_view_ids,
    )


def prepare_binding_finalize_serving_registration_from_context(
    context: RecipePublicationContext,
    *,
    tensors: dict[str, torch.Tensor],
    build_pipeline_version: str,
    representation_contract_hash: str,
    serving_manifest_ref: str | None = None,
    topology_admission_digest: str | None = None,
) -> Any:
    return prepare_binding_finalize_serving_registration(
        build_intent=build_binding_finalize_build_intent(
            context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        ),
        tensors=dict(tensors),
        representation_contract_hash=representation_contract_hash,
        logical_topology_json=context.logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        topology_admission_digest=topology_admission_digest,
    )


def build_binding_finalize_admission_facts_from_context(
    *,
    support_level: ServingSupportLevel,
    topology_admission_digest: str | None = None,
    same_binding_fast_path_validated: bool = True,
) -> ServingAdmissionFacts:
    return build_binding_finalize_admission_facts(
        support_level=support_level,
        topology_admission_digest=topology_admission_digest,
        same_binding_fast_path_validated=same_binding_fast_path_validated,
    )


def build_binding_finalize_publication_bundle_from_context(
    context: RecipePublicationContext,
    *,
    publication_subject: ServingPublicationSubject | BindingValueRef,
    canonical_index: CanonicalIndex,
    build_pipeline_version: str,
    representation_contract_hash: str,
    contract_family: str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    structural_view_ids: tuple[str, ...] = (),
    admission_facts: ServingAdmissionFacts | None = None,
) -> RepresentationPublishSpec:
    if admission_facts is None:
        raise ValueError(
            "binding finalize publication requires explicit admission_facts"
        )
    return build_binding_finalize_publication_bundle(
        build_intent=build_binding_finalize_build_intent(
            context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        ),
        contract_family=contract_family,
        publication_subject=publication_subject,
        canonical_index=canonical_index,
        representation_contract_hash=representation_contract_hash,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=context.logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        requirements=requirements,
        readiness_policy=readiness_policy,
        structural_view_ids=structural_view_ids,
        admission_facts=admission_facts,
    )


def complete_pure_transform_recipe_publication(
    recipe: Any,
    *,
    publication_context: RecipePublicationContext,
    source_artifact: Any,
    build_pipeline_version: str,
    source_tensors: Mapping[str, torch.Tensor] | None = None,
    materialization_device: str | torch.device = "cpu",
    representation_contract_hash: str | None = None,
    contract_family: str | None = None,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: Any = None,
    device: int | torch.device | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    readiness_policy: Any = None,
    structural_view_ids: tuple[str, ...] = (),
    source_contribution_device: str | int | None = None,
    source_contribution_artifacts: Any = None,
    timeout_s: float | None = None,
) -> Any:
    resolved_source_tensors = (
        load_source_tensors_for_recipe(
            recipe,
            source_artifact,
            target_device=materialization_device,
        )
        if source_tensors is None
        else {str(name): tensor for name, tensor in dict(source_tensors).items()}
    )
    serving_tensors = materialize_pure_transform_serving_tensors(
        recipe,
        resolved_source_tensors,
        target_device=materialization_device,
    )
    return tc.complete_pure_transform_publication(
        serving_tensors,
        build_intent=build_pure_transform_build_intent(
            publication_context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        ),
        source_artifact=source_artifact,
        contract_family=contract_family,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        device=device,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=publication_context.logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        readiness_policy=readiness_policy,
        structural_view_ids=tuple(
            str(view_id).strip()
            for view_id in structural_view_ids
            if str(view_id).strip()
        ),
        source_contribution_device=source_contribution_device,
        source_contribution_artifacts=source_contribution_artifacts,
        timeout_s=timeout_s,
    )


__all__ = [
    "RecipePublicationContext",
    "build_binding_finalize_admission_facts_from_context",
    "build_binding_finalize_build_intent",
    "build_binding_finalize_publication_bundle_from_context",
    "complete_pure_transform_recipe_publication",
    "build_pure_transform_build_intent",
    "build_pure_transform_publication_bundle_from_context",
    "build_pure_transform_publication_spec_from_context",
    "build_recipe_serving_build_intent",
    "prepare_binding_finalize_serving_registration_from_context",
    "prepare_pure_transform_serving_registration_from_context",
]
