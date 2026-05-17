#  Copyright (c) 2026, TensorCast Team.

"""Public TensorCast serving artifact runtime abstractions."""

from tensorcast.serving.binding_runtime import (
    bind_serving_artifact,
    build_materialization_execution_context,
    swap_serving_artifact,
)
from tensorcast.serving.builder.compiler import (
    CompiledServingRecipe,
    RecipeCompileIdentity,
    RecipeCompileInputs,
    TensorcastLogicalTopology,
    TensorcastSemanticValidationSpec,
    TensorcastServingFacts,
    TensorSchemaEntry,
    compute_recipe_compile_key,
)
from tensorcast.serving.builder.compiler import (
    compile_serving_recipe as compile_recipe_from_inputs,
)
from tensorcast.serving.builder.compiler import (
    compiled_recipe_realization_plan_count as builder_realization_plan_count,
)
from tensorcast.serving.builder.materialization import (
    allocate_tensors_from_schema,
    apply_copy_plan,
    tensorcast_view_slices_from_trace_plan,
    validate_dst_coverage,
    validate_source_tensor_names,
)
from tensorcast.serving.builder.publication import (
    RecipePublicationContext,
    build_binding_finalize_build_intent,
    build_pure_transform_build_intent,
)
from tensorcast.serving.builder.recipe_cache import (
    load_compiled_recipe_cache,
    write_compiled_recipe_cache,
)
from tensorcast.serving.builder.recipe_validation import (
    validate_recipe_for_builder_mode,
)
from tensorcast.serving.builder.semantic_validation import (
    evaluate_semantic_validation_spec,
)
from tensorcast.serving.builder.source_catalog import (
    SourceCatalog,
    resolve_source_artifact_ref,
    source_catalog_from_selected_safetensors,
)
from tensorcast.serving.builder.tensor_schema import (
    validate_tensor_schema_against_tensors,
)
from tensorcast.serving.builder.trace_cache import (
    dump_trace_plan_debug,
    load_trace_plan_cache,
    write_trace_plan_cache,
)
from tensorcast.serving.builder.trace_ir import TracePlan
from tensorcast.serving.config import (
    BootstrapSettings,
    DiagnosticsSettings,
    MaterializationSettings,
    ServingConfig,
    ServingSettings,
)
from tensorcast.serving.contract import (
    collect_runtime_tensor_schema,
    compute_runtime_representation_contract_hash,
    compute_runtime_tensor_schema_hash,
)
from tensorcast.serving.dto import (
    BootstrapSummary,
    FamilyReadiness,
    FrameworkAdapter,
    FrameworkIntegrationContext,
    PreparedServingArtifact,
    RuntimeTensorView,
    ServingPlacement,
)
from tensorcast.serving.local_ready import (
    LocalReadyBindingRealizationResult,
    binding_value_verification_state_name,
    build_binding_layout_for_recipe,
    canonical_index_entries_from_tensor_schema,
    canonical_index_from_recipe,
    compiled_recipe_realization_plan_count,
    compute_serving_binding_tensor_schema_hash,
    create_local_ready_binding,
    freeze_local_ready_binding,
    logical_topology_json_from_recipe,
    materialized_tensor_schema,
    prepare_local_ready_serving,
    prepare_same_binding_manifest_carrier,
    publication_context_from_recipe,
    realization_plan_proto_with_manifest,
    realize_local_ready_binding_from_source,
    serving_binding_tensor_schema,
    source_view_for_recipe,
    tensorcast_view_slice_count,
)
from tensorcast.serving.policy import ServingPolicy, ServingSelector
from tensorcast.serving.preload import (
    AttachedPreloadBinding,
    BindingPromotionResult,
    BorrowedPreloadLease,
    ExternalPreloadAuthority,
    ExternalPreloadExpectedDigests,
    ParsedExternalPreloadAuthority,
    PreloadSettings,
    RuntimePreloadAttachmentHandle,
    acquire_local_ready_preload_lease,
    acquire_preload_lease,
    acquire_retained_serving_binding,
    external_preload_extra_from_prefetched_binding,
    external_preload_extra_json,
    external_preload_mode,
    external_preload_trusted_reservation_bytes,
    parse_external_preload_authority,
    promote_current_value_and_wait,
)
from tensorcast.serving.recipe_build import (
    RecipeBuildIdentity,
    RecipeBuildSession,
    stable_recipe_build_hash,
)
from tensorcast.serving.recipe_build import (
    compute_recipe_cache_key as compute_recipe_build_cache_key,
)
from tensorcast.serving.recipe_build import (
    compute_trace_cache_key as compute_trace_build_cache_key,
)
from tensorcast.serving.recipe_build import (
    recipe_cache_path as recipe_build_cache_path,
)
from tensorcast.serving.recipe_build import (
    trace_cache_path as trace_build_cache_path,
)
from tensorcast.serving.resolver import (
    ResolvedServingArtifact,
    ServingArtifactResolver,
    is_reserved_serving_tensor_name,
    resolve_serving_artifact,
)
from tensorcast.serving.runtime import (
    DEFAULT_RUNTIME_PROFILE,
    RuntimeConfigProfile,
    RuntimeDaemonSettings,
    RuntimeGlobalStoreSettings,
    RuntimeSettings,
    resolve_runtime_config_profile,
)
from tensorcast.serving.runtime_contract import (
    MIN_SOURCE_BOUND_CONTRACT_VERSION,
    REQUIRED_SOURCE_BOUND_CAPABILITIES,
    SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4,
    SourceBoundContractState,
    read_source_bound_contract_state,
)
from tensorcast.serving.session import ServingBindingSession, ServingBindingState

__all__ = [
    "BootstrapSettings",
    "BootstrapSummary",
    "AttachedPreloadBinding",
    "allocate_tensors_from_schema",
    "apply_copy_plan",
    "BindingPromotionResult",
    "BorrowedPreloadLease",
    "CompiledServingRecipe",
    "DiagnosticsSettings",
    "DEFAULT_RUNTIME_PROFILE",
    "ExternalPreloadAuthority",
    "ExternalPreloadExpectedDigests",
    "FamilyReadiness",
    "FrameworkIntegrationContext",
    "FrameworkAdapter",
    "LocalReadyBindingRealizationResult",
    "MaterializationSettings",
    "MIN_SOURCE_BOUND_CONTRACT_VERSION",
    "ParsedExternalPreloadAuthority",
    "PreparedServingArtifact",
    "PreloadSettings",
    "RecipeBuildIdentity",
    "RecipeBuildSession",
    "RecipeCompileIdentity",
    "RecipeCompileInputs",
    "RecipePublicationContext",
    "ResolvedServingArtifact",
    "RuntimeConfigProfile",
    "RuntimeDaemonSettings",
    "RuntimeGlobalStoreSettings",
    "RuntimePreloadAttachmentHandle",
    "RuntimeSettings",
    "RuntimeTensorView",
    "is_reserved_serving_tensor_name",
    "REQUIRED_SOURCE_BOUND_CAPABILITIES",
    "SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4",
    "ServingBindingSession",
    "ServingBindingState",
    "ServingArtifactResolver",
    "ServingConfig",
    "ServingPlacement",
    "ServingPolicy",
    "ServingSelector",
    "ServingSettings",
    "SourceBoundContractState",
    "SourceCatalog",
    "TensorSchemaEntry",
    "TensorcastLogicalTopology",
    "TensorcastSemanticValidationSpec",
    "TensorcastServingFacts",
    "TracePlan",
    "binding_value_verification_state_name",
    "build_binding_layout_for_recipe",
    "bind_serving_artifact",
    "build_materialization_execution_context",
    "canonical_index_entries_from_tensor_schema",
    "canonical_index_from_recipe",
    "compile_recipe_from_inputs",
    "builder_realization_plan_count",
    "compiled_recipe_realization_plan_count",
    "compute_recipe_compile_key",
    "compute_recipe_build_cache_key",
    "compute_runtime_representation_contract_hash",
    "collect_runtime_tensor_schema",
    "compute_runtime_tensor_schema_hash",
    "compute_serving_binding_tensor_schema_hash",
    "compute_trace_build_cache_key",
    "create_local_ready_binding",
    "freeze_local_ready_binding",
    "build_binding_finalize_build_intent",
    "build_pure_transform_build_intent",
    "acquire_local_ready_preload_lease",
    "acquire_preload_lease",
    "acquire_retained_serving_binding",
    "external_preload_extra_from_prefetched_binding",
    "external_preload_extra_json",
    "external_preload_mode",
    "external_preload_trusted_reservation_bytes",
    "dump_trace_plan_debug",
    "evaluate_semantic_validation_spec",
    "load_compiled_recipe_cache",
    "load_trace_plan_cache",
    "parse_external_preload_authority",
    "logical_topology_json_from_recipe",
    "materialized_tensor_schema",
    "prepare_local_ready_serving",
    "prepare_same_binding_manifest_carrier",
    "promote_current_value_and_wait",
    "publication_context_from_recipe",
    "read_source_bound_contract_state",
    "recipe_build_cache_path",
    "realization_plan_proto_with_manifest",
    "realize_local_ready_binding_from_source",
    "resolve_serving_artifact",
    "resolve_runtime_config_profile",
    "resolve_source_artifact_ref",
    "serving_binding_tensor_schema",
    "source_catalog_from_selected_safetensors",
    "source_view_for_recipe",
    "stable_recipe_build_hash",
    "swap_serving_artifact",
    "tensorcast_view_slice_count",
    "tensorcast_view_slices_from_trace_plan",
    "trace_build_cache_path",
    "validate_recipe_for_builder_mode",
    "validate_dst_coverage",
    "validate_source_tensor_names",
    "validate_tensor_schema_against_tensors",
    "write_compiled_recipe_cache",
    "write_trace_plan_cache",
]
