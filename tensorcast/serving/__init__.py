#  Copyright (c) 2026, TensorCast Team.
"""Curated public facade for TensorCast serving integrations.

The recommended framework integration imports are the narrow modules
``tensorcast.serving.runtime``, ``tensorcast.serving.hosts`` and
``tensorcast.serving.testing``.  This package root exposes only the same stable
concepts lazily; builder/admin/binding helpers live in their explicit modules.
"""

from __future__ import annotations

import importlib
from typing import Any

_RUNTIME_EXPORTS = {
    "AdmissionRejectedError": (
        "tensorcast.serving.integration",
        "AdmissionRejectedError",
    ),
    "AttachFinalizeError": ("tensorcast.serving.integration", "AttachFinalizeError"),
    "AuthorityValidationError": (
        "tensorcast.serving.integration",
        "AuthorityValidationError",
    ),
    "BootstrapPolicy": ("tensorcast.serving.integration", "BootstrapPolicy"),
    "CapabilityMissingError": (
        "tensorcast.serving.integration",
        "CapabilityMissingError",
    ),
    "ConfigConflictError": ("tensorcast.serving.integration", "ConfigConflictError"),
    "OwnershipTransferError": (
        "tensorcast.serving.integration",
        "OwnershipTransferError",
    ),
    "PlacementAdmissionError": (
        "tensorcast.serving.integration",
        "PlacementAdmissionError",
    ),
    "PublishedReplicaProjection": (
        "tensorcast.serving.integration",
        "PublishedReplicaProjection",
    ),
    "ReplicaPublicationError": (
        "tensorcast.serving.integration",
        "ReplicaPublicationError",
    ),
    "ReplicaPublicationPolicy": (
        "tensorcast.serving.integration",
        "ReplicaPublicationPolicy",
    ),
    "ReloadResponseProjection": (
        "tensorcast.serving.integration",
        "ReloadResponseProjection",
    ),
    "RequestContext": ("tensorcast.serving.integration", "RequestContext"),
    "RuntimeAttachment": ("tensorcast.serving.integration", "RuntimeAttachment"),
    "RuntimeEndpointProjection": (
        "tensorcast.serving.integration",
        "RuntimeEndpointProjection",
    ),
    "RuntimeSwapError": ("tensorcast.serving.integration", "RuntimeSwapError"),
    "RuntimeWorkerView": ("tensorcast.serving.integration", "RuntimeWorkerView"),
    "SchemaMismatchError": ("tensorcast.serving.integration", "SchemaMismatchError"),
    "SelectorResolutionError": (
        "tensorcast.serving.integration",
        "SelectorResolutionError",
    ),
    "ServingConfig": ("tensorcast.serving.config", "ServingConfig"),
    "ServingArtifactSelector": ("tensorcast.serving.policy", "ServingSelector"),
    "ServingIntegrationError": (
        "tensorcast.serving.integration",
        "ServingIntegrationError",
    ),
    "ServingPolicy": ("tensorcast.serving.policy", "ServingPolicy"),
    "ServingRuntimeSession": (
        "tensorcast.serving.integration",
        "ServingRuntimeSession",
    ),
    "SourceSelectionProjection": (
        "tensorcast.serving.integration",
        "SourceSelectionProjection",
    ),
    "SourceProviderError": ("tensorcast.serving.integration", "SourceProviderError"),
    "TensorCastServingRuntimeError": (
        "tensorcast.serving.integration",
        "TensorCastServingRuntimeError",
    ),
    "WeightVersionProjection": (
        "tensorcast.serving.integration",
        "WeightVersionProjection",
    ),
    "merge_serving_reload_extra_config": (
        "tensorcast.serving.policy",
        "merge_serving_reload_extra_config",
    ),
    "normalize_serving_reload_request_payload": (
        "tensorcast.serving.policy",
        "normalize_serving_reload_request_payload",
    ),
}

_HOST_EXPORTS = {
    "AdmissionDecision": ("tensorcast.serving.hosts", "AdmissionDecision"),
    "AdmissionPolicy": ("tensorcast.serving.hosts", "AdmissionPolicy"),
    "AdmissionRequest": ("tensorcast.serving.hosts", "AdmissionRequest"),
    "CollectiveHost": ("tensorcast.serving.hosts", "CollectiveHost"),
    "DefaultAdmissionPolicy": ("tensorcast.serving.hosts", "DefaultAdmissionPolicy"),
    "FinalizeHookHost": ("tensorcast.serving.hosts", "FinalizeHookHost"),
    "FinalizePhase": ("tensorcast.serving.hosts", "FinalizePhase"),
    "FinalizePolicy": ("tensorcast.serving.hosts", "FinalizePolicy"),
    "FrameworkHost": ("tensorcast.serving.hosts", "FrameworkHost"),
    "FrameworkIdentity": ("tensorcast.serving.hosts", "FrameworkIdentity"),
    "IntegrationHost": ("tensorcast.serving.hosts", "IntegrationHost"),
    "MaterializationExecutionFacts": (
        "tensorcast.serving.hosts",
        "MaterializationExecutionFacts",
    ),
    "NativeLoadHost": ("tensorcast.serving.hosts", "NativeLoadHost"),
    "ObservabilitySink": ("tensorcast.serving.hosts", "ObservabilitySink"),
    "PlacementAdmissionFacts": ("tensorcast.serving.hosts", "PlacementAdmissionFacts"),
    "PlacementHost": ("tensorcast.serving.hosts", "PlacementHost"),
    "PlacementIdentityFacts": ("tensorcast.serving.hosts", "PlacementIdentityFacts"),
    "PlacementMemberFacts": ("tensorcast.serving.hosts", "PlacementMemberFacts"),
    "RecipeCachePolicy": ("tensorcast.serving.hosts", "RecipeCachePolicy"),
    "RecipeTraceHost": ("tensorcast.serving.hosts", "RecipeTraceHost"),
    "SourceCatalogProvider": ("tensorcast.serving.hosts", "SourceCatalogProvider"),
    "SourceCatalogRequest": ("tensorcast.serving.hosts", "SourceCatalogRequest"),
    "SourceDownloadPolicy": ("tensorcast.serving.hosts", "SourceDownloadPolicy"),
    "SourceHost": ("tensorcast.serving.hosts", "SourceHost"),
    "SourceSelector": ("tensorcast.serving.hosts", "SourceSelector"),
    "TensorCastEvent": ("tensorcast.serving.hosts", "TensorCastEvent"),
    "TensorSurfaceHost": ("tensorcast.serving.hosts", "TensorSurfaceHost"),
    "TorchTensorHost": ("tensorcast.serving.hosts", "TorchTensorHost"),
    "semantic_placement_digest": (
        "tensorcast.serving.hosts",
        "semantic_placement_digest",
    ),
    "serving_placement_from_framework_facts": (
        "tensorcast.serving.hosts",
        "serving_placement_from_framework_facts",
    ),
}

_CONFIG_EXPORTS = {
    "BootstrapSettings": ("tensorcast.serving.config", "BootstrapSettings"),
    "DiagnosticsSettings": ("tensorcast.serving.config", "DiagnosticsSettings"),
    "MaterializationSettings": ("tensorcast.serving.config", "MaterializationSettings"),
    "PreloadSettings": ("tensorcast.serving.preload", "PreloadSettings"),
    "RuntimeConfigProfile": ("tensorcast.serving.runtime", "RuntimeConfigProfile"),
    "RuntimeDaemonSettings": ("tensorcast.serving.runtime", "RuntimeDaemonSettings"),
    "RuntimeGlobalStoreSettings": (
        "tensorcast.serving.runtime",
        "RuntimeGlobalStoreSettings",
    ),
    "RuntimeSettings": ("tensorcast.serving.runtime", "RuntimeSettings"),
    "ServingSelector": ("tensorcast.serving.policy", "ServingSelector"),
    "ServingSettings": ("tensorcast.serving.config", "ServingSettings"),
    "DEFAULT_RUNTIME_PROFILE": (
        "tensorcast.serving.runtime",
        "DEFAULT_RUNTIME_PROFILE",
    ),
    "resolve_runtime_config_profile": (
        "tensorcast.serving.runtime",
        "resolve_runtime_config_profile",
    ),
}

_READINESS_EXPORTS = {
    "ReadinessInventoryAdmissionPolicy": (
        "tensorcast.serving.readiness",
        "ReadinessInventoryAdmissionPolicy",
    ),
    "coerce_finalize_class": ("tensorcast.serving.readiness", "coerce_finalize_class"),
    "coerce_serving_support_level": (
        "tensorcast.serving.readiness",
        "coerce_serving_support_level",
    ),
    "is_binding_finalize_publication_allowlisted": (
        "tensorcast.serving.readiness",
        "is_binding_finalize_publication_allowlisted",
    ),
    "is_pure_transform_publication_allowlisted": (
        "tensorcast.serving.readiness",
        "is_pure_transform_publication_allowlisted",
    ),
    "is_runtime_bind_swap_allowlisted": (
        "tensorcast.serving.readiness",
        "is_runtime_bind_swap_allowlisted",
    ),
    "readiness_family": ("tensorcast.serving.readiness", "readiness_family"),
    "readiness_post_bind_finalize_class": (
        "tensorcast.serving.readiness",
        "readiness_post_bind_finalize_class",
    ),
    "readiness_process_after_load_class": (
        "tensorcast.serving.readiness",
        "readiness_process_after_load_class",
    ),
    "readiness_publication_modes": (
        "tensorcast.serving.readiness",
        "readiness_publication_modes",
    ),
    "readiness_support_level": (
        "tensorcast.serving.readiness",
        "readiness_support_level",
    ),
    "serving_support_level_at_least": (
        "tensorcast.serving.readiness",
        "serving_support_level_at_least",
    ),
    "serving_support_level_display_name": (
        "tensorcast.serving.readiness",
        "serving_support_level_display_name",
    ),
}

_STATE_EXPORTS = {
    "ModelAttributeNames": ("tensorcast.serving.state", "ModelAttributeNames"),
    "ModelAttributeRuntimeState": (
        "tensorcast.serving.state",
        "ModelAttributeRuntimeState",
    ),
    "OneShotRuntimeHook": ("tensorcast.serving.state", "OneShotRuntimeHook"),
    "attachment_generation_key": (
        "tensorcast.serving.state",
        "attachment_generation_key",
    ),
}

_TESTING_EXPORTS = {
    "ConformanceResult": ("tensorcast.serving.testing", "ConformanceResult"),
    "assert_level1_runtime_conformance": (
        "tensorcast.serving.testing",
        "assert_level1_runtime_conformance",
    ),
    "assert_level2_local_bootstrap_conformance": (
        "tensorcast.serving.testing",
        "assert_level2_local_bootstrap_conformance",
    ),
    "assert_level3_retained_preload_conformance": (
        "tensorcast.serving.testing",
        "assert_level3_retained_preload_conformance",
    ),
    "assert_public_runtime_boundary": (
        "tensorcast.serving.testing",
        "assert_public_runtime_boundary",
    ),
}

_SCHEMA_EXPORTS = {
    "PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION": (
        "tensorcast.serving.hosts",
        "PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION",
    ),
    "PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION": (
        "tensorcast.serving.hosts",
        "PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION",
    ),
    "RECIPE_CACHE_POLICY_SCHEMA_VERSION": (
        "tensorcast.serving.hosts",
        "RECIPE_CACHE_POLICY_SCHEMA_VERSION",
    ),
    "PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION": (
        "tensorcast.serving.integration",
        "PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION",
    ),
    "RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION": (
        "tensorcast.serving.integration",
        "RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION",
    ),
    "RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION": (
        "tensorcast.serving.integration",
        "RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION",
    ),
    "SERVING_ARTIFACT_SELECTOR_SCHEMA_VERSION": (
        "tensorcast.serving.policy",
        "SERVING_SELECTOR_SCHEMA_VERSION",
    ),
    "SERVING_POLICY_SCHEMA_VERSION": (
        "tensorcast.serving.policy",
        "SERVING_POLICY_SCHEMA_VERSION",
    ),
    "SOURCE_CATALOG_REQUEST_SCHEMA_VERSION": (
        "tensorcast.serving.hosts",
        "SOURCE_CATALOG_REQUEST_SCHEMA_VERSION",
    ),
    "SOURCE_CATALOG_SCHEMA_VERSION": (
        "tensorcast.serving.hosts",
        "SOURCE_CATALOG_SCHEMA_VERSION",
    ),
    "SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION": (
        "tensorcast.serving.hosts",
        "SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION",
    ),
}

_DTO_EXPORTS = {
    "BootstrapEndpointProjection": (
        "tensorcast.serving.integration",
        "BootstrapEndpointProjection",
    ),
    "BootstrapSummary": ("tensorcast.serving.dto", "BootstrapSummary"),
    "BindingValueRefProjection": (
        "tensorcast.serving.integration",
        "BindingValueRefProjection",
    ),
    "FamilyReadiness": ("tensorcast.serving.dto", "FamilyReadiness"),
    "FrameworkIntegrationContext": (
        "tensorcast.serving.dto",
        "FrameworkIntegrationContext",
    ),
    "MaterializationDiagnosticsProjection": (
        "tensorcast.serving.integration",
        "MaterializationDiagnosticsProjection",
    ),
    "PreparedServingArtifact": ("tensorcast.serving.dto", "PreparedServingArtifact"),
    "ReloadRequestProjection": (
        "tensorcast.serving.integration",
        "ReloadRequestProjection",
    ),
    "RuntimeTensorView": ("tensorcast.serving.dto", "RuntimeTensorView"),
    "ServingPlacement": ("tensorcast.serving.dto", "ServingPlacement"),
    "SourceBoundContractProjection": (
        "tensorcast.serving.integration",
        "SourceBoundContractProjection",
    ),
}

_CURATED_EXPORTS = {
    **_RUNTIME_EXPORTS,
    **_HOST_EXPORTS,
    **_CONFIG_EXPORTS,
    **_READINESS_EXPORTS,
    **_STATE_EXPORTS,
    **_TESTING_EXPORTS,
    **_SCHEMA_EXPORTS,
    **_DTO_EXPORTS,
}

__all__ = sorted(_CURATED_EXPORTS)  # noqa: PLE0605 - curated facade is data-driven.


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _CURATED_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_CURATED_EXPORTS))
