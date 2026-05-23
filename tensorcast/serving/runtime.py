#  Copyright (c) 2026, TensorCast Team.

"""Framework-facing TensorCast serving runtime API."""

from __future__ import annotations

from tensorcast.serving._runtime_impl.lifecycle import ServingRuntimeSession
from tensorcast.serving.config import (
    ArtifactBindStartPlan,
    ReplicaPublicationPolicy,
    RetainedBindingAcquireStartPlan,
    ServingConfig,
    ServingStartPlan,
    ServingStartPlanError,
    SourceBootstrapToBindingStartPlan,
    plan_serving_start,
)
from tensorcast.serving.errors import (
    AdmissionRejectedError,
    ArtifactLocatorResolutionError,
    AttachFinalizeError,
    AuthorityValidationError,
    CapabilityMissingError,
    ConfigConflictError,
    OwnershipTransferError,
    PlacementAdmissionError,
    PolicyMismatchError,
    PublicationRequiredError,
    ReplicaPublicationError,
    RuntimeSwapError,
    SchemaMismatchError,
    ServingIntegrationError,
    SourceProviderError,
    TensorCastServingRuntimeError,
)
from tensorcast.serving.hosts import SourceSelector
from tensorcast.serving.policy import (
    ServingArtifactLocator,
    ServingPolicy,
    merge_serving_reload_extra_config,
    normalize_serving_reload_request_payload,
)
from tensorcast.serving.runtime_attachment import RuntimeAttachment
from tensorcast.serving.runtime_config import (
    DEFAULT_RUNTIME_PROFILE,
    RuntimeConfigProfile,
    RuntimeDaemonSettings,
    RuntimeGlobalStoreSettings,
    RuntimeSettings,
    resolve_runtime_config_profile,
)
from tensorcast.serving.runtime_intent import (
    BootstrapPolicy,
    ExistingServingArtifact,
    LocalSourceBootstrap,
    RequestContext,
    RetainedBindingAcquire,
)
from tensorcast.serving.runtime_view import RuntimeWorkerView

__all__ = [
    "AdmissionRejectedError",
    "ArtifactBindStartPlan",
    "ArtifactLocatorResolutionError",
    "AttachFinalizeError",
    "AuthorityValidationError",
    "BootstrapPolicy",
    "CapabilityMissingError",
    "ConfigConflictError",
    "DEFAULT_RUNTIME_PROFILE",
    "ExistingServingArtifact",
    "LocalSourceBootstrap",
    "OwnershipTransferError",
    "PlacementAdmissionError",
    "PolicyMismatchError",
    "PublicationRequiredError",
    "ReplicaPublicationError",
    "ReplicaPublicationPolicy",
    "RequestContext",
    "RetainedBindingAcquire",
    "RetainedBindingAcquireStartPlan",
    "RuntimeAttachment",
    "RuntimeConfigProfile",
    "RuntimeDaemonSettings",
    "RuntimeGlobalStoreSettings",
    "RuntimeSettings",
    "RuntimeSwapError",
    "RuntimeWorkerView",
    "SchemaMismatchError",
    "ServingArtifactLocator",
    "ServingConfig",
    "ServingIntegrationError",
    "ServingPolicy",
    "ServingRuntimeSession",
    "ServingStartPlan",
    "ServingStartPlanError",
    "SourceBootstrapToBindingStartPlan",
    "SourceProviderError",
    "SourceSelector",
    "TensorCastServingRuntimeError",
    "merge_serving_reload_extra_config",
    "normalize_serving_reload_request_payload",
    "plan_serving_start",
    "resolve_runtime_config_profile",
]
