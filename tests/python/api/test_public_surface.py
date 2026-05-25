#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import importlib
import importlib.util
import inspect

import tensorcast as tc
import tensorcast.artifact_runtime.diagnostics as tc_runtime_diagnostics
import tensorcast.artifact_runtime.readiness as tc_runtime_readiness
from tensorcast.api.store import (
    ArtifactRealizationHandle,
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
    PrefetchHandoff,
    PrefetchHandoffMemberFailure,
    PrefetchHandoffSet,
    RealizationTarget,
    RealizationTargetSet,
    RuntimeArtifactBuildIntent,
    RuntimeArtifactManifest,
    RuntimeArtifactPolicy,
    RuntimeBindingMemberRef,
    RuntimeBindingReadiness,
    RuntimeBindingResolvedLayout,
    RuntimeBindingSourceKind,
    RuntimeBindingSourceMemberRef,
    RuntimeBindingSourceRef,
    RuntimeBindingSourceReuseDecision,
    RuntimeBindingSourceReuseMode,
    RuntimeRealizationSpecCacheEntry,
    RuntimeTopologyRef,
    Store,
)
from tensorcast.artifact_runtime.attachment import (
    RuntimeAttachment,
    RuntimeBindingState,
)
from tensorcast.artifact_runtime.config import (
    RuntimeArtifactLocator,
    RuntimeStartPlanError,
    TensorCastRuntimeConfig,
    plan_runtime_start,
)
from tensorcast.artifact_runtime.diagnostics import RuntimeRealizationReport
from tensorcast.artifact_runtime.host import (
    RuntimeAdmissionDecision,
    RuntimeAdmissionPolicy,
    RuntimeAdmissionRequest,
    RuntimeHostCapabilities,
    RuntimePlacement,
    RuntimeProfile,
    RuntimeTensorView,
)
from tensorcast.artifact_runtime.intent import RuntimeRequestContext
from tensorcast.artifact_runtime.locator import ArtifactLocator
from tensorcast.artifact_runtime.policy import RuntimePolicy
from tensorcast.artifact_runtime.publication.actions import (
    RuntimeReplicaPublicationSettings,
    project_runtime_replica_publication_state,
    publish_runtime_replica,
    retire_runtime_replica,
    runtime_replica_publication_settings,
)
from tensorcast.artifact_runtime.reload import (
    merge_runtime_reload_extra_config,
    normalize_runtime_reload_request_payload,
    reload_runtime_attachment,
)
from tensorcast.artifact_runtime.state import (
    ModelAttributeNames,
    ModelAttributeRuntimeState,
    OneShotRuntimeHook,
)
from tensorcast.artifact_runtime.view import (
    BindingValueRefProjection,
    RuntimeEndpointProjection,
    RuntimeWorkerView,
    SourceSelectionProjection,
    WeightVersionProjection,
    aggregate_runtime_view_outputs,
)
from tensorcast.retained_realization import (
    RetainedRealizationClaim,
    RetainedRealizationExpectedDigests,
    retained_realization_claim_extra_from_handoff,
    retained_realization_claim_extra_json_from_handoff,
)


def test_tensorcast_exports_artifact_helpers() -> None:
    assert hasattr(tc, "artifact")
    assert callable(tc.artifact)
    assert hasattr(tc, "artifact_async")
    assert callable(tc.artifact_async)
    assert tc.ArtifactRealizationSpec is ArtifactRealizationSpec
    assert tc.ArtifactRealizationHandle is ArtifactRealizationHandle
    assert tc.ArtifactRealizationReport is ArtifactRealizationReport
    assert "ArtifactRealizationSpec" in tc.__all__
    assert tc.RetainedRealizationClaim is RetainedRealizationClaim
    assert "RetainedRealizationClaim" in tc.__all__
    assert tc.RetainedRealizationExpectedDigests is RetainedRealizationExpectedDigests
    assert "RetainedRealizationExpectedDigests" in tc.__all__
    assert callable(tc.parse_retained_realization_claim)
    assert (
        tc.retained_realization_claim_extra_from_handoff
        is retained_realization_claim_extra_from_handoff
    )
    assert "retained_realization_claim_extra_from_handoff" in tc.__all__
    assert (
        tc.retained_realization_claim_extra_json_from_handoff
        is retained_realization_claim_extra_json_from_handoff
    )
    assert "retained_realization_claim_extra_json_from_handoff" in tc.__all__
    assert tc.RuntimeAttachment is RuntimeAttachment
    assert tc.RuntimeBindingState is RuntimeBindingState
    assert tc.RuntimeRequestContext is RuntimeRequestContext
    assert tc.RuntimeAdmissionDecision is RuntimeAdmissionDecision
    assert tc.RuntimeAdmissionPolicy is RuntimeAdmissionPolicy
    assert tc.RuntimeAdmissionRequest is RuntimeAdmissionRequest
    assert tc.RuntimeHostCapabilities is RuntimeHostCapabilities
    assert tc.RuntimePlacement is RuntimePlacement
    assert tc.RuntimeProfile is RuntimeProfile
    assert tc.RuntimeTensorView is RuntimeTensorView
    assert tc.ArtifactLocator is ArtifactLocator
    assert tc.RuntimeArtifactLocator is RuntimeArtifactLocator
    assert tc.RuntimePolicy is RuntimePolicy
    assert tc.RuntimeRealizationReport is RuntimeRealizationReport
    assert tc.RuntimeArtifactBuildIntent is RuntimeArtifactBuildIntent
    assert tc.RuntimeArtifactManifest is RuntimeArtifactManifest
    assert tc.RuntimeArtifactPolicy is RuntimeArtifactPolicy
    assert tc.RealizationTarget is RealizationTarget
    assert tc.RealizationTargetSet is RealizationTargetSet
    assert tc.RuntimeBindingMemberRef is RuntimeBindingMemberRef
    assert tc.RuntimeBindingReadiness is RuntimeBindingReadiness
    assert tc.RuntimeBindingResolvedLayout is RuntimeBindingResolvedLayout
    assert tc.RuntimeBindingSourceKind is RuntimeBindingSourceKind
    assert tc.RuntimeBindingSourceMemberRef is RuntimeBindingSourceMemberRef
    assert tc.RuntimeBindingSourceRef is RuntimeBindingSourceRef
    assert tc.RuntimeBindingSourceReuseDecision is RuntimeBindingSourceReuseDecision
    assert tc.RuntimeBindingSourceReuseMode is RuntimeBindingSourceReuseMode
    assert tc.RuntimeRealizationSpecCacheEntry is RuntimeRealizationSpecCacheEntry
    assert tc.RuntimeTopologyRef is RuntimeTopologyRef
    assert tc.PrefetchHandoff is PrefetchHandoff
    assert tc.PrefetchHandoffMemberFailure is PrefetchHandoffMemberFailure
    assert tc.PrefetchHandoffSet is PrefetchHandoffSet
    for removed_name in (
        "ServingBindingTarget",
        "ServingBindingSetTarget",
        "PrefetchedServingBinding",
        "PrefetchedServingBindingSet",
        "ServingBuildIntent",
        "ServingArtifactManifest",
        "ServingRuntimePolicy",
        "ServingRealizationReport",
        "ServingBindingMemberRef",
        "ServingBindingReadiness",
        "ServingBindingResolvedLayout",
        "ServingBindingResolvedSpecCacheEntry",
        "ServingBindingSourceKind",
        "ServingBindingSourceMemberRef",
        "ServingBindingSourceRef",
        "ServingBindingSourceReuseDecision",
        "ServingBindingSourceReuseMode",
        "ServingTopologyRef",
        "RegisteredServingPublication",
        "PreparedServingRegistration",
        "ServingPublicationSubject",
        "ServingAdmissionFacts",
        "ServingSupportLevel",
        "build_serving_publication_bundle",
        "build_serving_publication_bundle_from_registered_artifact",
        "build_serving_manifest_ref",
        "SERVING_BUILD_DIGEST_VERSION",
        "compute_serving_tensor_schema_hash",
        "count_canonical_serving_tensors",
        "prepare_pure_transform_serving_registration",
        "prepare_binding_finalize_serving_registration",
        "prepare_serving_registration",
        "parse_serving_manifest_ref",
        "plan_serving_binding_source_reuse",
        "retained_realization_claim_extra_from_prefetched_binding",
        "retained_realization_claim_extra_json",
    ):
        assert removed_name not in tc.__all__
        assert not hasattr(tc, removed_name)
    assert tc.RuntimeStartPlanError is RuntimeStartPlanError
    assert tc.TensorCastRuntimeConfig is TensorCastRuntimeConfig
    assert tc.plan_runtime_start is plan_runtime_start
    assert tc.runtime is importlib.import_module("tensorcast.runtime")
    assert "runtime" in tc.__all__
    assert "serving" not in tc.__all__
    assert not hasattr(tc, "serving")
    assert "RuntimeAttachment" in tc.__all__
    assert "RuntimeBindingState" in tc.__all__
    assert "RuntimeRequestContext" in tc.__all__
    assert "RuntimeAdmissionDecision" in tc.__all__
    assert "RuntimeAdmissionPolicy" in tc.__all__
    assert "RuntimeAdmissionRequest" in tc.__all__
    assert "RuntimeHostCapabilities" in tc.__all__
    assert "RuntimePlacement" in tc.__all__
    assert "RuntimeProfile" in tc.__all__
    assert "RuntimeTensorView" in tc.__all__
    assert "ArtifactLocator" in tc.__all__
    assert "RuntimeArtifactLocator" in tc.__all__
    assert "RuntimePolicy" in tc.__all__
    assert "RuntimeRealizationReport" in tc.__all__
    assert "RuntimeArtifactBuildIntent" in tc.__all__
    assert "RuntimeArtifactManifest" in tc.__all__
    assert "RuntimeArtifactPolicy" in tc.__all__
    assert "RealizationTarget" in tc.__all__
    assert "RealizationTargetSet" in tc.__all__
    assert "RuntimeBindingMemberRef" in tc.__all__
    assert "RuntimeBindingReadiness" in tc.__all__
    assert "RuntimeBindingResolvedLayout" in tc.__all__
    assert "RuntimeBindingSourceKind" in tc.__all__
    assert "RuntimeBindingSourceMemberRef" in tc.__all__
    assert "RuntimeBindingSourceRef" in tc.__all__
    assert "RuntimeBindingSourceReuseDecision" in tc.__all__
    assert "RuntimeBindingSourceReuseMode" in tc.__all__
    assert "RuntimeRealizationSpecCacheEntry" in tc.__all__
    assert "RuntimeTopologyRef" in tc.__all__
    assert "PrefetchHandoff" in tc.__all__
    assert "PrefetchHandoffMemberFailure" in tc.__all__
    assert "PrefetchHandoffSet" in tc.__all__
    assert "RuntimeStartPlanError" in tc.__all__
    assert "TensorCastRuntimeConfig" in tc.__all__
    assert "plan_runtime_start" in tc.__all__
    assert "ServingRealizationReport" not in tc_runtime_diagnostics.__all__
    assert not hasattr(tc_runtime_diagnostics, "ServingRealizationReport")
    for removed_module in (
        "tensorcast.serving",
        "tensorcast.serving.runtime",
        "tensorcast.serving.diagnostics",
        "tensorcast.serving.config",
        "tensorcast.serving.contract",
        "tensorcast.serving.hosts",
        "tensorcast.serving.readiness",
        "tensorcast.serving.runtime_attachment",
        "tensorcast.serving.runtime_config",
        "tensorcast.serving.runtime_contract",
        "tensorcast.serving.runtime_intent",
        "tensorcast.serving.runtime_view",
        "tensorcast.serving.policy",
        "tensorcast.serving.session",
        "tensorcast.serving.source_catalog",
        "tensorcast.serving.state",
        "tensorcast.serving._runtime_impl",
        "tensorcast.serving._runtime_impl.lifecycle",
        "tensorcast.serving.admin",
        "tensorcast.serving.artifact_manifest",
        "tensorcast.serving.binding_runtime",
        "tensorcast.serving.builder",
        "tensorcast.serving.builder.tensor_parity",
        "tensorcast.serving.dto",
        "tensorcast.serving.errors",
        "tensorcast.serving.local_ready",
        "tensorcast.serving.replica_publication",
        "tensorcast.serving.resolver",
        "tensorcast.serving.retained_binding",
        "tensorcast.serving.testing",
    ):
        try:
            spec = importlib.util.find_spec(removed_module)
        except ModuleNotFoundError:
            spec = None
        assert spec is None
    assert tc.ModelAttributeNames is ModelAttributeNames
    assert tc.ModelAttributeRuntimeState is ModelAttributeRuntimeState
    assert tc.OneShotRuntimeHook is OneShotRuntimeHook
    assert "ModelAttributeNames" in tc.__all__
    assert "ModelAttributeRuntimeState" in tc.__all__
    assert "OneShotRuntimeHook" in tc.__all__
    assert tc.BindingValueRefProjection is BindingValueRefProjection
    assert tc.RuntimeEndpointProjection is RuntimeEndpointProjection
    assert tc.RuntimeWorkerView is RuntimeWorkerView
    assert tc.SourceSelectionProjection is SourceSelectionProjection
    assert tc.WeightVersionProjection is WeightVersionProjection
    assert "BindingValueRefProjection" in tc.__all__
    assert "RuntimeEndpointProjection" in tc.__all__
    assert "RuntimeWorkerView" in tc.__all__
    assert "SourceSelectionProjection" in tc.__all__
    assert "WeightVersionProjection" in tc.__all__
    assert tc.aggregate_runtime_view_outputs is aggregate_runtime_view_outputs
    assert "aggregate_runtime_view_outputs" in tc.__all__
    assert tc.publish_runtime_replica is publish_runtime_replica
    assert tc.project_runtime_replica_publication_state is (
        project_runtime_replica_publication_state
    )
    assert tc.retire_runtime_replica is retire_runtime_replica
    assert "publish_runtime_replica" in tc.__all__
    assert "retire_runtime_replica" in tc.__all__
    assert tc.RuntimeReplicaPublicationSettings is (RuntimeReplicaPublicationSettings)
    assert tc.runtime_replica_publication_settings is (
        runtime_replica_publication_settings
    )
    assert "RuntimeReplicaPublicationSettings" in tc.__all__
    assert "runtime_replica_publication_settings" in tc.__all__
    assert tc.reload_runtime_attachment is reload_runtime_attachment
    assert "reload_runtime_attachment" in tc.__all__
    assert tc.merge_runtime_reload_extra_config is merge_runtime_reload_extra_config
    assert tc.normalize_runtime_reload_request_payload is (
        normalize_runtime_reload_request_payload
    )
    assert "merge_runtime_reload_extra_config" in tc.__all__
    assert "normalize_runtime_reload_request_payload" in tc.__all__
    assert callable(tc_runtime_diagnostics.binding_layout_tensor_count)
    assert tc_runtime_readiness.ReadinessInventoryAdmissionPolicy is not None


def test_tensorcast_exports_programmable_primitives() -> None:
    assert hasattr(tc, "context")
    assert callable(tc.context)
    assert hasattr(tc, "CallContext")
    assert hasattr(tc, "CollectiveLoadGroup")
    assert hasattr(tc, "Operation")
    assert hasattr(tc, "plan")
    assert callable(tc.plan)
    assert hasattr(tc, "Plan")


def test_store_register_and_put_accept_policy_argument() -> None:
    for func in (
        Store.register,
        Store.register_async,
        Store.put,
        Store.put_async,
        tc.register,
        tc.register_async,
        tc.put,
        tc.put_async,
    ):
        sig = inspect.signature(func)
        assert "policy" in sig.parameters
        assert sig.parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
