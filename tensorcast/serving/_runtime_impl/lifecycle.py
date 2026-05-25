#  Copyright (c) 2026, TensorCast Team.
"""Serving lifecycle implementation for TensorCast framework integrations.

New framework integrations should prefer the narrow public modules
``tensorcast.serving.runtime``, ``tensorcast.serving.hosts`` and
``tensorcast.serving.testing``.  This module owns lifecycle orchestration and
keeps low-level helpers out of the framework-facing host/runtime modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

import tensorcast as tc
from tensorcast.api.store.common import canonical_index_to_bytes
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationHandle,
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
    RealizationTargetPlan,
    artifact_realization_report_to_dict,
    emit_artifact_realization_profile_event,
    envelope_for_runtime_attachment,
    model_runtime_report_for,
    report_for_runtime_attachment,
    resolve_artifact_selection,
)
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.serving import binding_runtime as tc_binding_runtime
from tensorcast.serving import config as tc_config
from tensorcast.serving import contract as tc_contract
from tensorcast.serving import diagnostics as tc_diagnostics
from tensorcast.serving import dto as tc_dto
from tensorcast.serving import errors as tc_errors
from tensorcast.serving import hosts as tc_hosts
from tensorcast.serving import local_ready as tc_local_ready
from tensorcast.serving import policy as tc_policy
from tensorcast.serving import readiness as tc_readiness
from tensorcast.serving import recipe_build as tc_recipe_build
from tensorcast.serving import replica_publication as tc_replica_publication
from tensorcast.serving import retained_binding as tc_retained_binding
from tensorcast.serving import runtime_attachment as tc_runtime_attachment
from tensorcast.serving import runtime_config as tc_runtime_config
from tensorcast.serving import runtime_contract as tc_runtime_contract
from tensorcast.serving import runtime_intent as tc_runtime_intent
from tensorcast.serving import runtime_view as tc_runtime_view
from tensorcast.serving import session as tc_session
from tensorcast.serving import source_catalog as tc_source_catalog
from tensorcast.serving.builder import compiler as tc_compiler
from tensorcast.serving.builder import materialization as tc_materialization
from tensorcast.serving.builder import publication as tc_publication
from tensorcast.serving.builder import recipe_cache as tc_recipe_cache
from tensorcast.serving.builder import recipe_validation as tc_recipe_validation
from tensorcast.serving.builder import semantic_validation as tc_semantic_validation
from tensorcast.serving.builder import tensor_schema as tc_tensor_schema
from tensorcast.serving.builder import trace_cache as tc_trace_cache
from tensorcast.serving.builder.compiler import TracePlan
from tensorcast.serving.resolver import (
    ResolvedServingArtifact,
    ServingArtifactResolver,
    canonical_index_from_descriptor,
    is_reserved_serving_tensor_name,
)
from tensorcast.types import (
    CollectivePolicy,
    FinalizeClass,
    ServingSupportLevel,
)

ArtifactError = tc.ArtifactError
BindingUpdateEpoch = tc.BindingUpdateEpoch
BindingReservationCapability = tc.BindingReservationCapability
BindingValueRef = tc.BindingValueRef
BuilderMode = tc.BuilderMode
CompiledServingRecipe = tc_compiler.CompiledServingRecipe
BindingFinalizeMaterializationResult = (
    tc_materialization.BindingFinalizeMaterializationResult
)
DEFAULT_RUNTIME_PROFILE = tc_runtime_config.DEFAULT_RUNTIME_PROFILE
LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION = (
    tc_local_ready.LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION
)

_LOGGER = logging.getLogger(__name__)
FamilyReadiness = tc_dto.FamilyReadiness
FrameworkIntegrationContext = tc_dto.FrameworkIntegrationContext
PreparedServingArtifact = tc_dto.PreparedServingArtifact
ServingBindingValue = tc_dto.ServingBindingValue
PublishedModelVersion = tc.PublishedModelVersion
ServingBindingPlan = tc_recipe_build.ServingBindingPlan
RecipeBuildCacheConfig = tc_recipe_build.RecipeBuildCacheConfig
RecipeBuildRunResult = tc_recipe_build.RecipeBuildRunResult
RecipeCacheLookupResult = tc_recipe_build.RecipeCacheLookupResult
RecipeCacheWriteResult = tc_recipe_build.RecipeCacheWriteResult
RecipeBuildSession = tc_recipe_build.RecipeBuildSession
COMPILED_RECIPE_MEMORY_CACHE = tc_recipe_build.COMPILED_RECIPE_MEMORY_CACHE
TRACE_PLAN_MEMORY_CACHE = tc_recipe_build.TRACE_PLAN_MEMORY_CACHE
RecipeCompileInputs = tc_compiler.RecipeCompileInputs
RecipePublicationContext = tc_publication.RecipePublicationContext
ParsedRetainedServingBindingAuthority = (
    tc_retained_binding.ParsedRetainedServingBindingAuthority
)
GroupRealizationAcquireRef = tc.GroupRealizationAcquireRef
RuntimeTensorView = tc_dto.RuntimeTensorView
SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4 = (
    tc_runtime_contract.SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4
)
SERVING_MANIFEST_TENSOR_NAME = tc.SERVING_MANIFEST_TENSOR_NAME
ServingBindingState = tc_session.ServingBindingState
ServingArtifactManifest = tc.ServingArtifactManifest
ServingConfig = tc_config.ServingConfig
ReplicaPublicationPolicy = tc_config.ReplicaPublicationPolicy
ServingBindingMemberRef = tc.ServingBindingMemberRef
ServingPlacement = tc_dto.ServingPlacement
ServingRuntimePolicy = tc.ServingRuntimePolicy
SourceBoundContractState = tc_runtime_contract.SourceBoundContractState
source_bound_contract_profile_fields = (
    tc_runtime_contract.source_bound_contract_profile_fields
)
SourceCatalog = tc_source_catalog.SourceCatalog
SOURCE_CATALOG_SCHEMA_VERSION = tc_source_catalog.SOURCE_CATALOG_SCHEMA_VERSION

AdmissionRejectedError = tc_errors.AdmissionRejectedError
ArtifactLocatorResolutionError = tc_errors.ArtifactLocatorResolutionError
AttachFinalizeError = tc_errors.AttachFinalizeError
AuthorityValidationError = tc_errors.AuthorityValidationError
CapabilityMissingError = tc_errors.CapabilityMissingError
ConfigConflictError = tc_errors.ConfigConflictError
ManifestMismatchError = tc_errors.ManifestMismatchError
OwnershipTransferError = tc_errors.OwnershipTransferError
PlacementAdmissionError = tc_errors.PlacementAdmissionError
PolicyMismatchError = tc_errors.PolicyMismatchError
PublicationRequiredError = tc_errors.PublicationRequiredError
ReplicaPublicationError = tc_errors.ReplicaPublicationError
RestoreBindingError = tc_errors.RestoreBindingError
RuntimeSwapError = tc_errors.RuntimeSwapError
SchemaMismatchError = tc_errors.SchemaMismatchError
ServingIntegrationError = tc_errors.ServingIntegrationError
ServingIntegrationNotImplementedError = tc_errors.ServingIntegrationNotImplementedError
SourceProviderError = tc_errors.SourceProviderError
SourceSubjectError = tc_errors.SourceSubjectError
TensorCastServingRuntimeError = tc_errors.TensorCastServingRuntimeError
_capability_missing = tc_errors.capability_missing

RuntimeAttachment = tc_runtime_attachment.RuntimeAttachment
RuntimeBindingState = tc_runtime_attachment.RuntimeBindingState
RuntimeBindingView = tc_runtime_attachment.RuntimeBindingView
RuntimeStateSeed = tc_runtime_attachment.RuntimeStateSeed

BindingValueRefProjection = tc_runtime_view.BindingValueRefProjection
MaterializationDiagnosticsProjection = (
    tc_runtime_view.MaterializationDiagnosticsProjection
)
PublishedReplicaProjection = tc_runtime_view.PublishedReplicaProjection
ReloadRequestProjection = tc_runtime_view.ReloadRequestProjection
ReloadResponseProjection = tc_runtime_view.ReloadResponseProjection
RuntimeEndpointProjection = tc_runtime_view.RuntimeEndpointProjection
RuntimeWorkerView = tc_runtime_view.RuntimeWorkerView
SourceBoundContractProjection = tc_runtime_view.SourceBoundContractProjection
SourceSelectionProjection = tc_runtime_view.SourceSelectionProjection
WeightVersionProjection = tc_runtime_view.WeightVersionProjection
source_selection_projection_from_artifact_realization_report = (
    tc_runtime_view.source_selection_projection_from_artifact_realization_report
)
source_selection_projection_from_execution_diagnostics = (
    tc_runtime_view.source_selection_projection_from_execution_diagnostics
)
source_selection_projection_from_materialization_diagnostics = (
    tc_runtime_view.source_selection_projection_from_materialization_diagnostics
)

# Host capability contracts live in hosts.py. Lifecycle uses module-local
# aliases only to keep the orchestration code readable.
AdmissionDecision = tc_hosts.AdmissionDecision
AdmissionPolicy = tc_hosts.AdmissionPolicy
AdmissionRequest = tc_hosts.AdmissionRequest
CollectiveHost = tc_hosts.CollectiveHost
DefaultAdmissionPolicy = tc_hosts.DefaultAdmissionPolicy
FinalizeHookHost = tc_hosts.FinalizeHookHost
FinalizePhase = tc_hosts.FinalizePhase
FinalizePolicy = tc_hosts.FinalizePolicy
FrameworkHost = tc_hosts.FrameworkHost
FrameworkIdentity = tc_hosts.FrameworkIdentity
IntegrationHost = tc_hosts.IntegrationHost
ManifestPolicy = tc_hosts.ManifestPolicy
MaterializationExecutionFacts = tc_hosts.MaterializationExecutionFacts
MaterializationPolicy = tc_hosts.MaterializationPolicy
NativeLoadHost = tc_hosts.NativeLoadHost
ObservabilitySink = tc_hosts.ObservabilitySink
PlacementAdmissionFacts = tc_hosts.PlacementAdmissionFacts
PlacementHost = tc_hosts.PlacementHost
PlacementIdentityFacts = tc_hosts.PlacementIdentityFacts
PlacementMemberFacts = tc_hosts.PlacementMemberFacts
RecipeCachePolicy = tc_hosts.RecipeCachePolicy
RecipeTraceHost = tc_hosts.RecipeTraceHost
RuntimeConfig = tc_hosts.RuntimeConfig
RuntimeProfile = tc_hosts.RuntimeProfile
SourceBoundContractProfile = tc_hosts.SourceBoundContractProfile
SourceCatalogPolicy = tc_hosts.SourceCatalogPolicy
SourceCatalogProvider = tc_hosts.SourceCatalogProvider
SourceCatalogRequest = tc_hosts.SourceCatalogRequest
SourceDownloadPolicy = tc_hosts.SourceDownloadPolicy
SourceHost = tc_hosts.SourceHost
SourceSelector = tc_hosts.SourceSelector
SourceSubjectCoordinator = tc_hosts.SourceSubjectCoordinator
TensorCastEvent = tc_hosts.TensorCastEvent
TensorSurfaceHost = tc_hosts.TensorSurfaceHost
TorchTensorHost = tc_hosts.TorchTensorHost
semantic_placement_digest = tc_hosts.semantic_placement_digest
serving_placement_from_framework_facts = tc_hosts.serving_placement_from_framework_facts
TensorcastSemanticValidationSpec = tc_compiler.TensorcastSemanticValidationSpec
TensorcastServingFacts = tc_compiler.TensorcastServingFacts
TensorSchemaEntry = tc_compiler.TensorSchemaEntry
read_source_bound_contract_state = tc_runtime_contract.read_source_bound_contract_state
resolve_runtime_config_profile = tc_runtime_config.resolve_runtime_config_profile

RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION = (
    tc_runtime_view.RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION
)
WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION = (
    tc_runtime_view.WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION
)
RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION = (
    tc_runtime_view.RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION
)
PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION = (
    tc_runtime_view.PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION
)
SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION = (
    tc_runtime_view.SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION
)
SERVING_ARTIFACT_LOCATOR_SCHEMA_VERSION = (
    tc_policy.SERVING_ARTIFACT_LOCATOR_SCHEMA_VERSION
)
binding_layout_debug_payload = tc_diagnostics.binding_layout_debug_payload
binding_layout_profile_fields = tc_diagnostics.binding_layout_profile_fields
binding_layout_tensor_count = tc_diagnostics.binding_layout_tensor_count
SERVING_POLICY_SCHEMA_VERSION = tc_policy.SERVING_POLICY_SCHEMA_VERSION
ServingArtifactLocator = tc_policy.ServingArtifactLocator
ServingPolicy = tc_policy.ServingPolicy
normalize_serving_reload_request_payload = (
    tc_policy.normalize_serving_reload_request_payload
)
merge_serving_reload_extra_config = tc_policy.merge_serving_reload_extra_config
load_source_tensors_for_recipe = tc_materialization.load_source_tensors_for_recipe
materialize_recipe_copy_plan_tensors = (
    tc_materialization.materialize_recipe_copy_plan_tensors
)
materialize_pure_transform_serving_tensors = (
    tc_materialization.materialize_pure_transform_serving_tensors
)
materialize_binding_finalize_serving_tensors = (
    tc_materialization.materialize_binding_finalize_serving_tensors
)
collect_serving_tensors_from_model = (
    tc_materialization.collect_serving_tensors_from_model
)
run_binding_finalize_semantic_validation = (
    tc_materialization.run_binding_finalize_semantic_validation
)
validate_binding_finalize_tensor_schema = (
    tc_materialization.validate_binding_finalize_tensor_schema
)
complete_pure_transform_recipe_publication_from_recipe = (
    tc_publication.complete_pure_transform_recipe_publication
)
PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION = 1
PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION = 1
SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION = 1
RECIPE_CACHE_POLICY_SCHEMA_VERSION = 1
SOURCE_CATALOG_REQUEST_SCHEMA_VERSION = 1


BootstrapPolicy = tc_runtime_intent.BootstrapPolicy
ServingIntent = tc_runtime_intent.ServingIntent
ExistingServingArtifact = tc_runtime_intent.ExistingServingArtifact
LocalSourceBootstrap = tc_runtime_intent.LocalSourceBootstrap
RetainedBindingAcquire = tc_runtime_intent.RetainedBindingAcquire
RequestContext = tc_runtime_intent.RequestContext


@dataclass(frozen=True)
class RuntimeBindingMaterialization:
    """Core primitive for adapter-driven attach/finalize/state ownership."""

    host: IntegrationHost
    profile_sink: Any | None = None
    state_factory: Any = RuntimeBindingState

    def attach_and_finalize(
        self,
        *,
        model: object,
        tensors: Mapping[str, object],
        binding_handle: object,
        context: FrameworkIntegrationContext,
        state_seed: RuntimeStateSeed,
        replace_meta_params: bool,
        target_device: Any,
        model_config: object | None = None,
        run_process_after_load: bool = True,
        run_post_bind_finalize: bool = True,
        expected_tensor_schema_hash: str | None = None,
        semantic_validation_spec: Any | None = None,
    ) -> RuntimeBindingState:
        owner: Any = binding_handle
        transferred = False
        try:
            self._emit("runtime_materialization.attach.start", state_seed)
            self._attach_bound_tensors(
                model,
                tensors,
                replace_meta_params=replace_meta_params,
            )
            canonical = self._collect_runtime_tensors(
                model,
                remove_duplicate=False,
            )
            if expected_tensor_schema_hash is not None:
                actual_tensor_schema_hash = self._compute_tensor_schema_hash(
                    canonical,
                    remove_duplicate=False,
                )
                if actual_tensor_schema_hash != expected_tensor_schema_hash:
                    raise SchemaMismatchError(
                        "TensorCast runtime tensor schema hash mismatch: "
                        f"expected={expected_tensor_schema_hash}, "
                        f"actual={actual_tensor_schema_hash}"
                    )
            invariants = self._snapshot_tensor_invariants(canonical)
            self._allocate_runtime_only_tensors(
                model,
                torch.device(target_device),
            )
            if run_process_after_load:
                self._maybe_run_hook(
                    "run_process_after_load",
                    model,
                    model_config,
                    torch.device(target_device),
                )
            if run_post_bind_finalize:
                self._maybe_run_hook(
                    "run_runtime_only_post_bind",
                    model,
                    model_config,
                    torch.device(target_device),
                )
            if semantic_validation_spec is not None:
                self._run_semantic_validation(
                    semantic_validation_spec,
                    model,
                    model_config,
                )
            after = self._collect_runtime_tensors(
                model,
                remove_duplicate=False,
            )
            self._validate_tensor_invariants(invariants, after)
            transfer_to_runtime = getattr(binding_handle, "transfer_to_runtime", None)
            if callable(transfer_to_runtime):
                owner = transfer_to_runtime()
                transferred = True
            view = state_seed.runtime_view()
            realization_handle = _runtime_attachment_realization_handle(
                report=state_seed.realization_report,
                binding_handle=binding_handle,
                owner=owner,
            )
            model_runtime_ref: dict[str, RuntimeBindingState] = {}
            model_runtime_handle = _model_runtime_realization_handle(
                context=context,
                target_device=target_device,
                runtime_attachment_handle=realization_handle,
                attach_fn=lambda **_kwargs: model_runtime_ref["state"],
            )
            try:
                state = self.state_factory(
                    binding=binding_handle,
                    artifact_ref=state_seed.artifact_ref,
                    runtime_view=view,
                    ownership_handle=owner,
                    release_contract=None
                    if realization_handle is None
                    else realization_handle.release_contract,
                    realization_handle=realization_handle,
                    model_runtime_handle=model_runtime_handle,
                )
            except Exception as exc:
                self._close_quietly(realization_handle or owner)
                raise OwnershipTransferError(
                    "TensorCast runtime binding state construction failed"
                ) from exc
            model_runtime_ref["state"] = state
            self._emit("runtime_materialization.attach.done", state_seed)
            return state
        except OwnershipTransferError:
            raise
        except SchemaMismatchError:
            self._close_quietly(owner)
            raise
        except Exception as exc:
            self._close_quietly(owner)
            if transferred:
                raise OwnershipTransferError(
                    "TensorCast runtime binding ownership transfer failed"
                ) from exc
            raise AttachFinalizeError(
                "TensorCast runtime binding attach/finalize failed"
            ) from exc

    def _maybe_run_hook(
        self,
        name: str,
        model: object,
        model_config: object | None,
        target_device: torch.device,
    ) -> None:
        hook_host = self.host.framework
        hook = getattr(hook_host, name, None)
        if callable(hook):
            hook(model, model_config, target_device)
            return
        phase = {
            "run_process_after_load": "process_after_load",
            "run_runtime_only_post_bind": "runtime_only_post_bind",
        }.get(name)
        if phase is None:
            return
        run_hook = getattr(hook_host, "run_finalize_hook", None)
        if callable(run_hook):
            run_hook(phase, model, model_config, target_device)

    def _run_semantic_validation(
        self,
        spec: Any,
        model: object,
        model_config: object | None,
    ) -> Any:
        if getattr(spec, "kind", None) == "none":
            return evaluate_semantic_validation_spec(spec, None)
        hook_host = self.host.framework
        semantic_probes = getattr(hook_host, "semantic_probes", None)
        actual_payload = (
            semantic_probes(model, model_config) if callable(semantic_probes) else None
        )
        return evaluate_semantic_validation_spec(spec, actual_payload)

    def _surface(self) -> TensorSurfaceHost:
        if self.host.tensor_surface is None:
            raise _capability_missing(
                "IntegrationHost requires TensorSurfaceHost for runtime "
                "tensor attach/schema/invariant operations",
                level="level1-runtime",
                capability="tensor_surface",
                operation="runtime_tensor_surface",
                required_methods=(
                    "attach_bound_tensors",
                    "collect_runtime_tensors",
                    "compute_runtime_tensor_schema_hash",
                    "snapshot_tensor_invariants",
                    "validate_tensor_invariants",
                ),
                next_action=(
                    "Pass IntegrationHost(tensor_surface=...) or use "
                    "TorchTensorHost for PyTorch module carriers."
                ),
            )
        return self.host.tensor_surface

    def _attach_bound_tensors(
        self,
        model: object,
        tensors: Mapping[str, object],
        *,
        replace_meta_params: bool,
    ) -> object:
        return self._surface().attach_bound_tensors(
            model,
            tensors,
            replace_meta_params=replace_meta_params,
        )

    def _collect_runtime_tensors(
        self,
        model: object,
        *,
        remove_duplicate: bool,
    ) -> Mapping[str, object]:
        return self._surface().collect_runtime_tensors(
            model,
            remove_duplicate=remove_duplicate,
        )

    def _compute_tensor_schema_hash(
        self,
        tensors: Mapping[str, object],
        *,
        remove_duplicate: bool,
    ) -> str:
        return self._surface().compute_runtime_tensor_schema_hash(
            tensors,
            remove_duplicate=remove_duplicate,
        )

    def _allocate_runtime_only_tensors(
        self,
        model: object,
        target_device: object,
    ) -> Mapping[str, object]:
        return self._surface().allocate_runtime_only_tensors(model, target_device)

    def _snapshot_tensor_invariants(self, tensors: Mapping[str, object]) -> object:
        return self._surface().snapshot_tensor_invariants(tensors)

    def _validate_tensor_invariants(
        self,
        before: object,
        after: Mapping[str, object],
    ) -> None:
        self._surface().validate_tensor_invariants(before, after)

    def _emit(self, event: str, state_seed: RuntimeStateSeed) -> None:
        sink = self.profile_sink
        if callable(sink):
            sink(
                {
                    "event": event,
                    "artifact_ref": state_seed.artifact_ref,
                    "readiness": state_seed.readiness,
                }
            )

    @staticmethod
    def _close_quietly(handle: object) -> None:
        _close_quietly(handle)


@dataclass(frozen=True)
class _HostMaterializationRequest:
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = ""
    require_materialization_options: bool = False


@dataclass(frozen=True)
class _DirectServingLoad:
    artifact_locator: Any | None = None
    policy: Any | None = None
    materialization: Any | None = None
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = "startup.direct_serving_artifact.bind"
    require_materialization_options: bool = False
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    expected_member: Any | None = None
    timeout_s: float | None = 30.0
    artifact_ref: str | None = None
    resolved_artifact: ResolvedServingArtifact | None = None
    model: Any | None = None


@dataclass(frozen=True)
class ServingLoadResult:
    model: Any | None = None
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    resolved_artifact: ResolvedServingArtifact | None = None
    binding_result: RuntimeBindingResult | None = None


@dataclass(frozen=True)
class _ServingReload:
    current_state: RuntimeBindingState | Any
    artifact_locator: Any | None = None
    policy: Any | None = None
    materialization: Any | None = None
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = "runtime_binding.swap"
    contract_identity: str | None = None
    require_materialization_options: bool = False
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    artifact_ref: str | None = None
    resolved_artifact: ResolvedServingArtifact | None = None
    model: Any | None = None


@dataclass(frozen=True)
class ServingReloadResult:
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    resolved_artifact: ResolvedServingArtifact | None = None
    binding_result: RuntimeBindingResult | None = None


@dataclass(frozen=True)
class _ServingArtifactPreflight:
    resolved_artifact: ResolvedServingArtifact
    serving_runtime_policy: Any | None


@dataclass(frozen=True)
class _RetainedBindingAcquire:
    authority: Any | None = None
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    expected_member: Any | None = None
    runtime: Any | None = None
    client: Any | None = None
    restore_fn: Any | None = None
    timeout_s: float | None = 30.0


@dataclass(frozen=True)
class RetainedBindingResult:
    model: Any | None = None
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    restored: RestoredRetainedBinding | None = None


@dataclass(frozen=True)
class _LocalReadyBootstrap:
    """Internal lowering payload for ``LocalSourceBootstrap``.

    This is deliberately private: framework integrations enter through
    ``ServingIntegration.start(LocalSourceBootstrap, context)`` and host facts.
    """

    source_selector: SourceSelector | Any | None = None
    bootstrap: Any | None = None
    materialization: Any | None = None
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = "bootstrap.same_binding_fast_path.tensorcast_realize"
    contract_identity: str | None = None
    require_materialization_options: bool = False
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    source_subject_coordinator: Any | None = None
    recipe: Any | None = None
    source_catalog: Any | None = None
    source_catalog_config: Any | None = None
    cache_config: Any | None = None
    cache_config_factory: Any | None = None
    source_subject: Any | None = None
    placement: Any | None = None
    source_artifact_ref: str | None = None
    serving_manifest_ref: str | None = None
    representation_contract_hash: str | None = None
    serving_build_digest: str | None = None
    model: Any | None = None
    manifest_tensor_name: str | None = None
    manifest_bytes: bytes | None = None
    build_recipe_from_framework_context: bool = False
    build_model_from_framework_context: bool = False
    build_manifest_carrier_from_framework_context: bool = False
    run_binding_finalize_hooks_when_required: bool = False
    options: Any | None = None
    binding_factory: Any | None = None
    family: str = ""
    tp_rank: int = 0
    tp_world_size: int = 1
    replace_meta_params: bool = True
    run_process_after_load: bool = False
    run_post_bind_finalize: bool = True
    run_semantic_validation: bool = False
    semantic_validation_spec: Any | None = None
    validate_representation_contract_hash: bool = False
    runtime_binding_schema_version: int | None = None
    serving_artifact_schema_version: int | None = None
    framework_name: str | None = None
    framework_version: str | None = None
    adapter_version: str | None = None
    serving_abi_version: str | None = None


@dataclass(frozen=True)
class _LocalReadyFinalize:
    """Internal payload for local-ready attach/finalize state construction."""

    model: Any
    recipe: Any
    binding: Any
    update_epoch: Any
    source_artifact_ref: str
    serving_manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    manifest_tensor_name: str
    source_bound_contract_state: Any
    source_bound_contract_path: str
    target_device: Any
    manifest_bytes: bytes | None = None
    framework_config: Any | None = None
    model_config: Any | None = None
    placement: Any | None = None
    family: str = ""
    tp_rank: int = 0
    tp_world_size: int = 1
    replace_meta_params: bool = True
    run_process_after_load: bool = False
    run_post_bind_finalize: bool = True
    run_semantic_validation: bool = False
    semantic_validation_spec: Any | None = None
    validate_representation_contract_hash: bool = False
    runtime_binding_schema_version: int | None = None
    serving_artifact_schema_version: int | None = None
    framework_name: str | None = None
    framework_version: str | None = None
    adapter_version: str | None = None
    serving_abi_version: str | None = None


@dataclass(frozen=True)
class LocalReadyServingResult:
    model: Any | None = None
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    prepared: PreparedServingArtifact | None = None
    binding_value: ServingBindingValue | None = None
    recipe: Any | None = None
    current_value: Any | None = None
    binding: Any | None = None
    update_epoch: Any | None = None
    layout: Any | None = None
    realization_entry_count: int | None = None
    realization: Any | None = None
    realization_report: ArtifactRealizationReport | None = None


@dataclass(frozen=True)
class RecipeBuildSessionRequest:
    source_subject: SourceSubject | Any | None = None
    framework_config: Any | None = None
    model_config: Any | None = None
    placement: ServingPlacement | None = None
    cache_config: Any | None = None
    identity: ServingBindingPlan | None = None
    trace_cache_schema_version: int | None = None
    tp_rank: int | None = None
    tp_world_size: int | None = None


@dataclass(frozen=True)
class RecipeBuildResult:
    session: RecipeBuildSession
    recipe: Any | None = None
    diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LocalReadyBindingContract:
    excluded_names: tuple[str, ...]
    canonical_tensor_names: tuple[str, ...]
    tensor_schema_hash: str
    representation_contract_hash: str
    mapped_copy_plan: tuple[Any, ...]
    realization_plan_proto: bytes
    realization_entry_count: int
    fallback_copy_plan: tuple[Any, ...]


@dataclass(frozen=True)
class LocalReadyMaterializationIdentity:
    source_artifact_ref: str
    source_metadata_fingerprint: str


@dataclass(frozen=True)
class LocalReadyManifestCarrierResult:
    representation_contract_hash: str
    manifest_bytes: bytes
    serving_manifest_ref: str
    serving_build_digest: str


def _binding_tensors(binding: Any) -> Mapping[str, torch.Tensor]:
    tensors = getattr(binding, "tensors", {})
    if tensors is None:
        return {}
    return dict(tensors)


def _canonical_index_bytes_from_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> bytes:
    entries: list[tc.CanonicalIndexEntry] = []
    cursor = 0
    for key, tensor in sorted(tensors.items(), key=lambda item: str(item[0])):
        name = str(key)
        size_bytes = int(tensor.element_size()) * int(tensor.numel())
        entries.append(
            CanonicalIndexEntry(
                name=name,
                dtype=tensor.dtype,
                shape=tuple(int(dim) for dim in tensor.shape),
                stride=tuple(int(dim) for dim in tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                segment_offset=cursor,
                size_bytes=size_bytes,
            )
        )
        cursor += size_bytes
    return canonical_index_to_bytes(
        CanonicalIndex(entries=tuple(entries), total_size_bytes=cursor, avbs_hash="")
    )


def _canonical_index_bytes_for_runtime_selection(
    *,
    resolved: ResolvedServingArtifact | Any | None,
    tensors: Mapping[str, torch.Tensor],
) -> bytes:
    descriptor = getattr(resolved, "descriptor", None)
    if descriptor is not None:
        try:
            return canonical_index_to_bytes(canonical_index_from_descriptor(descriptor))
        except (AttributeError, KeyError, TypeError, ValueError):
            _LOGGER.debug(
                "Failed to derive runtime canonical index from descriptor; using tensor metadata",
                exc_info=True,
            )
    return _canonical_index_bytes_from_tensors(tensors)


def _target_layout_digest_for_runtime_attachment(
    *,
    binding_layout_id: str | None,
    tensor_schema_hash: str,
) -> str:
    if binding_layout_id:
        return f"binding-layout:{binding_layout_id}"
    return f"runtime-schema:{tensor_schema_hash}"


def _runtime_attachment_report_for_resolved(
    *,
    resolved: ResolvedServingArtifact | Any,
    tensors: Mapping[str, torch.Tensor],
    binding_handle: Any | None,
    target_device: Any,
    tensor_schema_hash: str,
    execution_diagnostics: Any | None = None,
    materialization_diagnostics: Any | None = None,
) -> ArtifactRealizationReport:
    binding_layout_id = _optional_text(
        getattr(binding_handle, "binding_layout_id", None)
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device=target_device,
        target_layout_digest=_target_layout_digest_for_runtime_attachment(
            binding_layout_id=binding_layout_id,
            tensor_schema_hash=tensor_schema_hash,
        ),
        binding_layout_id=binding_layout_id,
    )
    envelope = envelope_for_runtime_attachment(tensors, retained=False)
    envelope.validate_for_target(target_plan)
    selection = resolve_artifact_selection(
        artifact_id=str(getattr(resolved, "artifact_ref", "") or ""),
        canonical_index_bytes=_canonical_index_bytes_for_runtime_selection(
            resolved=resolved,
            tensors=tensors,
        ),
        tensor_names=tuple(str(name) for name in tensors),
        artifact_profile="serving_artifact",
        authority_scope="daemon_mediated_runtime_attachment",
    )
    return report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        binding_handle=binding_handle,
        materialization_diagnostics=materialization_diagnostics,
        execution_diagnostics=execution_diagnostics,
        risk_labels=("binding_lifecycle",),
    )


def _runtime_attachment_report_for_retained(
    *,
    authority: tc_retained_binding.ParsedRetainedServingBindingAuthority,
    tensors: Mapping[str, torch.Tensor],
    binding_handle: Any | None,
    target_device: Any,
    tensor_schema_hash: str,
    reservation_bytes: int,
) -> ArtifactRealizationReport:
    binding_layout_id = _optional_text(
        getattr(binding_handle, "binding_layout_id", None)
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device=target_device,
        target_layout_digest=_target_layout_digest_for_runtime_attachment(
            binding_layout_id=binding_layout_id,
            tensor_schema_hash=tensor_schema_hash,
        ),
        binding_layout_id=binding_layout_id,
        copy_plan_digest=authority.expected.resolved_spec_digest,
    )
    envelope = envelope_for_runtime_attachment(
        tensors,
        retained=True,
        reservation_bytes=reservation_bytes,
    )
    envelope.validate_for_target(target_plan)
    artifact_id = (
        authority.serving_artifact_id
        or authority.local_serving_ref
        or authority.binding_value_ref.binding_value_id
    )
    selection = resolve_artifact_selection(
        artifact_id=str(artifact_id),
        canonical_index_bytes=_canonical_index_bytes_from_tensors(tensors),
        tensor_names=tuple(str(name) for name in tensors),
        artifact_profile="retained_binding",
        authority_scope="daemon_retained_runtime_attachment",
    )
    return report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        binding_handle=binding_handle,
        retained_authority=authority,
        risk_labels=("retained_acquire",),
    )


def _runtime_attachment_report_for_artifact_id(
    *,
    artifact_id: str,
    tensors: Mapping[str, torch.Tensor],
    binding_handle: Any | None,
    target_device: Any,
    tensor_schema_hash: str,
    artifact_profile: str,
    authority_scope: str,
    retained: bool = False,
    reservation_bytes: int = 0,
) -> ArtifactRealizationReport:
    binding_layout_id = _optional_text(
        getattr(binding_handle, "binding_layout_id", None)
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device=target_device,
        target_layout_digest=_target_layout_digest_for_runtime_attachment(
            binding_layout_id=binding_layout_id,
            tensor_schema_hash=tensor_schema_hash,
        ),
        binding_layout_id=binding_layout_id,
    )
    envelope = envelope_for_runtime_attachment(
        tensors,
        retained=retained,
        reservation_bytes=reservation_bytes,
    )
    envelope.validate_for_target(target_plan)
    selection = resolve_artifact_selection(
        artifact_id=str(artifact_id),
        canonical_index_bytes=_canonical_index_bytes_from_tensors(tensors),
        tensor_names=tuple(str(name) for name in tensors),
        artifact_profile=artifact_profile,
        authority_scope=authority_scope,
    )
    return report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        binding_handle=binding_handle,
        risk_labels=(artifact_profile,),
    )


def _close_quietly(handle: object) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            _LOGGER.exception("Failed to close runtime attachment handle")


def _runtime_attachment_realization_handle(
    *,
    report: ArtifactRealizationReport | None,
    binding_handle: Any,
    owner: Any | None = None,
) -> ArtifactRealizationHandle | None:
    if report is None:
        return None
    owner_handle = owner if owner is not None else binding_handle
    close = getattr(owner_handle, "close", None)
    close_fn = None
    if callable(close):

        def close_owner_handle() -> None:
            close()

        close_fn = close_owner_handle
    handle = ArtifactRealizationHandle(
        target_kind="runtime_attachment",
        report=report,
        binding_value=binding_handle,
        close_fn=close_fn,
    )
    emit_artifact_realization_profile_event(report)
    return handle


def _model_runtime_spec_for_context(
    *,
    context: FrameworkIntegrationContext,
    target_device: Any,
) -> ArtifactRealizationSpec:
    placement = getattr(context, "placement", None)
    framework = str(getattr(context, "framework_name", "") or "unknown_framework")
    return ArtifactRealizationSpec.model_runtime(
        framework=framework,
        device=target_device,
        topology=getattr(placement, "topology", None),
        member=getattr(placement, "member", None),
        adapter_version=_optional_text(getattr(context, "adapter_version", None)),
        runtime_abi_version=_optional_text(
            getattr(context, "serving_abi_version", None)
        ),
    )


def _model_runtime_realization_handle(
    *,
    context: FrameworkIntegrationContext,
    target_device: Any,
    runtime_attachment_handle: ArtifactRealizationHandle | None,
    attach_fn: Callable[..., RuntimeBindingState],
) -> ArtifactRealizationHandle | None:
    return _model_runtime_realization_handle_for_spec(
        spec=_model_runtime_spec_for_context(
            context=context,
            target_device=target_device,
        ),
        runtime_attachment_handle=runtime_attachment_handle,
        attach_fn=attach_fn,
    )


def _model_runtime_realization_handle_for_spec(
    *,
    spec: ArtifactRealizationSpec | None,
    runtime_attachment_handle: ArtifactRealizationHandle | None,
    attach_fn: Callable[..., RuntimeBindingState],
) -> ArtifactRealizationHandle | None:
    if runtime_attachment_handle is None:
        return None
    if spec is None:
        return None
    report = model_runtime_report_for(
        spec=spec,
        runtime_attachment_report=runtime_attachment_handle.report,
    )
    handle = ArtifactRealizationHandle(
        target_kind="model_runtime",
        report=report,
        attach_fn=attach_fn,
        release_contract=runtime_attachment_handle.release_contract,
    )
    emit_artifact_realization_profile_event(report)
    return handle


@dataclass(frozen=True)
class RuntimeBindingResult:
    """Attach-ready result from a serving bind or swap operation."""

    binding: Any
    tensors: Mapping[str, torch.Tensor]
    binding_layout_id: str | None = None
    operation_result: Any | None = None
    execution_diagnostics: Any | None = None
    materialization_diagnostics: Any | None = None

    @classmethod
    def from_binding(
        cls,
        binding: Any,
        *,
        operation_result: Any | None = None,
    ) -> RuntimeBindingResult:
        return cls(
            binding=binding,
            tensors=_binding_tensors(binding),
            binding_layout_id=getattr(binding, "binding_layout_id", None),
            operation_result=operation_result,
            execution_diagnostics=getattr(binding, "last_execution_diagnostics", None),
            materialization_diagnostics=getattr(
                binding,
                "last_materialization_diagnostics",
                None,
            ),
        )


@dataclass
class RestoredRetainedBinding:
    """Restored retained binding tensors before runtime ownership transfer."""

    _attached: tc_retained_binding.AttachedRetainedBinding
    _runtime_handle: (
        tc_retained_binding.RuntimeRetainedBindingAttachmentHandle | None
    ) = None

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._attached.tensors

    @property
    def binding_layout_id(self) -> str:
        return self._attached.binding_layout_id

    @property
    def binding_value_ref(self) -> tc.BindingValueRef:
        return self._attached.binding_value_ref

    @property
    def member_ref(self) -> tc.ServingBindingMemberRef:
        return self._attached.member_ref

    @property
    def reservation_bytes(self) -> int:
        return self._attached.reservation_bytes

    @property
    def authority(self) -> tc_retained_binding.ParsedRetainedServingBindingAuthority:
        return self._attached.authority

    @property
    def runtime_handle(
        self,
    ) -> tc_retained_binding.RuntimeRetainedBindingAttachmentHandle | None:
        return self._runtime_handle

    def transfer_to_runtime(
        self,
    ) -> tc_retained_binding.RuntimeRetainedBindingAttachmentHandle:
        if self._runtime_handle is None:
            self._runtime_handle = self._attached.transfer_to_runtime()
        return self._runtime_handle

    def close(self) -> None:
        if self._runtime_handle is None:
            self._attached.close()


@dataclass(frozen=True)
class SourceSubject:
    """Opaque framework-facing source subject wrapper."""

    artifact_ref: str
    subject: Any
    source_kind: str = "opaque"
    metadata_fingerprint: str | None = None

    def broadcast_payload(self) -> dict[str, Any]:
        if self.source_kind == "public_disk":
            subject_payload = _public_disk_source_payload(self.subject)
        else:
            subject_payload = self.subject
        return {
            "kind": self.source_kind,
            "artifact_ref": self.artifact_ref,
            "subject": subject_payload,
            "metadata_fingerprint": self.metadata_fingerprint,
        }

    def profile_fields(self) -> dict[str, Any]:
        source = self.subject
        fields: dict[str, Any] = {
            "artifact_ref": self.artifact_ref,
            "source_kind": self.source_kind,
        }
        if self.metadata_fingerprint is not None:
            fields["metadata_fingerprint"] = self.metadata_fingerprint
        canonical_index = getattr(source, "canonical_index_bytes", None)
        if canonical_index is not None:
            fields["canonical_index_bytes"] = len(canonical_index)
        source_index = getattr(source, "source_index_bytes", None)
        if source_index is not None:
            fields["source_index_bytes"] = len(bytes(source_index or b""))
        for name in ("format_kind", "metadata_capability"):
            value = getattr(source, name, None)
            if value is not None:
                fields[name] = str(value or "")
        return fields


def _public_disk_source_payload(source: Any) -> dict[str, Any]:
    return {
        "path": str(getattr(source, "path", "") or ""),
        "canonical_index_bytes": bytes(source.canonical_index_bytes),
        "artifact_id": str(getattr(source, "artifact_id", "") or ""),
        "generation": int(getattr(source, "generation", 0) or 0),
        "verify_checksums": bool(getattr(source, "verify_checksums", True)),
        "trusted_content_artifact_id": _optional_str(
            getattr(source, "trusted_content_artifact_id", None)
        ),
        "source_index_bytes": _optional_bytes(
            getattr(source, "source_index_bytes", None)
        ),
        "format_kind": _enum_wire_value(getattr(source, "format_kind", None)),
        "metadata_capability": _enum_wire_value(
            getattr(source, "metadata_capability", None)
        ),
        "resolution_strategy": _enum_wire_value(
            getattr(source, "resolution_strategy", None)
        ),
        "validation_mode": _enum_wire_value(getattr(source, "validation_mode", None)),
        "policy_id": _optional_str(getattr(source, "policy_id", None)),
        "exact_size_bytes": int(getattr(source, "exact_size_bytes", 0) or 0),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_text(value: Any) -> str | None:
    return _optional_str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _serving_realization_report(
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    value = diagnostics.get("serving_realization_report")
    if isinstance(value, Mapping):
        return value
    return None


def _nested_mapping(
    value: Mapping[str, object] | None,
    key: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    nested = value.get(key)
    if isinstance(nested, Mapping):
        return nested
    return None


def _nested_value(
    value: Mapping[str, object] | None,
    *path: str,
) -> object | None:
    current: object | None = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _artifact_locator_kind(artifact_locator: object) -> str:
    if isinstance(artifact_locator, Mapping):
        return str(artifact_locator.get("kind") or "")
    return str(getattr(artifact_locator, "kind", "") or "")


def _optional_bool(fields: Mapping[str, object], name: str, default: bool) -> bool:
    value = fields.get(name)
    if value is None:
        return default
    return bool(value)


def _optional_path(value: object | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _model_adjacent_cache_root(source_catalog: object) -> Path | None:
    raw_selected_files = getattr(source_catalog, "selected_files", ()) or ()
    selected_files = tuple(cast(Sequence[Any], raw_selected_files))
    if not selected_files:
        return None
    parent_paths: list[str] = []
    for entry in selected_files:
        path = getattr(entry, "path", None)
        if path is None:
            continue
        parent_paths.append(str(Path(path).expanduser().resolve().parent))
    if not parent_paths:
        return None
    return Path(os.path.commonpath(parent_paths)) / ".tensorcast" / "bootstrap_cache"


def _is_writable_or_creatable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.exists() and os.access(parent, os.W_OK)


def _recipe_build_cache_config_from_policy(
    policy: RecipeCachePolicy,
    *,
    source_catalog: object,
) -> RecipeBuildCacheConfig:
    fields = dict(policy.fields or {})
    explicit_cache_root = _optional_bool(fields, "explicit_cache_root", False)
    prefer_model_adjacent = _optional_bool(fields, "prefer_model_adjacent", True)
    cache_root = _optional_path(fields.get("cache_root"))

    roots: list[Path] = []
    if prefer_model_adjacent:
        model_adjacent = _model_adjacent_cache_root(source_catalog)
        if model_adjacent is not None:
            roots.append(model_adjacent)
    if cache_root is not None and (explicit_cache_root or not roots):
        roots.append(cache_root)
    roots = list(_unique_paths(roots))

    write_roots: list[Path] = []
    if prefer_model_adjacent:
        model_adjacent = _model_adjacent_cache_root(source_catalog)
        if model_adjacent is not None and _is_writable_or_creatable(model_adjacent):
            write_roots.append(model_adjacent)
    if cache_root is not None and (explicit_cache_root or not write_roots):
        write_roots.append(cache_root)
    write_roots = list(_unique_paths(write_roots))

    debug_output_dir = _optional_path(fields.get("debug_output_dir"))
    return RecipeBuildCacheConfig(
        cache_dirs=tuple(str(root / "trace_plans") for root in roots),
        trace_write_dirs=tuple(str(root / "trace_plans") for root in write_roots),
        recipe_cache_dirs=tuple(str(root / "compiled_recipes") for root in roots),
        recipe_cache_write_dirs=tuple(
            str(root / "compiled_recipes") for root in write_roots
        ),
        debug_output_dir=debug_output_dir,
        allow_cache=_optional_bool(fields, "allow_cache", True),
        allow_recipe_cache=_optional_bool(fields, "allow_recipe_cache", True),
        allow_trace=_optional_bool(fields, "allow_trace", True),
        trace_tp_slices=_optional_bool(fields, "trace_tp_slices", True),
        debug_dump_trace=_optional_bool(fields, "debug_dump_trace", False),
        synchronous_cache_write=_optional_bool(
            fields, "synchronous_cache_write", False
        ),
        synchronous_recipe_cache_write=_optional_bool(
            fields, "synchronous_recipe_cache_write", False
        ),
    )


def _collective_policy_value(policy: MaterializationPolicy) -> str:
    collective = str(policy.fields.get("collective", "auto") or "auto")
    return {
        "auto": "collective_first",
        "required": "require_collective",
        "disabled": "disable_collective",
        "collective_first": "collective_first",
        "require_collective": "require_collective",
        "disable_collective": "disable_collective",
    }.get(collective, collective)


def _execution_facts_payload(
    facts: MaterializationExecutionFacts,
) -> dict[str, object]:
    return {
        "tp_rank": facts.collective_rank,
        "tp_world_size": facts.collective_world_size,
        "same_node_tp": facts.same_node_tensor_parallel,
        "tp_ranks": tuple(int(rank) for rank in facts.tensor_parallel_ranks),
        "collective_world_size": facts.collective_world_size,
        "collective_rank": facts.collective_rank,
        "collective_context_unavailable": facts.collective_context_unavailable,
    }


def _framework_payload_mapping(payload: object | None) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    return {str(key): value for key, value in payload.items()}


def _optional_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    data = bytes(value)
    return data or None


def _enum_wire_value(value: Any) -> str | int | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", value)
    if isinstance(enum_value, (str, int)):
        return enum_value
    return str(enum_value)


def _source_subject_from_handle(source: Any) -> SourceSubject:
    artifact_ref = str(getattr(source, "artifact_id", "") or "")
    if not artifact_ref:
        raise RuntimeError("TensorCast source subject is missing a source artifact_id")
    return SourceSubject(
        artifact_ref=artifact_ref,
        subject=source,
        source_kind="public_disk",
    )


def resolve_source_subject(
    path: str,
    *,
    verify_checksums: bool,
) -> SourceSubject:
    return _source_subject_from_handle(
        tc.resolve_public_disk_source(
            path,
            verify_checksums=verify_checksums,
        )
    )


def source_subject_from_broadcast_payload(payload: Mapping[str, Any]) -> SourceSubject:
    payload_dict = dict(payload)
    if "kind" not in payload_dict:
        raise SourceSubjectError(
            "TensorCast source subject broadcast payload is missing kind"
        )
    kind = str(payload_dict.get("kind") or "")
    artifact_ref = str(payload_dict.get("artifact_ref") or "")
    if not artifact_ref:
        raise SourceSubjectError(
            "TensorCast source subject broadcast payload is missing artifact_ref"
        )
    source: Any
    if kind == "public_disk":
        subject_payload = payload_dict.get("subject")
        if not isinstance(subject_payload, Mapping):
            raise SourceSubjectError(
                "TensorCast public_disk source subject payload must be a mapping"
            )
        source = tc.PublicDiskSourceHandle(**dict(subject_payload))
    else:
        source = payload_dict.get("subject")
    return SourceSubject(
        artifact_ref=artifact_ref,
        subject=source,
        source_kind=kind,
        metadata_fingerprint=_optional_text(payload_dict.get("metadata_fingerprint")),
    )


def source_subject_broadcast_payload(subject: SourceSubject) -> dict[str, Any]:
    return subject.broadcast_payload()


def is_public_disk_source_subject(subject: Any) -> bool:
    return isinstance(subject, tc.PublicDiskSourceHandle)


def source_subject_slice_count(recipe: Any, subject: Any) -> int:
    if is_public_disk_source_subject(subject):
        return 0
    return tensorcast_view_slice_count(recipe)


def serving_binding_state_from_runtime_view(
    *,
    runtime_view: RuntimeBindingView,
    artifact_locator: Any,
    policy: Any,
    readiness: str | None = None,
) -> ServingBindingState:
    binding_value_ref = runtime_view.binding_value_ref
    to_ref = getattr(binding_value_ref, "to_binding_value_ref", None)
    if callable(to_ref):
        binding_value_ref = to_ref()
    if binding_value_ref is not None and not isinstance(
        binding_value_ref,
        BindingValueRef,
    ):
        if isinstance(binding_value_ref, Mapping):
            binding_value_ref = BindingValueRef.model_validate(dict(binding_value_ref))
        else:
            raise ServingIntegrationError(
                "RuntimeBindingView.binding_value_ref must be BindingValueRef or a mapping"
            )
    typed_binding_value_ref = cast(BindingValueRef | None, binding_value_ref)
    resolved_readiness = readiness or runtime_view.readiness or "loaded"
    state = "loaded" if resolved_readiness == "serving" else resolved_readiness
    return ServingBindingState(
        state=state,
        artifact_locator=artifact_locator,
        serving_artifact_ref=runtime_view.serving_artifact_ref,
        manifest_ref=getattr(policy, "manifest_ref", None),
        representation_contract_hash=(
            runtime_view.representation_contract_hash
            or getattr(policy, "representation_contract_hash", "")
        ),
        serving_build_digest=getattr(policy, "serving_build_digest", None),
        binding_value_ref=typed_binding_value_ref,
        local_serving_ref=runtime_view.local_serving_ref,
        readiness=resolved_readiness,
    )


def runtime_binding_state_from_runtime_view(
    *,
    binding: Any,
    runtime_view: RuntimeBindingView,
    artifact_ref: str | None = None,
    ownership_handle: Any | None = None,
    artifact_realization_report: ArtifactRealizationReport | None = None,
    model_runtime_spec: ArtifactRealizationSpec | None = None,
) -> RuntimeBindingState:
    realization_handle = _runtime_attachment_realization_handle(
        report=artifact_realization_report,
        binding_handle=binding,
        owner=ownership_handle,
    )
    model_runtime_ref: dict[str, RuntimeBindingState] = {}
    model_runtime_handle = _model_runtime_realization_handle_for_spec(
        spec=model_runtime_spec,
        runtime_attachment_handle=realization_handle,
        attach_fn=lambda **_kwargs: model_runtime_ref["state"],
    )
    state = RuntimeBindingState(
        binding=binding,
        artifact_ref=artifact_ref or runtime_view.serving_artifact_ref,
        runtime_view=runtime_view,
        ownership_handle=ownership_handle,
        release_contract=None
        if realization_handle is None
        else realization_handle.release_contract,
        realization_handle=realization_handle,
        model_runtime_handle=model_runtime_handle,
    )
    model_runtime_ref["state"] = state
    return state


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def execution_diagnostics_summary_fields(
    diagnostics: Any | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    fields = {
        "collective_requested": bool(
            getattr(diagnostics, "collective_requested", False)
        ),
        "collective_acknowledged": bool(
            getattr(diagnostics, "collective_acknowledged", False)
        ),
        "collective_used": bool(getattr(diagnostics, "collective_used", False)),
        "collective_policy": _enum_value(
            getattr(diagnostics, "collective_policy", None)
        ),
        "collective_failure_class": _enum_value(
            getattr(diagnostics, "collective_failure_class", None)
        ),
        "dominant_executor": getattr(diagnostics, "dominant_executor", None),
        "direct_write_supported": bool(
            getattr(diagnostics, "direct_write_supported", False)
        ),
        "fallback_bytes": int(getattr(diagnostics, "fallback_bytes", 0)),
        "residual_bytes": int(getattr(diagnostics, "residual_bytes", 0)),
        "actual_collective_committed_bytes": int(
            getattr(diagnostics, "actual_collective_committed_bytes", 0)
        ),
        "actual_local_typed_bytes": int(
            getattr(diagnostics, "actual_local_typed_bytes", 0)
        ),
        "actual_generic_backend_bytes": int(
            getattr(diagnostics, "actual_generic_backend_bytes", 0)
        ),
        "collective_skip_reason": getattr(diagnostics, "collective_skip_reason", None),
        "hash_rounds": int(getattr(diagnostics, "hash_rounds", 0)),
        "hash_backend": _enum_value(getattr(diagnostics, "hash_backend", None)),
        "hash_bytes": int(getattr(diagnostics, "hash_bytes", 0)),
        "hash_wall_time_ms": int(getattr(diagnostics, "hash_wall_time_ms", 0)),
        "hash_identity_forming": bool(
            getattr(diagnostics, "hash_identity_forming", False)
        ),
        "hash_location": _enum_value(getattr(diagnostics, "hash_location", None)),
        "identity_mint_strategy": _enum_value(
            getattr(diagnostics, "identity_mint_strategy", None)
        ),
    }
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def source_bound_plan_diagnostics_summary_fields(
    diagnostics: Any | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    fields = {
        "execution_plan_kind": getattr(diagnostics, "execution_plan_kind", None),
        "planned_collective_candidate_bytes": int(
            getattr(diagnostics, "planned_collective_candidate_bytes", 0)
        ),
        "planned_collective_admitted_bytes": int(
            getattr(diagnostics, "planned_collective_admitted_bytes", 0)
        ),
        "planned_local_typed_bytes": int(
            getattr(diagnostics, "planned_local_typed_bytes", 0)
        ),
        "planned_non_admitted_typed_bytes": int(
            getattr(diagnostics, "planned_non_admitted_typed_bytes", 0)
        ),
        "planned_generic_residual_bytes": int(
            getattr(diagnostics, "planned_generic_residual_bytes", 0)
        ),
        "collective_lowered_bytes": int(
            getattr(diagnostics, "collective_lowered_bytes", 0)
        ),
        "planner_reject_reason_buckets": dict(
            getattr(diagnostics, "planner_reject_reason_buckets", {})
        ),
        "planner_version": getattr(diagnostics, "planner_version", None),
        "plan_hash": getattr(diagnostics, "plan_hash", None),
        "estimated_collective_peak_temporary_bytes": int(
            getattr(diagnostics, "estimated_collective_peak_temporary_bytes", 0)
        ),
        "estimated_collective_batch_bytes": int(
            getattr(diagnostics, "estimated_collective_batch_bytes", 0)
        ),
        "estimated_collective_dedup_saving_bytes": int(
            getattr(diagnostics, "estimated_collective_dedup_saving_bytes", 0)
        ),
    }
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def is_runtime_binding_swap_capable(binding: Any) -> bool:
    return bool(
        getattr(binding, "swap_capable", False)
        or callable(getattr(binding, "swap", None))
    )


def local_ready_current_value_summary_fields(
    current_value: Any,
    *,
    require_local_serving_ref: bool = False,
) -> dict[str, Any]:
    local_serving_ref = getattr(current_value, "local_serving_ref", None)
    if require_local_serving_ref and not local_serving_ref:
        raise ServingIntegrationError(
            "TensorCast local-ready current value did not include local_serving_ref"
        )
    return {
        "binding_value_id": getattr(current_value, "binding_value_id", None),
        "verification_state": binding_value_verification_state_name(current_value),
        "local_serving_ref": local_serving_ref,
    }


def binding_value_ref_from_current_value(current_value: Any) -> Any | None:
    to_ref = getattr(current_value, "to_binding_value_ref", None)
    if callable(to_ref):
        return to_ref()
    binding_id = str(getattr(current_value, "binding_id", "") or "")
    binding_layout_id = str(getattr(current_value, "binding_layout_id", "") or "")
    binding_value_id = str(getattr(current_value, "binding_value_id", "") or "")
    if not (binding_id and binding_layout_id and binding_value_id):
        return None
    return BindingValueRef(
        binding_id=binding_id,
        binding_layout_id=binding_layout_id,
        binding_value_id=binding_value_id,
        seal_generation=int(getattr(current_value, "seal_generation", 0) or 0),
    )


def build_local_ready_prepared_artifact(
    *,
    source_artifact_ref: str,
    serving_manifest_ref: str,
    representation_contract_hash: str,
    serving_build_digest: str,
    tensor_schema_hash: str,
    current_value: Any,
    binding: Any,
    family: str,
    tp_rank: int,
    tp_world_size: int,
    source_bound_contract_state: SourceBoundContractState,
    source_bound_contract_path: str,
    artifact_realization_report: ArtifactRealizationReport | None = None,
    model_runtime_spec: ArtifactRealizationSpec | None = None,
) -> LocalReadyServingResult:
    current_value_fields = local_ready_current_value_summary_fields(
        current_value,
        require_local_serving_ref=True,
    )
    local_serving_ref = current_value_fields["local_serving_ref"]
    verification_state = str(
        current_value_fields["verification_state"] or "local_ready"
    )
    verification_job_id = getattr(current_value, "verification_job_id", None)
    binding_value_ref = binding_value_ref_from_current_value(current_value)
    binding_layout_id = getattr(binding, "binding_layout_id", None)
    execution_report = _strip_report_prefix(
        execution_diagnostics_summary_fields(
            getattr(binding, "last_execution_diagnostics", None),
            prefix="realize",
        ),
        prefix="realize",
    )
    plan_report = _strip_report_prefix(
        source_bound_plan_diagnostics_summary_fields(
            getattr(binding, "last_source_bound_plan_diagnostics", None),
            prefix="realize",
        ),
        prefix="realize",
    )
    realization_report = tc_diagnostics.ServingRealizationReport(
        source_artifact_ref=source_artifact_ref,
        serving_manifest_ref=serving_manifest_ref,
        representation_contract_hash=representation_contract_hash,
        serving_build_digest=serving_build_digest,
        tensor_schema_hash=tensor_schema_hash,
        family=family,
        tp_rank=int(tp_rank),
        tp_world_size=int(tp_world_size),
        source_bound_contract=tc_diagnostics.SourceContractReport(
            version=(source_bound_contract_state.source_bound_contract_version),
            capability_flags=tuple(
                source_bound_contract_state.source_bound_capability_names
            ),
            ready=source_bound_contract_state.source_bound_contract_ready,
            path=source_bound_contract_path,
        ),
        realization=tc_diagnostics.RealizationReport(
            binding_layout_id=binding_layout_id,
            binding_value=tc_diagnostics.BindingValueReport(
                verification_state=verification_state,
                verification_job_id=verification_job_id,
                local_serving_ref=local_serving_ref,
                binding_value_id=current_value_fields["binding_value_id"],
            ),
            execution=execution_report,
            plan=plan_report,
        ),
    )
    diagnostics = realization_report.to_runtime_diagnostics()
    if artifact_realization_report is not None:
        diagnostics["artifact_realization_report"] = (
            artifact_realization_report_to_dict(artifact_realization_report)
        )
    runtime_view = RuntimeBindingView(
        serving_artifact_ref=None,
        source_artifact_ref=source_artifact_ref,
        representation_contract_hash=representation_contract_hash,
        tensor_schema_hash=tensor_schema_hash,
        binding_value_ref=binding_value_ref,
        local_serving_ref=local_serving_ref,
        readiness="serving_local_ready",
        diagnostics=diagnostics,
    )
    runtime_state = runtime_binding_state_from_runtime_view(
        binding=binding,
        runtime_view=runtime_view,
        artifact_ref=source_artifact_ref,
        artifact_realization_report=artifact_realization_report,
        model_runtime_spec=model_runtime_spec,
    )
    prepared = PreparedServingArtifact(
        source_artifact_ref=source_artifact_ref,
        serving_artifact_ref=None,
        serving_manifest_ref=serving_manifest_ref,
        representation_contract_hash=representation_contract_hash,
        serving_build_digest=serving_build_digest,
        binding_value_ref=binding_value_ref,
        readiness="serving_local_ready",
        family=family,
        tensor_schema_hash=tensor_schema_hash,
        binding_layout_id=binding_layout_id,
        local_serving_ref=local_serving_ref,
        verification_state=verification_state,
        verification_job_id=verification_job_id,
        tp_rank=int(tp_rank),
        tp_world_size=int(tp_world_size),
    )
    return LocalReadyServingResult(
        runtime_state=runtime_state,
        runtime_view=runtime_view,
        prepared=prepared,
        binding_value=prepared.to_binding_value(),
        realization_report=artifact_realization_report,
    )


def _strip_report_prefix(fields: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    prefix_text = f"{prefix}_"
    return {
        key.removeprefix(prefix_text): value
        for key, value in fields.items()
        if key.startswith(prefix_text)
    }


def build_collective_group_id(
    *,
    artifact_ref: str,
    operation_scope: str,
    tp_ranks: tuple[int, ...],
    contract_identity: str | None = None,
) -> str:
    payload_dict: dict[str, Any] = {
        "artifact_ref": str(artifact_ref),
        "operation_scope": operation_scope,
        "tp_ranks": [int(rank) for rank in tp_ranks],
    }
    if contract_identity:
        payload_dict["contract_identity"] = str(contract_identity)
    payload = json.dumps(payload_dict, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"tensorcast-{digest}"


@dataclass(frozen=True)
class ServingIntegration:
    """Small service object for framework-facing serving lifecycle calls."""

    resolver: ServingArtifactResolver | None = None
    profile_sink: Any | None = None
    host: IntegrationHost | None = None

    @staticmethod
    def _lifecycle_not_implemented(method: str, phase: str) -> None:
        raise ServingIntegrationNotImplementedError(
            f"ServingIntegration.{method} request DTO is available, but the "
            f"deep core-owned lifecycle is scheduled for {phase}"
        )

    def resolve(self, artifact_ref: str, **kwargs: Any) -> ResolvedServingArtifact:
        return resolve_serving_artifact(
            artifact_ref,
            resolver=self.resolver,
            **kwargs,
        )

    def read_manifest(
        self,
        artifact: Any,
        *,
        artifact_ref: str,
    ) -> ResolvedServingArtifact:
        if self.resolver is None:
            raise ValueError("ServingIntegration.read_manifest requires resolver")
        return read_serving_artifact_manifest(
            artifact,
            artifact_ref=artifact_ref,
            resolver=self.resolver,
        )

    def cross_check(
        self,
        resolved_artifact: ResolvedServingArtifact,
        **kwargs: Any,
    ) -> ResolvedServingArtifact:
        if self.resolver is None:
            raise ValueError("ServingIntegration.cross_check requires resolver")
        return cross_check_serving_artifact(
            resolved_artifact,
            resolver=self.resolver,
            **kwargs,
        )

    def start(
        self,
        intent: ServingIntent,
        context: RequestContext,
    ) -> RuntimeAttachment:
        """Start serving from a public intent DTO."""

        decision = self._admit_intent(intent, context)
        if isinstance(intent, ExistingServingArtifact):
            self._reject_source_selector_for_existing_artifact(intent.artifact_locator)
            materialization_request = self._host_materialization_request(
                context,
                operation_scope="startup.direct_serving_artifact.bind",
            )
            load_result = self._load_existing_serving_artifact(
                _DirectServingLoad(
                    artifact_locator=intent.artifact_locator,
                    policy=intent.policy,
                    framework_config=context.framework_config,
                    model_config=context.model_config,
                    target_device=context.target_device,
                    timeout_s=context.timeout_s,
                    configured_collective_policy=(
                        materialization_request.configured_collective_policy
                    ),
                    source_bound_contract_state=(
                        materialization_request.source_bound_contract_state
                    ),
                    source_bound_contract_path=(
                        materialization_request.source_bound_contract_path
                    ),
                    execution_facts=materialization_request.execution_facts,
                    operation_scope=materialization_request.operation_scope,
                    require_materialization_options=(
                        materialization_request.require_materialization_options
                    ),
                )
            )
            if load_result.model is None or load_result.runtime_state is None:
                raise ServingIntegrationError(
                    "ServingIntegration.start returned no model/state for "
                    "ExistingServingArtifact"
                )
            return self._attachment_from_load_result(load_result, decision)
        if isinstance(intent, RetainedBindingAcquire):
            authority = intent.authority
            expected_member = authority.member
            if self.host is not None:
                placement = self._framework_context(
                    context.framework_config,
                    context.model_config,
                ).placement
                if (
                    placement is not None
                    and placement.member is not None
                    and placement.member != authority.member
                ):
                    raise AuthorityValidationError(
                        "ParsedRetainedServingBindingAuthority.member does not match "
                        "runtime placement",
                        details={
                            "authority_member": repr(authority.member),
                            "placement_member": repr(placement.member),
                        },
                    )
                if placement is not None and placement.member is not None:
                    expected_member = placement.member
            retained_result = self._restore_retained_for_intent(
                _RetainedBindingAcquire(
                    authority=authority,
                    framework_config=context.framework_config,
                    model_config=context.model_config,
                    target_device=context.target_device,
                    expected_member=expected_member,
                    timeout_s=context.timeout_s,
                )
            )
            if retained_result.model is None or retained_result.runtime_state is None:
                raise ServingIntegrationError(
                    "ServingIntegration.start returned no model/state for "
                    "RetainedBindingAcquire"
                )
            return self._attachment_from_retained_result(retained_result, decision)
        if isinstance(intent, LocalSourceBootstrap):
            local_ready_result = self._prepare_local_source_bootstrap(
                self._local_source_bootstrap_request(
                    intent,
                    context,
                    decision=decision,
                )
            )
            if (
                local_ready_result.model is None
                or local_ready_result.runtime_state is None
            ):
                raise ServingIntegrationError(
                    "ServingIntegration.start returned no model/state for "
                    "LocalSourceBootstrap"
                )
            self._run_local_ready_barrier(context)
            return self._attachment_from_local_ready_result(
                local_ready_result,
                decision,
            )
        raise ServingIntegrationError(
            f"Unsupported TensorCast serving intent: {type(intent).__name__}"
        )

    def reload(
        self,
        current_state: RuntimeBindingState | Any,
        intent: ExistingServingArtifact,
        context: RequestContext,
        *,
        model: object | None = None,
        contract_identity: str | None = None,
    ) -> RuntimeAttachment:
        """Reload an existing runtime binding from a public serving intent."""

        if not isinstance(intent, ExistingServingArtifact):
            raise ServingIntegrationError(
                "ServingIntegration.reload currently accepts "
                "ExistingServingArtifact intent only"
            )
        self._reject_source_selector_for_existing_artifact(intent.artifact_locator)
        decision = self._admit_intent(intent, context, reload=True)
        materialization_request = self._host_materialization_request(
            context,
            operation_scope="runtime_binding.swap",
        )
        result = self._reload_existing_serving_artifact(
            _ServingReload(
                current_state=current_state,
                artifact_locator=intent.artifact_locator,
                policy=intent.policy,
                framework_config=context.framework_config,
                model_config=context.model_config,
                target_device=context.target_device,
                contract_identity=contract_identity,
                model=model,
                configured_collective_policy=(
                    materialization_request.configured_collective_policy
                ),
                source_bound_contract_state=(
                    materialization_request.source_bound_contract_state
                ),
                source_bound_contract_path=(
                    materialization_request.source_bound_contract_path
                ),
                execution_facts=materialization_request.execution_facts,
                operation_scope=materialization_request.operation_scope,
                require_materialization_options=(
                    materialization_request.require_materialization_options
                ),
            )
        )
        if result.runtime_state is None:
            raise ServingIntegrationError(
                "ServingIntegration.reload returned no runtime state"
            )
        runtime_model = (
            model if model is not None else getattr(current_state, "model", None)
        )
        view = self._worker_view_from_state(
            result.runtime_state,
            decision=decision,
            include_reload_response=True,
        )
        return RuntimeAttachment(
            model=runtime_model,
            state=result.runtime_state,
            view=view,
        )

    def describe(self, state: RuntimeBindingState | Any) -> RuntimeWorkerView:
        """Return the typed endpoint/worker projection for core runtime state."""

        if isinstance(state, RuntimeWorkerView):
            return state
        return self._worker_view_from_state(state, decision=None)

    def _admit_intent(
        self,
        intent: ServingIntent,
        context: RequestContext,
        *,
        reload: bool = False,
    ) -> AdmissionDecision | None:
        if self.host is None:
            return None
        if context.model_config is None:
            raise ServingIntegrationError(
                "ServingIntegration host admission requires model_config"
            )
        framework_identity = self.host.framework.identity(context.model_config)
        placement_identity = self.host.placement.identity_facts(
            context.framework_config
        )
        placement_admission = self.host.placement.admission_facts(
            context.framework_config
        )
        request = AdmissionRequest(
            intent=intent,
            framework_identity=framework_identity,
            placement_identity=placement_identity,
            placement_admission=placement_admission,
            model_config=context.model_config,
            runtime_profile=self.host.runtime_profile or RuntimeProfile(),
        )
        policy = self.host.admission or DefaultAdmissionPolicy()
        decision = policy.admit(request)
        endpoint_fields = dict(decision.endpoint_fields)
        endpoint_fields.setdefault("family", decision.family)
        endpoint_fields.setdefault("tp_rank", placement_identity.tensor_parallel_rank)
        endpoint_fields.setdefault(
            "tp_world_size", placement_identity.tensor_parallel_size
        )
        decision = replace(decision, endpoint_fields=endpoint_fields)
        if reload:
            allowed = decision.reload_allowed
            action = "reload"
        elif isinstance(intent, LocalSourceBootstrap):
            allowed = decision.local_bootstrap_allowed
            action = "local bootstrap"
        else:
            allowed = decision.startup_allowed
            action = "startup"
        if not allowed:
            raise ServingIntegrationError(
                "TensorCast admission rejected "
                f"{action}: family={decision.family!r}, "
                f"support_level={decision.support_level!r}"
            )
        return decision

    @staticmethod
    def _reject_source_selector_for_existing_artifact(
        artifact_locator: object,
    ) -> None:
        if isinstance(artifact_locator, SourceSelector):
            raise ServingIntegrationError(
                "ExistingServingArtifact requires a durable serving artifact "
                "locator; local source selectors must use LocalSourceBootstrap"
            )
        if _artifact_locator_kind(artifact_locator) == "local_path":
            raise ServingIntegrationError(
                "ExistingServingArtifact rejects local_path artifact locators; use "
                "LocalSourceBootstrap for local source acquisition"
            )

    def _attachment_from_load_result(
        self,
        result: ServingLoadResult,
        decision: AdmissionDecision | None,
    ) -> RuntimeAttachment:
        state = result.runtime_state
        if state is None or result.model is None:
            raise ServingIntegrationError(
                "ServingLoadResult is missing model or runtime_state"
            )
        return RuntimeAttachment(
            model=result.model,
            state=state,
            view=self._worker_view_from_state(state, decision=decision),
        )

    def _attachment_from_retained_result(
        self,
        result: RetainedBindingResult,
        decision: AdmissionDecision | None,
    ) -> RuntimeAttachment:
        state = result.runtime_state
        if state is None or result.model is None:
            raise ServingIntegrationError(
                "RetainedBindingResult is missing model or runtime_state"
            )
        return RuntimeAttachment(
            model=result.model,
            state=state,
            view=self._worker_view_from_state(state, decision=decision),
        )

    def _attachment_from_local_ready_result(
        self,
        result: LocalReadyServingResult,
        decision: AdmissionDecision | None,
    ) -> RuntimeAttachment:
        state = result.runtime_state
        if state is None or result.model is None:
            raise ServingIntegrationError(
                "LocalReadyServingResult is missing model or runtime_state"
            )
        return RuntimeAttachment(
            model=result.model,
            state=state,
            view=self._worker_view_from_state(state, decision=decision),
            prepared=result.prepared,
            recipe=result.recipe,
        )

    def _local_source_bootstrap_request(
        self,
        intent: LocalSourceBootstrap,
        context: RequestContext,
        *,
        decision: AdmissionDecision | None,
    ) -> _LocalReadyBootstrap:
        if self.host is None:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires "
                "IntegrationHost"
            )
        if context.model_config is None:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires model_config"
            )
        profile = self.host.runtime_profile or RuntimeProfile()
        identity = self.host.framework.identity(context.model_config)
        placement_identity = self.host.placement.identity_facts(
            context.framework_config
        )
        placement = self._host_serving_placement(context.framework_config)
        recipe = getattr(intent, "recipe", None)
        model = getattr(intent, "model", None)
        coordinator = getattr(intent, "coordinator", None)
        if coordinator is None:
            coordinator = self._host_source_subject_coordinator(
                context.framework_config
            )
        source_catalog_config = getattr(intent, "source_catalog_config", None)
        if source_catalog_config is None:
            source_catalog_config = self._host_source_catalog_config(
                context.framework_config,
                context.model_config,
            )
        cache_config = intent.cache_policy
        if cache_config is None:
            cache_config = self._host_recipe_cache_policy(
                context.framework_config,
                context.model_config,
            )
        materialization_request = self._host_materialization_request(
            context,
            operation_scope="bootstrap.same_binding_fast_path.tensorcast_realize",
        )
        return _LocalReadyBootstrap(
            source_selector=intent.source_selector,
            bootstrap=intent.bootstrap_policy,
            cache_config=cache_config,
            source_subject_coordinator=coordinator,
            recipe=recipe,
            source_subject=getattr(intent, "source_subject", None),
            source_artifact_ref=getattr(intent, "source_artifact_ref", None),
            source_catalog_config=source_catalog_config,
            cache_config_factory=getattr(intent, "cache_config_factory", None),
            framework_config=context.framework_config,
            model_config=context.model_config,
            target_device=context.target_device,
            model=model,
            manifest_tensor_name=profile.manifest_policy.manifest_tensor_name,
            placement=placement,
            family=(decision.family if decision is not None else ""),
            tp_rank=placement_identity.tensor_parallel_rank,
            tp_world_size=placement_identity.tensor_parallel_size,
            build_recipe_from_framework_context=recipe is None,
            build_model_from_framework_context=model is None,
            build_manifest_carrier_from_framework_context=True,
            run_binding_finalize_hooks_when_required=True,
            run_post_bind_finalize=True,
            validate_representation_contract_hash=True,
            runtime_binding_schema_version=self._runtime_binding_schema_version(
                profile
            ),
            serving_artifact_schema_version=self._serving_artifact_schema_version(
                profile
            ),
            framework_name=identity.framework_name,
            framework_version=identity.framework_version,
            adapter_version=identity.adapter_version,
            serving_abi_version=identity.serving_abi_version,
            binding_factory=getattr(intent, "binding_factory", None),
            configured_collective_policy=(
                materialization_request.configured_collective_policy
            ),
            source_bound_contract_state=(
                materialization_request.source_bound_contract_state
            ),
            source_bound_contract_path=(
                materialization_request.source_bound_contract_path
            ),
            execution_facts=materialization_request.execution_facts,
            operation_scope=materialization_request.operation_scope,
            require_materialization_options=(
                materialization_request.require_materialization_options
            ),
        )

    def _host_source_subject_coordinator(
        self,
        framework_config: object | None,
    ) -> SourceSubjectCoordinator | None:
        if self.host is None or self.host.collective is None:
            return None
        return self.host.collective.source_subject_coordinator(framework_config)

    def _host_source_catalog_config(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> Any | None:
        if self.host is None or self.host.source is None:
            return None
        return self.host.source.source_catalog_config(
            framework_config,
            model_config,
        )

    def _host_recipe_cache_policy(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> RecipeCachePolicy | None:
        if self.host is None or self.host.source is None:
            return None
        policy = self.host.source.recipe_cache_policy(
            framework_config,
            model_config,
        )
        if policy is not None and not isinstance(policy, RecipeCachePolicy):
            raise ServingIntegrationError(
                "IntegrationHost.source.recipe_cache_policy must return "
                "RecipeCachePolicy or None"
            )
        return policy

    def _run_local_ready_barrier(self, context: RequestContext) -> None:
        if self.host is None or self.host.collective is None:
            return
        self.host.collective.local_ready_barrier(
            context.framework_config,
            context.target_device,
        )

    @staticmethod
    def _worker_view_from_state(
        state: RuntimeBindingState | Any,
        *,
        decision: AdmissionDecision | None,
        include_reload_response: bool = False,
    ) -> RuntimeWorkerView:
        runtime_view = getattr(state, "runtime_view", None)
        if runtime_view is None:
            raise ServingIntegrationError(
                "ServingIntegration.describe requires state.runtime_view"
            )
        endpoint_fields = dict(decision.endpoint_fields) if decision else {}
        return RuntimeWorkerView.from_runtime_view(
            runtime_view,
            family=str(
                endpoint_fields.get(
                    "family",
                    decision.family if decision is not None else "",
                )
            ),
            tp_rank=_optional_int(endpoint_fields.get("tp_rank")),
            tp_world_size=_optional_int(endpoint_fields.get("tp_world_size")),
            include_reload_response=include_reload_response,
        )

    def _host_materialization_request(
        self,
        context: RequestContext,
        *,
        operation_scope: str,
    ) -> _HostMaterializationRequest:
        if self.host is None:
            return _HostMaterializationRequest(operation_scope=operation_scope)
        profile = self.host.runtime_profile or RuntimeProfile()
        return _HostMaterializationRequest(
            configured_collective_policy=CollectivePolicy(
                _collective_policy_value(profile.materialization_policy)
            ),
            source_bound_contract_state=read_source_bound_contract_state(),
            source_bound_contract_path=SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4,
            execution_facts=_execution_facts_payload(
                self.host.placement.execution_facts(context.framework_config)
            ),
            operation_scope=operation_scope,
            require_materialization_options=True,
        )

    def _host_serving_placement(
        self,
        framework_config: object | None,
    ) -> ServingPlacement:
        if self.host is None:
            raise ServingIntegrationError(
                "ServingIntegration host placement requires IntegrationHost"
            )
        framework_payload = None
        framework_payload_fn = getattr(self.host.placement, "framework_payload", None)
        if callable(framework_payload_fn):
            framework_payload = _framework_payload_mapping(
                framework_payload_fn(framework_config)
            )
        identity_payload = None
        identity_payload_fn = getattr(self.host.placement, "identity_payload", None)
        if callable(identity_payload_fn):
            identity_payload = _framework_payload_mapping(
                identity_payload_fn(framework_config)
            )
        return serving_placement_from_framework_facts(
            identity_facts=self.host.placement.identity_facts(framework_config),
            admission_facts=self.host.placement.admission_facts(framework_config),
            member_facts=self.host.placement.member_facts(framework_config),
            framework_payload=framework_payload,
            identity_payload=identity_payload,
        )

    @staticmethod
    def _runtime_binding_schema_version(profile: RuntimeProfile) -> int:
        value = profile.runtime_config.fields.get("runtime_binding_schema_version", 1)
        return _optional_int(value) or 1

    @staticmethod
    def _serving_artifact_schema_version(profile: RuntimeProfile) -> int:
        value = profile.manifest_policy.fields.get(
            "serving_artifact_schema_version", None
        )
        if value is None:
            model_fields = getattr(ServingArtifactManifest, "model_fields", {})
            schema_field = model_fields.get("schema_version")
            value = getattr(schema_field, "default", 1)
        return _optional_int(value) or 1

    def _load_existing_serving_artifact(
        self, request: _DirectServingLoad
    ) -> ServingLoadResult:
        target_device = self._require_target_device(request.target_device)
        context = self._framework_context(
            request.framework_config,
            request.model_config,
        )
        preflight = self._preflight_serving_artifact(
            resolved_artifact=request.resolved_artifact,
            artifact_ref=request.artifact_ref,
            artifact_locator=request.artifact_locator,
            expected_tensor_schema_hash=None,
            policy=request.policy,
            placement=context.placement,
        )
        resolved = preflight.resolved_artifact
        policy = preflight.serving_runtime_policy
        model = request.model
        if model is None:
            self._prepare_model_construction(
                request.framework_config,
                request.model_config,
            )
            model = self._build_meta_model(
                request.framework_config,
                request.model_config,
            )
        self._assert_model_ready_for_runtime_binding(
            model,
            context="TensorCast direct serving artifact startup",
        )
        self._align_runtime_tensor_names(
            model,
            getattr(resolved, "tensor_names", ()),
        )
        current_tensors = self._collect_runtime_binding_tensors(
            model,
            remove_duplicate=False,
        )
        self._assert_tensor_names_match_expected(
            current_tensors,
            getattr(resolved, "tensor_names", ()),
        )
        tensor_schema_hash = self._compute_runtime_tensor_schema_hash(
            current_tensors,
            remove_duplicate=False,
        )
        preflight = self._preflight_serving_artifact(
            resolved_artifact=resolved,
            artifact_ref=request.artifact_ref,
            artifact_locator=request.artifact_locator,
            expected_tensor_schema_hash=tensor_schema_hash,
            policy=policy,
            placement=context.placement,
        )
        resolved = preflight.resolved_artifact
        policy = preflight.serving_runtime_policy
        manifest = getattr(resolved, "manifest", None)
        local_serving_ref = getattr(manifest, "local_serving_ref", None)
        if local_serving_ref:
            expected_member = request.expected_member
            if expected_member is None and context.placement is not None:
                expected_member = context.placement.member
            if expected_member is None:
                raise RestoreBindingError(
                    "ServingIntegration._load_existing_serving_artifact prepared "
                    "local-ready restore requires expected_member"
                )
            with restore_prepared_local_ready_binding(
                resolved_artifact=resolved,
                target_device=target_device,
                expected_member=expected_member,
                expected_tensor_schema_hash=tensor_schema_hash,
                expected_serving_build_digest=getattr(
                    manifest, "serving_build_digest", None
                ),
                caller_pid=os.getpid(),
                timeout_s=request.timeout_s,
            ) as restored:
                binding_result = RuntimeBindingResult.from_binding(restored)
                authority = getattr(restored, "authority", None)
                if authority is None:
                    artifact_report = _runtime_attachment_report_for_artifact_id(
                        artifact_id=str(getattr(resolved, "artifact_ref", "")),
                        tensors=binding_result.tensors,
                        binding_handle=restored,
                        target_device=target_device,
                        tensor_schema_hash=tensor_schema_hash,
                        artifact_profile="retained_binding",
                        authority_scope="daemon_retained_runtime_attachment",
                        retained=True,
                        reservation_bytes=int(restored.reservation_bytes),
                    )
                else:
                    artifact_report = _runtime_attachment_report_for_retained(
                        authority=authority,
                        tensors=binding_result.tensors,
                        binding_handle=restored,
                        target_device=target_device,
                        tensor_schema_hash=tensor_schema_hash,
                        reservation_bytes=restored.reservation_bytes,
                    )
                state_seed = self._state_seed(
                    resolved,
                    tensor_schema_hash=tensor_schema_hash,
                    execution_diagnostics=binding_result.execution_diagnostics,
                    materialization_diagnostics=(
                        binding_result.materialization_diagnostics
                    ),
                    binding_handle=restored,
                    artifact_realization_report=artifact_report,
                    readiness="serving_local_ready",
                )
                runtime_state = self._materializer().attach_and_finalize(
                    model=model,
                    tensors=binding_result.tensors,
                    binding_handle=restored,
                    context=context,
                    state_seed=state_seed,
                    replace_meta_params=True,
                    target_device=target_device,
                    model_config=request.model_config,
                )
        else:
            materialization = self._load_materialization_options(
                request,
                resolved,
            )
            binding_result = bind_serving_artifact(
                resolved_artifact=resolved,
                tensor_names=tuple(current_tensors.keys()),
                device=target_device,
                serving_runtime_policy=policy,
                options=materialization,
            )
            artifact_report = _runtime_attachment_report_for_resolved(
                resolved=resolved,
                tensors=binding_result.tensors,
                binding_handle=binding_result.binding,
                target_device=target_device,
                tensor_schema_hash=tensor_schema_hash,
                execution_diagnostics=binding_result.execution_diagnostics,
                materialization_diagnostics=binding_result.materialization_diagnostics,
            )
            state_seed = self._state_seed(
                resolved,
                tensor_schema_hash=tensor_schema_hash,
                execution_diagnostics=binding_result.execution_diagnostics,
                materialization_diagnostics=binding_result.materialization_diagnostics,
                binding_handle=binding_result.binding,
                artifact_realization_report=artifact_report,
            )
            runtime_state = self._materializer().attach_and_finalize(
                model=model,
                tensors=binding_result.tensors,
                binding_handle=binding_result.binding,
                context=context,
                state_seed=state_seed,
                replace_meta_params=True,
                target_device=target_device,
                model_config=request.model_config,
            )
        return ServingLoadResult(
            model=model,
            runtime_state=runtime_state,
            runtime_view=runtime_state.runtime_view,
            resolved_artifact=resolved,
            binding_result=binding_result,
        )

    def _reload_existing_serving_artifact(
        self, request: _ServingReload
    ) -> ServingReloadResult:
        target_device = (
            torch.device(request.target_device)
            if request.target_device is not None
            else None
        )
        binding = getattr(request.current_state, "binding", None)
        if binding is None:
            raise ServingIntegrationError(
                "ServingIntegration._reload_existing_serving_artifact requires current_state.binding"
            )
        if not is_runtime_binding_swap_capable(binding):
            raise ServingIntegrationError(
                "ServingIntegration._reload_existing_serving_artifact requires a "
                "swap-capable serving binding"
            )
        current_view = getattr(request.current_state, "runtime_view", None)
        expected_tensor_schema_hash = getattr(current_view, "tensor_schema_hash", None)
        runtime_tensors = None
        if request.model is not None:
            runtime_tensors = self._collect_runtime_binding_tensors(
                request.model,
                remove_duplicate=False,
            )
            expected_tensor_schema_hash = self._compute_runtime_tensor_schema_hash(
                runtime_tensors,
                remove_duplicate=False,
            )
            if target_device is None:
                for tensor in runtime_tensors.values():
                    target_device = torch.device(tensor.device)
                    break
        target_device = self._require_target_device(target_device)
        context = None
        if (
            request.model is not None
            or _artifact_locator_kind(request.artifact_locator) == "ranked_version_key"
        ):
            context = self._framework_context(
                request.framework_config,
                request.model_config,
            )
        placement = None if context is None else context.placement
        preflight = self._preflight_serving_artifact(
            resolved_artifact=request.resolved_artifact,
            artifact_ref=request.artifact_ref,
            artifact_locator=request.artifact_locator,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            policy=request.policy,
            placement=placement,
        )
        resolved = preflight.resolved_artifact
        policy = preflight.serving_runtime_policy
        materialization = self._reload_materialization_options(
            request,
            resolved,
        )
        binding_result = swap_serving_artifact(
            binding=binding,
            resolved_artifact=resolved,
            tensor_names=(
                None if runtime_tensors is None else tuple(runtime_tensors.keys())
            ),
            serving_runtime_policy=policy,
            options=materialization,
        )
        artifact_report = _runtime_attachment_report_for_resolved(
            resolved=resolved,
            tensors=binding_result.tensors,
            binding_handle=binding_result.binding,
            target_device=target_device,
            tensor_schema_hash=str(expected_tensor_schema_hash or ""),
            execution_diagnostics=binding_result.execution_diagnostics,
            materialization_diagnostics=binding_result.materialization_diagnostics,
        )
        state_seed = self._state_seed(
            resolved,
            tensor_schema_hash=str(expected_tensor_schema_hash or ""),
            execution_diagnostics=binding_result.execution_diagnostics,
            materialization_diagnostics=binding_result.materialization_diagnostics,
            binding_handle=binding_result.binding,
            artifact_realization_report=artifact_report,
        )
        if request.model is not None:
            context = context or self._framework_context(
                request.framework_config,
                request.model_config,
            )
            runtime_state = self._materializer().attach_and_finalize(
                model=request.model,
                tensors=binding_result.tensors,
                binding_handle=binding_result.binding,
                context=context,
                state_seed=state_seed,
                replace_meta_params=False,
                target_device=target_device,
                model_config=request.model_config,
                run_process_after_load=False,
            )
        else:
            realization_handle = _runtime_attachment_realization_handle(
                report=artifact_report,
                binding_handle=binding_result.binding,
                owner=(
                    getattr(request.current_state, "ownership_handle", None)
                    or binding_result.binding
                ),
            )
            runtime_state = RuntimeBindingState(
                binding=binding_result.binding,
                artifact_ref=state_seed.artifact_ref,
                runtime_view=state_seed.runtime_view(),
                ownership_handle=getattr(
                    request.current_state, "ownership_handle", None
                ),
                release_contract=None
                if realization_handle is None
                else realization_handle.release_contract,
                realization_handle=realization_handle,
            )
        return ServingReloadResult(
            runtime_state=runtime_state,
            runtime_view=runtime_state.runtime_view,
            resolved_artifact=resolved,
            binding_result=binding_result,
        )

    def _restore_retained_for_intent(
        self, request: _RetainedBindingAcquire
    ) -> RetainedBindingResult:
        target_device = self._require_target_device(request.target_device)
        authority = request.authority
        if authority is None:
            raise RestoreBindingError(
                "ServingIntegration._restore_retained_for_intent requires authority"
            )
        readiness = getattr(authority, "readiness", None)
        if readiness == "serving_reserved":
            raise RestoreBindingError(
                "TensorCast retained acquire readiness='serving_reserved' "
                "is not attachable"
            )
        if readiness in {
            "serving_group_prepared",
            "serving_group_published_ready",
        }:
            raise RestoreBindingError(
                "TensorCast retained acquire group readiness requires a "
                "published group-realization transaction authority"
            )
        if readiness == "serving_published_ready":
            raise RestoreBindingError(
                "TensorCast retained acquire readiness='serving_published_ready' "
                "requires a swap-capable serving binding handle"
            )
        model = self._build_meta_model(
            request.framework_config,
            request.model_config,
        )
        try:
            with restore_retained_binding(
                authority=authority,
                target_device=target_device,
                expected_member=request.expected_member,
                caller_pid=os.getpid(),
                timeout_s=request.timeout_s,
                runtime=request.runtime,
                client=request.client,
                restore_fn=request.restore_fn,
            ) as restored:
                expected = getattr(authority, "expected", None)
                expected_tensor_schema_hash = getattr(
                    expected, "tensor_schema_hash", None
                )
                artifact_report = _runtime_attachment_report_for_retained(
                    authority=authority,
                    tensors=restored.tensors,
                    binding_handle=restored,
                    target_device=target_device,
                    tensor_schema_hash=str(expected_tensor_schema_hash or ""),
                    reservation_bytes=restored.reservation_bytes,
                )
                state_seed = RuntimeStateSeed(
                    artifact_ref=(
                        getattr(authority, "serving_artifact_id", None)
                        or getattr(authority, "local_serving_ref", None)
                        or ""
                    ),
                    serving_artifact_ref=getattr(
                        authority, "serving_artifact_id", None
                    ),
                    tensor_schema_hash=str(expected_tensor_schema_hash or ""),
                    binding_value_ref=restored.binding_value_ref,
                    local_serving_ref=getattr(authority, "local_serving_ref", None),
                    readiness=str(
                        getattr(authority, "readiness", "") or "serving_local_ready"
                    ),
                    diagnostics={
                        "reservation_bytes": int(restored.reservation_bytes),
                        "verification_state": str(
                            getattr(authority, "verification_state", "") or ""
                        ),
                        "artifact_realization_report": (
                            artifact_realization_report_to_dict(artifact_report)
                        ),
                    },
                    realization_report=artifact_report,
                )
                runtime_state = self._materializer().attach_and_finalize(
                    model=model,
                    tensors=restored.tensors,
                    binding_handle=restored,
                    context=self._framework_context(
                        request.framework_config,
                        request.model_config,
                    ),
                    state_seed=state_seed,
                    replace_meta_params=True,
                    target_device=target_device,
                    model_config=request.model_config,
                    run_process_after_load=False,
                    expected_tensor_schema_hash=expected_tensor_schema_hash,
                )
                return RetainedBindingResult(
                    model=model,
                    runtime_state=runtime_state,
                    runtime_view=runtime_state.runtime_view,
                    restored=restored,
                )
        except (AttachFinalizeError, OwnershipTransferError, SchemaMismatchError):
            raise
        except Exception as exc:
            raise RestoreBindingError(
                "TensorCast retained binding acquire failed"
            ) from exc

    def _prepare_local_source_bootstrap(
        self, request: _LocalReadyBootstrap
    ) -> LocalReadyServingResult:
        if (
            request.recipe is None or request.source_subject is None
        ) and request.build_recipe_from_framework_context:
            request = self._local_ready_prepare_with_built_recipe(request)
        if request.recipe is None or request.source_subject is None:
            self._lifecycle_not_implemented("_prepare_local_source_bootstrap", "P5")
        if request.target_device is None:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires target_device"
            )
        if not request.manifest_tensor_name:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires manifest_tensor_name"
            )
        model = request.model
        if request.build_model_from_framework_context and model is None:
            if request.model_config is None:
                raise ServingIntegrationError(
                    "ServingIntegration.start(LocalSourceBootstrap) requires "
                    "model_config to build a framework model"
                )
            model = self._build_meta_model(
                request.framework_config, request.model_config
            )
        manifest_bytes = request.manifest_bytes
        serving_manifest_ref = request.serving_manifest_ref
        representation_contract_hash = request.representation_contract_hash
        serving_build_digest = request.serving_build_digest
        if request.build_manifest_carrier_from_framework_context and (
            manifest_bytes is None
            or not serving_manifest_ref
            or not representation_contract_hash
            or not serving_build_digest
        ):
            if request.model_config is None:
                raise ServingIntegrationError(
                    "ServingIntegration.start(LocalSourceBootstrap) requires "
                    "model_config to build a local-ready manifest carrier"
                )
            if request.placement is None:
                raise ServingIntegrationError(
                    "ServingIntegration.start(LocalSourceBootstrap) requires "
                    "placement to build a local-ready manifest carrier"
                )
            if request.runtime_binding_schema_version is None:
                raise ServingIntegrationError(
                    "ServingIntegration.start(LocalSourceBootstrap) requires "
                    "runtime_binding_schema_version to build a local-ready "
                    "manifest carrier"
                )
            if request.serving_artifact_schema_version is None:
                raise ServingIntegrationError(
                    "ServingIntegration.start(LocalSourceBootstrap) requires "
                    "serving_artifact_schema_version to build a local-ready "
                    "manifest carrier"
                )
            carrier = self.prepare_local_ready_manifest_carrier_from_framework_context(
                recipe=request.recipe,
                manifest_tensor_name=str(request.manifest_tensor_name),
                model_config=request.model_config,
                placement=request.placement,
                runtime_binding_schema_version=int(
                    request.runtime_binding_schema_version
                ),
                serving_artifact_schema_version=int(
                    request.serving_artifact_schema_version
                ),
                framework_name=request.framework_name,
                framework_version=request.framework_version,
                adapter_version=request.adapter_version,
                serving_abi_version=request.serving_abi_version,
            )
            manifest_bytes = carrier.manifest_bytes
            serving_manifest_ref = carrier.serving_manifest_ref
            representation_contract_hash = carrier.representation_contract_hash
            serving_build_digest = carrier.serving_build_digest
        options = request.options
        if model is not None:
            self._assert_model_ready_for_runtime_binding(
                model,
                context="TensorCast local-ready binding realization",
            )
            self._align_runtime_tensor_names(
                model,
                self.local_ready_materialized_tensor_names(request.recipe),
            )
            canonical_tensors = self._collect_runtime_binding_tensors(
                model,
                remove_duplicate=False,
            )
            runtime_only_names = self.runtime_only_tensor_names(model)
            contract = self.build_local_ready_binding_contract(
                recipe=request.recipe,
                canonical_tensors=canonical_tensors,
                runtime_only_tensor_names=runtime_only_names,
                manifest_tensor_name=str(request.manifest_tensor_name),
                manifest_bytes=manifest_bytes,
                representation_contract_hash_factory=lambda tensor_schema_hash: "",
            )
            self._assert_local_ready_contract_realizable(
                contract,
                context="TensorCast local-ready binding realization",
            )
            if options is None:
                options = self._local_ready_materialization_options(request)
        realization = prepare_local_ready_serving(
            recipe=request.recipe,
            source_subject=request.source_subject,
            target_device=torch.device(request.target_device),
            manifest_tensor_name=str(request.manifest_tensor_name),
            manifest_bytes=manifest_bytes,
            options=options,
            binding_factory=request.binding_factory,
        )
        realized = LocalReadyServingResult(
            recipe=request.recipe,
            binding=realization.binding,
            update_epoch=realization.update_epoch,
            layout=realization.layout,
            realization_entry_count=realization.realization_entry_count,
            realization=realization,
        )
        if self._local_ready_prepare_has_finalize_fields(
            request,
            model=model,
            serving_manifest_ref=serving_manifest_ref,
            representation_contract_hash=representation_contract_hash,
            serving_build_digest=serving_build_digest,
        ):
            run_process_after_load = request.run_process_after_load
            run_semantic_validation = request.run_semantic_validation
            if (
                request.run_binding_finalize_hooks_when_required
                and self.local_ready_requires_binding_finalize(request.recipe)
            ):
                run_process_after_load = True
                run_semantic_validation = True
            finalized = self._finalize_local_ready_runtime(
                _LocalReadyFinalize(
                    model=model,
                    recipe=request.recipe,
                    binding=realization.binding,
                    update_epoch=realization.update_epoch,
                    source_artifact_ref=str(request.source_artifact_ref),
                    serving_manifest_ref=str(serving_manifest_ref),
                    representation_contract_hash=str(representation_contract_hash),
                    serving_build_digest=str(serving_build_digest),
                    manifest_tensor_name=str(request.manifest_tensor_name),
                    source_bound_contract_state=request.source_bound_contract_state,
                    source_bound_contract_path=str(request.source_bound_contract_path),
                    target_device=request.target_device,
                    manifest_bytes=manifest_bytes,
                    framework_config=request.framework_config,
                    model_config=request.model_config,
                    placement=request.placement,
                    family=request.family,
                    tp_rank=request.tp_rank,
                    tp_world_size=request.tp_world_size,
                    replace_meta_params=request.replace_meta_params,
                    run_process_after_load=run_process_after_load,
                    run_post_bind_finalize=request.run_post_bind_finalize,
                    run_semantic_validation=run_semantic_validation,
                    semantic_validation_spec=request.semantic_validation_spec,
                    validate_representation_contract_hash=request.validate_representation_contract_hash,
                    runtime_binding_schema_version=request.runtime_binding_schema_version,
                    serving_artifact_schema_version=request.serving_artifact_schema_version,
                    framework_name=request.framework_name,
                    framework_version=request.framework_version,
                    adapter_version=request.adapter_version,
                    serving_abi_version=request.serving_abi_version,
                )
            )
            return LocalReadyServingResult(
                model=finalized.model,
                runtime_state=finalized.runtime_state,
                runtime_view=finalized.runtime_view,
                prepared=finalized.prepared,
                binding_value=finalized.binding_value,
                recipe=request.recipe,
                current_value=finalized.current_value,
                binding=finalized.binding,
                update_epoch=finalized.update_epoch,
                layout=realized.layout,
                realization_entry_count=realized.realization_entry_count,
                realization=realized.realization,
                realization_report=finalized.realization_report,
            )
        return realized

    def _local_ready_prepare_with_built_recipe(
        self,
        request: _LocalReadyBootstrap,
    ) -> _LocalReadyBootstrap:
        source_subject_record = request.source_subject
        if source_subject_record is None:
            source_subject_record = self._resolve_local_ready_source_subject(request)
        source_artifact_ref = request.source_artifact_ref or getattr(
            source_subject_record, "artifact_ref", None
        )
        if not source_artifact_ref:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) could not "
                "derive source_artifact_ref from source subject"
            )
        try:
            source_artifact_ref = tc_source_catalog.resolve_source_artifact_ref(
                source_artifact_ref
            )
        except ValueError as exc:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires "
                "a real source artifact identity"
            ) from exc
        source_realization_subject = getattr(
            source_subject_record, "subject", source_subject_record
        )
        placement = request.placement
        if placement is None and self.host is not None:
            placement = self._framework_context(
                request.framework_config,
                request.model_config,
            ).placement
        source_catalog = self._local_ready_source_catalog(
            request,
            source_subject=source_subject_record,
            source_artifact_ref=str(source_artifact_ref),
        )
        cache_config = self._local_ready_recipe_cache_config(
            request,
            source_catalog=source_catalog,
        )
        recipe = self._build_local_ready_recipe_from_framework_context(
            request,
            source_subject=source_subject_record,
            source_artifact_ref=str(source_artifact_ref),
            source_catalog=source_catalog,
            cache_config=cache_config,
            placement=placement,
        )
        return replace(
            request,
            recipe=recipe,
            source_catalog=source_catalog,
            cache_config=cache_config,
            source_subject=source_realization_subject,
            source_artifact_ref=str(source_artifact_ref),
            placement=placement,
        )

    def _resolve_local_ready_source_subject(
        self,
        request: _LocalReadyBootstrap,
    ) -> SourceSubject:
        if request.source_selector is None:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires "
                "source_selector when source_subject is not supplied"
            )
        verify_checksums = bool(
            getattr(request.bootstrap, "verify_source_checksums", False)
        )
        return self.resolve_source_subject(
            request.source_selector,
            verify_checksums=verify_checksums,
            coordinator=request.source_subject_coordinator,
        )

    def _local_ready_source_catalog(
        self,
        request: _LocalReadyBootstrap,
        *,
        source_subject: Any,
        source_artifact_ref: str,
    ) -> Any:
        try:
            expected_source_ref = tc_source_catalog.resolve_source_artifact_ref(
                source_artifact_ref
            )
        except ValueError as exc:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires "
                "a real source artifact identity"
            ) from exc
        if request.source_catalog is not None:
            self._validate_source_catalog_artifact_ref(
                request.source_catalog,
                expected_source_artifact_ref=expected_source_ref,
            )
            return request.source_catalog
        if self.host is not None and self.host.source_catalog is not None:
            if not isinstance(request.source_selector, SourceSelector):
                raise ServingIntegrationError(
                    "IntegrationHost.source_catalog requires a core SourceSelector"
                )
            if request.model_config is None:
                raise ServingIntegrationError(
                    "IntegrationHost.source_catalog requires model_config"
                )
            source_catalog = self.host.source_catalog.build_catalog(
                SourceCatalogRequest(
                    source_subject=source_subject,
                    source_selector=request.source_selector,
                    source_artifact_ref=expected_source_ref,
                    framework_identity=self.host.framework.identity(
                        request.model_config
                    ),
                    framework_config=request.framework_config,
                    model_config=request.model_config,
                    download_policy=(
                        request.source_catalog_config
                        if isinstance(
                            request.source_catalog_config, SourceDownloadPolicy
                        )
                        else None
                    ),
                    cache_policy=(
                        request.cache_config
                        if isinstance(request.cache_config, RecipeCachePolicy)
                        else None
                    ),
                    source_catalog_config=request.source_catalog_config,
                )
            )
            self._validate_source_catalog_artifact_ref(
                source_catalog,
                expected_source_artifact_ref=expected_source_ref,
            )
            return source_catalog
        raise _capability_missing(
            "ServingIntegration.start(LocalSourceBootstrap) requires "
            "IntegrationHost.source_catalog when recipe is not supplied",
            level="level2-local-bootstrap",
            capability="source_catalog",
            operation="local_bootstrap.source_catalog",
            required_methods=("build_catalog",),
            next_action=(
                "Add IntegrationHost(source_catalog=...) or provide a prepared "
                "recipe through the admin/offline bootstrap path."
            ),
        )

    @staticmethod
    def _validate_source_catalog_artifact_ref(
        source_catalog: Any,
        *,
        expected_source_artifact_ref: str,
    ) -> None:
        catalog_artifact_ref = getattr(source_catalog, "source_artifact_ref", None)
        if catalog_artifact_ref is None:
            raise ServingIntegrationError(
                "SourceCatalogProvider returned a catalog without a real "
                "source_artifact_ref"
            )
        try:
            catalog_source_ref = tc_source_catalog.resolve_source_artifact_ref(
                str(catalog_artifact_ref)
            )
        except ValueError as exc:
            raise ServingIntegrationError(
                "SourceCatalogProvider returned a catalog without a real "
                "source_artifact_ref"
            ) from exc
        if catalog_source_ref != expected_source_artifact_ref:
            raise ServingIntegrationError(
                "SourceCatalogProvider returned source_artifact_ref "
                f"{catalog_source_ref!r}, expected {expected_source_artifact_ref!r}"
            )

    @staticmethod
    def _local_ready_recipe_cache_config(
        request: _LocalReadyBootstrap,
        *,
        source_catalog: Any,
    ) -> Any:
        cache_config_factory = request.cache_config_factory
        if callable(cache_config_factory):
            return cache_config_factory(source_catalog=source_catalog)
        if isinstance(request.cache_config, RecipeCachePolicy):
            return _recipe_build_cache_config_from_policy(
                request.cache_config,
                source_catalog=source_catalog,
            )
        if request.cache_config is not None:
            return request.cache_config
        return RecipeBuildCacheConfig()

    def _build_local_ready_recipe_from_framework_context(
        self,
        request: _LocalReadyBootstrap,
        *,
        source_subject: Any,
        source_artifact_ref: str,
        source_catalog: Any,
        cache_config: Any,
        placement: Any | None,
    ) -> Any:
        if request.model_config is None:
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires "
                "model_config when recipe is not supplied"
            )
        adapter = self._recipe_framework_adapter(request.model_config)
        recipe_session = self.build_recipe_session(
            RecipeBuildSessionRequest(
                source_subject=source_subject,
                framework_config=request.framework_config,
                model_config=request.model_config,
                placement=placement,
                cache_config=cache_config,
            )
        )
        result = recipe_session.build_recipe(
            model_config=request.model_config,
            framework_config=request.framework_config,
            source_catalog=source_catalog,
            framework_adapter=adapter,
            build_meta_model=lambda: self._build_meta_model(
                request.framework_config,
                request.model_config,
            ),
            cache_config=cache_config,
            is_reserved_serving_tensor_name=is_reserved_serving_tensor_name,
            semantic_validation_spec=request.semantic_validation_spec,
            placement=placement,
            debug_extra={
                "source_artifact_ref": source_artifact_ref,
            },
            profile_sink=self.profile_sink,
        )
        return result.recipe

    @staticmethod
    def _local_ready_prepare_has_finalize_fields(
        request: _LocalReadyBootstrap,
        *,
        model: Any | None = None,
        serving_manifest_ref: str | None = None,
        representation_contract_hash: str | None = None,
        serving_build_digest: str | None = None,
    ) -> bool:
        return all(
            (
                model is not None,
                request.source_artifact_ref,
                serving_manifest_ref,
                representation_contract_hash,
                serving_build_digest,
                request.source_bound_contract_state is not None,
                request.source_bound_contract_path,
            )
        )

    def _local_ready_materialization_options(
        self,
        request: _LocalReadyBootstrap,
    ) -> Any | None:
        execution_facts = self._request_execution_facts(request)
        if (
            request.configured_collective_policy is None
            or request.source_bound_contract_state is None
            or not request.source_bound_contract_path
            or execution_facts is None
        ):
            if request.require_materialization_options:
                raise ServingIntegrationError(
                    "ServingIntegration.start(LocalSourceBootstrap) requires "
                    "materialization execution context"
                )
            return None
        if request.require_materialization_options and not getattr(
            request.source_bound_contract_state,
            "source_bound_contract_ready",
            False,
        ):
            raise ServingIntegrationError(
                "ServingIntegration.start(LocalSourceBootstrap) requires "
                "ready source-bound contract state"
            )
        identity = self.local_ready_materialization_identity(request.recipe)
        options, _profile = self.build_materialization_options(
            artifact_ref=identity.source_artifact_ref,
            operation_scope=request.operation_scope,
            configured_policy=request.configured_collective_policy,
            source_bound_contract_state=request.source_bound_contract_state,
            source_bound_contract_path=request.source_bound_contract_path,
            execution_facts=execution_facts,
            contract_identity=(
                request.contract_identity or identity.source_metadata_fingerprint
            ),
        )
        return options

    def _request_execution_facts(self, request: Any) -> Mapping[str, Any] | None:
        execution_facts = getattr(request, "execution_facts", None)
        if execution_facts is not None:
            return execution_facts
        if self.host is None:
            return None
        return _execution_facts_payload(
            self.host.placement.execution_facts(
                getattr(request, "framework_config", None)
            )
        )

    @staticmethod
    def _assert_local_ready_contract_realizable(
        contract: LocalReadyBindingContract,
        *,
        context: str,
    ) -> None:
        if contract.realization_entry_count <= 0:
            raise ServingIntegrationError(
                f"{context} requires a non-empty BindingRealizationPlan"
            )
        if not contract.fallback_copy_plan:
            return
        unsupported = ", ".join(
            f"{getattr(entry, 'op', '')}:{getattr(entry, 'dst_name', '')}"
            for entry in contract.fallback_copy_plan[:8]
        )
        if len(contract.fallback_copy_plan) > 8:
            unsupported = f"{unsupported}, ..." if unsupported else "..."
        raise ServingIntegrationError(
            f"{context} requires a fully representable BindingRealizationPlan; "
            f"unsupported_entries={len(contract.fallback_copy_plan)} "
            f"[{unsupported}]"
        )

    def _finalize_local_ready_runtime(
        self, request: _LocalReadyFinalize
    ) -> LocalReadyServingResult:
        target_device = self._require_target_device(request.target_device)
        if request.recipe is None:
            raise ServingIntegrationError(
                "ServingIntegration._finalize_local_ready_runtime requires recipe"
            )
        if request.model is None:
            raise ServingIntegrationError(
                "ServingIntegration._finalize_local_ready_runtime requires model"
            )
        if request.binding is None:
            raise ServingIntegrationError(
                "ServingIntegration._finalize_local_ready_runtime requires binding"
            )
        if request.update_epoch is None:
            raise ServingIntegrationError(
                "ServingIntegration._finalize_local_ready_runtime requires update_epoch"
            )
        if not request.manifest_tensor_name:
            raise ServingIntegrationError(
                "ServingIntegration._finalize_local_ready_runtime requires manifest_tensor_name"
            )
        try:
            framework_context = self._framework_context(
                request.framework_config,
                request.model_config,
            )
            self._assert_local_ready_binding_tensor_set(
                recipe=request.recipe,
                binding=request.binding,
                manifest_tensor_name=str(request.manifest_tensor_name),
            )
            tensor_schema_hash = self.local_ready_tensor_schema_hash(
                recipe=request.recipe,
                manifest_tensor_name=str(request.manifest_tensor_name),
                manifest_bytes=request.manifest_bytes,
            )
            self._validate_local_ready_representation_contract_hash(
                request,
                tensor_schema_hash=tensor_schema_hash,
            )
            semantic_validation_spec = self._local_ready_semantic_validation_spec(
                request
            )
            self._assert_local_ready_finalize_admitted(
                request,
                semantic_validation_spec=semantic_validation_spec,
            )
            self._materializer().attach_and_finalize(
                model=request.model,
                tensors=_binding_tensors(request.binding),
                binding_handle=request.binding,
                context=framework_context,
                state_seed=RuntimeStateSeed(
                    artifact_ref=str(request.source_artifact_ref),
                    source_artifact_ref=str(request.source_artifact_ref),
                    representation_contract_hash=str(
                        request.representation_contract_hash
                    ),
                    tensor_schema_hash=tensor_schema_hash,
                    readiness="serving_local_ready",
                ),
                replace_meta_params=bool(request.replace_meta_params),
                target_device=target_device,
                model_config=request.model_config,
                run_process_after_load=bool(request.run_process_after_load),
                run_post_bind_finalize=bool(request.run_post_bind_finalize),
                semantic_validation_spec=semantic_validation_spec,
            )
            tensors = self._collect_runtime_binding_tensors(
                request.model,
                remove_duplicate=False,
            )
            self.validate_local_ready_tensor_schema(
                recipe=request.recipe,
                tensors=tensors,
            )
            current_value = self.freeze_local_ready(
                binding=request.binding,
                update_epoch=request.update_epoch,
                source_artifact_ref=str(request.source_artifact_ref),
            )
            artifact_report = _runtime_attachment_report_for_artifact_id(
                artifact_id=str(request.source_artifact_ref),
                tensors=_binding_tensors(request.binding),
                binding_handle=request.binding,
                target_device=target_device,
                tensor_schema_hash=tensor_schema_hash,
                artifact_profile="local_ready_source_artifact",
                authority_scope="daemon_mediated_local_ready_runtime_attachment",
            )
            prepared = build_local_ready_prepared_artifact(
                source_artifact_ref=str(request.source_artifact_ref),
                serving_manifest_ref=str(request.serving_manifest_ref),
                representation_contract_hash=str(request.representation_contract_hash),
                serving_build_digest=str(request.serving_build_digest),
                tensor_schema_hash=tensor_schema_hash,
                current_value=current_value,
                binding=request.binding,
                family=str(request.family),
                tp_rank=int(request.tp_rank),
                tp_world_size=int(request.tp_world_size),
                source_bound_contract_state=request.source_bound_contract_state,
                source_bound_contract_path=str(request.source_bound_contract_path),
                artifact_realization_report=artifact_report,
                model_runtime_spec=_model_runtime_spec_for_context(
                    context=framework_context,
                    target_device=target_device,
                ),
            )
            return LocalReadyServingResult(
                model=request.model,
                runtime_state=prepared.runtime_state,
                runtime_view=prepared.runtime_view,
                prepared=prepared.prepared,
                binding_value=prepared.binding_value,
                recipe=request.recipe,
                current_value=current_value,
                binding=request.binding,
                update_epoch=request.update_epoch,
                realization_report=artifact_report,
            )
        except Exception:
            _close_quietly(request.binding)
            raise

    def _assert_local_ready_finalize_admitted(
        self,
        request: _LocalReadyFinalize,
        *,
        semantic_validation_spec: Any | None,
    ) -> None:
        if not self.local_ready_requires_binding_finalize(request.recipe):
            return
        if not request.run_process_after_load:
            raise ServingIntegrationError(
                "TensorCast representation-changing local-ready finalize "
                "requires process_after_load execution"
            )
        if not request.run_semantic_validation:
            raise ServingIntegrationError(
                "TensorCast representation-changing local-ready finalize "
                "requires explicit semantic validation"
            )
        if (
            semantic_validation_spec is None
            or getattr(semantic_validation_spec, "kind", "none") == "none"
        ):
            raise ServingIntegrationError(
                "TensorCast representation-changing local-ready finalize "
                "requires an explicit semantic validation spec"
            )
        if not request.validate_representation_contract_hash:
            raise ServingIntegrationError(
                "TensorCast representation-changing local-ready finalize "
                "requires representation contract validation"
            )
        if (
            request.source_bound_contract_state is None
            or not request.source_bound_contract_path
        ):
            raise ServingIntegrationError(
                "TensorCast representation-changing local-ready finalize "
                "requires same-binding contract proof"
            )
        if not getattr(
            request.source_bound_contract_state,
            "source_bound_contract_ready",
            False,
        ):
            raise ServingIntegrationError(
                "TensorCast representation-changing local-ready finalize "
                "requires ready same-binding contract proof"
            )

    @staticmethod
    def _local_ready_semantic_validation_spec(
        request: _LocalReadyFinalize,
    ) -> Any | None:
        if request.semantic_validation_spec is not None:
            return request.semantic_validation_spec
        if not request.run_semantic_validation:
            return None
        return getattr(request.recipe, "semantic_validation_spec", None)

    def _validate_local_ready_representation_contract_hash(
        self,
        request: _LocalReadyFinalize,
        *,
        tensor_schema_hash: str,
    ) -> None:
        if not request.validate_representation_contract_hash:
            return
        if request.model_config is None:
            raise ServingIntegrationError(
                "ServingIntegration local-ready representation validation "
                "requires model_config"
            )
        if request.placement is None:
            raise ServingIntegrationError(
                "ServingIntegration local-ready representation validation "
                "requires placement"
            )
        if request.runtime_binding_schema_version is None:
            raise ServingIntegrationError(
                "ServingIntegration local-ready representation validation "
                "requires runtime_binding_schema_version"
            )
        if request.serving_artifact_schema_version is None:
            raise ServingIntegrationError(
                "ServingIntegration local-ready representation validation "
                "requires serving_artifact_schema_version"
            )
        actual = self.local_ready_representation_contract_hash(
            tensor_schema_hash=tensor_schema_hash,
            model_config=request.model_config,
            placement=request.placement,
            runtime_binding_schema_version=int(request.runtime_binding_schema_version),
            serving_artifact_schema_version=int(
                request.serving_artifact_schema_version
            ),
            framework_name=request.framework_name,
            framework_version=request.framework_version,
            adapter_version=request.adapter_version,
            serving_abi_version=request.serving_abi_version,
        )
        expected = str(request.representation_contract_hash)
        if actual == expected:
            return
        raise ManifestMismatchError(
            "TensorCast local-ready manifest contract hash drifted after "
            f"finalize: expected={expected}, actual={actual}"
        )

    def build_local_ready_manifest_carrier(
        self,
        *,
        recipe: Any,
        manifest_tensor_name: str,
        representation_contract_hash: str,
        logical_topology_json_payload: str | None = None,
        topology_admission_digest: str | None = None,
    ) -> tuple[str, bytes]:
        return prepare_same_binding_manifest_carrier(
            recipe,
            manifest_tensor_name=manifest_tensor_name,
            representation_contract_hash=representation_contract_hash,
            logical_topology_json_payload=logical_topology_json_payload,
            topology_admission_digest=topology_admission_digest,
        )

    def build_local_ready_manifest_carrier_from_contract(
        self,
        *,
        recipe: Any,
        manifest_tensor_name: str,
        representation_contract_hash_factory: Any,
        topology: Any | None = None,
        framework_payload: Mapping[str, Any] | None = None,
    ) -> tuple[str, bytes]:
        base_canonical_index = canonical_index_from_recipe(recipe)
        tensor_schema_hash = compute_serving_tensor_schema_hash(
            base_canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        )
        representation_contract_hash = representation_contract_hash_factory(
            tensor_schema_hash
        )
        logical_topology_json_payload = logical_topology_json_from_recipe(
            recipe,
            topology=topology,
            framework_payload=dict(framework_payload or {}),
        )
        topology_admission_digest = _optional_text(
            getattr(topology, "schema_topology_digest", None)
        )
        return self.build_local_ready_manifest_carrier(
            recipe=recipe,
            manifest_tensor_name=manifest_tensor_name,
            representation_contract_hash=representation_contract_hash,
            logical_topology_json_payload=logical_topology_json_payload,
            topology_admission_digest=topology_admission_digest,
        )

    def local_ready_representation_contract_hash(
        self,
        *,
        tensor_schema_hash: str,
        model_config: Any,
        placement: Any,
        runtime_binding_schema_version: int,
        serving_artifact_schema_version: int,
        framework_name: str | None = None,
        framework_version: str | None = None,
        adapter_version: str | None = None,
        serving_abi_version: str | None = None,
    ) -> str:
        compute_hash = getattr(model_config, "compute_hash", None)
        model_hash = (
            compute_hash()
            if callable(compute_hash)
            else getattr(model_config, "model", "unknown")
        )
        model_name = str(getattr(model_config, "model", "unknown"))
        placement_identity = getattr(placement, "identity_payload", None)
        if placement_identity is None:
            stable_identity_payload = getattr(
                placement, "stable_identity_payload", None
            )
            if callable(stable_identity_payload):
                placement_identity = stable_identity_payload()
            else:
                placement_identity = {}
        source_identity = {
            "model_hash": model_hash,
            "model_name": model_name,
            "runtime_binding_schema_version": int(runtime_binding_schema_version),
            "serving_artifact_schema_version": int(serving_artifact_schema_version),
            "placement": placement_identity,
        }
        return compute_runtime_representation_contract_hash(
            tensor_schema_hash=str(tensor_schema_hash or ""),
            topology_ref=getattr(placement, "topology", None),
            member_ref=getattr(placement, "member", None),
            framework_name=framework_name
            or self._framework_identity(model_config).framework_name,
            framework_version=framework_version
            or self._framework_identity(model_config).framework_version,
            adapter_version=adapter_version
            or self._framework_identity(model_config).adapter_version,
            serving_abi_version=serving_abi_version
            or self._framework_identity(model_config).serving_abi_version,
            source_identity=source_identity,
        )

    def build_local_ready_manifest_carrier_from_framework_context(
        self,
        *,
        recipe: Any,
        manifest_tensor_name: str,
        model_config: Any,
        placement: Any,
        runtime_binding_schema_version: int,
        serving_artifact_schema_version: int,
        framework_name: str | None = None,
        framework_version: str | None = None,
        adapter_version: str | None = None,
        serving_abi_version: str | None = None,
    ) -> tuple[str, bytes]:
        return self.build_local_ready_manifest_carrier_from_contract(
            recipe=recipe,
            manifest_tensor_name=manifest_tensor_name,
            representation_contract_hash_factory=lambda tensor_schema_hash: self.local_ready_representation_contract_hash(
                tensor_schema_hash=tensor_schema_hash,
                model_config=model_config,
                placement=placement,
                runtime_binding_schema_version=runtime_binding_schema_version,
                serving_artifact_schema_version=serving_artifact_schema_version,
                framework_name=framework_name,
                framework_version=framework_version,
                adapter_version=adapter_version,
                serving_abi_version=serving_abi_version,
            ),
            topology=getattr(placement, "topology", None),
            framework_payload=getattr(placement, "framework_payload", {}),
        )

    def prepare_local_ready_manifest_carrier_from_framework_context(
        self,
        *,
        recipe: Any,
        manifest_tensor_name: str,
        model_config: Any,
        placement: Any,
        runtime_binding_schema_version: int,
        serving_artifact_schema_version: int,
        framework_name: str | None = None,
        framework_version: str | None = None,
        adapter_version: str | None = None,
        serving_abi_version: str | None = None,
    ) -> LocalReadyManifestCarrierResult:
        representation_contract_hash, manifest_bytes = (
            self.build_local_ready_manifest_carrier_from_framework_context(
                recipe=recipe,
                manifest_tensor_name=manifest_tensor_name,
                model_config=model_config,
                placement=placement,
                runtime_binding_schema_version=runtime_binding_schema_version,
                serving_artifact_schema_version=serving_artifact_schema_version,
                framework_name=framework_name,
                framework_version=framework_version,
                adapter_version=adapter_version,
                serving_abi_version=serving_abi_version,
            )
        )
        manifest = ServingArtifactManifest.from_bytes(manifest_bytes)
        return LocalReadyManifestCarrierResult(
            representation_contract_hash=representation_contract_hash,
            manifest_bytes=manifest_bytes,
            serving_manifest_ref=manifest.serving_manifest_ref,
            serving_build_digest=manifest.serving_build_digest,
        )

    def local_ready_tensor_schema_hash(
        self,
        *,
        recipe: Any,
        manifest_tensor_name: str,
        manifest_bytes: bytes | None = None,
    ) -> str:
        return compute_serving_binding_tensor_schema_hash(
            recipe,
            manifest_tensor_name=manifest_tensor_name,
            manifest_bytes=manifest_bytes,
        )

    def local_ready_materialized_tensor_names(
        self,
        recipe: Any,
    ) -> tuple[str, ...]:
        return tuple(str(entry.name) for entry in materialized_tensor_schema(recipe))

    def _assert_local_ready_binding_tensor_set(
        self,
        *,
        recipe: Any,
        binding: Any,
        manifest_tensor_name: str,
    ) -> None:
        expected_names = tuple(
            sorted(self.local_ready_materialized_tensor_names(recipe))
        )
        actual_names = tuple(
            sorted(
                str(name)
                for name in _binding_tensors(binding)
                if str(name) != manifest_tensor_name
            )
        )
        if actual_names == expected_names:
            return
        raise SchemaMismatchError(
            "TensorCast local-ready binding tensor set does not match recipe "
            "schema: "
            f"expected={list(expected_names)}, actual={list(actual_names)}"
        )

    def build_local_ready_binding_contract(
        self,
        *,
        recipe: Any,
        canonical_tensors: Mapping[str, Any],
        runtime_only_tensor_names: Sequence[str],
        manifest_tensor_name: str,
        representation_contract_hash_factory: Any,
        manifest_bytes: bytes | None = None,
    ) -> LocalReadyBindingContract:
        realization_plan_proto = bytes(
            getattr(recipe, "realization_plan_proto", b"") or b""
        )
        realization_entry_count = compiled_recipe_realization_plan_count(recipe)
        if realization_entry_count <= 0:
            raise ServingIntegrationError(
                "TensorCast local-ready binding contract requires a compiled "
                "recipe with a pre-lowered BindingRealizationPlan"
            )
        if not realization_plan_proto:
            raise ServingIntegrationError(
                "TensorCast local-ready binding contract requires compiled "
                "recipe realization_plan_proto; regenerate the compiled recipe cache"
            )
        validate_tensor_schema_against_tensors(
            recipe.tensor_schema,
            canonical_tensors,
        )
        tensor_schema_hash = self.local_ready_tensor_schema_hash(
            recipe=recipe,
            manifest_tensor_name=manifest_tensor_name,
            manifest_bytes=manifest_bytes,
        )
        return LocalReadyBindingContract(
            excluded_names=tuple(
                sorted(str(name) for name in runtime_only_tensor_names)
            ),
            canonical_tensor_names=tuple(
                sorted(str(name) for name in canonical_tensors)
            ),
            tensor_schema_hash=tensor_schema_hash,
            representation_contract_hash=representation_contract_hash_factory(
                tensor_schema_hash
            ),
            mapped_copy_plan=(),
            realization_plan_proto=realization_plan_proto,
            realization_entry_count=realization_entry_count,
            fallback_copy_plan=tuple(recipe.realization_fallback_plan),
        )

    def local_ready_recipe_summary_fields(self, recipe: Any) -> dict[str, int]:
        return RecipeBuildSession.recipe_summary_fields(recipe)

    def local_ready_materialization_identity(
        self,
        recipe: Any,
    ) -> LocalReadyMaterializationIdentity:
        return LocalReadyMaterializationIdentity(
            source_artifact_ref=str(recipe.source_artifact_ref),
            source_metadata_fingerprint=str(recipe.source_metadata_fingerprint),
        )

    def local_ready_requires_binding_finalize(self, recipe: Any) -> bool:
        serving_facts = getattr(recipe, "serving_facts", None)
        process_after_load_class = tc_readiness.coerce_finalize_class(
            getattr(serving_facts, "process_after_load_class", None),
            default=FinalizeClass.RUNTIME_ONLY,
        )
        return process_after_load_class == FinalizeClass.REPRESENTATION_CHANGING

    def validate_local_ready_tensor_schema(
        self,
        *,
        recipe: Any,
        tensors: Mapping[str, Any],
    ) -> None:
        validate_tensor_schema_against_tensors(recipe.tensor_schema, tensors)

    def freeze_local_ready(
        self,
        *,
        binding: Any,
        update_epoch: Any,
        source_artifact_ref: str,
    ) -> Any:
        return freeze_local_ready_binding(
            binding=binding,
            update_epoch=update_epoch,
            source_artifact_ref=source_artifact_ref,
        )

    def build_materialization_options(
        self,
        *,
        artifact_ref: str,
        operation_scope: str,
        configured_policy: CollectivePolicy,
        source_bound_contract_state: SourceBoundContractState,
        source_bound_contract_path: str,
        execution_facts: Mapping[str, Any],
        contract_identity: str | None = None,
    ) -> tuple[Any, dict[str, object]]:
        facts = dict(execution_facts)
        return tc_binding_runtime.build_materialization_execution_context(
            artifact_ref=artifact_ref,
            operation_scope=operation_scope,
            configured_policy=configured_policy,
            tp_rank=int(facts.get("tp_rank", 0) or 0),
            tp_world_size=int(facts.get("tp_world_size", 1) or 1),
            same_node_tp=bool(facts.get("same_node_tp", False)),
            tp_ranks=tuple(int(rank) for rank in facts.get("tp_ranks", ()) or ()),
            collective_world_size=int(
                facts.get("collective_world_size", facts.get("tp_world_size", 1)) or 1
            ),
            collective_rank=int(
                facts.get("collective_rank", facts.get("tp_rank", 0)) or 0
            ),
            source_bound_contract_profile_fields=source_bound_contract_profile_fields(
                source_bound_contract_state,
                source_bound_contract_path,
            ),
            build_group_id=build_collective_group_id,
            contract_identity=contract_identity,
            collective_context_unavailable=bool(
                facts.get("collective_context_unavailable", False)
            ),
        )

    def build_recipe_session(
        self, request: RecipeBuildSessionRequest
    ) -> RecipeBuildSession:
        identity = request.identity
        if identity is None:
            identity = self._recipe_build_identity(request)
        return RecipeBuildSession(identity)

    def _recipe_build_identity(
        self,
        request: RecipeBuildSessionRequest,
    ) -> ServingBindingPlan:
        model_config = request.model_config
        if model_config is None:
            self._lifecycle_not_implemented("build_recipe_session", "P2")
        adapter = self._recipe_framework_adapter(model_config)
        placement = request.placement
        if placement is None and self.host is not None:
            placement = self._framework_context(
                request.framework_config,
                model_config,
            ).placement
        serving_placement = getattr(placement, "serving_placement", placement)
        member = getattr(serving_placement, "member", None)
        stable_identity_payload = getattr(
            serving_placement, "stable_identity_payload", None
        )
        if callable(stable_identity_payload):
            placement_payload = stable_identity_payload()
        else:
            placement_payload = getattr(placement, "identity_payload", None)
            if placement_payload is None:
                placement_payload = getattr(serving_placement, "identity_payload", None)
        trace_cache_schema_version = request.trace_cache_schema_version
        if trace_cache_schema_version is None:
            trace_cache_schema_version = getattr(
                request.cache_config,
                "trace_cache_schema_version",
                1,
            )
        tp_rank = request.tp_rank
        if tp_rank is None:
            tp_rank = getattr(placement, "tp_rank", None)
        if tp_rank is None and member is not None:
            tp_rank = getattr(member, "member_index", None)
        tp_world_size = request.tp_world_size
        if tp_world_size is None:
            tp_world_size = getattr(placement, "tp_world_size", None)
        if tp_world_size is None and member is not None:
            tp_world_size = getattr(member, "member_count", None)
        compute_hash = getattr(model_config, "compute_hash", None)
        model_id = str(getattr(model_config, "model", "unknown"))
        framework_version = self._adapter_text(adapter, "framework_version")
        return ServingBindingPlan(
            model_hash=str(
                compute_hash()
                if callable(compute_hash)
                else getattr(model_config, "model", "unknown")
            ),
            model_id=model_id,
            model_revision=getattr(model_config, "revision", None),
            dtype=str(getattr(model_config, "dtype", "unknown")),
            runtime_version=framework_version,
            framework_name=self._adapter_text(adapter, "framework_name"),
            framework_version=framework_version,
            adapter_version=self._adapter_text(adapter, "adapter_version"),
            serving_abi_version=self._adapter_text(
                adapter,
                "serving_abi_version",
                model_config,
            ),
            trace_cache_schema_version=int(trace_cache_schema_version),
            tp_rank=int(tp_rank or 0),
            tp_world_size=int(tp_world_size or 1),
            topology_ref=getattr(serving_placement, "topology", None),
            member_ref=member,
            placement=placement_payload,
        )

    @staticmethod
    def _adapter_text(
        adapter: Any | None,
        method_name: str,
        *args: Any,
    ) -> str:
        method = getattr(adapter, method_name, None)
        if callable(method):
            return str(method(*args))
        return ""

    def resolve_source_subject(
        self,
        path: str | SourceSelector,
        *,
        verify_checksums: bool,
        coordinator: Any | None = None,
    ) -> SourceSubject:
        if isinstance(path, SourceSelector):
            if path.kind != "local_path":
                raise SourceSubjectError(
                    f"Unsupported TensorCast source selector kind: {path.kind}"
                )
            path = str(path.value)
        if coordinator is not None:
            should_coordinate = getattr(coordinator, "should_coordinate", None)
            if not callable(should_coordinate) or bool(should_coordinate()):
                return self._resolve_source_subject_with_coordinator(
                    path,
                    verify_checksums=verify_checksums,
                    coordinator=coordinator,
                )
        return resolve_source_subject(path, verify_checksums=verify_checksums)

    def _resolve_source_subject_with_coordinator(
        self,
        path: str,
        *,
        verify_checksums: bool,
        coordinator: Any,
    ) -> SourceSubject:
        source_rank = int(getattr(coordinator, "source_rank", 0) or 0)
        is_source_rank = getattr(coordinator, "is_source_rank", None)
        resolve_locally = bool(is_source_rank()) if callable(is_source_rank) else True
        subject = (
            resolve_source_subject(path, verify_checksums=verify_checksums)
            if resolve_locally
            else None
        )
        payload = None if subject is None else source_subject_broadcast_payload(subject)
        broadcast = getattr(coordinator, "broadcast_object", None)
        if not callable(broadcast):
            raise SourceSubjectError(
                "TensorCast source subject coordinator must provide "
                "broadcast_object(payload, src)"
            )
        payload = broadcast(payload, src=source_rank)
        if payload is None:
            raise SourceSubjectError(
                "TensorCast source subject coordinator returned no payload"
            )
        if not isinstance(payload, Mapping):
            raise SourceSubjectError(
                "TensorCast source subject coordinator must broadcast a mapping payload"
            )
        return source_subject_from_broadcast_payload(payload)

    def source_subject_broadcast_payload(
        self, subject: SourceSubject
    ) -> dict[str, Any]:
        return source_subject_broadcast_payload(subject)

    def source_subject_from_broadcast_payload(
        self, payload: Mapping[str, Any]
    ) -> SourceSubject:
        return source_subject_from_broadcast_payload(payload)

    def _framework_host(self) -> FrameworkHost:
        if self.host is not None:
            return self.host.framework
        raise _capability_missing(
            "ServingIntegration requires IntegrationHost.framework",
            level="level1-runtime",
            capability="framework",
            operation="framework_host",
            required_methods=(
                "identity",
                "build_runtime_model",
                "assert_model_ready_for_runtime_binding",
            ),
            next_action=(
                "Construct ServingRuntimeSession with IntegrationHost.framework."
            ),
        )

    def _framework_identity(
        self,
        model_config: Any | None,
    ) -> FrameworkIdentity:
        return self._framework_host().identity(model_config)

    def _build_meta_model(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> object:
        return self._framework_host().build_meta_model(
            framework_config,
            model_config,
        )

    def _surface(self) -> TensorSurfaceHost:
        if self.host is not None:
            if self.host.tensor_surface is None:
                raise _capability_missing(
                    "IntegrationHost requires TensorSurfaceHost for runtime "
                    "tensor operations",
                    level="level1-runtime",
                    capability="tensor_surface",
                    operation="runtime_tensor_surface",
                    required_methods=(
                        "attach_bound_tensors",
                        "collect_runtime_tensors",
                        "compute_runtime_tensor_schema_hash",
                    ),
                    next_action=(
                        "Pass IntegrationHost(tensor_surface=...) or use "
                        "TorchTensorHost for PyTorch module carriers."
                    ),
                )
            return self.host.tensor_surface
        raise _capability_missing(
            "ServingIntegration requires IntegrationHost.tensor_surface",
            level="level1-runtime",
            capability="tensor_surface",
            operation="runtime_tensor_surface",
            required_methods=(
                "attach_bound_tensors",
                "collect_runtime_tensors",
                "compute_runtime_tensor_schema_hash",
            ),
            next_action=(
                "Construct ServingRuntimeSession with IntegrationHost.tensor_surface."
            ),
        )

    @staticmethod
    def _require_target_device(target_device: Any | None) -> torch.device:
        if target_device is None:
            raise ServingIntegrationError(
                "ServingIntegration request requires target_device"
            )
        return torch.device(target_device)

    @staticmethod
    def _runtime_policy(policy: Any | None) -> Any | None:
        to_runtime_policy = getattr(policy, "to_runtime_policy", None)
        if callable(to_runtime_policy):
            return to_runtime_policy()
        return policy

    @staticmethod
    def _runtime_policy_with_placement(
        policy: Any | None, placement: Any | None
    ) -> Any | None:
        digest = _optional_text(
            getattr(
                getattr(placement, "topology", None), "schema_topology_digest", None
            )
        )
        if digest is None:
            return policy
        if policy is None:
            return ServingRuntimePolicy(
                require_manifest=True,
                expected_topology_admission_digest=digest,
            )
        model_copy = getattr(policy, "model_copy", None)
        if callable(model_copy):
            return model_copy(
                update={
                    "require_manifest": True,
                    "expected_topology_admission_digest": digest,
                }
            )
        return policy

    @classmethod
    def _runtime_policy_from_manifest(
        cls, policy: Any | None, resolved: Any, placement: Any | None = None
    ) -> Any | None:
        if policy is not None:
            return cls._runtime_policy_with_placement(policy, placement)
        manifest = getattr(resolved, "manifest", None)
        to_runtime_policy = getattr(manifest, "to_runtime_policy", None)
        if callable(to_runtime_policy):
            return cls._runtime_policy_with_placement(to_runtime_policy(), placement)
        return cls._runtime_policy_with_placement(None, placement)

    @staticmethod
    def _json_object_payload(value: Any, *, field_name: str) -> Any:
        try:
            payload = json.loads(str(value))
        except Exception as exc:
            raise ManifestMismatchError(
                f"TensorCast serving artifact {field_name} is invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ManifestMismatchError(
                f"TensorCast serving artifact {field_name} must be a JSON object"
            )
        return payload

    @classmethod
    def _validate_resolved_artifact_placement(
        cls,
        resolved_artifact: Any,
        *,
        placement: Any | None,
    ) -> None:
        manifest = getattr(resolved_artifact, "manifest", None)
        if manifest is None:
            return
        manifest_topology_digest = _optional_text(
            getattr(manifest, "topology_admission_digest", None)
        )
        placement_topology_digest = _optional_text(
            getattr(
                getattr(placement, "topology", None), "schema_topology_digest", None
            )
        )
        if manifest_topology_digest is not None:
            if placement_topology_digest is None:
                raise ManifestMismatchError(
                    "TensorCast serving artifact topology admission digest "
                    "requires current framework placement"
                )
            if manifest_topology_digest != placement_topology_digest:
                raise ManifestMismatchError(
                    "TensorCast serving artifact topology admission digest "
                    "mismatch: "
                    f"manifest={manifest_topology_digest}, "
                    f"current={placement_topology_digest}"
                )

        manifest_logical_topology = _optional_text(
            getattr(manifest, "logical_topology_json", None)
        )
        if manifest_logical_topology is None:
            return
        if placement is None:
            raise ManifestMismatchError(
                "TensorCast serving artifact logical topology requires current "
                "framework placement"
            )
        try:
            current_logical_topology = tc_contract.logical_topology_json(
                placement.topology,
                framework_payload=dict(getattr(placement, "framework_payload", {})),
            )
        except Exception as exc:
            raise ManifestMismatchError(
                "TensorCast serving artifact logical topology could not be "
                "computed from current framework placement"
            ) from exc
        if cls._json_object_payload(
            manifest_logical_topology, field_name="logical_topology_json"
        ) != cls._json_object_payload(
            current_logical_topology, field_name="current logical topology"
        ):
            raise ManifestMismatchError(
                "TensorCast serving artifact logical topology mismatch"
            )

    def _prepare_model_construction(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> None:
        host = self._framework_host()
        prepare = getattr(host, "prepare_model_construction", None)
        if callable(prepare):
            prepare(framework_config, model_config)

    def _assert_model_ready_for_runtime_binding(
        self,
        model: Any,
        *,
        context: str,
    ) -> None:
        host = self._framework_host()
        check = getattr(host, "assert_model_ready_for_runtime_binding", None)
        if callable(check):
            check(model, context=context)

    def _align_runtime_tensor_names(
        self,
        model: Any,
        expected_names: Sequence[str],
    ) -> int:
        return int(
            self._surface().align_runtime_tensor_names(model, expected_names) or 0
        )

    def _collect_runtime_binding_tensors(
        self,
        model: Any,
        *,
        remove_duplicate: bool,
    ) -> Mapping[str, Any]:
        return self._surface().collect_runtime_tensors(
            model,
            remove_duplicate=remove_duplicate,
        )

    def _compute_runtime_tensor_schema_hash(
        self,
        tensors: Mapping[str, Any],
        *,
        remove_duplicate: bool,
    ) -> str:
        return self._surface().compute_runtime_tensor_schema_hash(
            tensors,
            remove_duplicate=remove_duplicate,
        )

    def runtime_only_tensor_names(self, model: object) -> tuple[str, ...]:
        return self._surface().runtime_only_tensor_names(model)

    def support_level(
        self,
        model: object,
        model_config: object,
    ) -> ServingSupportLevel:
        host = self._framework_host()
        support_level = getattr(host, "support_level", None)
        if callable(support_level):
            return tc_readiness.coerce_serving_support_level(
                support_level(model, model_config),
                default=ServingSupportLevel.BLOCKED,
            )
        return ServingSupportLevel.BLOCKED

    def process_after_load_class(
        self,
        model: object,
        model_config: object,
    ) -> FinalizeClass:
        host = self._framework_host()
        process_after_load = getattr(host, "process_after_load_class", None)
        if callable(process_after_load):
            return tc_readiness.coerce_finalize_class(
                process_after_load(model, model_config),
                default=FinalizeClass.UNKNOWN_BLOCKED,
            )
        finalize_policy = getattr(host, "finalize_policy", None)
        if callable(finalize_policy):
            finalize_policy(model, model_config)
            return FinalizeClass.RUNTIME_ONLY
        return FinalizeClass.UNKNOWN_BLOCKED

    def post_bind_finalize_class(
        self,
        model: object,
        model_config: object,
    ) -> FinalizeClass:
        host = self._framework_host()
        post_bind_finalize = getattr(host, "post_bind_finalize_class", None)
        if callable(post_bind_finalize):
            return tc_readiness.coerce_finalize_class(
                post_bind_finalize(model, model_config),
                default=FinalizeClass.RUNTIME_ONLY,
            )
        finalize_policy = getattr(host, "finalize_policy", None)
        if callable(finalize_policy):
            finalize_policy(model, model_config)
            return FinalizeClass.RUNTIME_ONLY
        return FinalizeClass.RUNTIME_ONLY

    def trace_model_load(
        self,
        model: object,
        ordered_names: Sequence[str],
        meta_by_name: Mapping[str, object],
        *,
        debug_dump_trace: bool = False,
    ) -> TracePlan:
        host = self._framework_host()
        trace = getattr(host, "trace_model_load", None)
        if not callable(trace):
            raise _capability_missing(
                "ServingIntegration host requires RecipeTraceHost."
                "trace_model_load on recipe cache miss",
                level="level2-local-bootstrap",
                capability="recipe_trace",
                operation="local_bootstrap.trace_model_load",
                required_methods=("trace_model_load",),
                next_action=(
                    "Implement RecipeTraceHost.trace_model_load or precompute "
                    "a recipe through the admin/offline builder path."
                ),
            )
        return cast(
            TracePlan,
            trace(
                model,
                ordered_names,
                meta_by_name,
                debug_dump_trace=debug_dump_trace,
            ),
        )

    def cleanup_after_recipe_build(
        self,
        model: object,
        model_config: object,
        *,
        framework_config: object | None = None,
    ) -> None:
        host = self._framework_host()
        cleanup = getattr(host, "cleanup_after_recipe_build", None)
        if callable(cleanup):
            cleanup(
                model,
                model_config,
                framework_config=framework_config,
            )

    def semantic_probes(self, model: object, model_config: object) -> object:
        host = self._framework_host()
        semantic_probes = getattr(host, "semantic_probes", None)
        if callable(semantic_probes):
            return semantic_probes(model, model_config)
        return None

    def native_load_weights(self, model: object, weights: object) -> None:
        host = self._framework_host()
        native_load = getattr(host, "native_load_weights", None)
        if not callable(native_load):
            raise _capability_missing(
                "ServingIntegration host requires NativeLoadHost for native "
                "checkpoint/source loading",
                level="level2-local-bootstrap",
                capability="native_load",
                operation="local_bootstrap.native_load_weights",
                required_methods=("native_load_weights",),
                next_action=(
                    "Implement NativeLoadHost.native_load_weights for source "
                    "bootstrap cache misses."
                ),
            )
        native_load(model, weights)

    def _recipe_framework_adapter(self, model_config: Any | None) -> Any:
        identity = self._framework_identity(model_config)
        return SimpleNamespace(
            framework_name=lambda: str(identity.framework_name),
            framework_version=lambda: str(identity.framework_version),
            adapter_version=lambda: str(identity.adapter_version),
            serving_abi_version=lambda _model_config=None: str(
                identity.serving_abi_version
            ),
            support_level=self.support_level,
            runtime_only_tensor_names=self.runtime_only_tensor_names,
            process_after_load_class=self.process_after_load_class,
            post_bind_finalize_class=self.post_bind_finalize_class,
            trace_model_load=self.trace_model_load,
            cleanup_after_recipe_build=self.cleanup_after_recipe_build,
            semantic_probes=self.semantic_probes,
            native_load_weights=self.native_load_weights,
        )

    @staticmethod
    def _assert_tensor_names_match_expected(
        tensors: Mapping[str, Any],
        expected_names: Sequence[str],
    ) -> None:
        expected = {str(name) for name in expected_names}
        if not expected:
            return
        actual = {str(name) for name in tensors}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if not missing and not unexpected:
            return
        raise SchemaMismatchError(
            "TensorCast runtime tensor set does not match serving artifact: "
            f"missing_count={len(missing)}, unexpected_count={len(unexpected)}"
        )

    def _load_materialization_options(
        self,
        request: _DirectServingLoad,
        resolved: Any,
    ) -> Any | None:
        if request.materialization is not None:
            return request.materialization
        execution_facts = self._request_execution_facts(request)
        if (
            request.configured_collective_policy is None
            or request.source_bound_contract_state is None
            or not request.source_bound_contract_path
            or execution_facts is None
        ):
            if request.require_materialization_options:
                raise ServingIntegrationError(
                    "ServingIntegration._load_existing_serving_artifact requires "
                    "materialization execution context for direct bind"
                )
            return None
        if request.require_materialization_options and not getattr(
            request.source_bound_contract_state,
            "source_bound_contract_ready",
            False,
        ):
            raise ServingIntegrationError(
                "ServingIntegration._load_existing_serving_artifact requires ready "
                "source-bound contract state for direct bind"
            )
        manifest = getattr(resolved, "manifest", None)
        options, _profile = self.build_materialization_options(
            artifact_ref=str(getattr(resolved, "artifact_ref", "") or ""),
            operation_scope=request.operation_scope,
            configured_policy=request.configured_collective_policy,
            source_bound_contract_state=request.source_bound_contract_state,
            source_bound_contract_path=request.source_bound_contract_path,
            execution_facts=execution_facts,
            contract_identity=getattr(manifest, "representation_contract_hash", None),
        )
        return options

    def _reload_materialization_options(
        self,
        request: _ServingReload,
        resolved: Any,
    ) -> Any | None:
        if request.materialization is not None:
            return request.materialization
        execution_facts = self._request_execution_facts(request)
        if (
            request.configured_collective_policy is None
            or request.source_bound_contract_state is None
            or not request.source_bound_contract_path
            or execution_facts is None
        ):
            if request.require_materialization_options:
                raise ServingIntegrationError(
                    "ServingIntegration._reload_existing_serving_artifact requires "
                    "materialization execution context for swap"
                )
            return None
        if request.require_materialization_options and not getattr(
            request.source_bound_contract_state,
            "source_bound_contract_ready",
            False,
        ):
            raise ServingIntegrationError(
                "ServingIntegration._reload_existing_serving_artifact requires ready "
                "source-bound contract state for swap"
            )
        manifest = getattr(resolved, "manifest", None)
        options, _profile = self.build_materialization_options(
            artifact_ref=str(getattr(resolved, "artifact_ref", "") or ""),
            operation_scope=request.operation_scope,
            configured_policy=request.configured_collective_policy,
            source_bound_contract_state=request.source_bound_contract_state,
            source_bound_contract_path=request.source_bound_contract_path,
            execution_facts=execution_facts,
            contract_identity=(
                request.contract_identity
                or getattr(
                    manifest,
                    "representation_contract_hash",
                    None,
                )
            ),
        )
        return options

    def _resolved_artifact(
        self,
        *,
        resolved_artifact: ResolvedServingArtifact | None,
        artifact_ref: str | None,
        artifact_locator: Any | None,
        expected_tensor_schema_hash: str | None,
        serving_runtime_policy: Any | None,
        placement: ServingPlacement | None = None,
    ) -> ResolvedServingArtifact:
        if resolved_artifact is not None:
            if artifact_ref is not None and str(resolved_artifact.artifact_ref) != str(
                artifact_ref
            ):
                raise ManifestMismatchError(
                    "TensorCast resolved serving artifact ref mismatch: "
                    f"resolved={resolved_artifact.artifact_ref}, "
                    f"requested={artifact_ref}"
                )
            self._validate_resolved_artifact_placement(
                resolved_artifact,
                placement=placement,
            )
            if self.resolver is not None and expected_tensor_schema_hash:
                return cross_check_serving_artifact(
                    resolved_artifact,
                    resolver=self.resolver,
                    expected_tensor_schema_hash=expected_tensor_schema_hash,
                    serving_runtime_policy=serving_runtime_policy,
                )
            return resolved_artifact
        resolved_ref = artifact_ref
        if resolved_ref is None and artifact_locator is not None:
            resolve_artifact_ref = getattr(
                artifact_locator, "resolve_artifact_ref", None
            )
            if callable(resolve_artifact_ref):
                if _artifact_locator_kind(artifact_locator) == "ranked_version_key":
                    resolved_ref = resolve_artifact_ref(placement=placement)
                else:
                    resolved_ref = resolve_artifact_ref()
            else:
                resolved_ref = str(artifact_locator)
        if not resolved_ref:
            raise ServingIntegrationError(
                "ServingIntegration request requires resolved_artifact, "
                "artifact_ref, or artifact_locator"
            )
        resolved = resolve_serving_artifact(
            str(resolved_ref),
            resolver=self.resolver,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            serving_runtime_policy=serving_runtime_policy,
        )
        self._validate_resolved_artifact_placement(
            resolved,
            placement=placement,
        )
        return resolved

    def _preflight_serving_artifact(
        self,
        *,
        resolved_artifact: ResolvedServingArtifact | None,
        artifact_ref: str | None,
        artifact_locator: Any | None,
        expected_tensor_schema_hash: str | None,
        policy: Any | None,
        placement: ServingPlacement | None = None,
    ) -> _ServingArtifactPreflight:
        base_policy = self._runtime_policy(policy)
        resolved = self._resolved_artifact(
            resolved_artifact=resolved_artifact,
            artifact_ref=artifact_ref,
            artifact_locator=artifact_locator,
            expected_tensor_schema_hash=None,
            serving_runtime_policy=None,
            placement=placement,
        )
        serving_runtime_policy = self._runtime_policy_from_manifest(
            base_policy,
            resolved,
            placement=placement,
        )
        if expected_tensor_schema_hash is not None:
            resolved = self._resolved_artifact(
                resolved_artifact=resolved,
                artifact_ref=artifact_ref,
                artifact_locator=artifact_locator,
                expected_tensor_schema_hash=expected_tensor_schema_hash,
                serving_runtime_policy=serving_runtime_policy,
                placement=placement,
            )
        return _ServingArtifactPreflight(
            resolved_artifact=resolved,
            serving_runtime_policy=serving_runtime_policy,
        )

    def _framework_context(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> FrameworkIntegrationContext:
        identity = self._framework_identity(model_config)
        placement = None
        if self.host is not None:
            try:
                placement = self._host_serving_placement(framework_config)
            except Exception:
                placement = None
        return FrameworkIntegrationContext(
            framework_name=str(identity.framework_name),
            framework_version=str(identity.framework_version),
            adapter_version=str(identity.adapter_version),
            serving_abi_version=str(identity.serving_abi_version),
            placement=placement,
        )

    def _materializer(self) -> RuntimeBindingMaterialization:
        if self.host is None:
            raise _capability_missing(
                "ServingIntegration runtime materialization requires IntegrationHost",
                level="level1-runtime",
                capability="integration_host",
                operation="runtime_materialization",
                required_methods=("framework", "placement", "tensor_surface"),
                next_action=(
                    "Construct ServingRuntimeSession with an IntegrationHost "
                    "instead of calling lifecycle helpers without host facts."
                ),
            )
        return RuntimeBindingMaterialization(
            host=self.host,
            profile_sink=self.profile_sink,
        )

    @staticmethod
    def _state_seed(
        resolved: ResolvedServingArtifact,
        *,
        tensor_schema_hash: str,
        execution_diagnostics: Any | None,
        materialization_diagnostics: Any | None = None,
        binding_handle: Any | None = None,
        artifact_realization_report: ArtifactRealizationReport | None = None,
        readiness: str = "serving",
    ) -> RuntimeStateSeed:
        artifact_ref = str(getattr(resolved, "artifact_ref", "") or "")
        manifest = getattr(resolved, "manifest", None)
        representation_contract_hash = str(
            getattr(manifest, "representation_contract_hash", "") or ""
        )
        source_artifact_ref = getattr(manifest, "source_artifact_ref", None)
        serving_build_digest = getattr(manifest, "serving_build_digest", None)
        diagnostics = {}
        if materialization_diagnostics is not None:
            diagnostics["materialization"] = materialization_diagnostics
        if execution_diagnostics is not None:
            diagnostics["execution"] = execution_diagnostics
        if serving_build_digest:
            diagnostics["serving_build_digest"] = str(serving_build_digest)
        if artifact_realization_report is not None:
            diagnostics["artifact_realization_report"] = (
                artifact_realization_report_to_dict(artifact_realization_report)
            )
        binding_value_ref = getattr(binding_handle, "current_value", None)
        if binding_value_ref is None:
            binding_value_ref = getattr(binding_handle, "binding_value_ref", None)
        return RuntimeStateSeed(
            artifact_ref=artifact_ref,
            serving_artifact_ref=artifact_ref or None,
            source_artifact_ref=(
                str(source_artifact_ref) if source_artifact_ref else None
            ),
            representation_contract_hash=representation_contract_hash,
            tensor_schema_hash=str(tensor_schema_hash or ""),
            binding_value_ref=binding_value_ref,
            local_serving_ref=getattr(manifest, "local_serving_ref", None),
            readiness=readiness,
            diagnostics=diagnostics or None,
            realization_report=artifact_realization_report,
        )


def resolve_serving_artifact(
    artifact_ref: str,
    *,
    resolver: ServingArtifactResolver | None = None,
    manifest_tensor_name: str | None = None,
    schema_version: int | None = None,
    expected_tensor_schema_hash: str | None = None,
    serving_runtime_policy: Any | None = None,
) -> ResolvedServingArtifact:
    """Resolve a serving artifact and optionally cross-check runtime schema."""

    resolved_resolver = resolver or ServingArtifactResolver(
        manifest_tensor_name=manifest_tensor_name or tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=(
            schema_version
            if schema_version is not None
            else int(tc.ServingArtifactManifest.model_fields["schema_version"].default)
        ),
    )
    resolved = resolved_resolver.resolve(str(artifact_ref))
    if expected_tensor_schema_hash is not None:
        resolved_resolver.cross_check(
            resolved,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            serving_runtime_policy=serving_runtime_policy,
        )
    return resolved


def read_serving_artifact_manifest(
    artifact: Any,
    *,
    artifact_ref: str,
    resolver: ServingArtifactResolver,
) -> ResolvedServingArtifact:
    """Read a serving manifest from an already opened artifact handle."""

    return resolver.read_manifest(artifact, artifact_ref=str(artifact_ref))


def cross_check_serving_artifact(
    resolved_artifact: ResolvedServingArtifact,
    *,
    resolver: ServingArtifactResolver,
    expected_tensor_schema_hash: str,
    serving_runtime_policy: Any | None = None,
) -> ResolvedServingArtifact:
    """Validate manifest, descriptor schema, and runtime policy agreement."""

    return resolver.cross_check(
        resolved_artifact,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        serving_runtime_policy=serving_runtime_policy,
    )


@dataclass(frozen=True)
class ServingRuntimeSession:
    """Config-planned serving runtime lifecycle entrypoint."""

    serving_config: ServingConfig
    host: IntegrationHost
    integration: ServingIntegration
    profile_sink: Any | None = None

    @classmethod
    def from_config(
        cls,
        serving_config: ServingConfig | Mapping[str, Any],
        *,
        host: IntegrationHost,
        resolver: ServingArtifactResolver | None = None,
        profile_sink: Any | None = None,
    ) -> "ServingRuntimeSession":
        config = (
            serving_config
            if isinstance(serving_config, ServingConfig)
            else ServingConfig.from_mapping(serving_config)
        )
        return cls(
            serving_config=config,
            host=host,
            integration=ServingIntegration(
                resolver=resolver,
                profile_sink=profile_sink,
                host=host,
            ),
            profile_sink=profile_sink,
        )

    def start(self, context: RequestContext) -> RuntimeAttachment:
        intent = self._plan_start_intent(context)
        return self._start_intent(intent, context)

    def publish_current_replica(
        self,
        *,
        current_attachment: RuntimeAttachment,
        context: RequestContext | None = None,
        policy: ReplicaPublicationPolicy | Mapping[str, Any] | None = None,
    ) -> RuntimeAttachment:
        """Publish the current artifact-backed runtime attachment as a replica."""

        del context
        return tc_replica_publication.publish_current_replica(
            current_attachment=current_attachment,
            policy=self._replica_publication_policy(policy),
            ensure_runtime_initialized=self._ensure_runtime_initialized,
            profile_sink=self.profile_sink,
        )

    def project_current_replica_publication_state(
        self,
        *,
        current_attachment: RuntimeAttachment,
        state: str,
        reason: str | None = None,
        operation_id: str | None = None,
    ) -> RuntimeAttachment:
        """Return an attachment with a non-authoritative publication projection.

        This helper is intentionally limited to observational states.  The
        authoritative ``published`` and ``retired`` states still come only from
        publish/retire lifecycle operations that touch the daemon-owned binding.
        """

        return tc_replica_publication.project_current_replica_publication_state(
            current_attachment=current_attachment,
            state=state,
            reason=reason,
            operation_id=operation_id,
        )

    def retire_current_replica(
        self,
        *,
        current_attachment: RuntimeAttachment,
        reason: str = "retire",
        drain_timeout_s: float | None = None,
        context: RequestContext | None = None,
    ) -> RuntimeAttachment:
        """Retire the published replica tied to a runtime attachment."""

        del context
        return tc_replica_publication.retire_current_replica(
            current_attachment=current_attachment,
            reason=reason,
            drain_timeout_s=drain_timeout_s,
            default_drain_timeout_s=(
                self.serving_config.replica_publication.drain_timeout_s
            ),
            ensure_runtime_initialized=self._ensure_runtime_initialized,
            profile_sink=self.profile_sink,
        )

    def _start_intent(
        self,
        intent: ServingIntent,
        context: RequestContext,
    ) -> RuntimeAttachment:
        """Private/admin entrypoint for already lowered serving intents."""

        self._ensure_runtime_initialized()
        return self.integration.start(intent, context)

    def reload(
        self,
        *,
        current_attachment: RuntimeAttachment | RuntimeBindingState | Any,
        artifact_locator: ServingArtifactLocator,
        policy: ServingPolicy | None,
        context: RequestContext,
        model: object | None = None,
        contract_identity: str | None = None,
    ) -> RuntimeAttachment:
        self._reject_local_reload_artifact_locator(artifact_locator)
        if not isinstance(artifact_locator, ServingArtifactLocator):
            raise ConfigConflictError(
                "TensorCast serving reload requires a ServingArtifactLocator"
            )
        if policy is not None and not isinstance(policy, ServingPolicy):
            raise ConfigConflictError(
                "TensorCast serving reload requires a ServingPolicy or None"
            )
        if isinstance(current_attachment, RuntimeAttachment):
            self._reject_reload_with_active_publication(current_attachment)
        self._ensure_runtime_initialized()
        current_state = (
            current_attachment.state
            if isinstance(current_attachment, RuntimeAttachment)
            else current_attachment
        )
        runtime_model = (
            model if model is not None else getattr(current_attachment, "model", None)
        )
        return self.integration.reload(
            current_state,
            ExistingServingArtifact(artifact_locator=artifact_locator, policy=policy),
            context,
            model=runtime_model,
            contract_identity=contract_identity,
        )

    def describe(
        self,
        attachment_or_state: RuntimeAttachment | RuntimeBindingState | Any,
    ) -> RuntimeWorkerView:
        if isinstance(attachment_or_state, RuntimeAttachment):
            return attachment_or_state.view
        return self.integration.describe(attachment_or_state)

    def _ensure_runtime_initialized(self) -> None:
        self.serving_config.runtime.ensure_initialized()

    def _replica_publication_policy(
        self,
        policy: ReplicaPublicationPolicy | Mapping[str, Any] | None,
    ) -> ReplicaPublicationPolicy:
        if policy is None:
            return self.serving_config.replica_publication
        if isinstance(policy, ReplicaPublicationPolicy):
            return policy
        return ReplicaPublicationPolicy.model_validate(dict(policy))

    @staticmethod
    def _reject_reload_with_active_publication(
        current_attachment: RuntimeAttachment,
    ) -> None:
        tc_replica_publication.reject_reload_with_active_publication(current_attachment)

    def _plan_start_intent(self, context: RequestContext) -> ServingIntent:
        source_selector = self._source_selector_from_context(context)
        expected_member = None
        if (
            self.serving_config.retained_binding_acquire.mode == "external"
            and self.host is not None
        ):
            placement = self.integration._framework_context(
                context.framework_config,
                context.model_config,
            ).placement
            if placement is not None:
                expected_member = placement.member
        try:
            plan = tc_config.plan_serving_start(
                config=self.serving_config,
                source_selector=source_selector,
                expected_member=expected_member,
            )
        except tc_config.ServingStartPlanError as exc:
            raise ConfigConflictError(str(exc)) from exc

        if isinstance(plan, tc_config.RetainedBindingAcquireStartPlan):
            return RetainedBindingAcquire(plan.authority)
        if isinstance(plan, tc_config.ArtifactBindStartPlan):
            return ExistingServingArtifact(
                artifact_locator=plan.artifact_locator,
                policy=plan.policy,
            )
        if isinstance(plan, tc_config.SourceBootstrapToBindingStartPlan):
            return LocalSourceBootstrap(
                source_selector=plan.source_selector,
                bootstrap_policy=plan.bootstrap_policy,
            )
        raise ConfigConflictError(
            f"TensorCast serving planner returned unsupported plan: {plan!r}"
        )

    def _source_selector_from_context(
        self, context: RequestContext
    ) -> SourceSelector | None:
        if self.host.source is None:
            return None
        source_selector = getattr(self.host.source, "source_selector", None)
        if not callable(source_selector):
            return None
        selector = source_selector(context.framework_config, context.model_config)
        if selector is None:
            return None
        if not isinstance(selector, SourceSelector):
            raise ConfigConflictError(
                "IntegrationHost.source.source_selector must return "
                "SourceSelector or None"
            )
        return selector

    @staticmethod
    def _reject_local_reload_artifact_locator(artifact_locator: object) -> None:
        if (
            isinstance(artifact_locator, SourceSelector)
            or _artifact_locator_kind(artifact_locator) == "local_path"
        ):
            raise ConfigConflictError(
                "TensorCast serving reload requires a durable serving "
                "artifact locator, not a local source selector"
            )


def bind_serving_artifact(
    *,
    resolved_artifact: ResolvedServingArtifact,
    tensor_names: Sequence[str],
    device: Any,
    serving_runtime_policy: Any | None,
    options: Any | None,
) -> RuntimeBindingResult:
    """Bind a durable serving artifact and return an attach-ready result."""

    binding = tc_binding_runtime.bind_serving_artifact(
        resolved_artifact=resolved_artifact,
        tensor_names=tuple(tensor_names),
        device=device,
        serving_runtime_policy=serving_runtime_policy,
        options=options,
    )
    return RuntimeBindingResult.from_binding(binding)


def swap_serving_artifact(
    *,
    binding: Any,
    resolved_artifact: ResolvedServingArtifact,
    tensor_names: Sequence[str] | None = None,
    serving_runtime_policy: Any | None,
    options: Any | None,
) -> RuntimeBindingResult:
    """Swap an existing runtime binding to another serving artifact."""

    operation_result = tc_binding_runtime.swap_serving_artifact(
        binding=binding,
        resolved_artifact=resolved_artifact,
        tensor_names=tensor_names,
        serving_runtime_policy=serving_runtime_policy,
        options=options,
    )
    result_binding = operation_result if operation_result is not None else binding
    if not hasattr(result_binding, "tensors"):
        result_binding = binding
    return RuntimeBindingResult.from_binding(
        result_binding,
        operation_result=operation_result,
    )


@contextmanager
def restore_retained_binding(
    *,
    authority: tc_retained_binding.ParsedRetainedServingBindingAuthority | None = None,
    local_serving_ref: str | None = None,
    target_device: torch.device | str,
    expected_member: tc.ServingBindingMemberRef | None = None,
    expected_tensor_schema_hash: str | None = None,
    expected_serving_build_digest: str | None = None,
    expected_target_layout_hash: str | None = None,
    expected_daemon_id: str | None = None,
    expected_daemon_session_id: str | None = None,
    serving_artifact_id: str | None = None,
    caller_pid: int | None = None,
    runtime: Any | None = None,
    client: Any | None = None,
    restore_fn: Any | None = None,
    timeout_s: float | None = None,
) -> Iterator[RestoredRetainedBinding]:
    """Acquire and restore a retained binding value for framework attach.

    If the framework does not call ``transfer_to_runtime()``, the restored owner
    is released automatically when the context exits. After transfer, close
    ownership belongs to the returned runtime handle.
    """

    with tc_retained_binding.acquire_retained_serving_binding(
        authority=authority,
        local_serving_ref=local_serving_ref,
        target_device=target_device,
        expected_member=expected_member,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        expected_serving_build_digest=expected_serving_build_digest,
        expected_target_layout_hash=expected_target_layout_hash,
        expected_daemon_id=expected_daemon_id,
        expected_daemon_session_id=expected_daemon_session_id,
        serving_artifact_id=serving_artifact_id,
        caller_pid=caller_pid if caller_pid is not None else os.getpid(),
        runtime=runtime,
        client=client,
        timeout_s=timeout_s,
    ) as lease:
        attached = lease.restore(
            target_device=torch.device(target_device),
            restore_fn=restore_fn,
        )
        restored = RestoredRetainedBinding(attached)
        try:
            yield restored
        finally:
            restored.close()


@contextmanager
def restore_prepared_local_ready_binding(
    *,
    resolved_artifact: ResolvedServingArtifact,
    target_device: torch.device | str,
    expected_member: tc.ServingBindingMemberRef,
    expected_tensor_schema_hash: str,
    expected_serving_build_digest: str | None = None,
    caller_pid: int | None = None,
    timeout_s: float | None = None,
    runtime: Any | None = None,
    client: Any | None = None,
    restore_fn: Any | None = None,
) -> Iterator[RestoredRetainedBinding]:
    """Restore a local-ready retained value referenced by a serving manifest."""

    manifest = resolved_artifact.manifest
    local_serving_ref = getattr(manifest, "local_serving_ref", None)
    if manifest is None or not local_serving_ref:
        raise RuntimeError(
            "TensorCast prepared local-ready startup requires local_serving_ref "
            "in the serving artifact manifest"
        )
    serving_build_digest = (
        expected_serving_build_digest
        if expected_serving_build_digest is not None
        else getattr(manifest, "serving_build_digest", None)
    )
    if not serving_build_digest:
        raise RuntimeError(
            "TensorCast prepared local-ready startup requires serving_build_digest"
        )
    with restore_retained_binding(
        local_serving_ref=str(local_serving_ref),
        target_device=target_device,
        expected_member=expected_member,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        expected_serving_build_digest=str(serving_build_digest),
        serving_artifact_id=str(resolved_artifact.artifact_ref),
        caller_pid=caller_pid,
        timeout_s=timeout_s,
        runtime=runtime,
        client=client,
        restore_fn=restore_fn,
    ) as restored:
        yield restored


def evaluate_semantic_validation_spec(spec: Any, actual_payload: Any) -> Any:
    return tc_semantic_validation.evaluate_semantic_validation_spec(
        spec, actual_payload
    )


def validate_tensor_schema_against_tensors(
    tensor_schema: Any,
    tensors: Mapping[str, torch.Tensor],
) -> None:
    tc_tensor_schema.validate_tensor_schema_against_tensors(tensor_schema, tensors)


def collect_runtime_tensor_schema(
    tensors: Mapping[str, torch.Tensor],
    *,
    remove_duplicate: bool,
) -> Any:
    return tc_contract.collect_runtime_tensor_schema(
        tensors,
        remove_duplicate=remove_duplicate,
    )


def compute_runtime_tensor_schema_hash(schema: Any) -> str:
    return tc_contract.compute_runtime_tensor_schema_hash(schema)


def compute_runtime_representation_contract_hash(**kwargs: Any) -> str:
    return tc_contract.compute_runtime_representation_contract_hash(**kwargs)


def compute_serving_tensor_schema_hash(*args: Any, **kwargs: Any) -> str:
    return tc.compute_serving_tensor_schema_hash(*args, **kwargs)


def canonical_index_from_recipe(recipe: Any) -> Any:
    return tc_local_ready.canonical_index_from_recipe(recipe)


def materialized_tensor_schema(recipe: Any) -> Any:
    return tc_local_ready.materialized_tensor_schema(recipe)


def prepare_same_binding_manifest_carrier(*args: Any, **kwargs: Any) -> Any:
    return tc_local_ready.prepare_same_binding_manifest_carrier(*args, **kwargs)


def compute_serving_binding_tensor_schema_hash(*args: Any, **kwargs: Any) -> str:
    return tc_local_ready.compute_serving_binding_tensor_schema_hash(*args, **kwargs)


def prepare_local_ready_serving(*args: Any, **kwargs: Any) -> Any:
    return tc_local_ready.prepare_local_ready_serving(*args, **kwargs)


def freeze_local_ready_binding(*args: Any, **kwargs: Any) -> Any:
    return tc_local_ready.freeze_local_ready_binding(*args, **kwargs)


def tensorcast_view_slice_count(recipe: Any) -> int:
    return tc_local_ready.tensorcast_view_slice_count(recipe)


def compiled_recipe_realization_plan_count(recipe: Any) -> int:
    return tc_local_ready.compiled_recipe_realization_plan_count(recipe)


def binding_value_verification_state_name(value: Any) -> str:
    return tc_local_ready.binding_value_verification_state_name(value)


def logical_topology_json_from_recipe(*args: Any, **kwargs: Any) -> Any:
    return tc_local_ready.logical_topology_json_from_recipe(*args, **kwargs)


def publication_context_from_recipe(*args: Any, **kwargs: Any) -> Any:
    return tc_local_ready.publication_context_from_recipe(*args, **kwargs)


def resolve_source_artifact_ref(*args: Any, **kwargs: Any) -> Any:
    return tc_source_catalog.resolve_source_artifact_ref(*args, **kwargs)


def source_catalog_from_selected_safetensors(*args: Any, **kwargs: Any) -> Any:
    return tc_source_catalog.source_catalog_from_selected_safetensors(*args, **kwargs)


def compute_trace_build_cache_key(*args: Any, **kwargs: Any) -> str:
    return tc_recipe_build.compute_trace_cache_key(*args, **kwargs)


def compute_recipe_build_cache_key(*args: Any, **kwargs: Any) -> str:
    return tc_recipe_build.compute_recipe_cache_key(*args, **kwargs)


def trace_build_cache_path(*args: Any, **kwargs: Any) -> str:
    return tc_recipe_build.trace_cache_path(*args, **kwargs)


def recipe_build_cache_path(*args: Any, **kwargs: Any) -> str:
    return tc_recipe_build.recipe_cache_path(*args, **kwargs)


def stable_recipe_build_hash(*args: Any, **kwargs: Any) -> str:
    return tc_recipe_build.stable_recipe_build_hash(*args, **kwargs)


def load_trace_plan_cache(*args: Any, **kwargs: Any) -> Any:
    return tc_trace_cache.load_trace_plan_cache(*args, **kwargs)


def write_trace_plan_cache(*args: Any, **kwargs: Any) -> None:
    tc_trace_cache.write_trace_plan_cache(*args, **kwargs)


def dump_trace_plan_debug(*args: Any, **kwargs: Any) -> None:
    tc_trace_cache.dump_trace_plan_debug(*args, **kwargs)


def load_compiled_recipe_cache(*args: Any, **kwargs: Any) -> Any:
    return tc_recipe_cache.load_compiled_recipe_cache(*args, **kwargs)


def write_compiled_recipe_cache(*args: Any, **kwargs: Any) -> None:
    tc_recipe_cache.write_compiled_recipe_cache(*args, **kwargs)


def compute_recipe_compile_key(*args: Any, **kwargs: Any) -> str:
    return tc_compiler.compute_recipe_compile_key(*args, **kwargs)


def compute_recipe_compile_key_from_inputs(*args: Any, **kwargs: Any) -> str:
    return tc_compiler.compute_recipe_compile_key(*args, **kwargs)


def compile_recipe_from_inputs(*args: Any, **kwargs: Any) -> Any:
    return tc_compiler.compile_serving_recipe(*args, **kwargs)


def allocate_tensors_from_schema(*args: Any, **kwargs: Any) -> Any:
    return tc_materialization.allocate_tensors_from_schema(*args, **kwargs)


def apply_copy_plan(*args: Any, **kwargs: Any) -> Any:
    return tc_materialization.apply_copy_plan(*args, **kwargs)


def tensorcast_view_slices_from_trace_plan(*args: Any, **kwargs: Any) -> Any:
    return tc_materialization.tensorcast_view_slices_from_trace_plan(*args, **kwargs)


def validate_dst_coverage(*args: Any, **kwargs: Any) -> None:
    tc_materialization.validate_dst_coverage(*args, **kwargs)


def validate_source_tensor_names(*args: Any, **kwargs: Any) -> None:
    tc_materialization.validate_source_tensor_names(*args, **kwargs)


def validate_recipe_for_builder_mode(*args: Any, **kwargs: Any) -> None:
    tc_recipe_validation.validate_recipe_for_builder_mode(*args, **kwargs)


def build_pure_transform_build_intent(*args: Any, **kwargs: Any) -> Any:
    return tc_publication.build_pure_transform_build_intent(*args, **kwargs)


def complete_pure_transform_publication(*args: Any, **kwargs: Any) -> Any:
    return tc.complete_pure_transform_publication(*args, **kwargs)


def build_materialization_execution_context(*args: Any, **kwargs: Any) -> Any:
    return tc_binding_runtime.build_materialization_execution_context(*args, **kwargs)


def retained_binding_acquire_mode(*args: Any, **kwargs: Any) -> str:
    return tc_retained_binding.retained_binding_acquire_mode(*args, **kwargs)


def retained_serving_binding_trusted_reservation_bytes(
    *args: Any, **kwargs: Any
) -> int:
    return tc_retained_binding.retained_serving_binding_trusted_reservation_bytes(
        *args, **kwargs
    )


def retained_serving_binding_extra_from_prefetched_binding(
    *args: Any, **kwargs: Any
) -> Any:
    return tc_retained_binding.retained_serving_binding_extra_from_prefetched_binding(
        *args, **kwargs
    )


def parse_retained_serving_binding_authority(*args: Any, **kwargs: Any) -> Any:
    return tc_retained_binding.parse_retained_serving_binding_authority(*args, **kwargs)
