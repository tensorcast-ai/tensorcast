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
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

import tensorcast as tc
from tensorcast.serving import binding_runtime as tc_binding_runtime
from tensorcast.serving import config as tc_config
from tensorcast.serving import contract as tc_contract
from tensorcast.serving import dto as tc_dto
from tensorcast.serving import hosts as tc_hosts
from tensorcast.serving import local_ready as tc_local_ready
from tensorcast.serving import policy as tc_policy
from tensorcast.serving import preload as tc_preload
from tensorcast.serving import readiness as tc_readiness
from tensorcast.serving import recipe_build as tc_recipe_build
from tensorcast.serving import runtime as tc_runtime
from tensorcast.serving import runtime_contract as tc_runtime_contract
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
    is_reserved_serving_tensor_name,
)
from tensorcast.types import (
    CollectivePolicy,
    FinalizeClass,
    ServingSupportLevel,
    ServingTopologyRef,
)

ArtifactError = tc.ArtifactError
BootstrapSummary = tc_dto.BootstrapSummary
BindingUpdateEpoch = tc.BindingUpdateEpoch
BindingReservationCapability = tc.BindingReservationCapability
BindingValueRef = tc.BindingValueRef
BuilderMode = tc.BuilderMode
CompiledServingRecipe = tc_compiler.CompiledServingRecipe
BindingFinalizeMaterializationResult = (
    tc_materialization.BindingFinalizeMaterializationResult
)
DEFAULT_RUNTIME_PROFILE = tc_runtime.DEFAULT_RUNTIME_PROFILE
FamilyReadiness = tc_dto.FamilyReadiness
FrameworkIntegrationContext = tc_dto.FrameworkIntegrationContext
PreparedServingArtifact = tc_dto.PreparedServingArtifact
PublishedModelVersion = tc.PublishedModelVersion
RecipeBuildIdentity = tc_recipe_build.RecipeBuildIdentity
RecipeBuildCacheConfig = tc_recipe_build.RecipeBuildCacheConfig
RecipeBuildRunResult = tc_recipe_build.RecipeBuildRunResult
RecipeCacheLookupResult = tc_recipe_build.RecipeCacheLookupResult
RecipeCacheWriteResult = tc_recipe_build.RecipeCacheWriteResult
RecipeBuildSession = tc_recipe_build.RecipeBuildSession
COMPILED_RECIPE_MEMORY_CACHE = tc_recipe_build.COMPILED_RECIPE_MEMORY_CACHE
TRACE_PLAN_MEMORY_CACHE = tc_recipe_build.TRACE_PLAN_MEMORY_CACHE
RecipeCompileIdentity = tc_compiler.RecipeCompileIdentity
RecipeCompileInputs = tc_compiler.RecipeCompileInputs
RecipePublicationContext = tc_publication.RecipePublicationContext
ParsedExternalPreloadAuthority = tc_preload.ParsedExternalPreloadAuthority
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

# Host capability contracts live in hosts.py.  Integration imports aliases so
# lifecycle code and historical module exports use one canonical contract.
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
TensorcastLogicalTopology = tc_compiler.TensorcastLogicalTopology
TensorcastSemanticValidationSpec = tc_compiler.TensorcastSemanticValidationSpec
TensorcastServingFacts = tc_compiler.TensorcastServingFacts
TensorSchemaEntry = tc_compiler.TensorSchemaEntry
read_source_bound_contract_state = tc_runtime_contract.read_source_bound_contract_state
resolve_runtime_config_profile = tc_runtime.resolve_runtime_config_profile

RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION = 1
WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION = 1
RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION = 1
PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION = 1
SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION = 1
RETAINED_BINDING_AUTHORITY_SCHEMA_VERSION = 1
RETAINED_BINDING_READINESS_STATES = {
    "serving_reserved",
    "serving_local_ready",
    "serving_published_ready",
}
SERVING_ARTIFACT_SELECTOR_SCHEMA_VERSION = tc_policy.SERVING_SELECTOR_SCHEMA_VERSION
SERVING_POLICY_SCHEMA_VERSION = tc_policy.SERVING_POLICY_SCHEMA_VERSION
ServingArtifactSelector = tc_policy.ServingSelector
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


class TensorCastServingRuntimeError(RuntimeError):
    """Base class for machine-readable serving runtime failures."""

    code = "tensorcast_serving_runtime_error"
    operation = "serving_runtime"
    retryable = False
    worker_suspect = False

    def __init__(
        self,
        message: str = "",
        *,
        operation: str | None = None,
        retryable: bool | None = None,
        worker_suspect: bool | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation or self.operation
        self.retryable = self.retryable if retryable is None else retryable
        self.worker_suspect = (
            self.worker_suspect if worker_suspect is None else worker_suspect
        )
        self.details = dict(details or {})


class ServingIntegrationError(TensorCastServingRuntimeError):
    """Base class for structured serving integration failures."""


class ServingIntegrationNotImplementedError(ServingIntegrationError):
    """Raised when a deep core-owned lifecycle method is not implemented yet."""

    code = "not_implemented"
    operation = "serving_runtime"


class ConfigConflictError(ServingIntegrationError):
    """Serving config requests mutually exclusive lifecycle execution modes."""

    code = "config_conflict"
    operation = "config_planning"


class CapabilityMissingError(ServingIntegrationError):
    """Required host capability is absent for a requested lifecycle path."""

    code = "capability_missing"
    operation = "capability_validation"


def _capability_missing(
    message: str,
    *,
    level: str,
    capability: str,
    operation: str,
    required_methods: Sequence[str] = (),
    next_action: str,
) -> CapabilityMissingError:
    return CapabilityMissingError(
        message,
        operation=operation,
        details={
            "level": level,
            "capability": capability,
            "operation": operation,
            "required_methods": tuple(required_methods),
            "next_action": next_action,
        },
    )


class AdmissionRejectedError(ServingIntegrationError):
    """Core admission rejected a serving lifecycle request."""

    code = "admission_rejected"
    operation = "admission"


class PlacementAdmissionError(ServingIntegrationError):
    """Placement identity or semantic placement proof is invalid."""

    code = "placement_admission"
    operation = "placement_admission"


class SelectorResolutionError(ServingIntegrationError):
    """Durable serving selector could not resolve to a serving artifact."""

    code = "selector_resolution"
    operation = "selector_resolution"


class ManifestMismatchError(ServingIntegrationError):
    """Serving manifest content does not match requested runtime facts."""

    code = "manifest_mismatch"
    operation = "manifest_validation"


class PolicyMismatchError(ServingIntegrationError):
    """Serving runtime policy does not match the artifact manifest."""

    code = "policy_mismatch"
    operation = "policy_validation"


class AuthorityValidationError(ServingIntegrationError):
    """Retained binding authority failed validation."""

    code = "authority_validation"
    operation = "retained_acquire"


class SchemaMismatchError(ServingIntegrationError):
    """Runtime tensor schema does not match the serving artifact schema."""

    code = "schema_mismatch"
    operation = "schema_validation"
    worker_suspect = True


class AttachFinalizeError(ServingIntegrationError):
    """Framework attach, process-after-load, or finalize failed."""

    code = "attach_finalize"
    operation = "attach_finalize"
    worker_suspect = True


class RestoreBindingError(ServingIntegrationError):
    """Retained binding restore failed before runtime ownership transfer."""

    code = "restore_binding"
    operation = "retained_acquire"


class OwnershipTransferError(ServingIntegrationError):
    """Binding ownership transfer to runtime state failed."""

    code = "ownership_transfer"
    operation = "ownership_transfer"
    worker_suspect = True


class RuntimeSwapError(ServingIntegrationError):
    """Serving binding swap failed after execution started."""

    code = "runtime_swap"
    operation = "reload"
    worker_suspect = True


class SourceSubjectError(ServingIntegrationError):
    """Source selector resolution or broadcast payload handling failed."""

    code = "source_subject"
    operation = "source_provider"


class SourceProviderError(ServingIntegrationError):
    """Source provider, catalog, or cache policy failed."""

    code = "source_provider"
    operation = "source_provider"


class PublicationRequiredError(ServingIntegrationError):
    """A local-ready identity was used where durable publication is required."""

    code = "publication_required"
    operation = "selector_validation"


class ReplicaPublicationError(ServingIntegrationError):
    """Runtime-owned ephemeral replica publication failed."""

    code = "replica_publication"
    operation = "replica_publication"
    worker_suspect = True


@dataclass(frozen=True)
class BootstrapPolicy:
    fields: Mapping[str, object] = field(default_factory=dict)


class ServingIntent:
    """Marker base class for public serving lifecycle intent DTOs."""


@dataclass(frozen=True)
class ExistingServingArtifact(ServingIntent):
    selector: ServingArtifactSelector | object
    policy: ServingPolicy | object | None = None


@dataclass(frozen=True)
class LocalSourceBootstrap(ServingIntent):
    source_selector: SourceSelector
    bootstrap_policy: Any
    cache_policy: RecipeCachePolicy | None = None


@dataclass(frozen=True)
class _AdminLocalSourceBootstrap(LocalSourceBootstrap):
    """Private/admin local bootstrap request with prebuilt lifecycle inputs."""

    coordinator: SourceSubjectCoordinator | None = None
    source_catalog_config: object | None = None
    cache_config_factory: object | None = None
    recipe: object | None = None
    source_subject: object | None = None
    source_artifact_ref: str | None = None
    model: object | None = None
    binding_factory: object | None = None


@dataclass(frozen=True)
class RetainedBindingAuthority:
    group_id: str
    binding_value_ref: BindingValueRef
    reservation_capability: BindingReservationCapability
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    member: ServingBindingMemberRef
    reservation_bytes: int
    expected_target_layout_hash: str
    expected_tensor_schema_hash: str
    expected_serving_build_digest: str
    expected_resolved_spec_digest: str
    readiness: str
    verification_state: str = "local_only"
    local_serving_ref: str | None = None
    serving_artifact_id: str | None = None
    group_realization_acquire: GroupRealizationAcquireRef | None = None
    schema_version: int = RETAINED_BINDING_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RETAINED_BINDING_AUTHORITY_SCHEMA_VERSION:
            raise AuthorityValidationError(
                "RetainedBindingAuthority.schema_version is unsupported",
                details={
                    "expected": RETAINED_BINDING_AUTHORITY_SCHEMA_VERSION,
                    "actual": self.schema_version,
                },
            )
        self._require_non_empty("group_id", self.group_id)
        self._require_non_empty("daemon_id", self.daemon_id)
        self._require_non_empty("daemon_session_id", self.daemon_session_id)
        self._require_non_empty("device_uuid", self.device_uuid)
        self._require_non_empty(
            "expected_target_layout_hash", self.expected_target_layout_hash
        )
        self._require_non_empty(
            "expected_tensor_schema_hash", self.expected_tensor_schema_hash
        )
        self._require_non_empty(
            "expected_serving_build_digest", self.expected_serving_build_digest
        )
        self._require_non_empty(
            "expected_resolved_spec_digest", self.expected_resolved_spec_digest
        )
        self._require_non_empty("verification_state", self.verification_state)
        if self.readiness not in RETAINED_BINDING_READINESS_STATES:
            raise AuthorityValidationError(
                "RetainedBindingAuthority.readiness is unsupported",
                details={
                    "allowed": sorted(RETAINED_BINDING_READINESS_STATES),
                    "actual": self.readiness,
                },
            )
        if not isinstance(self.binding_value_ref, BindingValueRef):
            raise AuthorityValidationError(
                "RetainedBindingAuthority.binding_value_ref must be BindingValueRef"
            )
        if not isinstance(self.reservation_capability, BindingReservationCapability):
            raise AuthorityValidationError(
                "RetainedBindingAuthority.reservation_capability must be "
                "BindingReservationCapability"
            )
        if not isinstance(self.member, ServingBindingMemberRef):
            raise AuthorityValidationError(
                "RetainedBindingAuthority.member must be ServingBindingMemberRef"
            )
        if self.group_realization_acquire is not None and not isinstance(
            self.group_realization_acquire, GroupRealizationAcquireRef
        ):
            raise AuthorityValidationError(
                "RetainedBindingAuthority.group_realization_acquire must be "
                "GroupRealizationAcquireRef"
            )
        if self.reservation_bytes < 0:
            raise AuthorityValidationError(
                "RetainedBindingAuthority.reservation_bytes must be non-negative"
            )
        capability = self.reservation_capability
        expected_pairs = {
            "binding_value_ref": self.binding_value_ref,
            "daemon_id": self.daemon_id,
            "daemon_session_id": self.daemon_session_id,
            "device_uuid": self.device_uuid,
            "member": self.member,
            "reservation_bytes": self.reservation_bytes,
        }
        for field_name, expected in expected_pairs.items():
            actual = getattr(capability, field_name)
            if actual != expected:
                raise AuthorityValidationError(
                    "RetainedBindingAuthority.reservation_capability "
                    f"{field_name} mismatch",
                    details={
                        "expected": repr(expected),
                        "actual": repr(actual),
                    },
                )
        if self.member.group_id is not None and self.member.group_id != self.group_id:
            raise AuthorityValidationError(
                "RetainedBindingAuthority.member.group_id must match group_id",
                details={
                    "group_id": self.group_id,
                    "member_group_id": self.member.group_id,
                },
            )
        if self.readiness == "serving_published_ready" and not self.serving_artifact_id:
            raise AuthorityValidationError(
                "RetainedBindingAuthority.serving_artifact_id is required when "
                "readiness='serving_published_ready'"
            )

    @staticmethod
    def _require_non_empty(field_name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise AuthorityValidationError(
                f"RetainedBindingAuthority.{field_name} must be non-empty"
            )

    @classmethod
    def from_preload_authority(
        cls,
        authority: ParsedExternalPreloadAuthority,
    ) -> "RetainedBindingAuthority":
        if not isinstance(authority, ParsedExternalPreloadAuthority):
            raise AuthorityValidationError(
                "RetainedBindingAuthority.from_preload_authority requires "
                "ParsedExternalPreloadAuthority"
            )
        return cls(
            group_id=authority.group_id,
            binding_value_ref=authority.binding_value_ref,
            reservation_capability=authority.reservation_capability,
            daemon_id=authority.daemon_id,
            daemon_session_id=authority.daemon_session_id,
            device_uuid=authority.device_uuid,
            member=authority.member,
            reservation_bytes=authority.reservation_bytes,
            expected_target_layout_hash=authority.expected.target_layout_hash,
            expected_tensor_schema_hash=authority.expected.tensor_schema_hash,
            expected_serving_build_digest=(authority.expected.serving_build_digest),
            expected_resolved_spec_digest=(authority.expected.resolved_spec_digest),
            readiness=authority.readiness,
            verification_state=authority.verification_state,
            local_serving_ref=authority.local_serving_ref,
            serving_artifact_id=authority.serving_artifact_id,
            group_realization_acquire=authority.group_realization_acquire,
        )

    def to_preload_authority(self) -> ParsedExternalPreloadAuthority:
        return ParsedExternalPreloadAuthority(
            group_id=self.group_id,
            local_serving_ref=self.local_serving_ref,
            binding_value_ref=self.binding_value_ref,
            reservation_capability=self.reservation_capability,
            daemon_id=self.daemon_id,
            daemon_session_id=self.daemon_session_id,
            device_uuid=self.device_uuid,
            member=self.member,
            reservation_bytes=self.reservation_bytes,
            expected=tc_preload.ExternalPreloadExpectedDigests(
                target_layout_hash=self.expected_target_layout_hash,
                tensor_schema_hash=self.expected_tensor_schema_hash,
                serving_build_digest=self.expected_serving_build_digest,
                resolved_spec_digest=self.expected_resolved_spec_digest,
            ),
            readiness=self.readiness,
            verification_state=self.verification_state,
            serving_artifact_id=self.serving_artifact_id,
            group_realization_acquire=self.group_realization_acquire,
        )


@dataclass(frozen=True)
class RetainedBindingAcquire(ServingIntent):
    authority: RetainedBindingAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.authority, RetainedBindingAuthority):
            raise AuthorityValidationError(
                "RetainedBindingAcquire.authority must be RetainedBindingAuthority"
            )


@dataclass(frozen=True)
class RequestContext:
    framework_config: object | None = None
    model_config: object | None = None
    target_device: object | None = None
    timeout_s: float | None = 30.0


@dataclass(frozen=True)
class BindingValueRefProjection:
    binding_id: str
    binding_layout_id: str
    binding_value_id: str
    seal_generation: int

    @classmethod
    def from_value(cls, value: object) -> "BindingValueRefProjection | None":
        if value is None:
            return None
        if isinstance(value, Mapping):
            return cls(
                binding_id=str(value.get("binding_id", "") or ""),
                binding_layout_id=str(value.get("binding_layout_id", "") or ""),
                binding_value_id=str(value.get("binding_value_id", "") or ""),
                seal_generation=int(value.get("seal_generation", 0) or 0),
            )
        return cls(
            binding_id=str(getattr(value, "binding_id", "") or ""),
            binding_layout_id=str(getattr(value, "binding_layout_id", "") or ""),
            binding_value_id=str(getattr(value, "binding_value_id", "") or ""),
            seal_generation=int(getattr(value, "seal_generation", 0) or 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "binding_layout_id": self.binding_layout_id,
            "binding_value_id": self.binding_value_id,
            "seal_generation": self.seal_generation,
        }


@dataclass(frozen=True)
class SourceBoundContractProjection:
    fields: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return dict(self.fields)


@dataclass(frozen=True)
class MaterializationDiagnosticsProjection:
    fields: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return dict(self.fields)


@dataclass(frozen=True)
class ReloadRequestProjection:
    selector: Mapping[str, object] | None = None
    policy: Mapping[str, object] | None = None
    requested_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.selector is not None:
            payload["selector"] = dict(self.selector)
        if self.policy is not None:
            payload["policy"] = dict(self.policy)
        if self.requested_at is not None:
            payload["requested_at"] = self.requested_at
        return payload


@dataclass(frozen=True)
class PublishedReplicaProjection:
    state: str
    operation_id: str | None = None
    replica_id: str | None = None
    lease_id: str | None = None
    artifact_ref: str | None = None
    device_uuid: str | None = None
    owner_pid: int | None = None
    byte_space_kind: str | None = None
    byte_space_id: str | None = None
    binding_layout_id: str | None = None
    binding_value_ref: BindingValueRefProjection | None = None
    generation: str | None = None
    reason: str | None = None
    schema_version: int = PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "state": self.state,
        }
        optional: dict[str, object | None] = {
            "operation_id": self.operation_id,
            "replica_id": self.replica_id,
            "lease_id": self.lease_id,
            "artifact_ref": self.artifact_ref,
            "device_uuid": self.device_uuid,
            "owner_pid": self.owner_pid,
            "byte_space_kind": self.byte_space_kind,
            "byte_space_id": self.byte_space_id,
            "binding_layout_id": self.binding_layout_id,
            "generation": self.generation,
            "reason": self.reason,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        if self.binding_value_ref is not None:
            payload["binding_value_ref"] = self.binding_value_ref.to_dict()
        return payload


@dataclass(frozen=True)
class SourceSelectionProjection:
    selected_source_kind: str
    selected_replica_id: str | None = None
    selected_producer_worker_id: str | None = None
    selected_byte_space_kind: str | None = None
    selected_byte_space_id: str | None = None
    p2p_bytes: int = 0
    fallback_bytes: int = 0
    disk_bytes: int = 0
    reselection_attempts: int = 0
    reject_reason_bucket: str | None = None
    fallback_reason_bucket: str | None = None
    schema_version: int = SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "selected_source_kind": self.selected_source_kind,
            "p2p_bytes": self.p2p_bytes,
            "fallback_bytes": self.fallback_bytes,
            "disk_bytes": self.disk_bytes,
            "reselection_attempts": self.reselection_attempts,
        }
        optional: dict[str, object | None] = {
            "selected_replica_id": self.selected_replica_id,
            "selected_producer_worker_id": self.selected_producer_worker_id,
            "selected_byte_space_kind": self.selected_byte_space_kind,
            "selected_byte_space_id": self.selected_byte_space_id,
            "reject_reason_bucket": self.reject_reason_bucket,
            "fallback_reason_bucket": self.fallback_reason_bucket,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return payload


@dataclass(frozen=True)
class BootstrapEndpointProjection:
    fields: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.fields)


def _source_bound_projection_from_bootstrap(
    bootstrap_summary: BootstrapSummary | None,
) -> SourceBoundContractProjection | None:
    if bootstrap_summary is None:
        return None
    fields = {
        "version": getattr(bootstrap_summary, "source_bound_contract_version", 0),
        "capability_flags": list(
            getattr(bootstrap_summary, "source_bound_capability_flags", ())
        ),
        "ready": bool(getattr(bootstrap_summary, "source_bound_contract_ready", False)),
        "path": _optional_text(
            getattr(bootstrap_summary, "source_bound_contract_path", None)
        ),
    }
    if not any(value for value in fields.values()):
        return None
    return SourceBoundContractProjection(fields)


def _materialization_projection_from_fields(
    *,
    prefix: str,
    diagnostics: Mapping[str, object],
    bootstrap_summary: BootstrapSummary | None,
) -> MaterializationDiagnosticsProjection | None:
    fields: dict[str, object] = {}
    diagnostics_prefix = f"{prefix}_"
    for key, value in diagnostics.items():
        if key.startswith(diagnostics_prefix) and value is not None:
            fields[key[len(diagnostics_prefix) :]] = value
    if bootstrap_summary is not None:
        bootstrap_prefix = f"bootstrap_{prefix}_"
        for key, value in bootstrap_summary.to_dict().items():
            if key.startswith(bootstrap_prefix) and value is not None:
                fields[key[len(bootstrap_prefix) :]] = value
    if not fields:
        return None
    return MaterializationDiagnosticsProjection(fields)


def _reload_request_projection_from_diagnostics(
    diagnostics: Mapping[str, object],
) -> ReloadRequestProjection | None:
    value = diagnostics.get("reload_request")
    if value is None:
        return None
    if isinstance(value, ReloadRequestProjection):
        return value
    if not isinstance(value, Mapping):
        return None
    selector = value.get("selector")
    policy = value.get("policy")
    return ReloadRequestProjection(
        selector=dict(selector) if isinstance(selector, Mapping) else None,
        policy=dict(policy) if isinstance(policy, Mapping) else None,
        requested_at=_optional_text(value.get("requested_at")),
    )


def _published_replica_projection_from_value(
    value: object | None,
) -> PublishedReplicaProjection | None:
    if value is None:
        return None
    if isinstance(value, PublishedReplicaProjection):
        return value
    if not isinstance(value, Mapping):
        return None
    binding_value_ref = BindingValueRefProjection.from_value(
        value.get("binding_value_ref")
    )
    owner_pid = _optional_int(value.get("owner_pid"))
    return PublishedReplicaProjection(
        state=str(value.get("state") or ""),
        operation_id=_optional_text(value.get("operation_id")),
        replica_id=_optional_text(value.get("replica_id")),
        lease_id=_optional_text(value.get("lease_id")),
        artifact_ref=_optional_text(value.get("artifact_ref")),
        device_uuid=_optional_text(value.get("device_uuid")),
        owner_pid=owner_pid,
        byte_space_kind=_optional_text(value.get("byte_space_kind")),
        byte_space_id=_optional_text(value.get("byte_space_id")),
        binding_layout_id=_optional_text(value.get("binding_layout_id")),
        binding_value_ref=binding_value_ref,
        generation=_optional_text(value.get("generation")),
        reason=_optional_text(value.get("reason")),
    )


def _source_selection_projection_from_value(
    value: object | None,
) -> SourceSelectionProjection | None:
    if value is None:
        return None
    if isinstance(value, SourceSelectionProjection):
        return value
    if not isinstance(value, Mapping):
        return None
    return SourceSelectionProjection(
        selected_source_kind=str(value.get("selected_source_kind") or "unselected"),
        selected_replica_id=_optional_text(value.get("selected_replica_id")),
        selected_producer_worker_id=_optional_text(
            value.get("selected_producer_worker_id")
        ),
        selected_byte_space_kind=_optional_text(value.get("selected_byte_space_kind")),
        selected_byte_space_id=_optional_text(value.get("selected_byte_space_id")),
        p2p_bytes=_optional_int(value.get("p2p_bytes")) or 0,
        fallback_bytes=_optional_int(value.get("fallback_bytes")) or 0,
        disk_bytes=_optional_int(value.get("disk_bytes")) or 0,
        reselection_attempts=(_optional_int(value.get("reselection_attempts")) or 0),
        reject_reason_bucket=_optional_text(value.get("reject_reason_bucket")),
        fallback_reason_bucket=_optional_text(value.get("fallback_reason_bucket")),
    )


def _diagnostic_value(
    diagnostics: Any,
    name: str,
    default: object | None = None,
) -> object | None:
    if isinstance(diagnostics, Mapping):
        return diagnostics.get(name, default)
    return getattr(diagnostics, name, default)


def _dominant_reason_bucket(value: object | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    candidates: list[tuple[int, str]] = []
    for key, count in value.items():
        name = _optional_text(key)
        weight = _optional_int(count) or 0
        if name is not None and weight > 0:
            candidates.append((weight, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def source_selection_projection_from_materialization_diagnostics(
    diagnostics: Any | None,
) -> SourceSelectionProjection | None:
    """Project store materialization diagnostics into the runtime endpoint DTO."""

    if diagnostics is None:
        return None
    source = _optional_text(_diagnostic_value(diagnostics, "source"))
    if source is None:
        return None
    total_bytes = _optional_int(_diagnostic_value(diagnostics, "total_bytes")) or 0
    reselection_attempts = max(
        0,
        (_optional_int(_diagnostic_value(diagnostics, "retry_attempts")) or 1) - 1,
    )
    reason_bucket = _dominant_reason_bucket(
        _diagnostic_value(diagnostics, "retry_reason_buckets")
    )
    replica_id = _optional_text(
        _diagnostic_value(diagnostics, "ticket_replica_uuid")
    ) or _optional_text(_diagnostic_value(diagnostics, "replica_uuid"))
    if source == "p2p":
        return SourceSelectionProjection(
            selected_source_kind="published_memory_replica",
            selected_replica_id=replica_id,
            p2p_bytes=total_bytes,
            reselection_attempts=reselection_attempts,
            reject_reason_bucket=reason_bucket,
        )
    if source == "local_replica":
        return SourceSelectionProjection(
            selected_source_kind="local_memory_replica",
            selected_replica_id=replica_id,
            reselection_attempts=reselection_attempts,
            reject_reason_bucket=reason_bucket,
        )
    if source == "disk":
        return SourceSelectionProjection(
            selected_source_kind="canonical_fallback",
            fallback_bytes=total_bytes,
            disk_bytes=total_bytes,
            reselection_attempts=reselection_attempts,
            fallback_reason_bucket=reason_bucket,
        )
    return None


def source_selection_projection_from_execution_diagnostics(
    diagnostics: Any | None,
) -> SourceSelectionProjection | None:
    """Summarize daemon execution diagnostics as low-cardinality source choice."""

    if diagnostics is None:
        return None
    collective_bytes = (
        _optional_int(
            _diagnostic_value(diagnostics, "actual_collective_committed_bytes")
        )
        or 0
    )
    peer_transfer_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "collective_peer_transfer_bytes"))
        or 0
    )
    local_typed_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "actual_local_typed_bytes")) or 0
    )
    generic_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "actual_generic_backend_bytes"))
        or 0
    )
    fallback_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "fallback_bytes")) or 0
    )
    residual_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "residual_bytes")) or 0
    )
    skip_reason = _optional_text(
        _diagnostic_value(diagnostics, "collective_skip_reason")
    )
    if collective_bytes or peer_transfer_bytes:
        return SourceSelectionProjection(
            selected_source_kind="published_memory_replica",
            p2p_bytes=peer_transfer_bytes or collective_bytes,
            fallback_bytes=fallback_bytes,
            fallback_reason_bucket=skip_reason if fallback_bytes else None,
        )
    if local_typed_bytes:
        return SourceSelectionProjection(
            selected_source_kind="local_memory_replica",
            fallback_bytes=fallback_bytes,
            fallback_reason_bucket=skip_reason if fallback_bytes else None,
        )
    if fallback_bytes or residual_bytes or generic_bytes:
        return SourceSelectionProjection(
            selected_source_kind="canonical_fallback",
            fallback_bytes=fallback_bytes or residual_bytes or generic_bytes,
            fallback_reason_bucket=skip_reason,
        )
    return None


@dataclass(frozen=True)
class WeightVersionProjection:
    source_artifact_ref: str | None
    serving_artifact_ref: str | None
    serving_version_key: str | None
    serving_manifest_ref: str | None
    representation_contract_hash: str
    serving_build_digest: str | None
    tensor_schema_hash: str
    readiness: str
    family: str
    tp_rank: int | None
    tp_world_size: int | None
    binding_layout_id: str | None
    local_serving_ref: str | None
    binding_value_ref: BindingValueRefProjection | None
    verification_state: str
    verification_job_id: str | None
    source_bound_contract: SourceBoundContractProjection | None = None
    realize_diagnostics: MaterializationDiagnosticsProjection | None = None
    publish_diagnostics: MaterializationDiagnosticsProjection | None = None
    published_replica: PublishedReplicaProjection | None = None
    source_selection: SourceSelectionProjection | None = None
    reload_request: ReloadRequestProjection | None = None
    bootstrap_summary: BootstrapEndpointProjection | None = None
    schema_version: int = WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_artifact_ref": self.source_artifact_ref,
            "serving_artifact_ref": self.serving_artifact_ref,
            "serving_version_key": self.serving_version_key,
            "serving_manifest_ref": self.serving_manifest_ref,
            "representation_contract_hash": self.representation_contract_hash,
            "serving_build_digest": self.serving_build_digest,
            "tensor_schema_hash": self.tensor_schema_hash,
            "readiness": self.readiness,
            "family": self.family,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "binding_layout_id": self.binding_layout_id,
            "local_serving_ref": self.local_serving_ref,
            "binding_value_ref": (
                None
                if self.binding_value_ref is None
                else self.binding_value_ref.to_dict()
            ),
            "verification_state": self.verification_state,
            "verification_job_id": self.verification_job_id,
        }
        optional = {
            "source_bound_contract": self.source_bound_contract,
            "realize_diagnostics": self.realize_diagnostics,
            "publish_diagnostics": self.publish_diagnostics,
            "published_replica": self.published_replica,
            "source_selection": self.source_selection,
            "reload_request": self.reload_request,
            "bootstrap_summary": self.bootstrap_summary,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value.to_dict()
        return payload


@dataclass(frozen=True)
class ReloadResponseProjection:
    serving_artifact_ref: str | None
    representation_contract_hash: str
    serving_build_digest: str | None
    readiness: str
    schema_version: int = RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "serving_artifact_ref": self.serving_artifact_ref,
            "representation_contract_hash": self.representation_contract_hash,
            "serving_build_digest": self.serving_build_digest,
            "readiness": self.readiness,
        }


@dataclass(frozen=True)
class RuntimeEndpointProjection:
    weight_version: WeightVersionProjection
    reload_response: ReloadResponseProjection | None = None
    schema_version: int = RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION

    def to_weight_version_payload(self) -> dict[str, object]:
        return self.weight_version.to_dict()

    def to_reload_response_payload(self) -> dict[str, object] | None:
        if self.reload_response is None:
            return None
        return self.reload_response.to_dict()


@dataclass(frozen=True)
class RuntimeWorkerView:
    readiness: str
    serving_artifact_ref: str | None
    source_artifact_ref: str | None
    representation_contract_hash: str
    serving_build_digest: str | None
    tensor_schema_hash: str
    local_serving_ref: str | None
    binding_value_ref: BindingValueRefProjection | None
    verification_state: str
    verification_job_id: str | None
    endpoint: RuntimeEndpointProjection
    diagnostics: Mapping[str, object]

    @classmethod
    def from_runtime_view(
        cls,
        view: RuntimeBindingView,
        *,
        bootstrap_summary: BootstrapSummary | None = None,
        family: str = "",
        tp_rank: int | None = None,
        tp_world_size: int | None = None,
        include_reload_response: bool = False,
    ) -> "RuntimeWorkerView":
        diagnostics = dict(view.diagnostics or {})
        binding_value_ref = BindingValueRefProjection.from_value(view.binding_value_ref)
        serving_build_digest = _optional_text(
            diagnostics.get("serving_build_digest")
        ) or _optional_text(getattr(bootstrap_summary, "serving_build_digest", None))
        verification_state = str(
            diagnostics.get("verification_state")
            or getattr(bootstrap_summary, "verification_state", None)
            or "verified"
        )
        verification_job_id = _optional_text(
            diagnostics.get("verification_job_id")
        ) or _optional_text(getattr(bootstrap_summary, "verification_job_id", None))
        bootstrap_projection = None
        if bootstrap_summary is not None:
            bootstrap_projection = BootstrapEndpointProjection(
                bootstrap_summary.to_dict()
            )
        source_bound_projection = _source_bound_projection_from_bootstrap(
            bootstrap_summary
        )
        realize_projection = _materialization_projection_from_fields(
            prefix="realize",
            diagnostics=diagnostics,
            bootstrap_summary=bootstrap_summary,
        )
        publish_projection = _materialization_projection_from_fields(
            prefix="publish",
            diagnostics=diagnostics,
            bootstrap_summary=bootstrap_summary,
        )
        reload_projection = _reload_request_projection_from_diagnostics(diagnostics)
        published_replica = _published_replica_projection_from_value(
            diagnostics.get("published_replica")
        )
        source_selection = _source_selection_projection_from_value(
            diagnostics.get("source_selection")
        )
        weight_version = WeightVersionProjection(
            source_artifact_ref=view.source_artifact_ref,
            serving_artifact_ref=view.serving_artifact_ref,
            serving_version_key=_optional_text(diagnostics.get("serving_version_key")),
            serving_manifest_ref=_optional_text(diagnostics.get("serving_manifest_ref"))
            or _optional_text(getattr(bootstrap_summary, "serving_manifest_ref", None)),
            representation_contract_hash=view.representation_contract_hash,
            serving_build_digest=serving_build_digest,
            tensor_schema_hash=view.tensor_schema_hash,
            readiness=view.readiness,
            family=family or str(diagnostics.get("family") or ""),
            tp_rank=(
                tp_rank
                if tp_rank is not None
                else _optional_int(diagnostics.get("tp_rank"))
            ),
            tp_world_size=(
                tp_world_size
                if tp_world_size is not None
                else _optional_int(diagnostics.get("tp_world_size"))
            ),
            binding_layout_id=_optional_text(diagnostics.get("binding_layout_id"))
            or _optional_text(getattr(bootstrap_summary, "binding_layout_id", None)),
            local_serving_ref=view.local_serving_ref,
            binding_value_ref=binding_value_ref,
            verification_state=verification_state,
            verification_job_id=verification_job_id,
            source_bound_contract=source_bound_projection,
            realize_diagnostics=realize_projection,
            publish_diagnostics=publish_projection,
            published_replica=published_replica,
            source_selection=source_selection,
            reload_request=reload_projection,
            bootstrap_summary=bootstrap_projection,
        )
        reload_response = None
        if include_reload_response:
            reload_response = ReloadResponseProjection(
                serving_artifact_ref=view.serving_artifact_ref,
                representation_contract_hash=view.representation_contract_hash,
                serving_build_digest=serving_build_digest,
                readiness=view.readiness,
            )
        return cls(
            readiness=view.readiness,
            serving_artifact_ref=view.serving_artifact_ref,
            source_artifact_ref=view.source_artifact_ref,
            representation_contract_hash=view.representation_contract_hash,
            serving_build_digest=serving_build_digest,
            tensor_schema_hash=view.tensor_schema_hash,
            local_serving_ref=view.local_serving_ref,
            binding_value_ref=binding_value_ref,
            verification_state=verification_state,
            verification_job_id=verification_job_id,
            endpoint=RuntimeEndpointProjection(
                weight_version=weight_version,
                reload_response=reload_response,
            ),
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class RuntimeAttachment:
    model: object
    state: RuntimeBindingState
    view: RuntimeWorkerView
    bootstrap_summary: BootstrapSummary | None = None
    prepared: PreparedServingArtifact | None = None


@dataclass(frozen=True)
class RuntimeBindingView:
    """Read-only framework-facing view of core-owned runtime binding state."""

    serving_artifact_ref: str | None = None
    source_artifact_ref: str | None = None
    representation_contract_hash: str = ""
    tensor_schema_hash: str = ""
    binding_value_ref: Any | None = None
    local_serving_ref: str | None = None
    readiness: str = ""
    diagnostics: Mapping[str, Any] | None = None


@dataclass
class RuntimeBindingState:
    """Core-owned runtime binding lifecycle state placeholder."""

    binding: Any | None = None
    artifact_ref: str | None = None
    runtime_view: RuntimeBindingView | None = None
    ownership_handle: Any | None = None

    def close(self) -> None:
        handle = self.ownership_handle or self.binding
        close = getattr(handle, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class RuntimeStateSeed:
    """Core state facts known before framework tensor materialization."""

    artifact_ref: str | None = None
    serving_artifact_ref: str | None = None
    source_artifact_ref: str | None = None
    representation_contract_hash: str = ""
    tensor_schema_hash: str = ""
    binding_value_ref: Any | None = None
    local_serving_ref: str | None = None
    readiness: str = "loaded"
    diagnostics: Mapping[str, Any] | None = None

    def runtime_view(self) -> RuntimeBindingView:
        return RuntimeBindingView(
            serving_artifact_ref=self.serving_artifact_ref,
            source_artifact_ref=self.source_artifact_ref,
            representation_contract_hash=self.representation_contract_hash,
            tensor_schema_hash=self.tensor_schema_hash,
            binding_value_ref=self.binding_value_ref,
            local_serving_ref=self.local_serving_ref,
            readiness=self.readiness,
            diagnostics=self.diagnostics,
        )


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
        del context
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
            try:
                state = self.state_factory(
                    binding=binding_handle,
                    artifact_ref=state_seed.artifact_ref,
                    runtime_view=view,
                    ownership_handle=owner,
                )
            except Exception as exc:
                self._close_quietly(owner)
                raise OwnershipTransferError(
                    "TensorCast runtime binding state construction failed"
                ) from exc
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
    selector: Any | None = None
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
    selector: Any | None = None
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
    bootstrap_summary: BootstrapSummary | None = None
    prepared: PreparedServingArtifact | None = None
    current_value: Any | None = None
    binding: Any | None = None
    update_epoch: Any | None = None
    layout: Any | None = None
    realization_entry_count: int | None = None
    realization: Any | None = None


@dataclass(frozen=True)
class RecipeBuildSessionRequest:
    source_subject: SourceSubject | Any | None = None
    framework_config: Any | None = None
    model_config: Any | None = None
    placement: ServingPlacement | None = None
    cache_config: Any | None = None
    identity: RecipeBuildIdentity | None = None
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


def _close_quietly(handle: object) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


@dataclass(frozen=True)
class RuntimeBindingResult:
    """Attach-ready result from a serving bind or swap operation."""

    binding: Any
    tensors: Mapping[str, torch.Tensor]
    binding_layout_id: str | None = None
    operation_result: Any | None = None
    execution_diagnostics: Any | None = None

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
        )


@dataclass
class RestoredRetainedBinding:
    """Restored retained binding tensors before runtime ownership transfer."""

    _attached: tc_preload.AttachedPreloadBinding
    _runtime_handle: tc_preload.RuntimePreloadAttachmentHandle | None = None

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
    def runtime_handle(self) -> tc_preload.RuntimePreloadAttachmentHandle | None:
        return self._runtime_handle

    def transfer_to_runtime(self) -> tc_preload.RuntimePreloadAttachmentHandle:
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


def _selector_kind(selector: object) -> str:
    if isinstance(selector, Mapping):
        return str(selector.get("kind") or "")
    return str(getattr(selector, "kind", "") or "")


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
        legacy_source = tc.PublicDiskSourceHandle(**payload_dict)
        return _source_subject_from_handle(legacy_source)
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
    selector: Any,
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
                "RuntimeBindingView.binding_value_ref must be BindingValueRef-compatible"
            )
    typed_binding_value_ref = cast(BindingValueRef | None, binding_value_ref)
    resolved_readiness = readiness or runtime_view.readiness or "loaded"
    state = "loaded" if resolved_readiness == "serving" else resolved_readiness
    return ServingBindingState(
        state=state,
        selector=selector,
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
) -> RuntimeBindingState:
    return RuntimeBindingState(
        binding=binding,
        artifact_ref=artifact_ref or runtime_view.serving_artifact_ref,
        runtime_view=runtime_view,
        ownership_handle=ownership_handle,
    )


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
        "compatibility_lowered_bytes": int(
            getattr(diagnostics, "compatibility_lowered_bytes", 0)
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


def binding_layout_tensor_count(layout: Any) -> int:
    target_layout = getattr(layout, "target_layout", None)
    offsets = getattr(target_layout, "offsets", None)
    if offsets is None:
        return -1
    return len(offsets)


def binding_layout_profile_fields(layout: Any) -> dict[str, Any]:
    target_index_bytes = getattr(layout, "target_index_bytes", b"") or b""
    return {
        "target_index_bytes": len(target_index_bytes),
        "binding_tensor_count": binding_layout_tensor_count(layout),
    }


def binding_layout_debug_payload(
    layout: Any,
    *,
    target_device: Any,
    context: str,
    pid: int,
) -> dict[str, Any]:
    target_layout = layout.target_layout
    target_index_bytes = layout.target_index_bytes
    return {
        "context": str(context),
        "pid": int(pid),
        "target_device": str(target_device),
        "binding_layout_id": str(layout.binding_layout_id),
        "target_index_bytes_len": len(target_index_bytes),
        "target_index_sha256": hashlib.sha256(target_index_bytes).hexdigest(),
        "layout": {
            "layout_kind": int(target_layout.layout_kind),
            "index_kind": int(target_layout.index_kind),
            "tensor_spec_kind": int(target_layout.tensor_spec_kind),
            "logical_layout_hash": bytes(target_layout.logical_layout_hash).hex(),
            "view_id": str(target_layout.view_id),
            "storages": [
                {
                    "storage_id": str(storage.storage_id),
                    "device_id": int(storage.device_id),
                    "storage_length": int(storage.storage_length),
                    "mapping_base_offset": int(storage.mapping_base_offset),
                }
                for storage in target_layout.storages
            ],
            "offsets": [
                {
                    "name": str(offset.name),
                    "storage_id": str(offset.storage_id),
                    "storage_offset": int(offset.storage_offset),
                    "logical_length": int(offset.logical_length),
                }
                for offset in target_layout.offsets
            ],
        },
        "dst_specs": [
            {
                "name": str(spec.name),
                "dtype": str(spec.dtype),
                "shape": [int(v) for v in spec.shape],
                "stride": [int(v) for v in spec.stride],
                "storage_offset": int(spec.storage_offset),
                "logical_length": int(spec.logical_length),
            }
            for spec in layout.dst_specs
        ],
    }


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
    runtime_view = RuntimeBindingView(
        serving_artifact_ref=None,
        source_artifact_ref=source_artifact_ref,
        representation_contract_hash=representation_contract_hash,
        tensor_schema_hash=tensor_schema_hash,
        binding_value_ref=binding_value_ref,
        local_serving_ref=local_serving_ref,
        readiness="serving_local_ready",
        diagnostics={
            "verification_state": verification_state,
            "verification_job_id": verification_job_id,
            "serving_manifest_ref": serving_manifest_ref,
            "serving_build_digest": serving_build_digest,
            "family": family,
            "tp_rank": int(tp_rank),
            "tp_world_size": int(tp_world_size),
        },
    )
    runtime_state = runtime_binding_state_from_runtime_view(
        binding=binding,
        runtime_view=runtime_view,
        artifact_ref=source_artifact_ref,
    )
    bootstrap_summary = BootstrapSummary(
        source_artifact_ref=source_artifact_ref,
        serving_artifact_ref=None,
        serving_manifest_ref=serving_manifest_ref,
        representation_contract_hash=representation_contract_hash,
        serving_build_digest=serving_build_digest,
        binding_value_ref=binding_value_ref,
        readiness="serving_local_ready",
        binding_layout_id=binding_layout_id,
        local_serving_ref=local_serving_ref,
        verification_state=verification_state,
        verification_job_id=verification_job_id,
        source_bound_contract_version=source_bound_contract_state.source_bound_contract_version,
        source_bound_capability_flags=source_bound_contract_state.source_bound_capability_names,
        source_bound_contract_ready=source_bound_contract_state.source_bound_contract_ready,
        source_bound_contract_path=source_bound_contract_path,
        **execution_diagnostics_summary_fields(
            getattr(binding, "last_execution_diagnostics", None),
            prefix="realize",
        ),
        **source_bound_plan_diagnostics_summary_fields(
            getattr(binding, "last_source_bound_plan_diagnostics", None),
            prefix="realize",
        ),
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
        bootstrap_summary=bootstrap_summary,
    )
    return LocalReadyServingResult(
        runtime_state=runtime_state,
        runtime_view=runtime_view,
        bootstrap_summary=bootstrap_summary,
        prepared=prepared,
    )


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
            self._reject_source_selector_for_existing_artifact(intent.selector)
            materialization_request = self._host_materialization_request(
                context,
                operation_scope="startup.direct_serving_artifact.bind",
            )
            load_result = self._load_existing_serving_artifact(
                _DirectServingLoad(
                    selector=intent.selector,
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
            retained_authority = intent.authority
            authority = retained_authority.to_preload_authority()
            expected_member = retained_authority.member
            if self.host is not None:
                placement = self._framework_context(
                    context.framework_config,
                    context.model_config,
                ).placement
                if (
                    placement is not None
                    and placement.member is not None
                    and placement.member != retained_authority.member
                ):
                    raise AuthorityValidationError(
                        "RetainedBindingAuthority.member does not match "
                        "runtime placement",
                        details={
                            "authority_member": repr(retained_authority.member),
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
        self._reject_source_selector_for_existing_artifact(intent.selector)
        decision = self._admit_intent(intent, context, reload=True)
        materialization_request = self._host_materialization_request(
            context,
            operation_scope="runtime_binding.swap",
        )
        result = self._reload_existing_serving_artifact(
            _ServingReload(
                current_state=current_state,
                selector=intent.selector,
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
    def _reject_source_selector_for_existing_artifact(selector: object) -> None:
        if isinstance(selector, SourceSelector):
            raise ServingIntegrationError(
                "ExistingServingArtifact requires a durable serving artifact "
                "selector; local source selectors must use LocalSourceBootstrap"
            )
        if _selector_kind(selector) == "local_path":
            raise ServingIntegrationError(
                "ExistingServingArtifact rejects local_path selectors; use "
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
            view=self._worker_view_from_state(
                state,
                decision=decision,
                bootstrap_summary=result.bootstrap_summary,
            ),
            bootstrap_summary=result.bootstrap_summary,
            prepared=result.prepared,
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
        bootstrap_summary: BootstrapSummary | None = None,
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
            bootstrap_summary=bootstrap_summary,
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
        return serving_placement_from_framework_facts(
            identity_facts=self.host.placement.identity_facts(framework_config),
            admission_facts=self.host.placement.admission_facts(framework_config),
            member_facts=self.host.placement.member_facts(framework_config),
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
        policy = self._runtime_policy(request.policy)
        resolved = self._resolved_artifact(
            resolved_artifact=request.resolved_artifact,
            artifact_ref=request.artifact_ref,
            selector=request.selector,
            expected_tensor_schema_hash=None,
            serving_runtime_policy=policy,
        )
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
        policy = self._runtime_policy_from_manifest(policy, resolved)
        resolved = self._resolved_artifact(
            resolved_artifact=resolved,
            artifact_ref=request.artifact_ref,
            selector=request.selector,
            expected_tensor_schema_hash=tensor_schema_hash,
            serving_runtime_policy=policy,
        )
        context = self._framework_context(
            request.framework_config,
            request.model_config,
        )
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
                state_seed = self._state_seed(
                    resolved,
                    tensor_schema_hash=tensor_schema_hash,
                    execution_diagnostics=binding_result.execution_diagnostics,
                    binding_handle=restored,
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
            state_seed = self._state_seed(
                resolved,
                tensor_schema_hash=tensor_schema_hash,
                execution_diagnostics=binding_result.execution_diagnostics,
                binding_handle=binding_result.binding,
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
        policy = self._runtime_policy(request.policy)
        resolved = self._resolved_artifact(
            resolved_artifact=request.resolved_artifact,
            artifact_ref=request.artifact_ref,
            selector=request.selector,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            serving_runtime_policy=policy,
        )
        materialization = self._reload_materialization_options(
            request,
            resolved,
        )
        binding_result = swap_serving_artifact(
            binding=binding,
            resolved_artifact=resolved,
            serving_runtime_policy=policy,
            options=materialization,
        )
        state_seed = self._state_seed(
            resolved,
            tensor_schema_hash=str(expected_tensor_schema_hash or ""),
            execution_diagnostics=binding_result.execution_diagnostics,
            binding_handle=binding_result.binding,
        )
        if request.model is not None:
            runtime_state = self._materializer().attach_and_finalize(
                model=request.model,
                tensors=binding_result.tensors,
                binding_handle=binding_result.binding,
                context=self._framework_context(
                    request.framework_config,
                    request.model_config,
                ),
                state_seed=state_seed,
                replace_meta_params=False,
                target_device=target_device,
                model_config=request.model_config,
                run_process_after_load=False,
            )
        else:
            runtime_state = RuntimeBindingState(
                binding=binding_result.binding,
                artifact_ref=state_seed.artifact_ref,
                runtime_view=state_seed.runtime_view(),
                ownership_handle=getattr(
                    request.current_state, "ownership_handle", None
                ),
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
                    },
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
                bootstrap_summary=finalized.bootstrap_summary,
                prepared=finalized.prepared,
                current_value=finalized.current_value,
                binding=finalized.binding,
                update_epoch=finalized.update_epoch,
                layout=realized.layout,
                realization_entry_count=realized.realization_entry_count,
                realization=realized.realization,
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
        if request.source_catalog is not None:
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
                    source_artifact_ref=source_artifact_ref,
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
            catalog_artifact_ref = getattr(source_catalog, "source_artifact_ref", None)
            if catalog_artifact_ref is not None and str(catalog_artifact_ref) != str(
                source_artifact_ref
            ):
                raise ServingIntegrationError(
                    "SourceCatalogProvider returned source_artifact_ref "
                    f"{catalog_artifact_ref!r}, expected {source_artifact_ref!r}"
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
            semantic_validation_spec = request.semantic_validation_spec
            if semantic_validation_spec is None and request.run_semantic_validation:
                semantic_validation_spec = getattr(
                    request.recipe,
                    "semantic_validation_spec",
                    None,
                )
            self._materializer().attach_and_finalize(
                model=request.model,
                tensors=_binding_tensors(request.binding),
                binding_handle=request.binding,
                context=self._framework_context(
                    request.framework_config,
                    request.model_config,
                ),
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
            )
            return LocalReadyServingResult(
                model=request.model,
                runtime_state=prepared.runtime_state,
                runtime_view=prepared.runtime_view,
                bootstrap_summary=prepared.bootstrap_summary,
                prepared=prepared.prepared,
                current_value=current_value,
                binding=request.binding,
                update_epoch=request.update_epoch,
            )
        except Exception:
            _close_quietly(request.binding)
            raise

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
    ) -> tuple[str, bytes]:
        return prepare_same_binding_manifest_carrier(
            recipe,
            manifest_tensor_name=manifest_tensor_name,
            representation_contract_hash=representation_contract_hash,
            logical_topology_json_payload=logical_topology_json_payload,
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
        return self.build_local_ready_manifest_carrier(
            recipe=recipe,
            manifest_tensor_name=manifest_tensor_name,
            representation_contract_hash=representation_contract_hash,
            logical_topology_json_payload=logical_topology_json_payload,
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
        return (
            recipe.serving_facts.process_after_load_class
            == FinalizeClass.REPRESENTATION_CHANGING
        )

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
    ) -> RecipeBuildIdentity:
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
        return RecipeBuildIdentity(
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
    def _runtime_policy_from_manifest(policy: Any | None, resolved: Any) -> Any | None:
        if policy is not None:
            return policy
        manifest = getattr(resolved, "manifest", None)
        to_runtime_policy = getattr(manifest, "to_runtime_policy", None)
        if callable(to_runtime_policy):
            return to_runtime_policy()
        return None

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
        selector: Any | None,
        expected_tensor_schema_hash: str | None,
        serving_runtime_policy: Any | None,
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
            if self.resolver is not None and expected_tensor_schema_hash:
                return cross_check_serving_artifact(
                    resolved_artifact,
                    resolver=self.resolver,
                    expected_tensor_schema_hash=expected_tensor_schema_hash,
                    serving_runtime_policy=serving_runtime_policy,
                )
            return resolved_artifact
        resolved_ref = artifact_ref
        if resolved_ref is None and selector is not None:
            resolve_artifact_ref = getattr(selector, "resolve_artifact_ref", None)
            if callable(resolve_artifact_ref):
                resolved_ref = resolve_artifact_ref()
            else:
                resolved_ref = str(selector)
        if not resolved_ref:
            raise ServingIntegrationError(
                "ServingIntegration request requires resolved_artifact, artifact_ref, or selector"
            )
        return resolve_serving_artifact(
            str(resolved_ref),
            resolver=self.resolver,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
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
        binding_handle: Any | None = None,
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
        if execution_diagnostics is not None:
            diagnostics["execution"] = execution_diagnostics
            source_selection = source_selection_projection_from_execution_diagnostics(
                execution_diagnostics
            )
            if source_selection is not None:
                diagnostics["source_selection"] = source_selection
        if serving_build_digest:
            diagnostics["serving_build_digest"] = str(serving_build_digest)
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


_ACTIVE_PUBLICATION_STATES = {"publishing", "published", "retiring"}


def _nonempty_binding_value_ref(
    value: object | None,
) -> BindingValueRefProjection | None:
    projection = BindingValueRefProjection.from_value(value)
    if projection is None:
        return None
    if not (
        projection.binding_id
        and projection.binding_layout_id
        and projection.binding_value_id
    ):
        return None
    return projection


def _binding_value_refs_match(
    expected: BindingValueRefProjection | None,
    actual: BindingValueRefProjection | None,
) -> bool:
    if expected is None or actual is None:
        return True
    return expected.to_dict() == actual.to_dict()


def _publication_generation(attachment: RuntimeAttachment) -> str:
    weight_version = attachment.view.endpoint.weight_version
    binding_value_ref = weight_version.binding_value_ref
    if binding_value_ref is not None:
        payload: object = binding_value_ref.to_dict()
    else:
        payload = {
            "serving_artifact_ref": weight_version.serving_artifact_ref,
            "local_serving_ref": weight_version.local_serving_ref,
            "representation_contract_hash": weight_version.representation_contract_hash,
            "tensor_schema_hash": weight_version.tensor_schema_hash,
            "attachment_id": id(attachment),
        }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _state_publication_binding(state: RuntimeBindingState) -> object | None:
    return state.binding or state.ownership_handle


def _nested_attr(value: object | None, *names: str) -> object | None:
    current = value
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def _first_attr(value: object | None, *names: str) -> object | None:
    if value is None:
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _binding_published_lease_id(binding: object | None) -> str | None:
    return _optional_text(_first_attr(binding, "published_lease_id")) or _optional_text(
        _nested_attr(binding, "_slot", "published_lease_id")
    )


def _binding_published_replica_id(binding: object | None) -> str | None:
    return _optional_text(
        _first_attr(binding, "published_replica_id")
    ) or _optional_text(_nested_attr(binding, "_slot", "published_replica_id"))


def _published_projection_matches_binding(
    *,
    projection: PublishedReplicaProjection,
    attachment: RuntimeAttachment,
    binding: object,
) -> bool:
    lease_id = _binding_published_lease_id(binding)
    if projection.lease_id is not None and projection.lease_id != lease_id:
        return False
    replica_id = _binding_published_replica_id(binding)
    if (
        projection.replica_id is not None
        and replica_id is not None
        and projection.replica_id != replica_id
    ):
        return False
    artifact_ref = _optional_text(getattr(binding, "artifact_id", None))
    if artifact_ref is not None and projection.artifact_ref != artifact_ref:
        return False
    actual_ref = _nonempty_binding_value_ref(getattr(binding, "current_value", None))
    expected_ref = projection.binding_value_ref or (
        attachment.view.endpoint.weight_version.binding_value_ref
    )
    if actual_ref is not None and not _binding_value_refs_match(
        expected_ref, actual_ref
    ):
        return False
    return lease_id is not None or replica_id is not None


def _binding_byte_space_fields(binding: object | None) -> dict[str, str | None]:
    byte_space = getattr(binding, "byte_space", None)
    return {
        "byte_space_kind": (
            _optional_text(getattr(byte_space, "kind", None))
            or _optional_text(getattr(byte_space, "type", None))
        ),
        "byte_space_id": (
            _optional_text(getattr(byte_space, "id", None))
            or _optional_text(getattr(byte_space, "device_id", None))
            or _optional_text(getattr(byte_space, "name", None))
        ),
    }


def _call_operation_wait(
    operation: object,
    *,
    timeout_s: float,
) -> object | None:
    wait = getattr(operation, "wait", None)
    if callable(wait):
        return wait(timeout_s=timeout_s)
    result = getattr(operation, "result", None)
    if callable(result):
        return result(timeout_s=timeout_s)
    return operation


def _published_replica_projection_from_result(
    *,
    attachment: RuntimeAttachment,
    binding: object,
    operation: object | None,
    result: object | None,
    state: str,
    reason: str | None = None,
) -> PublishedReplicaProjection:
    current_value = getattr(binding, "current_value", None)
    binding_value_ref = (
        _nonempty_binding_value_ref(result)
        or _nonempty_binding_value_ref(current_value)
        or attachment.view.endpoint.weight_version.binding_value_ref
    )
    byte_space_fields = _binding_byte_space_fields(binding)
    return PublishedReplicaProjection(
        state=state,
        operation_id=_optional_text(getattr(operation, "operation_id", None)),
        replica_id=(
            _optional_text(_first_attr(result, "replica_id", "published_replica_id"))
            or _binding_published_replica_id(binding)
        ),
        lease_id=(
            _optional_text(_first_attr(result, "lease_id", "published_lease_id"))
            or _binding_published_lease_id(binding)
        ),
        artifact_ref=(
            _optional_text(_first_attr(result, "serving_artifact_id", "artifact_id"))
            or attachment.view.endpoint.weight_version.serving_artifact_ref
        ),
        device_uuid=(
            _optional_text(_first_attr(result, "device_uuid", "device_id"))
            or _optional_text(getattr(binding, "device_uuid", None))
        ),
        owner_pid=_optional_int(_first_attr(result, "owner_pid")) or os.getpid(),
        binding_layout_id=(
            _optional_text(_first_attr(result, "binding_layout_id"))
            or _optional_text(getattr(binding, "binding_layout_id", None))
            or attachment.view.endpoint.weight_version.binding_layout_id
        ),
        binding_value_ref=binding_value_ref,
        generation=_publication_generation(attachment),
        reason=reason,
        byte_space_kind=byte_space_fields["byte_space_kind"],
        byte_space_id=byte_space_fields["byte_space_id"],
    )


def _attachment_with_published_replica(
    attachment: RuntimeAttachment,
    projection: PublishedReplicaProjection,
) -> RuntimeAttachment:
    weight_version = replace(
        attachment.view.endpoint.weight_version,
        published_replica=projection,
    )
    endpoint = replace(attachment.view.endpoint, weight_version=weight_version)
    view = replace(attachment.view, endpoint=endpoint)
    return replace(attachment, view=view)


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
        if not isinstance(current_attachment, RuntimeAttachment):
            raise ReplicaPublicationError(
                "publish_current_replica requires a RuntimeAttachment"
            )
        publication_policy = self._replica_publication_policy(policy)
        if publication_policy.mode == "disabled":
            return current_attachment
        profile_start = time.perf_counter()
        self._ensure_runtime_initialized()
        weight_version = current_attachment.view.endpoint.weight_version
        if not weight_version.serving_artifact_ref:
            raise ReplicaPublicationError(
                "Replica publication requires an artifact-backed serving attachment"
            )
        binding = _state_publication_binding(current_attachment.state)
        if binding is None:
            raise ReplicaPublicationError(
                "Runtime attachment has no publication-capable binding"
            )
        actual_artifact_ref = _optional_text(getattr(binding, "artifact_id", None))
        if (
            actual_artifact_ref is not None
            and actual_artifact_ref != weight_version.serving_artifact_ref
        ):
            raise ReplicaPublicationError(
                "Runtime attachment publication artifact does not match "
                "the current weight version",
                details={
                    "expected_artifact_ref": weight_version.serving_artifact_ref,
                    "actual_artifact_ref": actual_artifact_ref,
                },
            )
        published = weight_version.published_replica
        if published is not None and published.state in _ACTIVE_PUBLICATION_STATES:
            if published.state == "published" and (
                _published_projection_matches_binding(
                    projection=published,
                    attachment=current_attachment,
                    binding=binding,
                )
            ):
                self._emit_publication_profile(
                    "runtime_publication.publish.replay",
                    attachment=current_attachment,
                    duration_s=time.perf_counter() - profile_start,
                    published_replica=published,
                )
                return current_attachment
            raise ReplicaPublicationError(
                "Runtime attachment active published replica does not match "
                "the current publication binding",
                details={
                    "published_replica_state": published.state,
                    "replica_id": published.replica_id,
                    "lease_id": published.lease_id,
                },
            )
        operation = None
        result = None
        try:
            publish_operation = getattr(binding, "publish_replica_operation", None)
            if callable(publish_operation):
                operation = publish_operation()
                result = _call_operation_wait(
                    operation,
                    timeout_s=publication_policy.timeout_s,
                )
            else:
                publish = getattr(binding, "publish_replica", None)
                if not callable(publish):
                    raise ReplicaPublicationError(
                        "Runtime binding does not expose publish_replica_operation"
                    )
                result = publish()
        except ReplicaPublicationError:
            self._emit_publication_profile(
                "runtime_publication.publish.failed",
                attachment=current_attachment,
                duration_s=time.perf_counter() - profile_start,
                error="replica_publication",
            )
            raise
        except Exception as exc:
            wrapped = ReplicaPublicationError(
                "Runtime replica publication failed",
                details={"reason": str(exc)},
            )
            self._emit_publication_profile(
                "runtime_publication.publish.failed",
                attachment=current_attachment,
                duration_s=time.perf_counter() - profile_start,
                error=str(exc),
            )
            raise wrapped from exc
        actual_ref = _nonempty_binding_value_ref(result) or _nonempty_binding_value_ref(
            getattr(binding, "current_value", None)
        )
        if not _binding_value_refs_match(weight_version.binding_value_ref, actual_ref):
            raise ReplicaPublicationError(
                "Runtime attachment publication result is stale",
                details={
                    "expected_binding_value_ref": None
                    if weight_version.binding_value_ref is None
                    else weight_version.binding_value_ref.to_dict(),
                    "actual_binding_value_ref": None
                    if actual_ref is None
                    else actual_ref.to_dict(),
                },
            )
        projection = _published_replica_projection_from_result(
            attachment=current_attachment,
            binding=binding,
            operation=operation,
            result=result,
            state="published",
        )
        self._emit_publication_profile(
            "runtime_publication.publish.done",
            attachment=current_attachment,
            duration_s=time.perf_counter() - profile_start,
            published_replica=projection,
        )
        return _attachment_with_published_replica(current_attachment, projection)

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
        if not isinstance(current_attachment, RuntimeAttachment):
            raise ReplicaPublicationError(
                "retire_current_replica requires a RuntimeAttachment"
            )
        weight_version = current_attachment.view.endpoint.weight_version
        published = weight_version.published_replica
        binding = _state_publication_binding(current_attachment.state)
        lease_id = _binding_published_lease_id(binding)
        active_projection = (
            published is not None and published.state in _ACTIVE_PUBLICATION_STATES
        )
        if not active_projection and lease_id is None:
            return current_attachment
        profile_start = time.perf_counter()
        self._ensure_runtime_initialized()
        if binding is None:
            raise ReplicaPublicationError(
                "Runtime attachment has no publication-capable binding"
            )
        retire = getattr(binding, "retire", None)
        if not callable(retire):
            raise ReplicaPublicationError("Runtime binding does not expose retire")
        drain_timeout = (
            drain_timeout_s
            if drain_timeout_s is not None
            else self.serving_config.replica_publication.drain_timeout_s
        )
        try:
            retire(drain_timeout_s=drain_timeout)
        except Exception as exc:
            wrapped = ReplicaPublicationError(
                "Runtime replica retirement failed",
                details={"reason": str(exc)},
            )
            self._emit_publication_profile(
                "runtime_publication.retire.failed",
                attachment=current_attachment,
                duration_s=time.perf_counter() - profile_start,
                published_replica=published,
                reason=reason,
                error=str(exc),
            )
            raise wrapped from exc
        projection = (
            published
            if published is not None
            else _published_replica_projection_from_result(
                attachment=current_attachment,
                binding=binding,
                operation=None,
                result=getattr(binding, "current_value", None),
                state="retired",
                reason=reason,
            )
        )
        projection = replace(projection, state="retired", reason=reason)
        self._emit_publication_profile(
            "runtime_publication.retire.done",
            attachment=current_attachment,
            duration_s=time.perf_counter() - profile_start,
            published_replica=projection,
            reason=reason,
        )
        return _attachment_with_published_replica(current_attachment, projection)

    def _emit_publication_profile(
        self,
        event: str,
        *,
        attachment: RuntimeAttachment,
        duration_s: float,
        published_replica: PublishedReplicaProjection | None = None,
        reason: str | None = None,
        error: str | None = None,
    ) -> None:
        sink = self.profile_sink
        if not callable(sink):
            return
        payload: dict[str, object] = {
            "event": event,
            "duration_s": duration_s,
            "serving_artifact_ref": attachment.view.endpoint.weight_version.serving_artifact_ref,
            "generation": _publication_generation(attachment),
        }
        if published_replica is not None:
            payload["published_replica_state"] = published_replica.state
            if published_replica.replica_id is not None:
                payload["replica_id"] = published_replica.replica_id
            if published_replica.lease_id is not None:
                payload["lease_id"] = published_replica.lease_id
        if reason is not None:
            payload["reason"] = reason
        if error is not None:
            payload["error"] = error
        sink(payload)

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
        selector: ServingArtifactSelector,
        policy: ServingPolicy | None,
        context: RequestContext,
        model: object | None = None,
        contract_identity: str | None = None,
    ) -> RuntimeAttachment:
        self._reject_local_reload_selector(selector)
        if not isinstance(selector, ServingArtifactSelector):
            raise ConfigConflictError(
                "TensorCast serving reload requires a ServingArtifactSelector"
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
            ExistingServingArtifact(selector=selector, policy=policy),
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
        published = current_attachment.view.endpoint.weight_version.published_replica
        if published is None or published.state not in _ACTIVE_PUBLICATION_STATES:
            return
        raise ReplicaPublicationError(
            "TensorCast serving reload requires retiring the active published "
            "replica before swap",
            operation="reload",
            details={
                "published_replica_state": published.state,
                "replica_id": published.replica_id,
                "lease_id": published.lease_id,
            },
        )

    def _plan_start_intent(self, context: RequestContext) -> ServingIntent:
        config = self.serving_config
        external = config.preload.mode == "external"
        selector = config.serving.selector
        has_selector = selector is not None
        bootstrap_mode = config.bootstrap.mode
        source_selector = self._source_selector_from_context(context)
        has_local_source = source_selector is not None

        if external and has_selector:
            raise ConfigConflictError(
                "TensorCast serving config cannot request both external "
                "preload and durable serving selector execution"
            )
        if bootstrap_mode == "required" and (external or has_selector):
            raise ConfigConflictError(
                "TensorCast bootstrap.mode='required' is mutually exclusive "
                "with external preload and durable serving selector execution"
            )
        if bootstrap_mode == "disabled" and not (external or has_selector):
            raise ConfigConflictError(
                "TensorCast bootstrap.mode='disabled' requires external "
                "preload authority or durable serving selector"
            )
        if external:
            expected_member = None
            if self.host is not None:
                placement = self._framework_context(
                    context.framework_config,
                    context.model_config,
                ).placement
                if placement is not None:
                    expected_member = placement.member
            authority = parse_external_preload_authority(
                config,
                expected_member=expected_member,
            )
            return RetainedBindingAcquire(
                RetainedBindingAuthority.from_preload_authority(authority)
            )
        if has_selector:
            return ExistingServingArtifact(
                selector=selector,
                policy=config.serving.policy,
            )
        if bootstrap_mode in {"auto", "required"} and has_local_source:
            return LocalSourceBootstrap(
                source_selector=source_selector,
                bootstrap_policy=config.bootstrap,
            )
        raise ConfigConflictError(
            "TensorCast serving config did not resolve to external preload, "
            "durable serving selector, or local source bootstrap"
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
    def _reject_local_reload_selector(selector: object) -> None:
        if (
            isinstance(selector, SourceSelector)
            or _selector_kind(selector) == "local_path"
        ):
            raise ConfigConflictError(
                "TensorCast serving reload requires a durable serving "
                "selector, not a local source selector"
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
    serving_runtime_policy: Any | None,
    options: Any | None,
) -> RuntimeBindingResult:
    """Swap an existing runtime binding to another serving artifact."""

    operation_result = tc_binding_runtime.swap_serving_artifact(
        binding=binding,
        resolved_artifact=resolved_artifact,
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
    authority: tc_preload.ParsedExternalPreloadAuthority | None = None,
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

    with tc_preload.acquire_retained_serving_binding(
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


def external_preload_mode(*args: Any, **kwargs: Any) -> str:
    return tc_preload.external_preload_mode(*args, **kwargs)


def external_preload_trusted_reservation_bytes(*args: Any, **kwargs: Any) -> int:
    return tc_preload.external_preload_trusted_reservation_bytes(*args, **kwargs)


def external_preload_extra_from_prefetched_binding(*args: Any, **kwargs: Any) -> Any:
    return tc_preload.external_preload_extra_from_prefetched_binding(*args, **kwargs)


def parse_external_preload_authority(*args: Any, **kwargs: Any) -> Any:
    return tc_preload.parse_external_preload_authority(*args, **kwargs)


__all__ = [
    "ArtifactError",
    "BindingUpdateEpoch",
    "BindingValueRef",
    "BindingFinalizeMaterializationResult",
    "BootstrapSummary",
    "BuilderMode",
    "CapabilityMissingError",
    "CollectiveHost",
    "CollectivePolicy",
    "ConfigConflictError",
    "DEFAULT_RUNTIME_PROFILE",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionRejectedError",
    "AdmissionRequest",
    "AuthorityValidationError",
    "BootstrapEndpointProjection",
    "BootstrapPolicy",
    "FamilyReadiness",
    "FrameworkHost",
    "FrameworkIdentity",
    "FrameworkIntegrationContext",
    "FinalizeClass",
    "FinalizeHookHost",
    "FinalizePhase",
    "FinalizePolicy",
    "AttachFinalizeError",
    "BindingValueRefProjection",
    "DefaultAdmissionPolicy",
    "ExistingServingArtifact",
    "IntegrationHost",
    "LocalSourceBootstrap",
    "ManifestPolicy",
    "MaterializationDiagnosticsProjection",
    "MaterializationExecutionFacts",
    "MaterializationPolicy",
    "NativeLoadHost",
    "ObservabilitySink",
    "PreparedServingArtifact",
    "PublishedModelVersion",
    "LocalReadyManifestCarrierResult",
    "LocalReadyBindingContract",
    "LocalReadyMaterializationIdentity",
    "LocalReadyServingResult",
    "ManifestMismatchError",
    "OwnershipTransferError",
    "PlacementAdmissionFacts",
    "PlacementHost",
    "PlacementIdentityFacts",
    "PlacementMemberFacts",
    "PolicyMismatchError",
    "PlacementAdmissionError",
    "PublishedReplicaProjection",
    "PublicationRequiredError",
    "ReplicaPublicationError",
    "ReplicaPublicationPolicy",
    "RecipeCachePolicy",
    "RecipeTraceHost",
    "ReloadRequestProjection",
    "ReloadResponseProjection",
    "ResolvedServingArtifact",
    "RequestContext",
    "RetainedBindingAcquire",
    "RetainedBindingAuthority",
    "RecipeBuildIdentity",
    "RecipeBuildCacheConfig",
    "RecipeBuildRunResult",
    "RecipeCacheLookupResult",
    "RecipeCacheWriteResult",
    "RecipeBuildResult",
    "RecipeBuildSessionRequest",
    "RecipeBuildSession",
    "COMPILED_RECIPE_MEMORY_CACHE",
    "TRACE_PLAN_MEMORY_CACHE",
    "RecipeCompileIdentity",
    "RecipeCompileInputs",
    "RecipePublicationContext",
    "ParsedExternalPreloadAuthority",
    "RestoreBindingError",
    "RestoredRetainedBinding",
    "RetainedBindingResult",
    "RuntimeBindingResult",
    "RuntimeBindingMaterialization",
    "RuntimeBindingState",
    "RuntimeStateSeed",
    "RuntimeBindingView",
    "RuntimeAttachment",
    "RuntimeConfig",
    "RuntimeEndpointProjection",
    "RuntimeProfile",
    "RuntimeSwapError",
    "RuntimeTensorView",
    "RuntimeWorkerView",
    "SERVING_MANIFEST_TENSOR_NAME",
    "SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4",
    "SchemaMismatchError",
    "SelectorResolutionError",
    "ServingArtifactSelector",
    "ServingIntegrationError",
    "ServingIntegrationNotImplementedError",
    "ServingArtifactResolver",
    "ServingArtifactManifest",
    "ServingBindingState",
    "ServingBindingMemberRef",
    "ServingConfig",
    "ServingIntegration",
    "ServingIntent",
    "ServingLoadResult",
    "ServingPlacement",
    "ServingPolicy",
    "ServingReloadResult",
    "ServingRuntimeSession",
    "SourceSelectionProjection",
    "ServingRuntimePolicy",
    "ServingSupportLevel",
    "ServingTopologyRef",
    "SourceBoundContractProjection",
    "SourceSelector",
    "SourceSubject",
    "SourceSubjectCoordinator",
    "SourceSubjectError",
    "SourceProviderError",
    "SourceBoundContractState",
    "SourceCatalog",
    "SourceCatalogPolicy",
    "SourceCatalogProvider",
    "SourceCatalogRequest",
    "SourceDownloadPolicy",
    "SourceHost",
    "TensorCastEvent",
    "TensorCastServingRuntimeError",
    "TensorSchemaEntry",
    "TensorSurfaceHost",
    "TensorcastLogicalTopology",
    "TensorcastSemanticValidationSpec",
    "TensorcastServingFacts",
    "TorchTensorHost",
    "TracePlan",
    "WeightVersionProjection",
    "RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION",
    "PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION",
    "RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION",
    "RETAINED_BINDING_AUTHORITY_SCHEMA_VERSION",
    "SERVING_ARTIFACT_SELECTOR_SCHEMA_VERSION",
    "SERVING_POLICY_SCHEMA_VERSION",
    "WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION",
    "PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION",
    "PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION",
    "SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION",
    "RECIPE_CACHE_POLICY_SCHEMA_VERSION",
    "SOURCE_CATALOG_REQUEST_SCHEMA_VERSION",
    "SOURCE_CATALOG_SCHEMA_VERSION",
    "SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION",
    "allocate_tensors_from_schema",
    "apply_copy_plan",
    "bind_serving_artifact",
    "binding_value_verification_state_name",
    "binding_value_ref_from_current_value",
    "build_local_ready_prepared_artifact",
    "build_pure_transform_build_intent",
    "build_collective_group_id",
    "build_materialization_execution_context",
    "canonical_index_from_recipe",
    "collect_runtime_tensor_schema",
    "collect_serving_tensors_from_model",
    "compile_recipe_from_inputs",
    "complete_pure_transform_publication",
    "complete_pure_transform_recipe_publication_from_recipe",
    "compiled_recipe_realization_plan_count",
    "compute_recipe_build_cache_key",
    "compute_recipe_compile_key",
    "compute_recipe_compile_key_from_inputs",
    "compute_runtime_representation_contract_hash",
    "compute_runtime_tensor_schema_hash",
    "compute_serving_binding_tensor_schema_hash",
    "compute_serving_tensor_schema_hash",
    "compute_trace_build_cache_key",
    "cross_check_serving_artifact",
    "dump_trace_plan_debug",
    "evaluate_semantic_validation_spec",
    "external_preload_extra_from_prefetched_binding",
    "external_preload_mode",
    "external_preload_trusted_reservation_bytes",
    "freeze_local_ready_binding",
    "is_public_disk_source_subject",
    "is_reserved_serving_tensor_name",
    "load_compiled_recipe_cache",
    "load_source_tensors_for_recipe",
    "load_trace_plan_cache",
    "logical_topology_json_from_recipe",
    "materialized_tensor_schema",
    "materialize_binding_finalize_serving_tensors",
    "materialize_pure_transform_serving_tensors",
    "materialize_recipe_copy_plan_tensors",
    "merge_serving_reload_extra_config",
    "normalize_serving_reload_request_payload",
    "prepare_local_ready_serving",
    "prepare_same_binding_manifest_carrier",
    "parse_external_preload_authority",
    "publication_context_from_recipe",
    "read_serving_artifact_manifest",
    "read_source_bound_contract_state",
    "recipe_build_cache_path",
    "resolve_serving_artifact",
    "resolve_runtime_config_profile",
    "resolve_source_subject",
    "resolve_source_artifact_ref",
    "restore_prepared_local_ready_binding",
    "restore_retained_binding",
    "source_selection_projection_from_execution_diagnostics",
    "source_selection_projection_from_materialization_diagnostics",
    "source_subject_broadcast_payload",
    "source_subject_from_broadcast_payload",
    "source_subject_slice_count",
    "source_bound_contract_profile_fields",
    "serving_binding_state_from_runtime_view",
    "serving_placement_from_framework_facts",
    "semantic_placement_digest",
    "run_binding_finalize_semantic_validation",
    "source_catalog_from_selected_safetensors",
    "stable_recipe_build_hash",
    "swap_serving_artifact",
    "tensorcast_view_slice_count",
    "tensorcast_view_slices_from_trace_plan",
    "trace_build_cache_path",
    "validate_dst_coverage",
    "validate_binding_finalize_tensor_schema",
    "validate_recipe_for_builder_mode",
    "validate_source_tensor_names",
    "validate_tensor_schema_against_tensors",
    "write_compiled_recipe_cache",
    "write_trace_plan_cache",
]
