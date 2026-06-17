#  Copyright (c) 2026, TensorCast Team.
"""Artifact-runtime host capability protocols for framework integrations.

This module is intentionally lightweight: importing it must not import the
runtime lifecycle implementation, builder stack, binding runtime, or store API.
Framework integrations should use these DTOs and protocols to describe facts
and capabilities; TensorCast core owns the lifecycle that consumes them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

from tensorcast.types import (
    SERVING_MANIFEST_TENSOR_NAME,
    RuntimeBindingMemberRef,
    RuntimeTopologyRef,
)

PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION = 1
PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION = 1
SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION = 1
RECIPE_CACHE_POLICY_SCHEMA_VERSION = 1
SOURCE_CATALOG_REQUEST_SCHEMA_VERSION = 1
SOURCE_CATALOG_SCHEMA_VERSION = 1


class RuntimeTensorView(BaseModel):
    """Framework-neutral tensor identity view without live tensor payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int = 0
    element_size: int | None = None


class RuntimePlacement(BaseModel):
    """Stable runtime placement identity shared with framework integrations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topology: RuntimeTopologyRef
    member: RuntimeBindingMemberRef
    framework_payload: dict[str, Any]
    identity_payload: dict[str, Any]

    def stable_identity_payload(self) -> dict[str, Any]:
        return {
            "topology": self.topology.model_dump(mode="python"),
            "member": self.member.model_dump(mode="python"),
            "framework_payload": self.framework_payload,
            "identity_payload": self.identity_payload,
        }


@dataclass(frozen=True)
class FrameworkIdentity:
    """Stable framework identity facts owned by a framework host."""

    framework_name: str
    framework_version: str
    adapter_version: str
    serving_abi_version: str

    def stable_identity_payload(self) -> dict[str, str]:
        return {
            "framework_name": str(self.framework_name),
            "framework_version": str(self.framework_version),
            "adapter_version": str(self.adapter_version),
            "serving_abi_version": str(self.serving_abi_version),
        }


class FrameworkHost(Protocol):
    """Framework-owned model construction and semantic preflight facts."""

    def identity(self, model_config: object) -> FrameworkIdentity: ...

    def prepare_model_construction(
        self,
        framework_config: object,
        model_config: object,
    ) -> None: ...

    def build_meta_model(
        self,
        framework_config: object,
        model_config: object,
    ) -> object: ...

    def build_runtime_model(
        self,
        framework_config: object,
        model_config: object,
        target_device: object,
    ) -> object: ...

    def assert_model_ready_for_runtime_binding(
        self,
        model: object,
        *,
        context: str,
    ) -> None: ...

    def semantic_probes(self, model: object, model_config: object) -> object: ...


class NativeLoadHost(Protocol):
    """Optional framework-native checkpoint/source loading capability."""

    def native_load_weights(self, model: object, weights: object) -> None: ...


class RecipeTraceHost(Protocol):
    """Optional recipe tracing capability required on recipe cache miss."""

    def trace_model_load(
        self,
        model: object,
        ordered_names: Sequence[str],
        meta_by_name: Mapping[str, object],
        *,
        debug_dump_trace: bool = False,
    ) -> object: ...

    def cleanup_after_recipe_build(
        self,
        model: object,
        model_config: object,
        *,
        framework_config: object | None = None,
    ) -> None: ...


FinalizePhase = str


@dataclass(frozen=True)
class FinalizePolicy:
    """Framework finalize hooks selected for a materialization phase."""

    run_process_after_load: bool = True
    run_post_bind_finalize: bool = True


class FinalizeHookHost(Protocol):
    """Optional framework-specific finalize hook capability."""

    def finalize_policy(
        self,
        model: object,
        model_config: object,
    ) -> FinalizePolicy: ...

    def run_finalize_hook(
        self,
        phase: FinalizePhase,
        model: object,
        model_config: object,
        target_device: object,
    ) -> None: ...


class TensorSurfaceHost(Protocol):
    """Tensor-level attach/schema/invariant contract for materialization."""

    def runtime_only_tensor_names(self, model: object) -> tuple[str, ...]: ...

    def align_runtime_tensor_names(
        self,
        model: object,
        expected_names: Sequence[str],
    ) -> int: ...

    def collect_runtime_tensors(
        self,
        model: object,
        *,
        remove_duplicate: bool = False,
    ) -> Mapping[str, object]: ...

    def collect_runtime_tensor_view(
        self,
        tensors: Mapping[str, object],
    ) -> tuple[RuntimeTensorView, ...]: ...

    def compute_runtime_tensor_schema_hash(
        self,
        tensors: Mapping[str, object],
        *,
        remove_duplicate: bool = False,
    ) -> str: ...

    def attach_bound_tensors(
        self,
        model: object,
        tensors: Mapping[str, object],
        *,
        replace_meta_params: bool,
    ) -> object: ...

    def allocate_runtime_only_tensors(
        self,
        model: object,
        target_device: object,
    ) -> Mapping[str, object]: ...

    def rehydrate_runtime_only_tensors(
        self,
        model: object,
        allocated: Mapping[str, object],
        target_device: object,
    ) -> Mapping[str, object]: ...

    def snapshot_tensor_invariants(
        self,
        tensors: Mapping[str, object],
    ) -> object: ...

    def validate_tensor_invariants(
        self,
        before: object,
        after: Mapping[str, object],
    ) -> None: ...


class TorchTensorHost:
    """Default PyTorch tensor surface implementation."""

    def runtime_only_tensor_names(self, model: object) -> tuple[str, ...]:
        del model
        return ()

    def align_runtime_tensor_names(
        self,
        model: object,
        expected_names: Sequence[str],
    ) -> int:
        from tensorcast.pytorch import module_binding as tc_module_binding

        excluded: set[str] = set()

        def set_excluded_names(_model: object, names: Sequence[str]) -> None:
            excluded.update(str(name) for name in names)

        return tc_module_binding.align_runtime_binding_exclude_names(
            cast(Any, model),
            expected_names,
            set_excluded_names,
            fail_on_missing=True,
        )

    def collect_runtime_tensors(
        self,
        model: object,
        *,
        remove_duplicate: bool = False,
    ) -> Mapping[str, object]:
        from tensorcast.pytorch import module_binding as tc_module_binding

        return tc_module_binding.collect_module_tensors(
            cast(Any, model),
            exclude_names=self.runtime_only_tensor_names(model),
            reject_reserved_tensor_names=True,
            remove_duplicate=remove_duplicate,
        )

    def collect_runtime_tensor_view(
        self,
        tensors: Mapping[str, object],
    ) -> tuple[RuntimeTensorView, ...]:
        import tensorcast.artifact_runtime.contract as tc_contract

        schema = tc_contract.collect_runtime_tensor_schema(
            cast(Any, tensors), remove_duplicate=False
        )
        return tuple(
            RuntimeTensorView(
                name=entry.name,
                dtype=entry.dtype,
                shape=entry.shape,
                stride=entry.stride,
                storage_offset=entry.storage_offset,
                element_size=entry.element_size,
            )
            for entry in schema
        )

    def compute_runtime_tensor_schema_hash(
        self,
        tensors: Mapping[str, object],
        *,
        remove_duplicate: bool = False,
    ) -> str:
        from tensorcast.pytorch import module_binding as tc_module_binding

        return tc_module_binding.compute_runtime_tensor_schema_hash(
            cast(Any, tensors),
            remove_duplicate=remove_duplicate,
        )

    def attach_bound_tensors(
        self,
        model: object,
        tensors: Mapping[str, object],
        *,
        replace_meta_params: bool,
    ) -> object:
        from tensorcast.pytorch import module_binding as tc_module_binding

        return tc_module_binding.attach_tensors_to_module(
            cast(Any, model),
            cast(Any, tensors),
            replace_meta_params=replace_meta_params,
            skip_reserved_tensor_names=True,
            fail_on_missing=False,
            fail_on_unexpected=True,
        )

    def allocate_runtime_only_tensors(
        self,
        model: object,
        target_device: Any,
    ) -> Mapping[str, object]:
        import torch

        from tensorcast.pytorch import module_binding as tc_module_binding

        device = torch.device(target_device)
        allocated = tc_module_binding.allocate_unbound_module_tensors(
            cast(Any, model),
            self.runtime_only_tensor_names(model),
            target_device=device,
        )
        allocated_objects: dict[str, object] = dict(allocated)
        allocated_objects.update(
            self.rehydrate_runtime_only_tensors(model, allocated_objects, device)
        )
        return allocated_objects

    def rehydrate_runtime_only_tensors(
        self,
        model: object,
        allocated: Mapping[str, object],
        target_device: Any,
    ) -> Mapping[str, object]:
        del model, allocated, target_device
        return {}

    def snapshot_tensor_invariants(
        self,
        tensors: Mapping[str, object],
    ) -> object:
        from tensorcast.pytorch import module_binding as tc_module_binding

        return tc_module_binding.snapshot_tensor_invariants(cast(Any, tensors))

    def validate_tensor_invariants(
        self,
        before: object,
        after: Mapping[str, object],
    ) -> None:
        from tensorcast.pytorch import module_binding as tc_module_binding

        tc_module_binding.validate_tensor_invariants(
            cast(Any, before), cast(Any, after)
        )


@dataclass(frozen=True)
class PlacementIdentityFacts:
    tensor_parallel_rank: int
    tensor_parallel_size: int
    pipeline_parallel_rank: int
    pipeline_parallel_size: int
    data_parallel_rank: int
    data_parallel_size: int
    placement_identity_schema_version: int = PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION


@dataclass(frozen=True)
class PlacementAdmissionFacts:
    placement_admission_schema_version: int = PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION
    expert_parallel_enabled: bool = False
    expert_parallel_rank: int | None = None
    expert_parallel_size: int | None = None
    expert_parallel_group_ranks: tuple[int, ...] = ()
    expert_placement_strategy: str | None = None
    expert_redundant_count: int = 0
    expert_mapping_digest: str | None = None
    expert_mapping_dynamic: bool = False
    eplb_enabled: bool = False
    eplb_physical_to_logical_digest: str | None = None
    dynamic_expert_remap_supported: bool = False
    semantic_placement_digests: Mapping[str, str] = field(default_factory=dict)

    def requires_framework_semantic_proof(self) -> bool:
        return bool(
            self.expert_parallel_enabled
            or self.expert_mapping_dynamic
            or self.eplb_enabled
            or self.eplb_physical_to_logical_digest
        )

    def missing_framework_semantic_proofs(self) -> tuple[str, ...]:
        missing: list[str] = []
        semantic_digests = dict(self.semantic_placement_digests or {})
        if (self.expert_parallel_enabled or self.expert_mapping_dynamic) and not (
            self.expert_mapping_digest or semantic_digests.get("expert_mapping")
        ):
            missing.append("expert_mapping")
        if self.eplb_enabled and not (
            self.eplb_physical_to_logical_digest
            or semantic_digests.get("eplb_physical_to_logical")
        ):
            missing.append("eplb_physical_to_logical")
        return tuple(missing)


@dataclass(frozen=True)
class PlacementMemberFacts:
    runtime_rank: int
    runtime_world_size: int
    member_id: str | None = None
    member_index: int | None = None
    member_count: int | None = None
    group_id_hint: str | None = None


@dataclass(frozen=True)
class MaterializationExecutionFacts:
    collective_rank: int | None = None
    collective_world_size: int | None = None
    same_node_tensor_parallel: bool | None = None
    tensor_parallel_ranks: tuple[int, ...] = ()
    collective_context_unavailable: bool = False


class PlacementHost(Protocol):
    """Framework-owned placement fact extraction."""

    def identity_facts(
        self,
        framework_config: object,
    ) -> PlacementIdentityFacts: ...

    def admission_facts(
        self,
        framework_config: object,
    ) -> PlacementAdmissionFacts: ...

    def member_facts(self, framework_config: object) -> PlacementMemberFacts: ...

    def execution_facts(
        self,
        framework_config: object,
    ) -> MaterializationExecutionFacts: ...


def runtime_placement_from_framework_facts(
    *,
    identity_facts: PlacementIdentityFacts,
    admission_facts: PlacementAdmissionFacts | None = None,
    member_facts: PlacementMemberFacts,
    framework_payload: Mapping[str, object] | None = None,
    identity_payload: Mapping[str, object] | None = None,
) -> RuntimePlacement:
    """Build core-owned runtime placement identity from host facts."""

    admission_facts = admission_facts or PlacementAdmissionFacts()
    placement_identity_payload = _stable_payload(
        {
            "placement_identity_schema_version": identity_facts.placement_identity_schema_version,
            "placement_admission_schema_version": admission_facts.placement_admission_schema_version,
            "tensor_parallel_rank": identity_facts.tensor_parallel_rank,
            "tensor_parallel_size": identity_facts.tensor_parallel_size,
            "pipeline_parallel_rank": identity_facts.pipeline_parallel_rank,
            "pipeline_parallel_size": identity_facts.pipeline_parallel_size,
            "data_parallel_rank": identity_facts.data_parallel_rank,
            "data_parallel_size": identity_facts.data_parallel_size,
            "expert_parallel_enabled": admission_facts.expert_parallel_enabled,
            "expert_parallel_rank": admission_facts.expert_parallel_rank,
            "expert_parallel_size": admission_facts.expert_parallel_size,
            "expert_parallel_group_ranks": admission_facts.expert_parallel_group_ranks,
            "expert_placement_strategy": admission_facts.expert_placement_strategy,
            "expert_redundant_count": admission_facts.expert_redundant_count,
            "expert_mapping_digest": admission_facts.expert_mapping_digest,
            "expert_mapping_dynamic": admission_facts.expert_mapping_dynamic,
            "eplb_enabled": admission_facts.eplb_enabled,
            "eplb_physical_to_logical_digest": admission_facts.eplb_physical_to_logical_digest,
            "semantic_placement_digests": dict(
                admission_facts.semantic_placement_digests
            ),
        }
    )
    topology_digest = _stable_digest(placement_identity_payload)
    member_count = member_facts.member_count
    if member_count is None:
        member_count = member_facts.runtime_world_size
    member_index = member_facts.member_index
    if member_index is None:
        member_index = member_facts.runtime_rank
    group_id = member_facts.group_id_hint
    if group_id is None:
        group_id = f"tensorcast:placement:{topology_digest[:16]}"
    member_id = member_facts.member_id
    if member_id is None:
        member_id = f"rank{member_facts.runtime_rank}"
    resolved_framework_payload = dict(
        framework_payload
        or {
            "family": "framework_parallelism",
            "version": "v1",
            "dimensions": [
                {
                    "name": "data_parallel",
                    "rank": identity_facts.data_parallel_rank,
                    "size": identity_facts.data_parallel_size,
                },
                {
                    "name": "pipeline_parallel",
                    "rank": identity_facts.pipeline_parallel_rank,
                    "size": identity_facts.pipeline_parallel_size,
                },
                {
                    "name": "tensor_parallel",
                    "rank": identity_facts.tensor_parallel_rank,
                    "size": identity_facts.tensor_parallel_size,
                },
            ],
            "expert_parallel_enabled": admission_facts.expert_parallel_enabled,
        }
    )
    resolved_identity_payload = dict(
        identity_payload or cast(Mapping[str, object], placement_identity_payload)
    )
    return RuntimePlacement(
        topology=RuntimeTopologyRef(
            schema_topology_digest=topology_digest,
            logical_topology_ref=(f"tensorcast://placement/{topology_digest[:16]}"),
        ),
        member=RuntimeBindingMemberRef(
            member_id=str(member_id),
            member_index=int(member_index),
            member_count=int(member_count),
            group_id=str(group_id),
        ),
        framework_payload=resolved_framework_payload,
        identity_payload=resolved_identity_payload,
    )


@dataclass(frozen=True)
class RuntimeConfig:
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializationPolicy:
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceBoundContractProfile:
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestPolicy:
    manifest_tensor_name: str = SERVING_MANIFEST_TENSOR_NAME
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceCatalogPolicy:
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeProfile:
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    materialization_policy: MaterializationPolicy = field(
        default_factory=MaterializationPolicy
    )
    source_bound_contract: SourceBoundContractProfile = field(
        default_factory=SourceBoundContractProfile
    )
    manifest_policy: ManifestPolicy = field(default_factory=ManifestPolicy)
    source_catalog_policy: SourceCatalogPolicy | None = None

    @classmethod
    def from_config(cls, runtime_config: object) -> "RuntimeProfile":
        return cls.from_runtime_config(runtime_config)

    @classmethod
    def from_runtime_config(cls, runtime_config: object) -> "RuntimeProfile":
        return cls(
            runtime_config=RuntimeConfig(
                _mapping_from_object(getattr(runtime_config, "runtime", None))
            ),
            materialization_policy=MaterializationPolicy(
                _mapping_from_object(getattr(runtime_config, "materialization", None))
            ),
            source_bound_contract=SourceBoundContractProfile(
                _mapping_from_object(
                    getattr(runtime_config, "source_bound_contract", None)
                )
            ),
            manifest_policy=ManifestPolicy(),
            source_catalog_policy=SourceCatalogPolicy(
                _mapping_from_object(getattr(runtime_config, "source_catalog", None))
            ),
        )


@dataclass(frozen=True)
class TensorCastEvent:
    name: str
    payload: Mapping[str, object] = field(default_factory=dict)


class ObservabilitySink(Protocol):
    def emit(self, event: TensorCastEvent) -> None: ...


@dataclass(frozen=True)
class SourceDownloadPolicy:
    fields: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION


@dataclass(frozen=True)
class RecipeCachePolicy:
    fields: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = RECIPE_CACHE_POLICY_SCHEMA_VERSION


@dataclass(frozen=True)
class SourceSelector:
    """Core-owned source selector before it resolves to a durable subject."""

    kind: str
    value: Any

    @classmethod
    def local_path(cls, path: str) -> "SourceSelector":
        return cls(kind="local_path", value=str(path))


@dataclass(frozen=True)
class SourceCatalogRequest:
    source_subject: object
    source_selector: SourceSelector
    source_artifact_ref: str
    framework_identity: FrameworkIdentity
    framework_config: object | None
    model_config: object
    download_policy: SourceDownloadPolicy | None = None
    cache_policy: RecipeCachePolicy | None = None
    source_catalog_config: object | None = None
    schema_version: int = SOURCE_CATALOG_REQUEST_SCHEMA_VERSION


class SourceCatalogProvider(Protocol):
    def build_catalog(self, request: SourceCatalogRequest) -> object: ...


class SourceHost(Protocol):
    """Optional framework source/cache hint provider."""

    def source_selector(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> SourceSelector | None: ...

    def source_catalog_config(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> object | None: ...

    def recipe_cache_policy(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> RecipeCachePolicy | None: ...


class SourceSubjectCoordinator(Protocol):
    def broadcast_object(self, payload: object, *, src: int) -> object: ...


class CollectiveHost(Protocol):
    """Optional framework collective primitives for local bootstrap."""

    def source_subject_coordinator(
        self,
        framework_config: object | None,
    ) -> SourceSubjectCoordinator | None: ...

    def local_ready_barrier(
        self,
        framework_config: object | None,
        target_device: object | None,
    ) -> None: ...


@dataclass(frozen=True)
class AdmissionRequest:
    intent: object
    framework_identity: FrameworkIdentity
    placement_identity: PlacementIdentityFacts
    placement_admission: PlacementAdmissionFacts
    model_config: object
    runtime_profile: RuntimeProfile


@dataclass(frozen=True)
class AdmissionDecision:
    family: str
    support_level: str
    startup_allowed: bool
    reload_allowed: bool
    local_bootstrap_allowed: bool
    endpoint_fields: Mapping[str, object] = field(default_factory=dict)


class AdmissionPolicy(Protocol):
    def admit(self, request: AdmissionRequest) -> AdmissionDecision: ...


class DefaultAdmissionPolicy:
    """Conservative admission policy used when a host omits one."""

    def admit(self, request: AdmissionRequest) -> AdmissionDecision:
        requires_proof = request.placement_admission.requires_framework_semantic_proof()
        return AdmissionDecision(
            family="generic",
            support_level=("generic_fail_closed" if requires_proof else "generic"),
            startup_allowed=not requires_proof,
            reload_allowed=not requires_proof,
            local_bootstrap_allowed=not requires_proof,
            endpoint_fields={},
        )


@dataclass(frozen=True)
class RuntimeHostCapabilities:
    framework: FrameworkHost
    placement: PlacementHost
    source_catalog: SourceCatalogProvider | None = None
    source: SourceHost | None = None
    tensor_surface: TensorSurfaceHost | None = None
    collective: CollectiveHost | None = None
    observability: ObservabilitySink | None = None
    runtime_profile: RuntimeProfile | None = None
    admission: AdmissionPolicy | None = None


IntegrationHost = RuntimeHostCapabilities
RuntimeAdmissionDecision = AdmissionDecision
RuntimeAdmissionPolicy = AdmissionPolicy
RuntimeAdmissionRequest = AdmissionRequest


def semantic_placement_digest(
    *,
    kind: str,
    payload: Mapping[str, object],
    schema_version: int = 1,
) -> str:
    """Build a core-owned semantic placement digest."""

    return _stable_digest(
        {
            "schema_version": int(schema_version),
            "kind": str(kind),
            "payload": dict(payload),
        }
    )


def _mapping_from_object(value: object | None) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return {str(key): dumped[key] for key in dumped}
    return {
        str(name): getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


def _stable_payload(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _stable_payload(dataclasses.asdict(cast(Any, value)))
    if hasattr(value, "model_dump"):
        model_dump = value.model_dump
        if callable(model_dump):
            return _stable_payload(model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_payload(value[key])
            for key in sorted(value, key=lambda item: str(item))
            if value[key] is not None
        }
    if isinstance(value, (list, tuple)):
        return [_stable_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_digest(value: object) -> str:
    payload = json.dumps(
        _stable_payload(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionRequest",
    "CollectiveHost",
    "DefaultAdmissionPolicy",
    "FinalizeHookHost",
    "FinalizePhase",
    "FinalizePolicy",
    "FrameworkHost",
    "FrameworkIdentity",
    "IntegrationHost",
    "ManifestPolicy",
    "MaterializationExecutionFacts",
    "MaterializationPolicy",
    "NativeLoadHost",
    "ObservabilitySink",
    "PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION",
    "PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION",
    "PlacementAdmissionFacts",
    "PlacementHost",
    "PlacementIdentityFacts",
    "PlacementMemberFacts",
    "RecipeCachePolicy",
    "RecipeTraceHost",
    "RECIPE_CACHE_POLICY_SCHEMA_VERSION",
    "RuntimeConfig",
    "RuntimeAdmissionDecision",
    "RuntimeAdmissionPolicy",
    "RuntimeAdmissionRequest",
    "RuntimeHostCapabilities",
    "RuntimePlacement",
    "RuntimeProfile",
    "SourceBoundContractProfile",
    "SourceCatalogPolicy",
    "SourceCatalogProvider",
    "SourceCatalogRequest",
    "SOURCE_CATALOG_REQUEST_SCHEMA_VERSION",
    "SOURCE_CATALOG_SCHEMA_VERSION",
    "SourceDownloadPolicy",
    "SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION",
    "SourceHost",
    "SourceSelector",
    "SourceSubjectCoordinator",
    "TensorCastEvent",
    "TensorSurfaceHost",
    "TorchTensorHost",
    "runtime_placement_from_framework_facts",
    "semantic_placement_digest",
]
