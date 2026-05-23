#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorcast.serving._runtime_impl.lifecycle as integration_mod
from tensorcast.pytorch.module_binding import TorchModuleAdapterMixin
from tensorcast.serving._runtime_impl.lifecycle import (
    PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION,
    PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION,
    RECIPE_CACHE_POLICY_SCHEMA_VERSION,
    SERVING_MANIFEST_TENSOR_NAME,
    SOURCE_CATALOG_REQUEST_SCHEMA_VERSION,
    SOURCE_CATALOG_SCHEMA_VERSION,
    SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION,
    AdmissionDecision,
    AdmissionRejectedError,
    AdmissionRequest,
    ArtifactLocatorResolutionError,
    AttachFinalizeError,
    AuthorityValidationError,
    BootstrapPolicy,
    CapabilityMissingError,
    ConfigConflictError,
    DefaultAdmissionPolicy,
    ExistingServingArtifact,
    FinalizeClass,
    FrameworkIdentity,
    IntegrationHost,
    LocalReadyBindingContract,
    LocalReadyManifestCarrierResult,
    LocalReadyMaterializationIdentity,
    LocalSourceBootstrap,
    ManifestMismatchError,
    MaterializationExecutionFacts,
    OwnershipTransferError,
    PlacementAdmissionError,
    PlacementAdmissionFacts,
    PlacementIdentityFacts,
    PlacementMemberFacts,
    RecipeBuildSessionRequest,
    RecipeCachePolicy,
    RequestContext,
    RestoreBindingError,
    RetainedBindingAcquire,
    RuntimeAttachment,
    RuntimeBindingMaterialization,
    RuntimeBindingResult,
    RuntimeBindingState,
    RuntimeBindingView,
    RuntimeProfile,
    RuntimeStateSeed,
    RuntimeWorkerView,
    SchemaMismatchError,
    ServingArtifactLocator,
    ServingBindingPlan,
    ServingConfig,
    ServingIntegration,
    ServingIntegrationError,
    ServingIntegrationNotImplementedError,
    ServingLoadResult,
    ServingPlacement,
    ServingReloadResult,
    ServingRuntimeSession,
    ServingSupportLevel,
    SourceCatalogRequest,
    SourceDownloadPolicy,
    SourceProviderError,
    SourceSelectionProjection,
    SourceSelector,
    SourceSubject,
    TensorcastSemanticValidationSpec,
    TensorSchemaEntry,
    _DirectServingLoad,
    _LocalReadyBootstrap,
    _LocalReadyFinalize,
    _RetainedBindingAcquire,
    _ServingReload,
    bind_serving_artifact,
    build_local_ready_prepared_artifact,
    is_runtime_binding_swap_capable,
    local_ready_current_value_summary_fields,
    restore_prepared_local_ready_binding,
    restore_retained_binding,
    runtime_binding_state_from_runtime_view,
    serving_placement_from_framework_facts,
    source_selection_projection_from_execution_diagnostics,
    source_selection_projection_from_materialization_diagnostics,
    source_subject_broadcast_payload,
    source_subject_from_broadcast_payload,
    swap_serving_artifact,
)
from tensorcast.serving._runtime_impl.lifecycle import (
    BindingValueRef as IntegrationBindingValueRef,
)
from tensorcast.serving._runtime_impl.lifecycle import (
    ServingBindingMemberRef as IntegrationServingBindingMemberRef,
)
from tensorcast.serving.admin import AdminLocalSourceBootstrap
from tensorcast.serving.contract import logical_topology_json
from tensorcast.serving.diagnostics import (
    binding_layout_debug_payload,
    binding_layout_profile_fields,
    binding_layout_tensor_count,
)
from tensorcast.serving.local_ready import (
    canonical_index_entries_from_tensor_schema,
    logical_topology_json_from_recipe,
)
from tensorcast.serving.retained_binding import (
    ParsedRetainedServingBindingAuthority,
    RetainedServingBindingExpectedDigests,
)
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    ServingBindingMemberRef,
    ServingTopologyRef,
)


class _Bound:
    def __init__(self) -> None:
        self.tensors = {"w": torch.ones((1,), dtype=torch.float32)}
        self.binding_layout_id = "layout-1"
        self.last_execution_diagnostics = {"executor": "fake"}
        self.swapped = None

    def swap(self, artifact, **kwargs):
        self.swapped = (artifact, kwargs)
        self.tensors = {"w": torch.full((1,), 2.0, dtype=torch.float32)}
        return self


class _Subset:
    def __init__(self, names):
        self.names = tuple(names)

    def bind(self, **kwargs):
        return _Bound()


class _Artifact:
    def subset(self, names):
        return _Subset(names)


def _matrix_placement(
    *,
    tp_size: int = 2,
    pp_size: int = 1,
    dp_size: int = 1,
    eplb_digest: str | None = None,
) -> ServingPlacement:
    framework_payload = {
        "family": "vllm_parallelism",
        "version": "v1",
        "dimensions": [
            {
                "name": "data_parallel",
                "rank": 0,
                "size": dp_size,
            },
            {
                "name": "pipeline_parallel",
                "rank": 0,
                "size": pp_size,
            },
            {
                "name": "tensor_parallel",
                "rank": 0,
                "size": tp_size,
            },
        ],
        "expert_parallel_enabled": False,
        "eplb_enabled": eplb_digest is not None,
        "eplb_physical_to_logical_digest": eplb_digest,
        "semantic_placement_digests": (
            {} if eplb_digest is None else {"eplb_physical_to_logical": eplb_digest}
        ),
    }
    admission = PlacementAdmissionFacts(
        eplb_enabled=eplb_digest is not None,
        eplb_physical_to_logical_digest=eplb_digest,
        semantic_placement_digests=framework_payload["semantic_placement_digests"],
    )
    return serving_placement_from_framework_facts(
        identity_facts=PlacementIdentityFacts(
            tensor_parallel_rank=0,
            tensor_parallel_size=tp_size,
            pipeline_parallel_rank=0,
            pipeline_parallel_size=pp_size,
            data_parallel_rank=0,
            data_parallel_size=dp_size,
        ),
        admission_facts=admission,
        member_facts=PlacementMemberFacts(
            runtime_rank=0,
            runtime_world_size=tp_size * pp_size * dp_size,
            member_id="dp0:pp0:tp0",
            member_index=0,
            member_count=tp_size * pp_size * dp_size,
        ),
        framework_payload=framework_payload,
        identity_payload=framework_payload,
    )


class _ContractFrameworkHost:
    def identity(self, model_config):
        del model_config
        return FrameworkIdentity(
            framework_name="fake",
            framework_version="1",
            adapter_version="adapter-1",
            serving_abi_version="abi-1",
        )

    def prepare_model_construction(self, framework_config, model_config):
        del framework_config, model_config

    def build_meta_model(self, framework_config, model_config):
        del framework_config, model_config
        return nn.Linear(1, 1)

    def build_runtime_model(self, framework_config, model_config, target_device):
        del framework_config, model_config
        return nn.Linear(1, 1).to(target_device)

    def assert_model_ready_for_runtime_binding(self, model, *, context):
        del model, context

    def semantic_probes(self, model, model_config):
        del model, model_config
        return {}


class _ContractPlacementHost:
    def identity_facts(self, framework_config):
        del framework_config
        return PlacementIdentityFacts(
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_rank=0,
            pipeline_parallel_size=1,
            data_parallel_rank=0,
            data_parallel_size=1,
        )

    def admission_facts(self, framework_config):
        del framework_config
        return PlacementAdmissionFacts()

    def member_facts(self, framework_config):
        del framework_config
        return PlacementMemberFacts(runtime_rank=0, runtime_world_size=1)

    def execution_facts(self, framework_config):
        del framework_config
        return integration_mod.MaterializationExecutionFacts()


def test_integration_host_contract_skeleton_and_default_admission():
    host = IntegrationHost(
        framework=_ContractFrameworkHost(),
        placement=_ContractPlacementHost(),
    )
    service = ServingIntegration(host=host)
    assert service.host is host

    decision = DefaultAdmissionPolicy().admit(
        AdmissionRequest(
            intent=ExistingServingArtifact(
                ServingArtifactLocator.artifact_ref("artifact:1")
            ),
            framework_identity=FrameworkIdentity(
                framework_name="fake",
                framework_version="1",
                adapter_version="adapter-1",
                serving_abi_version="abi-1",
            ),
            placement_identity=PlacementIdentityFacts(
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                pipeline_parallel_rank=0,
                pipeline_parallel_size=1,
                data_parallel_rank=0,
                data_parallel_size=1,
            ),
            placement_admission=PlacementAdmissionFacts(expert_parallel_enabled=True),
            model_config=object(),
            runtime_profile=RuntimeProfile(),
        )
    )
    assert decision.family == "generic"
    assert not decision.startup_allowed
    assert not decision.reload_allowed
    assert not decision.local_bootstrap_allowed


def test_framework_context_preserves_optional_host_placement_payloads():
    class _PayloadPlacementHost(_ContractPlacementHost):
        def framework_payload(self, framework_config):
            assert framework_config == "framework-config"
            return {
                "family": "vllm_parallelism",
                "version": "v1",
            }

        def identity_payload(self, framework_config):
            assert framework_config == "framework-config"
            return {
                "tp_rank": 0,
                "tp_world_size": 1,
            }

    integration = ServingIntegration(
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_PayloadPlacementHost(),
        )
    )

    context = integration._framework_context("framework-config", object())

    assert context.placement is not None
    assert context.placement.framework_payload == {
        "family": "vllm_parallelism",
        "version": "v1",
    }
    assert context.placement.identity_payload == {
        "tp_rank": 0,
        "tp_world_size": 1,
    }


def test_placement_admission_reports_missing_semantic_proofs():
    facts = PlacementAdmissionFacts(
        expert_parallel_enabled=True,
        eplb_enabled=True,
        expert_mapping_digest="expert-digest",
    )

    assert facts.requires_framework_semantic_proof()
    assert facts.missing_framework_semantic_proofs() == ("eplb_physical_to_logical",)

    complete = PlacementAdmissionFacts(
        expert_parallel_enabled=True,
        eplb_enabled=True,
        semantic_placement_digests={
            "expert_mapping": "expert-digest",
            "eplb_physical_to_logical": "eplb-digest",
        },
    )
    assert complete.missing_framework_semantic_proofs() == ()


def test_placement_identity_payload_includes_schema_versions():
    identity = PlacementIdentityFacts(
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
        pipeline_parallel_rank=0,
        pipeline_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
    )
    admission = PlacementAdmissionFacts(
        expert_parallel_enabled=True,
        semantic_placement_digests={"expert_mapping": "expert-digest"},
    )
    placement = serving_placement_from_framework_facts(
        identity_facts=identity,
        admission_facts=admission,
        member_facts=PlacementMemberFacts(
            runtime_rank=0,
            runtime_world_size=2,
        ),
    )

    payload = placement.stable_identity_payload()["identity_payload"]
    assert identity.placement_identity_schema_version == (
        PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION
    )
    assert admission.placement_admission_schema_version == (
        PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION
    )
    assert payload["placement_identity_schema_version"] == (
        PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION
    )
    assert payload["placement_admission_schema_version"] == (
        PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION
    )
    assert payload["semantic_placement_digests"] == {"expert_mapping": "expert-digest"}


def test_existing_serving_artifact_rejects_local_source_selector():
    service = ServingIntegration()
    with pytest.raises(ServingIntegrationError, match="LocalSourceBootstrap"):
        service.start(
            ExistingServingArtifact(SourceSelector.local_path("/tmp/model")),
            RequestContext(),
        )
    with pytest.raises(ServingIntegrationError, match="local_path"):
        service.start(
            ExistingServingArtifact(
                {
                    "kind": "local_path",
                    "value": "/tmp/model",
                }
            ),
            RequestContext(),
        )


def test_retained_binding_acquire_rejects_arbitrary_authority_object():
    with pytest.raises(
        ServingIntegrationError,
        match="ParsedRetainedServingBindingAuthority",
    ):
        RetainedBindingAcquire(SimpleNamespace(readiness="serving_local_ready"))

    authority = _authority()
    assert RetainedBindingAcquire(authority).authority is authority


def test_public_local_source_bootstrap_excludes_admin_override_fields():
    public_fields = {field.name for field in fields(LocalSourceBootstrap)}

    assert public_fields == {
        "source_selector",
        "bootstrap_policy",
        "cache_policy",
    }
    with pytest.raises(TypeError, match="recipe"):
        LocalSourceBootstrap(
            source_selector=SourceSelector.local_path("/tmp/model"),
            bootstrap_policy=BootstrapPolicy(),
            recipe=object(),  # type: ignore[call-arg]
        )


def test_serving_runtime_session_plans_direct_start_from_config(monkeypatch):
    captured = {}
    state = RuntimeBindingState(
        runtime_view=RuntimeBindingView(
            serving_artifact_ref="mi2:serving",
            readiness="serving",
        )
    )
    attachment = RuntimeAttachment(
        model=object(),
        state=state,
        view=RuntimeWorkerView.from_runtime_view(state.runtime_view),
    )

    def fake_start(self, intent, context):
        del self
        captured["intent"] = intent
        captured["context"] = context
        return attachment

    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: captured.setdefault("runtime_initialized", self),
    )
    monkeypatch.setattr(ServingIntegration, "start", fake_start)

    session = ServingRuntimeSession.from_config(
        ServingConfig.from_mapping(
            {
                "bootstrap": {
                    "mode": "disabled",
                },
                "serving": {
                    "artifact_locator": {
                        "kind": "artifact_ref",
                        "value": "mi2:serving",
                    },
                },
            }
        ),
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
        ),
    )
    result = session.start(RequestContext(model_config=object()))

    assert result is attachment
    assert captured["runtime_initialized"] is session.serving_config.runtime
    assert isinstance(captured["intent"], ExistingServingArtifact)
    assert captured["intent"].artifact_locator.kind == "artifact_ref"


def test_serving_runtime_session_private_intent_initializes_runtime(monkeypatch):
    captured = {}
    state = RuntimeBindingState(
        runtime_view=RuntimeBindingView(
            serving_artifact_ref="mi2:serving",
            readiness="serving",
        )
    )
    attachment = RuntimeAttachment(
        model=object(),
        state=state,
        view=RuntimeWorkerView.from_runtime_view(state.runtime_view),
    )

    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: captured.setdefault("runtime_initialized", self),
    )

    def fake_start(self, intent, context):
        del self
        captured["intent"] = intent
        captured["context"] = context
        return attachment

    monkeypatch.setattr(ServingIntegration, "start", fake_start)
    session = ServingRuntimeSession.from_config(
        {
            "bootstrap": {
                "mode": "disabled",
            },
            "serving": {
                "artifact_locator": {
                    "kind": "artifact_ref",
                    "value": "mi2:serving",
                },
            },
        },
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
        ),
    )
    intent = AdminLocalSourceBootstrap(
        source_selector=SourceSelector.local_path("/tmp/model"),
        bootstrap_policy=BootstrapPolicy(),
    )

    result = session._start_intent(intent, RequestContext(model_config=object()))

    assert result is attachment
    assert captured["runtime_initialized"] is session.serving_config.runtime
    assert captured["intent"] is intent


def test_serving_runtime_session_rejects_conflicting_start_config(monkeypatch):
    initialized = False

    def fail_if_initialized(self):
        del self
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        fail_if_initialized,
    )
    session = ServingRuntimeSession.from_config(
        ServingConfig.from_mapping(
            {
                "bootstrap": {
                    "mode": "required",
                },
                "serving": {
                    "artifact_locator": {
                        "kind": "artifact_ref",
                        "value": "mi2:serving",
                    },
                },
            }
        ),
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
        ),
    )

    with pytest.raises(ConfigConflictError, match="required"):
        session.start(RequestContext(model_config=object()))
    assert not initialized


def test_serving_runtime_session_uses_source_host_for_local_bootstrap(monkeypatch):
    captured = {}
    attachment = RuntimeAttachment(
        model=object(),
        state=RuntimeBindingState(runtime_view=RuntimeBindingView()),
        view=RuntimeWorkerView.from_runtime_view(RuntimeBindingView()),
    )

    class _Source:
        def source_selector(self, framework_config, model_config):
            captured["source_selector_args"] = (framework_config, model_config)
            return SourceSelector.local_path("/tmp/fakefw-model")

        def source_catalog_config(self, framework_config, model_config):
            del framework_config, model_config
            return None

        def recipe_cache_policy(self, framework_config, model_config):
            del framework_config, model_config
            return None

    def fake_start(self, intent, context):
        del self, context
        captured["intent"] = intent
        return attachment

    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: captured.setdefault("runtime_initialized", self),
    )
    monkeypatch.setattr(ServingIntegration, "start", fake_start)

    session = ServingRuntimeSession.from_config(
        ServingConfig.from_mapping(
            {
                "bootstrap": {
                    "mode": "required",
                },
            }
        ),
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
            source=_Source(),
        ),
    )
    result = session.start(
        RequestContext(
            framework_config="framework-config",
            model_config=SimpleNamespace(name="model-config"),
        )
    )

    assert result is attachment
    assert captured["runtime_initialized"] is session.serving_config.runtime
    assert captured["source_selector_args"][0] == "framework-config"
    assert isinstance(captured["intent"], LocalSourceBootstrap)
    assert captured["intent"].source_selector == SourceSelector.local_path(
        "/tmp/fakefw-model"
    )


def test_serving_runtime_session_rejects_local_reload_artifact_locator(monkeypatch):
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: pytest.fail("local artifact locator rejection must precede init"),
    )
    session = ServingRuntimeSession.from_config(
        ServingConfig.from_mapping(
            {
                "bootstrap": {
                    "mode": "disabled",
                },
                "serving": {
                    "artifact_locator": {
                        "kind": "artifact_ref",
                        "value": "mi2:serving",
                    },
                },
            }
        ),
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
        ),
    )
    attachment = RuntimeAttachment(
        model=object(),
        state=RuntimeBindingState(runtime_view=RuntimeBindingView()),
        view=RuntimeWorkerView.from_runtime_view(RuntimeBindingView()),
    )

    with pytest.raises(ConfigConflictError, match="durable"):
        session.reload(
            current_attachment=attachment,
            artifact_locator=SourceSelector.local_path("/tmp/model"),
            policy=None,
            context=RequestContext(),
        )
    with pytest.raises(ConfigConflictError, match="durable"):
        session.reload(
            current_attachment=attachment,
            artifact_locator={
                "kind": "local_path",
                "value": "/tmp/model",
            },
            policy=None,
            context=RequestContext(),
        )
    with pytest.raises(ConfigConflictError, match="ServingArtifactLocator"):
        session.reload(
            current_attachment=attachment,
            artifact_locator={
                "kind": "artifact_ref",
                "value": "mi2:serving-next",
            },
            policy=None,
            context=RequestContext(),
        )
    with pytest.raises(ConfigConflictError, match="ServingPolicy"):
        session.reload(
            current_attachment=attachment,
            artifact_locator=ServingArtifactLocator.artifact_ref("mi2:serving-next"),
            policy={"mode": "from_manifest"},
            context=RequestContext(),
        )


def test_admission_endpoint_uses_tensor_parallel_facts_not_member_index():
    class _MultiDimPlacementHost(_ContractPlacementHost):
        def identity_facts(self, framework_config):
            del framework_config
            return PlacementIdentityFacts(
                tensor_parallel_rank=1,
                tensor_parallel_size=2,
                pipeline_parallel_rank=1,
                pipeline_parallel_size=2,
                data_parallel_rank=0,
                data_parallel_size=1,
            )

        def member_facts(self, framework_config):
            del framework_config
            return PlacementMemberFacts(
                runtime_rank=3,
                runtime_world_size=4,
                member_id="dp0:pp1:tp1",
                member_index=3,
                member_count=4,
            )

    class _Admission:
        def admit(self, request):
            del request
            return AdmissionDecision(
                family="fake-family",
                support_level="runtime_bind_swap_ready",
                startup_allowed=True,
                reload_allowed=True,
                local_bootstrap_allowed=True,
                endpoint_fields={},
            )

    decision = ServingIntegration(
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_MultiDimPlacementHost(),
            admission=_Admission(),
        )
    )._admit_intent(
        ExistingServingArtifact(ServingArtifactLocator.artifact_ref("mi2:serving")),
        RequestContext(framework_config=object(), model_config=object()),
    )

    assert decision is not None
    assert decision.endpoint_fields["tp_rank"] == 1
    assert decision.endpoint_fields["tp_world_size"] == 2


def test_runtime_worker_view_projection_is_typed_not_diagnostics_only():
    runtime_view = RuntimeBindingView(
        serving_artifact_ref="mi2:serving",
        source_artifact_ref="mi2:source",
        representation_contract_hash="repr",
        tensor_schema_hash="schema",
        local_serving_ref="local:ready",
        readiness="serving_local_ready",
        diagnostics={
            "serving_build_digest": "build",
            "family": "fake-family",
            "tp_rank": 1,
            "tp_world_size": 2,
            "verification_state": "verified",
            "verification_job_id": "job-1",
            "source_selection": {
                "selected_source_kind": "published_memory_replica",
                "selected_replica_id": "replica-1",
                "selected_producer_worker_id": "worker-1",
                "selected_byte_space_kind": "cuda",
                "selected_byte_space_id": "0",
                "p2p_bytes": 1024,
                "fallback_bytes": 0,
                "disk_bytes": 0,
                "reselection_attempts": 1,
            },
        },
    )
    state = RuntimeBindingState(
        binding=object(),
        artifact_ref="mi2:serving",
        runtime_view=runtime_view,
    )

    worker_view = ServingIntegration().describe(state)

    assert isinstance(worker_view, RuntimeWorkerView)
    payload = worker_view.endpoint.to_weight_version_payload()
    assert payload["schema_version"] == 1
    assert payload["serving_artifact_ref"] == "mi2:serving"
    assert payload["source_artifact_ref"] == "mi2:source"
    assert payload["representation_contract_hash"] == "repr"
    assert payload["serving_build_digest"] == "build"
    assert payload["tensor_schema_hash"] == "schema"
    assert payload["readiness"] == "serving_local_ready"
    assert payload["family"] == "fake-family"
    assert payload["tp_rank"] == 1
    assert payload["tp_world_size"] == 2
    assert payload["source_selection"] == {
        "schema_version": 1,
        "selected_source_kind": "published_memory_replica",
        "selected_replica_id": "replica-1",
        "selected_producer_worker_id": "worker-1",
        "selected_byte_space_kind": "cuda",
        "selected_byte_space_id": "0",
        "p2p_bytes": 1024,
        "fallback_bytes": 0,
        "disk_bytes": 0,
        "reselection_attempts": 1,
    }
    assert worker_view.diagnostics["verification_job_id"] == "job-1"


def test_runtime_worker_view_accepts_typed_source_selection_projection():
    runtime_view = RuntimeBindingView(
        serving_artifact_ref="mi2:serving",
        representation_contract_hash="repr",
        tensor_schema_hash="schema",
        readiness="serving",
        diagnostics={
            "source_selection": SourceSelectionProjection(
                selected_source_kind="canonical_fallback",
                fallback_bytes=2048,
                fallback_reason_bucket="transport_unavailable",
            ),
        },
    )

    worker_view = RuntimeWorkerView.from_runtime_view(runtime_view)
    payload = worker_view.endpoint.to_weight_version_payload()

    assert payload["source_selection"] == {
        "schema_version": 1,
        "selected_source_kind": "canonical_fallback",
        "p2p_bytes": 0,
        "fallback_bytes": 2048,
        "disk_bytes": 0,
        "reselection_attempts": 0,
        "fallback_reason_bucket": "transport_unavailable",
    }


def test_source_selection_projection_from_materialization_diagnostics():
    p2p = source_selection_projection_from_materialization_diagnostics(
        SimpleNamespace(
            source="p2p",
            total_bytes=4096,
            replica_uuid="local-replica",
            ticket_replica_uuid="source-replica",
            retry_attempts=2,
            retry_reason_buckets={"source_visibility_stale": 1},
        )
    )

    assert p2p is not None
    assert p2p.to_dict() == {
        "schema_version": 1,
        "selected_source_kind": "published_memory_replica",
        "selected_replica_id": "source-replica",
        "p2p_bytes": 4096,
        "fallback_bytes": 0,
        "disk_bytes": 0,
        "reselection_attempts": 1,
        "reject_reason_bucket": "source_visibility_stale",
    }

    disk = source_selection_projection_from_materialization_diagnostics(
        SimpleNamespace(
            source="disk",
            total_bytes=2048,
            retry_attempts=3,
            retry_reason_buckets={"transport_unavailable": 2},
        )
    )

    assert disk is not None
    assert disk.to_dict() == {
        "schema_version": 1,
        "selected_source_kind": "canonical_fallback",
        "p2p_bytes": 0,
        "fallback_bytes": 2048,
        "disk_bytes": 2048,
        "reselection_attempts": 2,
        "fallback_reason_bucket": "transport_unavailable",
    }


def test_execution_diagnostics_seed_runtime_source_selection_projection():
    projection = source_selection_projection_from_execution_diagnostics(
        SimpleNamespace(
            actual_collective_committed_bytes=4096,
            collective_peer_transfer_bytes=1024,
            fallback_bytes=0,
        )
    )
    assert projection is not None
    assert projection.to_dict() == {
        "schema_version": 1,
        "selected_source_kind": "published_memory_replica",
        "p2p_bytes": 1024,
        "fallback_bytes": 0,
        "disk_bytes": 0,
        "reselection_attempts": 0,
    }

    resolved = SimpleNamespace(
        artifact_ref="mi2:serving",
        manifest=SimpleNamespace(
            representation_contract_hash="repr",
            source_artifact_ref="mi2:source",
            serving_build_digest="build",
        ),
    )
    seed = ServingIntegration._state_seed(
        resolved,
        tensor_schema_hash="schema",
        execution_diagnostics=SimpleNamespace(
            actual_collective_committed_bytes=4096,
            collective_peer_transfer_bytes=1024,
            fallback_bytes=0,
        ),
        binding_handle=SimpleNamespace(current_value=None),
    )
    worker_view = RuntimeWorkerView.from_runtime_view(seed.runtime_view())
    payload = worker_view.endpoint.to_weight_version_payload()

    assert payload["source_selection"] == {
        "schema_version": 1,
        "selected_source_kind": "published_memory_replica",
        "p2p_bytes": 1024,
        "fallback_bytes": 0,
        "disk_bytes": 0,
        "reselection_attempts": 0,
    }


def test_materialization_diagnostics_seed_runtime_source_selection_projection():
    resolved = SimpleNamespace(
        artifact_ref="mi2:serving",
        manifest=SimpleNamespace(
            representation_contract_hash="repr",
            source_artifact_ref="mi2:source",
            serving_build_digest="build",
        ),
    )

    seed = ServingIntegration._state_seed(
        resolved,
        tensor_schema_hash="schema",
        materialization_diagnostics={
            "source": "p2p",
            "total_bytes": 4096,
            "replica_id": "source-replica",
            "retry_attempts": 1,
            "retry_reason_buckets": {},
        },
        execution_diagnostics=SimpleNamespace(
            actual_collective_committed_bytes=0,
            collective_peer_transfer_bytes=0,
            fallback_bytes=4096,
        ),
        binding_handle=SimpleNamespace(current_value=None),
    )
    worker_view = RuntimeWorkerView.from_runtime_view(seed.runtime_view())
    payload = worker_view.endpoint.to_weight_version_payload()

    assert payload["source_selection"] == {
        "schema_version": 1,
        "selected_source_kind": "published_memory_replica",
        "selected_replica_id": "source-replica",
        "p2p_bytes": 4096,
        "fallback_bytes": 0,
        "disk_bytes": 0,
        "reselection_attempts": 0,
    }


def test_runtime_binding_result_captures_materialization_diagnostics():
    binding = SimpleNamespace(
        tensors={"weight": object()},
        binding_layout_id="layout-1",
        last_execution_diagnostics=None,
        last_materialization_diagnostics={
            "source": "disk",
            "total_bytes": 2048,
        },
    )

    result = RuntimeBindingResult.from_binding(binding)

    assert result.materialization_diagnostics == {
        "source": "disk",
        "total_bytes": 2048,
    }


def test_local_bootstrap_requires_host_source_catalog_provider():
    service = ServingIntegration(
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
        )
    )
    source_subject = SourceSubject(
        artifact_ref="mi2:source",
        subject=object(),
    )

    with pytest.raises(
        CapabilityMissingError, match="IntegrationHost.source_catalog"
    ) as exc_info:
        service._local_ready_source_catalog(
            _LocalReadyBootstrap(
                source_selector=SourceSelector.local_path("/tmp/model"),
                source_subject=source_subject,
                model_config=object(),
            ),
            source_subject=source_subject,
            source_artifact_ref="mi2:source",
        )
    assert exc_info.value.details["level"] == "level2-local-bootstrap"
    assert exc_info.value.details["capability"] == "source_catalog"
    assert exc_info.value.details["operation"] == "local_bootstrap.source_catalog"
    assert exc_info.value.details["required_methods"] == ("build_catalog",)


def test_source_catalog_provider_uses_integration_host():
    class _Provider:
        def __init__(self):
            self.request = None

        def build_catalog(self, request):
            self.request = request
            return SimpleNamespace(source_artifact_ref=request.source_artifact_ref)

    provider = _Provider()
    host = IntegrationHost(
        framework=_ContractFrameworkHost(),
        placement=_ContractPlacementHost(),
        source_catalog=provider,
    )
    service = ServingIntegration(host=host)
    source_subject = SourceSubject(
        artifact_ref="mi2:source",
        subject=object(),
    )

    catalog = service._local_ready_source_catalog(
        _LocalReadyBootstrap(
            source_selector=SourceSelector.local_path("/tmp/model"),
            source_subject=source_subject,
            model_config=object(),
        ),
        source_subject=source_subject,
        source_artifact_ref="mi2:source",
    )

    assert catalog.source_artifact_ref == "mi2:source"
    assert provider.request is not None
    assert provider.request.framework_identity.framework_name == "fake"
    assert provider.request.schema_version == SOURCE_CATALOG_REQUEST_SCHEMA_VERSION


@pytest.mark.parametrize(
    "provider_ref, expected",
    (
        (None, "without a real source_artifact_ref"),
        ("disk:/tmp/model", "without a real source_artifact_ref"),
        ("mi2:other:source", "expected 'mi2:source'"),
    ),
)
def test_source_catalog_provider_must_return_matching_source_artifact_ref(
    provider_ref, expected
):
    class _Provider:
        @staticmethod
        def build_catalog(request):
            del request
            if provider_ref is None:
                return SimpleNamespace()
            return SimpleNamespace(source_artifact_ref=provider_ref)

    service = ServingIntegration(
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
            source_catalog=_Provider(),
        )
    )
    source_subject = SourceSubject(
        artifact_ref="mi2:source",
        subject=object(),
    )

    with pytest.raises(ServingIntegrationError, match=expected):
        service._local_ready_source_catalog(
            _LocalReadyBootstrap(
                source_selector=SourceSelector.local_path("/tmp/model"),
                source_subject=source_subject,
                model_config=object(),
            ),
            source_subject=source_subject,
            source_artifact_ref="mi2:source",
        )


def test_local_ready_build_recipe_requires_real_source_subject_artifact_ref():
    service = ServingIntegration()

    with pytest.raises(ServingIntegrationError, match="real source artifact identity"):
        service._local_ready_prepare_with_built_recipe(
            _LocalReadyBootstrap(
                source_selector=SourceSelector.local_path("/tmp/model"),
                source_subject=SourceSubject(
                    artifact_ref="disk:/tmp/model",
                    subject=object(),
                ),
                model_config=object(),
            )
        )


def test_source_catalog_request_and_policy_schema_versions():
    download_policy = SourceDownloadPolicy(fields={"download_dir": "/tmp/download"})
    cache_policy = RecipeCachePolicy(fields={"cache_root": "/tmp/cache"})
    request = SourceCatalogRequest(
        source_subject=SourceSubject(
            artifact_ref="mi2:source",
            subject=object(),
        ),
        source_selector=SourceSelector.local_path("/tmp/model"),
        source_artifact_ref="mi2:source",
        framework_identity=FrameworkIdentity(
            framework_name="fake",
            framework_version="1",
            adapter_version="adapter-1",
            serving_abi_version="abi-1",
        ),
        framework_config=None,
        model_config=object(),
        download_policy=download_policy,
        cache_policy=cache_policy,
    )

    assert download_policy.schema_version == SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION
    assert cache_policy.schema_version == RECIPE_CACHE_POLICY_SCHEMA_VERSION
    assert request.schema_version == SOURCE_CATALOG_REQUEST_SCHEMA_VERSION
    assert request.download_policy is download_policy
    assert request.cache_policy is cache_policy


def test_recipe_cache_policy_builds_model_adjacent_cache_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_file = model_dir / "rank0.safetensors"
    weight_file.write_bytes(b"")
    source_catalog = SimpleNamespace(
        selected_files=(SimpleNamespace(path=str(weight_file)),)
    )
    policy = RecipeCachePolicy(
        fields={
            "cache_root": str(tmp_path / "default-cache"),
            "prefer_model_adjacent": True,
            "debug_output_dir": str(tmp_path / "debug"),
            "debug_dump_trace": True,
            "synchronous_cache_write": False,
        }
    )

    config = ServingIntegration._local_ready_recipe_cache_config(
        _LocalReadyBootstrap(cache_config=policy),
        source_catalog=source_catalog,
    )

    root = model_dir / ".tensorcast" / "bootstrap_cache"
    assert config.cache_dirs == (str(root / "trace_plans"),)
    assert config.recipe_cache_dirs == (str(root / "compiled_recipes"),)
    assert config.trace_write_dirs == (str(root / "trace_plans"),)
    assert config.recipe_cache_write_dirs == (str(root / "compiled_recipes"),)
    assert config.debug_output_dir == tmp_path / "debug"
    assert config.debug_dump_trace
    assert not config.synchronous_cache_write


def test_public_intent_attachment_type_exists():
    state = RuntimeBindingState(runtime_view=RuntimeBindingView())
    attachment = RuntimeAttachment(
        model=object(),
        state=state,
        view=RuntimeWorkerView.from_runtime_view(RuntimeBindingView()),
    )
    assert attachment.state is state
    assert attachment.recipe is None
    intent = LocalSourceBootstrap(
        source_selector=SourceSelector.local_path("/tmp/model"),
        bootstrap_policy=BootstrapPolicy(),
    )
    assert intent.source_selector.kind == "local_path"


def test_local_source_bootstrap_start_derives_request_from_host(monkeypatch):
    captured = {}
    coordinator = object()
    model = nn.Linear(1, 1)
    recipe = object()
    runtime_view = RuntimeBindingView(
        source_artifact_ref="mi2:source",
        representation_contract_hash="repr",
        tensor_schema_hash="schema",
        readiness="serving_local_ready",
    )
    runtime_state = RuntimeBindingState(
        binding=object(),
        artifact_ref="mi2:source",
        runtime_view=runtime_view,
    )

    class _Admission:
        def admit(self, request):
            captured["admission"] = request
            return AdmissionDecision(
                family="fake-family",
                support_level="runtime_bind_swap_ready",
                startup_allowed=True,
                reload_allowed=True,
                local_bootstrap_allowed=True,
            )

    class _Collective:
        def source_subject_coordinator(self, framework_config):
            captured["coordinator_framework_config"] = framework_config
            return coordinator

        def local_ready_barrier(self, framework_config, target_device):
            captured["barrier"] = (framework_config, target_device)

    class _Source:
        def source_catalog_config(self, framework_config, model_config):
            captured["source_catalog_config_args"] = (
                framework_config,
                model_config,
            )
            return "host-source-config"

        def recipe_cache_policy(self, framework_config, model_config):
            captured["recipe_cache_policy_args"] = (
                framework_config,
                model_config,
            )
            return RecipeCachePolicy(fields={"cache_root": "/tmp/cache"})

    def fake_prepare(self, request):
        del self
        captured["request"] = request
        return integration_mod.LocalReadyServingResult(
            model=model,
            runtime_state=runtime_state,
            runtime_view=runtime_view,
            prepared=integration_mod.PreparedServingArtifact(
                source_artifact_ref="mi2:source",
                serving_manifest_ref="manifest-ref",
                representation_contract_hash="repr",
                serving_build_digest="build",
                family="fake-family",
                tensor_schema_hash="schema",
            ),
            recipe=recipe,
        )

    monkeypatch.setattr(
        ServingIntegration, "_prepare_local_source_bootstrap", fake_prepare
    )

    attachment = ServingIntegration(
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
            tensor_surface=integration_mod.TorchTensorHost(),
            collective=_Collective(),
            source=_Source(),
            admission=_Admission(),
        )
    ).start(
        LocalSourceBootstrap(
            source_selector=SourceSelector.local_path("/tmp/model"),
            bootstrap_policy=BootstrapPolicy(),
        ),
        RequestContext(
            framework_config="framework-config",
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cpu"),
        ),
    )

    request = captured["request"]
    assert attachment.model is model
    assert attachment.prepared is not None
    assert attachment.recipe is recipe
    assert request.source_selector == SourceSelector.local_path("/tmp/model")
    assert request.source_subject_coordinator is coordinator
    assert captured["coordinator_framework_config"] == "framework-config"
    assert captured["barrier"] == ("framework-config", torch.device("cpu"))
    assert request.source_catalog_config == "host-source-config"
    assert isinstance(request.cache_config, RecipeCachePolicy)
    assert request.cache_config.fields["cache_root"] == "/tmp/cache"
    assert captured["source_catalog_config_args"] == (
        "framework-config",
        captured["admission"].model_config,
    )
    assert captured["recipe_cache_policy_args"] == (
        "framework-config",
        captured["admission"].model_config,
    )
    assert request.cache_config_factory is None
    assert request.manifest_tensor_name == SERVING_MANIFEST_TENSOR_NAME
    assert request.framework_name == "fake"
    assert request.adapter_version == "adapter-1"
    assert request.serving_abi_version == "abi-1"
    assert request.family == "fake-family"
    assert request.tp_rank == 0
    assert request.tp_world_size == 1
    assert request.build_recipe_from_framework_context
    assert request.build_model_from_framework_context
    assert request.build_manifest_carrier_from_framework_context
    assert request.validate_representation_contract_hash
    assert request.require_materialization_options
    assert request.source_bound_contract_path == (
        integration_mod.SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4
    )
    assert request.placement.member.member_index == 0


def test_host_start_fails_clearly_without_tensor_surface():
    class _Resolver:
        def resolve(self, artifact_ref):
            return SimpleNamespace(
                artifact=_Artifact(),
                artifact_ref=artifact_ref,
                tensor_names=(),
            )

    service = ServingIntegration(
        resolver=_Resolver(),
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
        ),
    )

    with pytest.raises(ServingIntegrationError, match="TensorSurfaceHost") as exc_info:
        service.start(
            ExistingServingArtifact(ServingArtifactLocator.artifact_ref("mi2:serving")),
            RequestContext(
                framework_config=object(),
                model_config=SimpleNamespace(model="fake"),
                target_device=torch.device("cpu"),
            ),
        )
    assert isinstance(exc_info.value, CapabilityMissingError)
    assert exc_info.value.details["level"] == "level1-runtime"
    assert exc_info.value.details["capability"] == "tensor_surface"
    assert exc_info.value.details["operation"] == "runtime_tensor_surface"


def test_integration_host_delegates_recipe_trace_capabilities():
    class _TraceFrameworkHost(_ContractFrameworkHost):
        def __init__(self):
            self.events = []

        def trace_model_load(
            self, model, ordered_names, meta_by_name, *, debug_dump_trace=False
        ):
            self.events.append(
                (
                    "trace",
                    model,
                    tuple(ordered_names),
                    dict(meta_by_name),
                    debug_dump_trace,
                )
            )
            return SimpleNamespace(trace=True)

        def cleanup_after_recipe_build(
            self, model, model_config, *, framework_config=None
        ):
            self.events.append(("cleanup", model, model_config, framework_config))

        def support_level(self, model, model_config):
            self.events.append(("support", model, model_config))
            return ServingSupportLevel.RUNTIME_BIND_SWAP_READY

        def process_after_load_class(self, model, model_config):
            self.events.append(("process_class", model, model_config))
            return FinalizeClass.RUNTIME_ONLY

        def post_bind_finalize_class(self, model, model_config):
            self.events.append(("post_class", model, model_config))
            return FinalizeClass.RUNTIME_ONLY

    framework = _TraceFrameworkHost()
    integration = ServingIntegration(
        host=IntegrationHost(
            framework=framework,
            placement=_ContractPlacementHost(),
            tensor_surface=integration_mod.TorchTensorHost(),
        )
    )

    assert integration.trace_model_load(
        "model",
        ("w",),
        {"w": object()},
        debug_dump_trace=True,
    ).trace
    integration.cleanup_after_recipe_build(
        "model",
        "model-config",
        framework_config="framework-config",
    )
    assert (
        integration.support_level("model", "model-config")
        is ServingSupportLevel.RUNTIME_BIND_SWAP_READY
    )
    assert (
        integration.process_after_load_class("model", "model-config")
        is FinalizeClass.RUNTIME_ONLY
    )
    assert (
        integration.post_bind_finalize_class("model", "model-config")
        is FinalizeClass.RUNTIME_ONLY
    )
    assert framework.events[0][0] == "trace"
    assert framework.events[1] == (
        "cleanup",
        "model",
        "model-config",
        "framework-config",
    )


def test_integration_host_fails_recipe_trace_miss_clearly():
    integration = ServingIntegration(
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
            tensor_surface=integration_mod.TorchTensorHost(),
        )
    )

    with pytest.raises(ServingIntegrationError, match="RecipeTraceHost"):
        integration.trace_model_load(object(), (), {})


def test_binding_layout_diagnostics_are_core_owned():
    layout = SimpleNamespace(
        binding_layout_id="layout-1",
        target_index_bytes=b"abc",
        target_layout=SimpleNamespace(
            layout_kind=1,
            index_kind=2,
            tensor_spec_kind=3,
            logical_layout_hash=b"\x01\x02",
            view_id="view-1",
            storages=[
                SimpleNamespace(
                    storage_id="storage-1",
                    device_id=0,
                    storage_length=8,
                    mapping_base_offset=0,
                )
            ],
            offsets=[
                SimpleNamespace(
                    name="w",
                    storage_id="storage-1",
                    storage_offset=0,
                    logical_length=4,
                )
            ],
        ),
        dst_specs=[
            SimpleNamespace(
                name="w",
                dtype="float32",
                shape=(1, 4),
                stride=(4, 1),
                storage_offset=0,
                logical_length=4,
            )
        ],
    )

    assert binding_layout_tensor_count(layout) == 1
    assert binding_layout_profile_fields(layout) == {
        "target_index_bytes": 3,
        "binding_tensor_count": 1,
    }
    payload = binding_layout_debug_payload(
        layout,
        target_device="cpu",
        context="unit",
        pid=123,
    )

    assert payload["binding_layout_id"] == "layout-1"
    assert payload["target_index_bytes_len"] == 3
    assert payload["layout"]["offsets"][0]["name"] == "w"
    assert payload["dst_specs"][0]["shape"] == [1, 4]


def test_runtime_binding_swap_capability_is_core_owned():
    assert is_runtime_binding_swap_capable(SimpleNamespace(swap=lambda _: None))
    assert is_runtime_binding_swap_capable(SimpleNamespace(swap_capable=True))
    assert not is_runtime_binding_swap_capable(SimpleNamespace())


def test_local_ready_current_value_summary_is_core_owned():
    from tensorcast.proto.daemon.v2 import store_daemon_pb2

    current_value = SimpleNamespace(
        binding_value_id="value-1",
        local_serving_ref="binding-local:binding-1:value-1",
        verification_state=(
            store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY
        ),
    )

    assert local_ready_current_value_summary_fields(
        current_value, require_local_serving_ref=True
    ) == {
        "binding_value_id": "value-1",
        "verification_state": "local_only",
        "local_serving_ref": "binding-local:binding-1:value-1",
    }
    with pytest.raises(integration_mod.ServingIntegrationError):
        local_ready_current_value_summary_fields(
            SimpleNamespace(binding_value_id="value-1"),
            require_local_serving_ref=True,
        )


def test_runtime_binding_state_from_view_is_core_owned():
    binding = _Bound()
    view = RuntimeBindingView(
        serving_artifact_ref="mi2:test:serving",
        representation_contract_hash="repr",
    )

    state = runtime_binding_state_from_runtime_view(
        binding=binding,
        runtime_view=view,
        ownership_handle="owner",
    )

    assert state.binding is binding
    assert state.artifact_ref == "mi2:test:serving"
    assert state.runtime_view is view
    assert state.ownership_handle == "owner"


def test_build_local_ready_prepared_artifact_returns_runtime_state_and_view():
    from tensorcast.proto.daemon.v2 import store_daemon_pb2

    binding = _Bound()
    binding.binding_layout_id = "layout-1"
    current_value = SimpleNamespace(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=3,
        local_serving_ref="binding-local:binding-1:value-1",
        verification_state=(
            store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY
        ),
        verification_job_id="verify-1",
    )
    result = build_local_ready_prepared_artifact(
        source_artifact_ref="mi2:test:source",
        serving_manifest_ref="manifest",
        representation_contract_hash="repr",
        serving_build_digest="build",
        tensor_schema_hash="schema",
        current_value=current_value,
        binding=binding,
        family="dummy",
        tp_rank=1,
        tp_world_size=2,
        source_bound_contract_state=SimpleNamespace(
            source_bound_contract_version=4,
            source_bound_capability_names=("a",),
            source_bound_contract_ready=True,
        ),
        source_bound_contract_path="/tmp/contract.json",
    )

    assert result.runtime_state.binding is binding
    assert result.runtime_state.artifact_ref == "mi2:test:source"
    assert result.runtime_view.source_artifact_ref == "mi2:test:source"
    assert result.runtime_view.serving_artifact_ref is None
    assert result.runtime_view.readiness == "serving_local_ready"
    assert result.runtime_view.local_serving_ref == ("binding-local:binding-1:value-1")
    assert result.runtime_view.tensor_schema_hash == "schema"
    report = result.runtime_view.diagnostics["serving_realization_report"]
    assert report["realization"]["binding_value"]["verification_state"] == "local_only"
    assert "verification_state" not in result.runtime_view.diagnostics
    assert result.binding_value is not None
    assert result.binding_value.readiness == "serving_local_ready"
    assert result.binding_value.local_serving_ref == "binding-local:binding-1:value-1"
    worker_view = RuntimeWorkerView.from_runtime_view(result.runtime_view)
    payload = worker_view.endpoint.to_weight_version_payload()
    assert payload["serving_manifest_ref"] == "manifest"
    assert payload["serving_build_digest"] == "build"
    assert payload["binding_layout_id"] == "layout-1"
    assert "bootstrap_summary" not in payload
    assert payload["source_bound_contract"] == {
        "version": 4,
        "capability_flags": ["a"],
        "ready": True,
        "path": "/tmp/contract.json",
    }
    assert payload["realize_diagnostics"]["collective_requested"] is False


def test_serving_integration_builds_local_ready_manifest_contract_in_core(monkeypatch):
    calls = []
    integration = ServingIntegration()
    recipe = SimpleNamespace(topology_ref=object(), member_ref=object())

    monkeypatch.setattr(
        integration_mod,
        "canonical_index_from_recipe",
        lambda seen_recipe: calls.append(("canonical", seen_recipe)) or "canonical",
    )
    monkeypatch.setattr(
        integration_mod,
        "compute_serving_tensor_schema_hash",
        lambda canonical, **kwargs: calls.append(("schema", canonical, kwargs))
        or "schema-hash",
    )
    monkeypatch.setattr(
        integration_mod,
        "logical_topology_json_from_recipe",
        lambda seen_recipe, **kwargs: calls.append(("topology", seen_recipe, kwargs))
        or '{"topology": true}',
    )
    monkeypatch.setattr(
        integration_mod,
        "prepare_same_binding_manifest_carrier",
        lambda seen_recipe, **kwargs: calls.append(("carrier", seen_recipe, kwargs))
        or ("manifest-ref", b"manifest"),
    )

    result = integration.build_local_ready_manifest_carrier_from_contract(
        recipe=recipe,
        manifest_tensor_name="__tensorcast_meta__.manifest",
        representation_contract_hash_factory=lambda tensor_schema_hash: (
            f"repr:{tensor_schema_hash}"
        ),
        topology="topology-ref",
        framework_payload={"rank": 0},
    )

    assert result == ("manifest-ref", b"manifest")
    assert calls == [
        ("canonical", recipe),
        (
            "schema",
            "canonical",
            {
                "manifest_tensor_name": "__tensorcast_meta__.manifest",
            },
        ),
        (
            "topology",
            recipe,
            {
                "topology": "topology-ref",
                "framework_payload": {
                    "rank": 0,
                },
            },
        ),
        (
            "carrier",
            recipe,
            {
                "manifest_tensor_name": "__tensorcast_meta__.manifest",
                "representation_contract_hash": "repr:schema-hash",
                "logical_topology_json_payload": '{"topology": true}',
                "topology_admission_digest": None,
            },
        ),
    ]


def test_local_ready_logical_topology_requires_topology_ref():
    recipe = SimpleNamespace(
        topology_ref=ServingTopologyRef(schema_topology_digest="a")
    )

    with pytest.raises(ValueError, match="requires ServingTopologyRef"):
        logical_topology_json_from_recipe(recipe)


def test_local_ready_logical_topology_allows_topology_insensitive_recipe():
    recipe = SimpleNamespace(topology_ref=None, member_ref=None)

    assert logical_topology_json_from_recipe(recipe) is None


def test_serving_integration_builds_local_ready_manifest_from_framework_context(
    monkeypatch,
):
    calls = []
    adapter = SimpleNamespace(
        framework_name=lambda: "fakefw",
        framework_version=lambda: "fakefw-v1",
        adapter_version=lambda: "adapter-v1",
        serving_abi_version=lambda _model_config: "abi-v1",
    )
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    recipe = SimpleNamespace(topology_ref=object(), member_ref=object())
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={"framework": "fakefw"},
        identity_payload={"rank": 0},
    )
    model_config = SimpleNamespace(
        model="fake-model",
        compute_hash=lambda: "model-hash",
    )

    monkeypatch.setattr(
        integration_mod,
        "canonical_index_from_recipe",
        lambda seen_recipe: calls.append(("canonical", seen_recipe)) or "canonical",
    )
    monkeypatch.setattr(
        integration_mod,
        "compute_serving_tensor_schema_hash",
        lambda canonical, **kwargs: calls.append(("schema", canonical, kwargs))
        or "schema-hash",
    )
    monkeypatch.setattr(
        integration_mod,
        "compute_runtime_representation_contract_hash",
        lambda **kwargs: calls.append(("repr", kwargs)) or "repr-hash",
    )
    monkeypatch.setattr(
        integration_mod,
        "logical_topology_json_from_recipe",
        lambda seen_recipe, **kwargs: calls.append(("topology", seen_recipe, kwargs))
        or '{"topology": true}',
    )
    monkeypatch.setattr(
        integration_mod,
        "prepare_same_binding_manifest_carrier",
        lambda seen_recipe, **kwargs: calls.append(("carrier", seen_recipe, kwargs))
        or ("manifest-ref", b"manifest"),
    )

    result = integration.build_local_ready_manifest_carrier_from_framework_context(
        recipe=recipe,
        manifest_tensor_name="__tensorcast_meta__.manifest",
        model_config=model_config,
        placement=placement,
        runtime_binding_schema_version=3,
        serving_artifact_schema_version=4,
    )

    assert result == ("manifest-ref", b"manifest")
    assert calls[2] == (
        "repr",
        {
            "tensor_schema_hash": "schema-hash",
            "topology_ref": placement.topology,
            "member_ref": placement.member,
            "framework_name": "fakefw",
            "framework_version": "fakefw-v1",
            "adapter_version": "adapter-v1",
            "serving_abi_version": "abi-v1",
            "source_identity": {
                "model_hash": "model-hash",
                "model_name": "fake-model",
                "runtime_binding_schema_version": 3,
                "serving_artifact_schema_version": 4,
                "placement": placement.identity_payload,
            },
        },
    )
    assert calls[-1] == (
        "carrier",
        recipe,
        {
            "manifest_tensor_name": "__tensorcast_meta__.manifest",
            "representation_contract_hash": "repr-hash",
            "logical_topology_json_payload": '{"topology": true}',
            "topology_admission_digest": "digest",
        },
    )


def test_serving_integration_prepares_manifest_carrier_result(monkeypatch):
    adapter = SimpleNamespace(
        framework_name=lambda: "fakefw",
        framework_version=lambda: "fakefw-v1",
        adapter_version=lambda: "adapter-v1",
        serving_abi_version=lambda _model_config: "abi-v1",
    )
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={},
        identity_payload={},
    )
    carrier_bytes = b"manifest-bytes"

    monkeypatch.setattr(
        ServingIntegration,
        "build_local_ready_manifest_carrier_from_framework_context",
        lambda _self, **_kwargs: ("repr-hash", carrier_bytes),
    )
    monkeypatch.setattr(
        integration_mod.ServingArtifactManifest,
        "from_bytes",
        lambda seen: SimpleNamespace(
            serving_manifest_ref=f"manifest:{seen!r}",
            serving_build_digest="build-digest",
        ),
    )

    result = integration.prepare_local_ready_manifest_carrier_from_framework_context(
        recipe=SimpleNamespace(),
        manifest_tensor_name="__tensorcast_meta__.manifest",
        model_config=SimpleNamespace(model="fake"),
        placement=placement,
        runtime_binding_schema_version=1,
        serving_artifact_schema_version=2,
    )

    assert isinstance(result, LocalReadyManifestCarrierResult)
    assert result.representation_contract_hash == "repr-hash"
    assert result.manifest_bytes == carrier_bytes
    assert result.serving_manifest_ref == "manifest:b'manifest-bytes'"
    assert result.serving_build_digest == "build-digest"


def test_serving_integration_builds_local_ready_binding_contract(monkeypatch):
    integration = ServingIntegration()
    monkeypatch.setattr(
        integration_mod,
        "compute_serving_binding_tensor_schema_hash",
        lambda *_args, **_kwargs: "schema-hash",
    )
    recipe = SimpleNamespace(
        tensor_schema=(
            TensorSchemaEntry(
                name="w",
                dtype="torch.float32",
                shape=(2,),
                stride=(1,),
            ),
        ),
        realization_plan_proto=b"plan",
        realization_plan_count=1,
        realization_plan=(),
        realization_fallback_plan=(),
    )

    contract = integration.build_local_ready_binding_contract(
        recipe=recipe,
        canonical_tensors={"w": torch.ones((2,), dtype=torch.float32)},
        runtime_only_tensor_names=("runtime_only",),
        manifest_tensor_name="__tensorcast_meta__.manifest",
        representation_contract_hash_factory=lambda tensor_schema_hash: (
            f"repr:{tensor_schema_hash}"
        ),
    )

    assert isinstance(contract, LocalReadyBindingContract)
    assert contract.excluded_names == ("runtime_only",)
    assert contract.canonical_tensor_names == ("w",)
    assert contract.tensor_schema_hash == "schema-hash"
    assert contract.representation_contract_hash == "repr:schema-hash"
    assert contract.realization_plan_proto == b"plan"
    assert contract.realization_entry_count == 1
    assert contract.fallback_copy_plan == ()


def test_serving_integration_owns_local_ready_recipe_fields():
    integration = ServingIntegration()
    recipe = SimpleNamespace(
        trace_plan=SimpleNamespace(
            copy_plan=(1, 2),
            expected_src_names={"src"},
            expected_dst_names={"dst"},
            tensorcast_slices=(),
        ),
        tensor_schema=("w",),
        realization_plan_count=4,
        realization_plan=(),
        realization_fallback_plan=(1,),
        source_artifact_ref="mi2:test:source",
        source_metadata_fingerprint="meta-fingerprint",
        serving_facts=SimpleNamespace(
            process_after_load_class=FinalizeClass.REPRESENTATION_CHANGING
        ),
    )

    assert integration.local_ready_recipe_summary_fields(recipe) == {
        "tensor_schema_count": 1,
        "copy_plan_count": 2,
        "expected_src_count": 1,
        "expected_dst_count": 1,
        "tensorcast_slice_count": 0,
        "realization_plan_count": 4,
        "realization_fallback_count": 1,
    }
    identity = integration.local_ready_materialization_identity(recipe)
    assert isinstance(identity, LocalReadyMaterializationIdentity)
    assert identity.source_artifact_ref == "mi2:test:source"
    assert identity.source_metadata_fingerprint == "meta-fingerprint"
    assert integration.local_ready_requires_binding_finalize(recipe)


def test_local_ready_canonical_index_uses_cumulative_segment_offsets():
    entries = canonical_index_entries_from_tensor_schema(
        (
            TensorSchemaEntry(
                name="a",
                dtype="torch.float32",
                shape=(2,),
                stride=(1,),
            ),
            TensorSchemaEntry(
                name="b",
                dtype="torch.float16",
                shape=(3,),
                stride=(1,),
            ),
            TensorSchemaEntry(
                name="c",
                dtype="torch.uint8",
                shape=(5,),
                stride=(1,),
            ),
        )
    )

    assert [entry.name for entry in entries] == ["a", "b", "c"]
    assert [entry.segment_offset for entry in entries] == [0, 8, 14]
    assert [entry.size_bytes for entry in entries] == [8, 6, 5]


class _MaterializedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty((1,), device="meta"))
        self.runtime_only = nn.Parameter(torch.empty((1,), device="meta"))


class _MaterializationAdapter(TorchModuleAdapterMixin):
    def __init__(self, *, fail_finalize: bool = False) -> None:
        self.fail_finalize = fail_finalize
        self.events: list[tuple[str, object | None, str]] = []

    def runtime_only_tensor_names(self, model: nn.Module) -> tuple[str, ...]:
        del model
        return ("runtime_only",)

    def build_meta_model(self, framework_config, model_config):
        del framework_config, model_config
        return _MaterializedModel()

    def framework_name(self) -> str:
        return "fakefw"

    def framework_version(self) -> str:
        return "fakefw-v1"

    def adapter_version(self) -> str:
        return "adapter-v1"

    def serving_abi_version(self, model_config) -> str:
        del model_config
        return "abi-v1"

    def run_process_after_load(self, model, model_config, target_device):
        assert not model.w.is_meta
        self.events.append(("process", model_config, str(target_device)))

    def run_runtime_only_post_bind(self, model, model_config, target_device):
        assert not model.runtime_only.is_meta
        self.events.append(("finalize", model_config, str(target_device)))
        if self.fail_finalize:
            raise RuntimeError("finalize failed")

    def semantic_probes(self, model, model_config):
        assert not model.w.is_meta
        return {"model_name": getattr(model_config, "name", None)}


class _AdapterFrameworkHost:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def identity(self, model_config):
        return FrameworkIdentity(
            framework_name=str(self.adapter.framework_name()),
            framework_version=str(self.adapter.framework_version()),
            adapter_version=str(self.adapter.adapter_version()),
            serving_abi_version=str(self.adapter.serving_abi_version(model_config)),
        )

    def prepare_model_construction(self, framework_config, model_config):
        prepare = getattr(self.adapter, "prepare_model_construction", None)
        if callable(prepare):
            prepare(framework_config, model_config)

    def build_meta_model(self, framework_config, model_config):
        build = getattr(self.adapter, "build_meta_model", None)
        if callable(build):
            return build(framework_config, model_config)
        return _MaterializedModel()

    def build_runtime_model(self, framework_config, model_config, target_device):
        build = getattr(self.adapter, "build_runtime_model", None)
        if callable(build):
            return build(framework_config, model_config, target_device)
        model = self.build_meta_model(framework_config, model_config)
        return model.to(target_device) if hasattr(model, "to") else model

    def assert_model_ready_for_runtime_binding(self, model, *, context):
        check = getattr(self.adapter, "assert_model_ready_for_runtime_binding", None)
        if callable(check):
            check(model, context=context)

    def semantic_probes(self, model, model_config):
        semantic_probes = getattr(self.adapter, "semantic_probes", None)
        return (
            semantic_probes(model, model_config) if callable(semantic_probes) else None
        )

    def run_process_after_load(self, model, model_config, target_device):
        hook = getattr(self.adapter, "run_process_after_load", None)
        if callable(hook):
            hook(model, model_config, target_device)

    def run_runtime_only_post_bind(self, model, model_config, target_device):
        hook = getattr(self.adapter, "run_runtime_only_post_bind", None)
        if callable(hook):
            hook(model, model_config, target_device)

    def support_level(self, model, model_config):
        support_level = getattr(self.adapter, "support_level", None)
        if callable(support_level):
            return support_level(model, model_config)
        return ServingSupportLevel.RUNTIME_BIND_SWAP_READY

    def process_after_load_class(self, model, model_config):
        process_after_load = getattr(self.adapter, "process_after_load_class", None)
        if callable(process_after_load):
            return process_after_load(model, model_config)
        return FinalizeClass.RUNTIME_ONLY

    def post_bind_finalize_class(self, model, model_config):
        post_bind_finalize = getattr(self.adapter, "post_bind_finalize_class", None)
        if callable(post_bind_finalize):
            return post_bind_finalize(model, model_config)
        return FinalizeClass.RUNTIME_ONLY

    def trace_model_load(
        self, model, ordered_names, meta_by_name, *, debug_dump_trace=False
    ):
        trace = getattr(self.adapter, "trace_model_load", None)
        if callable(trace):
            return trace(
                model, ordered_names, meta_by_name, debug_dump_trace=debug_dump_trace
            )
        raise CapabilityMissingError("trace_model_load unavailable")

    def cleanup_after_recipe_build(self, model, model_config, *, framework_config=None):
        cleanup = getattr(self.adapter, "cleanup_after_recipe_build", None)
        if callable(cleanup):
            cleanup(model, model_config, framework_config=framework_config)

    def native_load_weights(self, model, weights):
        native_load = getattr(self.adapter, "native_load_weights", None)
        if not callable(native_load):
            raise CapabilityMissingError("native_load_weights unavailable")
        native_load(model, weights)


class _AdapterTensorSurface:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def runtime_only_tensor_names(self, model):
        return self.adapter.runtime_only_tensor_names(model)

    def align_runtime_tensor_names(self, model, expected_names):
        align = getattr(self.adapter, "align_runtime_tensor_names", None)
        if callable(align):
            return align(model, expected_names)
        return integration_mod.TorchTensorHost().align_runtime_tensor_names(
            model, expected_names
        )

    def collect_runtime_tensors(self, model, *, remove_duplicate=False):
        return self.adapter.collect_runtime_binding_tensors(
            model, remove_duplicate=remove_duplicate
        )

    def collect_runtime_tensor_view(self, tensors):
        return integration_mod.TorchTensorHost().collect_runtime_tensor_view(tensors)

    def compute_runtime_tensor_schema_hash(self, tensors, *, remove_duplicate=False):
        return self.adapter.compute_runtime_tensor_schema_hash(
            tensors, remove_duplicate=remove_duplicate
        )

    def attach_bound_tensors(self, model, tensors, *, replace_meta_params):
        return self.adapter.attach_bound_tensors(
            model, tensors, replace_meta_params=replace_meta_params
        )

    def allocate_runtime_only_tensors(self, model, target_device):
        return self.adapter.allocate_runtime_only_tensors(model, target_device)

    def snapshot_tensor_invariants(self, tensors):
        return self.adapter.snapshot_tensor_invariants(tensors)

    def validate_tensor_invariants(self, before, after):
        self.adapter.validate_tensor_invariants(before, after)


def _host_for_adapter(adapter, **kwargs) -> IntegrationHost:
    return IntegrationHost(
        framework=kwargs.pop("framework", _AdapterFrameworkHost(adapter)),
        placement=kwargs.pop("placement", _ContractPlacementHost()),
        tensor_surface=kwargs.pop("tensor_surface", _AdapterTensorSurface(adapter)),
        **kwargs,
    )


class _TransferableBinding:
    def __init__(self) -> None:
        self.closed = False
        self.transferred = False
        self.runtime_handle = SimpleNamespace(closed=False)
        self.runtime_handle.close = lambda: setattr(self.runtime_handle, "closed", True)

    def transfer_to_runtime(self):
        self.transferred = True
        return self.runtime_handle

    def close(self):
        self.closed = True


def _member() -> ServingBindingMemberRef:
    return ServingBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )


def _binding_ref() -> BindingValueRef:
    return BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )


def _authority() -> ParsedRetainedServingBindingAuthority:
    member = _member()
    binding_ref = _binding_ref()
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4096,
        scope_digest="scope-1",
    )
    return ParsedRetainedServingBindingAuthority(
        group_id="group-1",
        local_serving_ref="binding-local:binding-1:value-1",
        binding_value_ref=binding_ref,
        reservation_capability=capability,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4096,
        expected=RetainedServingBindingExpectedDigests(
            target_layout_hash="layout-hash",
            tensor_schema_hash="schema-hash",
            serving_build_digest="build-digest",
            resolved_spec_digest="spec-digest",
        ),
        readiness="serving_local_ready",
        verification_state="local_only",
    )


def test_framework_boundary_reexports_serving_identity_types():
    assert IntegrationBindingValueRef is BindingValueRef
    assert IntegrationServingBindingMemberRef is ServingBindingMemberRef
    assert SERVING_MANIFEST_TENSOR_NAME.startswith("__tensorcast_meta__.")
    assert FinalizeClass.RUNTIME_ONLY.value == "runtime_only"
    assert ServingSupportLevel.RUNTIME_BIND_SWAP_READY.value


def test_retained_binding_authority_uses_parsed_retained_authority():
    parsed = _authority()

    assert parsed.binding_value_ref.binding_id == "binding-1"
    assert parsed.reservation_capability.capability_id == "capability-1"
    assert parsed.daemon_id == "daemon-1"
    assert parsed.daemon_session_id == "session-1"
    assert parsed.device_uuid == "gpu-0"
    assert parsed.member.group_id == "group-1"
    assert parsed.expected.target_layout_hash == "layout-hash"
    assert parsed.expected.tensor_schema_hash == "schema-hash"
    assert parsed.expected.serving_build_digest == "build-digest"
    assert parsed.expected.resolved_spec_digest == "spec-digest"
    assert parsed.readiness == "serving_local_ready"
    assert parsed.local_serving_ref == "binding-local:binding-1:value-1"


def test_serving_integration_p15_request_contract_smoke():
    closed: list[str] = []
    state = RuntimeBindingState(
        binding=SimpleNamespace(close=lambda: closed.append("binding")),
        artifact_ref="mi2:test:serving",
        runtime_view=RuntimeBindingView(
            serving_artifact_ref="mi2:test:serving",
            tensor_schema_hash="schema-hash",
            readiness="loaded",
        ),
    )
    state.close()
    assert closed == ["binding"]

    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={"framework": "fakefw"},
        identity_payload={"rank": 0},
    )
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(
        host=_host_for_adapter(adapter),
        profile_sink=lambda _event: None,
    )
    assert integration.host.framework.identity(None).framework_name == "fakefw"

    identity = ServingBindingPlan(
        model_hash="hash",
        model_id="fake-model",
        model_revision=None,
        dtype="torch.float16",
        runtime_version="fake-runtime-v1",
        framework_name="fakefw",
        framework_version="fakefw-v1",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        trace_cache_schema_version=1,
        tp_rank=0,
        tp_world_size=1,
        placement=placement.stable_identity_payload(),
    )
    session = integration.build_recipe_session(
        RecipeBuildSessionRequest(identity=identity)
    )
    assert session.recipe_cache_key(metadata_fingerprint="meta")

    request_and_method = (
        (
            _LocalReadyBootstrap(
                source_selector=SourceSelector.local_path("/tmp/model")
            ),
            integration._prepare_local_source_bootstrap,
        ),
    )
    for request, method in request_and_method:
        with pytest.raises(ServingIntegrationNotImplementedError):
            method(request)


def test_serving_integration_builds_recipe_session_identity_from_request():
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={"framework": "fakefw"},
        identity_payload={"rank": 0},
    )
    adapter = SimpleNamespace(
        framework_name=lambda: "fakefw",
        framework_version=lambda: "fakefw-v1",
        adapter_version=lambda: "adapter-v1",
        serving_abi_version=lambda _model_config: "abi-v1",
    )
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    model_config = SimpleNamespace(
        model="fake-model",
        revision="rev-a",
        dtype="torch.float16",
        compute_hash=lambda: "model-hash",
    )

    session = integration.build_recipe_session(
        RecipeBuildSessionRequest(
            model_config=model_config,
            placement=placement,
            trace_cache_schema_version=9,
        )
    )

    assert session.identity.model_hash == "model-hash"
    assert session.identity.model_id == "fake-model"
    assert session.identity.framework_name == "fakefw"
    assert session.identity.framework_version == "fakefw-v1"
    assert session.identity.adapter_version == "adapter-v1"
    assert session.identity.serving_abi_version == "abi-v1"
    assert session.identity.trace_cache_schema_version == 9
    assert session.identity.topology_ref is placement.topology
    assert session.identity.member_ref is placement.member
    assert session.identity.placement == placement.stable_identity_payload()


def test_serving_integration_load_and_reload_use_materialization():
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-a",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-a",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-a",
        ),
    )

    load_result = integration._load_existing_serving_artifact(
        _DirectServingLoad(
            resolved_artifact=resolved,
            framework_config=SimpleNamespace(name="framework"),
            model_config=SimpleNamespace(name="model"),
            target_device=torch.device("cpu"),
        )
    )

    assert isinstance(load_result, ServingLoadResult)
    assert isinstance(load_result.runtime_state, RuntimeBindingState)
    assert load_result.runtime_view.serving_artifact_ref == "mi2:test:serving-a"
    assert load_result.runtime_view.source_artifact_ref == "mi2:test:source"
    assert load_result.runtime_view.representation_contract_hash == "repr-a"
    assert load_result.runtime_view.readiness == "serving"
    assert torch.equal(
        load_result.model.w.detach(), torch.ones((1,), dtype=torch.float32)
    )
    assert adapter.events == [
        ("process", SimpleNamespace(name="model"), "cpu"),
        ("finalize", SimpleNamespace(name="model"), "cpu"),
    ]

    next_resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-b",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-b",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-b",
        ),
    )
    reload_result = integration._reload_existing_serving_artifact(
        _ServingReload(
            current_state=load_result.runtime_state,
            resolved_artifact=next_resolved,
            model=load_result.model,
            framework_config=SimpleNamespace(name="framework"),
            model_config=SimpleNamespace(name="model"),
        )
    )

    assert isinstance(reload_result, ServingReloadResult)
    assert reload_result.runtime_view.serving_artifact_ref == "mi2:test:serving-b"
    assert reload_result.runtime_view.representation_contract_hash == "repr-b"
    swapped_artifact = load_result.runtime_state.binding.swapped[0]
    assert isinstance(swapped_artifact, _Subset)
    assert swapped_artifact.names == ("w",)
    assert torch.equal(
        load_result.model.w.detach(), torch.full((1,), 2.0, dtype=torch.float32)
    )


def test_serving_integration_load_resolves_aligns_and_builds_materialization(
    monkeypatch,
):
    adapter = _MaterializationAdapter()
    align_calls = []
    adapter.align_runtime_tensor_names = (
        lambda model, names: align_calls.append(tuple(names)) or 0
    )
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving",
        tensor_names=("w",),
        tensor_schema_hash="schema-hash",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-a",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-a",
            to_runtime_policy=lambda: "manifest-policy",
        ),
    )
    calls = []

    class _Resolver:
        def resolve(self, artifact_ref):
            calls.append(("resolve", artifact_ref))
            return resolved

        def cross_check(self, resolved_artifact, **kwargs):
            calls.append(("cross_check", resolved_artifact, kwargs))
            return resolved_artifact

    def fake_build_options(self, **kwargs):
        calls.append(("options", kwargs))
        return "bind-options", {"profile": True}

    monkeypatch.setattr(
        ServingIntegration, "build_materialization_options", fake_build_options
    )

    result = ServingIntegration(
        resolver=_Resolver(),
        host=_host_for_adapter(adapter),
    )._load_existing_serving_artifact(
        _DirectServingLoad(
            artifact_ref="mi2:test:serving",
            target_device=torch.device("cpu"),
            configured_collective_policy="collective-policy",
            source_bound_contract_state=SimpleNamespace(
                source_bound_contract_ready=True
            ),
            source_bound_contract_path="/tmp/contract.json",
            execution_facts={"tp_rank": 0, "tp_world_size": 1},
            require_materialization_options=True,
        )
    )

    assert result.runtime_view.serving_artifact_ref == "mi2:test:serving"
    assert align_calls == [("w",)]
    assert calls == [
        ("resolve", "mi2:test:serving"),
        (
            "cross_check",
            resolved,
            {
                "expected_tensor_schema_hash": result.runtime_view.tensor_schema_hash,
                "serving_runtime_policy": "manifest-policy",
            },
        ),
        (
            "options",
            {
                "artifact_ref": "mi2:test:serving",
                "operation_scope": "startup.direct_serving_artifact.bind",
                "configured_policy": "collective-policy",
                "source_bound_contract_state": SimpleNamespace(
                    source_bound_contract_ready=True
                ),
                "source_bound_contract_path": "/tmp/contract.json",
                "execution_facts": {
                    "tp_rank": 0,
                    "tp_world_size": 1,
                },
                "contract_identity": "repr-a",
            },
        ),
    ]


def test_serving_integration_reload_resolves_and_cross_checks_artifact_ref():
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-next",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-next",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-next",
            to_runtime_policy=lambda: "manifest-policy",
        ),
    )
    calls = []

    class _Resolver:
        def resolve(self, artifact_ref):
            calls.append(("resolve", artifact_ref))
            return resolved

        def cross_check(self, resolved_artifact, **kwargs):
            calls.append(("cross_check", resolved_artifact, kwargs))
            return resolved_artifact

    binding = _Bound()
    integration = ServingIntegration(resolver=_Resolver())
    current_state = RuntimeBindingState(
        binding=binding,
        artifact_ref="mi2:test:serving-current",
        runtime_view=RuntimeBindingView(tensor_schema_hash="schema-hash"),
    )

    result = integration._reload_existing_serving_artifact(
        _ServingReload(
            current_state=current_state,
            artifact_ref="mi2:test:serving-next",
            target_device=torch.device("cpu"),
        )
    )

    assert result.resolved_artifact is resolved
    assert result.runtime_view.serving_artifact_ref == "mi2:test:serving-next"
    assert isinstance(binding.swapped[0], _Subset)
    assert binding.swapped[0].names == ("w",)
    assert calls == [
        ("resolve", "mi2:test:serving-next"),
        (
            "cross_check",
            resolved,
            {
                "expected_tensor_schema_hash": "schema-hash",
                "serving_runtime_policy": "manifest-policy",
            },
        ),
    ]


def test_serving_integration_resolves_ranked_locator_with_placement_member(
    monkeypatch,
):
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-rank-1",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-next",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-next",
        ),
    )
    calls = []

    class _Resolver:
        def resolve(self, artifact_ref):
            calls.append(("resolve", artifact_ref))
            return resolved

    member = ServingBindingMemberRef(
        member_id="dp0:pp0:tp1",
        member_index=1,
        member_count=2,
        group_id="group-1",
    )
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            group_id="group-1",
            schema_topology_digest="topology-digest",
            logical_topology_ref="fake://topology",
        ),
        member=member,
        framework_payload={"family": "fake"},
        identity_payload={"rank": 1},
    )

    class _RuntimeContext:
        def resolve_key_mapping_cached(self, *, key):
            calls.append(("key", key))
            return "mi2:test:serving-rank-1", None

    monkeypatch.setattr(
        "tensorcast.api.store.get_runtime_context", lambda: _RuntimeContext()
    )

    result = ServingIntegration(resolver=_Resolver())._resolved_artifact(
        resolved_artifact=None,
        artifact_ref=None,
        artifact_locator=ServingArtifactLocator.ranked_version_key(
            "models/demo/serving/v1"
        ),
        expected_tensor_schema_hash=None,
        serving_runtime_policy=None,
        placement=placement,
    )

    assert result is resolved
    assert calls == [
        ("key", "models/demo/serving/v1/members/dp0:pp0:tp1"),
        ("resolve", "mi2:test:serving-rank-1"),
    ]


def test_serving_integration_rejects_resolved_artifact_ref_mismatch():
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-rank-0",
        manifest=SimpleNamespace(),
    )

    with pytest.raises(ManifestMismatchError, match="artifact ref mismatch"):
        ServingIntegration()._resolved_artifact(
            resolved_artifact=resolved,
            artifact_ref="mi2:test:serving-rank-1",
            artifact_locator=None,
            expected_tensor_schema_hash=None,
            serving_runtime_policy=None,
        )


def test_serving_integration_accepts_matching_topology_digest_and_logical_topology():
    placement = _matrix_placement(tp_size=2, eplb_digest="eplb-a")
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-rank-0",
        manifest=SimpleNamespace(
            topology_admission_digest=placement.topology.schema_topology_digest,
            logical_topology_json=logical_topology_json(
                placement.topology,
                framework_payload=placement.framework_payload,
            ),
        ),
    )

    assert (
        ServingIntegration()._resolved_artifact(
            resolved_artifact=resolved,
            artifact_ref="mi2:test:serving-rank-0",
            artifact_locator=None,
            expected_tensor_schema_hash=None,
            serving_runtime_policy=None,
            placement=placement,
        )
        is resolved
    )


@pytest.mark.parametrize(
    ("current_placement", "match"),
    [
        (_matrix_placement(tp_size=4), "topology admission digest mismatch"),
        (_matrix_placement(pp_size=2), "topology admission digest mismatch"),
        (_matrix_placement(dp_size=2), "topology admission digest mismatch"),
        (
            _matrix_placement(tp_size=2, eplb_digest="eplb-b"),
            "topology admission digest mismatch",
        ),
    ],
)
def test_serving_integration_rejects_topology_mismatch_matrix(
    current_placement,
    match,
):
    published_placement = _matrix_placement(tp_size=2, eplb_digest="eplb-a")
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-rank-0",
        manifest=SimpleNamespace(
            topology_admission_digest=(
                published_placement.topology.schema_topology_digest
            ),
            logical_topology_json=logical_topology_json(
                published_placement.topology,
                framework_payload=published_placement.framework_payload,
            ),
        ),
    )

    with pytest.raises(ManifestMismatchError, match=match):
        ServingIntegration()._resolved_artifact(
            resolved_artifact=resolved,
            artifact_ref="mi2:test:serving-rank-0",
            artifact_locator=None,
            expected_tensor_schema_hash=None,
            serving_runtime_policy=None,
            placement=current_placement,
        )


def test_serving_integration_rejects_logical_topology_mismatch_without_digest():
    published_placement = _matrix_placement(tp_size=2)
    current_placement = _matrix_placement(tp_size=4)
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-rank-0",
        manifest=SimpleNamespace(
            logical_topology_json=logical_topology_json(
                published_placement.topology,
                framework_payload=published_placement.framework_payload,
            ),
        ),
    )

    with pytest.raises(ManifestMismatchError, match="logical topology mismatch"):
        ServingIntegration()._resolved_artifact(
            resolved_artifact=resolved,
            artifact_ref="mi2:test:serving-rank-0",
            artifact_locator=None,
            expected_tensor_schema_hash=None,
            serving_runtime_policy=None,
            placement=current_placement,
        )


def test_serving_integration_reload_builds_materialization_options(monkeypatch):
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-next",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-next",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-next",
        ),
    )
    calls = []

    class _Resolver:
        def resolve(self, artifact_ref):
            return resolved

        def cross_check(self, resolved_artifact, **kwargs):
            return resolved_artifact

    def fake_build_options(self, **kwargs):
        calls.append(kwargs)
        return "swap-options", {"profile": True}

    monkeypatch.setattr(
        ServingIntegration, "build_materialization_options", fake_build_options
    )
    binding = _Bound()
    current_state = RuntimeBindingState(
        binding=binding,
        artifact_ref="mi2:test:serving-current",
        runtime_view=RuntimeBindingView(tensor_schema_hash="schema-hash"),
    )

    ServingIntegration(resolver=_Resolver())._reload_existing_serving_artifact(
        _ServingReload(
            current_state=current_state,
            artifact_ref="mi2:test:serving-next",
            target_device=torch.device("cpu"),
            configured_collective_policy="collective-policy",
            source_bound_contract_state=SimpleNamespace(
                source_bound_contract_ready=True
            ),
            source_bound_contract_path="/tmp/contract.json",
            execution_facts={"tp_rank": 0, "tp_world_size": 1},
            contract_identity="contract-id",
            require_materialization_options=True,
        )
    )

    assert binding.swapped[1]["options"] == "swap-options"
    assert calls == [
        {
            "artifact_ref": "mi2:test:serving-next",
            "operation_scope": "runtime_binding.swap",
            "configured_policy": "collective-policy",
            "source_bound_contract_state": SimpleNamespace(
                source_bound_contract_ready=True
            ),
            "source_bound_contract_path": "/tmp/contract.json",
            "execution_facts": {
                "tp_rank": 0,
                "tp_world_size": 1,
            },
            "contract_identity": "contract-id",
        }
    ]


def test_serving_integration_reload_rejects_non_swap_capable_binding():
    integration = ServingIntegration()
    current_state = RuntimeBindingState(
        binding=SimpleNamespace(),
        artifact_ref="mi2:test:serving-current",
        runtime_view=RuntimeBindingView(tensor_schema_hash="schema-hash"),
    )

    with pytest.raises(ServingIntegrationError, match="swap-capable"):
        integration._reload_existing_serving_artifact(
            _ServingReload(
                current_state=current_state,
                artifact_ref="mi2:test:serving-next",
                target_device=torch.device("cpu"),
            )
        )


def test_serving_integration_reload_host_infers_target_device_from_model(monkeypatch):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-next",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-next",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-next",
            to_runtime_policy=lambda: "manifest-policy",
        ),
    )

    class _Resolver:
        def resolve(self, artifact_ref):
            assert artifact_ref == "mi2:test:serving-next"
            return resolved

        def cross_check(self, resolved_artifact, **_kwargs):
            return resolved_artifact

    monkeypatch.setattr(
        ServingIntegration,
        "build_materialization_options",
        lambda self, **_kwargs: ("swap-options", {}),
    )
    monkeypatch.setattr(
        integration_mod,
        "read_source_bound_contract_state",
        lambda: SimpleNamespace(source_bound_contract_ready=True),
    )
    binding = _Bound()
    current_state = RuntimeBindingState(
        binding=binding,
        artifact_ref="mi2:test:serving-current",
        runtime_view=RuntimeBindingView(tensor_schema_hash="schema-hash"),
    )

    attachment = ServingIntegration(
        resolver=_Resolver(),
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
            tensor_surface=integration_mod.TorchTensorHost(),
        ),
    ).reload(
        current_state,
        ExistingServingArtifact(
            ServingArtifactLocator.artifact_ref("mi2:test:serving-next")
        ),
        RequestContext(model_config=SimpleNamespace(model="fake")),
        model=_Model(),
    )

    assert attachment.state.runtime_view.serving_artifact_ref == "mi2:test:serving-next"
    assert binding.swapped[1]["options"] == "swap-options"


def test_serving_integration_load_prepared_local_ready_uses_restore(monkeypatch):
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    binding_ref = _binding_ref()

    class _PreparedRestored:
        tensors = {"w": torch.full((1,), 3.0, dtype=torch.float32)}
        binding_layout_id = "layout-1"
        binding_value_ref = binding_ref
        reservation_bytes = 2048

        def __init__(self) -> None:
            self.closed = False
            self.transferred = False
            self.runtime_handle = SimpleNamespace(close=lambda: None)

        def transfer_to_runtime(self):
            self.transferred = True
            return self.runtime_handle

        def close(self):
            self.closed = True

    restored = _PreparedRestored()

    @contextmanager
    def fake_restore_prepared(**kwargs):
        assert kwargs["expected_member"] == _member()
        yield restored

    monkeypatch.setattr(
        integration_mod, "restore_prepared_local_ready_binding", fake_restore_prepared
    )
    resolved = SimpleNamespace(
        artifact=_Artifact(),
        artifact_ref="mi2:test:serving-local",
        manifest=SimpleNamespace(
            representation_contract_hash="repr-local",
            source_artifact_ref="mi2:test:source",
            serving_build_digest="build-local",
            local_serving_ref="binding-local:binding-1:value-1",
        ),
    )

    result = integration._load_existing_serving_artifact(
        _DirectServingLoad(
            resolved_artifact=resolved,
            model_config=SimpleNamespace(name="model"),
            target_device=torch.device("cpu"),
            expected_member=_member(),
        )
    )

    assert result.runtime_view.readiness == "serving_local_ready"
    assert result.runtime_view.binding_value_ref == binding_ref
    assert torch.equal(result.model.w.detach(), torch.full((1,), 3.0))
    assert restored.transferred
    assert not restored.closed


def test_serving_integration_error_taxonomy_is_structured():
    assert issubclass(ManifestMismatchError, ServingIntegrationError)
    assert issubclass(SchemaMismatchError, ServingIntegrationError)
    assert issubclass(AdmissionRejectedError, ServingIntegrationError)
    assert issubclass(AuthorityValidationError, ServingIntegrationError)
    assert issubclass(CapabilityMissingError, ServingIntegrationError)
    assert issubclass(PlacementAdmissionError, ServingIntegrationError)
    assert issubclass(ArtifactLocatorResolutionError, ServingIntegrationError)
    assert issubclass(SourceProviderError, ServingIntegrationError)
    error = SchemaMismatchError(
        "bad schema",
        operation="reload",
        details={"artifact_locator": "mi2:serving"},
    )
    assert error.code == "schema_mismatch"
    assert error.operation == "reload"
    assert not error.retryable
    assert error.worker_suspect
    assert error.details == {"artifact_locator": "mi2:serving"}


def test_public_runtime_package_boundary_hides_admin_helpers():
    import tensorcast.serving as serving
    import tensorcast.serving.admin as serving_admin
    import tensorcast.serving.hosts as serving_hosts
    import tensorcast.serving.policy as serving_policy
    import tensorcast.serving.runtime as serving_runtime
    from tensorcast.serving.testing import (
        assert_framework_isolation,
        assert_public_runtime_boundary,
    )

    assert serving_runtime.ServingRuntimeSession is ServingRuntimeSession
    assert serving_runtime.ServingConfig is ServingConfig
    assert serving_runtime.ServingArtifactLocator is ServingArtifactLocator
    assert ServingArtifactLocator is serving_policy.ServingArtifactLocator
    assert serving_runtime.ServingPolicy is serving_policy.ServingPolicy
    assert integration_mod.ServingPolicy is serving_policy.ServingPolicy
    assert "ServingRuntimeSession" in serving_runtime.__all__
    assert "FrameworkAdapter" not in serving.__all__
    assert not hasattr(integration_mod, "FrameworkAdapter")
    assert not hasattr(ServingIntegration, "framework_adapter")
    assert "AdminLocalSourceBootstrap" not in serving_runtime.__all__
    assert "_AdminLocalSourceBootstrap" not in serving_runtime.__all__
    assert "bind_serving_artifact" not in serving_runtime.__all__
    assert not hasattr(ServingIntegration, "bind")
    assert not hasattr(ServingIntegration, "swap")
    assert not hasattr(ServingIntegration, "restore_retained")
    assert not hasattr(ServingIntegration, "restore_prepared_local_ready")
    assert serving_admin.AdminLocalSourceBootstrap is AdminLocalSourceBootstrap
    assert serving_hosts.IntegrationHost is IntegrationHost
    assert serving_hosts.SourceHost is integration_mod.SourceHost
    assert serving_hosts.RecipeCachePolicy is RecipeCachePolicy
    assert serving.PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION == (
        PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION
    )
    assert serving_hosts.PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION == (
        PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION
    )
    assert serving_hosts.SOURCE_CATALOG_REQUEST_SCHEMA_VERSION == (
        SOURCE_CATALOG_REQUEST_SCHEMA_VERSION
    )
    assert serving.SOURCE_CATALOG_SCHEMA_VERSION == SOURCE_CATALOG_SCHEMA_VERSION
    assert serving_hosts.SOURCE_CATALOG_SCHEMA_VERSION == (
        SOURCE_CATALOG_SCHEMA_VERSION
    )

    assert_public_runtime_boundary(serving_runtime)
    assert_framework_isolation(
        ("tensorcast.serving.runtime", "tensorcast.serving.hosts")
    )


def test_serving_root_facade_is_explicit_and_curated():
    import tensorcast.serving as serving

    hidden_names = {
        "bind_serving_artifact",
        "swap_serving_artifact",
        "restore_retained_binding",
        "build_materialization_execution_context",
        "compile_recipe_from_inputs",
        "load_compiled_recipe_cache",
        "write_compiled_recipe_cache",
        "materialize_recipe_copy_plan_tensors",
        "complete_pure_transform_recipe_publication",
        "build_binding_finalize_build_intent",
        "build_pure_transform_build_intent",
        "RecipeBuildSession",
        "RecipePublicationContext",
        "CompiledServingRecipe",
        "PublishedReplicaProjection",
        "ReloadResponseProjection",
        "RuntimeEndpointProjection",
        "SourceSelectionProjection",
        "WeightVersionProjection",
        "RuntimeAttachmentStore",
        "RuntimeAttachmentRecord",
        "ModelAttributeRuntimeState",
        "ReadinessInventoryAdmissionPolicy",
        "aggregate_runtime_view_outputs",
        "publication_aggregate",
    }
    assert hidden_names.isdisjoint(serving.__all__)
    assert "ServingRuntimeSession" not in serving.__all__
    assert "IntegrationHost" in serving.__all__
    assert "ConformanceResult" not in serving.__all__
    assert "ServingConfig" in serving.__all__
    assert "RuntimeSettings" in serving.__all__
    assert not hasattr(serving, "__getattr__")

    hidden_name = "CompiledServingRecipe"
    with pytest.raises(AttributeError):
        getattr(serving, hidden_name)


def test_source_subject_broadcast_round_trips_non_public_subjects():
    subject_payload = {"kind": "fake-source", "value": "payload"}
    subject = SourceSubject(
        artifact_ref="mi2:test:source",
        subject=subject_payload,
        source_kind="fake",
        metadata_fingerprint="meta",
    )

    payload = source_subject_broadcast_payload(subject)
    assert payload == {
        "kind": "fake",
        "artifact_ref": "mi2:test:source",
        "subject": subject_payload,
        "metadata_fingerprint": "meta",
    }

    restored = source_subject_from_broadcast_payload(payload)
    assert restored.artifact_ref == "mi2:test:source"
    assert restored.source_kind == "fake"
    assert restored.subject == subject_payload
    assert restored.profile_fields() == {
        "artifact_ref": "mi2:test:source",
        "source_kind": "fake",
        "metadata_fingerprint": "meta",
    }

    integration = ServingIntegration()
    payload = integration.source_subject_broadcast_payload(subject)
    assert integration.source_subject_from_broadcast_payload(payload) == restored


def test_serving_integration_resolve_source_subject_uses_coordinator(monkeypatch):
    subject = SourceSubject(
        artifact_ref="mi2:test:source",
        subject={"kind": "fake-source"},
        source_kind="fake",
        metadata_fingerprint="meta",
    )
    calls = []

    monkeypatch.setattr(
        integration_mod,
        "resolve_source_subject",
        lambda path, *, verify_checksums: subject,
    )

    class _Coordinator:
        source_rank = 0

        @staticmethod
        def should_coordinate():
            return True

        @staticmethod
        def is_source_rank():
            return True

        @staticmethod
        def broadcast_object(payload, *, src):
            calls.append((payload, src))
            return payload

    resolved = ServingIntegration().resolve_source_subject(
        SourceSelector.local_path("/tmp/model"),
        verify_checksums=True,
        coordinator=_Coordinator(),
    )

    assert resolved == subject
    assert calls == [
        (
            source_subject_broadcast_payload(subject),
            0,
        )
    ]


def test_runtime_binding_materialization_attaches_and_transfers_ownership():
    model = _MaterializedModel()
    adapter = _MaterializationAdapter()
    binding = _TransferableBinding()
    profile_events = []
    context = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={},
        identity_payload={},
    )

    state = RuntimeBindingMaterialization(
        host=_host_for_adapter(adapter),
        profile_sink=profile_events.append,
    ).attach_and_finalize(
        model=model,
        tensors={"w": torch.ones((2,), dtype=torch.float32)},
        binding_handle=binding,
        context=SimpleNamespace(placement=context),
        state_seed=RuntimeStateSeed(
            artifact_ref="mi2:test:serving",
            serving_artifact_ref="mi2:test:serving",
            representation_contract_hash="repr",
            tensor_schema_hash="schema",
            readiness="loaded",
        ),
        replace_meta_params=True,
        target_device=torch.device("cpu"),
        model_config=SimpleNamespace(name="model-config"),
    )

    assert state.binding is binding
    assert state.ownership_handle is binding.runtime_handle
    assert state.runtime_view.serving_artifact_ref == "mi2:test:serving"
    assert torch.equal(model.w.detach(), torch.ones((2,)))
    assert not model.runtime_only.is_meta
    assert binding.transferred
    assert not binding.closed
    assert adapter.events == [
        ("process", SimpleNamespace(name="model-config"), "cpu"),
        ("finalize", SimpleNamespace(name="model-config"), "cpu"),
    ]
    assert [event["event"] for event in profile_events] == [
        "runtime_materialization.attach.start",
        "runtime_materialization.attach.done",
    ]


def test_runtime_binding_materialization_closes_binding_on_finalize_failure():
    model = _MaterializedModel()
    binding = _TransferableBinding()

    with pytest.raises(AttachFinalizeError, match="attach/finalize failed"):
        adapter = _MaterializationAdapter(fail_finalize=True)
        RuntimeBindingMaterialization(
            host=_host_for_adapter(adapter),
        ).attach_and_finalize(
            model=model,
            tensors={"w": torch.ones((2,), dtype=torch.float32)},
            binding_handle=binding,
            context=SimpleNamespace(),
            state_seed=RuntimeStateSeed(artifact_ref="mi2:test:serving"),
            replace_meta_params=True,
            target_device=torch.device("cpu"),
        )

    assert binding.closed
    assert not binding.transferred


def test_runtime_binding_materialization_closes_runtime_handle_on_state_failure():
    model = _MaterializedModel()
    binding = _TransferableBinding()

    def failing_state_factory(**_kwargs):
        raise RuntimeError("state failed")

    with pytest.raises(OwnershipTransferError, match="state construction"):
        adapter = _MaterializationAdapter()
        RuntimeBindingMaterialization(
            host=_host_for_adapter(adapter),
            state_factory=failing_state_factory,
        ).attach_and_finalize(
            model=model,
            tensors={"w": torch.ones((2,), dtype=torch.float32)},
            binding_handle=binding,
            context=SimpleNamespace(),
            state_seed=RuntimeStateSeed(artifact_ref="mi2:test:serving"),
            replace_meta_params=True,
            target_device=torch.device("cpu"),
        )

    assert binding.transferred
    assert binding.runtime_handle.closed
    assert not binding.closed


class _LocalReadyBinding:
    def __init__(self, tensors=None) -> None:
        self.tensors = tensors or {"w": torch.ones((1,), dtype=torch.float32)}
        self.binding_layout_id = "layout-1"
        self.closed = False
        self.freeze_calls = []

    def freeze_current(self, **kwargs):
        from tensorcast.proto.daemon.v2 import store_daemon_pb2

        self.freeze_calls.append(kwargs)
        return SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id=self.binding_layout_id,
            binding_value_id="value-1",
            seal_generation=1,
            local_serving_ref="binding-local:binding-1:value-1",
            verification_state=(
                store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY
            ),
            verification_job_id="verify-1",
        )

    def close(self):
        self.closed = True


def _local_ready_recipe() -> SimpleNamespace:
    return SimpleNamespace(
        trace_plan=SimpleNamespace(expected_dst_names={"w"}),
        source_artifact_ref="mi2:test:source",
        source_metadata_fingerprint="meta-fingerprint",
        tensor_schema=(
            TensorSchemaEntry(
                name="w",
                dtype="torch.float32",
                shape=(1,),
                stride=(1,),
            ),
        ),
        realization_plan_proto=b"plan",
        realization_plan_count=1,
        realization_plan=(),
        realization_fallback_plan=(),
    )


def _representation_changing_local_ready_recipe() -> SimpleNamespace:
    recipe = _local_ready_recipe()
    recipe.serving_facts = SimpleNamespace(
        process_after_load_class=FinalizeClass.REPRESENTATION_CHANGING
    )
    recipe.semantic_validation_spec = TensorcastSemanticValidationSpec(
        kind="explicit",
        payload={"model_name": "model-config"},
    )
    return recipe


def _local_ready_finalize_request(**overrides) -> _LocalReadyFinalize:
    request = {
        "model": _MaterializedModel(),
        "recipe": _local_ready_recipe(),
        "binding": _LocalReadyBinding(),
        "update_epoch": "epoch-1",
        "source_artifact_ref": "mi2:test:source",
        "serving_manifest_ref": "manifest-ref",
        "representation_contract_hash": "repr",
        "serving_build_digest": "build",
        "manifest_tensor_name": "__tensorcast_meta__.manifest",
        "source_bound_contract_state": SimpleNamespace(
            source_bound_contract_version=4,
            source_bound_capability_names=("collective",),
            source_bound_contract_ready=True,
        ),
        "source_bound_contract_path": "/tmp/contract.json",
        "target_device": torch.device("cpu"),
        "model_config": SimpleNamespace(name="model-config"),
    }
    request.update(overrides)
    return _LocalReadyFinalize(**request)


def test_serving_integration_finalizes_local_ready_runtime_in_core():
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    model = _MaterializedModel()
    model_config = SimpleNamespace(name="model-config")
    binding = _LocalReadyBinding()
    recipe = _local_ready_recipe()
    source_bound_contract_state = SimpleNamespace(
        source_bound_contract_version=4,
        source_bound_capability_names=("collective",),
        source_bound_contract_ready=True,
    )

    result = integration._finalize_local_ready_runtime(
        _LocalReadyFinalize(
            model=model,
            recipe=recipe,
            binding=binding,
            update_epoch="epoch-1",
            source_artifact_ref="mi2:test:source",
            serving_manifest_ref="manifest-ref",
            representation_contract_hash="repr",
            serving_build_digest="build",
            manifest_tensor_name="__tensorcast_meta__.manifest",
            source_bound_contract_state=source_bound_contract_state,
            source_bound_contract_path="/tmp/contract.json",
            target_device=torch.device("cpu"),
            model_config=model_config,
            family="dummy",
            tp_rank=0,
            tp_world_size=1,
        )
    )

    assert result.model is model
    assert result.recipe is recipe
    assert result.binding is binding
    assert result.current_value.local_serving_ref == ("binding-local:binding-1:value-1")
    assert result.runtime_view.readiness == "serving_local_ready"
    assert result.runtime_view.source_artifact_ref == "mi2:test:source"
    assert torch.equal(model.w.detach(), torch.ones((1,)))
    assert not model.runtime_only.is_meta
    assert not binding.closed
    assert binding.freeze_calls == [
        {
            "update_epoch": "epoch-1",
            "source_artifact_ref": "mi2:test:source",
        }
    ]
    assert adapter.events == [("finalize", model_config, "cpu")]


def test_serving_integration_validates_local_ready_representation_contract(monkeypatch):
    calls = []
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    model = _MaterializedModel()
    model_config = SimpleNamespace(
        model="fake-model",
        compute_hash=lambda: "model-hash",
    )
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={"framework": "fakefw"},
        identity_payload={"rank": 0},
    )

    monkeypatch.setattr(
        integration_mod,
        "compute_runtime_representation_contract_hash",
        lambda **kwargs: calls.append(kwargs) or "repr",
    )

    result = integration._finalize_local_ready_runtime(
        _LocalReadyFinalize(
            model=model,
            recipe=_local_ready_recipe(),
            binding=_LocalReadyBinding(),
            update_epoch="epoch-1",
            source_artifact_ref="mi2:test:source",
            serving_manifest_ref="manifest-ref",
            representation_contract_hash="repr",
            serving_build_digest="build",
            manifest_tensor_name="__tensorcast_meta__.manifest",
            source_bound_contract_state=SimpleNamespace(
                source_bound_contract_version=4,
                source_bound_capability_names=("collective",),
                source_bound_contract_ready=True,
            ),
            source_bound_contract_path="/tmp/contract.json",
            target_device=torch.device("cpu"),
            model_config=model_config,
            placement=placement,
            validate_representation_contract_hash=True,
            runtime_binding_schema_version=3,
            serving_artifact_schema_version=4,
        )
    )

    assert result.current_value is not None
    assert len(calls) == 1
    assert calls[0]["topology_ref"] == placement.topology
    assert calls[0]["member_ref"] == placement.member
    assert calls[0]["framework_name"] == "fakefw"
    assert calls[0]["framework_version"] == "fakefw-v1"
    assert calls[0]["adapter_version"] == "adapter-v1"
    assert calls[0]["serving_abi_version"] == "abi-v1"
    assert calls[0]["source_identity"] == {
        "model_hash": "model-hash",
        "model_name": "fake-model",
        "runtime_binding_schema_version": 3,
        "serving_artifact_schema_version": 4,
        "placement": placement.identity_payload,
    }


def test_serving_integration_closes_local_ready_binding_on_representation_drift(
    monkeypatch,
):
    binding = _LocalReadyBinding()
    model_config = SimpleNamespace(model="fake-model")
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={"framework": "fakefw"},
        identity_payload={"rank": 0},
    )
    monkeypatch.setattr(
        integration_mod,
        "compute_runtime_representation_contract_hash",
        lambda **_kwargs: "actual",
    )

    with pytest.raises(ManifestMismatchError, match="contract hash drifted"):
        adapter = _MaterializationAdapter()
        ServingIntegration(
            host=_host_for_adapter(adapter),
        )._finalize_local_ready_runtime(
            _LocalReadyFinalize(
                model=_MaterializedModel(),
                recipe=_local_ready_recipe(),
                binding=binding,
                update_epoch="epoch-1",
                source_artifact_ref="mi2:test:source",
                serving_manifest_ref="manifest-ref",
                representation_contract_hash="expected",
                serving_build_digest="build",
                manifest_tensor_name="__tensorcast_meta__.manifest",
                source_bound_contract_state=SimpleNamespace(
                    source_bound_contract_version=4,
                    source_bound_capability_names=("collective",),
                    source_bound_contract_ready=True,
                ),
                source_bound_contract_path="/tmp/contract.json",
                target_device=torch.device("cpu"),
                model_config=model_config,
                placement=placement,
                validate_representation_contract_hash=True,
                runtime_binding_schema_version=3,
                serving_artifact_schema_version=4,
            )
        )

    assert binding.closed


def test_serving_integration_rejects_representation_changing_finalize_without_semantic_validation():
    binding = _LocalReadyBinding()

    with pytest.raises(ServingIntegrationError, match="explicit semantic validation"):
        ServingIntegration(
            host=_host_for_adapter(_MaterializationAdapter())
        )._finalize_local_ready_runtime(
            _local_ready_finalize_request(
                recipe=_representation_changing_local_ready_recipe(),
                binding=binding,
                run_process_after_load=True,
                run_semantic_validation=False,
            )
        )

    assert binding.closed


def test_serving_integration_rejects_representation_changing_finalize_without_contract_validation():
    binding = _LocalReadyBinding()

    with pytest.raises(
        ServingIntegrationError,
        match="requires representation contract validation",
    ):
        ServingIntegration(
            host=_host_for_adapter(_MaterializationAdapter())
        )._finalize_local_ready_runtime(
            _local_ready_finalize_request(
                recipe=_representation_changing_local_ready_recipe(),
                binding=binding,
                run_process_after_load=True,
                run_semantic_validation=True,
            )
        )

    assert binding.closed


def test_serving_integration_rejects_representation_changing_finalize_without_ready_contract(
    monkeypatch,
):
    binding = _LocalReadyBinding()
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={"framework": "fakefw"},
        identity_payload={"rank": 0},
    )
    monkeypatch.setattr(
        ServingIntegration,
        "local_ready_representation_contract_hash",
        lambda _self, **_kwargs: "repr",
    )

    with pytest.raises(
        ServingIntegrationError,
        match="ready same-binding contract proof",
    ):
        ServingIntegration(
            host=_host_for_adapter(_MaterializationAdapter())
        )._finalize_local_ready_runtime(
            _local_ready_finalize_request(
                recipe=_representation_changing_local_ready_recipe(),
                binding=binding,
                run_process_after_load=True,
                run_semantic_validation=True,
                validate_representation_contract_hash=True,
                placement=placement,
                runtime_binding_schema_version=3,
                serving_artifact_schema_version=4,
                source_bound_contract_state=SimpleNamespace(
                    source_bound_contract_version=4,
                    source_bound_capability_names=("collective",),
                    source_bound_contract_ready=False,
                ),
            )
        )

    assert binding.closed


def test_serving_integration_prepare_local_ready_owns_contract_and_options(monkeypatch):
    adapter = _MaterializationAdapter()
    align_calls = []
    adapter.align_runtime_tensor_names = (
        lambda model, names: align_calls.append(tuple(names)) or 0
    )
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    calls = []
    source_bound_contract_state = SimpleNamespace(
        source_bound_contract_ready=True,
        source_bound_contract_version=4,
        source_bound_capability_names=("collective",),
    )

    def fake_build_options(self, **kwargs):
        calls.append(("options", kwargs))
        return "realize-options", {"profile": True}

    def fake_prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return SimpleNamespace(
            binding=_LocalReadyBinding(),
            update_epoch="epoch-1",
            layout=SimpleNamespace(binding_layout_id="layout-1"),
            realization_entry_count=1,
        )

    monkeypatch.setattr(
        ServingIntegration, "build_materialization_options", fake_build_options
    )
    monkeypatch.setattr(integration_mod, "prepare_local_ready_serving", fake_prepare)

    result = integration._prepare_local_source_bootstrap(
        _LocalReadyBootstrap(
            source_selector=None,
            recipe=_local_ready_recipe(),
            source_subject=SimpleNamespace(),
            model=_MaterializedModel(),
            target_device=torch.device("cpu"),
            manifest_tensor_name="__tensorcast_meta__.manifest",
            source_artifact_ref="mi2:test:source",
            serving_manifest_ref="manifest-ref",
            representation_contract_hash="repr",
            serving_build_digest="build",
            configured_collective_policy="collective-policy",
            source_bound_contract_state=source_bound_contract_state,
            source_bound_contract_path="/tmp/contract.json",
            execution_facts={"tp_rank": 0, "tp_world_size": 1},
            contract_identity="contract-id",
            require_materialization_options=True,
        )
    )

    assert result.binding is not None
    assert result.runtime_view is not None
    assert result.current_value is not None
    assert result.realization_entry_count == 1
    assert align_calls == [("w",)]
    assert calls[0] == (
        "options",
        {
            "artifact_ref": "mi2:test:source",
            "operation_scope": "bootstrap.same_binding_fast_path.tensorcast_realize",
            "configured_policy": "collective-policy",
            "source_bound_contract_state": source_bound_contract_state,
            "source_bound_contract_path": "/tmp/contract.json",
            "execution_facts": {
                "tp_rank": 0,
                "tp_world_size": 1,
            },
            "contract_identity": "contract-id",
        },
    )
    assert calls[1][0] == "prepare"
    assert calls[1][1]["options"] == "realize-options"


def test_serving_integration_local_ready_gets_execution_facts_from_host(monkeypatch):
    adapter = _MaterializationAdapter()

    class _PlacementWithExecutionFacts(_ContractPlacementHost):
        def execution_facts(self, framework_config):
            del framework_config
            return MaterializationExecutionFacts(
                collective_rank=0,
                collective_world_size=1,
                same_node_tensor_parallel=True,
                tensor_parallel_ranks=(0,),
            )

    class _Surface(integration_mod.TorchTensorHost):
        def runtime_only_tensor_names(self, model):
            del model
            return ("runtime_only",)

    integration = ServingIntegration(
        host=_host_for_adapter(
            adapter,
            placement=_PlacementWithExecutionFacts(),
            tensor_surface=_Surface(),
        )
    )
    calls = []
    source_bound_contract_state = SimpleNamespace(
        source_bound_contract_ready=True,
        source_bound_contract_version=4,
        source_bound_capability_names=("collective",),
    )

    monkeypatch.setattr(
        ServingIntegration,
        "build_materialization_options",
        lambda self, **kwargs: calls.append(kwargs) or ("realize-options", {}),
    )
    monkeypatch.setattr(
        integration_mod,
        "prepare_local_ready_serving",
        lambda **kwargs: SimpleNamespace(
            binding=_LocalReadyBinding(),
            update_epoch="epoch-1",
            layout=SimpleNamespace(binding_layout_id="layout-1"),
            realization_entry_count=1,
        ),
    )

    integration._prepare_local_source_bootstrap(
        _LocalReadyBootstrap(
            source_selector=None,
            recipe=_local_ready_recipe(),
            source_subject=SimpleNamespace(),
            framework_config="framework-config",
            model=_MaterializedModel(),
            target_device=torch.device("cpu"),
            manifest_tensor_name="__tensorcast_meta__.manifest",
            source_artifact_ref="mi2:test:source",
            serving_manifest_ref="manifest-ref",
            representation_contract_hash="repr",
            serving_build_digest="build",
            configured_collective_policy="collective-policy",
            source_bound_contract_state=source_bound_contract_state,
            source_bound_contract_path="/tmp/contract.json",
            require_materialization_options=True,
        )
    )

    assert calls[0]["execution_facts"] == {
        "collective_context_unavailable": False,
        "collective_rank": 0,
        "collective_world_size": 1,
        "same_node_tp": True,
        "tp_rank": 0,
        "tp_ranks": (0,),
        "tp_world_size": 1,
    }


def test_serving_integration_prepare_local_ready_builds_recipe_from_source(monkeypatch):
    calls = []
    source_handle = SimpleNamespace()
    source_subject = SourceSubject(
        artifact_ref="mi2:test:source",
        source_kind="fake",
        subject=source_handle,
        metadata_fingerprint="meta-fingerprint",
    )
    source_catalog = SimpleNamespace(
        ordered_names=("w",),
        meta_by_name={},
        selected_files=(),
        source_artifact_ref="mi2:test:source",
        metadata_fingerprint="meta-fingerprint",
    )
    cache_config = SimpleNamespace(trace_cache_schema_version=9)

    class _Provider:
        def build_catalog(self, request):
            calls.append(("source_catalog", request))
            return source_catalog

    class _Surface(integration_mod.TorchTensorHost):
        def runtime_only_tensor_names(self, model):
            del model
            return ("runtime_only",)

    class _Session:
        def build_recipe(self, **kwargs):
            calls.append(("build_recipe", kwargs))
            return SimpleNamespace(
                recipe=_local_ready_recipe(), diagnostics={"compile_key": "recipe-key"}
            )

    monkeypatch.setattr(
        ServingIntegration,
        "resolve_source_subject",
        lambda self, selector, **kwargs: calls.append(("resolve", selector, kwargs))
        or source_subject,
    )
    monkeypatch.setattr(
        ServingIntegration,
        "build_recipe_session",
        lambda self, request: calls.append(("session", request)) or _Session(),
    )
    monkeypatch.setattr(
        ServingIntegration,
        "build_materialization_options",
        lambda self, **kwargs: calls.append(("options", kwargs))
        or ("realize-options", {}),
    )
    monkeypatch.setattr(
        integration_mod,
        "prepare_local_ready_serving",
        lambda **kwargs: calls.append(("prepare", kwargs))
        or SimpleNamespace(
            binding=_LocalReadyBinding(),
            update_epoch="epoch-1",
            layout=SimpleNamespace(binding_layout_id="layout-1"),
            realization_entry_count=1,
        ),
    )

    result = ServingIntegration(
        host=IntegrationHost(
            framework=_ContractFrameworkHost(),
            placement=_ContractPlacementHost(),
            source_catalog=_Provider(),
            tensor_surface=_Surface(),
        ),
    )._prepare_local_source_bootstrap(
        _LocalReadyBootstrap(
            source_selector=SourceSelector.local_path("/tmp/model"),
            bootstrap=SimpleNamespace(verify_source_checksums=True),
            source_subject_coordinator="coordinator",
            framework_config="framework-config",
            model_config=SimpleNamespace(name="model-config"),
            target_device=torch.device("cpu"),
            model=_MaterializedModel(),
            manifest_tensor_name="__tensorcast_meta__.manifest",
            source_catalog_config="source-config",
            cache_config_factory=lambda *, source_catalog: cache_config,
            serving_manifest_ref="manifest-ref",
            representation_contract_hash="repr",
            serving_build_digest="build",
            configured_collective_policy="collective-policy",
            source_bound_contract_state=SimpleNamespace(
                source_bound_contract_ready=True,
                source_bound_contract_version=4,
                source_bound_capability_names=("collective",),
            ),
            source_bound_contract_path="/tmp/contract.json",
            execution_facts={"tp_rank": 0},
            require_materialization_options=True,
            build_recipe_from_framework_context=True,
        )
    )

    assert result.runtime_view is not None
    assert calls[0] == (
        "resolve",
        SourceSelector.local_path("/tmp/model"),
        {
            "verify_checksums": True,
            "coordinator": "coordinator",
        },
    )
    assert calls[1][0] == "source_catalog"
    assert calls[1][1].source_artifact_ref == "mi2:test:source"
    assert calls[1][1].source_catalog_config == "source-config"
    assert calls[1][1].framework_identity.framework_name == "fake"
    assert calls[2][0] == "session"
    assert calls[2][1].source_subject is source_subject
    assert calls[2][1].placement is not None
    assert calls[2][1].placement.member.member_id == "rank0"
    assert calls[3][0] == "build_recipe"
    assert calls[3][1]["source_catalog"] is source_catalog
    assert calls[3][1]["cache_config"] is cache_config
    assert calls[5][0] == "prepare"
    assert calls[5][1]["source_subject"] is source_handle


def test_serving_integration_prepare_local_ready_builds_framework_context(monkeypatch):
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    calls = []
    source_bound_contract_state = SimpleNamespace(
        source_bound_contract_ready=True,
        source_bound_contract_version=4,
        source_bound_capability_names=("collective",),
    )
    recipe = _representation_changing_local_ready_recipe()
    model_config = SimpleNamespace(name="model-config")
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="digest",
            logical_topology_ref="fake://topology",
        ),
        member=_member(),
        framework_payload={"framework": "fakefw"},
        identity_payload={"rank": 0},
    )

    def fake_prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return SimpleNamespace(
            binding=_LocalReadyBinding(),
            update_epoch="epoch-1",
            layout=SimpleNamespace(binding_layout_id="layout-1"),
            realization_entry_count=1,
        )

    monkeypatch.setattr(
        ServingIntegration,
        "prepare_local_ready_manifest_carrier_from_framework_context",
        lambda self, **kwargs: calls.append(("carrier", kwargs))
        or LocalReadyManifestCarrierResult(
            representation_contract_hash="repr",
            manifest_bytes=b"manifest",
            serving_manifest_ref="manifest-ref",
            serving_build_digest="build",
        ),
    )
    monkeypatch.setattr(
        ServingIntegration,
        "build_materialization_options",
        lambda self, **kwargs: ("realize-options", {}),
    )
    monkeypatch.setattr(
        ServingIntegration,
        "local_ready_representation_contract_hash",
        lambda self, **kwargs: "repr",
    )
    monkeypatch.setattr(integration_mod, "prepare_local_ready_serving", fake_prepare)

    result = integration._prepare_local_source_bootstrap(
        _LocalReadyBootstrap(
            source_selector=None,
            recipe=recipe,
            source_subject=SimpleNamespace(),
            framework_config="framework-config",
            model_config=model_config,
            target_device=torch.device("cpu"),
            manifest_tensor_name="__tensorcast_meta__.manifest",
            source_artifact_ref="mi2:test:source",
            source_bound_contract_state=source_bound_contract_state,
            source_bound_contract_path="/tmp/contract.json",
            configured_collective_policy="collective-policy",
            execution_facts={"tp_rank": 0},
            require_materialization_options=True,
            placement=placement,
            build_model_from_framework_context=True,
            build_manifest_carrier_from_framework_context=True,
            run_binding_finalize_hooks_when_required=True,
            validate_representation_contract_hash=True,
            runtime_binding_schema_version=3,
            serving_artifact_schema_version=4,
        )
    )

    assert result.model is not None
    assert result.runtime_view is not None
    assert result.runtime_view.representation_contract_hash == "repr"
    assert result.layout is not None
    assert calls[0][0] == "carrier"
    assert calls[1][0] == "prepare"
    assert calls[1][1]["manifest_bytes"] == b"manifest"
    assert adapter.events == [
        ("process", model_config, "cpu"),
        ("finalize", model_config, "cpu"),
    ]


def test_serving_integration_finalizes_local_ready_runtime_runs_semantic_validation():
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    model_config = SimpleNamespace(name="model-config")

    result = integration._finalize_local_ready_runtime(
        _LocalReadyFinalize(
            model=_MaterializedModel(),
            recipe=_local_ready_recipe(),
            binding=_LocalReadyBinding(),
            update_epoch="epoch-1",
            source_artifact_ref="mi2:test:source",
            serving_manifest_ref="manifest-ref",
            representation_contract_hash="repr",
            serving_build_digest="build",
            manifest_tensor_name="__tensorcast_meta__.manifest",
            source_bound_contract_state=SimpleNamespace(
                source_bound_contract_version=4,
                source_bound_capability_names=(),
                source_bound_contract_ready=True,
            ),
            source_bound_contract_path="/tmp/contract.json",
            target_device=torch.device("cpu"),
            model_config=model_config,
            run_process_after_load=True,
            semantic_validation_spec=TensorcastSemanticValidationSpec(
                kind="explicit",
                payload={"model_name": "model-config"},
            ),
        )
    )

    assert result.current_value is not None
    assert adapter.events == [
        ("process", model_config, "cpu"),
        ("finalize", model_config, "cpu"),
    ]


def test_serving_integration_finalizes_local_ready_runtime_closes_on_error():
    binding = _LocalReadyBinding(tensors={"unexpected": torch.ones((1,))})

    with pytest.raises(SchemaMismatchError, match="tensor set"):
        adapter = _MaterializationAdapter()
        ServingIntegration(
            host=_host_for_adapter(adapter),
        )._finalize_local_ready_runtime(
            _LocalReadyFinalize(
                model=_MaterializedModel(),
                recipe=_local_ready_recipe(),
                binding=binding,
                update_epoch="epoch-1",
                source_artifact_ref="mi2:test:source",
                serving_manifest_ref="manifest-ref",
                representation_contract_hash="repr",
                serving_build_digest="build",
                manifest_tensor_name="__tensorcast_meta__.manifest",
                source_bound_contract_state=SimpleNamespace(
                    source_bound_contract_version=4,
                    source_bound_capability_names=(),
                    source_bound_contract_ready=True,
                ),
                source_bound_contract_path="/tmp/contract.json",
                target_device=torch.device("cpu"),
            )
        )

    assert binding.closed


def test_serving_integration_acquire_retained_binding_uses_materialization():
    client = _Client()
    adapter = _MaterializationAdapter()
    adapter.compute_runtime_tensor_schema_hash = (
        lambda _tensors, **_kwargs: "schema-hash"
    )
    adapter.allocate_runtime_only_tensors = (
        lambda model, _target_device: _allocate_cpu_runtime_only(model)
    )
    integration = ServingIntegration(host=_host_for_adapter(adapter))
    model_config = SimpleNamespace(name="model-config")

    result = integration._restore_retained_for_intent(
        _RetainedBindingAcquire(
            authority=_authority(),
            model_config=model_config,
            target_device=torch.device("cuda:0"),
            expected_member=_member(),
            runtime=_Runtime(client),
            restore_fn=lambda **_kwargs: {
                "w": torch.ones((1,), dtype=torch.float32),
            },
        )
    )

    assert result.model is not None
    assert torch.equal(result.model.w.detach(), torch.ones((1,)))
    assert not result.model.runtime_only.is_meta
    assert result.runtime_view.readiness == "serving_local_ready"
    assert result.runtime_view.tensor_schema_hash == "schema-hash"
    assert result.runtime_view.binding_value_ref == _binding_ref()
    assert adapter.events == [("finalize", model_config, "cuda:0")]
    assert client.released_tokens == []

    result.runtime_state.close()
    assert client.released_tokens == [b"lease"]


def test_serving_integration_acquire_retained_binding_rejects_published_ready():
    authority = _authority()
    authority = ParsedRetainedServingBindingAuthority(
        **{
            **authority.__dict__,
            "readiness": "serving_published_ready",
        }
    )
    adapter = _MaterializationAdapter()
    integration = ServingIntegration(host=_host_for_adapter(adapter))

    with pytest.raises(RestoreBindingError, match="swap-capable"):
        integration._restore_retained_for_intent(
            _RetainedBindingAcquire(
                authority=authority,
                target_device=torch.device("cuda:0"),
            )
        )


def _allocate_cpu_runtime_only(model):
    tensor = nn.Parameter(torch.empty((1,), dtype=torch.float32), requires_grad=False)
    model.runtime_only = tensor
    return {"runtime_only": tensor}


class _Client:
    def __init__(self) -> None:
        self.released_tokens: list[bytes] = []

    def acquire_binding_value(self, **_kwargs):
        return SimpleNamespace(
            reservation_bytes=4096,
            mem_handle=SimpleNamespace(lease_token=b"lease"),
            current_value=SimpleNamespace(
                binding_id="binding-1",
                binding_layout_id="layout-1",
                binding_value_id="value-1",
                seal_generation=1,
            ),
        )

    def release_placement_lease(self, *, lease_token: bytes, **_kwargs):
        self.released_tokens.append(bytes(lease_token))


class _Runtime:
    def __init__(self, client: _Client) -> None:
        self.client = client

    def ensure_client(self):
        return self.client


def test_bind_and_swap_return_attach_ready_results():
    resolved = SimpleNamespace(artifact=_Artifact(), tensor_names=("w",))
    result = bind_serving_artifact(
        resolved_artifact=resolved,
        tensor_names=("w",),
        device=torch.device("cuda:0"),
        serving_runtime_policy=None,
        options=None,
    )

    assert result.binding_layout_id == "layout-1"
    assert set(result.tensors) == {"w"}
    assert result.execution_diagnostics == {"executor": "fake"}

    class _SwapBinding(_Bound):
        def swap(self, artifact, **kwargs):
            self.swapped = (artifact, kwargs)

    binding = _SwapBinding()
    binding.tensors[SERVING_MANIFEST_TENSOR_NAME] = torch.ones((1,), dtype=torch.uint8)
    swap_result = swap_serving_artifact(
        binding=binding,
        resolved_artifact=resolved,
        serving_runtime_policy="policy",
        options="options",
    )

    assert swap_result.binding is binding
    assert isinstance(binding.swapped[0], _Subset)
    assert binding.swapped[0].names == ("w", SERVING_MANIFEST_TENSOR_NAME)
    assert binding.swapped[1] == {
        "serving_runtime_policy": "policy",
        "options": "options",
    }


def test_restore_retained_binding_releases_untransferred_attachment():
    client = _Client()

    with restore_retained_binding(
        authority=_authority(),
        target_device=torch.device("cuda:0"),
        runtime=_Runtime(client),
        restore_fn=lambda **_kwargs: {"w": torch.empty((1,))},
    ) as restored:
        assert restored.binding_layout_id == "layout-1"
        assert restored.binding_value_ref == _binding_ref()

    assert client.released_tokens == [b"lease"]


def test_restore_retained_binding_keeps_runtime_owned_attachment():
    client = _Client()

    with restore_retained_binding(
        authority=_authority(),
        target_device=torch.device("cuda:0"),
        runtime=_Runtime(client),
        restore_fn=lambda **_kwargs: {"w": torch.empty((1,))},
    ) as restored:
        runtime_handle = restored.transfer_to_runtime()

    assert runtime_handle.binding_layout_id == "layout-1"
    assert client.released_tokens == []
    runtime_handle.close()
    assert client.released_tokens == [b"lease"]


def test_restore_retained_binding_rejects_member_mismatch():
    expected_member = ServingBindingMemberRef(
        member_id="other-member",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )

    with (
        pytest.raises(RuntimeError, match="expected runtime placement"),
        restore_retained_binding(
            authority=_authority(),
            target_device=torch.device("cuda:0"),
            expected_member=expected_member,
            runtime=_Runtime(_Client()),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,))},
        ),
    ):
        pass


def test_restore_prepared_local_ready_requires_manifest_ref():
    resolved = SimpleNamespace(
        manifest=SimpleNamespace(local_serving_ref=None),
        artifact_ref="mi2:test:serving",
    )

    with (
        pytest.raises(RuntimeError, match="local_serving_ref"),
        restore_prepared_local_ready_binding(
            resolved_artifact=resolved,
            target_device=torch.device("cuda:0"),
            expected_member=_member(),
            expected_tensor_schema_hash="schema-hash",
        ),
    ):
        pass
