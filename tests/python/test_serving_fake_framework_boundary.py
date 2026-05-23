#  Copyright (c) 2026, TensorCast Team.

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import tensorcast.serving._runtime_impl.lifecycle as integration_mod
from tensorcast.serving._runtime_impl.lifecycle import (
    FrameworkIdentity,
    IntegrationHost,
    MaterializationExecutionFacts,
    PlacementAdmissionFacts,
    PlacementIdentityFacts,
    PlacementMemberFacts,
    ServingIntegration,
    SourceSelector,
)
from tensorcast.serving.admin import AdminLocalSourceBootstrap
from tensorcast.serving.builder.compiler import (
    CompiledServingRecipe,
    TensorcastSemanticValidationSpec,
    TensorcastServingFacts,
    TensorSchemaEntry,
)
from tensorcast.serving.builder.trace_ir import TracePlan
from tensorcast.serving.recipe_build import (
    RecipeBuildSession,
    ServingBindingPlan,
)
from tensorcast.serving.retained_binding import (
    ParsedRetainedServingBindingAuthority,
    RetainedServingBindingExpectedDigests,
)
from tensorcast.serving.runtime import (
    BootstrapPolicy,
    ExistingServingArtifact,
    RequestContext,
    RetainedBindingAcquire,
    ServingArtifactLocator,
)
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    FinalizeClass,
    ServingArtifactManifest,
    ServingBindingMemberRef,
    ServingSupportLevel,
)


class _FakeArtifactView:
    def __init__(self, parent, names=None):
        self.parent = parent
        self.names = tuple(names or ())

    def bind(self, **kwargs):
        binding = _FakeBinding()
        binding.names = self.names
        binding.kwargs = kwargs
        return binding


class _FakeArtifact:
    def subset(self, names):
        return _FakeArtifactView(self, names)


class _FakeBinding:
    def __init__(self):
        self.tensors = {"w": torch.ones((1,), dtype=torch.float16)}
        self.binding_layout_id = "layout-1"
        self.realized = None
        self.swapped = None
        self.closed = False

    def realize_from(self, source_view, *, realization_plan, options):
        self.realized = (source_view, realization_plan, options)
        return "epoch-1"

    def swap(self, artifact, **kwargs):
        self.swapped = (artifact, kwargs)
        self.tensors = {"w": torch.full((1,), 2.0, dtype=torch.float16)}
        return self

    def freeze_current(self, *, update_epoch, source_artifact_ref):
        return SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id=self.binding_layout_id,
            binding_value_id="value-1",
            seal_generation=1,
            update_epoch=update_epoch,
            source_artifact_ref=source_artifact_ref,
            local_serving_ref="binding-local:fake",
        )

    def close(self):
        self.closed = True


class _FakeRestoredRetainedBinding:
    def __init__(self):
        self.tensors = {"w": torch.ones((1,), dtype=torch.float16)}
        self.binding_layout_id = "layout-1"
        self.binding_value_ref = SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
        )
        self.reservation_bytes = 4096
        self.closed = False
        self.transferred = False

    def transfer_to_runtime(self):
        self.transferred = True
        return SimpleNamespace(close=lambda: None)

    def close(self):
        self.closed = True


def _retained_authority() -> ParsedRetainedServingBindingAuthority:
    member = ServingBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )
    binding_ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )
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
        local_serving_ref="binding-local:fake",
        binding_value_ref=binding_ref,
        reservation_capability=capability,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4096,
        expected=RetainedServingBindingExpectedDigests(
            target_layout_hash="layout-hash",
            tensor_schema_hash="fake-schema",
            serving_build_digest="build-digest",
            resolved_spec_digest="spec-digest",
        ),
        readiness="serving_local_ready",
        verification_state="local_only",
    )


class _FakeSource:
    def subset(self, names):
        return ("subset", tuple(names))


class _FakeRuntimeModel:
    def __init__(self):
        self.tensors = {"w": torch.empty((1,), dtype=torch.float16, device="meta")}


class _FakeFrameworkHost:
    def identity(self, model_config):
        del model_config
        return FrameworkIdentity(
            framework_name="fakefw",
            framework_version="fakefw-v1",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
        )

    def prepare_model_construction(self, framework_config, model_config):
        del framework_config, model_config

    def build_meta_model(self, framework_config, model_config):
        del framework_config, model_config
        return _FakeRuntimeModel()

    def build_runtime_model(self, framework_config, model_config, target_device):
        del framework_config, model_config, target_device
        return _FakeRuntimeModel()

    def assert_model_ready_for_runtime_binding(self, model, *, context):
        del context
        assert "w" in model.tensors

    def semantic_probes(self, model, model_config):
        del model, model_config
        return {}


class _FakePlacementHost:
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
        return PlacementMemberFacts(
            runtime_rank=0,
            runtime_world_size=1,
            member_id="member-0",
            member_index=0,
            member_count=1,
            group_id_hint="group-1",
        )

    def execution_facts(self, framework_config):
        del framework_config
        return MaterializationExecutionFacts(
            collective_rank=0,
            collective_world_size=1,
            tensor_parallel_ranks=(0,),
        )


class _FakeTensorSurface:
    def runtime_only_tensor_names(self, model):
        del model
        return ()

    def align_runtime_tensor_names(self, model, expected_names):
        assert set(expected_names) == set(model.tensors)
        return 0

    def collect_runtime_tensors(self, model, *, remove_duplicate=False):
        del remove_duplicate
        return dict(model.tensors)

    def collect_runtime_tensor_view(self, tensors):
        del tensors
        return ()

    def compute_runtime_tensor_schema_hash(self, tensors, *, remove_duplicate=False):
        del tensors, remove_duplicate
        return "fake-schema"

    def attach_bound_tensors(self, model, tensors, *, replace_meta_params):
        del replace_meta_params
        model.tensors.update(tensors)
        return model

    def allocate_runtime_only_tensors(self, model, target_device):
        del model, target_device
        return {}

    def snapshot_tensor_invariants(self, tensors):
        return tuple(sorted(tensors))

    def validate_tensor_invariants(self, before, after):
        assert before == tuple(sorted(after))


def _realization_plan_proto():
    from tensorcast.proto.daemon.v2 import store_daemon_pb2

    plan = store_daemon_pb2.BindingRealizationPlan()
    entry = plan.entries.add(dst_name="w")
    entry.op_kind = store_daemon_pb2.BINDING_REALIZATION_OP_KIND_COPY
    entry.source_name = "w"
    return plan.SerializeToString(deterministic=True)


def _recipe():
    return CompiledServingRecipe(
        compile_key="compile",
        source_artifact_ref="mi2:source",
        source_metadata_fingerprint="meta",
        serving_facts=TensorcastServingFacts(
            framework_name="fakefw",
            framework_version="fakefw-v1",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
            support_level=ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
            runtime_only_tensor_names=(),
            process_after_load_class=FinalizeClass.RUNTIME_ONLY,
            post_bind_finalize_class=FinalizeClass.RUNTIME_ONLY,
        ),
        trace_plan=TracePlan(
            copy_plan=[],
            expected_src_names={"w"},
            expected_dst_names={"w"},
            tensorcast_slices={},
            src_hull={},
        ),
        tensor_schema=(
            TensorSchemaEntry(
                name="w",
                dtype="torch.float16",
                shape=(1,),
                stride=(1,),
            ),
        ),
        source_hull=(),
        realization_plan=(),
        realization_fallback_plan=(),
        topology_ref=None,
        member_ref=None,
        semantic_validation_spec=TensorcastSemanticValidationSpec.empty(),
        realization_plan_proto=_realization_plan_proto(),
        realization_plan_count=1,
    )


def test_fake_second_framework_core_generated_ids_are_framework_neutral():
    group_id = integration_mod.build_collective_group_id(
        artifact_ref="mi2:fake:serving",
        operation_scope="fakefw.realize",
        tp_ranks=(0, 1),
        contract_identity="repr",
    )
    assert group_id.startswith("tensorcast-")
    assert "vllm" not in group_id

    _contract_hash, manifest_bytes = (
        integration_mod.prepare_same_binding_manifest_carrier(
            _recipe(),
            manifest_tensor_name="__tensorcast_meta__.manifest",
            representation_contract_hash="repr",
            topology_admission_digest="topology-digest",
        )
    )
    manifest = ServingArtifactManifest.from_bytes(manifest_bytes)
    lower_manifest = manifest_bytes.lower()
    assert integration_mod.LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION == (
        "tensorcast-bootstrap-v1"
    )
    assert manifest.topology_admission_digest == "topology-digest"
    assert (
        integration_mod.LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION.encode()
        in manifest_bytes
    )
    assert b"vllm" not in lower_manifest


def test_fake_second_framework_uses_host_intent_lifecycle(monkeypatch):
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
    )
    session = RecipeBuildSession(identity)
    assert session.recipe_cache_key(metadata_fingerprint="meta")

    monkeypatch.setattr(
        integration_mod,
        "read_source_bound_contract_state",
        lambda: SimpleNamespace(
            source_bound_contract_ready=True,
            source_bound_contract_version=4,
            source_bound_capability_names=("collective",),
        ),
    )
    monkeypatch.setattr(
        ServingIntegration,
        "build_materialization_options",
        lambda self, **kwargs: ("realize-options", kwargs),
    )
    direct_resolve_calls = []

    class _FakeResolver:
        def resolve(self, artifact_ref):
            direct_resolve_calls.append(("resolve", artifact_ref))
            return SimpleNamespace(
                artifact=_FakeArtifact(),
                artifact_ref=artifact_ref,
                tensor_names=("w",),
                manifest=SimpleNamespace(
                    representation_contract_hash="repr-direct",
                    source_artifact_ref="mi2:source",
                    serving_build_digest="build-direct",
                ),
            )

        def cross_check(self, resolved_artifact, **kwargs):
            direct_resolve_calls.append(("cross_check", kwargs))
            return resolved_artifact

    host = IntegrationHost(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )
    direct_attachment = ServingIntegration(
        resolver=_FakeResolver(),
        host=host,
    ).start(
        ExistingServingArtifact(ServingArtifactLocator.artifact_ref("mi2:serving")),
        RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cuda:0"),
        ),
    )
    direct_payload = direct_attachment.view.endpoint.to_weight_version_payload()
    assert direct_attachment.state.runtime_view.readiness == "serving"
    assert direct_payload["serving_artifact_ref"] == "mi2:serving"
    assert direct_payload["source_artifact_ref"] == "mi2:source"
    assert direct_resolve_calls[1][1]["expected_tensor_schema_hash"] == "fake-schema"
    reload_attachment = ServingIntegration(
        resolver=_FakeResolver(),
        host=host,
    ).reload(
        direct_attachment.state,
        ExistingServingArtifact(
            ServingArtifactLocator.artifact_ref("mi2:serving-next")
        ),
        RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        model=direct_attachment.model,
    )
    reload_payload = reload_attachment.view.endpoint.to_weight_version_payload()
    reload_response = reload_attachment.view.endpoint.to_reload_response_payload()
    assert reload_payload["serving_artifact_ref"] == "mi2:serving-next"
    assert reload_response == {
        "schema_version": 1,
        "serving_artifact_ref": "mi2:serving-next",
        "representation_contract_hash": "repr-direct",
        "serving_build_digest": "build-direct",
        "readiness": "serving",
    }
    assert direct_attachment.state.binding.swapped[1]["options"] == "realize-options"
    described = ServingIntegration(host=host).describe(reload_attachment.state)
    assert (
        described.endpoint.to_weight_version_payload()["serving_artifact_ref"]
        == "mi2:serving-next"
    )

    host_binding = _FakeBinding()
    host_model = _FakeRuntimeModel()
    attachment = ServingIntegration(host=host).start(
        AdminLocalSourceBootstrap(
            source_selector=SourceSelector.local_path("/tmp/fake-model"),
            bootstrap_policy=BootstrapPolicy(),
            recipe=_recipe(),
            source_subject=_FakeSource(),
            source_artifact_ref="mi2:source",
            model=host_model,
            binding_factory=lambda *args, **kwargs: host_binding,
        ),
        RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cuda:0"),
        ),
    )
    assert attachment.model is host_model
    assert attachment.state.runtime_view.readiness == "serving_local_ready"
    payload = attachment.view.endpoint.to_weight_version_payload()
    assert payload["source_artifact_ref"] == "mi2:source"
    assert payload["family"] == "generic"
    assert payload["tp_rank"] == 0
    assert attachment.prepared is not None
    assert host_binding.realized is not None
    assert host_binding.realized[2] == "realize-options"

    retained_calls = []
    restored = _FakeRestoredRetainedBinding()

    @contextmanager
    def fake_restore_retained(**kwargs):
        retained_calls.append(kwargs)
        yield restored

    monkeypatch.setattr(
        integration_mod, "restore_retained_binding", fake_restore_retained
    )
    retained_attachment = ServingIntegration(host=host).start(
        RetainedBindingAcquire(authority=_retained_authority()),
        RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cuda:0"),
        ),
    )
    retained_payload = retained_attachment.view.endpoint.to_weight_version_payload()
    assert retained_attachment.state.runtime_view.readiness == "serving_local_ready"
    assert retained_payload["local_serving_ref"] == "binding-local:fake"
    assert retained_payload["binding_value_ref"]["binding_value_id"] == "value-1"
    assert retained_calls[0]["expected_member"].member_index == 0
    assert restored.transferred


def test_fake_second_framework_uses_public_runtime_session(monkeypatch):
    import tensorcast.serving.hosts as tc_hosts
    import tensorcast.serving.runtime as tc_runtime
    from tensorcast.serving.testing import assert_framework_isolation

    monkeypatch.setattr(
        tc_runtime.RuntimeSettings, "ensure_initialized", lambda self: None
    )
    monkeypatch.setattr(
        integration_mod,
        "read_source_bound_contract_state",
        lambda: SimpleNamespace(
            source_bound_contract_ready=True,
            source_bound_contract_version=4,
            source_bound_capability_names=("collective",),
        ),
    )
    monkeypatch.setattr(
        integration_mod.ServingIntegration,
        "build_materialization_options",
        lambda self, **kwargs: ("runtime-options", kwargs),
    )

    class _Resolver:
        def resolve(self, artifact_ref):
            return SimpleNamespace(
                artifact=_FakeArtifact(),
                artifact_ref=artifact_ref,
                tensor_names=("w",),
                manifest=SimpleNamespace(
                    representation_contract_hash=f"repr:{artifact_ref}",
                    source_artifact_ref="mi2:source",
                    serving_build_digest=f"build:{artifact_ref}",
                ),
            )

        def cross_check(self, resolved_artifact, **kwargs):
            return resolved_artifact

    host = tc_hosts.IntegrationHost(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )
    session = tc_runtime.ServingRuntimeSession.from_config(
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
        host=host,
        resolver=_Resolver(),
    )

    attachment = session.start(
        tc_runtime.RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cuda:0"),
        )
    )
    reloaded = session.reload(
        current_attachment=attachment,
        artifact_locator=tc_runtime.ServingArtifactLocator.artifact_ref(
            "mi2:serving-next"
        ),
        policy=tc_runtime.ServingPolicy(),
        context=tc_runtime.RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        model=attachment.model,
    )

    assert (
        attachment.view.endpoint.to_weight_version_payload()["serving_artifact_ref"]
        == "mi2:serving"
    )
    assert (
        reloaded.view.endpoint.to_reload_response_payload()["serving_artifact_ref"]
        == "mi2:serving-next"
    )
    assert_framework_isolation(
        ("tensorcast.serving.runtime", "tensorcast.serving.hosts")
    )


def test_fake_second_framework_runtime_conformance_kit():
    import tensorcast.serving.hosts as tc_hosts
    import tensorcast.serving.runtime as tc_runtime
    from tensorcast.serving.testing import (
        assert_level1_runtime_conformance,
        assert_level2_local_bootstrap_conformance,
        assert_level3_retained_binding_conformance,
    )

    result = assert_level1_runtime_conformance(tc_runtime, tc_hosts)

    assert result.checks["direct_start"]
    assert result.checks["reload"]
    assert result.checks["describe"]
    assert result.checks["source_capability_not_required"]
    assert result.checks["source_catalog_not_required"]
    assert result.checks["rejects_local_reload_artifact_locator"]
    assert result.checks["rejects_untyped_reload_artifact_locator"]
    assert result.checks["rejects_untyped_reload_policy"]

    local = assert_level2_local_bootstrap_conformance(tc_runtime, tc_hosts)
    assert local.checks["missing_source_catalog_fails_closed"]
    assert local.checks["source_catalog_request_core_owned"]
    assert local.checks["recipe_build_receives_core_catalog"]
    assert local.checks["missing_trace_capability_is_explicit"]
    assert local.checks["local_path_is_not_reload_artifact_locator"]

    retained = assert_level3_retained_binding_conformance(tc_runtime, tc_hosts)
    assert retained.checks["retained_acquire_public_start"]
    assert retained.checks["retained_acquire_uses_host_member"]
    assert retained.checks["retained_acquire_transfers_ownership"]
    assert retained.checks["missing_authority_fails_closed"]
    assert retained.checks["authority_mismatch_fails_closed"]
    assert retained.checks["failure_path_used_retained_restore"]
    assert retained.checks["failure_cleanup_closes_untransferred_handle"]
    assert retained.checks["rejects_arbitrary_retained_authority"]


def test_conformance_failure_summary_includes_onboarding_hint():
    from tensorcast.serving.testing import ConformanceResult

    result = ConformanceResult(
        checks={"direct_start": False},
        messages={"direct_start": "provide a tensor surface"},
        level="level1-runtime",
    )

    try:
        result.assert_passed()
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected conformance failure")

    assert "level1-runtime" in message
    assert "direct_start" in message
    assert "provide a tensor surface" in message
