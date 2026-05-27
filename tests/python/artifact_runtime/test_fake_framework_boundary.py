#  Copyright (c) 2026, TensorCast Team.

import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import tensorcast as tc
import tensorcast.artifact_runtime.lifecycle as integration_mod
import tensorcast.artifact_runtime.recipe.local_ready as local_ready_mod
from tensorcast.api.store.artifact import Artifact
from tensorcast.artifact_runtime.admin import AdminLocalSourceBootstrap
from tensorcast.artifact_runtime.host import (
    FrameworkIdentity,
    IntegrationHost,
    MaterializationExecutionFacts,
    PlacementAdmissionFacts,
    PlacementIdentityFacts,
    PlacementMemberFacts,
    SourceSelector,
)
from tensorcast.artifact_runtime.intent import (
    BootstrapPolicy,
    ExistingRuntimeArtifact,
    RequestContext,
    RetainedBindingAcquire,
)
from tensorcast.artifact_runtime.lifecycle import ArtifactRuntimeIntegration
from tensorcast.artifact_runtime.locator import ArtifactLocator
from tensorcast.artifact_runtime.recipe.build import (
    RecipeBuildSession,
    RuntimeBindingPlan,
)
from tensorcast.artifact_runtime.recipe.compiler import (
    CompiledRuntimeRecipe,
    TensorcastRuntimeFacts,
    TensorcastSemanticValidationSpec,
    TensorSchemaEntry,
)
from tensorcast.artifact_runtime.recipe.trace_ir import TracePlan
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
    RetainedRealizationExpectedDigests,
)
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    FinalizeClass,
    RuntimeArtifactManifest,
    RuntimeBindingMemberRef,
    RuntimeSupportLevel,
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


def _retained_authority() -> ParsedRetainedRealizationAuthority:
    member = RuntimeBindingMemberRef(
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
    return ParsedRetainedRealizationAuthority(
        group_id="group-1",
        local_serving_ref="binding-local:fake",
        binding_value_ref=binding_ref,
        reservation_capability=capability,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4096,
        expected=RetainedRealizationExpectedDigests(
            target_layout_hash="layout-hash",
            tensor_schema_hash="fake-schema",
            runtime_build_digest="build-digest",
            resolved_spec_digest="spec-digest",
        ),
        readiness="runtime_local_ready",
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


class _FailingPlacementHost(_FakePlacementHost):
    def identity_facts(self, framework_config):
        del framework_config
        raise RuntimeError("placement unavailable")


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


def _recipe(source_artifact_ref="mi2:source"):
    return CompiledRuntimeRecipe(
        compile_key="compile",
        source_artifact_ref=source_artifact_ref,
        source_metadata_fingerprint="meta",
        runtime_facts=TensorcastRuntimeFacts(
            framework_name="fakefw",
            framework_version="fakefw-v1",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
            support_level=RuntimeSupportLevel.RUNTIME_BIND_SWAP_READY,
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
        local_ready_mod.prepare_same_binding_manifest_carrier(
            _recipe(),
            manifest_tensor_name="__tensorcast_meta__.manifest",
            representation_contract_hash="repr",
            topology_admission_digest="topology-digest",
        )
    )
    manifest = RuntimeArtifactManifest.from_bytes(manifest_bytes)
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
    identity = RuntimeBindingPlan(
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
        ArtifactRuntimeIntegration,
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
    direct_attachment = ArtifactRuntimeIntegration(
        resolver=_FakeResolver(),
        host=host,
    ).start(
        ExistingRuntimeArtifact(ArtifactLocator.artifact_ref("mi2:serving")),
        RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cuda:0"),
        ),
    )
    direct_payload = direct_attachment.view.endpoint.to_weight_version_payload()
    assert direct_attachment.state.runtime_view.readiness == "runtime_ready"
    assert direct_payload["serving_artifact_ref"] == "mi2:serving"
    assert direct_payload["source_artifact_ref"] == "mi2:source"
    assert direct_resolve_calls[1][1]["expected_tensor_schema_hash"] == "fake-schema"
    reload_attachment = ArtifactRuntimeIntegration(
        resolver=_FakeResolver(),
        host=host,
    ).reload(
        direct_attachment.state,
        ExistingRuntimeArtifact(ArtifactLocator.artifact_ref("mi2:serving-next")),
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
        "readiness": "runtime_ready",
    }
    assert direct_attachment.state.binding.swapped[1]["options"] == "realize-options"
    described = ArtifactRuntimeIntegration(host=host).describe(reload_attachment.state)
    assert (
        described.endpoint.to_weight_version_payload()["serving_artifact_ref"]
        == "mi2:serving-next"
    )

    host_binding = _FakeBinding()
    host_model = _FakeRuntimeModel()
    attachment = ArtifactRuntimeIntegration(host=host).start(
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
    assert attachment.state.runtime_view.readiness == "runtime_local_ready"
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
    retained_attachment = ArtifactRuntimeIntegration(host=host).start(
        RetainedBindingAcquire(authority=_retained_authority()),
        RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cuda:0"),
        ),
    )
    retained_payload = retained_attachment.view.endpoint.to_weight_version_payload()
    assert retained_attachment.state.runtime_view.readiness == "runtime_local_ready"
    assert retained_payload["local_serving_ref"] == "binding-local:fake"
    assert retained_payload["binding_value_ref"]["binding_value_id"] == "value-1"
    assert retained_calls[0]["expected_member"].member_index == 0
    assert restored.transferred


def test_artifact_realize_model_runtime_uses_direct_runtime_host(monkeypatch):
    monkeypatch.setattr(
        integration_mod,
        "read_source_bound_contract_state",
        lambda: SimpleNamespace(
            source_bound_contract_ready=True,
            source_bound_contract_version=4,
            source_bound_capability_names=("collective",),
        ),
    )
    materialization_calls = []

    def build_materialization_options(_self, **kwargs):
        materialization_calls.append(kwargs)
        return "realize-options", kwargs

    monkeypatch.setattr(
        ArtifactRuntimeIntegration,
        "build_materialization_options",
        build_materialization_options,
    )

    class _RecordingArtifactView:
        def __init__(self, parent, names):
            self.parent = parent
            self.names = tuple(names)

        def bind(self, **kwargs):
            binding = _FakeBinding()
            binding.last_materialization_diagnostics = {
                "source": "p2p",
                "operation_id": "op-direct",
                "total_bytes": 2,
                "retry_reason_buckets": {"none": 0},
                "ipc_open_sec": 0.001,
                "restore_tensors_sec": 0.002,
            }
            binding.last_execution_diagnostics = SimpleNamespace(
                actual_collective_committed_bytes=0,
                actual_local_typed_bytes=2,
                actual_generic_backend_bytes=0,
                fallback_bytes=0,
                residual_bytes=0,
                direct_write_supported=True,
                dominant_executor="local_typed",
            )
            self.parent.bind_calls.append((self.names, kwargs, binding))
            return binding

        def tensor_dict(self, **_kwargs):
            raise AssertionError("direct model-runtime path must not use TensorDict")

        def tensor_dict_with_diagnostics(self, **_kwargs):
            raise AssertionError("direct model-runtime path must not use TensorDict")

        def tensor_dict_into(self, *_args, **_kwargs):
            raise AssertionError("direct model-runtime path must not use TensorDict")

        def state_dict(self):
            raise AssertionError("direct model-runtime path must not build state dict")

    class _RecordingArtifact:
        def __init__(self):
            self.bind_calls = []

        def subset(self, names):
            return _RecordingArtifactView(self, names)

        def tensor_dict(self, **_kwargs):
            raise AssertionError("direct model-runtime path must not use TensorDict")

        def state_dict(self):
            raise AssertionError("direct model-runtime path must not build state dict")

    resolved_artifact = _RecordingArtifact()
    resolver_calls = []

    class _Resolver:
        def resolve(self, artifact_ref):
            resolver_calls.append(("resolve", artifact_ref))
            return SimpleNamespace(
                artifact=resolved_artifact,
                artifact_ref=artifact_ref,
                tensor_names=("w",),
                manifest=SimpleNamespace(
                    representation_contract_hash="repr-direct",
                    source_artifact_ref="mi2:source",
                    serving_build_digest="build-direct",
                ),
            )

        def cross_check(self, resolved, **kwargs):
            resolver_calls.append(("cross_check", kwargs))
            return resolved

    class _Store:
        pass

    def reject_runtime_session(*_args, **_kwargs):
        raise AssertionError("direct model-runtime path must not start a session")

    monkeypatch.setattr(
        integration_mod.ArtifactRuntimeSession,
        "from_config",
        classmethod(reject_runtime_session),
    )
    monkeypatch.setattr(
        integration_mod.ArtifactRuntimeSession,
        "start",
        reject_runtime_session,
    )

    artifact = Artifact(
        store_ref=weakref.ref(_Store()),
        artifact_id="mi2:serving",
        canonical_index_bytes=b"index",
    )
    host = tc.RuntimeHostCapabilities(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )
    profile_events = []

    handle = artifact.realize(
        tc.ArtifactRealizationSpec.model_runtime(
            framework="fakefw",
            device=torch.device("cuda:0"),
            adapter_version="adapter-v1",
            runtime_abi_version="abi-v1",
        ),
        runtime_host=host,
        runtime_context=RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        runtime_resolver=_Resolver(),
        profile_sink=profile_events.append,
    )
    attachment = handle.attachment()

    assert handle.attach() is attachment
    assert attachment.state.model_runtime_handle is handle
    assert handle.report.target_kind == "model_runtime"
    assert handle.report.artifact_id == "mi2:serving"
    assert handle.report.artifact_profile == "durable_artifact"
    assert handle.report.authority_scope == "daemon_mediated_durable"
    assert handle.report.source_selection_digest
    assert handle.report.model_runtime is not None
    assert handle.report.model_runtime.framework == "fakefw"
    assert handle.report.model_runtime.adapter_version == "adapter-v1"
    assert handle.report.model_runtime.runtime_abi_version == "abi-v1"
    assert handle.report.runtime_attach_sec is not None
    assert handle.report.runtime_attach_sec >= 0.0
    assert handle.report.runtime_finalize_sec is not None
    assert handle.report.runtime_finalize_sec >= 0.0
    assert handle.report.total_sec is not None
    assert handle.report.total_sec >= handle.report.runtime_attach_sec
    assert (
        attachment.state.realization_handle.report.target_kind == "runtime_attachment"
    )
    assert attachment.state.realization_handle.report.runtime_attach_sec == (
        handle.report.runtime_attach_sec
    )
    assert attachment.state.realization_handle.report.runtime_finalize_sec == (
        handle.report.runtime_finalize_sec
    )
    assert torch.equal(
        attachment.model.tensors["w"], torch.ones((1,), dtype=torch.float16)
    )
    assert resolved_artifact.bind_calls
    bind_names, bind_kwargs, _binding = resolved_artifact.bind_calls[0]
    assert bind_names == ("w",)
    assert bind_kwargs["device"] == torch.device("cuda:0")
    assert bind_kwargs["options"] == "realize-options"
    assert len(materialization_calls) == 1
    assert materialization_calls[0]["artifact_ref"] == "mi2:serving"
    assert (
        materialization_calls[0]["operation_scope"]
        == "startup.direct_artifact_runtime.bind"
    )
    assert materialization_calls[0][
        "source_bound_contract_state"
    ].source_bound_contract_ready
    assert handle.report.source == "p2p"
    assert handle.report.operation_id == "op-direct"
    assert handle.report.materialization_diagnostics["ipc_open_sec"] == 0.001
    assert handle.report.execution_commit is not None
    assert handle.report.execution_commit.actual_executor_path == "local_typed"
    assert handle.report.execution_commit.direct_write_bytes == 2
    assert handle.report.execution_commit.fallback_bytes == 0
    assert handle.report.envelope.copy_bytes == 0
    assert handle.report.envelope.temporary_replica_bytes == 0
    assert handle.report.envelope.retained_bytes == 0
    assert handle.report.envelope.cuda_ipc_open_count == 0
    assert [event["event"] for event in profile_events] == [
        "runtime_materialization.attach.start",
        "runtime_materialization.attach.done",
    ]
    assert resolver_calls[0] == ("resolve", "mi2:serving")
    assert resolver_calls[1][0] == "cross_check"

    serving_attachment = ArtifactRuntimeIntegration(
        resolver=_Resolver(),
        host=host,
    ).start(
        ExistingRuntimeArtifact(ArtifactLocator.artifact_ref("mi2:serving")),
        RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
            target_device=torch.device("cuda:0"),
        ),
    )
    serving_handle = serving_attachment.state.model_runtime_handle
    assert serving_handle.attach() is serving_attachment
    assert serving_handle.report.target_kind == handle.report.target_kind
    assert serving_handle.report.operation_backend == handle.report.operation_backend
    assert serving_handle.report.envelope == handle.report.envelope
    assert serving_handle.report.target_plan == handle.report.target_plan
    assert serving_handle.report.model_runtime == handle.report.model_runtime
    assert serving_handle.release_contract.release_policy == (
        handle.release_contract.release_policy
    )
    assert serving_handle.release_contract.release_strictness == (
        handle.release_contract.release_strictness
    )


def test_artifact_realize_model_runtime_uses_same_store_when_resolver_omitted(
    monkeypatch,
):
    import tensorcast.api.store as store_api
    import tensorcast.artifact_runtime.artifact.resolver as resolver_mod

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
        store_api,
        "artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct model-runtime resolver must use artifact store")
        ),
    )
    monkeypatch.setattr(
        resolver_mod.tc_artifact_manifest,
        "read_runtime_artifact_manifest_tensor",
        lambda *_args, **_kwargs: SimpleNamespace(
            representation_contract_hash="repr-direct",
            source_artifact_ref="mi2:source",
            serving_build_digest="build-direct",
            local_serving_ref=None,
        ),
    )
    manifest_cross_checks = []
    monkeypatch.setattr(
        resolver_mod.tc_artifact_manifest,
        "cross_check_runtime_artifact_manifest",
        lambda **kwargs: manifest_cross_checks.append(kwargs),
    )

    class _StoreArtifactView:
        def __init__(self, parent, names):
            self.parent = parent
            self.names = tuple(names)

        def bind(self, **kwargs):
            binding = _FakeBinding()
            self.parent.bind_calls.append((self.names, kwargs, binding))
            return binding

    class _StoreArtifact:
        def __init__(self):
            self.bind_calls = []
            self.descriptor = SimpleNamespace(
                artifact_id="mi2:serving",
                tensor_names=("w", tc.SERVING_MANIFEST_TENSOR_NAME),
                tensor_metas={
                    "w": SimpleNamespace(
                        shape=(1,),
                        dtype=torch.float16,
                        stride=(1,),
                        storage_offset=0,
                        size_bytes=2,
                    )
                },
                total_bytes=2,
            )

        def describe(self):
            return self.descriptor

        def subset(self, names):
            return _StoreArtifactView(self, names)

    opened_artifact = _StoreArtifact()
    store_calls = []

    class _Store:
        closed = False
        _runtime = object()
        _materialization = object()

        def artifact(self, **kwargs):
            store_calls.append(kwargs)
            return opened_artifact

    store = _Store()
    host = tc.RuntimeHostCapabilities(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )
    materialization_options = tc.GetArtifactOptions()
    artifact = Artifact(
        store_ref=weakref.ref(store),
        artifact_id="mi2:serving",
    )

    handle = artifact.realize(
        tc.ArtifactRealizationSpec.model_runtime(
            framework="fakefw",
            device=torch.device("cuda:0"),
            adapter_version="adapter-v1",
            runtime_abi_version="abi-v1",
            options=materialization_options,
        ),
        runtime_host=host,
        runtime_context=RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        runtime_resolver=None,
    )

    assert handle.report.target_kind == "model_runtime"
    assert store_calls == [{"ref": "mi2:serving"}]
    assert manifest_cross_checks
    bind_names, bind_kwargs, _binding = opened_artifact.bind_calls[0]
    assert bind_names == ("w",)
    assert bind_kwargs["options"] is materialization_options


def test_model_runtime_rejects_spec_context_device_mismatch():
    class _Store:
        pass

    artifact = Artifact(
        store_ref=weakref.ref(_Store()),
        artifact_id="mi2:serving",
    )

    with pytest.raises(tc.ArtifactError) as exc_info:
        artifact.realize(
            tc.ArtifactRealizationSpec.model_runtime(
                framework="fakefw",
                device=torch.device("cuda:0"),
            ),
            runtime_host=object(),
            runtime_context=RequestContext(target_device=torch.device("cuda:1")),
        )

    assert exc_info.value.status_code == "INVALID_ARGUMENT"
    assert "target_device facts disagree" in str(exc_info.value)


def test_model_runtime_rejects_host_placement_failure():
    class _Store:
        pass

    artifact = Artifact(
        store_ref=weakref.ref(_Store()),
        artifact_id="mi2:serving",
    )
    host = tc.RuntimeHostCapabilities(
        framework=_FakeFrameworkHost(),
        placement=_FailingPlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )

    with pytest.raises(
        integration_mod.ArtifactRuntimeIntegrationError,
        match="failed to collect runtime placement facts",
    ):
        artifact.realize(
            tc.ArtifactRealizationSpec.model_runtime(
                framework="fakefw",
                device=torch.device("cuda:0"),
            ),
            runtime_host=host,
            runtime_context=RequestContext(
                framework_config=SimpleNamespace(),
                model_config=SimpleNamespace(model="fake-model"),
            ),
            runtime_resolver=object(),
        )


def test_model_runtime_rejects_spec_host_member_mismatch():
    class _Store:
        pass

    artifact = Artifact(
        store_ref=weakref.ref(_Store()),
        artifact_id="mi2:serving",
    )
    host = tc.RuntimeHostCapabilities(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )

    with pytest.raises(tc.ArtifactError) as exc_info:
        artifact.realize(
            tc.ArtifactRealizationSpec.model_runtime(
                framework="fakefw",
                device=torch.device("cuda:0"),
                member=RuntimeBindingMemberRef(
                    member_id="member-1",
                    member_index=0,
                    member_count=1,
                    group_id="group-1",
                ),
            ),
            runtime_host=host,
            runtime_context=RequestContext(
                framework_config=SimpleNamespace(),
                model_config=SimpleNamespace(model="fake-model"),
            ),
            runtime_resolver=object(),
        )

    assert exc_info.value.status_code == "INVALID_ARGUMENT"
    assert "member facts disagree" in str(exc_info.value)


def test_model_runtime_options_and_runtime_artifact_policy_are_separate(
    monkeypatch,
):
    monkeypatch.setattr(
        integration_mod,
        "read_source_bound_contract_state",
        lambda: SimpleNamespace(
            source_bound_contract_ready=True,
            source_bound_contract_version=4,
            source_bound_capability_names=("collective",),
        ),
    )
    host = tc.RuntimeHostCapabilities(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )

    class _RecordingArtifactView:
        def __init__(self, parent, names):
            self.parent = parent
            self.names = tuple(names)

        def bind(self, **kwargs):
            binding = _FakeBinding()
            self.parent.bind_calls.append((self.names, kwargs, binding))
            return binding

    class _RecordingArtifact:
        def __init__(self):
            self.bind_calls = []

        def subset(self, names):
            return _RecordingArtifactView(self, names)

    class _Resolver:
        def __init__(self):
            self.cross_checks = []
            self.artifact = _RecordingArtifact()

        def resolve(self, artifact_ref):
            return SimpleNamespace(
                artifact=self.artifact,
                artifact_ref=artifact_ref,
                tensor_names=("w",),
                manifest=SimpleNamespace(
                    representation_contract_hash="repr-direct",
                    source_artifact_ref="mi2:source",
                    serving_build_digest="build-direct",
                ),
            )

        def cross_check(self, resolved_artifact, **kwargs):
            self.cross_checks.append(kwargs)
            return resolved_artifact

    def realize_with(runtime_artifact_policy=None):
        resolver = _Resolver()
        materialization_options = tc.GetArtifactOptions()

        class _Store:
            pass

        artifact = Artifact(
            store_ref=weakref.ref(_Store()),
            artifact_id="mi2:serving",
        )
        handle = artifact.realize(
            tc.ArtifactRealizationSpec.model_runtime(
                framework="fakefw",
                device=torch.device("cuda:0"),
                adapter_version="adapter-v1",
                runtime_abi_version="abi-v1",
                options=materialization_options,
                runtime_artifact_policy=runtime_artifact_policy,
            ),
            runtime_host=host,
            runtime_context=RequestContext(
                framework_config=SimpleNamespace(),
                model_config=SimpleNamespace(model="fake-model"),
            ),
            runtime_resolver=resolver,
        )
        assert handle.report.target_kind == "model_runtime"
        return resolver, materialization_options

    resolver, materialization_options = realize_with()
    policy_seen = resolver.cross_checks[0]["runtime_artifact_policy"]
    assert policy_seen is not materialization_options
    assert resolver.artifact.bind_calls[0][1]["options"] is materialization_options

    runtime_policy = tc.RuntimeArtifactPolicy(
        expected_representation_contract_hash="repr-direct",
    )
    resolver, materialization_options = realize_with(runtime_policy)
    policy_seen = resolver.cross_checks[0]["runtime_artifact_policy"]
    assert policy_seen.expected_representation_contract_hash == "repr-direct"
    bind_kwargs = resolver.artifact.bind_calls[0][1]
    assert bind_kwargs["runtime_artifact_policy"].expected_representation_contract_hash
    assert bind_kwargs["options"] is materialization_options


def test_artifact_realize_model_runtime_uses_local_ready_restore(monkeypatch):
    monkeypatch.setattr(
        integration_mod,
        "read_source_bound_contract_state",
        lambda: SimpleNamespace(
            source_bound_contract_ready=True,
            source_bound_contract_version=4,
            source_bound_capability_names=("collective",),
        ),
    )

    def reject_runtime_session(*_args, **_kwargs):
        raise AssertionError("direct model-runtime path must not start a session")

    monkeypatch.setattr(
        integration_mod.ArtifactRuntimeSession,
        "from_config",
        classmethod(reject_runtime_session),
    )
    monkeypatch.setattr(
        integration_mod.ArtifactRuntimeSession,
        "start",
        reject_runtime_session,
    )

    class _NoMaterializeArtifact:
        def subset(self, _names):
            raise AssertionError(
                "local-ready direct path must not bind source artifact"
            )

        def tensor_dict(self, **_kwargs):
            raise AssertionError("local-ready direct path must not use TensorDict")

        def state_dict(self):
            raise AssertionError("local-ready direct path must not build state dict")

    restored = _FakeRestoredRetainedBinding()
    restore_calls = []

    @contextmanager
    def fake_restore_prepared(**kwargs):
        restore_calls.append(kwargs)
        assert kwargs["expected_member"].member_id == "member-0"
        yield restored

    monkeypatch.setattr(
        integration_mod,
        "restore_prepared_local_ready_binding",
        fake_restore_prepared,
    )

    class _Resolver:
        def resolve(self, artifact_ref):
            return SimpleNamespace(
                artifact=_NoMaterializeArtifact(),
                artifact_ref=artifact_ref,
                tensor_names=("w",),
                manifest=SimpleNamespace(
                    representation_contract_hash="repr-local",
                    source_artifact_ref="mi2:source",
                    serving_build_digest="build-local",
                    local_serving_ref="binding-local:binding-1:value-1",
                ),
            )

        def cross_check(self, resolved, **_kwargs):
            return resolved

    class _Store:
        pass

    artifact = Artifact(
        store_ref=weakref.ref(_Store()),
        artifact_id="mi2:serving-local",
        canonical_index_bytes=b"index",
    )
    handle = artifact.realize(
        tc.ArtifactRealizationSpec.model_runtime(
            framework="fakefw",
            device=torch.device("cuda:0"),
        ),
        runtime_host=tc.RuntimeHostCapabilities(
            framework=_FakeFrameworkHost(),
            placement=_FakePlacementHost(),
            tensor_surface=_FakeTensorSurface(),
        ),
        runtime_context=RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        runtime_resolver=_Resolver(),
    )

    attachment = handle.attachment()
    assert attachment.state.runtime_view.readiness == "runtime_local_ready"
    assert attachment.state.runtime_view.local_serving_ref == (
        "binding-local:binding-1:value-1"
    )
    assert torch.equal(
        attachment.model.tensors["w"], torch.ones((1,), dtype=torch.float16)
    )
    assert restored.transferred
    assert not restored.closed
    assert restore_calls
    assert restore_calls[0]["resolved_artifact"].artifact_ref == "mi2:serving-local"
    assert handle.report.artifact_id == "mi2:serving-local"
    assert handle.report.artifact_profile == "durable_artifact"
    assert handle.report.authority_scope == "daemon_mediated_durable"
    assert handle.report.lifecycle_plan is not None
    assert handle.report.lifecycle_plan.retained is True
    assert handle.report.runtime_attach_sec is not None
    assert handle.report.runtime_attach_sec >= 0.0
    assert handle.report.runtime_finalize_sec is not None
    assert handle.report.runtime_finalize_sec >= 0.0
    assert handle.report.total_sec is not None
    assert handle.report.total_sec >= handle.report.runtime_attach_sec
    assert handle.report.envelope.retained_bytes == restored.reservation_bytes
    assert handle.report.envelope.release_policy == (
        "close_runtime_attachment",
        "release_placement_lease",
    )


def test_artifact_realize_model_runtime_uses_mounted_source_artifact(monkeypatch):
    source_artifact_ref = "msa1:test-source"
    calls = []
    host_binding = _FakeBinding()

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
        ArtifactRuntimeIntegration,
        "resolve_source_subject",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mounted-source artifact already owns the source subject")
        ),
    )
    monkeypatch.setattr(
        ArtifactRuntimeIntegration,
        "build_materialization_options",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mounted-source model_runtime must preserve spec options")
        ),
    )

    class _Provider:
        def build_catalog(self, request):
            calls.append(("catalog", request))
            return SimpleNamespace(
                source_artifact_ref=request.source_artifact_ref,
                metadata_fingerprint="meta",
                ordered_names=("w",),
                meta_by_name={},
                selected_files=(),
            )

    class _RecipeSession:
        def build_recipe(self, **kwargs):
            calls.append(("recipe", kwargs))
            return SimpleNamespace(
                recipe=_recipe(source_artifact_ref=source_artifact_ref),
                diagnostics={"compile_key": "compile"},
            )

    monkeypatch.setattr(
        ArtifactRuntimeIntegration,
        "build_recipe_session",
        lambda self, request: calls.append(("session", request)) or _RecipeSession(),
    )

    def fake_realize_local_ready_binding_from_source(**kwargs):
        calls.append(("prepare", kwargs))
        update_epoch = host_binding.realize_from(
            kwargs["source_subject"],
            realization_plan=kwargs["recipe"].realization_plan_proto,
            options=kwargs["options"],
        )
        return SimpleNamespace(
            binding=host_binding,
            update_epoch=update_epoch,
            layout=SimpleNamespace(binding_layout_id="layout-1"),
            realization_entry_count=1,
        )

    monkeypatch.setattr(
        local_ready_mod,
        "realize_local_ready_binding_from_source",
        fake_realize_local_ready_binding_from_source,
    )

    class _Store:
        pass

    source_handle = tc.PublicDiskSourceHandle(
        path="/tmp/fake-model",
        canonical_index_bytes=b"index",
        artifact_id=source_artifact_ref,
        generation=1,
    )
    artifact = Artifact(
        store_ref=weakref.ref(_Store()),
        artifact_id=source_artifact_ref,
        source_subject=source_handle,
    )
    host = tc.RuntimeHostCapabilities(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
        source_catalog=_Provider(),
    )
    materialization_options = tc.GetArtifactOptions()

    handle = artifact.realize(
        tc.ArtifactRealizationSpec.model_runtime(
            framework="fakefw",
            device=torch.device("cuda:0"),
            adapter_version="adapter-v1",
            options=materialization_options,
        ),
        runtime_host=host,
        runtime_context=RequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
    )
    attachment = handle.attachment()

    assert attachment.state.runtime_view.readiness == "runtime_local_ready"
    assert attachment.state.runtime_view.source_artifact_ref == source_artifact_ref
    assert handle.report.target_kind == "model_runtime"
    assert handle.report.model_runtime.framework == "fakefw"
    assert handle.report.model_runtime.adapter_version == "adapter-v1"
    assert handle.report.artifact_profile == "mounted_source"
    assert handle.report.authority_scope == "daemon_local_mounted_source"
    assert handle.report.logical_layout_hash
    assert calls[0][0] == "catalog"
    assert calls[0][1].source_selector == SourceSelector.local_path("/tmp/fake-model")
    assert calls[0][1].source_subject.subject is source_handle
    assert calls[0][1].source_artifact_ref == source_artifact_ref
    assert calls[2][0] == "recipe"
    assert calls[2][1]["source_catalog"].source_artifact_ref == source_artifact_ref
    assert calls[3][0] == "prepare"
    assert calls[3][1]["source_subject"] is source_handle
    assert host_binding.realized[2] is materialization_options


def test_fake_second_framework_uses_direct_artifact_runtime_api(monkeypatch):
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
        integration_mod.ArtifactRuntimeIntegration,
        "build_materialization_options",
        lambda self, **kwargs: ("runtime-options", kwargs),
    )

    def reject_runtime_session(*_args, **_kwargs):
        raise AssertionError("second-runtime proof must use artifact runtime API")

    monkeypatch.setattr(
        integration_mod.ArtifactRuntimeSession,
        "from_config",
        classmethod(reject_runtime_session),
    )
    monkeypatch.setattr(
        integration_mod.ArtifactRuntimeSession,
        "start",
        reject_runtime_session,
    )
    monkeypatch.setattr(
        integration_mod.ArtifactRuntimeSession,
        "reload",
        reject_runtime_session,
    )

    resolver_calls = []

    class _Resolver:
        def resolve(self, artifact_ref):
            resolver_calls.append(("resolve", artifact_ref))
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
            resolver_calls.append(("cross_check", kwargs))
            return resolved_artifact

    host = tc.RuntimeHostCapabilities(
        framework=_FakeFrameworkHost(),
        placement=_FakePlacementHost(),
        tensor_surface=_FakeTensorSurface(),
    )
    resolver = _Resolver()

    class _Store:
        pass

    artifact = Artifact(
        store_ref=weakref.ref(_Store()),
        artifact_id="mi2:serving",
    )

    handle = artifact.realize(
        tc.ArtifactRealizationSpec.model_runtime(
            framework="fakefw",
            device=torch.device("cuda:0"),
            adapter_version="adapter-v1",
            runtime_abi_version="abi-v1",
        ),
        runtime_host=host,
        runtime_context=tc.RuntimeRequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        runtime_resolver=resolver,
    )
    attachment = handle.attachment()
    reloaded = tc.reload_runtime_attachment(
        current_attachment=attachment,
        artifact_locator=tc.ArtifactLocator.artifact_ref("mi2:serving-next"),
        policy=tc.RuntimePolicy(),
        runtime_host=host,
        runtime_context=tc.RuntimeRequestContext(
            framework_config=SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        ensure_runtime_initialized=lambda: None,
        model=attachment.model,
        runtime_resolver=resolver,
    )

    assert (
        attachment.view.endpoint.to_weight_version_payload()["serving_artifact_ref"]
        == "mi2:serving"
    )
    assert (
        reloaded.view.endpoint.to_reload_response_payload()["serving_artifact_ref"]
        == "mi2:serving-next"
    )
    assert handle.report.target_kind == "model_runtime"
    assert handle.report.model_runtime.framework == "fakefw"
    assert reloaded.state.runtime_view.serving_artifact_ref == "mi2:serving-next"
    assert resolver_calls[0] == ("resolve", "mi2:serving")
    assert ("resolve", "mi2:serving-next") in resolver_calls


def test_fake_second_framework_artifact_runtime_conformance_kit():
    from tensorcast.artifact_runtime.testing import (
        assert_level1_artifact_runtime_conformance,
    )

    result = assert_level1_artifact_runtime_conformance(tc)

    assert result.checks["direct_start"]
    assert result.checks["artifact_realization_report"]
    assert result.checks["runtime_session_not_required"]
    assert result.checks["target_layout_from_runtime_binding"]
    assert result.checks["runtime_only_tensors_allocated"]
    assert result.checks["runtime_publication_actions"]
    assert result.checks["reload"]
    assert result.checks["describe"]
    assert result.checks["source_capability_not_required"]
    assert result.checks["source_catalog_not_required"]
    assert result.checks["resolver_uses_artifact_refs"]
    assert result.checks["rejects_local_reload_artifact_locator"]
    assert result.checks["rejects_untyped_reload_artifact_locator"]
    assert result.checks["rejects_untyped_reload_policy"]


def test_conformance_failure_summary_includes_onboarding_hint():
    from tensorcast.artifact_runtime.testing import ConformanceResult

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
