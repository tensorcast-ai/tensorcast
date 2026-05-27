#  Copyright (c) 2026, TensorCast Team.
"""Reusable conformance checks for framework artifact-runtime integrations."""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import torch

import tensorcast as tc
import tensorcast.artifact_runtime.lifecycle as _integration


@dataclass(frozen=True)
class ConformanceResult:
    """Result from a lightweight artifact runtime conformance check."""

    checks: Mapping[str, bool] = field(default_factory=dict)
    messages: Mapping[str, str] = field(default_factory=dict)
    level: str | None = None

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks.items() if not passed)

    def failure_summary(self) -> str:
        failed = self.failed_checks
        if not failed:
            return "TensorCast artifact-runtime conformance checks passed"
        lines = [
            "TensorCast artifact-runtime conformance checks failed"
            + (f" for {self.level}" if self.level else "")
            + ":"
        ]
        for name in failed:
            message = self.messages.get(name, "No remediation hint available")
            lines.append(f"- {name}: {message}")
        return "\n".join(lines)

    def assert_passed(self) -> None:
        if self.failed_checks:
            raise AssertionError(self.failure_summary())


def _result(
    *,
    level: str,
    checks: Mapping[str, bool],
    messages: Mapping[str, str],
) -> ConformanceResult:
    result = ConformanceResult(checks=checks, messages=messages, level=level)
    result.assert_passed()
    return result


_PUBLIC_BOUNDARY_MESSAGES = {
    "hides_runtime_session": (
        "Do not expose ArtifactRuntimeSession from the public runtime API; "
        "frameworks should use Artifact.realize(... model_runtime ...) and "
        "artifact-runtime actions."
    ),
    "has_attachment": (
        "Expose RuntimeAttachment as the framework-held lifecycle token."
    ),
    "has_request_context": (
        "Expose RequestContext so framework facts enter lifecycle calls through "
        "one typed context object."
    ),
    "hides_admin_local_bootstrap": (
        "Keep admin/local-bootstrap override DTOs out of the framework runtime "
        "module; route them through admin/offline surfaces."
    ),
    "hides_low_level_bind": (
        "Do not expose bind/swap/restore helpers from the runtime module; "
        "frameworks should use artifact-runtime start/reload/publication actions."
    ),
    "hides_serving_locator_policy": (
        "Keep serving-rooted locator and policy aliases out of "
        "the public runtime API; use ArtifactLocator, RuntimePolicy, and "
        "runtime reload helpers."
    ),
    "hides_legacy_config": (
        "Keep serving-rooted config and start-plan names out of "
        "the public runtime API; use TensorCastRuntimeConfig and "
        "plan_runtime_start."
    ),
    "hides_projection_dtos": (
        "Runtime endpoint projection DTOs live in tensorcast.artifact_runtime.view."
    ),
    "hides_state_helpers": (
        "Model attribute helpers live in tensorcast.artifact_runtime.state."
    ),
}

_ARTIFACT_RUNTIME_BOUNDARY_MESSAGES = {
    "has_artifact_realization_spec": (
        "Expose ArtifactRealizationSpec so frameworks can request model_runtime "
        "realization through the artifact API."
    ),
    "has_runtime_host": (
        "Expose RuntimeHostCapabilities as the framework-provided host surface."
    ),
    "has_runtime_context": (
        "Expose RuntimeRequestContext so framework facts enter runtime actions "
        "through one typed context object."
    ),
    "has_artifact_locator": (
        "Expose ArtifactLocator for durable artifact runtime reload requests."
    ),
    "has_runtime_policy": ("Expose RuntimePolicy for typed runtime reload admission."),
    "has_reload_action": (
        "Expose reload_runtime_attachment for runtime reload without a serving "
        "session object."
    ),
    "has_publication_actions": (
        "Expose runtime replica publish/retire actions without requiring a "
        "runtime session object."
    ),
    "hides_runtime_session": (
        "The tensorcast root runtime path must not expose ArtifactRuntimeSession; "
        "frameworks should use Artifact.realize(... model_runtime ...) instead."
    ),
    "hides_legacy_serving_dtos": (
        "Keep legacy serving-rooted DTO aliases off the tensorcast root runtime "
        "surface."
    ),
}

_FRAMEWORK_ISOLATION_MESSAGES = {
    "no_vllm_imports": (
        "Reference and conformance frameworks must not import vLLM. Move any "
        "needed generic fact extraction into TensorCast hosts or testing helpers."
    ),
    "no_internal_runtime_imports": (
        "Framework examples should not import TensorCast private/internal "
        "runtime modules."
    ),
    "no_serving_imports": (
        "Framework examples should not import the removed tensorcast.serving "
        "package; use tensorcast.artifact_runtime host/testing surfaces instead."
    ),
}


def assert_public_artifact_runtime_boundary(
    tc_module: ModuleType = tc,
) -> ConformanceResult:
    """Check that the root API exposes artifact-runtime, not serving-session, APIs."""

    public_names = set(getattr(tc_module, "__all__", ()))
    checks = {
        "has_artifact_realization_spec": "ArtifactRealizationSpec" in public_names,
        "has_runtime_host": "RuntimeHostCapabilities" in public_names,
        "has_runtime_context": "RuntimeRequestContext" in public_names,
        "has_artifact_locator": "ArtifactLocator" in public_names,
        "has_runtime_policy": "RuntimePolicy" in public_names,
        "has_reload_action": "reload_runtime_attachment" in public_names,
        "has_publication_actions": {
            "publish_runtime_replica",
            "retire_runtime_replica",
        }.issubset(public_names),
        "hides_runtime_session": "ArtifactRuntimeSession" not in public_names,
        "hides_legacy_serving_dtos": {
            "ServingBuildIntent",
            "ServingArtifactManifest",
            "ServingRuntimePolicy",
            "ServingBindingTarget",
            "ServingBindingSetTarget",
            "PrefetchedServingBinding",
            "PrefetchedServingBindingSet",
        }.isdisjoint(public_names),
    }
    return _result(
        level="public-artifact-runtime-boundary",
        checks=checks,
        messages=_ARTIFACT_RUNTIME_BOUNDARY_MESSAGES,
    )


def assert_public_runtime_boundary(runtime_module: ModuleType) -> ConformanceResult:
    """Check that runtime imports expose framework APIs, not admin helpers."""

    public_names = set(getattr(runtime_module, "__all__", ()))
    checks = {
        "hides_runtime_session": "ArtifactRuntimeSession" not in public_names,
        "has_attachment": "RuntimeAttachment" in public_names,
        "has_request_context": "RequestContext" in public_names,
        "hides_admin_local_bootstrap": "AdminLocalSourceBootstrap" not in public_names
        and "_AdminLocalSourceBootstrap" not in public_names,
        "hides_low_level_bind": "bind_runtime_artifact" not in public_names
        and "swap_runtime_artifact" not in public_names
        and "restore_retained_binding" not in public_names,
        "hides_serving_locator_policy": {
            "ServingArtifactLocator",
            "ServingPolicy",
            "merge_serving_reload_extra_config",
            "normalize_serving_reload_request_payload",
        }.isdisjoint(public_names),
        "hides_legacy_config": {
            "ServingConfig",
            "ServingStartPlan",
            "ServingStartPlanError",
            "plan_serving_start",
        }.isdisjoint(public_names)
        and "TensorCastRuntimeConfig" in public_names
        and "plan_runtime_start" in public_names,
        "hides_projection_dtos": {
            "PublishedReplicaProjection",
            "ReloadResponseProjection",
            "RuntimeEndpointProjection",
            "SourceSelectionProjection",
            "WeightVersionProjection",
        }.isdisjoint(public_names),
        "hides_state_helpers": {
            "ModelAttributeRuntimeState",
            "RuntimeAttachmentRecord",
            "RuntimeAttachmentStore",
        }.isdisjoint(public_names),
    }
    return _result(
        level="public-runtime-boundary",
        checks=checks,
        messages=_PUBLIC_BOUNDARY_MESSAGES,
    )


def assert_framework_isolation(module_names: Iterable[str]) -> ConformanceResult:
    """Check that a fake/reference framework avoids vLLM imports."""

    names = tuple(str(name) for name in module_names)
    checks = {
        "no_vllm_imports": not any(
            name == "vllm" or name.startswith("vllm.") for name in names
        ),
        "no_internal_runtime_imports": not any(
            name.startswith("tensorcast.serving.internal") for name in names
        ),
        "no_serving_imports": not any(
            name == "tensorcast.serving" or name.startswith("tensorcast.serving.")
            for name in names
        ),
    }
    return _result(
        level="framework-isolation",
        checks=checks,
        messages=_FRAMEWORK_ISOLATION_MESSAGES,
    )


class FakeArtifactView:
    def __init__(self, names: Iterable[str] = ()) -> None:
        self.names = tuple(names)

    def bind(self, **kwargs: Any) -> "FakeBinding":
        binding = FakeBinding()
        binding.names = self.names
        binding.bind_kwargs = kwargs
        return binding


class FakeArtifact:
    def subset(self, names: Iterable[str]) -> FakeArtifactView:
        return FakeArtifactView(names)


class FakeBinding:
    def __init__(self) -> None:
        self.tensors = {"w": torch.ones((1,), dtype=torch.float16)}
        self.binding_layout_id = "layout-1"
        self.current_value = SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
        )
        self.names: tuple[str, ...] = ()
        self.bind_kwargs: dict[str, Any] = {}
        self.swapped: tuple[object, dict[str, Any]] | None = None
        self.published_lease_id: str | None = None
        self.published_replica_id: str | None = None
        self.publish_calls = 0
        self.retire_calls: list[float | None] = []
        self.closed = False

    def swap(self, artifact: object, **kwargs: Any) -> "FakeBinding":
        self.swapped = (artifact, kwargs)
        self.tensors = {"w": torch.full((1,), 2.0, dtype=torch.float16)}
        return self

    def publish_replica(self) -> object:
        self.publish_calls += 1
        self.published_lease_id = "lease-1"
        self.published_replica_id = "replica-1"
        return SimpleNamespace(
            binding_id=self.current_value.binding_id,
            binding_layout_id=self.current_value.binding_layout_id,
            binding_value_id=self.current_value.binding_value_id,
            seal_generation=self.current_value.seal_generation,
            replica_id=self.published_replica_id,
            lease_id=self.published_lease_id,
            serving_artifact_id="mi2:serving",
            device_uuid="gpu-0",
        )

    def retire(self, *, drain_timeout_s: float | None = None) -> None:
        self.retire_calls.append(drain_timeout_s)
        self.published_lease_id = None
        self.published_replica_id = None

    def close(self) -> None:
        self.closed = True


class FakeRuntimeModel:
    def __init__(self) -> None:
        self.tensors = {"w": torch.empty((1,), dtype=torch.float16, device="meta")}


class FakeFrameworkHost:
    def identity(self, model_config: object) -> _integration.FrameworkIdentity:
        del model_config
        return _integration.FrameworkIdentity(
            framework_name="fakefw",
            framework_version="fakefw-v1",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
        )

    def prepare_model_construction(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> None:
        del framework_config, model_config

    def build_meta_model(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> FakeRuntimeModel:
        del framework_config, model_config
        return FakeRuntimeModel()

    def build_runtime_model(
        self,
        framework_config: object | None,
        model_config: object | None,
        target_device: object | None,
    ) -> FakeRuntimeModel:
        del framework_config, model_config, target_device
        return FakeRuntimeModel()

    def assert_model_ready_for_runtime_binding(
        self,
        model: FakeRuntimeModel,
        *,
        context: object,
    ) -> None:
        del context
        if "w" not in model.tensors:
            raise AssertionError("fake model missing runtime tensor 'w'")

    def semantic_probes(
        self,
        model: FakeRuntimeModel,
        model_config: object | None,
    ) -> dict[str, object]:
        del model, model_config
        return {}


class FakePlacementHost:
    def identity_facts(
        self,
        framework_config: object | None,
    ) -> _integration.PlacementIdentityFacts:
        del framework_config
        return _integration.PlacementIdentityFacts(
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_rank=0,
            pipeline_parallel_size=1,
            data_parallel_rank=0,
            data_parallel_size=1,
        )

    def admission_facts(
        self,
        framework_config: object | None,
    ) -> _integration.PlacementAdmissionFacts:
        del framework_config
        return _integration.PlacementAdmissionFacts()

    def member_facts(
        self,
        framework_config: object | None,
    ) -> _integration.PlacementMemberFacts:
        del framework_config
        return _integration.PlacementMemberFacts(
            runtime_rank=0,
            runtime_world_size=1,
            member_id="member-0",
            member_index=0,
            member_count=1,
            group_id_hint="group-1",
        )

    def execution_facts(
        self,
        framework_config: object | None,
    ) -> _integration.MaterializationExecutionFacts:
        del framework_config
        return _integration.MaterializationExecutionFacts(
            collective_rank=0,
            collective_world_size=1,
            tensor_parallel_ranks=(0,),
        )


class FakeTensorSurface:
    def runtime_only_tensor_names(self, model: FakeRuntimeModel) -> tuple[str, ...]:
        del model
        return ()

    def align_runtime_tensor_names(
        self,
        model: FakeRuntimeModel,
        expected_names: Iterable[str],
    ) -> int:
        if set(expected_names) != set(model.tensors):
            raise AssertionError("fake runtime tensor names do not match")
        return 0

    def collect_runtime_tensors(
        self,
        model: FakeRuntimeModel,
        *,
        remove_duplicate: bool = False,
    ) -> dict[str, object]:
        del remove_duplicate
        return dict(model.tensors)

    def collect_runtime_tensor_view(
        self,
        tensors: Mapping[str, object],
    ) -> tuple[object, ...]:
        del tensors
        return ()

    def compute_runtime_tensor_schema_hash(
        self,
        tensors: Mapping[str, object],
        *,
        remove_duplicate: bool = False,
    ) -> str:
        del tensors, remove_duplicate
        return "fake-schema"

    def attach_bound_tensors(
        self,
        model: FakeRuntimeModel,
        tensors: Mapping[str, object],
        *,
        replace_meta_params: bool,
    ) -> FakeRuntimeModel:
        del replace_meta_params
        model.tensors.update(cast(Mapping[str, torch.Tensor], tensors))
        return model

    def allocate_runtime_only_tensors(
        self,
        model: FakeRuntimeModel,
        target_device: torch.device,
    ) -> dict[str, object]:
        del model, target_device
        return {}

    def snapshot_tensor_invariants(
        self,
        tensors: Mapping[str, object],
    ) -> tuple[str, ...]:
        return tuple(sorted(tensors))

    def validate_tensor_invariants(
        self,
        before: tuple[str, ...],
        after: Mapping[str, object],
    ) -> None:
        if before != tuple(sorted(after)):
            raise AssertionError("fake tensor invariants changed")


class FakeRuntimeOnlyTensorSurface(FakeTensorSurface):
    def __init__(self) -> None:
        self.allocated: list[tuple[str, torch.device]] = []

    def runtime_only_tensor_names(self, model: FakeRuntimeModel) -> tuple[str, ...]:
        del model
        return ("cache",)

    def collect_runtime_tensors(
        self,
        model: FakeRuntimeModel,
        *,
        remove_duplicate: bool = False,
    ) -> dict[str, object]:
        del remove_duplicate
        return {
            name: tensor for name, tensor in model.tensors.items() if name != "cache"
        }

    def allocate_runtime_only_tensors(
        self,
        model: FakeRuntimeModel,
        target_device: torch.device,
    ) -> dict[str, object]:
        self.allocated.append(("cache", target_device))
        tensor = torch.zeros((1,), dtype=torch.float16)
        model.tensors["cache"] = tensor
        return {"cache": tensor}


class FakeRuntimeArtifactResolver:
    def resolve(self, artifact_ref: str) -> SimpleNamespace:
        return SimpleNamespace(
            artifact=FakeArtifact(),
            artifact_ref=artifact_ref,
            tensor_names=("w",),
            manifest=SimpleNamespace(
                representation_contract_hash=f"repr:{artifact_ref}",
                source_artifact_ref="mi2:source",
                serving_build_digest=f"build:{artifact_ref}",
            ),
        )

    def cross_check(
        self,
        resolved_artifact: SimpleNamespace,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        return resolved_artifact


class RecordingRuntimeArtifactResolver(FakeRuntimeArtifactResolver):
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def resolve(self, artifact_ref: str) -> SimpleNamespace:
        self.calls.append(("resolve", artifact_ref))
        return super().resolve(artifact_ref)

    def cross_check(
        self,
        resolved_artifact: SimpleNamespace,
        **kwargs: object,
    ) -> SimpleNamespace:
        self.calls.append(("cross_check", dict(kwargs)))
        return super().cross_check(resolved_artifact, **kwargs)


class _LocalPathLocator:
    kind = "local_path"
    value = "/tmp/fakefw-model"


def build_fake_artifact_runtime_host(
    tc_module: ModuleType = tc,
    *,
    tensor_surface: object | None = None,
) -> object:
    """Build a minimal non-vLLM host through the root artifact-runtime API."""

    return tc_module.RuntimeHostCapabilities(
        framework=FakeFrameworkHost(),
        placement=FakePlacementHost(),
        tensor_surface=tensor_surface or FakeTensorSurface(),
    )


_ARTIFACT_LEVEL1_MESSAGES = {
    "direct_start": (
        "Artifact model_runtime startup failed. Verify framework model "
        "construction, tensor surface attach/schema behavior, placement facts, "
        "and artifact resolver output."
    ),
    "artifact_realization_report": (
        "Artifact.realize(... model_runtime ...) must return a model_runtime "
        "realization report for the requested framework."
    ),
    "runtime_session_not_required": (
        "Level 1 artifact-runtime start/reload must not instantiate or call "
        "ArtifactRuntimeSession."
    ),
    "target_layout_from_runtime_binding": (
        "Model-runtime reports must carry target layout identity from the "
        "runtime attachment binding."
    ),
    "runtime_only_tensors_allocated": (
        "Runtime-only tensor allocation must be expressible through the neutral "
        "RuntimeHostCapabilities tensor surface."
    ),
    "runtime_publication_actions": (
        "Runtime publication must be represented by artifact-runtime "
        "publish/retire actions, not by a runtime session."
    ),
    "describe": (
        "RuntimeAttachment.view must expose the typed RuntimeWorkerView for the "
        "current attachment."
    ),
    "reload": (
        "Artifact runtime reload failed. Level 1 reload must use a typed "
        "ArtifactLocator and RuntimePolicy."
    ),
    "reload_identity_from_runtime_view": (
        "Reload response identity must come from the runtime view, not from the "
        "request payload."
    ),
    "source_capability_not_required": (
        "Level 1 direct artifact runtime start/reload must not require SourceHost."
    ),
    "source_catalog_not_required": (
        "Level 1 direct artifact runtime start/reload must not require "
        "SourceCatalogProvider."
    ),
    "resolver_uses_artifact_refs": (
        "Artifact runtime start/reload must resolve durable artifact refs through "
        "the supplied runtime resolver."
    ),
    "rejects_local_reload_artifact_locator": (
        "Reload must reject local source selectors; local paths belong to "
        "source bootstrap, not durable artifact runtime reload."
    ),
    "rejects_untyped_reload_artifact_locator": (
        "Reload must reject untyped artifact locator dictionaries on the public "
        "runtime path. Use ArtifactLocator."
    ),
    "rejects_untyped_reload_policy": (
        "Reload must reject untyped policy dictionaries on the public runtime "
        "path. Use RuntimePolicy."
    ),
}


@contextmanager
def _patched_direct_artifact_runtime():
    integration_module = cast(Any, _integration)
    original_contract_reader = integration_module.read_source_bound_contract_state
    original_materialization_options = (
        integration_module.ArtifactRuntimeIntegration.build_materialization_options
    )
    integration_module.read_source_bound_contract_state = lambda: SimpleNamespace(
        source_bound_contract_ready=True,
        source_bound_contract_version=4,
        source_bound_capability_names=("collective",),
    )
    integration_module.ArtifactRuntimeIntegration.build_materialization_options = (
        lambda self, **kwargs: ("fake-materialization-options", kwargs)
    )
    try:
        yield
    finally:
        integration_module.read_source_bound_contract_state = original_contract_reader
        integration_module.ArtifactRuntimeIntegration.build_materialization_options = (
            original_materialization_options
        )


@contextmanager
def _reject_artifact_runtime_session():
    session_cls = cast(Any, _integration.ArtifactRuntimeSession)
    original_from_config = session_cls.__dict__["from_config"]
    original_start = session_cls.__dict__["start"]
    original_reload = session_cls.__dict__["reload"]

    def reject_runtime_session(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("artifact-runtime conformance used ArtifactRuntimeSession")

    session_cls.from_config = classmethod(reject_runtime_session)
    session_cls.start = reject_runtime_session
    session_cls.reload = reject_runtime_session
    try:
        yield
    finally:
        session_cls.from_config = original_from_config
        session_cls.start = original_start
        session_cls.reload = original_reload


def assert_level1_artifact_runtime_conformance(
    tc_module: ModuleType = tc,
    *,
    host: object | None = None,
) -> ConformanceResult:
    """Run Level 1 durable model-runtime conformance through Artifact.realize."""

    from tensorcast.api.store.artifact import Artifact

    checks: dict[str, bool] = {}
    assert_public_artifact_runtime_boundary(tc_module)
    assert_framework_isolation((tc_module.__name__, __name__))

    class _Store:
        pass

    with _patched_direct_artifact_runtime(), _reject_artifact_runtime_session():
        store = _Store()
        tensor_surface = None if host is not None else FakeRuntimeOnlyTensorSurface()
        runtime_host = host or build_fake_artifact_runtime_host(
            tc_module,
            tensor_surface=tensor_surface,
        )
        model_config = SimpleNamespace(model="fake-model")
        identity = runtime_host.framework.identity(model_config)
        resolver = RecordingRuntimeArtifactResolver()
        store_ref: weakref.ReferenceType[Any] = weakref.ref(store)
        artifact = Artifact(
            store_ref=store_ref,
            artifact_id="mi2:serving",
        )
        handle = artifact.realize(
            tc_module.ArtifactRealizationSpec.model_runtime(
                framework=str(identity.framework_name),
                device=torch.device("cuda:0"),
                adapter_version=str(identity.adapter_version),
                runtime_abi_version=str(identity.serving_abi_version),
            ),
            runtime_host=runtime_host,
            runtime_context=tc_module.RuntimeRequestContext(
                framework_config=SimpleNamespace(),
                model_config=model_config,
            ),
            runtime_resolver=resolver,
        )
        attachment = handle.attachment()
        model_runtime_report = handle.report.model_runtime
        target_plan = handle.report.target_plan
        direct_payload = attachment.view.endpoint.to_weight_version_payload()
        checks["direct_start"] = (
            direct_payload.get("serving_artifact_ref") == "mi2:serving"
            and direct_payload.get("source_artifact_ref") == "mi2:source"
        )
        checks["artifact_realization_report"] = (
            handle.report.target_kind == "model_runtime"
            and model_runtime_report is not None
            and model_runtime_report.framework == str(identity.framework_name)
        )
        checks["target_layout_from_runtime_binding"] = (
            handle.report.target_layout_digest == "binding-layout:layout-1"
            and target_plan is not None
            and target_plan.target_layout_digest == "binding-layout:layout-1"
        )
        checks["runtime_only_tensors_allocated"] = (
            True
            if tensor_surface is None
            else (
                "cache" in attachment.model.tensors
                and ("cache", torch.device("cuda:0")) in tensor_surface.allocated
            )
        )
        publication_events: list[Mapping[str, object]] = []
        published = tc_module.publish_runtime_replica(
            current_attachment=attachment,
            policy=SimpleNamespace(
                mode="required",
                timeout_s=0.0,
                drain_timeout_s=0.0,
            ),
            ensure_runtime_initialized=lambda: None,
            profile_sink=publication_events.append,
        )
        published_replica = published.view.endpoint.weight_version.published_replica
        retired = tc_module.retire_runtime_replica(
            current_attachment=published,
            reason="conformance",
            drain_timeout_s=0.0,
            ensure_runtime_initialized=lambda: None,
            profile_sink=publication_events.append,
        )
        retired_replica = retired.view.endpoint.weight_version.published_replica
        published_binding = published.state.binding
        checks["runtime_publication_actions"] = (
            published_replica is not None
            and published_replica.state == "published"
            and published_replica.replica_id == "replica-1"
            and retired_replica is not None
            and retired_replica.state == "retired"
            and getattr(published_binding, "publish_calls", 0) == 1
            and getattr(published_binding, "retire_calls", ()) == [0.0]
            and [event["event"] for event in publication_events]
            == [
                "runtime_publication.publish.done",
                "runtime_publication.retire.done",
            ]
        )
        checks["describe"] = (
            attachment.view.endpoint.to_weight_version_payload().get(
                "serving_artifact_ref"
            )
            == "mi2:serving"
        )

        reloaded = tc_module.reload_runtime_attachment(
            current_attachment=retired,
            artifact_locator=tc_module.ArtifactLocator.artifact_ref("mi2:serving-next"),
            policy=tc_module.RuntimePolicy(),
            runtime_host=runtime_host,
            runtime_context=tc_module.RuntimeRequestContext(
                framework_config=SimpleNamespace(),
                model_config=SimpleNamespace(model="fake-model"),
            ),
            ensure_runtime_initialized=lambda: None,
            model=attachment.model,
            runtime_resolver=resolver,
        )
        reload_response = reloaded.view.endpoint.to_reload_response_payload()
        checks["reload"] = (
            reload_response is not None
            and reload_response.get("serving_artifact_ref") == "mi2:serving-next"
        )
        checks["reload_identity_from_runtime_view"] = (
            reload_response is not None
            and reloaded.state.runtime_view.serving_artifact_ref
            == reload_response.get("serving_artifact_ref")
        )
        checks["source_capability_not_required"] = True
        checks["source_catalog_not_required"] = True
        checks["resolver_uses_artifact_refs"] = (
            "resolve",
            "mi2:serving",
        ) in resolver.calls and ("resolve", "mi2:serving-next") in resolver.calls

        try:
            tc_module.reload_runtime_attachment(
                current_attachment=reloaded,
                artifact_locator=_LocalPathLocator(),
                policy=tc_module.RuntimePolicy(),
                runtime_host=runtime_host,
                runtime_context=tc_module.RuntimeRequestContext(),
                ensure_runtime_initialized=lambda: None,
            )
        except _integration.ConfigConflictError:
            checks["rejects_local_reload_artifact_locator"] = True
        else:
            checks["rejects_local_reload_artifact_locator"] = False

        try:
            tc_module.reload_runtime_attachment(
                current_attachment=reloaded,
                artifact_locator={
                    "kind": "artifact_ref",
                    "value": "mi2:serving-next",
                },
                policy=tc_module.RuntimePolicy(),
                runtime_host=runtime_host,
                runtime_context=tc_module.RuntimeRequestContext(),
                ensure_runtime_initialized=lambda: None,
            )
        except _integration.ConfigConflictError:
            checks["rejects_untyped_reload_artifact_locator"] = True
        else:
            checks["rejects_untyped_reload_artifact_locator"] = False

        try:
            tc_module.reload_runtime_attachment(
                current_attachment=reloaded,
                artifact_locator=tc_module.ArtifactLocator.artifact_ref(
                    "mi2:serving-next"
                ),
                policy={"mode": "from_manifest"},
                runtime_host=runtime_host,
                runtime_context=tc_module.RuntimeRequestContext(),
                ensure_runtime_initialized=lambda: None,
            )
        except _integration.ConfigConflictError:
            checks["rejects_untyped_reload_policy"] = True
        else:
            checks["rejects_untyped_reload_policy"] = False
        checks["runtime_session_not_required"] = True

    return _result(
        level="level1-artifact-runtime",
        checks=checks,
        messages=_ARTIFACT_LEVEL1_MESSAGES,
    )


__all__ = [
    "ConformanceResult",
    "FakeArtifact",
    "FakeArtifactView",
    "FakeBinding",
    "FakeFrameworkHost",
    "FakePlacementHost",
    "FakeRuntimeOnlyTensorSurface",
    "FakeRuntimeModel",
    "FakeRuntimeArtifactResolver",
    "FakeTensorSurface",
    "RecordingRuntimeArtifactResolver",
    "assert_framework_isolation",
    "assert_level1_artifact_runtime_conformance",
    "assert_public_artifact_runtime_boundary",
    "assert_public_runtime_boundary",
    "build_fake_artifact_runtime_host",
]
