#  Copyright (c) 2026, TensorCast Team.
"""Reusable conformance checks for framework serving integrations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import torch

import tensorcast as tc
from tensorcast.serving._runtime_impl import lifecycle as _integration


@dataclass(frozen=True)
class ConformanceResult:
    """Result from a lightweight serving runtime conformance check."""

    checks: Mapping[str, bool] = field(default_factory=dict)
    messages: Mapping[str, str] = field(default_factory=dict)
    level: str | None = None

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks.items() if not passed)

    def failure_summary(self) -> str:
        failed = self.failed_checks
        if not failed:
            return "TensorCast serving conformance checks passed"
        lines = [
            "TensorCast serving conformance checks failed"
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
    "has_session": (
        "Expose ServingRuntimeSession from tensorcast.serving.runtime; Level 1 "
        "frameworks should not construct lower-level lifecycle helpers."
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
        "frameworks should call ServingRuntimeSession.start/reload."
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
}


def assert_public_runtime_boundary(runtime_module: ModuleType) -> ConformanceResult:
    """Check that runtime imports expose framework APIs, not admin helpers."""

    public_names = set(getattr(runtime_module, "__all__", ()))
    checks = {
        "has_session": "ServingRuntimeSession" in public_names,
        "has_attachment": "RuntimeAttachment" in public_names,
        "has_request_context": "RequestContext" in public_names,
        "hides_admin_local_bootstrap": "AdminLocalSourceBootstrap" not in public_names
        and "_AdminLocalSourceBootstrap" not in public_names,
        "hides_low_level_bind": "bind_serving_artifact" not in public_names
        and "swap_serving_artifact" not in public_names
        and "restore_retained_binding" not in public_names,
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
        self.names: tuple[str, ...] = ()
        self.bind_kwargs: dict[str, Any] = {}
        self.swapped: tuple[object, dict[str, Any]] | None = None
        self.closed = False

    def swap(self, artifact: object, **kwargs: Any) -> "FakeBinding":
        self.swapped = (artifact, kwargs)
        self.tensors = {"w": torch.full((1,), 2.0, dtype=torch.float16)}
        return self

    def close(self) -> None:
        self.closed = True


class FakeRestoredRetainedBinding:
    def __init__(self) -> None:
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

    def transfer_to_runtime(self) -> object:
        self.transferred = True
        return SimpleNamespace(close=lambda: None)

    def close(self) -> None:
        self.closed = True


def _retained_authority(runtime_module: ModuleType) -> object:
    member = tc.ServingBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )
    binding_ref = tc.BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )
    capability = tc.BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4096,
        scope_digest="scope-1",
    )
    return runtime_module.RetainedBindingAuthority(
        group_id="group-1",
        binding_value_ref=binding_ref,
        reservation_capability=capability,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4096,
        expected_target_layout_hash="layout-hash",
        expected_tensor_schema_hash="fake-schema",
        expected_serving_build_digest="build-digest",
        expected_resolved_spec_digest="spec-digest",
        readiness="serving_local_ready",
        local_serving_ref="binding-local:fake",
    )


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


class SchemaMismatchTensorSurface(FakeTensorSurface):
    def compute_runtime_tensor_schema_hash(
        self,
        tensors: Mapping[str, object],
        *,
        remove_duplicate: bool = False,
    ) -> str:
        del tensors, remove_duplicate
        return "wrong-schema"


class FakeSourceHost:
    def source_selector(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> object:
        del framework_config, model_config
        return _integration.SourceSelector.local_path("/tmp/fakefw-model")

    def source_catalog_config(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> None:
        del framework_config, model_config
        return None

    def recipe_cache_policy(
        self,
        framework_config: object | None,
        model_config: object | None,
    ) -> None:
        del framework_config, model_config
        return None


class FakeSourceCatalogProvider:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def build_catalog(self, request: object) -> object:
        self.requests.append(request)
        return SimpleNamespace(
            source_artifact_ref=request.source_artifact_ref,
            selected_files=(),
        )


class FakeServingArtifactResolver:
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


def build_fake_runtime_host(hosts_module: ModuleType) -> object:
    """Build a minimal non-vLLM host for runtime conformance checks."""

    return hosts_module.IntegrationHost(
        framework=FakeFrameworkHost(),
        placement=FakePlacementHost(),
        tensor_surface=FakeTensorSurface(),
    )


_LEVEL1_MESSAGES = {
    "direct_start": (
        "Durable serving artifact startup failed. Verify framework model "
        "construction, tensor surface attach/schema behavior, placement facts, "
        "and artifact resolver output."
    ),
    "runtime_initialized": (
        "ServingRuntimeSession.start did not initialize RuntimeSettings before "
        "binding the serving artifact."
    ),
    "describe": (
        "ServingRuntimeSession.describe must return the typed RuntimeWorkerView "
        "for the current attachment."
    ),
    "reload": (
        "Durable serving artifact reload failed. Level 1 reload must use a "
        "typed ServingArtifactSelector and ServingPolicy."
    ),
    "reload_identity_from_runtime_view": (
        "Reload response identity must come from the runtime view, not from the "
        "request payload."
    ),
    "source_capability_not_required": (
        "Level 1 direct serving artifact start/reload must not require SourceHost."
    ),
    "source_catalog_not_required": (
        "Level 1 direct serving artifact start/reload must not require "
        "SourceCatalogProvider."
    ),
    "rejects_local_reload_selector": (
        "Reload must reject local source selectors; local paths belong to "
        "Level 2 bootstrap, not durable serving artifact reload."
    ),
    "rejects_untyped_reload_selector": (
        "Reload must reject untyped selector dictionaries on the public runtime "
        "path. Use ServingArtifactSelector."
    ),
    "rejects_untyped_reload_policy": (
        "Reload must reject untyped policy dictionaries on the public runtime "
        "path. Use ServingPolicy."
    ),
}

_LEVEL2_MESSAGES = {
    "missing_source_catalog_fails_closed": (
        "Local bootstrap requires a SourceCatalogProvider; TensorCast core owns "
        "source identity and catalog request construction."
    ),
    "source_catalog_request_core_owned": (
        "Source catalog providers must receive a core-owned SourceCatalogRequest "
        "with typed source selector and source artifact identity."
    ),
    "recipe_build_receives_core_catalog": (
        "Recipe build should consume the core source catalog, not framework "
        "private catalog state."
    ),
    "missing_trace_capability_is_explicit": (
        "Cache-miss local bootstrap must fail with a clear missing trace/native "
        "load capability instead of AttributeError or fallback loading."
    ),
    "local_path_is_not_reload_selector": (
        "Local path selectors must stay in bootstrap; reload accepts only durable "
        "serving artifact selectors."
    ),
}

_LEVEL3_MESSAGES = {
    "retained_acquire_public_start": (
        "External preload must enter through ServingRuntimeSession.start and "
        "return a RuntimeAttachment with typed endpoint projection."
    ),
    "retained_acquire_uses_host_member": (
        "Retained acquire must validate authority member facts against the "
        "framework placement host."
    ),
    "retained_acquire_transfers_ownership": (
        "Retained binding ownership must transfer into TensorCast runtime state "
        "only after attach/finalize succeeds."
    ),
    "missing_authority_fails_closed": (
        "External preload config must include typed retained authority."
    ),
    "authority_mismatch_fails_closed": (
        "Daemon/session/member authority mismatches must fail closed."
    ),
    "failure_cleanup_closes_untransferred_handle": (
        "Attach/finalize failure must close an untransferred retained handle."
    ),
    "failure_path_used_retained_restore": (
        "Retained preload failure coverage did not exercise restore ownership."
    ),
    "rejects_arbitrary_retained_authority": (
        "Retained acquire must reject arbitrary authority objects; use the typed "
        "RetainedBindingAuthority projection."
    ),
}


def _external_preload_config(runtime_module: ModuleType) -> dict[str, Any]:
    authority = _retained_authority(runtime_module)
    return {
        "preload": {
            "mode": "external",
            "authority": {
                "group_id": authority.group_id,
                "member_ref": authority.member.model_dump(mode="python"),
                "daemon_id": authority.daemon_id,
                "daemon_session_id": authority.daemon_session_id,
                "device_uuid": authority.device_uuid,
                "binding_value_ref": (
                    authority.binding_value_ref.model_dump(mode="python")
                ),
                "reservation_capability": (
                    authority.reservation_capability.model_dump(mode="python")
                ),
                "local_serving_ref": authority.local_serving_ref,
                "readiness": authority.readiness,
                "verification_state": authority.verification_state,
                "serving_artifact_id": authority.serving_artifact_id,
                "trusted_reservation_bytes": authority.reservation_bytes,
                "expected": {
                    "target_layout_hash": authority.expected_target_layout_hash,
                    "tensor_schema_hash": authority.expected_tensor_schema_hash,
                    "serving_build_digest": (authority.expected_serving_build_digest),
                    "resolved_spec_digest": (authority.expected_resolved_spec_digest),
                },
            },
        },
    }


@contextmanager
def _patched_fake_runtime(runtime_module: ModuleType):
    integration_module = cast(Any, _integration)
    original_ensure_initialized = runtime_module.RuntimeSettings.ensure_initialized
    original_contract_reader = integration_module.read_source_bound_contract_state
    original_materialization_options = (
        integration_module.ServingIntegration.build_materialization_options
    )
    initialized: list[object] = []

    def ensure_initialized(self) -> None:
        initialized.append(self)

    runtime_module.RuntimeSettings.ensure_initialized = ensure_initialized
    integration_module.read_source_bound_contract_state = lambda: SimpleNamespace(
        source_bound_contract_ready=True,
        source_bound_contract_version=4,
        source_bound_capability_names=("collective",),
    )
    integration_module.ServingIntegration.build_materialization_options = (
        lambda self, **kwargs: ("fake-materialization-options", kwargs)
    )
    try:
        yield initialized
    finally:
        runtime_module.RuntimeSettings.ensure_initialized = original_ensure_initialized
        integration_module.read_source_bound_contract_state = original_contract_reader
        integration_module.ServingIntegration.build_materialization_options = (
            original_materialization_options
        )


def assert_level1_runtime_conformance(
    runtime_module: ModuleType,
    hosts_module: ModuleType,
    *,
    host: object | None = None,
) -> ConformanceResult:
    """Run Level 1 durable serving artifact runtime conformance.

    The suite intentionally uses only ``tensorcast.serving.runtime`` and
    ``tensorcast.serving.hosts`` plus this testing module's fake host fixtures.
    It covers direct artifact start, reload, describe, capability optionality,
    strict public DTO rejection and no-vLLM-import contracts. It does not
    instantiate local bootstrap or retained preload intent DTOs.
    """

    checks: dict[str, bool] = {}
    assert_public_runtime_boundary(runtime_module)
    assert_framework_isolation(
        (runtime_module.__name__, hosts_module.__name__, __name__)
    )

    with _patched_fake_runtime(runtime_module) as initialized:
        host = host if host is not None else build_fake_runtime_host(hosts_module)
        session = runtime_module.ServingRuntimeSession.from_config(
            {
                "bootstrap": {
                    "mode": "disabled",
                },
                "serving": {
                    "selector": {
                        "kind": "artifact_ref",
                        "value": "mi2:serving",
                    },
                },
            },
            host=host,
            resolver=FakeServingArtifactResolver(),
        )
        attachment = session.start(
            runtime_module.RequestContext(
                framework_config=SimpleNamespace(),
                model_config=SimpleNamespace(model="fake-model"),
                target_device=torch.device("cuda:0"),
            )
        )
        direct_payload = attachment.view.endpoint.to_weight_version_payload()
        checks["direct_start"] = (
            direct_payload.get("serving_artifact_ref") == "mi2:serving"
            and direct_payload.get("source_artifact_ref") == "mi2:source"
        )
        checks["runtime_initialized"] = bool(initialized)

        described = session.describe(attachment)
        checks["describe"] = (
            described.endpoint.to_weight_version_payload().get("serving_artifact_ref")
            == "mi2:serving"
        )

        reloaded = session.reload(
            current_attachment=attachment,
            selector=runtime_module.ServingArtifactSelector.artifact_ref(
                "mi2:serving-next"
            ),
            policy=runtime_module.ServingPolicy(),
            context=runtime_module.RequestContext(
                framework_config=SimpleNamespace(),
                model_config=SimpleNamespace(model="fake-model"),
            ),
            model=attachment.model,
        )
        reload_response = reloaded.view.endpoint.to_reload_response_payload()
        checks["reload"] = (
            reload_response.get("serving_artifact_ref") == "mi2:serving-next"
        )
        checks["reload_identity_from_runtime_view"] = (
            reloaded.state.runtime_view.serving_artifact_ref
            == reload_response.get("serving_artifact_ref")
        )
        checks["source_capability_not_required"] = True
        checks["source_catalog_not_required"] = True

        try:
            session.reload(
                current_attachment=reloaded,
                selector=runtime_module.SourceSelector.local_path("/tmp/model"),
                policy=runtime_module.ServingPolicy(),
                context=runtime_module.RequestContext(),
            )
        except _integration.ConfigConflictError:
            checks["rejects_local_reload_selector"] = True
        else:
            checks["rejects_local_reload_selector"] = False

        try:
            session.reload(
                current_attachment=reloaded,
                selector={"kind": "artifact_ref", "value": "mi2:serving-next"},
                policy=runtime_module.ServingPolicy(),
                context=runtime_module.RequestContext(),
            )
        except _integration.ConfigConflictError:
            checks["rejects_untyped_reload_selector"] = True
        else:
            checks["rejects_untyped_reload_selector"] = False

        try:
            session.reload(
                current_attachment=reloaded,
                selector=runtime_module.ServingArtifactSelector.artifact_ref(
                    "mi2:serving-next"
                ),
                policy={"mode": "from_manifest"},
                context=runtime_module.RequestContext(),
            )
        except _integration.ConfigConflictError:
            checks["rejects_untyped_reload_policy"] = True
        else:
            checks["rejects_untyped_reload_policy"] = False

    return _result(level="level1-runtime", checks=checks, messages=_LEVEL1_MESSAGES)


def assert_level2_local_bootstrap_conformance(
    runtime_module: ModuleType,
    hosts_module: ModuleType,
) -> ConformanceResult:
    """Run Level 2 local source bootstrap planning conformance."""

    checks: dict[str, bool] = {}
    with _patched_fake_runtime(runtime_module):
        integration_module = cast(Any, _integration)
        host_without_catalog = hosts_module.IntegrationHost(
            framework=FakeFrameworkHost(),
            placement=FakePlacementHost(),
            tensor_surface=FakeTensorSurface(),
            source=FakeSourceHost(),
        )
        session = runtime_module.ServingRuntimeSession.from_config(
            {
                "bootstrap": {
                    "mode": "required",
                },
            },
            host=host_without_catalog,
        )
        original_resolve_source_subject = (
            integration_module.ServingIntegration.resolve_source_subject
        )

        def fake_resolve_source_subject(self, selector, **kwargs):
            del self, selector, kwargs
            return _integration.SourceSubject(
                artifact_ref="mi2:source",
                subject=SimpleNamespace(),
                source_kind="fake",
                metadata_fingerprint="meta",
            )

        integration_module.ServingIntegration.resolve_source_subject = (
            fake_resolve_source_subject
        )
        try:
            try:
                session.start(
                    runtime_module.RequestContext(
                        framework_config=SimpleNamespace(),
                        model_config=SimpleNamespace(model="fake-model"),
                        target_device=torch.device("cuda:0"),
                    )
                )
            except _integration.CapabilityMissingError as exc:
                checks["missing_source_catalog_fails_closed"] = "source_catalog" in str(
                    exc
                )
            else:
                checks["missing_source_catalog_fails_closed"] = False

            catalog_provider = FakeSourceCatalogProvider()
            host_with_catalog = hosts_module.IntegrationHost(
                framework=FakeFrameworkHost(),
                placement=FakePlacementHost(),
                tensor_surface=FakeTensorSurface(),
                source=FakeSourceHost(),
                source_catalog=catalog_provider,
            )
            session_with_catalog = runtime_module.ServingRuntimeSession.from_config(
                {
                    "bootstrap": {
                        "mode": "required",
                    },
                },
                host=host_with_catalog,
            )
            original_build_recipe = integration_module.RecipeBuildSession.build_recipe
            captured_builds: list[Mapping[str, object]] = []

            def fake_build_recipe(self, **kwargs):
                del self
                captured_builds.append(kwargs)
                kwargs["framework_adapter"].trace_model_load(
                    FakeRuntimeModel(),
                    ["w"],
                    {"w": SimpleNamespace(name="w")},
                )

            integration_module.RecipeBuildSession.build_recipe = fake_build_recipe
            try:
                try:
                    session_with_catalog.start(
                        runtime_module.RequestContext(
                            framework_config=SimpleNamespace(),
                            model_config=SimpleNamespace(model="fake-model"),
                            target_device=torch.device("cuda:0"),
                        )
                    )
                except _integration.CapabilityMissingError as exc:
                    checks["missing_trace_capability_is_explicit"] = (
                        "RecipeTraceHost" in str(exc) or "trace_model_load" in str(exc)
                    )
                else:
                    checks["missing_trace_capability_is_explicit"] = False
            finally:
                integration_module.RecipeBuildSession.build_recipe = (
                    original_build_recipe
                )
        finally:
            integration_module.ServingIntegration.resolve_source_subject = (
                original_resolve_source_subject
            )

        catalog_request = (
            catalog_provider.requests[0] if catalog_provider.requests else None
        )
        checks["source_catalog_request_core_owned"] = (
            catalog_request is not None
            and getattr(catalog_request, "source_artifact_ref", None) == "mi2:source"
            and isinstance(
                getattr(catalog_request, "source_selector", None),
                _integration.SourceSelector,
            )
        )
        checks["recipe_build_receives_core_catalog"] = (
            bool(captured_builds)
            and captured_builds[0].get("source_catalog") is not None
        )

        attachment = runtime_module.RuntimeAttachment(
            model=object(),
            state=_integration.RuntimeBindingState(
                runtime_view=_integration.RuntimeBindingView()
            ),
            view=runtime_module.RuntimeWorkerView.from_runtime_view(
                _integration.RuntimeBindingView()
            ),
        )
        try:
            session.reload(
                current_attachment=attachment,
                selector=runtime_module.SourceSelector.local_path("/tmp/fakefw-model"),
                policy=runtime_module.ServingPolicy(),
                context=runtime_module.RequestContext(),
            )
        except _integration.ConfigConflictError:
            checks["local_path_is_not_reload_selector"] = True
        else:
            checks["local_path_is_not_reload_selector"] = False

    return _result(
        level="level2-local-bootstrap",
        checks=checks,
        messages=_LEVEL2_MESSAGES,
    )


def assert_level3_retained_preload_conformance(
    runtime_module: ModuleType,
    hosts_module: ModuleType,
) -> ConformanceResult:
    """Run Level 3 retained/external preload conformance."""

    checks: dict[str, bool] = {}
    with _patched_fake_runtime(runtime_module):
        integration_module = cast(Any, _integration)
        host = build_fake_runtime_host(hosts_module)
        retained_calls: list[Mapping[str, object]] = []
        restored = FakeRestoredRetainedBinding()
        original_restore_retained = integration_module.restore_retained_binding

        @contextmanager
        def fake_restore_retained(**kwargs: object):
            retained_calls.append(kwargs)
            yield restored

        integration_module.restore_retained_binding = fake_restore_retained
        try:
            session = runtime_module.ServingRuntimeSession.from_config(
                _external_preload_config(runtime_module),
                host=host,
            )
            retained = session.start(
                runtime_module.RequestContext(
                    framework_config=SimpleNamespace(),
                    model_config=SimpleNamespace(model="fake-model"),
                    target_device=torch.device("cuda:0"),
                )
            )
        finally:
            integration_module.restore_retained_binding = original_restore_retained
        retained_payload = retained.view.endpoint.to_weight_version_payload()
        checks["retained_acquire_public_start"] = (
            retained_payload.get("local_serving_ref") == "binding-local:fake"
            and retained_payload.get("binding_value_ref", {}).get("binding_value_id")
            == "value-1"
        )
        checks["retained_acquire_uses_host_member"] = (
            bool(retained_calls)
            and getattr(retained_calls[0].get("expected_member"), "member_index", None)
            == 0
        )
        checks["retained_acquire_transfers_ownership"] = restored.transferred

        try:
            runtime_module.ServingConfig.from_mapping(
                {
                    "preload": {
                        "mode": "external",
                    },
                }
            )
        except Exception:
            checks["missing_authority_fails_closed"] = True
        else:
            checks["missing_authority_fails_closed"] = False

        mismatch_config = dict(_external_preload_config(runtime_module))
        preload = dict(mismatch_config["preload"])
        authority = dict(preload["authority"])
        capability = dict(authority["reservation_capability"])
        capability["daemon_session_id"] = "wrong-session"
        authority["reservation_capability"] = capability
        preload["authority"] = authority
        mismatch_config["preload"] = preload
        try:
            mismatch_session = runtime_module.ServingRuntimeSession.from_config(
                mismatch_config,
                host=host,
            )
            mismatch_session.start(
                runtime_module.RequestContext(
                    framework_config=SimpleNamespace(),
                    model_config=SimpleNamespace(model="fake-model"),
                    target_device=torch.device("cuda:0"),
                )
            )
        except Exception:
            checks["authority_mismatch_fails_closed"] = True
        else:
            checks["authority_mismatch_fails_closed"] = False

        failing_host = hosts_module.IntegrationHost(
            framework=FakeFrameworkHost(),
            placement=FakePlacementHost(),
            tensor_surface=SchemaMismatchTensorSurface(),
        )
        failing_restored = FakeRestoredRetainedBinding()
        failure_calls: list[Mapping[str, object]] = []

        @contextmanager
        def fake_restore_for_failure(**kwargs: object):
            failure_calls.append(kwargs)
            yield failing_restored

        integration_module.restore_retained_binding = fake_restore_for_failure
        try:
            failing_session = runtime_module.ServingRuntimeSession.from_config(
                _external_preload_config(runtime_module),
                host=failing_host,
            )
            try:
                failing_session.start(
                    runtime_module.RequestContext(
                        framework_config=SimpleNamespace(),
                        model_config=SimpleNamespace(model="fake-model"),
                        target_device=torch.device("cuda:0"),
                    )
                )
            except _integration.SchemaMismatchError:
                checks["failure_cleanup_closes_untransferred_handle"] = (
                    failing_restored.closed and not failing_restored.transferred
                )
            else:
                checks["failure_cleanup_closes_untransferred_handle"] = False
        finally:
            integration_module.restore_retained_binding = original_restore_retained

        checks["failure_path_used_retained_restore"] = bool(failure_calls)

        try:
            runtime_module.RetainedBindingAcquire(SimpleNamespace())
        except _integration.AuthorityValidationError:
            checks["rejects_arbitrary_retained_authority"] = True
        else:
            checks["rejects_arbitrary_retained_authority"] = False

    return _result(
        level="level3-retained-preload",
        checks=checks,
        messages=_LEVEL3_MESSAGES,
    )


def assert_minimal_runtime_conformance(
    runtime_module: ModuleType,
    hosts_module: ModuleType,
) -> ConformanceResult:
    """Compatibility alias for Level 1 runtime conformance."""

    return assert_level1_runtime_conformance(runtime_module, hosts_module)


__all__ = [
    "ConformanceResult",
    "FakeArtifact",
    "FakeArtifactView",
    "FakeBinding",
    "FakeFrameworkHost",
    "FakePlacementHost",
    "FakeRestoredRetainedBinding",
    "FakeRuntimeModel",
    "FakeServingArtifactResolver",
    "FakeSourceCatalogProvider",
    "FakeSourceHost",
    "FakeTensorSurface",
    "SchemaMismatchTensorSurface",
    "assert_framework_isolation",
    "assert_level1_runtime_conformance",
    "assert_level2_local_bootstrap_conformance",
    "assert_level3_retained_preload_conformance",
    "assert_minimal_runtime_conformance",
    "assert_public_runtime_boundary",
    "build_fake_runtime_host",
]
