#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import importlib.util

import pytest

import tensorcast.artifact_runtime.dto as serving_dto
from tensorcast.artifact_runtime.config import (
    MaterializationSettings,
    RetainedBindingAcquireSettings,
    RuntimeArtifactBindStartPlan,
    RuntimeRetainedRealizationStartPlan,
    RuntimeSourceBootstrapStartPlan,
    RuntimeStartPlanError,
    TensorCastRuntimeConfig,
    plan_runtime_start,
)
from tensorcast.artifact_runtime.locator import (
    ArtifactLocator,
    ranked_version_key_for_member,
)
from tensorcast.artifact_runtime.policy import (
    RuntimePolicy,
    merge_runtime_reload_extra_config,
)
from tensorcast.retained_realization import parse_retained_realization_authority
from tensorcast.types import RuntimeBindingMemberRef, RuntimeTopologyRef

FrameworkIntegrationContext = serving_dto.FrameworkIntegrationContext
PreparedRuntimeArtifact = serving_dto.PreparedRuntimeArtifact
RuntimeTensorView = serving_dto.RuntimeTensorView
RuntimeBindingValue = serving_dto.RuntimeBindingValue
RuntimePlacement = serving_dto.RuntimePlacement


def _find_spec_or_none(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None


def test_serving_public_package_is_removed() -> None:
    assert _find_spec_or_none("tensorcast.serving") is None
    assert _find_spec_or_none("tensorcast.serving.runtime") is None


def _retained_binding_acquire_config() -> dict:
    binding_value_ref = {
        "binding_id": "binding-1",
        "binding_layout_id": "layout-1",
        "binding_value_id": "value-1",
        "seal_generation": 1,
    }
    member = {
        "member_id": "member-0",
        "member_index": 0,
        "member_count": 1,
        "group_id": "group-1",
    }
    return {
        "retained_binding_acquire": {
            "mode": "external",
            "authority": {
                "group_id": "group-1",
                "member_ref": member,
                "daemon_id": "daemon-1",
                "daemon_session_id": "session-1",
                "device_uuid": "GPU-0",
                "binding_value_ref": binding_value_ref,
                "reservation_capability": {
                    "capability_id": "capability-1",
                    "binding_value_ref": binding_value_ref,
                    "daemon_id": "daemon-1",
                    "daemon_session_id": "session-1",
                    "device_uuid": "GPU-0",
                    "member": member,
                    "reservation_bytes": 4096,
                    "scope_digest": "scope-1",
                },
                "readiness": "runtime_published_ready",
                "serving_artifact_id": "mi2:test:serving",
                "trusted_reservation_bytes": 4096,
                "expected": {
                    "target_layout_hash": "layout-hash",
                    "tensor_schema_hash": "schema-hash",
                    "runtime_build_digest": "build-digest",
                    "resolved_spec_digest": "spec-digest",
                },
            },
        },
    }


def test_runtime_config_parses_nested_schema_defaults() -> None:
    config = TensorCastRuntimeConfig.from_mapping(
        {
            "runtime": {
                "mode": "CONNECT",
                "daemon": {
                    "show_logs": "true",
                },
                "global_store": {
                    "address": "127.0.0.1:50051",
                },
            },
            "runtime_artifact": {
                "artifact_locator": {
                    "kind": "version_key",
                    "value": " models/demo/serving/v1 ",
                },
            },
            "bootstrap": {
                "mode": "required",
                "verify_source_checksums": "false",
            },
            "materialization": {
                "collective": "required",
            },
        }
    )

    assert config.runtime.mode == "connect"
    assert config.runtime.daemon.show_logs is True
    assert config.runtime.global_store.resolved_mode("connect") == "connect"
    assert config.runtime_artifact.artifact_locator == ArtifactLocator(
        kind="version_key",
        value="models/demo/serving/v1",
    )
    assert config.to_mapping()["runtime_artifact"]["artifact_locator"] == {
        "kind": "version_key",
        "value": "models/demo/serving/v1",
        "schema_version": 1,
    }
    assert config.bootstrap.mode == "required"
    assert config.bootstrap.verify_source_checksums is False
    assert config.materialization.collective_policy_value() == "require_collective"


def test_materialization_auto_defaults_to_local_first_policy() -> None:
    config = TensorCastRuntimeConfig.from_mapping({})

    assert config.materialization.collective == "auto"
    assert config.materialization.collective_policy_value() == "disable_collective"


@pytest.mark.parametrize(
    ("collective", "expected_policy"),
    [
        ("collective_first", "collective_first"),
        ("required", "require_collective"),
        ("require_collective", "require_collective"),
        ("disabled", "disable_collective"),
        ("disable_collective", "disable_collective"),
    ],
)
def test_materialization_collective_policy_values(
    collective: str,
    expected_policy: str,
) -> None:
    settings = MaterializationSettings(collective=collective)

    assert settings.collective_policy_value() == expected_policy


def test_runtime_config_rejects_removed_serving_section() -> None:
    with pytest.raises(ValueError, match="serving.*removed"):
        TensorCastRuntimeConfig.from_mapping(
            {
                "serving": {
                    "artifact_locator": {
                        "kind": "artifact_ref",
                        "value": "mi2:test:serving",
                    },
                },
            }
        )


def test_runtime_config_rejects_runtime_artifact_selector_alias() -> None:
    with pytest.raises(ValueError, match="runtime_artifact.artifact_locator"):
        TensorCastRuntimeConfig.from_mapping(
            {
                "runtime_artifact": {
                    "selector": {
                        "kind": "artifact_ref",
                        "value": "mi2:test:serving",
                    },
                },
            }
        )


def test_runtime_config_emits_retained_binding_acquire_canonical_field() -> None:
    config = TensorCastRuntimeConfig.from_mapping(_retained_binding_acquire_config())

    assert isinstance(config.retained_binding_acquire, RetainedBindingAcquireSettings)
    mapping = config.to_mapping()
    assert "retained_binding_acquire" in mapping
    assert mapping["retained_binding_acquire"]["mode"] == "external"


def test_runtime_config_rejects_preload_key() -> None:
    with pytest.raises(ValueError, match="Unexpected TensorCast runtime config"):
        TensorCastRuntimeConfig.from_mapping(
            {
                "preload": {
                    "mode": "external",
                },
            }
        )


def test_ranked_version_key_locator_scopes_by_serving_member(monkeypatch) -> None:
    member = RuntimeBindingMemberRef(
        member_id="dp0:pp0:tp1",
        member_index=1,
        member_count=2,
        group_id="group-1",
    )
    locator = ArtifactLocator.ranked_version_key("models/demo/serving/v1/")

    assert (
        ranked_version_key_for_member(
            "models/demo/serving/v1/",
            member,
        )
        == "models/demo/serving/v1/members/dp0:pp0:tp1"
    )
    assert locator.resolve_version_key(member=member) == (
        "models/demo/serving/v1/members/dp0:pp0:tp1"
    )
    placement = RuntimePlacement(
        topology=RuntimeTopologyRef(
            schema_topology_digest="topology-digest",
            logical_topology_ref="fake://topology",
        ),
        member=member,
        framework_payload={},
        identity_payload={},
    )
    assert locator.resolve_version_key(placement=placement) == (
        "models/demo/serving/v1/members/dp0:pp0:tp1"
    )

    class _RuntimeContext:
        def resolve_key_mapping_cached(self, *, key):
            assert key == "models/demo/serving/v1/members/dp0:pp0:tp1"
            return "mi2:test:serving-rank-1", None

    monkeypatch.setattr(
        "tensorcast.api.store.get_runtime_context", lambda: _RuntimeContext()
    )

    assert locator.resolve_artifact_ref(member=member) == "mi2:test:serving-rank-1"


def test_artifact_locator_is_runtime_canonical_name() -> None:
    locator = ArtifactLocator.ranked_version_key("models/demo/serving/v1")

    assert isinstance(locator, ArtifactLocator)
    assert locator.kind == "ranked_version_key"
    assert locator.value == "models/demo/serving/v1"


def test_plan_runtime_start_classifies_three_canonical_variants() -> None:
    artifact_config = TensorCastRuntimeConfig.from_mapping(
        {
            "runtime_artifact": {
                "artifact_locator": {
                    "kind": "artifact_ref",
                    "value": "mi2:test:serving",
                },
            },
        }
    )
    artifact_plan = plan_runtime_start(
        config=artifact_config,
        source_selector=object(),
    )
    assert isinstance(artifact_plan, RuntimeArtifactBindStartPlan)
    assert artifact_plan.kind == "artifact_bind"
    assert artifact_plan.artifact_locator.value == "mi2:test:serving"

    source_selector = object()
    source_plan = plan_runtime_start(
        config=TensorCastRuntimeConfig.from_mapping({}),
        source_selector=source_selector,
    )
    assert isinstance(source_plan, RuntimeSourceBootstrapStartPlan)
    assert source_plan.kind == "source_bootstrap_to_binding"
    assert source_plan.source_selector is source_selector

    retained_plan = plan_runtime_start(
        config=TensorCastRuntimeConfig.from_mapping(_retained_binding_acquire_config()),
        source_selector=source_selector,
    )
    assert isinstance(retained_plan, RuntimeRetainedRealizationStartPlan)
    assert retained_plan.kind == "retained_binding_acquire"
    assert retained_plan.authority.binding_value_ref.binding_id == "binding-1"


def test_plan_runtime_start_reports_no_selected_candidate() -> None:
    with pytest.raises(RuntimeStartPlanError, match="rejected candidates"):
        plan_runtime_start(
            config=TensorCastRuntimeConfig.from_mapping({}),
            source_selector=None,
        )


def test_ranked_version_key_locator_requires_member() -> None:
    locator = ArtifactLocator.ranked_version_key("models/demo/serving/v1")

    with pytest.raises(ValueError, match="requires a member"):
        locator.resolve_version_key()


def test_runtime_config_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Unexpected TensorCast runtime config"):
        TensorCastRuntimeConfig.from_mapping({"unrelated": "unexpected"})


def test_runtime_policy_pinned_requires_identity_fields() -> None:
    with pytest.raises(ValueError, match="manifest_ref"):
        RuntimePolicy(mode="pinned")

    policy = RuntimePolicy(
        mode="pinned",
        manifest_ref="tensor:__tensorcast_meta__.manifest_json",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
    )

    assert policy.manifest_ref == "tensor:__tensorcast_meta__.manifest_json"


def test_merge_runtime_reload_extra_config_normalizes_wire_shape() -> None:
    extra = {
        "runtime": {
            "mode": "connect",
        },
        "runtime_artifact": {
            "policy": {
                "mode": "from_manifest",
            },
        },
    }

    merged = merge_runtime_reload_extra_config(
        extra,
        artifact_locator={
            "kind": "artifact_ref",
            "value": "mi2:test:serving",
        },
        policy={
            "mode": "pinned",
            "manifest_ref": "tensor:manifest",
            "representation_contract_hash": "repr-hash",
            "serving_build_digest": "build-digest",
        },
    )

    assert merged["runtime"] == {"mode": "connect"}
    assert merged["runtime_artifact"]["artifact_locator"] == {
        "kind": "artifact_ref",
        "value": "mi2:test:serving",
    }
    assert merged["runtime_artifact"]["policy"] == {
        "mode": "pinned",
        "manifest_ref": "tensor:manifest",
        "representation_contract_hash": "repr-hash",
        "serving_build_digest": "build-digest",
    }
    assert extra["runtime_artifact"]["policy"] == {"mode": "from_manifest"}


def test_runtime_config_parses_retained_binding_authority() -> None:
    config = TensorCastRuntimeConfig.from_mapping(
        {
            "retained_binding_acquire": {
                "mode": "external",
                "authority": {
                    "group_id": "group-1",
                    "member_ref": {
                        "member_id": "member-0",
                        "member_index": 0,
                        "member_count": 1,
                        "group_id": "group-1",
                    },
                    "daemon_id": "daemon-1",
                    "daemon_session_id": "session-1",
                    "device_uuid": "GPU-0",
                    "binding_value_ref": {
                        "binding_id": "binding-1",
                        "binding_layout_id": "layout-1",
                        "binding_value_id": "value-1",
                        "seal_generation": 1,
                    },
                    "reservation_capability": {
                        "capability_id": "capability-1",
                    },
                    "readiness": "runtime_published_ready",
                    "serving_artifact_id": "mi2:test:serving",
                    "trusted_reservation_bytes": 4096,
                    "expected": {
                        "target_layout_hash": "layout-hash",
                        "tensor_schema_hash": "schema-hash",
                        "runtime_build_digest": "build-digest",
                        "resolved_spec_digest": "spec-digest",
                    },
                },
            },
        }
    )

    assert config.retained_binding_acquire.mode == "external"
    assert config.retained_binding_acquire.authority is not None
    assert (
        config.retained_binding_acquire.authority.serving_artifact_id
        == "mi2:test:serving"
    )
    assert config.retained_binding_acquire.authority.trusted_reservation_bytes == 4096


def test_retained_binding_authority_parses_typed_refs() -> None:
    member = {
        "member_id": "member-0",
        "member_index": 0,
        "member_count": 1,
        "group_id": "group-1",
    }
    binding_value_ref = {
        "binding_id": "binding-1",
        "binding_layout_id": "layout-1",
        "binding_value_id": "value-1",
        "seal_generation": 1,
    }
    config = {
        "retained_binding_acquire": {
            "mode": "external",
            "authority": {
                "group_id": "group-1",
                "member_ref": member,
                "daemon_id": "daemon-1",
                "daemon_session_id": "session-1",
                "device_uuid": "GPU-0",
                "binding_value_ref": binding_value_ref,
                "reservation_capability": {
                    "capability_id": "capability-1",
                    "binding_value_ref": binding_value_ref,
                    "daemon_id": "daemon-1",
                    "daemon_session_id": "session-1",
                    "device_uuid": "GPU-0",
                    "member": member,
                    "reservation_bytes": 4096,
                    "scope_digest": "scope-1",
                },
                "local_serving_ref": "binding-local:binding-1:value-1",
                "readiness": "runtime_published_ready",
                "verification_state": "local_only",
                "serving_artifact_id": "mi2:test:serving",
                "trusted_reservation_bytes": 4096,
                "expected": {
                    "target_layout_hash": "layout-hash",
                    "tensor_schema_hash": "schema-hash",
                    "runtime_build_digest": "build-digest",
                    "resolved_spec_digest": "spec-digest",
                },
            },
        },
    }

    authority = parse_retained_realization_authority(config)

    assert authority.binding_value_ref.binding_id == "binding-1"
    assert authority.reservation_capability.reservation_bytes == 4096
    assert authority.member.member_id == "member-0"
    assert authority.expected.tensor_schema_hash == "schema-hash"


def test_prepared_serving_artifact_serializes_without_bootstrap_projection() -> None:
    binding_value_ref = {
        "binding_id": "binding-1",
        "binding_layout_id": "layout-1",
        "binding_value_id": "value-1",
        "seal_generation": 1,
    }
    prepared = PreparedRuntimeArtifact(
        source_artifact_ref="disk:/model",
        serving_artifact_ref=None,
        serving_manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        binding_value_ref=binding_value_ref,
        readiness="runtime_local_ready",
        family="dummy",
        tensor_schema_hash="schema-hash",
        binding_layout_id="layout-1",
        local_serving_ref="binding-local:binding-1:value-1",
    )

    payload = prepared.to_dict()

    assert payload["serving_manifest_ref"] == "tensor:manifest"
    assert payload["serving_artifact_ref"] is None
    assert payload["readiness"] == "runtime_local_ready"
    assert payload["binding_value_ref"] == binding_value_ref
    assert "bootstrap_summary" not in payload

    binding_value = prepared.to_binding_value()
    assert isinstance(binding_value, RuntimeBindingValue)
    assert binding_value.source_artifact_ref == "disk:/model"
    assert binding_value.readiness == "runtime_local_ready"
    assert binding_value.tensor_schema_hash == "schema-hash"
    assert binding_value.to_dict()["binding_value_ref"] == binding_value_ref


def test_prepared_serving_artifact_builds_reload_request() -> None:
    artifact = PreparedRuntimeArtifact(
        source_artifact_ref="disk:/model",
        serving_artifact_ref="mi2:test:serving",
        manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        family="demo",
        tensor_schema_hash="schema-hash",
    )

    assert artifact.serving_manifest_ref == "tensor:manifest"
    assert artifact.to_reload_request() == {
        "artifact_locator": {
            "kind": "artifact_ref",
            "value": "mi2:test:serving",
        },
        "policy": {
            "mode": "pinned",
            "manifest_ref": "tensor:manifest",
            "representation_contract_hash": "repr-hash",
            "serving_build_digest": "build-digest",
        },
    }


def test_local_ready_prepared_serving_artifact_cannot_build_reload_request() -> None:
    artifact = PreparedRuntimeArtifact(
        source_artifact_ref="disk:/model",
        serving_artifact_ref=None,
        manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        binding_value_ref={
            "binding_id": "binding-1",
            "binding_layout_id": "layout-1",
            "binding_value_id": "value-1",
            "seal_generation": 1,
        },
        readiness="runtime_local_ready",
        local_serving_ref="binding-local:binding-1:value-1",
        family="demo",
        tensor_schema_hash="schema-hash",
    )

    with pytest.raises(RuntimeError, match="cannot be used as a reload"):
        artifact.to_reload_request()

    payload = artifact.to_dict()
    assert payload["serving_artifact_ref"] is None
    assert payload["binding_value_ref"]["binding_value_id"] == "value-1"
    assert payload["readiness"] == "runtime_local_ready"
    assert payload["reload_request"] is None


def test_framework_context_and_runtime_tensor_view_are_identity_only() -> None:
    placement = RuntimePlacement(
        topology=RuntimeTopologyRef(
            schema_topology_digest="topology-digest",
            logical_topology_ref="vllm://parallelism?tp=2&pp=1&dp=1",
        ),
        member=RuntimeBindingMemberRef(
            member_id="dp0:pp0:tp1",
            member_index=1,
            member_count=2,
            group_id="group-1",
        ),
        framework_payload={
            "family": "vllm_parallelism",
            "version": "v1",
        },
        identity_payload={
            "tp_rank": 1,
            "tp_world_size": 2,
        },
    )
    context = FrameworkIntegrationContext(
        framework_name="vllm",
        framework_version="0.test",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        placement=placement,
        source_identity={"model": "demo"},
    )
    tensor_view = RuntimeTensorView(
        name="model.embed_tokens.weight",
        dtype="torch.float16",
        shape=(4, 8),
        stride=(8, 1),
        storage_offset=0,
        element_size=2,
    )

    assert context.stable_identity_payload()["framework_version"] == "0.test"
    assert (
        context.stable_identity_payload()["placement"]
        == placement.stable_identity_payload()
    )
    assert tensor_view.model_dump(mode="python") == {
        "name": "model.embed_tokens.weight",
        "dtype": "torch.float16",
        "shape": (4, 8),
        "stride": (8, 1),
        "storage_offset": 0,
        "element_size": 2,
    }
