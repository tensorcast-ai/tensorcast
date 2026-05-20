#  Copyright (c) 2026, TensorCast Team.

from types import SimpleNamespace

from tensorcast.serving.integration import (
    AdmissionRequest,
    FrameworkIdentity,
    PlacementAdmissionFacts,
    PlacementIdentityFacts,
    RuntimeProfile,
)
from tensorcast.serving.readiness import (
    ReadinessInventoryAdmissionPolicy,
    is_binding_finalize_publication_allowlisted,
    is_pure_transform_publication_allowlisted,
    is_runtime_bind_swap_allowlisted,
    serving_support_level_display_name,
)
from tensorcast.types import FinalizeClass, ServingSupportLevel


def _row(**overrides):
    values = {
        "family": "fake",
        "process_after_load_class": FinalizeClass.RUNTIME_ONLY,
        "post_bind_finalize_class": FinalizeClass.RUNTIME_ONLY,
        "support_level": ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        "pure_transform_candidate": True,
        "serving_only_runtime_allowed": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_readiness_helpers_accept_framework_inventory_rows() -> None:
    row = _row()

    assert serving_support_level_display_name(
        row.support_level) == "serving_bind_swap_ready"
    assert is_pure_transform_publication_allowlisted(row) is True
    assert is_runtime_bind_swap_allowlisted(row) is True
    assert is_binding_finalize_publication_allowlisted(row) is False


def test_binding_finalize_readiness_uses_core_rules() -> None:
    row = _row(
        process_after_load_class=FinalizeClass.REPRESENTATION_CHANGING,
        pure_transform_candidate=False,
    )

    assert is_pure_transform_publication_allowlisted(row) is False
    assert is_binding_finalize_publication_allowlisted(row) is True


def test_readiness_inventory_admission_policy_is_framework_neutral() -> None:
    policy = ReadinessInventoryAdmissionPolicy(lambda model_config: _row(
        family=model_config.family))
    decision = policy.admit(
        AdmissionRequest(
            intent=object(),
            framework_identity=FrameworkIdentity(
                framework_name="fakefw",
                framework_version="v1",
                adapter_version="adapter-v1",
                serving_abi_version="abi-v1",
            ),
            placement_identity=PlacementIdentityFacts(
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                pipeline_parallel_rank=0,
                pipeline_parallel_size=1,
                data_parallel_rank=0,
                data_parallel_size=1,
            ),
            placement_admission=PlacementAdmissionFacts(),
            model_config=SimpleNamespace(family="second-framework-family"),
            runtime_profile=RuntimeProfile(),
        ))

    assert decision.family == "second-framework-family"
    assert decision.startup_allowed is True
    assert decision.reload_allowed is True
    assert decision.local_bootstrap_allowed is True
