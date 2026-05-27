#  Copyright (c) 2026, TensorCast Team.
"""Minimal Level 1 TensorCast artifact-runtime framework integration."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

import torch

import tensorcast as tc
import tensorcast.artifact_runtime.host as tc_runtime_host
import tensorcast.artifact_runtime.testing as tc_testing


class ReferenceRuntimeModel:
    """Tiny model carrier used by the Level 1 conformance resolver."""

    def __init__(self) -> None:
        self.tensors = {
            "w": torch.empty((1,), dtype=torch.float16, device="meta"),
        }


class ReferenceFrameworkHost:
    """Framework-owned model construction and semantic facts."""

    def identity(self, model_config: object) -> tc_runtime_host.FrameworkIdentity:
        del model_config
        return tc_runtime_host.FrameworkIdentity(
            framework_name="referencefw",
            framework_version="0",
            adapter_version="level1-example",
            serving_abi_version="1",
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
    ) -> ReferenceRuntimeModel:
        del framework_config, model_config
        return ReferenceRuntimeModel()

    def build_runtime_model(
        self,
        framework_config: object | None,
        model_config: object | None,
        target_device: object | None,
    ) -> ReferenceRuntimeModel:
        del framework_config, model_config, target_device
        return ReferenceRuntimeModel()

    def assert_model_ready_for_runtime_binding(
        self,
        model: ReferenceRuntimeModel,
        *,
        context: object,
    ) -> None:
        del context
        if "w" not in model.tensors:
            raise AssertionError("reference model missing runtime tensor 'w'")

    def semantic_probes(
        self,
        model: ReferenceRuntimeModel,
        model_config: object | None,
    ) -> dict[str, object]:
        del model, model_config
        return {}


class ReferencePlacementHost:
    """Single-process placement facts for a Level 1 consumer."""

    def identity_facts(
        self,
        framework_config: object | None,
    ) -> tc_runtime_host.PlacementIdentityFacts:
        del framework_config
        return tc_runtime_host.PlacementIdentityFacts(
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
    ) -> tc_runtime_host.PlacementAdmissionFacts:
        del framework_config
        return tc_runtime_host.PlacementAdmissionFacts()

    def member_facts(
        self,
        framework_config: object | None,
    ) -> tc_runtime_host.PlacementMemberFacts:
        del framework_config
        return tc_runtime_host.PlacementMemberFacts(
            runtime_rank=0,
            runtime_world_size=1,
            member_id="member-0",
            member_index=0,
            member_count=1,
            group_id_hint="referencefw-single-process",
        )

    def execution_facts(
        self,
        framework_config: object | None,
    ) -> tc_runtime_host.MaterializationExecutionFacts:
        del framework_config
        return tc_runtime_host.MaterializationExecutionFacts(
            collective_rank=0,
            collective_world_size=1,
            tensor_parallel_ranks=(0,),
        )


class ReferenceTensorSurface:
    """Tensor attach/schema surface for the reference model carrier."""

    def runtime_only_tensor_names(
        self,
        model: ReferenceRuntimeModel,
    ) -> tuple[str, ...]:
        del model
        return ()

    def align_runtime_tensor_names(
        self,
        model: ReferenceRuntimeModel,
        expected_names: Iterable[str],
    ) -> int:
        if set(expected_names) != set(model.tensors):
            raise AssertionError("reference runtime tensor names do not match")
        return 0

    def collect_runtime_tensors(
        self,
        model: ReferenceRuntimeModel,
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
        model: ReferenceRuntimeModel,
        tensors: Mapping[str, object],
        *,
        replace_meta_params: bool,
    ) -> ReferenceRuntimeModel:
        del replace_meta_params
        model.tensors.update(tensors)
        return model

    def allocate_runtime_only_tensors(
        self,
        model: ReferenceRuntimeModel,
        target_device: object,
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
            raise AssertionError("reference tensor invariants changed")


def build_reference_host() -> tc.RuntimeHostCapabilities:
    """Build the minimal runtime host object a framework passes to TensorCast."""

    return tc.RuntimeHostCapabilities(
        framework=ReferenceFrameworkHost(),
        placement=ReferencePlacementHost(),
        tensor_surface=ReferenceTensorSurface(),
    )


def run_level1_conformance() -> tc_testing.ConformanceResult:
    """Run the TensorCast Level 1 conformance kit against this host."""

    return tc_testing.assert_level1_artifact_runtime_conformance(
        tc,
        host=build_reference_host(),
    )


def main() -> None:
    result = run_level1_conformance()
    print(json.dumps({"level": result.level, "checks": result.checks}, sort_keys=True))


if __name__ == "__main__":
    main()
