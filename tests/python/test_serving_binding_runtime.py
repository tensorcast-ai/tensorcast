#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

import tensorcast as tc
from tensorcast.serving.binding_runtime import (
    bind_serving_artifact,
    build_materialization_execution_context,
    swap_serving_artifact,
)
from tensorcast.types import CollectivePolicy


def test_bind_and_swap_serving_artifact_delegate_to_artifact_handles() -> None:
    calls: list[tuple[str, object]] = []

    class _Subset:

        def bind(self, **kwargs):
            calls.append(("bind", kwargs))
            return "binding"

    class _Artifact:

        def subset(self, names):
            calls.append(("subset", tuple(names)))
            return _Subset()

    class _Binding:

        def swap(self, artifact, **kwargs):
            calls.append(("swap", (artifact, kwargs)))
            return "swapped"

    resolved = SimpleNamespace(artifact=_Artifact())

    assert bind_serving_artifact(
        resolved_artifact=resolved,
        tensor_names=("a", "b"),
        device="cuda:0",
        serving_runtime_policy="policy",
        options="options",
    ) == "binding"
    assert swap_serving_artifact(
        binding=_Binding(),
        resolved_artifact=resolved,
        serving_runtime_policy="policy",
        options="options",
    ) == "swapped"

    assert calls == [
        ("subset", ("a", "b")),
        (
            "bind",
            {
                "device": "cuda:0",
                "serving_runtime_policy": "policy",
                "options": "options",
            },
        ),
        (
            "swap",
            (
                resolved.artifact,
                {
                    "serving_runtime_policy": "policy",
                    "options": "options",
                },
            ),
        ),
    ]


def test_materialization_execution_context_builds_collective_options() -> None:
    options, profile = build_materialization_execution_context(
        artifact_ref="mi2:test:serving",
        operation_scope="startup.bind",
        configured_policy=CollectivePolicy.COLLECTIVE_FIRST,
        tp_rank=1,
        tp_world_size=2,
        same_node_tp=True,
        tp_ranks=(0, 1),
        collective_world_size=2,
        collective_rank=1,
        source_bound_contract_profile_fields={"contract_ready": True},
        build_group_id=lambda **_kwargs: "group-1",
        contract_identity="repr-hash",
    )

    assert isinstance(options, tc.GetArtifactOptions)
    assert options.execution_topology.collective_group is not None
    assert options.execution_topology.collective_group.group_id == "group-1"
    assert profile["collective_requested"] is True
    assert profile["source_locality"] == "shared_source"


def test_materialization_execution_context_disables_collective_when_unavailable(
) -> None:
    options, profile = build_materialization_execution_context(
        artifact_ref="mi2:test:serving",
        operation_scope="startup.bind",
        configured_policy=CollectivePolicy.COLLECTIVE_FIRST,
        tp_rank=0,
        tp_world_size=2,
        same_node_tp=False,
        tp_ranks=(),
        collective_world_size=2,
        collective_rank=0,
        source_bound_contract_profile_fields={},
        build_group_id=lambda **_kwargs: "group-1",
        collective_context_unavailable=True,
    )

    assert options.execution_topology.collective_group is None
    assert profile["collective_requested"] is False
    assert profile["collective_reason"] == "collective_context_unavailable"
