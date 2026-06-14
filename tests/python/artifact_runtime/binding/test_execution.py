#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

import tensorcast as tc
from tensorcast.api._config import CollectivePolicyMode
from tensorcast.artifact_runtime.binding.execution import (
    bind_runtime_artifact,
    build_materialization_execution_context,
    swap_runtime_artifact,
)
from tensorcast.types import CollectivePolicy


def test_bind_and_swap_runtime_artifact_delegate_to_artifact_handles() -> None:
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
        tensors = {"a": object(), "b": object()}

        def swap(self, artifact, **kwargs):
            calls.append(("swap", (artifact, kwargs)))
            return "swapped"

    resolved = SimpleNamespace(artifact=_Artifact())

    assert (
        bind_runtime_artifact(
            resolved_artifact=resolved,
            tensor_names=("a", "b"),
            device="cuda:0",
            runtime_artifact_policy="policy",
            options="options",
        )
        == "binding"
    )
    assert (
        swap_runtime_artifact(
            binding=_Binding(),
            resolved_artifact=resolved,
            runtime_artifact_policy="policy",
            options="options",
        )
        == "swapped"
    )

    swapped_artifact = calls[3][1][0]
    assert isinstance(swapped_artifact, _Subset)
    assert calls == [
        ("subset", ("a", "b")),
        (
            "bind",
            {
                "device": "cuda:0",
                "runtime_artifact_policy": "policy",
                "options": "options",
            },
        ),
        ("subset", ("a", "b")),
        (
            "swap",
            (
                swapped_artifact,
                {
                    "runtime_artifact_policy": "policy",
                    "options": "options",
                },
            ),
        ),
    ]


def test_swap_runtime_artifact_prefers_binding_target_tensor_names() -> None:
    calls: list[tuple[str, object]] = []

    class _Subset:
        pass

    class _Artifact:
        def subset(self, names):
            calls.append(("subset", tuple(names)))
            return _Subset()

    class _Binding:
        tensors = {"a": object(), "__tensorcast_meta__.manifest_json": object()}

        def swap(self, artifact, **kwargs):
            calls.append(("swap", artifact))
            return "swapped"

    resolved = SimpleNamespace(artifact=_Artifact(), tensor_names=("a",))

    assert (
        swap_runtime_artifact(
            binding=_Binding(),
            resolved_artifact=resolved,
            tensor_names=("a",),
            runtime_artifact_policy=None,
            options=None,
        )
        == "swapped"
    )

    assert calls[0] == (
        "subset",
        ("a", "__tensorcast_meta__.manifest_json"),
    )
    assert isinstance(calls[1][1], _Subset)


def test_swap_runtime_artifact_prefers_binding_layout_tensor_order() -> None:
    calls: list[tuple[str, object]] = []

    class _Subset:
        pass

    class _Artifact:
        def subset(self, names):
            calls.append(("subset", tuple(names)))
            return _Subset()

    class _Binding:
        layout = SimpleNamespace(
            target_layout=SimpleNamespace(
                offsets=(
                    SimpleNamespace(name="a"),
                    SimpleNamespace(name=tc.SERVING_MANIFEST_TENSOR_NAME),
                    SimpleNamespace(name="b"),
                )
            )
        )
        tensors = {
            "b": object(),
            "a": object(),
            tc.SERVING_MANIFEST_TENSOR_NAME: object(),
        }

        def swap(self, artifact, **kwargs):
            calls.append(("swap", artifact))
            return "swapped"

    resolved = SimpleNamespace(artifact=_Artifact(), tensor_names=("a", "b"))

    assert (
        swap_runtime_artifact(
            binding=_Binding(),
            resolved_artifact=resolved,
            tensor_names=("b",),
            runtime_artifact_policy=None,
            options=None,
        )
        == "swapped"
    )

    assert calls[0] == (
        "subset",
        ("a", tc.SERVING_MANIFEST_TENSOR_NAME, "b"),
    )
    assert isinstance(calls[1][1], _Subset)


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
    assert (
        options.execution_topology.collective_policy
        is CollectivePolicyMode.COLLECTIVE_FIRST
    )
    assert profile["collective_requested"] is True
    assert profile["source_locality"] == "shared_source"


def test_materialization_execution_context_respects_disabled_collective_policy() -> (
    None
):
    options, profile = build_materialization_execution_context(
        artifact_ref="mi2:test:serving",
        operation_scope="startup.bind",
        configured_policy=CollectivePolicy.DISABLE_COLLECTIVE,
        tp_rank=1,
        tp_world_size=2,
        same_node_tp=True,
        tp_ranks=(0, 1),
        collective_world_size=2,
        collective_rank=1,
        source_bound_contract_profile_fields={},
        build_group_id=lambda **_kwargs: "group-1",
    )

    assert options.execution_topology.collective_group is None
    assert (
        options.execution_topology.collective_policy
        is CollectivePolicyMode.DISABLE_COLLECTIVE
    )
    assert profile["collective_requested"] is False
    assert profile["collective_reason"] == "collective_disabled"
    assert profile["source_locality"] == "shared_source"


def test_materialization_execution_context_disables_collective_when_unavailable() -> (
    None
):
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
