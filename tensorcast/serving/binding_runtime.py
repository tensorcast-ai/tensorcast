#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact bind/swap runtime facades."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import tensorcast as tc
from tensorcast.api._config import CollectivePolicyMode, SourceLocalityHint
from tensorcast.types import CollectivePolicy


def build_materialization_execution_context(
    *,
    artifact_ref: str,
    operation_scope: str,
    configured_policy: CollectivePolicy,
    tp_rank: int,
    tp_world_size: int,
    same_node_tp: bool,
    tp_ranks: Sequence[int],
    collective_world_size: int,
    collective_rank: int,
    source_bound_contract_profile_fields: Mapping[str, object],
    build_group_id: Callable[..., str],
    contract_identity: str | None = None,
    collective_context_unavailable: bool = False,
) -> tuple[Any, dict[str, object]]:
    profile_fields: dict[str, object] = {
        "tp_rank": int(tp_rank),
        "tp_world_size": int(tp_world_size),
        **dict(source_bound_contract_profile_fields),
        "collective_policy": CollectivePolicy.DISABLE_COLLECTIVE.value,
        "collective_requested": False,
        "same_node_tp": bool(same_node_tp),
    }
    if tp_ranks:
        profile_fields["tp_group_ranks"] = [int(rank) for rank in tp_ranks]
    if collective_context_unavailable:
        profile_fields["collective_reason"] = "collective_context_unavailable"

    source_locality = SourceLocalityHint.AUTO
    if tp_world_size <= 1:
        profile_fields.setdefault("collective_reason", "single_rank")
    elif not same_node_tp:
        profile_fields.setdefault("collective_reason", "tp_not_same_node")
    else:
        source_locality = SourceLocalityHint.SHARED_SOURCE

    collective = None
    if (
        tp_world_size > 1
        and same_node_tp
        and configured_policy is not CollectivePolicy.DISABLE_COLLECTIVE
    ):
        collective = tc.CollectiveLoadGroup(
            group_id=build_group_id(
                artifact_ref=artifact_ref,
                operation_scope=operation_scope,
                tp_ranks=tuple(int(rank) for rank in tp_ranks),
                contract_identity=contract_identity,
            ),
            world_size=int(collective_world_size),
            rank=int(collective_rank),
        )
        profile_fields.update(
            {
                "collective_policy": configured_policy.value,
                "collective_requested": True,
                "collective_group_id": collective.group_id,
                "collective_rank": collective.rank,
                "collective_world_size": collective.world_size,
            }
        )
    elif configured_policy is CollectivePolicy.DISABLE_COLLECTIVE:
        profile_fields.setdefault("collective_reason", "collective_disabled")
    profile_fields["source_locality"] = source_locality

    options = tc.GetArtifactOptions(
        execution_topology=tc.ExecutionTopologyContext(
            collective_group=collective,
            collective_policy=CollectivePolicyMode.parse(
                profile_fields["collective_policy"]
            ),
            source_locality=source_locality,
        )
    )
    return options, profile_fields


def bind_serving_artifact(
    *,
    resolved_artifact: Any,
    tensor_names: Sequence[str],
    device: Any,
    serving_runtime_policy: Any | None,
    options: Any | None,
) -> Any:
    return resolved_artifact.artifact.subset(list(tensor_names)).bind(
        device=device,
        serving_runtime_policy=serving_runtime_policy,
        options=options,
    )


def swap_serving_artifact(
    *,
    binding: Any,
    resolved_artifact: Any,
    serving_runtime_policy: Any | None,
    options: Any | None,
) -> Any:
    return binding.swap(
        resolved_artifact.artifact,
        serving_runtime_policy=serving_runtime_policy,
        options=options,
    )


__all__ = [
    "bind_serving_artifact",
    "build_materialization_execution_context",
    "swap_serving_artifact",
]
