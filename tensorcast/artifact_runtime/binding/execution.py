#  Copyright (c) 2026, TensorCast Team.

"""Runtime artifact bind/swap facades."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import tensorcast as tc
from tensorcast.api._config import CollectivePolicyMode, SourceLocalityHint
from tensorcast.types import CollectivePolicy


def _collective_policy_name(configured_policy: Any) -> str:
    if isinstance(configured_policy, CollectivePolicy):
        return configured_policy.value
    if isinstance(configured_policy, CollectivePolicyMode):
        return configured_policy.value
    if configured_policy is None:
        return "auto"
    normalized = str(configured_policy).strip().lower()
    return normalized or "auto"


def _parse_explicit_collective_policy(policy_name: str) -> CollectivePolicy:
    if policy_name in {"required", "require_collective"}:
        return CollectivePolicy.REQUIRE_COLLECTIVE
    if policy_name == "collective_first":
        return CollectivePolicy.COLLECTIVE_FIRST
    if policy_name in {"disabled", "disable_collective"}:
        return CollectivePolicy.DISABLE_COLLECTIVE
    raise ValueError(
        "Unknown TensorCast materialization collective policy "
        f"{policy_name!r}; expected auto, required, disabled, "
        "collective_first, require_collective, or disable_collective."
    )


def _resolve_collective_policy(
    configured_policy: Any,
    *,
    tp_world_size: int,
    same_node_tp: bool,
    collective_context_unavailable: bool,
) -> tuple[CollectivePolicy, str, bool, str | None]:
    configured_name = _collective_policy_name(configured_policy)
    if configured_name != "auto":
        return (
            _parse_explicit_collective_policy(configured_name),
            configured_name,
            False,
            None,
        )
    if collective_context_unavailable:
        return (
            CollectivePolicy.DISABLE_COLLECTIVE,
            configured_name,
            True,
            "collective_context_unavailable",
        )
    if tp_world_size <= 1:
        return (
            CollectivePolicy.DISABLE_COLLECTIVE,
            configured_name,
            True,
            "single_rank",
        )
    if not same_node_tp:
        return (
            CollectivePolicy.DISABLE_COLLECTIVE,
            configured_name,
            True,
            "tp_not_same_node",
        )
    return (
        CollectivePolicy.COLLECTIVE_FIRST,
        configured_name,
        True,
        None,
    )


def build_materialization_execution_context(
    *,
    artifact_ref: str,
    operation_scope: str,
    configured_policy: Any,
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
    resolved_policy, configured_policy_name, auto_resolved, auto_reason = (
        _resolve_collective_policy(
            configured_policy,
            tp_world_size=int(tp_world_size),
            same_node_tp=bool(same_node_tp),
            collective_context_unavailable=bool(collective_context_unavailable),
        )
    )
    profile_fields: dict[str, object] = {
        "tp_rank": int(tp_rank),
        "tp_world_size": int(tp_world_size),
        **dict(source_bound_contract_profile_fields),
        "collective_policy": resolved_policy.value,
        "collective_policy_configured": configured_policy_name,
        "collective_policy_auto_resolved": auto_resolved,
        "collective_requested": False,
        "same_node_tp": bool(same_node_tp),
    }
    if auto_reason is not None:
        profile_fields["collective_auto_reason"] = auto_reason
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
        and resolved_policy is not CollectivePolicy.DISABLE_COLLECTIVE
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
                "collective_policy": resolved_policy.value,
                "collective_requested": True,
                "collective_group_id": collective.group_id,
                "collective_rank": collective.rank,
                "collective_world_size": collective.world_size,
            }
        )
    elif resolved_policy is CollectivePolicy.DISABLE_COLLECTIVE:
        profile_fields.setdefault("collective_reason", "collective_disabled")
    else:
        profile_fields["collective_policy_rejected"] = resolved_policy.value
        profile_fields["collective_policy"] = CollectivePolicy.DISABLE_COLLECTIVE.value
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


def bind_runtime_artifact(
    *,
    resolved_artifact: Any,
    tensor_names: Sequence[str],
    device: Any,
    runtime_artifact_policy: Any | None,
    options: Any | None,
) -> Any:
    return resolved_artifact.artifact.subset(list(tensor_names)).bind(
        device=device,
        runtime_artifact_policy=runtime_artifact_policy,
        options=options,
    )


def _binding_layout_tensor_names(binding: Any) -> tuple[str, ...]:
    layout = getattr(binding, "layout", None)
    target_layout = getattr(layout, "target_layout", None)
    offsets = getattr(target_layout, "offsets", None)
    if offsets is None:
        return ()
    return tuple(
        str(offset.name) for offset in offsets if str(getattr(offset, "name", ""))
    )


def _binding_tensor_names(binding: Any) -> tuple[str, ...]:
    binding_tensors = getattr(binding, "tensors", None)
    if not isinstance(binding_tensors, Mapping):
        return ()
    return tuple(str(name) for name in binding_tensors)


def swap_runtime_artifact(
    *,
    binding: Any,
    resolved_artifact: Any,
    tensor_names: Sequence[str] | None = None,
    runtime_artifact_policy: Any | None,
    options: Any | None,
) -> Any:
    binding_layout_tensor_names = _binding_layout_tensor_names(binding)
    binding_tensor_names = _binding_tensor_names(binding)
    if binding_layout_tensor_names:
        tensor_names = binding_layout_tensor_names
    elif binding_tensor_names:
        tensor_names = binding_tensor_names
    elif tensor_names is None:
        tensor_names = tuple(getattr(resolved_artifact, "tensor_names", ()) or ())
    else:
        tensor_names = tuple(str(name) for name in tensor_names)
    artifact = (
        resolved_artifact.artifact.subset(list(tensor_names))
        if tensor_names
        else resolved_artifact.artifact
    )
    return binding.swap(
        artifact,
        runtime_artifact_policy=runtime_artifact_policy,
        options=options,
    )


__all__ = [
    "bind_runtime_artifact",
    "build_materialization_execution_context",
    "swap_runtime_artifact",
]
