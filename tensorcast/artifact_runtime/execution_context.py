#  Copyright (c) 2026, TensorCast Team.
"""Materialization execution context helpers for artifact runtime.

These helpers translate framework placement and source-bound contract facts into
the binding runtime execution options used by durable load, reload, and
local-ready realization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import tensorcast.artifact_runtime.binding.execution as tc_binding_runtime
from tensorcast.artifact_runtime.contract import (
    SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4,
    source_bound_contract_profile_fields,
)
from tensorcast.artifact_runtime.errors import ArtifactRuntimeIntegrationError
from tensorcast.artifact_runtime.host import (
    MaterializationExecutionFacts,
    MaterializationPolicy,
    RuntimeProfile,
)
from tensorcast.types import CollectivePolicy


@dataclass(frozen=True)
class HostMaterializationRequest:
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = ""
    require_materialization_options: bool = False


def collective_policy_value(policy: MaterializationPolicy) -> str:
    collective = str(policy.fields.get("collective", "disabled") or "disabled")
    return {
        "auto": "disable_collective",
        "required": "require_collective",
        "disabled": "disable_collective",
        "collective_first": "collective_first",
        "require_collective": "require_collective",
        "disable_collective": "disable_collective",
    }.get(collective, collective)


def execution_facts_payload(
    facts: MaterializationExecutionFacts,
) -> dict[str, object]:
    return {
        "tp_rank": facts.collective_rank,
        "tp_world_size": facts.collective_world_size,
        "same_node_tp": facts.same_node_tensor_parallel,
        "tp_ranks": tuple(int(rank) for rank in facts.tensor_parallel_ranks),
        "collective_world_size": facts.collective_world_size,
        "collective_rank": facts.collective_rank,
        "collective_context_unavailable": facts.collective_context_unavailable,
    }


def build_collective_group_id(
    *,
    artifact_ref: str,
    operation_scope: str,
    tp_ranks: tuple[int, ...],
    contract_identity: str | None = None,
) -> str:
    payload_dict: dict[str, Any] = {
        "artifact_ref": str(artifact_ref),
        "operation_scope": operation_scope,
        "tp_ranks": [int(rank) for rank in tp_ranks],
    }
    if contract_identity:
        payload_dict["contract_identity"] = str(contract_identity)
    payload = json.dumps(payload_dict, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"tensorcast-{digest}"


def host_materialization_request(
    *,
    host: Any | None,
    framework_config: object | None,
    operation_scope: str,
    read_contract_state: Callable[[], Any],
    source_bound_contract_path: str = SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4,
) -> HostMaterializationRequest:
    if host is None:
        return HostMaterializationRequest(operation_scope=operation_scope)
    profile = host.runtime_profile or RuntimeProfile()
    return HostMaterializationRequest(
        configured_collective_policy=CollectivePolicy(
            collective_policy_value(profile.materialization_policy)
        ),
        source_bound_contract_state=read_contract_state(),
        source_bound_contract_path=source_bound_contract_path,
        execution_facts=execution_facts_payload(
            host.placement.execution_facts(framework_config)
        ),
        operation_scope=operation_scope,
        require_materialization_options=True,
    )


def build_materialization_options(
    *,
    artifact_ref: str,
    operation_scope: str,
    configured_policy: CollectivePolicy,
    source_bound_contract_state: Any,
    source_bound_contract_path: str,
    execution_facts: Mapping[str, Any],
    contract_identity: str | None = None,
) -> tuple[Any, dict[str, object]]:
    facts = dict(execution_facts)
    return tc_binding_runtime.build_materialization_execution_context(
        artifact_ref=artifact_ref,
        operation_scope=operation_scope,
        configured_policy=configured_policy,
        tp_rank=int(facts.get("tp_rank", 0) or 0),
        tp_world_size=int(facts.get("tp_world_size", 1) or 1),
        same_node_tp=bool(facts.get("same_node_tp", False)),
        tp_ranks=tuple(int(rank) for rank in facts.get("tp_ranks", ()) or ()),
        collective_world_size=int(
            facts.get("collective_world_size", facts.get("tp_world_size", 1)) or 1
        ),
        collective_rank=int(facts.get("collective_rank", facts.get("tp_rank", 0)) or 0),
        source_bound_contract_profile_fields=source_bound_contract_profile_fields(
            source_bound_contract_state,
            source_bound_contract_path,
        ),
        build_group_id=build_collective_group_id,
        contract_identity=contract_identity,
        collective_context_unavailable=bool(
            facts.get("collective_context_unavailable", False)
        ),
    )


def request_materialization_options(
    *,
    explicit_options: Any | None,
    request: Any,
    artifact_ref: str,
    contract_identity: str | None,
    execution_facts: Mapping[str, Any] | None,
    build_options: Callable[..., tuple[Any, dict[str, object]]],
    missing_context_message: str,
    not_ready_message: str,
) -> Any | None:
    if explicit_options is not None:
        return explicit_options
    configured_collective_policy = getattr(
        request, "configured_collective_policy", None
    )
    source_bound_contract_state = getattr(request, "source_bound_contract_state", None)
    source_bound_contract_path = getattr(request, "source_bound_contract_path", None)
    require_materialization_options = bool(
        getattr(request, "require_materialization_options", False)
    )
    if (
        configured_collective_policy is None
        or source_bound_contract_state is None
        or not source_bound_contract_path
        or execution_facts is None
    ):
        if require_materialization_options:
            raise ArtifactRuntimeIntegrationError(missing_context_message)
        return None
    if require_materialization_options and not getattr(
        source_bound_contract_state,
        "source_bound_contract_ready",
        False,
    ):
        raise ArtifactRuntimeIntegrationError(not_ready_message)

    options, _profile = build_options(
        artifact_ref=artifact_ref,
        operation_scope=str(getattr(request, "operation_scope", "") or ""),
        configured_policy=configured_collective_policy,
        source_bound_contract_state=source_bound_contract_state,
        source_bound_contract_path=str(source_bound_contract_path),
        execution_facts=execution_facts,
        contract_identity=contract_identity,
    )
    return options


__all__ = [
    "HostMaterializationRequest",
    "build_collective_group_id",
    "build_materialization_options",
    "collective_policy_value",
    "execution_facts_payload",
    "host_materialization_request",
    "request_materialization_options",
]
