#  Copyright (c) 2025-2026, TensorCast Team.

import pytest
from pydantic import ValidationError

from tensorcast.api._config import (
    CollectivePolicyMode,
    ExecutionTopologyContext,
    GetArtifactOptions,
    RetrievalPolicy,
    RetrievalPreference,
    RetrievalPreset,
)
from tensorcast.api.context import CollectiveLoadGroup
from tensorcast.api.store.types import StoreOptions


def test_retrieval_policy_rejects_invalid_preference_gate() -> None:
    with pytest.raises(ValidationError):
        RetrievalPolicy(
            preference=RetrievalPreference.PREFER_DISK,
            allow_disk=False,
        )


def test_get_options_parse_source_preset() -> None:
    opts = GetArtifactOptions(source="disk_first")
    assert opts.source is not None
    assert opts.source.preference is RetrievalPreference.PREFER_DISK
    assert opts.source.allow_disk is True


def test_get_options_parse_topology() -> None:
    opts = GetArtifactOptions(
        execution_topology=ExecutionTopologyContext(
            collective_group=CollectiveLoadGroup(
                group_id="group-a",
                world_size=4,
                rank=2,
            ),
            collective_policy="require_collective",
        )
    )
    assert opts.execution_topology is not None
    assert opts.execution_topology.collective_group is not None
    assert opts.execution_topology.collective_group.rank == 2
    assert (
        opts.execution_topology.collective_policy
        is CollectivePolicyMode.REQUIRE_COLLECTIVE
    )


def test_execution_topology_keeps_unspecified_collective_policy() -> None:
    opts = GetArtifactOptions(
        execution_topology=ExecutionTopologyContext(
            collective_group=CollectiveLoadGroup(
                group_id="group-a",
                world_size=4,
                rank=2,
            )
        )
    )

    assert opts.execution_topology is not None
    assert opts.execution_topology.collective_group is not None
    assert opts.execution_topology.collective_policy is None


def test_collective_policy_parse_rejects_unspecified_value() -> None:
    with pytest.raises(ValueError, match="must be explicit"):
        CollectivePolicyMode.parse(None)


def test_store_options_accept_execution_scoped_defaults() -> None:
    opts = StoreOptions(get=GetArtifactOptions(source=RetrievalPreset.DISK_ONLY))
    assert opts.get is not None
    assert opts.get.source is not None
    assert opts.get.source.allow_p2p is False
