#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest
import torch

from tensorcast.api.store.serving_builder import compute_serving_tensor_schema_hash
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.serving.contract import (
    collect_runtime_tensor_schema,
    compute_runtime_representation_contract_hash,
    compute_runtime_tensor_schema_hash,
    logical_topology_json,
)
from tensorcast.types import ServingBindingMemberRef, ServingTopologyRef


def test_runtime_tensor_schema_hash_matches_serving_builder_contract() -> None:
    tensor = torch.empty((2, 3), dtype=torch.float16)
    schema = collect_runtime_tensor_schema(
        {"weights": tensor},
        remove_duplicate=False,
    )
    canonical_index = CanonicalIndex(
        entries=(
            CanonicalIndexEntry(
                name="weights",
                dtype=torch.float16,
                shape=(2, 3),
                stride=(3, 1),
                storage_offset=0,
                segment_offset=0,
                size_bytes=12,
            ),
        ),
        total_size_bytes=12,
        avbs_hash="",
    )

    assert compute_runtime_tensor_schema_hash(
        schema) == compute_serving_tensor_schema_hash(canonical_index)


def test_runtime_tensor_schema_requires_zero_storage_offset() -> None:
    view = torch.empty((4, ), dtype=torch.float32)[1:]

    with pytest.raises(ValueError, match="storage_offset == 0"):
        collect_runtime_tensor_schema({"view": view}, remove_duplicate=False)


def test_runtime_tensor_schema_duplicate_filter_is_explicit() -> None:
    tensor = torch.empty((2, ), dtype=torch.float32)

    full = collect_runtime_tensor_schema(
        {
            "a": tensor,
            "b": tensor,
        },
        remove_duplicate=False,
    )
    deduped = collect_runtime_tensor_schema(
        {
            "a": tensor,
            "b": tensor,
        },
        remove_duplicate=True,
    )

    assert tuple(entry.name for entry in full) == ("a", "b")
    assert tuple(entry.name for entry in deduped) == ("a", )


def test_logical_topology_json_is_canonicalized_by_core() -> None:
    topology = ServingTopologyRef(schema_topology_digest="topology-digest")

    payload_a = {
        "family":
        "vllm_tensor_parallel",
        "version":
        "v1",
        "dimensions": [
            {
                "name": "pipeline_parallel",
                "size": 1,
            },
            {
                "name": "tensor_parallel",
                "size": 2,
            },
        ],
    }
    payload_b = {
        "dimensions": [
            {
                "size": 2,
                "name": "tensor_parallel",
            },
            {
                "size": 1,
                "name": "pipeline_parallel",
            },
        ],
        "version":
        "v1",
        "family":
        "vllm_tensor_parallel",
    }

    assert logical_topology_json(topology, framework_payload=payload_a) == \
        logical_topology_json(topology, framework_payload=payload_b)


def test_runtime_representation_contract_hash_is_versioned_and_stable() -> None:
    topology = ServingTopologyRef(
        schema_topology_digest="topology-digest",
        logical_topology_ref="vllm://parallelism?tp=2",
    )
    member = ServingBindingMemberRef(
        member_id="tp1",
        member_index=1,
        member_count=2,
        group_id="group-1",
    )

    hash_a = compute_runtime_representation_contract_hash(
        tensor_schema_hash="schema-hash",
        topology_ref=topology,
        member_ref=member,
        framework_name="vllm",
        framework_version="1.0",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        source_identity={
            "model_hash": "model-hash",
            "model_name": "model-a",
        },
    )
    hash_b = compute_runtime_representation_contract_hash(
        tensor_schema_hash="schema-hash",
        topology_ref=topology,
        member_ref=member,
        framework_name="vllm",
        framework_version="1.0",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        source_identity={
            "model_name": "model-a",
            "model_hash": "model-hash",
        },
    )
    hash_c = compute_runtime_representation_contract_hash(
        tensor_schema_hash="schema-hash-other",
        topology_ref=topology,
        member_ref=member,
        framework_name="vllm",
        framework_version="1.0",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        source_identity={
            "model_name": "model-a",
            "model_hash": "model-hash",
        },
    )

    assert hash_a
    assert hash_a == hash_b
    assert hash_a != hash_c
