#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral runtime tensor schema validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from tensorcast.serving.builder.compiler import TensorSchemaEntry


def validate_tensor_schema_against_tensors(
    tensor_schema: Sequence[TensorSchemaEntry],
    tensors: Mapping[str, torch.Tensor],
) -> None:
    expected = {entry.name: entry for entry in tensor_schema}
    actual = dict(tensors)
    missing = set(expected) - set(actual)
    unexpected = set(actual) - set(expected)
    if missing or unexpected:
        raise RuntimeError(
            "TensorCast finalized tensor schema mismatch. "
            f"Missing={sorted(missing)}, Unexpected={sorted(unexpected)}"
        )
    for name, entry in expected.items():
        tensor = actual[name]
        _validate_tensor_schema_entry(name, entry, tensor)


def _validate_tensor_schema_entry(
    name: str,
    entry: TensorSchemaEntry,
    tensor: torch.Tensor,
) -> None:
    shape = tuple(int(dim) for dim in tensor.shape)
    stride = tuple(int(dim) for dim in tensor.stride())
    if shape != tuple(int(dim) for dim in entry.shape):
        raise RuntimeError(
            f"TensorCast finalized tensor shape mismatch for {name}: "
            f"{shape} != {entry.shape}"
        )
    if stride != tuple(int(dim) for dim in entry.stride):
        raise RuntimeError(
            f"TensorCast finalized tensor stride mismatch for {name}: "
            f"{stride} != {entry.stride}"
        )
    if str(tensor.dtype) != str(entry.dtype):
        raise RuntimeError(
            f"TensorCast finalized tensor dtype mismatch for {name}: "
            f"{tensor.dtype} != {entry.dtype}"
        )


__all__ = ["validate_tensor_schema_against_tensors"]
