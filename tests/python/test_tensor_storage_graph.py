#  Copyright (c) 2025, TensorCast Team.

import re
from typing import Iterable

import torch
import pytest

from tensorcast.api._tensor_graph import build_tensor_storage_graph


def _assert_group_shares_storage(graph, names: Iterable[str]) -> str:
    group_ids = {graph.aliases[name].storage_id for name in names}
    assert len(group_ids) == 1, f"Expected {names} to share storage, got ids={group_ids}"
    return next(iter(group_ids))


def test_build_tensor_storage_graph_complex_layout():
    # Shared CPU tensors with various views/slices/transposes
    cpu_base = torch.arange(0, 64, dtype=torch.float32).reshape(8, 8)
    cpu_view_perm = cpu_base.t()
    cpu_slice = cpu_base[:, 2:6]
    cpu_narrow = cpu_base.reshape(4, 16)[1]

    # Independent CPU tensors with different dtypes
    cpu_independent = torch.ones(10, dtype=torch.float32) * 3.14
    cpu_int = torch.arange(0, 32, dtype=torch.int16)
    cpu_int_view = cpu_int.view(4, 8)

    tensors: dict[str, torch.Tensor] = {
        "cpu_base": cpu_base,
        "cpu_view_perm": cpu_view_perm,
        "cpu_slice": cpu_slice,
        "cpu_narrow": cpu_narrow,
        "cpu_independent": cpu_independent,
        "cpu_int": cpu_int,
        "cpu_int_view": cpu_int_view,
    }

    # Optional GPU tensors if CUDA available
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        gpu_base = torch.linspace(0, 1, steps=48, device=device, dtype=torch.float16)
        gpu_view = gpu_base.view(12, 4)
        gpu_slice = gpu_base[8:32]
        tensors.update(
            {
                "gpu_base": gpu_base,
                "gpu_view": gpu_view,
                "gpu_slice": gpu_slice,
            }
        )

    graph = build_tensor_storage_graph(tensors)

    assert set(graph.aliases.keys()) == set(tensors.keys())

    # Validate CPU shared storage group
    cpu_shared_id = _assert_group_shares_storage(
        graph, ["cpu_base", "cpu_view_perm", "cpu_slice", "cpu_narrow"]
    )
    cpu_storage_entry = graph.storages[cpu_shared_id]
    assert cpu_storage_entry.device_id == -1
    assert cpu_storage_entry.size_bytes == cpu_base.untyped_storage().nbytes()
    assert re.match(r"^-?[\d]+:[0-9a-f]{16}:[0-9a-f]{16}$", cpu_shared_id)

    # Validate independent CPU tensors do not share storage
    independent_ids = {
        graph.aliases["cpu_independent"].storage_id,
        graph.aliases["cpu_int"].storage_id,
    }
    assert cpu_shared_id not in independent_ids
    assert len(independent_ids) == 2
    assert (
        graph.storages[graph.aliases["cpu_independent"].storage_id].size_bytes
        == cpu_independent.untyped_storage().nbytes()
    )

    # Validate integer tensor view shares storage with base
    int_shared_id = _assert_group_shares_storage(graph, ["cpu_int", "cpu_int_view"])
    assert int_shared_id != cpu_shared_id
    assert graph.storages[int_shared_id].size_bytes == cpu_int.untyped_storage().nbytes()

    # Validate tensor_meta_index mirrors tensor attributes
    for name, tensor in tensors.items():
        shape, stride, dtype, storage_offset = graph.tensor_meta_index[name]
        assert shape == list(tensor.shape)
        assert stride == list(tensor.stride())
        assert dtype == str(tensor.dtype)
        assert storage_offset == tensor.storage_offset()

    # Validate tensor_source_index points to the correct storage base pointer
    for group in [["cpu_base", "cpu_view_perm", "cpu_slice", "cpu_narrow"], ["cpu_int", "cpu_int_view"]]:
        sources = {graph.tensor_source_index[name] for name in group}
        assert len(sources) == 1

    assert graph.tensor_source_index["cpu_independent"] != graph.tensor_source_index["cpu_base"]

    if torch.cuda.is_available():
        gpu_group_id = _assert_group_shares_storage(
            graph, ["gpu_base", "gpu_view", "gpu_slice"]
        )
        gpu_entry = graph.storages[gpu_group_id]
        assert gpu_entry.device_id == torch.cuda.current_device()
        assert gpu_entry.size_bytes == tensors["gpu_base"].untyped_storage().nbytes()
        gpu_sources = {graph.tensor_source_index[name] for name in ["gpu_base", "gpu_view", "gpu_slice"]}
        assert len(gpu_sources) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_storage_graph_cuda_only_groups():
    device = torch.device("cuda", torch.cuda.current_device())
    base = torch.randn(256, dtype=torch.float16, device=device)
    view = base.view(64, 4)
    slice_a = base[16:128]
    slice_b = base[32:192:2]
    reshape_view = base.reshape(32, 8)
    independent = torch.ones(32, dtype=torch.float16, device=device)

    tensors = {
        "base": base,
        "view": view,
        "slice_a": slice_a,
        "slice_b": slice_b,
        "reshape_view": reshape_view,
        "independent": independent,
    }
    graph = build_tensor_storage_graph(tensors)

    shared_id = _assert_group_shares_storage(
        graph, ["base", "view", "slice_a", "slice_b", "reshape_view"]
    )
    storage_entry = graph.storages[shared_id]
    assert storage_entry.device_id == torch.cuda.current_device()
    assert storage_entry.size_bytes == base.untyped_storage().nbytes()

    independent_id = graph.aliases["independent"].storage_id
    assert independent_id != shared_id
    assert (
        graph.storages[independent_id].size_bytes == independent.untyped_storage().nbytes()
    )

    shared_offsets = {graph.tensor_source_index[name] for name in tensors if name != "independent"}
    assert len(shared_offsets) == 1

    for name, tensor in tensors.items():
        shape, stride, dtype, storage_offset = graph.tensor_meta_index[name]
        assert shape == list(tensor.shape)
        assert stride == list(tensor.stride())
        assert dtype == str(tensor.dtype)
        assert storage_offset == tensor.storage_offset()
