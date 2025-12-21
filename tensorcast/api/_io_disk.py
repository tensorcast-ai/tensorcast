#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os
from pathlib import Path

import torch

from tensorcast._c_ext import restore_tensors_from_disk, save_model_to_disk

from ._device import resolve_device
from ._indices import calculate_tensor_device_offsets, load_tensor_indices_from_dir
from ._tensor_graph import build_tensor_storage_graph


def save_dict(
    state_dict: dict[str, torch.Tensor],
    disk_path: str | os.PathLike,
    streaming_config: dict | None = None,
) -> dict:
    """Persist a PyTorch state_dict to disk using the unified writer.

    Args:
        state_dict: Mapping from tensor name to tensor.
        disk_path: Target directory for artifact files.
        streaming_config: Optional configuration dict for the writer
            (e.g., num_buffers, buffer_size_mb, enable_async_write).

    Returns:
        A descriptor dict with artifact_id, index_multihash, data_multihash,
        schema_version, encoding, and total_size.
    """
    graph = build_tensor_storage_graph(state_dict)
    tensor_names = sorted(graph.aliases.keys())
    tensor_data_index: dict[str, tuple[int, int]] = {
        name: graph.tensor_source_index[name] for name in tensor_names
    }
    meta_state_dict: dict[str, tuple[list[int], list[int], str, int]] = {}
    for name in tensor_names:
        alias = graph.aliases[name]
        meta_state_dict[name] = (
            list(alias.shape),
            list(alias.stride),
            alias.dtype,
            int(alias.storage_offset),
        )

    config = streaming_config or {}
    descriptor = save_model_to_disk(
        tensor_names, tensor_data_index, meta_state_dict, str(disk_path), config
    )
    return dict(descriptor)


def load_dict_from_disk(
    disk_path: str | os.PathLike,
    *,
    device_id: int | torch.device = 0,
) -> dict[str, torch.Tensor]:
    raw_disk_path = Path(str(disk_path))
    artifact_dir = raw_disk_path
    tensor_meta_index, tensor_data_index = load_tensor_indices_from_dir(artifact_dir)

    device_id_int: int = resolve_device(device_id)
    tensor_device_offsets, _ = calculate_tensor_device_offsets(
        tensor_data_index, device_id_int
    )
    per_tensor_offsets: dict[str, int] = dict(
        tensor_device_offsets.get(device_id_int, {})
    )

    target_device_for_local = device_id_int if torch.cuda.is_available() else -1

    state_dict = restore_tensors_from_disk(
        tensor_meta_index,
        str(artifact_dir),
        per_tensor_offsets,
        target_device_for_local,
    )
    return state_dict
