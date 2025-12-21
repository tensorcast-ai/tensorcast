#  Copyright (c) 2025, TensorCast Team.

"""Lazy loader for the tensorcast._C native extension."""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Mapping, MutableMapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_C_MODULE: ModuleType | None = None
_C_LOCK = threading.Lock()
_C_VALIDATED = False


def _load_c_ext() -> ModuleType:
    global _C_MODULE, _C_VALIDATED

    if _C_MODULE is None or not _C_VALIDATED:
        with _C_LOCK:
            if _C_MODULE is None:
                _C_MODULE = importlib.import_module("tensorcast._C")
            if not _C_VALIDATED:
                from tensorcast._build_config import validate_cuda_backend_consistency

                validate_cuda_backend_consistency()
                _C_VALIDATED = True

    return _C_MODULE


def get_c_ext() -> ModuleType:
    """Return the loaded C extension module, importing it if needed."""
    return _load_c_ext()


def inspect_or_generate_descriptor(
    disk_path: str | os.PathLike[str],
) -> Mapping[str, str | int]:
    return _load_c_ext().inspect_or_generate_descriptor(disk_path)


def build_canonical_index_from_safetensors(
    artifact: str | os.PathLike[str],
) -> bytes:
    return _load_c_ext().build_canonical_index_from_safetensors(artifact)


def get_device_uuid_map() -> dict[int | torch.device, str]:
    return _load_c_ext().get_device_uuid_map()


def get_cuda_memory_ptr(
    device_id: int | torch.device,
    cuda_ipc_handle: bytes | Any,
) -> int:
    return _load_c_ext().get_cuda_memory_ptr(device_id, cuda_ipc_handle)


def restore_tensors(
    meta_state_dict: Mapping[str, tuple[Sequence[int], Sequence[int], str, int]],
    memory_base_address: Mapping[int | torch.device, int],
    tensor_device_offsets: Mapping[int | torch.device, Mapping[str, int]],
    from_ipc_shm: bool,
) -> dict[str, torch.Tensor]:
    return _load_c_ext().restore_tensors(
        meta_state_dict, memory_base_address, tensor_device_offsets, from_ipc_shm
    )


def collect_tensor_storage_graph(
    tensors: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    return _load_c_ext().collect_tensor_storage_graph(tensors)


def compute_view_registration_plan(
    canonical_index_bytes: bytes,
    normalized_ops: Mapping[str, Any],
) -> Mapping[str, Any]:
    return _load_c_ext().compute_view_registration_plan(
        canonical_index_bytes, normalized_ops
    )


def get_cuda_memory_handle(device_id: int, memory_ptr: int) -> bytes:
    return _load_c_ext().get_cuda_memory_handle(device_id, memory_ptr)


def restore_tensors_from_disk(
    meta_state_dict: Mapping[str, tuple[Sequence[int], Sequence[int], str, int]],
    disk_path: str | os.PathLike[str],
    tensor_device_offsets: Mapping[str, int],
    device_id: int = -1,
) -> dict[str, torch.Tensor]:
    return _load_c_ext().restore_tensors_from_disk(
        meta_state_dict, disk_path, tensor_device_offsets, device_id
    )


def save_model_to_disk(
    tensor_names: Sequence[str],
    tensor_data: MutableMapping[str, tuple[int, int]],
    meta_state_dict: Mapping[str, tuple[Sequence[int], Sequence[int], str, int]],
    path: str | os.PathLike[str],
    config: Mapping[str, int | bool] | None = None,
) -> Mapping[str, str | int]:
    return _load_c_ext().save_model_to_disk(
        tensor_names, tensor_data, meta_state_dict, path, config
    )


__all__ = [
    "build_canonical_index_from_safetensors",
    "collect_tensor_storage_graph",
    "compute_view_registration_plan",
    "get_c_ext",
    "get_cuda_memory_handle",
    "get_cuda_memory_ptr",
    "get_device_uuid_map",
    "inspect_or_generate_descriptor",
    "restore_tensors",
    "restore_tensors_from_disk",
    "save_model_to_disk",
]
