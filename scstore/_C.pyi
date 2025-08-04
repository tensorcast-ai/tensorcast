"""Type stubs for scstore._C module (checkpoint functionality)"""

from typing import Dict, List, Tuple, Optional, Union, Sequence, Mapping, MutableMapping, Any
import os
import torch

# Type aliases for readability
_Ptr = int  # Raw memory address
_Offset = int  # File/device offset in bytes
_Size = int  # Size in bytes
_VerificationInfo = Dict[str, Union[int, List[int]]]
_DeviceId = Union[int, torch.device]
_PathLike = Union[str, os.PathLike[str]]

# -----------------------------------------------------------------------------
# Tensor save / load helpers
# -----------------------------------------------------------------------------

def save_tensors(
    tensor_names: Sequence[str],
    tensor_data: MutableMapping[str, Tuple[_Ptr, _Size]],
    path: _PathLike,
) -> Dict[str, _Offset]: ...


def save_tensors_streaming(
    tensor_names: Sequence[str],
    tensor_data: MutableMapping[str, Tuple[_Ptr, _Size]],
    path: _PathLike,
    config: Optional[Mapping[str, Union[int, bool]]] = None,
) -> Dict[str, _Offset]: ...


def restore_tensors(
    meta_state_dict: Mapping[str, Tuple[Sequence[int], Sequence[int], str, int]],
    memory_base_address: Mapping[_DeviceId, _Ptr],
    tensor_device_offsets: Mapping[_DeviceId, Mapping[str, _Offset]],
    from_ipc_shm: bool,
) -> Dict[str, torch.Tensor]: ...


def restore_tensors_from_model_path(
    meta_state_dict: Mapping[str, Tuple[Sequence[int], Sequence[int], str, int]],
    model_path: _PathLike,
    tensor_device_offsets: Mapping[str, _Offset],
    device_id: int = -1,
) -> Dict[str, torch.Tensor]: ...

# -----------------------------------------------------------------------------
# CUDA memory helpers
# -----------------------------------------------------------------------------

def allocate_cuda_memory(
    device_id: int,
    size: int,
) -> _Ptr: ...


def get_cuda_memory_handle(
    device_id: int,
    memory_ptr: _Ptr,
) -> bytes: ...


def get_device_uuid_map() -> Dict[_DeviceId, str]: ...


def get_cuda_memory_ptr(
    device_id: _DeviceId,
    cuda_ipc_handle: Union[bytes, Any],
) -> _Ptr: ...


def close_cuda_memory_handle(
    device_id: int,
    cuda_memory_ptr: _Ptr,
) -> bool: ...

# -----------------------------------------------------------------------------
# Verification utilities
# -----------------------------------------------------------------------------

def generate_model_verification_info(
    model_path: _PathLike,
    verification_level: int = 1,
) -> _VerificationInfo: ...


def verify_model_data_from_gpu(
    device_id: int,
    cuda_memory_ptr: _Ptr,
    memory_size: int,
    expected_verification: _VerificationInfo,
    verification_level: int,
) -> bool: ...