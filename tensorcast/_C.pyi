"""Type stubs for tensorcast._C module (checkpoint functionality)"""

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
# Build configuration
# -----------------------------------------------------------------------------


def is_fake_cuda() -> bool:
    """Returns True if the runtime Fake CUDA backend is active."""
    ...


# -----------------------------------------------------------------------------
# Tensor save / load helpers
# -----------------------------------------------------------------------------


def save_tensors(
    tensor_names: Sequence[str],
    tensor_data: MutableMapping[str, Tuple[_Ptr, _Size]],
    path: _PathLike,
) -> Dict[str, _Offset]: ...

def save_model_to_disk(
    tensor_names: Sequence[str],
    tensor_data: MutableMapping[str, Tuple[_Ptr, _Size]],
    meta_state_dict: Mapping[str, Tuple[Sequence[int], Sequence[int], str, int]],
    path: _PathLike,
    config: Optional[Mapping[str, Union[int, bool]]] = None,
) -> Mapping[str, Union[str, int]]: ...


def restore_tensors(
    meta_state_dict: Mapping[str, Tuple[Sequence[int], Sequence[int], str, int]],
    memory_base_address: Mapping[_DeviceId, _Ptr],
    tensor_device_offsets: Mapping[_DeviceId, Mapping[str, _Offset]],
    from_ipc_shm: bool,
    lease_token: bytes = b"",
    local_handle_socket_path: str = "",
) -> Dict[str, torch.Tensor]: ...

def restore_tensors_from_cpu_fd_with_lease(
    meta_state_dict: Mapping[str, Tuple[Sequence[int], Sequence[int], str, int]],
    fd: int,
    size_bytes: int,
    offset_bytes: int,
    tensor_device_offsets: Mapping[str, _Offset],
    lease_token: bytes,
    local_handle_socket_path: str,
) -> Dict[str, torch.Tensor]: ...


def restore_tensors_from_disk(
    meta_state_dict: Mapping[str, Tuple[Sequence[int], Sequence[int], str, int]],
    disk_path: _PathLike,
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


def get_cuda_memory_handle_with_offset(
    device_id: int,
    memory_ptr: _Ptr,
) -> Tuple[bytes, int]: ...


def get_device_uuid_map() -> Dict[_DeviceId, str]: ...


def get_cuda_memory_ptr(
    device_id: _DeviceId,
    cuda_ipc_handle: Union[bytes, Any],
) -> _Ptr: ...


def close_cuda_memory_handle(
    device_id: int,
    cuda_memory_ptr: _Ptr,
) -> bool: ...


def inspect_or_generate_descriptor(disk_path: _PathLike) -> Mapping[str, Union[str, int]]: ...


def build_canonical_index_from_safetensors(artifact: _PathLike) -> bytes: ...


# -----------------------------------------------------------------------------
# Verification utilities
# -----------------------------------------------------------------------------


def generate_artifact_verification_info(
    disk_path: _PathLike,
    verification_level: int = 1,
) -> _VerificationInfo: ...


def verify_artifact_data_from_gpu(
    device_id: int,
    cuda_memory_ptr: _Ptr,
    memory_size: int,
    expected_verification: _VerificationInfo,
    verification_level: int,
) -> bool: ...
