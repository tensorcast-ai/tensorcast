"""Type stubs for scstore._checkpoint_store module"""

from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum


# =======================
# Core Enumerations
# =======================


class ModelLocation(Enum):
    NONE: "ModelLocation"
    DISK: "ModelLocation"
    CPU: "ModelLocation"
    GPU: "ModelLocation"
    REMOTE: "ModelLocation"


class MemoryState(Enum):
    UNINITIALIZED: "MemoryState"
    UNALLOCATED: "MemoryState"
    ALLOCATED: "MemoryState"
    LOADING: "MemoryState"
    LOADED: "MemoryState"
    FAILED: "MemoryState"


class DeviceType(Enum):
    CPU: "DeviceType"
    GPU: "DeviceType"
    REMOTE: "DeviceType"
    DISK: "DeviceType"
    NONE: "DeviceType"


# =======================
# Data Structures
# =======================


class CommRegistrationInfo:
    model_size: int
    location: ModelLocation
    device_id: int
    comm_dev_type: int
    buffer_addresses: List[int]  # Read-only property
    buffer_sizes: List[int]
    remote_memory_keys: List[str]

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class MemCopyChunk:
    src_offset: int
    size: int
    dst_offset: int
    handle_idx: int

    def __init__(self) -> None: ...


class DeviceKey:
    type: DeviceType
    ordinal: int
    uuid: str

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class InstanceKey:
    model_id: str
    device: DeviceKey
    replica: int

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class ModelInfo:
    model_id: str
    size_bytes: int
    cpu_state: ModelLocation
    gpu_state: ModelLocation
    gpu_device_id: int
    gpu_device_uuid: str
    is_registered_for_comm: bool
    last_access_timestamp: float  # Read-only property
    load_timestamp: float  # Read-only property

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class PrepareMode(Enum):
    AUTO: "PrepareMode"
    COPY_ONLY: "PrepareMode"
    LOAD_ONLY: "PrepareMode"


class ModelHandle:
    def wait_ready(self, timeout_ms: int) -> None: ...

    @property
    def gpu_ptr(self) -> int: ...

    @property
    def ipc_handle_bytes(self) -> bytes: ...

    @property
    def instance_key(self) -> str: ...


# =======================
# Main Entry Point
# =======================


class CheckpointStore:
    def __init__(self) -> None: ...

    # ---- Unified prepare API ----
    def prepare(
        self,
        model_id: str,
        target_device: Union[DeviceKey, str, int] = "gpu:0",
        mode: PrepareMode = PrepareMode.AUTO,
        *,
        pinned_timeout_ms: Optional[int] = None
    ) -> ModelHandle: ...

    # ---- Multi-device helpers ----
    def get_loaded_devices(self, model_id: str) -> List[DeviceKey]: ...

    def list_device_models(self, device: DeviceKey) -> List[InstanceKey]: ...

    def wait_instance_ready(self, instance_key: InstanceKey) -> int: ...

    def unload_instance(self, instance_key: InstanceKey) -> int: ...

    def get_instance_state(self, instance_key: InstanceKey, memory_type: DeviceType) -> MemoryState: ...

    def get_instance_gpu_ptr(self, instance_key: InstanceKey) -> int: ...

    # ---- Distributed Memory Pool helpers ----
    def lock_chunks(
        self,
        instance_key: InstanceKey,
        chunk_indices: List[int],
    ) -> int: ...

    def unlock_chunks(
        self,
        instance_key: InstanceKey,
        chunk_indices: List[int],
        copied_gpu: bool,
    ) -> int: ...

    def enable_remote_instance_access(
        self,
        instance_key: InstanceKey,
        location: ModelLocation,
    ) -> CommRegistrationInfo: ...

    def disable_remote_instance_access(
        self,
        instance_key: InstanceKey,
        location: ModelLocation,
    ) -> bool: ...

    # ---- Memory / Metrics ----
    def clear_mem(self) -> int: ...

    def get_mem_pool_size(self) -> int: ...

    def get_chunk_size(self) -> int: ...

    def get_available_memory(self) -> int: ...

    def get_all_models_info(self) -> List[ModelInfo]: ...

    def get_gpu_memory_stats(self) -> List[Tuple[int, int]]: ...

    def __repr__(self) -> str: ...


# =======================
# Auxiliary Constructs
# =======================


class CommunicationManager:
    def __init__(
        self,
        listen_addr: str = "0.0.0.0",
        port: int = 9090,
        enable_rdma: bool = False,
    ) -> None: ...

    def is_enabled(self) -> bool: ...


def create_checkpoint_store(config: Dict[str, Any]) -> CheckpointStore: ...


def get_global_metrics_text() -> bytes: ...