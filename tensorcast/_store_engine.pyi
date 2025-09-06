"""Type stubs for tensorcast._store_engine module"""

from typing import List, Dict, Tuple, Optional, Any, Union, TypedDict
from enum import Enum


# =======================
# Core Enumerations
# =======================


class MemoryLocation(Enum):
    NONE: "MemoryLocation"
    DISK: "MemoryLocation"
    CPU: "MemoryLocation"
    GPU: "MemoryLocation"
    REMOTE: "MemoryLocation"


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
    artifact: int
    location: MemoryLocation
    device_id: int
    comm_dev_type: int
    buffer_addresses: List[int]  # Read-only property
    buffer_sizes: List[int]
    remote_memory_keys: List[str]

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class DeviceKey:
    type: DeviceType
    ordinal: int
    uuid: str

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class ReplicaKey:
    artifact_id: str
    device: DeviceKey
    replica: int

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class ReplicaInfo:
    artifact_id: str
    size_bytes: int
    cpu_state: MemoryLocation
    gpu_state: MemoryLocation
    gpu_device_id: int
    gpu_device_uuid: str
    is_registered_for_comm: bool
    last_access_timestamp: float  # Read-only property
    load_timestamp: float  # Read-only property

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class MaterializeMode(Enum):
    AUTO: "MaterializeMode"
    COPY_ONLY: "MaterializeMode"
    LOAD_ONLY: "MaterializeMode"


class ReplicaHandle:
    def wait_ready(self, timeout_ms: int) -> None: ...

    @property
    def gpu_ptr(self) -> int: ...

    @property
    def ipc_handle_bytes(self) -> bytes: ...

    @property
    def replica_key(self) -> str: ...


# =======================
# Main Entry Point
# =======================


class StoreEngine:
    # ---- Unified materialize_replica API ----
    def materialize_replica(
        self,
        target_device: Union[DeviceKey, str, int] = "gpu:0",
        mode: MaterializeMode = MaterializeMode.AUTO,
        *,
        pinned_timeout_ms: Optional[int] = None,
        artifact_id: Optional[str] = None,
        disk_path: Optional[str] = None,
    ) -> ReplicaHandle: ...

    # ---- Multi-device helpers ----
    def get_resident_devices(self, artifact_id: str) -> List[DeviceKey]: ...

    def list_device_replicas(self, device: DeviceKey) -> List[ReplicaKey]: ...

    def wait_replica_ready(self, replica_key: ReplicaKey) -> int: ...

    def unload_replica(self, replica_key: ReplicaKey) -> int: ...

    def get_replica_state(self, replica_key: ReplicaKey, memory_type: DeviceType) -> MemoryState: ...

    def get_replica_gpu_ptr(self, replica_key: ReplicaKey) -> int: ...

    def get_replica_size(self, replica_key: ReplicaKey) -> int: ...

    # ---- Distributed Memory Pool helpers ----
    def lock_chunks(
        self,
        replica_key: ReplicaKey,
        chunk_indices: List[int],
    ) -> int: ...

    def unlock_chunks(
        self,
        replica_key: ReplicaKey,
        chunk_indices: List[int],
        copied_gpu: bool,
    ) -> int: ...

    def enable_remote_replica_access(
        self,
        replica_key: ReplicaKey,
        location: MemoryLocation,
    ) -> CommRegistrationInfo: ...

    def disable_remote_replica_access(
        self,
        replica_key: ReplicaKey,
        location: MemoryLocation,
    ) -> bool: ...

    # ---- Memory / Metrics ----
    def clear_mem(self) -> int: ...

    def get_mem_pool_size(self) -> int: ...

    def get_chunk_size(self) -> int: ...

    def get_available_memory(self) -> int: ...

    def get_all_replicas_info(self) -> List[ReplicaInfo]: ...

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
    @staticmethod
    def from_config(listen_addr: str = "0.0.0.0", port: int = 9090, config: dict[str, object] = ...) -> "CommunicationManager": ...
    @staticmethod
    def from_yaml(listen_addr: str = "0.0.0.0", port: int = 9090, path: str = ...) -> "CommunicationManager": ...

    def is_enabled(self) -> bool: ...


def create_store_engine(config: Dict[str, Any]) -> StoreEngine: ...
