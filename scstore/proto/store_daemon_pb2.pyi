from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DeviceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEVICE_TYPE_DISK: _ClassVar[DeviceType]
    DEVICE_TYPE_CPU: _ClassVar[DeviceType]
    DEVICE_TYPE_GPU: _ClassVar[DeviceType]

class LoadModelStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOAD_MODEL_STATUS_ALLOCATED: _ClassVar[LoadModelStatus]
    LOAD_MODEL_STATUS_FAILED: _ClassVar[LoadModelStatus]

class VerificationStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VERIFICATION_STATUS_UNKNOWN: _ClassVar[VerificationStatus]
    VERIFICATION_STATUS_IN_PROGRESS: _ClassVar[VerificationStatus]
    VERIFICATION_STATUS_PASSED: _ClassVar[VerificationStatus]
    VERIFICATION_STATUS_FAILED: _ClassVar[VerificationStatus]

class ModelLocation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_LOCATION_NONE: _ClassVar[ModelLocation]
    MODEL_LOCATION_DISK: _ClassVar[ModelLocation]
    MODEL_LOCATION_CPU: _ClassVar[ModelLocation]
    MODEL_LOCATION_GPU: _ClassVar[ModelLocation]
    MODEL_LOCATION_REMOTE: _ClassVar[ModelLocation]
DEVICE_TYPE_DISK: DeviceType
DEVICE_TYPE_CPU: DeviceType
DEVICE_TYPE_GPU: DeviceType
LOAD_MODEL_STATUS_ALLOCATED: LoadModelStatus
LOAD_MODEL_STATUS_FAILED: LoadModelStatus
VERIFICATION_STATUS_UNKNOWN: VerificationStatus
VERIFICATION_STATUS_IN_PROGRESS: VerificationStatus
VERIFICATION_STATUS_PASSED: VerificationStatus
VERIFICATION_STATUS_FAILED: VerificationStatus
MODEL_LOCATION_NONE: ModelLocation
MODEL_LOCATION_DISK: ModelLocation
MODEL_LOCATION_CPU: ModelLocation
MODEL_LOCATION_GPU: ModelLocation
MODEL_LOCATION_REMOTE: ModelLocation

class GetServerConfigRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetServerConfigResponse(_message.Message):
    __slots__ = ("mem_pool_size", "chunk_size")
    MEM_POOL_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    mem_pool_size: int
    chunk_size: int
    def __init__(self, mem_pool_size: _Optional[int] = ..., chunk_size: _Optional[int] = ...) -> None: ...

class RegisterModelRequest(_message.Message):
    __slots__ = ("model_path",)
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    def __init__(self, model_path: _Optional[str] = ...) -> None: ...

class RegisterModelResponse(_message.Message):
    __slots__ = ("model_path", "model_size")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    MODEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    model_size: int
    def __init__(self, model_path: _Optional[str] = ..., model_size: _Optional[int] = ...) -> None: ...

class MemCopyHandle(_message.Message):
    __slots__ = ("cuda_ipc_handle",)
    CUDA_IPC_HANDLE_FIELD_NUMBER: _ClassVar[int]
    cuda_ipc_handle: bytes
    def __init__(self, cuda_ipc_handle: _Optional[bytes] = ...) -> None: ...

class LoadModelRequest(_message.Message):
    __slots__ = ("model_path", "replica_uuid", "device_uuid", "target_device_type", "pinned_allocation_timeout_ms", "pid", "keep_for_global", "size_bytes")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    REPLICA_UUID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_UUID_FIELD_NUMBER: _ClassVar[int]
    TARGET_DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PINNED_ALLOCATION_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    KEEP_FOR_GLOBAL_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    replica_uuid: str
    device_uuid: str
    target_device_type: DeviceType
    pinned_allocation_timeout_ms: int
    pid: int
    keep_for_global: bool
    size_bytes: int
    def __init__(self, model_path: _Optional[str] = ..., replica_uuid: _Optional[str] = ..., device_uuid: _Optional[str] = ..., target_device_type: _Optional[_Union[DeviceType, str]] = ..., pinned_allocation_timeout_ms: _Optional[int] = ..., pid: _Optional[int] = ..., keep_for_global: bool = ..., size_bytes: _Optional[int] = ...) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = ("model_path", "mem_handle", "status")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    MEM_HANDLE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    mem_handle: MemCopyHandle
    status: LoadModelStatus
    def __init__(self, model_path: _Optional[str] = ..., mem_handle: _Optional[_Union[MemCopyHandle, _Mapping]] = ..., status: _Optional[_Union[LoadModelStatus, str]] = ...) -> None: ...

class ConfirmModelRequest(_message.Message):
    __slots__ = ("model_path", "replica_uuid", "target_device_type")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    REPLICA_UUID_FIELD_NUMBER: _ClassVar[int]
    TARGET_DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    replica_uuid: str
    target_device_type: DeviceType
    def __init__(self, model_path: _Optional[str] = ..., replica_uuid: _Optional[str] = ..., target_device_type: _Optional[_Union[DeviceType, str]] = ...) -> None: ...

class ConfirmModelResponse(_message.Message):
    __slots__ = ("model_path", "code")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    code: int
    def __init__(self, model_path: _Optional[str] = ..., code: _Optional[int] = ...) -> None: ...

class UnloadModelRequest(_message.Message):
    __slots__ = ("model_path", "replica_uuid", "target_device_type", "pid", "version")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    REPLICA_UUID_FIELD_NUMBER: _ClassVar[int]
    TARGET_DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    replica_uuid: str
    target_device_type: DeviceType
    pid: int
    version: str
    def __init__(self, model_path: _Optional[str] = ..., replica_uuid: _Optional[str] = ..., target_device_type: _Optional[_Union[DeviceType, str]] = ..., pid: _Optional[int] = ..., version: _Optional[str] = ...) -> None: ...

class UnloadModelResponse(_message.Message):
    __slots__ = ("model_path", "code")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    code: int
    def __init__(self, model_path: _Optional[str] = ..., code: _Optional[int] = ...) -> None: ...

class ClearMemRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ClearMemResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetWorkerStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetWorkerStatusResponse(_message.Message):
    __slots__ = ("is_registered", "is_healthy", "is_shutting_down", "mem_pool_total_size", "mem_pool_available_size", "uptime_seconds", "worker_id")
    IS_REGISTERED_FIELD_NUMBER: _ClassVar[int]
    IS_HEALTHY_FIELD_NUMBER: _ClassVar[int]
    IS_SHUTTING_DOWN_FIELD_NUMBER: _ClassVar[int]
    MEM_POOL_TOTAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MEM_POOL_AVAILABLE_SIZE_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    is_registered: bool
    is_healthy: bool
    is_shutting_down: bool
    mem_pool_total_size: int
    mem_pool_available_size: int
    uptime_seconds: int
    worker_id: str
    def __init__(self, is_registered: bool = ..., is_healthy: bool = ..., is_shutting_down: bool = ..., mem_pool_total_size: _Optional[int] = ..., mem_pool_available_size: _Optional[int] = ..., uptime_seconds: _Optional[int] = ..., worker_id: _Optional[str] = ...) -> None: ...

class VerificationRequest(_message.Message):
    __slots__ = ("model_identifier", "replica_uuid", "timeout_ms")
    MODEL_IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    REPLICA_UUID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    model_identifier: str
    replica_uuid: str
    timeout_ms: int
    def __init__(self, model_identifier: _Optional[str] = ..., replica_uuid: _Optional[str] = ..., timeout_ms: _Optional[int] = ...) -> None: ...

class VerificationResponse(_message.Message):
    __slots__ = ("status", "err_msg")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERR_MSG_FIELD_NUMBER: _ClassVar[int]
    status: VerificationStatus
    err_msg: str
    def __init__(self, status: _Optional[_Union[VerificationStatus, str]] = ..., err_msg: _Optional[str] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_identifier", "model_size_bytes", "location", "loaded_timestamp", "last_access_timestamp", "replica_uuids", "is_registered_for_comm")
    MODEL_IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MODEL_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    LOADED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LAST_ACCESS_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    REPLICA_UUIDS_FIELD_NUMBER: _ClassVar[int]
    IS_REGISTERED_FOR_COMM_FIELD_NUMBER: _ClassVar[int]
    model_identifier: str
    model_size_bytes: int
    location: ModelLocation
    loaded_timestamp: int
    last_access_timestamp: int
    replica_uuids: _containers.RepeatedScalarFieldContainer[str]
    is_registered_for_comm: bool
    def __init__(self, model_identifier: _Optional[str] = ..., model_size_bytes: _Optional[int] = ..., location: _Optional[_Union[ModelLocation, str]] = ..., loaded_timestamp: _Optional[int] = ..., last_access_timestamp: _Optional[int] = ..., replica_uuids: _Optional[_Iterable[str]] = ..., is_registered_for_comm: bool = ...) -> None: ...

class GpuDeviceInfo(_message.Message):
    __slots__ = ("device_id", "device_uuid", "total_memory_bytes", "free_memory_bytes", "used_memory_bytes", "loaded_models")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_UUID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    FREE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    USED_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    LOADED_MODELS_FIELD_NUMBER: _ClassVar[int]
    device_id: int
    device_uuid: str
    total_memory_bytes: int
    free_memory_bytes: int
    used_memory_bytes: int
    loaded_models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, device_id: _Optional[int] = ..., device_uuid: _Optional[str] = ..., total_memory_bytes: _Optional[int] = ..., free_memory_bytes: _Optional[int] = ..., used_memory_bytes: _Optional[int] = ..., loaded_models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class MemoryPoolInfo(_message.Message):
    __slots__ = ("total_size_bytes", "available_bytes", "allocated_bytes", "allocated_chunks_count", "chunk_size_bytes")
    TOTAL_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_BYTES_FIELD_NUMBER: _ClassVar[int]
    ALLOCATED_BYTES_FIELD_NUMBER: _ClassVar[int]
    ALLOCATED_CHUNKS_COUNT_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    total_size_bytes: int
    available_bytes: int
    allocated_bytes: int
    allocated_chunks_count: int
    chunk_size_bytes: int
    def __init__(self, total_size_bytes: _Optional[int] = ..., available_bytes: _Optional[int] = ..., allocated_bytes: _Optional[int] = ..., allocated_chunks_count: _Optional[int] = ..., chunk_size_bytes: _Optional[int] = ...) -> None: ...

class CommunicationInfo(_message.Message):
    __slots__ = ("enabled", "total_transfers", "total_bytes_transferred", "total_transfer_errors")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TRANSFERS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_TRANSFERRED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TRANSFER_ERRORS_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    total_transfers: int
    total_bytes_transferred: int
    total_transfer_errors: int
    def __init__(self, enabled: bool = ..., total_transfers: _Optional[int] = ..., total_bytes_transferred: _Optional[int] = ..., total_transfer_errors: _Optional[int] = ...) -> None: ...

class GetDetailedStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetDetailedStatusResponse(_message.Message):
    __slots__ = ("is_registered", "is_healthy", "is_shutting_down", "uptime_seconds", "worker_id", "memory_pool_info", "gpu_devices", "cpu_models", "communication_info", "total_models_loaded", "total_model_size_bytes", "storage_path", "num_worker_threads")
    IS_REGISTERED_FIELD_NUMBER: _ClassVar[int]
    IS_HEALTHY_FIELD_NUMBER: _ClassVar[int]
    IS_SHUTTING_DOWN_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    MEMORY_POOL_INFO_FIELD_NUMBER: _ClassVar[int]
    GPU_DEVICES_FIELD_NUMBER: _ClassVar[int]
    CPU_MODELS_FIELD_NUMBER: _ClassVar[int]
    COMMUNICATION_INFO_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MODELS_LOADED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MODEL_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    STORAGE_PATH_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKER_THREADS_FIELD_NUMBER: _ClassVar[int]
    is_registered: bool
    is_healthy: bool
    is_shutting_down: bool
    uptime_seconds: int
    worker_id: str
    memory_pool_info: MemoryPoolInfo
    gpu_devices: _containers.RepeatedCompositeFieldContainer[GpuDeviceInfo]
    cpu_models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    communication_info: CommunicationInfo
    total_models_loaded: int
    total_model_size_bytes: int
    storage_path: str
    num_worker_threads: int
    def __init__(self, is_registered: bool = ..., is_healthy: bool = ..., is_shutting_down: bool = ..., uptime_seconds: _Optional[int] = ..., worker_id: _Optional[str] = ..., memory_pool_info: _Optional[_Union[MemoryPoolInfo, _Mapping]] = ..., gpu_devices: _Optional[_Iterable[_Union[GpuDeviceInfo, _Mapping]]] = ..., cpu_models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ..., communication_info: _Optional[_Union[CommunicationInfo, _Mapping]] = ..., total_models_loaded: _Optional[int] = ..., total_model_size_bytes: _Optional[int] = ..., storage_path: _Optional[str] = ..., num_worker_threads: _Optional[int] = ...) -> None: ...

class LoadedModelInfo(_message.Message):
    __slots__ = ("model_id", "version", "device_id", "ref_count", "pids", "size_bytes", "keep_for_global", "last_access_timestamp")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    REF_COUNT_FIELD_NUMBER: _ClassVar[int]
    PIDS_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    KEEP_FOR_GLOBAL_FIELD_NUMBER: _ClassVar[int]
    LAST_ACCESS_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    version: str
    device_id: int
    ref_count: int
    pids: _containers.RepeatedScalarFieldContainer[int]
    size_bytes: int
    keep_for_global: bool
    last_access_timestamp: int
    def __init__(self, model_id: _Optional[str] = ..., version: _Optional[str] = ..., device_id: _Optional[int] = ..., ref_count: _Optional[int] = ..., pids: _Optional[_Iterable[int]] = ..., size_bytes: _Optional[int] = ..., keep_for_global: bool = ..., last_access_timestamp: _Optional[int] = ...) -> None: ...

class GetLoadedModelsRequest(_message.Message):
    __slots__ = ("model_id_filter", "device_id_filter")
    MODEL_ID_FILTER_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FILTER_FIELD_NUMBER: _ClassVar[int]
    model_id_filter: str
    device_id_filter: int
    def __init__(self, model_id_filter: _Optional[str] = ..., device_id_filter: _Optional[int] = ...) -> None: ...

class GetLoadedModelsResponse(_message.Message):
    __slots__ = ("models", "total_models", "total_size_bytes")
    MODELS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MODELS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[LoadedModelInfo]
    total_models: int
    total_size_bytes: int
    def __init__(self, models: _Optional[_Iterable[_Union[LoadedModelInfo, _Mapping]]] = ..., total_models: _Optional[int] = ..., total_size_bytes: _Optional[int] = ...) -> None: ...

class LockChunksRequest(_message.Message):
    __slots__ = ("model_id", "chunk_indices")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDICES_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    chunk_indices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, model_id: _Optional[str] = ..., chunk_indices: _Optional[_Iterable[int]] = ...) -> None: ...

class LockChunksResponse(_message.Message):
    __slots__ = ("lock_token",)
    LOCK_TOKEN_FIELD_NUMBER: _ClassVar[int]
    lock_token: str
    def __init__(self, lock_token: _Optional[str] = ...) -> None: ...

class UnlockChunksRequest(_message.Message):
    __slots__ = ("lock_token",)
    LOCK_TOKEN_FIELD_NUMBER: _ClassVar[int]
    lock_token: str
    def __init__(self, lock_token: _Optional[str] = ...) -> None: ...

class UnlockChunksResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BeginRegisterTensorDictRequest(_message.Message):
    __slots__ = ("model_id", "device_id", "total_size", "enable_p2p", "ttl_ms", "tensor_index_key", "tensor_index_data")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    ENABLE_P2P_FIELD_NUMBER: _ClassVar[int]
    TTL_MS_FIELD_NUMBER: _ClassVar[int]
    TENSOR_INDEX_KEY_FIELD_NUMBER: _ClassVar[int]
    TENSOR_INDEX_DATA_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    device_id: int
    total_size: int
    enable_p2p: bool
    ttl_ms: int
    tensor_index_key: str
    tensor_index_data: TensorIndexData
    def __init__(self, model_id: _Optional[str] = ..., device_id: _Optional[int] = ..., total_size: _Optional[int] = ..., enable_p2p: bool = ..., ttl_ms: _Optional[int] = ..., tensor_index_key: _Optional[str] = ..., tensor_index_data: _Optional[_Union[TensorIndexData, _Mapping]] = ...) -> None: ...

class TensorIndexData(_message.Message):
    __slots__ = ("data", "schema_version", "encoding")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    schema_version: str
    encoding: str
    def __init__(self, data: _Optional[bytes] = ..., schema_version: _Optional[str] = ..., encoding: _Optional[str] = ...) -> None: ...

class BeginRegisterTensorDictResponse(_message.Message):
    __slots__ = ("registration_id", "daemon_ipc_handle", "device_id", "size")
    REGISTRATION_ID_FIELD_NUMBER: _ClassVar[int]
    DAEMON_IPC_HANDLE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    registration_id: str
    daemon_ipc_handle: bytes
    device_id: int
    size: int
    def __init__(self, registration_id: _Optional[str] = ..., daemon_ipc_handle: _Optional[bytes] = ..., device_id: _Optional[int] = ..., size: _Optional[int] = ...) -> None: ...

class CommitRegisteredTensorDictRequest(_message.Message):
    __slots__ = ("registration_id",)
    REGISTRATION_ID_FIELD_NUMBER: _ClassVar[int]
    registration_id: str
    def __init__(self, registration_id: _Optional[str] = ...) -> None: ...

class CommitRegisteredTensorDictResponse(_message.Message):
    __slots__ = ("registration_id", "model_id", "device_id", "size")
    REGISTRATION_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    registration_id: str
    model_id: str
    device_id: int
    size: int
    def __init__(self, registration_id: _Optional[str] = ..., model_id: _Optional[str] = ..., device_id: _Optional[int] = ..., size: _Optional[int] = ...) -> None: ...

class AbortRegisteredTensorDictRequest(_message.Message):
    __slots__ = ("registration_id",)
    REGISTRATION_ID_FIELD_NUMBER: _ClassVar[int]
    registration_id: str
    def __init__(self, registration_id: _Optional[str] = ...) -> None: ...

class AbortRegisteredTensorDictResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...
