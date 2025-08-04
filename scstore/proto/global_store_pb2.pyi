from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OK: _ClassVar[Status]
    NOT_FOUND: _ClassVar[Status]
    TIMED_OUT: _ClassVar[Status]
    TOO_MANY_REQUESTS: _ClassVar[Status]
    STATE_SYNC_REQUIRED: _ClassVar[Status]
    ERROR: _ClassVar[Status]

class ConnectionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONNECTED: _ClassVar[ConnectionStatus]
    RECONNECTING: _ClassVar[ConnectionStatus]
    DISCONNECTED: _ClassVar[ConnectionStatus]

class MemoryType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GPU: _ClassVar[MemoryType]
    RAM: _ClassVar[MemoryType]
    DISK: _ClassVar[MemoryType]

class ChunkState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHUNK_HOT: _ClassVar[ChunkState]
    CHUNK_LOCKED_TX: _ClassVar[ChunkState]
    CHUNK_COPIED_GPU: _ClassVar[ChunkState]
    CHUNK_COLD: _ClassVar[ChunkState]
    CHUNK_EVICTED: _ClassVar[ChunkState]
OK: Status
NOT_FOUND: Status
TIMED_OUT: Status
TOO_MANY_REQUESTS: Status
STATE_SYNC_REQUIRED: Status
ERROR: Status
CONNECTED: ConnectionStatus
RECONNECTING: ConnectionStatus
DISCONNECTED: ConnectionStatus
GPU: MemoryType
RAM: MemoryType
DISK: MemoryType
CHUNK_HOT: ChunkState
CHUNK_LOCKED_TX: ChunkState
CHUNK_COPIED_GPU: ChunkState
CHUNK_COLD: ChunkState
CHUNK_EVICTED: ChunkState

class RegisterWorkerRequest(_message.Message):
    __slots__ = ("node_id", "node_address", "grpc_port", "p2p_port", "mem_pool_total_size", "mem_pool_available_size", "is_recovery_registration", "previous_worker_id")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    GRPC_PORT_FIELD_NUMBER: _ClassVar[int]
    P2P_PORT_FIELD_NUMBER: _ClassVar[int]
    MEM_POOL_TOTAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MEM_POOL_AVAILABLE_SIZE_FIELD_NUMBER: _ClassVar[int]
    IS_RECOVERY_REGISTRATION_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    node_address: str
    grpc_port: int
    p2p_port: int
    mem_pool_total_size: int
    mem_pool_available_size: int
    is_recovery_registration: bool
    previous_worker_id: str
    def __init__(self, node_id: _Optional[str] = ..., node_address: _Optional[str] = ..., grpc_port: _Optional[int] = ..., p2p_port: _Optional[int] = ..., mem_pool_total_size: _Optional[int] = ..., mem_pool_available_size: _Optional[int] = ..., is_recovery_registration: bool = ..., previous_worker_id: _Optional[str] = ...) -> None: ...

class RegisterWorkerResponse(_message.Message):
    __slots__ = ("status", "worker_id", "heartbeat_interval_ms", "state_sync_required", "expected_state_version")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    STATE_SYNC_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    status: Status
    worker_id: str
    heartbeat_interval_ms: int
    state_sync_required: bool
    expected_state_version: int
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., worker_id: _Optional[str] = ..., heartbeat_interval_ms: _Optional[int] = ..., state_sync_required: bool = ..., expected_state_version: _Optional[int] = ...) -> None: ...

class WorkerHeartbeatRequest(_message.Message):
    __slots__ = ("worker_id", "mem_pool_available_size", "accepting_new_requests", "state_version", "state_checksum", "registered_models", "last_successful_sync", "global_store_status")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    MEM_POOL_AVAILABLE_SIZE_FIELD_NUMBER: _ClassVar[int]
    ACCEPTING_NEW_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    STATE_CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_MODELS_FIELD_NUMBER: _ClassVar[int]
    LAST_SUCCESSFUL_SYNC_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_STORE_STATUS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    mem_pool_available_size: int
    accepting_new_requests: bool
    state_version: int
    state_checksum: str
    registered_models: _containers.RepeatedScalarFieldContainer[str]
    last_successful_sync: int
    global_store_status: ConnectionStatus
    def __init__(self, worker_id: _Optional[str] = ..., mem_pool_available_size: _Optional[int] = ..., accepting_new_requests: bool = ..., state_version: _Optional[int] = ..., state_checksum: _Optional[str] = ..., registered_models: _Optional[_Iterable[str]] = ..., last_successful_sync: _Optional[int] = ..., global_store_status: _Optional[_Union[ConnectionStatus, str]] = ...) -> None: ...

class WorkerHeartbeatResponse(_message.Message):
    __slots__ = ("status", "state_sync_required", "expected_state_version", "obsolete_models", "server_timestamp")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STATE_SYNC_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    OBSOLETE_MODELS_FIELD_NUMBER: _ClassVar[int]
    SERVER_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    status: Status
    state_sync_required: bool
    expected_state_version: int
    obsolete_models: _containers.RepeatedScalarFieldContainer[str]
    server_timestamp: int
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., state_sync_required: bool = ..., expected_state_version: _Optional[int] = ..., obsolete_models: _Optional[_Iterable[str]] = ..., server_timestamp: _Optional[int] = ...) -> None: ...

class UnregisterWorkerRequest(_message.Message):
    __slots__ = ("worker_id", "is_graceful_shutdown")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    IS_GRACEFUL_SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    is_graceful_shutdown: bool
    def __init__(self, worker_id: _Optional[str] = ..., is_graceful_shutdown: bool = ...) -> None: ...

class UnregisterWorkerResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: Status
    def __init__(self, status: _Optional[_Union[Status, str]] = ...) -> None: ...

class ListActiveWorkersRequest(_message.Message):
    __slots__ = ("include_unavailable",)
    INCLUDE_UNAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    include_unavailable: bool
    def __init__(self, include_unavailable: bool = ...) -> None: ...

class ListActiveWorkersResponse(_message.Message):
    __slots__ = ("workers",)
    class WorkerInfo(_message.Message):
        __slots__ = ("worker_id", "node_id", "node_address", "grpc_port", "p2p_port", "mem_pool_total_size", "mem_pool_available_size", "accepting_new_requests", "last_heartbeat_timestamp", "state_version", "status")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        NODE_ID_FIELD_NUMBER: _ClassVar[int]
        NODE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
        GRPC_PORT_FIELD_NUMBER: _ClassVar[int]
        P2P_PORT_FIELD_NUMBER: _ClassVar[int]
        MEM_POOL_TOTAL_SIZE_FIELD_NUMBER: _ClassVar[int]
        MEM_POOL_AVAILABLE_SIZE_FIELD_NUMBER: _ClassVar[int]
        ACCEPTING_NEW_REQUESTS_FIELD_NUMBER: _ClassVar[int]
        LAST_HEARTBEAT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
        STATUS_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        node_id: str
        node_address: str
        grpc_port: int
        p2p_port: int
        mem_pool_total_size: int
        mem_pool_available_size: int
        accepting_new_requests: bool
        last_heartbeat_timestamp: int
        state_version: int
        status: ConnectionStatus
        def __init__(self, worker_id: _Optional[str] = ..., node_id: _Optional[str] = ..., node_address: _Optional[str] = ..., grpc_port: _Optional[int] = ..., p2p_port: _Optional[int] = ..., mem_pool_total_size: _Optional[int] = ..., mem_pool_available_size: _Optional[int] = ..., accepting_new_requests: bool = ..., last_heartbeat_timestamp: _Optional[int] = ..., state_version: _Optional[int] = ..., status: _Optional[_Union[ConnectionStatus, str]] = ...) -> None: ...
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    workers: _containers.RepeatedCompositeFieldContainer[ListActiveWorkersResponse.WorkerInfo]
    def __init__(self, workers: _Optional[_Iterable[_Union[ListActiveWorkersResponse.WorkerInfo, _Mapping]]] = ...) -> None: ...

class WorkerLocalState(_message.Message):
    __slots__ = ("worker_id", "state_version", "state_checksum", "local_replicas", "last_update_timestamp")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    STATE_CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    LOCAL_REPLICAS_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    state_version: int
    state_checksum: str
    local_replicas: _containers.RepeatedCompositeFieldContainer[ModelReplicaInfo]
    last_update_timestamp: int
    def __init__(self, worker_id: _Optional[str] = ..., state_version: _Optional[int] = ..., state_checksum: _Optional[str] = ..., local_replicas: _Optional[_Iterable[_Union[ModelReplicaInfo, _Mapping]]] = ..., last_update_timestamp: _Optional[int] = ...) -> None: ...

class ModelReplicaInfo(_message.Message):
    __slots__ = ("model_name", "replica_id", "memory_info", "max_concurrency", "current_requests", "is_available", "registered_timestamp")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    MEMORY_INFO_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENCY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    IS_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    replica_id: str
    memory_info: MemoryInfo
    max_concurrency: int
    current_requests: int
    is_available: bool
    registered_timestamp: int
    def __init__(self, model_name: _Optional[str] = ..., replica_id: _Optional[str] = ..., memory_info: _Optional[_Union[MemoryInfo, _Mapping]] = ..., max_concurrency: _Optional[int] = ..., current_requests: _Optional[int] = ..., is_available: bool = ..., registered_timestamp: _Optional[int] = ...) -> None: ...

class SynchronizeWorkerStateRequest(_message.Message):
    __slots__ = ("worker_id", "local_state", "force_full_sync")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_STATE_FIELD_NUMBER: _ClassVar[int]
    FORCE_FULL_SYNC_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    local_state: WorkerLocalState
    force_full_sync: bool
    def __init__(self, worker_id: _Optional[str] = ..., local_state: _Optional[_Union[WorkerLocalState, _Mapping]] = ..., force_full_sync: bool = ...) -> None: ...

class SynchronizeWorkerStateResponse(_message.Message):
    __slots__ = ("status", "new_state_version", "state_changes", "new_state_checksum")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NEW_STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    STATE_CHANGES_FIELD_NUMBER: _ClassVar[int]
    NEW_STATE_CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    status: Status
    new_state_version: int
    state_changes: _containers.RepeatedCompositeFieldContainer[StateChange]
    new_state_checksum: str
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., new_state_version: _Optional[int] = ..., state_changes: _Optional[_Iterable[_Union[StateChange, _Mapping]]] = ..., new_state_checksum: _Optional[str] = ...) -> None: ...

class StateChange(_message.Message):
    __slots__ = ("type", "replica_info", "reason")
    class ChangeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ADD_REPLICA: _ClassVar[StateChange.ChangeType]
        REMOVE_REPLICA: _ClassVar[StateChange.ChangeType]
        UPDATE_REPLICA: _ClassVar[StateChange.ChangeType]
    ADD_REPLICA: StateChange.ChangeType
    REMOVE_REPLICA: StateChange.ChangeType
    UPDATE_REPLICA: StateChange.ChangeType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REPLICA_INFO_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    type: StateChange.ChangeType
    replica_info: ModelReplicaInfo
    reason: str
    def __init__(self, type: _Optional[_Union[StateChange.ChangeType, str]] = ..., replica_info: _Optional[_Union[ModelReplicaInfo, _Mapping]] = ..., reason: _Optional[str] = ...) -> None: ...

class RequestFullStateSyncRequest(_message.Message):
    __slots__ = ("worker_id", "current_state_version")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    current_state_version: int
    def __init__(self, worker_id: _Optional[str] = ..., current_state_version: _Optional[int] = ...) -> None: ...

class RequestFullStateSyncResponse(_message.Message):
    __slots__ = ("status", "new_state_version", "expected_replicas", "new_state_checksum")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NEW_STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_REPLICAS_FIELD_NUMBER: _ClassVar[int]
    NEW_STATE_CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    status: Status
    new_state_version: int
    expected_replicas: _containers.RepeatedCompositeFieldContainer[ModelReplicaInfo]
    new_state_checksum: str
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., new_state_version: _Optional[int] = ..., expected_replicas: _Optional[_Iterable[_Union[ModelReplicaInfo, _Mapping]]] = ..., new_state_checksum: _Optional[str] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_name", "available_replicas")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_REPLICAS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    available_replicas: _containers.RepeatedCompositeFieldContainer[MemoryInfo]
    def __init__(self, model_name: _Optional[str] = ..., available_replicas: _Optional[_Iterable[_Union[MemoryInfo, _Mapping]]] = ...) -> None: ...

class GetModelInfoRequest(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...

class GetModelInfoResponse(_message.Message):
    __slots__ = ("status", "model_info")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
    status: Status
    model_info: ModelInfo
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., model_info: _Optional[_Union[ModelInfo, _Mapping]] = ...) -> None: ...

class ListModelReplicasRequest(_message.Message):
    __slots__ = ("model_name", "node_id", "node_address", "node_port", "memory_type", "device_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    NODE_PORT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    node_id: str
    node_address: str
    node_port: str
    memory_type: MemoryType
    device_id: int
    def __init__(self, model_name: _Optional[str] = ..., node_id: _Optional[str] = ..., node_address: _Optional[str] = ..., node_port: _Optional[str] = ..., memory_type: _Optional[_Union[MemoryType, str]] = ..., device_id: _Optional[int] = ...) -> None: ...

class ListModelReplicasResponse(_message.Message):
    __slots__ = ("model_replicas",)
    class ModelReplicasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: MemoryInfoList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[MemoryInfoList, _Mapping]] = ...) -> None: ...
    MODEL_REPLICAS_FIELD_NUMBER: _ClassVar[int]
    model_replicas: _containers.MessageMap[str, MemoryInfoList]
    def __init__(self, model_replicas: _Optional[_Mapping[str, MemoryInfoList]] = ...) -> None: ...

class RegisterModelReplicaRequest(_message.Message):
    __slots__ = ("model_name", "mem_info", "max_concurrency", "worker_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MEM_INFO_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENCY_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    mem_info: MemoryInfo
    max_concurrency: int
    worker_id: str
    def __init__(self, model_name: _Optional[str] = ..., mem_info: _Optional[_Union[MemoryInfo, _Mapping]] = ..., max_concurrency: _Optional[int] = ..., worker_id: _Optional[str] = ...) -> None: ...

class RegisterModelReplicaResponse(_message.Message):
    __slots__ = ("status", "model_name", "replica_id")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    status: Status
    model_name: str
    replica_id: str
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., model_name: _Optional[str] = ..., replica_id: _Optional[str] = ...) -> None: ...

class UpdateModelReplicaRequest(_message.Message):
    __slots__ = ("model_name", "replica_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    replica_id: str
    def __init__(self, model_name: _Optional[str] = ..., replica_id: _Optional[str] = ...) -> None: ...

class UpdateModelReplicaResponse(_message.Message):
    __slots__ = ("status", "model_name", "replica_id")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    status: Status
    model_name: str
    replica_id: str
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., model_name: _Optional[str] = ..., replica_id: _Optional[str] = ...) -> None: ...

class UnregisterModelReplicaRequest(_message.Message):
    __slots__ = ("model_name", "replica_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    replica_id: str
    def __init__(self, model_name: _Optional[str] = ..., replica_id: _Optional[str] = ...) -> None: ...

class UnregisterModelReplicaResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: Status
    def __init__(self, status: _Optional[_Union[Status, str]] = ...) -> None: ...

class MemoryInfo(_message.Message):
    __slots__ = ("node_id", "node_address", "node_port", "memory_size", "memory_type", "device_id", "remote_memory_keys", "buffer_sizes")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    NODE_PORT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_SIZE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    REMOTE_MEMORY_KEYS_FIELD_NUMBER: _ClassVar[int]
    BUFFER_SIZES_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    node_address: str
    node_port: int
    memory_size: int
    memory_type: MemoryType
    device_id: int
    remote_memory_keys: _containers.RepeatedScalarFieldContainer[str]
    buffer_sizes: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, node_id: _Optional[str] = ..., node_address: _Optional[str] = ..., node_port: _Optional[int] = ..., memory_size: _Optional[int] = ..., memory_type: _Optional[_Union[MemoryType, str]] = ..., device_id: _Optional[int] = ..., remote_memory_keys: _Optional[_Iterable[str]] = ..., buffer_sizes: _Optional[_Iterable[int]] = ...) -> None: ...

class MemoryInfoList(_message.Message):
    __slots__ = ("list",)
    LIST_FIELD_NUMBER: _ClassVar[int]
    list: _containers.RepeatedCompositeFieldContainer[MemoryInfo]
    def __init__(self, list: _Optional[_Iterable[_Union[MemoryInfo, _Mapping]]] = ...) -> None: ...

class RequestModelReplicaTransportRequest(_message.Message):
    __slots__ = ("model_name", "local_memory_info", "wait_timeout_ms", "source_node_id", "source_address", "source_port")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCAL_MEMORY_INFO_FIELD_NUMBER: _ClassVar[int]
    WAIT_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PORT_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    local_memory_info: MemoryInfo
    wait_timeout_ms: int
    source_node_id: str
    source_address: str
    source_port: int
    def __init__(self, model_name: _Optional[str] = ..., local_memory_info: _Optional[_Union[MemoryInfo, _Mapping]] = ..., wait_timeout_ms: _Optional[int] = ..., source_node_id: _Optional[str] = ..., source_address: _Optional[str] = ..., source_port: _Optional[int] = ...) -> None: ...

class RequestModelReplicaTransportResponse(_message.Message):
    __slots__ = ("status", "remote_memory_info", "transport_id")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REMOTE_MEMORY_INFO_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_ID_FIELD_NUMBER: _ClassVar[int]
    status: Status
    remote_memory_info: MemoryInfo
    transport_id: str
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., remote_memory_info: _Optional[_Union[MemoryInfo, _Mapping]] = ..., transport_id: _Optional[str] = ...) -> None: ...

class CompleteModelReplicaTransportRequest(_message.Message):
    __slots__ = ("transport_id",)
    TRANSPORT_ID_FIELD_NUMBER: _ClassVar[int]
    transport_id: str
    def __init__(self, transport_id: _Optional[str] = ...) -> None: ...

class CompleteModelReplicaTransportResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: Status
    def __init__(self, status: _Optional[_Union[Status, str]] = ...) -> None: ...

class GetModelReplicaRequest(_message.Message):
    __slots__ = ("model_name", "local_memory_info", "replica_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCAL_MEMORY_INFO_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    local_memory_info: MemoryInfo
    replica_id: str
    def __init__(self, model_name: _Optional[str] = ..., local_memory_info: _Optional[_Union[MemoryInfo, _Mapping]] = ..., replica_id: _Optional[str] = ...) -> None: ...

class GetModelReplicaResponse(_message.Message):
    __slots__ = ("status", "remote_memory_info", "replica_id")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REMOTE_MEMORY_INFO_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    status: Status
    remote_memory_info: MemoryInfo
    replica_id: str
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., remote_memory_info: _Optional[_Union[MemoryInfo, _Mapping]] = ..., replica_id: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: Status
    def __init__(self, status: _Optional[_Union[Status, str]] = ...) -> None: ...

class QueryChunkLocationsRequest(_message.Message):
    __slots__ = ("model_id", "chunk_indices")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDICES_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    chunk_indices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, model_id: _Optional[str] = ..., chunk_indices: _Optional[_Iterable[int]] = ...) -> None: ...

class ChunkLocation(_message.Message):
    __slots__ = ("chunk_idx", "node_id", "node_address", "p2p_port", "state", "node_load_ratio", "device_uuid", "replica")
    CHUNK_IDX_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    P2P_PORT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    NODE_LOAD_RATIO_FIELD_NUMBER: _ClassVar[int]
    DEVICE_UUID_FIELD_NUMBER: _ClassVar[int]
    REPLICA_FIELD_NUMBER: _ClassVar[int]
    chunk_idx: int
    node_id: str
    node_address: str
    p2p_port: int
    state: ChunkState
    node_load_ratio: float
    device_uuid: str
    replica: int
    def __init__(self, chunk_idx: _Optional[int] = ..., node_id: _Optional[str] = ..., node_address: _Optional[str] = ..., p2p_port: _Optional[int] = ..., state: _Optional[_Union[ChunkState, str]] = ..., node_load_ratio: _Optional[float] = ..., device_uuid: _Optional[str] = ..., replica: _Optional[int] = ...) -> None: ...

class QueryChunkLocationsResponse(_message.Message):
    __slots__ = ("status", "locations")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    status: Status
    locations: _containers.RepeatedCompositeFieldContainer[ChunkLocation]
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., locations: _Optional[_Iterable[_Union[ChunkLocation, _Mapping]]] = ...) -> None: ...

class ChunkStateUpdate(_message.Message):
    __slots__ = ("model_id", "chunk_idx", "state", "device_uuid", "replica")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_IDX_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_UUID_FIELD_NUMBER: _ClassVar[int]
    REPLICA_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    chunk_idx: int
    state: ChunkState
    device_uuid: str
    replica: int
    def __init__(self, model_id: _Optional[str] = ..., chunk_idx: _Optional[int] = ..., state: _Optional[_Union[ChunkState, str]] = ..., device_uuid: _Optional[str] = ..., replica: _Optional[int] = ...) -> None: ...

class BatchUpdateChunkStatesRequest(_message.Message):
    __slots__ = ("worker_id", "node_id", "updates")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    UPDATES_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    node_id: str
    updates: _containers.RepeatedCompositeFieldContainer[ChunkStateUpdate]
    def __init__(self, worker_id: _Optional[str] = ..., node_id: _Optional[str] = ..., updates: _Optional[_Iterable[_Union[ChunkStateUpdate, _Mapping]]] = ...) -> None: ...

class BatchUpdateChunkStatesResponse(_message.Message):
    __slots__ = ("status", "updates_applied")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPDATES_APPLIED_FIELD_NUMBER: _ClassVar[int]
    status: Status
    updates_applied: int
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., updates_applied: _Optional[int] = ...) -> None: ...
