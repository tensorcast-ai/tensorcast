from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StagerConfig(_message.Message):
    __slots__ = (
        "stage_cpu_for_rdma",
        "stage_chunk_mb_cpu",
        "stage_chunk_mb_gpu",
        "buffers_per_flow",
    )
    STAGE_CPU_FOR_RDMA_FIELD_NUMBER: _ClassVar[int]
    STAGE_CHUNK_MB_CPU_FIELD_NUMBER: _ClassVar[int]
    STAGE_CHUNK_MB_GPU_FIELD_NUMBER: _ClassVar[int]
    BUFFERS_PER_FLOW_FIELD_NUMBER: _ClassVar[int]
    stage_cpu_for_rdma: bool
    stage_chunk_mb_cpu: int
    stage_chunk_mb_gpu: int
    buffers_per_flow: int
    def __init__(
        self,
        stage_cpu_for_rdma: bool = ...,
        stage_chunk_mb_cpu: _Optional[int] = ...,
        stage_chunk_mb_gpu: _Optional[int] = ...,
        buffers_per_flow: _Optional[int] = ...,
    ) -> None: ...

class RdmaConfig(_message.Message):
    __slots__ = ("outstanding_wr", "ack_ttl_ms")
    OUTSTANDING_WR_FIELD_NUMBER: _ClassVar[int]
    ACK_TTL_MS_FIELD_NUMBER: _ClassVar[int]
    outstanding_wr: int
    ack_ttl_ms: int
    def __init__(
        self, outstanding_wr: _Optional[int] = ..., ack_ttl_ms: _Optional[int] = ...
    ) -> None: ...

class PoolConfig(_message.Message):
    __slots__ = ("preregister_mr", "pool_size_bytes", "chunk_bytes")
    PREREGISTER_MR_FIELD_NUMBER: _ClassVar[int]
    POOL_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    CHUNK_BYTES_FIELD_NUMBER: _ClassVar[int]
    preregister_mr: bool
    pool_size_bytes: int
    chunk_bytes: int
    def __init__(
        self,
        preregister_mr: bool = ...,
        pool_size_bytes: _Optional[int] = ...,
        chunk_bytes: _Optional[int] = ...,
    ) -> None: ...

class TransportConfig(_message.Message):
    __slots__ = ("tcp_conn_count",)
    TCP_CONN_COUNT_FIELD_NUMBER: _ClassVar[int]
    tcp_conn_count: int
    def __init__(self, tcp_conn_count: _Optional[int] = ...) -> None: ...

class AffinityConfig(_message.Message):
    __slots__ = ("enable",)
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    enable: bool
    def __init__(self, enable: bool = ...) -> None: ...

class SimpleNumaNode(_message.Message):
    __slots__ = ("id", "nics", "gpus", "is_default")
    ID_FIELD_NUMBER: _ClassVar[int]
    NICS_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    IS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    id: int
    nics: _containers.RepeatedScalarFieldContainer[str]
    gpus: _containers.RepeatedScalarFieldContainer[int]
    is_default: bool
    def __init__(
        self,
        id: _Optional[int] = ...,
        nics: _Optional[_Iterable[str]] = ...,
        gpus: _Optional[_Iterable[int]] = ...,
        is_default: bool = ...,
    ) -> None: ...

class SimpleNumaConfig(_message.Message):
    __slots__ = ("enable", "nodes")
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    enable: bool
    nodes: _containers.RepeatedCompositeFieldContainer[SimpleNumaNode]
    def __init__(
        self,
        enable: bool = ...,
        nodes: _Optional[_Iterable[_Union[SimpleNumaNode, _Mapping]]] = ...,
    ) -> None: ...

class CommunicatorConfig(_message.Message):
    __slots__ = (
        "enable_rdma",
        "stager",
        "rdma",
        "pool",
        "transport",
        "affinity",
        "simple_numa",
    )
    ENABLE_RDMA_FIELD_NUMBER: _ClassVar[int]
    STAGER_FIELD_NUMBER: _ClassVar[int]
    RDMA_FIELD_NUMBER: _ClassVar[int]
    POOL_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_FIELD_NUMBER: _ClassVar[int]
    AFFINITY_FIELD_NUMBER: _ClassVar[int]
    SIMPLE_NUMA_FIELD_NUMBER: _ClassVar[int]
    enable_rdma: bool
    stager: StagerConfig
    rdma: RdmaConfig
    pool: PoolConfig
    transport: TransportConfig
    affinity: AffinityConfig
    simple_numa: SimpleNumaConfig
    def __init__(
        self,
        enable_rdma: bool = ...,
        stager: _Optional[_Union[StagerConfig, _Mapping]] = ...,
        rdma: _Optional[_Union[RdmaConfig, _Mapping]] = ...,
        pool: _Optional[_Union[PoolConfig, _Mapping]] = ...,
        transport: _Optional[_Union[TransportConfig, _Mapping]] = ...,
        affinity: _Optional[_Union[AffinityConfig, _Mapping]] = ...,
        simple_numa: _Optional[_Union[SimpleNumaConfig, _Mapping]] = ...,
    ) -> None: ...
