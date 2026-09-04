#  Copyright (c) 2025-2026, TensorCast Team.
"""Public contracts for process-scoped region-backed artifact sessions."""

from __future__ import annotations

import ctypes
import ipaddress
import logging
import mmap
import os
import posixpath
import re
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Protocol

import grpc
import torch
from google.protobuf.message import Message
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from tensorcast.common.identity import build_byte_artifact_cgid
from tensorcast.common.selection_contract import build_artifact_selection
from tensorcast.common.selection_identity import (
    compute_byte_artifact_logical_layout_hash,
)
from tensorcast.daemon_ctl import get_daemon_client
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    HostSharedRegionAttachment,
    HostSharedRegionClass,
    LocalRegionHandle,
    RegionMemoryKind,
    ServerConfig,
)

logger = logging.getLogger(__name__)


class RegionTransferMode(str, Enum):
    """Memory ownership mode used for region-backed transfers."""

    SCRATCH = "scratch"
    ALLOCATOR = "allocator"


class RegionSessionHealth(str, Enum):
    """RPC health of an attached process session."""

    READY = "ready"
    FAILED = "failed"


class RegionSessionLifecycleState(str, Enum):
    """Resource lifecycle state of a process session."""

    ATTACHED = "attached"
    TERMINATED = "terminated"


class RegionSessionFailureCode(str, Enum):
    """Stable SDK classification for a fatal session failure."""

    TRANSPORT = "transport"
    DAEMON_STATUS = "daemon_status"
    REGION_SETUP = "region_setup"
    REGION_LOST = "region_lost"
    MALFORMED_RESPONSE = "malformed_response"
    INTERNAL = "internal"


class RegionSessionOperationKind(str, Enum):
    """Public operation category recorded with a fatal failure."""

    ALLOCATE = "allocate"
    EXISTS = "exists"
    GET_INTO = "get_into"
    PUT_FROM = "put_from"


class RegionArtifactInputError(ValueError):
    """Raised when an artifact transfer input violates the public contract."""


class RegionSessionAttachError(RuntimeError):
    """Raised when a process session cannot attach to its daemon."""


class RegionSessionTerminatedError(RuntimeError):
    """Raised when an operation is attempted after terminal shutdown."""


class _FrozenPublicModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _require_non_empty(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class ByteArtifactKeyspace(_FrozenPublicModel):
    """Caller vocabulary used to derive canonical byte-artifact identities."""

    namespace: str
    engine: str
    model_id: str
    model_version: str
    layout_id: str

    @field_validator(
        "namespace",
        "engine",
        "model_id",
        "model_version",
        "layout_id",
    )
    @classmethod
    def _validate_identity_field(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty(value, field_name=info.field_name)


class ScratchTransferOptions(_FrozenPublicModel):
    """Fixed-capacity SDK scratch arenas used for transfer staging."""

    mode: Literal[RegionTransferMode.SCRATCH] = RegionTransferMode.SCRATCH
    capacity_bytes: int = Field(gt=0)


class AllocatorTransferOptions(_FrozenPublicModel):
    """Daemon-backed allocations used directly as transfer buffers."""

    mode: Literal[RegionTransferMode.ALLOCATOR] = RegionTransferMode.ALLOCATOR


class RegionBackedArtifactSessionOptions(_FrozenPublicModel):
    """Immutable configuration for one process-scoped daemon attachment."""

    daemon_address: str
    session_name: str
    transfer: Annotated[
        ScratchTransferOptions | AllocatorTransferOptions,
        Field(discriminator="mode"),
    ]
    transfer_timeout_s: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    exists_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        allow_inf_nan=False,
    )
    region_name_prefix: str = "tensorcast_region_artifact"

    @field_validator("daemon_address", "session_name", "region_name_prefix")
    @classmethod
    def _validate_non_empty_field(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty(value, field_name=info.field_name)


class ByteArtifactSpec(_FrozenPublicModel):
    """One immutable byte artifact in a caller-owned keyspace."""

    keyspace: ByteArtifactKeyspace
    engine_key: bytes
    byte_length: int = Field(gt=0)

    @field_validator("engine_key")
    @classmethod
    def _validate_engine_key(cls, value: bytes) -> bytes:
        if not value:
            raise ValueError("engine_key must not be empty")
        return value


_MAX_POINTER_VALUE = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1


class HostMemorySpan:
    """An owned, contiguous host-memory range borrowed by a synchronous call."""

    __slots__ = ("_address", "_byte_length", "_owner")

    def __new__(cls) -> HostMemorySpan:
        raise TypeError("use HostMemorySpan.from_tensor() or from_address()")

    @classmethod
    def _from_validated_address(
        cls,
        *,
        address: int,
        byte_length: int,
        owner: object,
    ) -> HostMemorySpan:
        span = object.__new__(cls)
        span._address = address
        span._byte_length = byte_length
        span._owner = owner
        return span

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        offset_bytes: int,
        byte_length: int,
    ) -> HostMemorySpan:
        """Retain a contiguous CPU tensor and borrow a byte range from it."""
        if tensor.device.type != "cpu":
            raise RegionArtifactInputError("tensor must reside on CPU memory")
        if not tensor.is_contiguous():
            raise RegionArtifactInputError("tensor must be contiguous")
        if isinstance(offset_bytes, bool) or not isinstance(offset_bytes, int):
            raise RegionArtifactInputError("offset_bytes must be an integer")
        if isinstance(byte_length, bool) or not isinstance(byte_length, int):
            raise RegionArtifactInputError("byte_length must be an integer")
        if offset_bytes < 0:
            raise RegionArtifactInputError("offset_bytes must be non-negative")
        if byte_length <= 0:
            raise RegionArtifactInputError("byte_length must be positive")

        tensor_byte_length = int(tensor.numel()) * int(tensor.element_size())
        if offset_bytes > tensor_byte_length - byte_length:
            raise RegionArtifactInputError("tensor byte range is out of bounds")
        return cls.from_address(
            int(tensor.data_ptr()) + offset_bytes,
            byte_length,
            owner=tensor,
        )

    @classmethod
    def from_address(
        cls,
        address: int,
        byte_length: int,
        *,
        owner: object,
    ) -> HostMemorySpan:
        """Retain an explicit owner and borrow one of its host-memory ranges."""
        if isinstance(address, bool) or not isinstance(address, int):
            raise RegionArtifactInputError("address must be an integer")
        if isinstance(byte_length, bool) or not isinstance(byte_length, int):
            raise RegionArtifactInputError("byte_length must be an integer")
        if address <= 0:
            raise RegionArtifactInputError("address must be positive")
        if byte_length <= 0:
            raise RegionArtifactInputError("byte_length must be positive")
        if owner is None:
            raise RegionArtifactInputError("owner must not be None")
        if address > _MAX_POINTER_VALUE:
            raise RegionArtifactInputError("address exceeds the process pointer range")
        if byte_length > _MAX_POINTER_VALUE + 1 - address:
            raise RegionArtifactInputError("address range overflows the pointer range")
        return cls._from_validated_address(
            address=address,
            byte_length=byte_length,
            owner=owner,
        )

    @property
    def address(self) -> int:
        """Start address of the borrowed range."""
        return self._address

    @property
    def byte_length(self) -> int:
        """Length of the borrowed range in bytes."""
        return self._byte_length


@dataclass(frozen=True, slots=True)
class RegionArtifactTransfer:
    """Pair one byte artifact with an equally sized owned memory span."""

    artifact: ByteArtifactSpec
    span: HostMemorySpan

    def __post_init__(self) -> None:
        if self.artifact.byte_length != self.span.byte_length:
            raise RegionArtifactInputError(
                "artifact byte_length must equal span byte_length"
            )


class RegionSessionFailure(_FrozenPublicModel):
    """Stable first-failure record retained by a failed process session."""

    code: RegionSessionFailureCode
    message: str
    operation_kind: RegionSessionOperationKind
    operation_id: str | None
    occurred_at: datetime

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        return _require_non_empty(value, field_name="message")

    @field_validator("operation_id")
    @classmethod
    def _validate_operation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, field_name="operation_id")

    @field_validator("occurred_at")
    @classmethod
    def _normalize_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value.astimezone(timezone.utc)


class RegionSessionFailedError(RuntimeError):
    """Raised for the stable fatal error latched by a process session."""

    failure: RegionSessionFailure

    def __init__(self, failure: RegionSessionFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


class RegionArtifactExistsResult(_FrozenPublicModel):
    """Caller-ordered existence outcomes for one metadata request."""

    existence_mask: tuple[bool, ...]
    rpc_elapsed_s: float = Field(ge=0.0, allow_inf_nan=False)


class RegionArtifactTransferResult(_FrozenPublicModel):
    """Caller-ordered outcomes and timing for one transfer request."""

    success_mask: tuple[bool, ...]
    operation_id: str | None
    pack_elapsed_s: float = Field(ge=0.0, allow_inf_nan=False)
    copy_elapsed_s: float = Field(ge=0.0, allow_inf_nan=False)
    rpc_elapsed_s: float = Field(ge=0.0, allow_inf_nan=False)

    @field_validator("operation_id")
    @classmethod
    def _validate_operation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, field_name="operation_id")

    @model_validator(mode="after")
    def _validate_operation_id_presence(self) -> RegionArtifactTransferResult:
        if self.success_mask and self.operation_id is None:
            raise ValueError("a non-empty transfer result requires an operation_id")
        if not self.success_mask and self.operation_id is not None:
            raise ValueError("an empty transfer result must not have an operation_id")
        return self


@dataclass(frozen=True, slots=True)
class _OperationalOptionFingerprint:
    transfer_mode: RegionTransferMode
    scratch_capacity_bytes: int | None
    transfer_timeout_s: float | None
    exists_timeout_s: float

    @classmethod
    def from_options(
        cls,
        options: RegionBackedArtifactSessionOptions,
    ) -> _OperationalOptionFingerprint:
        transfer = options.transfer
        scratch_capacity_bytes = (
            transfer.capacity_bytes
            if isinstance(transfer, ScratchTransferOptions)
            else None
        )
        return cls(
            transfer_mode=transfer.mode,
            scratch_capacity_bytes=scratch_capacity_bytes,
            transfer_timeout_s=options.transfer_timeout_s,
            exists_timeout_s=options.exists_timeout_s,
        )

    def differing_fields(
        self,
        other: _OperationalOptionFingerprint,
    ) -> tuple[str, ...]:
        differences: list[str] = []
        if self.transfer_mode != other.transfer_mode:
            differences.append("transfer.mode")
        if self.scratch_capacity_bytes != other.scratch_capacity_bytes:
            differences.append("transfer.capacity_bytes")
        if self.transfer_timeout_s != other.transfer_timeout_s:
            differences.append("transfer_timeout_s")
        if self.exists_timeout_s != other.exists_timeout_s:
            differences.append("exists_timeout_s")
        return tuple(differences)


class _RegionAllocationLifecycle(str, Enum):
    BUILDING = "building"
    PROCESS_PINNED = "process_pinned"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class _RegionSlotGeometry:
    slot_bytes: int


@dataclass(slots=True)
class _RegionAllocationRecord:
    allocation_sequence: int
    handle: LocalRegionHandle
    capacity_bytes: int
    lifecycle: _RegionAllocationLifecycle = _RegionAllocationLifecycle.BUILDING
    attachment: HostSharedRegionAttachment | None = None
    file_descriptor: int | None = None
    mapped_region: mmap.mmap | None = None
    storage_root: object | None = None
    tensor_root: torch.Tensor | None = None
    base_address: int | None = None
    slot_geometry: _RegionSlotGeometry | None = None
    view_escaped: bool = False
    data_rpc_used: bool = False
    rollback_attempted: bool = False


@dataclass(frozen=True, slots=True)
class _ResolvedRegionSpan:
    record: _RegionAllocationRecord
    region_offset: int


@dataclass(frozen=True, slots=True)
class _AllocationRequest:
    shape: tuple[int, ...]
    dtype: torch.dtype
    element_count: int
    byte_length: int
    diagnostic_name: str


@dataclass(frozen=True, slots=True)
class _CompiledArtifact:
    spec: ByteArtifactSpec
    artifact_id: str
    selection: common_pb2.ArtifactSelection


@dataclass(frozen=True, slots=True)
class _CompiledTransfer:
    transfer: RegionArtifactTransfer
    artifact: _CompiledArtifact


@dataclass(frozen=True, slots=True)
class _ExpectedOutcome:
    artifact_id: str
    slot_index: int | None = None
    slot_generation: int | None = None


@dataclass(frozen=True, slots=True)
class _GeometryCandidate:
    record: _RegionAllocationRecord
    slot_bytes: int


@dataclass(slots=True)
class _PreparedDirectTransfer:
    operation_kind: RegionSessionOperationKind
    compiled_transfers: tuple[_CompiledTransfer, ...]
    request: (
        store_daemon_pb2.BatchGetIntoRegionRequest
        | store_daemon_pb2.BatchPutIfAbsentFromRegionRequest
    )
    geometry_candidates: tuple[_GeometryCandidate, ...]
    region_offsets: tuple[int, ...]
    operation_id: str | None = None
    slot_generation: int | None = None

    def expected_outcomes(self) -> tuple[_ExpectedOutcome, ...]:
        if self.slot_generation is None:
            raise RuntimeError("direct transfer has not been admitted")
        return tuple(
            _ExpectedOutcome(
                artifact_id=compiled.artifact.artifact_id,
                slot_index=int(offset.slot_index),
                slot_generation=self.slot_generation,
            )
            for compiled, offset in zip(
                self.compiled_transfers,
                self.request_layout.offsets,
                strict=True,
            )
        )

    @property
    def request_layout(self) -> store_daemon_pb2.TargetLayout:
        if self.operation_kind is RegionSessionOperationKind.GET_INTO:
            assert isinstance(
                self.request,
                store_daemon_pb2.BatchGetIntoRegionRequest,
            )
            return self.request.target_layout
        assert isinstance(
            self.request,
            store_daemon_pb2.BatchPutIfAbsentFromRegionRequest,
        )
        return self.request.source_layout


@dataclass(frozen=True, slots=True)
class _PreparedScratchLayout:
    compiled_transfers: tuple[_CompiledTransfer, ...]
    layout: store_daemon_pb2.TargetLayout
    packed_offsets: tuple[int, ...]


@dataclass(slots=True)
class _PreparedScratchTransfer:
    operation_kind: RegionSessionOperationKind
    compiled_transfers: tuple[_CompiledTransfer, ...]
    record: _RegionAllocationRecord
    request: (
        store_daemon_pb2.BatchGetIntoRegionRequest
        | store_daemon_pb2.BatchPutIfAbsentFromRegionRequest
    )
    packed_offsets: tuple[int, ...]
    operation_id: str | None = None

    def expected_outcomes(self) -> tuple[_ExpectedOutcome, ...]:
        return tuple(
            _ExpectedOutcome(artifact_id=compiled.artifact.artifact_id)
            for compiled in self.compiled_transfers
        )


@dataclass(frozen=True, slots=True)
class _ScratchTransferMetrics:
    direction: RegionSessionOperationKind
    artifact_count: int
    byte_count: int
    pack_elapsed_s: float
    copy_elapsed_s: float
    rpc_elapsed_s: float


@dataclass(frozen=True, slots=True)
class _SessionObservabilitySnapshot:
    session_name: str
    owner_pid: int
    endpoint: str
    transfer_mode: RegionTransferMode
    health: RegionSessionHealth
    health_transition_count: int
    admitted_operation_count: int
    completed_operation_count: int
    rejected_after_failure_count: int
    in_flight_operation_count: int
    get_transfer_count: int
    put_transfer_count: int
    artifact_count: int
    total_transfer_bytes: int
    unique_region_reference_count: int
    scratch_bytes_copied: int
    direct_bytes_submitted: int
    rpc_elapsed_s: float
    pack_elapsed_s: float
    copy_elapsed_s: float
    successful_item_count: int
    missed_item_count: int
    allocated_region_count: int
    allocated_region_bytes: int
    address_resolution_failure_count: int
    geometry_validation_failure_count: int
    suppressed_transparent_retry_count: int


@dataclass(slots=True)
class _SessionObservabilityCounters:
    health_transition_count: int = 0
    admitted_operation_count: int = 0
    completed_operation_count: int = 0
    rejected_after_failure_count: int = 0
    get_transfer_count: int = 0
    put_transfer_count: int = 0
    artifact_count: int = 0
    total_transfer_bytes: int = 0
    unique_region_reference_count: int = 0
    scratch_bytes_copied: int = 0
    direct_bytes_submitted: int = 0
    rpc_elapsed_s: float = 0.0
    pack_elapsed_s: float = 0.0
    copy_elapsed_s: float = 0.0
    successful_item_count: int = 0
    missed_item_count: int = 0
    address_resolution_failure_count: int = 0
    geometry_validation_failure_count: int = 0
    suppressed_transparent_retry_count: int = 0


@dataclass(frozen=True, slots=True)
class _ExistsPartition:
    compiled_artifacts: tuple[_CompiledArtifact, ...]
    request: store_daemon_pb2.BatchExistsRequest


class _OutcomeValidationError(RuntimeError):
    code: RegionSessionFailureCode

    def __init__(self, code: RegionSessionFailureCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class _StructuredRegionLostError(RuntimeError):
    """Internal typed evidence that a previously pinned region was lost."""


class _RolledBackRegionAllocationError(RuntimeError):
    pass


class _AmbiguousRegionSetupError(RuntimeError):
    pass


class _DaemonClient(Protocol):
    @property
    def _effective_grpc_message_limits(self) -> _GrpcMessageLimitsLike: ...

    def get_server_config(self) -> ServerConfig: ...

    def register_region(
        self,
        *,
        memory_kind: RegionMemoryKind,
        size_bytes: int,
        ttl_ms: int,
        daemon_managed: bool,
        host_shared_region_class: HostSharedRegionClass,
        region_name: str,
    ) -> LocalRegionHandle: ...

    def attach_host_shared_region(
        self,
        handle: LocalRegionHandle,
    ) -> HostSharedRegionAttachment: ...

    def release_host_shared_region(
        self,
        handle: LocalRegionHandle,
    ) -> bool: ...

    def unregister_region(self, region_id: str) -> bool: ...

    def batch_exists(
        self,
        *,
        selections: Sequence[common_pb2.ArtifactSelection],
        timeout_s: float,
        operation_id: str | None = None,
    ) -> store_daemon_pb2.BatchExistsResponse: ...

    def batch_get_into_region(
        self,
        *,
        selections: Sequence[common_pb2.ArtifactSelection],
        target_layout: store_daemon_pb2.TargetLayout,
        pid: int,
        device_uuid: str,
        operation_id: str | None = None,
        timeout_s: float | None = 600.0,
        retries: int = 1,
    ) -> store_daemon_pb2.BatchGetIntoRegionResponse: ...

    def batch_put_if_absent_from_region(
        self,
        *,
        items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
        source_layout: store_daemon_pb2.TargetLayout,
        pid: int,
        device_uuid: str,
        ttl_ms: int | None = None,
        operation_id: str | None = None,
        timeout_s: float | None = 600.0,
        retries: int = 1,
    ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse: ...


class _GrpcMessageLimitsLike(Protocol):
    max_send_message_bytes: int
    max_receive_message_bytes: int


_DAEMON_CLIENT_FACTORY: Callable[[str], _DaemonClient] = get_daemon_client
_LOCAL_HANDLE_PROBE_TIMEOUT_S = 1.0
_DAEMON_STARTUP_PHASE_READY = int(store_daemon_pb2.DAEMON_STARTUP_PHASE_READY)
_REGION_NAME_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_UINT64 = (1 << 64) - 1
_GRPC_FRAME_BYTES = 5
_WIRE_FIXED_HEADROOM_BYTES = 4096
_ORDINARY_OUTCOME_MESSAGE_BYTES = 32
_WIRE_OPERATION_ID_PLACEHOLDER = "0" * 32


def _map_shared_region(file_descriptor: int, byte_length: int) -> mmap.mmap:
    return mmap.mmap(
        file_descriptor,
        byte_length,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )


def _tensor_from_shared_mapping(
    mapped_region: mmap.mmap,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    element_count: int,
) -> torch.Tensor:
    return torch.frombuffer(
        mapped_region,
        dtype=dtype,
        count=element_count,
    ).reshape(shape)


_SHARED_REGION_MAPPER: Callable[[int, int], mmap.mmap] = _map_shared_region
_MAPPED_TENSOR_FACTORY: Callable[
    [mmap.mmap, tuple[int, ...], torch.dtype, int], torch.Tensor
] = _tensor_from_shared_mapping


def _copy_host_memory(
    destination_address: int,
    source_address: int,
    byte_length: int,
) -> None:
    ctypes.memmove(destination_address, source_address, byte_length)


_HOST_MEMORY_COPY: Callable[[int, int, int], None] = _copy_host_memory


def _sanitize_region_name_component(value: str) -> str:
    sanitized = _REGION_NAME_COMPONENT_PATTERN.sub("_", value.strip()).strip("_")
    return sanitized or "unnamed"


def _validate_allocation_request(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> _AllocationRequest:
    try:
        shape_snapshot = tuple(shape)
    except TypeError as exc:
        raise RegionArtifactInputError("shape must be an iterable of integers") from exc

    element_count = 1
    for dimension in shape_snapshot:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise RegionArtifactInputError("shape dimensions must be integers")
        if dimension < 0:
            raise RegionArtifactInputError("shape dimensions must be non-negative")
        if dimension != 0 and element_count > sys.maxsize // dimension:
            raise RegionArtifactInputError("shape element count overflows")
        element_count *= dimension
    if element_count == 0:
        raise RegionArtifactInputError("allocation byte length must be positive")

    if not isinstance(dtype, torch.dtype):
        raise RegionArtifactInputError("dtype must be a torch.dtype")
    try:
        dtype_probe = torch.empty((), dtype=dtype, device="cpu")
        element_size = int(dtype_probe.element_size())
        torch.frombuffer(bytearray(element_size), dtype=dtype, count=1)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RegionArtifactInputError(
            f"dtype {dtype!s} is not supported for CPU shared memory"
        ) from exc
    if element_size <= 0:
        raise RegionArtifactInputError("dtype element size must be positive")
    if element_count > sys.maxsize // element_size:
        raise RegionArtifactInputError("allocation byte length overflows")
    byte_length = element_count * element_size

    if not isinstance(name, str) or not name.strip():
        raise RegionArtifactInputError("name must not be empty")
    return _AllocationRequest(
        shape=shape_snapshot,
        dtype=dtype,
        element_count=element_count,
        byte_length=byte_length,
        diagnostic_name=_sanitize_region_name_component(name),
    )


def _lower_artifacts(
    artifacts: tuple[ByteArtifactSpec, ...],
) -> tuple[_CompiledArtifact, ...]:
    compiled: list[_CompiledArtifact] = []
    seen_artifact_ids: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, ByteArtifactSpec):
            raise RegionArtifactInputError(
                "artifact batches must contain ByteArtifactSpec values"
            )
        keyspace = artifact.keyspace
        try:
            artifact_id = build_byte_artifact_cgid(
                namespace=keyspace.namespace,
                engine=keyspace.engine,
                model_id=keyspace.model_id,
                model_version=keyspace.model_version,
                layout_id=keyspace.layout_id,
                engine_key=artifact.engine_key,
            )
            selection = build_artifact_selection(
                artifact_id=artifact_id,
                canonical_index_bytes=b"",
                layout_index_bytes=None,
                view_spec=None,
                tensor_names=None,
                view_subset_hash=None,
            )
        except ValueError as exc:
            raise RegionArtifactInputError(
                "byte artifact identity cannot be derived from the supplied keyspace"
            ) from exc
        if artifact_id in seen_artifact_ids:
            raise RegionArtifactInputError(
                "artifact batch contains a duplicate canonical artifact identity"
            )
        seen_artifact_ids.add(artifact_id)
        compiled.append(
            _CompiledArtifact(
                spec=artifact,
                artifact_id=artifact_id,
                selection=selection,
            )
        )
    return tuple(compiled)


def _lower_transfers(
    transfers: tuple[RegionArtifactTransfer, ...],
) -> tuple[_CompiledTransfer, ...]:
    for transfer in transfers:
        if not isinstance(transfer, RegionArtifactTransfer):
            raise RegionArtifactInputError(
                "transfer batches must contain RegionArtifactTransfer values"
            )
        if transfer.artifact.byte_length != transfer.span.byte_length:
            raise RegionArtifactInputError(
                "artifact byte_length must equal span byte_length"
            )
    compiled_artifacts = _lower_artifacts(
        tuple(transfer.artifact for transfer in transfers)
    )
    ordered_ranges = sorted(
        (
            transfer.span.address,
            transfer.span.address + transfer.span.byte_length,
        )
        for transfer in transfers
    )
    for previous, current in zip(ordered_ranges, ordered_ranges[1:], strict=False):
        if current[0] < previous[1]:
            raise RegionArtifactInputError(
                "host spans in one transfer batch must not overlap"
            )
    return tuple(
        _CompiledTransfer(transfer=transfer, artifact=artifact)
        for transfer, artifact in zip(transfers, compiled_artifacts, strict=True)
    )


def _new_byte_artifact_layout() -> store_daemon_pb2.TargetLayout:
    return store_daemon_pb2.TargetLayout(
        layout_kind=(store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED),
        index_kind=store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
        tensor_spec_kind=store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        logical_layout_hash=compute_byte_artifact_logical_layout_hash(),
    )


def _add_host_shared_storage(
    layout: store_daemon_pb2.TargetLayout,
    *,
    storage_id: str,
    record: _RegionAllocationRecord,
) -> None:
    storage = layout.storages.add(
        storage_id=storage_id,
        device_id=-1,
        storage_length=record.capacity_bytes,
        mapping_base_offset=0,
    )
    storage.region_ref.region_id = record.handle.region_id
    storage.region_ref.memory_kind = store_daemon_pb2.REGION_MEMORY_KIND_HOST_SHARED
    storage.region_ref.device_id = -1
    storage.region_ref.size_bytes = record.capacity_bytes


def _compile_scratch_layout(
    compiled_transfers: tuple[_CompiledTransfer, ...],
    record: _RegionAllocationRecord,
) -> _PreparedScratchLayout:
    if record.lifecycle is not _RegionAllocationLifecycle.PROCESS_PINNED:
        raise RegionArtifactInputError("scratch region is not process-pinned")
    if record.handle.host_shared_region_class is not HostSharedRegionClass.SCRATCH:
        raise RegionArtifactInputError("scratch layout requires a SCRATCH region")
    layout = _new_byte_artifact_layout()
    _add_host_shared_storage(layout, storage_id="storage-0", record=record)
    packed_offsets: list[int] = []
    cursor = 0
    for compiled in compiled_transfers:
        byte_length = compiled.artifact.spec.byte_length
        if cursor > record.capacity_bytes - byte_length:
            raise RegionArtifactInputError(
                "transfer batch exceeds scratch region capacity"
            )
        packed_offsets.append(cursor)
        layout.offsets.add(
            name=compiled.artifact.artifact_id,
            storage_id="storage-0",
            storage_offset=cursor,
            logical_length=byte_length,
        )
        cursor += byte_length
    return _PreparedScratchLayout(
        compiled_transfers=compiled_transfers,
        layout=layout,
        packed_offsets=tuple(packed_offsets),
    )


def _validate_scratch_capacity(
    compiled_transfers: tuple[_CompiledTransfer, ...],
    *,
    capacity_bytes: int,
) -> int:
    total_bytes = 0
    for compiled in compiled_transfers:
        byte_length = compiled.artifact.spec.byte_length
        if total_bytes > capacity_bytes - byte_length:
            raise RegionArtifactInputError(
                "transfer batch exceeds scratch region capacity"
            )
        total_bytes += byte_length
    return total_bytes


def _build_exists_request(
    artifacts: Sequence[_CompiledArtifact],
) -> store_daemon_pb2.BatchExistsRequest:
    request = store_daemon_pb2.BatchExistsRequest()
    for artifact in artifacts:
        request.selections.add().CopyFrom(artifact.selection)
    return request


def _estimate_response_bytes(
    expected: Sequence[_ExpectedOutcome],
    *,
    direct: bool,
) -> int:
    response = store_daemon_pb2.BatchGetIntoRegionResponse()
    for item in expected:
        outcome = response.outcomes.add(
            artifact_id=item.artifact_id,
            status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            message="x" * _ORDINARY_OUTCOME_MESSAGE_BYTES,
        )
        if direct:
            outcome.slot_index = _MAX_UINT64
            outcome.slot_generation = _MAX_UINT64
    return response.ByteSize() + _GRPC_FRAME_BYTES + _WIRE_FIXED_HEADROOM_BYTES


def _request_wire_bytes(request: Message) -> int:
    return int(request.ByteSize())


def _fits_wire_budget(
    request: Message,
    expected: Sequence[_ExpectedOutcome],
    *,
    limits: _GrpcMessageLimitsLike,
    direct: bool,
) -> bool:
    return (
        _request_wire_bytes(request) <= limits.max_send_message_bytes
        and _estimate_response_bytes(expected, direct=direct)
        <= limits.max_receive_message_bytes
    )


def _require_transfer_wire_budget(
    request: Message,
    expected: Sequence[_ExpectedOutcome],
    *,
    limits: _GrpcMessageLimitsLike,
    direct: bool,
) -> None:
    if not _fits_wire_budget(
        request,
        expected,
        limits=limits,
        direct=direct,
    ):
        raise RegionArtifactInputError(
            "transfer batch exceeds the effective gRPC wire budget"
        )


def _partition_exists_artifacts(
    compiled_artifacts: tuple[_CompiledArtifact, ...],
    *,
    limits: _GrpcMessageLimitsLike,
) -> tuple[_ExistsPartition, ...]:
    partitions: list[_ExistsPartition] = []
    start = 0
    while start < len(compiled_artifacts):
        low = start + 1
        high = len(compiled_artifacts)
        accepted_end = start
        accepted_request: store_daemon_pb2.BatchExistsRequest | None = None
        while low <= high:
            candidate_end = (low + high) // 2
            candidate_artifacts = compiled_artifacts[start:candidate_end]
            candidate_request = _build_exists_request(candidate_artifacts)
            candidate_expected = tuple(
                _ExpectedOutcome(artifact_id=artifact.artifact_id)
                for artifact in candidate_artifacts
            )
            if _fits_wire_budget(
                candidate_request,
                candidate_expected,
                limits=limits,
                direct=False,
            ):
                accepted_end = candidate_end
                accepted_request = candidate_request
                low = candidate_end + 1
            else:
                high = candidate_end - 1
        if accepted_request is None:
            raise RegionArtifactInputError(
                "one artifact exceeds the effective gRPC wire budget"
            )
        partitions.append(
            _ExistsPartition(
                compiled_artifacts=compiled_artifacts[start:accepted_end],
                request=accepted_request,
            )
        )
        start = accepted_end
    return tuple(partitions)


def _validate_batch_outcomes(
    outcomes: Sequence[store_daemon_pb2.BatchItemOutcome],
    expected: tuple[_ExpectedOutcome, ...],
    *,
    operation_kind: RegionSessionOperationKind,
) -> tuple[bool, ...]:
    expected_by_id = {item.artifact_id: item for item in expected}
    if len(expected_by_id) != len(expected):
        raise _OutcomeValidationError(
            RegionSessionFailureCode.MALFORMED_RESPONSE,
            "expected outcome identities are not unique",
        )
    resolved: dict[str, bool] = {}
    for outcome in outcomes:
        artifact_id = str(outcome.artifact_id)
        expected_item = expected_by_id.get(artifact_id)
        if expected_item is None:
            raise _OutcomeValidationError(
                RegionSessionFailureCode.MALFORMED_RESPONSE,
                "daemon response contains an unknown artifact outcome",
            )
        if artifact_id in resolved:
            raise _OutcomeValidationError(
                RegionSessionFailureCode.MALFORMED_RESPONSE,
                "daemon response contains a duplicate artifact outcome",
            )

        has_slot_index = outcome.HasField("slot_index")
        has_slot_generation = outcome.HasField("slot_generation")
        expects_slot = expected_item.slot_index is not None
        if has_slot_index != has_slot_generation or expects_slot != has_slot_index:
            raise _OutcomeValidationError(
                RegionSessionFailureCode.MALFORMED_RESPONSE,
                "daemon response contains malformed slot tokens",
            )
        if expects_slot and (
            int(outcome.slot_index) != expected_item.slot_index
            or int(outcome.slot_generation) != expected_item.slot_generation
        ):
            raise _OutcomeValidationError(
                RegionSessionFailureCode.MALFORMED_RESPONSE,
                "daemon response slot token does not match the request",
            )

        status = int(outcome.status)
        if status == store_daemon_pb2.BATCH_ITEM_STATUS_OK:
            resolved[artifact_id] = True
            continue
        if status == store_daemon_pb2.BATCH_ITEM_STATUS_MISS and operation_kind in (
            RegionSessionOperationKind.EXISTS,
            RegionSessionOperationKind.GET_INTO,
        ):
            resolved[artifact_id] = False
            continue
        raise _OutcomeValidationError(
            RegionSessionFailureCode.DAEMON_STATUS,
            f"daemon returned non-allowlisted item status {status}",
        )

    if len(resolved) != len(expected):
        raise _OutcomeValidationError(
            RegionSessionFailureCode.MALFORMED_RESPONSE,
            "daemon response is missing artifact outcomes",
        )
    return tuple(resolved[item.artifact_id] for item in expected)


def _failure_code_for_exception(cause: BaseException) -> RegionSessionFailureCode:
    current: BaseException | None = cause
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, _OutcomeValidationError):
            return current.code
        if isinstance(current, _StructuredRegionLostError):
            return RegionSessionFailureCode.REGION_LOST
        if isinstance(current, grpc.RpcError):
            return RegionSessionFailureCode.TRANSPORT
        current = current.__cause__ or current.__context__
    return RegionSessionFailureCode.INTERNAL


def _normalize_options(
    options: RegionBackedArtifactSessionOptions,
) -> RegionBackedArtifactSessionOptions:
    normalized = options.model_dump(mode="python")
    normalized["daemon_address"] = options.daemon_address.strip()
    normalized["session_name"] = options.session_name.strip()
    normalized["region_name_prefix"] = options.region_name_prefix.strip()
    return RegionBackedArtifactSessionOptions.model_validate(normalized)


def _normalize_host_port(target: str) -> tuple[str, str]:
    if target.startswith("["):
        closing_bracket = target.find("]")
        if (
            closing_bracket < 0
            or target[closing_bracket + 1 : closing_bracket + 2] != ":"
        ):
            raise ValueError("IPv6 daemon endpoint must use [host]:port syntax")
        host = target[1:closing_bracket]
        port_text = target[closing_bracket + 2 :]
    else:
        host, separator, port_text = target.rpartition(":")
        if not separator:
            raise ValueError("daemon endpoint must include a port")

    host = host.strip().rstrip(".").lower()
    if not host:
        raise ValueError("daemon endpoint host must not be empty")
    try:
        port = int(port_text.strip())
    except ValueError as exc:
        raise ValueError("daemon endpoint port must be an integer") from exc
    if port <= 0 or port > 65535:
        raise ValueError("daemon endpoint port must be between 1 and 65535")

    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        canonical_host = host
    else:
        canonical_host = parsed_ip.compressed
    rendered_host = f"[{canonical_host}]" if ":" in canonical_host else canonical_host
    return f"{rendered_host}:{port}", canonical_host


def _canonicalize_daemon_endpoint(address: str) -> tuple[str, str | None]:
    endpoint = address.strip()
    if endpoint.startswith("unix:"):
        socket_path = endpoint.removeprefix("unix:")
        while socket_path.startswith("//"):
            socket_path = socket_path[1:]
        if not socket_path.startswith("/"):
            raise ValueError("Unix daemon endpoint must contain an absolute path")
        normalized_path = posixpath.normpath(socket_path)
        return f"unix://{normalized_path}", None
    if endpoint.startswith("unix-abstract:"):
        abstract_name = endpoint.removeprefix("unix-abstract:").strip()
        if not abstract_name:
            raise ValueError("abstract Unix daemon endpoint must include a name")
        return f"unix-abstract:{abstract_name}", None
    if endpoint.startswith("dns:///"):
        normalized_target, host = _normalize_host_port(endpoint.removeprefix("dns:///"))
        return f"dns:///{normalized_target}", host
    if "://" in endpoint:
        raise ValueError("unsupported daemon endpoint scheme")
    return _normalize_host_port(endpoint)


def _is_node_local_host(host: str) -> bool:
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        if host == "localhost":
            return True
    else:
        if parsed_ip.is_unspecified:
            return False

    try:
        addresses = socket.getaddrinfo(host, 0, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for family, socket_type, protocol, _, socket_address in addresses:
        try:
            with socket.socket(family, socket_type, protocol) as probe:
                if family == socket.AF_INET6:
                    probe.bind((socket_address[0], 0, 0, socket_address[3]))
                else:
                    probe.bind((socket_address[0], 0))
        except OSError:
            continue
        return True
    return False


def _require_node_local_endpoint(canonical_address: str, host: str | None) -> None:
    if host is None or _is_node_local_host(host):
        return
    raise ValueError(f"daemon endpoint {canonical_address!r} is not node-local")


def _probe_local_handle_service(local_handle_socket_path: str) -> None:
    if not local_handle_socket_path.startswith("/"):
        raise ValueError("daemon local_handle_socket_path must be absolute")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(_LOCAL_HANDLE_PROBE_TIMEOUT_S)
        probe.connect(local_handle_socket_path)


_LOCAL_HANDLE_SERVICE_PROBE: Callable[[str], None] = _probe_local_handle_service


def _perform_attach_handshake(client: _DaemonClient) -> None:
    config = client.get_server_config()
    if config.startup_phase != _DAEMON_STARTUP_PHASE_READY:
        raise RuntimeError(
            f"StoreDaemon is not ready: startup_phase={config.startup_phase}"
        )
    if not config.cpu_shared_memory_enabled:
        raise RuntimeError("StoreDaemon CPU shared memory is disabled")
    local_handle_socket_path = config.local_handle_socket_path.strip()
    if not local_handle_socket_path:
        raise RuntimeError("StoreDaemon local_handle_socket_path is missing")
    _LOCAL_HANDLE_SERVICE_PROBE(local_handle_socket_path)


_PROCESS_SESSION_REGISTRY_LOCK = threading.Lock()
_PROCESS_SESSION_REGISTRY: dict[tuple[int, str], RegionBackedArtifactSession] = {}


def _find_process_session_locked(
    owner_pid: int,
) -> RegionBackedArtifactSession | None:
    for (registered_pid, _), session in _PROCESS_SESSION_REGISTRY.items():
        if registered_pid == owner_pid:
            return session
    return None


class RegionBackedArtifactSession:
    """One process-scoped attachment and sticky region-RPC failure domain."""

    _options: RegionBackedArtifactSessionOptions
    _canonical_daemon_address: str
    _owner_pid: int
    _client: _DaemonClient
    _operational_fingerprint: _OperationalOptionFingerprint
    _health: RegionSessionHealth
    _lifecycle_state: RegionSessionLifecycleState
    _failure: RegionSessionFailure | None
    _in_flight_count: int
    _allocation_records: list[_RegionAllocationRecord]
    _next_allocation_sequence: int
    _next_rpc_generation: int
    _scratch_get_record: _RegionAllocationRecord | None
    _scratch_put_record: _RegionAllocationRecord | None
    _scratch_metrics: dict[RegionSessionOperationKind, _ScratchTransferMetrics]
    _owned_helper_cleanups: list[Callable[[], None]]
    _helper_cleanup_started: bool
    _observability: _SessionObservabilityCounters

    def __new__(cls) -> RegionBackedArtifactSession:
        raise TypeError("use RegionBackedArtifactSession.attach()")

    @classmethod
    def _create_attached(
        cls,
        *,
        options: RegionBackedArtifactSessionOptions,
        canonical_daemon_address: str,
        owner_pid: int,
        client: _DaemonClient,
    ) -> RegionBackedArtifactSession:
        session = object.__new__(cls)
        session._options = options
        session._canonical_daemon_address = canonical_daemon_address
        session._owner_pid = owner_pid
        session._client = client
        session._operational_fingerprint = _OperationalOptionFingerprint.from_options(
            options
        )
        session._state_lock = threading.Lock()
        session._region_lock = threading.Lock()
        session._geometry_lock = threading.Lock()
        session._generation_lock = threading.Lock()
        session._scratch_get_lock = threading.Lock()
        session._scratch_put_lock = threading.Lock()
        session._observability_lock = threading.Lock()
        session._health = RegionSessionHealth.READY
        session._lifecycle_state = RegionSessionLifecycleState.ATTACHED
        session._failure: RegionSessionFailure | None = None
        session._in_flight_count = 0
        session._allocation_records = []
        session._next_allocation_sequence = 1
        session._next_rpc_generation = 1
        session._scratch_get_record: _RegionAllocationRecord | None = None
        session._scratch_put_record: _RegionAllocationRecord | None = None
        session._scratch_metrics: dict[
            RegionSessionOperationKind, _ScratchTransferMetrics
        ] = {}
        session._owned_helper_cleanups = []
        session._helper_cleanup_started = False
        session._observability = _SessionObservabilityCounters()
        return session

    @classmethod
    def attach(
        cls,
        options: RegionBackedArtifactSessionOptions,
    ) -> RegionBackedArtifactSession:
        """Attach this owner process to one ready, node-local StoreDaemon."""
        try:
            normalized_options = _normalize_options(options)
            canonical_address, endpoint_host = _canonicalize_daemon_endpoint(
                normalized_options.daemon_address
            )
        except Exception as exc:
            raise RegionSessionAttachError(
                "invalid region-backed artifact Session options"
            ) from exc

        owner_pid = os.getpid()
        requested_fingerprint = _OperationalOptionFingerprint.from_options(
            normalized_options
        )
        with _PROCESS_SESSION_REGISTRY_LOCK:
            existing = _find_process_session_locked(owner_pid)
            if existing is not None:
                if existing.lifecycle_state is RegionSessionLifecycleState.TERMINATED:
                    raise RegionSessionTerminatedError(
                        "the process-scoped region-backed artifact Session is terminated"
                    )
                if existing._canonical_daemon_address != canonical_address:
                    raise RegionSessionAttachError(
                        "this process is already attached to daemon endpoint "
                        f"{existing._canonical_daemon_address!r}; refusing "
                        f"{canonical_address!r}"
                    )
                differences = existing._operational_fingerprint.differing_fields(
                    requested_fingerprint
                )
                if differences:
                    raise RegionSessionAttachError(
                        "region-backed artifact Session options conflict in: "
                        + ", ".join(differences)
                    )
                return existing

            try:
                _require_node_local_endpoint(canonical_address, endpoint_host)
                client = _DAEMON_CLIENT_FACTORY(canonical_address)
                _perform_attach_handshake(client)
            except Exception as exc:
                raise RegionSessionAttachError(
                    f"failed to attach to StoreDaemon at {canonical_address!r}"
                ) from exc

            session = cls._create_attached(
                options=normalized_options,
                canonical_daemon_address=canonical_address,
                owner_pid=owner_pid,
                client=client,
            )
            _PROCESS_SESSION_REGISTRY[(owner_pid, canonical_address)] = session
            return session

    @property
    def health(self) -> RegionSessionHealth:
        with self._state_lock:
            return self._health

    @property
    def lifecycle_state(self) -> RegionSessionLifecycleState:
        with self._state_lock:
            return self._lifecycle_state

    @property
    def failure(self) -> RegionSessionFailure | None:
        with self._state_lock:
            return self._failure

    @property
    def _diagnostic_in_flight_count(self) -> int:
        with self._state_lock:
            return self._in_flight_count

    @property
    def _diagnostic_allocation_records(self) -> tuple[_RegionAllocationRecord, ...]:
        with self._region_lock:
            return tuple(self._allocation_records)

    @property
    def _diagnostic_scratch_metrics(self) -> tuple[_ScratchTransferMetrics, ...]:
        with self._region_lock:
            return tuple(
                self._scratch_metrics[direction]
                for direction in (
                    RegionSessionOperationKind.GET_INTO,
                    RegionSessionOperationKind.PUT_FROM,
                )
                if direction in self._scratch_metrics
            )

    @property
    def _diagnostic_observability(self) -> _SessionObservabilitySnapshot:
        with self._state_lock:
            health = self._health
            in_flight_count = self._in_flight_count
        with self._region_lock:
            pinned_records = tuple(
                record
                for record in self._allocation_records
                if record.lifecycle is _RegionAllocationLifecycle.PROCESS_PINNED
            )
        with self._observability_lock:
            counters = self._observability
            return _SessionObservabilitySnapshot(
                session_name=self._options.session_name,
                owner_pid=self._owner_pid,
                endpoint=self._canonical_daemon_address,
                transfer_mode=self._options.transfer.mode,
                health=health,
                health_transition_count=counters.health_transition_count,
                admitted_operation_count=counters.admitted_operation_count,
                completed_operation_count=counters.completed_operation_count,
                rejected_after_failure_count=counters.rejected_after_failure_count,
                in_flight_operation_count=in_flight_count,
                get_transfer_count=counters.get_transfer_count,
                put_transfer_count=counters.put_transfer_count,
                artifact_count=counters.artifact_count,
                total_transfer_bytes=counters.total_transfer_bytes,
                unique_region_reference_count=(counters.unique_region_reference_count),
                scratch_bytes_copied=counters.scratch_bytes_copied,
                direct_bytes_submitted=counters.direct_bytes_submitted,
                rpc_elapsed_s=counters.rpc_elapsed_s,
                pack_elapsed_s=counters.pack_elapsed_s,
                copy_elapsed_s=counters.copy_elapsed_s,
                successful_item_count=counters.successful_item_count,
                missed_item_count=counters.missed_item_count,
                allocated_region_count=len(pinned_records),
                allocated_region_bytes=sum(
                    record.capacity_bytes for record in pinned_records
                ),
                address_resolution_failure_count=(
                    counters.address_resolution_failure_count
                ),
                geometry_validation_failure_count=(
                    counters.geometry_validation_failure_count
                ),
                suppressed_transparent_retry_count=(
                    counters.suppressed_transparent_retry_count
                ),
            )

    def _raise_unavailable_locked(self) -> None:
        if self._health is RegionSessionHealth.FAILED:
            assert self._failure is not None
            with self._observability_lock:
                self._observability.rejected_after_failure_count += 1
            raise RegionSessionFailedError(self._failure)
        if self._lifecycle_state is RegionSessionLifecycleState.TERMINATED:
            raise RegionSessionTerminatedError(
                "the process-scoped region-backed artifact Session is terminated"
            )

    def _preliminary_state_gate(self) -> None:
        with self._state_lock:
            self._raise_unavailable_locked()

    def _final_rpc_admission(self) -> str:
        with self._state_lock:
            self._raise_unavailable_locked()
            operation_id = uuid.uuid4().hex
            self._in_flight_count += 1
            with self._observability_lock:
                self._observability.admitted_operation_count += 1
            return operation_id

    def _complete_rpc_admission(self) -> RegionSessionFailure | None:
        helper_cleanups: tuple[Callable[[], None], ...]
        with self._state_lock:
            assert self._in_flight_count > 0
            self._in_flight_count -= 1
            with self._observability_lock:
                self._observability.completed_operation_count += 1
            helper_cleanups = self._take_owned_helper_cleanups_locked()
            if self._health is RegionSessionHealth.FAILED:
                assert self._failure is not None
                failure = self._failure
            else:
                failure = None
        self._run_owned_helper_cleanups(helper_cleanups)
        return failure

    def _take_owned_helper_cleanups_locked(
        self,
    ) -> tuple[Callable[[], None], ...]:
        if (
            self._lifecycle_state is not RegionSessionLifecycleState.TERMINATED
            or self._in_flight_count != 0
            or self._helper_cleanup_started
        ):
            return ()
        self._helper_cleanup_started = True
        cleanups = tuple(self._owned_helper_cleanups)
        self._owned_helper_cleanups.clear()
        return cleanups

    @staticmethod
    def _run_owned_helper_cleanups(
        cleanups: tuple[Callable[[], None], ...],
    ) -> None:
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception:
                logger.exception("region-backed artifact Session helper cleanup failed")

    def _register_owned_helper_cleanup(self, cleanup: Callable[[], None]) -> None:
        """Register a private Session-owned helper cleanup before termination."""
        with self._state_lock:
            if self._lifecycle_state is RegionSessionLifecycleState.TERMINATED:
                raise RegionSessionTerminatedError(
                    "cannot register a helper after process-session termination"
                )
            self._owned_helper_cleanups.append(cleanup)

    def _latch_failure(
        self,
        failure: RegionSessionFailure,
        *,
        cause: BaseException | None = None,
    ) -> RegionSessionFailure:
        first_failure = False
        with self._state_lock:
            if self._failure is None:
                self._failure = failure
                self._health = RegionSessionHealth.FAILED
                first_failure = True
                with self._observability_lock:
                    self._observability.health_transition_count += 1
            retained = self._failure
        if first_failure:
            if cause is None:
                logger.error(
                    "region-backed artifact Session failed: code=%s "
                    "operation_kind=%s operation_id=%s",
                    failure.code.value,
                    failure.operation_kind.value,
                    failure.operation_id,
                )
            else:
                logger.exception(
                    "region-backed artifact Session failed: code=%s "
                    "operation_kind=%s operation_id=%s",
                    failure.code.value,
                    failure.operation_kind.value,
                    failure.operation_id,
                )
        return retained

    def _latch_region_setup_failure(
        self,
        *,
        operation_id: str,
        cause: BaseException,
        operation_kind: RegionSessionOperationKind = (
            RegionSessionOperationKind.ALLOCATE
        ),
    ) -> RegionSessionFailure:
        detail = str(cause).strip() or cause.__class__.__name__
        return self._latch_failure(
            RegionSessionFailure(
                code=RegionSessionFailureCode.REGION_SETUP,
                message=f"host shared-memory region setup failed: {detail}",
                operation_kind=operation_kind,
                operation_id=operation_id,
                occurred_at=datetime.now(timezone.utc),
            ),
            cause=cause,
        )

    def _allocate_sequence_and_name(
        self,
        request: _AllocationRequest,
    ) -> tuple[int, str]:
        with self._region_lock:
            allocation_sequence = self._next_allocation_sequence
            self._next_allocation_sequence += 1
        diagnostic_parts = (
            _sanitize_region_name_component(self._options.region_name_prefix),
            _sanitize_region_name_component(self._options.session_name),
            str(self._owner_pid),
            str(allocation_sequence),
            request.diagnostic_name,
        )
        return allocation_sequence, "-".join(diagnostic_parts)

    def _retain_building_record(
        self,
        *,
        allocation_sequence: int,
        handle: LocalRegionHandle,
        capacity_bytes: int,
    ) -> _RegionAllocationRecord:
        record = _RegionAllocationRecord(
            allocation_sequence=allocation_sequence,
            handle=handle,
            capacity_bytes=capacity_bytes,
        )
        with self._region_lock:
            self._allocation_records.append(record)
        return record

    @staticmethod
    def _validate_region_handle(
        handle: LocalRegionHandle,
        *,
        capacity_bytes: int,
        region_class: HostSharedRegionClass,
    ) -> None:
        if not handle.region_id:
            raise _AmbiguousRegionSetupError("registered region has no region id")
        if handle.memory_kind is not RegionMemoryKind.HOST_SHARED:
            raise _AmbiguousRegionSetupError(
                "registered region is not HOST_SHARED memory"
            )
        if handle.size_bytes != capacity_bytes:
            raise _AmbiguousRegionSetupError(
                "registered region capacity does not match the request"
            )
        if handle.ttl_ms != 0 or handle.expires_at is not None:
            raise _AmbiguousRegionSetupError(
                "registered region does not have non-expiring lifetime"
            )
        if not handle.daemon_managed:
            raise _AmbiguousRegionSetupError("registered region is not daemon-managed")
        if handle.host_shared_region_class is not region_class:
            raise _AmbiguousRegionSetupError(
                f"registered region is not a {region_class.value} region"
            )
        if not handle.attach_token:
            raise _AmbiguousRegionSetupError(
                "registered region does not contain an attach token"
            )

    @staticmethod
    def _validate_allocator_attachment(
        attachment: HostSharedRegionAttachment,
        *,
        handle: LocalRegionHandle,
        capacity_bytes: int,
    ) -> None:
        if attachment.region_id != handle.region_id:
            raise _AmbiguousRegionSetupError(
                "attached region id does not match registration"
            )
        if attachment.size_bytes != capacity_bytes:
            raise _AmbiguousRegionSetupError(
                "attached region capacity does not match registration"
            )
        if attachment.attach_token != handle.attach_token:
            raise _AmbiguousRegionSetupError(
                "attached region token does not match registration"
            )
        if attachment.fd < 0:
            raise _AmbiguousRegionSetupError(
                "attached region returned an invalid file descriptor"
            )
        try:
            attached_size = int(os.fstat(attachment.fd).st_size)
        except OSError as exc:
            raise _AmbiguousRegionSetupError(
                "attached region file descriptor cannot be inspected"
            ) from exc
        if attached_size != capacity_bytes:
            raise _AmbiguousRegionSetupError(
                "attached region file size does not match registration"
            )

    def _rollback_building_record(self, record: _RegionAllocationRecord) -> bool:
        with self._region_lock:
            if (
                record.lifecycle is not _RegionAllocationLifecycle.BUILDING
                or record.view_escaped
                or record.data_rpc_used
                or record.rollback_attempted
            ):
                return False
            record.rollback_attempted = True
            record.tensor_root = None
            record.storage_root = None

        mapped_region = record.mapped_region
        if mapped_region is not None:
            try:
                mapped_region.close()
            except (BufferError, OSError, ValueError):
                return False
            with self._region_lock:
                record.mapped_region = None

        file_descriptor = record.file_descriptor
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                return False
            with self._region_lock:
                record.file_descriptor = None

        try:
            released = self._client.release_host_shared_region(record.handle)
        except Exception:
            return False
        if not released:
            return False
        try:
            unregistered = self._client.unregister_region(record.handle.region_id)
        except Exception:
            return False
        if not unregistered:
            return False

        with self._region_lock:
            record.lifecycle = _RegionAllocationLifecycle.ROLLED_BACK
        return True

    def _create_region_tensor(
        self,
        request: _AllocationRequest,
        *,
        region_class: HostSharedRegionClass,
    ) -> tuple[_RegionAllocationRecord, torch.Tensor]:
        allocation_sequence, region_name = self._allocate_sequence_and_name(request)
        try:
            handle = self._client.register_region(
                memory_kind=RegionMemoryKind.HOST_SHARED,
                size_bytes=request.byte_length,
                ttl_ms=0,
                daemon_managed=True,
                host_shared_region_class=region_class,
                region_name=region_name,
            )
        except BaseException as exc:
            raise _AmbiguousRegionSetupError(
                "daemon region registration did not complete reliably"
            ) from exc

        record = self._retain_building_record(
            allocation_sequence=allocation_sequence,
            handle=handle,
            capacity_bytes=request.byte_length,
        )
        self._validate_region_handle(
            handle,
            capacity_bytes=request.byte_length,
            region_class=region_class,
        )

        try:
            attachment = self._client.attach_host_shared_region(handle)
        except BaseException as exc:
            raise _AmbiguousRegionSetupError(
                "local region attachment did not complete reliably"
            ) from exc
        with self._region_lock:
            record.attachment = attachment
            record.file_descriptor = attachment.fd
        self._validate_allocator_attachment(
            attachment,
            handle=handle,
            capacity_bytes=request.byte_length,
        )

        tensor: torch.Tensor | None = None
        try:
            mapped_region = _SHARED_REGION_MAPPER(
                attachment.fd,
                request.byte_length,
            )
            with self._region_lock:
                record.mapped_region = mapped_region
            tensor = _MAPPED_TENSOR_FACTORY(
                mapped_region,
                request.shape,
                request.dtype,
                request.element_count,
            )
            with self._region_lock:
                record.tensor_root = tensor
                record.storage_root = tensor.untyped_storage()

            tensor_byte_length = int(tensor.numel()) * int(tensor.element_size())
            base_address = int(tensor.data_ptr())
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise RuntimeError("mapped tensor is not dense contiguous CPU memory")
            if tensor.shape != request.shape or tensor.dtype is not request.dtype:
                raise RuntimeError(
                    "mapped tensor shape or dtype does not match request"
                )
            if tensor_byte_length != request.byte_length:
                raise RuntimeError("mapped tensor byte length does not match request")
            if len(mapped_region) != request.byte_length:
                raise RuntimeError("shared mapping does not cover the complete region")
            if base_address <= 0 or base_address % mmap.PAGESIZE != 0:
                raise RuntimeError("shared mapping base is not page-aligned")
            if request.byte_length > _MAX_POINTER_VALUE + 1 - base_address:
                raise RuntimeError("shared mapping address range overflows")
        except BaseException as exc:
            tensor = None
            with self._region_lock:
                record.tensor_root = None
                record.storage_root = None
            if self._rollback_building_record(record):
                raise _RolledBackRegionAllocationError(
                    "local tensor construction failed and the region was rolled back"
                ) from exc
            raise _AmbiguousRegionSetupError(
                "local tensor construction failed without an exact region rollback"
            ) from exc

        with self._region_lock:
            record.base_address = base_address
            record.view_escaped = True
            record.lifecycle = _RegionAllocationLifecycle.PROCESS_PINNED
        return record, tensor

    def _create_allocator_tensor(
        self,
        request: _AllocationRequest,
    ) -> torch.Tensor:
        _, tensor = self._create_region_tensor(
            request,
            region_class=HostSharedRegionClass.ALLOCATOR,
        )
        return tensor

    def _resolve_allocation_span(
        self,
        span: HostMemorySpan,
    ) -> _ResolvedRegionSpan:
        span_start = span.address
        span_length = span.byte_length
        if span_length > _MAX_POINTER_VALUE + 1 - span_start:
            raise RegionArtifactInputError("host span address range overflows")
        span_end = span_start + span_length
        with self._region_lock:
            matches = tuple(
                record
                for record in self._allocation_records
                if record.lifecycle is _RegionAllocationLifecycle.PROCESS_PINNED
                and record.base_address is not None
                and record.base_address <= span_start
                and span_end <= record.base_address + record.capacity_bytes
            )
        if not matches:
            raise RegionArtifactInputError(
                "host span is not fully contained in a Session allocation"
            )
        if len(matches) != 1:
            raise RegionArtifactInputError(
                "host span is ambiguously contained in multiple Session allocations"
            )
        record = matches[0]
        assert record.base_address is not None
        return _ResolvedRegionSpan(
            record=record,
            region_offset=span_start - record.base_address,
        )

    def allocate_host_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        name: str,
    ) -> torch.Tensor:
        """Allocate one process-pinned tensor from a daemon-managed host region."""
        self._preliminary_state_gate()
        try:
            if self._options.transfer.mode is not RegionTransferMode.ALLOCATOR:
                raise RegionArtifactInputError(
                    "allocate_host_tensor is available only in allocator mode"
                )
            request = _validate_allocation_request(shape, dtype, name)
        except RegionArtifactInputError:
            self._preliminary_state_gate()
            raise

        operation_id = self._final_rpc_admission()
        try:
            tensor = self._create_allocator_tensor(request)
        except _RolledBackRegionAllocationError as exc:
            failure = self._complete_rpc_admission()
            if failure is not None:
                raise RegionSessionFailedError(failure) from exc
            raise RegionArtifactInputError(str(exc)) from exc
        except BaseException as exc:
            failure = self._latch_region_setup_failure(
                operation_id=operation_id,
                cause=exc,
            )
            self._complete_rpc_admission()
            raise RegionSessionFailedError(failure) from exc

        failure = self._complete_rpc_admission()
        if failure is not None:
            raise RegionSessionFailedError(failure)
        return tensor

    def _scratch_capacity_bytes(self) -> int:
        transfer = self._options.transfer
        if not isinstance(transfer, ScratchTransferOptions):
            raise RegionArtifactInputError(
                "scratch arenas are available only in scratch mode"
            )
        return transfer.capacity_bytes

    def _create_scratch_arena(
        self,
        operation_kind: RegionSessionOperationKind,
    ) -> _RegionAllocationRecord:
        capacity_bytes = self._scratch_capacity_bytes()
        direction_name = (
            "scratch_get"
            if operation_kind is RegionSessionOperationKind.GET_INTO
            else "scratch_put"
        )
        request = _AllocationRequest(
            shape=(capacity_bytes,),
            dtype=torch.uint8,
            element_count=capacity_bytes,
            byte_length=capacity_bytes,
            diagnostic_name=direction_name,
        )
        record, _ = self._create_region_tensor(
            request,
            region_class=HostSharedRegionClass.SCRATCH,
        )
        return record

    def _ensure_scratch_arena(
        self,
        operation_kind: RegionSessionOperationKind,
    ) -> _RegionAllocationRecord:
        if operation_kind is RegionSessionOperationKind.GET_INTO:
            existing = self._scratch_get_record
        elif operation_kind is RegionSessionOperationKind.PUT_FROM:
            existing = self._scratch_put_record
        else:
            raise ValueError("scratch arena requires get or put operation kind")
        if existing is not None:
            return existing

        operation_id = self._final_rpc_admission()
        try:
            record = self._create_scratch_arena(operation_kind)
        except _RolledBackRegionAllocationError as exc:
            failure = self._complete_rpc_admission()
            if failure is not None:
                raise RegionSessionFailedError(failure) from exc
            raise RegionArtifactInputError(str(exc)) from exc
        except BaseException as exc:
            failure = self._latch_region_setup_failure(
                operation_id=operation_id,
                cause=exc,
                operation_kind=operation_kind,
            )
            self._complete_rpc_admission()
            raise RegionSessionFailedError(failure) from exc

        if operation_kind is RegionSessionOperationKind.GET_INTO:
            self._scratch_get_record = record
        else:
            self._scratch_put_record = record
        failure = self._complete_rpc_admission()
        if failure is not None:
            raise RegionSessionFailedError(failure)
        return record

    def _compile_scratch_transfer(
        self,
        compiled_transfers: tuple[_CompiledTransfer, ...],
        *,
        record: _RegionAllocationRecord,
        operation_kind: RegionSessionOperationKind,
    ) -> _PreparedScratchTransfer:
        prepared_layout = _compile_scratch_layout(compiled_transfers, record)
        if operation_kind is RegionSessionOperationKind.GET_INTO:
            get_request = store_daemon_pb2.BatchGetIntoRegionRequest(
                target_layout=prepared_layout.layout,
                pid=self._owner_pid,
                device_uuid="",
                operation_id=_WIRE_OPERATION_ID_PLACEHOLDER,
            )
            for compiled in compiled_transfers:
                get_request.selections.add().CopyFrom(compiled.artifact.selection)
            request: (
                store_daemon_pb2.BatchGetIntoRegionRequest
                | store_daemon_pb2.BatchPutIfAbsentFromRegionRequest
            ) = get_request
        elif operation_kind is RegionSessionOperationKind.PUT_FROM:
            put_request = store_daemon_pb2.BatchPutIfAbsentFromRegionRequest(
                source_layout=prepared_layout.layout,
                pid=self._owner_pid,
                device_uuid="",
                operation_id=_WIRE_OPERATION_ID_PLACEHOLDER,
            )
            for compiled in compiled_transfers:
                item = put_request.items.add()
                item.selection.CopyFrom(compiled.artifact.selection)
                item.invariant.layout_id = compiled.artifact.spec.keyspace.layout_id
                item.invariant.byte_length = compiled.artifact.spec.byte_length
                item.invariant.verification_mode = store_daemon_pb2.BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY
            request = put_request
        else:
            raise ValueError("scratch compilation requires get or put operation kind")

        expected = tuple(
            _ExpectedOutcome(artifact_id=compiled.artifact.artifact_id)
            for compiled in compiled_transfers
        )
        _require_transfer_wire_budget(
            request,
            expected,
            limits=self._client._effective_grpc_message_limits,
            direct=False,
        )
        return _PreparedScratchTransfer(
            operation_kind=operation_kind,
            compiled_transfers=compiled_transfers,
            record=record,
            request=request,
            packed_offsets=prepared_layout.packed_offsets,
        )

    def _admit_scratch_transfer(
        self,
        prepared: _PreparedScratchTransfer,
    ) -> str:
        operation_id = self._final_rpc_admission()
        prepared.operation_id = operation_id
        prepared.request.operation_id = operation_id
        with self._region_lock:
            prepared.record.data_rpc_used = True
        return operation_id

    def _record_scratch_transfer_metrics(
        self,
        prepared: _PreparedScratchTransfer,
        result: RegionArtifactTransferResult,
    ) -> None:
        observation = _ScratchTransferMetrics(
            direction=prepared.operation_kind,
            artifact_count=len(prepared.compiled_transfers),
            byte_count=sum(
                compiled.artifact.spec.byte_length
                for compiled in prepared.compiled_transfers
            ),
            pack_elapsed_s=result.pack_elapsed_s,
            copy_elapsed_s=result.copy_elapsed_s,
            rpc_elapsed_s=result.rpc_elapsed_s,
        )
        with self._region_lock:
            self._scratch_metrics[prepared.operation_kind] = observation

    @staticmethod
    def _prepare_exists(
        artifacts: tuple[ByteArtifactSpec, ...],
    ) -> tuple[_CompiledArtifact, ...]:
        return _lower_artifacts(artifacts)

    def _compile_direct_transfers(
        self,
        transfers: tuple[RegionArtifactTransfer, ...],
        *,
        operation_kind: RegionSessionOperationKind,
    ) -> _PreparedDirectTransfer:
        if operation_kind not in (
            RegionSessionOperationKind.GET_INTO,
            RegionSessionOperationKind.PUT_FROM,
        ):
            raise ValueError("direct compilation requires get or put operation kind")
        compiled_transfers = _lower_transfers(transfers)
        try:
            resolutions = tuple(
                self._resolve_allocation_span(compiled.transfer.span)
                for compiled in compiled_transfers
            )
        except RegionArtifactInputError:
            with self._observability_lock:
                self._observability.address_resolution_failure_count += 1
            raise

        unique_records: list[_RegionAllocationRecord] = []
        record_by_region_id: dict[str, _RegionAllocationRecord] = {}
        for resolution in resolutions:
            record = resolution.record
            region_id = record.handle.region_id
            existing = record_by_region_id.get(region_id)
            if existing is None:
                record_by_region_id[region_id] = record
                unique_records.append(record)
            elif existing is not record:
                raise RegionArtifactInputError(
                    "multiple allocation records share one daemon region id"
                )

        logical_base_by_region_id: dict[str, int] = {}
        logical_cursor = 0
        for record in unique_records:
            if record.capacity_bytes > _MAX_UINT64 - logical_cursor:
                raise RegionArtifactInputError(
                    "allocator layout storage capacities overflow uint64"
                )
            logical_base_by_region_id[record.handle.region_id] = logical_cursor
            logical_cursor += record.capacity_bytes

        geometry_lengths: dict[str, set[int]] = {
            record.handle.region_id: set() for record in unique_records
        }
        geometry_offsets: dict[str, list[int]] = {
            record.handle.region_id: [] for record in unique_records
        }
        for compiled, resolution in zip(
            compiled_transfers,
            resolutions,
            strict=True,
        ):
            region_id = resolution.record.handle.region_id
            geometry_lengths[region_id].add(compiled.artifact.spec.byte_length)
            geometry_offsets[region_id].append(resolution.region_offset)

        geometry_candidates: list[_GeometryCandidate] = []
        try:
            for record in unique_records:
                region_id = record.handle.region_id
                lengths = geometry_lengths[region_id]
                if len(lengths) != 1:
                    raise RegionArtifactInputError(
                        "all transfers in one allocator region must have equal length"
                    )
                slot_bytes = next(iter(lengths))
                if record.capacity_bytes % slot_bytes != 0:
                    raise RegionArtifactInputError(
                        "allocator region capacity is not aligned to transfer length"
                    )
                if any(
                    offset % slot_bytes != 0 for offset in geometry_offsets[region_id]
                ):
                    raise RegionArtifactInputError(
                        "allocator transfer offset is not aligned to transfer length"
                    )
                geometry_candidates.append(
                    _GeometryCandidate(record=record, slot_bytes=slot_bytes)
                )
        except RegionArtifactInputError:
            with self._observability_lock:
                self._observability.geometry_validation_failure_count += 1
            raise

        layout = _new_byte_artifact_layout()
        for storage_index, record in enumerate(unique_records):
            _add_host_shared_storage(
                layout,
                storage_id=f"storage-{storage_index}",
                record=record,
            )
        storage_id_by_region_id = {
            record.handle.region_id: f"storage-{storage_index}"
            for storage_index, record in enumerate(unique_records)
        }
        slot_bytes_by_region_id = {
            candidate.record.handle.region_id: candidate.slot_bytes
            for candidate in geometry_candidates
        }
        region_offsets: list[int] = []
        for compiled, resolution in zip(
            compiled_transfers,
            resolutions,
            strict=True,
        ):
            region_id = resolution.record.handle.region_id
            slot_bytes = slot_bytes_by_region_id[region_id]
            logical_offset = (
                logical_base_by_region_id[region_id] + resolution.region_offset
            )
            if logical_offset > _MAX_UINT64:
                raise RegionArtifactInputError(
                    "allocator transfer logical offset overflows uint64"
                )
            layout.offsets.add(
                name=compiled.artifact.artifact_id,
                storage_id=storage_id_by_region_id[region_id],
                storage_offset=logical_offset,
                logical_length=compiled.artifact.spec.byte_length,
                slot_index=resolution.region_offset // slot_bytes,
                slot_generation=_MAX_UINT64,
            )
            region_offsets.append(resolution.region_offset)

        if operation_kind is RegionSessionOperationKind.GET_INTO:
            get_request = store_daemon_pb2.BatchGetIntoRegionRequest(
                target_layout=layout,
                pid=self._owner_pid,
                device_uuid="",
                operation_id=_WIRE_OPERATION_ID_PLACEHOLDER,
            )
            for compiled in compiled_transfers:
                get_request.selections.add().CopyFrom(compiled.artifact.selection)
            request: (
                store_daemon_pb2.BatchGetIntoRegionRequest
                | store_daemon_pb2.BatchPutIfAbsentFromRegionRequest
            ) = get_request
        else:
            request = store_daemon_pb2.BatchPutIfAbsentFromRegionRequest(
                source_layout=layout,
                pid=self._owner_pid,
                device_uuid="",
                operation_id=_WIRE_OPERATION_ID_PLACEHOLDER,
            )
            for compiled in compiled_transfers:
                item = request.items.add()
                item.selection.CopyFrom(compiled.artifact.selection)
                item.invariant.layout_id = compiled.artifact.spec.keyspace.layout_id
                item.invariant.byte_length = compiled.artifact.spec.byte_length
                item.invariant.verification_mode = store_daemon_pb2.BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY

        expected = tuple(
            _ExpectedOutcome(
                artifact_id=compiled.artifact.artifact_id,
                slot_index=int(offset.slot_index),
                slot_generation=_MAX_UINT64,
            )
            for compiled, offset in zip(
                compiled_transfers,
                layout.offsets,
                strict=True,
            )
        )
        _require_transfer_wire_budget(
            request,
            expected,
            limits=self._client._effective_grpc_message_limits,
            direct=True,
        )
        return _PreparedDirectTransfer(
            operation_kind=operation_kind,
            compiled_transfers=compiled_transfers,
            request=request,
            geometry_candidates=tuple(geometry_candidates),
            region_offsets=tuple(region_offsets),
        )

    def _admit_direct_transfer(
        self,
        prepared: _PreparedDirectTransfer,
    ) -> tuple[str, int]:
        generation: int | None = None
        exhausted_failure: RegionSessionFailure | None = None
        first_exhausted_failure = False
        with self._state_lock:
            self._raise_unavailable_locked()
            with self._geometry_lock:
                for candidate in prepared.geometry_candidates:
                    existing = candidate.record.slot_geometry
                    if (
                        existing is not None
                        and existing.slot_bytes != candidate.slot_bytes
                    ):
                        raise RegionArtifactInputError(
                            "allocator region already has incompatible slot geometry"
                        )
                with self._generation_lock:
                    if self._next_rpc_generation > _MAX_UINT64:
                        failure = RegionSessionFailure(
                            code=RegionSessionFailureCode.INTERNAL,
                            message="direct transfer RPC generation is exhausted",
                            operation_kind=prepared.operation_kind,
                            operation_id=None,
                            occurred_at=datetime.now(timezone.utc),
                        )
                        if self._failure is None:
                            self._failure = failure
                            self._health = RegionSessionHealth.FAILED
                            first_exhausted_failure = True
                            with self._observability_lock:
                                self._observability.health_transition_count += 1
                        assert self._failure is not None
                        exhausted_failure = self._failure
                    else:
                        generation = self._next_rpc_generation
                        self._next_rpc_generation += 1
                if exhausted_failure is None:
                    for candidate in prepared.geometry_candidates:
                        if candidate.record.slot_geometry is None:
                            candidate.record.slot_geometry = _RegionSlotGeometry(
                                slot_bytes=candidate.slot_bytes
                            )
            if exhausted_failure is None:
                operation_id = uuid.uuid4().hex
                self._in_flight_count += 1
                with self._observability_lock:
                    self._observability.admitted_operation_count += 1

        if exhausted_failure is not None:
            if first_exhausted_failure:
                logger.error(
                    "region-backed artifact Session failed: code=%s "
                    "operation_kind=%s operation_id=%s",
                    exhausted_failure.code.value,
                    exhausted_failure.operation_kind.value,
                    exhausted_failure.operation_id,
                )
            raise RegionSessionFailedError(exhausted_failure)
        assert generation is not None

        with self._region_lock:
            for candidate in prepared.geometry_candidates:
                candidate.record.data_rpc_used = True
        prepared.operation_id = operation_id
        prepared.slot_generation = generation
        prepared.request.operation_id = operation_id
        for offset in prepared.request_layout.offsets:
            offset.slot_generation = generation
        return operation_id, generation

    def _latch_operational_failure(
        self,
        *,
        operation_kind: RegionSessionOperationKind,
        operation_id: str,
        cause: BaseException,
    ) -> RegionSessionFailure:
        detail = str(cause).strip() or cause.__class__.__name__
        return self._latch_failure(
            RegionSessionFailure(
                code=_failure_code_for_exception(cause),
                message=f"{operation_kind.value} operation failed: {detail}",
                operation_kind=operation_kind,
                operation_id=operation_id,
                occurred_at=datetime.now(timezone.utc),
            ),
            cause=cause,
        )

    def _execute_exists(
        self,
        partition: _ExistsPartition,
        operation_id: str,
    ) -> tuple[tuple[bool, ...], float]:
        rpc_start = time.monotonic()
        response = self._client.batch_exists(
            selections=partition.request.selections,
            timeout_s=self._options.exists_timeout_s,
            operation_id=operation_id,
        )
        rpc_elapsed_s = time.monotonic() - rpc_start
        expected = tuple(
            _ExpectedOutcome(artifact_id=artifact.artifact_id)
            for artifact in partition.compiled_artifacts
        )
        success_mask = _validate_batch_outcomes(
            response.outcomes,
            expected,
            operation_kind=RegionSessionOperationKind.EXISTS,
        )
        return success_mask, rpc_elapsed_s

    def _execute_get_into(
        self,
        prepared: _PreparedDirectTransfer,
    ) -> RegionArtifactTransferResult:
        if not isinstance(
            prepared.request,
            store_daemon_pb2.BatchGetIntoRegionRequest,
        ):
            raise RuntimeError("direct get received a put request")
        if prepared.operation_id is None or prepared.slot_generation is None:
            raise RuntimeError("direct get has not been admitted")
        rpc_started_at = time.monotonic()
        response = self._client.batch_get_into_region(
            selections=tuple(prepared.request.selections),
            target_layout=prepared.request.target_layout,
            pid=prepared.request.pid,
            device_uuid=prepared.request.device_uuid,
            operation_id=prepared.operation_id,
            timeout_s=self._options.transfer_timeout_s,
            retries=0,
        )
        rpc_elapsed_s = time.monotonic() - rpc_started_at
        success_mask = _validate_batch_outcomes(
            response.outcomes,
            prepared.expected_outcomes(),
            operation_kind=RegionSessionOperationKind.GET_INTO,
        )
        return RegionArtifactTransferResult(
            success_mask=success_mask,
            operation_id=prepared.operation_id,
            pack_elapsed_s=0.0,
            copy_elapsed_s=0.0,
            rpc_elapsed_s=rpc_elapsed_s,
        )

    def _execute_put_from(
        self,
        prepared: _PreparedDirectTransfer,
    ) -> RegionArtifactTransferResult:
        if not isinstance(
            prepared.request,
            store_daemon_pb2.BatchPutIfAbsentFromRegionRequest,
        ):
            raise RuntimeError("direct put received a get request")
        if prepared.operation_id is None or prepared.slot_generation is None:
            raise RuntimeError("direct put has not been admitted")
        rpc_started_at = time.monotonic()
        response = self._client.batch_put_if_absent_from_region(
            items=tuple(prepared.request.items),
            source_layout=prepared.request.source_layout,
            pid=prepared.request.pid,
            device_uuid=prepared.request.device_uuid,
            operation_id=prepared.operation_id,
            timeout_s=self._options.transfer_timeout_s,
            retries=0,
        )
        rpc_elapsed_s = time.monotonic() - rpc_started_at
        success_mask = _validate_batch_outcomes(
            response.outcomes,
            prepared.expected_outcomes(),
            operation_kind=RegionSessionOperationKind.PUT_FROM,
        )
        return RegionArtifactTransferResult(
            success_mask=success_mask,
            operation_id=prepared.operation_id,
            pack_elapsed_s=0.0,
            copy_elapsed_s=0.0,
            rpc_elapsed_s=rpc_elapsed_s,
        )

    def _execute_scratch_get_into(
        self,
        prepared: _PreparedScratchTransfer,
    ) -> RegionArtifactTransferResult:
        if not isinstance(
            prepared.request,
            store_daemon_pb2.BatchGetIntoRegionRequest,
        ):
            raise RuntimeError("scratch get received a put request")
        if prepared.operation_id is None:
            raise RuntimeError("scratch get has not been admitted")
        arena_address = prepared.record.base_address
        if arena_address is None:
            raise RuntimeError("scratch get arena has no mapped base address")

        rpc_started_at = time.monotonic()
        response = self._client.batch_get_into_region(
            selections=tuple(prepared.request.selections),
            target_layout=prepared.request.target_layout,
            pid=prepared.request.pid,
            device_uuid=prepared.request.device_uuid,
            operation_id=prepared.operation_id,
            timeout_s=self._options.transfer_timeout_s,
            retries=0,
        )
        rpc_elapsed_s = time.monotonic() - rpc_started_at
        success_mask = _validate_batch_outcomes(
            response.outcomes,
            prepared.expected_outcomes(),
            operation_kind=RegionSessionOperationKind.GET_INTO,
        )

        copy_started_at = time.monotonic()
        for compiled, packed_offset, succeeded in zip(
            prepared.compiled_transfers,
            prepared.packed_offsets,
            success_mask,
            strict=True,
        ):
            if succeeded:
                _HOST_MEMORY_COPY(
                    compiled.transfer.span.address,
                    arena_address + packed_offset,
                    compiled.artifact.spec.byte_length,
                )
        copy_elapsed_s = time.monotonic() - copy_started_at
        return RegionArtifactTransferResult(
            success_mask=success_mask,
            operation_id=prepared.operation_id,
            pack_elapsed_s=0.0,
            copy_elapsed_s=copy_elapsed_s,
            rpc_elapsed_s=rpc_elapsed_s,
        )

    def _execute_scratch_put_from(
        self,
        prepared: _PreparedScratchTransfer,
    ) -> RegionArtifactTransferResult:
        if not isinstance(
            prepared.request,
            store_daemon_pb2.BatchPutIfAbsentFromRegionRequest,
        ):
            raise RuntimeError("scratch put received a get request")
        if prepared.operation_id is None:
            raise RuntimeError("scratch put has not been admitted")
        arena_address = prepared.record.base_address
        if arena_address is None:
            raise RuntimeError("scratch put arena has no mapped base address")

        pack_started_at = time.monotonic()
        for compiled, packed_offset in zip(
            prepared.compiled_transfers,
            prepared.packed_offsets,
            strict=True,
        ):
            _HOST_MEMORY_COPY(
                arena_address + packed_offset,
                compiled.transfer.span.address,
                compiled.artifact.spec.byte_length,
            )
        pack_elapsed_s = time.monotonic() - pack_started_at

        rpc_started_at = time.monotonic()
        response = self._client.batch_put_if_absent_from_region(
            items=tuple(prepared.request.items),
            source_layout=prepared.request.source_layout,
            pid=prepared.request.pid,
            device_uuid=prepared.request.device_uuid,
            operation_id=prepared.operation_id,
            timeout_s=self._options.transfer_timeout_s,
            retries=0,
        )
        rpc_elapsed_s = time.monotonic() - rpc_started_at
        success_mask = _validate_batch_outcomes(
            response.outcomes,
            prepared.expected_outcomes(),
            operation_kind=RegionSessionOperationKind.PUT_FROM,
        )
        return RegionArtifactTransferResult(
            success_mask=success_mask,
            operation_id=prepared.operation_id,
            pack_elapsed_s=pack_elapsed_s,
            copy_elapsed_s=0.0,
            rpc_elapsed_s=rpc_elapsed_s,
        )

    def batch_exists(
        self,
        artifacts: Sequence[ByteArtifactSpec],
    ) -> RegionArtifactExistsResult:
        snapshot = tuple(artifacts)
        self._preliminary_state_gate()
        if not snapshot:
            return RegionArtifactExistsResult(existence_mask=(), rpc_elapsed_s=0.0)
        try:
            compiled_artifacts = self._prepare_exists(snapshot)
            partitions = _partition_exists_artifacts(
                compiled_artifacts,
                limits=self._client._effective_grpc_message_limits,
            )
        except RegionArtifactInputError:
            self._preliminary_state_gate()
            raise

        ordered_mask: list[bool] = []
        rpc_elapsed_s = 0.0
        for partition in partitions:
            operation_id = self._final_rpc_admission()
            try:
                partition_mask, partition_elapsed_s = self._execute_exists(
                    partition,
                    operation_id,
                )
            except BaseException as exc:
                failure = self._latch_operational_failure(
                    operation_kind=RegionSessionOperationKind.EXISTS,
                    operation_id=operation_id,
                    cause=exc,
                )
                self._complete_rpc_admission()
                raise RegionSessionFailedError(failure) from exc
            post_failure = self._complete_rpc_admission()
            if post_failure is not None:
                raise RegionSessionFailedError(post_failure)
            self._record_exists_result(
                partition,
                partition_mask,
                rpc_elapsed_s=partition_elapsed_s,
            )
            ordered_mask.extend(partition_mask)
            rpc_elapsed_s += partition_elapsed_s
        return RegionArtifactExistsResult(
            existence_mask=tuple(ordered_mask),
            rpc_elapsed_s=rpc_elapsed_s,
        )

    def _run_direct_transfer(
        self,
        snapshot: tuple[RegionArtifactTransfer, ...],
        *,
        operation_kind: RegionSessionOperationKind,
        executor: Callable[[_PreparedDirectTransfer], RegionArtifactTransferResult],
    ) -> RegionArtifactTransferResult:
        try:
            prepared = self._compile_direct_transfers(
                snapshot,
                operation_kind=operation_kind,
            )
        except RegionArtifactInputError:
            self._preliminary_state_gate()
            raise

        try:
            operation_id, _ = self._admit_direct_transfer(prepared)
        except RegionArtifactInputError:
            raise
        self._record_transfer_submission(prepared, direct=True)
        try:
            result = executor(prepared)
        except BaseException as exc:
            failure = self._latch_operational_failure(
                operation_kind=operation_kind,
                operation_id=operation_id,
                cause=exc,
            )
            self._complete_rpc_admission()
            raise RegionSessionFailedError(failure) from exc
        post_failure = self._complete_rpc_admission()
        if post_failure is not None:
            raise RegionSessionFailedError(post_failure)
        self._record_transfer_result(prepared, result, direct=True)
        return result

    def _run_scratch_transfer(
        self,
        snapshot: tuple[RegionArtifactTransfer, ...],
        *,
        operation_kind: RegionSessionOperationKind,
    ) -> RegionArtifactTransferResult:
        try:
            compiled_transfers = _lower_transfers(snapshot)
            _validate_scratch_capacity(
                compiled_transfers,
                capacity_bytes=self._scratch_capacity_bytes(),
            )
        except RegionArtifactInputError:
            self._preliminary_state_gate()
            raise

        direction_lock = (
            self._scratch_get_lock
            if operation_kind is RegionSessionOperationKind.GET_INTO
            else self._scratch_put_lock
        )
        with direction_lock:
            self._preliminary_state_gate()
            try:
                record = self._ensure_scratch_arena(operation_kind)
                prepared = self._compile_scratch_transfer(
                    compiled_transfers,
                    record=record,
                    operation_kind=operation_kind,
                )
            except RegionArtifactInputError:
                self._preliminary_state_gate()
                raise

            operation_id = self._admit_scratch_transfer(prepared)
            self._record_transfer_submission(prepared, direct=False)
            try:
                if operation_kind is RegionSessionOperationKind.GET_INTO:
                    result = self._execute_scratch_get_into(prepared)
                else:
                    result = self._execute_scratch_put_from(prepared)
            except BaseException as exc:
                failure = self._latch_operational_failure(
                    operation_kind=operation_kind,
                    operation_id=operation_id,
                    cause=exc,
                )
                self._complete_rpc_admission()
                raise RegionSessionFailedError(failure) from exc
            post_failure = self._complete_rpc_admission()
            if post_failure is not None:
                raise RegionSessionFailedError(post_failure)
            self._record_scratch_transfer_metrics(prepared, result)
            self._record_transfer_result(prepared, result, direct=False)
            return result

    def _record_exists_result(
        self,
        partition: _ExistsPartition,
        success_mask: tuple[bool, ...],
        *,
        rpc_elapsed_s: float,
    ) -> None:
        with self._observability_lock:
            counters = self._observability
            counters.artifact_count += len(partition.compiled_artifacts)
            counters.rpc_elapsed_s += rpc_elapsed_s
            counters.successful_item_count += sum(success_mask)
            counters.missed_item_count += len(success_mask) - sum(success_mask)

    def _record_transfer_submission(
        self,
        prepared: _PreparedDirectTransfer | _PreparedScratchTransfer,
        *,
        direct: bool,
    ) -> None:
        byte_count = sum(
            compiled.artifact.spec.byte_length
            for compiled in prepared.compiled_transfers
        )
        if direct:
            assert isinstance(prepared, _PreparedDirectTransfer)
            unique_region_count = len(prepared.geometry_candidates)
        else:
            unique_region_count = 1
        with self._observability_lock:
            counters = self._observability
            counters.artifact_count += len(prepared.compiled_transfers)
            counters.total_transfer_bytes += byte_count
            counters.unique_region_reference_count += unique_region_count
            counters.suppressed_transparent_retry_count += 1
            if prepared.operation_kind is RegionSessionOperationKind.GET_INTO:
                counters.get_transfer_count += 1
            else:
                counters.put_transfer_count += 1
            if direct:
                counters.direct_bytes_submitted += byte_count

    def _record_transfer_result(
        self,
        prepared: _PreparedDirectTransfer | _PreparedScratchTransfer,
        result: RegionArtifactTransferResult,
        *,
        direct: bool,
    ) -> None:
        successful_item_count = sum(result.success_mask)
        scratch_bytes_copied = 0
        if not direct:
            if prepared.operation_kind is RegionSessionOperationKind.PUT_FROM:
                scratch_bytes_copied = sum(
                    compiled.artifact.spec.byte_length
                    for compiled in prepared.compiled_transfers
                )
            else:
                scratch_bytes_copied = sum(
                    compiled.artifact.spec.byte_length
                    for compiled, succeeded in zip(
                        prepared.compiled_transfers,
                        result.success_mask,
                        strict=True,
                    )
                    if succeeded
                )
        with self._observability_lock:
            counters = self._observability
            counters.scratch_bytes_copied += scratch_bytes_copied
            counters.rpc_elapsed_s += result.rpc_elapsed_s
            counters.pack_elapsed_s += result.pack_elapsed_s
            counters.copy_elapsed_s += result.copy_elapsed_s
            counters.successful_item_count += successful_item_count
            counters.missed_item_count += (
                len(result.success_mask) - successful_item_count
            )

    def batch_get_into(
        self,
        transfers: Sequence[RegionArtifactTransfer],
    ) -> RegionArtifactTransferResult:
        snapshot = tuple(transfers)
        self._preliminary_state_gate()
        if not snapshot:
            return RegionArtifactTransferResult(
                success_mask=(),
                operation_id=None,
                pack_elapsed_s=0.0,
                copy_elapsed_s=0.0,
                rpc_elapsed_s=0.0,
            )
        if self._options.transfer.mode is RegionTransferMode.SCRATCH:
            return self._run_scratch_transfer(
                snapshot,
                operation_kind=RegionSessionOperationKind.GET_INTO,
            )
        return self._run_direct_transfer(
            snapshot,
            operation_kind=RegionSessionOperationKind.GET_INTO,
            executor=self._execute_get_into,
        )

    def batch_put_from(
        self,
        transfers: Sequence[RegionArtifactTransfer],
    ) -> RegionArtifactTransferResult:
        snapshot = tuple(transfers)
        self._preliminary_state_gate()
        if not snapshot:
            return RegionArtifactTransferResult(
                success_mask=(),
                operation_id=None,
                pack_elapsed_s=0.0,
                copy_elapsed_s=0.0,
                rpc_elapsed_s=0.0,
            )
        if self._options.transfer.mode is RegionTransferMode.SCRATCH:
            return self._run_scratch_transfer(
                snapshot,
                operation_kind=RegionSessionOperationKind.PUT_FROM,
            )
        return self._run_direct_transfer(
            snapshot,
            operation_kind=RegionSessionOperationKind.PUT_FROM,
            executor=self._execute_put_from,
        )

    def terminate_process_session(self) -> None:
        """Close admission permanently without releasing process-pinned memory."""
        with self._state_lock:
            self._lifecycle_state = RegionSessionLifecycleState.TERMINATED
            helper_cleanups = self._take_owned_helper_cleanups_locked()
        self._run_owned_helper_cleanups(helper_cleanups)


__all__ = [
    "AllocatorTransferOptions",
    "ByteArtifactKeyspace",
    "ByteArtifactSpec",
    "HostMemorySpan",
    "RegionArtifactExistsResult",
    "RegionArtifactInputError",
    "RegionArtifactTransfer",
    "RegionArtifactTransferResult",
    "RegionBackedArtifactSession",
    "RegionBackedArtifactSessionOptions",
    "RegionSessionAttachError",
    "RegionSessionFailedError",
    "RegionSessionFailure",
    "RegionSessionFailureCode",
    "RegionSessionHealth",
    "RegionSessionLifecycleState",
    "RegionSessionOperationKind",
    "RegionSessionTerminatedError",
    "RegionTransferMode",
    "ScratchTransferOptions",
]
