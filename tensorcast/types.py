#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, field_validator

from tensorcast.common.identity import ArtifactIdKind
from tensorcast.proto.daemon.v1 import (
    store_daemon_pb2 as store_daemon_pb2,
)

# ---------------------------------------------------------------------------
# Canonical typed models used across the Python SDK in place of raw dicts
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """Daemon server runtime configuration (client-facing subset)."""

    model_config = ConfigDict(frozen=True)

    # Canonical transfer slice size
    tx_slice_bytes: int
    mem_pool_size: int
    artifact_chunk_bytes: int = 0


# ----------------------------- Handshake models ----------------------------


class CoalescedHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["coalesced"] = "coalesced"
    daemon_ipc_handle: bytes


# CPU handshake variants removed


class LeaseHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["lease"] = "lease"


Handshake = Union[
    CoalescedHandshake,
    LeaseHandshake,
]


class BeginRegisterArtifactResult(BaseModel):
    """Result for BeginRegisterArtifact, replacing mixed dict outputs.

    Always includes a concrete handshake variant to avoid optional fields.
    """

    model_config = ConfigDict(frozen=True)

    registration_id: str
    device_id: int
    total_size: int
    handshake: Handshake


class ArtifactDescriptor(BaseModel):
    """Artifact descriptor (RFC-0007) returned at commit time."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    index_multihash: str | None = None
    data_multihash: str | None = None
    schema_version: str | None = None
    encoding: str | None = None
    total_size: int
    id_kind: ArtifactIdKind = ArtifactIdKind.MI2

    @field_validator(
        "index_multihash",
        "data_multihash",
        "schema_version",
        "encoding",
        mode="before",
    )
    @classmethod
    def _empty_str_is_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("id_kind", mode="before")
    @classmethod
    def _coerce_id_kind(cls, value: object) -> ArtifactIdKind:
        if isinstance(value, ArtifactIdKind):
            return value
        if isinstance(value, str):
            upper = value.upper()
            if upper == "MI2":
                return ArtifactIdKind.MI2
            if upper == "CGID":
                return ArtifactIdKind.CGID
        if isinstance(value, int):
            if value == 1:
                return ArtifactIdKind.MI2
            if value == 2:
                return ArtifactIdKind.CGID
        raise ValueError(f"Unsupported artifact id kind: {value!r}")


class CanonicalRange(BaseModel):
    """Canonical byte range populated during view registration."""

    model_config = ConfigDict(frozen=True)

    offset: int
    length: int


class CommitResult(BaseModel):
    """Commit result with descriptor and idempotency flag."""

    model_config = ConfigDict(frozen=True)

    descriptor: ArtifactDescriptor
    existed: bool = False
    view_id: str | None = None
    view_index_json: bytes | None = None
    view_data_hash: str | None = None
    canonical_ranges: tuple[CanonicalRange, ...] = ()
    allow_partial: bool = False


# ------------------------------ Plan models --------------------------------


class PlanBase(BaseModel):
    """Abstract base for plan variants with a typed applicator to the proto."""

    model_config = ConfigDict(frozen=True)

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        raise NotImplementedError


class CoalescedPlan(PlanBase):
    # Aliases supported for coalesced semantics
    kind: Literal["coalesced", "vram_coalesced"] = "coalesced"
    max_inflight_bytes: int = 512 * 1024 * 1024
    release_on_tensor_commit: bool = True

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        co = store_daemon_pb2.CoalescedOptions()
        co.max_inflight_bytes = int(self.max_inflight_bytes)
        co.release_on_tensor_commit = bool(self.release_on_tensor_commit)
        req.coalesced.CopyFrom(co)


# CPU plan removed


class LeasePlan(PlanBase):
    kind: Literal["lease", "vram_leased"] = "lease"
    min_tensor_bytes: int = 64 * 1024
    max_tensor_count: int = 8192
    lease_bytes_limit: int = 0
    in_place: bool = False

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        lo = store_daemon_pb2.LeaseOptions()
        lo.min_tensor_bytes = int(self.min_tensor_bytes)
        lo.max_tensor_count = int(self.max_tensor_count)
        lo.lease_bytes_limit = int(self.lease_bytes_limit)
        lo.in_place = bool(self.in_place)
        req.lease.CopyFrom(lo)


Plan = Union[CoalescedPlan, LeasePlan]


# ---------------------------- Segment feed model ---------------------------


class LeaseSegment(BaseModel):
    """A single lease segment exported from CUDA IPC and fed to the daemon.

    dst_offset is mandatory to eliminate ordering assumptions and reduce
    optional branching in downstream logic.
    """

    model_config = ConfigDict(frozen=True)

    device_id: int
    cuda_ipc_handle: bytes
    base_addr: int = 0
    length: int
    dst_offset: int


class RegisterStorage(BaseModel):
    """Deduplicated storage descriptor used during registration feeds."""

    model_config = ConfigDict(frozen=True)

    storage_id: str
    device_id: int
    cuda_ipc_handle: bytes
    storage_length: int


class RegisterTensorAlias(BaseModel):
    """Logical tensor metadata referencing a deduplicated storage."""

    model_config = ConfigDict(frozen=True)

    name: str
    storage_id: str
    storage_offset: int
    logical_length: int
    shape: list[int]
    stride: list[int]
    dtype: str


__all__ = [
    "ServerConfig",
    "CoalescedHandshake",
    "LeaseHandshake",
    "Handshake",
    "BeginRegisterArtifactResult",
    "ArtifactDescriptor",
    "CanonicalRange",
    "CommitResult",
    "PlanBase",
    "CoalescedPlan",
    "LeasePlan",
    "Plan",
    "LeaseSegment",
    "RegisterStorage",
    "RegisterTensorAlias",
]
