#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from datetime import datetime
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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


class StableDramHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["dram_stable"] = "dram_stable"
    staging_cuda_ipc_handle: bytes = b""


Handshake = Union[
    CoalescedHandshake,
    LeaseHandshake,
    StableDramHandshake,
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


LocalStableTierStatus = Literal["ready", "degraded", "skipped"]


class LocalStableTierResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: LocalStableTierStatus
    message: str | None = None


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
    local_stable_tier: LocalStableTierResult | None = None


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


class StableDramPlan(PlanBase):
    kind: Literal["dram_stable"] = "dram_stable"
    stage_on_gpu: bool = True
    release_gpu_on_commit: bool = True

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        opts = store_daemon_pb2.StableDramOptions()
        opts.stage_on_gpu = bool(self.stage_on_gpu)
        opts.release_gpu_on_commit = bool(self.release_gpu_on_commit)
        req.stable_dram.CopyFrom(opts)


Plan = Union[CoalescedPlan, LeasePlan, StableDramPlan]


# ---------------------------- Segment feed model ---------------------------


class LeaseSegment(BaseModel):
    """A single lease segment exported from CUDA IPC and fed to the daemon.

    A LeaseSegment maps a physical storage window (referenced by storage_id) into
    the artifact's logical address space.
    """

    model_config = ConfigDict(frozen=True)

    storage_id: str
    storage_offset: int = 0
    artifact_offset: int
    length: int

    @model_validator(mode="after")
    def _validate_offsets(self) -> "LeaseSegment":
        if not self.storage_id:
            raise ValueError("LeaseSegment.storage_id must not be empty")
        if self.storage_offset < 0:
            raise ValueError("LeaseSegment.storage_offset must be non-negative")
        if self.artifact_offset < 0:
            raise ValueError("LeaseSegment.artifact_offset must be non-negative")
        if self.length <= 0:
            raise ValueError("LeaseSegment.length must be positive")
        return self


class RegisterStorage(BaseModel):
    """Deduplicated storage descriptor used during registration feeds."""

    model_config = ConfigDict(frozen=True)

    storage_id: str
    device_id: int
    cuda_ipc_handle: bytes | None = None
    storage_length: int
    vram_region_id: str | None = None
    mapping_base_offset: int = 0

    @model_validator(mode="after")
    def _validate_source(self) -> "RegisterStorage":
        has_handle = self.cuda_ipc_handle is not None
        has_region = self.vram_region_id is not None
        if has_handle == has_region:
            raise ValueError(
                "RegisterStorage requires exactly one of cuda_ipc_handle or vram_region_id"
            )
        if self.mapping_base_offset < 0:
            raise ValueError("RegisterStorage.mapping_base_offset must be non-negative")
        return self


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


class VramRegionHandle(BaseModel):
    """Registered VRAM region descriptor returned by the daemon."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    ttl_ms: int
    expires_at: datetime | None = None


class DeregisterArtifactOutcome(BaseModel):
    """Result of a deregister_artifact invocation."""

    model_config = ConfigDict(frozen=True)

    drained: bool
    removed: bool
    released_region_ids: tuple[str, ...] = ()
    message: str | None = None


__all__ = [
    "ServerConfig",
    "CoalescedHandshake",
    "LeaseHandshake",
    "StableDramHandshake",
    "Handshake",
    "BeginRegisterArtifactResult",
    "ArtifactDescriptor",
    "CanonicalRange",
    "CommitResult",
    "PlanBase",
    "CoalescedPlan",
    "LeasePlan",
    "StableDramPlan",
    "Plan",
    "LeaseSegment",
    "RegisterStorage",
    "RegisterTensorAlias",
    "VramRegionHandle",
    "DeregisterArtifactOutcome",
]
