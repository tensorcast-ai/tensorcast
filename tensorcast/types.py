#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict

from tensorcast.proto.daemon.v1 import (
    store_daemon_pb2 as store_daemon_pb2,
)

# ---------------------------------------------------------------------------
# Canonical typed models used across the Python SDK in place of raw dicts
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """Daemon server runtime configuration (client-facing subset)."""

    model_config = ConfigDict(frozen=True)

    chunk_size: int
    mem_pool_size: int


# ----------------------------- Handshake models ----------------------------


class CoalescedHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["coalesced"] = "coalesced"
    daemon_ipc_handle: bytes


class CpuRingHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["cpu_ring"] = "cpu_ring"
    name: str
    ring_bytes: int


class CpuStreamHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["cpu_stream"] = "cpu_stream"
    stream_token: str


class CpuEmptyHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["cpu"] = "cpu"


class LeaseHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["lease"] = "lease"


Handshake = Union[
    CoalescedHandshake,
    CpuRingHandshake,
    CpuStreamHandshake,
    CpuEmptyHandshake,
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
    index_multihash: str
    data_multihash: str
    schema_version: str
    encoding: str
    total_size: int


class CommitResult(BaseModel):
    """Commit result with descriptor and idempotency flag."""

    model_config = ConfigDict(frozen=True)

    descriptor: ArtifactDescriptor
    existed: bool = False


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


class CpuPlan(PlanBase):
    # UMA/CPU VS streaming pathway
    kind: Literal["cpu", "uma"] = "cpu"
    # 1: SHM_RING, 2: GRPC_STREAM (daemon enum mapping handled by client)
    preferred_channel: Literal[1, 2] = 2
    ring_bytes: int = 0

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        dv = store_daemon_pb2.CpuOptions()
        if int(self.preferred_channel) == 1:
            dv.preferred_channel = store_daemon_pb2.CpuOptions.Channel.CHANNEL_SHM_RING
            if int(self.ring_bytes) > 0:
                dv.ring_bytes = int(self.ring_bytes)
        else:
            dv.preferred_channel = (
                store_daemon_pb2.CpuOptions.Channel.CHANNEL_GRPC_STREAM
            )
        req.cpu.CopyFrom(dv)


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


Plan = Union[CoalescedPlan, CpuPlan, LeasePlan]


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


__all__ = [
    "ServerConfig",
    "CoalescedHandshake",
    "CpuRingHandshake",
    "CpuStreamHandshake",
    "CpuEmptyHandshake",
    "LeaseHandshake",
    "Handshake",
    "BeginRegisterArtifactResult",
    "ArtifactDescriptor",
    "CommitResult",
    "PlanBase",
    "CoalescedPlan",
    "CpuPlan",
    "LeasePlan",
    "Plan",
    "LeaseSegment",
]
