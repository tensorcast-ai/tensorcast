#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

"""gRPC surface for memory tier operations."""

import time

import grpc

from tensorcast.global_store.models.memory_tier import (
    ChunkRange,
    MemoryTierLease,
    MemoryTierSnapshot,
)
from tensorcast.global_store.services.memory_tier_service import MemoryTierService
from tensorcast.logger import init_logger
from tensorcast.proto.memory_tier.v1 import memory_tier_pb2, memory_tier_pb2_grpc

logger = init_logger(__name__)


class MemoryTierGrpcServicer(memory_tier_pb2_grpc.MemoryTierServiceServicer):
    """Bridges protobuf requests to MemoryTierService."""

    def __init__(self, service: MemoryTierService):
        self.service = service

    def PublishMemoryTierStatus(self, request, context):
        status = request.status
        if not status.node_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "node_id is required")
        epoch_ns = status.epoch_ns or int(time.time_ns())
        snapshot = MemoryTierSnapshot(
            node_id=status.node_id,
            epoch_ns=epoch_ns,
            stable_total_bytes=status.stable_total_bytes,
            stable_used_bytes=status.stable_used_bytes,
            preemptible_total_bytes=status.preemptible_total_bytes,
            preemptible_marked_bytes=status.preemptible_marked_bytes,
            faults_per_sec=status.faults_per_sec,
            rehydrate_p99_ns=status.rehydrate_p99_ns,
            enable_preemptible=status.enable_preemptible,
            memory_tier_config_json=status.memory_tier_config_json or "{}",
        )
        self.service.publish_status(snapshot)
        return memory_tier_pb2.PublishMemoryTierStatusResponse()

    def RequestMemoryTierLease(self, request, context):
        if not request.node_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "node_id is required")
        if not request.artifact_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "artifact_id is required")
        if request.chunk_range.start < 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "chunk_range.start must be non-negative",
            )
        if request.chunk_range.count < 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "chunk_range.count must be non-negative",
            )
        if not request.chunk_ids and request.chunk_range.count == 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "chunk_range.count must be positive when chunk_ids empty",
            )
        chunk_range = ChunkRange(
            start=request.chunk_range.start,
            count=request.chunk_range.count,
        )
        lease = self.service.request_lease(
            node_id=request.node_id,
            kind=_lease_kind_to_string(request.kind),
            artifact_id=request.artifact_id,
            chunk_range=chunk_range,
            chunk_ids=list(request.chunk_ids),
            ledger_version=request.ledger_version,
            bytes_count=request.bytes,
            workload_id=request.workload_id or "default",
            request_id=request.request_id or _default_request_id(),
            issued_at_ns=request.issued_at_ns if request.issued_at_ns != 0 else None,
        )
        return memory_tier_pb2.RequestMemoryTierLeaseResponse(lease=_lease_to_pb(lease))

    def AcknowledgeMemoryTierLease(self, request, context):
        action = _ack_action_to_string(request.action)
        if action == "unknown":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "action is required")
        if not request.artifact_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "artifact_id is required")
        if action in {"acquired", "released"} and not request.chunk_ids:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "chunk_ids are required when acknowledging acquisition or release",
            )
        lease = self.service.acknowledge(
            lease_id=request.lease_id,
            action=action,
            artifact_id=request.artifact_id,
            chunk_ids=list(request.chunk_ids),
            ledger_version=request.ledger_version,
            chunk_range=ChunkRange(
                start=request.chunk_range.start, count=request.chunk_range.count
            ),
            bytes_count=request.bytes,
            request_id=request.request_id or _default_request_id(),
            ack_epoch_ns=request.ack_epoch_ns if request.ack_epoch_ns != 0 else None,
        )
        if lease is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "lease not found")
            raise RuntimeError("unreachable")  # context.abort raises
        return memory_tier_pb2.AcknowledgeMemoryTierLeaseResponse(
            lease=_lease_to_pb(lease)
        )

    def RevokeMemoryTierLease(self, request, context):
        lease = self.service.revoke(request.lease_id)
        if lease is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "lease not found")
            raise RuntimeError("unreachable")  # context.abort raises
        return memory_tier_pb2.RevokeMemoryTierLeaseResponse(lease=_lease_to_pb(lease))

    def ListOutstandingLeases(self, request, context):
        states = (
            [_lease_state_to_string(s) for s in request.states]
            if request.states
            else None
        )
        leases = self.service.list_outstanding(request.node_id, states)
        resp = memory_tier_pb2.ListOutstandingLeasesResponse()
        resp.leases.extend([_lease_to_pb(lease) for lease in leases])
        return resp

    def ListMemoryTierStatuses(self, request, context):
        snapshots = self.service.list_latest(
            request.node_id if request.node_id else None
        )
        resp = memory_tier_pb2.ListMemoryTierStatusesResponse()
        resp.statuses.extend([_snapshot_to_pb(snapshot) for snapshot in snapshots])
        return resp


def _lease_to_pb(lease: MemoryTierLease) -> memory_tier_pb2.MemoryTierLease:
    chunk_range_pb = memory_tier_pb2.ChunkRange(
        start=lease.chunk_range.start, count=lease.chunk_range.count
    )
    return memory_tier_pb2.MemoryTierLease(
        lease_id=lease.lease_id,
        node_id=lease.node_id,
        kind=_lease_kind_from_string(lease.kind),
        artifact_id=lease.artifact_id,
        chunk_range=chunk_range_pb,
        chunk_ids=list(lease.chunk_ids),
        ledger_version=lease.ledger_version,
        bytes=lease.bytes,
        workload_id=lease.workload_id,
        state=_lease_state_from_string(lease.state),
        request_id=lease.request_id,
        ack_epoch_ns=lease.ack_epoch_ns or 0,
        issued_at_ns=lease.issued_at_ns,
        expires_at_ns=lease.expires_at_ns or 0,
    )


def _lease_kind_to_string(kind: int) -> str:
    if kind == memory_tier_pb2.LEASE_KIND_STABLE:
        return "stable"
    if kind == memory_tier_pb2.LEASE_KIND_PREEMPTIBLE:
        return "preemptible"
    return "stable"


def _lease_kind_from_string(kind: str) -> memory_tier_pb2.LeaseKind:
    if kind == "preemptible":
        return memory_tier_pb2.LEASE_KIND_PREEMPTIBLE
    return memory_tier_pb2.LEASE_KIND_STABLE


def _lease_state_from_string(state: str) -> memory_tier_pb2.LeaseState:
    mapping = {
        "pending": memory_tier_pb2.LEASE_STATE_PENDING,
        "active": memory_tier_pb2.LEASE_STATE_ACTIVE,
        "revoking": memory_tier_pb2.LEASE_STATE_REVOKING,
        "expired": memory_tier_pb2.LEASE_STATE_EXPIRED,
    }
    return mapping.get(state, memory_tier_pb2.LEASE_STATE_PENDING)


def _lease_state_to_string(state: memory_tier_pb2.LeaseState) -> str:
    mapping = {
        memory_tier_pb2.LEASE_STATE_PENDING: "pending",
        memory_tier_pb2.LEASE_STATE_ACTIVE: "active",
        memory_tier_pb2.LEASE_STATE_REVOKING: "revoking",
        memory_tier_pb2.LEASE_STATE_EXPIRED: "expired",
    }
    return mapping.get(state, "pending")


def _ack_action_to_string(action: memory_tier_pb2.LeaseAckAction) -> str:
    if action == memory_tier_pb2.LEASE_ACK_ACTION_ACQUIRED:
        return "acquired"
    if action == memory_tier_pb2.LEASE_ACK_ACTION_RELEASED:
        return "released"
    return "unknown"


def _default_request_id() -> str:
    return f"mt_req_{int(time.time_ns())}"


def _snapshot_to_pb(snapshot: MemoryTierSnapshot) -> memory_tier_pb2.MemoryTierStatus:
    return memory_tier_pb2.MemoryTierStatus(
        node_id=snapshot.node_id,
        stable_total_bytes=snapshot.stable_total_bytes,
        stable_used_bytes=snapshot.stable_used_bytes,
        preemptible_total_bytes=snapshot.preemptible_total_bytes,
        preemptible_marked_bytes=snapshot.preemptible_marked_bytes,
        faults_per_sec=snapshot.faults_per_sec,
        rehydrate_p99_ns=snapshot.rehydrate_p99_ns,
        enable_preemptible=snapshot.enable_preemptible,
        memory_tier_config_json=snapshot.memory_tier_config_json,
        epoch_ns=snapshot.epoch_ns,
    )
