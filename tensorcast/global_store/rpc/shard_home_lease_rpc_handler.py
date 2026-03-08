#  Copyright (c) 2025-2026, TensorCast Team.

"""Shard-home lease RPC handler."""

from __future__ import annotations

from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.services.shard_home_lease_service import (
    ShardHomeLeaseService,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class ShardHomeLeaseRpcHandler:
    def __init__(
        self,
        *,
        shard_home_lease_service: ShardHomeLeaseService,
        datetime_to_timestamp: Callable[[object], timestamp_pb2.Timestamp | None],
        logger,
    ) -> None:
        self._svc = shard_home_lease_service
        self._datetime_to_timestamp = datetime_to_timestamp
        self._logger = logger

    def _lease_to_proto(self, lease: object) -> global_store_pb2.ShardHomeLease:
        # ShardHomeLease is a dataclass; keep this lightweight.
        holder = str(getattr(lease, "holder_daemon_id", "") or "")
        token = str(getattr(lease, "lease_token", "") or "")
        expires_at = getattr(lease, "expires_at", None)
        msg = global_store_pb2.ShardHomeLease(
            shard_id=int(getattr(lease, "shard_id", 0) or 0),
            holder_daemon_id=holder,
            lease_token=token,
            lease_generation=int(getattr(lease, "lease_generation", 0) or 0),
        )
        ts = self._datetime_to_timestamp(expires_at) if expires_at is not None else None
        if ts is not None:
            msg.expires_at.CopyFrom(ts)
        return msg

    def _lease_to_route_proto(self, lease: object) -> global_store_pb2.ShardHomeRoute:
        holder = str(getattr(lease, "holder_daemon_id", "") or "")
        expires_at = getattr(lease, "expires_at", None)
        msg = global_store_pb2.ShardHomeRoute(
            shard_id=int(getattr(lease, "shard_id", 0) or 0),
            holder_daemon_id=holder,
            lease_generation=int(getattr(lease, "lease_generation", 0) or 0),
        )
        ts = self._datetime_to_timestamp(expires_at) if expires_at is not None else None
        if ts is not None:
            msg.expires_at.CopyFrom(ts)
        return msg

    def acquire_shard_home_lease(
        self,
        request: global_store_pb2.AcquireShardHomeLeaseRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.AcquireShardHomeLeaseResponse:
        if not request.holder_daemon_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("holder_daemon_id is required")
            return global_store_pb2.AcquireShardHomeLeaseResponse()
        if request.ttl_ms <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("ttl_ms must be > 0")
            return global_store_pb2.AcquireShardHomeLeaseResponse()
        try:
            acquired, lease = self._svc.acquire(
                shard_id=int(request.shard_id),
                holder_daemon_id=str(request.holder_daemon_id),
                ttl_ms=int(request.ttl_ms),
            )
            lease_msg = self._lease_to_proto(lease)
            if not acquired:
                # Do not disclose the lease_token for a lease held by another daemon.
                lease_msg.lease_token = ""
            return global_store_pb2.AcquireShardHomeLeaseResponse(
                acquired=bool(acquired),
                lease=lease_msg,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("AcquireShardHomeLease failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.AcquireShardHomeLeaseResponse()

    def keepalive_shard_home_lease(
        self,
        request: global_store_pb2.KeepaliveShardHomeLeaseRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.KeepaliveShardHomeLeaseResponse:
        if not request.lease_token:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("lease_token is required")
            return global_store_pb2.KeepaliveShardHomeLeaseResponse()
        if request.ttl_ms <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("ttl_ms must be > 0")
            return global_store_pb2.KeepaliveShardHomeLeaseResponse()
        try:
            lease = self._svc.keepalive(
                lease_token=str(request.lease_token),
                ttl_ms=int(request.ttl_ms),
            )
            return global_store_pb2.KeepaliveShardHomeLeaseResponse(
                lease=self._lease_to_proto(lease),
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return global_store_pb2.KeepaliveShardHomeLeaseResponse()
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("KeepaliveShardHomeLease failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.KeepaliveShardHomeLeaseResponse()

    def batch_keepalive_shard_home_leases(
        self,
        request: global_store_pb2.BatchKeepaliveShardHomeLeasesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.BatchKeepaliveShardHomeLeasesResponse:
        if request.ttl_ms <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("ttl_ms must be > 0")
            return global_store_pb2.BatchKeepaliveShardHomeLeasesResponse()

        response = global_store_pb2.BatchKeepaliveShardHomeLeasesResponse()
        for entry in request.leases:
            outcome = response.outcomes.add(
                shard_id=int(entry.shard_id),
                lease_generation=int(entry.lease_generation),
                lease_token=str(entry.lease_token),
            )
            if not entry.lease_token:
                outcome.ok = False
                outcome.message = "lease_token is required"
                continue
            try:
                lease = self._svc.keepalive(
                    lease_token=str(entry.lease_token),
                    ttl_ms=int(request.ttl_ms),
                )
                if int(entry.shard_id) != int(lease.shard_id) or int(
                    entry.lease_generation
                ) != int(lease.lease_generation):
                    outcome.ok = False
                    outcome.lease.CopyFrom(self._lease_to_proto(lease))
                    outcome.message = "stale lease_generation for shard"
                    continue
                outcome.ok = True
                outcome.lease.CopyFrom(self._lease_to_proto(lease))
            except ValueError as exc:
                outcome.ok = False
                outcome.message = str(exc)
            except Exception as exc:  # noqa: BLE001
                outcome.ok = False
                outcome.message = str(exc)
        return response

    def release_shard_home_lease(
        self,
        request: global_store_pb2.ReleaseShardHomeLeaseRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ReleaseShardHomeLeaseResponse:
        if not request.lease_token:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("lease_token is required")
            return global_store_pb2.ReleaseShardHomeLeaseResponse(released=False)
        try:
            released = self._svc.release(lease_token=str(request.lease_token))
            return global_store_pb2.ReleaseShardHomeLeaseResponse(
                released=bool(released)
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("ReleaseShardHomeLease failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ReleaseShardHomeLeaseResponse(released=False)

    def get_shard_home_lease(
        self,
        request: global_store_pb2.GetShardHomeLeaseRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetShardHomeLeaseResponse:
        try:
            lease = self._svc.get_active(shard_id=int(request.shard_id))
            if lease is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("shard home lease not found")
                return global_store_pb2.GetShardHomeLeaseResponse()
            return global_store_pb2.GetShardHomeLeaseResponse(
                lease=self._lease_to_route_proto(lease)
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetShardHomeLease failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetShardHomeLeaseResponse()

    def batch_get_shard_home_leases(
        self,
        request: global_store_pb2.BatchGetShardHomeLeasesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.BatchGetShardHomeLeasesResponse:
        try:
            leases = self._svc.batch_get_active(
                shard_ids=[int(v) for v in request.shard_ids]
            )
            return global_store_pb2.BatchGetShardHomeLeasesResponse(
                leases=[self._lease_to_route_proto(lease) for lease in leases]
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("BatchGetShardHomeLeases failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.BatchGetShardHomeLeasesResponse()
