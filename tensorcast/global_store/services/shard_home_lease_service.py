#  Copyright (c) 2025-2026, TensorCast Team.

"""Shard-home lease service (byte artifact routing fencing)."""

from __future__ import annotations

from datetime import datetime, timezone

from tensorcast.global_store.models.shard_home_lease import ShardHomeLease
from tensorcast.global_store.repositories.shard_home_lease_repository import (
    ShardHomeLeaseRepository,
)


class ShardHomeLeaseService:
    def __init__(self, repository: ShardHomeLeaseRepository) -> None:
        self._repo = repository

    @staticmethod
    def _is_active(lease: ShardHomeLease) -> bool:
        if (
            not lease.holder_daemon_id
            or not lease.lease_token
            or lease.lease_generation <= 0
        ):
            return False
        if lease.expires_at is None:
            return False
        return lease.expires_at > datetime.now(timezone.utc)

    def get_active(self, *, shard_id: int) -> ShardHomeLease | None:
        lease = self._repo.get(shard_id=int(shard_id))
        if lease is None:
            return None
        return lease if self._is_active(lease) else None

    def batch_get_active(self, *, shard_ids: list[int]) -> list[ShardHomeLease]:
        leases = self._repo.batch_get(shard_ids=[int(v) for v in shard_ids])
        return [lease for lease in leases if self._is_active(lease)]

    def acquire(
        self,
        *,
        shard_id: int,
        holder_daemon_id: str,
        ttl_ms: int,
    ) -> tuple[bool, ShardHomeLease]:
        with self._repo.transaction() as cursor:
            return self._repo.acquire(
                shard_id=int(shard_id),
                holder_daemon_id=str(holder_daemon_id),
                ttl_ms=int(ttl_ms),
                cursor=cursor,
            )

    def keepalive(self, *, lease_token: str, ttl_ms: int) -> ShardHomeLease:
        pending_exc: ValueError | None = None
        with self._repo.transaction() as cursor:
            try:
                return self._repo.keepalive(
                    lease_token=str(lease_token),
                    ttl_ms=int(ttl_ms),
                    cursor=cursor,
                )
            except ValueError as exc:
                # BaseRepository.transaction wraps exceptions into DatabaseError.
                # Keep the failure typed (ValueError) so RPC handlers can map it
                # to NOT_FOUND without logging a transaction failure.
                pending_exc = exc
        raise pending_exc if pending_exc is not None else ValueError("keepalive failed")

    def release(self, *, lease_token: str) -> bool:
        with self._repo.transaction() as cursor:
            return bool(self._repo.release(lease_token=str(lease_token), cursor=cursor))
