#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for shard-home lease persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from tensorcast.global_store.models.shard_home_lease import ShardHomeLease
from tensorcast.global_store.repositories.base import BaseRepository


class ShardHomeLeaseRepository(BaseRepository):
    """Data access layer for `shard_home_leases`."""

    @staticmethod
    def _coerce_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _row_to_model(self, row: object) -> ShardHomeLease:
        shard_id, holder_daemon_id, lease_token, lease_generation, expires_at = row
        return ShardHomeLease(
            shard_id=int(shard_id),
            holder_daemon_id=str(holder_daemon_id or ""),
            lease_token=str(lease_token or ""),
            lease_generation=int(lease_generation or 0),
            expires_at=self._coerce_utc(expires_at) if expires_at is not None else None,
        )

    def get(self, *, shard_id: int, cursor=None) -> ShardHomeLease | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT shard_id,
                   holder_daemon_id,
                   lease_token,
                   lease_generation,
                   expires_at
            FROM shard_home_leases
            WHERE shard_id = ?
            """,
            [int(shard_id)],
        ).fetchone()
        if not row:
            return None
        return self._row_to_model(row)

    def batch_get(self, *, shard_ids: list[int], cursor=None) -> list[ShardHomeLease]:
        if not shard_ids:
            return []
        target = cursor if cursor is not None else self.get_cursor()
        placeholders = ", ".join(["?"] * len(shard_ids))
        rows = target.execute(
            f"""
            SELECT shard_id,
                   holder_daemon_id,
                   lease_token,
                   lease_generation,
                   expires_at
            FROM shard_home_leases
            WHERE shard_id IN ({placeholders})
            """,
            [int(v) for v in shard_ids],
        ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def acquire(
        self,
        *,
        shard_id: int,
        holder_daemon_id: str,
        ttl_ms: int,
        cursor=None,
    ) -> tuple[bool, ShardHomeLease]:
        """Acquire or refresh a shard-home lease.

        Returns (acquired, lease). When acquired is False, lease describes the
        current holder.
        """
        target = cursor if cursor is not None else self.get_cursor()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(milliseconds=max(0, int(ttl_ms)))

        refreshed_row = target.execute(
            """
            UPDATE shard_home_leases
            SET expires_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE shard_id = ?
              AND holder_daemon_id = ?
              AND lease_token <> ''
              AND lease_generation > 0
              AND expires_at > ?
            RETURNING shard_id, holder_daemon_id, lease_token, lease_generation, expires_at
            """,
            [expires_at, int(shard_id), str(holder_daemon_id), now],
        ).fetchone()
        if refreshed_row:
            return True, self._row_to_model(refreshed_row)

        new_token = uuid.uuid4().hex
        acquired_row = target.execute(
            """
            UPDATE shard_home_leases
            SET holder_daemon_id = ?,
                lease_token = ?,
                lease_generation = lease_generation + 1,
                expires_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE shard_id = ?
              AND (expires_at <= ? OR holder_daemon_id = '' OR lease_token = '')
            RETURNING shard_id, holder_daemon_id, lease_token, lease_generation, expires_at
            """,
            [str(holder_daemon_id), new_token, expires_at, int(shard_id), now],
        ).fetchone()
        if acquired_row:
            return True, self._row_to_model(acquired_row)

        # First acquire creates the row (or races).
        insert_token = uuid.uuid4().hex
        inserted_row = target.execute(
            """
            INSERT INTO shard_home_leases (
                shard_id, holder_daemon_id, lease_token, lease_generation, expires_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (shard_id) DO NOTHING
            RETURNING shard_id, holder_daemon_id, lease_token, lease_generation, expires_at
            """,
            [int(shard_id), str(holder_daemon_id), insert_token, 1, expires_at],
        ).fetchone()
        if inserted_row:
            return True, self._row_to_model(inserted_row)

        existing = self.get(shard_id=int(shard_id), cursor=target)
        if existing is None:
            raise ValueError("lease missing after acquire retry")

        active = (
            existing.expires_at is not None
            and existing.expires_at > now
            and bool(existing.holder_daemon_id)
            and bool(existing.lease_token)
            and existing.lease_generation > 0
        )
        if active and existing.holder_daemon_id != holder_daemon_id:
            return False, existing
        if active and existing.holder_daemon_id == holder_daemon_id:
            return True, existing

        # Last attempt: acquire again after an insert race or concurrent release.
        final_token = uuid.uuid4().hex
        final_row = target.execute(
            """
            UPDATE shard_home_leases
            SET holder_daemon_id = ?,
                lease_token = ?,
                lease_generation = lease_generation + 1,
                expires_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE shard_id = ?
              AND (expires_at <= ? OR holder_daemon_id = '' OR lease_token = '')
            RETURNING shard_id, holder_daemon_id, lease_token, lease_generation, expires_at
            """,
            [str(holder_daemon_id), final_token, expires_at, int(shard_id), now],
        ).fetchone()
        if final_row:
            return True, self._row_to_model(final_row)

        existing = self.get(shard_id=int(shard_id), cursor=target)
        if existing is None:
            raise ValueError("lease missing after acquire")
        return False, existing

    def keepalive(
        self, *, lease_token: str, ttl_ms: int, cursor=None
    ) -> ShardHomeLease:
        target = cursor if cursor is not None else self.get_cursor()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(milliseconds=max(0, int(ttl_ms)))
        row = target.execute(
            """
            SELECT shard_id, expires_at
            FROM shard_home_leases
            WHERE lease_token = ?
            """,
            [str(lease_token)],
        ).fetchone()
        if not row:
            raise ValueError("lease_token not found")
        shard_id = int(row[0])
        prev_expires_at = row[1]
        if prev_expires_at is None or self._coerce_utc(prev_expires_at) <= now:
            # Do not allow keepalive to revive an expired lease without bumping generation.
            raise ValueError("lease expired")

        target.execute(
            """
            UPDATE shard_home_leases
            SET expires_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE lease_token = ?
            """,
            [expires_at, str(lease_token)],
        )
        refreshed = self.get(shard_id=shard_id, cursor=target)
        if refreshed is None:
            raise ValueError("lease missing after keepalive")
        return refreshed

    def release(self, *, lease_token: str, cursor=None) -> bool:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT shard_id, lease_generation
            FROM shard_home_leases
            WHERE lease_token = ?
            """,
            [str(lease_token)],
        ).fetchone()
        if not row:
            return False
        shard_id = int(row[0])
        generation = int(row[1] or 0)
        now = datetime.now(timezone.utc)
        target.execute(
            """
            UPDATE shard_home_leases
            SET holder_daemon_id = '',
                lease_token = '',
                expires_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE shard_id = ? AND lease_token = ? AND lease_generation = ?
            """,
            [now, shard_id, str(lease_token), generation],
        )
        return True
