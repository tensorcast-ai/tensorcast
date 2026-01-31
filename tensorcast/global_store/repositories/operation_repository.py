#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for long-tail operations and coordinator leases (v2)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository


class OperationRepository(BaseRepository):
    """Data access layer for `operations`."""

    @staticmethod
    def _coerce_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def get(self, *, operation_id: str, cursor=None) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT operation_id,
                   kind,
                   target_artifact_id,
                   state,
                   status_proto,
                   snapshot_proto,
                   lease_owner,
                   lease_token,
                   lease_generation,
                   lease_expires_at,
                   created_at,
                   updated_at
            FROM operations
            WHERE operation_id = ?
            """,
            [operation_id],
        ).fetchone()
        if not row:
            return None
        return {
            "operation_id": str(row[0]),
            "kind": str(row[1]),
            "target_artifact_id": str(row[2]),
            "state": str(row[3]),
            "status_proto": bytes(row[4]),
            "snapshot_proto": (bytes(row[5]) if row[5] is not None else None),
            "lease_owner": (str(row[6]) if row[6] is not None else None),
            "lease_token": (str(row[7]) if row[7] is not None else None),
            "lease_generation": int(row[8]),
            "lease_expires_at": row[9],
            "created_at": row[10],
            "updated_at": row[11],
        }

    def _ensure_row(
        self,
        *,
        operation_id: str,
        kind: str,
        target_artifact_id: str,
        state: str,
        status_proto: bytes,
        cursor,
    ) -> None:
        existing = self.get(operation_id=operation_id, cursor=cursor)
        if existing is not None:
            if (
                existing["kind"] != kind
                or existing["target_artifact_id"] != target_artifact_id
            ):
                raise ValueError("operation_id conflict with different kind/target")
            return
        cursor.execute(
            """
            INSERT INTO operations (operation_id, kind, target_artifact_id, state, status_proto, snapshot_proto)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            [operation_id, kind, target_artifact_id, state, status_proto],
        )

    def acquire_lease(
        self,
        *,
        operation_id: str,
        kind: str,
        target_artifact_id: str,
        owner_id: str,
        ttl_ms: int,
        initial_state: str,
        initial_status_proto: bytes,
        cursor=None,
    ) -> tuple[bool, dict[str, Any]]:
        """Acquire or refresh a coordinator lease.

        Returns (acquired, lease_dict). When acquired is False, lease_dict
        describes the current holder.
        """
        target = cursor if cursor is not None else self.get_cursor()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(milliseconds=max(0, int(ttl_ms)))

        self._ensure_row(
            operation_id=operation_id,
            kind=kind,
            target_artifact_id=target_artifact_id,
            state=initial_state,
            status_proto=initial_status_proto,
            cursor=target,
        )

        row = self.get(operation_id=operation_id, cursor=target)
        if row is None:
            raise ValueError("operation missing after ensure")

        lease_expires_at = row.get("lease_expires_at")
        lease_owner = row.get("lease_owner")
        lease_token = row.get("lease_token")
        lease_generation = int(row.get("lease_generation") or 0)

        active = (
            bool(lease_token)
            and lease_expires_at is not None
            and lease_expires_at > now
        )
        if active and lease_owner and lease_owner != owner_id:
            return False, {
                "operation_id": operation_id,
                "lease_token": str(lease_token or ""),
                "owner_id": str(lease_owner),
                "lease_generation": lease_generation,
                "expires_at": lease_expires_at,
            }

        if active and lease_owner == owner_id and lease_token:
            target.execute(
                """
                UPDATE operations
                SET lease_expires_at = ?, updated_at = CURRENT_TIMESTAMP
                WHERE operation_id = ? AND lease_token = ?
                """,
                [expires_at, operation_id, lease_token],
            )
            row = self.get(operation_id=operation_id, cursor=target)
            if row is None:
                raise ValueError("operation missing after keepalive")
            return True, {
                "operation_id": operation_id,
                "lease_token": str(row["lease_token"] or ""),
                "owner_id": str(row["lease_owner"] or owner_id),
                "lease_generation": int(row["lease_generation"]),
                "expires_at": row["lease_expires_at"],
            }

        # Acquire a new lease
        new_token = uuid.uuid4().hex
        new_generation = lease_generation + 1
        target.execute(
            """
            UPDATE operations
            SET lease_owner = ?, lease_token = ?, lease_generation = ?, lease_expires_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE operation_id = ?
            """,
            [owner_id, new_token, new_generation, expires_at, operation_id],
        )
        row = self.get(operation_id=operation_id, cursor=target)
        if row is None:
            raise ValueError("operation missing after acquire")
        return True, {
            "operation_id": operation_id,
            "lease_token": str(row["lease_token"] or ""),
            "owner_id": str(row["lease_owner"] or owner_id),
            "lease_generation": int(row["lease_generation"]),
            "expires_at": row["lease_expires_at"],
        }

    def keepalive_lease(
        self,
        *,
        lease_token: str,
        ttl_ms: int,
        cursor=None,
    ) -> dict[str, Any]:
        target = cursor if cursor is not None else self.get_cursor()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(milliseconds=max(0, int(ttl_ms)))
        row = target.execute(
            """
            SELECT operation_id, lease_owner, lease_generation
            FROM operations
            WHERE lease_token = ?
            """,
            [lease_token],
        ).fetchone()
        if not row:
            raise ValueError("lease_token not found")
        operation_id = str(row[0])
        target.execute(
            """
            UPDATE operations
            SET lease_expires_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE lease_token = ?
            """,
            [expires_at, lease_token],
        )
        updated = self.get(operation_id=operation_id, cursor=target)
        if updated is None:
            raise ValueError("operation missing after keepalive")
        return {
            "operation_id": operation_id,
            "lease_token": str(updated["lease_token"] or lease_token),
            "owner_id": str(updated["lease_owner"] or ""),
            "lease_generation": int(updated["lease_generation"]),
            "expires_at": updated["lease_expires_at"],
        }

    def release_lease(self, *, lease_token: str, cursor=None) -> bool:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT operation_id
            FROM operations
            WHERE lease_token = ?
            """,
            [lease_token],
        ).fetchone()
        if not row:
            return False
        operation_id = str(row[0])
        target.execute(
            """
            UPDATE operations
            SET lease_owner = NULL,
                lease_token = NULL,
                lease_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE operation_id = ? AND lease_token = ?
            """,
            [operation_id, lease_token],
        )
        return True

    def update_operation(
        self,
        *,
        operation_id: str,
        lease_generation: int,
        state: str,
        status_proto: bytes,
        snapshot_proto: bytes | None,
        min_status_update_interval_ms: int = 0,
        cursor=None,
    ) -> None:
        target = cursor if cursor is not None else self.get_cursor()
        now = datetime.now(timezone.utc)
        row = self.get(operation_id=operation_id, cursor=target)
        if row is None:
            raise ValueError("operation not found")
        if int(row["lease_generation"]) != int(lease_generation):
            raise ValueError("stale lease_generation")
        lease_expires_at = row.get("lease_expires_at")
        if (
            lease_expires_at is None
            or self._coerce_utc(lease_expires_at) <= now
            or not row.get("lease_token")
        ):
            raise ValueError("operation lease expired")

        existing_snapshot = row.get("snapshot_proto")
        setting_snapshot = existing_snapshot is None and snapshot_proto is not None
        if existing_snapshot is not None:
            if snapshot_proto is not None and snapshot_proto != existing_snapshot:
                raise ValueError("snapshot mismatch with existing operation")
            snapshot_proto = existing_snapshot

        # Throttle in-place updates to avoid DB write amplification. Terminal
        # transitions should always persist.
        existing_state = str(row.get("state") or "")
        existing_updated_at = row.get("updated_at")
        if (
            int(min_status_update_interval_ms) > 0
            and existing_state == state
            and existing_updated_at is not None
            and not setting_snapshot
        ):
            updated_at = self._coerce_utc(existing_updated_at)
            age_ms = int((now - updated_at).total_seconds() * 1000)
            if age_ms < int(min_status_update_interval_ms):
                return

        target.execute(
            """
            UPDATE operations
            SET state = ?, status_proto = ?, snapshot_proto = ?, updated_at = CURRENT_TIMESTAMP
            WHERE operation_id = ? AND lease_generation = ?
            """,
            [state, status_proto, snapshot_proto, operation_id, int(lease_generation)],
        )
