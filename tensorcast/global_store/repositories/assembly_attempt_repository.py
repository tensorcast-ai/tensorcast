#  Copyright (c) 2026, TensorCast Team.

"""Repository for durable immutable assembly attempt rows."""

from __future__ import annotations

from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository


class AssemblyAttemptRepository(BaseRepository):
    """Data access layer for `assembly_attempts`."""

    def get(
        self,
        *,
        attempt_id: str | None = None,
        workspace_assembly_id: str | None = None,
        cursor=None,
    ) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        if bool(attempt_id) == bool(workspace_assembly_id):
            raise ValueError(
                "exactly one of attempt_id or workspace_assembly_id is required"
            )
        if attempt_id:
            row = target.execute(
                """
                SELECT attempt_id,
                       workspace_assembly_id,
                       layout_id,
                       attempt_intent_digest,
                       coordinator_operation_id,
                       attempt_record_proto,
                       created_at,
                       updated_at
                FROM assembly_attempts
                WHERE attempt_id = ?
                """,
                [attempt_id],
            ).fetchone()
        else:
            row = target.execute(
                """
                SELECT attempt_id,
                       workspace_assembly_id,
                       layout_id,
                       attempt_intent_digest,
                       coordinator_operation_id,
                       attempt_record_proto,
                       created_at,
                       updated_at
                FROM assembly_attempts
                WHERE workspace_assembly_id = ?
                """,
                [workspace_assembly_id],
            ).fetchone()
        if not row:
            return None
        return {
            "attempt_id": str(row[0]),
            "workspace_assembly_id": str(row[1]),
            "layout_id": str(row[2]),
            "attempt_intent_digest": str(row[3]),
            "coordinator_operation_id": str(row[4]),
            "attempt_record_proto": bytes(row[5]),
            "created_at": row[6],
            "updated_at": row[7],
        }

    def upsert(
        self,
        *,
        attempt_id: str,
        workspace_assembly_id: str,
        layout_id: str,
        attempt_intent_digest: str,
        coordinator_operation_id: str,
        attempt_record_proto: bytes,
        cursor=None,
    ) -> dict[str, Any]:
        target = cursor if cursor is not None else self.get_cursor()
        existing = self.get(attempt_id=attempt_id, cursor=target)
        if existing is not None:
            if (
                existing["workspace_assembly_id"] != workspace_assembly_id
                or existing["layout_id"] != layout_id
                or existing["attempt_intent_digest"] != attempt_intent_digest
                or existing["coordinator_operation_id"] != coordinator_operation_id
                or existing["attempt_record_proto"] != attempt_record_proto
            ):
                raise ValueError("assembly attempt conflict")
            return existing
        target.execute(
            """
            INSERT INTO assembly_attempts (
                attempt_id,
                workspace_assembly_id,
                layout_id,
                attempt_intent_digest,
                coordinator_operation_id,
                attempt_record_proto
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                attempt_id,
                workspace_assembly_id,
                layout_id,
                attempt_intent_digest,
                coordinator_operation_id,
                attempt_record_proto,
            ],
        )
        created = self.get(attempt_id=attempt_id, cursor=target)
        if created is None:
            raise ValueError("failed to create assembly attempt")
        return created


__all__ = ["AssemblyAttemptRepository"]
