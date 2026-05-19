#  Copyright (c) 2026, TensorCast Team.

"""Repository for group realization transaction state."""

from __future__ import annotations

import json
from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository

_TERMINAL_STATES = {"published", "aborted", "expired"}


def _required_part_ids_json(required_part_ids: list[str]) -> str:
    return json.dumps(
        sorted({str(part_id) for part_id in required_part_ids}),
        separators=(",", ":"),
    )


def _parse_required_part_ids(raw: str) -> list[str]:
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


class GroupRealizationConflictError(ValueError):
    """Raised when a semantic transaction or member fingerprint conflicts."""


class GroupRealizationRepository(BaseRepository):
    """Data access for group realization transactions and members."""

    def _row_to_transaction(self, row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "transaction_id": str(row[0]),
            "group_kind": str(row[1]),
            "group_id": str(row[2]),
            "epoch": int(row[3]),
            "version_set_id": str(row[4]),
            "realization_kind": str(row[5]),
            "transaction_fingerprint": bytes(row[6]),
            "required_part_ids": _parse_required_part_ids(str(row[7])),
            "total_parts": int(row[8]),
            "prepared_count": int(row[9]),
            "failed_count": int(row[10]),
            "published_count": int(row[11]),
            "state": str(row[12]),
            "deadline_unix_nanos": int(row[13]) if row[13] is not None else None,
            "namespace": row[14],
            "key": row[15],
            "key_generation": int(row[16]) if row[16] is not None else None,
            "manifest_hash": bytes(row[17]) if row[17] is not None else None,
            "failure_code": row[18],
            "failure_detail": row[19],
            "created_at": row[20],
            "updated_at": row[21],
            "last_state_change_at": row[22],
        }

    def _transaction_projection(self) -> str:
        return """
            transaction_id,
            group_kind,
            group_id,
            epoch,
            version_set_id,
            realization_kind,
            transaction_fingerprint,
            required_part_ids_json,
            total_parts,
            prepared_count,
            failed_count,
            published_count,
            state,
            deadline_unix_nanos,
            namespace,
            key,
            key_generation,
            manifest_hash,
            failure_code,
            failure_detail,
            created_at,
            updated_at,
            last_state_change_at
        """

    def _select_transaction_by_id(
        self, *, cursor, transaction_id: str
    ) -> dict[str, Any] | None:
        row = cursor.execute(
            f"""
            SELECT {self._transaction_projection()}
            FROM group_realization_transactions
            WHERE transaction_id = ?
            """,
            [transaction_id],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_transaction(row)

    def _select_transaction_by_slot(
        self,
        *,
        cursor,
        group_kind: str,
        group_id: str,
        epoch: int,
    ) -> dict[str, Any] | None:
        row = cursor.execute(
            f"""
            SELECT {self._transaction_projection()}
            FROM group_realization_transactions
            WHERE group_kind = ? AND group_id = ? AND epoch = ?
            """,
            [group_kind, group_id, int(epoch)],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_transaction(row)

    def _row_to_member(self, row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "transaction_id": str(row[0]),
            "part_id": str(row[1]),
            "daemon_id": str(row[2] or ""),
            "worker_id": row[3],
            "daemon_session_id": row[4],
            "materialization_attempt_id": row[5],
            "artifact_id": str(row[6]),
            "view_id": row[7],
            "requested_byte_space": str(row[8]),
            "selection_hash": bytes(row[9]),
            "member_fingerprint": bytes(row[10]) if row[10] is not None else None,
            "state": str(row[11]),
            "staged_binding_id": row[12],
            "staged_binding_value_id": row[13],
            "staging_token": row[14],
            "staging_epoch": int(row[15]) if row[15] is not None else None,
            "expected_previous_seal_generation": (
                int(row[16]) if row[16] is not None else None
            ),
            "prepared_value_hash": bytes(row[17]) if row[17] is not None else None,
            "source_replica_id": row[18],
            "source_export_generation": int(row[19]) if row[19] is not None else None,
            "child_transport_request_id": row[20],
            "failure_code": row[21],
            "failure_detail": row[22],
            "created_at": row[23],
            "updated_at": row[24],
        }

    def _member_projection(self) -> str:
        return """
            transaction_id,
            part_id,
            daemon_id,
            worker_id,
            daemon_session_id,
            materialization_attempt_id,
            artifact_id,
            view_id,
            requested_byte_space,
            selection_hash,
            member_fingerprint,
            state,
            staged_binding_id,
            staged_binding_value_id,
            staging_token,
            staging_epoch,
            expected_previous_seal_generation,
            prepared_value_hash,
            source_replica_id,
            source_export_generation,
            child_transport_request_id,
            failure_code,
            failure_detail,
            created_at,
            updated_at
        """

    def _select_member(
        self,
        *,
        cursor,
        transaction_id: str,
        part_id: str,
    ) -> dict[str, Any] | None:
        row = cursor.execute(
            f"""
            SELECT {self._member_projection()}
            FROM group_realization_members
            WHERE transaction_id = ? AND part_id = ?
            """,
            [transaction_id, part_id],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_member(row)

    def begin_or_join(
        self,
        *,
        transaction_id: str,
        group_kind: str,
        group_id: str,
        epoch: int,
        version_set_id: str,
        realization_kind: str,
        transaction_fingerprint: bytes,
        required_part_ids: list[str],
        total_parts: int,
        part: dict[str, Any],
        deadline_unix_nanos: int | None,
        namespace: str | None = None,
        key: str | None = None,
        key_generation: int | None = None,
        manifest_hash: bytes | None = None,
        daemon_id: str = "",
        daemon_session_id: str | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or replay a semantic transaction and joined member row."""
        required_ids = sorted({str(part_id) for part_id in required_part_ids})
        if len(required_ids) != int(total_parts):
            raise ValueError("total_parts must match required_part_ids")
        if str(part["part_id"]) not in required_ids:
            raise ValueError("part_id is not in required_part_ids")
        required_json = _required_part_ids_json(required_ids)
        with self.transaction() as cursor:
            existing = self._select_transaction_by_slot(
                cursor=cursor,
                group_kind=group_kind,
                group_id=group_id,
                epoch=epoch,
            )
            if existing is None:
                cursor.execute(
                    """
                    INSERT INTO group_realization_transactions (
                      transaction_id,
                      group_kind,
                      group_id,
                      epoch,
                      version_set_id,
                      realization_kind,
                      transaction_fingerprint,
                      required_part_ids_json,
                      total_parts,
                      state,
                      deadline_unix_nanos,
                      namespace,
                      key,
                      key_generation,
                      manifest_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'preparing', ?, ?, ?, ?, ?)
                    """,
                    [
                        transaction_id,
                        group_kind,
                        group_id,
                        int(epoch),
                        version_set_id,
                        realization_kind,
                        transaction_fingerprint,
                        required_json,
                        int(total_parts),
                        deadline_unix_nanos,
                        namespace,
                        key,
                        key_generation,
                        manifest_hash,
                    ],
                )
                existing = self._select_transaction_by_id(
                    cursor=cursor,
                    transaction_id=transaction_id,
                )
            elif bytes(existing["transaction_fingerprint"]) != bytes(
                transaction_fingerprint
            ):
                raise GroupRealizationConflictError(
                    "semantic slot already has a different transaction fingerprint"
                )

            if existing is None:
                raise ValueError("transaction missing after begin")
            if existing["transaction_id"] != transaction_id:
                transaction_id = str(existing["transaction_id"])
            member = self._select_member(
                cursor=cursor,
                transaction_id=transaction_id,
                part_id=str(part["part_id"]),
            )
            if member is None:
                cursor.execute(
                    """
                    INSERT INTO group_realization_members (
                      transaction_id,
                      part_id,
                      daemon_id,
                      worker_id,
                      daemon_session_id,
                      artifact_id,
                      view_id,
                      requested_byte_space,
                      selection_hash,
                      state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'joined')
                    """,
                    [
                        transaction_id,
                        part["part_id"],
                        daemon_id,
                        worker_id,
                        daemon_session_id,
                        part["artifact_id"],
                        part.get("view_id"),
                        part["requested_byte_space"],
                        part["selection_hash"],
                    ],
                )
            else:
                if (
                    member["artifact_id"] != part["artifact_id"]
                    or _text(member.get("view_id")) != _text(part.get("view_id"))
                    or member["requested_byte_space"] != part["requested_byte_space"]
                    or bytes(member["selection_hash"]) != bytes(part["selection_hash"])
                ):
                    raise GroupRealizationConflictError(
                        "member part identity does not match frozen version set"
                    )
                for field_name, incoming in (
                    ("daemon_id", daemon_id),
                    ("daemon_session_id", daemon_session_id),
                    ("worker_id", worker_id),
                ):
                    existing_value = _text(member.get(field_name))
                    incoming_value = _text(incoming)
                    if (
                        existing_value
                        and incoming_value
                        and existing_value != incoming_value
                    ):
                        raise GroupRealizationConflictError(
                            f"member {field_name} does not match existing join"
                        )
            return (
                self._select_transaction_by_id(
                    cursor=cursor,
                    transaction_id=transaction_id,
                )
                or existing
            )

    def report_prepared(
        self,
        *,
        transaction_id: str,
        part_id: str,
        member_fingerprint: bytes,
        daemon_id: str,
        daemon_session_id: str | None,
        worker_id: str | None,
        materialization_attempt_id: str | None,
        staged_binding_id: str | None,
        staged_binding_value_id: str | None,
        staging_token: str | None,
        staging_epoch: int | None,
        expected_previous_seal_generation: int | None,
        prepared_value_hash: bytes | None,
        source_replica_id: str | None,
        source_export_generation: int | None,
        child_transport_request_id: str | None,
        auto_publish_when_ready: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Mark a member prepared and update readiness/publish state atomically."""
        with self.transaction() as cursor:
            transaction = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            if transaction is None:
                raise KeyError("transaction not found")
            if transaction["state"] in _TERMINAL_STATES:
                raise ValueError("transaction is terminal")
            member = self._select_member(
                cursor=cursor,
                transaction_id=transaction_id,
                part_id=part_id,
            )
            if member is None:
                raise KeyError("member not found")
            if member["state"] in {"prepared", "published"}:
                if member["member_fingerprint"] == bytes(member_fingerprint):
                    return transaction, member
                raise GroupRealizationConflictError(
                    "member already prepared with a different fingerprint"
                )
            if member["state"] in {"failed", "cancelled", "expired"}:
                raise ValueError("member is terminal")

            cursor.execute(
                """
                UPDATE group_realization_members
                SET daemon_id = ?,
                    daemon_session_id = ?,
                    worker_id = ?,
                    materialization_attempt_id = ?,
                    member_fingerprint = ?,
                    state = 'prepared',
                    staged_binding_id = ?,
                    staged_binding_value_id = ?,
                    staging_token = ?,
                    staging_epoch = ?,
                    expected_previous_seal_generation = ?,
                    prepared_value_hash = ?,
                    source_replica_id = ?,
                    source_export_generation = ?,
                    child_transport_request_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE transaction_id = ? AND part_id = ?
                """,
                [
                    daemon_id,
                    daemon_session_id,
                    worker_id,
                    materialization_attempt_id,
                    member_fingerprint,
                    staged_binding_id,
                    staged_binding_value_id,
                    staging_token,
                    staging_epoch,
                    expected_previous_seal_generation,
                    prepared_value_hash,
                    source_replica_id,
                    source_export_generation,
                    child_transport_request_id,
                    transaction_id,
                    part_id,
                ],
            )
            prepared_count = int(transaction["prepared_count"]) + 1
            next_state = (
                "ready_to_publish"
                if prepared_count == int(transaction["total_parts"])
                and int(transaction["failed_count"]) == 0
                else "preparing"
            )
            cursor.execute(
                """
                UPDATE group_realization_transactions
                SET prepared_count = ?,
                    state = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    last_state_change_at = CASE
                      WHEN state != ? THEN CURRENT_TIMESTAMP
                      ELSE last_state_change_at
                    END
                WHERE transaction_id = ?
                """,
                [prepared_count, next_state, next_state, transaction_id],
            )
            if next_state == "ready_to_publish" and auto_publish_when_ready:
                published_txn = self._publish_ready_in_transaction(
                    cursor=cursor,
                    transaction_id=transaction_id,
                )
                published_member = self._select_member(
                    cursor=cursor,
                    transaction_id=transaction_id,
                    part_id=part_id,
                )
                if published_member is None:
                    raise ValueError("prepared member missing after auto publish")
                return published_txn, published_member
            prepared_txn = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            prepared_member = self._select_member(
                cursor=cursor,
                transaction_id=transaction_id,
                part_id=part_id,
            )
            if prepared_txn is None:
                raise ValueError("prepared state missing after update")
            if prepared_member is None:
                raise ValueError("prepared member missing after update")
            return prepared_txn, prepared_member

    def _source_visibility_issues(self, *, cursor, transaction_id: str) -> list[dict]:
        rows = cursor.execute(
            """
            SELECT m.part_id,
                   m.source_replica_id,
                   m.source_export_generation,
                   CAST(ar.replica_id AS VARCHAR) AS live_replica_id,
                   ar.is_available,
                   ar.export_generation,
                   ar.export_state
            FROM group_realization_members m
            LEFT JOIN artifact_replicas ar
              ON CAST(ar.replica_id AS VARCHAR) = m.source_replica_id
            WHERE m.transaction_id = ?
              AND m.state = 'prepared'
              AND m.source_replica_id IS NOT NULL
              AND m.source_replica_id != ''
            """,
            [transaction_id],
        ).fetchall()
        issues: list[dict] = []
        for row in rows:
            part_id = str(row[0])
            source_replica_id = str(row[1])
            expected_generation = int(row[2] or 0)
            if row[3] is None:
                issues.append(
                    {
                        "part_id": part_id,
                        "source_replica_id": source_replica_id,
                        "reason": "source_replica_not_found",
                    }
                )
                continue
            if not bool(row[4]):
                issues.append(
                    {
                        "part_id": part_id,
                        "source_replica_id": source_replica_id,
                        "reason": "source_replica_unavailable",
                    }
                )
            actual_generation = int(row[5] or 0)
            if actual_generation != expected_generation:
                issues.append(
                    {
                        "part_id": part_id,
                        "source_replica_id": source_replica_id,
                        "reason": "source_export_generation_mismatch",
                        "expected_generation": expected_generation,
                        "actual_generation": actual_generation,
                    }
                )
            if str(row[6] or "") != "EXPORTABLE":
                issues.append(
                    {
                        "part_id": part_id,
                        "source_replica_id": source_replica_id,
                        "reason": "source_replica_not_exportable",
                        "export_state": str(row[6] or ""),
                    }
                )
        return issues

    def _abort_for_source_visibility(
        self,
        *,
        cursor,
        transaction_id: str,
        issues: list[dict],
    ) -> None:
        detail = json.dumps(issues, sort_keys=True, separators=(",", ":"))
        cursor.execute(
            """
            UPDATE group_realization_transactions
            SET state = 'aborted',
                failure_code = 'source_visibility_stale',
                failure_detail = ?,
                updated_at = CURRENT_TIMESTAMP,
                last_state_change_at = CURRENT_TIMESTAMP
            WHERE transaction_id = ?
            """,
            [detail, transaction_id],
        )
        cursor.execute(
            """
            UPDATE group_realization_members
            SET state = 'failed',
                failure_code = 'source_visibility_stale',
                failure_detail = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE transaction_id = ? AND state = 'prepared'
            """,
            [detail, transaction_id],
        )

    def _publish_ready_in_transaction(
        self,
        *,
        cursor,
        transaction_id: str,
    ) -> dict[str, Any]:
        source_issues = self._source_visibility_issues(
            cursor=cursor,
            transaction_id=transaction_id,
        )
        if source_issues:
            self._abort_for_source_visibility(
                cursor=cursor,
                transaction_id=transaction_id,
                issues=source_issues,
            )
            updated = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            if updated is None:
                raise ValueError("transaction missing after source fence abort")
            return updated

        cursor.execute(
            """
            UPDATE group_realization_members
            SET state = 'published',
                updated_at = CURRENT_TIMESTAMP
            WHERE transaction_id = ? AND state = 'prepared'
            """,
            [transaction_id],
        )
        cursor.execute(
            """
            UPDATE group_realization_transactions
            SET state = 'published',
                published_count = total_parts,
                updated_at = CURRENT_TIMESTAMP,
                last_state_change_at = CURRENT_TIMESTAMP
            WHERE transaction_id = ?
            """,
            [transaction_id],
        )
        updated = self._select_transaction_by_id(
            cursor=cursor,
            transaction_id=transaction_id,
        )
        if updated is None:
            raise ValueError("transaction missing after publish")
        return updated

    def publish(
        self,
        *,
        transaction_id: str,
        require_ready_to_publish: bool,
    ) -> dict[str, Any]:
        """Publish a ready transaction and mark prepared members published."""
        with self.transaction() as cursor:
            transaction = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            if transaction is None:
                raise KeyError("transaction not found")
            if transaction["state"] == "published":
                return transaction
            if transaction["state"] in {"aborted", "expired"}:
                raise ValueError("transaction is terminal")
            ready = (
                transaction["state"] == "ready_to_publish"
                and int(transaction["prepared_count"])
                == int(transaction["total_parts"])
                and int(transaction["failed_count"]) == 0
            )
            if require_ready_to_publish and not ready:
                raise ValueError("transaction is not ready to publish")
            if not ready:
                raise ValueError("transaction is not ready to publish")
            return self._publish_ready_in_transaction(
                cursor=cursor,
                transaction_id=transaction_id,
            )

    def abort(
        self,
        *,
        transaction_id: str,
        failure_code: str,
        failure_detail: str,
    ) -> dict[str, Any]:
        with self.transaction() as cursor:
            transaction = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            if transaction is None:
                raise KeyError("transaction not found")
            if transaction["state"] == "published":
                raise ValueError("published transaction cannot be aborted")
            if transaction["state"] in {"aborted", "expired"}:
                return transaction
            cursor.execute(
                """
                UPDATE group_realization_transactions
                SET state = 'aborted',
                    failure_code = ?,
                    failure_detail = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    last_state_change_at = CURRENT_TIMESTAMP
                WHERE transaction_id = ?
                """,
                [failure_code, failure_detail, transaction_id],
            )
            cursor.execute(
                """
                UPDATE group_realization_members
                SET state = 'cancelled',
                    failure_code = ?,
                    failure_detail = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE transaction_id = ? AND state IN ('joined','preparing','prepared')
                """,
                [failure_code, failure_detail, transaction_id],
            )
            updated = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            if updated is None:
                raise ValueError("transaction missing after abort")
            return updated

    def expire_due(self, *, now_unix_nanos: int, batch_limit: int) -> int:
        """Expire due non-terminal transactions in an indexed, bounded batch."""
        limit = max(1, int(batch_limit))
        with self.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT transaction_id
                FROM group_realization_transactions
                WHERE state NOT IN ('published', 'aborted', 'expired')
                  AND deadline_unix_nanos IS NOT NULL
                  AND deadline_unix_nanos <= ?
                ORDER BY deadline_unix_nanos
                LIMIT ?
                """,
                [int(now_unix_nanos), limit],
            ).fetchall()
            transaction_ids = [str(row[0]) for row in rows]
            for transaction_id in transaction_ids:
                cursor.execute(
                    """
                    UPDATE group_realization_transactions
                    SET state = 'expired',
                        failure_code = 'deadline_exceeded',
                        failure_detail = 'group realization deadline expired',
                        updated_at = CURRENT_TIMESTAMP,
                        last_state_change_at = CURRENT_TIMESTAMP
                    WHERE transaction_id = ?
                    """,
                    [transaction_id],
                )
                cursor.execute(
                    """
                    UPDATE group_realization_members
                    SET state = 'expired',
                        failure_code = 'deadline_exceeded',
                        failure_detail = 'group realization deadline expired',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE transaction_id = ?
                      AND state IN ('joined','preparing','prepared')
                    """,
                    [transaction_id],
                )
            return len(transaction_ids)

    def get(self, *, transaction_id: str) -> dict[str, Any] | None:
        cursor = self.get_cursor()
        try:
            return self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
        finally:
            cursor.close()

    def get_by_slot(
        self,
        *,
        group_kind: str,
        group_id: str,
        epoch: int,
    ) -> dict[str, Any] | None:
        cursor = self.get_cursor()
        try:
            return self._select_transaction_by_slot(
                cursor=cursor,
                group_kind=group_kind,
                group_id=group_id,
                epoch=epoch,
            )
        finally:
            cursor.close()

    def get_diagnostic(self, *, transaction_id: str) -> dict[str, Any] | None:
        cursor = self.get_cursor()
        try:
            transaction = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            if transaction is None:
                return None
            rows = cursor.execute(
                f"""
                SELECT {self._member_projection()}
                FROM group_realization_members
                WHERE transaction_id = ?
                ORDER BY part_id
                """,
                [transaction_id],
            ).fetchall()
            members = [self._row_to_member(row) for row in rows]
            prepared_or_published = {
                member["part_id"]
                for member in members
                if member["state"] in {"prepared", "published"}
            }
            missing = [
                part_id
                for part_id in transaction["required_part_ids"]
                if part_id not in prepared_or_published
            ]
            return {
                "transaction": transaction,
                "members": members,
                "missing_part_ids": missing,
            }
        finally:
            cursor.close()

    def reconcile_counters(self, *, transaction_id: str) -> dict[str, int] | None:
        cursor = self.get_cursor()
        try:
            transaction = self._select_transaction_by_id(
                cursor=cursor,
                transaction_id=transaction_id,
            )
            if transaction is None:
                return None
            rows = cursor.execute(
                """
                SELECT state, COUNT(*)
                FROM group_realization_members
                WHERE transaction_id = ?
                GROUP BY state
                """,
                [transaction_id],
            ).fetchall()
            counts = {str(row[0]): int(row[1]) for row in rows}
            return {
                "prepared": counts.get("prepared", 0) + counts.get("published", 0),
                "failed": counts.get("failed", 0),
                "published": counts.get("published", 0),
            }
        finally:
            cursor.close()
