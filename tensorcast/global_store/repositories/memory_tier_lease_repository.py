#  Copyright (c) 2025-2026, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

"""Repository for memory tier lease persistence."""

import json
import time
from typing import Iterable, Sequence

from tensorcast.global_store.models.memory_tier import ChunkRange, MemoryTierLease
from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class MemoryTierLeaseRepository(BaseRepository):
    """Persistence helpers for memory_tier_leases table."""

    def create(self, lease: MemoryTierLease) -> MemoryTierLease:
        cursor = self.get_cursor()
        try:
            cursor.execute(
                """
                INSERT INTO memory_tier_leases (
                    lease_id, node_id, kind, artifact_id, chunk_range, chunk_ids,
                    ledger_version, bytes, workload_id, state, request_id,
                    ack_epoch_ns, issued_at_ns, expires_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    lease.lease_id,
                    lease.node_id,
                    lease.kind,
                    lease.artifact_id,
                    json.dumps(
                        {
                            "start": lease.chunk_range.start,
                            "count": lease.chunk_range.count,
                        }
                    ),
                    json.dumps(lease.chunk_ids),
                    lease.ledger_version,
                    lease.bytes,
                    lease.workload_id,
                    lease.state,
                    lease.request_id,
                    lease.ack_epoch_ns,
                    lease.issued_at_ns,
                    lease.expires_at_ns,
                ],
            )
            return lease
        finally:
            cursor.close()

    def find_by_id(self, lease_id: str) -> MemoryTierLease | None:
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT * FROM memory_tier_leases WHERE lease_id = ?
                """,
                [lease_id],
            ).fetchone()
            if not row:
                return None
            return self._row_to_model(row)
        finally:
            cursor.close()

    def find_by_request(self, node_id: str, request_id: str) -> MemoryTierLease | None:
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT * FROM memory_tier_leases
                WHERE node_id = ? AND request_id = ?
                ORDER BY issued_at_ns DESC
                LIMIT 1
                """,
                [node_id, request_id],
            ).fetchone()
            if not row:
                return None
            return self._row_to_model(row)
        finally:
            cursor.close()

    def upsert_request(self, lease: MemoryTierLease) -> MemoryTierLease:
        existing = self.find_by_request(lease.node_id, lease.request_id)
        if existing:
            return existing
        return self.create(lease)

    def acknowledge_acquired(
        self,
        lease_id: str,
        artifact_id: str,
        chunk_ids: Sequence[int],
        ledger_version: int,
        chunk_range: ChunkRange,
        bytes_count: int,
        request_id: str,
        ack_epoch_ns: int | None,
    ) -> MemoryTierLease | None:
        ack_ns = ack_epoch_ns or int(time.time_ns())
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                UPDATE memory_tier_leases
                SET state = 'active',
                    chunk_ids = ?,
                    chunk_range = ?,
                    ledger_version = ?,
                    bytes = ?,
                    ack_epoch_ns = ?,
                    request_id = ?
                WHERE lease_id = ? AND artifact_id = ?
                RETURNING *
                """,
                [
                    json.dumps(list(chunk_ids)),
                    json.dumps(
                        {"start": chunk_range.start, "count": chunk_range.count}
                    ),
                    ledger_version,
                    bytes_count,
                    ack_ns,
                    request_id,
                    lease_id,
                    artifact_id,
                ],
            ).fetchone()
            return self._row_to_model(row) if row else None
        finally:
            cursor.close()

    def acknowledge_released(
        self,
        lease_id: str,
        artifact_id: str,
        chunk_ids: Sequence[int],
        chunk_range: ChunkRange,
        ledger_version: int,
        bytes_count: int,
        request_id: str,
        ack_epoch_ns: int | None,
    ) -> MemoryTierLease | None:
        ack_ns = ack_epoch_ns or int(time.time_ns())
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                UPDATE memory_tier_leases
                SET state = 'expired',
                    chunk_ids = ?,
                    chunk_range = ?,
                    ledger_version = ?,
                    bytes = ?,
                    ack_epoch_ns = ?,
                    request_id = ?
                WHERE lease_id = ? AND artifact_id = ?
                RETURNING *
                """,
                [
                    json.dumps(list(chunk_ids)),
                    json.dumps(
                        {"start": chunk_range.start, "count": chunk_range.count}
                    ),
                    ledger_version,
                    bytes_count,
                    ack_ns,
                    request_id,
                    lease_id,
                    artifact_id,
                ],
            ).fetchone()
            return self._row_to_model(row) if row else None
        finally:
            cursor.close()

    def mark_revoking(self, lease_id: str) -> MemoryTierLease | None:
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                UPDATE memory_tier_leases
                SET state = 'revoking'
                WHERE lease_id = ?
                RETURNING *
                """,
                [lease_id],
            ).fetchone()
            return self._row_to_model(row) if row else None
        finally:
            cursor.close()

    def list_outstanding(
        self, node_id: str, states: Iterable[str] | None = None
    ) -> list[MemoryTierLease]:
        states = list(states) if states else ["pending", "active", "revoking"]
        placeholders = ", ".join(["?"] * len(states))
        cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                f"""
                SELECT * FROM memory_tier_leases
                WHERE node_id = ? AND state IN ({placeholders})
                ORDER BY issued_at_ns DESC
                """,
                [node_id, *states],
            ).fetchall()
            return [self._row_to_model(r) for r in rows]
        finally:
            cursor.close()

    def _row_to_model(self, row: tuple) -> MemoryTierLease:
        chunk_range_raw = row[4]
        chunk_ids_raw = row[5]
        try:
            cr_dict = (
                json.loads(chunk_range_raw)
                if chunk_range_raw
                else {"start": 0, "count": 0}
            )
            chunk_range = ChunkRange(
                start=int(cr_dict.get("start", 0)),
                count=int(cr_dict.get("count", 0)),
            )
        except Exception:
            chunk_range = ChunkRange()
        try:
            chunk_ids = [int(v) for v in json.loads(chunk_ids_raw or "[]")]
        except Exception:
            chunk_ids = []

        return MemoryTierLease(
            lease_id=row[0],
            node_id=row[1],
            kind=row[2],
            artifact_id=row[3],
            chunk_range=chunk_range,
            chunk_ids=chunk_ids,
            ledger_version=row[6],
            bytes=row[7],
            workload_id=row[8],
            state=row[9],
            request_id=row[10],
            ack_epoch_ns=row[11],
            issued_at_ns=row[12],
            expires_at_ns=row[13],
        )
