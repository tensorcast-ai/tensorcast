#  Copyright (c) 2026, TensorCast Team.

"""Repository for durable assembly slot-occupancy rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, SupportsIndex, SupportsInt, cast

from tensorcast.global_store.repositories.base import BaseRepository


class AssemblySlotOccupancyRepository(BaseRepository):
    """Data access layer for `assembly_slot_occupancies`."""

    _SELECT_COLUMNS = """
        attempt_id,
        slot_id,
        structural_view_id,
        binding_id,
        binding_value_id,
        coverage_plan_hash,
        contributor_daemon_id,
        coordinator_operation_id,
        coordinator_generation,
        lease_id,
        lease_generation,
        lease_expires_at,
        state,
        created_at,
        updated_at
    """

    def get(
        self,
        *,
        attempt_id: str,
        slot_id: str,
        cursor=None,
    ) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            f"""
            SELECT {self._SELECT_COLUMNS}
            FROM assembly_slot_occupancies
            WHERE attempt_id = ? AND slot_id = ?
            """,
            [attempt_id, slot_id],
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def claim_slot(
        self,
        *,
        attempt_id: str,
        slot_id: str,
        structural_view_id: str | None,
        binding_id: str,
        binding_value_id: str,
        coverage_plan_hash: str,
        contributor_daemon_id: str,
        coordinator_operation_id: str,
        coordinator_generation: int,
        lease_id: str,
        lease_generation: int,
        lease_expires_at: datetime | None,
        state: str = "accepted",
        cursor=None,
    ) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        target.execute(
            """
            INSERT INTO assembly_slot_occupancies (
                attempt_id,
                slot_id,
                structural_view_id,
                binding_id,
                binding_value_id,
                coverage_plan_hash,
                contributor_daemon_id,
                coordinator_operation_id,
                coordinator_generation,
                lease_id,
                lease_generation,
                lease_expires_at,
                state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (attempt_id, slot_id) DO NOTHING
            """,
            [
                attempt_id,
                slot_id,
                structural_view_id,
                binding_id,
                binding_value_id,
                coverage_plan_hash,
                contributor_daemon_id,
                coordinator_operation_id,
                int(coordinator_generation),
                lease_id,
                int(lease_generation),
                lease_expires_at,
                state,
            ],
        )
        row = self.get(attempt_id=attempt_id, slot_id=slot_id, cursor=target)
        if row is None:
            raise ValueError("failed to claim assembly slot occupancy")
        if self._matches_claim(
            row=row,
            structural_view_id=structural_view_id,
            binding_id=binding_id,
            binding_value_id=binding_value_id,
            coverage_plan_hash=coverage_plan_hash,
            contributor_daemon_id=contributor_daemon_id,
            coordinator_operation_id=coordinator_operation_id,
            coordinator_generation=int(coordinator_generation),
            lease_id=lease_id,
            lease_generation=int(lease_generation),
            state=state,
        ):
            return row

        target.execute(
            """
            UPDATE assembly_slot_occupancies
            SET structural_view_id = ?,
                binding_id = ?,
                binding_value_id = ?,
                coverage_plan_hash = ?,
                contributor_daemon_id = ?,
                coordinator_operation_id = ?,
                coordinator_generation = ?,
                lease_id = ?,
                lease_generation = ?,
                lease_expires_at = ?,
                state = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE attempt_id = ? AND slot_id = ?
              AND (
                NOT (
                  state = 'accepted'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > now()
                  AND EXISTS (
                    SELECT 1
                    FROM workers
                    WHERE workers.inactive_at IS NULL
                      AND (
                        workers.daemon_id = assembly_slot_occupancies.contributor_daemon_id
                        OR workers.worker_id = assembly_slot_occupancies.contributor_daemon_id
                      )
                  )
                )
                OR (
                  assembly_slot_occupancies.coordinator_operation_id = ?
                  AND assembly_slot_occupancies.coordinator_generation = ?
                )
              )
            """,
            [
                structural_view_id,
                binding_id,
                binding_value_id,
                coverage_plan_hash,
                contributor_daemon_id,
                coordinator_operation_id,
                int(coordinator_generation),
                lease_id,
                int(lease_generation),
                lease_expires_at,
                state,
                attempt_id,
                slot_id,
                coordinator_operation_id,
                int(coordinator_generation),
            ],
        )
        row = self.get(attempt_id=attempt_id, slot_id=slot_id, cursor=target)
        if row is None:
            raise ValueError("assembly slot occupancy disappeared during claim")
        if self._matches_claim(
            row=row,
            structural_view_id=structural_view_id,
            binding_id=binding_id,
            binding_value_id=binding_value_id,
            coverage_plan_hash=coverage_plan_hash,
            contributor_daemon_id=contributor_daemon_id,
            coordinator_operation_id=coordinator_operation_id,
            coordinator_generation=int(coordinator_generation),
            lease_id=lease_id,
            lease_generation=int(lease_generation),
            state=state,
        ):
            return row
        return None

    def list(
        self,
        *,
        attempt_id: str | None = None,
        slot_id: str | None = None,
        binding_id: str | None = None,
        binding_value_id: str | None = None,
        states: tuple[str, ...] | None = None,
        cursor=None,
    ) -> list[dict[str, Any]]:
        target = cursor if cursor is not None else self.get_cursor()
        clauses: list[str] = []
        params: list[object] = []
        if attempt_id:
            clauses.append("attempt_id = ?")
            params.append(attempt_id)
        if slot_id:
            clauses.append("slot_id = ?")
            params.append(slot_id)
        if binding_id:
            clauses.append("binding_id = ?")
            params.append(binding_id)
        if binding_value_id:
            clauses.append("binding_value_id = ?")
            params.append(binding_value_id)
        if states:
            placeholders = ",".join(["?"] * len(states))
            clauses.append(f"state IN ({placeholders})")
            params.extend(states)
        where_clause = ""
        if clauses:
            where_clause = "WHERE " + " AND ".join(clauses)
        rows = target.execute(
            f"""
            SELECT {self._SELECT_COLUMNS}
            FROM assembly_slot_occupancies
            {where_clause}
            ORDER BY attempt_id, slot_id
            """,
            params,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_state_if_current(
        self,
        *,
        attempt_id: str,
        slot_id: str,
        state: str,
        expected_lease_id: str | None = None,
        expected_lease_generation: int | None = None,
        current_states: tuple[str, ...] | None = None,
        lease_expires_at: datetime | None = None,
        cursor=None,
    ) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        clauses = ["attempt_id = ?", "slot_id = ?"]
        params: list[object] = [attempt_id, slot_id]
        if expected_lease_id is not None:
            clauses.append("lease_id = ?")
            params.append(expected_lease_id)
        if expected_lease_generation is not None:
            clauses.append("lease_generation = ?")
            params.append(int(expected_lease_generation))
        if current_states:
            placeholders = ",".join(["?"] * len(current_states))
            clauses.append(f"state IN ({placeholders})")
            params.extend(current_states)
        set_clauses = ["state = ?", "updated_at = CURRENT_TIMESTAMP"]
        set_params: list[object] = [state]
        if lease_expires_at is not None or state == "accepted":
            set_clauses.append("lease_expires_at = ?")
            set_params.append(lease_expires_at)
        target.execute(
            f"""
            UPDATE assembly_slot_occupancies
            SET {", ".join(set_clauses)}
            WHERE {" AND ".join(clauses)}
            """,
            [*set_params, *params],
        )
        return self.get(attempt_id=attempt_id, slot_id=slot_id, cursor=target)

    @staticmethod
    def _row_to_dict(row: tuple[object, ...]) -> dict[str, Any]:
        return {
            "attempt_id": str(row[0]),
            "slot_id": str(row[1]),
            "structural_view_id": (str(row[2]) if row[2] is not None else None),
            "binding_id": str(row[3]),
            "binding_value_id": str(row[4]),
            "coverage_plan_hash": str(row[5]),
            "contributor_daemon_id": str(row[6]),
            "coordinator_operation_id": str(row[7]),
            "coordinator_generation": int(
                cast(SupportsInt | SupportsIndex | str | bytes | bytearray, row[8])
            ),
            "lease_id": str(row[9]),
            "lease_generation": int(
                cast(SupportsInt | SupportsIndex | str | bytes | bytearray, row[10])
            ),
            "lease_expires_at": row[11],
            "state": str(row[12]),
            "created_at": row[13],
            "updated_at": row[14],
        }

    @staticmethod
    def _matches_claim(
        *,
        row: dict[str, Any],
        structural_view_id: str | None,
        binding_id: str,
        binding_value_id: str,
        coverage_plan_hash: str,
        contributor_daemon_id: str,
        coordinator_operation_id: str,
        coordinator_generation: int,
        lease_id: str,
        lease_generation: int,
        state: str,
    ) -> bool:
        return (
            (row.get("structural_view_id") or None) == structural_view_id
            and str(row["binding_id"]) == binding_id
            and str(row["binding_value_id"]) == binding_value_id
            and str(row["coverage_plan_hash"]) == coverage_plan_hash
            and str(row["contributor_daemon_id"]) == contributor_daemon_id
            and str(row["coordinator_operation_id"]) == coordinator_operation_id
            and int(row["coordinator_generation"]) == int(coordinator_generation)
            and str(row["lease_id"]) == lease_id
            and int(row["lease_generation"]) == int(lease_generation)
            and str(row["state"]) == state
        )


__all__ = ["AssemblySlotOccupancyRepository"]
