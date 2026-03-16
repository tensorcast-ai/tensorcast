#  Copyright (c) 2026, TensorCast Team.

"""Repository for durable assembly contributor occupancy rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository


class AssemblyContributionRepository(BaseRepository):
    """Data access layer for `assembly_contributions`."""

    _SELECT_COLUMNS = """
        assembly_id,
        view_id,
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
        assembly_id: str,
        view_id: str,
        cursor=None,
    ) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            f"""
            SELECT {self._SELECT_COLUMNS}
            FROM assembly_contributions
            WHERE assembly_id = ? AND view_id = ?
            """,
            [assembly_id, view_id],
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def upsert(
        self,
        *,
        assembly_id: str,
        view_id: str,
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
    ) -> dict[str, Any]:
        target = cursor if cursor is not None else self.get_cursor()
        target.execute(
            """
            INSERT INTO assembly_contributions (
                assembly_id,
                view_id,
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (assembly_id, view_id) DO UPDATE SET
                binding_id = excluded.binding_id,
                binding_value_id = excluded.binding_value_id,
                coverage_plan_hash = excluded.coverage_plan_hash,
                contributor_daemon_id = excluded.contributor_daemon_id,
                coordinator_operation_id = excluded.coordinator_operation_id,
                coordinator_generation = excluded.coordinator_generation,
                lease_id = excluded.lease_id,
                lease_generation = excluded.lease_generation,
                lease_expires_at = excluded.lease_expires_at,
                state = excluded.state,
                updated_at = now()
            """,
            [
                assembly_id,
                view_id,
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
        row = self.get(assembly_id=assembly_id, view_id=view_id, cursor=target)
        if row is None:
            raise ValueError("failed to upsert assembly contribution")
        return row

    def claim_slot(
        self,
        *,
        assembly_id: str,
        view_id: str,
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
        """Atomically claim `(assembly_id, view_id)` unless a durable live row exists."""
        target = cursor if cursor is not None else self.get_cursor()
        insert_params = [
            assembly_id,
            view_id,
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
        ]
        target.execute(
            """
            INSERT INTO assembly_contributions (
                assembly_id,
                view_id,
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (assembly_id, view_id) DO NOTHING
            """,
            insert_params,
        )
        row = self.get(assembly_id=assembly_id, view_id=view_id, cursor=target)
        if row is None:
            raise ValueError("failed to claim assembly contribution slot")
        if self._matches_claim(
            row=row,
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
            UPDATE assembly_contributions
            SET binding_id = ?,
                binding_value_id = ?,
                coverage_plan_hash = ?,
                contributor_daemon_id = ?,
                coordinator_operation_id = ?,
                coordinator_generation = ?,
                lease_id = ?,
                lease_generation = ?,
                lease_expires_at = ?,
                state = ?,
                updated_at = now()
            WHERE assembly_id = ? AND view_id = ?
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
                        workers.daemon_id = assembly_contributions.contributor_daemon_id
                        OR workers.worker_id = assembly_contributions.contributor_daemon_id
                      )
                  )
                )
                OR (
                  assembly_contributions.coordinator_operation_id = ?
                  AND assembly_contributions.coordinator_generation = ?
                )
              )
            """,
            [
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
                assembly_id,
                view_id,
                coordinator_operation_id,
                int(coordinator_generation),
            ],
        )
        row = self.get(assembly_id=assembly_id, view_id=view_id, cursor=target)
        if row is None:
            raise ValueError("assembly contribution slot disappeared during claim")
        if self._matches_claim(
            row=row,
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

    def update_state(
        self,
        *,
        assembly_id: str,
        view_id: str,
        state: str,
        cursor=None,
    ) -> dict[str, Any]:
        target = cursor if cursor is not None else self.get_cursor()
        target.execute(
            """
            UPDATE assembly_contributions
            SET state = ?, updated_at = now()
            WHERE assembly_id = ? AND view_id = ?
            """,
            [state, assembly_id, view_id],
        )
        row = self.get(assembly_id=assembly_id, view_id=view_id, cursor=target)
        if row is None:
            raise ValueError("assembly contribution not found")
        return row

    def update_state_if_current(
        self,
        *,
        assembly_id: str,
        view_id: str,
        state: str,
        expected_lease_id: str | None = None,
        expected_lease_generation: int | None = None,
        current_states: tuple[str, ...] | None = None,
        lease_expires_at: datetime | None = None,
        cursor=None,
    ) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        clauses = ["assembly_id = ?", "view_id = ?"]
        params: list[object] = [state, lease_expires_at, assembly_id, view_id]
        if expected_lease_id is not None:
            clauses.append("lease_id = ?")
            params.append(expected_lease_id)
        if expected_lease_generation is not None:
            clauses.append("lease_generation = ?")
            params.append(int(expected_lease_generation))
        if current_states:
            placeholders = ", ".join("?" for _ in current_states)
            clauses.append(f"state IN ({placeholders})")
            params.extend(current_states)
        row = target.execute(
            f"""
            UPDATE assembly_contributions
            SET state = ?, lease_expires_at = ?, updated_at = now()
            WHERE {" AND ".join(clauses)}
            RETURNING {self._SELECT_COLUMNS}
            """,
            params,
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_by_assembly(
        self,
        *,
        assembly_id: str,
        states: tuple[str, ...] | None = None,
        cursor=None,
    ) -> list[dict[str, Any]]:
        return self.list(
            assembly_id=assembly_id,
            states=states,
            cursor=cursor,
        )

    def list(
        self,
        *,
        assembly_id: str | None = None,
        view_id: str | None = None,
        binding_id: str | None = None,
        binding_value_id: str | None = None,
        states: tuple[str, ...] | None = None,
        cursor=None,
    ) -> list[dict[str, Any]]:
        target = cursor if cursor is not None else self.get_cursor()
        clauses = ["1 = 1"]
        params: list[object] = []
        if assembly_id is not None:
            clauses.append("assembly_id = ?")
            params.append(assembly_id)
        if view_id is not None:
            clauses.append("view_id = ?")
            params.append(view_id)
        if binding_id is not None:
            clauses.append("binding_id = ?")
            params.append(binding_id)
        if binding_value_id is not None:
            clauses.append("binding_value_id = ?")
            params.append(binding_value_id)
        if states:
            placeholders = ", ".join("?" for _ in states)
            clauses.append(f"state IN ({placeholders})")
            params.extend(states)
        rows = target.execute(
            f"""
            SELECT {self._SELECT_COLUMNS}
            FROM assembly_contributions
            WHERE {" AND ".join(clauses)}
            ORDER BY assembly_id ASC, view_id ASC
            """,
            params,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_by_binding_value(
        self,
        *,
        binding_id: str,
        binding_value_id: str,
        states: tuple[str, ...] | None = None,
        cursor=None,
    ) -> list[dict[str, Any]]:
        rows = self.list(
            binding_id=binding_id,
            binding_value_id=binding_value_id,
            states=states,
            cursor=cursor,
        )
        return [
            {"assembly_id": str(row["assembly_id"]), "view_id": str(row["view_id"])}
            for row in rows
        ]

    @staticmethod
    def _row_to_dict(row: tuple[object, ...]) -> dict[str, Any]:
        return {
            "assembly_id": str(row[0]),
            "view_id": str(row[1]),
            "binding_id": str(row[2]),
            "binding_value_id": str(row[3]),
            "coverage_plan_hash": str(row[4]),
            "contributor_daemon_id": str(row[5]),
            "coordinator_operation_id": str(row[6]),
            "coordinator_generation": int(row[7]),
            "lease_id": str(row[8]),
            "lease_generation": int(row[9]),
            "lease_expires_at": row[10],
            "state": str(row[11]),
            "created_at": row[12],
            "updated_at": row[13],
        }

    @staticmethod
    def _matches_claim(
        *,
        row: dict[str, Any],
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
            row["binding_id"] == binding_id
            and row["binding_value_id"] == binding_value_id
            and row["coverage_plan_hash"] == coverage_plan_hash
            and row["contributor_daemon_id"] == contributor_daemon_id
            and row["coordinator_operation_id"] == coordinator_operation_id
            and row["coordinator_generation"] == int(coordinator_generation)
            and row["lease_id"] == lease_id
            and row["lease_generation"] == int(lease_generation)
            and row["state"] == state
        )


__all__ = ["AssemblyContributionRepository"]
