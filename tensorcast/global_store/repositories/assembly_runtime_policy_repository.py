#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for mutable assembly runtime policies (versioned CAS)."""

from __future__ import annotations

from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository


class AssemblyRuntimePolicyRepository(BaseRepository):
    """Data access layer for `assembly_runtime_policies`."""

    def get(self, *, assembly_id: str, cursor=None) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT assembly_id, policy_version, policy_json, updated_at
            FROM assembly_runtime_policies
            WHERE assembly_id = ?
            """,
            [assembly_id],
        ).fetchone()
        if not row:
            return None
        return {
            "assembly_id": str(row[0]),
            "policy_version": int(row[1]),
            "policy_json": str(row[2]),
            "updated_at": row[3],
        }

    def update(
        self,
        *,
        assembly_id: str,
        policy_json: str,
        expected_policy_version: int,
        cursor=None,
    ) -> dict[str, Any]:
        target = cursor if cursor is not None else self.get_cursor()
        existing = self.get(assembly_id=assembly_id, cursor=target)
        if existing is None:
            if expected_policy_version != 0:
                raise ValueError("policy_version conflict (missing)")
            target.execute(
                """
                INSERT INTO assembly_runtime_policies (assembly_id, policy_version, policy_json)
                VALUES (?, 1, ?)
                """,
                [assembly_id, policy_json],
            )
            created = self.get(assembly_id=assembly_id, cursor=target)
            if created is None:
                raise ValueError("failed to create policy")
            return created

        if existing["policy_version"] != expected_policy_version:
            raise ValueError("policy_version conflict")
        new_version = int(expected_policy_version) + 1
        target.execute(
            """
            UPDATE assembly_runtime_policies
            SET policy_version = ?, policy_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE assembly_id = ? AND policy_version = ?
            """,
            [new_version, policy_json, assembly_id, expected_policy_version],
        )
        updated = self.get(assembly_id=assembly_id, cursor=target)
        if updated is None:
            raise ValueError("policy update failed")
        return updated
