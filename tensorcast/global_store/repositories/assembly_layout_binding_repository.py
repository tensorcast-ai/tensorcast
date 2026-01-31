#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for mutable assembly -> layout bindings (versioned CAS)."""

from __future__ import annotations

from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class AssemblyLayoutBindingRepository(BaseRepository):
    """Data access layer for `assembly_layout_bindings`."""

    def get(self, *, assembly_id: str, cursor=None) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT assembly_id, layout_id, binding_version, updated_at
            FROM assembly_layout_bindings
            WHERE assembly_id = ?
            """,
            [assembly_id],
        ).fetchone()
        if not row:
            return None
        return {
            "assembly_id": str(row[0]),
            "layout_id": str(row[1]),
            "binding_version": int(row[2]),
            "updated_at": row[3],
        }

    def update(
        self,
        *,
        assembly_id: str,
        layout_id: str,
        expected_binding_version: int,
        cursor=None,
    ) -> dict[str, Any]:
        target = cursor if cursor is not None else self.get_cursor()
        existing = self.get(assembly_id=assembly_id, cursor=target)
        if existing is None:
            if expected_binding_version != 0:
                raise ValueError("binding_version conflict (missing)")
            target.execute(
                """
                INSERT INTO assembly_layout_bindings (assembly_id, layout_id, binding_version)
                VALUES (?, ?, 1)
                """,
                [assembly_id, layout_id],
            )
            created = self.get(assembly_id=assembly_id, cursor=target)
            if created is None:
                raise ValueError("failed to create binding")
            return created

        if existing["binding_version"] != expected_binding_version:
            raise ValueError("binding_version conflict")

        new_version = int(expected_binding_version) + 1
        target.execute(
            """
            UPDATE assembly_layout_bindings
            SET layout_id = ?, binding_version = ?, updated_at = CURRENT_TIMESTAMP
            WHERE assembly_id = ? AND binding_version = ?
            """,
            [layout_id, new_version, assembly_id, expected_binding_version],
        )
        updated = self.get(assembly_id=assembly_id, cursor=target)
        if updated is None:
            raise ValueError("binding update failed")
        return updated
