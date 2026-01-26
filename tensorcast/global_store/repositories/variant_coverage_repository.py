#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for variant canonical coverage ranges."""

from __future__ import annotations

from typing import Iterable, Sequence

from tensorcast.global_store.repositories.base import BaseRepository


class VariantCoverageRepository(BaseRepository):
    """Data access layer for `variant_coverage_ranges`."""

    def get_ranges(
        self,
        *,
        artifact_id: str,
        view_id: str,
        cursor=None,
    ) -> list[tuple[int, int]]:
        target = cursor if cursor is not None else self.get_cursor()
        rows = target.execute(
            """
            SELECT range_offset, range_length
            FROM variant_coverage_ranges
            WHERE artifact_id = ? AND view_id = ?
            ORDER BY range_offset ASC
            """,
            [artifact_id, view_id],
        ).fetchall()
        return [(int(row[0]), int(row[1])) for row in rows]

    def replace_ranges(
        self,
        *,
        artifact_id: str,
        view_id: str,
        ranges: Iterable[tuple[int, int]],
        cursor=None,
    ) -> None:
        target = cursor if cursor is not None else self.get_cursor()
        target.execute(
            """
            DELETE FROM variant_coverage_ranges
            WHERE artifact_id = ? AND view_id = ?
            """,
            [artifact_id, view_id],
        )
        payload = [
            (artifact_id, view_id, int(offset), int(length))
            for offset, length in ranges
            if int(length) > 0
        ]
        if not payload:
            return
        target.executemany(
            """
            INSERT INTO variant_coverage_ranges (
                artifact_id,
                view_id,
                range_offset,
                range_length
            ) VALUES (?, ?, ?, ?)
            """,
            payload,
        )

    def find_overlaps(
        self,
        *,
        artifact_id: str,
        view_id: str,
        ranges: Sequence[tuple[int, int]],
        cursor=None,
    ) -> list[tuple[str, int, int]]:
        """Return (other_view_id, range_offset, range_length) overlaps."""
        target = cursor if cursor is not None else self.get_cursor()
        overlaps: list[tuple[str, int, int]] = []
        for offset, length in ranges:
            if length <= 0:
                continue
            start = int(offset)
            end = int(offset + length)
            rows = target.execute(
                """
                SELECT view_id, range_offset, range_length
                FROM variant_coverage_ranges
                WHERE artifact_id = ?
                  AND view_id != ?
                  AND range_offset < ?
                  AND (range_offset + range_length) > ?
                LIMIT 5
                """,
                [artifact_id, view_id, end, start],
            ).fetchall()
            for row in rows:
                overlaps.append((str(row[0]), int(row[1]), int(row[2])))
                if len(overlaps) >= 5:
                    return overlaps
        return overlaps
