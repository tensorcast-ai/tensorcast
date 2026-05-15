#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for progressive replica coverage and source assignments."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from tensorcast.global_store.exceptions import NotFoundError, ValidationError
from tensorcast.global_store.models import (
    ProgressiveAssignment,
    ProgressiveAssignmentState,
    ProgressiveClaimResult,
    ProgressiveCoverageIdentity,
    ProgressiveCoverageKind,
    ProgressiveCoverageReport,
    ProgressiveCoverageRow,
    ProgressiveCoverageState,
    ProgressiveExportState,
    ProgressiveSourceMemory,
    ProgressiveSourceTransport,
)
from tensorcast.global_store.repositories.base import BaseRepository

_COVERAGE_COLUMNS = """
    coverage_id, artifact_id, byte_space_kind, byte_space_id, selection_hash,
    logical_layout_hash, hash_space_kind, hash_space_id,
    canonical_index_multihash, coverage_order_hash, group_version_set_id,
    group_part_id, replica_id, daemon_id, daemon_session_id, worker_id,
    source_export_generation, coverage_epoch, coverage_kind, state,
    export_state, verified_units, verified_bytes, completed_units,
    completed_bytes, total_units, total_bytes, materialization_attempt_id,
    source_transport_id, source_domain, seed_transport_kind, deadline_at,
    created_at, updated_at
"""

_ASSIGNMENT_COLUMNS = """
    assignment_id, coverage_id, requester_daemon_id, requester_worker_id,
    requester_materialization_attempt_id, source_daemon_id, source_worker_id,
    source_domain, seed_transport_kind, start_unit, end_unit_exclusive,
    start_byte, end_byte_exclusive, source_export_generation, state,
    deadline_at, created_at, updated_at
"""

_COVERAGE_COLUMNS_C = ", ".join(
    f"c.{column.strip()}"
    for column in _COVERAGE_COLUMNS.replace("\n", " ").split(",")
    if column.strip()
)
_ASSIGNMENT_COLUMNS_A = ", ".join(
    f"a.{column.strip()}"
    for column in _ASSIGNMENT_COLUMNS.replace("\n", " ").split(",")
    if column.strip()
)

_ACTIVE_ASSIGNMENT_STATES = ("claimed", "reading")
_TERMINAL_ASSIGNMENT_STATES = ("succeeded", "failed", "expired", "cancelled")
_ACTIVE_COVERAGE_STATES = ("pending", "verified")
_SOURCE_MEMORY_FIELD_COUNT = 9
_ASSIGNMENT_FIELD_COUNT = 18


def _dt(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _text(value: Any) -> str:
    return str(value or "")


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _text_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item) for item in value)


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item) for item in value)


def _source_memory_from_fields(
    source_fields: tuple[Any, ...],
) -> ProgressiveSourceMemory | None:
    if len(source_fields) < _SOURCE_MEMORY_FIELD_COUNT:
        return None
    return ProgressiveSourceMemory(
        node_id=_text(source_fields[0]),
        node_address=_text(source_fields[1]),
        node_port=int(source_fields[2]),
        memory_size=int(source_fields[3]),
        memory_type=_text(source_fields[4]),
        device_id=int(source_fields[5]),
        transport=ProgressiveSourceTransport(
            remote_memory_keys=_text_tuple(source_fields[6]),
            buffer_sizes=_int_tuple(source_fields[7]),
            verification_json=_optional_text(source_fields[8]),
        ),
    )


class ProgressiveCoverageRepository(BaseRepository):
    """Data access for progressive partial-source dissemination."""

    @staticmethod
    def _identity_from_row(
        row: tuple[Any, ...], offset: int = 1
    ) -> ProgressiveCoverageIdentity:
        return ProgressiveCoverageIdentity(
            artifact_id=_text(row[offset]),
            byte_space_kind=_text(row[offset + 1]),
            byte_space_id=_text(row[offset + 2]),
            selection_hash=_text(row[offset + 3]),
            logical_layout_hash=_text(row[offset + 4]),
            hash_space_kind=_text(row[offset + 5]),
            hash_space_id=_text(row[offset + 6]),
            canonical_index_multihash=_text(row[offset + 7]),
            coverage_order_hash=_text(row[offset + 8]),
            group_version_set_id=_text(row[offset + 9]),
            group_part_id=_text(row[offset + 10]),
        )

    @classmethod
    def _row_to_coverage(cls, row: tuple[Any, ...]) -> ProgressiveCoverageRow:
        return ProgressiveCoverageRow(
            coverage_id=_text(row[0]),
            identity=cls._identity_from_row(row),
            replica_id=_text(row[12]),
            daemon_id=_text(row[13]),
            daemon_session_id=_optional_text(row[14]),
            worker_id=_text(row[15]),
            source_export_generation=int(row[16]),
            coverage_epoch=int(row[17]),
            coverage_kind=ProgressiveCoverageKind(_text(row[18])),
            state=ProgressiveCoverageState(_text(row[19])),
            export_state=ProgressiveExportState(_text(row[20])),
            verified_units=int(row[21]),
            verified_bytes=int(row[22]),
            completed_units=int(row[23]),
            completed_bytes=int(row[24]),
            total_units=int(row[25]),
            total_bytes=int(row[26]),
            materialization_attempt_id=_text(row[27]),
            source_transport_id=_optional_text(row[28]),
            source_domain=_text(row[29]),
            seed_transport_kind=_optional_text(row[30]),
            deadline_at=_dt(row[31]),
            created_at=_dt(row[32]),
            updated_at=_dt(row[33]),
        )

    @staticmethod
    def _row_to_assignment(
        row: tuple[Any, ...], *, replica_id: str = ""
    ) -> ProgressiveAssignment:
        if len(row) < _ASSIGNMENT_FIELD_COUNT:
            raise ValidationError(
                "progressive assignment query returned "
                f"{len(row)} fields; expected at least {_ASSIGNMENT_FIELD_COUNT}"
            )
        source_fields = row[_ASSIGNMENT_FIELD_COUNT:]
        source_memory = _source_memory_from_fields(source_fields)
        return ProgressiveAssignment(
            assignment_id=_text(row[0]),
            coverage_id=_text(row[1]),
            requester_daemon_id=_text(row[2]),
            requester_worker_id=_text(row[3]),
            requester_materialization_attempt_id=_text(row[4]),
            source_daemon_id=_text(row[5]),
            source_worker_id=_text(row[6]),
            source_domain=_text(row[7]),
            seed_transport_kind=_optional_text(row[8]),
            start_unit=int(row[9]),
            end_unit_exclusive=int(row[10]),
            start_byte=int(row[11]),
            end_byte_exclusive=int(row[12]),
            source_export_generation=int(row[13]),
            state=ProgressiveAssignmentState(_text(row[14])),
            deadline_at=row[15],
            created_at=_dt(row[16]),
            updated_at=_dt(row[17]),
            replica_id=replica_id,
            source_memory=source_memory,
        )

    @staticmethod
    def _identity_matches(
        left: ProgressiveCoverageIdentity, right: ProgressiveCoverageIdentity
    ) -> bool:
        return left == right

    def get_coverage(
        self, coverage_id: str, *, cursor=None
    ) -> ProgressiveCoverageRow | None:
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                f"SELECT {_COVERAGE_COLUMNS} FROM replica_progress_coverage WHERE coverage_id = ?",
                [coverage_id],
            ).fetchone()
            return self._row_to_coverage(row) if row else None
        finally:
            if owns_cursor:
                cursor.close()

    def get_coverage_by_attempt(
        self,
        *,
        materialization_attempt_id: str,
        replica_id: str,
        source_export_generation: int,
        cursor=None,
    ) -> ProgressiveCoverageRow | None:
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                f"""
                SELECT {_COVERAGE_COLUMNS}
                FROM replica_progress_coverage
                WHERE materialization_attempt_id = ?
                  AND replica_id = ?
                  AND source_export_generation = ?
                """,
                [materialization_attempt_id, replica_id, int(source_export_generation)],
            ).fetchone()
            return self._row_to_coverage(row) if row else None
        finally:
            if owns_cursor:
                cursor.close()

    def validate_worker_source_domain(
        self,
        *,
        worker_id: str,
        daemon_id: str,
        source_domain: str,
        role: str,
    ) -> None:
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT daemon_id, node_id
                FROM workers
                WHERE worker_id = ?
                """,
                [worker_id],
            ).fetchone()
        finally:
            cursor.close()
        if row is None:
            raise ValidationError(
                f"progressive {role} worker directory entry is required"
            )
        directory_daemon_id = _text(row[0])
        if directory_daemon_id != daemon_id:
            raise ValidationError(
                f"progressive {role} daemon_id does not match worker directory"
            )
        expected_domain = _text(row[1]).strip() or directory_daemon_id
        if expected_domain != source_domain:
            raise ValidationError(
                f"progressive {role} source_domain does not match worker directory"
            )

    def upsert_coverage(self, report: ProgressiveCoverageReport) -> None:
        with self.transaction() as tx:
            existing = self.get_coverage(report.coverage_id, cursor=tx)
            unique_existing = self.get_coverage_by_attempt(
                materialization_attempt_id=report.materialization_attempt_id,
                replica_id=report.replica_id,
                source_export_generation=report.source_export_generation,
                cursor=tx,
            )
            if (
                unique_existing is not None
                and unique_existing.coverage_id != report.coverage_id
            ):
                raise ValidationError(
                    "coverage row already exists for materialization attempt, "
                    "replica, and export generation with a different coverage_id"
                )
            if existing is None:
                self._insert_coverage(report, tx)
                return
            self._validate_coverage_update(existing, report)
            self._update_coverage(report, tx)

    def _insert_coverage(self, report: ProgressiveCoverageReport, tx) -> None:
        identity = report.identity
        tx.execute(
            """
            INSERT INTO replica_progress_coverage (
                coverage_id, artifact_id, byte_space_kind, byte_space_id,
                selection_hash, logical_layout_hash, hash_space_kind,
                hash_space_id, canonical_index_multihash, coverage_order_hash,
                group_version_set_id, group_part_id, replica_id, daemon_id,
                daemon_session_id, worker_id, source_export_generation,
                coverage_epoch, coverage_kind, state, export_state,
                verified_units, verified_bytes, completed_units, completed_bytes,
                total_units, total_bytes, materialization_attempt_id,
                source_transport_id, source_domain, seed_transport_kind, deadline_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                report.coverage_id,
                identity.artifact_id,
                identity.byte_space_kind,
                identity.byte_space_id,
                identity.selection_hash,
                identity.logical_layout_hash,
                identity.hash_space_kind,
                identity.hash_space_id,
                identity.canonical_index_multihash,
                identity.coverage_order_hash,
                identity.group_version_set_id,
                identity.group_part_id,
                report.replica_id,
                report.daemon_id,
                report.daemon_session_id,
                report.worker_id,
                int(report.source_export_generation),
                int(report.coverage_epoch),
                report.coverage_kind.value,
                report.state.value,
                report.export_state.value,
                int(report.verified_units),
                int(report.verified_bytes),
                int(report.completed_units),
                int(report.completed_bytes),
                int(report.total_units),
                int(report.total_bytes),
                report.materialization_attempt_id,
                report.source_transport_id,
                report.source_domain,
                report.seed_transport_kind,
                report.deadline_at,
            ],
        )

    def _update_coverage(self, report: ProgressiveCoverageReport, tx) -> None:
        tx.execute(
            """
            UPDATE replica_progress_coverage
            SET daemon_session_id = ?,
                daemon_id = ?,
                worker_id = ?,
                coverage_epoch = ?,
                state = ?,
                export_state = ?,
                verified_units = ?,
                verified_bytes = ?,
                completed_units = ?,
                completed_bytes = ?,
                total_units = ?,
                total_bytes = ?,
                source_transport_id = ?,
                source_domain = ?,
                seed_transport_kind = ?,
                deadline_at = ?,
                updated_at = now()
            WHERE coverage_id = ?
            """,
            [
                report.daemon_session_id,
                report.daemon_id,
                report.worker_id,
                int(report.coverage_epoch),
                report.state.value,
                report.export_state.value,
                int(report.verified_units),
                int(report.verified_bytes),
                int(report.completed_units),
                int(report.completed_bytes),
                int(report.total_units),
                int(report.total_bytes),
                report.source_transport_id,
                report.source_domain,
                report.seed_transport_kind,
                report.deadline_at,
                report.coverage_id,
            ],
        )

    def _validate_coverage_update(
        self,
        existing: ProgressiveCoverageRow,
        report: ProgressiveCoverageReport,
    ) -> None:
        if not self._identity_matches(existing.identity, report.identity):
            raise ValidationError("progressive coverage identity cannot change")
        if existing.replica_id != report.replica_id:
            raise ValidationError("progressive coverage replica_id cannot change")
        if existing.source_export_generation != report.source_export_generation:
            raise ValidationError(
                "progressive coverage source_export_generation cannot change"
            )
        if existing.materialization_attempt_id != report.materialization_attempt_id:
            raise ValidationError(
                "progressive coverage materialization_attempt_id cannot change"
            )
        if report.coverage_epoch <= existing.coverage_epoch:
            raise ValidationError("coverage_epoch must increase monotonically")
        if report.verified_units < existing.verified_units:
            raise ValidationError("verified_units cannot decrease")
        if report.verified_bytes < existing.verified_bytes:
            raise ValidationError("verified_bytes cannot decrease")
        if report.completed_units < existing.completed_units:
            raise ValidationError("completed_units cannot decrease")
        if report.completed_bytes < existing.completed_bytes:
            raise ValidationError("completed_bytes cannot decrease")
        if report.total_units != existing.total_units:
            raise ValidationError("total_units cannot change for coverage_id")
        if report.total_bytes != existing.total_bytes:
            raise ValidationError("total_bytes cannot change for coverage_id")

    def find_progressive_source(
        self,
        *,
        identity: ProgressiveCoverageIdentity,
        next_unit: int,
        max_units: int,
        requester_daemon_id: str,
        requester_worker_id: str,
        requester_source_domain: str,
        requester_materialization_attempt_id: str,
        request_fingerprint: bytes,
        deadline_at: datetime,
        heartbeat_timeout_seconds: float,
        max_outgoing_per_source: int,
        allow_cross_domain_seed_sources: bool,
        min_verified_bytes: int,
        min_assignment_bytes: int,
        max_assignment_bytes: int,
        max_assignments_per_materialization: int,
        candidate_scan_limit: int,
        source_domain_policy: str,
    ) -> ProgressiveClaimResult:
        with self.transaction() as tx:
            replay = self._find_assignment_by_fingerprint(
                request_fingerprint,
                tx,
                identity=identity,
                next_unit=next_unit,
                requester_daemon_id=requester_daemon_id,
                requester_worker_id=requester_worker_id,
                requester_materialization_attempt_id=requester_materialization_attempt_id,
            )
            if replay is not None:
                return ProgressiveClaimResult(assignment=replay, replayed=True)
            if self._request_fingerprint_exists(request_fingerprint, tx):
                return ProgressiveClaimResult(
                    assignment=None,
                    no_eligible_reason="request_fingerprint_conflict",
                )

            if requester_materialization_attempt_id:
                count_row = tx.execute(
                    """
                    SELECT COUNT(*)
                    FROM progressive_source_assignments
                    WHERE requester_daemon_id = ?
                      AND requester_worker_id = ?
                      AND requester_materialization_attempt_id = ?
                    """,
                    [
                        requester_daemon_id,
                        requester_worker_id,
                        requester_materialization_attempt_id,
                    ],
                ).fetchone()
                if (
                    count_row
                    and int(count_row[0]) >= max_assignments_per_materialization
                ):
                    return ProgressiveClaimResult(
                        assignment=None,
                        no_eligible_reason="max_assignments_per_materialization",
                    )

            candidates = self._load_candidates(
                identity=identity,
                next_unit=next_unit,
                heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                min_verified_bytes=min_verified_bytes,
                candidate_scan_limit=candidate_scan_limit,
                cursor=tx,
            )
            skipped_reason = "not_found"
            for row in candidates:
                coverage = self._row_to_coverage(row[:34])
                replica_id = coverage.replica_id
                if self._skip_for_domain_policy(
                    source_domain=coverage.source_domain,
                    seed_transport_kind=coverage.seed_transport_kind,
                    requester_source_domain=requester_source_domain,
                    allow_cross_domain_seed_sources=allow_cross_domain_seed_sources,
                    source_domain_policy=source_domain_policy,
                ):
                    skipped_reason = "source_domain_policy"
                    continue
                segment_end = self._segment_end(
                    coverage=coverage,
                    next_unit=next_unit,
                    max_units=max_units,
                    min_assignment_bytes=min_assignment_bytes,
                    max_assignment_bytes=max_assignment_bytes,
                )
                if segment_end <= next_unit:
                    skipped_reason = "segment_too_small"
                    continue
                self._ensure_counter(
                    tx,
                    replica_id=replica_id,
                    daemon_id=coverage.daemon_id,
                    source_export_generation=coverage.source_export_generation,
                )
                claimed = tx.execute(
                    """
                    UPDATE progressive_source_counters
                    SET active_assignments = active_assignments + 1,
                        last_assigned_at = now(),
                        updated_at = now()
                    WHERE source_replica_id = ?
                      AND source_export_generation = ?
                      AND active_assignments < ?
                    RETURNING active_assignments
                    """,
                    [
                        replica_id,
                        int(coverage.source_export_generation),
                        int(max_outgoing_per_source),
                    ],
                ).fetchone()
                if not claimed:
                    skipped_reason = "source_cap"
                    continue
                assignment_id = str(uuid.uuid4())
                tx.execute(
                    """
                    INSERT INTO progressive_source_assignments (
                        assignment_id, coverage_id, requester_daemon_id,
                        requester_worker_id, requester_materialization_attempt_id,
                        source_daemon_id, source_worker_id, source_domain,
                        seed_transport_kind, start_unit, end_unit_exclusive,
                        start_byte, end_byte_exclusive, source_export_generation,
                        state, deadline_at, request_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'claimed', ?, ?)
                    """,
                    [
                        assignment_id,
                        coverage.coverage_id,
                        requester_daemon_id,
                        requester_worker_id,
                        requester_materialization_attempt_id,
                        coverage.daemon_id,
                        coverage.worker_id,
                        coverage.source_domain,
                        coverage.seed_transport_kind,
                        int(next_unit),
                        int(segment_end),
                        int(next_unit),
                        int(segment_end),
                        int(coverage.source_export_generation),
                        deadline_at,
                        request_fingerprint,
                    ],
                )
                assignment = self._find_assignment_by_id(
                    assignment_id, tx, replica_id=replica_id
                )
                return ProgressiveClaimResult(assignment=assignment, replayed=False)
            return ProgressiveClaimResult(
                assignment=None, no_eligible_reason=skipped_reason
            )

    def _load_candidates(
        self,
        *,
        identity: ProgressiveCoverageIdentity,
        next_unit: int,
        heartbeat_timeout_seconds: float,
        min_verified_bytes: int,
        candidate_scan_limit: int,
        cursor,
    ) -> list[tuple[Any, ...]]:
        heartbeat_cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=max(0.0, float(heartbeat_timeout_seconds))
        )
        return cursor.execute(
            f"""
            SELECT {_COVERAGE_COLUMNS_C},
                   wl.last_heartbeat,
                   wl.accepting_new_requests,
                   w.inactive_at,
                   ar.export_state,
                   ar.export_generation
            FROM replica_progress_coverage c
            JOIN artifact_replicas ar ON CAST(ar.replica_id AS TEXT) = c.replica_id
            JOIN workers w ON w.worker_id = c.worker_id
            JOIN worker_liveness wl ON wl.worker_id = c.worker_id
            WHERE c.artifact_id = ?
              AND c.byte_space_kind = ?
              AND c.byte_space_id = ?
              AND c.selection_hash = ?
              AND c.logical_layout_hash = ?
              AND c.hash_space_kind = ?
              AND c.hash_space_id = ?
              AND c.canonical_index_multihash = ?
              AND c.coverage_order_hash = ?
              AND c.group_version_set_id = ?
              AND c.group_part_id = ?
              AND c.coverage_kind = 'byte_prefix'
              AND c.state = 'verified'
              AND c.export_state IN ('in_progress_exportable', 'complete_exportable')
              AND c.verified_units > ?
              AND c.verified_bytes >= ?
              AND c.verified_bytes >= ?
              AND c.deadline_at > now()
              AND w.daemon_id = c.daemon_id
              AND w.inactive_at IS NULL
              AND (CASE WHEN TRIM(w.node_id) = '' THEN w.daemon_id ELSE w.node_id END) = c.source_domain
              AND wl.accepting_new_requests = TRUE
              AND wl.last_heartbeat >= ?
              AND ar.worker_id = c.worker_id
              AND ar.export_state = 'EXPORTABLE'
              AND ar.export_generation = c.source_export_generation
            ORDER BY c.verified_bytes DESC, c.updated_at ASC
            LIMIT ?
            """,
            [
                identity.artifact_id,
                identity.byte_space_kind,
                identity.byte_space_id,
                identity.selection_hash,
                identity.logical_layout_hash,
                identity.hash_space_kind,
                identity.hash_space_id,
                identity.canonical_index_multihash,
                identity.coverage_order_hash,
                identity.group_version_set_id,
                identity.group_part_id,
                int(next_unit),
                int(next_unit),
                int(min_verified_bytes),
                heartbeat_cutoff,
                int(candidate_scan_limit),
            ],
        ).fetchall()

    @staticmethod
    def _segment_end(
        *,
        coverage: ProgressiveCoverageRow,
        next_unit: int,
        max_units: int,
        min_assignment_bytes: int,
        max_assignment_bytes: int,
    ) -> int:
        verified_end = min(int(coverage.verified_units), int(coverage.verified_bytes))
        if verified_end <= next_unit:
            return next_unit
        requested_limit = int(max_units) if int(max_units) > 0 else max_assignment_bytes
        bounded_max = max(1, min(int(max_assignment_bytes), requested_limit))
        segment_end = min(verified_end, int(next_unit) + bounded_max)
        terminal_tail = coverage.total_units > 0 and segment_end >= coverage.total_units
        if (
            segment_end - int(next_unit) < int(min_assignment_bytes)
            and not terminal_tail
        ):
            return next_unit
        return segment_end

    @staticmethod
    def _skip_for_domain_policy(
        *,
        source_domain: str,
        seed_transport_kind: str | None,
        requester_source_domain: str,
        allow_cross_domain_seed_sources: bool,
        source_domain_policy: str,
    ) -> bool:
        source = str(source_domain or "").strip()
        requester = str(requester_source_domain or "").strip()
        policy = str(source_domain_policy or "").strip().lower()
        if policy == "local_only" and source and requester and source != requester:
            return True
        if allow_cross_domain_seed_sources:
            return False
        seed_kind = str(seed_transport_kind or "").strip().lower()
        seed_like_kinds = {"tcp", "p2p"}
        return (
            seed_kind in seed_like_kinds
            and source != ""
            and requester != ""
            and source != requester
        )

    @staticmethod
    def _ensure_counter(
        tx,
        *,
        replica_id: str,
        daemon_id: str,
        source_export_generation: int,
    ) -> None:
        tx.execute(
            """
            INSERT INTO progressive_source_counters (
                source_replica_id, source_daemon_id, source_export_generation,
                active_assignments
            ) VALUES (?, ?, ?, 0)
            ON CONFLICT (source_replica_id, source_export_generation) DO NOTHING
            """,
            [replica_id, daemon_id, int(source_export_generation)],
        )

    def _find_assignment_by_fingerprint(
        self,
        request_fingerprint: bytes,
        cursor,
        *,
        identity: ProgressiveCoverageIdentity,
        next_unit: int,
        requester_daemon_id: str,
        requester_worker_id: str,
        requester_materialization_attempt_id: str,
    ) -> ProgressiveAssignment | None:
        row = cursor.execute(
            f"""
            SELECT {_ASSIGNMENT_COLUMNS_A}, c.replica_id,
                   ar.node_id, ar.node_address, ar.node_port, ar.memory_size,
                   ar.memory_type, ar.device_id, ar.remote_memory_keys,
                   ar.buffer_sizes, ar.verification_json
            FROM progressive_source_assignments a
            JOIN replica_progress_coverage c ON c.coverage_id = a.coverage_id
            JOIN artifact_replicas ar ON CAST(ar.replica_id AS TEXT) = c.replica_id
            WHERE a.request_fingerprint = ?
              AND a.requester_daemon_id = ?
              AND a.requester_worker_id = ?
              AND a.requester_materialization_attempt_id = ?
              AND a.start_unit = ?
              AND c.artifact_id = ?
              AND c.byte_space_kind = ?
              AND c.byte_space_id = ?
              AND c.selection_hash = ?
              AND c.logical_layout_hash = ?
              AND c.hash_space_kind = ?
              AND c.hash_space_id = ?
              AND c.canonical_index_multihash = ?
              AND c.coverage_order_hash = ?
              AND c.group_version_set_id = ?
              AND c.group_part_id = ?
            """,
            [
                request_fingerprint,
                requester_daemon_id,
                requester_worker_id,
                requester_materialization_attempt_id,
                int(next_unit),
                identity.artifact_id,
                identity.byte_space_kind,
                identity.byte_space_id,
                identity.selection_hash,
                identity.logical_layout_hash,
                identity.hash_space_kind,
                identity.hash_space_id,
                identity.canonical_index_multihash,
                identity.coverage_order_hash,
                identity.group_version_set_id,
                identity.group_part_id,
            ],
        ).fetchone()
        if not row:
            return None
        return self._row_to_assignment(row[:18] + row[19:], replica_id=_text(row[18]))

    @staticmethod
    def _request_fingerprint_exists(request_fingerprint: bytes, cursor) -> bool:
        row = cursor.execute(
            """
            SELECT 1
            FROM progressive_source_assignments
            WHERE request_fingerprint = ?
            LIMIT 1
            """,
            [request_fingerprint],
        ).fetchone()
        return row is not None

    def _find_assignment_by_id(
        self, assignment_id: str, cursor, *, replica_id: str = ""
    ) -> ProgressiveAssignment | None:
        row = cursor.execute(
            f"""
            SELECT {_ASSIGNMENT_COLUMNS_A}, c.replica_id,
                   ar.node_id, ar.node_address, ar.node_port, ar.memory_size,
                   ar.memory_type, ar.device_id, ar.remote_memory_keys,
                   ar.buffer_sizes, ar.verification_json
            FROM progressive_source_assignments a
            JOIN replica_progress_coverage c ON c.coverage_id = a.coverage_id
            JOIN artifact_replicas ar ON CAST(ar.replica_id AS TEXT) = c.replica_id
            WHERE a.assignment_id = ?
            """,
            [assignment_id],
        ).fetchone()
        if not row:
            return None
        source_replica_id = replica_id or _text(row[18])
        return self._row_to_assignment(
            row[:18] + row[19:], replica_id=source_replica_id
        )

    def complete_assignment(
        self,
        *,
        assignment_id: str,
        outcome: ProgressiveAssignmentState,
        outcome_detail: str | None,
    ) -> bool:
        if outcome.value not in _TERMINAL_ASSIGNMENT_STATES:
            raise ValidationError(
                "progressive assignment completion requires a terminal outcome"
            )
        with self.transaction() as tx:
            row = tx.execute(
                """
                SELECT a.state, c.replica_id, c.source_export_generation
                FROM progressive_source_assignments a
                JOIN replica_progress_coverage c ON c.coverage_id = a.coverage_id
                WHERE a.assignment_id = ?
                """,
                [assignment_id],
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    f"progressive assignment not found: {assignment_id}"
                )
            current_state = _text(row[0])
            if current_state in _TERMINAL_ASSIGNMENT_STATES:
                return False
            tx.execute(
                """
                UPDATE progressive_source_assignments
                SET state = ?, outcome_detail = ?, updated_at = now()
                WHERE assignment_id = ?
                """,
                [outcome.value, outcome_detail, assignment_id],
            )
            self._release_counter(
                tx,
                replica_id=_text(row[1]),
                source_export_generation=int(row[2]),
            )
            return True

    @staticmethod
    def _release_counter(
        tx,
        *,
        replica_id: str,
        source_export_generation: int,
    ) -> None:
        tx.execute(
            """
            UPDATE progressive_source_counters
            SET active_assignments = GREATEST(active_assignments - 1, 0),
                updated_at = now()
            WHERE source_replica_id = ?
              AND source_export_generation = ?
            """,
            [replica_id, int(source_export_generation)],
        )

    def retire_coverage(
        self,
        *,
        coverage_id: str | None,
        replica_id: str | None,
        daemon_id: str | None,
        source_export_generation: int | None,
        state: ProgressiveCoverageState,
        reason: str | None,
    ) -> tuple[int, int]:
        if state not in {
            ProgressiveCoverageState.FAILED,
            ProgressiveCoverageState.RETIRED,
        }:
            raise ValidationError("retire coverage requires FAILED or RETIRED state")
        with self.transaction() as tx:
            filters: list[str] = ["state IN ('pending', 'verified')"]
            args: list[Any] = []
            if coverage_id:
                filters.append("coverage_id = ?")
                args.append(coverage_id)
            if replica_id:
                filters.append("replica_id = ?")
                args.append(replica_id)
            if daemon_id:
                filters.append("daemon_id = ?")
                args.append(daemon_id)
            if source_export_generation is not None:
                filters.append("source_export_generation = ?")
                args.append(int(source_export_generation))
            where_sql = " AND ".join(filters)
            coverage_rows = tx.execute(
                f"""
                SELECT coverage_id, replica_id, source_export_generation
                FROM replica_progress_coverage
                WHERE {where_sql}
                """,
                args,
            ).fetchall()
            if not coverage_rows:
                return (0, 0)
            coverage_ids = [_text(row[0]) for row in coverage_rows]
            placeholders = ", ".join(["?"] * len(coverage_ids))
            tx.execute(
                f"""
                UPDATE replica_progress_coverage
                SET state = ?, updated_at = now()
                WHERE coverage_id IN ({placeholders})
                """,
                [state.value, *coverage_ids],
            )
            assignment_rows = tx.execute(
                f"""
                SELECT a.assignment_id, c.replica_id, c.source_export_generation
                FROM progressive_source_assignments a
                JOIN replica_progress_coverage c ON c.coverage_id = a.coverage_id
                WHERE a.coverage_id IN ({placeholders})
                  AND a.state IN ('claimed', 'reading')
                """,
                coverage_ids,
            ).fetchall()
            assignment_state = (
                ProgressiveAssignmentState.FAILED
                if state == ProgressiveCoverageState.FAILED
                else ProgressiveAssignmentState.CANCELLED
            )
            for row in assignment_rows:
                tx.execute(
                    """
                    UPDATE progressive_source_assignments
                    SET state = ?, outcome_detail = ?, updated_at = now()
                    WHERE assignment_id = ?
                    """,
                    [assignment_state.value, reason, _text(row[0])],
                )
                self._release_counter(
                    tx,
                    replica_id=_text(row[1]),
                    source_export_generation=int(row[2]),
                )
            return (len(coverage_rows), len(assignment_rows))

    def expire_progressive_state(
        self,
        *,
        coverage_batch_limit: int,
        assignment_batch_limit: int,
    ) -> tuple[int, int]:
        with self.transaction() as tx:
            assignment_rows = tx.execute(
                """
                SELECT a.assignment_id, c.replica_id, c.source_export_generation
                FROM progressive_source_assignments a
                JOIN replica_progress_coverage c ON c.coverage_id = a.coverage_id
                WHERE a.state IN ('claimed', 'reading')
                  AND a.deadline_at <= now()
                ORDER BY a.deadline_at ASC
                LIMIT ?
                """,
                [int(assignment_batch_limit)],
            ).fetchall()
            for row in assignment_rows:
                tx.execute(
                    """
                    UPDATE progressive_source_assignments
                    SET state = 'expired',
                        outcome_detail = 'assignment deadline expired',
                        updated_at = now()
                    WHERE assignment_id = ?
                    """,
                    [_text(row[0])],
                )
                self._release_counter(
                    tx,
                    replica_id=_text(row[1]),
                    source_export_generation=int(row[2]),
                )
            expired_assignment_count = len(assignment_rows)
            coverage_rows = tx.execute(
                """
                SELECT coverage_id
                FROM replica_progress_coverage
                WHERE state IN ('pending', 'verified')
                  AND deadline_at <= now()
                ORDER BY deadline_at ASC
                LIMIT ?
                """,
                [int(coverage_batch_limit)],
            ).fetchall()
            coverage_ids = [_text(row[0]) for row in coverage_rows]
            if coverage_ids:
                placeholders = ", ".join(["?"] * len(coverage_ids))
                coverage_assignment_rows = tx.execute(
                    f"""
                    SELECT a.assignment_id, c.replica_id, c.source_export_generation
                    FROM progressive_source_assignments a
                    JOIN replica_progress_coverage c ON c.coverage_id = a.coverage_id
                    WHERE a.coverage_id IN ({placeholders})
                      AND a.state IN ('claimed', 'reading')
                    """,
                    coverage_ids,
                ).fetchall()
                for row in coverage_assignment_rows:
                    tx.execute(
                        """
                        UPDATE progressive_source_assignments
                        SET state = 'cancelled',
                            outcome_detail = 'coverage deadline expired',
                            updated_at = now()
                        WHERE assignment_id = ?
                        """,
                        [_text(row[0])],
                    )
                    self._release_counter(
                        tx,
                        replica_id=_text(row[1]),
                        source_export_generation=int(row[2]),
                    )
                expired_assignment_count += len(coverage_assignment_rows)
                tx.execute(
                    f"""
                    UPDATE replica_progress_coverage
                    SET state = 'retired', updated_at = now()
                    WHERE coverage_id IN ({placeholders})
                    """,
                    coverage_ids,
                )
            return (len(coverage_ids), expired_assignment_count)
