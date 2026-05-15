#  Copyright (c) 2025-2026, TensorCast Team.

"""Service layer for progressive replica dissemination."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TypeVar

from tensorcast.global_store.config import get_config
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.metrics import (
    inc_progressive_assignment,
    inc_progressive_claim_db_conflict,
    inc_progressive_coverage_report,
    inc_progressive_report_throttled,
    inc_progressive_skipped_source,
    set_progressive_assignment_cleanup_batch_size,
    set_progressive_verified_bytes,
)
from tensorcast.global_store.models import (
    ProgressiveAssignmentState,
    ProgressiveClaimResult,
    ProgressiveCoverageIdentity,
    ProgressiveCoverageKind,
    ProgressiveCoverageReport,
    ProgressiveCoverageState,
    ProgressiveReportResult,
)
from tensorcast.global_store.repositories.base import is_transient_tx_conflict
from tensorcast.global_store.repositories.progressive_repository import (
    ProgressiveCoverageRepository,
)
from tensorcast.logger import init_logger

logger = init_logger(__name__)
T = TypeVar("T")


class ProgressiveReplicationService:
    """Business logic for progressive partial-source visibility and claims."""

    def __init__(self, repository: ProgressiveCoverageRepository) -> None:
        self.repository = repository
        self.config = get_config()
        self._claim_qps_lock = threading.Lock()
        self._claim_qps_windows: dict[str, tuple[int, int]] = defaultdict(
            lambda: (0, 0)
        )

    def _retry_transient_db_call(self, *, op_name: str, fn: Callable[[], T]) -> T:
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if not is_transient_tx_conflict(exc) or attempt == max_attempts - 1:
                    raise
                inc_progressive_claim_db_conflict(op_name)
                backoff_sec = min(0.2, 0.005 * (2**attempt))
                logger.warning(
                    "Retrying progressive DB conflict op=%s attempt=%s/%s backoff_s=%.3f error=%s",
                    op_name,
                    attempt + 1,
                    max_attempts,
                    backoff_sec,
                    exc,
                )
                time.sleep(backoff_sec)
        raise RuntimeError(f"{op_name} retry loop exhausted")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _require_text(value: str, *, field_name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValidationError(f"{field_name} is required")
        return normalized

    @staticmethod
    def _validate_identity(identity: ProgressiveCoverageIdentity) -> None:
        if identity.artifact_id.startswith("msa1:"):
            raise ValidationError(
                "progressive dissemination does not support msa1 artifacts"
            )
        required = {
            "artifact_id": identity.artifact_id,
            "byte_space_kind": identity.byte_space_kind,
            "selection_hash": identity.selection_hash,
            "logical_layout_hash": identity.logical_layout_hash,
            "hash_space_kind": identity.hash_space_kind,
            "canonical_index_multihash": identity.canonical_index_multihash,
            "coverage_order_hash": identity.coverage_order_hash,
        }
        for name, value in required.items():
            if not str(value or "").strip():
                raise ValidationError(
                    f"progressive coverage identity {name} is required"
                )
        if identity.byte_space_kind != "canonical":
            raise ValidationError(
                "progressive v1 supports canonical byte-prefix coverage only"
            )
        if identity.hash_space_kind != "canonical":
            raise ValidationError("progressive v1 supports canonical hash-space only")
        has_group_version = bool(identity.group_version_set_id)
        has_group_part = bool(identity.group_part_id)
        if has_group_version != has_group_part:
            raise ValidationError(
                "group_version_set_id and group_part_id must be supplied together"
            )

    def _validate_report(self, report: ProgressiveCoverageReport) -> None:
        self._validate_identity(report.identity)
        self._require_text(report.coverage_id, field_name="coverage_id")
        self._require_text(report.replica_id, field_name="replica_id")
        self._require_text(report.daemon_id, field_name="daemon_id")
        self._require_text(report.worker_id, field_name="worker_id")
        self._require_text(report.source_domain, field_name="source_domain")
        self._require_text(
            report.materialization_attempt_id,
            field_name="materialization_attempt_id",
        )
        if report.coverage_kind != ProgressiveCoverageKind.BYTE_PREFIX:
            raise ValidationError("progressive v1 supports byte-prefix coverage only")
        if report.source_export_generation <= 0:
            raise ValidationError("source_export_generation must be > 0")
        if report.verified_units > report.completed_units:
            raise ValidationError("verified_units cannot exceed completed_units")
        if report.verified_bytes > report.completed_bytes:
            raise ValidationError("verified_bytes cannot exceed completed_bytes")
        if report.total_units > 0 and report.completed_units > report.total_units:
            raise ValidationError("completed_units cannot exceed total_units")
        if report.total_bytes > 0 and report.completed_bytes > report.total_bytes:
            raise ValidationError("completed_bytes cannot exceed total_bytes")

    def _with_default_deadline(
        self, report: ProgressiveCoverageReport
    ) -> ProgressiveCoverageReport:
        ttl_ms = max(0, int(self.config.progressive_replication.coverage_ttl_ms))
        max_deadline = self._now() + timedelta(milliseconds=ttl_ms)
        deadline_at = report.deadline_at
        if deadline_at is not None:
            if deadline_at.tzinfo is None:
                deadline_at = deadline_at.replace(tzinfo=timezone.utc)
            deadline_at = min(deadline_at, max_deadline)
        else:
            deadline_at = max_deadline
        return ProgressiveCoverageReport(
            coverage_id=report.coverage_id,
            identity=report.identity,
            replica_id=report.replica_id,
            daemon_id=report.daemon_id,
            daemon_session_id=report.daemon_session_id,
            worker_id=report.worker_id,
            source_export_generation=report.source_export_generation,
            coverage_epoch=report.coverage_epoch,
            coverage_kind=report.coverage_kind,
            state=report.state,
            export_state=report.export_state,
            verified_units=report.verified_units,
            verified_bytes=report.verified_bytes,
            completed_units=report.completed_units,
            completed_bytes=report.completed_bytes,
            total_units=report.total_units,
            total_bytes=report.total_bytes,
            materialization_attempt_id=report.materialization_attempt_id,
            source_transport_id=report.source_transport_id,
            source_domain=report.source_domain,
            seed_transport_kind=report.seed_transport_kind,
            deadline_at=deadline_at,
        )

    def report_progressive_coverage(
        self, report: ProgressiveCoverageReport
    ) -> ProgressiveReportResult:
        if not self.config.progressive_replication.enabled:
            raise ValidationError("progressive replication is disabled")
        report = self._with_default_deadline(report)
        self._validate_report(report)
        self.repository.validate_worker_source_domain(
            worker_id=report.worker_id,
            daemon_id=report.daemon_id,
            source_domain=report.source_domain,
            role="source",
        )
        existing = self.repository.get_coverage(report.coverage_id)
        if existing is not None:
            if self._is_idempotent_report_replay(existing, report):
                return ProgressiveReportResult(
                    coverage_id=report.coverage_id,
                    state=existing.state,
                    updated=False,
                    reason="replayed",
                )
            self._validate_existing_update(existing, report)
            if self._should_throttle_report(existing, report):
                inc_progressive_report_throttled("coalesced")
                return ProgressiveReportResult(
                    coverage_id=report.coverage_id,
                    state=existing.state,
                    updated=False,
                    throttled=True,
                    reason="coalesced",
                )
        self._retry_transient_db_call(
            op_name="report_progressive_coverage",
            fn=lambda: self.repository.upsert_coverage(report),
        )
        inc_progressive_coverage_report(
            state=report.state.value,
            coverage_kind=report.coverage_kind.value,
        )
        set_progressive_verified_bytes(
            coverage_kind=report.coverage_kind.value,
            value=report.verified_bytes,
        )
        return ProgressiveReportResult(
            coverage_id=report.coverage_id,
            state=report.state,
            updated=True,
        )

    @staticmethod
    def _is_idempotent_report_replay(
        existing, report: ProgressiveCoverageReport
    ) -> bool:
        if int(report.coverage_epoch) != int(existing.coverage_epoch):
            return False
        return (
            existing.identity == report.identity
            and existing.replica_id == report.replica_id
            and existing.daemon_id == report.daemon_id
            and existing.worker_id == report.worker_id
            and existing.source_domain == report.source_domain
            and existing.source_export_generation == report.source_export_generation
            and existing.coverage_kind == report.coverage_kind
            and existing.state == report.state
            and existing.export_state == report.export_state
            and existing.verified_units == report.verified_units
            and existing.verified_bytes == report.verified_bytes
            and existing.completed_units == report.completed_units
            and existing.completed_bytes == report.completed_bytes
            and existing.total_units == report.total_units
            and existing.total_bytes == report.total_bytes
            and existing.materialization_attempt_id == report.materialization_attempt_id
            and existing.seed_transport_kind == report.seed_transport_kind
        )

    @staticmethod
    def _validate_existing_update(existing, report: ProgressiveCoverageReport) -> None:
        if existing.identity != report.identity:
            raise ValidationError("progressive coverage identity cannot change")
        if existing.replica_id != report.replica_id:
            raise ValidationError("progressive coverage replica_id cannot change")
        if existing.daemon_id != report.daemon_id:
            raise ValidationError("progressive coverage daemon_id cannot change")
        if existing.worker_id != report.worker_id:
            raise ValidationError("progressive coverage worker_id cannot change")
        if existing.source_domain != report.source_domain:
            raise ValidationError("progressive coverage source_domain cannot change")
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

    def _should_throttle_report(
        self,
        existing,
        report: ProgressiveCoverageReport,
    ) -> bool:
        terminal_states = {
            ProgressiveCoverageState.FAILED,
            ProgressiveCoverageState.RETIRED,
        }
        if report.state in terminal_states or existing.state != report.state:
            return False
        if existing.export_state != report.export_state:
            return False
        delta = int(report.verified_bytes) - int(existing.verified_bytes)
        min_delta = int(self.config.progressive_replication.min_report_delta_bytes)
        if delta >= min_delta:
            return False
        if existing.updated_at is None:
            return False
        updated_at = existing.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        elapsed_ms = (self._now() - updated_at).total_seconds() * 1000.0
        return elapsed_ms < int(
            self.config.progressive_replication.min_report_interval_ms
        )

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
        deadline_at: datetime | None,
    ) -> ProgressiveClaimResult:
        self._validate_identity(identity)
        self._require_text(requester_daemon_id, field_name="requester_daemon_id")
        self._require_text(requester_worker_id, field_name="requester_worker_id")
        self._require_text(
            requester_source_domain,
            field_name="requester_source_domain",
        )
        if not request_fingerprint:
            raise ValidationError("request_fingerprint is required")
        if int(next_unit) < 0:
            raise ValidationError("next_unit must be non-negative")
        if not self.config.progressive_replication.enabled:
            inc_progressive_skipped_source("progressive_disabled")
            return ProgressiveClaimResult(
                assignment=None,
                no_eligible_reason="progressive_disabled",
            )
        source_domain_policy = (
            str(self.config.progressive_replication.source_domain_policy)
            .strip()
            .lower()
        )
        if source_domain_policy not in {"local_only", "allow_all"}:
            raise ValidationError(
                "progressive source_domain_policy must be local_only or allow_all"
            )
        self.repository.validate_worker_source_domain(
            worker_id=requester_worker_id,
            daemon_id=requester_daemon_id,
            source_domain=requester_source_domain,
            role="requester",
        )
        if not self._admit_claim_qps(requester_daemon_id):
            inc_progressive_skipped_source("claim_qps_limited")
            return ProgressiveClaimResult(
                assignment=None,
                no_eligible_reason="claim_qps_limited",
            )
        ttl_ms = max(0, int(self.config.progressive_replication.assignment_ttl_ms))
        max_deadline = self._now() + timedelta(milliseconds=ttl_ms)
        if deadline_at is not None and deadline_at.tzinfo is None:
            deadline_at = deadline_at.replace(tzinfo=timezone.utc)
        assignment_deadline = (
            min(deadline_at, max_deadline) if deadline_at is not None else max_deadline
        )
        cfg = self.config.progressive_replication
        result = self._retry_transient_db_call(
            op_name="find_progressive_source",
            fn=lambda: self.repository.find_progressive_source(
                identity=identity,
                next_unit=int(next_unit),
                max_units=int(max_units),
                requester_daemon_id=requester_daemon_id,
                requester_worker_id=requester_worker_id,
                requester_source_domain=requester_source_domain,
                requester_materialization_attempt_id=requester_materialization_attempt_id,
                request_fingerprint=request_fingerprint,
                deadline_at=assignment_deadline,
                heartbeat_timeout_seconds=float(self.config.heartbeat_timeout_ms)
                / 1000.0,
                max_outgoing_per_source=int(cfg.max_outgoing_per_source),
                allow_cross_domain_seed_sources=bool(
                    cfg.allow_cross_domain_seed_sources
                ),
                min_verified_bytes=int(cfg.min_verified_bytes),
                min_assignment_bytes=int(cfg.min_assignment_bytes),
                max_assignment_bytes=int(cfg.max_assignment_bytes),
                max_assignments_per_materialization=int(
                    cfg.max_assignments_per_materialization
                ),
                candidate_scan_limit=int(cfg.assignment_candidate_scan_limit),
                source_domain_policy=source_domain_policy,
            ),
        )
        if result.assignment is None:
            inc_progressive_skipped_source(result.no_eligible_reason or "not_found")
            return result
        inc_progressive_assignment(
            assignment_state=result.assignment.state.value,
            coverage_kind=ProgressiveCoverageKind.BYTE_PREFIX.value,
        )
        return result

    def _admit_claim_qps(self, requester_daemon_id: str) -> bool:
        cap = int(self.config.progressive_replication.max_claim_qps_per_daemon)
        if cap <= 0:
            return True
        current_sec = int(time.time())
        with self._claim_qps_lock:
            window_sec, count = self._claim_qps_windows[requester_daemon_id]
            if window_sec != current_sec:
                self._claim_qps_windows[requester_daemon_id] = (current_sec, 1)
                return True
            if count >= cap:
                return False
            self._claim_qps_windows[requester_daemon_id] = (window_sec, count + 1)
            return True

    def complete_progressive_assignment(
        self,
        *,
        assignment_id: str,
        outcome: ProgressiveAssignmentState,
        outcome_detail: str | None,
    ) -> bool:
        self._require_text(assignment_id, field_name="assignment_id")
        released = self._retry_transient_db_call(
            op_name="complete_progressive_assignment",
            fn=lambda: self.repository.complete_assignment(
                assignment_id=assignment_id,
                outcome=outcome,
                outcome_detail=outcome_detail,
            ),
        )
        inc_progressive_assignment(
            assignment_state=outcome.value,
            coverage_kind=ProgressiveCoverageKind.BYTE_PREFIX.value,
        )
        return released

    def retire_progressive_coverage(
        self,
        *,
        coverage_id: str | None,
        replica_id: str | None,
        daemon_id: str | None,
        source_export_generation: int | None,
        state: ProgressiveCoverageState,
        reason: str | None,
    ) -> tuple[int, int]:
        return self._retry_transient_db_call(
            op_name="retire_progressive_coverage",
            fn=lambda: self.repository.retire_coverage(
                coverage_id=coverage_id,
                replica_id=replica_id,
                daemon_id=daemon_id,
                source_export_generation=source_export_generation,
                state=state,
                reason=reason,
            ),
        )

    def expire_progressive_state(
        self,
        *,
        coverage_batch_limit: int | None = None,
        assignment_batch_limit: int | None = None,
    ) -> tuple[int, int]:
        cfg = self.config.progressive_replication
        coverage_limit = max(1, int(coverage_batch_limit or cfg.cleanup_batch_limit))
        assignment_limit = max(
            1, int(assignment_batch_limit or cfg.cleanup_batch_limit)
        )
        expired_coverage, expired_assignments = self._retry_transient_db_call(
            op_name="expire_progressive_state",
            fn=lambda: self.repository.expire_progressive_state(
                coverage_batch_limit=coverage_limit,
                assignment_batch_limit=assignment_limit,
            ),
        )
        set_progressive_assignment_cleanup_batch_size(expired_assignments)
        return expired_coverage, expired_assignments
