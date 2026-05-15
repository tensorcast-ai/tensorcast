#  Copyright (c) 2025-2026, TensorCast Team.

"""Progressive replica dissemination domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ProgressiveCoverageKind(str, Enum):
    BYTE_PREFIX = "byte_prefix"


class ProgressiveCoverageState(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    RETIRED = "retired"


class ProgressiveExportState(str, Enum):
    NOT_EXPORTABLE = "not_exportable"
    IN_PROGRESS_EXPORTABLE = "in_progress_exportable"
    COMPLETE_EXPORTABLE = "complete_exportable"


class ProgressiveAssignmentState(str, Enum):
    CLAIMED = "claimed"
    READING = "reading"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProgressiveCoverageIdentity:
    artifact_id: str
    byte_space_kind: str
    byte_space_id: str
    selection_hash: str
    logical_layout_hash: str
    hash_space_kind: str
    hash_space_id: str
    canonical_index_multihash: str
    coverage_order_hash: str
    group_version_set_id: str = ""
    group_part_id: str = ""


@dataclass(frozen=True)
class ProgressiveCoverageReport:
    coverage_id: str
    identity: ProgressiveCoverageIdentity
    replica_id: str
    daemon_id: str
    daemon_session_id: str | None
    worker_id: str
    source_export_generation: int
    coverage_epoch: int
    coverage_kind: ProgressiveCoverageKind
    state: ProgressiveCoverageState
    export_state: ProgressiveExportState
    verified_units: int
    verified_bytes: int
    completed_units: int
    completed_bytes: int
    total_units: int
    total_bytes: int
    materialization_attempt_id: str
    source_transport_id: str | None
    source_domain: str
    seed_transport_kind: str | None
    deadline_at: datetime | None


@dataclass(frozen=True)
class ProgressiveCoverageRow:
    coverage_id: str
    identity: ProgressiveCoverageIdentity
    replica_id: str
    daemon_id: str
    daemon_session_id: str | None
    worker_id: str
    source_export_generation: int
    coverage_epoch: int
    coverage_kind: ProgressiveCoverageKind
    state: ProgressiveCoverageState
    export_state: ProgressiveExportState
    verified_units: int
    verified_bytes: int
    completed_units: int
    completed_bytes: int
    total_units: int
    total_bytes: int
    materialization_attempt_id: str
    source_transport_id: str | None
    source_domain: str
    seed_transport_kind: str | None
    deadline_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True)
class ProgressiveSourceTransport:
    remote_memory_keys: tuple[str, ...] = ()
    buffer_sizes: tuple[int, ...] = ()
    verification_json: str | None = None


@dataclass(frozen=True)
class ProgressiveSourceMemory:
    node_id: str
    node_address: str
    node_port: int
    memory_size: int
    memory_type: str
    device_id: int
    transport: ProgressiveSourceTransport


@dataclass(frozen=True)
class ProgressiveAssignment:
    assignment_id: str
    coverage_id: str
    replica_id: str
    source_daemon_id: str
    source_worker_id: str
    source_domain: str
    seed_transport_kind: str | None
    requester_daemon_id: str
    requester_worker_id: str
    requester_materialization_attempt_id: str
    start_unit: int
    end_unit_exclusive: int
    start_byte: int
    end_byte_exclusive: int
    source_export_generation: int
    state: ProgressiveAssignmentState
    deadline_at: datetime
    created_at: datetime | None
    updated_at: datetime | None
    source_memory: ProgressiveSourceMemory | None = None


@dataclass(frozen=True)
class ProgressiveClaimResult:
    assignment: ProgressiveAssignment | None
    replayed: bool = False
    no_eligible_reason: str = ""


@dataclass(frozen=True)
class ProgressiveReportResult:
    coverage_id: str
    state: ProgressiveCoverageState
    updated: bool
    throttled: bool = False
    reason: str = ""
