#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for progressive replica dissemination control-plane state."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import grpc
import pytest

from tensorcast.global_store.config.settings import (
    GlobalStoreConfig,
    ProgressiveReplicationConfig,
    set_config,
)
from tensorcast.global_store.exceptions import DatabaseError, ValidationError
from tensorcast.global_store.models import (
    ProgressiveAssignmentState,
    ProgressiveCoverageIdentity,
    ProgressiveCoverageKind,
    ProgressiveCoverageReport,
    ProgressiveCoverageState,
    ProgressiveExportState,
)
from tensorcast.global_store.repositories import (
    ProgressiveCoverageRepository,
    ReplicaRepository,
)
from tensorcast.global_store.services import ProgressiveReplicationService
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _configure_progressive(
    *,
    enabled: bool = True,
    min_report_delta_bytes: int = 0,
    min_report_interval_ms: int = 0,
    allow_cross_domain_seed_sources: bool = False,
    source_domain_policy: str = "local_only",
) -> None:
    set_config(
        GlobalStoreConfig(
            heartbeat_timeout_ms=30_000,
            progressive_replication=ProgressiveReplicationConfig(
                enabled=enabled,
                max_outgoing_per_source=1,
                coverage_ttl_ms=60_000,
                assignment_ttl_ms=30_000,
                allow_cross_domain_seed_sources=allow_cross_domain_seed_sources,
                min_verified_bytes=1,
                min_assignment_bytes=1,
                max_assignment_bytes=64,
                max_assignments_per_materialization=8,
                assignment_candidate_scan_limit=8,
                max_claim_qps_per_daemon=0,
                cleanup_batch_limit=16,
                source_domain_policy=source_domain_policy,
                min_report_delta_bytes=min_report_delta_bytes,
                min_report_interval_ms=min_report_interval_ms,
            ),
        )
    )


def _identity(*, selection_hash: str = "73656c") -> ProgressiveCoverageIdentity:
    return ProgressiveCoverageIdentity(
        artifact_id="mi2:index:data",
        byte_space_kind="canonical",
        byte_space_id="",
        selection_hash=selection_hash,
        logical_layout_hash="6c61796f7574",
        hash_space_kind="canonical",
        hash_space_id="",
        canonical_index_multihash="bafy-index",
        coverage_order_hash="6f72646572",
        group_version_set_id="gvs-1",
        group_part_id="rank-0",
    )


def _report(
    *,
    coverage_id: str = "coverage-1",
    replica_id: str = "11111111-1111-1111-1111-111111111111",
    source_export_generation: int = 7,
    coverage_epoch: int = 1,
    verified_bytes: int = 96,
    completed_bytes: int = 96,
    total_bytes: int = 128,
    materialization_attempt_id: str = "attempt-src",
    identity: ProgressiveCoverageIdentity | None = None,
    export_state: ProgressiveExportState = ProgressiveExportState.IN_PROGRESS_EXPORTABLE,
    source_domain: str = "dc-a",
    seed_transport_kind: str | None = None,
    deadline_at: datetime | None = None,
) -> ProgressiveCoverageReport:
    return ProgressiveCoverageReport(
        coverage_id=coverage_id,
        identity=identity or _identity(),
        replica_id=replica_id,
        daemon_id="daemon-src",
        daemon_session_id="session-src",
        worker_id="worker-src",
        source_export_generation=source_export_generation,
        coverage_epoch=coverage_epoch,
        coverage_kind=ProgressiveCoverageKind.BYTE_PREFIX,
        state=ProgressiveCoverageState.VERIFIED,
        export_state=export_state,
        verified_units=verified_bytes,
        verified_bytes=verified_bytes,
        completed_units=completed_bytes,
        completed_bytes=completed_bytes,
        total_units=total_bytes,
        total_bytes=total_bytes,
        materialization_attempt_id=materialization_attempt_id,
        source_transport_id=None,
        source_domain=source_domain,
        seed_transport_kind=seed_transport_kind,
        deadline_at=deadline_at or datetime.now(timezone.utc) + timedelta(seconds=60),
    )


def _insert_worker_directory(
    conn,
    *,
    worker_id: str,
    daemon_id: str,
    source_domain: str,
    node_address: str,
    grpc_port: int,
    p2p_port: int,
) -> None:
    conn.execute(
        """
        INSERT INTO workers (
            worker_id, daemon_id, node_id, node_address, grpc_port, p2p_port,
            mem_pool_total_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            worker_id,
            daemon_id,
            source_domain,
            node_address,
            grpc_port,
            p2p_port,
            1024,
        ],
    )


def _insert_requester_directory(conn, *, source_domain: str = "dc-a") -> None:
    _insert_worker_directory(
        conn,
        worker_id="worker-dst",
        daemon_id="daemon-dst",
        source_domain=source_domain,
        node_address="10.0.0.2",
        grpc_port=51051,
        p2p_port=51052,
    )


def _insert_source_directory(conn, *, replica_id: str) -> None:
    _insert_worker_directory(
        conn,
        worker_id="worker-src",
        daemon_id="daemon-src",
        source_domain="dc-a",
        node_address="10.0.0.1",
        grpc_port=50051,
        p2p_port=50052,
    )
    conn.execute(
        """
        INSERT INTO worker_liveness (
            worker_id, last_heartbeat, mem_pool_available_size,
            accepting_new_requests
        ) VALUES (?, ?, ?, TRUE)
        """,
        ["worker-src", datetime.now(timezone.utc), 1024],
    )
    _insert_source_replica(conn, replica_id=replica_id)


def _insert_source_replica(conn, *, replica_id: str) -> None:
    conn.execute(
        """
        INSERT INTO artifact_replicas (
            replica_id, artifact_id, view_id, node_id, node_address, node_port,
            memory_size, memory_type, device_id, max_concurrency, is_available,
            remote_memory_keys, buffer_sizes, export_state, export_generation,
            worker_id, is_memory_replica
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, FALSE, ?, ?, ?, ?, ?, TRUE)
        """,
        [
            replica_id,
            "mi2:index:data",
            "dc-a",
            "10.0.0.1",
            50052,
            128,
            "GPU",
            0,
            1,
            ["rkey"],
            [128],
            "EXPORTABLE",
            7,
            "worker-src",
        ],
    )
    conn.execute(
        "INSERT INTO replica_counters (replica_id, current_requests) VALUES (?, 0)",
        [replica_id],
    )


def test_schema_progressive_tables_and_constraints(db_connection) -> None:
    tables = {row[0].lower() for row in db_connection.execute("SHOW TABLES").fetchall()}
    assert {
        "replica_progress_coverage",
        "progressive_source_assignments",
        "progressive_source_counters",
    }.issubset(tables)

    repo = ProgressiveCoverageRepository(db_connection)
    repo.upsert_coverage(_report())
    with pytest.raises((DatabaseError, ValidationError)):
        repo.upsert_coverage(_report(coverage_id="coverage-other", coverage_epoch=2))


def test_progressive_coverage_never_feeds_ordinary_source_selection(
    db_connection,
) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    repo = ProgressiveCoverageRepository(db_connection)
    repo.upsert_coverage(_report(replica_id=replica_id))

    ordinary = ReplicaRepository(db_connection).find_available_for_transport(
        "mi2:index:data",
        heartbeat_timeout_seconds=30.0,
    )

    assert ordinary.replica is None
    assert ordinary.exportable_replicas == 1


def test_progressive_report_monotonic_identity_and_throttling(db_connection) -> None:
    _configure_progressive(
        min_report_delta_bytes=1024,
        min_report_interval_ms=3_600_000,
    )
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    _insert_worker_directory(
        db_connection,
        worker_id="worker-src",
        daemon_id="daemon-src",
        source_domain="dc-a",
        node_address="10.0.0.1",
        grpc_port=50051,
        p2p_port=50052,
    )
    first = service.report_progressive_coverage(_report(verified_bytes=64))
    assert first.updated is True

    throttled = service.report_progressive_coverage(
        _report(coverage_epoch=2, verified_bytes=65)
    )
    assert throttled.throttled is True

    with pytest.raises(ValidationError):
        service.report_progressive_coverage(
            _report(coverage_epoch=3, verified_bytes=63)
        )
    with pytest.raises(ValidationError):
        service.report_progressive_coverage(
            _report(coverage_epoch=3, identity=_identity(selection_hash="64696666"))
        )


def test_progressive_report_throttle_does_not_hide_identity_changes(
    db_connection,
) -> None:
    _configure_progressive(
        min_report_delta_bytes=1024,
        min_report_interval_ms=3_600_000,
    )
    replica_id = str(uuid.uuid4())
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    _insert_worker_directory(
        db_connection,
        worker_id="worker-src",
        daemon_id="daemon-src",
        source_domain="dc-a",
        node_address="10.0.0.1",
        grpc_port=50051,
        p2p_port=50052,
    )
    service.report_progressive_coverage(_report(replica_id=replica_id))

    with pytest.raises(ValidationError, match="replica_id cannot change"):
        service.report_progressive_coverage(
            _report(
                coverage_epoch=2,
                replica_id=str(uuid.uuid4()),
                verified_bytes=97,
                completed_bytes=97,
            )
        )
    with pytest.raises(ValidationError, match="source_export_generation"):
        service.report_progressive_coverage(
            _report(
                coverage_epoch=2,
                replica_id=replica_id,
                source_export_generation=8,
                verified_bytes=97,
                completed_bytes=97,
            )
        )
    with pytest.raises(ValidationError, match="total_units cannot change"):
        service.report_progressive_coverage(
            _report(
                coverage_epoch=2,
                replica_id=replica_id,
                verified_bytes=97,
                completed_bytes=97,
                total_bytes=256,
            )
        )


def test_progressive_report_export_state_change_bypasses_throttle(
    db_connection,
) -> None:
    _configure_progressive(
        min_report_delta_bytes=1024,
        min_report_interval_ms=3_600_000,
    )
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    _insert_worker_directory(
        db_connection,
        worker_id="worker-src",
        daemon_id="daemon-src",
        source_domain="dc-a",
        node_address="10.0.0.1",
        grpc_port=50051,
        p2p_port=50052,
    )
    service.report_progressive_coverage(
        _report(export_state=ProgressiveExportState.IN_PROGRESS_EXPORTABLE)
    )

    updated = service.report_progressive_coverage(
        _report(
            coverage_epoch=2,
            verified_bytes=97,
            completed_bytes=97,
            export_state=ProgressiveExportState.COMPLETE_EXPORTABLE,
        )
    )

    assert updated.updated is True
    assert updated.throttled is False


def test_progressive_report_requires_enabled_config(db_connection) -> None:
    _configure_progressive(enabled=False)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    _insert_worker_directory(
        db_connection,
        worker_id="worker-src",
        daemon_id="daemon-src",
        source_domain="dc-a",
        node_address="10.0.0.1",
        grpc_port=50051,
        p2p_port=50052,
    )

    with pytest.raises(ValidationError, match="progressive replication is disabled"):
        service.report_progressive_coverage(_report())

    row = db_connection.execute(
        "SELECT COUNT(*) FROM replica_progress_coverage"
    ).fetchone()
    assert row == (0,)


def test_progressive_report_replay_and_source_identity_are_strict(
    db_connection,
) -> None:
    _configure_progressive()
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    _insert_worker_directory(
        db_connection,
        worker_id="worker-src",
        daemon_id="daemon-src",
        source_domain="dc-a",
        node_address="10.0.0.1",
        grpc_port=50051,
        p2p_port=50052,
    )

    first = service.report_progressive_coverage(_report())
    assert first.updated is True
    replay = service.report_progressive_coverage(_report())
    assert replay.updated is False
    assert replay.reason == "replayed"

    _insert_worker_directory(
        db_connection,
        worker_id="worker-other",
        daemon_id="daemon-other",
        source_domain="dc-b",
        node_address="10.0.0.3",
        grpc_port=52051,
        p2p_port=52052,
    )
    with pytest.raises(ValidationError, match="daemon_id cannot change"):
        service.report_progressive_coverage(
            ProgressiveCoverageReport(
                **{
                    **_report(
                        coverage_epoch=2,
                        verified_bytes=97,
                        completed_bytes=97,
                    ).__dict__,
                    "daemon_id": "daemon-other",
                    "worker_id": "worker-other",
                    "source_domain": "dc-b",
                }
            )
        )


def test_progressive_deadlines_are_capped_by_policy(db_connection) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    _insert_requester_directory(db_connection)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )

    before_report = datetime.now(timezone.utc)
    service.report_progressive_coverage(
        _report(
            replica_id=replica_id,
            deadline_at=before_report + timedelta(days=30),
        )
    )
    coverage_deadline = db_connection.execute(
        """
        SELECT deadline_at
        FROM replica_progress_coverage
        WHERE coverage_id = 'coverage-1'
        """
    ).fetchone()[0]
    if coverage_deadline.tzinfo is None:
        coverage_deadline = coverage_deadline.replace(tzinfo=timezone.utc)
    assert coverage_deadline <= before_report + timedelta(seconds=61)

    before_claim = datetime.now(timezone.utc)
    claim = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-capped-deadline",
        deadline_at=before_claim + timedelta(days=30),
    )

    assert claim.assignment is not None
    assignment_deadline = claim.assignment.deadline_at
    if assignment_deadline.tzinfo is None:
        assignment_deadline = assignment_deadline.replace(tzinfo=timezone.utc)
    assert assignment_deadline <= before_claim + timedelta(seconds=31)


def test_progressive_invalid_source_domain_policy_fails_closed(
    db_connection,
) -> None:
    _configure_progressive(source_domain_policy="allow-al")
    _insert_requester_directory(db_connection)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )

    with pytest.raises(ValidationError, match="source_domain_policy"):
        service.find_progressive_source(
            identity=_identity(),
            next_unit=0,
            max_units=32,
            requester_daemon_id="daemon-dst",
            requester_worker_id="worker-dst",
            requester_source_domain="dc-a",
            requester_materialization_attempt_id="attempt-dst",
            request_fingerprint=b"claim-bad-policy",
            deadline_at=None,
        )


def test_progressive_requires_typed_source_domains(db_connection) -> None:
    _configure_progressive()
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )

    with pytest.raises(ValidationError, match="source_domain is required"):
        service.report_progressive_coverage(_report(source_domain=""))
    with pytest.raises(ValidationError, match="requester_source_domain is required"):
        service.find_progressive_source(
            identity=_identity(),
            next_unit=0,
            max_units=32,
            requester_daemon_id="daemon-dst",
            requester_worker_id="worker-dst",
            requester_source_domain="",
            requester_materialization_attempt_id="attempt-dst",
            request_fingerprint=b"claim-no-domain",
            deadline_at=None,
        )

    _insert_worker_directory(
        db_connection,
        worker_id="worker-src",
        daemon_id="daemon-src",
        source_domain="dc-a",
        node_address="10.0.0.1",
        grpc_port=50051,
        p2p_port=50052,
    )
    with pytest.raises(ValidationError, match="source_domain does not match"):
        service.report_progressive_coverage(_report(source_domain="dc-b"))

    _insert_requester_directory(db_connection, source_domain="dc-a")
    with pytest.raises(ValidationError, match="source_domain does not match"):
        service.find_progressive_source(
            identity=_identity(),
            next_unit=0,
            max_units=32,
            requester_daemon_id="daemon-dst",
            requester_worker_id="worker-dst",
            requester_source_domain="dc-b",
            requester_materialization_attempt_id="attempt-dst",
            request_fingerprint=b"claim-domain-mismatch",
            deadline_at=None,
        )


def test_progressive_disabled_claim_does_not_require_worker_directory(
    db_connection,
) -> None:
    _configure_progressive(enabled=False)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )

    result = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-disabled",
        deadline_at=None,
    )

    assert result.assignment is None
    assert result.no_eligible_reason == "progressive_disabled"


def test_progressive_claim_replay_and_counter_release(db_connection) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    _insert_requester_directory(db_connection)
    repo = ProgressiveCoverageRepository(db_connection)
    service = ProgressiveReplicationService(repo)
    assert service.report_progressive_coverage(_report(replica_id=replica_id)).updated

    claim = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-1",
        deadline_at=None,
    )

    assert claim.assignment is not None
    assert claim.assignment.replica_id == replica_id
    assert claim.assignment.start_byte == 0
    assert claim.assignment.end_byte_exclusive == 32

    replay = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-1",
        deadline_at=None,
    )
    assert replay.replayed is True
    assert replay.assignment is not None
    assert replay.assignment.assignment_id == claim.assignment.assignment_id

    fingerprint_conflict = service.find_progressive_source(
        identity=_identity(),
        next_unit=32,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-1",
        deadline_at=None,
    )
    assert fingerprint_conflict.assignment is None
    assert fingerprint_conflict.no_eligible_reason == "request_fingerprint_conflict"

    counter_row = db_connection.execute(
        """
        SELECT active_assignments
        FROM progressive_source_counters
        WHERE source_replica_id = ?
        """,
        [replica_id],
    ).fetchone()
    assert counter_row == (1,)

    released = service.complete_progressive_assignment(
        assignment_id=claim.assignment.assignment_id,
        outcome=ProgressiveAssignmentState.SUCCEEDED,
        outcome_detail="ok",
    )
    assert released is True
    replayed_completion = service.complete_progressive_assignment(
        assignment_id=claim.assignment.assignment_id,
        outcome=ProgressiveAssignmentState.SUCCEEDED,
        outcome_detail="ok",
    )
    assert replayed_completion is False
    counter_row = db_connection.execute(
        """
        SELECT active_assignments
        FROM progressive_source_counters
        WHERE source_replica_id = ?
        """,
        [replica_id],
    ).fetchone()
    assert counter_row == (0,)

    terminal_replay = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-1",
        deadline_at=None,
    )
    assert terminal_replay.replayed is True
    assert terminal_replay.assignment is not None
    assert terminal_replay.assignment.assignment_id == claim.assignment.assignment_id
    assert terminal_replay.assignment.state == ProgressiveAssignmentState.SUCCEEDED


@pytest.mark.parametrize("seed_transport_kind", ["tcp", "p2p"])
def test_cross_domain_seed_skipped_by_default(
    db_connection,
    seed_transport_kind: str,
) -> None:
    _configure_progressive(source_domain_policy="allow_all")
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    _insert_requester_directory(db_connection, source_domain="dc-b")
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    service.report_progressive_coverage(
        _report(
            replica_id=replica_id,
            source_domain="dc-a",
            seed_transport_kind=seed_transport_kind,
        )
    )

    claim = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-b",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-cross-domain",
        deadline_at=None,
    )

    assert claim.assignment is None
    assert claim.no_eligible_reason == "source_domain_policy"


def test_progressive_source_cap_is_separate_from_ordinary_counter(
    db_connection,
) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    _insert_requester_directory(db_connection)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    service.report_progressive_coverage(_report(replica_id=replica_id))

    first = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-cap-1",
        deadline_at=None,
    )
    assert first.assignment is not None

    second = service.find_progressive_source(
        identity=_identity(),
        next_unit=32,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-cap-2",
        deadline_at=None,
    )

    assert second.assignment is None
    assert second.no_eligible_reason == "source_cap"
    ordinary_counter = db_connection.execute(
        "SELECT current_requests FROM replica_counters WHERE replica_id = ?",
        [replica_id],
    ).fetchone()
    assert ordinary_counter == (0,)


def test_progressive_claim_requires_fresh_heartbeat_and_export_generation(
    db_connection,
) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    _insert_requester_directory(db_connection)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    service.report_progressive_coverage(_report(replica_id=replica_id))

    db_connection.execute(
        "UPDATE artifact_replicas SET export_generation = 8 WHERE replica_id = ?",
        [replica_id],
    )
    stale_generation = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-generation",
        deadline_at=None,
    )
    assert stale_generation.assignment is None
    assert stale_generation.no_eligible_reason == "not_found"

    db_connection.execute(
        "UPDATE artifact_replicas SET export_generation = 7 WHERE replica_id = ?",
        [replica_id],
    )
    db_connection.execute(
        """
        UPDATE worker_liveness
        SET last_heartbeat = ?
        WHERE worker_id = 'worker-src'
        """,
        [datetime.now(timezone.utc) - timedelta(minutes=10)],
    )
    stale_heartbeat = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-heartbeat",
        deadline_at=None,
    )
    assert stale_heartbeat.assignment is None
    assert stale_heartbeat.no_eligible_reason == "not_found"


def test_progressive_claim_rejects_stale_worker_directory_domain(
    db_connection,
) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    _insert_requester_directory(db_connection)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    service.report_progressive_coverage(_report(replica_id=replica_id))

    db_connection.execute(
        "UPDATE workers SET node_id = 'dc-b' WHERE worker_id = 'worker-src'"
    )
    stale_domain = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-stale-domain",
        deadline_at=None,
    )

    assert stale_domain.assignment is None
    assert stale_domain.no_eligible_reason == "not_found"


def test_progressive_expiration_releases_assignments_and_retires_coverage(
    db_connection,
) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    _insert_source_directory(db_connection, replica_id=replica_id)
    _insert_requester_directory(db_connection)
    service = ProgressiveReplicationService(
        ProgressiveCoverageRepository(db_connection)
    )
    service.report_progressive_coverage(_report(replica_id=replica_id))

    claim = service.find_progressive_source(
        identity=_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-expire",
        deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert claim.assignment is not None

    expired_coverage, expired_assignments = service.expire_progressive_state()
    assert expired_coverage == 0
    assert expired_assignments == 1
    counter_row = db_connection.execute(
        """
        SELECT active_assignments
        FROM progressive_source_counters
        WHERE source_replica_id = ?
        """,
        [replica_id],
    ).fetchone()
    assert counter_row == (0,)

    expired_replica_id = str(uuid.uuid4())
    _insert_source_replica(db_connection, replica_id=expired_replica_id)
    service.report_progressive_coverage(
        _report(
            coverage_id="coverage-expired",
            replica_id=expired_replica_id,
            deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    )
    expired_coverage, expired_assignments = service.expire_progressive_state()
    assert expired_coverage == 1
    assert expired_assignments == 0

    active_coverage_replica_id = str(uuid.uuid4())
    _insert_source_replica(db_connection, replica_id=active_coverage_replica_id)
    service.report_progressive_coverage(
        _report(
            coverage_id="coverage-active-expired",
            replica_id=active_coverage_replica_id,
            verified_bytes=128,
            completed_bytes=128,
            deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    )
    active_assignment_id = str(uuid.uuid4())
    db_connection.execute(
        """
        INSERT INTO progressive_source_counters (
            source_replica_id, source_daemon_id, source_export_generation,
            active_assignments
        ) VALUES (?, 'daemon-src', 7, 1)
        """,
        [active_coverage_replica_id],
    )
    db_connection.execute(
        """
        INSERT INTO progressive_source_assignments (
            assignment_id, coverage_id, requester_daemon_id,
            requester_worker_id, requester_materialization_attempt_id,
            source_daemon_id, source_worker_id, source_domain,
            seed_transport_kind, start_unit, end_unit_exclusive, start_byte,
            end_byte_exclusive, source_export_generation, state, deadline_at,
            request_fingerprint
        ) VALUES (?, 'coverage-active-expired', 'daemon-dst', 'worker-dst',
                  'attempt-dst-active-expired', 'daemon-src', 'worker-src',
                  'dc-a', NULL, 0, 32, 0, 32, 7, 'claimed', ?, ?)
        """,
        [
            active_assignment_id,
            datetime.now(timezone.utc) + timedelta(minutes=5),
            b"claim-active-expired-coverage",
        ],
    )
    expired_coverage, expired_assignments = service.expire_progressive_state()
    assert expired_coverage == 1
    assert expired_assignments == 1
    assignment_state = db_connection.execute(
        """
        SELECT state
        FROM progressive_source_assignments
        WHERE assignment_id = ?
        """,
        [active_assignment_id],
    ).fetchone()
    assert assignment_state == (ProgressiveAssignmentState.CANCELLED.value,)
    counter_row = db_connection.execute(
        """
        SELECT active_assignments
        FROM progressive_source_counters
        WHERE source_replica_id = ?
        """,
        [active_coverage_replica_id],
    ).fetchone()
    assert counter_row == (0,)


def _proto_identity() -> global_store_pb2.ProgressiveCoverageIdentity:
    identity = global_store_pb2.ProgressiveCoverageIdentity(
        artifact_id="mi2:index:data",
        selection_hash=b"sel",
        logical_layout_hash=b"layout",
        coverage_order_hash=b"order",
        group_version_set_id="gvs-1",
        group_part_id="rank-0",
    )
    identity.byte_space.kind = common_pb2.BYTE_SPACE_KIND_CANONICAL
    identity.hash_space.byte_space.kind = common_pb2.BYTE_SPACE_KIND_CANONICAL
    identity.hash_space.canonical_index_multihash = "bafy-index"
    return identity


def test_progressive_rpc_report_and_claim(test_context) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    from tensorcast.global_store.grpc_service import GlobalStoreServicer

    servicer = GlobalStoreServicer()
    _insert_source_directory(servicer.connection, replica_id=replica_id)
    _insert_requester_directory(servicer.connection)

    report = global_store_pb2.ReportProgressiveCoverageRequest(
        coverage_id="coverage-rpc",
        identity=_proto_identity(),
        replica_id=replica_id,
        daemon_id="daemon-src",
        daemon_session_id="session-src",
        worker_id="worker-src",
        source_export_generation=7,
        coverage_epoch=1,
        coverage_kind=global_store_pb2.PROGRESSIVE_COVERAGE_KIND_BYTE_PREFIX,
        verified_units=96,
        verified_bytes=96,
        completed_units=96,
        completed_bytes=96,
        total_units=128,
        total_bytes=128,
        materialization_attempt_id="attempt-src",
        state=global_store_pb2.PROGRESSIVE_COVERAGE_STATE_VERIFIED,
        export_state=global_store_pb2.PROGRESSIVE_EXPORT_STATE_IN_PROGRESS_EXPORTABLE,
        source_domain="dc-a",
        deadline_unix_nanos=int(
            (datetime.now(timezone.utc) + timedelta(seconds=60)).timestamp()
            * 1_000_000_000
        ),
    )
    report_resp = servicer.ReportProgressiveCoverage(report, test_context)
    assert report_resp.status == global_store_pb2.STATUS_OK
    assert report_resp.updated is True

    claim = global_store_pb2.FindProgressiveSourceRequest(
        identity=_proto_identity(),
        next_unit=0,
        max_units=32,
        requester_daemon_id="daemon-dst",
        requester_worker_id="worker-dst",
        requester_source_domain="dc-a",
        requester_materialization_attempt_id="attempt-dst",
        request_fingerprint=b"claim-rpc",
    )
    claim_resp = servicer.FindProgressiveSource(claim, test_context)
    assert claim_resp.status == global_store_pb2.STATUS_OK
    assert claim_resp.assignment.replica_id == replica_id
    assert claim_resp.assignment.end_byte_exclusive == 32
    assert claim_resp.assignment.source_memory_info.node_address == "10.0.0.1"
    assert claim_resp.assignment.source_memory_info.transport.export_generation == 7
    assert list(
        claim_resp.assignment.source_memory_info.transport.remote_memory_keys
    ) == ["rkey"]


def test_progressive_rpc_report_requires_explicit_visibility_state(
    test_context,
) -> None:
    _configure_progressive()
    replica_id = str(uuid.uuid4())
    from tensorcast.global_store.grpc_service import GlobalStoreServicer

    servicer = GlobalStoreServicer()
    _insert_source_directory(servicer.connection, replica_id=replica_id)

    report = global_store_pb2.ReportProgressiveCoverageRequest(
        coverage_id="coverage-rpc-unspecified",
        identity=_proto_identity(),
        replica_id=replica_id,
        daemon_id="daemon-src",
        daemon_session_id="session-src",
        worker_id="worker-src",
        source_export_generation=7,
        coverage_epoch=1,
        coverage_kind=global_store_pb2.PROGRESSIVE_COVERAGE_KIND_BYTE_PREFIX,
        verified_units=96,
        verified_bytes=96,
        completed_units=96,
        completed_bytes=96,
        total_units=128,
        total_bytes=128,
        materialization_attempt_id="attempt-src",
        source_domain="dc-a",
    )

    response = servicer.ReportProgressiveCoverage(report, test_context)

    assert response.status == global_store_pb2.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "state is required" in (test_context.details or "")
