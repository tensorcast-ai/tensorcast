#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for PendingTransportRequestRepository state transitions."""

from datetime import datetime, timedelta, timezone

from tensorcast.global_store.models import (
    PendingTransportRequest,
    PendingTransportState,
)


def _pending_request(
    *,
    request_id: str,
    request_fingerprint: str = "fp-default",
    artifact_id: str = "artifact-pending",
    deadline_at: datetime | None = None,
) -> PendingTransportRequest:
    return PendingTransportRequest(
        request_id=request_id,
        request_fingerprint=request_fingerprint,
        artifact_id=artifact_id,
        requested_view_id=None,
        source_node_id="source-node",
        source_address="192.168.10.1",
        source_port=9000,
        requester_worker_id=None,
        deadline_at=deadline_at,
    )


def _double(value: str) -> str:
    return f"{value}{value}"


def test_create_if_absent_with_cursor_is_idempotent(repositories):
    pending_repo = repositories["pending_transport_request"]
    first = _pending_request(
        request_id="pending-idempotent-1", request_fingerprint="fp-1"
    )

    with pending_repo.transaction() as tx:
        created = pending_repo.create_if_absent_with_cursor(first, tx)
        duplicate = _pending_request(
            request_id="pending-idempotent-1",
            request_fingerprint="fp-1",
            artifact_id="artifact-changed-but-idempotent",
        )
        resolved = pending_repo.create_if_absent_with_cursor(duplicate, tx)

    assert created.request_id == "pending-idempotent-1"
    assert resolved.request_id == created.request_id
    assert resolved.artifact_id == created.artifact_id
    assert resolved.state == PendingTransportState.ENQUEUED


def test_mark_dispatched_transitions_only_once(repositories):
    pending_repo = repositories["pending_transport_request"]
    with pending_repo.transaction() as tx:
        pending_repo.create_if_absent_with_cursor(
            _pending_request(request_id="pending-dispatch-once-1"),
            tx,
        )
        assert pending_repo.mark_dispatched("pending-dispatch-once-1", tx) is True
        assert pending_repo.mark_dispatched("pending-dispatch-once-1", tx) is False

    row = pending_repo.find_by_request_id("pending-dispatch-once-1")
    assert row is not None
    assert row.state == PendingTransportState.DISPATCHED
    assert row.dispatched_at is not None


def test_mark_cancelled_only_applies_to_enqueued_requests(repositories):
    pending_repo = repositories["pending_transport_request"]
    with pending_repo.transaction() as tx:
        pending_repo.create_if_absent_with_cursor(
            _pending_request(request_id="pending-cancelled-once-1"),
            tx,
        )
    assert pending_repo.mark_cancelled("pending-cancelled-once-1") is True
    assert pending_repo.mark_cancelled("pending-cancelled-once-1") is False

    with pending_repo.transaction() as tx:
        assert pending_repo.mark_dispatched("pending-cancelled-once-1", tx) is False

    row = pending_repo.find_by_request_id("pending-cancelled-once-1")
    assert row is not None
    assert row.state == PendingTransportState.CANCELLED


def test_expire_enqueued_deadlines_only_expires_due_rows(repositories):
    pending_repo = repositories["pending_transport_request"]
    now_utc = datetime.now(timezone.utc)

    with pending_repo.transaction() as tx:
        pending_repo.create_if_absent_with_cursor(
            _pending_request(
                request_id="pending-expire-due-1",
                deadline_at=now_utc - timedelta(seconds=10),
            ),
            tx,
        )
        pending_repo.create_if_absent_with_cursor(
            _pending_request(
                request_id="pending-expire-future-1",
                deadline_at=now_utc + timedelta(seconds=10),
            ),
            tx,
        )
        pending_repo.create_if_absent_with_cursor(
            _pending_request(request_id="pending-expire-no-deadline-1"),
            tx,
        )
        pending_repo.create_if_absent_with_cursor(
            _pending_request(
                request_id="pending-expire-dispatched-1",
                deadline_at=now_utc - timedelta(seconds=10),
            ),
            tx,
        )
        assert pending_repo.mark_dispatched("pending-expire-dispatched-1", tx) is True

    expired_ids = pending_repo.expire_enqueued_deadlines(now_utc=now_utc)
    assert expired_ids == ["pending-expire-due-1"]

    due = pending_repo.find_by_request_id("pending-expire-due-1")
    future = pending_repo.find_by_request_id("pending-expire-future-1")
    no_deadline = pending_repo.find_by_request_id("pending-expire-no-deadline-1")
    dispatched = pending_repo.find_by_request_id("pending-expire-dispatched-1")

    assert due is not None
    assert future is not None
    assert no_deadline is not None
    assert dispatched is not None
    assert due.state == PendingTransportState.EXPIRED
    assert future.state == PendingTransportState.ENQUEUED
    assert no_deadline.state == PendingTransportState.ENQUEUED
    assert dispatched.state == PendingTransportState.DISPATCHED


def test_purge_malformed_rows_does_not_drop_request_id_by_token_pattern(repositories):
    pending_repo = repositories["pending_transport_request"]
    request_id = "user-transport:alpha-any-transport:beta"

    with pending_repo.transaction() as tx:
        pending_repo.create_if_absent_with_cursor(
            _pending_request(request_id=request_id),
            tx,
        )
        purged = pending_repo.purge_malformed_rows(cursor=tx)
        assert purged == 0

    row = pending_repo.find_by_request_id(request_id)
    assert row is not None
    assert row.request_id == request_id
    assert row.state == PendingTransportState.ENQUEUED


def test_create_if_absent_normalizes_exact_doubled_tokens(repositories):
    pending_repo = repositories["pending_transport_request"]
    canonical_request_id = "suite:v3:rx4:r1:a0"
    canonical_group_id = "suite:v3"
    canonical_group_kind = "tp_version"
    canonical_group_part_id = "rx4:r1"
    canonical_fingerprint = "fp-dedupe-1"
    canonical_artifact_id = "cgid:artifact-v3"

    request = PendingTransportRequest(
        request_id=_double(canonical_request_id),
        request_fingerprint=_double(canonical_fingerprint),
        artifact_id=_double(canonical_artifact_id),
        requested_view_id=None,
        source_node_id=_double("node-a"),
        source_address=_double("192.168.10.1"),
        source_port=9000,
        requester_worker_id=_double("worker_a"),
        group_id=_double(canonical_group_id),
        group_kind=_double(canonical_group_kind),
        group_total_parts=28,
        group_part_id=_double(canonical_group_part_id),
        group_priority=0,
        group_epoch=9,
    )

    with pending_repo.transaction() as tx:
        created = pending_repo.create_if_absent_with_cursor(request, tx)

    assert created.request_id == canonical_request_id
    assert created.request_fingerprint == canonical_fingerprint
    assert created.artifact_id == canonical_artifact_id
    assert created.source_node_id == "node-a"
    assert created.source_address == "192.168.10.1"
    assert created.requester_worker_id == "worker_a"
    assert created.group_id == canonical_group_id
    assert created.group_kind == canonical_group_kind
    assert created.group_part_id == canonical_group_part_id

    resolved = pending_repo.find_by_request_id(canonical_request_id)
    assert resolved is not None
    assert resolved.request_id == canonical_request_id
    assert resolved.group_kind == canonical_group_kind


def test_purge_malformed_rows_heals_exact_doubled_tokens(repositories):
    pending_repo = repositories["pending_transport_request"]
    request_id = "heal:v3:rx4:r1:a0"

    with pending_repo.transaction() as tx:
        pending_repo.create_if_absent_with_cursor(
            PendingTransportRequest(
                request_id=request_id,
                request_fingerprint="fp-heal",
                artifact_id="cgid:heal-artifact",
                requested_view_id=None,
                source_node_id="node-a",
                source_address="192.168.10.1",
                source_port=9000,
                requester_worker_id="worker-a",
                group_id="heal:v3",
                group_kind="tp_version",
                group_total_parts=28,
                group_part_id="rx4:r1",
                group_priority=0,
                group_epoch=1,
            ),
            tx,
        )
        tx.execute(
            """
            UPDATE pending_transport_requests
            SET request_id = ?,
                request_fingerprint = ?,
                artifact_id = ?,
                source_node_id = ?,
                source_address = ?,
                requester_worker_id = ?,
                group_id = ?,
                group_kind = ?,
                group_part_id = ?
            WHERE request_id = ?
            """,
            [
                _double(request_id),
                _double("fp-heal"),
                _double("cgid:heal-artifact"),
                _double("node-a"),
                _double("192.168.10.1"),
                _double("worker-a"),
                _double("heal:v3"),
                _double("tp_version"),
                _double("rx4:r1"),
                request_id,
            ],
        )
        reconciled = pending_repo.purge_malformed_rows(cursor=tx)
        assert reconciled == 1

    healed = pending_repo.find_by_request_id(request_id)
    assert healed is not None
    assert healed.request_id == request_id
    assert healed.request_fingerprint == "fp-heal"
    assert healed.artifact_id == "cgid:heal-artifact"
    assert healed.source_node_id == "node-a"
    assert healed.source_address == "192.168.10.1"
    assert healed.requester_worker_id == "worker-a"
    assert healed.group_id == "heal:v3"
    assert healed.group_kind == "tp_version"
    assert healed.group_part_id == "rx4:r1"
