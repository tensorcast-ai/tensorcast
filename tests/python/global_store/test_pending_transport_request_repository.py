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


def test_create_if_absent_with_cursor_is_idempotent(repositories):
    pending_repo = repositories["pending_transport_request"]
    first = _pending_request(request_id="pending-idempotent-1", request_fingerprint="fp-1")

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
