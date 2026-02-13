#  Copyright (c) 2025-2026, TensorCast Team.

"""Shared helpers for control-plane idempotency."""

from __future__ import annotations

import json
from dataclasses import dataclass

import grpc

from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.repositories.idempotency_repository import (
    IDEMPOTENCY_PENDING_STATUS,
    IdempotencyRecord,
    IdempotencyRepository,
)


@dataclass(frozen=True)
class BeginIdempotentOperationResult:
    should_execute: bool
    payload_mismatch: bool
    timed_out_waiting_for_replay: bool
    client_request_id: str
    request_fingerprint: str
    replay_record: IdempotencyRecord | None


def begin_idempotent_operation(
    *,
    idempotency_repository: IdempotencyRepository,
    operation_kind: str,
    client_request_id: str,
    request_payload: bytes,
) -> BeginIdempotentOperationResult:
    request_fingerprint = idempotency_repository.fingerprint_payload(request_payload)
    reserved = idempotency_repository.reserve_operation(
        client_request_id=client_request_id,
        operation_kind=operation_kind,
        request_fingerprint=request_fingerprint,
    )
    if reserved:
        return BeginIdempotentOperationResult(
            should_execute=True,
            payload_mismatch=False,
            timed_out_waiting_for_replay=False,
            client_request_id=client_request_id,
            request_fingerprint=request_fingerprint,
            replay_record=None,
        )

    record = idempotency_repository.get_record(client_request_id)
    if record is None:
        return BeginIdempotentOperationResult(
            should_execute=True,
            payload_mismatch=False,
            timed_out_waiting_for_replay=False,
            client_request_id=client_request_id,
            request_fingerprint=request_fingerprint,
            replay_record=None,
        )
    if record.operation_kind != operation_kind:
        gs_metrics.inc_idempotency_replay_conflict(operation_kind=operation_kind)
        return BeginIdempotentOperationResult(
            should_execute=False,
            payload_mismatch=True,
            timed_out_waiting_for_replay=False,
            client_request_id=client_request_id,
            request_fingerprint=request_fingerprint,
            replay_record=None,
        )
    if record.request_fingerprint != request_fingerprint:
        gs_metrics.inc_idempotency_replay_conflict(operation_kind=operation_kind)
        return BeginIdempotentOperationResult(
            should_execute=False,
            payload_mismatch=True,
            timed_out_waiting_for_replay=False,
            client_request_id=client_request_id,
            request_fingerprint=request_fingerprint,
            replay_record=None,
        )
    if record.response_status == IDEMPOTENCY_PENDING_STATUS:
        record = idempotency_repository.wait_for_completed_record(
            client_request_id=client_request_id
        )
        if record is None:
            return BeginIdempotentOperationResult(
                should_execute=False,
                payload_mismatch=False,
                timed_out_waiting_for_replay=True,
                client_request_id=client_request_id,
                request_fingerprint=request_fingerprint,
                replay_record=None,
            )
    gs_metrics.inc_idempotency_replay_hit(operation_kind=operation_kind)
    return BeginIdempotentOperationResult(
        should_execute=False,
        payload_mismatch=False,
        timed_out_waiting_for_replay=False,
        client_request_id=client_request_id,
        request_fingerprint=request_fingerprint,
        replay_record=record,
    )


def encode_stored_grpc_status(
    code: grpc.StatusCode | None,
    details: str | None,
) -> str:
    effective_code = code if code is not None else grpc.StatusCode.OK
    payload = {
        "grpc_code": effective_code.name,
        "grpc_details": details or "",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_stored_grpc_status(value: str) -> tuple[grpc.StatusCode, str]:
    try:
        payload = json.loads(value)
    except Exception:  # noqa: BLE001
        return grpc.StatusCode.INTERNAL, "invalid idempotency status payload"
    if not isinstance(payload, dict):
        return grpc.StatusCode.INTERNAL, "invalid idempotency status payload"
    code_name = str(payload.get("grpc_code", "INTERNAL"))
    details = str(payload.get("grpc_details", ""))
    code = getattr(grpc.StatusCode, code_name, grpc.StatusCode.INTERNAL)
    return code, details
