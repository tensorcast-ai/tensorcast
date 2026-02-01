#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import base64
from dataclasses import dataclass

from tensorcast.api.context import CallContext
from tensorcast.daemon_ctl import DaemonCtl, get_daemon_client
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def _encode_capability_token(token: bytes) -> str:
    raw = base64.urlsafe_b64encode(token).decode("ascii")
    return raw.rstrip("=")


def _decode_capability_token(token: str) -> bytes:
    raw = token.strip()
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _ctx_timeout_s(ctx: CallContext | None) -> float | None:
    if ctx is None or ctx.deadline_ms is None:
        return None
    timeout_s = float(ctx.deadline_ms) / 1000.0
    if timeout_s <= 0:
        raise RuntimeError("CallContext deadline exceeded")
    return max(0.001, timeout_s)


def _resolve_client(daemon_endpoint: str | None, client: DaemonCtl | None) -> DaemonCtl:
    if client is not None:
        return client
    if daemon_endpoint is None:
        return get_daemon_client()
    return get_daemon_client(daemon_endpoint)


@dataclass(frozen=True, slots=True)
class RetentionHandle:
    handle_id: str
    expires_at_ms: int
    capability_token: str
    charged_bytes: int
    diagnostics: str | None = None


def _handle_from_proto(
    handle: store_daemon_pb2.RetentionHandle,
) -> RetentionHandle:
    token = bytes(handle.capability_token)
    if not token:
        raise RuntimeError("RetentionHandle missing capability_token")
    diagnostics = handle.diagnostics or None
    return RetentionHandle(
        handle_id=handle.handle_id,
        expires_at_ms=int(handle.expires_at_ms),
        capability_token=_encode_capability_token(token),
        charged_bytes=int(handle.charged_bytes),
        diagnostics=diagnostics,
    )


def acquire_retention_handle(
    *,
    selection: common_pb2.ArtifactSelection,
    policy: store_daemon_pb2.StorePolicy | None = None,
    ttl_ms: int | None = None,
    ctx: CallContext | None = None,
    daemon_endpoint: str | None = None,
    client: DaemonCtl | None = None,
) -> RetentionHandle:
    if ttl_ms is not None and int(ttl_ms) <= 0:
        raise ValueError("ttl_ms must be positive when provided")
    ctl = _resolve_client(daemon_endpoint, client)
    timeout_s = _ctx_timeout_s(ctx)
    resp = ctl.acquire_retention_handle(
        selection=selection,
        policy=policy,
        ttl_ms=ttl_ms,
        timeout_s=timeout_s,
    )
    if not resp.HasField("handle"):
        raise RuntimeError("AcquireRetentionHandle returned empty handle")
    return _handle_from_proto(resp.handle)


def renew_retention_handle(
    *,
    handle_token: str,
    extend_ttl_ms: int,
    ctx: CallContext | None = None,
    daemon_endpoint: str | None = None,
    client: DaemonCtl | None = None,
) -> RetentionHandle:
    if int(extend_ttl_ms) <= 0:
        raise ValueError("extend_ttl_ms must be positive")
    ctl = _resolve_client(daemon_endpoint, client)
    timeout_s = _ctx_timeout_s(ctx)
    resp = ctl.renew_retention_handle(
        handle_token=_decode_capability_token(handle_token),
        extend_ttl_ms=int(extend_ttl_ms),
        timeout_s=timeout_s,
    )
    if not resp.HasField("handle"):
        raise RuntimeError("RenewRetentionHandle returned empty handle")
    return _handle_from_proto(resp.handle)


def release_retention_handle(
    *,
    handle_token: str,
    ctx: CallContext | None = None,
    daemon_endpoint: str | None = None,
    client: DaemonCtl | None = None,
) -> bool:
    ctl = _resolve_client(daemon_endpoint, client)
    timeout_s = _ctx_timeout_s(ctx)
    resp = ctl.release_retention_handle(
        handle_token=_decode_capability_token(handle_token),
        timeout_s=timeout_s,
    )
    return bool(resp.released)


__all__ = [
    "RetentionHandle",
    "acquire_retention_handle",
    "renew_retention_handle",
    "release_retention_handle",
]
