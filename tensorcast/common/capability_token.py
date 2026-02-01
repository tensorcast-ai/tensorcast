#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from google.protobuf.message import Message

from tensorcast.proto.common.v1 import capability_token_pb2


@dataclass(frozen=True, slots=True)
class CapabilityTokenKey:
    version: int
    secret: bytes


@dataclass(frozen=True, slots=True)
class CapabilityTokenConfig:
    active: CapabilityTokenKey
    previous: tuple[CapabilityTokenKey, ...] = ()


def _serialize_deterministic(message: Message) -> bytes:
    return message.SerializeToString(deterministic=True)


def serialize_scope(message: Message) -> bytes:
    return _serialize_deterministic(message)


class CapabilityTokenManager:
    def __init__(self, config: CapabilityTokenConfig) -> None:
        self._config = config

    def configured(self) -> bool:
        return bool(self._config.active.version and self._config.active.secret)

    def mint(
        self,
        *,
        issuer_daemon_id: str,
        audience: capability_token_pb2.CapabilityAudience,
        scope: bytes,
        expires_at_ms: int,
        queue_epoch: capability_token_pb2.QueueEpochFencing | None = None,
    ) -> bytes:
        if not self.configured():
            raise ValueError("capability token keys are not configured")
        if not issuer_daemon_id:
            raise ValueError("issuer_daemon_id is required")
        if audience == capability_token_pb2.CAPABILITY_AUDIENCE_UNSPECIFIED:
            raise ValueError("audience is required")
        if not scope:
            raise ValueError("scope is required")
        if expires_at_ms <= 0:
            raise ValueError("expires_at_ms is required")

        env = capability_token_pb2.CapabilityTokenEnvelope(
            token_version=int(self._config.active.version),
            issuer_daemon_id=str(issuer_daemon_id),
            audience=audience,
            scope=bytes(scope),
            expires_at_ms=int(expires_at_ms),
        )
        if queue_epoch is not None:
            env.queue_epoch.CopyFrom(queue_epoch)
        auth = self._compute_auth_tag(env, self._config.active.secret)
        env.auth_tag = auth
        return _serialize_deterministic(env)

    def verify(
        self,
        token: bytes,
        *,
        expected_audience: capability_token_pb2.CapabilityAudience,
        expected_issuer: str,
        now_ms: int,
        require_not_expired: bool,
    ) -> capability_token_pb2.CapabilityTokenEnvelope:
        if not self.configured():
            raise ValueError("capability token keys are not configured")
        env = capability_token_pb2.CapabilityTokenEnvelope()
        env.ParseFromString(token)
        if (
            env.token_version == 0
            or not env.issuer_daemon_id
            or env.audience == capability_token_pb2.CAPABILITY_AUDIENCE_UNSPECIFIED
            or not env.scope
            or not env.auth_tag
        ):
            raise ValueError("capability token missing required fields")
        if expected_issuer and env.issuer_daemon_id != expected_issuer:
            raise PermissionError("capability token issuer mismatch")
        if (
            expected_audience != capability_token_pb2.CAPABILITY_AUDIENCE_UNSPECIFIED
            and env.audience != expected_audience
        ):
            raise PermissionError("capability token audience mismatch")
        if require_not_expired and env.expires_at_ms <= int(now_ms):
            raise PermissionError("capability token expired")

        key = self._key_for_version(env.token_version)
        if key is None:
            raise PermissionError("capability token key version not recognized")
        expected = self._compute_auth_tag(env, key.secret)
        if not hmac.compare_digest(expected, env.auth_tag):
            raise PermissionError("capability token auth tag mismatch")
        return env

    @staticmethod
    def looks_like_envelope(token: bytes) -> bool:
        env = capability_token_pb2.CapabilityTokenEnvelope()
        try:
            env.ParseFromString(token)
        except Exception:  # noqa: BLE001
            return False
        return bool(
            env.token_version
            and env.issuer_daemon_id
            and env.audience != capability_token_pb2.CAPABILITY_AUDIENCE_UNSPECIFIED
            and env.scope
            and env.auth_tag
        )

    def _key_for_version(self, version: int) -> CapabilityTokenKey | None:
        if version == self._config.active.version:
            return self._config.active
        for key in self._config.previous:
            if version == key.version:
                return key
        return None

    @staticmethod
    def _compute_auth_tag(
        env: capability_token_pb2.CapabilityTokenEnvelope, secret: bytes
    ) -> bytes:
        unsigned_env = capability_token_pb2.CapabilityTokenEnvelope()
        unsigned_env.CopyFrom(env)
        unsigned_env.auth_tag = b""
        payload = _serialize_deterministic(unsigned_env)
        return hmac.new(secret, payload, hashlib.sha256).digest()
