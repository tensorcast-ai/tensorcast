#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Mapping

from tensorcast.api.errors import ArtifactError


def _encode_capability_token(token: bytes) -> str:
    raw = base64.urlsafe_b64encode(token).decode("ascii")
    return raw.rstrip("=")


def _decode_capability_token(token: str) -> bytes:
    raw = token.strip()
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Serializable reference to engine-owned buffers."""

    instance_id: str
    name: str
    tensors: Mapping[str, str]
    layout_hash: str | None = None
    expires_at_ms: int = 0
    capability_token: str = ""

    def is_expired(self, *, now_ms: int | None = None) -> bool:
        if self.expires_at_ms <= 0:
            return True
        now_value = now_ms if now_ms is not None else int(time.time() * 1000)
        return now_value >= int(self.expires_at_ms)

    def validate(self, *, now_ms: int | None = None) -> None:
        if not self.instance_id:
            raise ArtifactError(
                "TargetSpec.instance_id is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not self.capability_token:
            raise ArtifactError(
                "TargetSpec.capability_token is required",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self.is_expired(now_ms=now_ms):
            raise ArtifactError(
                "TargetSpec capability has expired",
                status_code="DEADLINE_EXCEEDED",
                retryable=True,
            )

    def to_proto(self):
        from tensorcast.proto.plan.v1 import plan_pb2

        proto = plan_pb2.TargetSpec(
            instance_id=str(self.instance_id),
            name=str(self.name),
        )
        proto.tensors.update({str(k): str(v) for k, v in self.tensors.items()})
        if self.layout_hash is not None:
            proto.layout_hash = str(self.layout_hash)
        if self.expires_at_ms:
            proto.expires_at_ms = int(self.expires_at_ms)
        if self.capability_token:
            proto.capability_token = _decode_capability_token(self.capability_token)
        return proto

    @staticmethod
    def from_proto(proto) -> "TargetSpec":
        capability = (
            _encode_capability_token(bytes(proto.capability_token))
            if proto.capability_token
            else ""
        )
        layout_hash = str(proto.layout_hash) if proto.layout_hash else None
        expires_at_ms = int(proto.expires_at_ms) if proto.expires_at_ms else 0
        tensors = {str(k): str(v) for k, v in proto.tensors.items()}
        return TargetSpec(
            instance_id=str(proto.instance_id),
            name=str(proto.name),
            tensors=tensors,
            layout_hash=layout_hash,
            expires_at_ms=expires_at_ms,
            capability_token=capability,
        )


__all__ = [
    "TargetSpec",
]
