#  Copyright (c) 2025-2026, TensorCast Team.

"""Common helper utilities for Global Store gRPC services."""

from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import datetime, timezone
from pathlib import PurePosixPath

from google.protobuf import timestamp_pb2


def timestamp_to_datetime(ts: timestamp_pb2.Timestamp | None) -> datetime | None:
    """Convert protobuf Timestamp to timezone-aware datetime (UTC)."""
    if ts is None:
        return None
    return ts.ToDatetime(tzinfo=timezone.utc)


def datetime_to_timestamp(dt: datetime | None) -> timestamp_pb2.Timestamp | None:
    """Convert datetime to protobuf Timestamp (UTC)."""
    if dt is None:
        return None
    normalized = (
        dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    )
    proto = timestamp_pb2.Timestamp()
    proto.FromDatetime(normalized)
    return proto


def coerce_db_datetime(value: object) -> datetime | None:
    """Best-effort conversion for DuckDB timestamp outputs."""
    if value is None:
        return None
    if isinstance(value, datetime):
        candidate = value
    elif isinstance(value, str):
        candidate = datetime.fromisoformat(value)
    else:
        raise ValueError(f"Unsupported datetime value: {value!r}")
    return (
        candidate.astimezone(timezone.utc)
        if candidate.tzinfo
        else candidate.replace(tzinfo=timezone.utc)
    )


def is_safe_relative_path(path: str) -> bool:
    if not path:
        return False
    if "\\" in path:
        return False
    pure = PurePosixPath(path)
    if pure.is_absolute():
        return False
    return ".." not in pure.parts


def multibase_sha256_to_hex(value: str) -> str | None:
    """Convert multibase base32 multihash (sha2-256) to lowercase hex digest."""
    if not value or value[0] != "b":
        return None
    payload = value[1:]
    if not payload:
        return None
    padding_needed = (-len(payload)) % 8
    padded = payload + ("=" * padding_needed)
    try:
        decoded = base64.b32decode(padded.upper(), casefold=True)
    except binascii.Error:
        return None
    if len(decoded) != 34 or decoded[0] != 0x12 or decoded[1] != 0x20:
        return None
    digest = decoded[2:]
    if len(digest) != 32:
        return None
    return digest.hex()


def sha256_digest_to_multibase(digest: bytes) -> str | None:
    """Convert raw SHA-256 digest bytes to multibase base32 multihash."""
    if len(digest) != 32:
        return None
    multihash = b"\x12\x20" + digest
    b32 = base64.b32encode(multihash).decode("ascii").lower().rstrip("=")
    return f"b{b32}"


def index_bytes_to_multibase_sha256(data: bytes) -> str | None:
    if not data:
        return None
    digest = hashlib.sha256(data).digest()
    return sha256_digest_to_multibase(digest)


def hex_sha256_to_multibase(value: str) -> str | None:
    if not value:
        return None
    try:
        digest = bytes.fromhex(value)
    except ValueError:
        return None
    return sha256_digest_to_multibase(digest)
