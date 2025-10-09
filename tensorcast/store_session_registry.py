#  Copyright (c) 2025, TensorCast Team.

"""Persistent metadata registry for Store session instances.

Records each Store session under ``~/.tensorcast/store_sessions`` so that
machine operators can inspect active clients via the CLI.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "StoreSessionRecord",
    "store_sessions_root",
    "write_record",
    "load_record",
    "iter_records",
]


@dataclass(slots=True)
class StoreSessionRecord:
    """Serialized representation of a Store session."""

    session_id: str
    pid: int
    daemon_endpoint: str
    created_at: float
    last_activity_at: float
    closed_at: float | None = None
    active_leases: int = 0
    pending_futures: int = 0
    capabilities: dict[str, object] = field(default_factory=dict)

    def mark_activity(self) -> None:
        self.last_activity_at = time.time()

    def mark_closed(self) -> None:
        self.closed_at = time.time()


def store_sessions_root() -> Path:
    root = Path.home() / ".tensorcast" / "store_sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _record_path(session_id: str) -> Path:
    return store_sessions_root() / f"{session_id}.json"


def write_record(record: StoreSessionRecord) -> None:
    payload = asdict(record)
    # Provide human-friendly ISO timestamps alongside raw floats for CLI UX.
    payload["created_at_iso"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created_at)
    )
    payload["last_activity_at_iso"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.last_activity_at)
    )
    if record.closed_at is not None:
        payload["closed_at_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.closed_at)
        )
    path = _record_path(record.session_id)
    tmp_path = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def load_record(session_id: str) -> StoreSessionRecord | None:
    path = _record_path(session_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return _record_from_payload(payload)


def _record_from_payload(payload: dict[str, Any]) -> StoreSessionRecord | None:
    try:
        session_id = str(payload["session_id"])
        pid = int(payload["pid"])
        daemon_endpoint = str(payload["daemon_endpoint"])
        created_at = float(payload["created_at"])
        last_activity_at = float(payload["last_activity_at"])
    except (KeyError, TypeError, ValueError):
        return None
    closed_at_raw = payload.get("closed_at")
    closed_at = float(closed_at_raw) if closed_at_raw is not None else None
    active_leases = int(payload.get("active_leases", 0) or 0)
    pending_futures = int(payload.get("pending_futures", 0) or 0)
    capabilities_obj = payload.get("capabilities")
    capabilities: dict[str, object]
    capabilities = dict(capabilities_obj) if isinstance(capabilities_obj, dict) else {}
    return StoreSessionRecord(
        session_id=session_id,
        pid=pid,
        daemon_endpoint=daemon_endpoint,
        created_at=created_at,
        last_activity_at=last_activity_at,
        closed_at=closed_at,
        active_leases=active_leases,
        pending_futures=pending_futures,
        capabilities=capabilities,
    )


def iter_records() -> Iterator[StoreSessionRecord]:
    root = store_sessions_root()
    for entry in sorted(root.glob("*.json")):
        record = load_record(entry.stem)
        if record is not None:
            yield record
